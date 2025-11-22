import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silence HF tokenizer fork warning

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)  # avoid fork-after-tokenizer
except RuntimeError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import time
import datetime
import argparse
from pathlib import Path
import random
import contextlib
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.utils.logging import set_verbosity_error  # hide HF warnings if desired
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score

# optional: hide most HF warnings
set_verbosity_error()

# Silence SentencePiece fast-tokenizer byte-fallback notice (we’re intentionally using fast)
warnings.filterwarnings(
    "ignore",
    message=".*uses the byte fallback option.*",
    category=UserWarning,
    module="transformers.convert_slow_tokenizer",
)

# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "roberta-large": "roberta-large",
    "roberta-base": "roberta-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "bert-base-uncased": "bert-base-uncased",
    "bert-large-uncased": "bert-large-uncased",
    "scibert-uncased": "allenai/scibert_scivocab_uncased",
    "scibert-cased": "allenai/scibert_scivocab_cased",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune a transformer for token classification")
parser.add_argument("--model_key", type=str, required=True, choices=PRESETS.keys(),
                    help="Which preset model to fine-tune")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--data_file", type=str, default="CR_ECSS_dataset.json", help="Dataset JSON file")
args = parser.parse_args()

MODEL_KEY = args.model_key
model_name_or_path = PRESETS[MODEL_KEY]
epochs = args.epochs
batch_num = args.batch_size
seed_val = args.seed
data_file = args.data_file

fine_tuning_runs = 1
dataset = pd.read_json(data_file)

# AMP preferences
use_amp = True  # global switch for mixed precision

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def format_time(elapsed: float) -> str:
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def unique_path_with_suffix(base: Path) -> Path:
    """If base exists, append _1, _2, ... until a free path is found."""
    if not base.exists():
        return base
    i = 1
    while True:
        cand = base.with_name(f"{base.stem}_{i}{base.suffix}")
        if not cand.exists():
            return cand
        i += 1

def tokenize_and_align_labels(examples, labels, tokenizer, max_len=512, stride=0):
    """
    Align word-level labels to wordpiece tokens using FAST tokenizers (requires use_fast=True).
    """
    enc = tokenizer(
        examples,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_len,
        stride=stride if stride else 0,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )

    word_piece_labels = []
    label_all_tokens = True
    for i, label in enumerate(labels):
        word_ids = enc.word_ids(batch_index=i)  # only available for fast tokenizers
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        word_piece_labels.append(label_ids)

    enc["labels"] = word_piece_labels
    return enc

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)
set_seed(seed_val)

# For speed (not fully deterministic)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True  # enable if you need strict determinism (and set benchmark=False)

# ─────────────────────────────────────────────────────────────────────────────
# Device & AMP
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f"Device: {device} | GPUs: {n_gpu}")

# prefer bf16 if supported, else fp16
amp_dtype = torch.float16
if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16

def autocast_ctx(enabled: bool):
    if not (enabled and torch.cuda.is_available()):
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)  # PyTorch ≥ 2.0
    except AttributeError:
        return torch.cuda.amp.autocast(dtype=amp_dtype)

# new GradScaler API, with safe fallback for older torch
try:
    from torch.amp import GradScaler as _GradScaler
    scaler = _GradScaler("cuda", enabled=(use_amp and torch.cuda.is_available() and amp_dtype == torch.float16))
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available() and amp_dtype == torch.float16))

# ─────────────────────────────────────────────────────────────────────────────
# Label mapping
# ─────────────────────────────────────────────────────────────────────────────
tag_vals = dataset["labels"].unique()
tag2idx = {tag: i for i, tag in enumerate(tag_vals)}
tag2name = {v: k for k, v in tag2idx.items()}
tag2name[-100] = "None"

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
for run_idx in range(fine_tuning_runs):
    print(f"\n==================== Fine Tuning Round {run_idx + 1} ====================")

    # Timestamped naming
    ts = now_stamp()
    out_dir = Path("runs") / MODEL_KEY / f"finetuned_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_json_base = Path(f"train_results_{MODEL_KEY.replace('/', '__')}_{ts}.json")
    report_csv_base = Path(f"report_{MODEL_KEY.replace('/', '__')}_{ts}.csv")

    # Inspect config and set tokenizer/model flags
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    model_type = getattr(cfg, "model_type", "").lower()
    name_l = model_name_or_path.lower()
    is_modernbert = ("modernbert" in model_type) or ("modernbert" in name_l)

    extra_model_kwargs = {}
    if is_modernbert:
        extra_model_kwargs["reference_compile"] = False
        extra_model_kwargs["attn_implementation"] = "eager"

    # We MUST use FAST tokenizer (word_ids need it). Keep add_prefix_space for RoBERTa/ModernBERT.
    needs_prefix_space = (model_type in {"roberta"} or "roberta" in name_l or "modernbert" in name_l)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,                         # << required for word_ids()
        add_prefix_space=needs_prefix_space,   # needed for byte-level BPE
    )

    # Max length from tokenizer (fallback to 512)
    MAX_LEN = (
        tokenizer.model_max_length
        if (getattr(tokenizer, "model_max_length", None) and tokenizer.model_max_length < 10000)
        else 512
    )
    STRIDE = 0  # set >0 if you want sliding windows for very long sentences

    # Build sentences/labels per sentence_id
    sentence_ids = dataset["sentence_id"].unique()
    sentences = [[w for w in dataset[dataset["sentence_id"] == sid]["words"].values] for sid in sentence_ids]
    labels_list = [[tag2idx[lbl] for lbl in dataset[dataset["sentence_id"] == sid]["labels"].values] for sid in sentence_ids]

    encoded = tokenize_and_align_labels(sentences, labels_list, tokenizer, max_len=MAX_LEN, stride=STRIDE)
    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]
    labels = encoded["labels"]

    # Sample inspect (trim to non-pad tokens, show first ~80 tokens)
    for j in range(min(3, len(input_ids))):
        ids = input_ids[j]
        mask = attention_masks[j]
        L = int(np.sum(mask)) if hasattr(mask, "__len__") else len(ids)
        toks = tokenizer.convert_ids_to_tokens(ids[:L])
        print(f"No.{j}, len:{len(ids)}")
        print("texts:", " ".join(toks[:80]), "..." if L > 80 else "")
        lab_names = [tag2name.get(x, "UNK") for x in labels[j][:L]]
        print("labels:", " ".join(lab_names[:80]), "..." if L > 80 else "")

    # Split
    tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks = train_test_split(
        input_ids, labels, attention_masks, random_state=seed_val, test_size=0.213, shuffle=True
    )

    tr_inputs = torch.as_tensor(tr_inputs)
    val_inputs = torch.as_tensor(val_inputs)
    tr_tags = torch.as_tensor(tr_tags)
    val_tags = torch.as_tensor(val_tags)
    tr_masks = torch.as_tensor(tr_masks)
    val_masks = torch.as_tensor(val_masks)

    # Dataloaders (no workers → no fork → no tokenizer warning)
    num_workers = 0
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    valid_data = TensorDataset(val_inputs, val_masks, val_tags)

    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=batch_num,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_dataloader = DataLoader(
        valid_data,
        sampler=SequentialSampler(valid_data),
        batch_size=batch_num,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model with proper label maps
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(tag2idx),
        id2label={i: t for t, i in tag2idx.items()},
        label2id={t: i for t, i in tag2idx.items()},
        **extra_model_kwargs,
    ).to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Optim + sched
    def build_optimizer(model, lr=3e-5, wd=0.01):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": wd},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        return AdamW(optimizer_grouped_parameters, lr=lr)

    optimizer = build_optimizer(model, lr=3e-5, wd=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    training_stats = []
    train_loss_hist = []
    total_t0 = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # TRAIN / EVAL
    # ─────────────────────────────────────────────────────────────────────────
    for epoch_i in range(epochs):
        print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
        print("Training...")
        t0 = time.time()
        total_train_loss = 0.0

        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device, non_blocking=True)
            b_input_mask = batch[1].to(device, non_blocking=True)
            b_labels = batch[2].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # forward (no labels) + manual loss to avoid DP scalar gather
            with autocast_ctx(use_amp):
                out = model(b_input_ids, attention_mask=b_input_mask)
                logits = out.logits  # [B, T, C]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    b_labels.view(-1),
                    ignore_index=-100,
                )

            if hasattr(scaler, "is_enabled") and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_train_loss += float(loss.detach())

        avg_train_loss = total_train_loss / max(1, len(train_dataloader))
        training_time = format_time(time.time() - t0)
        print(f"\n  Average training loss: {avg_train_loss:.4f}")
        print(f"  Training epoch took: {training_time}")
        train_loss_hist.append(avg_train_loss)

        # ----------------------------
        # Validation
        # ----------------------------
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()

        total_eval_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in valid_dataloader:
                b_input_ids = batch[0].to(device, non_blocking=True)
                b_input_mask = batch[1].to(device, non_blocking=True)
                b_labels = batch[2].to(device, non_blocking=True)

                out = model(b_input_ids, attention_mask=b_input_mask)
                logits = out.logits
                vloss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    b_labels.view(-1),
                    ignore_index=-100,
                )
                total_eval_loss += float(vloss.detach())

                # predictions
                preds = torch.argmax(logits, dim=2)  # [B, T]
                preds = preds.detach().cpu().numpy()
                label_ids = b_labels.detach().cpu().numpy()
                input_mask = b_input_mask.detach().cpu().numpy()

                for i_m, m in enumerate(input_mask):
                    t1, t2 = [], []
                    for j, flag in enumerate(m):
                        if flag:
                            if tag2name.get(label_ids[i_m][j], "None") != "None":
                                t1.append(tag2name[label_ids[i_m][j]])
                                t2.append(tag2name[preds[i_m][j]])
                        else:
                            break
                    y_true.append(t1)
                    y_pred.append(t2)

        # Flatten & metrics
        y_true_words = [w for sent in y_true for w in sent]
        y_pred_words = [w for sent in y_pred for w in sent]

        labels_for_scores = [lab for lab in set(y_true_words) if lab != "O"]
        if labels_for_scores:
            pr, rc, f1s, support = precision_recall_fscore_support(
                y_true_words, y_pred_words, labels=labels_for_scores, zero_division=0
            )
            f1_scores = {lab: float(f1s[i]) for i, lab in enumerate(labels_for_scores)}
            examples = {lab: int(support[i]) for i, lab in enumerate(labels_for_scores)}
            f1_scores["weighted"] = float(
                f1_score(y_true_words, y_pred_words, average="weighted", labels=labels_for_scores, zero_division=0)
            )
            examples["sum"] = int(np.sum([examples[k] for k in examples.keys()]))
        else:
            f1_scores = {"weighted": 0.0}
            examples = {"sum": 0}

        avg_val_loss = total_eval_loss / max(1, len(valid_dataloader))
        print(f"  F1_score (weighted): {f1_scores['weighted']:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        validation_time = format_time(time.time() - t0)
        print(f"  Validation took: {validation_time}")

        # persist per-epoch stats (JSON lines-like in a list)
        stat_row = {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "F1 score": f1_scores["weighted"],
            "examples_sum": examples["sum"],
            "Label_F1_scores": f1_scores,
            "examples": examples,
            "Training Time": training_time,
            "Validation Time": validation_time,
            "model_key": MODEL_KEY,
            "timestamp": ts,
        }
        if epoch_i == 0:
            training_stats = [stat_row]
        else:
            training_stats.append(stat_row)

        with open(unique_path_with_suffix(train_json_base), "w+", encoding="utf-8") as file:
            pd.DataFrame(training_stats).to_json(file, orient="records", force_ascii=False)

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # Final classification report
    y_true_words = [w for sent in y_true for w in sent]
    y_pred_words = [w for sent in y_pred for w in sent]
    report_dict = classification_report(
        y_true_words,
        y_pred_words,
        digits=3,
        labels=[label for label in set(y_true_words) if label != "O"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = unique_path_with_suffix(report_csv_base)
    report_df.to_csv(report_path, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    # Save final model & tokenizer alongside results
    save_dir = out_dir
    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model & tokenizer saved to: {save_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# Filenames (summary echo)
# ─────────────────────────────────────────────────────────────────────────────
ts = now_stamp()
safe_key = MODEL_KEY.replace("/", "__")
train_json = unique_path_with_suffix(Path(f"train_results_{safe_key}_{ts}.json"))
report_csv = unique_path_with_suffix(Path(f"report_{safe_key}_{ts}.csv"))
save_dir = Path("runs") / MODEL_KEY / f"finetuned_{ts}"
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Training {MODEL_KEY} → results: {train_json}, report: {report_csv}, model dir: {save_dir}")
