import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import shutil
from pathlib import Path
import argparse
import math
import inspect
import json
import time

from datasets import Dataset, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    pipeline,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import transformers
import torch

# ----------------------------
# Presets
# ----------------------------
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

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        "MLM trainer (DeBERTa / RoBERTa / ModernBERT / BERT / SciBERT) with resume + perplexity + CSV logs + best-save."
    )
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="deberta-v3-large",
                   help="Model preset to use.")
    p.add_argument("--model-name", default=None, help="HF model id (overrides --preset).")
    p.add_argument("--train-epochs", type=int, default=10, help="Number of training epochs.")
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="Accumulate gradients across this many batches before an optimizer step.")
    p.add_argument("--max-length", type=int, default=512,
                   help="Maximum sequence length for tokenization (pad/truncate to this).")
    p.add_argument("--overwrite-run-dir", action="store_true",
                   help="If set, delete the existing run dir before starting.")
    p.add_argument("--resume-from", default="auto",
                   help="Resume from checkpoint: 'auto' (default), 'none', or a path to checkpoint-* directory.")
    p.add_argument("--save-total-limit", type=int, default=3,
                   help="Max number of checkpoint-* folders to keep in output_dir. Default: 3.")
    p.add_argument("--data-file", default="Sentences_WikiBooksAbstracts.txt",
                   help="Path to a newline-delimited text file.")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="Fraction of lines used for validation (0.0..0.5).")
    p.add_argument("--early-stop-patience", type=int, default=2,
                   help="Early stopping patience in eval epochs. Use 0 to disable early stopping.")
    p.add_argument("--skip-final-save", action="store_true",
                   help="If set, do not export a 'final' snapshot (use 'best' and checkpoints only).")
    p.add_argument("--log-every", type=int, default=200,
                   help="Trainer logging_steps value.")
    return p.parse_args()

args = parse_args()

# ----------------------------
# Resolve model & flags
# ----------------------------
model_name = args.model_name or PRESETS[args.preset]
train_epochs = args.train_epochs
per_dev_bs = args.per_device_train_batch_size
grad_acc = max(1, args.gradient_accumulation_steps)
max_length = args.max_length
val_ratio = min(max(args.val_ratio, 0.0), 0.5)
early_stop_patience = max(0, args.early_stop_patience)

name_l = model_name.lower()
IS_ROBERTA = "roberta" in name_l
IS_DEBERTA = "deberta" in name_l
IS_MODERNBERT = "modernbert" in name_l
IS_SCIBERT = "scibert" in name_l
IS_BERT = ("bert" in name_l) and not (IS_ROBERTA or IS_DEBERTA or IS_MODERNBERT or IS_SCIBERT)

# ----------------------------
# Rank/World
# ----------------------------
is_rank0 = os.environ.get("LOCAL_RANK") in (None, "0") and os.environ.get("RANK", "0") == "0"
world_size = int(os.environ.get("WORLD_SIZE", "1"))

# datasets tqdm control
try:
    from datasets.utils.logging import disable_progress_bar, enable_progress_bar
    (enable_progress_bar if is_rank0 else disable_progress_bar)()
except Exception:
    pass

# ----------------------------
# Run folder
# ----------------------------
def slug(mid: str) -> str:
    return mid.replace("/", "__").replace(":", "-").replace(" ", "_")

run_dir = Path("runs") / slug(model_name)
if args.overwrite_run_dir and is_rank0 and run_dir.exists():
    shutil.rmtree(run_dir)
run_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Tokenizer
# ----------------------------
def load_tokenizer(name: str):
    if IS_DEBERTA:
        try:
            return AutoTokenizer.from_pretrained(name, use_fast=False)
        except Exception:
            if is_rank0:
                print("note: sentencepiece not found for slow tokenizer; using fast tokenizer. (pip install sentencepiece)")
            return AutoTokenizer.from_pretrained(name, use_fast=True)
    try:
        return AutoTokenizer.from_pretrained(name, use_fast=True)
    except ImportError as e:
        msg = str(e).lower()
        if "protobuf" in msg or "sentencepiece" in msg:
            if is_rank0:
                print("note: fast tokenizer deps missing; falling back to slow. (pip install 'protobuf==4.25.3' sentencepiece)")
            return AutoTokenizer.from_pretrained(name, use_fast=False)
        raise

tokenizer = load_tokenizer(model_name)

# ----------------------------
# Model + grad checkpointing
# ----------------------------
model = AutoModelForMaskedLM.from_pretrained(model_name)
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

GC_MODE = "off"
if hasattr(model, "gradient_checkpointing_enable"):
    if IS_DEBERTA:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            GC_MODE = "non-reentrant"
        except TypeError:
            GC_MODE = "disabled (transformers<4.36)"
    else:
        try:
            model.gradient_checkpointing_enable()
            GC_MODE = "default"
        except Exception:
            GC_MODE = "off"

def _count_params(m):
    t = sum(p.numel() for p in m.parameters())
    t_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return t, t_train

# ----------------------------
# Environment & config summary
# ----------------------------
if is_rank0:
    fam = (
        "DeBERTa" if IS_DEBERTA else
        "RoBERTa" if IS_ROBERTA else
        "ModernBERT" if IS_MODERNBERT else
        "SciBERT" if IS_SCIBERT else
        "BERT" if IS_BERT else
        "Other"
    )
    total, trainable = _count_params(model)
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            devices.append(f"cuda:{i} {prop.name} {prop.total_memory // (1024**3)}GB")
    else:
        devices.append("cpu")

    print("=== TRAIN CONFIG ===")
    print(f"family: {fam}")
    print(f"model: {model_name}")
    print(f"transformers: {transformers.__version__}   torch: {torch.__version__}")
    print(f"devices: {', '.join(devices)}   world_size: {world_size}")
    print(f"run_dir: {run_dir}")
    print(f"epochs: {train_epochs}   per_device_bs: {per_dev_bs}   grad_acc: {grad_acc}")
    print(f"effective_global_bs: {per_dev_bs * grad_acc * world_size}")
    print(f"max_length: {max_length}   mask_token: {tokenizer.mask_token}")
    print(f"grad_checkpointing: {GC_MODE}")
    print(f"early_stop_patience: {early_stop_patience} (0 disables)")
    print("====================")

# Optional demo (rank-0 only)
try:
    fm = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    if is_rank0:
        print(f"demo: {fm(f'This {tokenizer.mask_token} works with {model_name}.')}")
except Exception as e:
    if is_rank0:
        print(f"note: fill-mask demo skipped: {e}")

# Persist initial state (useful for reproducibility)
model.save_pretrained(run_dir)
tokenizer.save_pretrained(run_dir)
if is_rank0:
    print(f"saved initial model+tokenizer to: {run_dir}")

# ----------------------------
# Data (uses --max-length) + Train/Val split
# ----------------------------
file_path = os.path.abspath(args.data_file)
if not Path(file_path).exists():
    raise FileNotFoundError(f"--data-file not found: {file_path}")

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

texts = [line.strip() for line in lines if line.strip()]
if is_rank0:
    print(f"data: {len(texts)} non-empty lines from {file_path}")

features = Features({"text": Value("string")})
full_ds = Dataset.from_dict({"text": texts}, features=features)

if val_ratio > 0.0:
    split = full_ds.train_test_split(test_size=val_ratio, seed=42, shuffle=True)
    train_ds, eval_ds = split["train"], split["test"]
else:
    train_ds, eval_ds = full_ds, full_ds

def tok_fn(batch):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=max_length
    )

train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
eval_tok  = eval_ds.map(tok_fn,  batched=True, remove_columns=["text"])

if is_rank0:
    print(f"dataset: train={len(train_tok)}  eval={len(eval_tok)}")

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# ----------------------------
# Callbacks (CSV logs, epoch summaries, per-epoch eval/save, early stop, best save)
# ----------------------------
def _safe_exp(x):
    try:
        return float(math.exp(float(x)))
    except Exception:
        return ""

class CSVLogger(TrainerCallback):
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.header = ["step", "epoch", "loss", "train_ppl", "learning_rate", "eval_loss", "eval_ppl"]
        self.header_written = False
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero: return
        train_loss = logs.get("loss", "")
        eval_loss = logs.get("eval_loss", "")
        row = {
            "step": state.global_step,
            "epoch": getattr(state, "epoch", None),
            "loss": train_loss,
            "train_ppl": _safe_exp(train_loss) if train_loss != "" else "",
            "learning_rate": logs.get("learning_rate", ""),
            "eval_loss": eval_loss,
            "eval_ppl": _safe_exp(eval_loss) if eval_loss != "" else "",
        }
        import csv as _csv
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=self.header)
            if not self.header_written:
                w.writeheader(); self.header_written = True
            w.writerow(row)

class EpochLossLogger(TrainerCallback):
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.header = ["epoch", "last_train_loss", "train_ppl", "eval_loss", "eval_ppl", "global_step"]
        self.header_written = False
        self.last_train_loss = None
        self.last_eval_loss = None
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self.last_eval_loss = metrics["eval_loss"]
            if hasattr(state, "is_world_process_zero") and state.is_world_process_zero:
                import csv as _csv
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=self.header)
                    if not self.header_written:
                        w.writeheader(); self.header_written = True
                    w.writerow({
                        "epoch": int(state.epoch) if state.epoch is not None else "",
                        "last_train_loss": self.last_train_loss if self.last_train_loss is not None else "",
                        "train_ppl": _safe_exp(self.last_train_loss) if self.last_train_loss is not None else "",
                        "eval_loss": self.last_eval_loss if self.last_eval_loss is not None else "",
                        "eval_ppl": _safe_exp(self.last_eval_loss) if self.last_eval_loss is not None else "",
                        "global_step": state.global_step,
                    })
        return control

class PerEpochTqdm(TrainerCallback):
    """Progress bar that advances on optimizer steps (respects grad accumulation) and forces one eval per epoch."""
    def __init__(self, opt_steps_per_epoch: int, num_epochs: int, is_rank0: bool):
        self.opt_steps_per_epoch = max(1, opt_steps_per_epoch)
        self.num_epochs = num_epochs
        self.is_rank0 = is_rank0
        self.pbar = None
        self._last_global_step = None
    def on_epoch_begin(self, args, state, control, **kwargs):
        if not self.is_rank0: return
        from tqdm.auto import tqdm
        cur_epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        self.pbar = tqdm(total=self.opt_steps_per_epoch,
                         desc=f"Epoch {cur_epoch}/{int(args.num_train_epochs)} (optimizer steps)",
                         leave=False)
        self._last_global_step = state.global_step
    def on_step_end(self, args, state, control, **kwargs):
        if not self.is_rank0 or self.pbar is None:
            return control
        if self._last_global_step is None:
            self._last_global_step = state.global_step
        if state.global_step != self._last_global_step:
            self.pbar.update(1)
            self._last_global_step = state.global_step
        return control
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.is_rank0 and self.pbar is not None:
            self.pbar.close(); self.pbar = None
        control.should_evaluate = True
        return control

class SimpleEarlyStop(TrainerCallback):
    def __init__(self, patience=2, tol=1e-6):
        self.patience, self.tol = patience, tol
        self.best, self.bad = float("inf"), 0
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics: return control
        cur = metrics["eval_loss"]
        if cur < self.best - self.tol:
            self.best, self.bad = cur, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                control.should_training_stop = True
        return control

class BestCheckpointSaver(TrainerCallback):
    def __init__(self, base_dir: Path, model, tokenizer, metric="eval_loss", mode="min", tol=1e-6):
        self.metric, self.mode, self.tol = metric, mode, tol
        self.out_dir = Path(base_dir) / "best"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.tokenizer = tokenizer

        # load previous best if present
        self.meta_path = self.out_dir / "best_meta.json"
        self.best_val = None
        try:
            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if meta.get("metric") == self.metric:
                        self.best_val = float(meta.get("value"))
        except Exception:
            self.best_val = None

    def _is_improved(self, cur: float) -> bool:
        if self.best_val is None:
            return True
        return (cur < self.best_val - self.tol) if self.mode == "min" else (cur > self.best_val + self.tol)

    def _write_meta(self, cur: float, state):
        meta = {
            "metric": self.metric,
            "value": float(cur),
            "epoch": int(state.epoch) if getattr(state, "epoch", None) is not None else None,
            "global_step": int(state.global_step) if hasattr(state, "global_step") else None,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric not in metrics:
            return control
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return control

        cur = float(metrics[self.metric])
        if self._is_improved(cur):
            mdl = self.model.module if hasattr(self.model, "module") else self.model
            mdl.save_pretrained(self.out_dir)
            self.tokenizer.save_pretrained(self.out_dir)
            self.best_val = cur
            self._write_meta(cur, state)
            if is_rank0:
                print(f"saved BEST ({self.metric}={cur:.6f}) -> {self.out_dir}")
        return control

# ----------------------------
# TrainingArguments (per-epoch save/eval where supported)
# ----------------------------
TA_SIG = inspect.signature(TrainingArguments.__init__)
ACCEPTED_TA_KEYS = set(TA_SIG.parameters.keys())
def TARGS(**kwargs):
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in ACCEPTED_TA_KEYS})

supports_epoch_strat = "evaluation_strategy" in ACCEPTED_TA_KEYS and "save_strategy" in ACCEPTED_TA_KEYS

base_kwargs = dict(
    output_dir=str(run_dir),
    overwrite_output_dir=True,
    num_train_epochs=train_epochs,
    per_device_train_batch_size=per_dev_bs,
    gradient_accumulation_steps=grad_acc,
    dataloader_num_workers=max(1, os.cpu_count() // 4),
    logging_steps=max(1, args.log_every),
)

epoch_kwargs = dict(
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=args.save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

legacy_kwargs = dict(
    evaluate_during_training=True,
    save_steps=10_000,
    eval_steps=10_000,
    save_total_limit=args.save_total_limit,
)

training_args = TARGS(**base_kwargs, **(epoch_kwargs if supports_epoch_strat else legacy_kwargs))

# ----------------------------
# Info: steps per epoch
# ----------------------------
dummy_trainer = Trainer(model=model, args=training_args, train_dataset=train_tok, data_collator=collator)
batches_per_epoch = max(1, len(dummy_trainer.get_train_dataloader()))
opt_steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_acc))

if is_rank0:
    print(f"batches_per_epoch(per process): {batches_per_epoch}")
    print(f"optimizer_steps_per_epoch(with grad_acc={grad_acc}): {opt_steps_per_epoch}")

# ----------------------------
# Build trainer + callbacks
# ----------------------------
csv_log_file = run_dir / "training_log.csv"
epoch_log_file = run_dir / "epoch_log.csv"

class ForceEpochEvalSave(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kw):
        control.should_evaluate = True
        control.should_save = True
        return control

callbacks = [
    CSVLogger(csv_log_file),
    EpochLossLogger(epoch_log_file),
    BestCheckpointSaver(run_dir, model, tokenizer),
    PerEpochTqdm(opt_steps_per_epoch, train_epochs, is_rank0),
]
if early_stop_patience > 0:
    callbacks.append(SimpleEarlyStop(patience=early_stop_patience, tol=1e-6))
if not supports_epoch_strat:
    callbacks.append(ForceEpochEvalSave())

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=collator,
    callbacks=callbacks,
)

# ----------------------------
# Resume logic
# ----------------------------
resume_ckpt = None
resume_arg = (args.resume_from or "auto").lower()
if resume_arg == "auto":
    try:
        last = get_last_checkpoint(str(run_dir))
        if last is not None:
            resume_ckpt = last
            if is_rank0:
                print(f"resume: auto -> {resume_ckpt}")
    except Exception:
        resume_ckpt = None
elif resume_arg != "none":
    ckpt_path = Path(resume_arg)
    if ckpt_path.exists():
        resume_ckpt = str(ckpt_path)
        if is_rank0:
            print(f"resume: from path -> {resume_ckpt}")
    else:
        if is_rank0:
            print(f"warn: --resume-from not found: {ckpt_path}; starting fresh")

# ----------------------------
# Train
# ----------------------------
try:
    trainer.train(resume_from_checkpoint=resume_ckpt)
finally:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

# ----------------------------
# Final save
# ----------------------------
# When load_best_model_at_end=True (supported installs), the in-memory model is the best one.
# You already have:
#   - checkpoints/ (epoch-wise)
#   - runs/.../best/ (explicit best snapshot)
# A 'final' export is optional; it clones the currently loaded (best) weights into runs/.../final/
if is_rank0 and not args.skip_final_save:
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    try:
        trainer.save_model(str(final_dir))
    except Exception:
        mdl = model.module if hasattr(model, "module") else model
        mdl.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved FINAL -> {final_dir}")
    print(f"logs: steps={csv_log_file}  epochs={epoch_log_file}")
elif is_rank0 and args.skip_final_save:
    print("final save skipped (use --skip-final-save to control).")
    print(f"best dir: {run_dir / 'best'}")
    print(f"logs: steps={csv_log_file}  epochs={epoch_log_file}")
