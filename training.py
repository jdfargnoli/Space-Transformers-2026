import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import shutil
from pathlib import Path
import argparse
import math

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
from packaging import version
import transformers
import torch

# ----------------------------
# Presets & CLI (DEFAULT = DeBERTa v3 LARGE)
# ----------------------------
PRESETS = {
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "roberta-large": "roberta-large",
    "roberta-base": "roberta-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
    "modernbert-base": "answerdotai/ModernBERT-base",
}

def parse_args():
    p = argparse.ArgumentParser(
        "MLM trainer (DeBERTa / RoBERTa / ModernBERT) with CSV logs + best-save."
    )
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="deberta-v3-large",
                   help="Model preset to use.")
    p.add_argument("--model-name", default=None, help="HF model id (overrides --preset).")
    p.add_argument("--train-epochs", type=int, default=10)
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="Accumulate gradients across this many batches before an optimizer step.")
    p.add_argument("--max-length", type=int, default=512,
                   help="Maximum sequence length for tokenization (pad/truncate to this).")
    p.add_argument("--overwrite-run-dir", action="store_true")
    p.add_argument("--data-file", default="Sentences_WikiBooksAbstracts.txt")
    return p.parse_args()

args = parse_args()
model_name = args.model_name or PRESETS[args.preset]
train_epochs = args.train_epochs
per_dev_bs = args.per_device_train_batch_size
grad_acc = max(1, args.gradient_accumulation_steps)
max_length = args.max_length

IS_DEBERTA = "deberta" in model_name.lower()
IS_ROBERTA = "roberta" in model_name.lower()
IS_MODERNBERT = "modernbert" in model_name.lower()

# ----------------------------
# Rank-0 helper & datasets tqdm control
# ----------------------------
is_rank0 = os.environ.get("LOCAL_RANK") in (None, "0") and os.environ.get("RANK", "0") == "0"
world_size = int(os.environ.get("WORLD_SIZE", "1"))
try:
    from datasets.utils.logging import disable_progress_bar, enable_progress_bar
    (enable_progress_bar if is_rank0 else disable_progress_bar)()
except Exception:
    pass

# ----------------------------
# Dynamic run folder from model id
# ----------------------------
def slug(mid: str) -> str:
    return mid.replace("/", "__").replace(":", "-").replace(" ", "_")

run_dir = Path("runs") / slug(model_name)
if args.overwrite_run_dir and is_rank0 and run_dir.exists():
    shutil.rmtree(run_dir)
run_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Tokenizer (DeBERTa → prefer slow to avoid byte-fallback warning)
# ----------------------------
def load_tokenizer(name: str):
    if IS_DEBERTA:
        try:
            return AutoTokenizer.from_pretrained(name, use_fast=False)  # needs sentencepiece
        except Exception:
            if is_rank0:
                print("⚠️ sentencepiece not found for slow tokenizer; using fast tokenizer. (pip install sentencepiece)")
            return AutoTokenizer.from_pretrained(name, use_fast=True)
    # RoBERTa / ModernBERT
    try:
        return AutoTokenizer.from_pretrained(name, use_fast=True)
    except ImportError as e:
        msg = str(e).lower()
        if "protobuf" in msg or "sentencepiece" in msg:
            if is_rank0:
                print("⚠️ Fast tokenizer deps missing; falling back to slow. (pip install 'protobuf==4.25.3' sentencepiece)")
            return AutoTokenizer.from_pretrained(name, use_fast=False)
        raise

tokenizer = load_tokenizer(model_name)

# ----------------------------
# Model + safe gradient checkpointing
# ----------------------------
model = AutoModelForMaskedLM.from_pretrained(model_name)
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

GC_MODE = "off"
if hasattr(model, "gradient_checkpointing_enable"):
    if IS_DEBERTA:
        try:
            # Prefer non-reentrant (Transformers >= 4.36)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            GC_MODE = "non-reentrant"
        except TypeError:
            # Older Transformers → leave GC off for DeBERTa to avoid DDP/AMP issues
            GC_MODE = "disabled (transformers<4.36)"
    else:
        try:
            model.gradient_checkpointing_enable()
            GC_MODE = "default"
        except Exception:
            GC_MODE = "off"

if is_rank0:
    fam = "DeBERTa" if IS_DEBERTA else ("RoBERTa" if IS_ROBERTA else ("ModernBERT" if IS_MODERNBERT else "Other"))
    eff_global_bs = per_dev_bs * grad_acc * world_size
    print(f"Family: {fam}")
    print(f"Resolved model: {model_name}")
    print(f"Epochs: {train_epochs}")
    print(f"Run folder: {run_dir}")
    print(f"WORLD_SIZE (GPUs): {world_size}")
    print(f"Per-device batch size: {per_dev_bs}")
    print(f"Grad accumulation steps: {grad_acc}")
    print(f"Effective global batch size: {eff_global_bs}")
    print(f"Max seq length: {max_length}")
    print(f"Mask token: {tokenizer.mask_token}")
    print(f"Gradient checkpointing: {GC_MODE}")

# Optional demo (rank-0 only)
try:
    fm = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    if is_rank0:
        print(fm(f"This {tokenizer.mask_token} works with {model_name}."))
except Exception as e:
    if is_rank0:
        print(f"[fill-mask demo skipped] {e}")

# Initial save
model.save_pretrained(run_dir)
tokenizer.save_pretrained(run_dir)
if is_rank0:
    print(f"Initial save to: {run_dir}")

# ----------------------------
# Data (uses --max-length)
# ----------------------------
file_path = os.path.abspath(args.data_file)
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

data = {"text": [line.strip() for line in lines]}
features = Features({"text": Value("string")})
dataset = Dataset.from_dict(data, features=features)

tokenized_dataset = dataset.map(
    lambda e: tokenizer(
        e["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ),
    batched=True
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# ----------------------------
# Callbacks (CSV logs, epoch summaries, progress bar, early stop, best save)
# ----------------------------
class CSVLogger(TrainerCallback):
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.header = ["step", "epoch", "loss", "learning_rate", "eval_loss"]
        self.header_written = False
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero: return
        row = {
            "step": state.global_step,
            "epoch": getattr(state, "epoch", None),
            "loss": logs.get("loss", ""),
            "learning_rate": logs.get("learning_rate", ""),
            "eval_loss": logs.get("eval_loss", ""),
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
        self.header = ["epoch", "last_train_loss", "eval_loss", "global_step"]
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
                        "eval_loss": self.last_eval_loss if self.last_eval_loss is not None else "",
                        "global_step": state.global_step,
                    })
        return control

class PerEpochTqdm(TrainerCallback):
    """
    Progress bar that advances on optimizer steps (respects gradient accumulation),
    and forces one eval per epoch.
    """
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
        # Only advance when an optimizer step has occurred (global_step changed)
        if self._last_global_step is None:
            self._last_global_step = state.global_step
        if state.global_step != self._last_global_step:
            self.pbar.update(1)
            self._last_global_step = state.global_step
        return control
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.is_rank0 and self.pbar is not None:
            self.pbar.close(); self.pbar = None
        control.should_evaluate = True  # one eval per epoch
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
        self.best_val = None
        self.out_dir = Path(base_dir) / "best"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.tokenizer = tokenizer
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric not in metrics: return control
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero: return control
        cur = metrics[self.metric]
        improved = (self.best_val is None) or (cur < self.best_val - self.tol) if self.mode == "min" else (cur > self.best_val + self.tol)
        if improved:
            self.best_val = cur
            mdl = self.model.module if hasattr(self.model, "module") else self.model
            mdl.save_pretrained(self.out_dir)
            self.tokenizer.save_pretrained(self.out_dir)
            if is_rank0:
                print(f"💾 Saved new BEST ({self.metric}={cur:.6f}) to {self.out_dir}")
        return control

# ----------------------------
# TrainingArguments (add gradient accumulation)
# ----------------------------
training_args = TrainingArguments(
    output_dir=str(run_dir),
    overwrite_output_dir=True,
    num_train_epochs=train_epochs,
    per_device_train_batch_size=per_dev_bs,
    gradient_accumulation_steps=grad_acc,   # <<<
    save_steps=5_000,
    save_total_limit=2,
    logging_steps=2000,
    dataloader_num_workers=max(1, os.cpu_count() // 4),
    # evaluation/saving cadence handled by callbacks
)

# ----------------------------
# Info: steps per epoch
# ----------------------------
# Local dataloader length (per process)
dummy_trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, data_collator=collator)
batches_per_epoch = max(1, len(dummy_trainer.get_train_dataloader()))
opt_steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_acc))

if is_rank0:
    print(f"Batches per epoch (per process): {batches_per_epoch}")
    print(f"Optimizer steps per epoch (with grad_acc={grad_acc}): {opt_steps_per_epoch}")

# ----------------------------
# Trainer
# ----------------------------
csv_log_file = run_dir / "training_log.csv"
epoch_log_file = run_dir / "epoch_log.csv"

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # per your setup
    data_collator=collator,
    callbacks=[
        CSVLogger(csv_log_file),
        EpochLossLogger(epoch_log_file),
        SimpleEarlyStop(patience=2, tol=1e-6),
        BestCheckpointSaver(run_dir, model, tokenizer),
    ],
)

# Progress bar in optimizer steps
trainer.add_callback(PerEpochTqdm(opt_steps_per_epoch, train_epochs, is_rank0))

trainer.train()

# Final snapshot (rank-0)
if is_rank0:
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    try:
        trainer.save_model(str(final_dir))
    except Exception:
        mdl = model.module if hasattr(model, "module") else model
        mdl.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ Saved FINAL to: {final_dir}")
    print(f"Step CSV:  {csv_log_file}")
    print(f"Epoch CSV: {epoch_log_file}")
