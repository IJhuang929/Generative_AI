"""Fine-tuning script for hallucination type classification.

Trains a causal LM (with PEFT/LoRA) on the 5-class hallucination dataset
defined in ``classes.json``. Inputs are built via the chat-template
prompt in ``prompt_template.format_chat_messages``; supervision is the
full token sequence (standard causal-LM next-token prediction).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

# WeightedCETrainer reads outputs.logits in compute_loss; from unsloth
# 2024.11+ logits are returned empty unless this env var is set BEFORE
# the unsloth import below.
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import numpy as np
import pandas as pd
from unsloth import FastLanguageModel  # type: ignore[import-not-found]
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from prompt_template import format_chat_messages, parse_model_output
from preprocess import load_split_with_evidence
from pdf_parser import release_embedder

NUM_LABELS = 5


def compute_class_weights(label_series: pd.Series) -> torch.Tensor:
    """Compute balanced class weights for the 5-way label distribution.

    Uses scikit-learn's ``compute_class_weight(class_weight="balanced", ...)``,
    which assigns weight ``n_samples / (n_classes * n_samples_in_class)`` to
    each class. Rare classes (e.g. label 4 ≈ 2.3% of train) end up with
    weights well above 1.0, while majority classes get weights below 1.0.

    Args:
        label_series: A pandas Series of integer labels in ``{0, 1, 2, 3, 4}``.

    Returns:
        A ``torch.float32`` tensor of shape ``(5,)`` indexed by class id.
        Class ids absent from ``label_series`` get weight ``1.0``.
    """
    classes_present = np.array(sorted(label_series.unique()))
    weights_present = compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=label_series.to_numpy(),
    )
    full = np.ones(NUM_LABELS, dtype=np.float32)
    for cls, w in zip(classes_present, weights_present):
        full[int(cls)] = float(w)

    print("=" * 50)
    print("Class weights (balanced):")
    for cls in range(NUM_LABELS):
        marker = "" if cls in classes_present else "  [missing → 1.0]"
        print(f"  Label {cls}: {full[cls]:.4f}{marker}")
    print("=" * 50)
    return torch.tensor(full, dtype=torch.float32)


def oversample_to_median(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """將每個類別的樣本數過採樣到中位數，避免多數類膨脹太多。

    策略：取 5 個類別樣本數的中位數作為目標數量。
    樣本數低於中位數的類別做 replacement=True 的 resample。
    樣本數高於中位數的類別保持不變（不做 downsample）。

    Args:
        df: 包含 label 欄位的 DataFrame
        random_state: 隨機種子

    Returns:
        過採樣後 shuffle 的 DataFrame
    """
    from sklearn.utils import resample

    counts = df["label"].value_counts()
    all_counts = pd.Series(0, index=range(NUM_LABELS))
    all_counts.update(counts)
    target = int(np.median(all_counts[all_counts > 0].values))

    print(f"[oversample] target count per class (median): {target}")
    parts = []
    for label in range(NUM_LABELS):
        subset = df[df["label"] == label]
        if len(subset) == 0:
            print(f"  Label {label}: 0 samples, skipping")
            continue
        if len(subset) < target:
            subset = resample(
                subset, replace=True,
                n_samples=target, random_state=random_state
            )
            print(f"  Label {label}: {counts.get(label, 0)} → {target} (upsampled)")
        else:
            print(f"  Label {label}: {len(subset)} (kept)")
        parts.append(subset)

    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


class HallucinationDataset(TorchDataset):
    """Causal-LM dataset for hallucination classification.

    Each item is a chat-formatted prompt ending with the assistant's class
    name. Only the assistant answer tokens are supervised — every prompt
    token (system prompt, instructions, evidence, review sentence) is masked
    to ``-100`` so the model is trained on producing the *answer*, not on
    reconstructing the prompt itself.

    Notes:
        - Long sequences are right-truncated to ``max_length`` and padded
          with ``tokenizer.pad_token_id`` on the right.
        - Pad positions in ``labels`` are set to ``-100``.
        - Prompt positions in ``labels`` are set to ``-100``; only assistant
          answer tokens contribute to the cross-entropy loss.
    """

    _ID2NAME = {
        0: "Attribution Failure",
        1: "Entity",
        2: "Number",
        3: "Overgeneralization",
        4: "Temporal",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 700,  # evidence is 1000
    ) -> None:
        """Initialize the dataset.

        Args:
            df: DataFrame with columns ``text``, ``evidence``, ``label``.
            tokenizer: A HuggingFace tokenizer with a configured chat template
                and a ``pad_token`` (set caller-side if missing).
            max_length: Maximum tokenized sequence length. Sequences longer
                than this are right-truncated.
        """
        required = {"text", "evidence", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df is missing required columns: {sorted(missing)}")

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id is None — set tokenizer.pad_token "
                "(e.g. = tokenizer.eos_token) before constructing the dataset."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row["text"])
        evidence = str(row["evidence"])
        label_id = int(row["label"])

        base_messages = format_chat_messages(text, evidence)
        full_messages = base_messages + [
            {"role": "assistant", "content": self._ID2NAME[label_id]}
        ]

        prompt_ids = self.tokenizer.apply_chat_template(
            base_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]
        attn_len = len(full_ids)
        pad_n = self.max_length - attn_len
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = torch.tensor(full_ids + [pad_id] * pad_n, dtype=torch.long)
        attention_mask = torch.tensor([1] * attn_len + [0] * pad_n, dtype=torch.long)

        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), attn_len)
        labels[:prompt_len] = -100              # mask prompt: train only on the answer
        labels[attention_mask == 0] = -100      # mask pad

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "class_label": torch.tensor(label_id, dtype=torch.long),
        }


CLASS_NAMES: tuple[str, ...] = (
    "Attribution Failure",
    "Entity",
    "Number",
    "Overgeneralization",
    "Temporal",
)


class WeightedCETrainer(Trainer):
    """``Trainer`` that scales each sample's CE loss by its class weight.

    Because :class:`HallucinationDataset` masks every prompt token to
    ``-100``, the model's per-sample CE is already restricted to the few
    answer tokens — so weighting at the *sample* level is sufficient and
    much simpler than the previous per-token / last-occurrence scheme.
    For sample ``b`` of class ``c``::

        sample_loss[b] = mean( CE(answer tokens) )           # ignore_index=-100
        batch_loss     = mean_b ( sample_loss[b] * w[c_b] )

    where ``w`` is the balanced class-weight tensor returned by
    :func:`compute_class_weights`.

    Notes:
        - A per-sample CE loop avoids materializing the ``(B, T-1, V)``
          flattened logits, which OOMs T4 16GB at ``max_length=768`` with
          Qwen2.5's ~152k vocab.
        - Samples whose answer was truncated away (no valid label tokens)
          contribute zero loss and are excluded from the mean.
    """

    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor,
        oversample: bool = False,
        oversample_alpha: float = 1.0,
        sample_class_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the trainer with optional class-balanced oversampling.

        Args:
            class_weights: Per-class weight tensor of shape ``(NUM_LABELS,)``
                applied at the loss level. Pass all-ones to disable.
            oversample: If True, override the default sampler with a
                :class:`WeightedRandomSampler` that draws each sample with
                probability proportional to ``1 / count(class) ** alpha``.
            oversample_alpha: Strength of the inverse-frequency weighting.
                ``0.0`` = natural distribution, ``1.0`` = exactly balanced
                (rare classes drawn ~``majority/minority`` × more often),
                ``0.5`` = sqrt-balanced (a common compromise that avoids
                overfitting the minority class while still helping recall).
            sample_class_labels: Required when ``oversample=True``. A 1-D
                ``LongTensor`` of class ids (length == ``len(train_dataset)``)
                in the same order as the train dataset.
        """
        super().__init__(*args, **kwargs)
        if class_weights.shape != (NUM_LABELS,):
            raise ValueError(
                f"class_weights must have shape ({NUM_LABELS},); "
                f"got {tuple(class_weights.shape)}"
            )
        self.class_weights: torch.Tensor = class_weights.detach().clone().float()

        self.oversample = bool(oversample)
        self.oversample_alpha = float(oversample_alpha)
        if self.oversample and sample_class_labels is None:
            raise ValueError(
                "oversample=True requires sample_class_labels (labels parallel "
                "to train_dataset)."
            )
        self.sample_class_labels: torch.Tensor | None = (
            sample_class_labels.detach().clone().long()
            if sample_class_labels is not None
            else None
        )

    def _get_train_sampler(self, *args: Any, **kwargs: Any):
        """Return a class-balanced :class:`WeightedRandomSampler` if enabled.

        Falls back to the default Trainer sampler otherwise. The sampler
        draws each example with probability proportional to ``1 / count[c]
        ** alpha``, normalized so the per-sample weights sum to 1. Drawn
        with replacement, so minority-class examples are repeated within
        an epoch — pair with ``class_weights == 1`` to avoid double-correction.
        """
        if not self.oversample or self.sample_class_labels is None:
            return super()._get_train_sampler(*args, **kwargs)

        from torch.utils.data import WeightedRandomSampler

        labels = self.sample_class_labels
        counts = torch.bincount(labels, minlength=NUM_LABELS).float().clamp(min=1.0)
        # weight per CLASS: 1 / count^alpha. alpha=0 → uniform (natural),
        # alpha=1 → exactly balanced, alpha=0.5 → sqrt-balanced.
        class_w = counts.pow(-self.oversample_alpha)
        sample_w = class_w[labels].double()  # double() per WeightedRandomSampler convention

        n_samples = len(labels)
        sampler = WeightedRandomSampler(
            weights=sample_w,
            num_samples=n_samples,
            replacement=True,
        )
        # Friendly diagnostic: show expected per-class draws per epoch.
        with torch.no_grad():
            probs = sample_w / sample_w.sum()
            per_class_p = torch.zeros(NUM_LABELS, dtype=torch.float64)
            per_class_p.scatter_add_(0, labels.long(), probs)
            expected = (per_class_p * n_samples).tolist()
        print(
            f"[oversample] alpha={self.oversample_alpha}, "
            f"expected draws/epoch per class: "
            f"{[f'{c}:{int(round(e))}' for c, e in enumerate(expected)]}"
        )
        return sampler

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # Unsloth's per-epoch eval flips the model into a code path that
        # returns an "empty logits" sentinel (raises on __getitem__). The env
        # var is sometimes consulted per-forward, so re-assert it here.
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        class_label = inputs.pop("class_label", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)

        # `logits` may be Unsloth's empty-logits sentinel — not None, but
        # raises NotImplementedError on indexing. Probe with a tiny slice;
        # if it raises, fall back to the base trainer (model's own CE loss).
        logits_usable = logits is not None
        if logits_usable:
            try:
                _ = logits[..., :1, :1]
            except (NotImplementedError, AttributeError, TypeError):
                logits_usable = False

        if not logits_usable or labels is None or class_label is None:
            if class_label is not None:
                inputs["class_label"] = class_label
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        B = shift_labels.size(0)

        cw = self.class_weights.to(logits.device, non_blocking=True)
        cl = class_label.to(logits.device)

        # Collect per-sample losses in a list and stack them — using
        # `logits.new_zeros(...)[b] = ce` would silently drop gradients,
        # because in-place __setitem__ on a leaf tensor with requires_grad=False
        # never attaches a grad_fn to the destination.
        sample_losses: list[torch.Tensor] = []
        sample_weights: list[torch.Tensor] = []
        for b in range(B):
            if (shift_labels[b] != -100).any():
                sample_losses.append(
                    F.cross_entropy(
                        shift_logits[b],
                        shift_labels[b],
                        ignore_index=-100,
                        reduction="mean",
                    )
                )
                sample_weights.append(cw[cl[b]])

        if sample_losses:
            losses_t = torch.stack(sample_losses)
            weights_t = torch.stack(sample_weights).to(losses_t.dtype)
            loss = (losses_t * weights_t).mean()
        else:
            # No valid label tokens in this batch — emit zero loss that still
            # depends on logits so backward has a valid graph.
            loss = logits.sum() * 0.0

        if return_outputs:
            return loss, outputs
        return loss
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        import gc
        from unsloth import FastLanguageModel

        # clean up before eval to free memory for the forward pass; eval can be OOM otherwise due to the large max_length and Qwen2.5's vocab size.
        gc.collect()
        torch.cuda.empty_cache()

        # inference mode can be more memory-efficient for evaluation, especially with large max_length and vocab size; switch to it before the forward pass, then back to training mode afterward.
        FastLanguageModel.for_inference(self.model)

        # do the actual evaluation (calls compute_loss, which relies on the env var to get logits)
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # switch back to training mode for the next epoch; if the model was wrapped in a PEFT adapter, also call FastLanguageModel.for_training to re-enable the training code paths that PEFT relies on.
        FastLanguageModel.for_training(self.model)
        
        # 5. clean up for next step not OOM
        torch.cuda.empty_cache()
        
        return metrics


class ReduceLROnPlateauCallback(TrainerCallback):
    """
    Monitor eval_loss and reduce the learning rate when it has not improved
    for a consecutive number of evaluations equal to `patience`.

    This callback requires using an evaluation strategy (i.e., eval_strategy != 'no').
    """
    def __init__(
        self,
        patience: int = 2,
        factor: float = 0.5,
        min_delta: float = 1e-4,
        min_lr: float = 1e-6,
        verbose: bool = True,
    ):
        self.patience = patience
        self.factor = factor
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.verbose = verbose
        self._best_loss = float("inf")
        self._bad_evals = 0

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if eval_loss < self._best_loss - self.min_delta:
            self._best_loss = eval_loss
            self._bad_evals = 0
        else:
            self._bad_evals += 1
            if self.verbose:
                print(
                    f"\n[ReduceLROnPlateau] eval_loss not improved for {self._bad_evals} eval(s) "
                    f"({self._bad_evals}/{self.patience}) "
                    f"best={self._best_loss:.4f}, current={eval_loss:.4f}"
                )

        if self._bad_evals >= self.patience:
            self._bad_evals = 0
            optimizer = kwargs.get("optimizer")
            if optimizer is None:
                return
            for pg in optimizer.param_groups:
                old_lr = pg["lr"]
                new_lr = max(old_lr * self.factor, self.min_lr)
                pg["lr"] = new_lr
            if self.verbose:
                print(
                    f"[ReduceLROnPlateau] → lr {old_lr:.2e} → {new_lr:.2e}"
                )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for :func:`main`.

    Kept as a free function so the parser can be reused by tests / notebooks
    without invoking the full training pipeline.

    Returns:
        A configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for 5-way hallucination type classification.",
    )
    parser.add_argument("--data_dir", type=str, default=".",
                        help="Directory containing train.csv, dev.csv and "
                             "paper_evidence/{split}/*.pdf. Evidence is built "
                             "on the fly (cached to {split}_processed.csv).")
    parser.add_argument("--evidence_top_k", type=int, default=3,
                        help="Top-k paragraphs to retrieve per row.")
    parser.add_argument("--evidence_max_chars", type=int, default=1000,
                        help="Per-row evidence character cap.")
    parser.add_argument("--no_evidence_cache", action="store_true",
                        help="Disable reading/writing {split}_processed.csv "
                             "cache; always parse PDFs from scratch.")
    parser.add_argument("--output_dir", type=str, default="./adapter_checkpoint",
                        help="Where to save the trained PEFT adapter.")
    parser.add_argument("--max_length", type=int, default=768,
                        help="Maximum tokenized sequence length.")
    parser.add_argument("--r", type=int, default=32,
                        help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha (scaling).")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device train/eval batch size.")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Linear warmup ratio over total steps.")
    parser.add_argument("--oversample", action="store_true",
                        help="Use a class-balanced WeightedRandomSampler for "
                             "training. Implies class_weights=1 (the loss-level "
                             "and sampler-level corrections compose multiplicatively, "
                             "so combining them over-corrects).")
    parser.add_argument("--oversample_alpha", type=float, default=1.0,
                        help="Strength of inverse-frequency sampling: 0=natural, "
                             "1=fully balanced, 0.5=sqrt-balanced (compromise that "
                             "limits minority-class overfitting). Only used when "
                             "--oversample is set.")
    parser.add_argument("--no_class_weights", action="store_true",
                        help="Force class_weights to 1.0 even without --oversample. "
                             "Useful for ablation.")
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="ReduceLROnPlateau: number of evals with no improvement "
                             "before lr is reduced.")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="ReduceLROnPlateau: multiplicative factor by which lr "
                             "is reduced.")
    parser.add_argument("--lr_min_lr", type=float, default=1e-6,
                        help="ReduceLROnPlateau: lower bound on the learning rate.")
    parser.add_argument("--wandb_project", type=str, default="hw3-hallucination",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (auto-generated if omitted).")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging entirely.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Smoke-test the full pipeline on a tiny subset "
                             "(50 train / 20 dev, 1 epoch). Forces --no_wandb "
                             "and prints a component checklist at the end.")
    parser.add_argument("--max_grad_norm", type=float, default=0.3,
                        help="Gradient clipping norm(QLoRA 0.3)。")
    return parser


def _print_dry_run_checklist(
    checks: dict[str, bool],
    macro_f1: float | None,
) -> None:
    """Print the dry-run component checklist with ✓/✗ per stage."""
    def mark(ok: bool) -> str:
        return "✓" if ok else "✗"

    print("=" * 40)
    print("DRY RUN Checklist")
    print("=" * 40)
    print(f"[{mark(checks['model_load'])}] 模型載入成功")
    print(f"[{mark(checks['lora'])}] QLoRA adapter 套用成功")
    print(f"[{mark(checks['class_weights'])}] class_weights 計算正確")
    print(f"[{mark(checks['dataset'])}] HallucinationDataset 建立成功")
    print(f"[{mark(checks['forward_pass'])}] WeightedCETrainer forward pass 無誤")
    print(f"[{mark(checks['train'])}] 3 epoch 訓練完成")
    f1_str = f"（Macro F1 = {macro_f1:.4f}）" if macro_f1 is not None else ""
    print(f"[{mark(checks['eval'])}] dev 評估完成{f1_str}")
    print(f"[{mark(checks['config_save'])}] training_config.json 儲存成功")
    print("[✓] wandb disabled（dry_run 固定關閉）")
    print("=" * 40)


def main() -> None:
    """Run the full QLoRA training pipeline end-to-end.

    Steps:
        1. Parse CLI args.
        2. Load the 4-bit Qwen2.5-3B-Instruct base model + tokenizer via
           Unsloth's ``FastLanguageModel`` (Unsloth patches transformers on
           import — keep the import inside ``main`` so callers who only
           need the dataset / trainer classes don't pay that cost).
        3. Wrap the model with a LoRA adapter targeting attention + FFN
           projections, with gradient checkpointing.
        4. Build train / eval :class:`HallucinationDataset` instances from
           ``train_processed.csv`` / ``dev_processed.csv``.
        5. Compute balanced class weights and per-class answer-token ids.
        6. Configure ``TrainingArguments`` (epoch-based eval & save, fp16 for
           T4-class GPUs, no external loggers).
        7. Train via the default ``Trainer`` and save the adapter.
    """
    args = _build_arg_parser().parse_args()

    import wandb  # type: ignore[import-not-found]

    if args.dry_run:
        args.no_wandb = True
        args.epochs = 3
        print("[dry-run] Forcing --no_wandb=True, epochs=3; "
              "train/dev will be sliced to 50/20.")

    checks: dict[str, bool] = {
        "model_load": False,
        "lora": False,
        "class_weights": False,
        "dataset": False,
        "forward_pass": False,
        "train": False,
        "eval": False,
        "config_save": False,
    }
    macro_f1: float | None = None

    try:
        print(f"[1/6] Loading base model (max_seq_length={args.max_length}, 4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            max_seq_length=args.max_length,
            dtype=None,
            load_in_4bit=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        checks["model_load"] = True

        print(f"[2/6] Applying LoRA (r={args.r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * n_train_params / n_total_params
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        else:
            print(f"trainable params: {n_train_params:,} || all params: {n_total_params:,} "
                  f"|| trainable%: {trainable_pct:.4f}")
        checks["lora"] = True

        data_dir = Path(args.data_dir)
        print(f"[3/6] Building evidence from {data_dir}/ (cache={'off' if args.no_evidence_cache else 'on'})...")
        train_df = load_split_with_evidence(
            str(data_dir), "train",
            top_k=args.evidence_top_k,
            max_chars=args.evidence_max_chars,
            cache=not args.no_evidence_cache,
        )
        dev_df = load_split_with_evidence(
            str(data_dir), "dev",
            top_k=args.evidence_top_k,
            max_chars=args.evidence_max_chars,
            cache=not args.no_evidence_cache,
        )
        print(f"       train={len(train_df)}, dev={len(dev_df)}")
        # MiniLM is no longer needed; free its ~80MB before LM training begins.
        release_embedder()

        if args.dry_run:
            train_df = train_df.head(50).reset_index(drop=True)
            dev_df = dev_df.head(20).reset_index(drop=True)
            print(f"       [dry-run] sliced -> train={len(train_df)}, dev={len(dev_df)}")

        print("[3.5/6] Oversampling minority classes to median count...")
        train_df_balanced = oversample_to_median(train_df)
        print(f"       train size: {len(train_df)} → {len(train_df_balanced)}")

        # Strategy selection:
        #   --oversample           -> sampler-level balancing, class_weights=1
        #   --no_class_weights     -> ablation: no balancing at all
        #   default                -> loss-level class weighting (legacy behavior)
        disable_class_weights = args.oversample or args.no_class_weights
        if disable_class_weights:
            reason = "oversample" if args.oversample else "no_class_weights"
            print(f"[class_weights] disabled ({reason}); using all-ones.")
            class_weights = torch.ones(NUM_LABELS, dtype=torch.float32)
        else:
            class_weights = compute_class_weights(train_df_balanced["label"])
        checks["class_weights"] = True

        sample_class_labels: torch.Tensor | None = None
        if args.oversample:
            sample_class_labels = torch.tensor(
                train_df_balanced["label"].astype(int).tolist(), dtype=torch.long
            )

        print(f"[4/6] Tokenizing datasets (max_length={args.max_length})...")
        train_dataset = HallucinationDataset(train_df_balanced, tokenizer, max_length=args.max_length)
        eval_dataset = HallucinationDataset(dev_df, tokenizer, max_length=args.max_length)
        if args.dry_run:
            sample0 = train_dataset[0]
            decoded = tokenizer.decode(sample0["input_ids"], skip_special_tokens=False)
            print("-" * 50)
            print("[dry-run] First training prompt (decoded):")
            print(decoded)
            print("-" * 50)
        checks["dataset"] = True

        steps_per_epoch = math.ceil(
            len(train_dataset) / (args.batch_size * args.grad_accum)
        )
        total_train_steps = steps_per_epoch * args.epochs
        warmup_steps = int(args.warmup_ratio * total_train_steps)

        class_counts_train = [int((train_df_balanced["label"] == c).sum()) for c in range(NUM_LABELS)]
        class_counts_dev = [int((dev_df["label"] == c).sum()) for c in range(NUM_LABELS)]

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        else:
            gpu_name = "cpu"
            gpu_mem_gb = 0.0

        print("[5/6] Configuring TrainingArguments...")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=10 if args.dry_run else 20,
            report_to="wandb" if not args.no_wandb else "none",
            save_total_limit=2,
            remove_unused_columns=False,
        )

        if not args.no_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    # 模型
                    "model": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                    "max_length": args.max_length,

                    # QLoRA
                    "r": args.r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"],
                    "bias": "none",
                    "use_gradient_checkpointing": "unsloth",
                    "trainable_params": n_train_params,
                    "total_params": n_total_params,
                    "trainable_pct": trainable_pct,

                    # 訓練
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "effective_batch_size": args.batch_size * args.grad_accum,
                    "epochs": args.epochs,
                    "warmup_ratio": args.warmup_ratio,
                    "bf16": training_args.bf16,
                    "fp16": training_args.fp16,
                    "seed": training_args.seed,
                    "weight_decay": training_args.weight_decay,
                    "max_grad_norm": args.max_grad_norm,
                    "lr_scheduler_type": str(training_args.lr_scheduler_type),
                    "optim": str(training_args.optim),
                    "adam_beta1": training_args.adam_beta1,
                    "adam_beta2": training_args.adam_beta2,
                    "steps_per_epoch": steps_per_epoch,
                    "total_train_steps": total_train_steps,
                    "warmup_steps": warmup_steps,

                    # 資料
                    "class_weights": class_weights.tolist(),
                    "class_counts_train": class_counts_train,
                    "class_counts_dev": class_counts_dev,
                    "train_samples": len(train_df),
                    "dev_samples": len(dev_df),
                    "oversample": args.oversample,
                    "oversample_alpha": args.oversample_alpha,
                    "no_class_weights": args.no_class_weights,

                    # 硬體
                    "gpu_name": gpu_name,
                    "gpu_mem_gb": gpu_mem_gb,
                },
            )

        wandb_url = wandb.run.url if wandb.run is not None else None

        print("[6/6] Building WeightedCETrainer and starting training...")
        plateau_callback = ReduceLROnPlateauCallback(
            patience=args.lr_patience,
            factor=args.lr_factor,
            min_lr=args.lr_min_lr,
        )
        trainer = WeightedCETrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            class_weights=class_weights,
            oversample=args.oversample,
            oversample_alpha=args.oversample_alpha,
            sample_class_labels=sample_class_labels,
            callbacks=[plateau_callback],
        )

        if args.dry_run:
            sample = train_dataset[0]
            batch = {k: v.unsqueeze(0).to(model.device) for k, v in sample.items()}
            with torch.no_grad():
                test_loss = trainer.compute_loss(model, batch)
            print(f"[dry-run] forward-pass loss: {float(test_loss):.4f}")
            checks["forward_pass"] = True

        train_start = time.time()
        trainer.train()
        train_seconds = time.time() - train_start
        checks["forward_pass"] = True
        checks["train"] = True

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving adapter + tokenizer to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        macro_f1, per_class_f1 = _eval_dev_set(
            model=model,
            tokenizer=tokenizer,
            dev_df=dev_df,
            max_length=args.max_length,
            n_eval=1777,
            wandb_enabled=not args.no_wandb,
            wandb_module=wandb,
        )
        checks["eval"] = True

        if args.oversample:
            sampling_strategy = f"oversample(alpha={args.oversample_alpha})"
        elif args.no_class_weights:
            sampling_strategy = "none"
        else:
            sampling_strategy = "class_weights"

        config_path = output_dir / "training_config.json"
        config_payload: dict[str, Any] = {
            **vars(args),
            "sampling_strategy": sampling_strategy,
            "class_weights": class_weights.tolist(),
            "class_counts_train": class_counts_train,
            "dev_macro_f1": float(macro_f1),
            "dev_class_f1": {name: float(per_class_f1[i]) for i, name in enumerate(CLASS_NAMES)},
            "train_seconds": float(train_seconds),
            "wandb_run_url": wandb_url,
        }
        with config_path.open("w") as f:
            json.dump(config_payload, f, indent=2)
        print(f"Saved training config to {config_path}")
        checks["config_save"] = True

        if not args.no_wandb:
            wandb.finish()

        if args.dry_run:
            return

        print("=" * 40)
        print("Training Complete!")
        print(f"Dev Macro F1     : {macro_f1:.4f}")
        print(f"Adapter saved to : {output_dir}")
        print(f"Config saved to  : {config_path}")
        print(f"WandB Run        : {wandb_url or 'disabled'}")
        print("=" * 40)
    finally:
        if args.dry_run:
            _print_dry_run_checklist(checks, macro_f1)


def _eval_dev_set(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    dev_df: pd.DataFrame,
    max_length: int,
    n_eval: int,
    wandb_enabled: bool,
    wandb_module: Any,
) -> tuple[float, np.ndarray]:
    """Evaluate the trained model on the first ``n_eval`` rows of dev.

    Generates a single-token completion per example, parses it via
    :func:`prompt_template.parse_model_output`, then computes Macro F1,
    per-class precision/recall/F1, and a confusion matrix. When wandb is
    enabled, scalar metrics and the confusion matrix are also logged.

    Args:
        model: The trained (PEFT-wrapped) causal LM.
        tokenizer: Matching tokenizer.
        dev_df: Dev dataframe; only the first ``n_eval`` rows are used.
        max_length: Truncation cap for the prompt (matches training).
        n_eval: Number of dev rows to evaluate.
        wandb_enabled: Whether to log metrics to wandb.
        wandb_module: The imported ``wandb`` module (or any object exposing
            ``.log`` and ``.plot.confusion_matrix``); ignored if
            ``wandb_enabled`` is False.

    Returns:
        ``(macro_f1, per_class_f1)`` where ``per_class_f1`` is a numpy
        array of length 5 indexed by class id.
    """
    

    print(f"Evaluating on first {n_eval} dev samples...")
    FastLanguageModel.for_inference(model)

    eval_subset = dev_df.head(n_eval).reset_index(drop=True)
    true_labels: list[int] = []
    pred_labels: list[int] = []
    parse_failures = 0
    class_names_lower = [n.lower() for n in CLASS_NAMES]

    for i in range(len(eval_subset)):
        row = eval_subset.iloc[i]
        messages = format_chat_messages(str(row["text"]), str(row["evidence"]))
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = out[0, enc["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded_lower = decoded.lower()
        if not any(n in decoded_lower for n in class_names_lower):
            parse_failures += 1
        pred_labels.append(parse_model_output(decoded))
        true_labels.append(int(row["label"]))

    label_ids = list(range(NUM_LABELS))
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=label_ids, average=None, zero_division=0
    )
    macro_f1 = float(
        f1_score(true_labels, pred_labels, average="macro", labels=label_ids, zero_division=0)
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=label_ids)
    cm_row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.where(cm_row_sums == 0, 1, cm_row_sums)

    print("\nClassification report:")
    print(classification_report(
        true_labels, pred_labels,
        labels=label_ids,
        target_names=list(CLASS_NAMES),
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(f"Parse failures: {parse_failures} / {len(eval_subset)}")

    if wandb_enabled:
        log_payload: dict[str, Any] = {
            "dev/macro_f1": macro_f1,
            "dev/parse_failures": parse_failures,
            "dev/parse_failure_rate": parse_failures / max(1, len(eval_subset)),
        }
        for cls_id, name in enumerate(CLASS_NAMES):
            key = name.replace(" ", "_")
            log_payload[f"dev/class_f1/{key}"] = float(per_class_f1[cls_id])
            log_payload[f"dev/class_precision/{key}"] = float(per_class_p[cls_id])
            log_payload[f"dev/class_recall/{key}"] = float(per_class_r[cls_id])

        if torch.cuda.is_available():
            log_payload["dev/gpu_peak_mem_gb"] = (
                torch.cuda.max_memory_allocated() / 1e9
            )

        wandb_module.log(log_payload)

        wandb_module.log({
            "dev/confusion_matrix": wandb_module.plot.confusion_matrix(
                probs=None,
                y_true=true_labels,
                preds=pred_labels,
                class_names=list(CLASS_NAMES),
            )
        })

        cm_norm_table = wandb_module.Table(
            columns=["true_class"] + [f"pred_{n}" for n in CLASS_NAMES]
        )
        for i, name in enumerate(CLASS_NAMES):
            cm_norm_table.add_data(
                name, *[float(cm_norm[i, j]) for j in range(NUM_LABELS)]
            )
        wandb_module.log({"dev/confusion_matrix_normalized": cm_norm_table})

    return macro_f1, per_class_f1


if __name__ == "__main__":
    main()
