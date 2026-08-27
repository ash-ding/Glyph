"""Supervised fine-tuning: the only way capability gets into weights here.

Phase 1 fixes the training algorithm on purpose.  With the algorithm frozen,
the model the agent gets is a deterministic function of (data, target,
hyper-parameters), which is what makes the three-way regret decomposition
mean anything.  Let the algorithm vary too and "the agent designed well" and
"RL happened to help" stop being separable.

All seven target roles are trained through one loss -- token cross-entropy on
the response.  R4 (proposal distribution) needs candidates weighted by their
verifier score rather than a single best answer, so examples carry a
`weight`; that keeps the proposal semantics without a second code path.  R5
(value/verifier) emits its score as tokens for the same reason.  One path
means an arm cannot differ from another by accident.

Loss is masked to the answer: the prompt is given, not predicted.  Weighting
is per example and applied after per-example normalisation, so a long answer
does not quietly count for more than a short one.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Example:
    prompt: str
    answer: str
    weight: float = 1.0        # R4: the verifier's score for this candidate


@dataclass
class HParams:
    """What the agent is allowed to choose, and nothing else."""
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    full_finetune: bool = False
    lr: float | None = None     # None -> 1e-4 for LoRA, 1e-5 for full
    epochs: int = 2
    batch_size: int = 32
    max_len: int = 256
    warmup_frac: float = 0.03
    seed: int = 0

    def resolved_lr(self) -> float:
        if self.lr is not None:
            return self.lr
        return 1e-5 if self.full_finetune else 1e-4


class _Packed(Dataset):
    def __init__(self, examples: list[Example], tok, max_len: int):
        self.rows = []
        for ex in examples:
            p = tok(ex.prompt, add_special_tokens=False)["input_ids"]
            a = tok(ex.answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = (p + a)[:max_len]
            labels = ([-100] * len(p) + a)[:max_len]
            if all(t == -100 for t in labels):
                continue            # the prompt filled max_len; nothing to learn
            self.rows.append((ids, labels, float(ex.weight)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _collate(batch, pad_id: int):
    width = max(len(ids) for ids, _, _ in batch)
    n = len(batch)
    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    labels = torch.full((n, width), -100, dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    weights = torch.tensor([w for _, _, w in batch], dtype=torch.float)
    for k, (ids, lab, _) in enumerate(batch):
        input_ids[k, :len(ids)] = torch.tensor(ids)
        labels[k, :len(lab)] = torch.tensor(lab)
        attn[k, :len(ids)] = 1
    return input_ids, labels, attn, weights


def _weighted_loss(logits: torch.Tensor, labels: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
    """Per-example mean CE over answer tokens, then weighted across the batch.

    Normalising inside the example before weighting is the point: without it
    a long answer contributes more gradient than a short one and the weights
    the agent chose stop meaning what it thinks they mean.
    """
    logits = logits[:, :-1].float()
    target = labels[:, 1:]
    mask = target != -100
    safe = target.masked_fill(~mask, 0)

    tok_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), safe.reshape(-1), reduction="none"
    ).view(target.shape)
    tok_loss = tok_loss * mask

    counts = mask.sum(dim=1).clamp(min=1)
    per_example = tok_loss.sum(dim=1) / counts
    w = weights.to(per_example.device)
    return (per_example * w).sum() / w.sum().clamp(min=1e-6)


def train(examples: list[Example], hp: HParams, *, base_model: str,
          out_dir: str | Path, ledger=None, device: str = "cuda",
          log_every: int = 50) -> dict:
    """Train and save.  Returns a record of what was actually run.

    GPU seconds go through the ledger, measured rather than estimated -- the
    crossover figure is denominated in them, and a formula would drift from
    what the machine actually did.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = _Packed(examples, tok, hp.max_len)
    if len(ds) == 0:
        raise ValueError("no trainable examples: every answer was truncated away")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if not hp.full_finetune:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=hp.lora_rank, lora_alpha=hp.lora_alpha, lora_dropout=hp.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))

    loader = DataLoader(ds, batch_size=hp.batch_size, shuffle=True,
                        collate_fn=lambda b: _collate(b, tok.pad_token_id),
                        generator=torch.Generator().manual_seed(hp.seed))
    steps = max(1, len(loader) * hp.epochs)
    lr = hp.resolved_lr()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=hp.warmup_frac)

    timer = ledger.gpu_timer("student_training") if ledger is not None else None
    if timer is not None:
        timer.__enter__()
    model.train()
    step = 0
    losses: list[float] = []
    try:
        for _ in range(hp.epochs):
            for input_ids, labels, attn, weights in loader:
                logits = model(input_ids=input_ids.to(device),
                               attention_mask=attn.to(device)).logits
                loss = _weighted_loss(logits, labels.to(device), weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                losses.append(float(loss.item()))
                if log_every and step % log_every == 0:
                    print(f"    step {step}/{steps}  loss "
                          f"{sum(losses[-log_every:]) / log_every:.4f}", flush=True)
    finally:
        if timer is not None:
            timer.__exit__(None, None, None)

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    record = {"base_model": base_model, "out_dir": str(out_dir),
              "examples": len(ds), "dropped": len(examples) - len(ds),
              "steps": step, "hparams": asdict(hp), "lr": lr,
              "final_loss": sum(losses[-20:]) / max(1, len(losses[-20:])),
              "adapter": "full" if hp.full_finetune else f"lora-r{hp.lora_rank}"}
    (out_dir / "train_record.json").write_text(json.dumps(record, indent=2))
    return record
