"""Experiment identity, bounded configuration, and synchronized cost accounting."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import torch


METHODS = ("ppo_sim", "ppo_wm_frozen", "ppo_wm_frozen_hard", "ppo_wm_online", "supermimic", "bptt_wm_frozen", "pretrain")


@dataclass
class Config:
    method: str = "supermimic"
    seed: int = 0
    subset: str = "locomotion"
    num_envs: int = 512
    buffer_steps: int = 256
    iterations: int = 100
    horizon: int = 32
    hist: int = 10
    goal_frames: int = 10
    wm_window: int = 8
    wm_batch: int = 256
    wm_batches: int = 2
    pi_batch: int = 512
    warmup_steps: int = 64
    wm_only_iters: int = 50
    collect_steps: float = 2.5
    lr: float = 2e-4
    actor_lr: float | None = None
    lr_decay_iters: int = 250000
    gamma: float = 0.99
    lam: float = 0.95
    initial_std: float = 0.1
    ppo_epochs: int = 5
    minibatches: int = 4
    clip: float = 0.2
    entropy_coef: float = 0.0
    target_kl: float = 0.02
    ppo_wm_survival: str = "soft"
    amp: str = "none"
    compile: bool = True
    checkpoint_every: int = 25
    max_wall_seconds: float = 1800
    bootstrap: str | None = None
    motion_dir: str = "data/lafan1_g1"

    def validate(self):
        if self.method not in METHODS or self.subset not in ("locomotion", "all40"):
            raise ValueError("unknown method or subset")
        for key in ("num_envs", "buffer_steps", "iterations", "horizon", "hist", "goal_frames",
                    "wm_window", "wm_batch", "wm_batches", "pi_batch", "collect_steps",
                    "ppo_epochs", "minibatches", "checkpoint_every"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.warmup_steps < self.wm_window or self.buffer_steps < self.wm_window:
            raise ValueError("warmup/buffer must contain a full WM window")
        if not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0 or self.amp not in ("none", "bf16"):
            raise ValueError("finite positive time budget and supported precision required")
        if self.ppo_wm_survival not in ("hard", "soft"):
            raise ValueError("ppo_wm_survival must be hard or soft")
        if not 0 < self.gamma < 1 or not 0 <= self.lam <= 1 or not 0 < self.clip < 1:
            raise ValueError("invalid discount/GAE/clipping range")
        if self.seed < 0 or self.wm_only_iters < 0 or self.lr_decay_iters < 1:
            raise ValueError("invalid seed/warmup/decay schedule")
        for key in ("lr", "initial_std", "collect_steps"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.actor_lr is not None and (not math.isfinite(self.actor_lr) or self.actor_lr <= 0):
            raise ValueError("actor_lr must be finite and positive, or null to inherit lr")
        if "_frozen" in self.method and not self.bootstrap:
            raise ValueError("frozen methods require the SAME pretrain checkpoint")
        if self.method == "ppo_wm_frozen_hard" and self.ppo_wm_survival != "hard":
            raise ValueError("hard frozen comparator must use hard termination")
        if self.method == "ppo_sim" and self.bootstrap:
            raise ValueError("PPO-Sim must not be charged for or load WM pretraining")
        return self


def set_learning_rates(config, iteration, actor_optimizer, critic_optimizer, wm_optimizer=None):
    """Schedule the actor independently; retain the original shared-rate default."""
    if iteration < 1:
        raise ValueError("iterations are one-based")
    fraction = min((iteration - 1) / config.lr_decay_iters, 1.)
    scale = .03 + .97 * .5 * (1 + math.cos(math.pi * fraction))
    actor_base = config.lr if config.actor_lr is None else config.actor_lr
    rates = {"lr": config.lr * scale, "actor_lr": actor_base * scale}
    for optimizer, rate in ((actor_optimizer, rates["actor_lr"]),
                            (critic_optimizer, rates["lr"]), (wm_optimizer, rates["lr"])):
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = rate
    return rates


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def motion_files(config):
    files = sorted(Path(config.motion_dir).glob("*.csv"))
    if len(files) != 40:
        raise ValueError(f"Expected the complete 40-clip source, got {len(files)}")
    if config.subset == "locomotion":
        files = [p for p in files if p.stem.startswith(("walk", "run", "sprint"))]
    if not files:
        raise ValueError("empty motion subset")
    return files


def identity(config, *, hash_bootstrap=True):
    files = motion_files(config)
    root = Path(__file__).resolve().parents[2]
    sources = sorted(p for folder in ("supermimic", "experiments/ablation")
                     for p in (root / folder).glob("*.py"))
    return {
        "config": asdict(config),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sources},
        "motions": {p.name: sha256(p) for p in files},
        "assets": {str(p.relative_to(root)): sha256(p)
                   for directory in ("data/g1_description", "data/g1_usd", "data/isaac-assets")
                   for p in sorted((root / directory).rglob("*")) if p.is_file()},
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-"),
        "cuda": torch.version.cuda, "gpu_count": 1,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("mujoco", "mujoco-warp", "warp-lang", "newton", "isaaclab", "isaaclab-newton")},
        "bootstrap_sha256": sha256(config.bootstrap) if config.bootstrap and hash_bootstrap else None,
        "evaluation": "native-full-clips-first-frame-fresh-process-v1",
    }


class Ledger:
    """Single allocated GPU, synchronized phase timing, cold start included.

    Evaluation runs separately and has its own ledger. Pretraining is charged in
    full to EVERY standalone WM method, even when physically reused across runs.
    Actual campaign spend can deduplicate bootstrap_sha256; do not deduplicate
    the standalone-method curves. Timings include failed attempt logs separately.
    """

    def __init__(self, started=None, inherited=None):
        self.started = time.perf_counter() if started is None else started
        self.inherited = dict(inherited or {})
        self.seconds = {k: 0.0 for k in ("collection", "wm_training", "imagined_rollout", "policy_optimization")}
        self.simulator_steps = 0
        self.imagined_steps = 0

    @contextmanager
    def phase(self, name):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.seconds[name] += time.perf_counter() - start

    def snapshot(self):
        torch.cuda.synchronize()
        actual = time.perf_counter() - self.started
        total = actual + self.inherited.get("train_wall_seconds", 0)
        out = {f"{k}_seconds": v + self.inherited.get(f"{k}_seconds", 0)
               for k, v in self.seconds.items()}
        out.update(simulator_steps=self.simulator_steps + self.inherited.get("simulator_steps", 0),
                   imagined_steps=self.imagined_steps + self.inherited.get("imagined_steps", 0),
                   train_wall_seconds=total, actual_run_seconds=actual, train_gpu_hours=total / 3600,
                   actual_run_gpu_hours=actual / 3600,
                   peak_torch_memory_bytes=torch.cuda.max_memory_allocated())
        out["simulator_steps_per_second"] = out["simulator_steps"] / max(out["collection_seconds"], 1e-12)
        out["imagined_steps_per_second"] = out["imagined_steps"] / max(out["imagined_rollout_seconds"], 1e-12)
        return out


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def append_csv(path, row):
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def freeze(module, frozen):
    for parameter in module.parameters():
        parameter.requires_grad_(not frozen)
