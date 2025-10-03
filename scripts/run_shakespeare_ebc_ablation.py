import argparse
import json
import math
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import subprocess
import uuid

# Ensure matplotlib cache path is writable before importing pyplot
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# Avoid JAX processes greedily reserving nearly all GPU memory per worker
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.22")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from configs import parse_config_from_json
from data_loaders import get_data_loader
from models import create_model
from optimizers import get_optimizer
from trainer import Trainer


def clone_args(args, **overrides):
    new = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        setattr(new, k, v)
    return new


class AblationLogger:
    def __init__(self, config):
        self.config = config
        self.steps: List[int] = []
        self.losses: List[float] = []
        self.train_accs: List[float] = []
        self.ebc_c: List[float] = []
        self.ebc_applied_kl: List[float] = []
        self.ebc_delta_ctrl: List[float] = []
        self.ebc_rho: List[float] = []
        self.ebc_tau: List[float] = []
        self.ebc_tau_scale: List[float] = []
        self.ebc_surr_scale: List[float] = []
        self.ebc_S: List[float] = []
        self.ebc_S_raw: List[float] = []
        self.ebc_beta_mean: List[float] = []
        self.ebc_beta_max: List[float] = []
        self.ebc_lr: List[float] = []
        self.wall_times: List[float] = []
        self._t0 = time.time()

    def log_training(self, step, loss, accuracy, log):
        self.steps.append(int(step))
        self.losses.append(float(loss))
        self.train_accs.append(float(accuracy))
        c = None
        applied_kl = None
        delta_ctrl = None
        rho = None
        tau = None
        tau_scale = None
        S = None
        S_raw = None
        beta_mean = None
        beta_max = None
        lr = None
        if isinstance(log, dict) and "ebc" in log and isinstance(log["ebc"], dict):
            c = log["ebc"].get("c")
            applied_kl = log["ebc"].get("applied_kl")
            delta_ctrl = log["ebc"].get("delta_ctrl")
            rho = log["ebc"].get("rho")
            tau = log["ebc"].get("tau")
            tau_scale = log["ebc"].get("tau_scale")
            S = log["ebc"].get("S")
            S_raw = log["ebc"].get("S_raw")
            beta_mean = log["ebc"].get("beta_mean")
            beta_max = log["ebc"].get("beta_max")
            lr = log["ebc"].get("lr")
            surr_scale = log["ebc"].get("surr_scale")
        self.ebc_c.append(None if c is None else float(c))
        self.ebc_applied_kl.append(None if applied_kl is None else float(applied_kl))
        self.ebc_delta_ctrl.append(None if delta_ctrl is None else float(delta_ctrl))
        self.ebc_rho.append(None if rho is None else float(rho))
        self.ebc_tau.append(None if tau is None else float(tau))
        self.ebc_tau_scale.append(None if tau_scale is None else float(tau_scale))
        self.ebc_surr_scale.append(None if surr_scale is None else float(surr_scale))
        self.ebc_S.append(None if S is None else float(S))
        self.ebc_S_raw.append(None if S_raw is None else float(S_raw))
        self.ebc_beta_mean.append(None if beta_mean is None else float(beta_mean))
        self.ebc_beta_max.append(None if beta_max is None else float(beta_max))
        self.ebc_lr.append(None if lr is None else float(lr))
        self.wall_times.append(time.time() - self._t0)

    def log_validation(self, step, metrics):
        # For now we rely on final validate() at the end
        pass

    def get_results(self):
        # compute c statistics
        cs = [c for c in self.ebc_c if c is not None]
        c_p10 = c_p50 = c_p90 = None
        eta_eff_mean = eta_eff_median = None
        if cs:
            import numpy as _np
            arr = _np.array(cs)
            c_p10 = float(_np.percentile(arr, 10))
            c_p50 = float(_np.percentile(arr, 50))
            c_p90 = float(_np.percentile(arr, 90))
            lr = float(getattr(self.config, "lr", 0.0))
            eta_eff_mean = float(arr.mean() * lr)
            eta_eff_median = float(_np.median(arr) * lr)
        return {
            "steps": self.steps,
            "train_losses": self.losses,
            "train_accs": self.train_accs,
            "ebc_c": self.ebc_c,
            "ebc_applied_kl": self.ebc_applied_kl,
            "ebc_delta_ctrl": self.ebc_delta_ctrl,
            "ebc_rho": self.ebc_rho,
            "ebc_tau": self.ebc_tau,
            "ebc_tau_scale": self.ebc_tau_scale,
            "ebc_surr_scale": self.ebc_surr_scale,
            "ebc_S": self.ebc_S,
            "ebc_S_raw": self.ebc_S_raw,
            "ebc_beta_mean": self.ebc_beta_mean,
            "ebc_beta_max": self.ebc_beta_max,
            "ebc_lr": self.ebc_lr,
            "c_p10": c_p10,
            "c_p50": c_p50,
            "c_p90": c_p90,
            "eta_eff_mean": eta_eff_mean,
            "eta_eff_median": eta_eff_median,
            "wall_times": self.wall_times,
        }


class TaskFailed(RuntimeError):
    def __init__(self, meta: Dict[str, Any], returncode: int, log_path: Path, task: Tuple[Dict[str, Any], Dict[str, Any]]):
        message = f"Subprocess failed for {meta} with return code {returncode}"
        super().__init__(message)
        self.meta = meta
        self.returncode = returncode
        self.log_path = log_path
        self.task = task


def make_config(
    args,
    optimizer: str,
    ebc: bool,
    spectral: bool,
    delta: float = None,
    aggregate: str = None,
    run_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
):
    # Project mapping
    if spectral:
        project = {"default": "spec_normalize"}
    else:
        project = {"default": "none"}

    # Resolve run directory for artifacts/checkpoints
    out_root = Path(run_dir) if run_dir is not None else Path(args.output_dir)
    ckpt_dir = out_root / "ckpts"
    if job_id:
        ckpt_dir = ckpt_dir / job_id

    cfg = dict(
        # Data/model
        data="shakespeare",
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        d_embed=args.d_embed,
        num_blocks=args.num_blocks,
        softmax_scale=1.0,
        final_scale=1.0,
        residual_scale=1.0,
        scales_learnable=False,
        zero_init=False,
        max_embed_inflation_factor=1.0,
        use_unembed=True,
        layernorm_substitute="none",
        # Train
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        accum_steps=1,
        lr=args.lr,
        wd=args.wd,
        spectral_wd=0.0,
        w_max=1.0,
        schedule="none",
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        val_iters=args.val_iters,
        num_checkpoints=0,
        seed=args.seed,
        # Optimizer
        optimizer=optimizer,
        beta1=0.9,
        beta2=0.999,
        # Dtypes
        model_dtype=args.model_dtype,
        project_dtype=args.project_dtype,
        project=project,
        # EBC
        ebc_enable=bool(ebc),
        ebc_target_kl=args.ebc_target_kl if delta is None else delta,
        ebc_update_every=args.ebc_update_every,
        ebc_probe_layers=args.ebc_probe_layers,
        ebc_beta_ema=args.ebc_beta_ema,
        ebc_safety=args.ebc_safety,
        ebc_aggregate=args.ebc_aggregate if aggregate is None else aggregate,
        ebc_center_logits=True,
        ebc_include_embed_out=False,
        # EBC controller (optional)
        ebc_ctrl_enable=bool(getattr(args, "ebc_ctrl_enable", False)),
        ebc_ctrl_period=int(getattr(args, "ebc_ctrl_period", 20)),
        ebc_ctrl_kp=float(getattr(args, "ebc_ctrl_kp", 0.15)),
        ebc_ctrl_ki=float(getattr(args, "ebc_ctrl_ki", 0.02)),
        ebc_ctrl_ema_halflife=float(getattr(args, "ebc_ctrl_ema_halflife", 250)),
        ebc_delta_star=float(args.ebc_delta_star) if getattr(args, "ebc_delta_star", None) is not None else args.ebc_target_kl,
        ebc_ctrl_delta_min=float(getattr(args, "ebc_ctrl_delta_min", 0.01)),
        ebc_ctrl_delta_max=float(getattr(args, "ebc_ctrl_delta_max", 0.30)),
        ebc_ctrl_log_every=bool(getattr(args, "ebc_ctrl_log_every", False)),
        # EBC guard params
        ebc_guard_warmup_steps=int(getattr(args, "ebc_guard_warmup_steps", 100)),
        ebc_tau_scale_floor=float(getattr(args, "ebc_tau_scale_floor", 0.05)),
        ebc_tau_max_shrink_per_probe=float(getattr(args, "ebc_tau_max_shrink_per_probe", 4.0)),
        ebc_tau_recover_rate=float(getattr(args, "ebc_tau_recover_rate", 1.10)),
        ebc_tau_shrink_exponent=float(getattr(args, "ebc_tau_shrink_exponent", 0.5)),
        # EBC robust betas
        ebc_beta_huber_delta=float(getattr(args, "ebc_beta_huber_delta", 0.0)),
        ebc_beta_full_sweep=int(getattr(args, "ebc_beta_full_sweep", 200)),
        # Misc
        jit=bool(args.jit),  # enable with --jit; default False due to RoPE tracer note
        output_dir=str(out_root),
        spectral_backend=getattr(args, "spectral_backend", "auto"),
        # Checkpoint/resume
        resume=bool(getattr(args, "resume", True)),
        ckpt_interval=int(getattr(args, "ckpt_interval", 200)),
        ckpt_dir=str(ckpt_dir),
    )
    return cfg


def run_one(config_dict: Dict[str, Any]):
    config = parse_config_from_json(config_dict)
    train_loader, val_loader, loss_fn = get_data_loader(config)
    model = create_model(config)
    optimizer = get_optimizer(config)

    logger = AblationLogger(config)
    key = jax.random.PRNGKey(config.seed)
    key, sub = jax.random.split(key)
    params = model.initialize(sub)
    opt_state = optimizer.init_state(params)

    # Resume support: load latest checkpoint if present
    ckpt_dir = Path(getattr(config, "ckpt_dir", Path(config.output_dir) / "ckpts"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    done_path = ckpt_dir / "done.json"
    if getattr(config, "resume", True) and done_path.exists():
        with open(done_path, "r") as f:
            saved = json.load(f)
        return saved

    latest_ckpt = None
    if getattr(config, "resume", True):
        for p in sorted(ckpt_dir.glob("ckpt_*.pkl")):
            latest_ckpt = p
        if latest_ckpt is not None:
            import pickle
            with open(latest_ckpt, "rb") as f:
                state = pickle.load(f)
            # Restore trainer-related counters after instantiation below
            params = state.get("params", params)
            opt_state = state.get("opt_state", opt_state)
            key = state.get("key", key)

    trainer = Trainer(model, optimizer, train_loader, val_loader, loss_fn, config, logger)
    # If we loaded a checkpoint, restore trainer internals and step
    if getattr(config, "resume", True) and latest_ckpt is not None:
        trainer.restore_state(state)

    t0 = time.time()
    params, opt_state, key = trainer.train(params, opt_state, key)
    wall_clock = time.time() - t0

    # Final validation
    val_metrics = trainer.validate(params)
    ppl = float(math.exp(float(val_metrics["loss"])) if val_metrics["loss"] < 20 else float("inf"))

    history = logger.get_results()
    # Clip statistics
    cs = [c for c in history["ebc_c"] if c is not None]
    no_clip_rate = None
    clip_rate = None
    avg_c = None
    c_p10 = history.get("c_p10")
    c_p50 = history.get("c_p50")
    c_p90 = history.get("c_p90")
    eta_eff_mean = history.get("eta_eff_mean")
    eta_eff_median = history.get("eta_eff_median")
    if cs:
        no_clip_rate = float(sum(1 for c in cs if c >= 0.999) / len(cs))
        clip_rate = 1.0 - no_clip_rate
        avg_c = float(sum(cs) / len(cs))

    result = {
        "wall_clock": wall_clock,
        "val_loss": float(val_metrics["loss"]),
        "val_acc": float(val_metrics["accuracy"]),
        "ppl": ppl,
        "history": history,
        "accept_rate": no_clip_rate,
        "no_clip_rate": no_clip_rate,
        "clip_rate": clip_rate,
        "avg_c": avg_c,
        "c_p10": c_p10,
        "c_p50": c_p50,
        "c_p90": c_p90,
        "eta_eff_mean": eta_eff_mean,
        "eta_eff_median": eta_eff_median,
    }
    # Mark job as done for resume detection
    try:
        with open(done_path, "w") as f:
            json.dump(result, f)
    except Exception:
        pass
    return result


def plot_run(history: Dict[str, Any], file_prefix: str, out_dir: Path, display_title: Optional[str] = None):
    steps = history["steps"]
    losses = history["train_losses"]
    cs = history["ebc_c"]

    title = display_title or file_prefix

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_loss.png", dpi=160)

    if any(c is not None for c in cs):
        plt.figure(figsize=(8, 3))
        plt.plot(steps, [1.0 if c is None else c for c in cs])
        plt.xlabel("Step")
        plt.ylabel("c (clip)")
        plt.ylim(0, 1.05)
        plt.title(f"{title} clip factor")
        plt.tight_layout()
        plt.savefig(out_dir / f"{file_prefix}_clip.png", dpi=160)


def job_label(meta: Dict[str, Any]) -> str:
    base = f"{meta['optimizer']}_{meta['spectral']}"
    if meta["ebc"] == "off":
        return f"{base}_baseline"
    delta = meta.get("delta")
    agg = meta.get("aggregate")
    delta_str = "na" if delta is None else str(delta).replace(".", "p")
    agg_str = agg or ""
    return f"{base}_ebc_d{delta_str}_{agg_str}"


def job_display(meta: Dict[str, Any]) -> str:
    base = f"{meta['optimizer']} ({meta['spectral']})"
    if meta["ebc"] == "off":
        return f"{base} baseline"
    return f"{base} EBC δ={meta.get('delta')} {meta.get('aggregate')}"


def _run_task(task: Tuple[Dict[str, Any], Dict[str, Any]]):
    meta, cfg = task
    res = run_one(cfg)
    return meta, res


def run_auto_pipeline(args, out_dir: Path):
    """Automatic LR sweep + EBC calibration + final runs.

    Strategy:
    - For each base (optimizer x spectral), sweep LR bracket for baseline (EBC off) for warmup steps.
    - Pick LR with lowest warmup val loss per base.
    - For each base, sweep deltas/aggregates for EBC for warmup steps using the chosen LR; pick candidate whose clip rate falls within [accept_lo, accept_hi] (interpreted as clip-rate band) and best val; fallback to best val not worse than baseline by max_val_degrade; otherwise skip EBC for that base.
    - Run final full steps for the baseline and EBC winner per base.
    """
    optimizers = [s.strip() for s in args.optimizers.split(",") if s.strip()]
    spectral_opts = [s.strip() for s in args.spectral.split(",") if s.strip()]

    adam_lrs = [float(x) for x in args.adam_lrs.split(",") if x]
    muon_lrs = [float(x) for x in args.muon_lrs.split(",") if x]
    lr_bracket = {"adam": adam_lrs, "muon": muon_lrs}
    delta_candidates = [float(x) for x in args.ebc_deltas_auto.split(",") if x]
    agg_candidates = [s.strip() for s in args.ebc_aggregates_auto.split(",") if s.strip()]
    lr_scales = [float(x) for x in args.ebc_lr_scales.split(",") if x]

    # Detect available GPUs and clamp concurrency
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip() != "":
            gpu_list = [x for x in visible.split(",") if x.strip() != ""]
            avail_gpus = len(gpu_list)
        else:
            # JAX will raise if no plugin; handle defensively
            avail_gpus = len([d for d in jax.devices() if d.platform == "gpu"])
    except Exception:
        avail_gpus = 0
    if avail_gpus <= 0:
        print("[auto] Warning: no GPUs detected; falling back to CPU. This will be very slow.")
        effective_gpus = 0
    else:
        effective_gpus = min(int(args.num_gpus), avail_gpus)
        if effective_gpus < int(args.num_gpus):
            print(f"[auto] Requested num_gpus={args.num_gpus} but only {avail_gpus} visible; using {effective_gpus}.")

    warm_args = clone_args(
        args,
        steps=int(args.auto_warmup_steps),
        val_interval=int(args.auto_val_interval),
        val_iters=int(args.auto_val_iters),
    )
    # Warm dataset once in the parent
    warm_cfg = make_config(warm_args, optimizers[0], ebc=False, spectral=False)
    _warm_config = parse_config_from_json(warm_cfg)
    _warm_train, _warm_val, _ = get_data_loader(_warm_config)
    del _warm_train, _warm_val

    # Queue helpers
    def launch_queue(task_list: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        # Use multi-GPU orchestration if requested; else run inline sequentially
        if effective_gpus <= 1:
            results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for t in task_list:
                results.append(_run_task(t))
            return results

        tmp_dir = out_dir / "_auto_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        pending: List[Tuple[Dict[str, Any], Dict[str, Any]]] = list(task_list)
        active: List[Tuple[subprocess.Popen, Path, Dict[str, Any], int, Tuple[Dict[str, Any], Dict[str, Any]]]] = []
        results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        retry_counts: Dict[str, int] = {}
        max_task_retries = max(0, int(getattr(args, "task_retry", 1)))
        next_task_idx = 0

        # Helper to launch the next pending task on a specific GPU id
        def launch_next(gid: int):
            nonlocal next_task_idx
            if next_task_idx >= len(pending):
                return False
            task = pending[next_task_idx]
            proc = _launch_single_task_subprocess(task, gid, tmp_dir)
            active.append(proc)
            next_task_idx += 1
            return True

        # Initial fill
        for gid in range(min(effective_gpus, len(pending))):
            launch_next(gid)

        while active or next_task_idx < len(pending):
            try:
                finished = _drain_finished(active)
            except TaskFailed as err:
                meta_key = json.dumps(err.meta, sort_keys=True)
                attempt = retry_counts.get(meta_key, 0)
                if attempt < max_task_retries:
                    retry_counts[meta_key] = attempt + 1
                    pending.append(err.task)
                    time.sleep(min(2.0 * (attempt + 1), 10.0))
                    continue
                raise RuntimeError(f"Subprocess failed for {err.meta} after {attempt} retries (rc={err.returncode})") from err

            for meta, res, gid in finished:
                results.append((meta, res))
                launch_next(gid)

            if active:
                time.sleep(0.2)
            elif next_task_idx < len(pending):
                # If GPUs freed but nothing active, launch pending tasks on available slots
                for gid in range(min(effective_gpus, len(pending) - next_task_idx)):
                    launch_next(gid)

        return results

    def _clip_value(metrics: Dict[str, Any]) -> float:
        clip = metrics.get("clip_rate")
        return 0.0 if clip is None else float(clip)

    def _summarize_entry(meta: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "meta": meta,
            "metrics": metrics,
            "optimizer": meta.get("optimizer"),
            "spectral": meta.get("spectral"),
            "aggregate": meta.get("aggregate"),
            "delta": float(meta.get("delta")) if meta.get("delta") is not None else None,
            "lr": float(meta.get("lr")) if meta.get("lr") is not None else None,
            "stage": meta.get("stage"),
            "clip_rate": _clip_value(metrics),
            "avg_c": metrics.get("avg_c"),
            "val_loss": float(metrics.get("val_loss", float("inf"))),
        }
        return entry

    clip_lo = float(args.accept_lo)
    clip_hi = float(args.accept_hi)
    clip_target = args.clip_target if args.clip_target is not None else 0.5 * (clip_lo + clip_hi)
    skip_threshold = getattr(args, "auto_skip_threshold", 0.2)
    delta_min = float(args.auto_delta_min)
    delta_max = float(args.auto_delta_max)
    delta_expand = float(args.auto_delta_expand)
    delta_iters = int(args.auto_delta_iters)

    # Stage 1: LR sweep per base
    lr_tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for opt in optimizers:
        for spec in spectral_opts:
            spec_bool = (spec == "spec")
            for lr in lr_bracket.get(opt, [args.lr]):
                meta = {"stage": "lr", "optimizer": opt, "spectral": spec, "ebc": "off", "delta": None, "aggregate": None, "lr": lr}
                label = f"lr_{job_label(meta)}_lr{str(lr).replace('.', 'p')}"
                cfg = make_config(clone_args(warm_args, lr=lr), opt, ebc=False, spectral=spec_bool, run_dir=out_dir, job_id=label)
                lr_tasks.append((meta, cfg))

    lr_results = launch_queue(lr_tasks)

    # Summarize and pick best LR per base
    best_lr: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
    baseline_val: Dict[Tuple[str, str], float] = {}
    lr_records: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    print("[auto] LR sweep results:")
    for meta, res in lr_results:
        key = (meta["optimizer"], meta["spectral"])
        v = res["val_loss"]
        lr_records.setdefault(key, []).append(_summarize_entry(meta, res))
        if key not in best_lr or v < best_lr[key][0]:
            best_lr[key] = (v, meta)
            baseline_val[key] = v
        print(f"  {key} lr={meta['lr']}: val_loss={v:.4f}")

    global_best_val = min(baseline_val.values()) if baseline_val else float("inf")
    skip_bases: Dict[Tuple[str, str], bool] = {}
    for key, val in baseline_val.items():
        skip = bool(val > global_best_val * (1 + skip_threshold)) if math.isfinite(global_best_val) else False
        skip_bases[key] = skip
        if skip:
            print(
                f"  -> Skipping calibration for {key} (baseline {val:.4f} vs best {global_best_val:.4f}, threshold {(1 + skip_threshold):.2f}x)"
            )

    # Stage 2: EBC calibration using chosen LR

    def calibrate_base(opt: str, spec: str, base_lr: float, base_val: float):
        spec_bool = (spec == "spec")
        entries: List[Dict[str, Any]] = []
        seen: set[Tuple[str, float]] = set()

        def record(results: List[Tuple[Dict[str, Any], Dict[str, Any]]]):
            for meta, metrics in results:
                entry = _summarize_entry(meta, metrics)
                entries.append(entry)
                if entry["delta"] is not None:
                    seen.add((entry["aggregate"], round(float(entry["delta"]), 8)))

        def run_initial_grid():
            task_batch: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for delta in delta_candidates:
                for agg in agg_candidates:
                    meta = {
                        "stage": "calib",
                        "optimizer": opt,
                        "spectral": spec,
                        "ebc": "on",
                        "delta": float(delta),
                        "aggregate": agg,
                        "lr": base_lr,
                    }
                    label = f"calib_{job_label(meta)}"
                    cfg = make_config(
                        clone_args(warm_args, lr=base_lr),
                        opt,
                        ebc=True,
                        spectral=spec_bool,
                        delta=float(delta),
                        aggregate=agg,
                        run_dir=out_dir,
                        job_id=label,
                    )
                    task_batch.append((meta, cfg))
            if task_batch:
                record(launch_queue(task_batch))

        def run_single(delta: float, aggregate: str, tag: str):
            key_id = (aggregate, round(float(delta), 8))
            if key_id in seen:
                return None
            meta = {
                "stage": tag,
                "optimizer": opt,
                "spectral": spec,
                "ebc": "on",
                "delta": float(delta),
                "aggregate": aggregate,
                "lr": base_lr,
            }
            label = f"{tag}_{job_label(meta)}_d{str(delta).replace('.', 'p')}"
            cfg = make_config(
                clone_args(warm_args, lr=base_lr),
                opt,
                ebc=True,
                spectral=spec_bool,
                delta=float(delta),
                aggregate=aggregate,
                run_dir=out_dir,
                job_id=label,
            )
            record(launch_queue([(meta, cfg)]))
            return entries[-1]

        run_initial_grid()
        if not entries:
            return None, entries, {"status": "no_candidates"}

        focus_agg = None
        best_val = float("inf")
        for agg in agg_candidates:
            agg_entries = [e for e in entries if e.get("aggregate") == agg]
            if not agg_entries:
                continue
            best_entry = min(agg_entries, key=lambda e: e["val_loss"])
            if best_entry["val_loss"] < best_val:
                best_val = best_entry["val_loss"]
                focus_agg = agg
        if focus_agg is None:
            focus_agg = agg_candidates[0] if agg_candidates else args.ebc_aggregate

        def focus_list() -> List[Dict[str, Any]]:
            return sorted(
                [e for e in entries if e.get("aggregate") == focus_agg and e.get("delta") is not None],
                key=lambda e: e["delta"],
            )

        attempts = 0
        while attempts < delta_iters:
            current = focus_list()
            if not current:
                break
            if any(clip_lo <= e["clip_rate"] <= clip_hi for e in current):
                break

            bracket_delta = None
            for left, right in zip(current, current[1:]):
                c_left = left["clip_rate"]
                c_right = right["clip_rate"]
                if (c_left > clip_hi and c_right < clip_lo) or (c_left < clip_lo and c_right > clip_hi):
                    bracket_delta = math.sqrt(left["delta"] * right["delta"])
                    break
            if bracket_delta is None:
                clip_values = [e["clip_rate"] for e in current]
                if clip_values and all(c > clip_hi for c in clip_values):
                    bracket_delta = min(current[-1]["delta"] * delta_expand, delta_max)
                elif clip_values and all(c < clip_lo for c in clip_values):
                    bracket_delta = max(current[0]["delta"] / delta_expand, delta_min)
            if bracket_delta is None:
                break
            if bracket_delta < delta_min or bracket_delta > delta_max:
                break
            if any(abs(bracket_delta - e["delta"]) / max(e["delta"], 1e-8) < 1e-3 for e in current):
                break
            if run_single(bracket_delta, focus_agg, tag="adelta") is None:
                break
            attempts += 1

        focused = focus_list()
        if not focused:
            return None, entries, {"status": "no_focus", "focus_aggregate": focus_agg}

        max_allowed = base_val * (1 + args.max_val_degrade)
        feasible = [e for e in focused if e["val_loss"] <= max_allowed]
        used_limit = True
        if not feasible:
            feasible = focused
            used_limit = False
        if not feasible:
            return None, entries, {"status": "no_feasible", "focus_aggregate": focus_agg}

        in_band = [e for e in feasible if clip_lo <= e["clip_rate"] <= clip_hi]
        if in_band:
            chosen = min(in_band, key=lambda e: e["val_loss"])
            status = "in_band"
        else:
            chosen = min(feasible, key=lambda e: (abs(e["clip_rate"] - clip_target), e["val_loss"]))
            status = "nearest"

        notes = {
            "status": status,
            "focus_aggregate": focus_agg,
            "used_val_limit": used_limit,
            "samples": [
                {
                    "delta": e["delta"],
                    "aggregate": e["aggregate"],
                    "clip_rate": e["clip_rate"],
                    "val_loss": e["val_loss"],
                }
                for e in focused
            ],
        }
        notes["winner"] = {
            "delta": chosen["delta"],
            "aggregate": chosen["aggregate"],
            "clip_rate": chosen["clip_rate"],
            "val_loss": chosen["val_loss"],
        }

        winner_info = {
            "delta": chosen["delta"],
            "aggregate": chosen["aggregate"],
            "lr": base_lr,
            "clip_rate": chosen["clip_rate"],
            "avg_c": chosen["avg_c"],
            "val_loss": chosen["val_loss"],
        }
        return winner_info, entries, notes

    winners: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = { (o, s): None for o in optimizers for s in spectral_opts }
    calibration_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    print("[auto] EBC calibration results:")
    for opt in optimizers:
        for spec in spectral_opts:
            key = (opt, spec)
            if key not in best_lr:
                calibration_records[key] = {"status": "no_lr"}
                continue
            if skip_bases.get(key, False):
                calibration_records[key] = {"status": "skipped"}
                print(f"  {key} skipped (baseline triage)")
                continue
            base_lr = best_lr[key][1]["lr"]
            base_val = baseline_val.get(key, float("inf"))
            winner_info, entries, notes = calibrate_base(opt, spec, base_lr, base_val)
            entries_sorted = sorted(
                [e for e in entries if e.get("delta") is not None],
                key=lambda e: (e["aggregate"], e["delta"]),
            )
            if entries_sorted:
                for entry in entries_sorted:
                    print(
                        f"  {key} delta={entry['delta']} {entry['aggregate']} lr={entry['lr']}: val={entry['val_loss']:.4f} clip_rate={entry['clip_rate']}"
                    )
            else:
                print(f"  {key} produced no EBC samples")
            calibration_records[key] = {
                "status": notes.get("status", "unknown"),
                "entries": [
                    {
                        "delta": e["delta"],
                        "aggregate": e["aggregate"],
                        "clip_rate": e["clip_rate"],
                        "val_loss": e["val_loss"],
                        "lr": e["lr"],
                        "stage": e.get("stage"),
                    }
                    for e in entries_sorted
                ],
                "notes": notes,
            }
    winners[key] = winner_info
    if winner_info is None:
        print(f"  -> No viable delta found for {key}")

    # Stage 2c: Controller/guard sweep (optional)
    chosen_ctrl: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = { (o, s): None for o in optimizers for s in spectral_opts }
    if getattr(args, "auto_ctrl_search", False):
        print("[auto] Controller/guard sweep:")
        kps = [float(x) for x in str(getattr(args, "ctrl_kps", "")).split(",") if x.strip()]
        kis = [float(x) for x in str(getattr(args, "ctrl_kis", "")).split(",") if x.strip()]
        hls = [float(x) for x in str(getattr(args, "ctrl_halflifes", "")).split(",") if x.strip()]
        dmins = [float(x) for x in str(getattr(args, "ctrl_delta_mins", "")).split(",") if x.strip()]
        dmaxs = [float(x) for x in str(getattr(args, "ctrl_delta_maxs", "")).split(",") if x.strip()]
        floors = [float(x) for x in str(getattr(args, "guard_floors", "")).split(",") if x.strip()]
        exps = [float(x) for x in str(getattr(args, "guard_shrink_exponents", "")).split(",") if x.strip()]

        ctrl_tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for opt in optimizers:
            for spec in spectral_opts:
                key = (opt, spec)
                w = winners.get(key)
                if not w or skip_bases.get(key, False):
                    continue
                spec_bool = (spec == "spec")
                base_lr = best_lr[key][1]["lr"] if key in best_lr else args.lr
                for kp in kps:
                    for ki in kis:
                        for hl in hls:
                            for dm in dmins:
                                for dx in dmaxs:
                                    for fl in floors:
                                        for ex in exps:
                                            meta = {
                                                "stage": "ctrl",
                                                "optimizer": opt,
                                                "spectral": spec,
                                                "ebc": "on",
                                                "delta": w["delta"],
                                                "aggregate": w["aggregate"],
                                                "lr": base_lr,
                                                "kp": kp,
                                                "ki": ki,
                                                "hl": hl,
                                                "dmin": dm,
                                                "dmax": dx,
                                                "floor": fl,
                                                "exp": ex,
                                            }
                                            label = (
                                                f"ctrl_{job_label(meta)}_kp{str(kp).replace('.', 'p')}_ki{str(ki).replace('.', 'p')}"
                                            )
                                            cfg = make_config(
                                                clone_args(
                                                    warm_args,
                                                    lr=base_lr,
                                                    ebc_ctrl_kp=kp,
                                                    ebc_ctrl_ki=ki,
                                                    ebc_ctrl_ema_halflife=hl,
                                                    ebc_ctrl_delta_min=dm,
                                                    ebc_ctrl_delta_max=dx,
                                                    ebc_guard_warmup_steps=args.ebc_guard_warmup_steps,
                                                    ebc_tau_scale_floor=fl,
                                                    ebc_tau_max_shrink_per_probe=args.ebc_tau_max_shrink_per_probe,
                                                    ebc_tau_recover_rate=args.ebc_tau_recover_rate,
                                                    ebc_tau_shrink_exponent=ex,
                                                ),
                                                opt,
                                                ebc=True,
                                                spectral=spec_bool,
                                                delta=w["delta"],
                                                aggregate=w["aggregate"],
                                                run_dir=out_dir,
                                                job_id=label,
                                            )
                                            ctrl_tasks.append((meta, cfg))
        ctrl_results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        if ctrl_tasks:
            ctrl_results = launch_queue(ctrl_tasks)

        # Score and choose
        # Objective: minimize (|clip_rate-target|, KL tracking error, warmup val)
        chosen_ctrl = { (o, s): None for o in optimizers for s in spectral_opts }
        if ctrl_results:
            clip_target = clip_target
            def _score(meta, res):
                hist = res.get("history", {})
                kl = [k for k in hist.get("ebc_applied_kl", []) if k is not None]
                dstar = [d for d in hist.get("ebc_delta_ctrl", []) if d is not None]
                rho = [r for r in hist.get("ebc_rho", []) if r is not None]
                # tracking error (MAE)
                if kl and dstar and len(kl) == len(dstar):
                    import numpy as _np
                    mae = float(_np.mean(_np.abs(_np.array(kl) - _np.array(dstar))))
                else:
                    mae = 1e3
                # violation fraction
                viol = 0.0
                if rho:
                    viol = float(sum(1 for r in rho if r > 1.2) / len(rho))
                # clip deviation
                clip = res.get("clip_rate")
                if clip is None:
                    clip = 1.0
                dev = abs(float(clip) - float(clip_target))
                # val loss
                v = float(res.get("val_loss", 1e6))
                # weighted score
                return 3.0*dev + 5.0*mae + 1.0*viol + 0.5*v

            print("[auto] Controller candidates:")
            buckets: Dict[Tuple[str, str], List[Tuple[float, Dict[str, Any], Dict[str, Any]]]] = {}
            for meta, res in ctrl_results:
                key = (meta["optimizer"], meta["spectral"])
                s = _score(meta, res)
                buckets.setdefault(key, []).append((s, meta, res))
                print(
                    f"  {key} kp={meta['kp']} ki={meta['ki']} hl={meta['hl']} dmin={meta['dmin']} dmax={meta['dmax']} floor={meta['floor']} exp={meta['exp']} -> score={s:.4f}, val={res.get('val_loss'):.4f}, clip={res.get('clip_rate')}"
                )
            for key, lst in buckets.items():
                lst.sort(key=lambda t: t[0])
                best_score, best_meta, best_res = lst[0]
                chosen_ctrl[key] = {
                    "kp": best_meta["kp"],
                    "ki": best_meta["ki"],
                    "hl": best_meta["hl"],
                    "dmin": best_meta["dmin"],
                    "dmax": best_meta["dmax"],
                    "floor": best_meta["floor"],
                    "exp": best_meta["exp"],
                }
                print(f"[auto] Chosen ctrl for {key}: {chosen_ctrl[key]}")

    # Stage 2b: LR-scale sweep for EBC + paired control baselines
    scale_tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    scale_ctrl_tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for opt in optimizers:
        for spec in spectral_opts:
            key = (opt, spec)
            winner_info = winners.get(key)
            if not winner_info or skip_bases.get(key, False):
                continue
            spec_bool = (spec == "spec")
            base_lr = best_lr[key][1]["lr"] if key in best_lr else args.lr
            delta = winner_info["delta"]
            aggregate = winner_info["aggregate"]
            for s in lr_scales:
                lr_scaled = base_lr * s
                meta_e = {
                    "stage": "scale",
                    "optimizer": opt,
                    "spectral": spec,
                    "ebc": "on",
                    "delta": delta,
                    "aggregate": aggregate,
                    "lr": lr_scaled,
                    "lr_scale": s,
                }
                label_e = f"scale_{job_label(meta_e)}_x{str(s).replace('.', 'p')}"
                # If controller search chose settings, inject them
                if getattr(args, "auto_ctrl_search", False) and chosen_ctrl.get(key):
                    ctrl = chosen_ctrl[key]
                    ctrl_args = clone_args(
                        warm_args,
                        lr=lr_scaled,
                        ebc_ctrl_kp=ctrl["kp"],
                        ebc_ctrl_ki=ctrl["ki"],
                        ebc_ctrl_ema_halflife=ctrl["hl"],
                        ebc_ctrl_delta_min=ctrl["dmin"],
                        ebc_ctrl_delta_max=ctrl["dmax"],
                        ebc_guard_warmup_steps=args.ebc_guard_warmup_steps,
                        ebc_tau_scale_floor=ctrl["floor"],
                        ebc_tau_max_shrink_per_probe=args.ebc_tau_max_shrink_per_probe,
                        ebc_tau_recover_rate=args.ebc_tau_recover_rate,
                        ebc_tau_shrink_exponent=ctrl["exp"],
                    )
                else:
                    ctrl_args = clone_args(warm_args, lr=lr_scaled)
                cfg_e = make_config(
                    ctrl_args,
                    opt,
                    ebc=True,
                    spectral=spec_bool,
                    delta=delta,
                    aggregate=aggregate,
                    run_dir=out_dir,
                    job_id=label_e,
                )
                scale_tasks.append((meta_e, cfg_e))

                meta_c = {
                    "stage": "scale_ctrl",
                    "optimizer": opt,
                    "spectral": spec,
                    "ebc": "off",
                    "delta": None,
                    "aggregate": None,
                    "lr": lr_scaled,
                    "lr_scale": s,
                }
                label_c = f"scale_ctrl_{job_label(meta_c)}_x{str(s).replace('.', 'p')}"
                cfg_c = make_config(
                    clone_args(warm_args, lr=lr_scaled),
                    opt,
                    ebc=False,
                    spectral=spec_bool,
                    run_dir=out_dir,
                    job_id=label_c,
                )
                scale_ctrl_tasks.append((meta_c, cfg_c))

    scale_results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    scale_ctrl_results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    if scale_tasks:
        scale_results = launch_queue(scale_tasks)
    if scale_ctrl_tasks:
        scale_ctrl_results = launch_queue(scale_ctrl_tasks)

    scale_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    chosen_scale: Dict[Tuple[str, str], Optional[float]] = {}

    if scale_results:
        print("[auto] EBC LR scale results:")
    scale_entries: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for meta, res in scale_results:
        key = (meta["optimizer"], meta["spectral"])
        entry = _summarize_entry(meta, res)
        scale_entries.setdefault(key, []).append(entry)
        print(
            f"  {key} scale={meta.get('lr_scale', 1)}: val={res['val_loss']:.4f} clip_rate={res.get('clip_rate')}"
        )

    ctrl_entries: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if scale_ctrl_results:
        print("[auto] Control LR scale baselines:")
    for meta, res in scale_ctrl_results:
        key = (meta["optimizer"], meta["spectral"])
        entry = _summarize_entry(meta, res)
        ctrl_entries.setdefault(key, []).append(entry)
        print(
            f"  {key} ctrl scale={meta.get('lr_scale', 1)}: val={res['val_loss']:.4f}"
        )

    for opt in optimizers:
        for spec in spectral_opts:
            key = (opt, spec)
            winner_info = winners.get(key)
            if not winner_info or skip_bases.get(key, False):
                chosen_scale[key] = None
                continue
            entries = scale_entries.get(key, [])
            base_val = baseline_val.get(key, float("inf"))
            limit = base_val * (1 + args.scale_max_val_degrade)
            best = None
            # Prefer highest scale within clip band and val limit
            candidates = []
            for entry in entries:
                scale = float(entry["meta"].get("lr_scale", 1.0))
                val_loss = entry["val_loss"]
                clip = entry["clip_rate"]
                in_band = clip_lo <= clip <= clip_hi
                within = val_loss <= limit
                if within:
                    candidates.append((scale, in_band, val_loss, entry))
            chosen = None
            if candidates:
                # sort by (band priority, scale descending, val ascending)
                candidates.sort(key=lambda x: (not x[1], -x[0], x[2]))
                chosen = candidates[0]
            elif entries:
                # fallback to lowest val even if exceeding limit
                entries.sort(key=lambda e: e["val_loss"])
                entry = entries[0]
                chosen = (float(entry["meta"].get("lr_scale", 1.0)), False, entry["val_loss"], entry)

            chosen_scale[key] = chosen[0] if chosen else None
            scale_records[key] = {
                "entries": [
                    {
                        "lr_scale": float(e["meta"].get("lr_scale", 1.0)),
                        "clip_rate": e["clip_rate"],
                        "val_loss": e["val_loss"],
                    }
                    for e in sorted(entries, key=lambda e: float(e["meta"].get("lr_scale", 1.0)))
                ],
                "controls": [
                    {
                        "lr_scale": float(e["meta"].get("lr_scale", 1.0)),
                        "val_loss": e["val_loss"],
                    }
                    for e in sorted(ctrl_entries.get(key, []), key=lambda e: float(e["meta"].get("lr_scale", 1.0)))
                ],
            }

    # Stage 3: Final full runs for baseline + winners
    final_steps = int(args.final_steps or args.steps)
    final_args = clone_args(args, steps=final_steps)
    final_tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for opt in optimizers:
        for spec in spectral_opts:
            key = (opt, spec)
            if skip_bases.get(key, False):
                continue
            spec_bool = (spec == "spec")
            base_lr = best_lr[key][1]["lr"] if key in best_lr else args.lr
            base_meta = {
                "stage": "final",
                "optimizer": opt,
                "spectral": spec,
                "ebc": "off",
                "delta": None,
                "aggregate": None,
                "lr": base_lr,
            }
            base_label = f"final_{job_label(base_meta)}"
            if getattr(args, "auto_ctrl_search", False) and chosen_ctrl.get(key):
                ctrl = chosen_ctrl[key]
                final_base_args = clone_args(
                    final_args,
                    lr=base_lr,
                    ebc_ctrl_kp=ctrl["kp"],
                    ebc_ctrl_ki=ctrl["ki"],
                    ebc_ctrl_ema_halflife=ctrl["hl"],
                    ebc_ctrl_delta_min=ctrl["dmin"],
                    ebc_ctrl_delta_max=ctrl["dmax"],
                    ebc_guard_warmup_steps=args.ebc_guard_warmup_steps,
                    ebc_tau_scale_floor=ctrl["floor"],
                    ebc_tau_max_shrink_per_probe=args.ebc_tau_max_shrink_per_probe,
                    ebc_tau_recover_rate=args.ebc_tau_recover_rate,
                    ebc_tau_shrink_exponent=ctrl["exp"],
                )
            else:
                final_base_args = clone_args(final_args, lr=base_lr)
            base_cfg = make_config(
                final_base_args,
                opt,
                ebc=False,
                spectral=spec_bool,
                run_dir=out_dir,
                job_id=base_label,
            )
            final_tasks.append((base_meta, base_cfg))

            winner_info = winners.get(key)
            if not winner_info:
                continue
            scale_choice = chosen_scale.get(key)
            lr_scale = scale_choice if scale_choice is not None else 1.0
            lr_ebc = base_lr * lr_scale
            ebc_meta = {
                "stage": "final",
                "optimizer": opt,
                "spectral": spec,
                "ebc": "on",
                "delta": winner_info["delta"],
                "aggregate": winner_info["aggregate"],
                "lr": lr_ebc,
                "lr_scale": lr_scale,
            }
            ebc_label = f"final_{job_label(ebc_meta)}"
            if getattr(args, "auto_ctrl_search", False) and chosen_ctrl.get(key):
                ctrl = chosen_ctrl[key]
                final_ebc_args = clone_args(
                    final_args,
                    lr=lr_ebc,
                    ebc_ctrl_kp=ctrl["kp"],
                    ebc_ctrl_ki=ctrl["ki"],
                    ebc_ctrl_ema_halflife=ctrl["hl"],
                    ebc_ctrl_delta_min=ctrl["dmin"],
                    ebc_ctrl_delta_max=ctrl["dmax"],
                    ebc_guard_warmup_steps=args.ebc_guard_warmup_steps,
                    ebc_tau_scale_floor=ctrl["floor"],
                    ebc_tau_max_shrink_per_probe=args.ebc_tau_max_shrink_per_probe,
                    ebc_tau_recover_rate=args.ebc_tau_recover_rate,
                    ebc_tau_shrink_exponent=ctrl["exp"],
                )
            else:
                final_ebc_args = clone_args(final_args, lr=lr_ebc)
            ebc_cfg = make_config(
                final_ebc_args,
                opt,
                ebc=True,
                spectral=spec_bool,
                delta=winner_info["delta"],
                aggregate=winner_info["aggregate"],
                run_dir=out_dir,
                job_id=ebc_label,
            )
            final_tasks.append((ebc_meta, ebc_cfg))

            if lr_scale and lr_scale > 1.0:
                ctrl_meta = {
                    "stage": "final",
                    "optimizer": opt,
                    "spectral": spec,
                    "ebc": "off",
                    "delta": None,
                    "aggregate": None,
                    "lr": lr_ebc,
                    "lr_scale": lr_scale,
                }
                ctrl_label = f"final_ctrl_{job_label(ctrl_meta)}"
                ctrl_cfg = make_config(
                    clone_args(final_args, lr=lr_ebc),
                    opt,
                    ebc=False,
                    spectral=spec_bool,
                    run_dir=out_dir,
                    job_id=ctrl_label,
                )
                final_tasks.append((ctrl_meta, ctrl_cfg))

    final_results = launch_queue(final_tasks) if final_tasks else []

    # Write summaries/plots
    print("optimizer,spectral,ebc,delta,aggregate,wall_clock,val_loss,val_acc,ppl,clip_rate,avg_c")
    out_summary: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for meta, res in final_results:
        out_summary.append((meta, res))
        clip_display = "" if res["clip_rate"] is None else f"{res['clip_rate']:.2f}"
        avgc_display = "" if res["avg_c"] is None else f"{res['avg_c']:.3f}"
        print(
            f"{meta['optimizer']},{meta['spectral']},{meta['ebc']},{meta.get('delta')},{meta.get('aggregate')},{res['wall_clock']:.2f},{res['val_loss']:.4f},{res['val_acc']:.4f},{res['ppl']:.2f},{clip_display},{avgc_display}"
        )
        plot_run(res["history"], job_label(meta), out_dir, job_display(meta))

    # Save CSV summary for finals
    csv_path = out_dir / "summary_final.csv"
    with open(csv_path, "w") as f:
        f.write("optimizer,spectral,ebc,delta,aggregate,lr,wall_clock,val_loss,val_acc,ppl,clip_rate,avg_c\n")
        for meta, res in out_summary:
            f.write(
                f"{meta['optimizer']},{meta['spectral']},{meta['ebc']},{'' if meta['delta'] is None else meta['delta']},{'' if meta['aggregate'] is None else meta['aggregate']},{meta.get('lr')},{res['wall_clock']:.2f},{res['val_loss']:.6f},{res['val_acc']:.6f},{res['ppl']:.3f},{res['clip_rate']},{res['avg_c']}\n"
            )

    # Dump JSON
    json_path = out_dir / "results_final.json"
    with open(json_path, "w") as f:
        json.dump([
            {"meta": meta, "metrics": res} for meta, res in out_summary
        ], f, indent=2)

    # Decisions dump
    decisions = {
        "best_lr": {
            "|".join(k): {
                "lr": best_lr[k][1]["lr"],
                "val_loss": best_lr[k][0],
            }
            for k in best_lr
        },
        "baseline_val": {"|".join(k): baseline_val.get(k) for k in best_lr},
        "skip_bases": {"|".join(k): bool(skip_bases.get(k, False)) for k in best_lr},
        "calibration": {"|".join(k): calibration_records.get(k, {}) for k in winners},
        "winners": {"|".join(k): winners.get(k) for k in winners},
        "chosen_scale": {"|".join(k): chosen_scale.get(k) for k in chosen_scale},
        "scale_records": {"|".join(k): scale_records.get(k, {}) for k in winners},
        "settings": {
            "clip_band": [clip_lo, clip_hi],
            "clip_target": clip_target,
            "max_val_degrade": args.max_val_degrade,
            "scale_max_val_degrade": getattr(args, "scale_max_val_degrade", None),
            "warmup_steps": args.auto_warmup_steps,
            "final_steps": final_steps,
            "ebc_lr_scales": lr_scales,
            "skip_threshold": skip_threshold,
        },
    }
    with open(out_dir / "auto_decisions.json", "w") as f:
        json.dump(decisions, f, indent=2)


def _write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f)


def _read_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)


def print_status(run_dir: Path):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        print(f"Run directory {run_dir} does not exist")
        return

    ckpt_root = run_dir / "ckpts"
    if not ckpt_root.exists():
        print(f"No checkpoints found under {ckpt_root}")
        return

    entries = []
    for job_dir in sorted(ckpt_root.iterdir()):
        if not job_dir.is_dir():
            continue
        done_path = job_dir / "done.json"
        latest_path = job_dir / "latest.pkl"
        status = "PENDING"
        detail = ""
        if done_path.exists():
            status = "DONE"
            try:
                data = _read_json(done_path)
                val_loss = data.get("val_loss")
                if val_loss is not None:
                    detail = f"val_loss={val_loss:.4f}"
            except Exception:
                detail = "done.json unreadable"
        elif latest_path.exists():
            status = "RESUMABLE"
        else:
            log_glob = list(job_dir.glob("*.log"))
            if log_glob:
                status = "RUNNING"
        entries.append((job_dir.name, status, detail))

    if not entries:
        print(f"No job directories found under {ckpt_root}")
        return

    print(f"Status for {run_dir}:")
    print(f"{'job':<60} {'status':<10} detail")
    for name, status, detail in entries:
        print(f"{name:<60} {status:<10} {detail}")


def _launch_single_task_subprocess(task: Tuple[Dict[str, Any], Dict[str, Any]], gpu_id: int, tmp_dir: Path, python_exe: str = None) -> Tuple[subprocess.Popen, Path, Dict[str, Any], int, Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Launch a subprocess pinned to one GPU to run a single task and write result JSON.

    Returns (proc, out_path, meta)
    """
    python_exe = python_exe or os.environ.get("PYTHON", "python")
    meta, cfg = task
    uid = uuid.uuid4().hex
    in_path = tmp_dir / f"task_{uid}_in.json"
    out_path = tmp_dir / f"task_{uid}_out.json"
    _write_json(in_path, {"meta": meta, "config": cfg})

    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # Respect parent's CUDA_VISIBLE_DEVICES mapping if present (may be UUIDs or a subset of indices)
    parent_visible = env.get("CUDA_VISIBLE_DEVICES")
    if parent_visible and parent_visible.strip() != "":
        ids = [s.strip() for s in parent_visible.split(",") if s.strip()]
        chosen = ids[gpu_id % len(ids)]
        env["CUDA_VISIBLE_DEVICES"] = chosen
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [python_exe, "-m", "scripts.run_shakespeare_ebc_ablation", "--single_task", str(in_path), "--single_task_out", str(out_path)]
    proc = subprocess.Popen(cmd, env=env)
    return proc, out_path, meta, gpu_id, task


def _drain_finished(procs: List[Tuple[subprocess.Popen, Path, Dict[str, Any], int, Tuple[Dict[str, Any], Dict[str, Any]]]]) -> List[Tuple[Dict[str, Any], Dict[str, Any], int]]:
    finished: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []
    still_running: List[Tuple[subprocess.Popen, Path, Dict[str, Any], int, Tuple[Dict[str, Any], Dict[str, Any]]]] = []
    for p, outp, meta, gid, task in procs:
        ret = p.poll()
        if ret is None:
            still_running.append((p, outp, meta, gid, task))
        else:
            if ret != 0:
                raise TaskFailed(meta, ret, outp, task)
            res = _read_json(outp)
            finished.append((meta, res, gid))
    procs[:] = still_running
    return finished


def main():
    p = argparse.ArgumentParser(description="Shakespeare EBC ablation")
    p.add_argument("--output_dir", type=Path, default=Path("outputs/ebc_ablation"))
    p.add_argument("--run_dir", type=Path, default=None, help="Optional: reuse an existing run directory to resume an interrupted sweep")
    p.add_argument("--status", action="store_true", help="Print status summary for a run directory and exit")
    # Model/data
    p.add_argument("--vocab_size", type=int, default=65)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_embed", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=3)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    # Train
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--val_iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    # Dtype / JIT
    p.add_argument("--model_dtype", type=str, default="float32", choices=["float32", "bfloat16", "float64"], help="Computation/param dtype")
    p.add_argument("--project_dtype", type=str, default="float32", choices=["float32", "bfloat16", "float64"], help="Projection ops dtype")
    p.add_argument("--spectral_backend", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Where to compute spectral normalization estimates")
    p.add_argument("--jit", action="store_true", help="Enable JIT compile of loss/backprop")
    # EBC defaults
    p.add_argument("--ebc_target_kl", type=float, default=0.05)
    p.add_argument("--ebc_update_every", type=int, default=20)
    p.add_argument("--ebc_probe_layers", type=int, default=2)
    p.add_argument("--ebc_beta_ema", type=float, default=0.9)
    p.add_argument("--ebc_safety", type=float, default=1.05)
    p.add_argument("--ebc_aggregate", type=str, default="l1", choices=["l1", "l2"])
    # EBC controller (optional)
    p.add_argument("--ebc_ctrl_enable", action="store_true", help="Enable applied-KL PI controller for EBC")
    p.add_argument("--ebc_ctrl_period", type=int, default=20, help="Probe/update controller every N steps")
    p.add_argument("--ebc_ctrl_kp", type=float, default=0.15, help="Proportional gain for KL controller")
    p.add_argument("--ebc_ctrl_ki", type=float, default=0.02, help="Integral gain for KL controller (EMA)")
    p.add_argument("--ebc_ctrl_ema_halflife", type=float, default=250, help="EMA half-life in steps for KL error integral")
    p.add_argument("--ebc_delta_star", type=float, default=None, help="Target per-token KL (nats/token) for controller; defaults to --ebc_target_kl")
    p.add_argument("--ebc_ctrl_delta_min", type=float, default=0.01, help="Min bound for controller's delta")
    p.add_argument("--ebc_ctrl_delta_max", type=float, default=0.30, help="Max bound for controller's delta")
    p.add_argument("--ebc_ctrl_log_every", action="store_true", help="Force logging applied-KL/rho each log step (diagnostic)")
    # EBC guard parameters (shrink-only tau scaling)
    p.add_argument("--ebc_guard_warmup_steps", type=int, default=100)
    p.add_argument("--ebc_tau_scale_floor", type=float, default=0.05)
    p.add_argument("--ebc_tau_max_shrink_per_probe", type=float, default=4.0)
    p.add_argument("--ebc_tau_recover_rate", type=float, default=1.10)
    p.add_argument("--ebc_tau_shrink_exponent", type=float, default=0.5)
    # EBC robust beta knobs
    p.add_argument("--ebc_beta_huber_delta", type=float, default=0.0, help="Huber clipping delta for beta residuals (0 to disable)")
    p.add_argument("--ebc_beta_full_sweep", type=int, default=200, help="Full beta refresh period in steps (0 to disable)")
    # Grid options
    p.add_argument("--optimizers", type=str, default="adam,muon")
    p.add_argument("--spectral", type=str, default="none,spec", help="none or spec, comma-separated")
    p.add_argument("--ebc", type=str, default="off,on", help="off or on, comma-separated")
    p.add_argument("--deltas", type=str, default="0.05,0.1", help="EBC KL targets when ebc=on")
    p.add_argument("--aggregates", type=str, default="l1", help="EBC aggregates when ebc=on (comma: l1,l2)")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel processes to run (within one GPU)")
    # Sharding across multiple OS processes/GPUs
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards (processes across GPUs)")
    p.add_argument("--shard_index", type=int, default=0, help="This process's shard index [0..num_shards-1]")
    # Auto tuning pipeline
    p.add_argument("--auto", action="store_true", help="Run automatic LR + EBC calibration + final runs")
    p.add_argument("--auto_warmup_steps", type=int, default=300)
    p.add_argument("--auto_val_interval", type=int, default=150)
    p.add_argument("--auto_val_iters", type=int, default=2)
    p.add_argument("--adam_lrs", type=str, default="1e-4,3e-4,1e-3")
    p.add_argument("--muon_lrs", type=str, default="3e-3,1e-2,2e-2")
    p.add_argument("--ebc_deltas_auto", type=str, default="0.5,1.0,2.0")
    p.add_argument("--ebc_aggregates_auto", type=str, default="l2,l1")
    p.add_argument("--ebc_lr_scales", type=str, default="1,1.5,2,3", help="LR multipliers to test for EBC headroom after delta selection")
    p.add_argument("--accept_lo", type=float, default=0.2, help="Lower bound on desired clip rate during calibration")
    p.add_argument("--accept_hi", type=float, default=0.6, help="Upper bound on desired clip rate during calibration")
    p.add_argument("--clip_target", type=float, default=None, help="Preferred clip rate target; defaults to midpoint of [accept_lo, accept_hi]")
    p.add_argument("--max_val_degrade", type=float, default=0.02, help="Max allowed relative val loss degrade vs baseline at warmup (e.g., 0.02 = 2%)")
    p.add_argument("--scale_max_val_degrade", type=float, default=0.03, help="Max allowed relative warmup val loss degrade vs baseline when scaling LR under EBC")
    # Adaptive delta search knobs
    p.add_argument("--auto_delta_min", type=float, default=0.01, help="Lower bound for adaptive delta search")
    p.add_argument("--auto_delta_max", type=float, default=8.0, help="Upper bound for adaptive delta search")
    p.add_argument("--auto_delta_expand", type=float, default=2.0, help="Multiplicative factor when expanding/shrinking delta adaptively")
    p.add_argument("--auto_delta_iters", type=int, default=3, help="Max additional attempts per base to adaptively search delta")
    p.add_argument("--auto_skip_threshold", type=float, default=0.2, help="Skip optimizers whose baseline val is worse than best by this relative fraction")
    p.add_argument("--task_retry", type=int, default=2, help="Number of retries for failed subprocess jobs in --auto mode")
    p.add_argument("--final_steps", type=int, default=None, help="Override final training steps for winners (defaults to --steps)")
    p.add_argument("--num_gpus", type=int, default=1, help="For --auto: concurrent tasks pinned one per GPU using CUDA_VISIBLE_DEVICES 0..N-1")
    # Checkpointing / resume
    p.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoints if present (default on)")
    p.add_argument("--ckpt_interval", type=int, default=200, help="Save a checkpoint every N optimizer steps")
    # Single-task worker mode (internal)
    p.add_argument("--single_task", type=Path, default=None)
    p.add_argument("--single_task_out", type=Path, default=None)

    # Optional controller/guard sweep (Stage 2c)
    p.add_argument("--auto_ctrl_search", action="store_true", help="Enable controller and guard grid search after delta selection")
    p.add_argument("--ctrl_kps", type=str, default="0.1,0.15,0.2")
    p.add_argument("--ctrl_kis", type=str, default="0.01,0.02")
    p.add_argument("--ctrl_halflifes", type=str, default="150,250,400")
    p.add_argument("--ctrl_delta_mins", type=str, default="0.005,0.01,0.02")
    p.add_argument("--ctrl_delta_maxs", type=str, default="0.10,0.15")
    p.add_argument("--guard_floors", type=str, default="0.05,0.10")
    p.add_argument("--guard_shrink_exponents", type=str, default="0.5,0.75")

    args = p.parse_args()
    # Single-task worker mode
    if args.single_task is not None:
        payload = _read_json(args.single_task)
        meta = payload["meta"]
        cfg = payload["config"]
        res = run_one(cfg)
        if args.single_task_out is None:
            print(json.dumps({"meta": meta, "result": res}))
        else:
            _write_json(args.single_task_out, res)
        return

    if args.status:
        target_dir = args.run_dir if args.run_dir is not None else args.output_dir
        print_status(target_dir)
        return

    if args.run_dir is not None:
        out_dir = args.run_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = args.output_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizers = [s.strip() for s in args.optimizers.split(",") if s.strip()]
    spectral_opts = [s.strip() for s in args.spectral.split(",") if s.strip()]
    ebc_opts = [s.strip() for s in args.ebc.split(",") if s.strip()]
    deltas = [float(s.strip()) for s in args.deltas.split(",") if s.strip()]
    aggregates = [s.strip() for s in args.aggregates.split(",") if s.strip()]

    summary_rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    print("optimizer,spectral,ebc,delta,aggregate,wall_clock,val_loss,val_acc,ppl,clip_rate,avg_c")

    # Warm dataset once (avoids parallel download race producing empty memmaps)
    warm_cfg_dict = make_config(args, optimizers[0], ebc=False, spectral=False, run_dir=out_dir, job_id="_warm")
    warm_config = parse_config_from_json(warm_cfg_dict)
    _warm_train, _warm_val, _ = get_data_loader(warm_config)
    del _warm_train
    del _warm_val

    tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for opt in optimizers:
        for spec in spectral_opts:
            spec_bool = spec == "spec"
            meta = {"optimizer": opt, "spectral": spec, "ebc": "off", "delta": None, "aggregate": None}
            label = job_label(meta)
            cfg = make_config(args, opt, ebc=False, spectral=spec_bool, run_dir=out_dir, job_id=label)
            tasks.append((meta, cfg))

            if "on" in ebc_opts:
                for delta in deltas:
                    for agg in aggregates:
                        meta = {
                            "optimizer": opt,
                            "spectral": spec,
                            "ebc": "on",
                            "delta": delta,
                            "aggregate": agg,
                        }
                        label = job_label(meta)
                        cfg = make_config(args, opt, ebc=True, spectral=spec_bool, delta=delta, aggregate=agg, run_dir=out_dir, job_id=label)
                        tasks.append((meta, cfg))

    def handle_result(meta: Dict[str, Any], res: Dict[str, Any]):
        summary_rows.append((meta, res))
        clip_display = "" if res["clip_rate"] is None else f"{res['clip_rate']:.2f}"
        avgc_display = "" if res["avg_c"] is None else f"{res['avg_c']:.3f}"
        print(
            f"{meta['optimizer']},{meta['spectral']},{meta['ebc']},{meta.get('delta')},{meta.get('aggregate')},{res['wall_clock']:.2f},{res['val_loss']:.4f},{res['val_acc']:.4f},{res['ppl']:.2f},{clip_display},{avgc_display}"
        )
        label = job_label(meta)
        title = job_display(meta)
        plot_run(res["history"], label, out_dir, title)

    # Shard tasks across multiple processes (one per GPU recommended)
    if args.num_shards > 1:
        tasks = [t for i, t in enumerate(tasks) if (i % args.num_shards) == args.shard_index]

    # Auto pipeline
    if args.auto:
        run_auto_pipeline(args, out_dir)
        print(f"Saved results to {out_dir}")
        return

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = [pool.submit(_run_task, t) for t in tasks]
            for fut in as_completed(futures):
                meta, res = fut.result()
                handle_result(meta, res)
    else:
        for meta, cfg in tasks:
            res = run_one(cfg)
            handle_result(meta, res)

    # Save CSV summary
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("optimizer,spectral,ebc,delta,aggregate,wall_clock,val_loss,val_acc,ppl,clip_rate,avg_c\n")
        for meta, res in summary_rows:
            f.write(
                f"{meta['optimizer']},{meta['spectral']},{meta['ebc']},{'' if meta['delta'] is None else meta['delta']},{'' if meta['aggregate'] is None else meta['aggregate']},{res['wall_clock']:.2f},{res['val_loss']:.6f},{res['val_acc']:.6f},{res['ppl']:.3f},{res['clip_rate']},{res['avg_c']}\n"
            )

    # Simple Pareto: val PPL vs avg clip (EBC only)
    xs, ys, labels = [], [], []
    for meta, res in summary_rows:
        if meta["ebc"] == "on" and res["avg_c"] is not None:
            xs.append(res["avg_c"])  # larger avg_c -> less clipping
            ys.append(res["ppl"])
            labels.append(f"{meta['optimizer']}-{meta['spectral']}-d{meta['delta']}-{meta['aggregate']}")

    if xs:
        plt.figure(figsize=(6, 4))
        plt.scatter(xs, ys)
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y), fontsize=8)
        plt.xlabel("avg c (higher = fewer clips)")
        plt.ylabel("Val perplexity")
        plt.title("EBC: PPL vs avg clip")
        plt.tight_layout()
        plt.savefig(out_dir / "pareto_ppl_vs_avgc.png", dpi=160)

    # Dump JSON of configs and metrics for reproducing
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(
            [
                {
                    "optimizer": meta["optimizer"],
                    "spectral": meta["spectral"],
                    "ebc": meta["ebc"],
                    "delta": meta["delta"],
                    "aggregate": meta["aggregate"],
                    "metrics": res,
                }
                for meta, res in summary_rows
            ],
            f,
            indent=2,
        )

    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
