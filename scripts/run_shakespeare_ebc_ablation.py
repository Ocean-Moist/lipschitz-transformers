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
        self.wall_times: List[float] = []
        self._t0 = time.time()

    def log_training(self, step, loss, accuracy, log):
        self.steps.append(int(step))
        self.losses.append(float(loss))
        self.train_accs.append(float(accuracy))
        c = None
        if isinstance(log, dict) and "ebc" in log and isinstance(log["ebc"], dict):
            c = log["ebc"].get("c")
        self.ebc_c.append(None if c is None else float(c))
        self.wall_times.append(time.time() - self._t0)

    def log_validation(self, step, metrics):
        # For now we rely on final validate() at the end
        pass

    def get_results(self):
        return {
            "steps": self.steps,
            "train_losses": self.losses,
            "train_accs": self.train_accs,
            "ebc_c": self.ebc_c,
            "wall_times": self.wall_times,
        }


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
        # Misc
        jit=bool(args.jit),  # enable with --jit; default False due to RoPE tracer note
        output_dir=str(out_root),
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
    # Accept rate / avg clip
    cs = [c for c in history["ebc_c"] if c is not None]
    accept_rate = None
    avg_c = None
    if cs:
        accept_rate = float(sum(1 for c in cs if c >= 0.999) / len(cs))
        avg_c = float(sum(cs) / len(cs))

    result = {
        "wall_clock": wall_clock,
        "val_loss": float(val_metrics["loss"]),
        "val_acc": float(val_metrics["accuracy"]),
        "ppl": ppl,
        "history": history,
        "accept_rate": accept_rate,
        "avg_c": avg_c,
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
    - For each base, sweep deltas/aggregates for EBC for warmup steps using the chosen LR; pick candidate with accept_rate in [accept_lo, accept_hi] and best val; fallback to best val not worse than baseline by max_val_degrade; otherwise skip EBC for that base.
    - Run final full steps for the baseline and EBC winner per base.
    """
    optimizers = [s.strip() for s in args.optimizers.split(",") if s.strip()]
    spectral_opts = [s.strip() for s in args.spectral.split(",") if s.strip()]

    adam_lrs = [float(x) for x in args.adam_lrs.split(",") if x]
    muon_lrs = [float(x) for x in args.muon_lrs.split(",") if x]
    lr_bracket = {"adam": adam_lrs, "muon": muon_lrs}
    delta_candidates = [float(x) for x in args.ebc_deltas_auto.split(",") if x]
    agg_candidates = [s.strip() for s in args.ebc_aggregates_auto.split(",") if s.strip()]

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
        active: List[Tuple[subprocess.Popen, Path, Dict[str, Any], int]] = []
        results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        next_task_idx = 0
        # Pre-fill
        for gid in range(min(effective_gpus, len(task_list))):
            proc = _launch_single_task_subprocess(task_list[next_task_idx], gid, tmp_dir)
            active.append(proc)
            next_task_idx += 1

        while active:
            finished = _drain_finished(active)
            for meta, res, gid in finished:
                results.append((meta, res))
                # launch next
                if next_task_idx < len(task_list):
                    proc = _launch_single_task_subprocess(task_list[next_task_idx], gid, tmp_dir)
                    active.append(proc)
                    next_task_idx += 1
            # be gentle
            time.sleep(0.2)
        return results

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
    print("[auto] LR sweep results:")
    for meta, res in lr_results:
        key = (meta["optimizer"], meta["spectral"])
        v = res["val_loss"]
        if key not in best_lr or v < best_lr[key][0]:
            best_lr[key] = (v, meta)
            baseline_val[key] = v
        print(f"  {key} lr={meta['lr']}: val_loss={v:.4f}")

    # Stage 2: EBC calibration using chosen LR
    ebc_tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for opt in optimizers:
        for spec in spectral_opts:
            spec_bool = (spec == "spec")
            key = (opt, spec)
            lr = best_lr[key][1]["lr"] if key in best_lr else args.lr
            for delta in delta_candidates:
                for agg in agg_candidates:
                    meta = {"stage": "calib", "optimizer": opt, "spectral": spec, "ebc": "on", "delta": delta, "aggregate": agg, "lr": lr}
                    label = f"calib_{job_label(meta)}"
                    cfg = make_config(clone_args(warm_args, lr=lr), opt, ebc=True, spectral=spec_bool, delta=delta, aggregate=agg, run_dir=out_dir, job_id=label)
                    ebc_tasks.append((meta, cfg))

    ebc_results = launch_queue(ebc_tasks)

    # Pick EBC winner per base
    winners: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = { (o,s): None for o in optimizers for s in spectral_opts }
    print("[auto] EBC calibration results:")
    for meta, res in ebc_results:
        key = (meta["optimizer"], meta["spectral"])
        ar = res.get("accept_rate")
        avgc = res.get("avg_c")
        v = res["val_loss"]
        print(f"  {key} delta={meta['delta']} {meta['aggregate']} lr={meta['lr']}: val={v:.4f} acc_rate={ar} avg_c={avgc}")

    # Decide best per base using band + loss
    grouped: Dict[Tuple[str,str], List[Tuple[Dict[str,Any], Dict[str,Any]]]] = {}
    for meta, res in ebc_results:
        key = (meta["optimizer"], meta["spectral"])
        grouped.setdefault(key, []).append((meta, res))

    for key, items in grouped.items():
        base_val = baseline_val.get(key, float("inf"))
        band = []
        best_any = None
        for meta, res in items:
            ar = res.get("accept_rate")
            if ar is not None and args.accept_lo <= ar <= args.accept_hi:
                band.append((res["val_loss"], meta))
            # track best val overall
            if (best_any is None) or (res["val_loss"] < best_any[0]):
                best_any = (res["val_loss"], meta)
        pick = None
        if band:
            # pick lowest val in band
            pick = sorted(band, key=lambda x: x[0])[0][1]
        else:
            # allow small degrade vs baseline
            if best_any and (best_any[0] <= base_val * (1 + args.max_val_degrade)):
                pick = best_any[1]
        winners[key] = pick

    # Stage 3: Final full runs for baseline + winners
    final_steps = int(args.final_steps or args.steps)
    final_args = clone_args(args, steps=final_steps)
    final_tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for opt in optimizers:
        for spec in spectral_opts:
            spec_bool = (spec == "spec")
            key = (opt, spec)
            lr = best_lr[key][1]["lr"] if key in best_lr else args.lr
            # baseline
            base_meta = {"stage": "final", "optimizer": opt, "spectral": spec, "ebc": "off", "delta": None, "aggregate": None, "lr": lr}
            base_label = f"final_{job_label(base_meta)}"
            base_cfg = make_config(clone_args(final_args, lr=lr), opt, ebc=False, spectral=spec_bool, run_dir=out_dir, job_id=base_label)
            final_tasks.append((base_meta, base_cfg))
            # ebc winner
            if winners.get(key):
                m = winners[key]
                ebc_meta = {"stage": "final", "optimizer": opt, "spectral": spec, "ebc": "on", "delta": m["delta"], "aggregate": m["aggregate"], "lr": lr}
                ebc_label = f"final_{job_label(ebc_meta)}"
                ebc_cfg = make_config(clone_args(final_args, lr=lr), opt, ebc=True, spectral=spec_bool, delta=m["delta"], aggregate=m["aggregate"], run_dir=out_dir, job_id=ebc_label)
                final_tasks.append((ebc_meta, ebc_cfg))

    final_results = launch_queue(final_tasks)

    # Write summaries/plots
    print("optimizer,spectral,ebc,delta,aggregate,wall_clock,val_loss,val_acc,ppl,accept_rate,avg_c")
    out_summary: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for meta, res in final_results:
        out_summary.append((meta, res))
        accept_display = "" if res["accept_rate"] is None else f"{res['accept_rate']:.2f}"
        avgc_display = "" if res["avg_c"] is None else f"{res['avg_c']:.3f}"
        print(
            f"{meta['optimizer']},{meta['spectral']},{meta['ebc']},{meta.get('delta')},{meta.get('aggregate')},{res['wall_clock']:.2f},{res['val_loss']:.4f},{res['val_acc']:.4f},{res['ppl']:.2f},{accept_display},{avgc_display}"
        )
        plot_run(res["history"], job_label(meta), out_dir, job_display(meta))

    # Save CSV summary for finals
    csv_path = out_dir / "summary_final.csv"
    with open(csv_path, "w") as f:
        f.write("optimizer,spectral,ebc,delta,aggregate,lr,wall_clock,val_loss,val_acc,ppl,accept_rate,avg_c\n")
        for meta, res in out_summary:
            f.write(
                f"{meta['optimizer']},{meta['spectral']},{meta['ebc']},{'' if meta['delta'] is None else meta['delta']},{'' if meta['aggregate'] is None else meta['aggregate']},{meta.get('lr')},{res['wall_clock']:.2f},{res['val_loss']:.6f},{res['val_acc']:.6f},{res['ppl']:.3f},{res['accept_rate']},{res['avg_c']}\n"
            )

    # Dump JSON
    json_path = out_dir / "results_final.json"
    with open(json_path, "w") as f:
        json.dump([
            {"meta": meta, "metrics": res} for meta, res in out_summary
        ], f, indent=2)

    # Decisions dump
    decisions = {
        "best_lr": {"|".join(k): v[1]["lr"] for k, v in best_lr.items()},
        "winners": {"|".join(k): winners[k] for k in winners},
        "settings": {
            "accept_band": [args.accept_lo, args.accept_hi],
            "max_val_degrade": args.max_val_degrade,
            "warmup_steps": args.auto_warmup_steps,
            "final_steps": final_steps,
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


def _launch_single_task_subprocess(task: Tuple[Dict[str, Any], Dict[str, Any]], gpu_id: int, tmp_dir: Path, python_exe: str = None) -> Tuple[subprocess.Popen, Path, Dict[str, Any], int]:
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
    return proc, out_path, meta, gpu_id


def _drain_finished(procs: List[Tuple[subprocess.Popen, Path, Dict[str, Any], int]]) -> List[Tuple[Dict[str, Any], Dict[str, Any], int]]:
    finished: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []
    still_running: List[Tuple[subprocess.Popen, Path, Dict[str, Any], int]] = []
    for p, outp, meta, gid in procs:
        ret = p.poll()
        if ret is None:
            still_running.append((p, outp, meta, gid))
        else:
            if ret != 0:
                raise RuntimeError(f"Subprocess failed for {meta} with return code {ret}")
            res = _read_json(outp)
            finished.append((meta, res, gid))
    procs[:] = still_running
    return finished


def main():
    p = argparse.ArgumentParser(description="Shakespeare EBC ablation")
    p.add_argument("--output_dir", type=Path, default=Path("outputs/ebc_ablation"))
    p.add_argument("--run_dir", type=Path, default=None, help="Optional: reuse an existing run directory to resume an interrupted sweep")
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
    p.add_argument("--jit", action="store_true", help="Enable JIT compile of loss/backprop")
    # EBC defaults
    p.add_argument("--ebc_target_kl", type=float, default=0.05)
    p.add_argument("--ebc_update_every", type=int, default=20)
    p.add_argument("--ebc_probe_layers", type=int, default=2)
    p.add_argument("--ebc_beta_ema", type=float, default=0.9)
    p.add_argument("--ebc_safety", type=float, default=1.05)
    p.add_argument("--ebc_aggregate", type=str, default="l1", choices=["l1", "l2"])
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
    p.add_argument("--accept_lo", type=float, default=0.1)
    p.add_argument("--accept_hi", type=float, default=0.7)
    p.add_argument("--max_val_degrade", type=float, default=0.02, help="Max allowed relative val loss degrade vs baseline at warmup (e.g., 0.02 = 2%)")
    p.add_argument("--final_steps", type=int, default=None, help="Override final training steps for winners (defaults to --steps)")
    p.add_argument("--num_gpus", type=int, default=1, help="For --auto: concurrent tasks pinned one per GPU using CUDA_VISIBLE_DEVICES 0..N-1")
    # Checkpointing / resume
    p.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoints if present (default on)")
    p.add_argument("--ckpt_interval", type=int, default=200, help="Save a checkpoint every N optimizer steps")
    # Single-task worker mode (internal)
    p.add_argument("--single_task", type=Path, default=None)
    p.add_argument("--single_task_out", type=Path, default=None)

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
    print("optimizer,spectral,ebc,delta,aggregate,wall_clock,val_loss,val_acc,ppl,accept_rate,avg_c")

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
        accept_display = "" if res["accept_rate"] is None else f"{res['accept_rate']:.2f}"
        avgc_display = "" if res["avg_c"] is None else f"{res['avg_c']:.3f}"
        print(
            f"{meta['optimizer']},{meta['spectral']},{meta['ebc']},{meta.get('delta')},{meta.get('aggregate')},{res['wall_clock']:.2f},{res['val_loss']:.4f},{res['val_acc']:.4f},{res['ppl']:.2f},{accept_display},{avgc_display}"
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
        f.write("optimizer,spectral,ebc,delta,aggregate,wall_clock,val_loss,val_acc,ppl,accept_rate,avg_c\n")
        for meta, res in summary_rows:
            f.write(
                f"{meta['optimizer']},{meta['spectral']},{meta['ebc']},{'' if meta['delta'] is None else meta['delta']},{'' if meta['aggregate'] is None else meta['aggregate']},{res['wall_clock']:.2f},{res['val_loss']:.6f},{res['val_acc']:.6f},{res['ppl']:.3f},{res['accept_rate']},{res['avg_c']}\n"
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
