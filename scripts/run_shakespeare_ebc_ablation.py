import argparse
import json
import math
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

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


def make_config(args, optimizer: str, ebc: bool, spectral: bool, delta: float = None, aggregate: str = None):
    # Project mapping
    if spectral:
        project = {"default": "spec_normalize"}
    else:
        project = {"default": "none"}

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
        model_dtype="float32",
        project_dtype="float32",
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
        jit=False,  # keep False to avoid Rope cache tracer issues under JIT
        output_dir=str(args.output_dir),
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

    trainer = Trainer(model, optimizer, train_loader, val_loader, loss_fn, config, logger)

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

    return {
        "wall_clock": wall_clock,
        "val_loss": float(val_metrics["loss"]),
        "val_acc": float(val_metrics["accuracy"]),
        "ppl": ppl,
        "history": history,
        "accept_rate": accept_rate,
        "avg_c": avg_c,
    }


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


def main():
    p = argparse.ArgumentParser(description="Shakespeare EBC ablation")
    p.add_argument("--output_dir", type=Path, default=Path("outputs/ebc_ablation"))
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
    p.add_argument("--workers", type=int, default=1, help="Number of parallel processes to run")

    args = p.parse_args()
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
    warm_cfg_dict = make_config(args, optimizers[0], ebc=False, spectral=False)
    warm_config = parse_config_from_json(warm_cfg_dict)
    _warm_train, _warm_val, _ = get_data_loader(warm_config)
    del _warm_train
    del _warm_val

    tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for opt in optimizers:
        for spec in spectral_opts:
            spec_bool = spec == "spec"
            meta = {"optimizer": opt, "spectral": spec, "ebc": "off", "delta": None, "aggregate": None}
            cfg = make_config(args, opt, ebc=False, spectral=spec_bool)
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
                        cfg = make_config(args, opt, ebc=True, spectral=spec_bool, delta=delta, aggregate=agg)
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

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            for meta, res in pool.map(_run_task, tasks):
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
