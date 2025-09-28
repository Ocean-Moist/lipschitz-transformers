import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from configs import parse_config_from_json
from models import create_model
from optimizers import get_optimizer
from trainer import Trainer


def toy_lm_loader(vocab_size=32, batch_size=8, seq_len=16, seed=0):
    key = jax.random.PRNGKey(seed)

    def iterator():
        nonlocal key
        while True:
            key, sk = jax.random.split(key)
            x = jax.random.randint(sk, (batch_size, seq_len), 0, vocab_size)
            y = x
            yield x, y

    return iterator()


def lm_cross_entropy(model, w, inputs, targets):
    logits = model(inputs, w)
    logits = logits.astype(jnp.float32)
    logZ = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    log_probs = logits - logZ
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return jnp.mean(nll)


def run_history(cfg: Dict[str, Any]) -> Tuple[list, list, list, list]:
    config = parse_config_from_json(cfg)
    train_loader = toy_lm_loader(config.vocab_size, config.batch_size, config.seq_len, config.seed)
    val_loader = toy_lm_loader(config.vocab_size, config.batch_size, config.seq_len, config.seed + 1)

    model = create_model(config)
    optimizer = get_optimizer(config)

    class LogAll:
        def __init__(self):
            self.steps, self.losses, self.accs, self.cs = [], [], [], []
        def log_training(self, step, loss, acc, log):
            self.steps.append(step)
            self.losses.append(float(loss))
            self.accs.append(float(acc))
            c = None
            if isinstance(log, dict) and "ebc" in log and isinstance(log["ebc"], dict):
                c = log["ebc"].get("c")
            self.cs.append(c)
        def log_validation(self, step, metrics):
            pass
        def get_results(self):
            return {}

    logger = LogAll()

    key = jax.random.PRNGKey(config.seed)
    key, sub = jax.random.split(key)
    params = model.initialize(sub)
    opt_state = optimizer.init_state(params)

    trainer = Trainer(model, optimizer, train_loader, val_loader, lm_cross_entropy, config, logger)
    params, opt_state, key = trainer.train(params, opt_state, key)
    return logger.steps, logger.losses, logger.accs, logger.cs


def base_cfg():
    return dict(
        data="shakespeare",
        vocab_size=32,
        num_heads=2,
        d_embed=16,
        num_blocks=1,
        softmax_scale=1.0,
        final_scale=1.0,
        residual_scale=1.0,
        zero_init=False,
        use_unembed=True,
        layernorm_substitute="none",
        batch_size=8,
        seq_len=16,
        steps=60,
        accum_steps=1,
        lr=1e-3,
        wd=0.0,
        spectral_wd=0.0,
        w_max=1.0,
        schedule="none",
        log_interval=1,
        val_interval=10_000,
        val_iters=1,
        num_checkpoints=0,
        seed=0,
        optimizer="adam",
        beta1=0.9,
        beta2=0.999,
        model_dtype="float32",
        project_dtype="float32",
        project={"default": "none"},
        ebc_enable=False,
        jit=False,
        output_dir="outputs",
    )


def plot_curves():
    # 1) Baseline vs EBC variants (default LR)
    cfgs = []
    labels = []
    base = base_cfg()
    cfgs.append(base)
    labels.append("baseline")

    ebc1 = {**base, "ebc_enable": True, "ebc_target_kl": 0.05}
    cfgs.append(ebc1)
    labels.append("EBC δ=0.05 L1")

    ebc2 = {**base, "ebc_enable": True, "ebc_target_kl": 0.2}
    cfgs.append(ebc2)
    labels.append("EBC δ=0.2 L1")

    ebc3 = {**base, "ebc_enable": True, "ebc_target_kl": 0.2, "ebc_aggregate": "l2"}
    cfgs.append(ebc3)
    labels.append("EBC δ=0.2 L2")

    series = []
    for cfg in cfgs:
        steps, losses, accs, cs = run_history(cfg)
        series.append((steps, losses, accs, cs))

    plt.figure(figsize=(8, 5))
    for (steps, losses, _, _), lab in zip(series, labels):
        plt.plot(steps, losses, label=lab)
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title("EBC variants (default LR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/ebc_loss_default_lr.png", dpi=160)

    # Clip factor for EBC runs
    plt.figure(figsize=(8, 3))
    for (steps, _, _, cs), lab in zip(series[1:], labels[1:]):
        plt.plot(steps, [c if c is not None else 1.0 for c in cs], label=lab)
    plt.xlabel("Step")
    plt.ylabel("c (clip)")
    plt.ylim(0, 1.05)
    plt.title("EBC clip factor")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/ebc_clip_default_lr.png", dpi=160)

    # 2) High LR stability: baseline vs EBC
    hlr_base = {**base, "lr": 5e-2}
    hlr_ebc = {**base, "lr": 5e-2, "ebc_enable": True, "ebc_target_kl": 0.1}
    s1 = run_history(hlr_base)
    s2 = run_history(hlr_ebc)

    plt.figure(figsize=(8, 5))
    plt.plot(s1[0], s1[1], label="baseline (lr=0.05)")
    plt.plot(s2[0], s2[1], label="EBC (lr=0.05, δ=0.1)")
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title("High LR stability: EBC vs baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/ebc_loss_high_lr.png", dpi=160)


if __name__ == "__main__":
    plot_curves()

