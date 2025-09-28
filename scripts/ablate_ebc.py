import time
import itertools
import jax
import jax.numpy as jnp

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
            # Next-token LM (shift targets left by one; last token arbitrary)
            y = x
            yield x, y

    return iterator()


def lm_cross_entropy(model, w, inputs, targets):
    logits = model(inputs, w)  # [B, T, V]
    logits = logits.astype(jnp.float32)
    logZ = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    log_probs = logits - logZ
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return jnp.mean(nll)


def run_job(config_dict):
    config = parse_config_from_json(config_dict)

    train_loader = toy_lm_loader(
        vocab_size=config.vocab_size, batch_size=config.batch_size, seq_len=config.seq_len, seed=config.seed
    )
    val_loader = toy_lm_loader(
        vocab_size=config.vocab_size, batch_size=config.batch_size, seq_len=config.seq_len, seed=config.seed + 1
    )

    model = create_model(config)
    optimizer = get_optimizer(config)

    class DummyLogger:
        def __init__(self, config):
            self.config = config
            self.history = []
        def log_training(self, step, loss, acc, log):
            self.history.append((step, float(loss), float(acc), log))
        def log_validation(self, step, metrics):
            pass
        def get_results(self):
            return {"history": self.history}

    logger = DummyLogger(config)

    key = jax.random.PRNGKey(config.seed)
    key, sub = jax.random.split(key)
    params = model.initialize(sub)
    opt_state = optimizer.init_state(params)

    trainer = Trainer(model, optimizer, train_loader, val_loader, lm_cross_entropy, config, logger)
    params, opt_state, key = trainer.train(params, opt_state, key)
    results = logger.get_results()
    # Extract last EBC clip factor if present
    last = results["history"][-1] if results["history"] else None
    ebc_c = last[3].get("ebc", {}).get("c") if last else None
    return {
        "final_loss": last[1] if last else None,
        "final_acc": last[2] if last else None,
        "ebc_c": ebc_c,
    }


def make_config(optimizer: str, ebc_enable: bool, spectral: bool):
    project = {"default": "spec_normalize"} if spectral else {"default": "none"}
    return dict(
        # Model/data
        data="shakespeare",
        vocab_size=32,
        num_heads=2,
        d_embed=16,
        num_blocks=1,
        softmax_scale=1.0,
        final_scale=1.0,
        residual_scale=1.0,
        scales_learnable=False,
        zero_init=False,
        max_embed_inflation_factor=1.0,
        use_unembed=True,
        layernorm_substitute="none",
        # Training
        batch_size=8,
        seq_len=16,
        steps=30,
        accum_steps=1,
        lr=1e-3,
        wd=0.0,
        spectral_wd=0.0,
        w_max=1.0,
        schedule="none",
        log_interval=10,
        val_interval=1000,
        val_iters=1,
        num_checkpoints=0,
        seed=0,
        jit=False,
        # Optimizer
        optimizer=optimizer,
        beta1=0.9,
        beta2=0.999,
        # Dtypes
        model_dtype="float32",
        project_dtype="float32",
        project=project,
        # EBC
        ebc_enable=ebc_enable,
        ebc_target_kl=0.05,
        ebc_update_every=10,
        ebc_probe_layers=1,
        ebc_beta_ema=0.9,
        ebc_safety=1.05,
        ebc_aggregate="l1",
        ebc_center_logits=True,
        ebc_include_embed_out=False,
        # Output
        output_dir="outputs",
    )


def main():
    grid = list(itertools.product(["adam", "muon"], [False, True], [False, True]))
    print("optimizer,ebc,spectral,final_loss,final_acc,ebc_c")
    for optimizer, ebc, spectral in grid:
        cfg = make_config(optimizer, ebc, spectral)
        res = run_job(cfg)
        print(
            f"{optimizer},{int(ebc)},{int(spectral)},{res['final_loss']},{res['final_acc']},{res['ebc_c']}"
        )


if __name__ == "__main__":
    main()
