from modula.atom import orthogonalize, hard_cap, soft_cap, pure_svd
from modula.atom import spectral_hammer, spectral_weight_decay, spectral_normalize
import jax.numpy as jnp


class Config:
    """Configuration container with attribute access."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def copy(self):
        """Return a copy of the configuration."""
        return {k: v for k, v in self.__dict__.items()}


# Project function mapping
PROJECT_FUNCTIONS = {
    "none": lambda x, **kwargs: x,
    "orthogonal": orthogonalize,
    "hard_cap": hard_cap,
    "soft_cap": soft_cap,
    "pure_svd": pure_svd,
    "spec_hammer": spectral_hammer,
    "spec_wd": spectral_weight_decay,
    "spec_normalize": spectral_normalize,
}

# Data type mapping
DTYPES = {
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
    "float64": jnp.float64,
}


def parse_config_from_json(config_dict):
    """Convert JSON config dict to Config object with proper type conversions."""
    config = Config(**config_dict)

    # Set reference to mappings
    config.project_fn_map = PROJECT_FUNCTIONS
    config.dtype = DTYPES[config.model_dtype]
    config.project_dtype = DTYPES[config.project_dtype]

    # EBC defaults
    if not hasattr(config, "ebc_enable"):
        config.ebc_enable = False
    if not hasattr(config, "ebc_target_kl"):
        config.ebc_target_kl = 0.05  # nats/token
    if not hasattr(config, "ebc_update_every"):
        config.ebc_update_every = 20
    if not hasattr(config, "ebc_probe_layers"):
        config.ebc_probe_layers = 2
    if not hasattr(config, "ebc_beta_ema"):
        config.ebc_beta_ema = 0.9
    if not hasattr(config, "ebc_beta_huber_delta"):
        # Set to 0 to disable Huber; >0 applies clipped residual against EMA
        config.ebc_beta_huber_delta = 0.0
    if not hasattr(config, "ebc_beta_full_sweep"):
        # Full beta refresh period in steps (0 to disable)
        config.ebc_beta_full_sweep = 200
    if not hasattr(config, "ebc_safety"):
        config.ebc_safety = 1.05
    if not hasattr(config, "ebc_aggregate"):
        config.ebc_aggregate = "l1"
    if not hasattr(config, "ebc_center_logits"):
        config.ebc_center_logits = True
    if not hasattr(config, "ebc_include_embed_out"):
        config.ebc_include_embed_out = False

    # EBC controller defaults (disabled by default)
    if not hasattr(config, "ebc_ctrl_enable"):
        config.ebc_ctrl_enable = False
    if not hasattr(config, "ebc_ctrl_period"):
        config.ebc_ctrl_period = 20
    if not hasattr(config, "ebc_ctrl_kp"):
        config.ebc_ctrl_kp = 0.15
    if not hasattr(config, "ebc_ctrl_ki"):
        config.ebc_ctrl_ki = 0.02
    if not hasattr(config, "ebc_ctrl_ema_halflife"):
        config.ebc_ctrl_ema_halflife = 250
    if not hasattr(config, "ebc_delta_star"):
        # target applied per-token KL (nats/token) that controller will track
        config.ebc_delta_star = config.ebc_target_kl
    if not hasattr(config, "ebc_ctrl_delta_min"):
        config.ebc_ctrl_delta_min = 0.01
    if not hasattr(config, "ebc_ctrl_delta_max"):
        config.ebc_ctrl_delta_max = 0.30
    if not hasattr(config, "ebc_ctrl_log_every"):
        # If True, compute and log applied_KL/rho every log_interval for diagnostics
        config.ebc_ctrl_log_every = False

    # Trainer defaults
    if not hasattr(config, "pre_dualize"):
        config.pre_dualize = False
    if not hasattr(config, "post_dualize"):
        config.post_dualize = False
    if not hasattr(config, "jit"):
        config.jit = True
    if not hasattr(config, "spectral_backend"):
        config.spectral_backend = "auto"

    return config
