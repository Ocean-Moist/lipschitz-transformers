import jax
import jax.numpy as jnp

from modula.compound import GPT
from modula.ebc import (
    get_scope_indices,
    estimate_beta_jvp,
    apply_ebc_clipping,
    centered_logit_norm,
)


def _make_tiny_gpt():
    return GPT(
        vocab_size=32,
        num_heads=2,
        d_embed=16,
        num_blocks=1,
        blocks_mass=1,
        dtype=jnp.float32,
        project_dtype=jnp.float32,
        softmax_scale=1.0,
        final_scale=1.0,
        residual_scale=1.0,
        zero_init=False,
        project=None,
        max_embed_inflation_factor=1.0,
        use_unembed=True,
        layernorm_substitute="none",
    )


def _random_updates_like(params, key, scale=1e-3):
    ups = []
    for i, w in enumerate(params):
        key, sub = jax.random.split(key)
        ups.append(scale * jax.random.normal(sub, w.shape, dtype=w.dtype))
    return ups


def test_ebc_clipping_scales_to_tau_predicted():
    key = jax.random.PRNGKey(0)
    model = _make_tiny_gpt()
    key, sub = jax.random.split(key)
    params = model.initialize(sub)

    # Probe batch (tokens)
    key, sub = jax.random.split(key)
    inputs = jax.random.randint(sub, (2, 4), 0, 32)

    # Synthetic proposed update
    key, sub = jax.random.split(key)
    updates = _random_updates_like(params, sub, scale=5e-3)

    trackers = model.trackers()
    scope = get_scope_indices(trackers, include_embed_out=False)

    # Estimate betas for scoped layers via JVP along update direction
    betas = [0.0 for _ in trackers]
    for idx in scope:
        b = estimate_beta_jvp(model, params, idx, updates[idx], inputs, center_logits=True)
        betas[idx] = float(b)

    # Compute a tau that forces clipping to a predictable factor
    # First compute S without clipping (safety=1.0; L1 aggregate)
    clipped, S, c = apply_ebc_clipping(updates, betas, tau=jnp.array(1e9), scope_indices=scope, safety=1.0, aggregate="l1")
    assert float(c) == 1.0
    assert float(S) > 0

    tau = 0.5 * S  # expect c ~ 0.5
    clipped, S2, c2 = apply_ebc_clipping(updates, betas, tau=tau, scope_indices=scope, safety=1.0, aggregate="l1")

    # Predicted budget should be clipped at tau
    assert 0 < float(c2) < 1.0
    assert float(S2 * c2 / c2) == float(S2)  # no-op check for jit values
    assert float(c2 * S) <= float(tau) + 1e-6

    # Optional: check actual logit change roughly respects tau (allow slack for nonlinearity)
    # Compute logits difference after applying clipped updates
    params_new = [w + dw for (w, dw) in zip(params, clipped)]
    logits_old = model(inputs, params)
    logits_new = model(inputs, params_new)
    dlogits = logits_new - logits_old
    measured = centered_logit_norm(dlogits)
    assert float(measured) <= float(tau) * 1.5  # conservative slack
