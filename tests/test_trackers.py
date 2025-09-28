import jax
import jax.numpy as jnp

from modula.compound import GPT


def test_gpt_trackers_count_matches_params():
    # Tiny GPT
    model = GPT(
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

    key = jax.random.PRNGKey(0)
    params = model.initialize(key)

    trackers = model.trackers()

    assert isinstance(trackers, list)
    assert len(trackers) == len(params), (
        f"Trackers length {len(trackers)} must match params length {len(params)}"
    )

    # Expect some well-known labels to be present
    assert "embed" in trackers
    assert "out" in trackers
    # Per-layer labels (single block)
    assert any(l.startswith("q0") for l in trackers)
    assert any(l.startswith("k0") for l in trackers)
    assert any(l.startswith("v0") for l in trackers)
    assert any(l.startswith("w0") for l in trackers)
    assert any(l.startswith("mlp_in0") for l in trackers)
    assert any(l.startswith("mlp_out0") for l in trackers)

