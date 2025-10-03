import jax
import jax.numpy as jnp
from typing import List, Tuple


def pytree_rms_norm(tree) -> jnp.ndarray:
    leaves, _ = jax.tree_util.tree_flatten(tree)
    if not leaves:
        return jnp.array(0.0)
    total_sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return jnp.sqrt(total_sq)


def centered_logit_norm(delta_logits: jnp.ndarray) -> jnp.ndarray:
    mean_shift = jnp.mean(delta_logits, axis=-1, keepdims=True)
    centered = delta_logits - mean_shift
    return jnp.sqrt(jnp.sum(jnp.square(centered)))


def tau_from_kl(delta_per_token: float, total_tokens: int) -> jnp.ndarray:
    if delta_per_token <= 0 or total_tokens <= 0:
        return jnp.inf
    return jnp.sqrt(4.0 * float(total_tokens) * float(delta_per_token))


def get_scope_indices(trackers: List[str], include_embed_out: bool = False) -> List[int]:
    scope = []
    for i, label in enumerate(trackers):
        if include_embed_out:
            scope.append(i)
        else:
            if label not in ("embed", "out"):
                scope.append(i)
    return scope


def estimate_beta_jvp(
    model,
    params,
    layer_idx: int,
    dW_layer,
    inputs,
    center_logits: bool = True,
) -> jnp.ndarray:
    norm_l = pytree_rms_norm(dW_layer)
    if float(norm_l) < 1e-12:
        return jnp.array(0.0)
    U_l = jax.tree_util.tree_map(lambda x: x / norm_l, dW_layer)

    def f(full_params):
        return model(inputs, full_params)

    zeros_like_params = jax.tree_util.tree_map(jnp.zeros_like, params)

    def insert_layer(tree, idx, val):
        out = list(tree)
        out[idx] = val
        return out

    tangents = insert_layer(zeros_like_params, layer_idx, U_l)
    _, dlogits = jax.jvp(f, (params,), (tangents,))
    return centered_logit_norm(dlogits) if center_logits else jnp.sqrt(jnp.sum(dlogits**2))


def apply_ebc_clipping(
    updates,
    betas: List[float],
    tau: jnp.ndarray,
    scope_indices: List[int],
    safety: float = 1.05,
    aggregate: str = "l1",
) -> Tuple[object, jnp.ndarray, jnp.ndarray]:
    impacts = []
    for idx in scope_indices:
        beta = jnp.array(betas[idx]) * safety
        norm_u = pytree_rms_norm(updates[idx])
        impacts.append(beta * norm_u)

    if not impacts:
        return updates, jnp.array(0.0), jnp.array(1.0)

    impacts_arr = jnp.stack(impacts)
    if aggregate == "l1":
        # Provably safe separable surrogate
        S = jnp.sum(impacts_arr)
        eff_tau = tau
    elif aggregate == "l2":
        # Safe L2 requires comparing sqrt(L) * S2 to tau
        # Equivalently, compare S2 to tau / sqrt(L)
        S = jnp.sqrt(jnp.sum(jnp.square(impacts_arr)))
        L = max(1, len(scope_indices))
        eff_tau = tau / jnp.sqrt(jnp.array(L, dtype=tau.dtype))
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")

    c = jnp.where(S > eff_tau, eff_tau / (S + 1e-12), 1.0)
    clipped = jax.tree_util.tree_map(lambda g: c * g, updates)
    return clipped, S, c
