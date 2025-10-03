import time
import os
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

from modula.ebc import (
    tau_from_kl,
    get_scope_indices,
    estimate_beta_jvp,
    apply_ebc_clipping,
)


class Trainer:
    def __init__(
        self, model, optimizer, train_loader, val_loader, loss_fn, config, logger
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.logger = logger
        self.step = 0

        # Compile loss and gradient function (optionally JIT)
        value_and_grad = jax.value_and_grad(partial(loss_fn, model))
        self.loss_and_grad = jax.jit(value_and_grad) if getattr(config, "jit", True) else value_and_grad

        # EBC state (lightweight)
        self._ebc_enabled = bool(getattr(config, "ebc_enable", False))
        self._ebc_probe_inputs = None  # set on first batch
        self._ebc_betas = None  # list[float], one per Atom weight
        self._ebc_scope_idx = None  # indices of layers in budget
        self._ebc_probe_pos = 0  # round-robin pointer
        self._ebc_last = None  # last (tau,S,c)
        # EBC PI controller state
        self._ebc_ctrl_enable = bool(getattr(self.config, "ebc_ctrl_enable", False))
        self._ebc_ctrl_period = int(getattr(self.config, "ebc_ctrl_period", 20))
        self._ebc_ctrl_kp = float(getattr(self.config, "ebc_ctrl_kp", 0.15))
        self._ebc_ctrl_ki = float(getattr(self.config, "ebc_ctrl_ki", 0.02))
        self._ebc_ctrl_halflife = float(getattr(self.config, "ebc_ctrl_ema_halflife", 250))
        # controller tracks per-token KL target delta* (nats/token)
        self._ebc_ctrl_delta = float(getattr(self.config, "ebc_delta_star", getattr(self.config, "ebc_target_kl", 0.05)))
        self._ebc_ctrl_logdelta = float(jnp.log(max(self._ebc_ctrl_delta, 1e-8)))
        self._ebc_ctrl_ema_e = 0.0
        # EBC robust beta settings
        self._ebc_beta_huber_delta = float(getattr(self.config, "ebc_beta_huber_delta", 0.0))
        self._ebc_beta_full_sweep = int(getattr(self.config, "ebc_beta_full_sweep", 200))

    def _ckpt_paths(self):
        ckdir = Path(getattr(self.config, "ckpt_dir", Path(self.config.output_dir) / "ckpts"))
        ckdir.mkdir(parents=True, exist_ok=True)
        latest = ckdir / "latest.pkl"
        return ckdir, latest

    def _save_checkpoint(self, params, opt_state, key):
        ckdir, latest = self._ckpt_paths()
        state = {
            "step": int(self.step),
            "params": jax.device_get(params),
            "opt_state": jax.device_get(opt_state),
            "key": jax.device_get(key),
            "ebc": {
                "probe_inputs": None if self._ebc_probe_inputs is None else jax.device_get(self._ebc_probe_inputs),
                "betas": self._ebc_betas,
                "scope_idx": self._ebc_scope_idx,
                "probe_pos": self._ebc_probe_pos,
                "last": self._ebc_last,
            },
        }
        tmp = ckdir / f"ckpt_{self.step:06d}.pkl"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Update latest pointer
        try:
            with open(latest, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def restore_state(self, state):
        # Restore trainer internals and return params/opt_state/key
        self.step = int(state.get("step", 0))
        e = state.get("ebc", {}) or {}
        self._ebc_probe_inputs = e.get("probe_inputs", None)
        self._ebc_betas = e.get("betas", None)
        self._ebc_scope_idx = e.get("scope_idx", None)
        self._ebc_probe_pos = int(e.get("probe_pos", 0))
        self._ebc_last = e.get("last", None)
        return state.get("params"), state.get("opt_state"), state.get("key")

    def train(self, params, opt_state, key):
        """Run one training epoch."""
        accum_step = 0
        accum_loss = 0.0
        accum_grad = jax.tree.map(jnp.zeros_like, params)
        log = {}

        for inputs, targets in self.train_loader:
            # Forward and backward pass
            loss, grads = self.loss_and_grad(params, inputs, targets)

            # Gradient accumulation
            accum_grad = jax.tree.map(jnp.add, accum_grad, grads)
            accum_loss += loss.item()
            accum_step += 1

            # Skip update if still accumulating
            if accum_step % self.config.accum_steps != 0:
                continue

            # Average accumulated gradients
            grads = jax.tree.map(lambda g: g / self.config.accum_steps, accum_grad)
            accum_grad = jax.tree.map(jnp.zeros_like, params)
            loss = accum_loss / self.config.accum_steps
            accum_loss = 0.0

            # Apply pre-dualization if needed
            if self.config.pre_dualize:
                grads = self.model.dualize(grads)

            # Update optimizer state and get parameter updates (raw, pre-EBC)
            _, opt_state, updates_raw = self.optimizer.update(params, grads, opt_state)
            # Default updates are raw; EBC may override below
            updates = updates_raw

            # Apply post-dualization if needed (to raw updates)
            if self.config.post_dualize:
                updates_raw = self.model.dualize(updates_raw)

            # Get learning rate from schedule
            lr = self.get_lr(self.step)

            # EBC: estimate budget and clip updates (before weight decay/projection)
            if self._ebc_enabled:
                # Initialize EBC state on first use
                if self._ebc_betas is None:
                    trackers = self.model.trackers()
                    self._ebc_scope_idx = get_scope_indices(
                        trackers, include_embed_out=self.config.ebc_include_embed_out
                    )
                    self._ebc_betas = [0.0 for _ in trackers]

                # Capture a fixed probe batch once
                if self._ebc_probe_inputs is None:
                    self._ebc_probe_inputs = inputs

                # Compute tau from target KL and token count
                # Assume inputs is (B, T) for GPT; else fallback to batch size
                total_tokens = (
                    int(self._ebc_probe_inputs.shape[0] * self._ebc_probe_inputs.shape[1])
                    if self._ebc_probe_inputs.ndim >= 2
                    else int(self._ebc_probe_inputs.shape[0])
                )
                # controller can adjust the per-token KL target (delta)
                target_delta = (
                    float(self._ebc_ctrl_delta) if self._ebc_ctrl_enable else float(self.config.ebc_target_kl)
                )
                tau = tau_from_kl(target_delta, total_tokens)

                # Update beta estimates every N steps, probing K layers round-robin
                if (self.step % self.config.ebc_update_every) == 0:
                    k = max(1, int(self.config.ebc_probe_layers))
                    for i in range(k):
                        if not self._ebc_scope_idx:
                            break
                        idx = self._ebc_scope_idx[(self._ebc_probe_pos + i) % len(self._ebc_scope_idx)]
                        beta_hat = estimate_beta_jvp(
                            self.model,
                            params,
                            idx,
                            updates_raw[idx],
                            self._ebc_probe_inputs,
                            center_logits=self.config.ebc_center_logits,
                        )
                        # EMA update
                        ema = float(self.config.ebc_beta_ema)
                        prev = float(self._ebc_betas[idx])
                        bh = float(beta_hat)
                        if self._ebc_beta_huber_delta > 0.0:
                            # Huber on residual relative to EMA
                            r = bh - prev
                            delta = self._ebc_beta_huber_delta
                            if abs(r) > delta:
                                r = delta if r > 0 else -delta
                            bh = prev + r
                        self._ebc_betas[idx] = float(ema) * prev + (1.0 - float(ema)) * bh
                    self._ebc_probe_pos = (self._ebc_probe_pos + k) % max(1, len(self._ebc_scope_idx))

                # Periodic full sweep to limit beta drift/noise
                if self._ebc_beta_full_sweep and (self.step % self._ebc_beta_full_sweep == 0):
                    ema = float(self.config.ebc_beta_ema)
                    for idx in (self._ebc_scope_idx or []):
                        beta_hat = estimate_beta_jvp(
                            self.model,
                            params,
                            idx,
                            updates_raw[idx],
                            self._ebc_probe_inputs,
                            center_logits=self.config.ebc_center_logits,
                        )
                        prev = float(self._ebc_betas[idx])
                        bh = float(beta_hat)
                        if self._ebc_beta_huber_delta > 0.0:
                            r = bh - prev
                            delta = self._ebc_beta_huber_delta
                            if abs(r) > delta:
                                r = delta if r > 0 else -delta
                            bh = prev + r
                        self._ebc_betas[idx] = float(ema) * prev + (1.0 - float(ema)) * bh

                # Apply clipping
                updates_clipped, S, c = apply_ebc_clipping(
                    updates_raw,
                    self._ebc_betas,
                    tau,
                    self._ebc_scope_idx,
                    safety=float(self.config.ebc_safety),
                    aggregate=self.config.ebc_aggregate,
                )
                updates = updates_clipped
                # Stash EBC internals for logging
                self._ebc_last = {
                    "tau": float(tau),
                    "S": float(S),
                    "c": float(c),
                    "scoped_layers": int(len(self._ebc_scope_idx) if self._ebc_scope_idx is not None else 0),
                }

                # Periodic controller probe: shadow-apply current c*updates and measure applied KL
                if self._ebc_ctrl_enable and (self.step % max(1, self._ebc_ctrl_period) == 0):
                    # shadow params before mutation
                    params_before = params
                    # create shadow-updated params using RAW optimizer updates scaled by current c
                    c_updates = jax.tree.map(lambda g: c * g, updates_raw)
                    params_plus = self.model.step(params_before, c_updates, lr)

                    # logits on probe batch (center per token)
                    z = self.model(self._ebc_probe_inputs, params_before)
                    z_p = self.model(self._ebc_probe_inputs, params_plus)
                    # center per token
                    def center_logits(logits):
                        mean_shift = jnp.mean(logits, axis=-1, keepdims=True)
                        return logits - mean_shift
                    dz = center_logits(z_p - z)
                    T = jnp.linalg.norm(dz)
                    # applied per-token KL
                    def kl_per_token(z0, z1):
                        p = jax.nn.log_softmax(z0)
                        q = jax.nn.log_softmax(z1)
                        # KL(p||q) = sum p*(log p - log q)
                        p_prob = jnp.exp(p)
                        kl_tok = jnp.sum(p_prob * (p - q), axis=-1)
                        return jnp.mean(kl_tok)
                    applied_kl = float(kl_per_token(z, z_p))

                    # guard on applied quantities (shrink-only)
                    eps = 1e-12
                    # recompute surrogate S on raw updates (input-independent given betas)
                    _clipped_probe, S_probe_val, _ = apply_ebc_clipping(
                        updates_raw,
                        self._ebc_betas,
                        jnp.array(1e9, dtype=jnp.float32),
                        self._ebc_scope_idx,
                        safety=float(self.config.ebc_safety),
                        aggregate=self.config.ebc_aggregate,
                    )
                    S_probe = float(S_probe_val)
                    rho = float(T / max(float(c) * S_probe + eps, eps))
                    if rho > 1.0:
                        # shrinking tau is equivalent to shrinking delta by rho^2
                        self._ebc_ctrl_logdelta -= 2.0 * float(jnp.log(rho))

                    # PI controller in log space for per-token KL target
                    delta_star = float(self._ebc_ctrl_delta)
                    e = float(delta_star - applied_kl)
                    # EMA update with half-life in steps; update when we probe (periodic)
                    # alpha chosen so EMA halves after 'halflife' steps
                    # approximate per-probe alpha by accounting for probe spacing
                    import math
                    hl = max(1.0, self._ebc_ctrl_halflife)
                    dt = max(1.0, float(self._ebc_ctrl_period))
                    alpha = 1.0 - math.exp(-math.log(2.0) * dt / hl)
                    self._ebc_ctrl_ema_e = (1.0 - alpha) * self._ebc_ctrl_ema_e + alpha * e
                    self._ebc_ctrl_logdelta = self._ebc_ctrl_logdelta + self._ebc_ctrl_kp * e + self._ebc_ctrl_ki * self._ebc_ctrl_ema_e
                    # clamp delta
                    delta_min = float(getattr(self.config, "ebc_ctrl_delta_min", 0.01))
                    delta_max = float(getattr(self.config, "ebc_ctrl_delta_max", 0.30))
                    new_delta = float(jnp.clip(jnp.exp(self._ebc_ctrl_logdelta), delta_min, delta_max))
                    # store
                    self._ebc_ctrl_delta = new_delta
                    # extend logging
                    self._ebc_last["applied_kl"] = float(applied_kl)
                    self._ebc_last["rho"] = rho
                    self._ebc_last["delta_ctrl"] = self._ebc_ctrl_delta

            else:
                # No EBC: use raw optimizer updates
                updates = updates_raw

            # Update parameters with weight decay and projection
            key, subkey = jax.random.split(key)
            params = self.model.decay_step_project(
                params,
                updates,
                w_max=self.config.w_max,
                wd=self.config.wd,
                spectral_wd=getattr(self.config, "spectral_wd", 0),
                lr=lr,
                key=subkey,
            )

            self.step += 1

            # Log periodically
            if self.step % self.config.log_interval == 0:
                # Calculate training accuracy
                logits = self.model(inputs, params)
                train_preds = jnp.argmax(logits, axis=-1)
                train_acc = jnp.mean(train_preds == targets)
                log = self.model.log(params, updates)

                if self._ebc_enabled and self._ebc_last is not None:
                    # Merge EBC internals
                    log = {**log, "ebc": self._ebc_last}

                self.logger.log_training(self.step, loss, train_acc, log)

            # Validate periodically
            if self.step % self.config.val_interval == 0:
                val_metrics = self.validate(params)
                self.logger.log_validation(self.step, val_metrics)

            # Checkpoint periodically
            if getattr(self.config, "ckpt_interval", 0) and (self.step % int(self.config.ckpt_interval) == 0):
                self._save_checkpoint(params, opt_state, key)

            # Check if training complete
            if self.step >= self.config.steps:
                break

        # Final checkpoint
        if getattr(self.config, "ckpt_interval", 0):
            self._save_checkpoint(params, opt_state, key)
        return params, opt_state, key

    def validate(self, params):
        """Run validation and return metrics."""
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_step = 0

        for inputs, targets in self.val_loader:
            loss, _ = self.loss_and_grad(params, inputs, targets)
            logits = self.model(inputs, params)
            val_loss_sum += float(loss)
            preds = jnp.argmax(logits, axis=-1)
            val_acc_sum += jnp.mean(preds == targets)
            val_step += 1

            if val_step >= self.config.val_iters:
                break

        return {"loss": val_loss_sum / val_step, "accuracy": val_acc_sum / val_step}

    def get_lr(self, step):
        """Get learning rate based on schedule."""
        schedule_fn = {
            "linear": lambda s: (self.config.steps - s) / self.config.steps,
            "cosine": lambda s: 0.5 * (1 + jnp.cos(jnp.pi * s / self.config.steps)),
            "sqrt": lambda s: 1 / (1 + (s // 512) ** 0.5),
            "none": lambda s: 1,
        }[self.config.schedule]

        return self.config.lr * schedule_fn(step)
