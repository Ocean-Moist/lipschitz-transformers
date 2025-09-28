import time
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

            # Update optimizer state and get parameter updates
            _, opt_state, updates = self.optimizer.update(params, grads, opt_state)

            # Apply post-dualization if needed
            if self.config.post_dualize:
                updates = self.model.dualize(updates)

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
                tau = tau_from_kl(self.config.ebc_target_kl, total_tokens)

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
                            updates[idx],
                            self._ebc_probe_inputs,
                            center_logits=self.config.ebc_center_logits,
                        )
                        # EMA update
                        ema = float(self.config.ebc_beta_ema)
                        self._ebc_betas[idx] = float(ema) * float(self._ebc_betas[idx]) + (1.0 - float(ema)) * float(beta_hat)
                    self._ebc_probe_pos = (self._ebc_probe_pos + k) % max(1, len(self._ebc_scope_idx))

                # Apply clipping
                updates, S, c = apply_ebc_clipping(
                    updates,
                    self._ebc_betas,
                    tau,
                    self._ebc_scope_idx,
                    safety=float(self.config.ebc_safety),
                    aggregate=self.config.ebc_aggregate,
                )
                # Stash EBC internals for logging
                self._ebc_last = {
                    "tau": float(tau),
                    "S": float(S),
                    "c": float(c),
                    "scoped_layers": int(len(self._ebc_scope_idx) if self._ebc_scope_idx is not None else 0),
                }

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

            # Check if training complete
            if self.step >= self.config.steps:
                break

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
