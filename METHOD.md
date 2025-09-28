# Ellipsotope-Budgeted Clipping (EBC): A Tractable Logit-Space Trust Region

### Abstract

Controlling the step size during optimization is crucial for the stable training of deep neural networks. Ideally, this control should be implemented as a trust region in the output space (logits or probabilities), as the loss function (e.g., cross-entropy) is directly sensitive to changes in this space. However, enforcing such a constraint has traditionally been intractable due to the complexity of the parameter-to-logit mapping. We introduce Ellipsotope-Budgeted Clipping (EBC), a practical, drop-in optimizer wrapper that enforces a logit-space trust region. EBC leverages the recent observation that the geometry of attainable logits in Transformers is characterized by an additive structure across layers (a Minkowski sum of ellipsoids). By transferring this structure to the local update regime, we derive a tractable bound on the total logit change as a sum of per-layer contributions. EBC enforces this bound via weighted clipping, where the weights are online estimates of per-layer sensitivities. This approach provides a geometrically principled alternative to parameter-space clipping, directly targeting the dynamics relevant to the loss.

-----

## 1\. Introduction

The stability of training dynamics in large neural networks remains a central challenge. Optimization algorithms like AdamW often require careful tuning of learning rates and stabilization techniques, such as gradient clipping, to avoid divergence or loss spikes. These techniques typically operate in *parameter space*, constraining the norm of the gradient or the proposed update ($\|\Delta W\|$).

However, the objective function—typically cross-entropy loss, which minimizes the KL divergence between the empirical distribution and the model distribution—is defined in the *output space* (probabilities or their preceding logits). The mapping from parameters $W$ to logits $\ell(W)$ is highly non-linear and complex. A small step in parameter space can induce a large, unpredictable change in the output distribution.

The principled approach to managing this uncertainty is a **trust region** enforced in the output space. We seek to ensure that for any update $\Delta W$, the resulting change in logits remains bounded:

$$
\|\Delta \ell\| = \|\ell(W+\Delta W) - \ell(W)\| \le \tau.
$$This directly controls the change in the model's predictions, providing a more robust stabilization mechanism than parameter-space constraints.

### The Intractability Barrier

Traditionally, enforcing a logit-space trust region has been intractable for large models. A naive implementation requires:

1.  **Expensive Computation:** Calculating the exact $\|\Delta \ell\|$ requires an extra forward pass *after* the update is proposed but *before* it is applied.
2.  **Intractable Objects:** Approximating the change using the first-order Taylor expansion, $\|\Delta \ell\| \approx \|J \cdot \Delta W\|$, requires the full Jacobian $J$ (logits w.r.t. parameters). For a model with $P$ parameters and a vocabulary $V$, this object is massive and impractical to materialize or even operate with efficiently.

This intractability has forced practitioners to rely on parameter-space heuristics.

## 2\. The Geometry of Logit Updates

Recent analysis provides a geometric insight that allows us to bypass the intractability barrier. *The Geometry of LLM Logits* [2] demonstrates that in standard Transformers with LayerNorm and residual connections, the set of attainable logits across all inputs is contained within an **ellipsotope**—a Minkowski sum of layer-wise ellipsoids.

This reveals a crucial property: the logits exhibit an **additive structure** across the network's depth. We leverage this insight by transferring the additive structure from the *global geometry* of attainable logits to the *local geometry* of training updates.

### Linearization and Tractable Bounds

Let $W=(W^{(1)},\dots,W^{(L)})$ be the parameters grouped by layer (or block). For a proposed update $\Delta W = (\Delta W^{(1)},\dots,\Delta W^{(L)})$, the resulting change in logits can be linearized as:

$$\\Delta \\ell ;\\approx; \\sum\_{\\ell=1}^L J\_\\ell ,\\Delta W^{(\\ell)},
$$where $J_\ell$ is the Jacobian of the logits with respect to the weights of block $\ell$. Critically, this linearization preserves the additive structure.

We can now bound the norm of the total logit change using the triangle inequality and operator norm sub-multiplicativity:

$$
\|\Delta \ell\| \;\approx\; \Big\|\sum_{\ell=1}^L J_\ell \,\Delta W^{(\ell)}\Big\|
\;\le\; \sum_{\ell=1}^L \|J_\ell\|\,\|\Delta W^{(\ell)}\|.
$$This is the tractability breakthrough. We have replaced the intractable global Jacobian $J$ with a sum of per-layer contributions. This bound is structurally identical to the Minkowski sum that defines the ellipsotope, grounding the approach in the network's inherent geometry.

## 3\. Ellipsotope-Budgeted Clipping (EBC)

We utilize the derived bound to create a practical algorithm, Ellipsotope-Budgeted Clipping (EBC). EBC acts as a wrapper around any base optimizer (e.g., AdamW, SGD).

### 3.1 The Logit-Space Trust Region

We aim to enforce the logit-space trust region $\|\Delta \ell\| \le \tau$. Using the tractable bound, we define a surrogate constraint:

$$\\sum\_{\\ell=1}^L \\beta\_\\ell ,|\\Delta W^{(\\ell)}| ;\\le; \\tau,
\\tag{★}
$$where $\beta_\ell$ is an online estimate of the layer-wise Jacobian norm $\|J_\ell\|$.

If the proposed updates $\Delta W^{(\ell)}_{\text{raw}}$ violate this constraint, we compute the total budgeted impact $S=\sum_\ell \beta_\ell \,\|\Delta W^{(\ell)}_{\text{raw}}\|$. If $S > \tau$, we scale down all updates uniformly by a factor $c=\tau/S$.

This procedure resembles a global norm clip, but crucially, it is a **weighted** clip where the weights ($\beta_\ell$) reflect the sensitivity of the logits to changes in each layer.

### 3.2 Online Sensitivity Estimation ($\beta_\ell$)

The constraint (★) requires estimates of the per-layer Jacobian norms $\beta_\ell \approx \|J_\ell\|$. Computing the exact operator norm is expensive. However, for the purpose of the trust region, we only need the sensitivity along the *direction of the proposed update*.

We estimate this directional sensitivity efficiently online. For a layer $\ell$ and its proposed update $\Delta W^{(\ell)}_{\text{raw}}$, we define the normalized direction $U^{(\ell)} = \Delta W^{(\ell)}_{\text{raw}} / \|\Delta W^{(\ell)}_{\text{raw}}\|$. We then estimate the directional Jacobian norm:

$$
\widehat\beta_\ell = \|J_\ell U^{(\ell)}\|.
$$This can be computed efficiently using two methods:

1.  **Jacobian-Vector Products (JVP):** If the framework supports efficient forward-mode automatic differentiation, $\|J_\ell U^{(\ell)}\|$ can be computed directly without extra forward passes or weight perturbations.
2.  **Finite Differences (FD):** Alternatively, we can use a small probe step $\varepsilon$:
$$

```
$$\\widehat\\beta\_\\ell \\approx |\\ell(W+\\varepsilon U^{(\\ell)})-\\ell(W)| / \\varepsilon.
$$
$$This requires one extra forward pass with only layer $\ell$ perturbed.
```

To minimize overhead, we update $\widehat\beta_\ell$ sporadically (e.g., every $T=20$ steps for a subset of layers) and maintain an Exponential Moving Average (EMA): $\beta_\ell \leftarrow \lambda \beta_\ell + (1-\lambda)\widehat\beta_\ell$.

### 3.3 The EBC Algorithm

The EBC wrapper operates as follows on each training step:

1.  **Propose Updates:** Compute raw updates $\Delta W^{(\ell)}_{\text{raw}}$ from the base optimizer.
2.  **Estimate Sensitivities (Sporadic):** If required by the schedule, update the directional sensitivity estimates $\beta_\ell$ for a subset of layers using JVP or FD.
3.  **Compute Budgeted Impact:** Calculate $S=\sum_\ell \beta_\ell \,\|\Delta W^{(\ell)}_{\text{raw}}\|$.
4.  **Weighted Clip:** If $S \le \tau$, accept the updates. Otherwise, scale all updates by $c=\tau/S$: $\Delta W^{(\ell)} \leftarrow c\,\Delta W^{(\ell)}_{\text{raw}}$.
5.  **Apply Updates:** $W^{(\ell)} \leftarrow W^{(\ell)} + \Delta W^{(\ell)}$.

### 3.4 Interpretation and Refinements

#### Connection to KL Divergence

The L2 norm in logit space is a meaningful metric because it bounds the KL divergence between the output distributions before and after the update. Locally, $D_{\mathrm{KL}}(p(z)\|p(z+\delta z)) \approx \tfrac12 \delta z^\top F(z) \delta z$, where $F(z)$ is the Fisher information matrix. Since the spectral norm of $F(z)$ is bounded, controlling $\|\delta z\|_2^2$ directly controls the KL step.

#### Logit Centering

The softmax function is invariant to constant shifts in the logits. To respect this invariance and align the L2 norm with the Fisher geometry, it is crucial to center the logit changes per token before computing the norm used in the $\beta_\ell$ estimation:

$$
\tilde{\delta z} = \delta z - \tfrac{1}{V}(\mathbf{1}^\top \delta z)\mathbf{1}.
$$#### Setting the Trust Radius ($\tau$)

By using centered logits and recognizing the bound $D_{\mathrm{KL}} \lesssim \tfrac14 \|\tilde{\delta z}\|_2^2$, we can select $\tau$ based on a desired average per-token KL divergence target ($\delta$). For a batch with $B$ tokens:

$$\\tau \\approx \\sqrt{4,B,\\delta}.
$$This provides an interpretable hyperparameter (e.g., $\delta=0.05$ nats/token).

## 4\. Connections and Extensions

### 4.1 Modular Norms and Layer Budgeting

EBC aligns with the *modular norm* perspective [1], which argues for principled budgeting of learning rates across composed layers based on their sensitivities. In EBC, the $\beta_\ell$ values dynamically allocate the total logit-space budget $\tau$ among the layers. Layers that have a larger impact on the logits (high $\beta_\ell$) are constrained more strictly, providing a data-driven alternative to layer-wise adaptive methods like LARS or LAMB.

### 4.2 Synergy with Manifold Optimization

The accuracy of the sensitivity estimates $\beta_\ell$ depends on the conditioning of the layer mappings. *Modular Manifolds* [1] suggests constraining weight matrices to well-behaved manifolds, such as the Stiefel manifold (enforcing orthogonality or controlling singular values), often implemented via Muon-style updates or matrix retraction (e.g., polar decomposition).

EBC synergizes well with these approaches. By stabilizing the spectral properties of the layers, manifold constraints ensure that the $\beta_\ell$ estimates remain stable and accurate, potentially reducing the need for aggressive clipping. An optional retraction step can be added after the EBC update is applied.

## 5\. Conclusion

Ellipsotope-Budgeted Clipping (EBC) provides a tractable and geometrically principled method for enforcing a trust region in the logit space. By leveraging the additive structure of logit geometry in modern Transformers, EBC decomposes the intractable global sensitivity into a sum of manageable per-layer sensitivities. This results in a drop-in optimizer wrapper that stabilizes training dynamics by directly controlling the change in the output distribution, offering a robust alternative to traditional parameter-space clipping.

-----

### References

[1] Thinking Machines Lab. *Modular Manifolds* (Stiefel constraints, modular norms and layer budgeting).
[2] Hugo ʕ•ᴥ•ʔ Bear. *The Geometry of LLM Logits* (ellipsotope outer bound for logits under LayerNorm).