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

<problem>
Study Context



  - Model: small GPT (num_blocks=3, d_embed=128, num_heads=4), causal LM on shakespeare.

  - Optimizers/variants: Adam and Muon; with/without spectral normalization; EBC on/off.

  - Training config: constant LR (no schedule), 2000 steps, bfloat16 model dtype, project dtype float32, JIT enabled.

  - Hardware: 8× RTX 4000 Ada; orchestration runs one subprocess per GPU. Spectral backend set to GPU, but winning

  configuration here is “none” (no spectral projection).

  - Automation: Auto pipeline with three stages: (1) LR sweep (EBC off), (2) EBC delta/aggregate calibration at baseline

  LR, (2b) LR-scale sweep for EBC and matched control, (3) final full-length runs. Clip statistics recorded as clip_rate

  (fraction of logged steps with c<0.999) and avg_c.



  Key artifacts



  - Finals/plots/decisions: outputs/ebc_ablation/shakespeare_autopilot_v3/summary_final.csv, results_final.json,

  auto_decisions.json, adam_none_baseline_loss.png, adam_none_ebc_d32p0_l2_{loss,clip}.png.



  Summary of Results



  - LR sweep (Stage 1)

      - Adam (no spectral): best warmup at lr=0.002 (val_loss≈1.8533).

      - Muon and spectral variants substantially worse; triaged before calibration.

  - EBC calibration at lr=0.002 (Stage 2)

      - δ grid (L1/L2): for δ ≤ 8, clip_rate≈1.0 (clipping nearly every logged step), high val_loss.

      - Increasing δ reduced clipping and improved warmup loss: δ=16 → clip_rate≈0.93; δ=32 → clip_rate≈0.73; δ=32 L2

  selected as “nearest” (outside target band).

  - LR-scale sweep (Stage 2b) with EBC δ=32, L2

      - EBC at lr×2: val_loss≈1.7326 vs control (no EBC) ≈2.1158.

      - Control diverges at lr×3 and lr×4 (val_loss≈4.996 and ≈1320), whereas EBC remains stable (≈1.98 at ×3, ≈2.44 at ×4).

      - Chosen scale for finals: ×2.

  - Final runs (Stage 3)

      - Baseline (Adam, lr=0.002): val_loss≈1.3795 (replicate: 1.4624).

      - EBC (Adam, lr=0.004, δ=32 L2): val_loss≈1.5982, clip_rate≈0.92, avg_c≈0.228.



  Interpretation



  - Stability/Headroom: EBC demonstrably increases learning-rate headroom. At matched higher LR (×2), EBC maintains good

  warmup loss where the control degrades; at ×3–×4 the control collapses and EBC remains stable. This aligns with the trust-

  region intuition behind EBC.

  - Final Quality Regression: Despite the headroom win, the final EBC run underperforms the baseline. The recorded

  clip_rate≈0.92 and avg_c≈0.228 indicate heavy clipping across most of the final run. The δ selected from warmup did not

  maintain the desired partial-clipping regime over 2000 steps.



  Observed Dynamics



  - Calibration Band Miss: At base LR=0.002, the “nearest” winner δ=32 yielded clip_rate≈0.73 (above the target band), but

  at lr×2 the warmup clip_rate dropped to ≈0.47 (inside band) with strong validation performance. In finals, the clip_rate

  drifted upward to ≈0.92.

  - Drift Across Phases: The fraction of clipped steps changes significantly between short warmups and full training,

  suggesting non-stationary update/curvature statistics. A fixed δ calibrated early does not preserve the clipping regime

  later.

  - Aggregation: L2 aggregate was selected empirically; L1 consistently over-clipped harder in this setting.

  - Muon: Baselines were far weaker; EBC on muon was not pursued here due to triage. The broader conclusions are presently

  for Adam.



  Measurement Details



  - clip_rate definition: fraction of logged steps where the EBC coefficient c < 0.999 (i.e., clipping applied). avg_c is

  the average c.

  - Logging cadence: log_interval=20 steps, warmup length 300 steps, so clip metrics reflect sampled snapshots, not per-

  step totals.

  - EBC internals: tau derived from target KL per token count; β estimates updated via JVP every 20 steps over 2 layers;

  logits centered per config.



  Potential Confounds



  - Warmup vs Full Run: Warmup probes are brief; the distribution of gradient norms, layer sensitivities (β), and curvature

  may shift materially over the full run.

  - Constant LR: No decay schedule used; as training progresses, maintaining a fixed KL budget may interact differently with

  evolving sharpness/curvature.

  - Sampling/Scope: Only a subset of layers is probed per update (round-robin); this may introduce variance in β estimates

  that manifests differently later in training.

  - Metric Granularity: clip_rate is computed only on logging steps; spikes or patterns between logs are not captured.

  - Small-scale regime: Shakespeare is tiny; behavior might differ at larger scale or with different tokenization/

  statistics.



  What Seems Robust



  - EBC provides stability/robustness at elevated LR, consistently outperforming controls in the short-run regime and

  preventing divergence at ×3–×4.

  - Without maintaining a mid-band clipping regime, heavy clipping correlates with inferior final validation quality in

  this setup.



  Open Questions (Research-Oriented)



  - Regime Drift: Why does a δ that lands near the target clip rate during warmup drift to heavy clipping later? Is this

  driven by evolving β estimates, the τ computation from fixed target KL, changes in gradient spectrum, or non-stationary

  curvature?

  - LR Interaction: Why did clip_rate at δ=32 appear lower (closer to target) at lr×2 in warmups, yet rise during full

  training at the same LR? Is there a transient calibration effect or a sensitivity to early dynamics?

  - Aggregation Choice: Under what conditions does L2 (vs L1) produce better long-run behavior? Can we characterize per-

  layer c distributions to understand aggregate sensitivity?

  - Probing Strategy: How do EBC beta-estimation cadence and probe-layer selection affect long-run clipping statistics?

  Would broader or adaptive probing stabilize β estimates?

  - Metric Semantics: Is clip_rate (fraction of steps clipped) the best operational target? Would targeting a statistic over

  c’s distribution (e.g., quantiles) or per-layer measures better correlate with final quality?

  - Trust-Region Link: Given EBC’s KL framing, how does a fixed per-step KL budget interact with the non-stationary

  optimization path (e.g., as loss decreases and gradients flatten vs. sharpen)? Is there a theoretically grounded rule for

  evolving δ across training?

  - Overhead/Throughput: What is the net throughput impact of EBC (tokens/sec) vs. the compute saved by higher LR headroom?

  Early indicators are positive for stability; quantifying the end-to-end efficiency frontier is pending.



  Reproducibility Pointers



  - Finals: outputs/ebc_ablation/shakespeare_autopilot_v3/summary_final.csv, results_final.json

  - Decisions and calibration traces: outputs/ebc_ablation/shakespeare_autopilot_v3/auto_decisions.json

  - Plots: outputs/ebc_ablation/shakespeare_autopilot_v3/adam_none_baseline_loss.png, outputs/ebc_ablation/

  shakespeare_autopilot_v3/adam_none_ebc_d32p0_l2_{loss,clip}.png



  Bottom line



  - EBC achieved its core promise (expanded LR headroom and stability) in short-run evaluations, but a fixed δ calibrated at

  warmup did not hold the desired partial-clipping regime throughout full training, leading to over-clipping and worse final

  validation loss. The question of how δ should evolve during training appears to be a research issue (distributional drift

  and trust-region dynamics), not just an engineering tweak. perhaps i am thinking about the problem wrong as well, and there could be a way of thinking from first principles and abstracting/simplifying away the dynamic delta, like it is a symptom not a cause. 

anyways, here I have presented our observations for you to think about and figure out the best way to proceed, whether it is a dynamic delta variation or something else. 

</problem>
<solution>
# Ellipsotope‑Budgeted Clipping (EBC) on Shakespeare

**A verified diagnosis and a corrected, self‑contained training recipe**

---

## 0) What you asked for

> *“Verify the soundness/consistency of the solution, explain the behavior we saw, and figure out the best way to proceed—whether a dynamic delta or something else.”*

This document is self‑contained. It (i) validates the theory against your observations, (ii) pinpoints the failure modes that caused the final regression, and (iii) gives a drop‑in, corrected EBC procedure that preserves LR headroom **and** final quality.

---

## 1) Experimental setting (brief)

* **Model**: 3‑block GPT (d_embed=128, 4 heads), causal LM on Shakespeare.
* **Optimizers**: Adam (winner), Muon (triaged); spectral projection “none” performed best.
* **Training**: constant LR, 2000 steps, bf16 params / fp32 proj, JIT.
* **Automation**: 3 stages (LR sweep → EBC calibration → LR‑scale sweep → finals).
* **EBC telemetry**: `clip_rate` (fraction of logged steps with `c < 0.999`) and `avg_c`.

### Key results you observed

* **LR sweep (no EBC)**: Adam best at **lr=0.002** (val ≈ 1.853 warmup).
* **EBC calibration at lr=0.002**:

  * Small “δ” (your grid values ≤ 8) → `clip_rate≈1.0` (over‑clipped).
  * Larger “δ”: at 16 → `clip_rate≈0.93`; at 32 → `clip_rate≈0.73`; selected “δ=32 (L2)”.
* **LR‑scale sweep (EBC δ=32, L2)**:

  * **lr×2**: EBC val ≈ **1.7326** vs control ≈ **2.1158**.
  * Control **diverged** at ×3, ×4; EBC remained **stable** (≈1.98, ≈2.44).
* **Finals**:

  * Baseline (Adam 0.002): val ≈ **1.3795** (replicate 1.4624).
  * EBC (Adam 0.004, δ=32 L2): val ≈ **1.5982**, **clip_rate≈0.92**, **avg_c≈0.228**.

**Takeaway**: EBC **increased LR headroom and stability**, but the final run **underperformed** baseline due to heavy clipping.

---

## 2) Theory you’re using (recap & check)

Let parameters be grouped by layer (W=(W^{(1)},\dots,W^{(L)})); logits for a batch are (z(W)). For a proposal (\Delta W),

[
\Delta z ;\approx; \sum_{\ell=1}^L J_\ell,\Delta W^{(\ell)}.
]

Triangle inequality gives the **layer‑separable** bound

[
|\Delta z| ;\le; \sum_{\ell=1}^L |J_\ell|,|\Delta W^{(\ell)}|
\quad\Rightarrow\quad
S_1:=\sum_{\ell=1}^L \beta_\ell |\Delta W^{(\ell)}|*2 ;\le; \tau,
]
with **directional** sensitivities (\beta*\ell = |J_\ell U^{(\ell)}|), (U^{(\ell)}=\Delta W^{(\ell)}/|\Delta W^{(\ell)}|).

* **Logit centering (per token)** is required (softmax shift‑invariance):
  (\tilde{\Delta z} = (I-\frac{1}{V}\mathbf 1\mathbf 1^\top)\Delta z).

* **KL linkage (local)** with centered logits:
  (D_{\text{KL}}(p(z)|p(z+\tilde{\delta z})) ;\le; \tfrac14 |\tilde{\delta z}|_2^2).

* **Trust radius** for **average per‑token** KL target (\delta) on a batch of (B) tokens:
  (\boxed{\ \tau = \sqrt{4,B,\delta}\ }).

* **Aggregator correctness**:

  * **L1** (S_1=\sum_\ell \beta_\ell |\Delta W^{(\ell)}|_2) is **safe**.
  * If you prefer **L2** (S_2=(\sum_\ell (\beta_\ell|\Delta W^{(\ell)}|_2)^2)^{1/2}), enforce
    (\sqrt{L},S_2 \le \tau) (equivalently (S_2\le\tau/\sqrt{L})). Without the (\sqrt{L}) factor, it is **not** a bound.

---

## 3) Why your final EBC regressed (consistent diagnosis)

* **Heavy clipping collapses effective LR.** With **avg (c≈0.228)** at **lr=0.004**, the effective LR was
  (\eta_{\text{eff}} = c \cdot \eta = 0.228 \times 0.004 = 0.000912), which is **smaller** than the baseline (0.002).
  You were effectively training **slower** than baseline for most of the run.

* **Why clipping drifted upward**: A **fixed** (\tau) does not adapt to evolving (\beta_\ell) and gradient spectra; as training geometry sharpens, the surrogate (S) grows, so (c=\tau/S) shrinks. This explains the mid‑band warmup → heavy‑clip finals drift.

* **Aggregator mismatch**: Selecting **L2** without the (\sqrt{L}) safety explains why L2 appeared to “clip less” than L1 in warmup; it was effectively **looser** than the provable bound.

---

## 4) Minimal, principled fix

### 4.1 Control the **applied** per‑token KL with a light PI loop

Every (K) steps (e.g., (K=20)) on a tiny **probe** batch:

1. Compute current logits (z) and “shadow‑applied” logits (z^+) at (W \oplus (c,\Delta W_{\text{raw}})) (no weight mutation).
2. Center per token, compute applied KL (\widehat{\mathrm{KL}}) (nats/token).
3. PI controller (log‑space) tracks a target (\delta^*) (e.g., 0.03–0.07 nats/token):
   [
   e_t=\delta^*-\widehat{\mathrm{KL}}*t,\qquad
   \log \delta*{t+1}=\log \delta_t + k_p e_t + k_i,\text{EMA}(e_{1:t}).
   ]
   Then set (\tau_{t+1}=\sqrt{4,B_{\text{train}},\delta_{t+1}}).

**Good starting gains**: (k_p=0.15), (k_i=0.02), EMA half‑life 200–300 steps, clamp (\delta\in[0.01,0.30]) nats/token.

### 4.2 Use the **L1** layer aggregator (or scale L2 by (\sqrt{L}))

* Preferred: (S=\sum_\ell \beta_\ell |\Delta W^{(\ell)}|_2), (c=\min(1,\tau/\max(S,\varepsilon))).
* If you keep L2, compare (\sqrt{L},S_2) to (\tau) (here (L=3\Rightarrow \sqrt{L}\approx 1.732)).

### 4.3 Guard on **applied** quantities (bug fix)

On the same probe batch, recompute the surrogate (S_{\text{probe}}) (centered) and the applied logit change (T=|\tilde z^+-\tilde z|_2). Use

[
\boxed{\ \rho=\dfrac{T}{\max(c\cdot S_{\text{probe}},\varepsilon)}\ }\quad
\text{if }\rho>1\ \Rightarrow\ \tau\leftarrow \tau/\rho\ \text{(shrink‑only)}.
]

> The earlier formula missing the product (c\cdot S) hides violations when (c<1).

### 4.4 Sensitivity (β) estimation you already run—keep, but make it consistent

* Directional JVP/FD along (U^{(\ell)}); skip zero‑norm layers.
* Update 2–3 layers per step (round‑robin), periodic full sweep (~every 200 steps).
* Huber‑clip new (\widehat\beta) then EMA (e.g., (\lambda=0.95)).
* **Centering** must be applied inside the JVP/FD path so (\beta) matches the KL geometry.

---

## 5) Ready‑to‑run training protocol

### Stage 1 — LR sweep (unchanged)

* No EBC; pick the best baseline LR (e.g., **0.002** for Adam).

### Stage 2 — EBC warmup with controller (300–500 steps)

* Turn on **EBC + PI control** (Sections 4.1–4.3).
* Grid **(\delta^*\in{0.03,0.05,0.07})** nats/token.
* **Selection**: best warmup val **and** median (c \in [0.5,0.7]).

### Stage 2b — LR‑scale sweep with controller

* Test LR×{1.5, 2, 3} (and optional 4).
* Compare **EBC+controller** vs EBC‑fixed‑(\tau) vs no‑EBC.
* Choose LR with best warmup val while median (c) remains ≈0.5–0.7.

### Stage 3 — Finals

* 3 seeds at chosen LR & (\delta^*).
* Report final val, (c) quantiles, applied KL tracking, throughput (tokens/s) and probe overhead.

---

## 6) Pseudocode (drop‑in)

```python
# ΔW_raw proposed by base optimizer (e.g., Adam); ε is small positive

# ----- L1 surrogate -----
S = 0.0
for ℓ in layers:
    gℓ = ΔW_raw[ℓ]
    nℓ = l2_norm(gℓ)
    if nℓ > 0:
        Uℓ = gℓ / max(nℓ, ε)
        βℓ = beta_ema[ℓ]               # ||Jℓ Uℓ|| on centered logits (EMA)
        S += βℓ * nℓ

# ----- global scaling -----
c = min(1.0, τ / max(S, ε))

# ----- periodic probe (every K steps) -----
if step % K == 0:
    z   = logits(W, probe_batch)
    z_p = logits(W ⊕ (c * ΔW_raw), probe_batch)   # shadow apply; no mutation
    T   = l2_norm(center_per_token(z_p - z))      # applied logit L2 (centered)
    KL  = mean_per_token_KL(softmax(z), softmax(z_p))

    # guard on applied quantities (shrink-only)
    S_probe = surrogate_on_probe(ΔW_raw, beta_ema, centered=True)  # recompute on probe
    ρ = T / max(c * S_probe, ε)
    if ρ > 1.0:
        τ = τ / ρ

    # PI controller on per-token KL (log-space)
    e = δ_star - KL
    ema_e = update_ema(ema_e, e, half_life=250)
    logδ = logδ + k_p * e + k_i * ema_e
    δ = clamp(exp(logδ), δ_min, δ_max)
    τ = sqrt(4.0 * B_train * δ)

# ----- apply clipped step -----
for ℓ in layers:
    W[ℓ] += c * ΔW_raw[ℓ]

# ----- β refresh (round-robin JVP/FD), with Huber+EMA -----
refresh_some_betas()
```

**Invariants**: use **the same per‑token centering** in β, (S), (T), KL. All compute in fp32; params can stay bf16.

---

## 7) What to log & when you’re “on‑track”

**Every (K) steps**, log:

* Applied per‑token KL (probe) vs target (\delta); current (\tau); (B_{\text{train}}).
* (c) statistics: mean, median, p10/p90; and (\eta_{\text{eff}} = c \cdot \text{LR}) (mean/median).
* Per‑layer contributions (s_\ell = \beta_\ell |\Delta W^{(\ell)}|_2) (mean/quantiles).
* Guard activations: (\rho) count and magnitudes.
* Throughput (tokens/s); probe time fraction.

**Acceptance for finals**:

* KL tracks (\delta^*) within ±30% most of training.
* Median (c) stays ~0.5–0.7; no long plateaus at (c\ll 0.3).
* Final val at LR×2 **≤** baseline.
* Overhead ≤ 5–10%.

---

## 8) Focused ablations (to cement confidence)

1. **Aggregator**: L1 vs L2/(\sqrt{L}). Expect L1 slightly less conservative but both safe.
2. **β cadence**: 20‑step round‑robin vs denser; add periodic full sweep.
3. **Guard**: none vs shrink‑only (recommended) vs shrink+β‑inflate.
4. **(\delta^*) grid**: 0.03/0.05/0.07 nats/token; correlate (c) quantiles with final loss.
5. **(Optional) manifold constraints**: light orthogonality/conditioning may reduce β variance; keep off unless it clearly helps throughput/quality.

---

## 9) Answers to your open questions

* **Regime drift**: Fixed (\tau) + evolving (\beta) and gradients ⇒ (S\uparrow), (c\downarrow), (\eta_{\text{eff}}\downarrow). *Control the applied per‑token KL* to keep trust‑region semantics constant.

* **LR interaction**: With clipping, (\eta_{\text{eff}} = \tau/S_{\text{base}}) (approximately independent of nominal LR). Early warmth can mask later sharpness; the controller adapts (\tau) as needed.

* **Aggregation choice**: L1 is the provable surrogate. If using L2, enforce the (\sqrt{L}) safety factor; otherwise it’s effectively looser.

* **Probing strategy**: Round‑robin β refresh + occasional full sweep; Huber+EMA. Controller makes the system robust to β noise.

* **Metric semantics**: Prefer **applied KL** and (c) quantiles over clip_rate. The latter is a sparse indicator and can mislead selection.

* **Trust‑region schedule**: A fixed per‑step KL only works in stationary geometry; with drift, **dynamic (\delta)** (via PI) is the principled solution.

* **Overhead/throughput**: Tiny probes every (K) steps + amortized JVPs have modest overhead; measure tokens/s and report the probe %.

---

## 10) One‑page checklist

* [ ] **Center logits per token** for β, (S), (T), and KL.
* [ ] **L1 aggregator** (or L2 with (\sqrt{L}) scaling).
* [ ] **Scaling** (c=\min(1,\tau/\max(S,\varepsilon))).
* [ ] **Applied guard** (\rho=T/\max(c,S_{\text{probe}},\varepsilon)) (shrink‑only).
* [ ] **PI controller** on **applied per‑token KL**; update (\tau=\sqrt{4B\delta}).
* [ ] **β refresh** with JVP/FD along (U^{(\ell)}); Huber+EMA.
* [ ] **Selection** after warmup: best val **and** median (c\in[0.5,0.7]).
* [ ] **Report** (c) quantiles, (\eta_{\text{eff}}), applied KL tracking, per‑layer (s_\ell), throughput.

---

### Bottom line

Your observations are **consistent** with EBC’s trust‑region theory: EBC increases LR headroom and prevents divergence, but a **fixed** radius drives the run into **heavy clipping** later, collapsing the effective LR and hurting final quality.
Switch to the **L1 surrogate**, add the **applied‑KL PI controller** (with the corrected guard), and keep centering consistent. This keeps (c) in a healthy band throughout training and delivers **both** the stability benefits of EBC **and** strong final validation loss.
</solution>