# 01 Methods 与 Results 草稿（可粘贴骨架）

> 使用说明：本文件是**可直接改写进论文的骨架**，每个数字后面标注其证据编号（见 `00_证据清单.md`）。
> 凡本文件未出现的数字，不要写进论文。

---

## 3. Method

### 3.1 Problem setting

We forecast hourly PM2.5 at a **target station** using two information sources:
(i) the target station's own history, and (ii) the histories of its neighbouring stations.
Formally, for history length `L` and horizon `H`, the input is `X ∈ R^{S×L}` (S = number of collocated stations,
1 feature per station) and the target is `y ∈ R^{1×H}` (the target station's PM2.5 at the next `H` hours).
We evaluate on a fixed task grid `L ∈ {24, 48, 72, 168}` × `H ∈ {1, 3, 6, 12, 24}`.

### 3.2 Backbone and spatial residual（方向 0）

A PatchTST backbone processes **only the target station's channel** and produces
`base_prediction = PatchTST(center_x)`.
Neighbour information enters through a **bounded prediction-end residual**:

```
forecast_residual = α · tanh( spatial_forecast_head( neighbour_context ) )
prediction        = base_prediction + forecast_residual
```

Three components define the released configuration:
1. **Sparse top-k gating** over neighbours (`k = 5`, chosen on the validation period);
2. **Station identity bias** — a learnable per-neighbour prior added to the gate logits;
3. **Degraded initialisation** — the backbone is initialised from a degraded PatchTST and then **frozen**;
   only the spatial branch is trained.

Two properties make the design auditable: the residual head is **zero-initialised**, so at initialisation
(and whenever `α = 0`) the model is *exactly* the frozen backbone; and `forward_components(x)` /
`spatial_components(x)` expose the gate and residual for intervention studies.

> **Scope statement (must appear in the paper)**: the frozen-backbone condition is **part of the method**,
> not an implementation detail. Jointly training the backbone with the spatial head degrades results
> severely (−23.47% on average; −37.6% / −52.8% at L=72 / 168) [B4].

### 3.3 Training protocol（所有实验共用）

AdamW (`lr = 1e-3`, `weight_decay = 1e-4`), `ReduceLROnPlateau(factor = 0.5, patience = 3, mode = min)`,
MSE loss, gradient clipping 1.0. Epochs / early-stopping patience / batch size:
40 / 8 / 256 for `L ≤ 48`, 30 / 6 / 512 for `L > 48`. Model selection uses the **validation split only**;
the test split is evaluated **once**. All comparisons are **paired by seed**.

### 3.4 Data splits

Temporal 70/10/20 train/validation/test split. Neighbour screening (which stations enter the neighbourhood pool)
uses the **training period only** (leakage-free protocol). The earlier full-series screening variant is retained
only as a control and is *not* used for the paper's numbers [B7].

### 3.5 Baselines

**Single-station (same information as `center_x` only, never neighbours).** Two families:
- **Recurrent**: GRU/LSTM over the target station's channel, capacity curve `hidden ∈ {8,16,32,40,48,64}`
  (≈220 – 13.3k trainable parameters); the arm with `hidden = 40` (≈5.2k) is the **trainable-budget-matched**
  point for our spatial branch (4,588–5,418).
- **Traditional**: persistence, daily naive, climatology, AR, Ridge, linear spatial regression (deterministic arms).

**Transformer family (added to close a reviewer gap).**
- **Informer**: the repository's `LTSF_Informer` (ProbSparse attention), capacity-matched grid
  `d_model ∈ {8,12,16,32}` with `d_layers = 1`, `factor = 5`, `distil = True`. Informer's parameter count is
  **independent of (L, H)**, which makes exact capacity matching possible: `d_model = 12, e_layers = 2`
  gives **5,125** parameters, inside our branch's range.
- **TST** (tsai): included **as-is** only. tsai's TST head is `Linear(q_len × d_model → c_out)`, so its parameter
  count grows with both `L` and `H` (969 – 105,648); it therefore **cannot** be capacity-matched and is reported
  separately [B8].

All baselines use the **same training loop, optimiser, schedule, early-stopping rule, splits and seeds** as the
main experiments; only the model class differs. Compliance checks (10 items) include a *skeleton-reproduction gate*:
the reimplemented baseline runner reproduces the archived GRU reference result bit-for-bit
(RMSE 20.28444099426269) [C].

---

## 4. Results

### 4.1 The spatial residual improves the frozen backbone on both headline tasks

With five new seeds (2047–2051): mean paired RMSE reduction vs the same-seed degraded PatchTST
**3.11876%** (24→1) and **0.82199%** (168→6); 5/5 seeds in both tasks; one-sided exact sign test
**p = 0.03125** [A1]. The five-round attempt sequence, including the four unsuccessful rounds,
is reported in full [A2], and neighbour-intervention ablations confirm the gain depends on neighbour values [A3].

### 4.2 The gain is present across the task grid（20/20 configurations）

Under the leakage-free protocol the reduction is positive in **20/20** configurations (mean **2.013%**),
from **2.907%–4.285%** at `L = 24` to **−0.176%** at `L = 168, H = 24` [A4].
The single non-improving cell is reported as-is.

### 4.3 Frequency-domain enhancement (attached experiment, honestly bounded)

Adding a frequency branch on top of the frozen ST model improves RMSE by **0.054%–1.840%**
(mean **0.635%**), for a cumulative **0.100%–6.047%** reduction vs PatchTST [A5].
We also report the **equal-capacity time-domain control**, which is *stronger* than the frequency branch
on average [B5]; we therefore treat frequency as an engineering add-on, not as a contribution.

### 4.4 Cross-city generalisation

Transferring the frozen structure to **8 previously unused centre stations in Guangzhou** (new seeds 7001–7005,
confirmation period never used for structure selection): pooled paired RMSE reduction **2.2072%** (24→1, 40/40 pairs)
and **1.8224%** (168→6, 37/40 pairs); **8/8 stations** and **5/5 blocks** agree in direction; SMAPE improves in both tasks [A6].

### 4.5 Comparison with Transformer baselines（方向 21）

**Matched trainable budget.** At 5,125 trainable parameters (`informer_d12_e2`, our branch has 4,588–5,418),
the Informer is **4.66% worse** than our model. Across the Informer capacity grid our model is better at
**every** tested capacity: +24.11% (1,609), +16.95% (5,777), and **+9.01% at 21,793 parameters — 4.2× our budget** [A7].
The advantage holds under **both** selection criteria: validation-selected **+5.13%**, test-selected (upper bound)
**+3.80%** [A8].

**Where the advantage lives.** Per-lead analysis shows the advantage is concentrated at short lead times
(H=6: we win 5–6 of 6 leads) and largely disappears at H=24 (6–9 of 24 leads) [A9].

**TST (as-is).** TST is within ±1.4% of our model (−0.49% to −1.36%) while using up to **20×** our trainable
parameters; we report it as a non-capacity-aligned reference [B8].

### 4.6 What the comparison does *not* show（必须保留）

At the same trainable budget, the **recurrent single-station baselines remain stronger**: GRU
(hidden=40, ≈5.2k) −4.32%, GRU (h32) −4.41%, LSTM (h16) −3.35%; the best traditional baselines
(Ridge −2.16%, AR −1.82%) are also slightly better [B1, B2]. A same-information-set multi-station GRU
(18 stations, 21k trainable) is better by **−2.82%** [B3]. We therefore position our contribution as
(a) a controlled, auditable spatial mechanism, (b) task-grid coverage, and (c) cross-city generalisation —
**not** as state-of-the-art accuracy over all baselines.

### 4.7 A methodological finding about model selection

Across the 12-arm recurrent capacity curve, the **pooled** Spearman correlation between validation loss and test
RMSE is 0.97, but the **within-configuration** correlation is ≈0 (−0.002 mean, −0.206 median): model selection by
validation loss is informative *across* tasks and approximately uninformative *within* a task for that family.
Informer behaves differently (within-configuration mean 0.790). Consequence: any claim of the form
"we compared against the best baseline" must state the selection criterion and report both criteria [A10, A8].

---

## 5. Limitations（草稿）

1. **Recurrent baselines are stronger at equal budget** (§4.6). Our model's advantage is over Transformer
   baselines and over its own degraded backbone, not over recurrent models.
2. **Long-horizon deficit.** The spatial branch loses effect at long lead times (§4.5) and at `L = 168, H = 24`
   the grid cell is negative [A4].
3. **Frozen backbone is a prerequisite** (§3.2): joint training degrades severely [B4], so the method does not
   benefit from end-to-end fine-tuning.
4. **Frequency branch is not an equal-capacity win** [B5].
5. **Single target pollutant and single centre station per city**; both cities' data have been previously used,
   so the cross-city result is an external-validity check rather than a blinded test [B7, D5].
6. **Leakage boundary.** The released numbers use training-period neighbour screening; the earlier variant used
   the full series and is kept only as a control [B7].

## 6. Reproducibility statement（草稿）

All experiments use fixed seeds and a single evaluation of the test split; every reported number can be
recomputed from the archived predictions (SHA256-verified archives under `~/.herdr/artifacts/PatchTST/`).
Baseline runs pass a 10-item compliance check including an independent recomputation of every metric
(max relative discrepancy 3.4e-16) and a bit-level reproduction of a previously archived reference run.
The 20-run re-execution of the headline confirmation differs by 0.0.
