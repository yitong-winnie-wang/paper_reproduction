# Reproduction reference notes: Keskar et al. (ICLR 2017), Figure 3 panels F1, C1, and C2

**Purpose.** These are detailed technical notes for writing a reproduction report, **not** a claim of exact replication. They document (i) what the paper specifies, (ii) what the actual PyTorch notebooks implement or assume, (iii) what the completed checkpoint and measurement files contain, and (iv) how the measured results compare with the published figure and Table 2. Wherever a detail is not specified by the paper, it is identified as an **implementation choice** rather than attributed to the authors.

**Evidence examined (no external implementation or web search):**

1. Attached paper: Nitish Shirish Keskar et al., *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*, ICLR 2017, particularly §§2.2–2.2.1 (pp. 3–5), Figure 3 (p. 6), Table 2 (p. 4), Appendix A and B (p. 12), and Appendix D (p. 13).
2. `Keskar_Figure3_F1_C1_C2_Colab.ipynb` — the original from-scratch training/evaluation notebook.
3. `Keskar_Figure3_C2_Only_Recovery.ipynb` — recovery notebook used to preserve the completed F1/C1 measurements and evaluate C2 with per-point autosaving.
4. User's completed archive, `keskar_fig3_f1_c1_c2-20261021T013525Z-1-001.zip`: all six `*_best.pt` and six `*_progress.pt` checkpoint files, three `*_freshly_measured_curve.csv` files, `experiment_config.json`, the final figure PNG, and saved original-BN endpoint diagnostics for F1/C1. The numbers reported below were checked against these files, not merely transcribed from screenshots.

**Bottom line.** All six SB/LB training runs reached a *notebook-defined* stopping condition. All **183 interpolation locations** (61 each for F1, C1, C2) were evaluated using trained checkpoint weights; each location has four measured quantities. The resulting panels reproduce the qualitative phenomenon of high loss sensitivity around LB solutions and yield higher SB than LB *test accuracy* in all three networks. However, **C1 endpoint test accuracy is ~12 percentage points below the paper in both regimes; C2's LB result and extrapolation are also notably different.** This is best described as a **qualitative, partial reproduction**, not exact quantitative replication.

---

## 1. Precise task and paper-grounded experimental design

### 1.1 Which three panels?

| Original panel | Paper network | Dataset | Training examples | Test examples | Small-batch (SB) size | Large-batch (LB) size |
|---|---|---|---:|---:|---:|---:|
| Figure 3(a) | **F1** | MNIST, 28×28 grayscale, 10 classes | 60,000 | 10,000 | **256** | **6,000** |
| Figure 3(c) | **C1** | CIFAR-10, 32×32 RGB, 10 classes | 50,000 | 10,000 | **256** | **5,000** |
| Figure 3(d) | **C2** | CIFAR-10, 32×32 RGB, 10 classes | 50,000 | 10,000 | **256** | **5,000** |

**Paper sources:** Table 1 identifies models/datasets; Appendix A/Table 5 gives dataset dimensions; §2.2 states 256-example SB batches and LB batches equal to 10% of the training set. The paper reports **Adam in both regimes**, and says training continued until the loss stopped improving, without a fixed budget. Its published Table 2 summaries average **five** runs from random starting points, while Figure 3 uses **one randomly chosen SB/LB pair** from those trials.

### 1.2 What is plotted?

Let the optimized *trainable* parameter vectors be \(\theta_{\mathrm{SB}}\) and \(\theta_{\mathrm{LB}}\). For each architecture, the reproduced experiment evaluates

\[
\theta(\alpha)=(1-\alpha)\theta_{\mathrm{SB}}+\alpha\theta_{\mathrm{LB}},\quad -1\leq\alpha\leq 2.
\]

- \(\alpha=0\) is the small-batch checkpoint; \(\alpha=1\) is the large-batch checkpoint. Values in \((0,1)\) interpolate; values below 0 or above 1 **extrapolate** along that direction. Thus the displayed domain is wider than the segment connecting the solutions.
- Each \(\alpha\) produces four **measured** quantities: full-training-set mean cross-entropy, full-test-set mean cross-entropy, full-training-set top-1 classification accuracy (%), and full-test-set top-1 classification accuracy (%).
- **Left vertical axis:** cross-entropy (CE). **Right vertical axis:** accuracy (%). **Blue:** loss; **red:** accuracy. **Solid:** training; **dashed:** testing. This recreates the Figure 3 convention, although plot layout/axis limits are not identical.
- The paper uses mean cross-entropy as its objective (§2.2). Appendix D's separate **curvilinear** interpolation (Figure 7) and the paper's multidimensional sharpness metric (Eq. 4) **were not reproduced**.

The notebook measures a **one-dimensional, parameterization-dependent slice**. It does **not** measure the Hessian spectrum, general sharpness in all directions, or a causal relationship between sharpness and generalization.

---

## 2. Exact architectures implemented, and the architecture uncertainties

### 2.1 F1: MNIST fully connected classifier

**Paper (Appendix B.1):** 784 inputs, five batch-normalized hidden layers of 512 neurons with ReLU, and 10 softmax outputs.

**Code implements:** `Flatten(28×28→784)` → five repetitions of `Linear(...,512) → BatchNorm1d(512) → ReLU(inplace=True)` → `Linear(512,10)` raw logits. Width sequence: **784 → 512 → 512 → 512 → 512 → 512 → 10**. `torch.nn.functional.cross_entropy` accepts the unnormalized logits and computes stable log-softmax internally; putting an explicit softmax before it would be incorrect. No dropout is used in F1.

**Trainable parameter count checked from checkpoint:** **1,462,794**. As with the other models, final output logits have no batch normalization or dropout.

### 2.2 C1: CIFAR-10 shallow modified AlexNet

**Paper (Appendix B.3):** two stages, each described as `[64,5,5,2]–MaxPool(3)` (64 filters, 5×5 convolutions, convolution stride 2, then max pooling of size 3), followed by hidden dense layers of **384** and **192**, then 10 outputs. BN/ReLU are specified; dropout retention probability is 0.5 on the two dense hidden layers.

**Code's *specific* layer sequence:**

1. `Conv2d(3,64,kernel_size=5,stride=2,padding=2) → BatchNorm2d(64) → ReLU → MaxPool2d(kernel_size=3,stride=2,padding=0)`.
2. `Conv2d(64,64,kernel_size=5,stride=2,padding=2) → BatchNorm2d(64) → ReLU → MaxPool2d(kernel_size=3,stride=2,padding=0)`.
3. Flatten **64×1×1 → 64** features.
4. `Linear(64,384) → BatchNorm1d(384) → ReLU → Dropout(p=0.5)`.
5. `Linear(384,192) → BatchNorm1d(192) → ReLU → Dropout(p=0.5)`.
6. `Linear(192,10)` raw logits.

**Feature-map sizes for the implemented padding/strides:** input **32×32** → first conv **16×16** → first pool **7×7** → second conv **4×4** → second pool **1×1**. This calculation is crucial: **C1 reaches a 1×1 spatial map before its dense layers.**

**Trainable parameter count checked from checkpoint:** **209,546**.

**Important uncertainty / possible explanation of the accuracy shortfall:** The paper does **not** give the padding values or C1 pooling stride (nor a complete framework-specific layer graph). The chosen `stride=2` on both `MaxPool(3)` layers, together with the chosen convolution padding, reduces the map to 1×1. Another legitimate interpretation could retain more spatial features and yield a significantly different network. This is an **especially plausible** contributor to C1's lower accuracy, **but it has not been isolated experimentally**. Do not assert that this definitely caused the discrepancy or that the authors necessarily used another stride.

### 2.3 C2: CIFAR-10 deep modified VGG

**Paper (Appendix B.4):** five blocks containing 2, 2, 3, 3, and 3 convolutions of 3×3, stride 1, with 64, 128, 256, 512, and 512 filters respectively; a `MaxPool(2)` after each block; a dense layer of 512; final 10-class output. The paragraph under the C2/C4 heading mistakenly calls the described architecture “C3”; the notebook reasonably interprets this as **C2**, following the heading and Table 1.

**Code implements:**

| Block | Code's convolutions | Normalization / activation | After block |
|---|---|---|---|
| 1 | 2 × `Conv2d(...,64,3,stride=1,padding=1)` | `BatchNorm2d → ReLU` after each conv | `MaxPool2d(2,stride=2)` |
| 2 | 2 × `Conv2d(...,128,3,stride=1,padding=1)` | same | same |
| 3 | 3 × `Conv2d(...,256,3,stride=1,padding=1)` | same | same |
| 4 | 3 × `Conv2d(...,512,3,stride=1,padding=1)` | same | same |
| 5 | 3 × `Conv2d(...,512,3,stride=1,padding=1)` | same | same |

Spatial map progression is **32 → 16 → 8 → 4 → 2 → 1** after the five pools. Final head: flatten **512×1×1** → `Linear(512,512) → BatchNorm1d(512) → ReLU → Dropout(p=0.5)` → `Linear(512,10)` logits.

**Trainable parameter count checked from checkpoint:** **14,991,946**; **13 convolutional layers**. The paper does not state the chosen “same” convolution padding, exact BN placement, PyTorch layer initialization, or the precise dropout ordering.

### 2.4 Batch normalization defaults and initialization

For all networks, hidden layers are implemented as **Linear/Conv → BN → ReLU**, with dropout **after** the chosen dense-layer ReLU. PyTorch BN defaults are `eps=1e-5`, `momentum=0.1`, `affine=True`, and `track_running_stats=True`; its affine scales begin at 1 and shifts at 0. These defaults apply **during training**, not the separate recalibration procedure below.

Linear/Conv layers use their **native PyTorch default initialization**, not a custom replication of the paper's unspecified uniform distribution. In PyTorch's standard modules this is a Kaiming-uniform-style weight initialization with corresponding default biases. Model-family seeds differ; within each family, the two optimizer regimes receive **identical initial model parameters**. Thus the notebook controls initialization within each SB/LB pair, but does **not** establish what initialization was used in the authors' original Figure 3.

---

## 3. Datasets, preprocessing, and evaluation populations

- **Data source:** standard `torchvision.datasets.MNIST` and `torchvision.datasets.CIFAR10`; dataset sizes are those in Appendix A. Both train and test datasets are used in full for curve evaluation.
- **Stored input representation:** original `uint8` image data. MNIST is expanded to `N×1×28×28`; CIFAR-10 is transposed from `NHWC` to `NCHW` (`N×3×32×32`). Targets are integer class IDs of dtype `torch.long`.
- **Only numerical pixel transform:** transfer to device, convert to `float32`, **divide by 255.0**, producing inputs in `[0,1]`.
- **Not applied:** mean subtraction, per-channel standardization, whitening/ZCA, cropping, flipping, rotations, translations, color jitter, random augmentation, a validation split, or pretrained features.
- **Paper comparison:** Appendix A says MNIST and CIFAR datasets were used without preprocessing; converting the byte encoding into normalized floating-point numerical inputs is an **implementation convention not literally specified in the text**. Appendix E studies data augmentation separately; those Appendix E interventions were **not** used in the Figure 3 baseline reproduction.
- **Metric calculation:** for each evaluation loader, accumulate `cross_entropy(logits, labels, reduction="sum")` and correct argmax predictions over **every sample**. Divide the summed loss by number of examples; report top-1 `correct / count × 100`. `model.eval()` disables dropout and uses currently stored BN statistics. Training metrics use 60,000 (F1) or 50,000 (C1/C2) examples; every test measurement uses 10,000. **No subset approximation** is made to the plotted training/test metrics; only BN calibration uses a subset.
- **No test leakage:** test samples/labels are not used to train weights, choose checkpoints, or estimate BN statistics.

---

## 4. Training: optimizer, batching, checkpoint selection, and reproducibility

### 4.1 Adam, loss, and optimizer steps

| Parameter / policy | Actual value or behavior | Specified by paper? |
|---|---|---|
| Optimizer | PyTorch `torch.optim.Adam` for SB and LB | **Adam yes**, these settings no |
| Learning rate | **0.001**, fixed | Not stated |
| Adam betas | **(0.9, 0.999)** | Not stated |
| Adam epsilon | **1e-8** | Not stated |
| Weight decay | **0.0** | Not stated |
| LR scheduler | **None** | Not stated |
| Gradient clipping | **None** | Not stated |
| Loss | Mean categorical cross-entropy of raw logits | Mean CE yes |
| Epochs | Up to **100** | Paper has no fixed cap |
| Significant training-CE improvement | More than **1e-4** absolute | Not stated |
| Plateau patience | **15** consecutive epochs without significant improvement | Not stated |
| Checkpoint selection | **Best full-training-set CE**, examined after every epoch | Precise policy not stated |
| Checkpoint frequency | Every **1 epoch** for progress; best checkpoint on qualifying improvement | Not stated |
| Number of independent SB/LB pairs | **1 per architecture** | Paper averaged 5 runs, plotted one random pair |

**Important checkpoint detail:** “Best” is defined by an improvement greater than `min_delta=1e-4`, not necessarily the mathematically lowest CE ever encountered if a smaller later reduction occurred. The saved **best epoch may precede the stop epoch** by up to the plateau waiting period. The published paper trained until CE ceased to improve, but did not publish these tolerances or select checkpoints in this exact documented manner. A stop at `epoch_cap` is *not* evidence of the paper's unconstrained convergence criterion.

**Training loop semantics:** each epoch shuffles the complete training set; data are grouped into *effective* SB or LB batches; for each effective batch, optimizer gradients are zeroed, each physical microbatch contributes a per-example summed CE gradient divided by the **actual effective-batch example count**, and **one Adam update** is made after all microbatches. Thus all examples contribute with their correct effective-batch weight, including undersized last effective batches.

### 4.2 Effective versus physical batch sizes (T4 accommodation)

| Architecture | SB effective | LB effective | Physical microbatch | Evaluation batch | BN-calibration batch | Adam steps/epoch (SB/LB) |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 256 | 6,000 | **256** | **1,024** | **256** | **235 / 10** |
| C1 | 256 | 5,000 | **128** | **256** | **128** | **196 / 10** |
| C2 | 256 | 5,000 | **32** | **64** | **32** | **196 / 10** |

The number of optimizer updates per epoch is `ceil(training_examples / effective_batch)`. **Balanced** `torch.tensor_split` creates nearly equal physical pieces, avoiding a final piece of size 1 that would cause problems for BN. Examples: C1's 5,000-sample LB batch becomes **40 microbatches of 125**; C2's becomes **157 microbatches of approximately 31–32**; and C2's 256-sample SB batch becomes **eight microbatches of 32**. F1's physical size 256 means a normal SB effective batch need not be split.

The gradient accumulation reproduces **effective-batch sample weighting**, *but not exactly the gradient of an unsplit, large-BN-batch network*. BN training statistics come from **physical microbatches** (32 for C2, about 125 for C1's LB batches), not the complete 5,000/6,000 effective batch. This approximation is necessary for one T4's memory budget and is a substantive possible source of endpoint differences. This caveat applies even though Adam is stepped only once per effective batch.

### 4.3 Seeds, ordering, computation, and environment

- Global seed: **1729** for Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA when available.
- Model initialization seeds: **F1 = 1729**, **C1 = 11729**, **C2 = 21729**. `initialize_identically` re-seeds before creating each SB or LB model. Identical initialization within each pair was **verified from the stored `initial_digest` hashes**; initializations are different across model families.
- Epoch training shuffle uses a new `torch.Generator` seeded with `model_initial_seed + epoch`, **plus 1,000,000 for LB**. Thus SB and LB use **different data orderings** despite sharing initial weights. Both have independent freshly initialized Adam optimizer state.
- Model parameters and evaluation input are **float32**. **No AMP/mixed precision**; TF32 matmul/convolution acceleration is explicitly disabled (`allow_tf32=False`); `cudnn.benchmark=False`.
- **No claim of bitwise determinism:** no explicit `torch.use_deterministic_algorithms(True)` or full deterministic cuDNN enforcement; Colab/PyTorch/CUDA/torchvision versions are **not pinned** in the notebook or recorded in the archived `experiment_config.json`. Actual runtime versions and exact hardware for each stage should **not be invented**. The workflow was designed for **Colab with one T4**, and a CPU-to-GPU switch was part of the recovery process.
- DataLoaders: `num_workers=1`, `prefetch_factor=1` when workers are used, `persistent_workers=True`, and `pin_memory=True` if CUDA is available; nonblocking CPU→device copies when possible. These are performance-oriented choices, not paper prescriptions.
- Output and download directories: `/content/drive/MyDrive/keskar_fig3_f1_c1_c2` (Google Drive persistent outputs), `/content/keskar_fig3_f1_c1_c2` (local fallback), `/content/keskar_fig3_datasets` (datasets).

### 4.4 Checkpoint and recovery mechanics

Each network/regime creates two files:

- `F1_SB_best.pt`, `F1_LB_best.pt`, etc.: trained **model state including BN buffers**, network/regime label, best epoch, training CE/accuracy, training-signature hash, and initial-model hash. These are the **six endpoints actually used to produce the curves**.
- `F1_SB_progress.pt`, etc.: current model, **Adam optimizer state**, completed epoch, best loss, plateau count, stop reason (`plateau` or `epoch_cap`), CPU/CUDA RNG state, and signature. These permit resuming training at an epoch boundary without retraining completed epochs.

Saves use a temporary file and rename to reduce risks from partial writes. `read_and_validate_endpoints` checks network identity, SB/LB regime, configuration signature, model state keys, and shared-initialization digest before interpolation. Training signatures *exclude* the epoch cap to permit extending a capped run, while including batch sizes, learning parameters, microbatch size, initialization seed, plateau settings, and input divisor.

**What happened in this completed run:** the original Figure 3 interpolation was interrupted during C2. **F1/C1's finished measured points were saved to CSV before runtime changes.** The separate recovery notebook loaded those existing CSVs and six saved endpoint checkpoints, evaluated only C2, and **atomically rewrote C2's CSV after every new alpha point**, allowing restart/resume. The final combined PNG was drawn from the three **previously measured CSVs**, not by evaluating all three networks again in one plotting call. This recovery mechanism does **not** change the calculation of an individual point, but the report should accurately describe the final visualization as using **checkpoint-derived cached measurements**.

---

## 5. BN recalibration and initialization: two deliberate methodological decisions

### 5.1 How to handle BN at parameters that were never trained

The paper does **not** specify how BN running means/variances are handled at interpolated \(\alpha\). Three possible procedures were considered:

1. **Interpolate the SB/LB stored BN running means and variances** together with trainable parameters. This preserves the endpoints' original statistics but need not describe the activation distributions of intermediate networks.
2. **Keep one endpoint's running statistics** throughout the path. Simple, but arbitrarily favors that endpoint.
3. **Re-estimate BN statistics at every \(\alpha\)** from training data, which more directly normalizes the interpolated network and does not use test samples, but makes each measurement depend on the calibration protocol.

**Implemented:** option **3**, applied *also at* \(\alpha=0\) and \(\alpha=1\), so every plotted point uses one consistent policy:

1. Copy the interpolated **trainable parameters** into a model using `torch.lerp`, including extrapolation for \(\alpha<0\) and \(\alpha>1\). **Do not interpolate running-statistic buffers or the BN `num_batches_tracked` buffers.** BN trainable scale/shift *are* interpolated because they are named parameters.
2. Set the whole model to **evaluation mode** so all dropout layers are disabled; reset each BN layer's running statistics/count; switch **only BN modules** to training mode.
3. Temporarily set BN `momentum=None` for cumulative **equal-calibration-microbatch** averaging. Run **2,048 fixed training examples** in forward-only/no-gradient mode. Inputs are selected once by a deterministic permutation using seed `1729 + 333333 = 335062`. For CIFAR-10 this is the same subset of indices for C1 and C2. BN-calibration minibatches are **256** for F1, **128** for C1, **32** for C2, which divide 2,048 exactly: **8**, **16**, **64** calibration batches, respectively.
4. Restore BN `momentum` settings, call `model.eval()`, and calculate all four metrics over the **full** train/test datasets.

**Caveat:** cumulative BN statistics average *per-microbatch* moments, and the calibrating forward pass normalizes using each microbatch's statistics. This is a deterministic, approximate inference convention, **not** an exact full-training-set population-moment calculation. Crucially, the measured curve is not literally a line in the enlarged space of **trainable weights plus BN running buffers**: only trainable weights follow the straight line; BN buffers are **recalculated as functions of the interpolated weights**. A report should explicitly state this difference.

### 5.2 Same or independent starting weights?

**Implemented:** share the same initialized model parameters within each F1/C1/C2 SB/LB pair. Only training batch regime, its consequent step count/noise/BN-microbatch grouping, and distinct shuffle stream change. Adam states are not shared. Initial hashes in the archived endpoints confirm equality within each pair.

**Why:** the paired initialization reduces a major confound; a path between independently initialized networks can reflect unrelated solutions and hidden-unit permutation symmetries as much as the batch-size intervention. **Limitation:** the paper reports five random initializations but does *not* say that the selected SB/LB Figure 3 pair shared one initial vector. Consequently, these curves need not be geometrically identical to the published randomly chosen pair's curves. Parameter interpolation is also not invariant to model reparameterization, so even identical initialization does not turn a 1-D slice into a rigorous all-direction sharpness measure.

### 5.3 Why endpoint metrics differ between tables/CSV and checkpoint logs

At \(\alpha=0\) and 1, the trainable parameters equal those of the saved checkpoints. But the plotted curve uses **freshly recalibrated BN running statistics**, not the checkpoint's **training-time BN buffers**. This explains why `*_best.pt` training CE and its saved/evaluated original-BN metrics need not coincide exactly with the plot's endpoints. The original paper's Table 2 BN evaluation convention is unknown. Do **not** silently mix original-BN metrics and recalibrated endpoints in a quantitative table.

---

## 6. What actually finished: verified training-run metadata

The following values were read from the six `*_progress.pt` stop records and six `*_best.pt` checkpoints in the ZIP. “Best training CE” and “best training accuracy” refer to the selected original checkpoint/training-time BN state, **not** the recalibrated Figure 3 endpoint.

| Network | Regime | Last epoch | Termination | Best epoch | Best training CE (saved) | Best training acc. (saved) | Model parameters |
|---|---|---:|---|---:|---:|---:|---:|
| F1 | SB | **79** | plateau | **64** | **0.00015874** | **99.9983%** | 1,462,794 |
| F1 | LB | **69** | plateau | **54** | **0.00030598** | **99.9950%** | 1,462,794 |
| C1 | SB | **96** | plateau | **81** | **0.06268192** | **98.0140%** | 209,546 |
| C1 | LB | **88** | plateau | **73** | **0.14343487** | **95.3140%** | 209,546 |
| C2 | SB | **100** | **epoch cap** | **87** | **0.00306669** | **99.9080%** | 14,991,946 |
| C2 | LB | **71** | plateau | **56** | **0.00391533** | **99.8940%** | 14,991,946 |

**Interpretation:** Five of the six runs met the notebook's plateau criterion; **C2 SB reached the 100-epoch safety cap with a plateau counter of 13/15**. Its best stored checkpoint is epoch 87. It is inaccurate to say that every run met the paper's unconstrained convergence criterion. All six checkpoints and associated progress states are present; *no training rerun was needed* after disconnection.

**Optimizer-step counts from completed epochs:** given the fixed full-dataset training loop, each run executed **18,565** steps (F1 SB), **690** (F1 LB), **18,816** (C1 SB), **880** (C1 LB), **19,600** (C2 SB), and **710** (C2 LB) through its recorded last epoch. Their *best* checkpoints were selected earlier than the last epoch in every case, as listed above. These update totals follow the source code and verified stop epochs; they are not hyperparameters copied from the paper. Per epoch, LB receives only 10 Adam updates versus 196–235 for SB, even though all training examples are processed in each epoch.

**Synthetic sanity checks (enabled):** before the expensive training stage, the original notebook creates one model of each type, feeds **two randomly generated input images** of the expected size, verifies output shape `(2,10)` and finite logits, and prints the parameter counts. These random inputs are **only an implementation smoke test**, never training examples, test data, interpolation metrics, or pretrained parameters.

**No five-run statistics:** one SB/LB pair per architecture was produced. There is **no confidence interval, standard deviation, or uncertainty estimate** for the reproduction's accuracy gaps. A single sample cannot establish how frequently the behavior occurs over initialization/seed choices.

---

## 7. Measured Figure 3 endpoint values (BN-recalibrated evaluation)

These values come **directly from the three CSV files**, at \(\alpha=0\) (SB) and \(\alpha=1\) (LB). Values are rounded for this table; the CSVs contain greater numerical precision.

| Network | Regime / alpha | Train CE | Test CE | Train accuracy | Test accuracy |
|---|---|---:|---:|---:|---:|
| F1 | SB / 0 | **0.0001893** | **0.0767580** | **99.9967%** | **98.48%** |
| F1 | LB / 1 | **0.0003643** | **0.0728129** | **99.9933%** | **98.32%** |
| C1 | SB / 0 | **0.0447634** | **1.8928047** | **98.5940%** | **67.32%** |
| C1 | LB / 1 | **0.1148020** | **1.5820377** | **96.5480%** | **65.07%** |
| C2 | SB / 0 | **0.0029544** | **0.7897766** | **99.9040%** | **86.71%** |
| C2 | LB / 1 | **0.0038129** | **0.9292034** | **99.9040%** | **81.51%** |

### 7.1 Test-accuracy reproduction versus original paper (Table 2)

| Network | Paper SB test acc. | Our SB | Difference (ours − paper) | Paper LB test acc. | Our LB | Difference (ours − paper) |
|---|---:|---:|---:|---:|---:|---:|
| **F1** | **98.03 ± 0.07%** | **98.48%** | **+0.45 pp** | **97.81 ± 0.07%** | **98.32%** | **+0.51 pp** |
| **C1** | **80.04 ± 0.12%** | **67.32%** | **−12.72 pp** | **77.26 ± 0.42%** | **65.07%** | **−12.19 pp** |
| **C2** | **89.24 ± 0.12%** | **86.71%** | **−2.53 pp** | **87.26 ± 0.07%** | **81.51%** | **−5.75 pp** |

*“pp” denotes absolute **percentage points**, not a relative-percent change. Paper ± values are published **five-trial standard deviations**; they are **not** uncertainty estimates for the single reproduction run. Our columns use the recalibrated Figure 3 endpoints; see §5.3 for the evaluation-policy mismatch.*

### 7.2 Test generalization gaps, SB accuracy minus LB accuracy

| Network | Paper Table 2 gap | Our gap | Difference in gap (ours − paper) |
|---|---:|---:|---:|
| **F1** | **0.22 pp** | **0.16 pp** | **−0.06 pp** |
| **C1** | **2.78 pp** | **2.25 pp** | **−0.53 pp** |
| **C2** | **1.98 pp** | **5.20 pp** | **+3.22 pp** |

**Key distinction:** All three reproduce the *sign* of the reported test-accuracy gap. **C2** exhibits a much larger gap than the paper; **C1** reproduces a gap of comparable order but has much lower absolute accuracies. The F1 difference is very small in both datasets and could be sensitive to trial-to-trial variation and evaluation policy. With only one run, the report should avoid significance claims.

### 7.3 Training accuracy comparison with Table 2

The published paper's Table 2 reports these training-accuracy means: **F1 SB/LB 99.66%/99.92%; C1 99.89%/99.66%; C2 99.99%/99.99%**. Our *recalibrated* endpoints are **F1 99.9967%/99.9933%, C1 98.594%/96.548%, C2 99.904%/99.904%**.

The relatively low **C1 LB training accuracy** matters: its **96.548%** is ~**3.11 pp** below the paper's **99.66%**, despite its checkpoint being judged complete by our plateau rule. Thus C1's problem is **not solely a test-set generalization effect**; even the large-batch *training fit* differs. C1 SB is also about **1.30 pp** below the paper's training result. F1 and C2 reach much closer-to-perfect training fits (depending on evaluation state).

### 7.4 Raw original-BN checkpoints versus plotted recalibrated endpoints

The ZIP includes `F1_original_endpoint_metrics.json` and `C1_original_endpoint_metrics.json` captured from the initial run, but **does not include C2's analogous original-BN full-test-set evaluation**. Do not invent it.

| Network/regime | Raw-checkpoint train CE | Raw-checkpoint test CE | Raw train accuracy | Raw test accuracy | Recalibrated test accuracy |
|---|---:|---:|---:|---:|---:|
| F1 SB | 0.0001587 | 0.0745242 | 99.9983% | **98.58%** | **98.48%** |
| F1 LB | 0.0003060 | 0.0731100 | 99.9950% | **98.27%** | **98.32%** |
| C1 SB | 0.0626819 | 1.8235480 | 98.0140% | **67.00%** | **67.32%** |
| C1 LB | 0.1434349 | 1.5664470 | 95.3140% | **64.19%** | **65.07%** |

These differences are real consequences of changing BN statistics for inference. Example: **C1 LB rises from 64.19% with saved BN buffers to 65.07% after recalibration**, while both methods remain well below the paper's C1 values. **Changing BN alone does not eliminate the main accuracy discrepancy.** For C2, only its stored *training* metric at the best epoch is verified; its raw-BN **test** metric is not available in the uploaded archive.

**Subtle but noteworthy:** for F1 and C1, the LB endpoint has **lower test CE** than the SB endpoint (**F1 0.0728 vs 0.0768; C1 1.5820 vs 1.8928**) while it has *lower test accuracy*. Cross-entropy and zero-one accuracy are different criteria, so these observations are not contradictory. C2 has the more intuitive alignment (LB has both higher test CE and lower test accuracy). Avoid writing that the LB test loss is always worse in this reproduction.

---

## 8. The measured curves: numerical landmarks and visual comparison to original Figure 3

The following landmarks come from **actual CSV rows**. The “paper figure” shape/magnitude comparisons after them are **approximate visual readings from the printed Figure 3**, not paper-supplied numerical coordinates. Do not present the paper's graphical estimates as exact published numbers.

### 8.1 Measured intermediate barriers and extrapolation

| Network | Largest *interior* training CE for 0 < α < 1 | Test CE there | Train/test accuracy there | Training CE at α=2 | Test CE at α=2 | Train/test accuracy at α=2 |
|---|---|---:|---|---:|---:|---|
| **F1** | **0.00672** at **α=0.90** | 0.07831 | 99.77% / 98.06% | **2.9143** | **2.8610** | **35.66% / 35.34%** |
| **C1** | **2.0060** at **α=0.80** | 2.6592 | 55.81% / 47.98% | **5.8795** | **5.9410** | **9.02% / 8.56%** |
| **C2** | **2.7093** at **α=0.90** | 2.8371 | 38.54% / 36.46% | **16.7235** | **16.8853** | **11.84% / 11.86%** |

These “interior maximum” entries select only sampled values in \(0<\alpha<1\) and should **not** be interpreted as the global maximum over the whole \([-1,2]\) path. F1's large extrapolation loss is, for example, higher before \(\alpha=2\) than at the right edge.

### 8.2 F1: similar cliff, different right-hand extrapolation

- **Original Figure 3(a):** train and test CE are almost flat through the SB endpoint and LB endpoint, with a sharp climb just to the right of LB (roughly \(\alpha\gtrsim1.1\)). The paper's loss reaches about **11–12** near \(\alpha=2\), and accuracy collapses toward zero. These are **visual estimates** from the figure.
- **Our F1:** SB and LB are both near-perfect on the training set and around **98.5%** on the test set. CE is low over much of the line. After LB, CE rises sharply (**train CE 0.659 at α=1.10**, **3.768 at 1.20**) and reaches a local high near **4.405 at α=1.25**, then falls to **2.914 at α=2**. Training accuracy falls to around **1%** near \(\alpha=1.3\)–1.35 and then *recovers* to **35.66%** by \(\alpha=2\).
- **Assessment:** the **cliff just beyond LB** matches the paper qualitatively; the **nonmonotone loss and accuracy recovery** on the far right do **not** match its visible extrapolation shape. This is not a reason to alter or smooth the measured data.

### 8.3 C1: two low-loss endpoints but a taller barrier and lower accuracies

- **Original Figure 3(c):** SB and LB appear as low-training-loss regions, separated by a modest bump near roughly \(\alpha=0.7\)–0.8. The red accuracy curves dip near that bump but remain substantially higher than in our plot. The test accuracy at the two endpoints is near the paper's Table 2 values, **80.04%/77.26%**.
- **Our C1:** train CE rises from **0.0448 (SB)** to **2.0060 at α=0.8**, then falls to **0.1148 at LB**. Training accuracy at that interior peak is only **55.81%** (test **47.98%**), and test accuracy at the endpoints is only **67.32%/65.07%**. CE grows strongly again beyond LB, reaching **5.8795 train / 5.9410 test** at \(\alpha=2\).
- **Assessment:** the **two-minimum-plus-interior-barrier** and steep post-LB rise are reproduced, but the **interior barrier is more disruptive**, and the **~12-pp endpoint test-accuracy shortfall** is the major quantitative reproduction failure. A critical additional finding is that C1 LB **training** accuracy is below the paper's, too. The under-specified C1 pooling/padding choices and consequent **1×1** feature map are specific, testable hypotheses; do not claim an established diagnosis.

### 8.4 C2: similar narrow LB valley, peak shifted toward LB, stronger extrapolation

- **Original Figure 3(d):** two low-training-loss endpoints with a **narrow loss bump around roughly \(\alpha=0.7\)–0.8** (visual estimate), then recovery by \(\alpha=1\); loss rises again after LB. The published right edge has a training-CE scale of roughly **6**. Original endpoint test accuracies were **89.24%/87.26%** (Table 2).
- **Our C2:** SB has **train CE 0.00295/test accuracy 86.71%**, LB has **train CE 0.00381/test accuracy 81.51%**. The interior peak is later, at **α=0.90**, where train/test CE reach **2.7093/2.8371** and accuracy falls to **38.54%/36.46%**. The loss then drops sharply again at **α=1.00**. Extrapolation reaches **16.7235/16.8853** train/test CE at \(\alpha=2\), much larger than in the original panel.
- **Assessment:** the distinctive **narrow, strongly direction-sensitive LB minimum** and loss-barrier geometry resemble the paper, but the **peak location**, **LB test accuracy**, **5.20-pp gap** (vs published 1.98 pp), and large right-hand loss are materially different.

### 8.5 Cautions when describing “sharpness”

- A high CE near—but not at—the LB solution demonstrates **sensitivity along this particular SB↔LB direction**. It is **not** the paper's Eq. (4) sharpness metric, nor a measured Hessian eigenvalue.
- The horizontal variable is \(\alpha\), **not Euclidean parameter distance**. Because endpoint separation \(\|\theta_{\mathrm{LB}}-\theta_{\mathrm{SB}}\|\) can differ across models, the apparent width measured in \(\alpha\) is **not a normalized physical radius**. Comparing numerical slopes between F1/C1/C2 without accounting for direction length is not a robust cross-architecture sharpness ranking.
- BN recalibration may introduce additional dependence of the measured function on \(\alpha\). The shape reflects the **selected inference/BN policy** as well as the trainable parameter direction.
- The paper's original figure depicts **one selected pair**; this reproduction also has **one paired seed**, not a multi-seed estimate. Similar qualitative behavior does not establish generality beyond this run.
- A bump between two trained endpoints does not imply **every possible connection** between them has high loss; another nonlinear or symmetry-adjusted path may behave differently.

---

## 9. Specific differences from the paper and what could plausibly explain them

| Difference / under-specified element | What the reproduction did | Consequence / how to discuss it |
|---|---|---|
| **Optimization schedule** | Fixed Adam LR 0.001, no decay/scheduler, PyTorch default betas/epsilon | These values are not documented by the paper; training trajectory and solution basin can change. |
| **Stopping rule** | Patience 15, min improvement 1e-4, 100-epoch cap; best checkpoint by training CE | The paper says “until no improvement,” without a numeric tolerance or cap. **C2 SB stopped by cap, not plateau.** |
| **C1 padding/pool stride** | Convolution pad 2; two `MaxPool(3,stride=2)` layers → final 1×1 map | A concrete possible cause of **under-capacity or different representation** relative to the paper's unspecified implementation; needs an ablation to substantiate. |
| **BN evaluation policy** | Reset/recalibrate on fixed 2,048 training samples separately at every alpha, including endpoints | Can move endpoint accuracy/loss and curve shape; the paper's policy is undocumented. Raw-vs-recalibrated F1/C1 differences are measured (§7.4). |
| **BN during effective LB training** | Accumulated gradients with BN on smaller physical chunks | **Not equivalent** to computing a 5,000/6,000-example BN forward pass. Particularly consequential for C2 with 32-image microbatches. |
| **Initialization** | Same initial parameters for paired SB/LB, different seeded shuffle streams; PyTorch default initializer | Reduces one confound but may not reflect the paper's randomly selected pair or uniform-start distribution. Changes interpolation direction and valleys. |
| **Trials** | Single run/pair per model | No reproduction standard deviations; published Table 2 values average 5 trials. |
| **Input scale** | uint8 divided by 255; no per-channel normalization/augmentation | The paper does not spell out input numerical scale; may impact convergence/feature learning. |
| **BN/activation/dropout placement** | Conv/Linear→BN→ReLU, dropout p=0.5 after specified dense layers | Reasonable implementation of the verbal architecture, but exact original graph is unavailable from the PDF. |
| **Data/eval performance settings** | PyTorch/torchvision on Colab; 1 worker; architecture-specific eval/calibration microbatch | Not a paper setting; reproducibility depends on library/hardware versions and floating-point kernels. |
| **Figure panel count and visual scaling** | Only three panels in a 1×3 figure, automatic CE axis limits, fixed accuracy 0–100 | Paper has six panels and individually chosen CE/accuracy axis ranges. Figure size/axis range differences are **presentation**, not necessarily different model behavior. |
| **Recovery plotting** | F1/C1 computed in first evaluation; C2 recomputed separately and saved each alpha; final image from CSV | All numerical points still originate from trained checkpoints; final rendering reads cached measured values. This provenance should be reported transparently. |

**Priorities if further experiments are possible:** (1) verify/ablate C1's pooling/padding and resulting spatial dimensions; (2) test alternative BN policies while keeping the **same six checkpoints**; (3) compare BN behavior under different training microbatches, which *does* require retraining; (4) evaluate multiple common-initialization seeds; (5) extend C2 SB past the cap or test training schedules. None of these tests has been performed in the supplied results, so they should appear as **limitations or follow-up proposals**, not explanations proven by the current data.

---

## 10. Exact configuration preserved with the final figure

This is the configuration from **the archived `experiment_config.json`**, reproduced to make the choices independently checkable. The `recovery` field records the actual interruption workaround. All numeric values in the plot come from checkpoint-derived measurements; the settings below control the experiment or figure style rather than supplying curve values.

```json
{
  "training_and_evaluation": {
    "seed": 1729,
    "networks": ["F1", "C1", "C2"],
    "use_google_drive": true,
    "drive_dir": "/content/drive/MyDrive/keskar_fig3_f1_c1_c2",
    "local_dir": "/content/keskar_fig3_f1_c1_c2",
    "data_dir": "/content/keskar_fig3_datasets",
    "num_workers": 1,
    "pin_memory": true,
    "lr": 0.001,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "weight_decay": 0.0,
    "max_epochs": 100,
    "patience": 15,
    "min_delta": 0.0001,
    "save_every_epochs": 1,
    "bn_calibration_examples": 2048,
    "alpha_min": -1.0,
    "alpha_sb": 0.0,
    "alpha_lb": 1.0,
    "alpha_max": 2.0,
    "alpha_grid_points": 61,
    "input_divisor": 255.0,
    "test_run_synthetic_checks": true,
    "models": {
      "F1": {"dataset": "MNIST", "small_batch": 256, "large_batch": 6000,
             "physical_batch": 256, "eval_batch": 1024, "calib_batch": 256},
      "C1": {"dataset": "CIFAR10", "small_batch": 256, "large_batch": 5000,
             "physical_batch": 128, "eval_batch": 256, "calib_batch": 128},
      "C2": {"dataset": "CIFAR10", "small_batch": 256, "large_batch": 5000,
             "physical_batch": 32, "eval_batch": 64, "calib_batch": 32}
    }
  },
  "plot_style": {
    "rows": 1,
    "figsize": [18, 5],
    "dpi": 180,
    "line_width": 1.7,
    "train_ls": "-",
    "test_ls": "--",
    "loss_color": "#225bb3",
    "accuracy_color": "#ca3030",
    "accuracy_ylim": [0, 100],
    "grid_alpha": 0.35,
    "file_name": "figure3_F1_C1_C2_computed.png"
  },
  "recovery": "F1/C1 evaluated originally; C2 evaluated with per-alpha autosave"
}
```

The alpha grid is **61 equally spaced values** on `[-1,2]` (spacing **0.05**) and programmatically ensures 0 and 1 are present. The plotting code contains **no hard-coded measured accuracy or loss values**; the stored CSV files have columns:

```text
network,alpha,train_ce,test_ce,train_acc,test_acc
```

Additional settings implicit in the code rather than the JSON: native PyTorch BN defaults (`eps=1e-5`, momentum `0.1` during training); BN calibration momentum `None`; calibration random-index seed offset `333333`; family seed offsets `0/10000/20000`; LB shuffle seed offset `1000000`; default PyTorch Conv/Linear initialization; no mixed precision/TF32; no augmentation, no scheduler, no gradient clipping, no output softmax layer, and no BN on the final logits. Some of these remain library-version-dependent because exact package versions were not archived.

---

## 11. Recommended claims and wording for a reproduction report

### Findings that the archive directly supports

1. All six trained endpoint checkpoint pairs exist, and SB/LB initialization hashes match within F1/C1/C2.
2. The two stated effective batch regimes, Adam implementation, input transformations, architectures, plateau rules, and BN policy are documented in the supplied notebook/configuration.
3. Each of the three final curves contains **61 experimentally evaluated points**; loss/accuracy are based on complete train/test datasets after training-data-only BN recalibration.
4. SB **test accuracy** exceeds LB test accuracy for all three: **F1 +0.16 pp, C1 +2.25 pp, C2 +5.20 pp**.
5. C1/C2 show low training CE at both endpoints, higher CE between them, and steep sensitivity near the LB endpoint along the chosen direction. F1 shows a particularly pronounced post-LB cliff.
6. F1's test accuracies are slightly above the original means; C1 significantly underperforms; C2 SB modestly and LB substantially underperform.
7. C1's lower training accuracy, its implemented 1×1 final conv map, C2's `epoch_cap`, and BN-microbatch approximation are material qualifications to reproducing the original conditions.

### Conclusions to **avoid**

- “We exactly reproduced Keskar et al.'s Figure 3,” or “We matched all reported accuracies.” We **did not**.
- “We computed the paper's sharpness metric / Hessian eigenvalues.” We **did not**.
- “The LB solutions are sharper in every direction.” One directional slice does **not** establish this.
- “Gradient accumulation is identical to an unsplit LB batch.” **Not** with per-microbatch BN.
- “The paper used the same seed/Adam settings/BN calibration/padding/pool strides.” These are **undocumented in the attached PDF**.
- “All models converged under the paper's unconstrained stopping rule.” **C2 SB hit a 100-epoch cap**.
- “The C1 mismatch was definitely caused by pooling.” This is a **hypothesis**, not the output of an ablation.
- “The generalization gap is always accompanied by larger test cross-entropy.” Our **F1/C1** data contradict that blanket claim.
- “The figure itself demonstrates causality between sharpness and generalization.” It is illustrative/observational evidence, not a controlled causal proof.

### Possible careful summary paragraph (adapt into your own voice)

> We reproduced Figure 3(a), 3(c), and 3(d) of Keskar et al. (2017) for the F1, C1, and C2 networks using independently written PyTorch models trained with Adam at effective batch sizes of 256 and 10% of the training set. For each network, the two runs shared an initialization, and we evaluated mean cross-entropy and classification accuracy on the complete training and test splits at 61 parameter-interpolation positions from α = −1 to α = 2. We recalibrated BN statistics from a fixed training-only subset at each interpolation point. The reproduction recovered the qualitative behavior of low-loss endpoints and sharp sensitivity along the SB-to-LB direction, especially near the LB solutions, and found higher SB test accuracy in all three models. However, the reproduction was not quantitatively exact: notably, C1 test accuracy was about 12 percentage points below the original results in both regimes, and C2 showed a larger generalization gap and more extreme extrapolation loss. Differences may reflect under-specified architectural and normalization conventions, the use of microbatch gradient accumulation, initialization choices, and the finite training budget. These interpretations require targeted ablations rather than being proven by the present single-pair experiment.

---

## 12. Files to retain as evidence / reproducibility package

Inside the archived `keskar_fig3_f1_c1_c2/` directory:

- `figure3_F1_C1_C2_computed.png` — final three-panel figure (F1, C1, C2).
- `F1_freshly_measured_curve.csv`, `C1_freshly_measured_curve.csv`, `C2_freshly_measured_curve.csv` — exact numerical points used to plot the final curves (all **61 rows** each); these are especially important if replotting in a report.
- `F1_SB_best.pt`, `F1_LB_best.pt`, `C1_SB_best.pt`, `C1_LB_best.pt`, `C2_SB_best.pt`, `C2_LB_best.pt` — trained endpoint **model weights and BN buffers**.
- Corresponding six `*_progress.pt` files — optimizer/RNG/progress states that substantiate training epoch and stop reason.
- `experiment_config.json` — settings, plotting layout, and recovery provenance.
- `F1_original_endpoint_metrics.json`, `C1_original_endpoint_metrics.json` — original-BN endpoint diagnostics useful for describing recalibration sensitivity. **No corresponding C2 original-BN endpoint test evaluation is archived.**
- Retain both the **original training notebook** and the **C2 recovery notebook** alongside this folder: the ZIP contains resulting artifacts, **not the notebook source**.

**Scope of evidence:** We verified that each curve file contains exactly 61 records, checked the six best-checkpoint epochs and their parameter counts, and checked each progress record's stop reason. We did **not** independently rerun the many-hour training or full dataset inference from scratch in this analysis. The data and conclusion refer to the specific saved runs, not statistical replications over different seeds.

---

## 13. Compact source map for citations in the eventual report

| Source within attached paper | What it establishes |
|---|---|
| **Table 1, p. 4** | F1 = fully connected/MNIST; C1 = shallow CNN/CIFAR-10; C2 = deep CNN/CIFAR-10 |
| **§2.2, p. 4** | Mean cross-entropy; SB=256; LB=10% of training data; Adam; five trials; convergence description |
| **Table 2, p. 4** | Published train/test accuracies, means and SDs for SB/LB |
| **§2.2.1, pp. 4–5** | Linear interpolation formula, \(\alpha\in[-1,2]\), random paired endpoints |
| **Figure 3 and caption, p. 6** | Correct figure panels, dual-axis semantics, solid/dashed conventions and endpoint alpha values |
| **Appendix A/Table 5, p. 12** | Training/test example counts, image sizes, classes, “no preprocessing” statement |
| **Appendix B.1, B.3, B.4, p. 12** | F1/C1/C2 architecture descriptions and their unresolved implementation details |
| **Appendix D/Figure 7, p. 13** | Separate curved-path experiment **not** reproduced here |
| **Appendix E, pp. 13 onward** | Data augmentation is a **separate** mitigation experiment, not a basis for assuming augmentation in Figure 3 |

**Primary citation for your report:** Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* International Conference on Learning Representations (ICLR). Cite the relevant PDF sections/tables in addition to this bibliographic entry when presenting paper-derived numbers.
