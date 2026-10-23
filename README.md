# Paper Reproductions — CS 6362 Assignment 2

Reproductions of one figure from each of three ICLR 2017 papers. Each paper has its own
folder, its own notebook, and its own saved results. Everything in this repository was
produced by the notebooks here; no numbers were copied from the papers.

| Folder | Paper | Figure reproduced |
|---|---|---|
| `paper1_zhang/` | Zhang, Bengio, Hardt, Recht, Vinyals — *Understanding deep learning requires rethinking generalization* | Figure 1, panels (a), (b), (c) |
| `paper2_keskar/` | Keskar, Mudigere, Nocedal, Smelyanskiy, Tang — *On large-batch training for deep learning: generalization gap and sharp minima* | Figure 3, panels F1, C1, C2 |
| `paper3_higgins/` | Higgins et al. — *β-VAE: learning basic visual concepts with a constrained variational framework* | Figure 7, panels A and B |

## How to run

Every reproduction is a single self-contained notebook that runs top to bottom on a
Colab GPU runtime. Nothing needs to be installed beyond what Colab ships with, and no
local setup is required.

1. Open the notebook for the paper you want in Google Colab.
2. Set the runtime to GPU (Runtime → Change runtime type → T4 GPU).
3. Run all cells.

The notebooks download their own data. Each one writes its results into a `results/`
directory as it goes and reads them back when it draws the figure, so the plotting cells
can be re-run without retraining. The `results/` directories in this repository hold the
exact runs that the reports describe — if you want to see the figures without spending
the GPU hours, skip the training cells and run the plotting cells against the committed
CSVs.

Runtimes below are for a Colab T4.

---

## `paper1_zhang/` — Figure 1

**What the figure shows.** Three panels from the randomization tests. Panel (a) is the
training loss of a small Inception on CIFAR-10 under five conditions: true labels, random
labels, shuffled pixels, random pixels, and Gaussian inputs. Panel (b) is the time to fit
the training set as a function of the label corruption rate *p*, relative to *p* = 0, for
three architectures. Panel (c) is the test error at convergence against the same *p*.

**Data.** CIFAR-10, downloaded by the notebook. 50,000 training images, 10,000 test.

**Models.** A small Inception, a small AlexNet, and a 1×512 MLP, all following the
descriptions in Appendix A of the paper. No weight decay, no data augmentation, no
dropout — the figure's runs are the unregularized ones.

**Runs.** Five input conditions for panel (a); six corruption rates
(*p* = 0, 0.2, 0.4, 0.6, 0.8, 1.0) × three architectures for panels (b) and (c). Each run
trains for up to 100 epochs and stops early once training accuracy reaches 99.9%.

**Output.** One `run_metrics.csv` per run holding per-epoch training loss, training
accuracy, test loss and test accuracy. The three panels are assembled from these.

**Runtime.** About 6 hours in total.

**Note on panel (b).** The paper does not define "time to overfit". The notebook records
enough per-epoch detail to recompute the panel under several convergence criteria; the
report discusses what changes under each. The committed figure uses training
accuracy ≥ 99%.

---

## `paper2_keskar/` — Figure 3

**What the figure shows.** Cross-entropy and accuracy along the straight line between a
small-batch solution and a large-batch solution, for three of the paper's networks. The
interpolation is w(α) = (1 − α)·w_s + α·w_ℓ, swept over α ∈ [−1, 2].

**Data.** MNIST for F1; CIFAR-10 for C1 and C2. Both downloaded by the notebook.

**Networks.** F1 is the fully connected network of Appendix B.1. C1 and C2 are the
convolutional networks of Appendix B.3 and B.4.

**Batch sizes.** Small batch is 256. Large batch is 10% of the training set, reached by
gradient accumulation so that the optimizer step matches the stated effective batch.

**Batch normalization.** The interpolated weights w(α) are not a point the network was
ever trained at, so its batch-norm running statistics do not apply there. The notebook
recalibrates the running statistics at every α with a forward pass over the training set
before measuring loss and accuracy. Without this the curves are not comparable to the
paper's.

**Output.** One CSV per network holding, for each α, the training and test cross-entropy
and accuracy. The three panels are drawn from these.

**Runtime.** About 3 hours in total.

**Ablation.** `paper2_keskar/` also contains the C1 pooling-stride ablation that the
report discusses. Appendix B.3 leaves the pooling stride unspecified; the two readings
give networks of 209,546 and 799,370 parameters and test accuracies about 10 points
apart. Both are in the results directory.

---

## `paper3_higgins/` — Figure 7

**What the figure shows.** Panels A and B compare a β-VAE at β = 4 against a plain VAE
(β = 1) on dSprites. Each panel holds the latent traversals and the per-latent responses
to the ground-truth generative factors.

**Data.** dSprites, 737,280 binary 64×64 images, downloaded from the DeepMind
repository by the notebook.

**Model.** The encoder and decoder of the paper's Table 1, with 10 latent dimensions.
The reconstruction term is a Bernoulli log-likelihood summed over the 4,096 pixels and
averaged over the batch; the KL term is the Gaussian closed form. The scaling of these
two terms relative to each other is what β multiplies, and the paper does not state it,
so it is written out explicitly in the notebook.

**Runs.** β = 1 and β = 4, seed 7. The notebook is parameterized over β and seed, so
other values can be run by changing one cell.

**Output.** `run_metrics.csv` per run with per-epoch reconstruction loss, KL, and the
per-dimension KL used to identify which latents are active. Traversal images and factor
response plots are written alongside.

**Runtime.** About 2 hours per β.


