# SAR-to-EO Image Translation — Technical Report

**Author:** Krishna Kant
**Assignment:** GalaxEye Space — Satellite AI Research Intern
**Task:** Given a Sentinel-1 SAR (radar) patch, generate the corresponding Sentinel-2 EO (optical RGB) patch.
**Code:** https://github.com/codehunter17/sar2eo

---

## 1. Abstract

This work tackles SAR-to-EO image translation: mapping a single-channel Sentinel-1 SAR (VV) patch to a three-channel Sentinel-2 optical RGB patch. The problem is fundamentally **ill-posed** — SAR encodes surface roughness and dielectric properties, not colour or spectral reflectance — so a given SAR input is consistent with many plausible optical outputs. I adopt a **Pix2Pix conditional GAN** (U-Net generator + 70×70 PatchGAN discriminator, LSGAN + L1 objective), the standard, compute-efficient baseline for paired image-to-image translation that fits comfortably on a single ≤16 GB GPU.

I train on the Kaggle *Sentinel-1&2 Image Pairs (terrain-segregated)* dataset, holding out **two entire terrains** to measure generalisation: `grassland` (validation) and `urban` (test). I run a **controlled ablation** isolating the adversarial term — *L1-only* versus *L1 + adversarial (full Pix2Pix)* — with everything else fixed.

On the test split the full model reaches **LPIPS 0.733, FID 384.6, SSIM 0.139, PSNR 12.71**. The headline finding is that the full GAN and the L1-only ablation are **nearly identical** across all four metrics. The loss curves explain why: the discriminator loss collapses toward zero early in training, so the adversarial gradient vanishes and Pix2Pix effectively degenerates into L1-supervised regression. The report analyses this dynamic, the large gap between validation (`grassland`) and test (`urban`) performance, and the divergence between pixel and perceptual metrics, and proposes concrete next steps (discriminator regularisation / TTUR, perceptual losses, and conditional diffusion) to recover a useful adversarial signal.

---

## 2. Literature Survey

**Image-to-image translation with cGANs.** Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks* (Pix2Pix, CVPR 2017), established the paired-translation template used here: a U-Net generator, a PatchGAN discriminator that classifies overlapping local patches as real/fake, and a combined objective of an adversarial term plus an L1 reconstruction term weighted by λ≈100. The U-Net's skip connections (Ronneberger et al., MICCAI 2015) preserve high-frequency spatial structure that a plain encoder-decoder loses at the bottleneck — important for remote-sensing imagery where edges and field boundaries matter. PatchGAN keeps the discriminator local and parameter-light, encouraging sharp textures without modelling global layout.

**GAN objective stability.** Vanilla GANs minimise a Jensen-Shannon divergence and are prone to vanishing gradients when the discriminator becomes too confident. Mao et al., *Least Squares GAN* (ICCV 2017), replace the cross-entropy with a least-squares (MSE) loss, penalising samples by their distance to the decision boundary and yielding more stable gradients; this is the adversarial loss I use. Even so, a discriminator that wins decisively still starves the generator of useful gradient — a failure mode directly observed in this project and discussed in §4. Remedies in the literature include two-time-scale update rules (Heusel et al., TTUR, 2017), spectral normalisation (Miyato et al., 2018), and one-sided label smoothing (Salimans et al., 2016).

**SAR-to-optical translation specifically.** The SEN1-2 (Schmitt et al., 2018) and SEN12MS (Schmitt et al., 2019) datasets catalysed learning-based SAR→optical work. Representative approaches apply Pix2Pix/CycleGAN-style translation to despeckle and colourise SAR (e.g. Wang et al.; Fuentes Reyes et al., 2019), generally reporting that (a) pixel metrics like PSNR/SSIM are weak predictors of visual quality on this task, and (b) cross-region generalisation is the hardest part — models that memorise one biome transfer poorly to another. More recent work explores conditional diffusion models for SAR→EO, trading compute for sample quality and better multi-modal coverage of the output distribution. These observations motivate two of my design choices: ranking on perceptual metrics (LPIPS/FID), and a **terrain-disjoint** train/val/test split rather than a random one.

**Perceptual evaluation.** Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS, CVPR 2018), showed that distances in deep feature space track human perceptual similarity far better than pixel error. Together with FID (Fréchet distance between Inception feature statistics of real vs generated sets), LPIPS is the primary ranking signal here; SSIM/PSNR are reported but, as discussed in §4, reward conservative mean-seeking predictions that look nothing like real optical imagery.

**Gap addressed.** Within a free-tier compute budget I provide a clean, reproducible Pix2Pix baseline with a *controlled* adversarial-vs-L1 ablation on a terrain-disjoint split, and — rather than reporting only the headline metric — diagnose *why* the adversarial term failed to help (discriminator collapse) and lay out the specific stabilisation strategies that would address it.

---

## 3. Methodology

### 3.1 Dataset and splits

I use the **Kaggle *Sentinel-1&2 Image Pairs, segregated by terrain*** dataset (requiemonk) — pre-paired, co-registered Sentinel-1 SAR and Sentinel-2 optical patches at 256×256, grouped into terrain folders. It is the smallest of the three permitted sources and requires no institutional access, appropriate for a free-tier compute budget. No other external remote-sensing data was used.

**Split strategy (terrain-disjoint).** Because adjacent satellite patches are frequently near-duplicates, a naïve random split leaks information between train and evaluation and inflates scores. I therefore split on **terrain type**: `grassland` is held out entirely as the **validation** split and `urban` as the **test** split; all remaining terrains are used only for training. This makes both evaluation splits genuinely out-of-distribution with respect to land cover — a deliberately harder and more honest measure of generalisation, consistent with GalaxEye's own held-out-scene evaluation.

### 3.2 Preprocessing and normalisation

SAR and optical modalities differ in dynamic range and statistics, so each is normalised separately. SAR VV input is single-channel, already dB-scaled and min-max normalised to [0, 255] (matching the inference I/O contract); it is divided by 255 to [0, 1] then normalised with mean = std = 0.5 to **[−1, 1]**. Optical RGB is normalised the same way (per-channel mean = std = 0.5), so the generator's `tanh` output in [−1, 1] maps directly back to valid pixels. Training-time augmentation is a **horizontal flip only**; vertical flips and random crops are disabled because SAR geometry is orientation-sensitive and patches are already 256×256.

### 3.3 Architecture

**Generator — U-Net**, depth `num_downs = 8` (bottleneck at 1×1 for a 256² input), base width `ngf = 64`, batch normalisation, dropout in the deeper layers for regularisation, and a `tanh` output. Skip connections concatenate each encoder feature map with the matching decoder stage, preserving fine spatial detail.

**Discriminator — PatchGAN**, base width `ndf = 64`, `n_layers = 3`, giving a **70×70** receptive field; it classifies overlapping local patches rather than the whole image, favouring high-frequency texture realism with few parameters.

### 3.4 Loss

The generator objective combines the **LSGAN** adversarial loss (MSE form, more stable than BCE) with an **L1 reconstruction** term: `L_G = L_adv + λ_L1 · L_1`, with **λ_L1 = 100** (Isola et al.). The discriminator minimises the LSGAN real/fake loss. (One implementation detail mattered: the LSGAN target tensors must be created on the prediction's device; an earlier CPU/GPU mismatch was fixed with `torch.full_like`, committed to the repo.)

### 3.5 Controlled ablation

The single variable isolated is the **adversarial term**:

- **Model A — L1-only:** generator trained with the L1 term alone, no discriminator (`mode: l1_only`).
- **Model B — full Pix2Pix:** L1 + adversarial (`mode: full_gan`).

Everything else — architecture, data, split, optimiser, schedule, seed, epochs — is held identical, so any difference is attributable to the adversarial loss.

### 3.6 Training configuration

| Setting | Value |
|---|---|
| Optimiser | Adam (β₁ = 0.5, β₂ = 0.999) |
| Learning rate | 2×10⁻⁴ (constant first half, then linear decay to 0) |
| Batch size | 4 (fits ≤16 GB VRAM at 256²) |
| Epochs | 40 per model |
| λ_L1 | 100 |
| Image size | 256×256 |
| Random seed | 42 |
| Augmentation | horizontal flip |
| Hardware | Kaggle free tier, NVIDIA Tesla T4 (×2), 16 GB each |

Per-epoch training and validation L1 loss, generator loss and discriminator loss are logged to `loss_log.csv`; the loss-curve plot is saved to `loss_curve.png`. The full pipeline (data prep → Model B → Model A → evaluation of both on val + test) runs end-to-end headless on Kaggle.

> **Compute-budget note.** The configured `FULL_EPOCHS` was reduced from 200 to **40**. A full-dataset run of two models for 200 epochs plus evaluation exceeded Kaggle's 12-hour batch limit (an earlier run was killed at the wall). 40 epochs is sufficient for Pix2Pix to converge here (the curves flatten well before epoch 40) and guarantees the entire pipeline — both models *and* evaluation — completes within budget. This is a deliberate, documented trade-off, in line with the brief's guidance that a well-reasoned run within compute constraints is preferred over chasing marginal gains.

---

## 4. Results

### 4.1 Quantitative metrics

All metrics are computed by `eval.py` on the held-out terrains. **Primary (perceptual): LPIPS ↓, FID ↓. Secondary (pixel): SSIM ↑, PSNR ↑.**

**Validation split — terrain = `grassland`**

| Model | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|
| Model A — L1-only (ablation) | 0.7896 | 285.22 | **0.2531** | **14.10** |
| Model B — Pix2Pix (full GAN) | 0.8006 | **280.63** | 0.2483 | 13.86 |

**Test split — terrain = `urban`**

| Model | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|
| Model A — L1-only (ablation) | 0.7390 | 386.30 | **0.1402** | 12.66 |
| Model B — Pix2Pix (full GAN) | **0.7327** | **384.55** | 0.1393 | **12.71** |

### 4.2 Ablation analysis — the adversarial term did not help

The two models are **statistically indistinguishable**: differences are within ~0.01 LPIPS, ~5 FID, <0.01 SSIM and ~0.2 dB PSNR, and the winner flips between metrics and splits. On `grassland` the L1-only model is marginally better on LPIPS/SSIM/PSNR while the GAN edges FID; on `urban` the GAN is marginally better on LPIPS/FID. There is **no meaningful advantage to the adversarial term** in this run.

The loss curves (`loss_curve.png`) explain it. The **discriminator loss falls to ≈0 early in training and stays there** (logged values reach `D ≈ 0.0000`), meaning the PatchGAN classifies real vs generated patches almost perfectly. When the discriminator wins this decisively, the generator's adversarial gradient effectively vanishes (the LSGAN term saturates), so the generator is driven almost entirely by the L1 term. In other words, **Model B collapsed into Model A.** This is a classic GAN failure mode (discriminator over-powering the generator), and the ablation — far from being a null result — *cleanly demonstrates* it: with the adversarial signal dead, the two configurations must converge to the same place, which is exactly what the numbers show.

*(Figure: generator and discriminator loss curves for both models — see `submission_assets/`. The G curve decreases and flattens; the D curve drops to ≈0 within the first few epochs and remains there, the signature of the collapse.)*

### 4.3 Pixel-vs-perceptual gap

The metrics disagree by construction. SSIM/PSNR are modest-but-not-catastrophic (SSIM ≈ 0.25 on grassland) while LPIPS/FID are poor (LPIPS ≈ 0.8, FID ≈ 281). This is the expected signature of an L1-dominated model on an ill-posed task: minimising L1 over many plausible optical outputs yields the **per-pixel mean** — a smooth, desaturated, low-frequency image. Such an image scores *reasonably* on pixel metrics (it is close on average and structurally aligned) but *badly* on perceptual metrics (it lacks the texture and colour statistics of real optical imagery, which is exactly what LPIPS and FID measure). This is precisely why the brief ranks on perceptual metrics, and why the vanished adversarial term — the only component that pushes toward realistic high-frequency detail — is the key weakness of this run.

### 4.4 Error profile — where the model fails

**Validation (`grassland`) vs test (`urban`).** Performance degrades sharply from grassland to urban: FID rises from ~281 to ~385 and SSIM falls from ~0.25 to ~0.14. Grassland is comparatively low-frequency and homogeneous, so a mean-seeking generator can approximate it; **urban scenes are high-texture and building-dense**, with strong SAR backscatter from layover/double-bounce that has no simple optical analogue, and they are out-of-distribution relative to the training terrains. The model has no mechanism to hallucinate plausible street/roof structure, so it falls back to blurred grey-green fields — visually wrong, and heavily penalised by FID. This terrain gap is the clearest evidence that the model generalises by **smoothing**, not by learning terrain-specific structure.

**Typical failure modes (qualitative triplets, `submission_assets/`):** (1) loss of high-frequency texture — fields and forests become uniform colour blobs; (2) colour desaturation toward a mean grey-green; (3) on urban tiles, complete loss of building/road geometry; (4) water and shadow regions (very low SAR return) collapse to flat dark patches. Successes are mostly large homogeneous land-cover (open grassland / bare soil) where the mean prediction happens to be close to the truth.

*(Figures: ≥5 SAR → generated → ground-truth triplets covering both success (grassland / bare soil) and failure (urban, water) cases — see `submission_assets/`.)*

### 4.5 Loss-curve interpretation

The generator loss decreases monotonically and flattens before epoch 40, indicating convergence and justifying the 40-epoch budget. Train and validation L1 track each other closely with no widening gap, so there is **no significant overfitting** — unsurprising given the model is under-fitting the *perceptual* target rather than over-fitting the pixels. The discriminator loss collapses to ≈0 within the first few epochs (see §4.2). For an adversarial method the G/D curves are shown separately rather than as a single loss.

---

## 5. Future Work

If I were continuing this as a first-month deliverable, in priority order:

1. **Fix the discriminator collapse (highest impact).** The adversarial signal is the missing ingredient for perceptual quality. Concrete levers: a **two-time-scale update rule** (lower the D learning rate, e.g. D 1×10⁻⁴ / G 2×10⁻⁴), **spectral normalisation** on the discriminator, **one-sided label smoothing**, and/or reducing PatchGAN capacity or updating D less frequently. The goal is to keep D in the informative regime (D-loss ≈ 0.3–0.6) so the generator keeps receiving gradient.

2. **Add a perceptual / feature-matching loss.** A VGG-based perceptual term or discriminator feature-matching (as in pix2pixHD) directly optimises the deep-feature statistics that LPIPS/FID measure, and is far more robust than the adversarial term alone.

3. **Train longer on more data, with LR decay.** With the collapse fixed, scale from 40 to 100–200 epochs and incorporate additional permitted data (SEN1-2) for terrain diversity, which §4.4 shows is the main generalisation bottleneck.

4. **Stronger / multi-modal models.** A **conditional diffusion model** (e.g. an SAR-conditioned latent diffusion / Palette-style approach) better covers the multi-modal output distribution of this ill-posed problem and is the current state of the art for SAR→optical; a compute-heavier but higher-ceiling direction.

5. **Task-aware evaluation.** Beyond LPIPS/FID, evaluate on a downstream proxy (e.g. land-cover segmentation agreement between generated and real optical) to measure whether outputs are *useful*, not just realistic.

---

## 6. Conclusion

I built a clean, reproducible Pix2Pix baseline for SAR→EO translation and evaluated it honestly on a **terrain-disjoint** split with a controlled L1-vs-adversarial ablation. The model produces structurally-aligned but perceptually weak optical predictions (test LPIPS 0.733, FID 384.6, SSIM 0.139, PSNR 12.71), and the ablation shows the adversarial term provided no benefit — because the discriminator collapsed and Pix2Pix degenerated into L1 regression. The pixel-vs-perceptual gap and the grassland-vs-urban gap together show the model generalises by smoothing rather than by synthesising realistic terrain-specific texture.

**Honest limitations:** the headline metrics are modest; the adversarial component — the part that should drive realism — did not function; and 40 epochs on a single terrain-segregated dataset is a constrained setup. None of this is hidden by the numbers, and the most valuable output of the project is a precise diagnosis of *why* it underperformed and a concrete, prioritised plan (discriminator stabilisation, perceptual loss, more data/epochs, diffusion) to fix it. Within a free-tier budget, the work demonstrates correct end-to-end methodology, careful evaluation design, and the research judgement to interpret a negative ablation result.

---

## 7. Time and Resource Log

**Machine (training):** Kaggle free tier, NVIDIA Tesla **T4 (×2)**, 16 GB VRAM each, headless "Save & Run All" batch execution.

**Per-epoch / total training time:** ~2.5–6 min per epoch (faster once the dataset is OS-cached); the **complete pipeline** — data preparation → Model B (40 epochs) → Model A (40 epochs) → evaluation of both on validation + test (inference + LPIPS/FID/SSIM/PSNR) — ran end-to-end in **~6.2 hours** (22,307 s wall-clock) in a single batch session.

**Approximate time by activity:**

| Activity | Time |
|---|---|
| Data exploration & understanding the SAR/optical modalities | ~2 h |
| Literature reading (Pix2Pix, LSGAN, LPIPS, SAR→optical) | ~3 h |
| Implementation (dataloaders, U-Net, PatchGAN, losses, train/eval/infer) | ~10 h |
| Debugging & getting the pipeline to run headless on Kaggle | ~4 h |
| Training (compute, unattended) | ~6.2 h wall-clock |
| Evaluation & results analysis | ~2 h |
| Report writing | ~3 h |

**Constraints and how they shaped decisions:** the dominant constraint was the **free-tier GPU budget** (Kaggle's 12-hour batch limit and weekly GPU quota). An initial 200-epoch full-dataset run of two models exceeded the 12-hour wall and was killed before evaluation, which directly motivated reducing to **40 epochs** to guarantee the whole pipeline — including the ablation and evaluation — finishes within budget. The same constraint motivated choosing the smallest permitted dataset and a Pix2Pix-class model (rather than diffusion). These trade-offs favour a *complete, reproducible* result over a longer run that risks producing nothing.

---

## References

**Datasets**

- requiemonk (2022). *Sentinel-1&2 Image Pairs Segregated by Terrain.* Kaggle. https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain
- Schmitt, M., et al. (2018). *SEN1-2.* TU Munich. https://mediatum.ub.tum.de/1436631
- Schmitt, M., et al. (2019). *SEN12MS.* https://mediatum.ub.tum.de/1474000

**Papers**

- Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR. https://arxiv.org/abs/1611.07004
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. https://arxiv.org/abs/1505.04597
- Mao, X., et al. (2017). *Least Squares Generative Adversarial Networks.* ICCV. https://arxiv.org/abs/1611.04076
- Zhang, R., et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).* CVPR. https://arxiv.org/abs/1801.03924
- Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID/TTUR).* NeurIPS. https://arxiv.org/abs/1706.08500
