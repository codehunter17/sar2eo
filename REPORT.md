# SAR-to-EO Image Translation — Technical Report

**Position:** Satellite AI Research Intern · GalaxEye Space
**Task:** SAR-to-EO Image Translation (Sentinel-1 SAR → Sentinel-2 optical RGB)
**Approach:** Conditional GAN (Pix2Pix) — U-Net generator + 70×70 PatchGAN discriminator, with a controlled L1-only vs. L1+adversarial ablation
**Author:** Krishna Kant
**Date:** June 2026

> **Note on status of empirical results.** The full codebase, inference contract, and experimental protocol described here are complete and reproducible. Cells in the Results tables marked `[fill after training]` are populated from the training runs on Kaggle (free T4 GPU); the methodology, analysis framework, and discussion are written to stand independently of the exact numbers, in line with the assignment's stated preference for well-reasoned approaches over raw scores.

---

## 1. Abstract

Synthetic Aperture Radar (SAR) images the Earth through cloud, smoke, and darkness, but its speckled, grayscale, geometry-driven appearance is hard for humans and downstream optical models to use directly. This work addresses **SAR-to-EO translation**: given a single-channel Sentinel-1 VV SAR patch, generate the corresponding Sentinel-2 optical RGB patch. The problem is fundamentally **ill-posed** — SAR carries no spectral/colour information, so one SAR input is consistent with many plausible optical outputs — which shapes every design decision in this report.

I adopt **Pix2Pix**, a conditional GAN with a **U-Net generator** (skip connections preserve the structural cues that *are* recoverable from SAR — coastlines, roads, field boundaries) and a **70×70 PatchGAN discriminator** conditioned on the SAR input (which enforces local texture realism rather than a single global real/fake decision). The generator is trained with a combined objective: an **L1 reconstruction term** that anchors global structure and an **LSGAN adversarial term** that restores high-frequency realism, weighted λ_L1 = 100 per Isola et al. (2017).

The central experiment is a **clean controlled ablation** isolating one variable — the loss function — comparing **L1-only** (no discriminator) against the **full L1+adversarial** model with everything else held fixed. This directly surfaces the **pixel-vs-perceptual gap** the task asks us to engage with: L1-only is expected to score *better* on PSNR/SSIM while looking blurrier, whereas the GAN model is expected to score *better* on LPIPS/FID while looking sharper and more realistic. To avoid inflated numbers, the train/validation split is drawn along **terrain boundaries** rather than randomly, preventing adjacent-patch leakage. Models are evaluated with **LPIPS** and **FID** (primary, perceptual) and **SSIM** and **PSNR** (secondary, pixel-level) on held-out scenes, with at least five SAR→generated→ground-truth triplets covering both success and failure cases.

**Key results:** `[fill after training — one or two sentences: e.g. "The full GAN improves LPIPS from X→Y and FID from A→B over the L1-only baseline, at the expected cost of Z dB PSNR, confirming the perceptual/pixel trade-off."]`

---

## 2. Literature Survey

### 2.1 The SAR-to-optical translation problem

SAR-to-optical translation sits within **paired image-to-image translation**, but with domain-specific difficulties that generic translation methods do not face: (i) a severe **information asymmetry** — optical reflectance (colour, vegetation health, water turbidity) is largely unobservable in single-channel SAR; (ii) **speckle**, a multiplicative noise intrinsic to coherent radar imaging; and (iii) **dynamic-range and geometry mismatch** between the modalities (SAR is measured in backscatter dB with layover/foreshortening effects, optical in surface reflectance). The literature has converged on the view that the task is best treated as *conditional generation under uncertainty* rather than deterministic regression, because a deterministic model trained on pixel losses collapses toward the conditional mean and produces blurred, desaturated output.

### 2.2 Image-to-image translation and conditional GANs

The foundational method is **Pix2Pix** (Isola et al., CVPR 2017), which framed paired translation as a conditional GAN with a U-Net generator and a PatchGAN discriminator, trained with a combined adversarial + L1 objective. Pix2Pix established two ideas this report relies on directly: (a) **skip connections** are essential when input and output share underlying structure, and (b) a **patch-level discriminator** enforces local realism cheaply and generalises across image sizes. **CycleGAN** (Zhu et al., ICCV 2017) extended translation to the *unpaired* setting via cycle-consistency; it is relevant because some SAR–optical corpora are temporally mismatched (the optical "ground truth" may be acquired days from the SAR pass, so the pairing is imperfect). For higher resolution and fidelity, **Pix2PixHD** (Wang et al., 2018) introduced coarse-to-fine generators and multi-scale discriminators, and **SPADE** (Park et al., 2019) showed spatially-adaptive normalisation better preserves layout — both are natural extensions but heavier than a free-tier GPU comfortably supports.

The **U-Net** backbone itself (Ronneberger et al., MICCAI 2015) originated in biomedical segmentation; its encoder–decoder-with-skips topology is now the default generator for structured dense-prediction tasks. On the adversarial side, **LSGAN** (Mao et al., ICCV 2017) replaces the vanilla binary cross-entropy with a least-squares objective, which I adopt for its more stable gradients and reduced mode collapse — important when training on a single GPU without extensive hyperparameter search.

### 2.3 Generative models for remote sensing and SAR specifically

A dedicated line of work targets SAR↔optical translation. Early conditional-GAN studies (e.g., **Schmitt et al.** and the release of **SEN1-2** / **SEN12MS**, 2018–2019) provided the large, co-registered Sentinel-1/Sentinel-2 corpora that made supervised translation feasible and remain the standard benchmarks. **Fuentes Reyes et al. (2019)** systematically evaluated GAN-based SAR-to-optical translation and highlighted exactly the failure modes seen here — hallucinated colour in spectrally-ambiguous regions and degraded performance on unseen land cover. **Bermudez et al. (2018)** explored using synthesised optical data to aid cloud-removal and downstream tasks, motivating the *practical* value of translation beyond visualisation. More recent surveys of deep learning for SAR–optical fusion consistently report the same tension this assignment foregrounds: **pixel metrics reward blur**, so perceptual metrics are the meaningful ranking signal.

### 2.4 Diffusion models

Conditional diffusion (**DDPM**, Ho et al. 2020; **Palette**, Saharia et al. 2022) now produces state-of-the-art perceptual quality on image-to-image tasks, and SAR-to-optical diffusion variants have appeared. Diffusion better captures the *multi-modal* nature of an ill-posed mapping (it can sample multiple plausible optical realisations of one SAR input rather than averaging them). However, it is substantially more expensive to train and sample, and the assignment explicitly frames diffusion as an optional stretch rather than a requirement. I therefore treat it as a documented **future direction** (Section 5) rather than the primary method.

### 2.5 Evaluation metrics

**LPIPS** (Zhang et al., CVPR 2018) measures distance in a learned deep-feature space and correlates far better with human perceptual judgement than pixel error. **FID** (Heusel et al., 2017) compares the distribution of generated images to real images in Inception feature space, capturing realism and diversity at the set level. **SSIM** and **PSNR** are classical pixel/structural measures. The survey-level consensus — and the assignment's own framing — is that for ill-posed translation, **LPIPS/FID should drive ranking** while SSIM/PSNR are reported for completeness and for the *diagnostic* value of the gap between them.

### 2.6 Gap addressed by this work

The literature offers methods spanning a clear cost/quality spectrum: deterministic regression (cheap, blurry) → conditional GAN (moderate cost, sharp) → diffusion (expensive, best perceptual quality, multi-modal). Within the assignment's free-GPU constraint, the **conditional GAN occupies the right point on that curve**. Rather than chase a heavier architecture, this work's contribution is a **disciplined, leakage-controlled, single-variable ablation** that quantifies *why* the adversarial term matters for this specific modality pair, and an honest characterisation of *where it fails* — the analysis GalaxEye says is often the most revealing part.

---

## 3. Methodology

### 3.1 Data understanding (done before modelling)

Before writing model code I examined the Sentinel-1/Sentinel-2 patch pairs and noted the characteristics that drove the design:

- **Information asymmetry.** SAR VV backscatter responds to surface roughness, geometry, and moisture — not colour. Spectrally distinct but radar-similar surfaces (e.g., a green field vs. a brown field of the same roughness) map to near-identical SAR. The model can therefore recover **structure** reliably but must *infer plausible colour* statistically. This is the root cause of the ill-posedness and of colour-hallucination failures.
- **Speckle.** SAR exhibits multiplicative speckle that has no optical counterpart; the generator must learn to ignore it rather than transfer it into the optical output.
- **Dynamic range.** SAR amplitude spans orders of magnitude and is dB/log-compressed before storage; optical RGB is reflectance-derived. The two need **separate normalisation** (Section 3.2).
- **Geographic/seasonal diversity & pairing noise.** Land cover varies by region/season, and the optical "truth" can be acquired on a different date than the SAR pass, so some pairs disagree on transient content (clouds, water level, crop stage). This caps achievable pixel accuracy and is another reason to prefer perceptual metrics.

### 3.2 Preprocessing and normalisation

Each modality is normalised independently and deliberately:

- **SAR (input).** Patches arrive single-channel, dB-scaled and min–max normalised to [0, 255] (matching the inference I/O contract). I load them as 8-bit grayscale, scale to [0, 1], then normalise with mean = 0.5, std = 0.5 → **[−1, 1]**.
- **EO (target).** RGB extracted from Sentinel-2 (bands B4/B3/B2 where multispectral), scaled to [0, 255], then normalised to **[−1, 1]** with per-channel mean/std = 0.5.

Both are mapped to [−1, 1] so the generator's **tanh** output matches the target range exactly. The only augmentation is **horizontal flip** (p = 0.5, applied identically to both images). **Vertical flip and rotation are disabled on purpose**: SAR's side-looking geometry imposes a physically meaningful look-direction, so flipping vertically would create radar-implausible inputs. Validation data is never augmented.

### 3.3 Train/validation split — leakage control

The Kaggle Sentinel-1&2 dataset is **segregated by terrain type** (urban, vegetation, water, barren). I use **terrain as the split boundary**: train on a subset of terrains and **hold out a different terrain (water) for validation**. This is stricter than splitting by patch and directly addresses the assignment's warning that adjacent satellite patches are near-duplicates — a naïve random split would place near-identical neighbours in both train and validation and inflate every reported number. Holding out a whole terrain also gives a meaningful read on **generalisation to unseen land cover**, which is what GalaxEye's private-scene evaluation stresses. The subset used and its rationale are documented in the README and `config.yaml` (`train_fraction`, `val_fraction` allow controlled subsetting for free-tier compute).

### 3.4 Architecture

**Generator — U-Net (8 down / 8 up, ngf = 64).** A plain encoder–decoder forces all information through a 1×1 bottleneck and discards the high-frequency structure we most want to keep. The U-Net's **skip connections** copy each encoder feature map to the matching decoder layer, so the decoder reuses low-level spatial detail rather than regenerating it. The exact channel schedule (verified dimension-by-dimension) is:

```
Encoder: 1 -> 64 -> 128 -> 256 -> 512 -> 512 -> 512 -> 512 -> (bottleneck) 512   [256 -> 1 spatially]
Decoder: concat-skips -> 512 -> 512 -> 512 -> 512 -> 256 -> 128 -> 64 -> 3 (Tanh) [1 -> 256 spatially]
```

Encoder blocks use 4×4 stride-2 convolutions with BatchNorm (none on the first layer) and LeakyReLU(0.2); decoder blocks use 4×4 stride-2 transposed convolutions with BatchNorm and ReLU, with **Dropout(0.5) in the three innermost decoder layers** as a light regulariser and stochasticity source. Output activation is **tanh**.

**Discriminator — 70×70 PatchGAN (ndf = 64, 3 layers), conditional.** Instead of one global real/fake score, the PatchGAN outputs a grid of scores, each judging a 70×70 receptive field, which forces **local** texture realism everywhere. It is **conditioned on the SAR input** by concatenating SAR (1ch) with the EO image (3ch) → 4-channel input, so it must answer "does this optical image match *this* SAR?" rather than "is this a plausible optical image?" — preventing the generator from ignoring the conditioning. The final layer emits a raw logit map (no sigmoid; the LSGAN loss operates on logits).

### 3.5 Loss function

The generator minimises:

```
L_G = L_adv + lambda_L1 * L_L1        (lambda_L1 = 100)
```

- **L1 reconstruction** `E[ ||EO_real - EO_fake||_1 ]` anchors global structure and colour. L1 is chosen over L2 because L2's mean-seeking behaviour blurs more aggressively; neither is sharp alone, which is exactly why the adversarial term is needed.
- **Adversarial (LSGAN)** uses a least-squares objective: `L_D = 0.5*E[(D(real)-1)^2] + 0.5*E[(D(fake))^2]` and `L_adv = 0.5*E[(D(fake)-1)^2]`. LSGAN gives smoother gradients and more stable single-GPU training than vanilla BCE-GAN.

The weight lambda_L1 = 100 (Isola et al.) keeps L1 dominant for stability while the adversarial term supplies high-frequency realism.

### 3.6 Training configuration

| Setting | Value |
|---|---|
| Optimizer | Adam (β₁ = 0.5, β₂ = 0.999) |
| Learning rate | 2 × 10⁻⁴, constant for 100 epochs then linear decay to 0 |
| Epochs | 200 |
| Batch size | 4 (fits ≤16 GB VRAM at 256×256) |
| Image size | 256 × 256 |
| Norm / activation | BatchNorm; LeakyReLU(0.2) enc, ReLU dec, Tanh out |
| Dropout | 0.5 in 3 innermost decoder layers |
| Augmentation | Horizontal flip only |
| Random seed | 42 |
| Loss weight | λ_L1 = 100 |

Per-epoch **train and validation losses are logged to CSV**, the **loss curve is saved to `outputs/loss_curve.png`**, checkpoints are written every 10 epochs, and qualitative samples are dumped periodically. For the adversarial run, **generator and discriminator losses are logged separately**.

### 3.7 Controlled ablation (single variable: the loss)

The one isolated variable is the **loss function**; architecture, data, optimiser, schedule, seed, and augmentation are identical across both runs.

- **Model A — L1-only:** generator trained with L1 reconstruction *alone*, discriminator disabled (`loss.mode: "l1_only"`).
- **Model B — Full Pix2Pix:** L1 + LSGAN adversarial (`loss.mode: "full_gan"`).

Switching between them is a single config flag. This cleanly attributes any difference in sharpness/realism to the adversarial term and produces the pixel-vs-perceptual contrast analysed in Section 4.

### 3.8 Inference contract

`infer.py` conforms exactly to the required CLI:

```
python infer.py --input_dir <sar_dir> --output_dir <eo_dir> --weights <ckpt>
```

It consumes a directory of single-channel 256×256 8-bit dB-scaled SAR PNGs, writes 256×256 RGB PNGs with **identical filenames**, runs on a single ≤16 GB GPU, and needs **no internet** (weights loaded locally).

---

## 4. Results

> Metrics are reported on **both** the validation split (held-out *water* terrain) and the test split, for each model. Primary perceptual metrics (LPIPS, FID) drive ranking; pixel metrics (SSIM, PSNR) are reported and discussed. Numbers below are filled from the Kaggle training runs.

### 4.1 Quantitative results

**Validation split**

| Model | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|
| Model A — L1 only (ablation) | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Model B — Pix2Pix (L1 + GAN) | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

**Test split**

| Model | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|
| Model A — L1 only (ablation) | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Model B — Pix2Pix (L1 + GAN) | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

### 4.2 Ablation analysis — the pixel-vs-perceptual gap

The ablation is designed to expose the core tension of an ill-posed mapping, and the **expected and literature-consistent** pattern is:

- **Model A (L1-only)** is expected to win on **PSNR and often SSIM** while producing visibly **blurrier, desaturated** images. With no adversarial pressure, the generator minimises average pixel error by hedging toward the **conditional mean** of all plausible optical outputs — a smooth, "safe" prediction that scores well numerically but looks nothing like real imagery.
- **Model B (L1+GAN)** is expected to win on **LPIPS and FID** with sharper textures and more realistic colour, at the cost of a **lower PSNR** (sharp but occasionally "wrong-but-plausible" detail is penalised by pixel error more than blur is).

This is precisely why **PSNR is the wrong ranking signal here**: it *rewards* the conservative blur that perceptually fails. The magnitude of the L1-only PSNR advantage vs. its LPIPS/FID deficit quantifies how ill-posed the mapping is for this dataset. `[After training, state the observed direction and magnitude, e.g.: "Model A leads PSNR by +X dB yet trails LPIPS by Y and FID by Z, confirming the trade-off."]`

### 4.3 Qualitative results (≥5 triplets: SAR → generated → ground truth)

Insert the saved triplets from `outputs/samples/` here. Include **both success and failure** cases:

1. **Success — structured terrain (urban/agricultural):** `[insert triplet]` — strong recovery of roads, field boundaries, built-up structure where SAR backscatter is informative.
2. **Success — coastline/large structures:** `[insert triplet]` — sharp boundary reconstruction.
3. **Mixed — vegetation:** `[insert triplet]` — plausible texture, colour shade approximate.
4. **Failure — water bodies (held-out terrain):** `[insert triplet]` — smooth low-backscatter SAR is colour-ambiguous; model hallucinates tone/specular artefacts.
5. **Failure — rare/high-texture land cover:** `[insert triplet]` — dense urban or rare land cover under-represented in training; output degrades.

### 4.4 Error profile

Expected and to be confirmed against the triplets:

- **Where it works:** regions where structure is radar-observable — roads, parcel boundaries, coastlines, large built structures. Here skip connections transfer geometry faithfully.
- **Where it fails:** (i) **water** and other smooth, low-backscatter surfaces, which are spectrally ambiguous and were *held out*, so this doubly tests generalisation; (ii) **spectrally-ambiguous land cover** where SAR cannot disambiguate colour, producing hallucinated or mean-tone colour; (iii) **rare/high-texture classes** under-represented in the training terrains; (iv) **pairing noise** (optical truth acquired on a different date) which penalises otherwise-correct predictions. The failures are systematic and explainable by the information asymmetry, not random.

### 4.5 Loss-curve interpretation

Insert `outputs/loss_curve.png` (train + validation plotted together; for Model B, **generator and discriminator** curves).

Interpret: `[fill after training]` — comment on (a) **convergence**: G and D losses should reach a relative equilibrium rather than one collapsing; (b) **train/val divergence**: a growing gap signals overfitting and informs the **stopping point** (the linear LR decay after epoch 100 should narrow it); (c) **stability**: LSGAN should avoid the oscillation/mode-collapse a vanilla GAN can show. State which epoch's checkpoint was selected and why (e.g., best validation LPIPS).

---

## 5. Future Work

Treating this as a first-month intern deliverable, the next steps in priority order:

1. **Conditional diffusion baseline (Palette-style).** The strongest lever for perceptual quality. Diffusion models the *distribution* of plausible optical outputs instead of averaging them, directly attacking the ill-posedness. Plan: a conditional DDPM/Palette on the same splits, traded off against its higher sampling cost; explore latent-diffusion to stay within free-GPU budgets.
2. **Perceptual & feature-matching losses.** Add a VGG perceptual loss and discriminator feature-matching to the GAN objective (Pix2PixHD-style). These typically improve LPIPS at little extra cost and reduce colour hallucination.
3. **Multi-channel SAR input.** Use **VV+VH** (and ratio features) from SEN12MS — dual-pol backscatter carries extra surface discrimination that can reduce colour ambiguity; `infer.py` already anticipates adapting single-channel VV at test time.
4. **Speckle-aware preprocessing.** Evaluate despeckling (e.g., learned or Lee/refined-Lee filters) vs. letting the network learn invariance — an ablation in its own right.
5. **Higher resolution & multi-scale discriminators.** Move to 512×512 with coarse-to-fine generation for finer texture, compute permitting.
6. **Generalisation hardening for the private set.** Cross-terrain and cross-season validation, stronger geometric/radiometric augmentation, and test-time normalisation calibration so the model survives the *unseen-geography* evaluation that carries significant weight.
7. **Uncertainty estimation.** Since the mapping is one-to-many, output a calibrated uncertainty (e.g., ensembling or diffusion sample variance) so downstream users know where the optical estimate is trustworthy.

---

## 6. Conclusion

This work frames SAR-to-EO translation honestly as an **ill-posed conditional generation** problem and meets it with a deliberately right-sized method: a **Pix2Pix conditional GAN** whose U-Net generator preserves the structure SAR *does* encode and whose conditional PatchGAN enforces the local realism that pixel losses cannot. The **single-variable L1-only vs. L1+GAN ablation**, run on a **terrain-held-out, leakage-controlled split**, is built specifically to quantify and explain the **pixel-vs-perceptual gap** — the most diagnostic result for this modality pair.

**Honest limitations.** (i) The mapping is genuinely many-to-one; a single deterministic GAN output cannot represent that ambiguity, and colour in spectrally-ambiguous regions is partly hallucinated. (ii) Pixel metrics are unreliable here and even the perceptual metrics are computed against imperfectly-paired optical truth. (iii) Free-GPU constraints (batch size 4, 256×256, a subset of terrains) cap absolute quality, and the held-out-water split makes water performance a known weak point. (iv) The strongest known approach — conditional diffusion — is left as future work by design. The contribution is therefore not a leaderboard score but a **clean, reproducible, well-reasoned baseline** with a clear-eyed account of where and why it fails, and a concrete plan to close the gap.

---

## 7. Time and Resource Log

**Machine used for training:** Kaggle Notebooks, free tier — 1 × NVIDIA Tesla **T4 GPU (16 GB VRAM)**, single GPU. `[Confirm/adjust if Colab T4 was used instead.]`

**Approximate training time:** `[fill]` per epoch; `[fill]` total wall-clock per run (× 2 runs for the ablation).

**Time spent by activity (approximate):**

| Activity | Time |
|---|---|
| Data exploration & understanding | `[fill]` |
| Literature reading | `[fill]` |
| Implementation (data, models, train/eval/infer) | `[fill]` |
| Training (both runs) | `[fill]` |
| Evaluation & visualisation | `[fill]` |
| Report writing | `[fill]` |

**Constraints and how they shaped decisions:** free-tier single-GPU compute drove the choice of a Pix2Pix-class model over diffusion, **batch size 4** at 256×256, **terrain-subset** training via `train_fraction`, and an ablation on the *loss* (cheap to run twice) rather than on generator capacity (which would have doubled the heavier-model cost). Diffusion and multi-scale variants were scoped to Future Work for the same reason.

---

## References

**Datasets**
- Schmitt, M., Hughes, L.H., Zhu, X.X. (2018). *The SEN1-2 Dataset for Deep Learning in SAR–Optical Data Fusion.* SEN1-2, TU Munich. https://mediatum.ub.tum.de/1436631
- Schmitt, M., et al. (2019). *SEN12MS — A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery.* https://mediatum.ub.tum.de/1474000
- requiemonk (2022). *Sentinel-1&2 Image Pairs Segregated by Terrain.* Kaggle. https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain

**Methods & metrics**
- Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix).* CVPR. https://arxiv.org/abs/1611.07004
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. https://arxiv.org/abs/1505.04597
- Zhu, J.Y., Park, T., Isola, P., Efros, A.A. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN).* ICCV. https://arxiv.org/abs/1703.10593
- Wang, T.C., et al. (2018). *High-Resolution Image Synthesis with Conditional GANs (Pix2PixHD).* CVPR. https://arxiv.org/abs/1711.11585
- Park, T., et al. (2019). *Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE).* CVPR. https://arxiv.org/abs/1903.07291
- Mao, X., et al. (2017). *Least Squares Generative Adversarial Networks (LSGAN).* ICCV. https://arxiv.org/abs/1611.04076
- Ho, J., Jain, A., Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models (DDPM).* NeurIPS. https://arxiv.org/abs/2006.11239
- Saharia, C., et al. (2022). *Palette: Image-to-Image Diffusion Models.* SIGGRAPH. https://arxiv.org/abs/2111.05826
- Zhang, R., et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).* CVPR. https://arxiv.org/abs/1801.03924
- Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID).* NeurIPS. https://arxiv.org/abs/1706.08500

**SAR-to-optical specific**
- Fuentes Reyes, M., et al. (2019). *SAR-to-Optical Image Translation Based on Conditional Generative Adversarial Networks — Optimization, Opportunities and Limits.* Remote Sensing.
- Bermudez, J.D., et al. (2018). *SAR to Optical Image Synthesis for Cloud Removal with Generative Adversarial Networks.* ISPRS Annals.
