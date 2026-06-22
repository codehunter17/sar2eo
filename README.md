# SAR-to-EO Image Translation

**Task:** Given a Sentinel-1 SAR (radar) image, generate the corresponding Sentinel-2 EO (optical RGB) image.  
**Method:** Pix2Pix — conditional GAN with U-Net generator and PatchGAN discriminator.  
**Dataset:** Kaggle Sentinel-1&2 Image Pairs (terrain-segregated).

---

## Requirements

- Python 3.10
- CUDA-capable GPU (≤16 GB VRAM — tested on Kaggle T4 free tier)
- See `requirements.txt` for all pinned dependencies

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/sar2eo.git
cd sar2eo

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**Colab / Kaggle** (GPU already available):
```python
!git clone https://github.com/<your-username>/sar2eo.git
%cd sar2eo
!pip install -r requirements.txt
```

---

## Dataset Structure

**Dataset used:** [Sentinel-1&2 Image Pairs Segregated by Terrain](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)  
**Why this dataset:** Pre-paired, terrain-segregated, manageable size, no institutional access required.  
**Split strategy:** We use terrain type as the split boundary. Terrain types used for training: `urban`, `vegetation`, `barren`. Terrain type held out for validation: `water`. This prevents adjacent-patch leakage that would occur with a naïve random split.

Expected directory layout after download and organisation:

```
dataset/
├── train/
│   ├── s1/          ← Sentinel-1 SAR patches (256×256 PNG)
│   └── s2/          ← Sentinel-2 RGB patches (256×256 PNG)
└── val/
    ├── s1/
    └── s2/
```

Update `data_root` in `config.yaml` to point to this directory.

---

## Training

**Full model (L1 + adversarial loss):**
```bash
python train.py --config config.yaml
```

**Ablation — L1 only (no discriminator):**
```bash
# Edit config.yaml: set loss.mode to "l1_only"
python train.py --config config.yaml
```

Checkpoints are saved to `outputs/checkpoints/` every 10 epochs.  
Loss values are logged to `outputs/loss_log.csv`.  
Loss curve is saved to `outputs/loss_curve.png`.  
Qualitative samples are saved to `outputs/samples/`.

---

## Inference

Conforms exactly to the GalaxEye I/O contract:

```bash
python infer.py \
  --input_dir  <path/to/sar_patches> \
  --output_dir <path/to/eo_output> \
  --weights    outputs/final_weights.pth
```

- **Input:** directory of 256×256 8-bit grayscale PNG (dB-scaled, normalised to [0, 255])
- **Output:** directory of 256×256 RGB PNG, same filenames as inputs
- Runs offline — no internet access required at inference time

---

## Evaluation

```bash
# First run inference to generate predictions
python infer.py --input_dir dataset/val/s1 --output_dir outputs/val_preds --weights outputs/final_weights.pth

# Then compute metrics
python eval.py --pred_dir outputs/val_preds --gt_dir dataset/val/s2 --save_json outputs/metrics.json
```

Computes: **LPIPS** (↓), **FID** (↓), **SSIM** (↑), **PSNR** (↑)

---

## Model Weights

**Final checkpoint:** [Download from Hugging Face Hub](https://huggingface.co/<your-username>/sar2eo)

> The weights link will be added after training is complete. The same link is submitted in the GalaxEye submission form.

---

## Results

### Validation Split

| Model | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ |
|-------|---------|-------|--------|--------|
| L1 only (ablation) | — | — | — | — |
| Pix2Pix (full GAN) | — | — | — | — |

> Results will be filled in after training. See `outputs/metrics.json`.

### Training Loss Curve

![Loss Curve](outputs/loss_curve.png)

---

## Citation / References

**Datasets:**
```
Zhu, X.X., et al. (2018). So2Sat LCZ42: A Benchmark Dataset for Local Climate Zone Classification.
SEN1-2 dataset. TU Munich. https://mediatum.ub.tum.de/1436631

Schmitt, M., et al. (2019). SEN12MS - A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery.
https://mediatum.ub.tum.de/1474000

requiemonk (2022). Sentinel-1&2 Image Pairs Segregated by Terrain. Kaggle.
https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain
```

**Papers:**
```
Isola, P., Zhu, J.Y., Zhou, T., & Efros, A.A. (2017). Image-to-Image Translation with Conditional
Adversarial Networks. CVPR 2017. https://arxiv.org/abs/1611.07004

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical
Image Segmentation. MICCAI 2015. https://arxiv.org/abs/1505.04597

Mao, X., et al. (2017). Least Squares Generative Adversarial Networks. ICCV 2017.
https://arxiv.org/abs/1611.04076

Zhang, R., et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).
CVPR 2018. https://arxiv.org/abs/1801.03924
```
