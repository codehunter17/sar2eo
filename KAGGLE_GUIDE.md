# Kaggle Training Runbook — SAR-to-EO

A click-by-click guide to train both models and produce every result the GalaxEye
assignment needs. No data is downloaded to your computer — everything runs on
Kaggle's free GPU.

---

## 0. One-time: push the code to GitHub (required deliverable anyway)
The notebook clones your repo, so put the project on GitHub first (public).
See `GITHUB_GUIDE` steps if you need them. You'll paste the repo URL into the notebook.

---

## 1. Create the notebook
1. Go to https://www.kaggle.com → **Create → New Notebook**.
2. Top-right **⋮ (settings) → Accelerator → GPU T4 x1**. Also set **Internet → On**
   (needed once to `pip install` and `git clone`).
3. **File → Import Notebook → Upload** `sar2eo_kaggle.ipynb` (in your project folder).

## 2. Attach the dataset
1. Right panel **+ Add Input**.
2. Search **`sentinel12-image-pairs-segregated-by-terrain`** (author *requiemonk*).
3. Click **+** to add it. It mounts read-only under `/kaggle/input/...` (no download).

## 3. Point the notebook at your repo
- In **cell 1 (Settings)** set `REPO_URL` to your public GitHub repo URL.
- (Optional) change `VAL_TERRAIN` / `TEST_TERRAIN` if you want different held-out terrains.
- Leave `SMOKE = True` for the first pass.

## 4. Smoke test (≈ a few minutes)
- **Run All**. With `SMOKE=True` it trains a few epochs on a small subset just to
  prove data prep → train → infer → eval all work and `final_weights.pth` is written.
- If the results table prints at the end with numbers (even bad ones), the pipeline is healthy.

## 5. Full run
1. In cell 1 set **`SMOKE = False`**.
2. **Run All** again. This trains the real models:
   - Model B — `config_full.yaml` (L1 + adversarial)
   - Model A — `config_l1.yaml` (L1 only, the ablation)
3. Expect a few hours total on a T4 (both runs). The 9-hour Kaggle session is enough;
   if you're tight on time, lower `FULL_EPOCHS` (e.g. 100) — note it in the report's time log.

## 6. Grab the outputs
From the **Output** tab (or the file browser on the right) download:
- `outputs_full/final_weights.pth`  ← the model you submit
- `submission_assets/`  ← loss curves, loss CSVs, and sample triplets
- `outputs_full/metrics_val.json`, `metrics_test.json` and the `outputs_l1/...` equivalents

## 7. Host the weights publicly (Hugging Face)
1. https://huggingface.co → **New model** (public). You're already `KeenHunter`.
2. Upload `final_weights.pth`.
3. Copy the public download URL → paste into the **README** *and* the submission form.

## 8. Fill in the report + README
Using the printed results table, loss curves, and 5 triplets:
- Replace every `[fill after training]` cell in `REPORT.md`, then regenerate the PDF
  (or ask me to regenerate it for you).
- Fill the README results table.

## 9. Build the ZIP
`Krishna_Kant_GalaxEye.zip` containing: `Technical_Report.pdf`, the time/resource log,
loss-curve plot(s), and the qualitative triplet images. **Do not** include the weights
(those go via the public link).

---

### Troubleshooting
- **"No /kaggle/input dataset found"** → you didn't Add the dataset (step 2).
- **`prepare_data` says 0 pairs / wrong terrains** → run a quick `!ls /kaggle/input/*`
  to see the real folder names, then set `VAL_TERRAIN`/`TEST_TERRAIN` to match.
- **CUDA out of memory** → lower `batch_size` to 2 in `config.yaml` (or the config cell).
- **Session time limit** → reduce `FULL_EPOCHS`; the model still demonstrates the ablation.
