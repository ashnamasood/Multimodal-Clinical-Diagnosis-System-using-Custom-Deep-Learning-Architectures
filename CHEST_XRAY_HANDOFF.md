# Chest X-ray Branch — Ashna Instructions

Paste this file into Copilot and follow the steps in order.

## Background (read first)

- `train_chest_xray.py` on `main` includes inference fixes (commit `80157d8` or later).
- Inference uses **per-class thresholds**, not raw top-1 class probability.
- Training reloads **best validation-loss weights** before threshold tuning, test eval, and the final checkpoint save.
- Temporary debug logging may write to `debug-9f5fc4.log` — you can ignore it.

---

## Step 1 — Sync repo and install dependencies

From the project root:

```powershell
cd path\to\Multimodal-Clinical-Diagnosis-System-using-Custom-Deep-Learning-Architectures
git pull origin main
pip install -r requirements.txt
```

**Confirm:** `git log -1 --oneline` shows commit `80157d8` or newer.

---

## Step 2 — Verify you have the updated script

Open [`train_chest_xray.py`](train_chest_xray.py) and confirm these exist (search in file):

- `_format_clinical_prediction`
- `reloaded_best`

If either is missing, run `git pull origin main` again.

---

## Step 3 — Prepare NIH dataset

Extract NIH ChestX-ray14 so this layout works (default `--data-root`):

```
%USERPROFILE%\Downloads\archive\
├── Data_Entry_2017.csv
├── images_001\images\*.png
├── images_002\images\*.png
└── ...
```

If your dataset is elsewhere, use that path in place of `<YOUR_NIH_ARCHIVE_PATH>` in the commands below.

---

## Step 4 — Sanity training (500 images, 10 epochs)

Run a quick pipeline check (~30–60 min depending on GPU):

```powershell
python train_chest_xray.py --data-root <YOUR_NIH_ARCHIVE_PATH> --max-images 500 --epochs 10 --batch-size 8
```

Example if data is in Downloads:

```powershell
python train_chest_xray.py --data-root %USERPROFILE%\Downloads\archive --max-images 500 --epochs 10 --batch-size 8
```

**Confirm in terminal:**

- `Restored best-validation checkpoint from outputs\chest_xray_cnn_best.pt`
- `Optimized per-class thresholds` with values **not all 0.500**
- Final test metrics block (micro/macro F1)

**Confirm these files exist:**

```
outputs/
├── chest_xray_cnn.pt
├── chest_xray_cnn_best.pt
├── chest_xray_classes.json
├── test_metrics.json
└── test_previews/
```

If training fails with out-of-memory, retry with `--batch-size 4`.

---

## Step 5 — Full training (only if Step 4 looks OK)

```powershell
python train_chest_xray.py --data-root <YOUR_NIH_ARCHIVE_PATH> --max-images 0 --epochs 100 --batch-size 8
```

**Target:** micro F1 roughly 0.7–0.8 on full NIH data (see [`README.md`](README.md)).

**Confirm:** `outputs/test_metrics.json` is updated with new `macro_f1`, `micro_f1`, and `optimized_thresholds`.

---

## Step 6 — Inference on new X-rays

1. Place PNG/JPG files in [`input/`](input/) (sample images may already be there).
2. Run:

```powershell
python train_chest_xray.py --inference-only --checkpoint outputs/chest_xray_cnn.pt --external-test-folder input
```

**How to read terminal output (main behavior change):**

| Terminal output | Meaning |
|-----------------|--------|
| `Effusion (0.62), Pneumonia (0.55)` | Both passed optimized thresholds — multi-label prediction |
| `No Finding (low confidence)` | Nothing passed threshold; model is uncertain |
| `No Finding (weak signal: Infiltration 0.31)` | Nothing passed threshold; highest raw score only |

**Confirm artifacts:**

- Previews: `outputs/external_test_previews/*.png`
- CSV: `outputs/external_test_previews/external_test_results.csv` (`predicted_label` uses threshold logic)
- Predictions are **not** the same single disease for every image

---

## Step 7 — Optional: labeled external evaluation

If you have a CSV with `filename` and `Finding Labels` columns:

```powershell
python train_chest_xray.py --inference-only --checkpoint outputs/chest_xray_cnn.pt --external-test-folder input --external-labels-csv path\to\labels.csv
```

**Confirm:** `outputs/external_test_previews/external_test_metrics.json` is created.

---

## Step 8 — Done checklist

Before reporting the chest branch complete:

- [ ] `outputs/test_metrics.json` exists with `optimized_thresholds` and reasonable `macro_f1` / `micro_f1`
- [ ] `outputs/chest_xray_cnn.pt` loads without error in `--inference-only`
- [ ] Inference does not always print the same single disease for every image
- [ ] Grad-CAM previews exist under `outputs/test_previews/` or `outputs/external_test_previews/`
- [ ] Share `outputs/test_metrics.json` and 1–2 preview PNGs with Nauman for review
