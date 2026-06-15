# Multimodal Clinical Diagnosis System using Custom Deep Learning Architectures

Research demo combining chest X-ray, skin lesion, and heart disease models behind a FastAPI web app.

## Modalities

| Modality | Status | Training script |
|----------|--------|-----------------|
| Chest X-ray | Pneumonia classifier (NORMAL vs PNEUMONIA) | `XRay-Pneumonia/train_xray_cnn.py` |
| Skin cancer | Ready | `Heart&SkinCancer/train_skin_cnn.py` |
| Heart disease | Ready | `Heart&SkinCancer/train_heart_tabular.py` |

## Setup

```bash
pip install -r requirements.txt
```

## Train chest X-ray model

Dataset lives in `XRay-Pneumonia/chest_xray/chest_xray/` (Kaggle pneumonia export):

```bash
python XRay-Pneumonia/train_xray_cnn.py
```

**Overnight on CPU (recommended):** keep the laptop plugged in and disable sleep, then:

```powershell
powershell -ExecutionPolicy Bypass -File XRay-Pneumonia/run_training.ps1
```

Progress is logged to `XRay-Pneumonia/outputs/xray/training.log`. When finished, check `metrics.json` and run `python XRay-Pneumonia/test_model.py`.

Checkpoints are written to `XRay-Pneumonia/outputs/xray/xray_cnn.pt`.

**Note:** Images are pediatric (ages 1–5). Results may not generalize to adult X-rays.

## Run the web app

```bash
uvicorn webapp.backend:app --host 0.0.0.0 --port 8001
```

Open `http://localhost:8001` in your browser.

## Train heart and skin models

```bash
python Heart&SkinCancer/train_skin_cnn.py
python Heart&SkinCancer/train_heart_tabular.py
```

## Smoke test X-ray model

```bash
python XRay-Pneumonia/test_model.py
python multimodal_inference.py --chest-xray XRay-Pneumonia/chest_xray/chest_xray/test/PNEUMONIA/person100_bacteria_285.jpeg
```

## Project layout

```
├── webapp/                         # FastAPI backend + static frontend
├── Heart&SkinCancer/               # Skin CNN + heart tabular models
├── XRay-Pneumonia/                 # Chest X-ray dataset + training + inference
│   ├── chest_xray/chest_xray/      # Kaggle dataset (train/test)
│   ├── train_xray_cnn.py
│   ├── inference.py
│   └── outputs/xray/
├── input/                          # Sample images for manual testing
└── outputs/                        # Legacy root outputs (gitignored)
```

## Disclaimer

For research and education only. Not for clinical use.
