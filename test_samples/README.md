# Test samples

Small inputs so teammates can run inference without downloading full datasets.

## X-ray (`test_samples/xray/`)

| File | Expected label |
|------|----------------|
| `normal.jpeg` | NORMAL |
| `normal_02.jpeg` … `normal_04.jpeg` | NORMAL |
| `pneumonia.jpeg` | PNEUMONIA |
| `pneumonia_02.jpeg` … `pneumonia_04.jpeg` | PNEUMONIA |
| `pneumonia_bacterial.jpeg` | PNEUMONIA |

## Skin (`test_samples/skin/`)

Raw lesion photos exported from HMNIST (same source the model was trained on). **Not** Grad-CAM previews — safe to upload in the web app.

| File pattern | Class |
|--------------|-------|
| `lesion_nv.jpg`, `lesion_nv_02.jpg` | nv (melanocytic nevi) |
| `lesion_mel.jpg`, `lesion_mel_02.jpg` | mel (melanoma) |
| `lesion_bcc.jpg`, `lesion_bcc_02.jpg` | bcc |
| `lesion_bkl.jpg`, `lesion_bkl_02.jpg` | bkl |
| `lesion_akiec.jpg`, `lesion_akiec_02.jpg` | akiec |
| `lesion_df.jpg`, `lesion_df_02.jpg` | df |
| `lesion_vasc.jpg`, `lesion_vasc_02.jpg` | vasc |

Regenerate skin samples after pulling (requires local HMNIST CSV):

```powershell
python test_samples/export_skin_samples.py
```

## Heart (`test_samples/heart/`)

| File | Profile |
|------|---------|
| `sample_low_risk.json` | Low-risk vitals |
| `sample_moderate_risk.json` | Mixed risk factors |
| `sample_high_risk.json` | Higher-risk chest pain profile |

## Manifest

`manifest.json` lists every sample and expected X-ray labels for automated checks.

## Quick smoke test (all modalities)

From repo root:

```powershell
python test_samples/run_smoke_tests.py
```

## Web app

```powershell
uvicorn webapp.backend:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and upload any file from `test_samples/xray/` or `test_samples/skin/`. Paste heart JSON from `test_samples/heart/`.

Checkpoints: `XRay-Pneumonia/outputs/xray/xray_cnn.pt` and `Heart&SkinCancer/outputs/{heart,skin}/`.
