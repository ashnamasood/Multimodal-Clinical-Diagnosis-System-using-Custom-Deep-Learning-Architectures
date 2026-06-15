# Presentation Assets

Organized images for the 8-slide FusionNet deck. Copy directly into PowerPoint or Google Slides.

**Web app (for screenshots):** `http://127.0.0.1:8000`  
**Start server:** `uvicorn webapp.backend:app --host 127.0.0.1 --port 8000`

---

## Folder map

| Folder | Slide | Contents |
|--------|-------|----------|
| `slide01/` | 1 — Title & overview | Logo, dashboard illustration, feature icon |
| `slide03/` | 3 — Datasets | X-ray + skin sample images |
| `slide04/` | 4 — Preprocessing | Normal + pneumonia X-ray inputs |
| `slide05/` | 5 — Methodology | Feature SVG icons |
| `slide06/` | 6 — X-ray results | Grad-CAM PNGs (true positive + false positive) |
| `slide07/` | 7 — Skin, heart, fusion | Skin samples + **accuracy bar chart** |
| `slide08/` | 8 — Demo | Demo upload files + Grad-CAM reference |
| `screenshots/` | 1, 3, 5, 7, 8 | **You capture these** from the live web app |

---

## Files included (ready to insert)

### slide01/
- `logo.svg`
- `dashboard-illustration.svg`
- `feature-diagnostic.svg`

### slide03/
- `xray_pneumonia.jpeg` — pneumonia chest X-ray
- `xray_normal.jpeg` — normal chest X-ray
- `skin_lesion_mel.jpg` — melanoma-class lesion
- `skin_lesion_nv.jpg` — benign nevus (optional comparison)

### slide04/
- `xray_normal.jpeg`
- `xray_pneumonia.jpeg`

### slide05/
- `feature-diagnostic.svg`
- `feature-fast.svg`
- `feature-private.svg`

### slide06/
- `gradcam_true_positive.png` — pred PNEUMONIA, true PNEUMONIA
- `gradcam_false_positive.png` — pred PNEUMONIA, true NORMAL
- `gradcam_true_positive_alt.png` — extra true-positive example
- `gradcam_true_normal.png` — pred NORMAL, true NORMAL

### slide07/
- `accuracy_comparison_chart.png` — bar chart: X-ray 82% · Skin 55% · Heart 87%
- `skin_lesion_mel.jpg`
- `skin_lesion_nv.jpg`

### slide08/
- `demo_xray_pneumonia.jpeg` — upload during live demo
- `demo_skin_mel.jpg` — upload during live demo
- `gradcam_reference.png` — backup if live Grad-CAM fails

---

## Screenshots to capture (save into `screenshots/`)

| Filename | Slide | How to capture |
|----------|-------|----------------|
| `slide01_overview.png` | 1 | Open app → **Overview** tab → full-page screenshot |
| `slide03_heart_form.png` | 3 | **Diagnosis** tab → cardiac vitals form section |
| `slide05_models_tab.png` | 5 | **Models** tab |
| `slide07_fusion_results.png` | 7 | Run analysis → crop fusion risk + recommendation |
| `slide08_demo_full.png` | 8 | Run analysis with demo inputs → full results panel |

---

## Live demo inputs (slide 8)

Upload these from `slide08/`:
1. `demo_xray_pneumonia.jpeg`
2. `demo_skin_mel.jpg`
3. Click **Run analysis** → show Grad-CAM + fusion risk
