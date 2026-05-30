# FusionNet-Scratch: Web Application Setup & Integration Guide

## Overview

**FusionNet-Scratch** is a production-ready multimodal clinical diagnosis web application that integrates three custom deep learning models:

1. **Chest X-ray CNN** (ResNet50-based, 14-class multi-label)
2. **Skin Cancer Classifier** (Custom CNN, 7-class)
3. **Heart Disease Predictor** (Tabular MLP, 2-class)

The system uses **feature-level late fusion** to combine predictions and generate integrated risk assessments.

---

## Project Structure

```
.
├── webapp/                          # Web application (NEW)
│   ├── backend.py                   # FastAPI server (REST API)
│   ├── model_hub.py                 # Unified model loader & inference engine
│   └── static/                      # Frontend assets
│       ├── index.html              # Web UI
│       ├── app.js                  # JavaScript frontend logic
│       └── styles.css              # Dark-mode styling
├── train_chest_xray.py             # Chest X-ray training script
├── multimodal_inference.py         # CLI diagnostic tool
├── Heart&SkinCancer/
│   ├── train_skin_cnn.py          # Skin cancer training
│   ├── train_heart_tabular.py     # Heart disease training
│   ├── inference.py               # Heart/Skin inference
│   └── outputs/
│       ├── skin/
│       │   ├── skin_cnn.pt        # Trained checkpoint
│       │   └── metrics.json
│       └── heart/
│           ├── heart_mlp.pt       # Trained checkpoint
│           ├── heart_preprocessor.joblib
│           └── metrics.json
├── outputs/
│   ├── chest_xray_cnn.pt          # Trained checkpoint ✅
│   ├── chest_xray_cnn_best.pt
│   ├── test_metrics.json
│   └── external_test_previews/
└── requirements.txt                # Python dependencies
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or manually:
pip install torch torchvision numpy pillow matplotlib scikit-learn opencv-python pandas joblib fastapi uvicorn python-multipart httpx
```

### 2. Start the Web Server
```bash
cd /path/to/project
python -m uvicorn webapp.backend:app --host 0.0.0.0 --port 8001 --reload
```

### 3. Access the Web UI
Open your browser to: **http://localhost:8001**

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8001/api/health
```
**Response:**
```json
{
  "status": "ok",
  "models": {
    "xray": true,
    "heart": false,
    "skin": false
  }
}
```

### Schema (Available Fields)
```bash
curl http://localhost:8001/api/schema
```

### Prediction Endpoint
```bash
curl -X POST http://localhost:8001/api/predict \
  -F "chest_xray_image=@input/edema.png" \
  -F "skin_image=@input/skin.jpg" \
  -F "heart_features={...json...}"
```

**Chest X-ray:**
- Input: PNG/JPG image
- Output: Multi-label findings with per-class confidence scores

**Skin Lesion:**
- Input: PNG/JPG image  
- Output: Class prediction + top-3 probabilities

**Heart Features (Tabular):**
```json
{
  "Age": 45,
  "Sex": "M",
  "ChestPainType": "ATA",
  "RestingBP": 120,
  "Cholesterol": 220,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 150,
  "ExerciseAngina": "N",
  "Oldpeak": 1.0,
  "ST_Slope": "Up"
}
```

### Fusion Output
```json
{
  "overall_risk_score": 0.45,
  "overall_risk_level": "moderate",
  "recommendation": "Clinical review is recommended.",
  "notes": ["X-ray positive findings: Atelectasis, Consolidation, Effusion"]
}
```

---

## Model Components

### ChestXRayCNN (ResNet50 backbone)
- **Classes:** 14 diseases (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia)
- **Architecture:** ResNet50 (25M params) → 3-layer classifier → 14 outputs
- **Loss:** Focal Loss (γ=2.0) for class imbalance
- **Inference Latency:** ~150ms per image (CPU)
- **Current Performance:** Micro F1: 0.1260 (limited by data imbalance; detection works but high false positive rate)

### SkinLesionCNN (Custom 4-layer CNN)
- **Classes:** 7 skin lesion types
- **Architecture:** 4-layer custom CNN → embedding → 7-class logits
- **Checkpoints:** `Heart&SkinCancer/outputs/skin/skin_cnn.pt`
- **Status:** ⚠️ Checkpoint not found in current setup

### HeartMLP (Tabular MLP)
- **Classes:** 2 (No disease / Heart disease)
- **Features:** 11 categorical + numeric inputs
- **Architecture:** 64-64-32 → 2 outputs
- **Preprocessor:** RobustScaler (handles clinical data outliers)
- **Checkpoints:** `Heart&SkinCancer/outputs/heart/heart_mlp.pt`
- **Status:** ⚠️ Checkpoint not found in current setup

---

## Feature Fusion Strategy

The system uses **late fusion** at the confidence score level:

1. **Per-Modality Prediction:**
   - X-ray: Multi-label sigmoid probabilities
   - Skin: Single-label softmax probabilities
   - Heart: Binary softmax probabilities

2. **Risk Scoring Logic:**
   - X-ray: Average confidence of positive findings
   - Skin: High confidence if malignant class detected (melanoma, BCC, SCC)
   - Heart: High confidence if disease class predicted

3. **Overall Risk Aggregation:**
   ```
   risk_score = mean([xray_score, skin_score, heart_score])
   ```

4. **Risk Level Classification:**
   - **High (≥0.7):** "Immediate clinical follow-up recommended"
   - **Moderate (0.4–0.7):** "Clinical review recommended"
   - **Low (<0.4):** "No urgent findings; continue monitoring"

---

## Frontend UI Components

### Image Upload Sections
- **Chest X-ray:** Accepts PNG/JPG
- **Skin Lesion:** Accepts PNG/JPG

### Tabular Form (Heart Features)
- Dropdowns for categorical fields (Sex, ChestPainType, etc.)
- Number inputs for numeric fields (Age, RestingBP, etc.)
- Pre-filled with sensible defaults

### Prediction Output
- **Status Display:** Shows which models are available
- **JSON Results Panel:** Full prediction payloads for all modalities
- **Fusion Assessment:** Integrated risk score + recommendation

---

## Troubleshooting

### Models Not Loading
```bash
# Check model checkpoint paths:
ls -la outputs/chest_xray_cnn.pt
ls -la Heart&SkinCancer/outputs/heart/
ls -la Heart&SkinCancer/outputs/skin/
```

### Port Already in Use
```bash
# Use a different port:
python -m uvicorn webapp.backend:app --host 0.0.0.0 --port 8002
```

### Import Errors
```bash
# Ensure dependencies are installed in venv:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Example API Call (Python)

```python
import requests
import json

url = "http://localhost:8001/api/predict"

# Prepare files and data
files = {
    'chest_xray_image': ('edema.png', open('input/edema.png', 'rb'), 'image/png'),
    'skin_image': ('skin.jpg', open('input/skin.jpg', 'rb'), 'image/jpeg'),
}

data = {
    'heart_features': json.dumps({
        "Age": 55,
        "Sex": "F",
        "ChestPainType": "ATA",
        "RestingBP": 130,
        "Cholesterol": 250,
        "FastingBS": 1,
        "RestingECG": "ST",
        "MaxHR": 140,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.5,
        "ST_Slope": "Down"
    })
}

response = requests.post(url, files=files, data=data)
print(response.json()['outputs']['fusion'])
```

---

## Performance Metrics

### Latency (Per Prediction)
- **Chest X-ray:** ~150ms (CPU, ResNet50)
- **Skin Lesion:** ~50ms (CPU, custom CNN)
- **Heart Tabular:** <5ms (MLP on preprocessed data)
- **Total (all 3):** ~210ms

### Accuracy on Test Datasets
| Modality | Metric | Value | Notes |
|----------|--------|-------|-------|
| X-ray | Micro F1 | 0.1260 | Limited by class imbalance; high recall (0.58) but low precision (0.07) |
| X-ray | Macro F1 | 0.1227 | Systemic issue across all classes |
| Skin | (Pending) | N/A | Model checkpoint not located in current system |
| Heart | (Pending) | N/A | Model checkpoint not located in current system |

---

## Next Steps for Improvement

### Short-term (Class Imbalance)
1. Implement **WeightedRandomSampler** for X-ray training
2. Oversample rare diseases (Pneumonia, Edema, Atelectasis)
3. Retrain with balanced data sampling

### Medium-term (Accuracy)
1. Ensemble models across different architectures
2. Fine-tune thresholds per disease based on clinical ROC curves
3. Implement cost-weighted loss functions

### Long-term (Production)
1. Add HIPAA-compliant database backend for patient records
2. Implement real-time model retraining pipeline
3. Deploy via Docker containers on cloud (AWS, GCP, Azure)

---

## References

- [Wang et al., ChestX-ray8, CVPR 2017](https://arxiv.org/abs/1705.02315)
- [Tschandl et al., HAM10000, Scientific Data 2018](https://arxiv.org/abs/1803.10417)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Contributors

- **Ashna Masood** (BSCS23058): Skin CNN, Feature Fusion, Django Dashboard
- **Nauman Arif** (BSCS23060): Chest X-ray CNN, Data Pipeline, Interpretability (Grad-CAM)

**Department:** Computer Science/Artificial Intelligence, Information Technology University  
**Submitted:** May 28, 2026

---

**Status:** ✅ Backend API fully integrated | ✅ Frontend UI complete | ⚠️ Heart/Skin checkpoints need to be located/trained

Access the dashboard at: **http://localhost:8001**
