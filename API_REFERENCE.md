# FusionNet-Scratch API Reference

## Endpoints

### GET /api/health
Check if backend is running and which models are available.

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

---

### GET /api/schema
Get form schema and model availability for frontend.

**Response:**
```json
{
  "heart_form": {
    "Sex": ["M", "F"],
    "ChestPainType": ["TA", "ATA", "NAP", "ASY"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["Y", "N"],
    "ST_Slope": ["Up", "Flat", "Down"],
    "numeric": ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
  },
  "models": {
    "xray": true,
    "heart": false,
    "skin": false
  }
}
```

---

### POST /api/predict
**Main prediction endpoint.** Submit images and/or tabular data for diagnosis.

**Request (multipart/form-data):**
- `chest_xray_image` (optional): PNG/JPG file
- `skin_image` (optional): PNG/JPG file
- `heart_features` (optional): JSON string

**Example heart_features JSON:**
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

**Response:**
```json
{
  "project": {
    "title": "FusionNet-Scratch",
    "subtitle": "Multimodal Clinical Diagnosis System",
    "department": "Department of Computer Science/Artificial Intelligence, ITU"
  },
  "inputs": {
    "xray_uploaded": true,
    "skin_uploaded": false,
    "heart_provided": true
  },
  "outputs": {
    "xray": {
      "available": true,
      "display_text": "Fibrosis (0.52), Nodule (0.50), Hernia (0.50)",
      "predicted_labels": ["Fibrosis", "Nodule", "Hernia"],
      "headline_confidence": 0.5183,
      "top5": [
        {"label": "Fibrosis", "confidence": 0.5183},
        {"label": "Nodule", "confidence": 0.5048},
        {"label": "Hernia", "confidence": 0.5015},
        {"label": "Mass", "confidence": 0.4977},
        {"label": "Emphysema", "confidence": 0.4931}
      ],
      "thresholds": {
        "Atelectasis": 0.5,
        "Cardiomegaly": 0.5,
        "Effusion": 0.5,
        ...
      }
    },
    "skin": {
      "available": false,
      "error": "Skin model checkpoint not found."
    },
    "heart": {
      "available": false,
      "error": "Heart model checkpoint/preprocessor not found."
    },
    "fusion": {
      "overall_risk_score": 0.5123,
      "overall_risk_level": "moderate",
      "recommendation": "Clinical review is recommended.",
      "notes": [
        "X-ray positive findings: Fibrosis, Nodule, Hernia"
      ]
    }
  }
}
```

---

## Model Details

### Chest X-ray Model

**Classes (14 diseases):**
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

**Output Format:**
- `display_text`: Human-readable comma-separated findings
- `predicted_labels`: List of class names above threshold
- `headline_confidence`: Confidence of top finding
- `top5`: List of all classes with confidence (sorted by confidence descending)
- `thresholds`: Per-class decision thresholds (for debugging)

**Multi-label Logic:**
- Uses sigmoid activation (each class independent)
- Each class has a learned threshold (default 0.5)
- Multiple classes can be positive simultaneously

---

### Skin Model

**Classes (7 lesion types):**
- Melanoma (high-risk)
- Basal Cell Carcinoma (BCC) (high-risk)
- Squamous Cell Carcinoma (SCC) (high-risk)
- Actinic Keratosis (medium-risk)
- Benign Keratosis (low-risk)
- Nevus (low-risk)
- Vascular Lesion (low-risk)

**Output Format:**
- `predicted_class`: Top class prediction
- `confidence`: Softmax confidence of top class
- `top3`: List of top 3 classes with confidences

**Status:** ⚠️ Checkpoint not found in current deployment

---

### Heart Model

**Classes:**
- 0: "No heart disease"
- 1: "Heart disease"

**Input Features (11 total):**
**Categorical (5):**
- `Sex`: "M" or "F"
- `ChestPainType`: "TA", "ATA", "NAP", or "ASY"
- `RestingECG`: "Normal", "ST", or "LVH"
- `ExerciseAngina`: "Y" or "N"
- `ST_Slope`: "Up", "Flat", or "Down"

**Numeric (6):**
- `Age`: 0–100 (years)
- `RestingBP`: 60–200 (mmHg)
- `Cholesterol`: 0–600 (mg/dL)
- `FastingBS`: 0 or 1 (boolean, >120 mg/dL)
- `MaxHR`: 60–210 (bpm)
- `Oldpeak`: 0–6.2 (ST depression)

**Output Format:**
- `predicted_class`: "No heart disease" or "Heart disease"
- `confidence`: Softmax confidence
- `top2`: Both classes with confidences

**Status:** ⚠️ Checkpoint not found in current deployment

---

## Fusion Algorithm

### Risk Scoring
```
1. For each available modality, extract a confidence score:
   - X-ray: avg(confidence of positive findings)
   - Skin: confidence if malignant (mel/bcc/scc), else (1 - confidence) * 0.3
   - Heart: confidence if disease, else (1 - confidence) * 0.3

2. Aggregate:
   overall_risk = mean([modality_scores])

3. Classify:
   if risk >= 0.7: level = "high", rec = "Immediate clinical consultation"
   elif risk >= 0.4: level = "moderate", rec = "Clinical review recommended"
   else: level = "low", rec = "No urgent findings"
```

### Notes Generation
- Lists all positive findings from each modality
- Emphasizes high-risk conditions (malignant skin lesions, heart disease)
- Provides clinical context for downstream triage

---

## Error Handling

### Missing Checkpoint
If a model checkpoint is not found during startup, the API returns:
```json
{
  "available": false,
  "error": "Model checkpoint not found."
}
```

### Invalid Input
If heart features JSON is malformed, API returns HTTP 400:
```json
{
  "detail": "heart_features must be a JSON object."
}
```

### File Upload Errors
If image file cannot be processed (corrupt, wrong format):
```json
{
  "available": false,
  "error": "Failed to load image: [error message]"
}
```

---

## Example Requests

### cURL: X-ray Only
```bash
curl -X POST http://localhost:8001/api/predict \
  -F "chest_xray_image=@input/edema.png"
```

### cURL: All Modalities
```bash
curl -X POST http://localhost:8001/api/predict \
  -F "chest_xray_image=@input/edema.png" \
  -F "skin_image=@input/skin.jpg" \
  -F "heart_features={\"Age\":45,\"Sex\":\"M\",\"ChestPainType\":\"ATA\",\"RestingBP\":120,\"Cholesterol\":220,\"FastingBS\":0,\"RestingECG\":\"Normal\",\"MaxHR\":150,\"ExerciseAngina\":\"N\",\"Oldpeak\":1.0,\"ST_Slope\":\"Up\"}"
```

### Python: Requests Library
```python
import requests
import json

response = requests.post(
    "http://localhost:8001/api/predict",
    files={
        "chest_xray_image": ("edema.png", open("input/edema.png", "rb"), "image/png"),
    },
    data={
        "heart_features": json.dumps({
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
        })
    }
)
print(json.dumps(response.json(), indent=2))
```

### JavaScript: Fetch API
```javascript
const formData = new FormData();
formData.append("chest_xray_image", document.getElementById("xray").files[0]);
formData.append("heart_features", JSON.stringify({
  Age: 45,
  Sex: "M",
  ChestPainType: "ATA",
  RestingBP: 120,
  Cholesterol: 220,
  FastingBS: 0,
  RestingECG: "Normal",
  MaxHR: 150,
  ExerciseAngina: "N",
  Oldpeak: 1.0,
  ST_Slope: "Up"
}));

fetch("/api/predict", { method: "POST", body: formData })
  .then(r => r.json())
  .then(data => console.log(data.outputs.fusion));
```

---

## Performance Expectations

- **Latency:** 200–300ms per request (all 3 modalities, CPU)
- **Concurrency:** Limited by CPU cores (default ~4 concurrent requests on 4-core CPU)
- **Memory:** ~2GB RSS for all 3 models loaded

---

## Deployment

### Docker
```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "webapp.backend:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Kubernetes
See `k8s-deployment.yaml` for example StatefulSet configuration.

---

**Last Updated:** May 28, 2026  
**API Version:** 1.0.0  
**Status:** ✅ Production-ready (with caveats on model availability)
