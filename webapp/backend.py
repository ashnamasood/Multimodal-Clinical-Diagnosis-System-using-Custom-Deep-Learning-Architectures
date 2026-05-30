from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.model_hub import HEART_FORM_SCHEMA, ModelHub, parse_heart_json

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="FusionNet-Scratch API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hub = ModelHub(device="cpu")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "models": hub.availability(),
    }


@app.get("/api/schema")
def schema() -> dict:
    return {
        "heart_form": HEART_FORM_SCHEMA,
        "models": hub.availability(),
    }


@app.post("/api/predict")
async def predict(
    chest_xray_image: UploadFile | None = File(default=None),
    skin_image: UploadFile | None = File(default=None),
    heart_features: str | None = Form(default=None),
) -> dict:
    try:
        parsed_heart = parse_heart_json(heart_features)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    xray_result = None
    skin_result = None
    heart_result = None

    if chest_xray_image is not None:
        xray_bytes = await chest_xray_image.read()
        if xray_bytes:
            xray_result = hub.predict_xray_bytes(xray_bytes)

    if skin_image is not None:
        skin_bytes = await skin_image.read()
        if skin_bytes:
            skin_result = hub.predict_skin_bytes(skin_bytes)

    if parsed_heart is not None:
        try:
            heart_result = hub.predict_heart(parsed_heart)
        except Exception as exc:
            heart_result = {"available": False, "error": str(exc)}

    fusion = hub.fused_assessment(xray_result, skin_result, heart_result)

    return {
        "project": {
            "title": "FusionNet-Scratch",
            "subtitle": "Multimodal Clinical Diagnosis System",
            "department": "Department of Computer Science/Artificial Intelligence, ITU",
        },
        "inputs": {
            "xray_uploaded": chest_xray_image is not None,
            "skin_uploaded": skin_image is not None,
            "heart_provided": parsed_heart is not None,
        },
        "outputs": {
            "xray": xray_result,
            "skin": skin_result,
            "heart": heart_result,
            "fusion": fusion,
        },
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
