from fastapi import FastAPI
from inference.schemas import PredictRequest, PredictResponse
from inference.predictor import Predictor
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/current/model.joblib")
METADATA_PATH = os.environ.get("METADATA_PATH", "/models/current/metadata.json")

predictor = Predictor(MODEL_PATH, METADATA_PATH)

app = FastAPI(title="actual-ml-inference")


@app.get("/health")
def health():
    return {"ok": True, "model_version": predictor.model_version}


@app.get("/model-info")
def model_info():
    return {
        "model_version": predictor.model_version,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    result = predictor.predict_one(
        transaction_description=req.transaction_description,
        country=req.country or "unknown",
        currency=req.currency or "unknown",
    )
    return PredictResponse(**result)
