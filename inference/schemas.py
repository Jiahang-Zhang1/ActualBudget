from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    transaction_id: str | None = None
    transaction_description: str
    country: str | None = "unknown"
    currency: str | None = "unknown"


class CategoryScore(BaseModel):
    category_id: str
    score: float


class PredictResponse(BaseModel):
    predicted_category_id: str
    confidence: float
    top_categories: List[CategoryScore]
    model_version: str
