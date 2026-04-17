from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class Predictor:
    def __init__(self, model_path: str, metadata_path: str | None = None):
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)
        self.metadata = {}

        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.model_version = self.metadata.get(
            "model_version",
            os.environ.get("MODEL_VERSION", self.model_path.stem),
        )

    def _score_matrix(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)

        if hasattr(self.pipeline, "decision_function"):
            scores = self.pipeline.decision_function(X)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            return scores

        preds = self.pipeline.predict(X)
        classes = self.pipeline.named_steps["clf"].classes_
        scores = np.zeros((len(preds), len(classes)), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for i, p in enumerate(preds):
            scores[i, class_to_idx[p]] = 1.0
        return scores

    def predict_one(self, transaction_description: str, country: str, currency: str):
        X = pd.DataFrame(
            [
                {
                    "transaction_description": transaction_description or "",
                    "country": country or "unknown",
                    "currency": currency or "unknown",
                }
            ]
        )

        scores = self._score_matrix(X)
        classes = self.pipeline.named_steps["clf"].classes_
        row = scores[0]
        top_idx = np.argsort(row)[::-1][:3]

        top_categories = [
            {"category_id": str(classes[i]), "score": float(row[i])} for i in top_idx
        ]

        return {
            "predicted_category_id": top_categories[0]["category_id"],
            "confidence": top_categories[0]["score"],
            "top_categories": top_categories,
            "model_version": self.model_version,
        }
