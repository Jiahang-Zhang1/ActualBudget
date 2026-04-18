from __future__ import annotations

from typing import Iterable


DEFAULT_MODEL_TO_ACTUAL = {
    "Food & Dining": "Food",
    "Transportation": "General",
    "Shopping & Retail": "General",
    "Entertainment & Recreation": "General",
    "Healthcare & Medical": "General",
    "Utilities & Services": "Bills",
    "Financial Services": "Savings",
    "Income": "Income",
    "Government & Legal": "General",
    "Charity & Donations": "General",
}


def normalize_label(label: str | None) -> str:
    return (label or "").strip()


def load_label_mapping_from_cfg(cfg: dict) -> dict[str, str]:
    mapping = dict(DEFAULT_MODEL_TO_ACTUAL)
    user_mapping = cfg.get("label_mapping", {}) or {}
    mapping.update(user_mapping)
    return mapping


def map_model_label_to_actual(label: str, mapping: dict[str, str]) -> str:
    clean = normalize_label(label)
    return mapping.get(clean, clean)


def map_labels(labels: Iterable[str], mapping: dict[str, str]) -> list[str]:
    return [map_model_label_to_actual(label, mapping) for label in labels]
