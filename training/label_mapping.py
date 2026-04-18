MODEL_TO_ACTUAL = {
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

def map_model_label_to_actual(label: str) -> str:
    return MODEL_TO_ACTUAL.get(label, label)
