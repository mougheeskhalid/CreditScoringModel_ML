"""
predict.py
-----------
Loads the saved best model (models/best_model.joblib) and scores one or more
new loan applicants. This is the "deployment" piece -- run this any time you
want a creditworthiness prediction without retraining.

Run:
    python predict.py

Edit the `new_applicants` list below with your own data to test other cases.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "best_model.joblib")
META_PATH = os.path.join("models", "model_metadata.joblib")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No saved model found. Run `python credit_scoring_model.py` first."
        )
    pipeline = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return pipeline, meta


def predict(pipeline, applicants_df):
    preds = pipeline.predict(applicants_df)
    probs = pipeline.predict_proba(applicants_df)[:, 1]
    out = applicants_df.copy()
    out["predicted_creditworthy"] = preds
    out["probability_good_credit"] = probs.round(3)
    out["label"] = out["predicted_creditworthy"].map({1: "GOOD credit risk", 0: "BAD credit risk"})
    return out


if __name__ == "__main__":
    pipeline, meta = load_model()
    print(f"Loaded model: {meta['best_model_name']}")

    # Example applicants -- replace with real data or load from a CSV
    new_applicants = pd.DataFrame([
        {
            "age": 34, "annual_income": 72000, "employment_length_years": 8,
            "num_credit_accounts": 4, "credit_utilization": 0.22,
            "missed_payments_2yrs": 0, "loan_amount": 12000,
            "debt_to_income_ratio": 0.18, "home_ownership": "MORTGAGE",
            "loan_purpose": "debt_consolidation",
        },
        {
            "age": 25, "annual_income": 28000, "employment_length_years": 1,
            "num_credit_accounts": 6, "credit_utilization": 0.85,
            "missed_payments_2yrs": 3, "loan_amount": 9000,
            "debt_to_income_ratio": 0.55, "home_ownership": "RENT",
            "loan_purpose": "credit_card",
        },
    ])

    results = predict(pipeline, new_applicants)
    print("\nPredictions:")
    print(results[["age", "annual_income", "debt_to_income_ratio", "label", "probability_good_credit"]]
          .to_string(index=False))
