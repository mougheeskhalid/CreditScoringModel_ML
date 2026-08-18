"""
credit_scoring_model.py
------------------------
CodeAlpha Machine Learning Internship - Task 1: Credit Scoring Model

Predicts an individual's creditworthiness (good credit risk vs bad) from
financial history features, using:
    - Logistic Regression
    - Decision Tree
    - Random Forest

Evaluates every model with Precision, Recall, F1-Score and ROC-AUC, saves
comparison plots to outputs/, and saves the best-performing model to
models/best_model.joblib for reuse in predict.py.

Run:
    python credit_scoring_model.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, still writes PNG files fine
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

DATA_PATH = os.path.join("data", "credit_data.csv")
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
TARGET_COL = "creditworthy"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {DATA_PATH}")
    return df


def build_preprocessor(df):
    numeric_features = [
        "age", "annual_income", "employment_length_years",
        "num_credit_accounts", "credit_utilization",
        "missed_payments_2yrs", "loan_amount", "debt_to_income_ratio",
    ]
    categorical_features = ["home_ownership", "loan_purpose"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    return preprocessor, numeric_features, categorical_features


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
    }


def evaluate_model(name, pipeline, X_test, y_test, results, roc_data):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc,
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Bad Risk (0)", "Good Risk (1)"]))
    print(f"ROC-AUC: {auc:.4f}")

    return y_pred, auc


def plot_roc_curves(roc_data, path):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved ROC curve comparison to {path}")


def plot_confusion_matrix(y_test, y_pred, model_name, path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad Risk", "Good Risk"],
                yticklabels=["Bad Risk", "Good Risk"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} (Best Model)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {path}")


def plot_feature_importance(pipeline, numeric_features, categorical_features, path):
    rf_model = pipeline.named_steps["classifier"]
    if not hasattr(rf_model, "feature_importances_"):
        return
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_names = numeric_features + cat_names

    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh([all_names[i] for i in order][::-1], importances[order][::-1], color="#4C72B0")
    plt.xlabel("Importance")
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    preprocessor, numeric_features, categorical_features = build_preprocessor(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    results = []
    roc_data = {}
    fitted_pipelines = {}

    for name, model in get_models().items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        evaluate_model(name, pipeline, X_test, y_test, results, roc_data)

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print("\n================ MODEL COMPARISON ================")
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    best_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC = {results_df.iloc[0]['ROC-AUC']:.4f})")

    # Plots
    plot_roc_curves(roc_data, os.path.join(OUTPUT_DIR, "roc_curves.png"))
    best_pred = best_pipeline.predict(X_test)
    plot_confusion_matrix(y_test, best_pred, best_name, os.path.join(OUTPUT_DIR, "confusion_matrix_best_model.png"))
    if best_name == "Random Forest":
        plot_feature_importance(best_pipeline, numeric_features, categorical_features,
                                 os.path.join(OUTPUT_DIR, "feature_importance.png"))
    else:
        # Also always save RF feature importance as a bonus insight, if RF was trained
        plot_feature_importance(fitted_pipelines["Random Forest"], numeric_features, categorical_features,
                                 os.path.join(OUTPUT_DIR, "feature_importance.png"))

    # Save best model + column metadata
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump({"best_model_name": best_name, "columns": list(X.columns)},
                os.path.join(MODEL_DIR, "model_metadata.joblib"))
    print(f"\nSaved best model pipeline to {MODEL_DIR}/best_model.joblib")
    print("Done. Check the outputs/ folder for plots and models/ for the saved model.")


if __name__ == "__main__":
    main()
