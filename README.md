# CodeAlpha_CreditScoringModel

**CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model**

Predicts whether a loan applicant is a **good** or **bad** credit risk from
financial-history features (income, debt-to-income ratio, missed payments,
credit utilization, etc.) using three classifiers: **Logistic Regression**,
**Decision Tree**, and **Random Forest**.

## Project structure

```
CodeAlpha_CreditScoringModel/
├── generate_dataset.py        # builds the synthetic credit dataset -> data/credit_data.csv
├── credit_scoring_model.py    # main script: trains + evaluates all 3 models
├── predict.py                 # loads the saved model and scores new applicants
├── requirements.txt
├── data/
│   └── credit_data.csv        # already generated for you (3,000 rows)
├── models/                    # created after you run credit_scoring_model.py
└── outputs/                   # plots + comparison table land here
```

## About the dataset

Real credit-bureau datasets (Kaggle's "Give Me Some Credit", UCI German
Credit Data, etc.) require an account/download step, so this project
**generates a synthetic dataset** with the same kind of features and the
same real-world logic baked in — higher income and no missed payments push
a person toward "good risk," high debt-to-income and high credit
utilization push toward "bad risk," plus random noise so it isn't trivially
easy. `data/credit_data.csv` is already included and generated with a fixed
random seed, so your results will match this README exactly. Swap in a real
dataset later by replacing that CSV (keep the same column names, or update
the `numeric_features` / `categorical_features` lists in
`credit_scoring_model.py`).

## Setup & run — VS Code

1. **Extract the zip** and open the folder in VS Code (`File → Open Folder…`).
2. **Open a terminal** in VS Code: `` Ctrl + ` `` (backtick), or Terminal → New Terminal.
3. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **(Optional) Regenerate the dataset** — not required, `data/credit_data.csv` is already there:
   ```bash
   python generate_dataset.py
   ```
6. **Train and evaluate all 3 models:**
   ```bash
   python credit_scoring_model.py
   ```
   This prints a classification report + ROC-AUC for each model, saves a
   comparison table to `outputs/model_comparison.csv`, saves 3 plots to
   `outputs/`, and saves the best model to `models/best_model.joblib`.
7. **Test the trained model on new applicants:**
   ```bash
   python predict.py
   ```
   Edit the `new_applicants` list inside `predict.py` to try your own inputs.

If VS Code shows Python errors before you install requirements, that's
expected — select the `venv` interpreter (`Ctrl+Shift+P` → *Python: Select
Interpreter* → pick the one inside `venv/`) so VS Code stops flagging
imports it can't find yet.

## Results (from the included dataset)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.722 | 0.669 | 0.456 | 0.543 | **0.768** |
| Random Forest | 0.702 | 0.658 | 0.364 | 0.469 | 0.750 |
| Decision Tree | 0.685 | 0.577 | 0.484 | 0.526 | 0.682 |

Logistic Regression comes out on top on ROC-AUC and gets saved as
`models/best_model.joblib`. `outputs/feature_importance.png` (from the
Random Forest) shows **credit utilization** and **debt-to-income ratio** as
the strongest predictors — matches real-world credit scoring intuition.

## Internship submission checklist (from the task PDF)

- [ ] Push this folder to a GitHub repo named `CodeAlpha_CreditScoringModel`
- [ ] Post a short video explaining the project on LinkedIn, tag @CodeAlpha, include the GitHub link
- [ ] Submit through the WhatsApp-group submission form
- [ ] Remember: at least 2–3 tasks total are needed for the completion certificate — this is only Task 1
