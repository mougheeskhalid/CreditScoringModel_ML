"""
generate_dataset.py
--------------------
Generates a synthetic but realistic "credit history" dataset and saves it to
data/credit_data.csv

Why synthetic data?
Real credit bureau data is private / paid (Kaggle's "Give Me Some Credit",
UCI's German Credit Data, etc. require an account or download step). To keep
this project 100% self-contained and runnable offline, we generate data that
follows the same logical relationships real credit data has (higher income
and on-time payments -> better score, high debt-to-income and missed
payments -> worse score) plus random noise, so the models have something
genuine to learn.

You can swap this file out for a real dataset later -- just make sure the
final CSV has the same column names used in credit_scoring_model.py, or
update TARGET_COL / FEATURE lists there.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 3000

def generate_data(n_samples=N_SAMPLES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 65, n_samples)
    income = rng.normal(55000, 22000, n_samples).clip(12000, 220000)
    employment_length = rng.integers(0, 35, n_samples)
    num_credit_accounts = rng.integers(0, 15, n_samples)
    credit_utilization = rng.uniform(0, 1, n_samples)          # % of credit limit used
    missed_payments_2yrs = rng.poisson(0.6, n_samples).clip(0, 10)
    loan_amount = rng.normal(15000, 9000, n_samples).clip(500, 60000)
    debt_to_income = (rng.uniform(0.05, 0.6, n_samples) +
                       (missed_payments_2yrs * 0.02)).clip(0.02, 0.95)
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], n_samples, p=[0.45, 0.4, 0.15])
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "car", "other"],
        n_samples, p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )

    # --- Underlying "true" creditworthiness score (latent variable) ---
    score = (
        0.9 * (income / 100000)
        - 3.2 * debt_to_income
        - 0.9 * missed_payments_2yrs
        - 1.8 * credit_utilization
        + 0.035 * employment_length
        + 0.015 * (age - 21)
        + rng.normal(0, 0.22, n_samples)          # noise (kept modest, not zero, so it's not a trivial split)
    )

    # Convert latent score into a binary label via logistic function
    prob_good = 1 / (1 + np.exp(-(score + 0.4)))
    creditworthy = (rng.uniform(0, 1, n_samples) < prob_good).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "employment_length_years": employment_length,
        "num_credit_accounts": num_credit_accounts,
        "credit_utilization": credit_utilization.round(3),
        "missed_payments_2yrs": missed_payments_2yrs,
        "loan_amount": loan_amount.round(2),
        "debt_to_income_ratio": debt_to_income.round(3),
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "creditworthy": creditworthy,   # target: 1 = good credit risk, 0 = bad
    })
    return df


if __name__ == "__main__":
    df = generate_data()
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "credit_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print("\nClass balance:")
    print(df["creditworthy"].value_counts(normalize=True).round(3))
    print("\nPreview:")
    print(df.head())
