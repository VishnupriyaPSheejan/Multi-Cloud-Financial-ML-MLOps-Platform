import pandas as pd

FEATURES = [
    "age","monthly_income","credit_utilization","payment_delay_days",
    "loan_count","debt_to_income","account_tenure_months",
    "monthly_transactions","savings_balance"
]

def build_features(df):
    x = df[FEATURES].copy()
    x["income_log"] = __import__("numpy").log1p(x["monthly_income"])
    x["savings_log"] = __import__("numpy").log1p(x["savings_balance"])
    x["utilization_dti_interaction"] = x["credit_utilization"] * x["debt_to_income"]
    return x
