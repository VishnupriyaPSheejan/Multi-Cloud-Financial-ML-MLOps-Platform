import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

def test_feature_pipeline():
    import pandas as pd
    from features import build_features
    df=pd.DataFrame([{"age":40,"monthly_income":5000,"credit_utilization":.4,"payment_delay_days":1,
                      "loan_count":2,"debt_to_income":.3,"account_tenure_months":48,
                      "monthly_transactions":30,"savings_balance":10000}])
    x=build_features(df)
    assert x.shape==(1,12)
    assert x.isna().sum().sum()==0
