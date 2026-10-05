from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def generate_monitoring_report():
    df=pd.read_csv(ROOT/"data/sample/customers.csv")
    metrics={
        "rows":len(df),
        "missing_values":int(df.isna().sum().sum()),
        "high_risk_rate":round(df.high_credit_risk.mean(),4),
        "mean_utilization":round(df.credit_utilization.mean(),4),
        "mean_dti":round(df.debt_to_income.mean(),4)
    }
    out=ROOT/"models/monitoring_report.json"
    out.write_text(json.dumps(metrics,indent=2))
    return metrics
if __name__=="__main__":
    print(json.dumps(generate_monitoring_report(),indent=2))
