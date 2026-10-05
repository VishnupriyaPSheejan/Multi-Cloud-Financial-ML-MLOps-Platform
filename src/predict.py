from pathlib import Path
import joblib
import pandas as pd
from features import build_features

ROOT=Path(__file__).resolve().parents[1]
MODEL=joblib.load(ROOT/"models/credit_risk_model.joblib")

def predict_one(payload):
    df=pd.DataFrame([payload])
    probability=float(MODEL.predict_proba(build_features(df))[:,1][0])
    segment="High" if probability>=.65 else ("Medium" if probability>=.35 else "Low")
    drivers=[]
    if payload["credit_utilization"]>.60: drivers.append("high credit utilization")
    if payload["payment_delay_days"]>5: drivers.append("payment delays")
    if payload["debt_to_income"]>.45: drivers.append("high debt-to-income ratio")
    if payload["loan_count"]>=3: drivers.append("multiple active loans")
    if not drivers: drivers.append("no dominant high-risk driver detected")
    return {"credit_risk_probability":round(probability,4),"risk_segment":segment,"risk_drivers":drivers}
