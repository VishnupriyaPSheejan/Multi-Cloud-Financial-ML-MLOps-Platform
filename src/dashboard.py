from pathlib import Path
import json, joblib, pandas as pd, streamlit as st
import plotly.express as px
from predict import predict_one

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/sample/customers.csv"
MODEL=ROOT/"models/credit_risk_model.joblib"

st.set_page_config(page_title="Financial ML Risk Platform",layout="wide")
st.title("🏦 Multi-Cloud Financial ML & MLOps Platform")
st.caption("Synthetic retail-banking portfolio | Cloud-neutral demo | No cloud credentials required")

if not DATA.exists() or not MODEL.exists():
    st.warning("Run: python src/generate_data.py && python src/train.py")
    st.stop()

df=pd.read_csv(DATA)
metrics=json.loads((ROOT/"models/metrics.json").read_text())

c1,c2,c3,c4=st.columns(4)
c1.metric("Customers",f"{len(df):,}")
c2.metric("High-risk rate",f"{df.high_credit_risk.mean()*100:.1f}%")
c3.metric("ROC-AUC",metrics["roc_auc"])
c4.metric("Recall",metrics["recall"])

left,right=st.columns(2)
with left:
    fig=px.histogram(df,x="credit_utilization",color="high_credit_risk",nbins=30,title="Credit Utilization Distribution")
    st.plotly_chart(fig,use_container_width=True)
with right:
    risk=df.assign(risk_segment=pd.cut(df.credit_utilization,[-1,.35,.65,2],labels=["Low","Medium","High"]))
    fig=px.scatter(df,x="debt_to_income",y="monthly_income",color="high_credit_risk",
                   title="Debt-to-Income vs Income",opacity=.55)
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Customer Risk Simulator")
cols=st.columns(3)
age=cols[0].slider("Age",21,75,42)
income=cols[1].number_input("Monthly income",1800.,30000.,7200.)
util=cols[2].slider("Credit utilization",0.,1.,.64)
cols=st.columns(3)
delay=cols[0].slider("Payment delay days",0,30,8)
loans=cols[1].slider("Loan count",0,8,3)
dti=cols[2].slider("Debt-to-income",0.,.95,.48)
cols=st.columns(3)
tenure=cols[0].slider("Account tenure (months)",1,180,64)
tx=cols[1].slider("Monthly transactions",0,150,31)
savings=cols[2].number_input("Savings balance",0.,200000.,8500.)

if st.button("Predict Credit Risk",type="primary"):
    result=predict_one({"age":age,"monthly_income":income,"credit_utilization":util,
                        "payment_delay_days":delay,"loan_count":loans,"debt_to_income":dti,
                        "account_tenure_months":tenure,"monthly_transactions":tx,
                        "savings_balance":savings})
    st.success(f"Risk segment: {result['risk_segment']} | Probability: {result['credit_risk_probability']:.1%}")
    st.write("**Risk drivers:** "+", ".join(result["risk_drivers"]))
