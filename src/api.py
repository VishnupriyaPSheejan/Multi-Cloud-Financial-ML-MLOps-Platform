from fastapi import FastAPI
from pydantic import BaseModel, Field
from predict import predict_one

app=FastAPI(title="Multi-Cloud Financial ML API",version="1.0.0")

class Customer(BaseModel):
    age:int=Field(ge=18,le=100)
    monthly_income:float=Field(gt=0)
    credit_utilization:float=Field(ge=0,le=1)
    payment_delay_days:int=Field(ge=0)
    loan_count:int=Field(ge=0)
    debt_to_income:float=Field(ge=0,le=1)
    account_tenure_months:int=Field(ge=1)
    monthly_transactions:int=Field(ge=0)
    savings_balance:float=Field(ge=0)

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/predict")
def predict(customer:Customer):
    return predict_one(customer.model_dump())
