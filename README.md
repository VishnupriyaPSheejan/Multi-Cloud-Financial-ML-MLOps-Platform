# Multi-Cloud Financial ML & MLOps Platform

A production-style portfolio project demonstrating **cloud-agnostic financial machine learning and MLOps** across AWS and Google Cloud concepts.

The project uses a synthetic retail-banking dataset to predict customer credit-risk probability and demonstrate:

- Python + SQL-style analytics
- Predictive modelling with scikit-learn
- AWS and GCP architecture patterns
- Terraform Infrastructure as Code templates
- MLflow-compatible experiment tracking
- Dockerized inference
- FastAPI model serving
- Streamlit business dashboard
- Model evaluation and monitoring
- CI/CD-ready repository structure
- Reproducible synthetic data generation

> **Important:** The default demo runs entirely locally. No AWS/GCP account, credentials, cloud spend, or paid API is required.

## Business problem

A retail bank wants to identify customers who may become higher credit-risk so relationship teams can proactively review exposure and offer appropriate support.

The platform predicts a `credit_risk_probability` and converts it into:

- Low / Medium / High risk segment
- Risk drivers
- Portfolio-level risk distribution
- Model performance metrics
- Customer-level predictions

The architecture is deliberately cloud-neutral. The same logical ML lifecycle can map to:

### AWS
S3 → Glue → Redshift → SageMaker → CloudWatch

### GCP
Cloud Storage → BigQuery → Vertex AI → Cloud Monitoring

### Shared engineering layer
Python → Docker → FastAPI → MLflow → Airflow → Terraform → GitHub Actions

## Architecture

```text
                    ┌─────────────────────────────┐
                    │ Synthetic Banking Data      │
                    │ Customers + Transactions   │
                    └──────────────┬──────────────┘
                                   │
                          Data / Feature Pipeline
                                   │
                    ┌──────────────▼──────────────┐
                    │ Feature Engineering         │
                    │ Python + Pandas + SQL       │
                    └──────────────┬──────────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             │                                           │
      AWS reference path                         GCP reference path
 S3 → Glue → Redshift → SageMaker       GCS → BigQuery → Vertex AI
             │                                           │
             └─────────────────────┬─────────────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │ ML Training       │
                         │ LogisticRegression│
                         │ + calibration     │
                         └─────────┬─────────┘
                                   │
                         ┌─────────▼─────────┐
                         │ Evaluation         │
                         │ ROC-AUC / PR-AUC   │
                         │ F1 / Recall / Brier│
                         └─────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ FastAPI Model Serving       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Streamlit Risk Dashboard    │
                    └─────────────────────────────┘
```

## Run locally

### Option 1 — Python

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python src/generate_data.py
python src/train.py
uvicorn src.api:app --reload
```

In another terminal:

```bash
streamlit run src/dashboard.py
```

API:
`http://localhost:8000/docs`

Dashboard:
`http://localhost:8501`

### Option 2 — Docker

```bash
docker compose up --build
```

Then open:

- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs

## Demo API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "monthly_income": 7200,
    "credit_utilization": 0.64,
    "payment_delay_days": 8,
    "loan_count": 3,
    "debt_to_income": 0.48,
    "account_tenure_months": 64,
    "monthly_transactions": 31,
    "savings_balance": 8500
  }'
```

## Project structure

```text
multi-cloud-financial-mlops-platform/
├── data/
│   └── sample/                 # generated demo datasets
├── models/                     # trained model artifact
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── api.py
│   ├── dashboard.py
│   └── monitoring.py
├── tests/
│   └── test_pipeline.py
├── terraform/
│   ├── aws/
│   └── gcp/
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Cloud portability

The Terraform folders intentionally demonstrate the infrastructure boundary without requiring a cloud account for the demo.

For an actual deployment:

**AWS**
- S3 for raw/curated data
- Glue for transformation
- Redshift for warehouse analytics
- SageMaker for model lifecycle
- CloudWatch for monitoring

**GCP**
- Cloud Storage for raw/curated data
- BigQuery for warehouse analytics
- Vertex AI for model lifecycle
- Cloud Monitoring for observability

The ML code itself is independent of either provider.

## Why this matters for banking

The same architecture can support:

- Credit-risk prediction
- Customer churn
- Fraud-risk scoring
- Next-best-action
- Loan propensity
- Customer segmentation
- Revenue prediction

## Disclaimer

This is an educational portfolio project using synthetic data. It is not a credit decisioning system and must not be used for real lending decisions.
