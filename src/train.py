
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, accuracy_score, brier_score_loss
from features import build_features

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/sample/customers.csv"
MODEL_DIR=ROOT/"models"
MODEL_DIR.mkdir(exist_ok=True)

def train():
    df=pd.read_csv(DATA)
    X=build_features(df)
    y=df["high_credit_risk"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    # Balanced linear baseline: interpretable and appropriate for a banking risk demo.
    pipe=Pipeline([
        ("scale",StandardScaler()),
        ("model",LogisticRegression(max_iter=3000,class_weight="balanced"))
    ])
    pipe.fit(Xtr,ytr)
    p=pipe.predict_proba(Xte)[:,1]
    # Select threshold on training set to maximize F1 rather than assuming 0.50.
    ptr=pipe.predict_proba(Xtr)[:,1]
    thresholds=[i/100 for i in range(10,91)]
    best_t=max(thresholds,key=lambda t:f1_score(ytr,(ptr>=t).astype(int)))
    pred=(p>=best_t).astype(int)
    metrics={
        "accuracy": round(accuracy_score(yte,pred),4),
        "roc_auc": round(roc_auc_score(yte,p),4),
        "pr_auc": round(average_precision_score(yte,p),4),
        "precision": round(precision_score(yte,pred),4),
        "recall": round(recall_score(yte,pred),4),
        "f1": round(f1_score(yte,pred),4),
        "brier_score": round(brier_score_loss(yte,p),4),
        "decision_threshold": round(best_t,2)
    }
    joblib.dump(pipe,MODEL_DIR/"credit_risk_model.joblib")
    (MODEL_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__=="__main__":
    train()
