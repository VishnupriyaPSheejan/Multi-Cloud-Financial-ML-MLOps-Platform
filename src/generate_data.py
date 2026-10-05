from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample"
DATA.mkdir(parents=True, exist_ok=True)

def main(seed=42, n=6000):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 75, n)
    income = np.maximum(1800, rng.lognormal(np.log(5200), 0.45, n))
    utilization = np.clip(rng.beta(2.2, 3.2, n), 0.01, 0.99)
    delays = rng.poisson(2.4, n)
    loans = rng.integers(0, 6, n)
    dti = np.clip(rng.beta(2.4, 4.0, n) + 0.04*loans, 0.03, 0.95)
    tenure = rng.integers(3, 180, n)
    transactions = rng.poisson(32, n) + 2
    savings = np.maximum(100, rng.lognormal(np.log(9000), 0.9, n))

    score = (
        -1.8
        + 4.2 * utilization
        + 0.14 * delays
        + 3.2 * dti
        + 0.28 * loans
        - 0.00010 * income
        - 0.004 * tenure
        - 0.000012 * savings
        - 0.020 * transactions
        + rng.normal(0, 0.30, n)
    )
    prob = 1/(1+np.exp(-score))
    risk = rng.binomial(1, prob)

    customers = pd.DataFrame({
        "customer_id": [f"CUST-{i:06d}" for i in range(1,n+1)],
        "age": age,
        "monthly_income": income.round(2),
        "credit_utilization": utilization.round(4),
        "payment_delay_days": delays,
        "loan_count": loans,
        "debt_to_income": dti.round(4),
        "account_tenure_months": tenure,
        "monthly_transactions": transactions,
        "savings_balance": savings.round(2),
        "high_credit_risk": risk
    })
    customers.to_csv(DATA/"customers.csv", index=False)

    transactions_df = pd.DataFrame({
        "transaction_id": [f"TXN-{i:08d}" for i in range(1,n*3+1)],
        "customer_id": rng.choice(customers.customer_id, n*3),
        "amount": np.round(rng.lognormal(4.3, 1.0, n*3),2),
        "channel": rng.choice(["card","online","branch","atm","wallet"], n*3, p=[.35,.30,.12,.15,.08]),
        "category": rng.choice(["grocery","travel","utilities","retail","dining","health","education"], n*3)
    })
    transactions_df.to_csv(DATA/"transactions.csv", index=False)
    print(f"Generated {n} customers and {len(transactions_df)} transactions in {DATA}")

if __name__ == "__main__":
    main()
