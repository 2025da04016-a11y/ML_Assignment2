"""
Train 5 classification models on the Telco Customer Churn dataset,
evaluate them with the required metrics, and save the trained
models + preprocessing objects for use in the Streamlit app.

Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv
Source : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
(download this CSV yourself and place it in the project root as
 telco_churn.csv before running this script)

Run:
    python model/train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "telco_churn.csv")
MODEL_DIR = os.path.dirname(__file__)


def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Telco churn quirk: TotalCharges is read as object due to blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop the customer ID column — it's an identifier, not a feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical features
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y, encoders


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Some models need predict_proba for AUC; fall back to decision_function if absent
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def main():
    X, y, encoders = load_and_preprocess(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        # Tree-based models don't need scaling, but scaling doesn't hurt them either —
        # using scaled data everywhere keeps the pipeline simple and consistent.
        model.fit(X_train_scaled, y_train)
        results[name] = evaluate(name, model, X_test_scaled, y_test)

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, filename))

    # Save shared preprocessing objects so the Streamlit app can reuse them
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.joblib"))

    # Save metrics table for the README / report
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(MODEL_DIR, "..", "metrics_summary.csv"))
    print("\nSaved metrics_summary.csv and all trained models.")

    # Save a small test_data.csv (unscaled, unlabeled features + true label)
    # for the "upload CSV" feature in the Streamlit app.
    test_export = X_test.copy()
    test_export["Churn"] = y_test.values
    test_export.to_csv(os.path.join(MODEL_DIR, "..", "test_data.csv"), index=False)
    print("Saved test_data.csv for use in the Streamlit app / submission.")


if __name__ == "__main__":
    main()
