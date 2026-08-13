"""
Streamlit app: Telco Customer Churn — Classification Model Demo

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub, then deploy on
https://streamlit.io/cloud pointing at this file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    return scaler, encoders, feature_columns, models


def preprocess_uploaded(df, encoders, feature_columns):
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen categories gracefully by mapping to the first known class
            df[col] = df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # Keep only the columns the model was trained on, in the right order
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        st.error(f"Uploaded file is missing expected columns: {missing}")
        st.stop()

    return df[feature_columns]


def main():
    st.set_page_config(page_title="Telco Churn — Model Demo", layout="wide")
    st.title("Telco Customer Churn — Classification Model Demo")
    st.write(
        "Upload test data (CSV) with the same columns as the original "
        "Telco Customer Churn dataset, including the `Churn` column, "
        "pick a model, and view its performance."
    )

    scaler, encoders, feature_columns, models = load_artifacts()

    # --- Sidebar controls ---
    st.sidebar.header("Controls")
    model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

    uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file from the sidebar to see predictions and metrics.")
        return

    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded data preview")
    st.dataframe(raw_df.head())

    if "Churn" not in raw_df.columns:
        st.error("Uploaded CSV must include a 'Churn' column (Yes/No or 1/0) to compute metrics.")
        st.stop()

    # True labels
    if pd.api.types.is_numeric_dtype(raw_df["Churn"]):
        y_true = raw_df["Churn"]
    else:
        y_true = raw_df["Churn"].astype(str).map({"Yes": 1, "No": 0})

    X = preprocess_uploaded(raw_df.drop(columns=["Churn"]), encoders, feature_columns)
    X_scaled = scaler.transform(X)

    model = models[model_name]
    y_pred = model.predict(X_scaled)
    y_score = (
        model.predict_proba(X_scaled)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_scaled)
    )

    st.subheader(f"Results — {model_name}")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    st.dataframe(pd.DataFrame(report).T)


if __name__ == "__main__":
    main()
