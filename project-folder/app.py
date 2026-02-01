"""
Author: Ashok Jayavelu
Bits ID: 2025AB05128
Description:
As per the assignment requirement, used sklearn library and generated ML models by:
1. Downloading dataset from GitHub
2. Manual upload from local
3. Selecting and running different classification models
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from model.data_preprocessing import clean_and_prepare_data


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Machine Learning Classification App",
    layout="wide"
)

# ---------------------------------------------------
# Styling (Professional White & Grey)
# ---------------------------------------------------
st.markdown("""
<style>
body { background-color: #f5f5f5; }
h1, h2, h3 { color: #222222; }
section[data-testid="stSidebar"] {
    background-color: #eeeeee;
    border-right: 1px solid #dddddd;
}
.stButton>button {
    background-color: #666666;
    color: white;
    border-radius: 6px;
    font-weight: 500;
}
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 6px;
}
[data-testid="stInfo"] {
    background-color: #f0f0f0;
    border-left: 4px solid #999999;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("Machine Learning Classification Assignment - 2025AB05128")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.header("Machine Learning Controls")

# ---------------------------------------------------
# GitHub RAW Dataset URL (CORRECT PATH)
# ---------------------------------------------------
GITHUB_RAW_CSV = (
    "https://raw.githubusercontent.com/"
    "ashokj-bits2025aiml/machine_learning/main/"
    "project-folder/heart.csv"
)

# ---- Download Button ----
try:
    response = requests.get(GITHUB_RAW_CSV)

    if response.status_code == 200:
        st.sidebar.download_button(
            label="⬇ Download Dataset from Git",
            data=response.content,
            file_name="heart.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.error("Dataset not found on GitHub")

except Exception:
    st.sidebar.error("Unable to connect to GitHub")

# ---- Upload Dataset ----
st.sidebar.markdown("**Upload downloaded dataset**")
uploaded_csv = st.sidebar.file_uploader("", type=["csv"])

# ---- Model Selection ----
model_selected = st.sidebar.selectbox(
    "Select Machine Learning Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

run_model = st.sidebar.button("🚀 Run Model")

# ---------------------------------------------------
# Main Logic
# ---------------------------------------------------
if uploaded_csv is not None:

    data = pd.read_csv(uploaded_csv)

    st.subheader("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    if run_model:

        # ---- Preprocessing ----
        X, y = clean_and_prepare_data(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ---- Model Selection ----
        if model_selected == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000)

        elif model_selected == "Decision Tree":
            clf = DecisionTreeClassifier(random_state=42)

        elif model_selected == "K-Nearest Neighbors":
            clf = KNeighborsClassifier()

        elif model_selected == "Naive Bayes":
            clf = GaussianNB()

        elif model_selected == "Random Forest":
            clf = RandomForestClassifier(random_state=42)

        else:
            clf = XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42
            )

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        probabilities = (
            clf.predict_proba(X_test)[:, 1]
            if hasattr(clf, "predict_proba")
            else None
        )

        # ---- Metrics ----
        results = {
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, probabilities)
            if probabilities is not None else "N/A",
            "Precision": precision_score(y_test, predictions, average="weighted"),
            "Recall": recall_score(y_test, predictions, average="weighted"),
            "F1 Score": f1_score(y_test, predictions, average="weighted"),
            "MCC": matthews_corrcoef(y_test, predictions)
        }

        metrics_df = pd.DataFrame(
            results.items(),
            columns=["Metric", "Value"]
        )
        metrics_df.index += 1

        st.subheader("Evaluation Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        # ---- Confusion Matrix ----
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

        # ---- Classification Report ----
        st.subheader("Classification Report")
        report = classification_report(
            y_test,
            predictions,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

else:
    st.info("⬅ Please download the dataset from GitHub and upload it to proceed.")
