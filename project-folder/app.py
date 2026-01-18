import streamlit as st
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Import model training functions
from model.logistic import train_logistic
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost_model import train_xgboost
from model.data_preprocessing import clean_and_prepare_data


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="2025ab05128 - ML Assignment 2 – Binary Classification",
    layout="wide"
)

st.title("📊 Machine Learning Assignment - Ashok Jayavelu 2025ab05128")
st.write("Binary Classification - Dataset will be uploaded below (heart disease)")

# --------------------------------------------------
# Dataset Upload
# --------------------------------------------------
st.header("📁 Upload Dataset (only in CSV)")
uploaded_file = st.file_uploader(
    "Upload dataset (CSV format only)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Data Cleaning & Preparation
    # --------------------------------------------------
    X, y = clean_and_prepare_data(df)

    # --------------------------------------------------
    # Model Selection
    # --------------------------------------------------
    st.header("🤖 Select Machine Learning Model")

    model_name = st.selectbox(
        "Choose a classification model",
        (
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    # --------------------------------------------------
    # Run Model
    # --------------------------------------------------
    if st.button("🚀 Run Model"):

        if model_name == "Logistic Regression":
            y_test, y_pred, y_prob = train_logistic(X, y)

        elif model_name == "Decision Tree":
            y_test, y_pred, y_prob = train_decision_tree(X, y)

        elif model_name == "K-Nearest Neighbors":
            y_test, y_pred, y_prob = train_knn(X, y)

        elif model_name == "Naive Bayes":
            y_test, y_pred, y_prob = train_naive_bayes(X, y)

        elif model_name == "Random Forest":
            y_test, y_pred, y_prob = train_random_forest(X, y)

        else:
            y_test, y_pred, y_prob = train_xgboost(X, y)

        # --------------------------------------------------
        # Evaluation Metrics
        # --------------------------------------------------
        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
            st.metric("Precision", round(precision_score(y_test, y_pred), 4))

        with col2:
            st.metric("Recall", round(recall_score(y_test, y_pred), 4))
            st.metric("F1 Score", round(f1_score(y_test, y_pred), 4))

        with col3:
            # ✅ FIXED AUC HANDLING
            if y_prob is not None:
                if len(y_prob.shape) == 2:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_prob)

                st.metric("AUC Score", round(auc, 4))
            else:
                st.metric("AUC Score", "Not Available")

            st.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 4))

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted 0", "Predicted 1"],
            index=["Actual 0", "Actual 1"]
        )
        st.table(cm_df)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------
        st.subheader("📄 Classification Report")
        st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload a CSV file to continue.")
