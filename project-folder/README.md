# Machine Learning Assignment – 2

**Course:** Machine Learning
**Programme:** M.Tech (AIML)
**Student Name:** Ashok Jayavelu
**Roll Number:** 2025ab05128

---

## a. Problem Statement

The objective of this assignment is to implement multiple **machine learning classification models**, evaluate their performance using standard metrics, and deploy the models using an **interactive Streamlit web application**.

The assignment demonstrates an **end-to-end machine learning workflow**, including:

* Dataset selection
* Model implementation
* Model evaluation
* User interface development
* Cloud deployment using Streamlit Community Cloud

---

## b. Dataset Description

* **Dataset Name:** Heart Disease Dataset
* **Source:** Public dataset from Kaggle
* **Problem Type:** Binary Classification
* **Target Variable:** Presence of heart disease (0 = No, 1 = Yes)
* **Number of Features:** ≥ 12
* **Number of Instances:** ≥ 500

The dataset consists of patient health-related attributes such as age, cholesterol level, blood pressure, and other clinical parameters used to predict the presence of heart disease.

---

## c. Models Used and Evaluation Metrics

The following **six classification models** were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

### Evaluation Metrics Used

For each model, the following metrics were calculated:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

### Model Performance Comparison Table

| ML Model                 | Accuracy | AUC    | Precision | Recall | F1 Score | MCC     |
| ------------------------ | -------- | ------ | --------- | ------ | -------- | ------- |
| Logistic Regression      | 0.8033   | 0.8712 | 0.8035    | 0.8033 | 0.8025   | 0.6031  |
| Decision Tree            | 0.8033   | 0.8019 | 0.8033    | 0.8033 | 0.8033   | 0.6039  |
| KNN                      | 0.7869   | 0.8377 | 0.7880    | 0.7869 | 0.7855   | 0.5702  |
| Naive Bayes              | 0.7869   | 0.8842 | 0.7914    | 0.7869 | 0.7872   | 0.5771  | 
| Random Forest (Ensemble) | 0.7541   | 0.8588 | 0.7537    | 0.7541 | 0.7537   | 0.5038  |
| XGBoost (Ensemble)       | 0.7213   | 0.8323 | 0.7208    | 0.7213 | 0.7209   | 0.4376  |

---

## Model-wise Observations

| ML Model                 | Observation                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------- |
| Logistic Regression      | Performs well as a baseline model with stable and interpretable results.                |
| Decision Tree            | Captures non-linear relationships but may overfit the dataset.                          |
| KNN                      | Sensitive to the choice of K and distance metric; performance varies with data scaling. |
| Naive Bayes              | Fast and efficient but assumes feature independence, which may reduce accuracy.         |
| Random Forest (Ensemble) | Provides improved performance due to ensemble averaging and reduced overfitting.        |
| XGBoost (Ensemble)       | Achieves the best overall performance due to gradient boosting and regularization.      |

---

## Streamlit Application Features

The deployed Streamlit application includes the following mandatory features:

* CSV dataset upload option
* Model selection dropdown
* Display of evaluation metrics
* Confusion matrix and classification report

---

## Project Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- runtime.txt
│-- README.md
│-- model/
│   │-- logistic.py
│   │-- decision_tree.py
│   │-- knn.py
│   │-- naive_bayes.py
│   │-- random_forest.py
│   │-- xgboost_model.py
│   │-- data_preprocessing.py
```

---

## Deployment Details

* **Platform:** Streamlit Community Cloud
* **Python Version:** 3.10
* **Deployment Type:** Free Tier
* **GitHub link** https://github.com/ashokj-bits2025aiml/machine_learning/tree/main/project-folder
* **Live App Link:** https://machinelearning-ukbynakebxswsrqnxbrzmb.streamlit.app/ *

---

**End of README**
