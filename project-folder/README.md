# 📊 Machine Learning Assignment 2– Binary Classification (Streamlit App)

This project is a **Streamlit-based Machine Learning application** developed as part of **ML Assignment 2**.
It allows users to upload a dataset, select a binary classification model, train the model, and evaluate its performance using multiple metrics.

## 👨‍🎓 Author

**Ashok Jayavelu**
Roll Number: **2025ab05128**
Course: *Machine Learning*
---

## 🚀 Features

* Upload any **CSV dataset** for binary classification
* Automatic **data cleaning and preprocessing**
* Train and evaluate the following models:

  * Logistic Regression
  * Decision Tree
  * K-Nearest Neighbors (KNN)
  * Naive Bayes
  * Random Forest
  * XGBoost
* Display comprehensive evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * AUC Score
  * Matthews Correlation Coefficient (MCC)
* Visualize:

  * Confusion Matrix
  * Classification Report

---

## 🧠 Models Used

The application supports multiple supervised learning algorithms implemented using **scikit-learn** and **XGBoost**.
Each model is modularized and stored inside the `model/` directory.

---

## 📁 Project Structure

```
project-folder/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python version for Streamlit Cloud
├── heart.csv                # Sample dataset (optional)
├── model/
│   ├── data_preprocessing.py
│   ├── logistic.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   └── xgboost_model.py
```

---

## ⚙️ Requirements

* **Python 3.10** (mandatory for Streamlit Cloud compatibility)
* Required libraries are listed in `requirements.txt`

### `requirements.txt`

```
streamlit==1.50.0
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.3.2
scipy==1.11.4
xgboost==2.1.4
```

---

## 🐍 Python Version (Important)

This project **must run on Python 3.10**.

The following file is required for Streamlit Cloud:

### `runtime.txt`

```
python-3.10
```

---

## ▶️ How to Run Locally

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd project-folder
   ```

2. Create and activate a virtual environment:

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deploying on Streamlit Cloud

1. Push the project to GitHub
2. Ensure these files exist in the root:

   * `app.py`
   * `requirements.txt`
   * `runtime.txt`
3. Go to **Streamlit Cloud**
4. Create a new app and select your repository
5. Deploy 🚀

---

## 📊 Dataset Requirements

* Input file must be in **CSV format**
* Target variable should be **binary (0/1)**
* Feature preprocessing is handled automatically

---

## 📝 Evaluation Metrics Explained

* **Accuracy** – Overall correctness of the model
* **Precision** – Correct positive predictions
* **Recall** – Ability to detect positive cases
* **F1 Score** – Balance between precision and recall
* **AUC** – Model’s ability to distinguish classes
* **MCC** – Robust metric for imbalanced datasets

---




