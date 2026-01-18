import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_prepare_data(df):
    """
    Cleans the heart disease dataset and returns X, y
    """

    # -----------------------------
    # Step 1: Basic validation
    # -----------------------------
    if "target" not in df.columns:
        raise ValueError("Dataset must contain 'target' column")

    # -----------------------------
    # Step 2: Remove duplicates
    # -----------------------------
    df = df.drop_duplicates()

    # -----------------------------
    # Step 3: Handle missing values
    # -----------------------------
    if df.isnull().sum().any():
        df = df.fillna(df.median())

    # -----------------------------
    # Step 4: Feature-Target split
    # -----------------------------
    X = df.drop("target", axis=1)
    y = df["target"]

    # -----------------------------
    # Step 5: Feature Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y
