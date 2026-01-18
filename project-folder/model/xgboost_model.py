from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="logloss",   # IMPORTANT to avoid warnings
        use_label_encoder=False
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)

    # 🔑 XGBoost SUPPORTS probability
    y_prob = xgb.predict_proba(X_test)

    return y_test, y_pred, y_prob
