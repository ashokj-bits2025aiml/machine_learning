from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    # 🔑 Random Forest SUPPORTS probabilities
    y_prob = rf.predict_proba(X_test)

    return y_test, y_pred, y_prob
