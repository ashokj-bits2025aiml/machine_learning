from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

    # 🔑 Naive Bayes SUPPORTS probability
    y_prob = nb.predict_proba(X_test)

    return y_test, y_pred, y_prob
