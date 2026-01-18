from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    # 🔑 THIS LINE IS IMPORTANT
    y_prob = knn.predict_proba(X_test)

    return y_test, y_pred, y_prob
