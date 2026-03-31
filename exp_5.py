import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------
# LOAD DATA (simple substitute)
# ---------------------------
data = load_digits()   # handwritten digits dataset
X = data.data
y = data.target

# normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------
# PLA (FROM SCRATCH)
# ---------------------------
class PLA:
    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.W = np.zeros((len(self.classes), X.shape[1]))

        for idx, c in enumerate(self.classes):
            y_binary = np.where(y == c, 1, -1)
            w = np.zeros(X.shape[1])

            for _ in range(self.epochs):
                for i in range(len(X)):
                    if y_binary[i] * np.dot(w, X[i]) <= 0:
                        w += self.lr * y_binary[i] * X[i]

            self.W[idx] = w

    def predict(self, X):
        scores = np.dot(X, self.W.T)
        return self.classes[np.argmax(scores, axis=1)]

# train PLA
pla = PLA(lr=0.01, epochs=10)
pla.fit(X_train, y_train)
pla_pred = pla.predict(X_test)

print("PLA Accuracy:", accuracy_score(y_test, pla_pred))

# ---------------------------
# MLP (WITH TUNING)
# ---------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    batch_size=64,
    max_iter=200
)

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)

print("MLP Accuracy:", accuracy_score(y_test, mlp_pred))

# ---------------------------
# METRICS
# ---------------------------
print("\nMLP Classification Report:\n")
print(classification_report(y_test, mlp_pred))
