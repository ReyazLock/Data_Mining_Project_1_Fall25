import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Concrete_Data.csv")

X = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

train_idx = list(range(0, 501)) + list(range(631, len(data)))
test_idx = list(range(501, 631))
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

def gradient_descent(X, y, lr=1e-7, epochs=100000, tol=1e-6):
    n, d = X.shape
    m = np.random.randn(d) * 0.01
    b = 0.0
    losses = []
    prev_loss = float('inf')

    for epoch in range(epochs):
        y_pred = np.dot(X, m) + b
        error = y_pred - y
        loss = np.mean(error ** 2)
        losses.append(loss)

        grad_m = (2 / n) * np.dot(X.T, error)
        grad_b = (2 / n) * np.sum(error)

        m -= lr * grad_m
        b -= lr * grad_b

        if abs(prev_loss - loss) < tol:
            print(f"Converged at epoch {epoch}")
            break
        prev_loss = loss

    return m, b, losses

m, b, losses = gradient_descent(X_train, y_train)

y_train_pred = np.dot(X_train, m) + b
y_test_pred = np.dot(X_test, m) + b

mse_train = np.mean((y_train_pred - y_train) ** 2)
mse_test = np.mean((y_test_pred - y_test) ** 2)

r2_train = 1 - np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
r2_test = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print("Final model (standardized X):")
for i, coef in enumerate(m, start=1):
    print(f"m_{i} = {coef:.6f}")
print(f"b = {b:.6f}")

print("\nTraining Results:")
print(f"  MSE = {mse_train:.3f}")
print(f"  R^2 = {r2_train:.3f}")

print("\nTesting Results:")
print(f"  MSE = {mse_test:.3f}")
print(f"  R^2 = {r2_test:.3f}")

plt.figure(figsize=(7, 4))
plt.plot(losses, linewidth=2)
plt.title("Gradient Descent Convergence (MSE vs Iteration)")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()
