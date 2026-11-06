import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# Load PCA-reduced datasets for Problem 1 and Problem 2
# Each CSV must have PCA features + a "label" column
# ============================================================

p1_train = pd.read_csv("problem1_train.csv")
p1_valid = pd.read_csv("problem1_valid.csv")
p2_train = pd.read_csv("problem2_train.csv")
p2_valid = pd.read_csv("problem2_valid.csv")

# Split features/labels
def split_xy(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

X1_train, y1_train = split_xy(p1_train)
X1_valid, y1_valid = split_xy(p1_valid)

X2_train, y2_train = split_xy(p2_train)
X2_valid, y2_valid = split_xy(p2_valid)

# Standardize before SVM
scaler_p1 = StandardScaler()
scaler_p2 = StandardScaler()

X1_train_scaled = scaler_p1.fit_transform(X1_train)
X1_valid_scaled = scaler_p1.transform(X1_valid)

X2_train_scaled = scaler_p2.fit_transform(X2_train)
X2_valid_scaled = scaler_p2.transform(X2_valid)

# ============================================================
# Helper function: train + evaluate + runtime measurement
# ============================================================

def evaluate_model(model, X_train, y_train, X_valid, y_valid, model_name):
    print(f"\n============================")
    print(f"Training {model_name}")
    print(f"============================")

    # ---- Training time ----
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # ---- Train accuracy + runtime ----
    t0 = time.time()
    train_pred = model.predict(X_train)
    train_pred_time = time.time() - t0
    train_acc = accuracy_score(y_train, train_pred)

    # ---- Validation accuracy + runtime ----
    t0 = time.time()
    valid_pred = model.predict(X_valid)
    valid_pred_time = time.time() - t0
    valid_acc = accuracy_score(y_valid, valid_pred)

    # ---- Print results ----
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {valid_acc:.4f}")
    print(f"Training Time: {train_time:.4f} sec")
    print(f"Prediction Time (Train): {train_pred_time:.6f} sec")
    print(f"Prediction Time (Valid): {valid_pred_time:.6f} sec")

    return {
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "train_time": train_time,
        "pred_time_train": train_pred_time,
        "pred_time_valid": valid_pred_time
    }


# ============================================================
# Define final chosen models with selected hyperparameters
# ============================================================

# Problem 1
rf_p1 = RandomForestClassifier(n_estimators=200, random_state=42)
svm_p1 = SVC(C=100)

# Problem 2
rf_p2 = RandomForestClassifier(n_estimators=100, random_state=42)
svm_p2 = SVC(C=100)

# ============================================================
# Run all four models
# ============================================================

results = {}

results["Problem1_RF"] = evaluate_model(
    rf_p1, X1_train, y1_train, X1_valid, y1_valid, "Problem 1 - Random Forest"
)

results["Problem1_SVM"] = evaluate_model(
    svm_p1, X1_train_scaled, y1_train, X1_valid_scaled, y1_valid, "Problem 1 - SVM"
)

results["Problem2_RF"] = evaluate_model(
    rf_p2, X2_train, y2_train, X2_valid, y2_valid, "Problem 2 - Random Forest"
)

results["Problem2_SVM"] = evaluate_model(
    svm_p2, X2_train_scaled, y2_train, X2_valid_scaled, y2_valid, "Problem 2 - SVM"
)

print("\n\n===============================")
print("All models trained successfully!")
print("===============================")

for key, val in results.items():
    print(f"\nModel: {key}")
    for metric, v in val.items():
        print(f"  {metric}: {v:.4f}")
