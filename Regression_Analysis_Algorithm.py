import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("Concrete_Data.csv")
data.columns = [
    'Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer',
    'CoarseAggregate', 'FineAggregate', 'Age', 'CompressiveStrength'
]

train_data = data.iloc[:900]
test_data = data.iloc[900:]

X_train = train_data.drop(columns=['CompressiveStrength'])
y_train = train_data['CompressiveStrength']
X_test = test_data.drop(columns=['CompressiveStrength'])
y_test = test_data['CompressiveStrength']

def standardize(df_train, df_test):
    mean = df_train.mean()
    std = df_train.std()
    return (df_train - mean) / std, (df_test - mean) / std

X_train_std, X_test_std = standardize(X_train, X_test)

X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

def run_regression(X_train, X_test, y_train, y_test, label):
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train_const).fit()
    y_train_pred = model.predict(X_train_const)
    y_test_pred = model.predict(X_test_const)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"=== {label} ===")
    print(f"Training MSE: {mse_train:.3f}, Testing MSE: {mse_test:.3f}")
    print(f"Training R^2: {r2_train:.3f}, Testing R^2: {r2_test:.3f}")

    result = pd.DataFrame({
        'Feature': model.params.index,
        'Coefficient': model.params.values,
        'p_value': model.pvalues.values
    })
    result['Model'] = label
    return result

results_raw = run_regression(X_train, X_test, y_train, y_test, 'Raw Values')
results_std = run_regression(X_train_std, X_test_std, y_train, y_test, 'Standardized')
results_log = run_regression(X_train_log, X_test_log, y_train, y_test, 'Log-transformed')

p_value_summary = pd.concat([results_raw, results_std, results_log], axis=0)
p_value_summary.to_csv("p_value_summary.csv", index=False)

print("\nAll model p-values saved to p_value_summary.csv")
