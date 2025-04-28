import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os 

DATA_PATH = os.path.join("..", "..", "AvailableData", "GHG_cleaned_v1.csv")
df = pd.read_csv(DATA_PATH)

features = ['CH4', 'CO2', 'F-Gas', 'N2O']
target = 'GHG'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, 
                     learning_rate=0.1, max_depth=3, random_state=42)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("📊 Evaluation Metrics")
print(f"🔸 MSE (Mean Squared Error): {mse:.4f}")
print(f"🔸 MAE (Mean Absolute Error): {mae:.4f}")
print(f"🔸 R² (R-squared): {r2:.4f}")

residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, residuals_train, color='dodgerblue', edgecolor='black')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("XgBoost Residual Plot Train")
plt.xlabel("Train")
plt.ylabel("Residuals Train")

plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, residuals_test, color='deepskyblue', edgecolor='black')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("XgBoost Residual Plot Test")
plt.xlabel("Test")
plt.ylabel("Residuals Test")

plt.tight_layout()
plt.show()
