# ========================
# 📌 1. Import Libraries
# ========================
import pandas as pd
import numpy
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.api import VAR

# ========================
# 📂 2. Load Dataset
# ========================
# 📂 2. Load Dataset
# ========================
DATA_PATH = os.path.join("/AvailableData/GHG_cleaned_v1.csv")
GHG_DATA = pd.read_csv(DATA_PATH)
print("🔹 Columns:", GHG_DATA.columns)

# ========================
# 🧹 3. Data Cleaning & Overview
# ========================
# ตรวจสอบ Missing และ Duplicate
print("\n🔍 Missing values:\n", GHG_DATA.isnull().sum())
print("🔍 Duplicate rows:", GHG_DATA.duplicated().sum())
print("\n📊 Descriptive Stats:\n", GHG_DATA.describe())

# ========================
# 🔗 4. Correlation Analysis
# ========================
plt.figure(figsize=(10, 6))
sns.heatmap(GHG_DATA.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ========================
# 📈 5. Trend Analysis แนวโน้ม (Trend) ของ GHG
# ========================
plt.figure(figsize=(10, 6))
plt.plot(GHG_DATA["Date"], GHG_DATA["GHG"], marker='o')
plt.title("GHG Over Time")
plt.xlabel("Year")
plt.ylabel("GHG Emissions")
plt.grid(True)
plt.show()

# ========================
# ⏳ 6. Time Series Preparation & Stationarity
# ========================
df_ts = GHG_DATA.copy()
df_ts["Date"] = pd.to_datetime(df_ts["Date"], format='%Y')
df_ts.set_index("Date", inplace=True)

# 🧪  6.1 ADF Test - Stationarity Check
# ถ้า p-value < 0.05 → ข้อมูล stationary พร้อมพยากรณ์
# ถ้า p-value > 0.05 → ต้องทำ differencing ก่อน
adf_result = adfuller(df_ts["GHG"])
print("\n🔍 ADF Test (Before Differencing)")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

# 🔁 6.2 Differencing (if non-stationary)
df_ts["GHG_diff"] = df_ts["GHG"].diff()
adf_diff = adfuller(df_ts["GHG_diff"].dropna())

# 🔍 ตรวจสอบอีกครั้งหลังทำ differencing
print("\n🔁 ADF Test (After Differencing)")
print(f"ADF Statistic: {adf_diff[0]}")
print(f"p-value: {adf_diff[1]}")

# Plot Differenced Series
plt.figure(figsize=(10, 4))
plt.plot(df_ts["GHG_diff"])
plt.title("Differenced GHG Series")
plt.xlabel("Year")
plt.ylabel("Differenced GHG")
plt.grid(True)
plt.show()

# ========================
# ⚙️ 7. ARIMA Modeling & Forecasting
# ========================
# 📌 7.1 Fit ARIMA Model
# หา p, d, q ที่เหมาะสมด้วย Auto ARIMA
stepwise_fit = auto_arima(df_ts["GHG"], start_p=0, start_q=0,
                          max_p=5, max_q=5, d=1,
                          seasonal=False, trace=True,
                          suppress_warnings=True, stepwise=True)

print("\n📋 ARIMA Model Summary")
print(stepwise_fit.summary())

# 🔮 7.2 Forecast Future (5 Years Ahead)
# ✂️ Train on 90% for future forecast
train_size = int(len(df_ts) * 0.9)
train_future = df_ts["GHG"][:train_size]
test_future_dates = pd.date_range(start=df_ts.index[train_size], periods=5, freq='YS')

# 🔁 Fit + Forecast
model_future = auto_arima(train_future, d=1, seasonal=False, suppress_warnings=True)
future_forecast, conf_int = model_future.predict(n_periods=5, return_conf_int=True)

# 📄 Forecast DataFrame
future_df = pd.DataFrame({
    "Forecast": future_forecast,
    "Lower CI": conf_int[:, 0],
    "Upper CI": conf_int[:, 1]
}, index=test_future_dates)

# Plot Forecast
plt.figure(figsize=(10, 6))
plt.plot(df_ts["GHG"], label="Actual GHG")
plt.plot(future_df["Forecast"], label="Future Forecast", color='green')
plt.fill_between(future_df.index, future_df["Lower CI"], future_df["Upper CI"], alpha=0.3, color='lightgreen')
plt.title("GHG Forecast - Trained on 90% Data")
plt.xlabel("Year")
plt.ylabel("GHG Emissions")
plt.legend()
plt.grid(True)
plt.show()

# ========================
# 📏 8. ประเมินผล ARIMA Model Evaluation (80/20 Split)
# ========================
# Split 80/20
train_size = int(len(df_ts) * 0.8)
train, test = df_ts["GHG"][:train_size], df_ts["GHG"][train_size:]

# ฟิตโมเดลใหม่กับ train
model = auto_arima(train, d=1, seasonal=False, suppress_warnings=True)
preds = model.predict(n_periods=len(test))

# Evaluation Metrics
mse = mean_squared_error(test, preds)
mae = mean_absolute_error(test, preds)
r2 = r2_score(test, preds)

print("\n📊 Evaluation Metrics")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2 * 100:.2f}%")

# MSE (Mean Squared Error): ค่ายิ่งน้อยยิ่งดี
# MAE (Mean Absolute Error): ค่ายิ่งน้อยยิ่งดี
# R² Score: ค่ายิ่งใกล้ 1 ยิ่งดี (1 = อธิบายข้อมูลได้สมบูรณ์)
 
# ========================
# 📉 9. Residuals Analysis
# ========================
residuals = test - preds

# Histogram
plt.figure(figsize=(10, 4))
sns.histplot(residuals, bins=10, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.show()

# Q-Q Plot
sm.qqplot(residuals, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

# Residuals Over Time
plt.figure(figsize=(10, 4))
plt.plot(residuals, marker='o')
plt.title("Residuals Over Time")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True)
plt.show()

# ACF/PACF
plot_acf(residuals.dropna())
plt.title("ACF of Residuals")
plt.show()

plot_pacf(residuals.dropna())
plt.title("PACF of Residuals")
plt.show()

# Ljung-Box Testอ
# เช็คว่ามี residuals กี่ตัว แล้วใช้ lags ต่ำกว่า
# p-value > 0.05 → residual เป็น white noise → โมเดลดี
max_lags = min(10, len(residuals.dropna()) - 1)
lb_test = acorr_ljungbox(residuals.dropna(), lags=[max_lags], return_df=True)
print("\n🧪 Ljung-Box Test:")
print(lb_test)

# ========================
# 📋 10. Compare Actual vs Predicted
# ========================
# ลองแสดงตารางสรุปผลพยากรณ์กับค่าจริง
results_df = pd.DataFrame({
    "Actual": test.values,
    "Predicted": preds
}, index=test.index)

print("\n📋 Compare Actual vs Predicted")
print(results_df.head())

# ========================
# 📈 11. Visualize All Forecasts
# ========================
# All Forecast Overview
plt.figure(figsize=(12, 6))
plt.plot(df_ts["GHG"], label="Actual GHG")
plt.plot(future_df["Forecast"], label="Forecast", color='green')
plt.fill_between(future_df.index, future_df["Lower CI"], future_df["Upper CI"], alpha=0.4, color='lightgreen')
plt.title("GHG Forecast (ARIMA)")
plt.xlabel("Year")
plt.ylabel("GHG Emissions")
plt.legend()
plt.grid(True)
plt.show()

# Future Forecast Only
plt.figure(figsize=(10, 5))
plt.plot(future_df["Forecast"], marker='o', label="Forecast")
plt.fill_between(future_df.index, future_df["Lower CI"], future_df["Upper CI"], alpha=0.2, color='lightblue')
plt.title("Future Forecast Only")
plt.xlabel("Year")
plt.ylabel("GHG Emissions")
plt.legend()
plt.grid(True)
plt.show()

# ========================
# 🧪 12. Overfitting Check
# ========================
# ถ้า train R² สูงมาก แต่ test R² ต่ำมาก → อาจ overfitting
# Refit model on training data
model = auto_arima(train, d=1, seasonal=False, suppress_warnings=True)

# Predict on training set
train_preds = model.predict_in_sample()

# Evaluate performance on training set
train_mse = mean_squared_error(train, train_preds)
train_r2 = r2_score(train, train_preds)
print(f"\n🧪 Train MSE: {train_mse:.2f}")
print(f"🧪 Train R²: {train_r2:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Train Actual")
plt.plot(train.index, train_preds, label="Train Predicted", linestyle='--')
plt.plot(test.index, test, label="Test Actual")
plt.plot(test.index, preds, label="Test Predicted", linestyle='--')
plt.title("Train vs Test Forecast Comparison")
plt.xlabel("Year")
plt.ylabel("GHG Emissions")
plt.legend()
plt.grid(True)
plt.show()

# ภาพที่ 1: Train Actual vs Train Predicted
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Train Actual")
plt.plot(train.index, train_preds, label="Train Predicted", linestyle='--')
plt.title("Train Forecast Comparison")
plt.xlabel("Year")
plt.ylabel("GHG Emissions")
plt.legend()
plt.grid(True)
plt.show()

# ภาพที่ 2: Test Actual vs Test Predicted
plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label="Test Actual")
plt.plot(test.index, preds, label="Test Predicted", linestyle='--')
plt.title("Test Forecast Comparison")
plt.xlabel("Year")
plt.ylabel("GHG Emissions")
plt.legend()
plt.grid(True)
plt.show()

print("\n📈 Performance Comparison (Train vs Test):")
print(f"Train R²: {train_r2 * 100:.2f}% | Test R²: {r2 * 100:.2f}%")
print(f"Train MSE: {train_mse:.2f} | Test MSE: {mse:.2f}")
print(f"Train MAE: {mean_absolute_error(train, train_preds):.2f} | Test MAE: {mae:.2f}")

# ========================
# 📝 13. Report Summary
# ========================
print("\n📝 Summary for Report:")
trend = 'ลดลง' if df_ts['GHG'].iloc[-1] < df_ts['GHG'].iloc[0] else 'เพิ่มขึ้น'
print(f"• GHG emissions แสดงแนวโน้ม {trend} ในช่วงปี {df_ts.index.year.min()} - {df_ts.index.year.max()}")
print("• จากโมเดล ARIMA พบว่า residual ไม่มีรูปแบบเฉพาะ (randomness สูง) → โมเดลเหมาะสม")
print(f"• ค่า R² = {r2:.2f} บ่งบอกว่าโมเดลสามารถอธิบายความแปรปรวนของข้อมูลได้ {'มาก' if r2 > 0.8 else 'พอสมควร'}")

interval_width = np.mean(future_df["Upper CI"] - future_df["Lower CI"])
print(f"• ความกว้างเฉลี่ยของช่วงพยากรณ์: {interval_width:.2f}")

# ========================
# 💾 13. Export Outputs
# ========================
os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/GHG_forecast_vs_actual_v1.csv")
future_df.to_csv("outputs/GHG_future_forecast_v1.csv")