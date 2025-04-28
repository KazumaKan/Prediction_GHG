# ===================== 📚 1. Import Library =====================
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from scipy import stats
from sklearn.inspection import PartialDependenceDisplay

# ===================== 🔍 2. Load and Explore Data =====================
DATA_PATH = os.path.join("..", "..", "AvailableData", "GHG_cleaned_v1.csv")
df = pd.read_csv(DATA_PATH)
print(df.columns)

# ===================== 🧹 3. เลือกคอลัมน์ที่ต้องการ =====================
df = df[['GHG', 'CH4', 'CO2', 'F-Gas', 'N2O']]

# ===================== 🔴 4. ตรวจสอบ Missing Values =====================
missing_values = df.isnull().sum()
print(f"Missing values:\n{missing_values}")

# ===================== 🎯 5. เลือก Features และ Target =====================
X = df[['CH4', 'CO2', 'F-Gas', 'N2O']]  # Feature
y = df['GHG']  # Target

# ===================== 📊 6. แบ่งข้อมูลเป็น Train และ Test =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== ⚖️ 7. Scaling ข้อมูล =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 🧑‍💻 8. ใช้ Pipeline สำหรับการสเกลและการฝึกโมเดล =====================
pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))
pipeline.fit(X_train, y_train)

# ===================== 🧪 9. Cross-validation และประเมินผล =====================
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE Scores: {cv_scores}")
print(f"Mean Cross-Validation MSE: {cv_scores.mean()}")

# ===================== 📈 10. ทำนายผลและประเมินผลโมเดล =====================
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# วัดผลของโมเดลด้วย MSE, MAE, และ R-squared
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# แสดงผลการประเมิน
print("Training Metrics:")
print(f"MSE: {mse_train:.2f}")
print(f"MAE: {mae_train:.2f}")
print(f"R-squared: {r2_train:.2%}")

print("\nTesting Metrics:")
print(f"MSE: {mse_test:.2f}")
print(f"MAE: {mae_test:.2f}")
print(f"R-squared: {r2_test:.2%}")

# ===================== 📊 11. วิเคราะห์ Feature Importance =====================
rf = pipeline.named_steps['randomforestregressor']  # Extract the Random Forest model from the pipeline
feature_importance = rf.feature_importances_
feature_names = X.columns

# แสดงกราฟความสำคัญของฟีเจอร์ต่าง ๆ
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, hue=feature_names, palette='viridis')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()

# ===================== 📉 12. แสดง Partial Dependence Plots =====================
features = [0, 1, 2, 3]
feature_names = X_train.columns  # สมมติว่า X_train เป็น DataFrame ที่มีชื่อคอลัมน์

fig = plt.figure(figsize=(10, 8)) # กำหนดขนาด Figure
axes = fig.subplots(2, 2) # สร้าง Layout แบบ 2x2 Subplots

for i, ax in enumerate(axes.flatten()):
    if i < len(features):
        PartialDependenceDisplay.from_estimator(pipeline, X_train, features=[features[i]],
                                                feature_names=feature_names, ax=ax)
        ax.set_title(f"Partial Dependence: {feature_names[features[i]]}")
    else:
        fig.delaxes(ax) # ลบ Subplot ที่ไม่จำเป็น (ถ้าจำนวน Features ไม่เต็ม Layout)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # ปรับ Layout โดยอัตโนมัติ
fig.suptitle('Partial Dependence Plots', fontsize=16)
plt.show()

# ===================== 🔍 13. ตรวจสอบ Underfitting และ Overfitting =====================
train_r2 = r2_train
test_r2 = r2_test

# สร้างกราฟเปรียบเทียบ R-squared ระหว่างข้อมูลฝึกและทดสอบ
plt.figure(figsize=(8, 6))
plt.bar(['Training', 'Testing'], [train_r2, test_r2], color=['green', 'red'])
plt.ylim(0, 1)
plt.title('Comparison of R-squared: Training vs Testing')
plt.ylabel('R-squared')
plt.show()

# ประเมินผลโมเดล
print("\nModel Evaluation:")
if r2_train > 0.95 and r2_test < 0.8:
    print("Overfitting: Model performs very well on training data but poorly on test data.")
elif r2_train < 0.7 and r2_test < 0.7:
    print("Underfitting: Model performs poorly on both training and test data.")
else:
    print("Model performance is reasonable.")

# ===================== 📊 14. วิเคราะห์ Residuals =====================
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

# สร้างกราฟ Residuals
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_train, y=residuals_train)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values (Train)")
plt.ylabel("Residuals (Train)")
plt.title("Residual Plot (Train)")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_test, y=residuals_test)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values (Test)")
plt.ylabel("Residuals (Test)")
plt.title("Residual Plot (Test)")

plt.tight_layout()
plt.show()

# ===================== 🧪 15. การทดสอบ Normality ของ Residuals =====================
stat_train, p_value_train = stats.shapiro(residuals_train)
stat_test, p_value_test = stats.shapiro(residuals_test)

print(f"Shapiro-Wilk Test for Train Residuals: Stat={stat_train}, p-value={p_value_train}")
print(f"Shapiro-Wilk Test for Test Residuals: Stat={stat_test}, p-value={p_value_test}")

# ===================== 📊 16. Histogram of Residuals =====================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(residuals_train, kde=True)
plt.xlabel("Residuals (Train)")
plt.title("Histogram of Residuals (Train)")

plt.subplot(1, 2, 2)
sns.histplot(residuals_test, kde=True)
plt.xlabel("Residuals (Test)")
plt.title("Histogram of Residuals (Test)")

plt.tight_layout()
plt.show()

# ===================== 🔮 17. ทำนายค่าของ GHG สำหรับอนาคต =====================
y_pred_future = pipeline.predict(X_test)

# ===================== 📊 18. แสดงผลการทำนายเทียบกับค่าจริง =====================
results_df = pd.DataFrame({
    'Actual GHG': y_test,
    'Predicted GHG': y_pred_future
})

# แสดงผลการทำนาย
print(results_df.head())

# ===================== 📈 19. Plot กราฟเปรียบเทียบผลลัพธ์ =====================
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df['Actual GHG'], label="Actual GHG", color='b', linewidth=2)
plt.plot(results_df.index, results_df['Predicted GHG'], label="Predicted GHG", color='r', linestyle='--', linewidth=2)

plt.xlabel('Index', fontsize=12)
plt.ylabel('GHG', fontsize=12)
plt.title('Actual vs Predicted GHG', fontsize=14)

plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# ===================== 💾 20. Save Model =====================
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(pipeline, os.path.join(model_dir, "random_forest_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "Random.pkl"))
print("Model saved successfully!")
