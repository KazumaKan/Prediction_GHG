import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# อ่านไฟล์ CSV
file_path = "../RawData/DataSet_Total_Gas.csv"  # ตรวจสอบว่าไฟล์อยู่ตำแหน่งนี้จริง
df = pd.read_csv(file_path)

# แสดงข้อมูล 5 แถวแรก เพื่อตรวจสอบว่าโหลดมาถูกต้อง
print(df.head())

#********************************************************************************************************************
#🔸ตรวจสอบค่าที่หายไป & แก้ไข Missing Values (ค่าหาย)
print("🔸 Missing Values ในแต่ละคอลัมน์ 🔸")
print(df.isnull().sum())

# แสดงแถวที่มี Missing Values
df_missing = df[df.isnull().any(axis=1)]
print("\n🔸 ข้อมูลที่มีค่าหายไป 🔸")
print(df_missing)

# แสดงแถวที่มี Missing Values  (ใช้ Heatmap)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="Reds", cbar=True, linewidths=0.5, linecolor="black")
plt.title("🔴 Heatmap show Missing Values of Dataset", fontsize=14)
plt.show()

#🔹 วิธีแก้ไขข้อมูล
#ลบแถวที่มีค่าหายไป
df.dropna(inplace=True)

# ตรวจสอบอีกครั้งว่ามีค่าหายไปหรือไม่
print("🔸 Missing Values ในแต่ละคอลัมน์ หลังจากลบแล้ว 🔸")
print(df.isnull().sum())

# แสดง Heatmap อีกครั้ง
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="Reds", cbar=True, linewidths=0.5, linecolor="black")
plt.title("🔴 Heatmap after Removing Missing Values", fontsize=14)
plt.show()

#********************************************************************************************************************
# 🔸ตรวจสอบ & แก้ไข Data Inconsistency (ข้อมูลไม่สอดคล้องกัน)
# 📌 เงื่อนไขที่ต้องตรวจสอบ:
# ค่าของ GHG ต้องเป็นผลรวมของ CH4 + CO2 + F-Gas + N2O

# คำนวณค่าผลรวมที่ถูกต้อง
df["GHG_Calculated"] = df["CH4"] + df["CO2"] + df["F-Gas"] + df["N2O"]

# ตรวจสอบว่าค่า GHG ตรงกับค่าที่คำนวณได้หรือไม่
df["Inconsistency"] = df["GHG"] != df["GHG_Calculated"]

# แสดงเฉพาะแถวที่มีความไม่สอดคล้องกัน
inconsistent_rows = df[df["Inconsistency"]]
print("🔹 ข้อมูลที่มีความไม่สอดคล้องกัน 🔹")
print(inconsistent_rows)

# แสดงจำนวนแถวที่ไม่สอดคล้องกัน
print(f"\n🔹 จำนวนแถวที่ไม่สอดคล้องกัน: {len(inconsistent_rows)} แถว")

# 🔸 ตรวจสอบประเภทของข้อมูลในแต่ละคอลัมน์
print("🔹 ประเภทข้อมูลของแต่ละคอลัมน์ 🔹")
print(df.dtypes)
#---------------------------------------------------------
#🔹 วิธีแก้ไขข้อมูล
# แก้ไขค่าที่ผิดพลาด
df.loc[df["Inconsistency"], "GHG"] = df["GHG_Calculated"]

# ลบคอลัมน์ที่ใช้คำนวณออก
df.drop(columns=["GHG_Calculated", "Inconsistency"], inplace=True)

# แสดงผลลัพธ์ที่แก้ไขแล้ว
print("\n✅ ข้อมูลที่แก้ไขแล้ว ✅")
print(df)

# ตรวจสอบอีกครั้งหลังจากการแก้ไข
df["GHG_Calculated"] = df["CH4"] + df["CO2"] + df["F-Gas"] + df["N2O"]
df["Inconsistency"] = df["GHG"] != df["GHG_Calculated"]

# แสดงผลลัพธ์ของแถวที่ยังคงมีความไม่สอดคล้องกัน
inconsistent_rows_after_fix = df[df["Inconsistency"]]

#🔹 ตรวจสอบอีกที
# หากไม่มีข้อมูลไม่สอดคล้องกันจะมีแถวที่ว่าง
if len(inconsistent_rows_after_fix) == 0:
    print("✅ ไม่มีข้อมูลที่ไม่สอดคล้องกันหลังจากการแก้ไข")
else:
    print("🔹 ข้อมูลที่ยังคงไม่สอดคล้องกัน 🔹")
    print(inconsistent_rows_after_fix)

# ลบคอลัมน์ที่ใช้คำนวณออก
df.drop(columns=["GHG_Calculated", "Inconsistency"], inplace=True)

#********************************************************************************************************************
#🔸ตรวจสอบ & แก้ไข Data Redundancy (ข้อมูลซ้ำซ้อน)
# แสดงข้อมูลก่อนการลบข้อมูลซ้ำ
print("🔹 ข้อมูลก่อนการลบข้อมูลซ้ำ 🔹")
print(df)

# ตรวจสอบข้อมูลซ้ำ
duplicates = df[df.duplicated()]
# แสดงข้อความหากไม่มีข้อมูลซ้ำ
if duplicates.empty:
    print("\n🔹 ไม่มีข้อมูลที่ซ้ำ 🔹")
else:
    print("\n🔸 ข้อมูลที่ซ้ำ 🔸")
    print(duplicates)
    
#---------------------------------------------------------
#🔹 วิธีแก้ไขข้อมูล_ถ้ามีลบข้อมูลซ้ำ
df_no_duplicates = df.drop_duplicates()

# แสดงข้อมูลหลังการลบข้อมูลซ้ำ
print("\n✅ ข้อมูลหลังจากการลบข้อมูลซ้ำ ✅")
print(df_no_duplicates)

#---------------------------------------------------------
#💠จำลองข้อมูลที่ซ้ำ-โดยการคัดลอกข้อมูลทั้งแถวของปี 2012 โดยการคัดลอกแถวนี้และเพิ่มลงไปใน DataFrame
# สร้างข้อมูลซ้ำโดยการคัดลอกแถวของปี 2012
df_duplicate = df[df['Date'] == 2012].copy()

# เพิ่มข้อมูลซ้ำลงไปใน DataFrame
df_with_duplicates = pd.concat([df, df_duplicate], ignore_index=True)

# ตรวจสอบข้อมูลซ้ำ
duplicates = df_with_duplicates[df_with_duplicates.duplicated(keep=False)]  # keep=False เพื่อให้แสดงแถวทั้งหมดที่ซ้ำ

# แสดงข้อความหากไม่มีข้อมูลซ้ำ
if duplicates.empty:
    print("\n🔸 ไม่มีข้อมูลที่ซ้ำ 🔸")
else:
    print("\n🔸 ข้อมูลที่ซ้ำ 🔸")
    print(duplicates)
    
#🔹 วิธีแก้ไขข้อมูล-ลบแถวที่ซ้ำโดย เก็บแถวแรกที่พบและลบแถวที่ซ้ำ
# แสดงข้อมูลก่อนการลบข้อมูลซ้ำ
    print("🔹 ข้อมูลก่อนการลบข้อมูลซ้ำ 🔹")
    print(df_with_duplicates)

# ลบข้อมูลที่ซ้ำและเก็บแถวแรก
df_without_duplicates = df_with_duplicates.drop_duplicates(keep='first')

# แสดงข้อมูลหลังจากการลบซ้ำ
print("\n🔹 ข้อมูลหลังการลบข้อมูลซ้ำ 🔹")
print(df_without_duplicates)

# ตรวจสอบข้อมูลซ้ำอีกครั้ง
duplicates_after_drop = df_without_duplicates[df_without_duplicates.duplicated(keep=False)]

# แสดงข้อความหากไม่มีข้อมูลซ้ำ
if duplicates_after_drop.empty:
    print("\n🔹 ไม่มีข้อมูลที่ซ้ำ 🔹")
else:
    print("\n🔸 ข้อมูลที่ซ้ำ 🔸")
    print(duplicates_after_drop)

#********************************************************************************************************************
#🔸ตรวจสอบ Outliers (ข้อมูลแปลกประหลาด)
# ใช้วิธีหาด้วย IQR Method
# ฟังก์ชันหาค่า Outlier ด้วย IQR
def find_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# หา Outliers ในทุกคอลัมน์
outlier_results = {}
for col in ["GHG", "CH4", "CO2", "F-Gas", "N2O"]:
    outliers = find_outliers_iqr(df, col)
    outlier_results[col] = outliers

# แสดงค่า Outliers ที่พบ
for col, outliers in outlier_results.items():
    if not outliers.empty:
        print(f"🔴 Outliers ในคอลัมน์ {col}:")
        print(outliers)
        print("-" * 50)
    else:
        print(f"✅ ไม่มี Outliers ในคอลัมน์ {col}")

# แสดง Box Plot ของแต่ละคอลัมน์
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[["GHG", "CH4", "CO2", "F-Gas", "N2O"]])
plt.title("Box Plot แสดง Outliers ในแต่ละคอลัมน์", fontsize=14)
plt.show()

#---------------------------------------------------------
#🔹 วิธีแก้ไขข้อมูล_ใช้ Log Transformation ลดผลกระทบ Outliers
#ใช้ np.log1p(x) แปลงค่า
#ใช้ np.expm1(x) แปลงค่ากลับ
#แสดงกราฟ Box Plot ก่อนทำ Log Transform
plt.figure(figsize=(8,5))
sns.boxplot(x=df["F-Gas"])
plt.title("🔴 Box Plot ของ F-Gas (ก่อนใช้ Log Transform)")
plt.show()

#ใช้ Log Transform กับคอลัมน์ F-Gas
df["F-Gas_Log"] = np.log1p(df["F-Gas"])  # log(1 + F-Gas)

#แสดงกราฟ Box Plot หลังทำ Log Transform
plt.figure(figsize=(8,5))
sns.boxplot(x=df["F-Gas_Log"])
plt.title("🟢 Box Plot ของ F-Gas (หลังใช้ Log Transform)")
plt.show()

#วิธีแปลงค่ากลับไปเป็นค่าปกติ
df["F-Gas_Reverted"] = np.expm1(df["F-Gas_Log"])  # แปลงกลับ exp(F-Gas_Log) - 1

#ตรวจสอบค่าก่อนและหลังแปลงกลับ
df[["F-Gas", "F-Gas_Log", "F-Gas_Reverted"]].head()
print(df.columns)

df[["F-Gas", "F-Gas_Log", "F-Gas_Reverted"]].head()

#*การแปลงค่ากลับไปใช้ค่าเดิมทำได้โดยใช้ np.expm1()*
df["F-Gas_Reverted"] = np.expm1(df["F-Gas_Log"])  # แปลงค่ากลับ exp(F-Gas_Log) - 1
df["diff"] = df["F-Gas"] - df["F-Gas_Reverted"] 
print(df[["F-Gas", "F-Gas_Reverted", "diff"]].head(10))
#ลบคอลัมน์ที่ไม่ได้ใช้
df.drop(columns=["F-Gas_Log", "diff","F-Gas_Reverted"], inplace=True)
print(df.head())

#********************************************************************************************************************
#🔸การแก้ไข Imbalanced Dataset (ข้อมูลไม่สมดุล)
#ดูค่าเฉลี่ย (mean) และค่ามัธยฐาน (median) ของแต่ละคลาส
print(df.describe())

# สร้างลูปแสดงกราฟสำหรับแต่ละคอลัมน์ใน DataFrame
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:  # ตรวจสอบให้แสดงเฉพาะคอลัมน์ที่มีชนิดข้อมูลตัวเลข
        plt.figure(figsize=(8, 5))  # กำหนดขนาดกราฟ
        sns.histplot(df[column], bins=20, kde=True)  # สร้างกราฟ histplot พร้อม kde
        plt.title(f"Distribution of {column}")  # ชื่อกราฟตามชื่อคอลัมน์
        plt.show()  # แสดงกราฟ

input("Press Enter to exit...")


# ปัดค่าทศนิยมในคอลัมน์ตัวเลขให้เหลือ 2 ตำแหน่ง
df = df.round(2)

# บันทึก DataFrame เป็นไฟล์ CSV โดยไม่เก็บ index
# 📂 บันทึก DataFrame เป็นไฟล์ CSV
output_file_path = "/AvailableData/GHG_cleaned_v1.csv"
df.to_csv(output_file_path, index=False)

