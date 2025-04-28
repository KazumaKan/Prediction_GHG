# GHG Prediction Model Evaluation 🌍📊

This project is part of the CPE408 course, where we evaluate and compare the performance of three different models for predicting Greenhouse Gas (GHG) emissions. The goal is to assess the accuracy and efficiency of each model and provide insights into how they can be applied for environmental impact analysis.

## 📚 Project Overview
In this repository, we implement and compare three distinct machine learning and statistical models to predict GHG emissions:

1. **Random Forest** 🌲 - An ensemble learning method that uses multiple decision trees to improve prediction accuracy and handle complex, non-linear data.
2. **XGBoost** 🚀 - A powerful gradient boosting algorithm known for its efficiency and high predictive performance, especially on structured/tabular data.
3. **ARIMA** 📈 - A time series forecasting model that uses historical data to predict future values, particularly useful for sequential and time-dependent GHG data.

We evaluate these models based on multiple performance metrics, including **Mean Absolute Error (MAE)**, **Root Mean Squared Error (MSE)**, and **R² Score**, to determine which model offers the best prediction capability.

## ⚙️ Features
- **Data Preprocessing**: Handling missing values, feature scaling, and splitting the data into training and test sets.
- **Model Training**: Training and fine-tuning the Random Forest, XGBoost, and ARIMA models.
- **Model Evaluation**: Comparing model performance using standard evaluation metrics.
- **Visualization**: Plotting results to visualize the accuracy and effectiveness of each model.

## 🌐 Dataset
The dataset used for this project comes from the **World Bank**. It includes historical data on Greenhouse Gas (GHG) emissions across different countries, which is used as the basis for training and testing the models.

## 💡 Key Insights
The objective of this evaluation is to determine the best model for predicting GHG emissions by comparing their performance on real-world datasets. This will help in understanding how different types of models perform on environmental data and contribute to making informed decisions on sustainable practices.

## 🚀 Requirements
- Python 3.x
- Libraries : pandas, numpy, matplotlib, seaborn, joblib, scipy, scikit-learn, statsmodels, pmdarima, xgboost

## 📊 Results
The results will be presented in a clear and concise manner, including:
- A comparison of the models' performance.
- Visualizations of the prediction accuracy.

## 📥 How to Run
1. Clone this repository.
2. Install the required libraries
3. Run code: *python/python3/py* + Name project

### What’s included:
- The project description and overview.
- Models used (Random Forest, XGBoost, ARIMA).
- Dataset information (World Bank).
- Steps on how to run the project.
