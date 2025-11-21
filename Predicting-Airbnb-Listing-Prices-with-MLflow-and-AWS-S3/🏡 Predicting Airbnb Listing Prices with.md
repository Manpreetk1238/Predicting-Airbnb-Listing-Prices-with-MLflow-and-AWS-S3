📘 Airbnb Price Prediction with MLflow & AWS S3
📝 Project Overview

This project builds an end-to-end machine learning pipeline to predict Airbnb nightly listing prices using real listing data. The workflow includes data retrieval from AWS S3, preprocessing, exploratory analysis, feature engineering, model training, evaluation, and MLflow experiment tracking. The goal is to deliver a clean, reproducible, and industry-standard pipeline for price prediction.

🚀 Objectives

Retrieve Airbnb dataset from AWS S3

Perform data cleaning, handling missing values, and outlier removal

Create new features such as distance to city center and amenity counts

Build machine learning models to predict nightly price

Track experiments using MLflow

Register and compare models based on RMSE, MAE, and R²

Provide visual insights through EDA and model evaluation plots


☁️ 1. Data Source – AWS S3

The raw Airbnb data is stored in:

s3://your-bucket-name/airbnb/raw_data/listings.csv


You can download it using boto3 or s3fs:

import pandas as pd
df = pd.read_csv("s3://your-bucket-name/airbnb/raw_data/listings.csv")

🧹 2. Data Cleaning & Preprocessing

Key steps:

Removed irrelevant columns

Cleaned missing values

Capped extreme price outliers

Fixed inconsistent datatypes

Parsed dates and numerical fields

🧩 3. Feature Engineering

Created additional features to enhance model performance:

amenities_count (parsed amenity list)

dist_to_center_km (approx distance to Manhattan center)

reviews_per_month_filled (filled missing values)

📊 4. Exploratory Data Analysis (EDA)

Basic exploratory analysis was performed to understand distributions, correlations, and trends in the dataset.
This includes:

Price distribution analysis

Identifying skewness and extreme values

Understanding feature relationships

Detecting patterns across neighbourhoods and room types
🤖 5. Model Development

Models trained:

Linear Regression

Random Forest Regressor

Train/test split: 80/20
Scaling applied where necessary.

🧪 6. MLflow Experiment Tracking

Each model run logs:

Parameters

Metrics (RMSE, MAE, R²)

Feature importance plots

Trained model artifacts

Example MLflow code:

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")


MLflow UI displays:

Experiment comparisons

Run metrics

Registered best model

📈 7. Model Performance Metrics
| Model              | RMSE        | MAE        | R²       |
|-------------------|-------------|------------|----------|
| Linear Regression | 9070.24     | 55.98      | 0.3614   |
| Random Forest     | 8373.88     | 52.44      | 0.4104   |


Random Forest performed better across all evaluation metrics.

🏁 Conclusion

This project demonstrates a complete machine learning pipeline integrated with AWS S3 and MLflow. The Random Forest model achieved the best accuracy and is registered as the final production candidate. The repository provides a reproducible, industry-level workflow suitable for scaling into a real pricing system.

📦 How to Run

Clone the repo:

git clone https://github.com/yourname/airbnb-price-prediction.git


Install dependencies:

pip install -r requirements.txt


Start MLflow UI:

mlflow ui


Run the Jupyter notebook.

