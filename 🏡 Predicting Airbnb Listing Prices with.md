🏡 Airbnb Price Prediction with MLflow and AWS S3
⭐ 1. Project Overview

This project develops a complete, reproducible machine learning pipeline to predict Airbnb listing prices using AWS S3 for data storage and MLflow for experiment tracking.

The workflow includes:

Retrieving raw Airbnb data from S3

Cleaning & preprocessing noisy dataset

Exploratory data analysis (EDA)

Feature engineering (amenities, distance, reviews, etc.)

Training multiple regression models

Tracking all runs using MLflow

Registering the best model in the MLflow Model Registry

Structuring the project for production-grade reproducibility

📌 2. Business Problem

Listing prices vary widely even among similar properties.
StayWise aims to:

Provide hosts with smart, data-driven price recommendations

Improve revenue predictability

Increase platform competitiveness

This model predicts optimal nightly prices based on:

Location

Room type

Reviews

Host activity

Amenities

Distance to city center

🗂️ 3. Project Architecture
AWS S3 (Raw Dataset)
         ↓
 Data Loading (boto3 / s3fs)
         ↓
 Data Cleaning & Processing
         ↓
 Exploratory Data Analysis (EDA)
         ↓
 Feature Engineering
         ↓
 Model Training (LR, RF, XGBoost)
         ↓
 MLflow Experiment Tracking
         ↓
 Model Evaluation
         ↓
 MLflow Model Registry (Best Model)

📁 4. Dataset Description

Key attributes in the dataset:

id

name

host_id

host_name

neighbourhood_group

neighbourhood

latitude

longitude

room_type

price

minimum_nights

number_of_reviews

last_review

reviews_per_month

calculated_host_listings_count

availability_365

price_capped (engineered)

The dataset contains missing values, inconsistencies, and outliers — all of which are processed in the pipeline.

☁️ 5. AWS S3 Data Access
Example S3 Path
s3://your-bucket-name/airbnb/raw_data/listings.csv

Load Data from S3
import pandas as pd
df = pd.read_csv("s3://your-bucket-name/airbnb/raw_data/listings.csv")

🧹 6. Data Cleaning
Issues handled:

Missing values (reviews_per_month, last_review)

Dropping duplicates

Converting data types

Treating outliers

Capping price with price_capped

Example Code
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df.drop_duplicates(inplace=True)
df['price_capped'] = df['price'].clip(0, 500)

📊 7. Exploratory Data Analysis (EDA)

EDA focused on price distribution, spatial patterns, and review behavior.

Plots Included:

Price Distribution

Boxplot: Price by Neighbourhood Group

Correlation Heatmap

Scatter Plot: Latitude–Longitude vs Price

Example Code
plt.hist(df['price_capped'], bins=50)
plt.title("Distribution of Capped Price")
plt.xlabel("Price")
plt.ylabel("Frequency")


🧠 8. Feature Engineering
Engineered Features:

amenities_list

amenities_count

dist_to_center_km

availability_indicator

price_capped

Example: Distance Feature
def approx_distance(lat, lon, c_lat=40.7128, c_lon=-74.0060):
    return ((lat - c_lat)**2 + (lon - c_lon)**2)**0.5 * 111

df['dist_to_center_km'] = approx_distance(df['latitude'], df['longitude'])

Example: Amenities Parsing
def parse_amenities(a):
    if pd.isna(a): 
        return []
    s = a.replace("'", "").replace('"', "").strip("[]")
    return [i.strip() for i in s.split(",") if i.strip()]

df["amenities_list"] = df["amenities"].map(parse_amenities)
df["amenities_count"] = df["amenities_list"].apply(len)

🤖 9. Model Development

Trained models include:

Linear Regression

Random Forest Regressor

XGBoost Regressor

Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Preprocessing Pipeline

One-hot encoding of categorical variables

Scaling numeric values

Combining into a full sklearn Pipeline

📈 10. MLflow Experiment Tracking

Every model run logs:

Parameters Logged

Model name

Hyperparameters

Feature list

Metrics Logged

RMSE

MAE

R² Score

Artifacts Logged

Plots

Model pickle file

Preprocessing pipeline

Example MLflow Code
with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.sklearn.log_model(model, "model")

MLflow UI (Insert Your Screenshots Here)
assets/mlflow_experiment.png
assets/mlflow_metrics.png
assets/mlflow_registry.png

📦 11. Model Registry

The best model was:

Registered in MLflow Model Registry

Assigned a unique version

Tagged with metadata

Ready for deployment

📉 12. Results & Insights
Key Findings

Distance to the city center strongly affects price

Neighbourhood groups show significant variation

XGBoost & Random Forest outperform Linear Regression

Feature engineering improves model accuracy




▶️ 13. How to Run the Project
Step 1 — Install Dependencies
pip install -r requirements.txt

Step 2 — Start MLflow UI
mlflow ui

Step 3 — Run Training
python src/train_model.py

