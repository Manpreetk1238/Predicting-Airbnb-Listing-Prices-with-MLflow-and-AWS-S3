** Predicting Airbnb Listing Prices with MLflow & AWS S3**
** 1. Project Overview**

This project builds a complete, reproducible machine learning pipeline that predicts Airbnb listing prices using AWS S3 for cloud-based data storage and MLflow for experiment tracking.
The workflow includes data loading, cleaning, exploratory analysis, feature engineering, model training, experiment tracking, and registering the best model.

** 2. Business Problem**

Airbnb listing prices vary widely even among comparable properties.
The goal of this project is to help StayWise generate data-driven price recommendations by developing a regression model that predicts nightly prices based on key listing attributes, such as:

Location

Room type

Reviews and host activity

Amenities

Distance to key city landmarks

This model supports hosts in setting competitive prices while improving overall platform performance.

** 3. Project Architecture**
AWS S3 (Raw Dataset)
        ↓
    Data Loading
        ↓
Data Cleaning & Processing
        ↓
   Exploratory Analysis
        ↓
  Feature Engineering
        ↓
  Model Development
        ↓
 MLflow Experiment Tracking
        ↓
     Model Evaluation
        ↓
 MLflow Model Registry

** 4. Dataset Description**

The dataset contains information about Airbnb listings, including:

Host details

Neighbourhood group and coordinates

Room type and availability

Reviews and listing history

Engineered features such as capped price and distance to city center

The dataset required cleaning due to missing values, formatting inconsistencies, and outliers.

** 5. AWS S3 Data Access**

The raw CSV file is stored in an AWS S3 bucket and accessed directly using a cloud-compatible file system.
Using S3 ensures scalability, version control, and reproducibility across environments.

** 6. Data Cleaning**

Cleaning steps performed:

Handled missing values in review columns

Converted date columns

Removed duplicate listings

Applied price capping to reduce skewed distributions

Standardized formatting

** 7. Exploratory Data Analysis (EDA)**

EDA focused on understanding:

Price distribution

Neighbourhood-based price variation

Correlation between numerical features

Effects of room types and reviews

Geographical pricing patterns

These insights guided feature engineering and model selection.

** 8. Feature Engineering**

To enhance predictive power, several new features were created, including:

Parsed amenities list

Count of total amenities

Approximate distance to city center

Availability metrics

Capped price values

These engineered features improved model interpretability and accuracy.

** 9. Model Development**

Two regression models were developed and compared:

Linear Regression – baseline model

Random Forest Regressor – ensemble model with better performance

A train/test split was used, and a complete preprocessing pipeline handled encoding and scaling.

** 10. MLflow Experiment Tracking**

MLflow was used to track:

Logged Parameters

Model name

Feature sets

Training configurations

Logged Metrics

RMSE

MAE

R²

Artifacts

Trained model files

Preprocessing pipeline

Metrics summary

Insert MLflow Screenshots Here

MLflow experiment runs

Metrics comparison

Registered model page

**📦 11. Model Registry**

The best-performing model was:

Versioned and stored in MLflow Model Registry

Assigned a unique model version

Ready for deployment with reproducible configurations

**📉 12. Results & Key Insights**

Key observations include:

Distance to city center is a strong predictor of price

Neighbourhood groups significantly influence pricing

Random Forest outperformed Linear Regression

Feature engineering improved model accuracy

MLflow streamlined experiment comparison and tracking

