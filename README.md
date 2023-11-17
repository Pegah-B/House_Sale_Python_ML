## Overview
This project aims to predict house prices using machine learning techniques. The dataset used in this analysis contains real estate listings in the US broken by State and zip code, downloaded from https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

## Files
1. House_Sales_EDA.ipynb
   Explores the dataset structure, features, and distributions. Handles missing values and outliers, and performs necessary data transformations. Utilizes data visualization for exploring correlations and relationships between features. Outputs the cleansed and preprocessed data to "Cleansed_Data.csv."

2. House_Sales_ML.ipynb
   Uses "Cleansed_Data.csv" for training machine learning models using random forest and lasso regression models for house price prediction. Including various metrics for evaluation. 

4. realtor-data.csv
   Dataset used in this project, downloaded from Kaggle.

5. Preprocessed_Data.csv
   Cleaned and preprocessed data from House_Sales_EDA.ipynb.

6. requirements.txt
   Includes dependencies for the project. To install dependencies, use:
     ```
     pip install -r requirements.txt
     ```

## Dataset Information
The dataset used in this analysis contains real estate listings in the US broken down by State and zip code, downloaded from [Kaggle](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset).

## Requirements
- Python
- Jupyter Notebooks
- Dependencies: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, etc.
