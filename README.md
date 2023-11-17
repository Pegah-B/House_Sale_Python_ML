## Overview
This project aims to predict house prices using machine learning techniques. The dataset used in this analysis contains real estate listings in the US broken by State and zip code, downloaded from https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

## Files
1. House_Sales_EDA.ipynb
   
   &nbsp;&nbsp; Explores the dataset structure, features, and distributions. Handles missing values and outliers, and performs necessary data transformations. Utilizes data visualization for exploring correlations and relationships between features. Outputs the cleansed and preprocessed data to "Cleansed_Data.csv."

3. House_Sales_ML.ipynb
   
   &nbsp;&nbsp;Uses "Cleansed_Data.csv" for training machine learning models using random forest and lasso regression models for house price prediction. Including various metrics for evaluation. 

5. realtor-data.csv
   
   &nbsp;&nbsp; Dataset used in this project, downloaded from Kaggle.

7. Cleansed_Data.csv.csv
   
   &nbsp;&nbsp; Cleaned and preprocessed data from "House_Sales_EDA.ipynb".

9. requirements.txt
    
   &nbsp;&nbsp; Includes dependencies for the project. To install dependencies, use:
     ```
     pip install -r requirements.txt
     ```


## Requirements
- Python
- Jupyter Notebooks
- Dependencies: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, etc.
