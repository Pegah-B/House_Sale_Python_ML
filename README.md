# Real Estate Price Prediction Model: EDA, Machine Learning, and Deployment 

## Overview
This project aims to predict house prices using machine learning techniques. The dataset used in this analysis contains real estate listings in the US broken by State and zip code, downloaded from https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

The project consists of three main parts:

1- Exploratory Data Analysis (EDA), Preprocessing, and Data Visualization 

2- Model Training: A machine learning pipeline that preprocesses the dataset and builds prediction models

3- Model Deployment: A web application built with Flask to serve the prediction models


## Files

1. **Data and Notebooks**
   
 &nbsp;&nbsp; Contains the raw dataset (```realtor-data.csv```), preprocessed data (```Preprocessed_Data.csv```), and Jupyter notebooks for the exploratory data analysis and preprocessing:

- ```House_Sales_EDA.ipynb```: Explores the dataset structure, features, and distributions, handles missing values and outliers, and visualizes relationships between features.
- ```House_Sale_Preprocessing.ipynb```: Preprocesses the data by handling missing values, outliers, and scaling features, then outputs a cleaned dataset (Preprocessed_Data.csv).
- ```House_Sales_ML.ipynb```: Trains two machine learning models (Random Forest and Lasso Regression) on the preprocessed data and evaluates their performance using relevant metrics.

2. **ML Model**
   
 &nbsp;&nbsp; Contains the pre-trained machine learning models and related metadata.
```model_training.py```: Script to train and save the models.
```ml_model_v1.pickle```: Random Forest Regressor (Version 1).
```ml_model_v2.pickle```: XGBoost Regressor (Version 2).
```data_columns.json```: JSON file listing the feature columns used during model training.
```state_city.json```: JSON file listing state and city info


3. **Model Deployment**
   
 &nbsp;&nbsp; Contains the Flask web application that serves the trained machine learning models.
```app.py```: Main Flask application script that handles user input and serves predictions.
```util.py```: Contains helper functions for loading models and making predictions.
```templates/index.html```: HTML page for user interaction, allowing input of house features for price prediction.

## 
To run the project locally:

1. Clone the project repository: ```git clone <repository-url>``` and ```cd real_estate_price_prediction```.
2. Install dependencies: ```pip install -r requirements.txt```.
3. Explore data and run Jupyter Notebooks (EDA, Preprocessing, and Model Training): Navigate to the "Data and Notbooks" folder and run ```jupyter notebook_name.ipynb```.
4. Model Training: Navigate to the "ML Model folder" and run ```model_training.py``` to create two trained model versions ```ml_model_v1.pickle``` (Random forest) and ```ml_model_v2.pickle``` (XGBoost). 
5. Run the Flask Web Application for Model Deployment: Navigate to the "Model Deployment" folder and start the Flask application by ```python app.py```.
6. Access the web app: By default, the app runs on ```http://localhost:8000```. Open your web browser and visit ```http://localhost:8000```.


## Requirements
- Python
- Jupyter Notebooks
- Dependencies: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, Flask 
