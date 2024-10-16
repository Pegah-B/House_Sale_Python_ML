import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('Preprocessed_Data.csv')
#one-hot encoding
df = pd.get_dummies(df, columns=['status'])
df = pd.get_dummies(df, columns=['city'])
df = pd.get_dummies(df, columns=['state'])
#Features and Target variable
X = df.drop(['price'], axis=1)
y = df['price']

#Save Columns Names as JSON
data_columns = [col for col in X.columns]
import json
columns = {'data_columns' : data_columns}
with open ('data_columns.json' , 'w') as f:
    json.dump(columns , f)

#Group Cities by State and Save as JSON
state_city_dict = df.groupby('state')['city'].apply(lambda x: list(set(x))).to_dict()
import json
with open('state_city.json', 'w') as json_file:
    json.dump(state_city_dict, json_file)
    
#Split the Dataset for Training and Testing 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Train Models
ml_model = {
    'RF'  : RandomForestRegressor(),
    'XGB' : XGBRegressor()
}
# Hyperparameters selected based on GridSearch done in the notebook file "House_Sales_ML.ipynb"
hyper_params = {
    'RF': {
        'bootstrap': True, 
        'ccp_alpha': 0.0, 
        'criterion': 'squared_error',
        'max_depth': None, 
        'max_features': 1.0,
        'max_leaf_nodes': None,
        'max_samples': None, 
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 1, 
        'min_samples_split': 10, 
        'min_weight_fraction_leaf': 0.0, 
        'n_estimators': 200, 
        'n_jobs': None, 
        'oob_score': False, 
        'random_state': None, 
        'verbose': 0, 
        'warm_start': False
    },
    'XGB': {
        'objective': 'reg:squarederror',
        'base_score': None,
        'booster': None, 
        'callbacks': None, 
        'colsample_bylevel': None,
        'colsample_bynode': None, 
        'colsample_bytree': None, 
        'device': None, 
        'early_stopping_rounds': None,
        'enable_categorical': False,
        'eval_metric': None, 
        'feature_types': None, 
        'gamma': 0.1,
        'grow_policy': None,
        'importance_type': None, 
        'interaction_constraints': None,
        'learning_rate': 0.5, 
        'max_bin': None,
        'max_cat_threshold': None,
        'max_cat_to_onehot': None, 
        'max_delta_step': None, 
        'max_depth': 7, 
        'max_leaves': None,
        'min_child_weight': None,
        'missing': np.nan, 
        'monotone_constraints': None,
        'multi_strategy': None,
        'n_estimators': 200,
        'n_jobs': None,
        'num_parallel_tree': None, 
        'random_state': None,
        'reg_alpha': None,
        'reg_lambda': None,
        'sampling_method': None,
        'scale_pos_weight': None,
        'subsample': None, 
        'tree_method': None, 
        'validate_parameters': None,
        'verbosity': None
    }
}
#Save ML Models
model_filenames = {
    'RF': 'ml_model_v1.pickle',
    'XGB': 'ml_model_v2.pickle'
}
for model_name, model in ml_model.items():
    print(f"Training started for {model_name} model...")
    model.set_params(**hyper_params[model_name])
    model.fit(x_train, y_train)

    with open(model_filenames[model_name], 'wb') as file:
        pickle.dump(model, file)

    print(f"{model_name} model trained and saved as {model_filenames[model_name]}.")
