import pickle
import json
import numpy as np
import pandas as pd


def predict_price(params):

    x = np.zeros((1,(len(data_columns))))

    x[0,0] = params.get('bed') 
    x[0,1] = params.get('bath') 
    x[0,2] = params.get('acre_lot') 
    x[0,3] = params.get('zip_code') 
    x[0,4] = params.get('house_size') 

    status_col = 'status_' + params.get('status')
    x[0,data_columns.index(status_col)] = 1

    state_col = 'state_' + params.get('state')
    x[0,data_columns.index(state_col)] = 1

    city_col = 'city_' + params.get('city')
    x[0,data_columns.index(city_col)] = 1

    x_df = pd.DataFrame(x, columns=data_columns)
    y_pred = np.expm1(model.predict(x_df)) 

    return y_pred


def load_artifacts(model_version):
    print('loading saved artifacts')
    global data_columns
    global model

    model_path = f'../ML Model/ml_model_v{model_version}.pickle'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open('../ML Model/data_columns.json', 'r') as f:
        data_columns = json.load(f)['data_columns']    


if __name__ == '__main__':  
    model_version = 1  
    load_artifacts(model_version)  
    params = {
    'bed': 3,
    'bath': 2,
    'acre_lot': 2,
    'zip_code' : 90214,
    'house_size': 1500,
    'status': 'for_sale',
    'state': 'New York',
    'city': 'Dover'
    }
    y_pred = predict_price(params)
    print(f'Price Prediction: {y_pred[0] :0.2f}')

    