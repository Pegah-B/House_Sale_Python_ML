from flask import Flask, request, jsonify, render_template
import json
import util

app = Flask(__name__)

with open('../Ml Model/state_city.json') as f:
    state_city_data = json.load(f)

@app.route('/')
def home():
    states = list(state_city_data.keys())
    return render_template('index.html', states=states, state_city_data=state_city_data)

@app.route('/get_cities/<state>', methods=['GET'])
def get_cities(state):
    cities = state_city_data.get(state, [])
    return jsonify(cities)

@app.route('/real_estate_price_prediction', methods = ['POST'])
def predict_price():
    
    params = {
        'status': request.form['status'],
        'bed': int(request.form['bed']),
        'bath': int(request.form['bath']),
        'acre_lot': float(request.form['acre_lot']),   
        'state': request.form['state'], 
        'city': request.form['city'],
        'zip_code': int(request.form['zip_code']),
        'house_size': float(request.form['house_size'])
         }
    
    y_pred = util.predict_price(params)
    prediction = f'Price Prediction: {y_pred[0] :0.2f}'    

    return render_template('index.html', params=params, prediction=prediction)


if __name__ == '__main__' : 
    print("Starting Python Flask Server")
    model_version = 1
    util.load_artifacts(model_version)
    app.run(port = 8000, debug=True)

    