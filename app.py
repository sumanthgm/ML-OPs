import joblib

import pandas as pd

from flask import request, Flask, jsonify

maintenance_predictor = joblib.load('models/model-v1.joblib')

maintenance_predictor_api = Flask("Equipment Maintenance Predictor")


@maintenance_predictor_api.get('/')
def home():
    return 'Welcome to the equipment maintenance predictor!'


@maintenance_predictor_api.post('/v1/equipment')
def predict_failure():

    equipment_state = request.get_json()

    sample = {
        'Air temperature [K]': equipment_state['Air temperature [K]'],
        'Process temperature [K]': equipment_state['Process temperature [K]'],
        'Rotational speed [rpm]': equipment_state['Rotational speed [rpm]'],
        'Torque [Nm]': equipment_state['Torque [Nm]'],
        'Tool wear [min]': equipment_state['Tool wear [min]'],
        'Type': equipment_state['Type']
    }

    data_point = pd.DataFrame([sample])

    prediction = maintenance_predictor.predict(data_point).tolist()
    
    if prediction == 1:
        prediction_label = 'yes'
    else:
        prediction_label = 'no' 

    return jsonify({'Failure expected?': prediction_label})


@maintenance_predictor_api.post('/v1/equipmentbatch')
def predict_failure_onbatch():

    equipment_states_file = request.files['file']

    input_data = pd.read_csv(equipment_states_file)

    predictions = maintenance_predictor.predict(input_data).tolist()
    
    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    maintenance_predictor_api.run(debug=True, host='0.0.0.0', port=8000)