from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open('insurance_cost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():

    # Return prediction
    return jsonify({'predicted_charges': 'Irene Busah'})

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.json
    features = np.array([[
        data['age'], 
        data['sex'],
        data['bmi'], 
        data['children'], 
        data['smoker'],
        data['region']
    ]])

    # Make prediction
    prediction = model.predict(features)

    # Return prediction
    return jsonify({'predicted_charges': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
