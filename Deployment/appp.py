from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('flood_prediction_model.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Function to preprocess input data
def preprocess_input(data):
    scaled_data = scaler.transform([data])
    return scaled_data

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = [float(x) for x in request.form.values()]

    # Preprocess input data
    scaled_data = preprocess_input(data)

    # Make prediction
    prediction_prob = model.predict([scaled_data])[0]
    predicted_class = np.argmax(prediction_prob)

    # Render template with prediction result
    return render_template('index.html', prediction=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)
