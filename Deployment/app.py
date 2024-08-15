from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load model dan scaler
model = tf.keras.models.load_model('flood_prediction_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = [float(data['Tn']), float(data['Tx']), float(data['Tavg']), float(data['RH_avg']), float(data['RR']), float(data['ss']), float(data['ff_x'])]
    features = np.array(features).reshape(1, -1)
    
    # Normalisasi fitur
    scaled_features = scaler.transform(features)
    
    prediction_prob = model.predict(scaled_features)
    prediction = (prediction_prob > 0.5).astype(int)
    
    output = 'Flood' if prediction[0][0] == 1 else 'No Flood'
    
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
