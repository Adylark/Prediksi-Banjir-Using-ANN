from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


model = load_model('mymodel.h5')

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the request
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])
    feature5 = float(request.form['feature5'])
    feature6 = float(request.form['feature6'])
    feature7 = float(request.form['feature7'])

    # Preprocess the input features
    # Make predictions using the loaded model
    prediction = model.predict([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]])

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)