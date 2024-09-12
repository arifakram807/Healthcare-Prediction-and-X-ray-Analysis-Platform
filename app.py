from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from werkzeug.utils import secure_filename
import os
from PIL import Image

app = Flask(__name__)

# Paths
MODEL_PATH = 'D:/AI_Training/chest_xray_data/models/xray_model.h5'
UPLOAD_FOLDER = 'D:/AI_Training/chest_xray_data/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model for length of stay prediction
with open('models/model.pkl', 'rb') as f:
    length_of_stay_model = pickle.load(f)

# Load the column names for length of stay model
with open('models/columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# Load the trained model for X-ray prediction
xray_model = tf.keras.models.load_model(MODEL_PATH)

# Home route to display the main page
@app.route('/')
def home():
    return render_template('home.html')

# Prediction form route for length of stay
@app.route('/predict-form')
def predict_form():
    return render_template('index.html')

# Prediction route for length of stay
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract health condition flags
    health_flags = ['dialysisrenalendstage', 'asthma', 'irondef', 'pneum',
                    'substancedependence', 'psychologicaldisordermajor', 'depress',
                    'psychother', 'fibrosisandother', 'malnutrition', 'hemo']

    total_issues = sum(int(data.get(flag, 0)) for flag in health_flags)
    health_risk_weights = {
        'dialysisrenalendstage': 3, 'asthma': 1, 'irondef': 1, 'pneum': 2,
        'substancedependence': 2, 'psychologicaldisordermajor': 2, 'depress': 1,
        'psychother': 1, 'fibrosisandother': 2, 'malnutrition': 2, 'hemo': 1
    }
    health_risk_score = sum(health_risk_weights[flag] * int(data.get(flag, 0)) for flag in health_flags)

    # Extract other input features
    input_data = {
        'rcount': data['rcount'],
        'hematocrit': data['hematocrit'],
        'neutrophils': data.get('neutrophils', 0),
        'sodium': data['sodium'],
        'glucose': data['glucose'],
        'bloodureanitro': data.get('bloodureanitro', 0),
        'creatinine': data['creatinine'],
        'bmi': data['bmi'],
        'pulse': data['pulse'],
        'respiration': data['respiration'],
        'total_issues': total_issues,
        'bmi_glucose': float(data['bmi']) * float(data['glucose']),
        'bmi_creatinine': float(data['bmi']) * float(data['creatinine']),
        'health_risk_score': health_risk_score,
        'gender_M': int(data.get('gender_M', 0)),
        'gender_F': 1 - int(data.get('gender_M', 0))
    }

    input_df = pd.DataFrame([input_data]).reindex(columns=model_columns, fill_value=0)
    prediction = length_of_stay_model.predict(input_df.values)[0]
    rounded_prediction = round(prediction, 6)

    return render_template('result.html', predicted_length_of_stay=rounded_prediction)

# Route for X-ray prediction page
@app.route('/xray', methods=['GET'])
def xray_page():
    return render_template('predict_xray.html')

# Route to handle X-ray prediction
@app.route('/predict_xray', methods=['POST'])
def predict_xray():
    if 'xrayImage' not in request.files:
        return redirect(request.url)

    # Save uploaded X-ray image
    file = request.files['xrayImage']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess the image
    image = Image.open(file_path).convert('L')
    image = image.resize((300, 300))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=[0, -1])

    # Make prediction using the X-ray model
    prediction = xray_model.predict(image)

    # Map the predictions to the disease labels
    disease_labels = ['No Finding', 'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
                      'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural Thickening', 'Cardiomegaly',
                      'Nodule', 'Mass', 'Hernia']

    predicted_diseases = [disease_labels[i] for i in range(len(disease_labels)) if prediction[0][i] > 0.4]  # Adjust threshold if necessary
    if not predicted_diseases:
        predicted_diseases = ["No findings detected."]

    return render_template('xray_result.html', filename=filename, predictions=predicted_diseases)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
