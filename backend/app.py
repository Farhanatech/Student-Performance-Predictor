from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
CORS(app) # Taake frontend error na de

# 1. Model load karein
model = joblib.load('student_performance_model.pkl')

# 2. Polynomial transformation (wahi jo training mein thi)
poly = PolynomialFeatures(degree=3)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Data nikalna frontend se
    hours = float(data['study_hours'])
    attendance = float(data['attendance'])
    past = float(data['past_scores'])
    extra = 1 if data['extracurricular'] == 'Yes' else 0

    # Input ko format karna
    features = np.array([[hours, attendance, past, extra]])
    
    # Polynomial mein convert karna (Zaroori step!)
    # Note: Asal mein humein training wala poly object save karna chahiye tha, 
    # par abhi hum yahan fit_transform use kar rahe hain demo ke liye.
    features_poly = poly.fit_transform(features)
    
    # Prediction
    prediction = model.predict(features_poly)
    
    return jsonify({'score': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)