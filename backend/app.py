from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
CORS(app) 


model = joblib.load('student_performance_model.pkl')


poly = PolynomialFeatures(degree=3)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
   
    hours = float(data['study_hours'])
    attendance = float(data['attendance'])
    past = float(data['past_scores'])
    extra = 1 if data['extracurricular'] == 'Yes' else 0

   
    features = np.array([[hours, attendance, past, extra]])
    
    # Using the same polynomial transformation that was fitted during training
    features_poly = poly.transform(features)
    
    # Prediction
    prediction = model.predict(features_poly)
    
    return jsonify({'score': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)