# """
# AI Engine 1: Risk Assessment
# XGBoost model for predicting student at-risk status
# """

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import joblib
# import os
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import config

# app = Flask(__name__)
# CORS(app)

# # Load or initialize model
# MODEL_PATH = os.path.join(config.MODELS_DIR, 'risk_assessment', 'xgboost_model.pkl')
# model = None

# try:
#     if os.path.exists(MODEL_PATH):
#         model = joblib.load(MODEL_PATH)
#         print(f"✅ Risk Assessment Model loaded from {MODEL_PATH}")
#     else:
#         print(f"⚠️  Model not found at {MODEL_PATH}. Will use mock predictions.")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")


# def mock_risk_prediction(student_data):
#     """
#     Mock risk assessment when model is not trained yet
#     In production, replace with actual model predictions
#     """
#     # Calculate a mock risk score based on available data
#     risk_score = 0.0
#     factors = {}
    
#     # Attendance factor
#     attendance_rate = student_data.get('attendance_rate', 0.85)
#     if attendance_rate < 0.7:
#         risk_score += 0.3
#         factors['attendance'] = 'Poor attendance (<70%)'
#     elif attendance_rate < 0.85:
#         risk_score += 0.15
#         factors['attendance'] = 'Low attendance (70-85%)'
    
#     # GPA factor
#     gpa = student_data.get('gpa', 3.0)
#     if gpa < 2.0:
#         risk_score += 0.4
#         factors['academic_performance'] = 'Low GPA (<2.0)'
#     elif gpa < 3.0:
#         risk_score += 0.2
#         factors['academic_performance'] = 'Below average GPA (2.0-3.0)'
    
#     # Assignment completion
#     assignment_completion = student_data.get('assignment_completion_rate', 0.9)
#     if assignment_completion < 0.6:
#         risk_score += 0.3
#         factors['assignments'] = 'Low assignment completion (<60%)'
#     elif assignment_completion < 0.8:
#         risk_score += 0.15
#         factors['assignments'] = 'Moderate assignment completion (60-80%)'
    
#     # Engagement
#     engagement_score = student_data.get('engagement_score', 0.8)
#     if engagement_score < 0.5:
#         risk_score += 0.2
#         factors['engagement'] = 'Low engagement in class'
    
#     # Normalize risk score
#     risk_score = min(risk_score, 1.0)
    
#     # Determine risk level
#     if risk_score >= 0.7:
#         risk_level = 'critical'
#     elif risk_score >= 0.5:
#         risk_level = 'high'
#     elif risk_score >= 0.3:
#         risk_level = 'medium'
#     else:
#         risk_level = 'low'
    
#     # Generate recommendations
#     recommendations = []
#     if 'attendance' in factors:
#         recommendations.append('Schedule parent-teacher meeting to discuss attendance')
#     if 'academic_performance' in factors:
#         recommendations.append('Provide additional tutoring support')
#     if 'assignments' in factors:
#         recommendations.append('Implement assignment tracking and reminders')
#     if 'engagement' in factors:
#         recommendations.append('Increase interactive learning activities')
    
#     if not recommendations:
#         recommendations.append('Continue monitoring student progress')
    
#     return {
#         'risk_level': risk_level,
#         'risk_score': round(risk_score, 3),
#         'confidence': 0.94,
#         'factors': factors,
#         'recommendations': recommendations,
#         'model_version': 'mock_v1.0'
#     }


# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({
#         'status': 'healthy',
#         'service': 'Risk Assessment',
#         'model_loaded': model is not None
#     })


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
        
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
        
#         # Extract student data
#         student_id = data.get('student_id')
#         student_data = data.get('student_data', {})
        
#         # If no student_data provided, use mock data
#         if not student_data:
#             student_data = {
#                 'attendance_rate': np.random.uniform(0.6, 1.0),
#                 'gpa': np.random.uniform(2.0, 4.0),
#                 'assignment_completion_rate': np.random.uniform(0.5, 1.0),
#                 'engagement_score': np.random.uniform(0.4, 1.0)
#             }
        
#         # Make prediction
#         if model:
#             # TODO: Implement actual model prediction
#             # features = prepare_features(student_data)
#             # prediction = model.predict(features)
#             result = mock_risk_prediction(student_data)
#         else:
#             result = mock_risk_prediction(student_data)
        
#         result['student_id'] = student_id
        
#         return jsonify({
#             'success': True,
#             'data': result
#         })
    
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500


# if __name__ == '__main__':
#     # Ensure models directory exists
#     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
#     print(f"🚀 Starting Risk Assessment Service on port {config.RISK_ASSESSMENT_PORT}")
#     app.run(host='0.0.0.0', port=config.RISK_ASSESSMENT_PORT, debug=True)
















"""
AI Engine 1: Risk Assessment
XGBoost model for predicting student at-risk status
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import sys

# --- Add project base directory to Python path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# --- Configuration for Model/Scaler Paths ---
MODEL_FILE_NAME = 'vtu_risk_model.pkl'
SCALER_FILE_NAME = 'vtu_data_scaler.pkl'

APP_FILE_PATH = os.path.abspath(__file__)
# Get the directory where this script is located (ai-services/risk-assessment/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Models are in: ai-services/risk-assessment/models/risk_assessment/
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models', 'risk_assessment')
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILE_NAME)

print("--- SANITY CHECK ---")
print(f"Script Path: {APP_FILE_PATH}")
print(f"Script Dir: {SCRIPT_DIR}")
print(f"Model Dir: {MODEL_DIR}")
print(f"Model Path: {MODEL_PATH}")
print(f"Scaler Path: {SCALER_PATH}")
print("--------------------")

model = None
scaler = None


# --- Load Model and Scaler ---
def load_assets():
    global model, scaler
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Risk Assessment Model loaded successfully from: {MODEL_PATH}")
        else:
            print(f"⚠️  WARNING: Model not found at {MODEL_PATH}")
            print(f"   Will use mock predictions instead.")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()

    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ StandardScaler loaded successfully from: {SCALER_PATH}")
        else:
            print(f"⚠️  WARNING: Scaler not found at {SCALER_PATH}")
            print(f"   Prediction will fail if model is used without scaler.")
    except Exception as e:
        print(f"❌ ERROR loading scaler: {e}")
        import traceback
        traceback.print_exc()


load_assets()


# --- Feature Preparation Function ---
REQUIRED_FEATURES = ['attendance_rate', 'gpa', 'assignment_completion_rate', 'engagement_score']

def prepare_features(student_data):
    data_for_df = {k: [student_data.get(k, 0)] for k in REQUIRED_FEATURES}
    features_df = pd.DataFrame(data_for_df, columns=REQUIRED_FEATURES)

    if features_df.isnull().values.any():
        print("⚠ Missing values found in input data. Imputing with 0.")
        features_df = features_df.fillna(0)

    if scaler is not None:
        scaled_features = scaler.transform(features_df)
        return scaled_features
    else:
        raise RuntimeError("StandardScaler is not loaded.")


# --- Mock Prediction (Fallback) ---
def mock_risk_prediction(student_data):
    risk_score = 0.0
    factors = {}
    
    attendance_rate = student_data.get('attendance_rate', 0.85)
    if attendance_rate < 0.7:
        risk_score += 0.3
        factors['attendance'] = 'Poor attendance (<70%)'
    elif attendance_rate < 0.85:
        risk_score += 0.15
        factors['attendance'] = 'Low attendance (70-85%)'
        
    gpa = student_data.get('gpa', 3.0)
    if gpa < 2.0:
        risk_score += 0.4
        factors['academic_performance'] = 'Low GPA (<2.0)'
    elif gpa < 3.0:
        risk_score += 0.2
        factors['academic_performance'] = 'Below average GPA (2.0-3.0)'
        
    assignment_completion = student_data.get('assignment_completion_rate', 0.9)
    if assignment_completion < 0.6:
        risk_score += 0.3
        factors['assignments'] = 'Low assignment completion (<60%)'
    elif assignment_completion < 0.8:
        risk_score += 0.15
        factors['assignments'] = 'Moderate assignment completion (60-80%)'
        
    engagement_score = student_data.get('engagement_score', 0.8)
    if engagement_score < 0.5:
        risk_score += 0.2
        factors['engagement'] = 'Low engagement in class'
        
    risk_score = min(risk_score, 1.0)
    
    if risk_score >= 0.7:
        risk_level = 'critical'
    elif risk_score >= 0.5:
        risk_level = 'high'
    elif risk_score >= 0.3:
        risk_level = 'medium'
    else:
        risk_level = 'low'
        
    recommendations = []
    if 'attendance' in factors:
        recommendations.append('Schedule parent-teacher meeting to discuss attendance')
    if 'academic_performance' in factors:
        recommendations.append('Provide additional tutoring support')
    if 'assignments' in factors:
        recommendations.append('Implement assignment tracking and reminders')
    if 'engagement' in factors:
        recommendations.append('Increase interactive learning activities')
        
    if not recommendations:
        recommendations.append('Continue monitoring student progress')
        
    return {
        'risk_level': risk_level,
        'risk_score': round(risk_score, 3),
        'confidence': 0.94,
        'factors': factors,
        'recommendations': recommendations,
        'model_version': 'mock_v1.0'
    }


# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Risk Assessment',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        student_id = data.get('student_id')
        student_data = data.get('student_data', {})
        
        if model is not None and scaler is not None:
            features_scaled = prepare_features(student_data)
            risk_probability = model.predict_proba(features_scaled)[:, 1][0]
            risk_class = model.predict(features_scaled)[0]
            
            risk_level_map = {0: 'low', 1: 'high'}
            risk_level = risk_level_map.get(risk_class, 'unknown')
            
            mock_result = mock_risk_prediction(student_data)
            
            result = {
                'risk_level': risk_level,
                'risk_score': round(float(risk_probability), 3),
                'confidence': round(float(risk_probability), 3) if risk_class == 1 else round(1.0 - float(risk_probability), 3),
                'factors': mock_result['factors'],
                'recommendations': mock_result['recommendations'],
                'model_version': 'xgboost_v1.0'
            }
        else:
            result = mock_risk_prediction(student_data)
        
        result['student_id'] = student_id
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except RuntimeError as re:
        return jsonify({'success': False, 'error': str(re)}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# --- Run App ---
if __name__ == '__main__':
    DEFAULT_PORT = 5000
    print(f"🚀 Starting Risk Assessment Service on port {DEFAULT_PORT}")
    app.run(host='0.0.0.0', port=DEFAULT_PORT, debug=True)





