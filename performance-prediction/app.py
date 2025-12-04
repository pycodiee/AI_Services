# """
# AI Engine 6: Performance Prediction
# Time-series forecasting using Facebook Prophet for GPA prediction
# """

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# import os
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import config

# app = Flask(__name__)
# CORS(app)

# # Try to import Prophet
# try:
#     from prophet import Prophet
#     PROPHET_AVAILABLE = True
# except Exception as e:
#     print(f"⚠️  Prophet not available: {e}. Using mock predictions.")
#     PROPHET_AVAILABLE = False


# def generate_mock_historical_data(student_id, current_gpa=3.6):
#     """Generate mock historical GPA data"""
#     # Generate 6 terms of historical data
#     terms = []
#     base_date = datetime.now() - timedelta(days=365*2)  # 2 years ago
    
#     # Simulate GPA progression
#     gpa_values = []
#     base_gpa = current_gpa - 0.8  # Start lower
    
#     for i in range(6):
#         # Add slight upward trend with some variation
#         gpa = base_gpa + (i * 0.15) + np.random.uniform(-0.1, 0.1)
#         gpa = min(max(gpa, 2.0), 4.0)  # Clamp between 2.0 and 4.0
#         gpa_values.append(round(gpa, 2))
        
#         term_date = base_date + timedelta(days=i*120)  # ~4 months per term
#         terms.append({
#             'ds': term_date,
#             'y': gpa
#         })
    
#     return pd.DataFrame(terms)


# def predict_with_prophet(historical_data, periods=4):
#     """Predict future GPA using Prophet"""
#     try:
#         # Initialize and fit Prophet model
#         model = Prophet(
#             yearly_seasonality=False,
#             weekly_seasonality=False,
#             daily_seasonality=False,
#             changepoint_prior_scale=0.05
#         )
        
#         model.fit(historical_data)
        
#         # Make future dataframe
#         future = model.make_future_dataframe(periods=periods, freq='120D')  # 120 days ~4 months per term
        
#         # Predict
#         forecast = model.predict(future)
        
#         # Extract predictions
#         predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
#         return predictions
#     except Exception as e:
#         print(f"Prophet prediction error: {e}")
#         return None


# def mock_prediction(current_gpa, periods=4):
#     """Mock prediction when Prophet is not available"""
#     predictions = []
#     base_date = datetime.now()
    
#     # Simple linear projection with slight improvement
#     improvement_per_term = 0.1
    
#     for i in range(1, periods + 1):
#         predicted_gpa = current_gpa + (i * improvement_per_term)
#         predicted_gpa = min(predicted_gpa, 4.0)  # Cap at 4.0
        
#         # Add confidence intervals
#         lower_bound = max(predicted_gpa - 0.2, 2.0)
#         upper_bound = min(predicted_gpa + 0.2, 4.0)
        
#         term_date = base_date + timedelta(days=i*120)
        
#         predictions.append({
#             'ds': term_date,
#             'yhat': predicted_gpa,
#             'yhat_lower': lower_bound,
#             'yhat_upper': upper_bound
#         })
    
#     return pd.DataFrame(predictions)


# def format_predictions(predictions_df, current_gpa):
#     """Format predictions for API response"""
#     result = []
    
#     # Add current term
#     result.append({
#         'term': 'Current',
#         'date': datetime.now().strftime('%Y-%m'),
#         'predicted_gpa': round(current_gpa, 2),
#         'lower_bound': round(current_gpa, 2),
#         'upper_bound': round(current_gpa, 2),
#         'is_actual': True
#     })
    
#     # Add future predictions
#     term_names = ['Next Term', 'Term +2', 'Term +3', 'Term +4', 'Term +5', 'Term +6']
    
#     for idx, row in predictions_df.iterrows():
#         if idx < len(term_names):
#             result.append({
#                 'term': term_names[idx],
#                 'date': row['ds'].strftime('%Y-%m') if hasattr(row['ds'], 'strftime') else str(row['ds'])[:7],
#                 'predicted_gpa': round(float(row['yhat']), 2),
#                 'lower_bound': round(float(row['yhat_lower']), 2),
#                 'upper_bound': round(float(row['yhat_upper']), 2),
#                 'is_actual': False
#             })
    
#     return result


# def generate_insights(predictions):
#     """Generate insights from predictions"""
#     if len(predictions) < 2:
#         return []
    
#     current = predictions[0]['predicted_gpa']
#     final = predictions[-1]['predicted_gpa']
#     change = final - current
    
#     insights = []
    
#     if change > 0.3:
#         insights.append({
#             'type': 'positive',
#             'message': f'Excellent trajectory! Your GPA is projected to improve by {abs(change):.2f} points.'
#         })
#     elif change > 0:
#         insights.append({
#             'type': 'positive',
#             'message': f'Steady progress! Your GPA is projected to improve by {abs(change):.2f} points.'
#         })
#     elif change < -0.2:
#         insights.append({
#             'type': 'warning',
#             'message': f'Attention needed! Projected decline of {abs(change):.2f} points. Consider seeking additional support.'
#         })
#     else:
#         insights.append({
#             'type': 'neutral',
#             'message': 'Your GPA is projected to remain stable. Keep up the consistent work!'
#         })
    
#     # Add recommendation based on final predicted GPA
#     if final >= 3.7:
#         insights.append({
#             'type': 'info',
#             'message': 'You\'re on track for honors! Maintain your current study habits.'
#         })
#     elif final < 2.5:
#         insights.append({
#             'type': 'warning',
#             'message': 'Academic intervention recommended. Connect with your advisor and teachers.'
#         })
    
#     return insights


# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({
#         'status': 'healthy',
#         'service': 'Performance Prediction',
#         'prophet_available': PROPHET_AVAILABLE
#     })


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
        
#         student_id = data.get('student_id', 'unknown')
#         periods = data.get('periods', 4)
#         current_gpa = data.get('current_gpa', 3.6)
        
#         # Generate or get historical data
#         historical_data = generate_mock_historical_data(student_id, current_gpa)
        
#         # Make predictions
#         if PROPHET_AVAILABLE:
#             predictions_df = predict_with_prophet(historical_data, periods)
#             if predictions_df is None:
#                 predictions_df = mock_prediction(current_gpa, periods)
#         else:
#             predictions_df = mock_prediction(current_gpa, periods)
        
#         # Format results
#         predictions = format_predictions(predictions_df, current_gpa)
#         insights = generate_insights(predictions)
        
#         return jsonify({
#             'success': True,
#             'data': {
#                 'student_id': student_id,
#                 'current_gpa': round(current_gpa, 2),
#                 'predictions': predictions,
#                 'insights': insights,
#                 'model_version': 'prophet_v1.0' if PROPHET_AVAILABLE else 'mock_v1.0',
#                 'confidence': 0.85
#             }
#         })
    
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500


# if __name__ == '__main__':
#     print(f"🚀 Starting Performance Prediction Service on port {config.PERFORMANCE_PREDICTION_PORT}")
#     app.run(host='0.0.0.0', port=config.PERFORMANCE_PREDICTION_PORT, debug=True)




























"""
AI Engine 6: Performance Prediction (Dynamic Prophet Model)

This server is designed to work with the PerformancePrediction.tsx frontend.
It does NOT use pre-trained .pkl files.
Instead, it receives a student's historical data list (from a CSV or manual entry)
and trains a new Prophet model on-the-fly for every API call.
"""

import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# --- Try to import Prophet ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    print("✅ Prophet library loaded successfully.")
except ImportError:
    PROPHET_AVAILABLE = False
    print("❌ WARNING: Prophet library not found. API will use mock predictions.")
    print("   Please install with: pip install prophet")

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- 3. Mock Prediction (Fallback) ---
def mock_prediction(historical_df, periods):
    """Fallback prediction if Prophet is not installed."""
    print("... using MOCK prediction function.")
    
    # Create historical data for chart
    historical_data = historical_df.rename(columns={'y': 'actualGPA'})
    historical_data['term'] = historical_data['ds'].dt.strftime('Term %Y-%m')
    
    # Generate mock forecast
    forecast_entries = []
    last_gpa = historical_df['y'].iloc[-1]
    last_date = historical_df['ds'].iloc[-1]
    
    for i in range(periods):
        future_date = last_date + pd.DateOffset(months=(i + 1) * 6) # Assume 6-month terms
        predicted_gpa = np.clip(last_gpa + (np.random.rand() - 0.4) * 0.1, 2.0, 4.0)
        lower = np.clip(predicted_gpa - 0.15 - (i * 0.05), 2.0, 4.0)
        upper = np.clip(predicted_gpa + 0.15 + (i * 0.05), 2.0, 4.0)
        
        forecast_entries.append({
            'term': f'Future Term {i+1}',
            'predictedGPA': round(predicted_gpa, 2),
            'lowerBound': round(lower, 2),
            'upperBound': round(upper, 2)
        })
        last_gpa = predicted_gpa # Trend continues

    # Combine historical and forecast
    combined_data = historical_data.to_dict('records') + forecast_entries
    return combined_data

# --- 2. Real Prophet Prediction ---
def train_and_predict_dynamic(historical_df, periods):
    """
    Trains a new Prophet model on the provided data and returns a forecast.
    """
    if not PROPHET_AVAILABLE:
        return mock_prediction(historical_df, periods)

    print(f"... training new Prophet model on {len(historical_df)} data points.")
    
    # Initialize and fit the model
    # We use interval_width=0.95 to get the 95% confidence interval
    model = Prophet(interval_width=0.95, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.add_seasonality(name='academic_year', period=365.25, fourier_order=1)
    
    model.fit(historical_df)
    
    # Create future dataframe to predict
    # 'freq=MS' stands for Month-Start frequency. We ask for 6-month terms (periods*6)
    future_df = model.make_future_dataframe(periods=periods, freq='6MS') 
    
    # Generate the forecast
    forecast_df = model.predict(future_df)
    
    print("... forecast complete.")
    
    # --- Format Data for Chart ---
    # Merge historical data with forecast data
    combined_df = forecast_df.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical_df.set_index('ds')['y'])
    combined_df = combined_df.reset_index()

    # Format for the frontend chart
    chart_data = []
    for _, row in combined_df.iterrows():
        entry = {
            'term': row['ds'].strftime('%b %Y'), # Format date like "Jan 2024"
            'actualGPA': row['y'] if pd.notna(row['y']) else None,
            'predictedGPA': round(row['yhat'], 2) if pd.notna(row['yhat']) else None,
            'lowerBound': round(row['yhat_lower'], 2) if pd.notna(row['yhat_lower']) else None,
            'upperBound': round(row['yhat_upper'], 2) if pd.notna(row['yhat_upper']) else None,
        }
        chart_data.append(entry)
        
    return chart_data


# --- 1. API Endpoints ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Performance Prediction (Dynamic Prophet)',
        'prophet_available': PROPHET_AVAILABLE
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get historical data list from the frontend
        # This matches the `gradeEntries` from the .tsx file
        historical_entries = data.get('historicalData') 
        if not historical_entries or len(historical_entries) < 2:
            return jsonify({'error': 'Not enough historical data. Need at least 2 entries.'}), 400
        
        # Debug: Log the received data structure
        print(f"[DEBUG] Received {len(historical_entries)} historical entries")
        print(f"[DEBUG] First entry structure: {historical_entries[0] if historical_entries else 'None'}")

        # Get number of periods to forecast
        periods = int(data.get('periods', 4)) # Default to 4 terms

        # --- Convert frontend data to DataFrame ---
        # The .tsx file sends { term: "Fall 2023", gpa: "3.2" }
        # Prophet needs { ds: "2023-09-01", y: 3.2 }
        
        data_list = []
        base_date = datetime.now()
        
        for idx, entry in enumerate(historical_entries):
            # Validate entry structure
            if not isinstance(entry, dict):
                return jsonify({'error': f'Invalid entry format at index {idx}. Expected dictionary.'}), 400
            
            term_str = entry.get('term', '').strip()
            gpa_str = entry.get('gpa', '').strip()
            
            # Validate that we have both term and gpa
            if not term_str:
                return jsonify({'error': f'Missing "term" field at index {idx}'}), 400
            if not gpa_str:
                return jsonify({'error': f'Missing "gpa" field at index {idx}'}), 400
            
            # Validate and convert GPA
            try:
                gpa_value = float(gpa_str)
                if gpa_value < 0 or gpa_value > 4.0:
                    return jsonify({'error': f'Invalid GPA value {gpa_value} at index {idx}. GPA must be between 0 and 4.0.'}), 400
            except (ValueError, TypeError) as e:
                return jsonify({'error': f'Invalid GPA value "{gpa_str}" at index {idx}. Could not convert to float. Error: {str(e)}'}), 400
            
            date_obj = None
            
            # Try multiple date parsing strategies
            try:
                # Strategy 1: Try parsing as ISO date (YYYY-MM-DD)
                date_obj = datetime.strptime(term_str, '%Y-%m-%d')
            except ValueError:
                try:
                    # Strategy 2: Try parsing as "Fall 2023" or "Spring 2024"
                    term_lower = term_str.lower()
                    if 'fall' in term_lower or 'autumn' in term_lower:
                        year = int(term_str.split()[-1])
                        date_obj = datetime(year, 9, 1)  # September for Fall
                    elif 'spring' in term_lower:
                        year = int(term_str.split()[-1])
                        date_obj = datetime(year, 1, 1)  # January for Spring
                    elif 'summer' in term_lower:
                        year = int(term_str.split()[-1])
                        date_obj = datetime(year, 6, 1)  # June for Summer
                    elif 'winter' in term_lower:
                        year = int(term_str.split()[-1])
                        date_obj = datetime(year, 12, 1)  # December for Winter
                    else:
                        # Strategy 3: Try to extract year and estimate
                        import re
                        year_match = re.search(r'\d{4}', term_str)
                        if year_match:
                            year = int(year_match.group())
                            # Estimate based on position (earlier entries are further back)
                            months_back = (len(historical_entries) - idx - 1) * 6
                            date_obj = datetime(year, 1, 1) - pd.DateOffset(months=months_back)
                        else:
                            raise ValueError("Could not parse year")
                except (ValueError, IndexError):
                    # Strategy 4: Fallback - estimate based on position
                    # Assume 6 months per term, working backwards from now
                    months_back = (len(historical_entries) - idx - 1) * 6
                    date_obj = base_date - pd.DateOffset(months=months_back)
                    print(f"Warning: Could not parse term '{term_str}', using estimated date: {date_obj.strftime('%Y-%m-%d')}")

            # Add to data list with validated values
            data_list.append({
                'ds': date_obj,
                'y': gpa_value
            })

        historical_df = pd.DataFrame(data_list)
        historical_df['ds'] = pd.to_datetime(historical_df['ds'])
        historical_df = historical_df.sort_values(by='ds')
        
        # --- Get Prediction ---
        forecast_data = train_and_predict_dynamic(historical_df, periods)
        
        return jsonify({
            'success': True,
            'data': {
                'student_id': data.get('studentId', 'student'),
                'forecast': forecast_data, # This is the list the frontend chart expects
                'model_version': 'prophet_dynamic_v1.1' if PROPHET_AVAILABLE else 'mock_v1.1'
            }
        })
        
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# --- 4. Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5006))
    print(f"--- DYNAMIC PERFORMANCE PREDICTION ENGINE (PROPHET) ---")
    print(f"🚀 Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)