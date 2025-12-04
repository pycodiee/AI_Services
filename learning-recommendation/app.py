from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random 

# Mock configuration settings 
class MockConfig:
    LEARNING_RECOMMENDATION_PORT = 5000 
config = MockConfig()

app = Flask(__name__)
CORS(app)

# --- RL CONFIGURATION (The Engine Parameters) ---
MASTERY_THRESHOLD = 75

WEIGHTS = {
    "test": 0.50,      # Risk Prediction Model Score
    "assignment": 0.25, # Latest Assignment Score
    "interest": 0.15,    # Student Interest/Engagement Factor
    "time": 0.10       # Time Spent on Topic / Time Since Last Interaction
}

# --- STATIC MOCK DATA (To satisfy all required frontend fields) ---
CATEGORIES = ['Computer Science', 'Engineering', 'Mathematics', 'General']
DIFFICULTIES = ['Beginner', 'Intermediate', 'Advanced']
TAGS = ['RL', 'Adaptive', 'Priority', 'Core Concept']
THUMBNAILS = ['💻', '🧮', '⚙️', '📈']

# --- RL PRIORITY ENGINE CORE LOGIC (Copied from dashboard_utils.py) ---
def calculate_priority_score(module_data, weights):
    # Calculates the high-value 'priority_score'
    risk_score = module_data.get("test_score", 0)
    assignment_score = module_data.get("assignment_score", 100)
    interest_rating = module_data.get("interest_rating", 1)
    time_spent = module_data.get("time_spent", 0.1)

    risk_factor = (100 - risk_score) 
    assignment_factor = (100 - assignment_score)
    time_factor = 1 / time_spent 
    interest_factor = interest_rating * 20

    priority_score = (
        weights["test"] * risk_factor +
        weights["assignment"] * assignment_factor +
        weights["interest"] * interest_factor +
        weights["time"] * time_factor
    )
    
    return priority_score

def find_improvement_areas(profile, threshold, weights):
    """Calculates priority for all modules below the mastery threshold."""
    all_modules_data = []
    
    for module in profile["modules"]:
        if module["test_score"] < threshold: 
            module["priority_score"] = calculate_priority_score(module, weights)
            all_modules_data.append(module)
            
    all_modules_data.sort(key=lambda x: x["priority_score"], reverse=True)
    
    return all_modules_data

# --- FLASK ENDPOINTS ---

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'RL Adaptive Recommendation Engine (Adapter Mode)'
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Runs the RL Priority Engine and converts the output to the LearningModule 
    schema required by the static frontend.
    """
    try:
        data = request.get_json()
        
        # CRITICAL ASSUMPTION: The POST request MUST contain the 'modules_data' key
        # with the student's current scores for each syllabus topic.
        if 'modules_data' not in data:
            # Fallback for testing: using mock data if the expected key is not found
            student_modules = [
                {"name": "Data Structures & Algorithms", "test_score": 30, "assignment_score": 75, "interest_rating": 4, "time_spent": 0.5},
                {"name": "Fluid Dynamics Course", "test_score": 60, "assignment_score": 50, "interest_rating": 5, "time_spent": 0.2},
                {"name": "Advanced Calculus: Derivatives", "test_score": 85, "assignment_score": 90, "interest_rating": 2, "time_spent": 2.0},
            ]
        else:
            student_modules = data['modules_data']


        student_profile = {
            "student_name": data.get("user_id", "Unknown"),
            "subject": data.get("subject_context", "General"), 
            "modules": student_modules
        }
        
        # 1. Run RL Engine
        prioritized_topics = find_improvement_areas(
            student_profile, 
            MASTERY_THRESHOLD, 
            WEIGHTS
        )
        
        # 2. ADAPTER LAYER: Convert RL Output to Frontend Schema
        recommendations_output = []
        for i, topic in enumerate(prioritized_topics):
            rl_score = topic['priority_score']
            
            # CRITICAL: Scale RL score to fit the frontend's expected matchScore (0-100)
            # This ensures the progress bar looks reasonable.
            # Using a simplified scaling method: Max score is set to 250 for scaling.
            match_score = min(99, 50 + int(rl_score / 5)) 

            recommendations_output.append({
                # Mapped RL data
                'id': str(i + 1), 
                'title': topic['name'], 
                'matchScore': match_score, 
                'description': f"Priority is HIGH! Risk Score: {topic['test_score']}%, Latest Score: {topic['assignment_score']}%. RL Priority Index: {round(rl_score, 2)}",

                # Static/Mock fields to satisfy the remaining FE requirements
                'category': random.choice(CATEGORIES),
                'difficulty': random.choice(DIFFICULTIES),
                'duration': f"{random.randint(4, 12)} Weeks",
                'completionRate': random.randint(50, 95),
                'enrolledStudents': random.randint(100, 5000),
                'rating': round(4.0 + random.random() * 0.9, 1),
                'tags': random.sample(TAGS, 2),
                'thumbnail': random.choice(THUMBNAILS),
            })
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': student_profile['student_name'],
                'recommendations': recommendations_output, 
                'model_version': 'rl_priority_engine_adapter_v1.0'
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Internal Server Error: {str(e)}",
            'detail': "Ensure the POST body contains the 'modules_data' key with the required RL factors (name, test_score, assignment_score, interest_rating, time_spent)."
        }), 500


if __name__ == '__main__':
    print(f"🚀 Starting RL Adaptive Recommendation Service (Adapter) on port {config.LEARNING_RECOMMENDATION_PORT}")
    app.run(host='0.0.0.0', port=config.LEARNING_RECOMMENDATION_PORT, debug=True)
















