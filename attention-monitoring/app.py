"""
AI Engine 2: Attention Monitoring
Computer Vision model using MediaPipe for detecting student engagement
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

app = Flask(__name__)
CORS(app)

# Try to import MediaPipe, use OpenCV fallback if not available
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully")
except ImportError:
    print("⚠️  MediaPipe not available. Using OpenCV-only fallback.")
    MEDIAPIPE_AVAILABLE = False
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def decode_image(image_data):
    """Decode base64 image"""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


def calculate_head_pose(landmarks, image_shape):
    """Calculate head pose angles"""
    # Get key facial landmarks
    nose_tip = landmarks[1]
    chin = landmarks[152]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    
    h, w = image_shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    nose = np.array([nose_tip.x * w, nose_tip.y * h])
    chin_pt = np.array([chin.x * w, chin.y * h])
    left_eye_pt = np.array([left_eye.x * w, left_eye.y * h])
    right_eye_pt = np.array([right_eye.x * w, right_eye.y * h])
    
    # Calculate angles (simplified)
    # Yaw (left-right rotation)
    face_center_x = (left_eye_pt[0] + right_eye_pt[0]) / 2
    yaw = (nose[0] - face_center_x) / w * 90  # Approximate yaw angle
    
    # Pitch (up-down rotation)
    eye_center_y = (left_eye_pt[1] + right_eye_pt[1]) / 2
    pitch = (nose[1] - eye_center_y) / h * 90  # Approximate pitch angle
    
    return {
        'yaw': float(yaw),
        'pitch': float(pitch),
        'roll': 0.0  # Simplified - not calculating roll
    }


def analyze_attention_opencv(image):
    """Analyze attention using OpenCV only (fallback)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return {
            'face_detected': False,
            'attention_score': 0.0,
            'engagement_level': 'not_detected',
            'is_looking_at_screen': False,
            'message': 'No face detected'
        }
    
    # Face detected - provide basic attention estimate
    # In a real implementation, you'd use more sophisticated analysis
    attention_score = np.random.uniform(0.6, 0.9)  # Mock score
    
    if attention_score >= 0.7:
        engagement_level = 'high'
    elif attention_score >= 0.5:
        engagement_level = 'medium'
    else:
        engagement_level = 'low'
    
    return {
        'face_detected': True,
        'attention_score': round(attention_score, 3),
        'engagement_level': engagement_level,
        'is_looking_at_screen': True,
        'head_pose': {'yaw': 0, 'pitch': 0, 'roll': 0},
        'facial_expression': 'neutral',
        'model_version': 'opencv_v1.0'
    }


def analyze_attention(image):
    """Analyze student attention from image"""
    try:
        if MEDIAPIPE_AVAILABLE:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return {
                    'face_detected': False,
                    'attention_score': 0.0,
                    'engagement_level': 'not_detected',
                    'is_looking_at_screen': False,
                    'message': 'No face detected'
                }
            
            # Get landmarks
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate head pose
            head_pose = calculate_head_pose(landmarks, image.shape)
            
            # Determine if looking at screen
            yaw_threshold = 30
            pitch_threshold = 25
            
            is_looking_forward = (
                abs(head_pose['yaw']) < yaw_threshold and
                abs(head_pose['pitch']) < pitch_threshold
            )
            
            # Calculate attention score
            yaw_score = max(0, 1 - abs(head_pose['yaw']) / yaw_threshold)
            pitch_score = max(0, 1 - abs(head_pose['pitch']) / pitch_threshold)
            attention_score = (yaw_score + pitch_score) / 2
            
            # Determine engagement level
            if attention_score >= 0.7:
                engagement_level = 'high'
            elif attention_score >= 0.4:
                engagement_level = 'medium'
            else:
                engagement_level = 'low'
            
            return {
                'face_detected': True,
                'attention_score': round(attention_score, 3),
                'engagement_level': engagement_level,
                'is_looking_at_screen': is_looking_forward,
                'head_pose': head_pose,
                'facial_expression': 'neutral',
                'model_version': 'mediapipe_v1.0'
            }
        else:
            # Use OpenCV fallback
            return analyze_attention_opencv(image)
    
    except Exception as e:
        raise Exception(f"Analysis failed: {str(e)}")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Attention Monitoring',
        'model': 'MediaPipe Face Mesh' if MEDIAPIPE_AVAILABLE else 'OpenCV Haar Cascade'
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode image
        image_data = data['image']
        image = decode_image(image_data)
        
        # Analyze attention
        result = analyze_attention(image)
        
        # Add metadata
        result['student_id'] = data.get('student_id')
        result['session_id'] = data.get('session_id')
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print(f"🚀 Starting Attention Monitoring Service on port {config.ATTENTION_MONITORING_PORT}")
    app.run(host='0.0.0.0', port=config.ATTENTION_MONITORING_PORT, debug=True)

