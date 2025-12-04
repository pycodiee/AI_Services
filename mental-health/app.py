"""
AI Engine 4: Mental Health Analysis
Sentiment Analysis using Transformers for student well-being monitoring
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

app = Flask(__name__)
CORS(app)

# Try to import transformers, use mock if not available
try:
    from transformers import pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Could not load transformer model: {e}. Using mock analysis.")
    sentiment_analyzer = None
    MODEL_LOADED = False


def mock_sentiment_analysis(text):
    """Mock sentiment analysis when model is not available"""
    # Simple keyword-based sentiment
    negative_keywords = ['sad', 'anxious', 'stressed', 'depressed', 'worried', 'tired', 'overwhelmed', 'frustrated', 'angry', 'scared']
    positive_keywords = ['happy', 'excited', 'confident', 'motivated', 'great', 'good', 'wonderful', 'amazing', 'love', 'enjoy']
    
    text_lower = text.lower()
    
    negative_count = sum(1 for word in negative_keywords if word in text_lower)
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    
    if negative_count > positive_count:
        sentiment = 'NEGATIVE'
        score = 0.6 + (negative_count * 0.1)
    elif positive_count > negative_count:
        sentiment = 'POSITIVE'
        score = 0.6 + (positive_count * 0.1)
    else:
        sentiment = 'NEUTRAL'
        score = 0.5
    
    score = min(score, 0.99)
    
    return [{
        'label': sentiment,
        'score': score
    }]


def analyze_mental_health(text):
    """Analyze mental health from text"""
    try:
        # Perform sentiment analysis
        if sentiment_analyzer:
            results = sentiment_analyzer(text[:512])[0]  # Limit text length
        else:
            results = mock_sentiment_analysis(text)[0]
        
        sentiment_label = results['label']
        confidence = results['score']
        
        # Convert to sentiment score (-1 to 1)
        if sentiment_label == 'POSITIVE':
            sentiment_score = confidence
        elif sentiment_label == 'NEGATIVE':
            sentiment_score = -confidence
        else:
            sentiment_score = 0.0
        
        # Determine emotion and stress level
        if sentiment_score < -0.6:
            emotion = 'stressed'
            stress_level = 'high'
            requires_attention = True
        elif sentiment_score < -0.3:
            emotion = 'anxious'
            stress_level = 'moderate'
            requires_attention = True
        elif sentiment_score < 0.3:
            emotion = 'neutral'
            stress_level = 'low'
            requires_attention = False
        elif sentiment_score < 0.6:
            emotion = 'content'
            stress_level = 'low'
            requires_attention = False
        else:
            emotion = 'happy'
            stress_level = 'low'
            requires_attention = False
        
        # Generate recommendations
        recommendations = []
        if stress_level == 'high':
            recommendations.extend([
                'Consider scheduling a counseling session',
                'Engage in stress-relief activities (meditation, exercise)',
                'Talk to a trusted friend or family member'
            ])
        elif stress_level == 'moderate':
            recommendations.extend([
                'Take regular breaks during study sessions',
                'Practice mindfulness and relaxation techniques',
                'Maintain a healthy sleep schedule'
            ])
        else:
            recommendations.append('Keep up the positive mindset!')
        
        return {
            'sentiment_score': round(sentiment_score, 3),
            'confidence': round(confidence, 3),
            'emotion': emotion,
            'stress_level': stress_level,
            'requires_attention': requires_attention,
            'recommendations': recommendations,
            'text_analyzed': text[:100] + '...' if len(text) > 100 else text,
            'model_version': 'distilbert_v1.0' if MODEL_LOADED else 'mock_v1.0'
        }
    
    except Exception as e:
        raise Exception(f"Analysis failed: {str(e)}")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Mental Health Analysis',
        'model_loaded': MODEL_LOADED
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Analyze mental health
        result = analyze_mental_health(text)
        
        # Add metadata
        result['student_id'] = data.get('student_id')
        
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
    print(f"🚀 Starting Mental Health Analysis Service on port {config.MENTAL_HEALTH_PORT}")
    app.run(host='0.0.0.0', port=config.MENTAL_HEALTH_PORT, debug=True)

