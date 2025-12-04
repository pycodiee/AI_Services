"""
AI Engine 5: Chatbot
Multilingual AI tutor using Open Router API Gateway + Translation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import requests
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

app = Flask(__name__)
CORS(app)

# Open Router API Configuration
OPENROUTER_AVAILABLE = bool(config.OPENROUTER_API_KEY)
OPENROUTER_BASE_URL = config.OPENROUTER_BASE_URL
OPENROUTER_MODEL = config.OPENROUTER_MODEL

if OPENROUTER_AVAILABLE:
    print(f"✅ Open Router API configured - Model: {OPENROUTER_MODEL}")
    print(f"   Base URL: {OPENROUTER_BASE_URL}")
    print(f"   API Key: {'*' * (len(config.OPENROUTER_API_KEY) - 4) + config.OPENROUTER_API_KEY[-4:] if len(config.OPENROUTER_API_KEY) > 4 else '***'}")
else:
    print(f"⚠️  Open Router API key not found. Using mock responses.")
    print(f"   To enable Open Router, set OPENROUTER_API_KEY in your .env file")

# Try to import translation
try:
    from googletrans import Translator
    translator = Translator()
    TRANSLATION_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Translation not available: {e}")
    translator = None
    TRANSLATION_AVAILABLE = False

# Conversation storage (in-memory, use Redis in production)
conversations = {}

# Mock responses for common questions (fallback when API unavailable)
MOCK_RESPONSES = {
    'math': """I'd love to help you with math! I can assist with:
• Solving equations step-by-step (linear, quadratic, systems)
• Explaining mathematical concepts clearly
• Working through problems with detailed solutions
• Providing practice examples

For example, if you ask "solve 2x + 5 = 13", I'll show you:
Step 1: Subtract 5 from both sides → 2x = 8
Step 2: Divide by 2 → x = 4
And explain why each step works!

What math problem would you like help with? 🧮""",
    'equation': """I can help solve equations step-by-step! Here's how I approach it:

**For linear equations like 3x + 7 = 22:**
1. Isolate the variable term
2. Perform inverse operations
3. Simplify both sides
4. Check your answer

**For quadratic equations:**
1. Set equation to zero
2. Factor or use quadratic formula
3. Solve for each solution
4. Verify answers

**For systems of equations:**
1. Use substitution or elimination
2. Solve for one variable
3. Substitute back to find the other
4. Check both equations

Share your equation and I'll solve it step-by-step with explanations! 📐""",
    'study': "Effective study techniques include: 1) Spaced repetition - review material over increasing intervals, 2) Active recall - test yourself instead of just re-reading, 3) Pomodoro technique - study in 25-minute focused sessions, 4) Teach others - explaining concepts helps solidify understanding, and 5) Create mind maps to connect concepts visually. What subject would you like study tips for?",
    'stress': "I understand exam stress can be overwhelming. Here are some helpful strategies: 1) Start preparing early to avoid last-minute cramming, 2) Practice deep breathing exercises (4-7-8 breathing technique), 3) Get adequate sleep (7-9 hours), 4) Exercise regularly to reduce anxiety, 5) Break study sessions into manageable chunks, and 6) Remember that one exam doesn't define you. Would you like to talk more about what's causing stress?",
    'explain': """I excel at providing clear explanations! I can help explain:
• Math concepts (algebra, calculus, geometry)
• Science topics (physics, chemistry, biology)
• Any subject matter you're studying

I'll use:
✓ Simple, clear language
✓ Step-by-step breakdowns
✓ Examples and analogies
✓ Visual descriptions when helpful

What concept would you like me to explain? 🎓""",
    'default': """I'm EduGuardian AI, your educational assistant! I can help with:

🧮 **Math Problem Solving**: Solve equations step-by-step, show all work
📐 **Equation Solving**: Linear, quadratic, systems - with detailed steps
📚 **Explanations**: Clear explanations of any concept
📝 **Study Help**: Homework, assignments, learning strategies
😌 **Support**: Motivation and stress management

**Example questions you can ask:**
• "Solve 3x + 5 = 20"
• "Explain how photosynthesis works"
• "How do I solve quadratic equations?"
• "Explain derivatives step-by-step"

What would you like help with?"""
}


def get_mock_response(message):
    """Generate mock response based on keywords (fallback when API unavailable)"""
    message_lower = message.lower()
    
    # Check for equation solving requests
    if any(word in message_lower for word in ['solve', 'equation', 'x =', 'find x', 'solve for']):
        return MOCK_RESPONSES['equation']
    # Check for explanation requests
    elif any(word in message_lower for word in ['explain', 'how does', 'what is', 'tell me about', 'describe']):
        return MOCK_RESPONSES['explain']
    # Check for math topics
    elif any(word in message_lower for word in ['math', 'calculus', 'algebra', 'geometry', 'trigonometry', 'derivative', 'integral']):
        return MOCK_RESPONSES['math']
    # Check for study techniques
    elif any(word in message_lower for word in ['study', 'learn', 'technique', 'method', 'how to study']):
        return MOCK_RESPONSES['study']
    # Check for stress/anxiety
    elif any(word in message_lower for word in ['stress', 'anxiety', 'exam', 'test', 'worried', 'nervous']):
        return MOCK_RESPONSES['stress']
    else:
        return MOCK_RESPONSES['default']


def chat_with_openrouter(message, conversation_history):
    """Chat using Open Router API Gateway"""
    try:
        if not OPENROUTER_AVAILABLE:
            print("⚠️  OpenRouter API key not configured, using mock response")
            return get_mock_response(message)
        
        # Enhanced system prompt for educational assistance
        system_prompt = """You are EduGuardian AI, an expert educational assistant and tutor. Your capabilities include:

1. **Math Problem Solving**: Solve equations step-by-step, show work clearly, explain each step
2. **Equation Solving**: Handle linear, quadratic, polynomial, and system of equations with detailed steps
3. **Explanations**: Provide clear, comprehensive explanations of concepts in math, science, and other subjects
4. **Step-by-Step Guidance**: Break down complex problems into manageable steps
5. **Educational Support**: Help with homework, assignments, and learning strategies
6. **Motivation**: Encourage students while being empathetic

**Important Guidelines:**
- For math problems: Show ALL steps clearly, use proper mathematical notation
- For equations: Solve systematically, explain why each step is taken
- For explanations: Use examples, analogies, and visual descriptions when helpful
- Be encouraging and patient - students learn at different paces
- If a student asks "how to solve" or "explain", provide detailed step-by-step guidance
- Use emojis sparingly to make responses friendly but keep focus on education
- Responses should be clear and detailed enough to help students learn independently

Remember: Your goal is to help students UNDERSTAND, not just give answers. Guide them through the learning process."""
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation history (last 10 messages to keep context manageable)
        messages.extend(conversation_history[-10:])
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Prepare request to Open Router
        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://eduguardian.ai",
            "X-Title": "EduGuardian AI",
            "Content-Type": "application/json"
        }
        
        # Payload with proper configuration
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.7,
            # Add these fields to help with 503 errors
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        
        print(f"📤 Sending request to Open Router API...")
        print(f"   Model: {OPENROUTER_MODEL}")
        print(f"   Message length: {len(message)} chars")
        print(f"   URL: {url}")
        
        # Make request with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=30
                )
                
                print(f"   Response Status: {response.status_code}")
                
                # If successful, break the retry loop
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response content
                    if "choices" in result and len(result["choices"]) > 0:
                        ai_response = result["choices"][0]["message"]["content"]
                        print(f"✅ Received response from Open Router ({len(ai_response)} chars)")
                        return ai_response
                    else:
                        print(f"⚠️  Unexpected response format: {result}")
                        return get_mock_response(message)
                
                # Handle specific error codes
                elif response.status_code == 503:
                    error_data = response.json() if response.text else {}
                    error_message = error_data.get('error', {}).get('message', 'Service unavailable')
                    
                    print(f"❌ OpenRouter 503 Error: {error_message}")
                    print(f"   Response: {response.text[:500]}")
                    
                    # Check if it's a rate limit or model availability issue
                    if "rate limit" in error_message.lower():
                        print("   Issue: Rate limit exceeded")
                        if attempt < max_retries - 1:
                            print(f"   Retrying in {retry_delay} seconds... (attempt {attempt + 2}/{max_retries})")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                    elif "model" in error_message.lower():
                        print(f"   Issue: Model '{OPENROUTER_MODEL}' may not be available")
                        print(f"   Suggestion: Check if the model name is correct")
                        print(f"   Common models: meta-llama/llama-3-8b-instruct, anthropic/claude-2")
                        break
                    else:
                        print("   Issue: Service temporarily unavailable")
                        if attempt < max_retries - 1:
                            print(f"   Retrying in {retry_delay} seconds... (attempt {attempt + 2}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                    break
                
                elif response.status_code == 401:
                    print(f"❌ Authentication Error: Invalid API key")
                    print(f"   Please check your OPENROUTER_API_KEY in .env file")
                    break
                
                elif response.status_code == 429:
                    print(f"❌ Rate Limit Error")
                    if attempt < max_retries - 1:
                        print(f"   Retrying in {retry_delay} seconds... (attempt {attempt + 2}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    break
                
                else:
                    print(f"❌ HTTP Error {response.status_code}: {response.text[:200]}")
                    break
                    
            except requests.exceptions.Timeout:
                print(f"❌ Request timeout")
                if attempt < max_retries - 1:
                    print(f"   Retrying in {retry_delay} seconds... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                break
            
            except requests.exceptions.ConnectionError as e:
                print(f"❌ Connection error: {e}")
                break
        
        # If all retries failed, use mock response
        print("⚠️  All retry attempts failed, using mock response")
        return get_mock_response(message)
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling Open Router API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"   Response Status: {e.response.status_code}")
                print(f"   Response Text: {e.response.text[:200]}")
            except:
                pass
        return get_mock_response(message)
    except Exception as e:
        print(f"❌ Unexpected error in chat_with_openrouter: {e}")
        import traceback
        traceback.print_exc()
        return get_mock_response(message)


def translate_text(text, target_lang):
    """Translate text to target language"""
    try:
        if not translator or target_lang == 'en':
            return text
        
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Chatbot',
        'openrouter_available': OPENROUTER_AVAILABLE,
        'openrouter_model': OPENROUTER_MODEL if OPENROUTER_AVAILABLE else None,
        'translation_available': TRANSLATION_AVAILABLE,
        'base_url': OPENROUTER_BASE_URL
    })


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message']
        user_id = data.get('user_id', 'anonymous')
        conversation_id = data.get('conversation_id', f"{user_id}_{int(datetime.now().timestamp())}")
        language = data.get('language', 'en')
        
        # Get or create conversation
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        conversation_history = conversations[conversation_id]
        
        # Translate input if needed
        if language != 'en' and TRANSLATION_AVAILABLE:
            message_en = translate_text(message, 'en')
        else:
            message_en = message
        
        # Get response from Open Router
        response = chat_with_openrouter(message_en, conversation_history)
        
        # Translate response if needed
        if language != 'en' and TRANSLATION_AVAILABLE:
            response_translated = translate_text(response, language)
        else:
            response_translated = response
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": message_en})
        conversation_history.append({"role": "assistant", "content": response})
        conversations[conversation_id] = conversation_history
        
        return jsonify({
            'success': True,
            'data': {
                'message': response_translated,
                'conversation_id': conversation_id,
                'language': language,
                'timestamp': datetime.now().isoformat(),
                'model_version': OPENROUTER_MODEL if OPENROUTER_AVAILABLE else 'mock_v1.0'
            }
        })
    
    except Exception as e:
        print(f"❌ Error in /chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get conversation history"""
    if conversation_id in conversations:
        return jsonify({
            'success': True,
            'data': {
                'conversation_id': conversation_id,
                'messages': conversations[conversation_id]
            }
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Conversation not found'
        }), 404


if __name__ == '__main__':
    print(f"🚀 Starting Chatbot Service on port {config.CHATBOT_PORT}")
    app.run(host='0.0.0.0', port=config.CHATBOT_PORT, debug=True)