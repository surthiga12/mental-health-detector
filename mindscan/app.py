"""
MindScan - Mental Health Detection Chatbot
A Flask web application for mental health conversation and assessment.
"""

import os
import secrets
import logging
import random
from typing import Dict, List, Any, Tuple, Optional
from flask import Flask, request, jsonify, render_template, session

# Import all required modules
from text_processor import TextPreprocessor
from classifier import MentalHealthClassifier
from advanced_classifier import AdvancedMentalHealthClassifier
from dataset_manager import MentalHealthDatasetManager
from recommender import RecommendationSystem
from conversation import ConversationHistory
from crisis_detector import CrisisDetector
from ml_predictor import SimpleMLPredictor
from textblob import TextBlob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Initialize components
try:
    # Download required NLTK data
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    text_processor = TextPreprocessor()
    classifier = MentalHealthClassifier()
    advanced_classifier = AdvancedMentalHealthClassifier()
    dataset_manager = MentalHealthDatasetManager()
    recommender = RecommendationSystem()
    conversation_history = ConversationHistory()
    crisis_detector = CrisisDetector()
    ml_predictor = SimpleMLPredictor()
    
    # Try to load pre-trained models
    try:
        advanced_classifier.load_model('models/advanced_mental_health_model.pkl')
        logger.info("Advanced model loaded successfully")
    except FileNotFoundError:
        logger.info("No pre-trained advanced model found. Will use rule-based system.")
    
    logger.info("Components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    raise

# Conversation flow
conversation_starters = [
    "How has your day been so far?",
    "What's on your mind today?",
    "How are you feeling at the moment?",
    "Would you like to talk about what's happening in your life?",
    "Is there something specific you'd like to discuss?"
]

follow_up_questions = {
    'normal': [
        "That's absolutely wonderful to hear! What's been going especially well for you?",
        "I'm so glad you're feeling positive! What activities or experiences have been bringing you the most joy?",
        "That's fantastic! I love hearing such positive energy. How do you maintain this wonderful mindset?",
        "How amazing! Your happiness is truly uplifting. Would you like to share more about what's been making you feel so great?"
    ],
    'moderate': [
        "I understand things might be challenging. What helps you cope when you feel this way?",
        "Would you like to tell me more about what's bothering you?",
        "How long have you been feeling this way?",
        "Is there someone in your life you can talk to about these feelings?"
    ],
    'high': [
        "I'm here to listen and support you. Have you talked to anyone else about these feelings?",
        "Your feelings matter, and you're not alone. Would you be open to speaking with a counselor?",
        "I hear how difficult this is. What can I do to help you feel safer right now?",
        "These feelings are temporary, and help is available. Can we explore some immediate support options?"
    ]
}

empathetic_responses = {
    'normal': [
        "It's wonderful that you're feeling positive! Your happiness brightens my day too.",
        "I'm really glad to hear you're doing well! That's fantastic news.",
        "That's absolutely amazing! It's great to see you in such good spirits.",
        "How wonderful! Your positive energy is truly uplifting.",
        "That's fantastic! I love hearing when someone is feeling happy and well."
    ],
    'moderate': [
        "I hear you, and what you're feeling is valid.",
        "It's okay to have mixed feelings about things.",
        "Thank you for sharing that with me.",
        "It takes courage to talk about these feelings."
    ],
    'high': [
        "I'm really concerned about what you're going through.",
        "You don't have to face this alone - help is available.",
        "I want you to know that your life matters.",
        "These feelings are real and deserve attention."
    ]
}

# Chatbot responses based on risk level
responses = {
    'normal': [
        "That's absolutely wonderful to hear! Your happiness is contagious and I'm so glad you're feeling this way.",
        "How fantastic! I love hearing positive news like this. Maintaining good mental health is so important and you're doing great!",
        "That's amazing! It's such a joy to hear when someone is feeling happy and positive. What's been contributing to these good feelings?",
        "Wonderful! Your positive energy really comes through and it's heartwarming. I'd love to hear more about what's going well for you!"
    ],
    'moderate': [
        "I understand you're going through some challenges. Would you like to talk more about it?",
        "Thank you for sharing. These feelings are valid, and it's okay to seek support when needed.",
        "I hear you, and it's brave of you to share your feelings. Would you like to explore some coping strategies?"
    ],
    'high': [
        "I'm very concerned about your safety right now. If you're having thoughts of suicide, please know that you can call 1056 (KIRAN Mental Health Helpline) or 112 (Emergency) right away. Would you be willing to do that?",
        "Your life matters, and what you're going through sounds incredibly painful. Please reach out to emergency services or call 1056 (KIRAN) - they are trained to help and want to support you through this.",
        "I hear how much pain you're in. This is an emergency, and you deserve immediate support. Please call 1056 (KIRAN), 112 (Emergency), or 9152987821 (Suicide Prevention) - they are there for you 24/7 and want to help."
    ]
}

def generate_contextual_response(message: str, risk_level: str, detected_issues: List[str] = None) -> str:
    """Generate a contextual response based on message content and risk level."""
    if detected_issues is None:
        detected_issues = []
        
    message_lower = message.lower()
    
    # Check for positive emotions and expressions
    positive_keywords = [
        'happy', 'great', 'awesome', 'wonderful', 'fantastic', 'amazing', 'excellent',
        'good', 'better', 'fine', 'well', 'joy', 'joyful', 'excited', 'cheerful',
        'pleased', 'satisfied', 'content', 'delighted', 'thrilled', 'positive',
        'optimistic', 'upbeat', 'energetic', 'motivated', 'confident', 'proud',
        'love', 'loving', 'grateful', 'blessed', 'lucky', 'successful'
    ]
    
    positive_expressions = [
        'feeling good', 'doing well', 'going great', 'pretty good', 'really good',
        'much better', 'feeling better', 'feeling happy', 'feel amazing',
        'life is good', 'things are good', 'everything is great', 'having fun'
    ]
    
    # Check for positive sentiment
    has_positive_words = any(word in message_lower for word in positive_keywords)
    has_positive_expressions = any(expr in message_lower for expr in positive_expressions)
    
    if has_positive_words or has_positive_expressions:
        positive_responses = [
            "That's absolutely wonderful to hear! Your happiness truly brightens my day. What's been contributing to these positive feelings?",
            "How fantastic! I love hearing such positive energy. What's been going particularly well for you?",
            "That's amazing! It's so refreshing to hear when someone is feeling this good. What's been the source of your happiness?",
            "Wonderful news! Your positive outlook is truly inspiring. I'd love to hear more about what's been making you feel so great!",
            "That's such great news! I'm genuinely happy for you. What's been the highlight that's brought you this joy?"
        ]
        return random.choice(positive_responses)
    
    # Default responses based on risk level
    if risk_level == 'high':
        return random.choice([
            "I'm really concerned about what you're going through. You don't have to face this alone.",
            "What you're experiencing sounds incredibly difficult. I'm here to support you through this.",
            "Thank you for sharing something so personal and painful with me. Your courage in reaching out matters."
        ])
    elif risk_level == 'moderate':
        return random.choice([
            "I can hear that you're going through a challenging time. What's been the hardest part for you?",
            "It sounds like you're dealing with some significant stress. How long have you been feeling this way?",
            "Thank you for opening up about what's been troubling you. What would be most helpful to talk about right now?"
        ])
    else:
        return random.choice([
            "I'm here to listen and understand what you're going through. What's been on your mind lately?",
            "Thank you for sharing with me. How are these feelings affecting your day-to-day life?",
            "I appreciate you trusting me with your thoughts. What would you like to explore together?"
        ])

def detect_question_intent(message: str) -> str:
    """Detect if the user is asking a specific type of question."""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['how', 'what', 'why', 'when', 'where', 'who']):
        if any(word in message_lower for word in ['feel', 'feeling', 'help', 'better', 'cope', 'deal']):
            return 'help_seeking'
        elif any(word in message_lower for word in ['depression', 'anxiety', 'stress', 'mental health']):
            return 'information_seeking'
    
    if message_lower.endswith('?'):
        return 'general_question'
    
    return 'statement'

def get_conversation_response(risk_level: str, sentiment_trend: str, message_count: int) -> str:
    """Generate a contextual response based on conversation history."""
    if message_count == 0:
        return random.choice(conversation_starters)
    
    if sentiment_trend == 'improving':
        return "I notice you seem to be feeling a bit better. What's been helpful for you?"
    elif sentiment_trend == 'declining':
        return "I notice things might be getting more difficult. Would you like to talk about what's changed?"
    
    if message_count < 3:
        opening_responses = [
            "Thanks for continuing to share with me. What else is on your mind?",
            "I'm glad you're opening up. How else can I support you today?",
            "I appreciate you talking with me. What would you like to explore next?"
        ]
        return random.choice(opening_responses)
    
    return random.choice(follow_up_questions[risk_level])

@app.route('/')
def home():
    """Render the chat interface."""
    return render_template('chat.html')

@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {'status': 'healthy'}

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and maintain conversation context."""
    try:
        # Validate request
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("Received request without message")
            return jsonify({
                'error': 'Missing message'
            }), 400
        
        # Get session ID or create new one
        session_id = session.get('session_id')
        if not session_id:
            session_id = secrets.token_hex(8)
            session['session_id'] = session_id
            logger.info(f"Created new session: {session_id}")
        
        # Get message and log
        message = data['message']
        logger.info(f"Processing message: {message[:50]}...")
        
        # Process message and determine sentiment
        try:
            # Extract features from text
            features = text_processor.extract_features(message)
            text_lower = message.lower()
            
            # Enhanced greeting and conversation detection
            greeting_patterns = [
                'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 
                'greetings', 'howdy', 'what\'s up', 'how are you', 'how\'re you'
            ]
            farewell_patterns = ['bye', 'goodbye', 'see you', 'farewell', 'talk later', 'gotta go']
            gratitude_patterns = ['thanks', 'thank you', 'appreciate', 'grateful']
            
            # Enhanced positive sentiment detection
            positive_patterns = [
                'happy', 'great', 'awesome', 'wonderful', 'fantastic', 'amazing', 'excellent',
                'good', 'better', 'fine', 'well', 'joy', 'joyful', 'excited', 'cheerful',
                'pleased', 'satisfied', 'content', 'delighted', 'thrilled', 'positive',
                'optimistic', 'upbeat', 'energetic', 'motivated', 'confident', 'proud'
            ]
            
            # Check for conversation patterns
            is_greeting = any(pattern in text_lower for pattern in greeting_patterns)
            is_farewell = any(pattern in text_lower for pattern in farewell_patterns)
            is_gratitude = any(pattern in text_lower for pattern in gratitude_patterns)
            is_positive = any(pattern in text_lower for pattern in positive_patterns)
            
            if is_greeting and len(message.split()) <= 5:  # Short greeting
                risk_level = 'normal'
                greeting_responses = [
                    "Hello! I'm here to listen and support you. How are you feeling today?",
                    "Hi there! I'm glad you're here. What's on your mind?",
                    "Hello! I'm here to help. How has your day been going?",
                    "Hi! Thanks for reaching out. How are you doing right now?"
                ]
                chatbot_response = random.choice(greeting_responses)
            elif is_farewell:
                risk_level = 'normal'
                farewell_responses = [
                    "Take care of yourself! Remember, I'm here if you need to talk again.",
                    "Goodbye! Don't hesitate to reach out if you need support.",
                    "Take care! Remember that your mental health matters.",
                    "See you later! I hope you have a peaceful day."
                ]
                chatbot_response = random.choice(farewell_responses)
            elif is_gratitude:
                risk_level = 'normal'
                gratitude_responses = [
                    "You're very welcome! I'm here to support you whenever you need it.",
                    "I'm glad I could help! How are you feeling about everything now?",
                    "You don't need to thank me - supporting you is what I'm here for!",
                    "I'm happy to help! Is there anything else you'd like to talk about?"
                ]
                chatbot_response = random.choice(gratitude_responses)
            elif is_positive and not crisis_detector.detect_crisis(message):
                # Positive sentiment detected
                risk_level = 'normal'
                positive_responses = [
                    "That's absolutely wonderful to hear! I'm so glad you're feeling happy. What's been contributing to these positive feelings?",
                    "How fantastic! It's great to see you in such good spirits. What's been going well for you lately?",
                    "That's amazing! Your happiness is contagious. What's been making you feel so positive?",
                    "I love hearing that you're happy! It's wonderful when life feels good. What's been the highlight of your day?"
                ]
                chatbot_response = random.choice(positive_responses)
            else:
                # Check for crisis first
                crisis_detected = crisis_detector.detect_crisis(message)
                if crisis_detected:
                    risk_level = 'high'
                    chatbot_response = crisis_detector.get_crisis_response()
                    logger.warning(f"CRISIS DETECTED for session {session_id}: {message[:50]}...")
                else:
                    # Use ML model if available
                    ml_prediction = None
                    if ml_predictor.is_available():
                        ml_prediction = ml_predictor.predict(message)
                        risk_level = ml_prediction['risk_level']
                        detected_issues = ml_prediction.get('detected_issues', [])
                    
                    # Fall back to advanced classifier
                    if ml_prediction is None:
                        if advanced_classifier.is_trained:
                            prediction = advanced_classifier.predict(message)
                            risk_level = prediction['risk_level']
                            detected_issues = prediction.get('detected_issues', [])
                        else:
                            # Basic rule-based classification
                            risk_level = classifier.classify_basic(message)
                            detected_issues = []
                    
                    # Use TextBlob for sentiment analysis
                    blob = TextBlob(message)
                    polarity = blob.sentiment.polarity
                    
                    # Adjust risk level based on sentiment
                    if polarity < -0.5 and risk_level == 'normal':
                        risk_level = 'moderate'
                    
                    # Generate contextual response
                    chatbot_response = generate_contextual_response(message, risk_level, detected_issues)
            
            # Update conversation history
            conversation_history.add_message(
                session_id=session_id,
                message=message,
                sentiment=risk_level,
                confidence=0.8
            )
            
            # Get conversation context
            analysis = conversation_history.analyze_trend(session_id)
            sentiment_trend = analysis['trend']
            message_count = analysis['messages_count']
            
            # Get empathetic response and recommendations
            empathetic_response = random.choice(empathetic_responses[risk_level])
            recommendations = recommender.get_recommendations(risk_level)
            
            logger.info(f"Session {session_id}: Risk level {risk_level}, Sentiment trend {sentiment_trend}")
            
            return jsonify({
                'chatbot_response': chatbot_response,
                'empathetic_response': empathetic_response,
                'assessment': {
                    'risk_level': risk_level,
                    'confidence': 0.8,
                    'sentiment_trend': sentiment_trend
                },
                'recommendations': recommendations
            })
            
        except Exception as inner_e:
            logger.error(f"Error processing message content: {str(inner_e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'chatbot_response': "I'm here to help. Could you please tell me more about how you're feeling?",
            'empathetic_response': "I want to understand and support you better.",
            'assessment': {'risk_level': 'normal', 'confidence': 0.5},
            'recommendations': [
                {
                    'title': 'Share Your Thoughts',
                    'description': 'Feel free to express yourself. I\'m here to listen and understand.'
                }
            ]
        })

@app.route('/train_advanced', methods=['POST'])
def train_advanced_model() -> Dict[str, Any]:
    """Train the advanced ML classifier with multiple algorithms."""
    try:
        data = request.get_json()
        
        if data and data.get('use_sample_data'):
            # Generate comprehensive dataset
            logger.info("Generating comprehensive dataset...")
            texts, labels = dataset_manager.create_comprehensive_dataset(size=600)
            
            # Augment the dataset for better training
            texts, labels = dataset_manager.augment_dataset(texts, labels, multiplier=2)
            
            # Save the generated dataset
            dataset_manager.save_dataset(texts, labels, "training_dataset.csv")
            
        elif data and 'data' in data:
            # Use provided data
            texts = []
            labels = []
            for item in data['data']:
                if 'text' in item and 'label' in item:
                    texts.append(item['text'])
                    labels.append(item['label'])
        else:
            return jsonify({
                'error': 'No training data provided. Use "use_sample_data": true or provide "data" array.'
            }), 400
        
        if len(texts) < 10:
            return jsonify({
                'error': 'Insufficient training data. Need at least 10 samples.'
            }), 400
        
        # Train the advanced classifier
        logger.info(f"Training advanced model with {len(texts)} samples...")
        results = advanced_classifier.train(texts, labels)
        
        # Save the trained model
        os.makedirs('models', exist_ok=True)
        advanced_classifier.save_model('models/advanced_mental_health_model.pkl')
        
        # Get dataset statistics
        stats = dataset_manager.get_dataset_statistics(labels)
        
        return jsonify({
            'message': 'Advanced model training completed successfully',
            'results': {
                'ensemble_score': results['ensemble_score'],
                'training_samples': results['training_samples'],
                'feature_count': results['feature_count'],
                'dataset_stats': stats
            },
            'model_saved': True
        })
        
    except Exception as e:
        logger.error(f"Error training advanced model: {str(e)}")
        return jsonify({
            'error': f'Training failed: {str(e)}'
        }), 500

@app.route('/model/status', methods=['GET'])
def model_status() -> Dict[str, Any]:
    """Get the status of all models."""
    try:
        return jsonify({
            'advanced_model_trained': advanced_classifier.is_trained,
            'ml_predictor_available': ml_predictor.is_available(),
            'crisis_detector_active': True,
            'rule_based_system': True,
            'available_endpoints': [
                '/chat - Main chat interface',
                '/train_advanced - Train advanced ML model',
                '/model/status - This endpoint'
            ]
        })
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return jsonify({
            'error': f'Status check failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
