# MindSync with Enhanced Natural Conversation and Multi-Language Responses
# pip install flask flask-socketio opencv-python librosa soundfile numpy scikit-learn transformers torch speechrecognition pyttsx3 pillow textblob tensorflow eventlet wave googletrans==4.0.0-rc1

import os
import cv2
import numpy as np
import librosa
import threading
import wave
import io
import base64
import tempfile
import struct
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
import json
import logging
from datetime import datetime, timedelta
import warnings
from textblob import TextBlob
import re
import time
import queue
from PIL import Image
import speech_recognition as sr
import pyttsx3
import random
from collections import defaultdict, Counter
from googletrans import Translator

# Try to import FER with error handling
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FER library not available: {e}")
    FER_AVAILABLE = False

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")
    TF_AVAILABLE = False

from collections import deque
import random

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mindsync_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# --- INTEGRATED CAMERA MODULE ---
class FacialExpressionDetector:
    """A dedicated class to handle all camera and facial expression detection logic."""
    def __init__(self):
        self.is_active = False
        self.emotion_detector = None
        self.face_cascade = None
        self.emotion_history = deque(maxlen=100)  # Store emotion history
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                logger.warning("Face cascade classifier not loaded properly.")
                self.face_cascade = None

            if FER_AVAILABLE:
                try:
                    self.emotion_detector = FER()
                    logger.info("FER: Facial emotion detector initialized successfully.")
                except Exception as e:
                    logger.error(f"FER: Initialization error: {e}")
                    self.emotion_detector = None
            else:
                logger.warning("FER: Library not available. Facial detection will be limited.")
        except Exception as e:
            logger.error(f"FER: Critical initialization error: {e}")
            self.emotion_detector = None
            self.face_cascade = None

    def analyze_frame(self, frame_data):
        """Decodes a base64 frame and analyzes it for facial expressions."""
        if not self.is_active: return
        try:
            image_data = frame_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: logger.error("FER: Could not decode frame."); return
            if self.emotion_detector:
                emotions = self.emotion_detector.detect_emotions(frame)
                if emotions:
                    emotion_data = emotions[0]['emotions']
                    dominant_emotion = max(emotion_data, key=emotion_data.get)
                    # Store emotion with timestamp
                    self.emotion_history.append({
                        'emotion': dominant_emotion,
                        'timestamp': datetime.now(),
                        'confidence': emotion_data[dominant_emotion]
                    })
                    socketio.emit('facial_emotion_update', {'emotion': dominant_emotion})
                    logger.info(f"FER: Detected emotion: {dominant_emotion}")
                    return
            if self.face_cascade:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0: 
                    # Store face detected with timestamp
                    self.emotion_history.append({
                        'emotion': 'face detected',
                        'timestamp': datetime.now(),
                        'confidence': 0.5
                    })
                    socketio.emit('facial_emotion_update', {'emotion': 'face detected'}); 
                    logger.info("FER: Fallback - Face detected.")
                else: 
                    self.emotion_history.append({
                        'emotion': 'no face',
                        'timestamp': datetime.now(),
                        'confidence': 0.5
                    })
                    socketio.emit('facial_emotion_update', {'emotion': 'no face'})
            else: 
                self.emotion_history.append({
                    'emotion': 'detector not ready',
                    'timestamp': datetime.now(),
                    'confidence': 0.5
                })
                socketio.emit('facial_emotion_update', {'emotion': 'detector not ready'})
        except Exception as e:
            logger.error(f"FER: Error processing frame: {e}")
            self.emotion_history.append({
                'emotion': 'error',
                'timestamp': datetime.now(),
                'confidence': 0.5
            })
            socketio.emit('facial_emotion_update', {'emotion': 'error'})

    def start(self): 
        self.is_active = True; 
        logger.info("FER: Camera detector started."); 
        socketio.emit('camera_status', {'active': True})
    
    def stop(self): 
        self.is_active = False; 
        logger.info("FER: Camera detector stopped."); 
        socketio.emit('camera_status', {'active': False})
    
    def get_emotion_summary(self):
        """Get a summary of emotions detected during the session."""
        if not self.emotion_history:
            return {"status": "No facial data available"}
        
        # Count emotions
        emotion_counts = Counter([entry['emotion'] for entry in self.emotion_history])
        total_entries = len(self.emotion_history)
        
        # Calculate percentages
        emotion_percentages = {
            emotion: (count / total_entries) * 100 
            for emotion, count in emotion_counts.items()
        }
        
        # Get dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Calculate average confidence
        avg_confidence = sum(entry['confidence'] for entry in self.emotion_history) / total_entries
        
        # Get time range
        if len(self.emotion_history) > 1:
            start_time = self.emotion_history[0]['timestamp']
            end_time = self.emotion_history[-1]['timestamp']
            duration = (end_time - start_time).total_seconds() / 60  # in minutes
        else:
            duration = 0
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_percentages": emotion_percentages,
            "average_confidence": avg_confidence,
            "duration_minutes": duration,
            "total_detections": total_entries
        }
# --- END OF INTEGRATED CAMERA MODULE ---


class RealTimeMindSync:
    def __init__(self):
        self.is_listening = False
        self.privacy_mode = True
        self.conversation_history = deque(maxlen=20)
        self.current_emotion = 'neutral'
        self.emotion_confidence = 0.5
        self.connected_clients = 0
        self.current_language = 'en'
        self.session_topics = set()
        self.last_response_time = 0
        self.response_cooldown = 1.0
        self.last_user_input = ""
        self.last_bot_response = ""
        self.user_name = None
        self.user_preferences = {}
        self.conversation_context = {
            'user_feelings': [], 'topics_discussed': [], 'emotional_trend': [],
            'last_responses': deque(maxlen=10), 'conversation_start_time': datetime.now(), 'total_exchanges': 0
        }
        self.response_history = defaultdict(int)
        self.recent_responses = deque(maxlen=5)
        
        self.features = {'voice_recognition': True, 'camera': False, 'privacy': True, 'voice_text_display': True, 'emotion_detection': True, 'crisis_alert': True, 'history': True}
        
        self.speech_recognition_languages = {
            'en': 'en-US', 'en-uk': 'en-GB', 'hi': 'hi-IN', 'ta': 'ta-IN', 'te': 'te-IN', 'bn': 'bn-BD', 
            'es': 'es-ES', 'fr': 'fr-FR', 'de': 'de-DE', 'it': 'it-IT', 'pt': 'pt-BR', 'ru': 'ru-RU',
            'zh': 'zh-CN', 'ja': 'ja-JP', 'ar': 'ar-SA',
            # Additional Indian languages
            'gu': 'gu-IN', 'kn': 'kn-IN', 'ml': 'ml-IN', 'mr': 'mr-IN', 'pa': 'pa-IN', 'ur': 'ur-IN'
        }
        
        # Updated language mappings to include all supported languages
        self.available_languages = {
            'en': 'English (US)', 'en-uk': 'English (UK)', 'fr': 'Français (French)', 'de': 'Deutsch (German)', 
            'es': 'Español (Spanish)', 'it': 'Italiano (Italian)', 'pt': 'Português (Portuguese)', 
            'ru': 'Русский (Russian)', 'zh': '中文 (Chinese Mandarin)', 'ja': '日本語 (Japanese)', 
            'hi': 'हिन्दी (Hindi)', 'ar': 'العربية (Arabic)',
            # Additional Indian languages
            'ta': 'தமிழ் (Tamil)', 'te': 'తెలుగు (Telugu)', 'bn': 'বাংলা (Bengali)',
            'gu': 'ગુજરાતી (Gujarati)', 'kn': 'ಕನ್ನಡ (Kannada)', 'ml': 'മലയാളം (Malayalam)',
            'mr': 'मराठी (Marathi)', 'pa': 'ਪੰਜਾਬੀ (Punjabi)', 'ur': 'اردو (Urdu)'
        }
        
        # Google Translate language codes
        self.google_translate_codes = {
            'en': 'en', 'en-uk': 'en', 'fr': 'fr', 'de': 'de', 'es': 'es', 'it': 'it', 
            'pt': 'pt', 'ru': 'ru', 'zh': 'zh', 'ja': 'ja', 'hi': 'hi', 'ar': 'ar',
            # Additional Indian languages
            'ta': 'ta', 'te': 'te', 'bn': 'bn', 'gu': 'gu', 'kn': 'kn', 'ml': 'ml',
            'mr': 'mr', 'pa': 'pa', 'ur': 'ur'
        }
        
        self.initialize_speech_components()
        
        # Initialize Google Translator
        try:
            self.translator = Translator()
            logger.info("Google Translator initialized successfully")
        except Exception as e:
            logger.error(f"Google Translator initialization error: {e}")
            self.translator = None
        
        # --- MULTI-LANGUAGE RESPONSES ---
        self.crisis_keywords = {
            'en': ['suicide', 'kill myself', 'end it all', 'no point living', 'want to die', 'hurt myself', 'self harm', 'ending my life'],
            'hi': ['आत्महत्या', 'खुद को मारना', 'मरना चाहता हूं', 'जीने का मन नहीं', 'सुसाइड'],
            'es': ['suicidio', 'matarme', 'acabar con todo', 'no quiero vivir', 'quiero morir'],
            'fr': ['suicide', 'se tuer', 'en finir', 'je ne veux plus vivre', 'je veux mourir'],
            'de': ['Selbstmord', 'mich umbringen', 'alles beenden', 'will nicht mehr leben', 'will sterben'],
            'ar': ['انتحار', 'قتل نفسي', 'إنهاء كل شيء', 'لا فائدة من الحياة', 'أريد الموت'],
            'ru': ['самоубийство', 'убить себя', 'положить всему конец', 'нет смысла жить', 'хочу умереть'],
            'zh': ['自杀', '杀死自己', '结束一切', '活着没意义', '想死'],
            'ja': ['自殺', '自分を殺す', 'すべてを終わらせる', '生きる意味がない', '死にたい'],
            'it': ['suicidio', 'uccidermi', 'finire tutto', 'nessun punto nel vivere', 'voglio morire'],
            'pt': ['suicídio', 'matar a mim mesmo', 'acabar com tudo', 'sem sentido viver', 'quero morrer'],
            # Additional Indian languages
            'ta': ['தற்கொலை', 'நானே கொல்வது', 'எல்லாவற்றையும் முடிப்பது', 'வாழ்வதில் பயனில்லை', 'இறக்க விரும்புகிறேன்'],
            'te': ['ఆత్మహత్య', 'నేను నాను చంపుకోవడం', 'అన్నింటినీ ముగించడం', 'జీవించడానికి అర్థం లేదు', 'చనిపోవాలనుకుంటున్నాను'],
            'bn': ['আত্মহত্যা', 'নিজেকে মারা', 'সবকিছু শেষ করা', 'বেঁচে থাকার কোনো মানে নেই', 'আমি মারা যেতে চাই'],
            'gu': ['આત્મહત્યા', 'હું મારે છું', 'બધું સમાપ્ત કરવું', 'જીવવાનો કોઈ અર્થ નથી', 'હું મરવા માંગુ છું'],
            'kn': ['ಆತ್ಮಹತ್ಯೆ', 'ನಾನು ನನ್ನನ್ನು ಕೊಲ್ಲುವುದು', 'ಎಲ್ಲವನ್ನೂ ಮುಗಿಸುವುದು', 'ಬದುಕುವುದಕ್ಕೆ ಅರ್ಥವಿಲ್ಲ', 'ನಾನು ಸಾಯಲು ಬಯಸುತ್ತೇನೆ'],
            'ml': ['ആത്മഹത്യ', 'ഞാൻ സ്വയം കൊല്ലുക', 'എല്ലാം അവസാനിപ്പിക്കുക', 'ജീവിക്കാൻ അർത്ഥമില്ല', 'ഞാൻ മരിക്കാൻ ആഗ്രഹിക്കുന്നു'],
            'mr': ['आत्महत्या', 'मी स्वतःचा बळी देतो', 'सर्वकाही संपवणे', 'जगण्याचा कोणताही अर्थ नाही', 'मी मरू इच्छितो'],
            'pa': ['ਆਤਮ-ਹੱਤਿਆ', 'ਮੈਂ ਆਪਣੇ ਆਪ ਨੂੰ ਮਾਰਦਾ ਹਾਂ', 'ਸਭ ਕੁਝ ਖਤਮ ਕਰਨਾ', 'ਜੀਣ ਦਾ ਕੋਈ ਮਤਲਬ ਨਹੀਂ', 'ਮੈਂ ਮਰਨਾ ਚਾਹੁੰਦਾ ਹਾਂ'],
            'ur': ['خودکشی', 'میں اپنے آپ کو ماروں گا', 'سبھی کچھ ختم کرنا', 'زندگی کرنے کا کوئی فائدہ نہیں', 'میں مرنا چاہتا ہوں']
        }

        self.supportive_responses = {
            'depression': [
                "I hear you're going through a tough time. Remember, storms don't last forever. You're stronger than you realize.",
                "It's okay to feel this way. Even the darkest night will end and the sun will rise. Small steps forward are still progress."
            ],
            'anxiety': [
                "I understand you're feeling anxious. Remember, anxiety is like a rocking chair - it gives you something to do but doesn't get you anywhere. Let's focus on what you can control.",
                "Breathe deeply. Anxiety is temporary, but your strength is permanent. You've overcome challenges before, and you can do it again."
            ],
            'positive': [
                "That's wonderful to hear! It's great that you're feeling positive today.",
                "I'm glad you're feeling good! Positive emotions are worth celebrating."
            ],
            'neutral': [
                "Thank you for sharing with me. Every conversation matters, and I'm here to listen and support you.",
                "I appreciate you taking time to check in with your mental health. Self-awareness is the first step to wellbeing."
            ]
        }
        
        self.crisis_responses = [
            "I'm really concerned about you right now. Please reach out for immediate help. In the US, you can call or text 988 for the Suicide & Crisis Lifeline.",
            "It sounds like you're going through an extremely difficult time. Please don't go through this alone. There are people who want to help you. You can connect with trained counselors by calling 988."
        ]

        self.greetings = {
            'morning': "Good morning",
            'afternoon': "Good afternoon",
            'evening': "Good evening"
        }
        
        # Text templates in English (base language)
        self.text_templates = {
            'greeting_followup': "I'm here to listen and support you. How are you feeling today?",
            'nice_to_meet_you': "It's nice to meet you, {name}! I'm here to help you feel better. What's on your mind?",
            'anxiety_opening': "I hear you're dealing with anxiety. That can be really challenging. Would you like to tell me more about what's making you anxious?",
            'depression_opening': "I'm sorry to hear you're feeling down. Depression can be overwhelming, but you're not alone. What's been most difficult for you lately?",
            'sad_opening': "Thank you for sharing that with me. It takes courage to open up about difficult feelings. What's been weighing on you?",
            'happy_opening': "I'm glad to hear something positive! It's wonderful when we have good moments. What's been bringing you joy?",
            'neutral_opening': "I appreciate you sharing that with me. I'm here to support you through whatever you're experiencing. Can you tell me more?",
            'repeated_input': "I notice you mentioned that before. It must be really important to you. Would you like to explore it more deeply?",
            'anxiety_support': "Anxiety can feel so overwhelming. What does your anxiety feel like in your body right now?",
            'depression_support': "Depression can feel like a heavy blanket. What's been the hardest part of your day recently?",
            'general_support': "Thank you for sharing that with me. I'm here to listen and support you.",
            'emotional_support': "I can hear the {emotion} in your words. It takes courage to share these feelings. I'm here with you in this moment.",
            'positive_response': "I love hearing the {emotion} in your voice! It's wonderful when we have these bright moments. What's contributing to these good feelings?",
            'question_response': "That's a thoughtful question. What are your own thoughts about it?",
            'perspective_response': "Thank you for sharing your perspective. What makes you feel that way?",
            'neutral_followup': "Thank you for sharing that with me. When you say '{user_input}', how does that make you feel?",
            'report_generated': "Your mental health report has been generated based on our conversation and facial expression analysis. Please review it carefully.",
            'report_crisis': "Your report indicates you may be experiencing significant mental health challenges. Please consider reaching out to a mental health professional.",
            'report_positive': "Your report shows positive mental health indicators. Keep up the good work with these practices!",
            'report_mixed': "Your report shows mixed mental health indicators. Consider focusing on the areas where you're struggling.",
            'report_neutral': "Your report shows neutral mental health indicators. Consider exploring your feelings more deeply."
        }
        
        # UI text translations for all supported languages
        self.ui_translations = {
            'en': {
                'app_title': 'MindSync',
                'app_subtitle': 'Mental Health Assistant',
                'connecting': 'Connecting...',
                'connected': 'Connected',
                'disconnected': 'Disconnected',
                'mindsync_assistant': 'MindSync Assistant',
                'type_message': 'Type your message...',
                'send': 'Send',
                'voice_recognition': 'Voice Recognition',
                'ready': 'Ready',
                'listening': 'Listening...',
                'click_mic': 'Click the microphone to start speaking',
                'speak_now': 'Speak now. Click the microphone again to stop.',
                'controls': 'Controls',
                'camera': 'Camera',
                'camera_on': 'On',
                'camera_off': 'Off',
                'stop_camera': 'Stop Camera',
                'privacy': 'Privacy',
                'privacy_on': 'On',
                'privacy_off': 'Off',
                'history': 'History',
                'hide_history': 'Hide History',
                'generate_report': 'Generate Report',
                'test_connection': 'Test Connection',
                'emotion_analysis': 'Emotion Analysis',
                'text_based': 'Text-based:',
                'confidence': 'Confidence:',
                'facial': 'Facial:',
                'emotion_distribution': 'Emotion Distribution',
                'topics_discussed': 'Topics Discussed',
                'facial_expression_analysis': 'Facial Expression Analysis',
                'dominant_facial_emotion': 'Dominant Facial Emotion',
                'average_confidence': 'Average Confidence',
                'facial_analysis_duration': 'Facial Analysis Duration',
                'recommendations': 'Recommendations',
                'overall_assessment': 'Overall Assessment',
                'summary': 'Summary',
                'session_duration': 'Session Duration',
                'total_exchanges': 'Total Exchanges',
                'dominant_emotion': 'Dominant Emotion',
                'emotional_trend': 'Emotional Trend',
                'risk_level': 'Risk Level',
                'report_generated_on': 'Report generated on:',
                'minutes': 'minutes',
                'error_microphone': 'Microphone is not available. Please check your microphone settings.',
                'error_permission': 'Microphone permission was denied. Please allow microphone access.',
                'error_camera': 'Could not access camera. Please check permissions.',
                'error_camera_denied': 'Camera access denied. Please allow camera permissions and try again.',
                'error_camera_not_found': 'No camera found. Please connect a camera and try again.',
                'error_camera_in_use': 'Camera is already in use by another application.',
                'error_browser_support': 'Your browser does not support speech recognition. Please try using Chrome, Edge, or Safari.',
                'error_occurred': 'Error occurred in recognition:',
                'no_speech': 'No speech was detected. Please try again.',
                'network_error': 'Network error occurred. Please check your internet connection.',
                'webcam_test_successful': 'WebSocket test successful:',
                'test_message_sent': 'WebSocket test message sent. Check console for response.',
                'improving': 'improving',
                'declining': 'declining',
                'stable': 'stable',
                'insufficient_data': 'insufficient_data',
                'low': 'low',
                'medium': 'medium',
                'high': 'high'
            },
            'ta': {
                'app_title': 'மைண்ட்சிங்க்',
                'app_subtitle': 'மன ஆரோக்கிய உதவியாளர்',
                'connecting': 'இணைக்கிறது...',
                'connected': 'இணைக்கப்பட்டது',
                'disconnected': 'துண்டிக்கப்பட்டது',
                'mindsync_assistant': 'மைண்ட்சிங்க் உதவியாளர்',
                'type_message': 'உங்கள் செய்தியை தட்டச்சு செய்யவும்...',
                'send': 'அனுப்பு',
                'voice_recognition': 'குரல் அங்கீகாரம்',
                'ready': 'தயார்',
                'listening': 'கேட்கிறது...',
                'click_mic': 'பேச மைக்ரோஃபோனை கிளிக் செய்யவும்',
                'speak_now': 'இப்போது பேசுங்கள். நிறுத்த மீண்டும் மைக்ரோஃபோனை கிளிக் செய்யவும்.',
                'controls': 'கட்டுப்பாடுகள்',
                'camera': 'கேமரா',
                'camera_on': 'ஆன்',
                'camera_off': 'ஆஃப்',
                'stop_camera': 'கேமராவை நிறுத்து',
                'privacy': 'தனியுரிமை',
                'privacy_on': 'ஆன்',
                'privacy_off': 'ஆஃப்',
                'history': 'வரலாறு',
                'hide_history': 'வரலாற்றை மறை',
                'generate_report': 'அறிக்கையை உருவாக்கு',
                'test_connection': 'இணைப்பை சோதிக்க',
                'emotion_analysis': 'உணர்ச்சி பகுப்பாய்வு',
                'text_based': 'உரை அடிப்படையில்:',
                'confidence': 'நம்பிக்கை:',
                'facial': 'முக வெளிப்பாடு:',
                'emotion_distribution': 'உணர்ச்சி விநியோகம்',
                'topics_discussed': 'விவாதிக்கப்பட்ட தலைப்புகள்',
                'facial_expression_analysis': 'முக வெளிப்பாடு பகுப்பாய்வு',
                'dominant_facial_emotion': 'முக்கிய முக உணர்ச்சி',
                'average_confidence': 'சராசரி நம்பிக்கை',
                'facial_analysis_duration': 'முக பகுப்பாய்வு காலம்',
                'recommendations': 'பரிந்துரைகள்',
                'overall_assessment': 'ஒட்டுமொத்த மதிப்பாய்வு',
                'summary': 'சுருக்கம்',
                'session_duration': 'அமர்வு காலம்',
                'total_exchanges': 'மொத்த பரிமாற்றங்கள்',
                'dominant_emotion': 'முக்கிய உணர்ச்சி',
                'emotional_trend': 'உணர்ச்சி போக்கு',
                'risk_level': 'ஆபத்து நிலை',
                'report_generated_on': 'அறிக்கை உருவாக்கப்பட்டது:',
                'minutes': 'நிமிடங்கள்',
                'error_microphone': 'மைக்ரோஃபோன் கிடைக்கவில்லை. தயவுசெய்து உங்கள் மைக்ரோஃபோன் அமைப்புகளை சரிபார்க்கவும்.',
                'error_permission': 'மைக்ரோஃபோன் அனுமதி மறுக்கப்பட்டது. தயவுசெய்து மைக்ரோஃபோன் அணுகலை அனுமதிக்கவும்.',
                'error_camera': 'கேமராவை அணுக முடியவில்லை. தயவுசெய்து அனுமதிகளை சரிபார்க்கவும்.',
                'error_camera_denied': 'கேமரா அணுகல் மறுக்கப்பட்டது. தயவுசெய்து கேமரா அனுமதிகளை அனுமதித்து மீண்டும் முயற்சிக்கவும்.',
                'error_camera_not_found': 'கேமரா கிடைக்கவில்லை. தயவுசெய்து ஒரு கேமராவை இணைத்து மீண்டும் முயற்சிக்கவும்.',
                'error_camera_in_use': 'கேமரா ஏற்கனவே மற்றொரு பயன்பாட்டில் பயன்படுத்தப்படுகிறது.',
                'error_browser_support': 'உங்கள் உலாவி குரல் அங்கீகாரத்தை ஆதரிக்கவில்லை. தயவுசெய்து குரோம், எட்ஜ் அல்லது சஃபாரியைப் பயன்படுத்த முயற்சிக்கவும்.',
                'error_occurred': 'அங்கீகாரத்தில் பிழை ஏற்பட்டது:',
                'no_speech': 'பேச்சு கண்டறியப்படவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.',
                'network_error': 'நெட்வொர்க் பிழை ஏற்பட்டது. தயவுசெய்து உங்கள் இணைய இணைப்பை சரிபார்க்கவும்.',
                'webcam_test_successful': 'வெப்சாக்கெட் சோதனை வெற்றிகரமாக முடிந்தது:',
                'test_message_sent': 'வெப்சாக்கெட் சோதனை செய்தி அனுப்பப்பட்டது. பதிலுக்கு கன்சோலை சரிபார்க்கவும்.',
                'improving': 'மேம்படுகிறது',
                'declining': 'குறைகிறது',
                'stable': 'நிலையானது',
                'insufficient_data': 'போதுமான தரவு இல்லை',
                'low': 'குறைந்த',
                'medium': 'நடுத்தர',
                'high': 'உயர்ந்த'
            },
            'hi': {
                'app_title': 'माइंडसिंक',
                'app_subtitle': 'मानसिक स्वास्थ्य सहायक',
                'connecting': 'कनेक्ट हो रहा है...',
                'connected': 'कनेक्ट हो गया',
                'disconnected': 'डिस्कनेक्ट हो गया',
                'mindsync_assistant': 'माइंडसिंक सहायक',
                'type_message': 'अपना संदेश टाइप करें...',
                'send': 'भेजें',
                'voice_recognition': 'आवाज़ पहचान',
                'ready': 'तैयार',
                'listening': 'सुन रहा है...',
                'click_mic': 'बोलने के लिए माइक्रोफोन पर क्लिक करें',
                'speak_now': 'अभी बोलें। रुकने के लिए फिर से माइक्रोफोन पर क्लिक करें।',
                'controls': 'नियंत्रण',
                'camera': 'कैमरा',
                'camera_on': 'चालू',
                'camera_off': 'बंद',
                'stop_camera': 'कैमरा बंद करें',
                'privacy': 'गोपनीयता',
                'privacy_on': 'चालू',
                'privacy_off': 'बंद',
                'history': 'इतिहास',
                'hide_history': 'इतिहास छिपाएं',
                'generate_report': 'रिपोर्ट जेनरेट करें',
                'test_connection': 'कनेक्शन टेस्ट करें',
                'emotion_analysis': 'भावना विश्लेषण',
                'text_based': 'टेक्स्ट आधारित:',
                'confidence': 'विश्वास:',
                'facial': 'चेहरे का:',
                'emotion_distribution': 'भावना वितरण',
                'topics_discussed': 'चर्चा किए गए विषय',
                'facial_expression_analysis': 'चेहरे की अभिव्यक्ति विश्लेषण',
                'dominant_facial_emotion': 'प्रमुख चेहरे की भावना',
                'average_confidence': 'औसत विश्वास',
                'facial_analysis_duration': 'चेहरा विश्लेषण अवधि',
                'recommendations': 'सिफारिशें',
                'overall_assessment': 'समग्र मूल्यांकन',
                'summary': 'सारांश',
                'session_duration': 'सत्र अवधि',
                'total_exchanges': 'कुल एक्सचेंज',
                'dominant_emotion': 'प्रमुख भावना',
                'emotional_trend': 'भावनात्मक प्रवृत्ति',
                'risk_level': 'जोखिम स्तर',
                'report_generated_on': 'रिपोर्ट जेनरेट की गई:',
                'minutes': 'मिनट',
                'error_microphone': 'माइक्रोफोन उपलब्ध नहीं है। कृपया अपने माइक्रोफोन सेटिंग्स जांचें।',
                'error_permission': 'माइक्रोफोन अनुमति अस्वीकार कर दी गई। कृपया माइक्रोफोन एक्सेस की अनुमति दें।',
                'error_camera': 'कैमरा एक्सेस नहीं किया जा सका। कृपया अनुमतियां जांचें।',
                'error_camera_denied': 'कैमरा एक्सेस अस्वीकार कर दिया गया। कृपया कैमरा अनुमतियां दें और फिर से कोशिश करें।',
                'error_camera_not_found': 'कोई कैमरा नहीं मिला। कृपया एक कैमरा कनेक्ट करें और फिर से कोशिश करें।',
                'error_camera_in_use': 'कैमरा पहले से ही दूसरे एप्लिकेशन द्वारा उपयोग में है।',
                'error_browser_support': 'आपका ब्राउज़र वॉयस रिकग्निशन का समर्थन नहीं करता। कृपया क्रोम, एज या सफारी का उपयोग करने का प्रयास करें।',
                'error_occurred': 'पहचान में त्रुटि हुई:',
                'no_speech': 'कोई भाषण का पता नहीं चला। कृपया फिर से कोशिश करें।',
                'network_error': 'नेटवर्क त्रुटि हुई। कृपया अपना इंटरनेट कनेक्शन जांचें।',
                'webcam_test_successful': 'वेबसॉकेट टेस्ट सफल:',
                'test_message_sent': 'वेबसॉकेट टेस्ट संदेश भेजा गया। प्रतिक्रिया के लिए कंसोल जांचें।',
                'improving': 'सुधार हो रहा है',
                'declining': 'गिर रहा है',
                'stable': 'स्थिर',
                'insufficient_data': 'अपर्याप्त डेटा',
                'low': 'कम',
                'medium': 'मध्यम',
                'high': 'उच्च'
            },
            'es': {
                'app_title': 'MindSync',
                'app_subtitle': 'Asistente de Salud Mental',
                'connecting': 'Conectando...',
                'connected': 'Conectado',
                'disconnected': 'Desconectado',
                'mindsync_assistant': 'Asistente MindSync',
                'type_message': 'Escribe tu mensaje...',
                'send': 'Enviar',
                'voice_recognition': 'Reconocimiento de Voz',
                'ready': 'Listo',
                'listening': 'Escuchando...',
                'click_mic': 'Haz clic en el micrófono para empezar a hablar',
                'speak_now': 'Habla ahora. Vuelve a hacer clic en el micrófono para detener.',
                'controls': 'Controles',
                'camera': 'Cámara',
                'camera_on': 'Encendida',
                'camera_off': 'Apagada',
                'stop_camera': 'Detener Cámara',
                'privacy': 'Privacidad',
                'privacy_on': 'Activada',
                'privacy_off': 'Desactivada',
                'history': 'Historial',
                'hide_history': 'Ocultar Historial',
                'generate_report': 'Generar Informe',
                'test_connection': 'Probar Conexión',
                'emotion_analysis': 'Análisis de Emociones',
                'text_based': 'Basado en texto:',
                'confidence': 'Confianza:',
                'facial': 'Facial:',
                'emotion_distribution': 'Distribución de Emociones',
                'topics_discussed': 'Temas Discutidos',
                'facial_expression_analysis': 'Análisis de Expresiones Faciales',
                'dominant_facial_emotion': 'Emoción Facial Dominante',
                'average_confidence': 'Confianza Promedio',
                'facial_analysis_duration': 'Duración del Análisis Facial',
                'recommendations': 'Recomendaciones',
                'overall_assessment': 'Evaluación General',
                'summary': 'Resumen',
                'session_duration': 'Duración de la Sesión',
                'total_exchanges': 'Intercambios Totales',
                'dominant_emotion': 'Emoción Dominante',
                'emotional_trend': 'Tendencia Emocional',
                'risk_level': 'Nivel de Riesgo',
                'report_generated_on': 'Informe generado en:',
                'minutes': 'minutos',
                'error_microphone': 'El micrófono no está disponible. Por favor, verifica la configuración de tu micrófono.',
                'error_permission': 'Se denegó el permiso del micrófono. Por favor, permite el acceso al micrófono.',
                'error_camera': 'No se pudo acceder a la cámara. Por favor, verifica los permisos.',
                'error_camera_denied': 'Se denegó el acceso a la cámara. Por favor, permite los permisos de la cámara e inténtalo de nuevo.',
                'error_camera_not_found': 'No se encontró ninguna cámara. Por favor, conecta una cámara e inténtalo de nuevo.',
                'error_camera_in_use': 'La cámara ya está siendo utilizada por otra aplicación.',
                'error_browser_support': 'Tu navegador no soporta el reconocimiento de voz. Por favor, intenta usar Chrome, Edge o Safari.',
                'error_occurred': 'Ocurrió un error en el reconocimiento:',
                'no_speech': 'No se detectó habla. Por favor, inténtalo de nuevo.',
                'network_error': 'Ocurrió un error de red. Por favor, verifica tu conexión a internet.',
                'webcam_test_successful': 'Prueba de WebSocket exitosa:',
                'test_message_sent': 'Mensaje de prueba de WebSocket enviado. Revisa la consola para la respuesta.',
                'improving': 'mejorando',
                'declining': 'empeorando',
                'stable': 'estable',
                'insufficient_data': 'datos_insuficientes',
                'low': 'bajo',
                'medium': 'medio',
                'high': 'alto'
            },
            'fr': {
                'app_title': 'MindSync',
                'app_subtitle': 'Assistant de Santé Mentale',
                'connecting': 'Connexion...',
                'connected': 'Connecté',
                'disconnected': 'Déconnecté',
                'mindsync_assistant': 'Assistant MindSync',
                'type_message': 'Tapez votre message...',
                'send': 'Envoyer',
                'voice_recognition': 'Reconnaissance Vocale',
                'ready': 'Prêt',
                'listening': 'Écoute...',
                'click_mic': 'Cliquez sur le microphone pour commencer à parler',
                'speak_now': 'Parlez maintenant. Cliquez à nouveau sur le microphone pour arrêter.',
                'controls': 'Contrôles',
                'camera': 'Caméra',
                'camera_on': 'Activée',
                'camera_off': 'Désactivée',
                'stop_camera': 'Arrêter la Caméra',
                'privacy': 'Confidentialité',
                'privacy_on': 'Activée',
                'privacy_off': 'Désactivée',
                'history': 'Historique',
                'hide_history': 'Masquer l\'Historique',
                'generate_report': 'Générer un Rapport',
                'test_connection': 'Tester la Connexion',
                'emotion_analysis': 'Analyse des Émotions',
                'text_based': 'Basé sur le texte:',
                'confidence': 'Confiance:',
                'facial': 'Facial:',
                'emotion_distribution': 'Distribution des Émotions',
                'topics_discussed': 'Sujets Discutés',
                'facial_expression_analysis': 'Analyse des Expressions Faciales',
                'dominant_facial_emotion': 'Émotion Faciale Dominante',
                'average_confidence': 'Confiance Moyenne',
                'facial_analysis_duration': 'Durée de l\'Analyse Faciale',
                'recommendations': 'Recommandations',
                'overall_assessment': 'Évaluation Générale',
                'summary': 'Résumé',
                'session_duration': 'Durée de la Session',
                'total_exchanges': 'Échanges Totaux',
                'dominant_emotion': 'Émotion Dominante',
                'emotional_trend': 'Tendance Émotionnelle',
                'risk_level': 'Niveau de Risque',
                'report_generated_on': 'Rapport généré le:',
                'minutes': 'minutes',
                'error_microphone': 'Le microphone n\'est pas disponible. Veuillez vérifier les paramètres de votre microphone.',
                'error_permission': 'L\'autorisation du microphone a été refusée. Veuillez autoriser l\'accès au microphone.',
                'error_camera': 'Impossible d\'accéder à la caméra. Veuillez vérifier les autorisations.',
                'error_camera_denied': 'L\'accès à la caméra a été refusé. Veuillez autoriser les permissions de la caméra et réessayer.',
                'error_camera_not_found': 'Aucune caméra trouvée. Veuillez connecter une caméra et réessayer.',
                'error_camera_in_use': 'La caméra est déjà utilisée par une autre application.',
                'error_browser_support': 'Votre navigateur ne supporte pas la reconnaissance vocale. Veuillez essayer d\'utiliser Chrome, Edge ou Safari.',
                'error_occurred': 'Erreur lors de la reconnaissance:',
                'no_speech': 'Aucune parole détectée. Veuillez réessayer.',
                'network_error': 'Une erreur réseau s\'est produite. Veuillez vérifier votre connexion internet.',
                'webcam_test_successful': 'Test WebSocket réussi:',
                'test_message_sent': 'Message de test WebSocket envoyé. Vérifiez la console pour la réponse.',
                'improving': 'en amélioration',
                'declining': 'en déclin',
                'stable': 'stable',
                'insufficient_data': 'données_insuffisantes',
                'low': 'faible',
                'medium': 'moyen',
                'high': 'élevé'
            },
            'de': {
                'app_title': 'MindSync',
                'app_subtitle': 'Mentalgesundheits-Assistent',
                'connecting': 'Verbinden...',
                'connected': 'Verbunden',
                'disconnected': 'Getrennt',
                'mindsync_assistant': 'MindSync-Assistent',
                'type_message': 'Nachricht eingeben...',
                'send': 'Senden',
                'voice_recognition': 'Spracherkennung',
                'ready': 'Bereit',
                'listening': 'Hört zu...',
                'click_mic': 'Klicken Sie auf das Mikrofon, um zu sprechen',
                'speak_now': 'Sprechen Sie jetzt. Klicken Sie erneut auf das Mikrofon, um zu stoppen.',
                'controls': 'Steuerelemente',
                'camera': 'Kamera',
                'camera_on': 'Ein',
                'camera_off': 'Aus',
                'stop_camera': 'Kamera stoppen',
                'privacy': 'Datenschutz',
                'privacy_on': 'Ein',
                'privacy_off': 'Aus',
                'history': 'Verlauf',
                'hide_history': 'Verlauf ausblenden',
                'generate_report': 'Bericht erstellen',
                'test_connection': 'Verbindung testen',
                'emotion_analysis': 'Emotionsanalyse',
                'text_based': 'Textbasiert:',
                'confidence': 'Vertrauen:',
                'facial': 'Gesicht:',
                'emotion_distribution': 'Emotionsverteilung',
                'topics_discussed': 'Diskutierte Themen',
                'facial_expression_analysis': 'Gesichtsausdrucksanalyse',
                'dominant_facial_emotion': 'Dominante Gesichtsemotion',
                'average_confidence': 'Durchschnittliches Vertrauen',
                'facial_analysis_duration': 'Dauer der Gesichtsanalyse',
                'recommendations': 'Empfehlungen',
                'overall_assessment': 'Gesamtbewertung',
                'summary': 'Zusammenfassung',
                'session_duration': 'Sitzungsdauer',
                'total_exchanges': 'Gesamtaustausch',
                'dominant_emotion': 'Dominante Emotion',
                'emotional_trend': 'Emotionale Tendenz',
                'risk_level': 'Risikostufe',
                'report_generated_on': 'Bericht erstellt am:',
                'minutes': 'Minuten',
                'error_microphone': 'Mikrofon nicht verfügbar. Bitte überprüfen Sie Ihre Mikrofoneinstellungen.',
                'error_permission': 'Mikrofonberechtigung verweigert. Bitte erlauben Sie den Mikrofonzugriff.',
                'error_camera': 'Kein Zugriff auf die Kamera möglich. Bitte überprüfen Sie die Berechtigungen.',
                'error_camera_denied': 'Kamerazugriff verweigert. Bitte erlauben Sie die Kameraberechtigungen und versuchen Sie es erneut.',
                'error_camera_not_found': 'Keine Kamera gefunden. Bitte verbinden Sie eine Kamera und versuchen Sie es erneut.',
                'error_camera_in_use': 'Die Kamera wird bereits von einer anderen Anwendung verwendet.',
                'error_browser_support': 'Ihr Browser unterstützt die Spracherkennung nicht. Bitte versuchen Sie, Chrome, Edge oder Safari zu verwenden.',
                'error_occurred': 'Fehler bei der Erkennung:',
                'no_speech': 'Keine Sprache erkannt. Bitte versuchen Sie es erneut.',
                'network_error': 'Netzwerkfehler aufgetreten. Bitte überprüfen Sie Ihre Internetverbindung.',
                'webcam_test_successful': 'WebSocket-Test erfolgreich:',
                'test_message_sent': 'WebSocket-Testnachricht gesendet. Überprüfen Sie die Konsole auf die Antwort.',
                'improving': 'verbessernd',
                'declining': 'abnehmend',
                'stable': 'stabil',
                'insufficient_data': 'unzureichende_daten',
                'low': 'niedrig',
                'medium': 'mittel',
                'high': 'hoch'
            },
            'zh': {
                'app_title': 'MindSync',
                'app_subtitle': '心理健康助手',
                'connecting': '连接中...',
                'connected': '已连接',
                'disconnected': '已断开',
                'mindsync_assistant': 'MindSync助手',
                'type_message': '输入您的消息...',
                'send': '发送',
                'voice_recognition': '语音识别',
                'ready': '准备就绪',
                'listening': '正在听...',
                'click_mic': '点击麦克风开始说话',
                'speak_now': '现在说话。再次点击麦克风停止。',
                'controls': '控制',
                'camera': '摄像头',
                'camera_on': '开启',
                'camera_off': '关闭',
                'stop_camera': '停止摄像头',
                'privacy': '隐私',
                'privacy_on': '开启',
                'privacy_off': '关闭',
                'history': '历史记录',
                'hide_history': '隐藏历史记录',
                'generate_report': '生成报告',
                'test_connection': '测试连接',
                'emotion_analysis': '情绪分析',
                'text_based': '基于文本:',
                'confidence': '信心:',
                'facial': '面部:',
                'emotion_distribution': '情绪分布',
                'topics_discussed': '讨论的主题',
                'facial_expression_analysis': '面部表情分析',
                'dominant_facial_emotion': '主要面部情绪',
                'average_confidence': '平均信心',
                'facial_analysis_duration': '面部分析持续时间',
                'recommendations': '建议',
                'overall_assessment': '总体评估',
                'summary': '摘要',
                'session_duration': '会话持续时间',
                'total_exchanges': '总交流次数',
                'dominant_emotion': '主要情绪',
                'emotional_trend': '情绪趋势',
                'risk_level': '风险等级',
                'report_generated_on': '报告生成于:',
                'minutes': '分钟',
                'error_microphone': '麦克风不可用。请检查您的麦克风设置。',
                'error_permission': '麦克风权限被拒绝。请允许麦克风访问。',
                'error_camera': '无法访问摄像头。请检查权限。',
                'error_camera_denied': '摄像头访问被拒绝。请允许摄像头权限并重试。',
                'error_camera_not_found': '未找到摄像头。请连接摄像头并重试。',
                'error_camera_in_use': '摄像头已被其他应用程序使用。',
                'error_browser_support': '您的浏览器不支持语音识别。请尝试使用Chrome、Edge或Safari。',
                'error_occurred': '识别中发生错误:',
                'no_speech': '未检测到语音。请重试。',
                'network_error': '发生网络错误。请检查您的互联网连接。',
                'webcam_test_successful': 'WebSocket测试成功:',
                'test_message_sent': 'WebSocket测试消息已发送。检查控制台以获取响应。',
                'improving': '改善中',
                'declining': '下降中',
                'stable': '稳定',
                'insufficient_data': '数据不足',
                'low': '低',
                'medium': '中',
                'high': '高'
            },
            'ja': {
                'app_title': 'MindSync',
                'app_subtitle': 'メンタルヘルスアシスタント',
                'connecting': '接続中...',
                'connected': '接続済み',
                'disconnected': '切断されました',
                'mindsync_assistant': 'MindSyncアシスタント',
                'type_message': 'メッセージを入力...',
                'send': '送信',
                'voice_recognition': '音声認識',
                'ready': '準備完了',
                'listening': '聞いています...',
                'click_mic': 'マイクをクリックして話し始める',
                'speak_now': '今話してください。停止するには再度マイクをクリックしてください。',
                'controls': 'コントロール',
                'camera': 'カメラ',
                'camera_on': 'オン',
                'camera_off': 'オフ',
                'stop_camera': 'カメラを停止',
                'privacy': 'プライバシー',
                'privacy_on': 'オン',
                'privacy_off': 'オフ',
                'history': '履歴',
                'hide_history': '履歴を非表示',
                'generate_report': 'レポートを生成',
                'test_connection': '接続をテスト',
                'emotion_analysis': '感情分析',
                'text_based': 'テキストベース:',
                'confidence': '信頼度:',
                'facial': '顔:',
                'emotion_distribution': '感情分布',
                'topics_discussed': '議論されたトピック',
                'facial_expression_analysis': '顔の表情分析',
                'dominant_facial_emotion': '主要な顔の感情',
                'average_confidence': '平均信頼度',
                'facial_analysis_duration': '顔分析の持続時間',
                'recommendations': '推奨事項',
                'overall_assessment': '全体的な評価',
                'summary': '要約',
                'session_duration': 'セッション期間',
                'total_exchanges': '総交流回数',
                'dominant_emotion': '主要な感情',
                'emotional_trend': '感情的傾向',
                'risk_level': 'リスクレベル',
                'report_generated_on': 'レポート生成日時:',
                'minutes': '分',
                'error_microphone': 'マイクが利用できません。マイク設定を確認してください。',
                'error_permission': 'マイクのアクセス許可が拒否されました。マイクへのアクセスを許可してください。',
                'error_camera': 'カメラにアクセスできません。権限を確認してください。',
                'error_camera_denied': 'カメラへのアクセスが拒否されました。カメラの権限を許可して再試行してください。',
                'error_camera_not_found': 'カメラが見つかりません。カメラを接続して再試行してください。',
                'error_camera_in_use': 'カメラは既に他のアプリケーションで使用されています。',
                'error_browser_support': 'お使いのブラウザは音声認識をサポートしていません。Chrome、Edge、またはSafariの使用をお試しください。',
                'error_occurred': '認識でエラーが発生しました:',
                'no_speech': '音声が検出されませんでした。もう一度お試しください。',
                'network_error': 'ネットワークエラーが発生しました。インターネット接続を確認してください。',
                'webcam_test_successful': 'WebSocketテスト成功:',
                'test_message_sent': 'WebSocketテストメッセージが送信されました。応答についてはコンソールを確認してください。',
                'improving': '改善中',
                'declining': '低下中',
                'stable': '安定',
                'insufficient_data': 'データ不足',
                'low': '低',
                'medium': '中',
                'high': '高'
            },
            'ar': {
                'app_title': 'مايندسينك',
                'app_subtitle': 'مساعد الصحة النفسية',
                'connecting': 'جاري الاتصال...',
                'connected': 'متصل',
                'disconnected': 'منقطع',
                'mindsync_assistant': 'مساعد مايندسينك',
                'type_message': 'اكتب رسالتك...',
                'send': 'إرسال',
                'voice_recognition': 'التعرف على الصوت',
                'ready': 'جاهز',
                'listening': 'يستمع...',
                'click_mic': 'انقر على الميكروفون لبدء التحدث',
                'speak_now': 'تحدث الآن. انقر مرة أخرى على الميكروفون للتوقف.',
                'controls': 'عناصر التحكم',
                'camera': 'الكاميرا',
                'camera_on': 'مفعلة',
                'camera_off': 'معطلة',
                'stop_camera': 'إيقاف الكاميرا',
                'privacy': 'الخصوصية',
                'privacy_on': 'مفعلة',
                'privacy_off': 'معطلة',
                'history': 'السجل',
                'hide_history': 'إخفاء السجل',
                'generate_report': 'إنشاء تقرير',
                'test_connection': 'اختبار الاتصال',
                'emotion_analysis': 'تحليل المشاعر',
                'text_based': 'قائم على النص:',
                'confidence': 'الثقة:',
                'facial': 'الوجه:',
                'emotion_distribution': 'توزيع المشاعر',
                'topics_discussed': 'المواضيع التي تمت مناقشتها',
                'facial_expression_analysis': 'تحليل تعابير الوجه',
                'dominant_facial_emotion': 'المشاعر الوجهية المهيمنة',
                'average_confidence': 'متوسط الثقة',
                'facial_analysis_duration': 'مدة تحليل الوجه',
                'recommendations': 'التوصيات',
                'overall_assessment': 'التقييم العام',
                'summary': 'ملخص',
                'session_duration': 'مدة الجلسة',
                'total_exchanges': 'إجمالي التبادلات',
                'dominant_emotion': 'المشاعر المهيمنة',
                'emotional_trend': 'الاتجاه العاطفي',
                'risk_level': 'مستوى المخاطرة',
                'report_generated_on': 'تم إنشاء التقرير في:',
                'minutes': 'دقائق',
                'error_microphone': 'الميكروفون غير متاح. يرجى التحقق من إعدادات الميكروفون.',
                'error_permission': 'تم رفض إذن الميكروفون. يرجى السماح بالوصول إلى الميكروفون.',
                'error_camera': 'لا يمكن الوصول إلى الكاميرا. يرجى التحقق من الأذونات.',
                'error_camera_denied': 'تم رفض الوصول إلى الكاميرا. يرجى السماح بأذونات الكاميرا والمحاولة مرة أخرى.',
                'error_camera_not_found': 'لم يتم العثور على كاميرا. يرجى توصيل كاميرا والمحاولة مرة أخرى.',
                'error_camera_in_use': 'الكاميرا قيد الاستخدام بالفعل بواسطة تطبيق آخر.',
                'error_browser_support': 'متصفحك لا يدعم التعرف على الصوت. يرجى محاولة استخدام Chrome أو Edge أو Safari.',
                'error_occurred': 'حدث خطأ في التعرف:',
                'no_speech': 'لم يتم اكتشاف كلام. يرجى المحاولة مرة أخرى.',
                'network_error': 'حدث خطأ في الشبكة. يرجى التحقق من اتصال الإنترنت الخاص بك.',
                'webcam_test_successful': 'اختبار WebSocket ناجح:',
                'test_message_sent': 'تم إرسال رسالة اختبار WebSocket. تحقق من وحدة التحكم للحصول على الرد.',
                'improving': 'يتحسن',
                'declining': 'يتدهور',
                'stable': 'مستقر',
                'insufficient_data': 'بيانات غير كافية',
                'low': 'منخفض',
                'medium': 'متوسط',
                'high': 'مرتفع'
            },
            'pt': {
                'app_title': 'MindSync',
                'app_subtitle': 'Assistente de Saúde Mental',
                'connecting': 'Conectando...',
                'connected': 'Conectado',
                'disconnected': 'Desconectado',
                'mindsync_assistant': 'Assistente MindSync',
                'type_message': 'Digite sua mensagem...',
                'send': 'Enviar',
                'voice_recognition': 'Reconhecimento de Voz',
                'ready': 'Pronto',
                'listening': 'Ouvindo...',
                'click_mic': 'Clique no microfone para começar a falar',
                'speak_now': 'Fale agora. Clique novamente no microfone para parar.',
                'controls': 'Controles',
                'camera': 'Câmera',
                'camera_on': 'Ligada',
                'camera_off': 'Desligada',
                'stop_camera': 'Parar Câmera',
                'privacy': 'Privacidade',
                'privacy_on': 'Ligada',
                'privacy_off': 'Desligada',
                'history': 'Histórico',
                'hide_history': 'Ocultar Histórico',
                'generate_report': 'Gerar Relatório',
                'test_connection': 'Testar Conexão',
                'emotion_analysis': 'Análise de Emoções',
                'text_based': 'Baseado em texto:',
                'confidence': 'Confiança:',
                'facial': 'Facial:',
                'emotion_distribution': 'Distribuição de Emoções',
                'topics_discussed': 'Tópicos Discutidos',
                'facial_expression_analysis': 'Análise de Expressões Faciais',
                'dominant_facial_emotion': 'Emoção Facial Dominante',
                'average_confidence': 'Confiança Média',
                'facial_analysis_duration': 'Duração da Análise Facial',
                'recommendations': 'Recomendações',
                'overall_assessment': 'Avaliação Geral',
                'summary': 'Resumo',
                'session_duration': 'Duração da Sessão',
                'total_exchanges': 'Trocas Totais',
                'dominant_emotion': 'Emoção Dominante',
                'emotional_trend': 'Tendência Emocional',
                'risk_level': 'Nível de Risco',
                'report_generated_on': 'Relatório gerado em:',
                'minutes': 'minutos',
                'error_microphone': 'Microfone não disponível. Por favor, verifique as configurações do seu microfone.',
                'error_permission': 'Permissão do microfone negada. Por favor, permita o acesso ao microfone.',
                'error_camera': 'Não foi possível acessar a câmera. Por favor, verifique as permissões.',
                'error_camera_denied': 'Acesso à câmera negado. Por favor, permita as permissões da câmera e tente novamente.',
                'error_camera_not_found': 'Nenhuma câmera encontrada. Por favor, conecte uma câmera e tente novamente.',
                'error_camera_in_use': 'A câmera já está sendo usada por outro aplicativo.',
                'error_browser_support': 'Seu navegador não suporta reconhecimento de voz. Por favor, tente usar Chrome, Edge ou Safari.',
                'error_occurred': 'Ocorreu um erro no reconhecimento:',
                'no_speech': 'Nenhuma fala detectada. Por favor, tente novamente.',
                'network_error': 'Ocorreu um erro de rede. Por favor, verifique sua conexão com a internet.',
                'webcam_test_successful': 'Teste de WebSocket bem-sucedido:',
                'test_message_sent': 'Mensagem de teste de WebSocket enviada. Verifique o console para a resposta.',
                'improving': 'melhorando',
                'declining': 'declinando',
                'stable': 'estável',
                'insufficient_data': 'dados_insuficientes',
                'low': 'baixo',
                'medium': 'médio',
                'high': 'alto'
            },
            'ru': {
                'app_title': 'MindSync',
                'app_subtitle': 'Помощник по психическому здоровью',
                'connecting': 'Подключение...',
                'connected': 'Подключено',
                'disconnected': 'Отключено',
                'mindsync_assistant': 'Помощник MindSync',
                'type_message': 'Введите ваше сообщение...',
                'send': 'Отправить',
                'voice_recognition': 'Распознавание голоса',
                'ready': 'Готов',
                'listening': 'Слушает...',
                'click_mic': 'Нажмите на микрофон, чтобы начать говорить',
                'speak_now': 'Говорите сейчас. Нажмите снова на микрофон, чтобы остановиться.',
                'controls': 'Управление',
                'camera': 'Камера',
                'camera_on': 'Включена',
                'camera_off': 'Выключена',
                'stop_camera': 'Остановить камеру',
                'privacy': 'Конфиденциальность',
                'privacy_on': 'Включена',
                'privacy_off': 'Выключена',
                'history': 'История',
                'hide_history': 'Скрыть историю',
                'generate_report': 'Создать отчет',
                'test_connection': 'Проверить соединение',
                'emotion_analysis': 'Анализ эмоций',
                'text_based': 'На основе текста:',
                'confidence': 'Уверенность:',
                'facial': 'Лицо:',
                'emotion_distribution': 'Распределение эмоций',
                'topics_discussed': 'Обсуждаемые темы',
                'facial_expression_analysis': 'Анализ выражения лица',
                'dominant_facial_emotion': 'Доминирующая эмоция лица',
                'average_confidence': 'Средняя уверенность',
                'facial_analysis_duration': 'Продолжительность анализа лица',
                'recommendations': 'Рекомендации',
                'overall_assessment': 'Общая оценка',
                'summary': 'Резюме',
                'session_duration': 'Продолжительность сессии',
                'total_exchanges': 'Общее количество обменов',
                'dominant_emotion': 'Доминирующая эмоция',
                'emotional_trend': 'Эмоциональный тренд',
                'risk_level': 'Уровень риска',
                'report_generated_on': 'Отчет создан:',
                'minutes': 'минуты',
                'error_microphone': 'Микрофон недоступен. Пожалуйста, проверьте настройки микрофона.',
                'error_permission': 'Разрешение на микрофон отклонено. Пожалуйста, разрешите доступ к микрофону.',
                'error_camera': 'Не удалось получить доступ к камере. Пожалуйста, проверьте разрешения.',
                'error_camera_denied': 'Доступ к камере отклонен. Пожалуйста, разрешите разрешения камеры и попробуйте снова.',
                'error_camera_not_found': 'Камера не найдена. Пожалуйста, подключите камеру и попробуйте снова.',
                'error_camera_in_use': 'Камера уже используется другим приложением.',
                'error_browser_support': 'Ваш браузер не поддерживает распознавание голоса. Пожалуйста, попробуйте использовать Chrome, Edge или Safari.',
                'error_occurred': 'Ошибка в распознавании:',
                'no_speech': 'Речь не обнаружена. Пожалуйста, попробуйте снова.',
                'network_error': 'Произошла сетевая ошибка. Пожалуйста, проверьте ваше интернет-соединение.',
                'webcam_test_successful': 'Тест WebSocket успешен:',
                'test_message_sent': 'Тестовое сообщение WebSocket отправлено. Проверьте консоль для ответа.',
                'improving': 'улучшающийся',
                'declining': 'ухудшающийся',
                'stable': 'стабильный',
                'insufficient_data': 'недостаточно_данных',
                'low': 'низкий',
                'medium': 'средний',
                'high': 'высокий'
            },
            'it': {
                'app_title': 'MindSync',
                'app_subtitle': 'Assistente di Salute Mentale',
                'connecting': 'Connessione...',
                'connected': 'Connesso',
                'disconnected': 'Disconnesso',
                'mindsync_assistant': 'Assistente MindSync',
                'type_message': 'Digita il tuo messaggio...',
                'send': 'Invia',
                'voice_recognition': 'Riconoscimento Vocale',
                'ready': 'Pronto',
                'listening': 'In ascolto...',
                'click_mic': 'Clicca sul microfono per iniziare a parlare',
                'speak_now': 'Parla ora. Clicca di nuovo sul microfono per fermarti.',
                'controls': 'Controlli',
                'camera': 'Fotocamera',
                'camera_on': 'Accesa',
                'camera_off': 'Spenta',
                'stop_camera': 'Ferma Fotocamera',
                'privacy': 'Privacy',
                'privacy_on': 'Attiva',
                'privacy_off': 'Disattiva',
                'history': 'Cronologia',
                'hide_history': 'Nascondi Cronologia',
                'generate_report': 'Genera Report',
                'test_connection': 'Testa Connessione',
                'emotion_analysis': 'Analisi delle Emozioni',
                'text_based': 'Basato sul testo:',
                'confidence': 'Fiducia:',
                'facial': 'Facciale:',
                'emotion_distribution': 'Distribuzione delle Emozioni',
                'topics_discussed': 'Argomenti Discussi',
                'facial_expression_analysis': 'Analisi delle Espressioni Facciali',
                'dominant_facial_emotion': 'Emozione Facciale Dominante',
                'average_confidence': 'Fiducia Media',
                'facial_analysis_duration': 'Durata dell\'Analisi Facciale',
                'recommendations': 'Raccomandazioni',
                'overall_assessment': 'Valutazione Complessiva',
                'summary': 'Riepilogo',
                'session_duration': 'Durata della Sessione',
                'total_exchanges': 'Scambi Totali',
                'dominant_emotion': 'Emozione Dominante',
                'emotional_trend': 'Trend Emotivo',
                'risk_level': 'Livello di Rischio',
                'report_generated_on': 'Report generato il:',
                'minutes': 'minuti',
                'error_microphone': 'Microfono non disponibile. Per favore, controlla le impostazioni del microfono.',
                'error_permission': 'Autorizzazione del microfono negata. Per favore, consenti l\'accesso al microfono.',
                'error_camera': 'Impossibile accedere alla fotocamera. Per favore, controlla i permessi.',
                'error_camera_denied': 'Accesso alla fotocamera negato. Per favore, consenti i permessi della fotocamera e riprova.',
                'error_camera_not_found': 'Nessuna fotocamera trovata. Per favore, collega una fotocamera e riprova.',
                'error_camera_in_use': 'La fotocamera è già in uso da un\'altra applicazione.',
                'error_browser_support': 'Il tuo browser non supporta il riconoscimento vocale. Per favore, prova a usare Chrome, Edge o Safari.',
                'error_occurred': 'Errore nel riconoscimento:',
                'no_speech': 'Nessun parlato rilevato. Per favore, riprova.',
                'network_error': 'Errore di rete. Per favore, controlla la tua connessione internet.',
                'webcam_test_successful': 'Test WebSocket riuscito:',
                'test_message_sent': 'Messaggio di test WebSocket inviato. Controlla la console per la risposta.',
                'improving': 'migliorando',
                'declining': 'peggiorando',
                'stable': 'stabile',
                'insufficient_data': 'dati_insufficienti',
                'low': 'basso',
                'medium': 'medio',
                'high': 'alto'
            },
            'te': {
                'app_title': 'మైండ్‌సింక్',
                'app_subtitle': 'మానసిక ఆరోగ్య సహాయకుడు',
                'connecting': 'అనుసంధానిస్తోంది...',
                'connected': 'అనుసంధానించబడింది',
                'disconnected': 'అనుసంధానం తీసివేయబడింది',
                'mindsync_assistant': 'మైండ్‌సింక్ సహాయకుడు',
                'type_message': 'మీ సందేశాన్ని టైప్ చేయండి...',
                'send': 'పంపండి',
                'voice_recognition': 'వాయిస్ రికగ్నిషన్',
                'ready': 'సిద్ధంగా ఉంది',
                'listening': 'వింటోంది...',
                'click_mic': 'మాట్లాడటానికి మైక్‌ను నొక్కండి',
                'speak_now': 'ఇప్పుడు మాట్లాడండి. ఆపడానికి మళ్ళీ మైక్‌ను నొక్కండి.',
                'controls': 'నియంత్రణలు',
                'camera': 'కెమెరా',
                'camera_on': 'ఆన్',
                'camera_off': 'ఆఫ్',
                'stop_camera': 'కెమెరాను ఆపండి',
                'privacy': 'గోప్యత',
                'privacy_on': 'ఆన్',
                'privacy_off': 'ఆఫ్',
                'history': 'చరిత్ర',
                'hide_history': 'చరిత్రను దాచండి',
                'generate_report': 'నివేదికను రూపొందించండి',
                'test_connection': 'అనుసంధానాన్ని పరీక్షించండి',
                'emotion_analysis': 'భావోద్వేగ విశ్లేషణ',
                'text_based': 'టెక్స్ట్ ఆధారితం:',
                'confidence': 'విశ్వాసం:',
                'facial': 'ముఖ వ్యక్తీకరణ:',
                'emotion_distribution': 'భావోద్వేగ పంపిణీ',
                'topics_discussed': 'చర్చించబడిన అంశాలు',
                'facial_expression_analysis': 'ముఖ వ్యక్తీకరణ విశ్లేషణ',
                'dominant_facial_emotion': 'ప్రధాన ముఖ భావోద్వేగం',
                'average_confidence': 'సగటు విశ్వాసం',
                'facial_analysis_duration': 'ముఖ విశ్లేషణ వ్యవధి',
                'recommendations': 'సిఫార్సులు',
                'overall_assessment': 'మొత్తం మదింపు',
                'summary': 'సారాంశం',
                'session_duration': 'సెషన్ వ్యవధి',
                'total_exchanges': 'మొత్తం మార్పిడి',
                'dominant_emotion': 'ప్రధాన భావోద్వేగం',
                'emotional_trend': 'భావోద్వేగ ధోరణి',
                'risk_level': 'రిస్క్ స్థాయి',
                'report_generated_on': 'నివేదిక రూపొందించబడింది:',
                'minutes': 'నిమిషాలు',
                'error_microphone': 'మైక్రోఫోన్ అందుబాటులో లేదు. దయచేసి మీ మైక్రోఫోన్ సెట్టింగ్‌లను తనిఖీ చేయండి.',
                'error_permission': 'మైక్రోఫోన్ అనుమతి నిరాకరించబడింది. దయచేసి మైక్రోఫోన్ యాక్సెస్‌ను అనుమతించండి.',
                'error_camera': 'కెమెరాకు యాక్సెస్ చేయడం సాధ్యం కాలేదు. దయచేసి అనుమతులను తనిఖీ చేయండి.',
                'error_camera_denied': 'కెమెరా యాక్సెస్ నిరాకరించబడింది. దయచేసి కెమెరా అనుమతులను అనుమతించి మళ్ళీ ప్రయత్నించండి.',
                'error_camera_not_found': 'కెమెరా కనుగొనబడలేదు. దయచేసి ఒక కెమెరాను కనెక్ట్ చేసి మళ్ళీ ప్రయత్నించండి.',
                'error_camera_in_use': 'కెమెరా ఇప్పటికే మరొక అప్లికేషన్ ద్వారా ఉపయోగించబడుతోంది.',
                'error_browser_support': 'మీ బ్రౌజర్ వాయిస్ రికగ్నిషన్‌ను మద్దతు ఇవ్వదు. దయచేసి క్రోమ్, ఎడ్జ్ లేదా సఫారిని ఉపయోగించడానికి ప్రయత్నించండి.',
                'error_occurred': 'గుర్తింపులో లోపం ఏర్పడింది:',
                'no_speech': 'మాట గుర్తించబడలేదు. దయచేసి మళ్ళీ ప్రయత్నించండి.',
                'network_error': 'నెట్‌వర్క్ లోపం ఏర్పడింది. దయచేసి మీ ఇంటర్నెట్ కనెక్షన్‌ను తనిఖీ చేయండి.',
                'webcam_test_successful': 'వెబ్‌సాకెట్ పరీక్ష విజయవంతమైంది:',
                'test_message_sent': 'వెబ్‌సాకెట్ పరీక్ష సందేశం పంపబడింది. స్పందన కోసం కన్సోల్‌ను తనిఖీ చేయండి.',
                'improving': 'మెరుగుపడుతోంది',
                'declining': 'క్షీణిస్తోంది',
                'stable': 'స్థిరంగా ఉంది',
                'insufficient_data': 'తగినంత డేటా లేదు',
                'low': 'తక్కువ',
                'medium': 'మధ్యస్థం',
                'high': 'ఎక్కువ'
            },
            'bn': {
                'app_title': 'মাইন্ডসিঙ্ক',
                'app_subtitle': 'মানসিক স্বাস্থ্য সহকারী',
                'connecting': 'সংযোগ করা হচ্ছে...',
                'connected': 'সংযুক্ত',
                'disconnected': 'বিচ্ছিন্ন',
                'mindsync_assistant': 'মাইন্ডসিঙ্ক সহকারী',
                'type_message': 'আপনার বার্তা টাইপ করুন...',
                'send': 'পাঠান',
                'voice_recognition': 'ভয়েস রিকগনিশন',
                'ready': 'প্রস্তুত',
                'listening': 'শোনা হচ্ছে...',
                'click_mic': 'কথা বলতে মাইক্রোফোনে ক্লিক করুন',
                'speak_now': 'এখন কথা বলুন। বন্ধ করতে আবার মাইক্রোফোনে ক্লিক করুন।',
                'controls': 'নিয়ন্ত্রণ',
                'camera': 'ক্যামেরা',
                'camera_on': 'চালু',
                'camera_off': 'বন্ধ',
                'stop_camera': 'ক্যামেরা বন্ধ করুন',
                'privacy': 'গোপনীয়তা',
                'privacy_on': 'চালু',
                'privacy_off': 'বন্ধ',
                'history': 'ইতিহাস',
                'hide_history': 'ইতিহাস লুকান',
                'generate_report': 'রিপোর্ট তৈরি করুন',
                'test_connection': 'সংযোগ পরীক্ষা করুন',
                'emotion_analysis': 'আবেগ বিশ্লেষণ',
                'text_based': 'টেক্সট ভিত্তিক:',
                'confidence': 'আত্মবিশ্বাস:',
                'facial': 'মুখের:',
                'emotion_distribution': 'আবেগ বণ্টন',
                'topics_discussed': 'আলোচিত বিষয়',
                'facial_expression_analysis': 'মুখের অভিব্যক্তি বিশ্লেষণ',
                'dominant_facial_emotion': 'প্রধান মুখের আবেগ',
                'average_confidence': 'গড় আত্মবিশ্বাস',
                'facial_analysis_duration': 'মুখের বিশ্লেষণের সময়কাল',
                'recommendations': 'সুপারিশ',
                'overall_assessment': 'সামগ্রিক মূল্যায়ন',
                'summary': 'সারসংক্ষেপ',
                'session_duration': 'সেশনের সময়কাল',
                'total_exchanges': 'মোট বিনিময়',
                'dominant_emotion': 'প্রধান আবেগ',
                'emotional_trend': 'আবেগের প্রবণতা',
                'risk_level': 'ঝুঁকির স্তর',
                'report_generated_on': 'রিপোর্ট তৈরি হয়েছে:',
                'minutes': 'মিনিট',
                'error_microphone': 'মাইক্রোফোন উপলব্ধ নয়। দয়া করে আপনার মাইক্রোফোন সেটিংস চেক করুন।',
                'error_permission': 'মাইক্রোফোন অনুমতি অস্বীকার করা হয়েছে। দয়া করে মাইক্রোফোন অ্যাক্সেসের অনুমতি দিন।',
                'error_camera': 'ক্যামেরায় অ্যাক্সেস করা যায়নি। দয়া করে অনুমতি চেক করুন।',
                'error_camera_denied': 'ক্যামেরা অ্যাক্সেস অস্বীকার করা হয়েছে। দয়া করে ক্যামেরা অনুমতি দিন এবং আবার চেষ্টা করুন।',
                'error_camera_not_found': 'কোনো ক্যামেরা পাওয়া যায়নি। দয়া করে একটি ক্যামেরা সংযোগ করুন এবং আবার চেষ্টা করুন।',
                'error_camera_in_use': 'ক্যামেরা ইতিমধ্যে অন্য অ্যাপ্লিকেশন দ্বারা ব্যবহৃত হচ্ছে।',
                'error_browser_support': 'আপনার ব্রাউজার ভয়েস রিকগনিশন সমর্থন করে না। দয়া করে ক্রোম, এজ বা সাফারি ব্যবহার করার চেষ্টা করুন।',
                'error_occurred': 'স্বীকৃতিতে ত্রুটি ঘটেছে:',
                'no_speech': 'কোনো বক্তৃতা সনাক্ত হয়নি। দয়া করে আবার চেষ্টা করুন।',
                'network_error': 'নেটওয়ার্ক ত্রুটি ঘটেছে। দয়া করে আপনার ইন্টারনেট সংযোগ চেক করুন।',
                'webcam_test_successful': 'ওয়েবসকেট টেস্ট সফল:',
                'test_message_sent': 'ওয়েবসকেট টেস্ট বার্তা পাঠানো হয়েছে। প্রতিক্রিয়ার জন্য কনসোল চেক করুন।',
                'improving': 'উন্নতি হচ্ছে',
                'declining': 'পতন হচ্ছে',
                'stable': 'স্থিতিশীল',
                'insufficient_data': 'অপর্যাপ্ত_ডেটা',
                'low': 'কম',
                'medium': 'মাঝারি',
                'high': 'উচ্চ'
            },
            'gu': {
                'app_title': 'માઇન્ડસિંક',
                'app_subtitle': 'માનસિક સ્વાસ્થ્ય સહાયક',
                'connecting': 'કનેક્ટ કરી રહ્યા છીએ...',
                'connected': 'કનેક્ટ થયેલ છે',
                'disconnected': 'ડિસ્કનેક્ટ થયેલ છે',
                'mindsync_assistant': 'માઇન્ડસિંક સહાયક',
                'type_message': 'તમારો સંદેશ ટાઇપ કરો...',
                'send': 'મોકલો',
                'voice_recognition': 'અવાજ ઓળખ',
                'ready': 'તૈયાર',
                'listening': 'સાંભળી રહ્યા છીએ...',
                'click_mic': 'બોલવા માટે માઇક્રોફોન પર ક્લિક કરો',
                'speak_now': 'હવે બોલો. બંધ કરવા માટે ફરીથી માઇક્રોફોન પર ક્લિક કરો.',
                'controls': 'નિયંત્રણો',
                'camera': 'કેમેરો',
                'camera_on': 'ચાલુ',
                'camera_off': 'બંધ',
                'stop_camera': 'કેમેરો બંધ કરો',
                'privacy': 'ગોપનીયતા',
                'privacy_on': 'ચાલુ',
                'privacy_off': 'બંધ',
                'history': 'ઇતિહાસ',
                'hide_history': 'ઇતિહાસ છુપાવો',
                'generate_report': 'અહેવાલ બનાવો',
                'test_connection': 'જોડાણ ચકાસો',
                'emotion_analysis': 'લાગણી વિશ્લેષણ',
                'text_based': 'ટેક્સ્ટ આધારિત:',
                'confidence': 'વિશ્વાસ:',
                'facial': 'ચહેરાનું:',
                'emotion_distribution': 'લાગણી વિતરણ',
                'topics_discussed': 'ચર્ચા કરેલ વિષયો',
                'facial_expression_analysis': 'ચહેરાની અભિવ્યક્તિ વિશ્લેષણ',
                'dominant_facial_emotion': 'મુખ્ય ચહેરાની લાગણી',
                'average_confidence': 'સરેરાશ વિશ્વાસ',
                'facial_analysis_duration': 'ચહેરાનું વિશ્લેષણ સમયગાળો',
                'recommendations': 'ભલામણો',
                'overall_assessment': 'સમગ્ર મૂલ્યાંકન',
                'summary': 'સારાંશ',
                'session_duration': 'સત્ર સમયગાળો',
                'total_exchanges': 'કુલ વિનિમયો',
                'dominant_emotion': 'મુખ્ય લાગણી',
                'emotional_trend': 'લાગણીશીલ પ્રવૃત્તિ',
                'risk_level': 'જોખમ સ્તર',
                'report_generated_on': 'અહેવાલ બનાવ્યો:',
                'minutes': 'મિનિટો',
                'error_microphone': 'માઇક્રોફોન ઉપલબ્ધ નથી. કૃપા કરીને તમારા માઇક્રોફોન સેટિંગ્સ તપાસો.',
                'error_permission': 'માઇક્રોફોન પરવાનગી નકારવામાં આવી છે. કૃપા કરીને માઇક્રોફોન ઍક્સેસની પરવાનગી આપો.',
                'error_camera': 'કેમેરામાં ઍક્સેસ કરી શકાયું નથી. કૃપા કરીને પરવાનગીઓ તપાસો.',
                'error_camera_denied': 'કેમેરા ઍક્સેસ નકારવામાં આવ્યો છે. કૃપા કરીને કેમેરા પરવાનગીઓ આપો અને ફરીથી પ્રયાસ કરો.',
                'error_camera_not_found': 'કોઈ કેમેરો મળ્યો નથી. કૃપા કરીને એક કેમેરો કનેક્ટ કરો અને ફરીથી પ્રયાસ કરો.',
                'error_camera_in_use': 'કેમેરો પહેલેથી જ અન્ય એપ્લિકેશન દ્વારા ઉપયોગમાં છે.',
                'error_browser_support': 'તમારો બ્રાઉઝર વોઇસ રિકગ્નિશનને સપોર્ટ કરતો નથી. કૃપા કરીને ક્રોમ, એજ અથવા સફારીનો ઉપયોગ કરવાનો પ્રયાસ કરો.',
                'error_occurred': 'ઓળખમાં ભૂલ થઈ:',
                'no_speech': 'કોઈ વાણી શોધાઈ નથી. કૃપા કરીને ફરીથી પ્રયાસ કરો.',
                'network_error': 'નેટવર્ક ભૂલ થઈ. કૃપા કરીને તમારું ઇન્ટરનેટ જોડાણ તપાસો.',
                'webcam_test_successful': 'વેબસોકેટ ટેસ્ટ સફળ:',
                'test_message_sent': 'વેબસોકેટ ટેસ્ટ સંદેશ મોકલાયો. જવાબ માટે કન્સોલ તપાસો.',
                'improving': 'સુધરી રહ્યું છે',
                'declining': 'ઘટી રહ્યું છે',
                'stable': 'સ્થિર',
                'insufficient_data': 'અપૂરતી_માહિતી',
                'low': 'ઓછું',
                'medium': 'મધ્યમ',
                'high': 'વધુ'
            },
            'kn': {
                'app_title': 'ಮೈಂಡ್‌ಸಿಂಕ್',
                'app_subtitle': 'ಮಾನಸಿಕ ಆರೋಗ್ಯ ಸಹಾಯಕ',
                'connecting': 'ಸಂಪರ್ಕಿಸಲಾಗುತ್ತಿದೆ...',
                'connected': 'ಸಂಪರ್ಕಿಸಲಾಗಿದೆ',
                'disconnected': 'ಸಂಪರ್ಕ ಕಡಿತಗೊಂಡಿದೆ',
                'mindsync_assistant': 'ಮೈಂಡ್‌ಸಿಂಕ್ ಸಹಾಯಕ',
                'type_message': 'ನಿಮ್ಮ ಸಂದೇಶವನ್ನು ಟೈಪ್ ಮಾಡಿ...',
                'send': 'ಕಳುಹಿಸಿ',
                'voice_recognition': 'ಧ್ವನಿ ಗುರುತಿಸುವಿಕೆ',
                'ready': 'ಸಿದ್ಧ',
                'listening': 'ಕೇಳುತ್ತಿದೆ...',
                'click_mic': 'ಮಾತನಾಡಲು ಮೈಕ್ರೋಫೋನ್ ಮೇಲೆ ಕ್ಲಿಕ್ ಮಾಡಿ',
                'speak_now': 'ಈಗ ಮಾತನಾಡಿ. ನಿಲ್ಲಿಸಲು ಮತ್ತೆ ಮೈಕ್ರೋಫೋನ್ ಮೇಲೆ ಕ್ಲಿಕ್ ಮಾಡಿ.',
                'controls': 'ನಿಯಂತ್ರಣಗಳು',
                'camera': 'ಕ್ಯಾಮೆರಾ',
                'camera_on': 'ಆನ್',
                'camera_off': 'ಆಫ್',
                'stop_camera': 'ಕ್ಯಾಮೆರಾ ನಿಲ್ಲಿಸಿ',
                'privacy': 'ಗೌಪ್ಯತೆ',
                'privacy_on': 'ಆನ್',
                'privacy_off': 'ಆಫ್',
                'history': 'ಇತಿಹಾಸ',
                'hide_history': 'ಇತಿಹಾಸವನ್ನು ಮರೆಮಾಡಿ',
                'generate_report': 'ವರದಿಯನ್ನು ರಚಿಸಿ',
                'test_connection': 'ಸಂಪರ್ಕವನ್ನು ಪರೀಕ್ಷಿಸಿ',
                'emotion_analysis': 'ಭಾವನೆಗಳ ವಿಶ್ಲೇಷಣೆ',
                'text_based': 'ಪಠ್ಯ ಆಧಾರಿತ:',
                'confidence': 'ವಿಶ್ವಾಸ:',
                'facial': 'ಮುಖದ:',
                'emotion_distribution': 'ಭಾವನೆಗಳ ವಿತರಣೆ',
                'topics_discussed': 'ಚರ್ಚಿಸಿದ ವಿಷಯಗಳು',
                'facial_expression_analysis': 'ಮುಖದ ಅಭಿವ್ಯಕ್ತಿ ವಿಶ್ಲೇಷಣೆ',
                'dominant_facial_emotion': 'ಪ್ರಮುಖ ಮುಖದ ಭಾವನೆ',
                'average_confidence': 'ಸರಾಸರಿ ವಿಶ್ವಾಸ',
                'facial_analysis_duration': 'ಮುಖದ ವಿಶ್ಲೇಷಣೆಯ ಅವಧಿ',
                'recommendations': 'ಶಿಫಾರಸುಗಳು',
                'overall_assessment': 'ಒಟ್ಟಾರೆ ಮೌಲ್ಯಮಾಪನ',
                'summary': 'ಸಾರಾಂಶ',
                'session_duration': 'ಅಧಿವೇಶನದ ಅವಧಿ',
                'total_exchanges': 'ಒಟ್ಟು ವಿನಿಮಯಗಳು',
                'dominant_emotion': 'ಪ್ರಮುಖ ಭಾವನೆ',
                'emotional_trend': 'ಭಾವನಾತ್ಮಕ ಪ್ರವೃತ್ತಿ',
                'risk_level': 'ಅಪಾಯದ ಮಟ್ಟ',
                'report_generated_on': 'ವರದಿಯನ್ನು ರಚಿಸಲಾಗಿದೆ:',
                'minutes': 'ನಿಮಿಷಗಳು',
                'error_microphone': 'ಮೈಕ್ರೋಫೋನ್ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಮೈಕ್ರೋಫೋನ್ ಸೆಟ್ಟಿಂಗ್‌ಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
                'error_permission': 'ಮೈಕ್ರೋಫೋನ್ ಅನುಮತಿಯನ್ನು ನಿರಾಕರಿಸಲಾಗಿದೆ. ದಯವಿಟ್ಟು ಮೈಕ್ರೋಫೋನ್ ಪ್ರವೇಶಕ್ಕೆ ಅನುಮತಿಸಿ.',
                'error_camera': 'ಕ್ಯಾಮೆರಾಕ್ಕೆ ಪ್ರವೇಶಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಅನುಮತಿಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
                'error_camera_denied': 'ಕ್ಯಾಮೆರಾ ಪ್ರವೇಶವನ್ನು ನಿರಾಕರಿಸಲಾಗಿದೆ. ದಯವಿಟ್ಟು ಕ್ಯಾಮೆರಾ ಅನುಮತಿಗಳನ್ನು ನೀಡಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
                'error_camera_not_found': 'ಯಾವುದೇ ಕ್ಯಾಮೆರಾ ಕಂಡುಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ಒಂದು ಕ್ಯಾಮೆರಾವನ್ನು ಸಂಪರ್ಕಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
                'error_camera_in_use': 'ಕ್ಯಾಮೆರಾವನ್ನು ಈಗಾಗಲೇ ಇನ್ನೊಂದು ಅಪ್ಲಿಕೇಶನ್ ಬಳಸುತ್ತಿದೆ.',
                'error_browser_support': 'ನಿಮ್ಮ ಬ್ರೌಸರ್ ಧ್ವನಿ ಗುರುತಿಸುವಿಕೆಯನ್ನು ಬೆಂಬಲಿಸುವುದಿಲ್ಲ. ದಯವಿಟ್ಟು ಕ್ರೋಮ್, ಎಡ್ಜ್ ಅಥವಾ ಸಫಾರಿ ಬಳಸಲು ಪ್ರಯತ್ನಿಸಿ.',
                'error_occurred': 'ಗುರುತಿಸುವಿಕೆಯಲ್ಲಿ ದೋಷ ಸಂಭವಿಸಿದೆ:',
                'no_speech': 'ಯಾವುದೇ ಭಾಷಣ ಪತ್ತೆಯಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
                'network_error': 'ನೆಟ್‌ವರ್ಕ್ ದೋಷ ಸಂಭವಿಸಿದೆ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಇಂಟರ್ನೆಟ್ ಸಂಪರ್ಕವನ್ನು ಪರಿಶೀಲಿಸಿ.',
                'webcam_test_successful': 'ವೆಬ್‌ಸಾಕೆಟ್ ಪರೀಕ್ಷೆ ಯಶಸ್ವಿಯಾಗಿದೆ:',
                'test_message_sent': 'ವೆಬ್‌ಸಾಕೆಟ್ ಪರೀಕ್ಷಾ ಸಂದೇಶವನ್ನು ಕಳುಹಿಸಲಾಗಿದೆ. ಪ್ರತಿಕ್ರಿಯೆಗಾಗಿ ಕನ್ಸೋಲ್ ಅನ್ನು ಪರಿಶೀಲಿಸಿ.',
                'improving': 'ಸುಧಾರಿಸುತ್ತಿದೆ',
                'declining': 'ಕ್ಷೀಣಿಸುತ್ತಿದೆ',
                'stable': 'ಸ್ಥಿರ',
                'insufficient_data': 'ಅಪರ್ಯಾಪ್ತ_ಡೇಟಾ',
                'low': 'ಕಡಿಮೆ',
                'medium': 'ಮಧ್ಯಮ',
                'high': 'ಹೆಚ್ಚು'
            },
            'ml': {
                'app_title': 'മൈൻഡ്സിങ്ക്',
                'app_subtitle': 'മാനസികാരോഗ്യ സഹായി',
                'connecting': 'ബന്ധിപ്പിക്കുന്നു...',
                'connected': 'ബന്ധിപ്പിച്ചു',
                'disconnected': 'വിച്ഛേദിച്ചു',
                'mindsync_assistant': 'മൈൻഡ്സിങ്ക് സഹായി',
                'type_message': 'നിങ്ങളുടെ സന്ദേശം ടൈപ്പ് ചെയ്യുക...',
                'send': 'അയയ്ക്കുക',
                'voice_recognition': 'ശബ്ദ തിരിച്ചറിയൽ',
                'ready': 'തയ്യാറാണ്',
                'listening': 'കേൾക്കുന്നു...',
                'click_mic': 'സംസാരിക്കാൻ മൈക്രോഫോണിൽ ക്ലിക്ക് ചെയ്യുക',
                'speak_now': 'ഇപ്പോൾ സംസാരിക്കുക. നിർത്താൻ വീണ്ടും മൈക്രോഫോണിൽ ക്ലിക്ക് ചെയ്യുക.',
                'controls': 'നിയന്ത്രണങ്ങൾ',
                'camera': 'ക്യാമറ',
                'camera_on': 'ഓൺ',
                'camera_off': 'ഓഫ്',
                'stop_camera': 'ക്യാമറ നിർത്തുക',
                'privacy': 'സ്വകാര്യത',
                'privacy_on': 'ഓൺ',
                'privacy_off': 'ഓഫ്',
                'history': 'ചരിത്രം',
                'hide_history': 'ചരിത്രം മറയ്ക്കുക',
                'generate_report': 'റിപ്പോർട്ട് സൃഷ്ടിക്കുക',
                'test_connection': 'കണക്ഷൻ പരിശോധിക്കുക',
                'emotion_analysis': 'വികാര വിശകലനം',
                'text_based': 'ടെക്സ്റ്റ് അടിസ്ഥാനത്തിൽ:',
                'confidence': 'ആത്മവിശ്വാസം:',
                'facial': 'മുഖം:',
                'emotion_distribution': 'വികാര വിതരണം',
                'topics_discussed': 'ചർച്ച ചെയ്ത വിഷയങ്ങൾ',
                'facial_expression_analysis': 'മുഖ പ്രകടന വിശകലനം',
                'dominant_facial_emotion': 'പ്രധാന മുഖ വികാരം',
                'average_confidence': 'ശരാശരി ആത്മവിശ്വാസം',
                'facial_analysis_duration': 'മുഖ വിശകലന ദൈർഘ്യം',
                'recommendations': 'ശുപാർശകൾ',
                'overall_assessment': 'മൊത്തത്തിലുള്ള വിലയിരുത്തൽ',
                'summary': 'സംഗ്രഹം',
                'session_duration': 'സെഷൻ ദൈർഘ്യം',
                'total_exchanges': 'ആകെ കൈമാറ്റങ്ങൾ',
                'dominant_emotion': 'പ്രധാന വികാരം',
                'emotional_trend': 'വികാര പ്രവണത',
                'risk_level': 'അപകട നില',
                'report_generated_on': 'റിപ്പോർട്ട് സൃഷ്ടിച്ചത്:',
                'minutes': 'മിനിറ്റുകൾ',
                'error_microphone': 'മൈക്രോഫോൺ ലഭ്യമല്ല. ദയവായി നിങ്ങളുടെ മൈക്രോഫോൺ ക്രമീകരണങ്ങൾ പരിശോധിക്കുക.',
                'error_permission': 'മൈക്രോഫോൺ അനുമതി നിരസിച്ചു. ദയവായി മൈക്രോഫോൺ ആക്സസ്സ് അനുവദിക്കുക.',
                'error_camera': 'ക്യാമറയിലേക്ക് പ്രവേശിക്കാൻ കഴിഞ്ഞില്ല. ദയവായി അനുമതികൾ പരിശോധിക്കുക.',
                'error_camera_denied': 'ക്യാമറ ആക്സസ്സ് നിരസിച്ചു. ദയവായി ക്യാമറ അനുമതികൾ അനുവദിച്ച് വീണ്ടും ശ്രമിക്കുക.',
                'error_camera_not_found': 'ക്യാമറ കണ്ടെത്തിയില്ല. ദയവായി ഒരു ക്യാമറ കണക്റ്റ് ചെയ്ത് വീണ്ടും ശ്രമിക്കുക.',
                'error_camera_in_use': 'ക്യാമറ ഇതിനകം മറ്റൊരു അപ്ലിക്കേഷൻ ഉപയോഗിക്കുന്നു.',
                'error_browser_support': 'നിങ്ങളുടെ ബ്രൗസർ വോയ്‌സ് റെക്കഗ്നിഷൻ പിന്തുണയ്ക്കുന്നില്ല. ദയവായി ക്രോം, എഡ്ജ് അല്ലെങ്കിൽ സഫാരി ഉപയോഗിക്കാൻ ശ്രമിക്കുക.',
                'error_occurred': 'തിരിച്ചറിയലിൽ പിശക് സംഭവിച്ചു:',
                'no_speech': 'വാക്ക് കണ്ടെത്തിയില്ല. ദയവായി വീണ്ടും ശ്രമിക്കുക.',
                'network_error': 'നെറ്റ്‌വർക്ക് പിശക് സംഭവിച്ചു. ദയവായി നിങ്ങളുടെ ഇന്റർനെറ്റ് കണക്ഷൻ പരിശോധിക്കുക.',
                'webcam_test_successful': 'വെബ്‌സോക്കറ്റ് ടെസ്റ്റ് വിജയിച്ചു:',
                'test_message_sent': 'വെബ്‌സോക്കറ്റ് ടെസ്റ്റ് സന്ദേശം അയച്ചു. മറുപടിക്കായി കൺസോൾ പരിശോധിക്കുക.',
                'improving': 'മെച്ചപ്പെടുന്നു',
                'declining': 'ക്ഷയിക്കുന്നു',
                'stable': 'സ്ഥിരം',
                'insufficient_data': 'അപര്യാപ്തമായ_ഡാറ്റ',
                'low': 'കുറവ്',
                'medium': 'ഇടത്തരം',
                'high': 'ഉയർന്ന'
            },
            'mr': {
                'app_title': 'माइंडसिंक',
                'app_subtitle': 'मानसिक आरोग्य सहाय्यक',
                'connecting': 'कनेक्ट करत आहे...',
                'connected': 'कनेक्ट केले',
                'disconnected': 'डिस्कनेक्ट केले',
                'mindsync_assistant': 'माइंडसिंक सहाय्यक',
                'type_message': 'तुमचा संदेश टाइप करा...',
                'send': 'पाठवा',
                'voice_recognition': 'आवाज ओळख',
                'ready': 'तयार',
                'listening': 'ऐकत आहे...',
                'click_mic': 'बोलण्यासाठी मायक्रोफोनवर क्लिक करा',
                'speak_now': 'आता बोला. थांबवण्यासाठी पुन्हा मायक्रोफोनवर क्लिक करा.',
                'controls': 'नियंत्रणे',
                'camera': 'कॅमेरा',
                'camera_on': 'चालू',
                'camera_off': 'बंद',
                'stop_camera': 'कॅमेरा थांबवा',
                'privacy': 'गोपनीयता',
                'privacy_on': 'चालू',
                'privacy_off': 'बंद',
                'history': 'इतिहास',
                'hide_history': 'इतिहास लपवा',
                'generate_report': 'अहवाल तयार करा',
                'test_connection': 'कनेक्शन तपासा',
                'emotion_analysis': 'भावना विश्लेषण',
                'text_based': 'मजकूर आधारित:',
                'confidence': 'आत्मविश्वास:',
                'facial': 'चेहऱ्याचे:',
                'emotion_distribution': 'भावना वितरण',
                'topics_discussed': 'चर्चा केलेले विषय',
                'facial_expression_analysis': 'चेहरा अभिव्यक्ती विश्लेषण',
                'dominant_facial_emotion': 'प्रमुख चेहरा भावना',
                'average_confidence': 'सरासरी आत्मविश्वास',
                'facial_analysis_duration': 'चेहरा विश्लेषण कालावधी',
                'recommendations': 'शिफारसी',
                'overall_assessment': 'एकूण मूल्यांकन',
                'summary': 'सारांश',
                'session_duration': 'सत्र कालावधी',
                'total_exchanges': 'एकूण विनिमय',
                'dominant_emotion': 'प्रमुख भावना',
                'emotional_trend': 'भावनात्मक ट्रेंड',
                'risk_level': 'जोखमी स्तर',
                'report_generated_on': 'अहवाल तयार केला:',
                'minutes': 'मिनिटे',
                'error_microphone': 'मायक्रोफोन उपलब्ध नाही. कृपया तुमचे मायक्रोफोन सेटिंग्ज तपासा.',
                'error_permission': 'मायक्रोफोन परवानगी नकारली आहे. कृपया मायक्रोफोन प्रवेशाची परवानगी द्या.',
                'error_camera': 'कॅमेऱ्यावर प्रवेश करता आले नाही. कृपया परवानग्या तपासा.',
                'error_camera_denied': 'कॅमेऱ्यावर प्रवेश नकारला. कृपया कॅमेरा परवानग्या द्या आणि पुन्हा प्रयत्न करा.',
                'error_camera_not_found': 'कोणतेही कॅमेरा सापडले नाही. कृपया एक कॅमेरा कनेक्ट करा आणि पुन्हा प्रयत्न करा.',
                'error_camera_in_use': 'कॅमेरा आधीपासून दुसर्‍या अ‍ॅप्लिकेशनद्वारे वापरला जात आहे.',
                'error_browser_support': 'तुमचा ब्राउझर व्हॉइस रिकग्निशनला सपोर्ट करत नाही. कृपया क्रोम, एज किंवा सफारी वापरण्याचा प्रयत्न करा.',
                'error_occurred': 'ओळखमध्ये त्रुटी आली:',
                'no_speech': 'कोणतेही भाषण आढळले नाही. कृपया पुन्हा प्रयत्न करा.',
                'network_error': 'नेटवर्क त्रुटी आली. कृपया तुमचे इंटरनेट कनेक्शन तपासा.',
                'webcam_test_successful': 'वेबसॉकेट चाचणी यशस्वी:',
                'test_message_sent': 'वेबसॉकेट चाचणी संदेश पाठवला. प्रतिसादासाठी कंसोल तपासा.',
                'improving': 'सुधारत आहे',
                'declining': 'क्षयमान आहे',
                'stable': 'स्थिर',
                'insufficient_data': 'अपुरेसे_डेटा',
                'low': 'कमी',
                'medium': 'मध्यम',
                'high': 'जास्त'
            },
            'pa': {
                'app_title': 'ਮਾਈਂਡਸਿੰਕ',
                'app_subtitle': 'ਮਾਨਸਿਕ ਸਿਹਤ ਸਹਾਇਕ',
                'connecting': 'ਕਨੈਕਟ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...',
                'connected': 'ਕਨੈਕਟ ਹੈ',
                'disconnected': 'ਡਿਸਕਨੈਕਟ ਹੈ',
                'mindsync_assistant': 'ਮਾਈਂਡਸਿੰਕ ਸਹਾਇਕ',
                'type_message': 'ਆਪਾਣਾ ਸੁਨੇਹਾ ਟਾਈਪ ਕਰੋ...',
                'send': 'ਭੇਜੋ',
                'voice_recognition': 'ਆਵਾਜ਼ ਪਛਾਣ',
                'ready': 'ਤਿਆਰ',
                'listening': 'ਸੁਣ ਰਿਹਾ ਹੈ...',
                'click_mic': 'ਗੱਲ ਕਰਨ ਲਈ ਮਾਈਕ੍ਰੋਫੋਨ \'ਤੇ ਕਲਿੱਕ ਕਰੋ',
                'speak_now': 'ਹੁਣ ਗੱਲ ਕਰੋ। ਰੁਕਣ ਲਈ ਦੁਬਾਰਾ ਮਾਈਕ੍ਰੋਫੋਨ \'ਤੇ ਕਲਿੱਕ ਕਰੋ।',
                'controls': 'ਕੰਟਰੋਲ',
                'camera': 'ਕੈਮਰਾ',
                'camera_on': 'ਚਾਲੂ',
                'camera_off': 'ਬੰਦ',
                'stop_camera': 'ਕੈਮਰਾ ਰੋਕੋ',
                'privacy': 'ਪ੍ਰਾਈਵੇਸੀ',
                'privacy_on': 'ਚਾਲੂ',
                'privacy_off': 'ਬੰਦ',
                'history': 'ਇਤਿਹਾਸ',
                'hide_history': 'ਇਤਿਹਾਸ ਲੁਕਾਓ',
                'generate_report': 'ਰਿਪੋਰਟ ਬਣਾਓ',
                'test_connection': 'ਕਨੈਕਸ਼ਨ ਦੀ ਜਾਂਚ ਕਰੋ',
                'emotion_analysis': 'ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ',
                'text_based': 'ਟੈਕਸਟ ਅਧਾਰਤ:',
                'confidence': 'ਵਿਸ਼ਵਾਸ:',
                'facial': 'ਚਿਹਰੇ ਦਾ:',
                'emotion_distribution': 'ਭਾਵਨਾ ਵੰਡ',
                'topics_discussed': 'ਚਰਚਾ ਕੀਤੇ ਵਿਸ਼ੇ',
                'facial_expression_analysis': 'ਚਿਹਰੇ ਦੀ ਪ੍ਰਗਟਾਵਾ ਵਿਸ਼ਲੇਸ਼ਣ',
                'dominant_facial_emotion': 'ਪ੍ਰਮੁੱਖ ਚਿਹਰੇ ਦੀ ਭਾਵਨਾ',
                'average_confidence': 'ਔਸਤ ਵਿਸ਼ਵਾਸ',
                'facial_analysis_duration': 'ਚਿਹਰੇ ਦੇ ਵਿਸ਼ਲੇਸ਼ਣ ਦੀ ਮਿਆਦ',
                'recommendations': 'ਸਿਫ਼ਾਰਸ਼ਾਂ',
                'overall_assessment': 'ਸਮੁੱਚਾ ਮੁਲਾਂਕਣ',
                'summary': 'ਸਾਰ',
                'session_duration': 'ਸੈਸ਼ਨ ਮਿਆਦ',
                'total_exchanges': 'ਕੁੱਲ ਐਕਸਚੇਂਜ',
                'dominant_emotion': 'ਪ੍ਰਮੁੱਖ ਭਾਵਨਾ',
                'emotional_trend': 'ਭਾਵਨਾਤਮਕ ਰੁਝਾਨ',
                'risk_level': 'ਜੋਖਮ ਪੱਧਰ',
                'report_generated_on': 'ਰਿਪੋਰਟ ਬਣਾਈ ਗਈ:',
                'minutes': 'ਮਿੰਟ',
                'error_microphone': 'ਮਾਈਕ੍ਰੋਫੋਨ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੀ ਮਾਈਕ੍ਰੋਫੋਨ ਸੈਟਿੰਗਾਂ ਦੀ ਜਾਂਚ ਕਰੋ।',
                'error_permission': 'ਮਾਈਕ੍ਰੋਫੋਨ ਦੀ ਇਜਾਜ਼ਤ ਤੋਂ ਇਨਕਾਰ ਕੀਤਾ ਗਿਆ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਮਾਈਕ੍ਰੋਫੋਨ ਪਹੁੰਚ ਦੀ ਇਜਾਜ਼ਤ ਦਿਓ।',
                'error_camera': 'ਕੈਮਰੇ ਤੱਕ ਪਹੁੰਚ ਨਹੀਂ ਕੀਤੀ ਜਾ ਸਕੀ। ਕਿਰਪਾ ਕਰਕੇ ਇਜਾਜ਼ਤਾਂ ਦੀ ਜਾਂਚ ਕਰੋ।',
                'error_camera_denied': 'ਕੈਮਰੇ ਦੀ ਪਹੁੰਚ ਤੋਂ ਇਨਕਾਰ ਕੀਤਾ ਗਿਆ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕੈਮਰਾ ਇਜਾਜ਼ਤਾਂ ਦਿਓ ਅਤੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।',
                'error_camera_not_found': 'ਕੋਈ ਕੈਮਰਾ ਨਹੀਂ ਲੱਭਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ ਕੈਮਰਾ ਕਨੈਕਟ ਕਰੋ ਅਤੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।',
                'error_camera_in_use': 'ਕੈਮਰਾ ਪਹਿਲਾਂ ਹੀ ਦੂਜੀ ਐਪਲੀਕੇਸ਼ਨ ਦੁਆਰਾ ਵਰਤਿਆ ਜਾ ਰਿਹਾ ਹੈ।',
                'error_browser_support': 'ਤੁਹਾਡਾ ਬ੍ਰਾਊਜ਼ਰ ਵਾਇਸ ਰਿਕਗਨਿਸ਼ਨ ਦਾ ਸਮਰਥਨ ਨਹੀਂ ਕਰਦਾ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕਰੋਮ, ਐਜ ਜਾਂ ਸਫਾਰੀ ਵਰਤਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ।',
                'error_occurred': 'ਪਛਾਣ ਵਿੱਚ ਗਲਤੀ ਆਈ:',
                'no_speech': 'ਕੋਈ ਭਾਸ਼ਣ ਦਾ ਪਤਾ ਨਹੀਂ ਲੱਗਾ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।',
                'network_error': 'ਨੈੱਟਵਰਕ ਗਲਤੀ ਆਈ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੀ ਇੰਟਰਨੈੱਟ ਕਨੈਕਸ਼ਨ ਦੀ ਜਾਂਚ ਕਰੋ।',
                'webcam_test_successful': 'ਵੈੱਬਸਾਕੇਟ ਟੈਸਟ ਸਫਲ:',
                'test_message_sent': 'ਵੈੱਬਸਾਕੇਟ ਟੈਸਟ ਸੁਨੇਹਾ ਭੇਜਿਆ ਗਿਆ। ਜਵਾਬ ਲਈ ਕਨਸੋਲ ਦੀ ਜਾਂਚ ਕਰੋ।',
                'improving': 'ਸੁਧਾਰ ਰਿਹਾ ਹੈ',
                'declining': 'ਘਟ ਰਿਹਾ ਹੈ',
                'stable': 'ਸਥਿਰ',
                'insufficient_data': 'ਅਪੂਰਨ_ਡਾਟਾ',
                'low': 'ਘੱਟ',
                'medium': 'ਦਰਮਿਆਨਾ',
                'high': 'ਵੱਧ'
            },
            'ur': {
                'app_title': 'مائنڈسنک',
                'app_subtitle': 'ذہنی صحت کا معاون',
                'connecting': 'کنکٹ ہو رہا ہے...',
                'connected': 'کنکٹ ہو گیا',
                'disconnected': 'ڈسکنکٹ ہو گیا',
                'mindsync_assistant': 'مائنڈسنک معاون',
                'type_message': 'اپنا پیغام ٹائپ کریں...',
                'send': 'بھیجیں',
                'voice_recognition': 'آواز کی پہچان',
                'ready': 'تیار ہے',
                'listening': 'سن رہا ہے...',
                'click_mic': 'بات کرنے کے لیے مائیکروفون پر کلک کریں',
                'speak_now': 'اب بات کریں۔ روکنے کے لیے دوبارہ مائیکروفون پر کلک کریں۔',
                'controls': 'کنٹرولز',
                'camera': 'کیمرہ',
                'camera_on': 'آن',
                'camera_off': 'آف',
                'stop_camera': 'کیمرہ روکیں',
                'privacy': 'رازداری',
                'privacy_on': 'آن',
                'privacy_off': 'آف',
                'history': 'تاریخ',
                'hide_history': 'تاریخ چھپائیں',
                'generate_report': 'رپورٹ بنائیں',
                'test_connection': 'کنکشن ٹیسٹ کریں',
                'emotion_analysis': 'جذبات کا تجزیہ',
                'text_based': 'متن پر مبنی:',
                'confidence': 'اعتماد:',
                'facial': 'چہرے کا:',
                'emotion_distribution': 'جذبات کی تقسیم',
                'topics_discussed': 'بحث کے موضوعات',
                'facial_expression_analysis': 'چہرے کی اظہار کا تجزیہ',
                'dominant_facial_emotion': 'غالب چہرے کا جذبہ',
                'average_confidence': 'اوسط اعتماد',
                'facial_analysis_duration': 'چہرے کے تجزیے کی مدت',
                'recommendations': 'تجاویز',
                'overall_assessment': 'کل جائزہ',
                'summary': 'خلاصہ',
                'session_duration': 'سیشن کی مدت',
                'total_exchanges': 'کل تبادلے',
                'dominant_emotion': 'غالب جذبہ',
                'emotional_trend': 'جذباتی رجحان',
                'risk_level': 'خطرے کی سطح',
                'report_generated_on': 'رپورٹ بنائی گئی:',
                'minutes': 'منٹ',
                'error_microphone': 'مائیکروفون دستیاب نہیں ہے۔ براہ کرم اپنے مائیکروفون سیٹنگز چیک کریں۔',
                'error_permission': 'مائیکروفون کی اجازت مسترد کردی گئی۔ براہ کرم مائیکروفون تک رسائی کی اجازت دیں۔',
                'error_camera': 'کیمرے تک رسائی نہیں ہو سکی۔ براہ کرم اجازت چیک کریں۔',
                'error_camera_denied': 'کیمرے تک رسائی مسترد کردی گئی۔ براہ کرم کیمرے کی اجازت دیں اور دوبارہ کوشش کریں۔',
                'error_camera_not_found': 'کوئی کیمرہ نہیں ملا۔ براہ کرم ایک کیمرہ کنکٹ کریں اور دوبارہ کوشش کریں۔',
                'error_camera_in_use': 'کیمرہ پہلے ہی دوسرے ایپلیکیشن کے ذریعے استعمال ہو رہا ہے۔',
                'error_browser_support': 'آپ کا براؤزر آواز کی پہچان کو سپورٹ نہیں کرتا۔ براہ کرم کروم، ایج یا سفاری استعمال کرنے کی کوشش کریں۔',
                'error_occurred': 'پہچان میں خرابی واقع ہوئی:',
                'no_speech': 'کوئی تقریر کا پتہ نہیں چلا۔ براہ کرم دوبارہ کوشش کریں۔',
                'network_error': 'نیٹ ورک خرابی واقع ہوئی۔ براہ کرم اپنی انٹرنیٹ کنکشن چیک کریں۔',
                'webcam_test_successful': 'ویب ساکٹ ٹیسٹ کامیاب:',
                'test_message_sent': 'ویب ساکٹ ٹیسٹ پیغام بھیجا گیا۔ جواب کے لیے کنسول چیک کریں۔',
                'improving': 'بہتر ہو رہا ہے',
                'declining': 'گر رہا ہے',
                'stable': 'مستحکم',
                'insufficient_data': 'ناکافی_ڈیٹا',
                'low': 'کم',
                'medium': 'درمیانہ',
                'high': 'زیادہ'
            }
        }
        
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        except Exception as e:
            logger.error(f"TTS engine initialization error: {e}")
            self.tts_engine = None
    
    def initialize_speech_components(self):
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 300
            logger.info("Speech recognizer initialized successfully")
        except Exception as e:
            logger.error(f"Speech components initialization error: {e}")
            self.recognizer = None
    
    def translate_text(self, text, target_language):
        """Translate text to the target language using Google Translate."""
        if not self.translator or target_language == 'en':
            return text
        
        try:
            # Get the Google Translate language code
            google_lang_code = self.google_translate_codes.get(target_language, 'en')
            
            # Translate the text
            translated = self.translator.translate(text, dest=google_lang_code)
            return translated.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def detect_crisis_keywords(self, text):
        text_lower = text.lower()
        keywords = self.crisis_keywords.get(self.current_language, self.crisis_keywords['en'])
        for keyword in keywords:
            if keyword.lower() in text_lower: return True
        return False
    
    def get_emotion_from_text(self, text):
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.4: return 'happy', abs(polarity)
            elif polarity < -0.4: return 'sad', abs(polarity)
            elif polarity > 0.1: return 'positive', abs(polarity)
            elif polarity < -0.1: return 'negative', abs(polarity)
            else: return 'neutral', 0.5
        except Exception as e:
            logger.error(f"Text emotion analysis error: {e}")
            return 'neutral', 0.5
    
    def extract_topics(self, text):
        topics = []
        text_lower = text.lower()
        topic_keywords = {
            'anxiety': ['anxious', 'anxiety', 'worry', 'nervous', 'panic', 'stress', 'overwhelmed', 'tense'],
            'depression': ['depressed', 'sad', 'down', 'blue', 'unhappy', 'hopeless', 'empty', 'numb'],
            'relationships': ['relationship', 'friend', 'family', 'partner', 'breakup', 'argue', 'fight', 'love'],
            'work': ['work', 'job', 'career', 'boss', 'colleague', 'stressful', 'unemployed', 'fired'],
            'sleep': ['sleep', 'insomnia', 'tired', 'restless', 'awake', 'nightmare', 'exhausted'],
            'self-esteem': ['confidence', 'self-worth', 'insecure', 'doubt', 'believe', 'ashamed', 'proud'],
            'health': ['health', 'doctor', 'medicine', 'sick', 'pain', 'illness', 'hospital'],
            'hobbies': ['hobby', 'interest', 'enjoy', 'fun', 'relax', 'music', 'movie', 'book', 'game'],
            'future': ['future', 'goal', 'dream', 'plan', 'hope', 'worry about', 'afraid of']
        }
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords): topics.append(topic)
        return topics
    
    def extract_name(self, text):
        patterns = [r"my name is (\w+)", r"i'm (\w+)", r"i am (\w+)", r"call me (\w+)", r"everyone calls me (\w+)"]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                name = match.group(1).capitalize()
                self.user_name = name
                return name
        return None
    
    def learn_preference(self, topic, preference):
        if topic not in self.user_preferences: self.user_preferences[topic] = []
        if preference not in self.user_preferences[topic]: self.user_preferences[topic].append(preference)
    
    def get_personalized_greeting(self):
        hour = datetime.now().hour
        time_of_day = 'morning' if 5 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening'
        
        greeting = self.greetings.get(time_of_day, "Hello")
        
        if self.user_name: 
            return f"{greeting}, {self.user_name}! "
        else: 
            return f"{greeting}! "
    
    def get_contextual_response(self, user_input, emotion, confidence):
        topics = self.extract_topics(user_input)
        self.conversation_context['topics_discussed'].extend(topics)
        self.conversation_context['user_feelings'].append((emotion, confidence))
        self.conversation_context['total_exchanges'] += 1
        
        if not self.user_name and self.conversation_context['total_exchanges'] < 5:
            self.extract_name(user_input)
        
        if self.detect_crisis_keywords(user_input):
            return self.get_crisis_response()
        
        if self.conversation_context['total_exchanges'] <= 3:
            return self.get_initial_conversation_response(user_input, emotion, topics)
        
        if user_input.lower() == self.last_user_input.lower():
            return self.get_repeated_input_response()
        
        self.last_user_input = user_input
        
        if topics:
            return self.get_topic_specific_response(topics[0], user_input, emotion)
        elif emotion in ['sad', 'negative']:
            return self.get_emotional_support_response(user_input, emotion)
        elif emotion in ['happy', 'positive']:
            return self.get_positive_response(user_input, emotion)
        else:
            return self.get_neutral_response(user_input, emotion)
    
    def get_initial_conversation_response(self, user_input, emotion, topics):
        if self.conversation_context['total_exchanges'] == 1:
            return f"{self.get_personalized_greeting()}{self.get_text('greeting_followup')}"
        
        if self.user_name and self.conversation_context['total_exchanges'] == 2:
            return self.get_text('nice_to_meet_you', name=self.user_name)
        
        if topics:
            primary_topic = topics[0]
            if primary_topic == 'anxiety': return self.get_text('anxiety_opening')
            elif primary_topic == 'depression': return self.get_text('depression_opening')
            elif primary_topic == 'relationships': return self.get_text('relationships_opening')
            elif primary_topic == 'work': return self.get_text('work_opening')
        
        if emotion in ['sad', 'negative']: return self.get_text('sad_opening')
        elif emotion in ['happy', 'positive']: return self.get_text('happy_opening')
        else: return self.get_text('neutral_opening')
    
    def get_repeated_input_response(self):
        return self.get_text('repeated_input')
    
    def get_topic_specific_response(self, topic, user_input, emotion):
        if topic == 'anxiety': return self.get_text('anxiety_support')
        elif topic == 'depression': return self.get_text('depression_support')
        else: return self.get_text('general_support')
    
    def get_emotional_support_response(self, user_input, emotion):
        return self.get_text('emotional_support', emotion=emotion)
    
    def get_positive_response(self, user_input, emotion):
        return self.get_text('positive_response', emotion=emotion)
    
    def get_neutral_response(self, user_input, emotion):
        if '?' in user_input: return self.get_text('question_response')
        elif any(word in user_input.lower() for word in ['think', 'feel', 'believe']): return self.get_text('perspective_response')
        else: return self.get_text('neutral_followup', user_input=user_input)
    
    def select_varied_response(self, responses):
        available_responses = [r for r in responses if r not in self.recent_responses]
        if not available_responses: available_responses = responses
        response = random.choice(available_responses)
        self.response_history[response] += 1
        self.recent_responses.append(response)
        return response

    def get_text(self, key, **kwargs):
        """Helper function to get text in the current language with optional formatting."""
        # Get the English text template
        text_template = self.text_templates.get(key, "Response not found.")
        
        # Format the template with provided kwargs
        try:
            formatted_text = text_template.format(**kwargs)
        except Exception as e:
            logger.error(f"Text formatting error: {e}")
            formatted_text = text_template
        
        # Translate to the current language
        translated_text = self.translate_text(formatted_text, self.current_language)
        
        return translated_text
    
    def get_ui_text(self, key):
        """Helper function to get UI text in the current language."""
        # Get the UI text in the current language
        ui_text = self.ui_translations.get(self.current_language, {}).get(key, "")
        
        # If no translation is available, fall back to English
        if not ui_text:
            ui_text = self.ui_translations.get('en', {}).get(key, key)
        
        return ui_text

    def get_response(self, user_input):
        emotion, confidence = self.get_emotion_from_text(user_input)
        self.current_emotion = emotion
        self.emotion_confidence = confidence
        response = self.get_contextual_response(user_input, emotion, confidence)
        return response
    
    def get_crisis_response(self):
        # Get a crisis response in English
        response = self.select_varied_response(self.crisis_responses)
        
        # Translate to the current language
        translated_response = self.translate_text(response, self.current_language)
        
        return translated_response
    
    def speak_response(self, text):
        if not self.tts_engine: return
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
    
    def process_recognized_text(self, text, is_voice_input=False):
        if not text or not text.strip(): return
        text = text.strip()
        logger.info(f"Processing recognized text: {text}")
        response = self.get_response(text)
        socketio.emit('bot_response', {'response': response, 'emotion': self.current_emotion, 'confidence': self.emotion_confidence})
        
        # Only speak response if the input was from voice recognition
        if is_voice_input and self.features['voice_text_display']: 
            self.speak_response(response)
        
        self.conversation_history.append({'timestamp': datetime.now().isoformat(), 'user_input': text, 'response': response, 'emotion': self.current_emotion})
    
    def toggle_privacy(self, privacy_mode):
        self.privacy_mode = privacy_mode
        socketio.emit('privacy_status', {'privacy_mode': privacy_mode})
    
    def change_language(self, language_code):
        if language_code in self.available_languages:
            self.current_language = language_code
            return True, f"Language changed to {self.available_languages[language_code]}"
        return False, "Language not supported"
    
    def get_conversation_history(self):
        return list(self.conversation_history)
    
    def generate_mental_health_report(self):
        """Generate a comprehensive mental health report based on conversation and facial expression data."""
        # Get conversation data
        conversation_data = list(self.conversation_history)
        
        # Get facial expression data
        facial_data = camera_detector.get_emotion_summary()
        
        # Initialize report structure
        report = {
            'timestamp': datetime.now().isoformat(),
            'user_name': self.user_name,
            'session_duration': 0,
            'conversation_analysis': {},
            'facial_analysis': facial_data,
            'overall_assessment': '',
            'recommendations': [],
            'risk_level': 'low',
            'summary': ''
        }
        
        # Calculate session duration
        if conversation_data:
            start_time = datetime.fromisoformat(conversation_data[0]['timestamp'])
            end_time = datetime.fromisoformat(conversation_data[-1]['timestamp'])
            report['session_duration'] = (end_time - start_time).total_seconds() / 60  # in minutes
        
        # Analyze conversation emotions
        emotions = [entry['emotion'] for entry in conversation_data]
        emotion_counts = Counter(emotions)
        total_exchanges = len(conversation_data)
        
        if total_exchanges > 0:
            # Calculate emotion percentages
            emotion_percentages = {
                emotion: (count / total_exchanges) * 100 
                for emotion, count in emotion_counts.items()
            }
            
            # Determine dominant emotion
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            
            # Analyze emotional trend
            if len(emotions) > 1:
                first_half = emotions[:len(emotions)//2]
                second_half = emotions[len(emotions)//2:]
                
                first_half_positive = sum(1 for e in first_half if e in ['happy', 'positive'])
                second_half_positive = sum(1 for e in second_half if e in ['happy', 'positive'])
                
                if second_half_positive > first_half_positive:
                    emotional_trend = 'improving'
                elif second_half_positive < first_half_positive:
                    emotional_trend = 'declining'
                else:
                    emotional_trend = 'stable'
            else:
                emotional_trend = 'insufficient_data'
            
            # Analyze topics discussed
            all_topics = []
            for entry in conversation_data:
                topics = self.extract_topics(entry['user_input'])
                all_topics.extend(topics)
            
            topic_counts = Counter(all_topics)
            dominant_topics = [topic for topic, count in topic_counts.most_common(3)]
            
            # Check for crisis indicators
            crisis_indicators = sum(1 for entry in conversation_data 
                                  if self.detect_crisis_keywords(entry['user_input']))
            
            # Store conversation analysis
            report['conversation_analysis'] = {
                'total_exchanges': total_exchanges,
                'dominant_emotion': dominant_emotion,
                'emotion_percentages': emotion_percentages,
                'emotional_trend': emotional_trend,
                'dominant_topics': dominant_topics,
                'crisis_indicators': crisis_indicators
            }
            
            # Determine overall assessment and recommendations
            if crisis_indicators > 0:
                report['risk_level'] = 'high'
                report['overall_assessment'] = 'CRISIS: Immediate attention recommended'
                report['recommendations'] = [
                    'Contact a mental health professional immediately',
                    'Call or text 988 (US) for the Suicide & Crisis Lifeline',
                    'Reach out to a trusted friend or family member',
                    'Remove any means of self-harm from your environment',
                    'Avoid making major life decisions while in crisis'
                ]
                report['summary'] = self.translate_text(
                    "Your conversation indicates you may be experiencing a mental health crisis. "
                    "It's important to seek immediate help from a mental health professional. "
                    "Your life is valuable, and there are people who want to support you through this difficult time.",
                    self.current_language
                )
            elif dominant_emotion in ['sad', 'negative'] and emotion_percentages.get(dominant_emotion, 0) > 60:
                report['risk_level'] = 'medium'
                report['overall_assessment'] = 'CONCERN: Elevated negative emotions detected'
                report['recommendations'] = [
                    'Practice mindfulness and meditation',
                    'Engage in regular physical activity',
                    'Maintain a consistent sleep schedule',
                    'Connect with supportive friends or family',
                    'Consider speaking with a mental health professional',
                    'Limit alcohol and caffeine intake',
                    'Try journaling to express your feelings'
                ]
                report['summary'] = self.translate_text(
                    "Your conversation shows a predominance of negative emotions. "
                    "This is a common experience, and there are many effective strategies to help improve your mood. "
                    "Consider implementing some of the recommended techniques and do not hesitate to seek professional support.",
                    self.current_language
                )
            elif dominant_emotion in ['happy', 'positive'] and emotion_percentages.get(dominant_emotion, 0) > 60:
                report['risk_level'] = 'low'
                report['overall_assessment'] = 'POSITIVE: Good emotional balance detected'
                report['recommendations'] = [
                    'Continue practices that are contributing to your positive mood',
                    'Share your positive experiences with others',
                    'Maintain your current self-care routine',
                    'Consider helping others who may be struggling',
                    'Practice gratitude regularly',
                    'Continue engaging in activities you enjoy'
                ]
                report['summary'] = self.translate_text(
                    "Your conversation shows a predominance of positive emotions, which is excellent for your mental wellbeing. "
                    "Continue the practices that are contributing to your positive state of mind. "
                    "Remember that maintaining mental health is an ongoing process.",
                    self.current_language
                )
            else:
                report['risk_level'] = 'low'
                report['overall_assessment'] = 'STABLE: Mixed emotional indicators'
                report['recommendations'] = [
                    'Practice self-awareness to better understand your emotions',
                    'Try to identify triggers for negative emotions',
                    'Develop a consistent self-care routine',
                    'Consider speaking with a mental health professional for guidance',
                    'Practice stress management techniques',
                    'Ensure you are getting adequate sleep and nutrition'
                ]
                report['summary'] = self.translate_text(
                    "Your conversation shows a mix of emotional states. "
                    "This is normal, but developing greater emotional awareness can help you navigate these feelings more effectively. "
                    "Consider implementing some of the recommended techniques to enhance your emotional wellbeing.",
                    self.current_language
                )
            
            # Incorporate facial expression data if available
            if 'dominant_emotion' in facial_data:
                facial_emotion = facial_data['dominant_emotion']
                
                # Check if facial and text emotions align
                if facial_emotion in ['happy', 'positive'] and dominant_emotion in ['happy', 'positive']:
                    report['summary'] += " " + self.translate_text(
                        "Your facial expressions align with your positive emotional state, suggesting genuine feelings of wellbeing.",
                        self.current_language
                    )
                elif facial_emotion in ['sad', 'angry', 'fear'] and dominant_emotion in ['sad', 'negative']:
                    report['summary'] += " " + self.translate_text(
                        "Your facial expressions align with your negative emotional state, suggesting genuine feelings of distress.",
                        self.current_language
                    )
                    if report['risk_level'] == 'low':
                        report['risk_level'] = 'medium'
                        report['overall_assessment'] = 'CONCERN: Facial expressions indicate distress'
                
                # Add facial-specific recommendations
                if facial_emotion in ['sad', 'angry', 'fear'] and facial_data['emotion_percentages'].get(facial_emotion, 0) > 60:
                    report['recommendations'].append(
                        self.translate_text(
                            "Consider relaxation techniques to reduce facial tension",
                            self.current_language
                        )
                    )
        
        return report

# --- Initialize the assistant and the new detector ---
assistant = RealTimeMindSync()
camera_detector = FacialExpressionDetector()

# Flask routes
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MindSync - Mental Health Assistant</title>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css">
    <style>
        :root { --primary-color: #4a6fa5; --secondary-color: #6b8cae; --accent-color: #f7931e; --text-color: #333; --bg-color: #f5f7fa; --card-bg: #ffffff; --shadow: 0 4px 6px rgba(0, 0, 0, 0.1); --border-radius: 8px; }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: var(--bg-color); color: var(--text-color); line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { background-color: var(--primary-color); color: white; padding: 20px 0; box-shadow: var(--shadow); }
        .header-content { display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto; padding: 0 20px; }
        .logo { font-size: 24px; font-weight: bold; }
        .status-indicator { display: flex; align-items: center; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background-color: #ccc; margin-right: 8px; }
        .status-dot.active { background-color: #4caf50; }
        .main-content { display: grid; grid-template-columns: 1fr 300px; gap: 20px; margin-top: 20px; }
        .chat-container { background-color: var(--card-bg); border-radius: var(--border-radius); box-shadow: var(--shadow); overflow: hidden; display: flex; flex-direction: column; height: 600px; }
        .chat-header { background-color: var(--secondary-color); color: white; padding: 15px; display: flex; justify-content: space-between; align-items: center; }
        .chat-messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 80%; padding: 12px 16px; border-radius: var(--border-radius); word-wrap: break-word; }
        .message.user { align-self: flex-end; background-color: var(--primary-color); color: white; }
        .message.bot { align-self: flex-start; background-color: #e9ecef; color: var(--text-color); }
        .message-time { font-size: 12px; opacity: 0.7; margin-top: 5px; }
        .typing-indicator { display: flex; align-items: center; padding: 12px 16px; background-color: #e9ecef; border-radius: var(--border-radius); max-width: 80px; }
        .typing-indicator span { height: 8px; width: 8px; background-color: #999; border-radius: 50%; display: inline-block; margin: 0 2px; animation: bounce 1.4s infinite ease-in-out both; }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        .chat-input { display: flex; padding: 15px; border-top: 1px solid #e9ecef; }
        .chat-input input { flex: 1; padding: 10px 15px; border: 1px solid #ddd; border-radius: var(--border-radius); font-size: 16px; }
        .chat-input button { background-color: var(--primary-color); color: white; border: none; border-radius: var(--border-radius); padding: 10px 15px; margin-left: 10px; cursor: pointer; transition: background-color 0.3s; }
        .chat-input button:hover { background-color: var(--secondary-color); }
        .sidebar { display: flex; flex-direction: column; gap: 20px; }
        .card { background-color: var(--card-bg); border-radius: var(--border-radius); box-shadow: var(--shadow); padding: 20px; }
        .card-title { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: var(--primary-color); }
        .controls { display: flex; flex-direction: column; gap: 10px; }
        .control-btn { background-color: var(--primary-color); color: white; border: none; border-radius: var(--border-radius); padding: 10px 15px; cursor: pointer; transition: background-color 0.3s; display: flex; align-items: center; justify-content: space-between; }
        .control-btn:hover { background-color: var(--secondary-color); }
        .control-btn.active { background-color: var(--accent-color); }
        .emotion-display { display: flex; flex-direction: column; gap: 10px; }
        .emotion-item { display: flex; justify-content: space-between; }
        .emotion-value { font-weight: bold; }
        .language-selector { margin-top: 10px; }
        .language-selector select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: var(--border-radius); }
        .video-container { position: relative; width: 100%; height: 200px; background-color: #000; border-radius: var(--border-radius); overflow: hidden; margin-top: 10px; display: none; }
        .video-container.show { display: block; }
        .video-container video { width: 100%; height: 100%; object-fit: cover; }
        .voice-text-display { background-color: #f8f9fa; border-radius: var(--border-radius); padding: 10px; margin-top: 10px; min-height: 60px; display: none; }
        .voice-text-display.show { display: block; }
        .voice-text-content { font-style: italic; margin-bottom: 5px; }
        .voice-text-content.listening { color: var(--accent-color); }
        .voice-text-timestamp { font-size: 12px; color: #6c757d; text-align: right; }
        .mic-button { width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; transition: all 0.3s ease; }
        .mic-button.listening { background-color: #dc3545; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); } 70% { box-shadow: 0 0 0 20px rgba(220, 53, 69, 0); } 100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); } }
        .status-ready { background-color: #28a745; }
        .status-listening { background-color: #ffc107; animation: blink 1s infinite; }
        @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
        .report-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.5); z-index: 1000; }
        .report-content { background-color: white; margin: 50px auto; padding: 20px; border-radius: var(--border-radius); width: 80%; max-width: 800px; max-height: 80vh; overflow-y: auto; }
        .report-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .report-title { font-size: 24px; font-weight: bold; color: var(--primary-color); }
        .close-btn { background: none; border: none; font-size: 24px; cursor: pointer; }
        .risk-level { padding: 5px 10px; border-radius: 20px; color: white; font-weight: bold; display: inline-block; margin-bottom: 15px; }
        .risk-low { background-color: #28a745; }
        .risk-medium { background-color: #ffc107; color: #212529; }
        .risk-high { background-color: #dc3545; }
        .report-section { margin-bottom: 20px; }
        .report-section-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: var(--secondary-color); }
        .recommendation-list { list-style-type: none; padding-left: 0; }
        .recommendation-list li { padding: 8px 0; border-bottom: 1px solid #eee; }
        .recommendation-list li:last-child { border-bottom: none; }
        .emotion-bar { height: 20px; background-color: #e9ecef; border-radius: 10px; margin-bottom: 5px; }
        .emotion-fill { height: 100%; border-radius: 10px; }
        .emotion-label { display: flex; justify-content: space-between; margin-bottom: 10px; }
        @media (max-width: 768px) { .main-content { grid-template-columns: 1fr; } .sidebar { order: 2; } .report-content { width: 95%; margin: 20px auto; } }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div class="logo" id="app-title">MindSync</div>
            <div class="status-indicator">
                <div class="status-dot" id="connection-status"></div>
                <span id="connection-text">Connecting...</span>
            </div>
        </div>
    </header>
    
    <div class="container">
        <div class="main-content">
            <div class="chat-container">
                <div class="chat-header">
                    <div id="mindsync-assistant">MindSync Assistant</div>
                    <div id="emotion-display">
                        <span id="emotion-type">Neutral</span> | 
                        <span id="emotion-confidence">50%</span>
                    </div>
                </div>
                
                <div class="chat-messages" id="chat-messages"></div>
                
                <div class="chat-input">
                    <input type="text" id="chat-input" placeholder="Type your message...">
                    <button id="send-btn">Send</button>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="card">
                    <div class="card-title" id="voice-recognition">Voice Recognition</div>
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <div>
                            <span class="status-indicator status-ready" id="statusIndicator"></span>
                            <span id="statusText">Ready</span>
                        </div>
                        <select class="form-select language-selector" id="languageSelect">
                            <option value="en-US">English (US)</option>
                            <option value="en-GB">English (UK)</option>
                            <option value="es-ES">Spanish</option>
                            <option value="fr-FR">French</option>
                            <option value="de-DE">German</option>
                            <option value="it-IT">Italian</option>
                            <option value="pt-BR">Portuguese</option>
                            <option value="ru-RU">Russian</option>
                            <option value="zh-CN">Chinese (Mandarin)</option>
                            <option value="ja-JP">Japanese</option>
                            <option value="ko-KR">Korean</option>
                            <option value="ar-SA">Arabic</option>
                            <option value="hi-IN">Hindi</option>
                            <!-- Additional Indian languages -->
                            <option value="ta-IN">Tamil</option>
                            <option value="te-IN">Telugu</option>
                            <option value="bn-BD">Bengali</option>
                            <option value="gu-IN">Gujarati</option>
                            <option value="kn-IN">Kannada</option>
                            <option value="ml-IN">Malayalam</option>
                            <option value="mr-IN">Marathi</option>
                            <option value="pa-IN">Punjabi</option>
                            <option value="ur-IN">Urdu</option>
                        </select>
                    </div>
                    
                    <button id="micButton" class="btn btn-primary mic-button">
                        <i class="bi bi-mic-fill fs-2"></i>
                    </button>
                    
                    <div class="text-center mt-3">
                        <small id="helpText" class="text-muted">Click the microphone to start speaking</small>
                    </div>
                    
                    <div class="voice-text-display" id="voice-text-display">
                        <div class="voice-text-content" id="voice-text-content"></div>
                        <div class="voice-text-timestamp" id="voice-text-timestamp"></div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title" id="controls">Controls</div>
                    <div class="controls">
                        <button class="control-btn" id="camera-btn">
                            <span id="camera">Camera</span>
                            <span id="camera-status">Off</span>
                        </button>
                        <button class="control-btn" id="privacy-btn">
                            <span id="privacy">Privacy</span>
                            <span id="privacy-status">On</span>
                        </button>
                        <button class="control-btn" id="history-btn">
                            <span id="history">History</span>
                        </button>
                        <button class="control-btn" id="report-btn">
                            <span id="generate-report">Generate Report</span>
                        </button>
                        <button class="control-btn" id="test-btn">
                            <span id="test-connection">Test Connection</span>
                        </button>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title" id="emotion-analysis">Emotion Analysis</div>
                    <div class="emotion-display">
                        <div class="emotion-item">
                            <span id="text-based">Text-based:</span>
                            <span class="emotion-value" id="current-emotion">Neutral</span>
                        </div>
                        <div class="emotion-item">
                            <span id="confidence">Confidence:</span>
                            <span class="emotion-value" id="current-confidence">50%</span>
                        </div>
                        <div class="emotion-item">
                            <span id="facial">Facial:</span>
                            <span class="emotion-value" id="facial-emotion">-</span>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title" id="camera-title">Camera</div>
                    <div class="video-container" id="video-container">
                        <video id="video" autoplay playsinline></video>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Report Modal -->
    <div class="report-modal" id="report-modal">
        <div class="report-content">
            <div class="report-header">
                <div class="report-title" id="report-title">Mental Health Report</div>
                <button class="close-btn" id="close-report">&times;</button>
            </div>
            <div id="report-body">
                <!-- Report content will be dynamically inserted here -->
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let isVoiceActive = false, isCameraActive = false, isPrivacyOn = true, isHistoryVisible = false, isTyping = false;
        let stream = null, videoInterval = null, recognition, finalTranscript = '';
        let currentLanguage = 'en';
        let uiTexts = {};
        
        const connectionStatus = document.getElementById('connection-status');
        const connectionText = document.getElementById('connection-text');
        const chatMessages = document.getElementById('chat-messages');
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const micButton = document.getElementById('micButton');
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const helpText = document.getElementById('helpText');
        const voiceTextDisplay = document.getElementById('voice-text-display');
        const voiceTextContent = document.getElementById('voice-text-content');
        const voiceTextTimestamp = document.getElementById('voice-text-timestamp');
        const languageSelect = document.getElementById('languageSelect');
        const cameraBtn = document.getElementById('camera-btn');
        const cameraStatus = document.getElementById('camera-status');
        const privacyBtn = document.getElementById('privacy-btn');
        const privacyStatus = document.getElementById('privacy-status');
        const historyBtn = document.getElementById('history-btn');
        const reportBtn = document.getElementById('report-btn');
        const testBtn = document.getElementById('test-btn');
        const emotionType = document.getElementById('emotion-type');
        const emotionConfidence = document.getElementById('emotion-confidence');
        const currentEmotion = document.getElementById('current-emotion');
        const currentConfidence = document.getElementById('current-confidence');
        const facialEmotion = document.getElementById('facial-emotion');
        const videoContainer = document.getElementById('video-container');
        const video = document.getElementById('video');
        const reportModal = document.getElementById('report-modal');
        const reportBody = document.getElementById('report-body');
        const closeReport = document.getElementById('close-report');
        
        // Function to update UI text based on language
        function updateUITexts(language) {
            // Request UI texts from server
            socket.emit('get_ui_texts', { language: language });
        }
        
        // Function to apply UI texts
        function applyUITexts(texts) {
            uiTexts = texts;
            
            // Update all UI elements with the new language
            document.getElementById('app-title').textContent = texts.app_title || 'MindSync';
            document.getElementById('connection-text').textContent = texts.connected || 'Connected';
            document.getElementById('mindsync-assistant').textContent = texts.mindsync_assistant || 'MindSync Assistant';
            document.getElementById('chat-input').placeholder = texts.type_message || 'Type your message...';
            document.getElementById('send-btn').textContent = texts.send || 'Send';
            document.getElementById('voice-recognition').textContent = texts.voice_recognition || 'Voice Recognition';
            document.getElementById('statusText').textContent = texts.ready || 'Ready';
            document.getElementById('helpText').textContent = texts.click_mic || 'Click the microphone to start speaking';
            document.getElementById('camera').textContent = texts.camera || 'Camera';
            document.getElementById('privacy').textContent = texts.privacy || 'Privacy';
            document.getElementById('history').textContent = texts.history || 'History';
            document.getElementById('generate-report').textContent = texts.generate_report || 'Generate Report';
            document.getElementById('test-connection').textContent = texts.test_connection || 'Test Connection';
            document.getElementById('emotion-analysis').textContent = texts.emotion_analysis || 'Emotion Analysis';
            document.getElementById('text-based').textContent = texts.text_based || 'Text-based:';
            document.getElementById('confidence').textContent = texts.confidence || 'Confidence:';
            document.getElementById('facial').textContent = texts.facial || 'Facial:';
            document.getElementById('camera-title').textContent = texts.camera || 'Camera';
            document.getElementById('report-title').textContent = texts.app_title + ' ' + (texts.generate_report || 'Report') || 'MindSync Report';
            
            // Update status texts
            if (isVoiceActive) {
                document.getElementById('statusText').textContent = texts.listening || 'Listening...';
                document.getElementById('helpText').textContent = texts.speak_now || 'Speak now. Click the microphone again to stop.';
            }
            
            if (isCameraActive) {
                document.getElementById('camera-status').textContent = texts.camera_on || 'On';
            } else {
                document.getElementById('camera-status').textContent = texts.camera_off || 'Off';
            }
            
            if (isPrivacyOn) {
                document.getElementById('privacy-status').textContent = texts.privacy_on || 'On';
            } else {
                document.getElementById('privacy-status').textContent = texts.privacy_off || 'Off';
            }
            
            if (isHistoryVisible) {
                document.getElementById('history').textContent = texts.hide_history || 'Hide History';
            } else {
                document.getElementById('history').textContent = texts.history || 'History';
            }
        }
        
        socket.onAny((eventName, ...args) => { console.log(`Received event: ${eventName}`, args); });
        
        socket.on('connect', () => {
            console.log('Socket connected successfully');
            connectionStatus.classList.add('active'); 
            connectionText.textContent = uiTexts.connected || 'Connected';
            
            // Request initial UI texts
            updateUITexts(currentLanguage);
            
            addBotMessage(uiTexts.greeting_followup || 'Connected to MindSync. How are you feeling today?');
        });
        
        socket.on('disconnect', () => { 
            connectionStatus.classList.remove('active'); 
            connectionText.textContent = uiTexts.disconnected || 'Disconnected'; 
        });
        
        socket.on('bot_response', (data) => { 
            removeTypingIndicator(); 
            addBotMessage(data.response); 
            updateEmotionDisplay(data.emotion, data.confidence); 
        });
        
        socket.on('camera_status', (data) => {
            isCameraActive = data.active;
            if (isCameraActive) { 
                cameraBtn.innerHTML = `<span>${uiTexts.stop_camera || 'Stop Camera'}</span><span id="camera-status">${uiTexts.camera_on || 'On'}</span>`; 
                cameraBtn.classList.add('active'); 
                startVideo(); 
            }
            else { 
                cameraBtn.innerHTML = `<span>${uiTexts.camera || 'Camera'}</span><span id="camera-status">${uiTexts.camera_off || 'Off'}</span>`; 
                cameraBtn.classList.remove('active'); 
                stopVideo(); 
            }
        });
        
        socket.on('privacy_status', (data) => {
            isPrivacyOn = data.privacy_mode;
            if (isPrivacyOn) { 
                privacyBtn.innerHTML = `<span>${uiTexts.privacy || 'Privacy'}: ${uiTexts.privacy_on || 'On'}</span><span id="privacy-status">${uiTexts.privacy_on || 'On'}</span>`; 
                privacyBtn.classList.add('active'); 
            }
            else { 
                privacyBtn.innerHTML = `<span>${uiTexts.privacy || 'Privacy'}: ${uiTexts.privacy_off || 'Off'}</span><span id="privacy-status">${uiTexts.privacy_off || 'Off'}</span>`; 
                privacyBtn.classList.remove('active'); 
            }
        });
        
        socket.on('emotion_update', (data) => { updateEmotionDisplay(data.emotion, data.confidence); });
        
        socket.on('facial_emotion_update', (data) => {
            console.log('Facial emotion update:', data);
            if (facialEmotion) { 
                facialEmotion.textContent = data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1); 
            }
        });

        socket.on('test_connection', (data) => { 
            addBotMessage(`${uiTexts.webcam_test_successful || 'WebSocket test successful:'} ${data.message}`); 
        });
        
        socket.on('error', (data) => { 
            removeTypingIndicator(); 
            addBotMessage(`${uiTexts.error_occurred || 'Error occurred:'} ${data.message}`); 
        });
        
        socket.on('ui_texts', (data) => {
            applyUITexts(data.texts);
        });
        
        socket.on('language_changed', (data) => {
            if (data.success) {
                currentLanguage = data.language;
                updateUITexts(currentLanguage);
            }
        });
        
        socket.on('mental_health_report', (data) => {
            displayReport(data);
        });
        
        function initializeSpeechRecognition() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) { 
                addBotMessage(uiTexts.error_browser_support || 'Your browser does not support speech recognition. Please try using Chrome, Edge, or Safari.'); 
                micButton.disabled = true; 
                return; 
            }
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition; 
            recognition = new SpeechRecognition();
            recognition.continuous = true; 
            recognition.interimResults = true; 
            recognition.lang = languageSelect.value;
            recognition.onstart = function() { 
                isVoiceActive = true; 
                micButton.classList.add('listening'); 
                statusIndicator.className = 'status-indicator status-listening'; 
                statusText.textContent = uiTexts.listening || 'Listening...'; 
                helpText.textContent = uiTexts.speak_now || 'Speak now. Click the microphone again to stop.'; 
                voiceTextDisplay.classList.add('show'); 
                voiceTextContent.textContent = uiTexts.listening || 'Listening...'; 
                voiceTextContent.classList.add('listening'); 
                voiceTextTimestamp.textContent = ''; 
            };
            recognition.onresult = function(event) { 
                let interimTranscript = ''; 
                for (let i = event.resultIndex; i < event.results.length; i++) { 
                    const transcript = event.results[i][0].transcript; 
                    if (event.results[i].isFinal) { 
                        finalTranscript += transcript + ' '; 
                        voiceTextDisplay.classList.add('show'); 
                        voiceTextContent.textContent = finalTranscript; 
                        voiceTextContent.classList.remove('listening'); 
                        voiceTextTimestamp.textContent = `Recognized at ${getCurrentTime()}`; 
                        socket.emit('recognized_text', { text: finalTranscript.trim(), is_voice_input: true }); 
                        addUserMessage(finalTranscript.trim()); 
                        showTypingIndicator(); 
                        finalTranscript = ''; 
                    } else { 
                        interimTranscript += transcript; 
                    } 
                } 
                if (interimTranscript) { 
                    voiceTextContent.textContent = finalTranscript + interimTranscript; 
                    voiceTextContent.classList.add('listening'); 
                } 
            };
            recognition.onerror = function(event) { 
                console.error('Speech recognition error:', event.error); 
                let errorMessage = uiTexts.error_occurred || 'Error occurred in recognition:'; 
                switch(event.error) { 
                    case 'no-speech': 
                        errorMessage = uiTexts.no_speech || 'No speech was detected. Please try again.'; 
                        break; 
                    case 'audio-capture': 
                        errorMessage = uiTexts.error_microphone || 'Microphone is not available. Please check your microphone settings.'; 
                        break; 
                    case 'not-allowed': 
                        errorMessage = uiTexts.error_permission || 'Microphone permission was denied. Please allow microphone access.'; 
                        break; 
                    case 'network': 
                        errorMessage = uiTexts.network_error || 'Network error occurred. Please check your internet connection.'; 
                        break; 
                    default: 
                        errorMessage += event.error; 
                } 
                statusIndicator.className = 'status-indicator'; 
                statusText.textContent = uiTexts.error_occurred || 'Error'; 
                helpText.textContent = errorMessage; 
                micButton.classList.remove('listening'); 
                isVoiceActive = false; 
            };
            recognition.onend = function() { 
                isVoiceActive = false; 
                micButton.classList.remove('listening'); 
                statusIndicator.className = 'status-indicator status-ready'; 
                statusText.textContent = uiTexts.ready || 'Ready'; 
                helpText.textContent = uiTexts.click_mic || 'Click the microphone to start speaking'; 
            };
        }
        
        micButton.addEventListener('click', function() { 
            if (isVoiceActive) { 
                recognition.stop(); 
            } else { 
                finalTranscript = ''; 
                recognition.lang = languageSelect.value; 
                recognition.start(); 
            } 
        });
        
        cameraBtn.addEventListener('click', () => { 
            if (isCameraActive) { 
                socket.emit('stop_camera'); 
            } else { 
                socket.emit('start_camera'); 
            } 
        });
        
        privacyBtn.addEventListener('click', () => { 
            isPrivacyOn = !isPrivacyOn; 
            socket.emit('toggle_privacy', { privacy_mode: isPrivacyOn }); 
        });
        
        historyBtn.addEventListener('click', () => { 
            isHistoryVisible = !isHistoryVisible; 
            historyBtn.innerHTML = isHistoryVisible ? 
                `<span>${uiTexts.hide_history || 'Hide History'}</span>` : 
                `<span>${uiTexts.history || 'History'}</span>`; 
        });
        
        reportBtn.addEventListener('click', () => { 
            socket.emit('generate_report'); 
        });
        
        testBtn.addEventListener('click', () => { 
            socket.emit('test_connection', { message: 'Test message from client' }); 
            addBotMessage(uiTexts.test_message_sent || 'WebSocket test message sent. Check console for response.'); 
        });
        
        languageSelect.addEventListener('change', () => {
            if (isVoiceActive) { 
                recognition.stop(); 
                setTimeout(() => { 
                    recognition.lang = languageSelect.value; 
                    recognition.start(); 
                }, 100); 
            }
            const langCode = languageSelect.value.split('-')[0];
            socket.emit('change_language', { language: langCode });
        });
        
        sendBtn.addEventListener('click', sendMessage);
        
        chatInput.addEventListener('keypress', (e) => { 
            if (e.key === 'Enter') { 
                sendMessage(); 
            } 
        });
        
        closeReport.addEventListener('click', () => { 
            reportModal.style.display = 'none'; 
        });
        
        function sendMessage() { 
            const message = chatInput.value.trim(); 
            if (message) { 
                addUserMessage(message); 
                showTypingIndicator(); 
                // Send text input with is_voice_input: false
                socket.emit('recognized_text', { text: message, is_voice_input: false }); 
                chatInput.value = ''; 
            } 
        }
        
        function addUserMessage(message) { 
            const messageDiv = document.createElement('div'); 
            messageDiv.className = 'message user'; 
            const contentDiv = document.createElement('div'); 
            contentDiv.className = 'message-content'; 
            contentDiv.textContent = message; 
            const timeDiv = document.createElement('div'); 
            timeDiv.className = 'message-time'; 
            timeDiv.textContent = getCurrentTime(); 
            messageDiv.appendChild(contentDiv); 
            messageDiv.appendChild(timeDiv); 
            chatMessages.appendChild(messageDiv); 
            scrollToBottom(); 
        }
        
        function addBotMessage(message) { 
            const messageDiv = document.createElement('div'); 
            messageDiv.className = 'message bot'; 
            const contentDiv = document.createElement('div'); 
            contentDiv.className = 'message-content'; 
            contentDiv.textContent = message; 
            const timeDiv = document.createElement('div'); 
            timeDiv.className = 'message-time'; 
            timeDiv.textContent = getCurrentTime(); 
            messageDiv.appendChild(contentDiv); 
            messageDiv.appendChild(timeDiv); 
            chatMessages.appendChild(messageDiv); 
            scrollToBottom(); 
        }
        
        function showTypingIndicator() { 
            if (isTyping) return; 
            isTyping = true; 
            const typingDiv = document.createElement('div'); 
            typingDiv.className = 'message bot typing-indicator'; 
            typingDiv.id = 'typing-indicator'; 
            for (let i = 0; i < 3; i++) { 
                const span = document.createElement('span'); 
                typingDiv.appendChild(span); 
            } 
            chatMessages.appendChild(typingDiv); 
            scrollToBottom(); 
        }
        
        function removeTypingIndicator() { 
            isTyping = false; 
            const typingIndicator = document.getElementById('typing-indicator'); 
            if (typingIndicator) { 
                typingIndicator.remove(); 
            } 
        }
        
        function getCurrentTime() { 
            const now = new Date(); 
            return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); 
        }
        
        function scrollToBottom() { 
            chatMessages.scrollTop = chatMessages.scrollHeight; 
        }
        
        function updateEmotionDisplay(emotion, confidence) { 
            emotionType.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1); 
            emotionConfidence.textContent = `${Math.round(confidence * 100)}%`; 
            currentEmotion.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1); 
            currentConfidence.textContent = `${Math.round(confidence * 100)}%`; 
        }
        
        function displayReport(report) {
            // Clear previous report content
            reportBody.innerHTML = '';
            
            // Create report sections
            const reportDate = document.createElement('p');
            reportDate.textContent = `${uiTexts.report_generated_on || 'Report generated on:'} ${new Date(report.timestamp).toLocaleString()}`;
            reportBody.appendChild(reportDate);
            
            // Risk level
            const riskLevel = document.createElement('div');
            riskLevel.className = `risk-level risk-${report.risk_level}`;
            riskLevel.textContent = `${uiTexts.risk_level || 'Risk Level'}: ${report.risk_level.toUpperCase()}`;
            reportBody.appendChild(riskLevel);
            
            // Overall assessment
            const assessmentSection = document.createElement('div');
            assessmentSection.className = 'report-section';
            const assessmentTitle = document.createElement('div');
            assessmentTitle.className = 'report-section-title';
            assessmentTitle.textContent = uiTexts.overall_assessment || 'Overall Assessment';
            assessmentSection.appendChild(assessmentTitle);
            
            const assessmentText = document.createElement('p');
            assessmentText.textContent = report.overall_assessment;
            assessmentSection.appendChild(assessmentText);
            reportBody.appendChild(assessmentSection);
            
            // Summary
            const summarySection = document.createElement('div');
            summarySection.className = 'report-section';
            const summaryTitle = document.createElement('div');
            summaryTitle.className = 'report-section-title';
            summaryTitle.textContent = uiTexts.summary || 'Summary';
            summarySection.appendChild(summaryTitle);
            
            const summaryText = document.createElement('p');
            summaryText.textContent = report.summary;
            summarySection.appendChild(summaryText);
            reportBody.appendChild(summarySection);
            
            // Conversation analysis
            if (report.conversation_analysis && Object.keys(report.conversation_analysis).length > 0) {
                const conversationSection = document.createElement('div');
                conversationSection.className = 'report-section';
                const conversationTitle = document.createElement('div');
                conversationTitle.className = 'report-section-title';
                conversationTitle.textContent = 'Conversation Analysis';
                conversationSection.appendChild(conversationTitle);
                
                // Session duration
                const durationText = document.createElement('p');
                durationText.textContent = `${uiTexts.session_duration || 'Session Duration'}: ${Math.round(report.session_duration)} ${uiTexts.minutes || 'minutes'}`;
                conversationSection.appendChild(durationText);
                
                // Total exchanges
                const exchangesText = document.createElement('p');
                exchangesText.textContent = `${uiTexts.total_exchanges || 'Total Exchanges'}: ${report.conversation_analysis.total_exchanges}`;
                conversationSection.appendChild(exchangesText);
                
                // Dominant emotion
                const dominantEmotionText = document.createElement('p');
                dominantEmotionText.textContent = `${uiTexts.dominant_emotion || 'Dominant Emotion'}: ${report.conversation_analysis.dominant_emotion}`;
                conversationSection.appendChild(dominantEmotionText);
                
                // Emotional trend
                const trendText = document.createElement('p');
                trendText.textContent = `${uiTexts.emotional_trend || 'Emotional Trend'}: ${uiTexts[report.conversation_analysis.emotional_trend] || report.conversation_analysis.emotional_trend}`;
                conversationSection.appendChild(trendText);
                
                // Emotion percentages
                if (report.conversation_analysis.emotion_percentages) {
                    const emotionPercentTitle = document.createElement('div');
                    emotionPercentTitle.className = 'report-section-title';
                    emotionPercentTitle.textContent = uiTexts.emotion_distribution || 'Emotion Distribution';
                    emotionPercentTitle.style.fontSize = '16px';
                    emotionPercentTitle.style.marginTop = '10px';
                    conversationSection.appendChild(emotionPercentTitle);
                    
                    for (const [emotion, percentage] of Object.entries(report.conversation_analysis.emotion_percentages)) {
                        const emotionLabel = document.createElement('div');
                        emotionLabel.className = 'emotion-label';
                        emotionLabel.innerHTML = `
                            <span>${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                            <span>${Math.round(percentage)}%</span>
                        `;
                        conversationSection.appendChild(emotionLabel);
                        
                        const emotionBar = document.createElement('div');
                        emotionBar.className = 'emotion-bar';
                        
                        const emotionFill = document.createElement('div');
                        emotionFill.className = 'emotion-fill';
                        emotionFill.style.width = `${percentage}%`;
                        
                        // Set color based on emotion
                        if (emotion === 'happy' || emotion === 'positive') {
                            emotionFill.style.backgroundColor = '#28a745';
                        } else if (emotion === 'sad' || emotion === 'negative') {
                            emotionFill.style.backgroundColor = '#dc3545';
                        } else {
                            emotionFill.style.backgroundColor = '#6c757d';
                        }
                        
                        emotionBar.appendChild(emotionFill);
                        conversationSection.appendChild(emotionBar);
                    }
                }
                
                // Dominant topics
                if (report.conversation_analysis.dominant_topics && report.conversation_analysis.dominant_topics.length > 0) {
                    const topicsTitle = document.createElement('div');
                    topicsTitle.className = 'report-section-title';
                    topicsTitle.textContent = uiTexts.topics_discussed || 'Topics Discussed';
                    topicsTitle.style.fontSize = '16px';
                    topicsTitle.style.marginTop = '10px';
                    conversationSection.appendChild(topicsTitle);
                    
                    const topicsList = document.createElement('ul');
                    topicsList.style.paddingLeft = '20px';
                    
                    for (const topic of report.conversation_analysis.dominant_topics) {
                        const topicItem = document.createElement('li');
                        topicItem.textContent = topic.charAt(0).toUpperCase() + topic.slice(1);
                        topicsList.appendChild(topicItem);
                    }
                    
                    conversationSection.appendChild(topicsList);
                }
                
                reportBody.appendChild(conversationSection);
            }
            
            // Facial analysis
            if (report.facial_analysis && report.facial_analysis.status !== "No facial data available") {
                const facialSection = document.createElement('div');
                facialSection.className = 'report-section';
                const facialTitle = document.createElement('div');
                facialTitle.className = 'report-section-title';
                facialTitle.textContent = uiTexts.facial_expression_analysis || 'Facial Expression Analysis';
                facialSection.appendChild(facialTitle);
                
                // Dominant facial emotion
                const dominantFacialText = document.createElement('p');
                dominantFacialText.textContent = `${uiTexts.dominant_facial_emotion || 'Dominant Facial Emotion'}: ${report.facial_analysis.dominant_emotion}`;
                facialSection.appendChild(dominantFacialText);
                
                // Average confidence
                const confidenceText = document.createElement('p');
                confidenceText.textContent = `${uiTexts.average_confidence || 'Average Confidence'}: ${Math.round(report.facial_analysis.average_confidence * 100)}%`;
                facialSection.appendChild(confidenceText);
                
                // Duration
                const durationText = document.createElement('p');
                durationText.textContent = `${uiTexts.facial_analysis_duration || 'Facial Analysis Duration'}: ${Math.round(report.facial_analysis.duration_minutes)} ${uiTexts.minutes || 'minutes'}`;
                facialSection.appendChild(durationText);
                
                // Emotion percentages
                if (report.facial_analysis.emotion_percentages) {
                    const emotionPercentTitle = document.createElement('div');
                    emotionPercentTitle.className = 'report-section-title';
                    emotionPercentTitle.textContent = uiTexts.emotion_distribution || 'Facial Emotion Distribution';
                    emotionPercentTitle.style.fontSize = '16px';
                    emotionPercentTitle.style.marginTop = '10px';
                    facialSection.appendChild(emotionPercentTitle);
                    
                    for (const [emotion, percentage] of Object.entries(report.facial_analysis.emotion_percentages)) {
                        const emotionLabel = document.createElement('div');
                        emotionLabel.className = 'emotion-label';
                        emotionLabel.innerHTML = `
                            <span>${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                            <span>${Math.round(percentage)}%</span>
                        `;
                        facialSection.appendChild(emotionLabel);
                        
                        const emotionBar = document.createElement('div');
                        emotionBar.className = 'emotion-bar';
                        
                        const emotionFill = document.createElement('div');
                        emotionFill.className = 'emotion-fill';
                        emotionFill.style.width = `${percentage}%`;
                        
                        // Set color based on emotion
                        if (emotion === 'happy' || emotion === 'positive') {
                            emotionFill.style.backgroundColor = '#28a745';
                        } else if (emotion === 'sad' || emotion === 'negative' || emotion === 'angry' || emotion === 'fear') {
                            emotionFill.style.backgroundColor = '#dc3545';
                        } else {
                            emotionFill.style.backgroundColor = '#6c757d';
                        }
                        
                        emotionBar.appendChild(emotionFill);
                        facialSection.appendChild(emotionBar);
                    }
                }
                
                reportBody.appendChild(facialSection);
            }
            
            // Recommendations
            if (report.recommendations && report.recommendations.length > 0) {
                const recommendationsSection = document.createElement('div');
                recommendationsSection.className = 'report-section';
                const recommendationsTitle = document.createElement('div');
                recommendationsTitle.className = 'report-section-title';
                recommendationsTitle.textContent = uiTexts.recommendations || 'Recommendations';
                recommendationsSection.appendChild(recommendationsTitle);
                
                const recommendationsList = document.createElement('ul');
                recommendationsList.className = 'recommendation-list';
                
                for (const recommendation of report.recommendations) {
                    const recommendationItem = document.createElement('li');
                    recommendationItem.textContent = recommendation;
                    recommendationsList.appendChild(recommendationItem);
                }
                
                recommendationsSection.appendChild(recommendationsList);
                reportBody.appendChild(recommendationsSection);
            }
            
            // Show the modal
            reportModal.style.display = 'block';
        }
        
        async function startVideo() {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                if (videoDevices.length === 0) { throw new Error('No camera devices found'); }
                let constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } };
                if (videoDevices.length > 1) { const frontCamera = videoDevices.find(device => device.label.toLowerCase().includes('front') || device.label.toLowerCase().includes('webcam')); if (frontCamera) { constraints.video.deviceId = frontCamera.deviceId; } }
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    videoContainer.classList.add('show');
                    videoInterval = setInterval(() => {
                        if (video.readyState === video.HAVE_ENOUGH_DATA) {
                            const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight; const ctx = canvas.getContext('2d'); ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                            const imageData = canvas.toDataURL('image/jpeg', 0.7);
                            socket.emit('video_frame', { frame: imageData });
                        }
                    }, 2000);
                };
            } catch (err) { 
                console.error('Error accessing camera:', err); 
                let errorMessage = uiTexts.error_camera || 'Could not access camera. Please check permissions.'; 
                if (err.name === 'NotAllowedError') { 
                    errorMessage = uiTexts.error_camera_denied || 'Camera access denied. Please allow camera permissions and try again.'; 
                } else if (err.name === 'NotFoundError') { 
                    errorMessage = uiTexts.error_camera_not_found || 'No camera found. Please connect a camera and try again.'; 
                } else if (err.name === 'NotReadableError') { 
                    errorMessage = uiTexts.error_camera_in_use || 'Camera is already in use by another application.'; 
                } 
                addBotMessage(errorMessage); 
                socket.emit('stop_camera'); 
            }
        }
        
        function stopVideo() {
            if (stream) { stream.getTracks().forEach(track => { track.stop(); }); stream = null; }
            if (videoInterval) { clearInterval(videoInterval); videoInterval = null; }
            videoContainer.classList.remove('show');
        }
        
        document.addEventListener('DOMContentLoaded', function() { 
            initializeSpeechRecognition(); 
            // Request initial UI texts
            updateUITexts(currentLanguage);
        });
    </script>
</body>
</html>
''')

@app.route('/api/save-text', methods=['POST'])
def save_text():
    data = request.json
    text = data.get('text', '')
    if text: assistant.process_recognized_text(text)
    return jsonify({"status": "success", "text": text})

# --- MODIFIED SOCKET.IO EVENT HANDLERS FOR CAMERA ---
@socketio.on('connect')
def handle_connect():
    assistant.connected_clients += 1
    logger.info(f"Client connected. Total clients: {assistant.connected_clients}")
    emit('status_update', {'connected': True, 'features': assistant.features, 'current_language': assistant.current_language, 'available_languages': assistant.available_languages})
    emit('bot_response', {'response': assistant.get_text('greeting_followup'), 'emotion': 'neutral', 'confidence': 0.5})

@socketio.on('disconnect')
def handle_disconnect():
    assistant.connected_clients -= 1
    logger.info(f"Client disconnected. Total clients: {assistant.connected_clients}")

@socketio.on('recognized_text')
def handle_recognized_text(data):
    text = data.get('text', '')
    is_voice_input = data.get('is_voice_input', False)
    if text: assistant.process_recognized_text(text, is_voice_input)

@socketio.on('send_message')
def handle_message(data):
    message = data.get('message', '')
    if message: assistant.process_recognized_text(message, False)  # Text messages are not voice input

@socketio.on('start_camera')
def handle_start_camera():
    camera_detector.start()

@socketio.on('stop_camera')
def handle_stop_camera():
    camera_detector.stop()

@socketio.on('video_frame')
def handle_video_frame(data):
    frame_data = data.get('frame', '')
    if frame_data: camera_detector.analyze_frame(frame_data)

@socketio.on('toggle_privacy')
def handle_toggle_privacy(data):
    privacy_mode = data.get('privacy_mode', True)
    assistant.toggle_privacy(privacy_mode)

@socketio.on('change_language')
def handle_change_language(data):
    language_code = data.get('language', 'en')
    success, message = assistant.change_language(language_code)
    emit('language_changed', {'success': success, 'message': message, 'language': language_code, 'language_name': assistant.available_languages.get(language_code, 'English')})

@socketio.on('get_ui_texts')
def handle_get_ui_texts(data):
    language = data.get('language', 'en')
    ui_texts = assistant.ui_translations.get(language, assistant.ui_translations.get('en', {}))
    emit('ui_texts', {'texts': ui_texts})

@socketio.on('get_conversation_history')
def handle_get_conversation_history():
    history = assistant.get_conversation_history()
    emit('conversation_history', {'history': history})

@socketio.on('generate_report')
def handle_generate_report():
    report = assistant.generate_mental_health_report()
    emit('mental_health_report', report)
    
    # Send a message to the user about the report
    if report['risk_level'] == 'high':
        emit('bot_response', {'response': assistant.get_text('report_crisis'), 'emotion': 'concerned', 'confidence': 0.9})
    elif report['risk_level'] == 'medium':
        emit('bot_response', {'response': assistant.get_text('report_mixed'), 'emotion': 'supportive', 'confidence': 0.8})
    elif report['risk_level'] == 'low' and report['conversation_analysis'].get('dominant_emotion') in ['happy', 'positive']:
        emit('bot_response', {'response': assistant.get_text('report_positive'), 'emotion': 'encouraging', 'confidence': 0.8})
    else:
        emit('bot_response', {'response': assistant.get_text('report_neutral'), 'emotion': 'neutral', 'confidence': 0.7})

@socketio.on('test_connection')
def handle_test_connection(data):
    message = data.get('message', '')
    emit('test_connection', {'message': f"Received: {message}"})

@socketio.on('error')
def handle_error(data):
    logger.error(f"Socket.IO error: {data}")
    emit('error', {'message': str(data)})

if __name__ == '__main__':
    if FER_AVAILABLE and camera_detector.emotion_detector: print("--- MindSync Starting ---\nStatus: Facial Expression Detection is ENABLED.")
    else: print("--- MindSync Starting ---\nWARNING: Facial Expression Detection is DISABLED. Check 'fer' and 'tensorflow' installation.")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, use_reloader=False)
