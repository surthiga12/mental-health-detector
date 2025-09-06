"""
Simple ML Predictor for the trained Reddit model
"""

import pickle
import re
from typing import Dict, Any
import os

class SimpleMLPredictor:
    """Simple predictor using the trained Reddit model."""
    
    def __init__(self, model_path: str = 'models/simple_reddit_model.pkl'):
        """Initialize the predictor."""
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.vectorizer = data['vectorizer']
                print(f"✅ ML model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model file not found: {self.model_path}")
                self.model = None
                self.vectorizer = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.vectorizer = None
    
    def clean_text(self, text: str) -> str:
        """Clean text for prediction."""
        if not text:
            return ""
        
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict mental health risk level."""
        if not self.model or not self.vectorizer:
            return {
                'risk_level': 'moderate',
                'confidence': 0.5,
                'probabilities': {'normal': 0.33, 'moderate': 0.34, 'high': 0.33},
                'source': 'fallback'
            }
        
        try:
            # Clean and vectorize text
            clean_text = self.clean_text(text)
            if not clean_text:
                return {
                    'risk_level': 'normal',
                    'confidence': 0.6,
                    'probabilities': {'normal': 0.6, 'moderate': 0.3, 'high': 0.1},
                    'source': 'empty_text'
                }
            
            # Vectorize
            text_vec = self.vectorizer.transform([clean_text])
            
            # Predict
            prediction = self.model.predict(text_vec)[0]
            probabilities = self.model.predict_proba(text_vec)[0]
            
            # Map prediction to risk level
            risk_levels = ['normal', 'moderate', 'high']
            risk_level = risk_levels[prediction]
            confidence = max(probabilities)
            
            # Create probability dict
            prob_dict = {
                'normal': float(probabilities[0]),
                'moderate': float(probabilities[1]),
                'high': float(probabilities[2])
            }
            
            return {
                'risk_level': risk_level,
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'source': 'reddit_trained_model'
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'risk_level': 'moderate',
                'confidence': 0.5,
                'probabilities': {'normal': 0.33, 'moderate': 0.34, 'high': 0.33},
                'source': 'error_fallback'
            }
    
    def is_available(self) -> bool:
        """Check if the model is available."""
        return self.model is not None and self.vectorizer is not None

# Global instance
ml_predictor = SimpleMLPredictor()
