from typing import List, Dict, Any, Union
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthClassifier:
    """Mental health classification using multiple models."""
    
    def __init__(self):
        """Initialize the classifier with multiple models."""
        self.models = {
            'logistic': LogisticRegression(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = LogisticRegression(random_state=42)
        self.best_model_name = 'logistic'
        
        # Initialize basic scaler and model
        self.initialize_basic_model()
        
    def initialize_basic_model(self):
        """Initialize a basic model with default training data."""
        try:
            # Create simple sample data for basic initialization
            X = np.array([
                [1, 1],  # normal
                [1, 2],  # normal
                [2, 2],  # moderate
                [3, 3],  # high
                [2, 3],  # moderate
            ])
            y = np.array([0, 0, 1, 2, 1])  # 0: normal, 1: moderate, 2: high
            
            # Fit scaler
            self.scaler.fit(X)
            
            # Train model
            self.best_model.fit(X, y)
            logger.info("Basic model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing basic model: {str(e)}")
            raise
    def train(self, features: List[Dict[str, float]], labels: List[int]) -> Dict[str, float]:
        """
        Train multiple models and select the best performing one.
        
        Args:
            features (List[Dict[str, float]]): List of feature dictionaries
            labels (List[int]): List of labels
            
        Returns:
            Dict[str, float]: Performance metrics for each model
        """
        try:
            # Convert features to numpy array
            X = np.array([[v for v in f.values()] for f in features])
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and evaluate each model
            results = {}
            best_score = 0
            
            for name, model in self.models.items():
                logger.info(f"Training {name} model...")
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                results[name] = score
                
                if score > best_score:
                    best_score = score
                    self.best_model = model
                    self.best_model_name = name
            
            logger.info(f"Best model: {self.best_model_name} with accuracy {best_score:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        """
        Predict mental health risk level for new data.
        
        Args:
            features (Dict[str, Any]): Feature dictionary for one instance
            
        Returns:
            Dict[str, Union[str, float]]: Prediction results including risk level and confidence
        """
        try:
            # Extract numeric features
            word_count = features.get('word_count', 0)
            text_length = features.get('text_length', 0)
            
            # Create feature vector
            X = np.array([[word_count, text_length]])
            X_scaled = self.scaler.transform(X)
            
            # Get prediction and probability
            try:
                prediction = self.best_model.predict(X_scaled)[0]
                probabilities = self.best_model.predict_proba(X_scaled)[0]
                confidence = max(probabilities)
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                prediction = 0  # Default to normal
                confidence = 0.5
            
            # Map numeric prediction to risk level
            risk_levels = {0: "normal", 1: "moderate", 2: "high"}
            risk_level = risk_levels.get(prediction, "normal")
            
            # Get text for keyword analysis
            text = str(features.get('processed_text', '')).lower()
            
            # Define keyword lists
            crisis_keywords = ['die', 'death', 'suicide', 'kill', 'end it all']
            positive_keywords = ['good', 'happy', 'great', 'wonderful', 'blessed', 'amazing', 'joy', 'excited']
            negative_keywords = ['sad', 'depressed', 'anxious', 'worried', 'scared', 'hopeless']
            
            # Check for keyword matches
            has_crisis_words = any(word in text for word in crisis_keywords)
            has_positive_words = any(word in text for word in positive_keywords)
            has_negative_words = any(word in text for word in negative_keywords)
            
            # Rule-based overrides
            if has_crisis_words:
                return {
                    'risk_level': 'high',
                    'confidence': 0.95,
                    'reason': 'Crisis keywords detected'
                }
            elif has_positive_words and not has_negative_words:
                return {
                    'risk_level': 'normal',
                    'confidence': 0.90,
                    'reason': 'Positive sentiment detected'
                }
            elif has_negative_words:
                return {
                    'risk_level': 'moderate',
                    'confidence': 0.85,
                    'reason': 'Negative sentiment detected'
                }
                
            # If no keywords matched, return default assessment
            return {
                'risk_level': 'normal',
                'confidence': 0.75,
                'reason': 'No specific indicators detected'
            }
            
            # Override based on keyword detection
            if has_crisis_words:
                return {
                    'risk_level': 'high',
                    'confidence': 0.95,
                    'model_used': 'keyword_override'
                }
            elif has_positive_words and not any(word in text_lower for word in ['not', 'no', 'dont']):
                return {
                    'risk_level': 'normal',
                    'confidence': 0.90,
                    'model_used': 'keyword_override'
                }
            
            if self.best_model is None:
                raise ValueError("Model not trained yet")
            
            # Convert features to numpy array
            X = np.array([[v for v in features.values()]])
            X_scaled = self.scaler.transform(X)
            
            # Get prediction and probability
            prediction = self.best_model.predict(X_scaled)[0]
            probabilities = self.best_model.predict_proba(X_scaled)[0]
            confidence = max(probabilities)
            
            # Map prediction to risk level
            risk_levels = {0: "normal", 1: "moderate", 2: "high"}
            
            # Check for positive sentiment words
            positive_words = ['good', 'happy', 'great', 'wonderful', 'blessed', 'amazing', 'joy', 'excited', 'love', 'peaceful']
            text_lower = str(features.get('text', '')).lower()
            word_count = len(text_lower.split())
            positive_count = sum(1 for word in positive_words if word in text_lower)
            
            # If significant positive sentiment is detected, override to normal
            if positive_count > 0 and positive_count / word_count > 0.1:
                risk_level = "normal"
            else:
                risk_level = risk_levels.get(prediction, "unknown")
            
            return {
                "risk_level": risk_level,
                "confidence": confidence,
                "model_used": self.best_model_name
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model and scaler to disk.
        
        Args:
            path (str): Path to save the model
        """
        try:
            if self.best_model is None:
                raise ValueError("No trained model to save")
            
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'model_name': self.best_model_name
            }
            joblib.dump(model_data, path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, path: str) -> 'MentalHealthClassifier':
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            MentalHealthClassifier: Loaded classifier instance
        """
        try:
            model_data = joblib.load(path)
            classifier = cls()
            classifier.best_model = model_data['model']
            classifier.scaler = model_data['scaler']
            classifier.best_model_name = model_data['model_name']
            logger.info(f"Model loaded from {path}")
            return classifier
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
