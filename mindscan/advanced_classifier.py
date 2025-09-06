"""
Enhanced Mental Health Classifier with Multiple Algorithms
Supports various machine learning approaches for better accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import logging
from typing import Dict, List, Tuple, Any
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

class AdvancedMentalHealthClassifier:
    """
    Advanced mental health classifier using multiple algorithms and enhanced features.
    """
    
    def __init__(self):
        """Initialize the advanced classifier."""
        self.models = {}
        self.vectorizer = None
        self.ensemble_model = None
        self.feature_names = []
        self.is_trained = False
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing with multiple techniques.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic and psychological features from text.
        """
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Sentiment analysis
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Psychological indicators
        first_person_pronouns = ['i', 'me', 'my', 'myself', 'mine']
        negative_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor']
        
        words = text.lower().split()
        features['first_person_ratio'] = sum(1 for word in words if word in first_person_pronouns) / len(words) if words else 0
        features['negative_word_ratio'] = sum(1 for word in words if word in negative_words) / len(words) if words else 0
        
        # Mental health keywords
        depression_keywords = ['sad', 'depressed', 'down', 'hopeless', 'worthless', 'empty', 'lonely']
        anxiety_keywords = ['anxious', 'worried', 'nervous', 'panic', 'fear', 'stress', 'overwhelmed']
        
        features['depression_keyword_count'] = sum(1 for word in words if word in depression_keywords)
        features['anxiety_keyword_count'] = sum(1 for word in words if word in anxiety_keywords)
        
        return features
    
    def prepare_training_data(self, texts: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with enhanced features.
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create TF-IDF features
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            tfidf_features = self.vectorizer.fit_transform(processed_texts).toarray()
        else:
            tfidf_features = self.vectorizer.transform(processed_texts).toarray()
        
        # Extract linguistic features
        linguistic_features = []
        for text in texts:
            features = self.extract_linguistic_features(text)
            linguistic_features.append(list(features.values()))
        
        linguistic_features = np.array(linguistic_features)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, linguistic_features])
        
        return combined_features, np.array(labels)
    
    def train_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train individual machine learning models.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Test score
            test_score = model.score(X_test, y_test)
            
            # Predictions for detailed analysis
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': test_score,
                'classification_report': classification_report(y_test, y_pred),
                'predictions': y_pred
            }
            
            self.models[name] = model
            
            logger.info(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            logger.info(f"{name} - Test Score: {test_score:.4f}")
        
        return results
    
    def create_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> VotingClassifier:
        """
        Create an ensemble model combining multiple algorithms.
        """
        # Use the best performing individual models
        estimators = [
            ('rf', self.models['random_forest']),
            ('svm', self.models['svm']),
            ('lr', self.models['logistic_regression'])
        ]
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability-based voting
        )
        
        # Train ensemble
        self.ensemble_model.fit(X, y)
        
        return self.ensemble_model
    
    def train(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """
        Train the advanced classifier with multiple models.
        """
        logger.info("Starting advanced training...")
        
        # Prepare data
        X, y = self.prepare_training_data(texts, labels)
        
        # Train individual models
        individual_results = self.train_individual_models(X, y)
        
        # Create ensemble model
        ensemble_model = self.create_ensemble_model(X, y)
        
        # Test ensemble performance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ensemble_score = ensemble_model.score(X_test, y_test)
        
        self.is_trained = True
        
        results = {
            'individual_models': individual_results,
            'ensemble_score': ensemble_score,
            'training_samples': len(texts),
            'feature_count': X.shape[1]
        }
        
        logger.info(f"Ensemble Model Score: {ensemble_score:.4f}")
        logger.info("Advanced training completed!")
        
        return results
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict mental health risk using the ensemble model.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare the text
        X, _ = self.prepare_training_data([text], [0])  # Dummy label
        
        # Get predictions from ensemble
        prediction = self.ensemble_model.predict(X)[0]
        probabilities = self.ensemble_model.predict_proba(X)[0]
        
        # Get individual model predictions for comparison
        individual_predictions = {}
        for name, model in self.models.items():
            individual_predictions[name] = {
                'prediction': model.predict(X)[0],
                'probability': model.predict_proba(X)[0].tolist()
            }
        
        # Convert prediction to risk level
        risk_levels = ['normal', 'moderate', 'high']
        risk_level = risk_levels[prediction]
        confidence = max(probabilities)
        
        return {
            'risk_level': risk_level,
            'confidence': confidence,
            'probabilities': {
                'normal': probabilities[0],
                'moderate': probabilities[1],
                'high': probabilities[2]
            },
            'individual_predictions': individual_predictions
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'ensemble_model': self.ensemble_model,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.vectorizer = model_data['vectorizer']
        self.ensemble_model = model_data['ensemble_model']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")

def create_sample_dataset() -> Tuple[List[str], List[int]]:
    """
    Create a sample dataset for testing the advanced classifier.
    """
    texts = [
        # Normal (0)
        "I had a great day today! Everything went well at work.",
        "Just finished a nice workout. Feeling good about life.",
        "Looking forward to the weekend with my family.",
        "Had lunch with friends, it was really enjoyable.",
        "Completed my project successfully. Very satisfied.",
        
        # Moderate (1)
        "Feeling a bit stressed about work lately.",
        "I'm worried about my upcoming exam next week.",
        "Things have been challenging but I'm managing okay.",
        "Sometimes I feel overwhelmed but I push through.",
        "Having some relationship issues but nothing too serious.",
        
        # High (2)
        "I feel completely hopeless and don't see the point anymore.",
        "Can't stop crying and feel worthless all the time.",
        "Everything is falling apart and I can't handle it.",
        "I hate myself and wish I could just disappear.",
        "Life is meaningless and I'm considering ending it all."
    ]
    
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    
    return texts, labels

if __name__ == "__main__":
    # Example usage
    classifier = AdvancedMentalHealthClassifier()
    
    # Create sample data
    texts, labels = create_sample_dataset()
    
    # Train the classifier
    results = classifier.train(texts, labels)
    
    # Test prediction
    test_text = "I'm feeling really down and don't know what to do"
    prediction = classifier.predict(test_text)
    
    print(f"Prediction: {prediction}")
