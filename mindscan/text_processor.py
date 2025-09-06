from typing import Dict, List, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing class for mental health detection."""
    
    def __init__(self):
        """Initialize the text preprocessor with NLTK components."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text: str) -> str:
        """
        Preprocess the input text by cleaning, tokenizing, and lemmatizing.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Split into words (simpler tokenization)
        tokens = text.split()
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract linguistic features from the text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Dictionary of extracted features
        """
        try:
            # Preprocess the text
            processed_text = self.preprocess(text)
            
            # Basic feature extraction
            features = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'processed_text': processed_text
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {
                'text_length': 0,
                'word_count': 0,
                'processed_text': ''
            }
        processed_text = self.preprocess(text)
        tokens = processed_text.split()
        
        # Calculate basic features without POS tagging
        features = {
            'text_length': len(text),
            'word_count': len(tokens),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'unique_words': len(set(tokens)),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'negative_words': sum(1 for word in tokens if word in ['not', 'no', 'never', 'cant', 'wont', 'dont'])
        }
        
        return features
