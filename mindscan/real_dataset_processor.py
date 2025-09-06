"""
Real Dataset Processor for Mental Health Detection
Handles various dataset formats and preprocessing for machine learning
"""

import pandas as pd
import numpy as np
import os
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

logger = logging.getLogger(__name__)

class RealDatasetProcessor:
    """
    Process real-world mental health datasets for training ML models.
    """
    
    def __init__(self, datasets_dir: str = "datasets"):
        """Initialize the dataset processor."""
        self.datasets_dir = datasets_dir
        self.processed_data = {}
        self.label_encoders = {}
        
    def detect_dataset_format(self, filepath: str) -> str:
        """
        Automatically detect the format and type of dataset.
        """
        df = pd.read_csv(filepath, nrows=5)  # Read first 5 rows to analyze
        columns = [col.lower() for col in df.columns]
        
        # Reddit-style datasets
        if any(col in columns for col in ['post', 'title', 'selftext', 'subreddit']):
            return 'reddit'
        
        # DAIC-WOZ style (interview transcripts)
        elif any(col in columns for col in ['transcript', 'phq8', 'phq_score']):
            return 'daic_woz'
        
        # Generic text classification
        elif 'text' in columns and any(col in columns for col in ['label', 'class', 'category']):
            return 'generic_text'
        
        # eRisk style (sequential posts)
        elif any(col in columns for col in ['date', 'timestamp', 'user_id']):
            return 'erisk'
        
        else:
            return 'unknown'
    
    def load_reddit_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess Reddit-style mental health dataset.
        """
        logger.info(f"Loading Reddit dataset: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} samples with columns: {list(df.columns)}")
            
            # Common Reddit dataset column mappings
            text_columns = ['text', 'post', 'title', 'selftext', 'body', 'content']
            label_columns = ['label', 'class', 'category', 'subreddit', 'condition']
            
            # Find the text column
            text_col = None
            for col in df.columns:
                if col.lower() in text_columns:
                    text_col = col
                    break
            
            # If no direct text column, combine available text fields
            if text_col is None:
                available_text_cols = []
                for col in df.columns:
                    if df[col].dtype == 'object' and col.lower() in ['title', 'selftext', 'body']:
                        available_text_cols.append(col)
                
                if available_text_cols:
                    # Combine text columns
                    df['combined_text'] = df[available_text_cols].fillna('').agg(' '.join, axis=1)
                    text_col = 'combined_text'
                else:
                    raise ValueError("No text column found in dataset")
            
            # Find the label column
            label_col = None
            for col in df.columns:
                if col.lower() in label_columns:
                    label_col = col
                    break
            
            if label_col is None:
                raise ValueError("No label column found in dataset")
            
            # Clean the data
            df = df.dropna(subset=[text_col, label_col])
            df[text_col] = df[text_col].astype(str)
            
            # Process labels
            df = self._process_labels(df, label_col)
            
            logger.info(f"Processed dataset: {len(df)} samples")
            logger.info(f"Label distribution: {df['processed_label'].value_counts().to_dict()}")
            
            return df[[text_col, 'processed_label']].rename(columns={text_col: 'text', 'processed_label': 'label'})
            
        except Exception as e:
            logger.error(f"Error loading Reddit dataset: {str(e)}")
            raise
    
    def load_generic_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load generic text classification dataset.
        """
        logger.info(f"Loading generic dataset: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            
            # Find text and label columns
            text_col = None
            label_col = None
            
            for col in df.columns:
                if col.lower() in ['text', 'content', 'message', 'post']:
                    text_col = col
                if col.lower() in ['label', 'class', 'category', 'target']:
                    label_col = col
            
            if text_col is None or label_col is None:
                raise ValueError("Could not identify text and label columns")
            
            # Clean and process
            df = df.dropna(subset=[text_col, label_col])
            df[text_col] = df[text_col].astype(str)
            df = self._process_labels(df, label_col)
            
            logger.info(f"Processed generic dataset: {len(df)} samples")
            
            return df[[text_col, 'processed_label']].rename(columns={text_col: 'text', 'processed_label': 'label'})
            
        except Exception as e:
            logger.error(f"Error loading generic dataset: {str(e)}")
            raise
    
    def _process_labels(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """
        Process and standardize labels to 0 (normal), 1 (moderate), 2 (high).
        """
        original_labels = df[label_col].unique()
        logger.info(f"Original labels: {original_labels}")
        
        # Create label mapping based on common patterns
        label_mapping = {}
        
        for label in original_labels:
            label_str = str(label).lower().strip()
            
            # High risk indicators
            if any(keyword in label_str for keyword in [
                'depression', 'suicidal', 'suicide', 'severe', 'high', 'crisis',
                'selfharm', 'self_harm', 'bipolar', 'ptsd', 'panic', 'schizophrenia'
            ]):
                label_mapping[label] = 2
            
            # Moderate risk indicators
            elif any(keyword in label_str for keyword in [
                'anxiety', 'stress', 'moderate', 'mild', 'worried', 'overwhelmed',
                'sad', 'lonely', 'insomnia', 'ocd', 'adhd'
            ]):
                label_mapping[label] = 1
            
            # Normal/control indicators
            elif any(keyword in label_str for keyword in [
                'normal', 'control', 'healthy', 'no', 'none', 'low', '0'
            ]):
                label_mapping[label] = 0
            
            # Numeric labels
            elif label_str.isdigit():
                num_label = int(label_str)
                if num_label <= 1:
                    label_mapping[label] = 0
                elif num_label <= 3:
                    label_mapping[label] = 1
                else:
                    label_mapping[label] = 2
            
            # Default mapping based on position
            else:
                # If we can't determine, assign based on order
                label_list = sorted(original_labels)
                idx = label_list.index(label)
                if len(label_list) <= 2:
                    label_mapping[label] = min(idx, 1)
                else:
                    label_mapping[label] = min(idx, 2)
        
        df['processed_label'] = df[label_col].map(label_mapping)
        
        # Handle any unmapped labels
        df['processed_label'] = df['processed_label'].fillna(0).astype(int)
        
        logger.info(f"Label mapping: {label_mapping}")
        return df
    
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Automatically load and process any dataset format.
        """
        filepath = os.path.join(self.datasets_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        # Detect format
        dataset_format = self.detect_dataset_format(filepath)
        logger.info(f"Detected dataset format: {dataset_format}")
        
        # Load based on format
        if dataset_format == 'reddit':
            return self.load_reddit_dataset(filepath)
        elif dataset_format in ['generic_text', 'unknown']:
            return self.load_generic_dataset(filepath)
        else:
            # Try generic loading as fallback
            return self.load_generic_dataset(filepath)
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset characteristics and quality.
        """
        analysis = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'text_stats': {
                'avg_length': df['text'].str.len().mean(),
                'min_length': df['text'].str.len().min(),
                'max_length': df['text'].str.len().max(),
                'avg_words': df['text'].str.split().str.len().mean()
            },
            'data_quality': {
                'duplicates': df.duplicated().sum(),
                'missing_text': df['text'].isna().sum(),
                'empty_text': (df['text'].str.strip() == '').sum()
            }
        }
        
        logger.info(f"Dataset Analysis: {analysis}")
        return analysis
    
    def clean_dataset(self, df: pd.DataFrame, min_length: int = 10, max_length: int = 2000) -> pd.DataFrame:
        """
        Clean and filter dataset for better quality.
        """
        initial_size = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Remove very short or very long texts
        df = df[df['text'].str.len().between(min_length, max_length)]
        
        # Remove texts that are mostly special characters or numbers
        df = df[df['text'].str.contains(r'[a-zA-Z]', regex=True)]
        
        # Remove empty or whitespace-only texts
        df = df[df['text'].str.strip() != '']
        
        final_size = len(df)
        logger.info(f"Cleaned dataset: {initial_size} -> {final_size} samples ({final_size/initial_size*100:.1f}% retained)")
        
        return df.reset_index(drop=True)
    
    def balance_dataset(self, df: pd.DataFrame, strategy: str = 'undersample') -> pd.DataFrame:
        """
        Balance the dataset across different labels.
        """
        label_counts = df['label'].value_counts()
        logger.info(f"Original distribution: {label_counts.to_dict()}")
        
        if strategy == 'undersample':
            # Undersample to match the smallest class
            min_count = label_counts.min()
            balanced_dfs = []
            
            for label in df['label'].unique():
                label_df = df[df['label'] == label].sample(n=min_count, random_state=42)
                balanced_dfs.append(label_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif strategy == 'oversample':
            # Simple oversampling by repeating samples
            max_count = label_counts.max()
            balanced_dfs = []
            
            for label in df['label'].unique():
                label_df = df[df['label'] == label]
                if len(label_df) < max_count:
                    # Repeat samples to reach max_count
                    repeats = max_count // len(label_df)
                    remainder = max_count % len(label_df)
                    
                    repeated_df = pd.concat([label_df] * repeats, ignore_index=True)
                    if remainder > 0:
                        extra_df = label_df.sample(n=remainder, random_state=42)
                        repeated_df = pd.concat([repeated_df, extra_df], ignore_index=True)
                    
                    balanced_dfs.append(repeated_df)
                else:
                    balanced_dfs.append(label_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        else:
            balanced_df = df  # No balancing
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        new_counts = balanced_df['label'].value_counts()
        logger.info(f"Balanced distribution: {new_counts.to_dict()}")
        
        return balanced_df
    
    def prepare_for_training(self, filename: str, 
                           clean: bool = True, 
                           balance: bool = True,
                           balance_strategy: str = 'undersample',
                           test_size: float = 0.2) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Complete pipeline to prepare dataset for training.
        """
        logger.info(f"Preparing dataset {filename} for training...")
        
        # Load dataset
        df = self.load_dataset(filename)
        
        # Analyze dataset
        analysis = self.analyze_dataset(df)
        
        # Clean dataset
        if clean:
            df = self.clean_dataset(df)
        
        # Balance dataset
        if balance:
            df = self.balance_dataset(df, strategy=balance_strategy)
        
        # Split into train/test
        train_df, test_df = train_test_split(
            df, test_size=test_size, 
            stratify=df['label'], 
            random_state=42
        )
        
        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return (
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            test_df['text'].tolist(),
            test_df['label'].tolist()
        )
    
    def save_processed_dataset(self, df: pd.DataFrame, filename: str):
        """Save processed dataset."""
        filepath = os.path.join(self.datasets_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed dataset to {filepath}")

if __name__ == "__main__":
    # Example usage
    processor = RealDatasetProcessor()
    
    # List available datasets
    datasets_dir = "datasets"
    if os.path.exists(datasets_dir):
        files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
        print(f"Available datasets: {files}")
        
        if files:
            # Process the first available dataset
            filename = files[0]
            print(f"Processing {filename}...")
            
            train_texts, train_labels, test_texts, test_labels = processor.prepare_for_training(filename)
            print(f"Prepared {len(train_texts)} training samples and {len(test_texts)} test samples")
