"""
Dataset downloader and preprocessor for mental health detection
Handles multiple public datasets and creates training data
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import os
import json
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class MentalHealthDatasetManager:
    """
    Manages downloading and preprocessing of mental health datasets.
    """
    
    def __init__(self, data_dir: str = "datasets"):
        """Initialize the dataset manager."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def create_reddit_style_dataset(self) -> Tuple[List[str], List[int]]:
        """
        Create a comprehensive dataset mimicking Reddit mental health posts.
        This is for demonstration - in practice you'd use real datasets.
        """
        
        # Normal posts (label 0)
        normal_posts = [
            "Just got promoted at work! So excited for this new opportunity.",
            "Had an amazing weekend hiking with friends. Nature is so refreshing.",
            "Finished reading a great book today. Highly recommend it to everyone.",
            "Cooking has become my new hobby. Made pasta from scratch tonight.",
            "Grateful for my supportive family. They mean the world to me.",
            "Started a new exercise routine and feeling more energetic.",
            "Volunteered at the animal shelter today. Such a rewarding experience.",
            "Planning a vacation with my partner. Can't wait to explore new places.",
            "Successfully completed my certification course. Hard work pays off.",
            "Enjoying this beautiful sunny day. Perfect weather for a walk.",
        ]
        
        # Moderate risk posts (label 1)
        moderate_posts = [
            "Work has been really stressful lately. Barely getting any sleep.",
            "Feeling anxious about my upcoming job interview next week.",
            "Been having some arguments with my partner. Relationship is tough.",
            "College assignments are piling up and I'm feeling overwhelmed.",
            "Money is tight this month. Worried about paying all the bills.",
            "Parents are getting divorced and it's affecting me more than expected.",
            "Social situations make me nervous. Hard to make new friends.",
            "Been feeling moody and irritable for the past few weeks.",
            "Health issues have been causing me stress and worry.",
            "Feeling like I'm not good enough at my job sometimes.",
        ]
        
        # High risk posts (label 2)
        high_risk_posts = [
            "Everything feels pointless. Don't see any reason to keep going.",
            "Can't stop crying and feel completely worthless as a person.",
            "Nobody understands me and I feel so alone in this world.",
            "Life is just pain and suffering. Wish it would all end.",
            "I hate myself and everything about my pathetic existence.",
            "Been thinking about ending it all. Nothing matters anymore.",
            "Feel like a burden to everyone around me. They'd be better off.",
            "Can't eat or sleep. Lost all interest in things I used to enjoy.",
            "Every day is a struggle just to get out of bed and function.",
            "Drowning in depression and anxiety. Can't see a way out.",
        ]
        
        # Combine all posts
        all_texts = normal_posts + moderate_posts + high_risk_posts
        all_labels = [0] * len(normal_posts) + [1] * len(moderate_posts) + [2] * len(high_risk_posts)
        
        return all_texts, all_labels
    
    def augment_dataset(self, texts: List[str], labels: List[int], multiplier: int = 3) -> Tuple[List[str], List[int]]:
        """
        Augment the dataset by creating variations of existing texts.
        """
        augmented_texts = []
        augmented_labels = []
        
        # Synonyms for data augmentation
        synonyms = {
            'sad': ['depressed', 'down', 'blue', 'miserable', 'unhappy'],
            'happy': ['glad', 'joyful', 'cheerful', 'pleased', 'content'],
            'anxious': ['worried', 'nervous', 'concerned', 'uneasy', 'troubled'],
            'angry': ['mad', 'furious', 'irritated', 'annoyed', 'upset'],
            'stressed': ['overwhelmed', 'pressured', 'burdened', 'strained', 'tense']
        }
        
        for i in range(multiplier):
            for text, label in zip(texts, labels):
                # Original text
                if i == 0:
                    augmented_texts.append(text)
                    augmented_labels.append(label)
                else:
                    # Create variations
                    modified_text = text
                    
                    # Replace some words with synonyms
                    for word, syns in synonyms.items():
                        if word in modified_text.lower():
                            replacement = np.random.choice(syns)
                            modified_text = modified_text.replace(word, replacement)
                    
                    # Add some noise (typos, abbreviations)
                    if np.random.random() < 0.3:
                        words = modified_text.split()
                        if words:
                            # Random word replacement with common abbreviations
                            abbrevs = {'you': 'u', 'are': 'r', 'to': '2', 'for': '4'}
                            for j, word in enumerate(words):
                                if word.lower() in abbrevs and np.random.random() < 0.5:
                                    words[j] = abbrevs[word.lower()]
                            modified_text = ' '.join(words)
                    
                    augmented_texts.append(modified_text)
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
    
    def create_comprehensive_dataset(self, size: int = 1000) -> Tuple[List[str], List[int]]:
        """
        Create a comprehensive dataset with various mental health scenarios.
        """
        
        # Base templates for different categories
        templates = {
            0: [  # Normal
                "Had a {adjective} day at {place}. Feeling {emotion} about everything.",
                "Just {activity} and feeling really {emotion}. Life is {adjective}.",
                "Spent time with {people} today. Such a {adjective} experience.",
                "{activity} has been going well. Very {emotion} with the progress.",
                "Looking forward to {future_activity}. Feeling {emotion} and {adjective}."
            ],
            1: [  # Moderate
                "Been feeling {emotion} about {stressor} lately. It's been {adjective}.",
                "{stressor} is causing me some {emotion}. Finding it {adjective} to handle.",
                "Having some {problem} with {area}. Feeling {emotion} about it.",
                "Work/life balance has been {adjective}. Feeling {emotion} sometimes.",
                "Been {emotion} more than usual. Things feel {adjective} right now."
            ],
            2: [  # High risk
                "Everything feels {negative_adj}. Don't see any {positive_concept} anymore.",
                "Can't stop feeling {negative_emotion}. Life seems {negative_adj}.",
                "Feel like a {negative_self} to everyone. Wish I could {negative_action}.",
                "Every day is {negative_adj}. Can't find any {positive_concept} in anything.",
                "Been thinking about {concerning_thought}. Nothing seems {positive_adj} anymore."
            ]
        }
        
        # Word lists for templates
        words = {
            'adjective': ['great', 'wonderful', 'amazing', 'good', 'nice', 'pleasant'],
            'place': ['work', 'school', 'home', 'the gym', 'the park'],
            'emotion': ['happy', 'content', 'satisfied', 'proud', 'grateful'],
            'activity': ['exercising', 'reading', 'cooking', 'studying', 'working'],
            'people': ['friends', 'family', 'colleagues', 'classmates'],
            'future_activity': ['the weekend', 'my vacation', 'the project', 'the event'],
            'stressor': ['work', 'school', 'relationships', 'finances', 'health'],
            'problem': ['issues', 'difficulties', 'challenges', 'troubles'],
            'area': ['my job', 'my studies', 'my relationship', 'my family'],
            'negative_adj': ['pointless', 'hopeless', 'meaningless', 'worthless'],
            'positive_concept': ['hope', 'point', 'meaning', 'purpose', 'joy'],
            'negative_emotion': ['crying', 'hurting', 'suffering', 'breaking down'],
            'negative_self': ['burden', 'failure', 'disappointment', 'waste'],
            'negative_action': ['disappear', 'give up', 'end it all'],
            'concerning_thought': ['ending everything', 'giving up completely', 'not being here'],
            'positive_adj': ['worthwhile', 'meaningful', 'hopeful']
        }
        
        texts = []
        labels = []
        
        # Generate texts for each category
        for label in [0, 1, 2]:
            count = size // 3
            for _ in range(count):
                template = np.random.choice(templates[label])
                
                # Fill template with random words
                filled_template = template
                for placeholder in words:
                    if f'{{{placeholder}}}' in filled_template:
                        word = np.random.choice(words[placeholder])
                        filled_template = filled_template.replace(f'{{{placeholder}}}', word)
                
                texts.append(filled_template)
                labels.append(label)
        
        # Shuffle the dataset
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)
    
    def save_dataset(self, texts: List[str], labels: List[int], filename: str):
        """Save dataset to CSV file."""
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filename: str) -> Tuple[List[str], List[int]]:
        """Load dataset from CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath)
        
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        logger.info(f"Dataset loaded from {filepath}: {len(texts)} samples")
        return texts, labels
    
    def get_dataset_statistics(self, labels: List[int]) -> Dict[str, int]:
        """Get statistics about the dataset."""
        unique, counts = np.unique(labels, return_counts=True)
        stats = {
            'total_samples': len(labels),
            'normal_samples': counts[0] if 0 in unique else 0,
            'moderate_samples': counts[1] if 1 in unique else 0,
            'high_risk_samples': counts[2] if 2 in unique else 0
        }
        
        logger.info(f"Dataset statistics: {stats}")
        return stats

if __name__ == "__main__":
    # Example usage
    manager = MentalHealthDatasetManager()
    
    # Create comprehensive dataset
    texts, labels = manager.create_comprehensive_dataset(size=300)
    
    # Augment the dataset
    augmented_texts, augmented_labels = manager.augment_dataset(texts, labels, multiplier=2)
    
    # Save dataset
    manager.save_dataset(augmented_texts, augmented_labels, "mental_health_dataset.csv")
    
    # Get statistics
    stats = manager.get_dataset_statistics(augmented_labels)
    print(f"Created dataset with {stats['total_samples']} samples")
