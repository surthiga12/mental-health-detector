"""
Fixed Crisis Detector - Immediate safety implementation
"""

import re
from typing import Dict, Any

class CrisisDetector:
    """Robust crisis detection system."""
    
    def __init__(self):
        # Simple but comprehensive crisis patterns
        self.crisis_phrases = [
            "dont want to live", "don't want to live", "do not want to live",
            "want to die", "wish i was dead", "kill myself", "end my life",
            "suicide", "suicidal", "self harm", "self-harm", 
            "no reason to live", "life is meaningless", "nothing matters",
            "can't go on", "cant go on", "give up on life",
            "better off dead", "hate myself", "worthless", "hopeless",
            "want to disappear", "end it all", "no point in living"
        ]
        
        # Crisis word combinations
        self.crisis_words = ["die", "death", "kill", "end", "suicide", "harm"]
        self.self_references = ["myself", "i want", "i wish", "i feel", "i am", "i don't", "i dont"]
    
    def detect_crisis(self, text: str) -> bool:
        """Detect crisis indicators in text."""
        text_lower = text.lower().strip()
        
        # Direct phrase matching
        for phrase in self.crisis_phrases:
            if phrase in text_lower:
                return True
        
        # Combination detection
        has_crisis_word = any(word in text_lower for word in self.crisis_words)
        has_self_ref = any(ref in text_lower for ref in self.self_references)
        
        if has_crisis_word and has_self_ref:
            # Additional check for negative context that's NOT negation
            positive_words = ["want to help", "save", "prevent", "support"]
            if not any(pos in text_lower for pos in positive_words):
                return True
        
        return False
    
    def get_crisis_response(self) -> str:
        """Get appropriate crisis response."""
        responses = [
            "I'm very concerned about what you've shared. Your life has value and you matter. Please reach out for immediate help: Call 1056 (KIRAN Mental Health Helpline) or 112 (Emergency). You don't have to face this alone.",
            
            "What you're experiencing sounds incredibly difficult and I'm worried about you. Please contact crisis support immediately: 1056 (KIRAN) or 9152987821 (Suicide Prevention India). There are people who want to help you through this.",
            
            "I hear that you're in serious pain right now. These feelings can change with proper support. Please reach out for immediate help: Call 1056 (KIRAN), 112 (Emergency), or go to your nearest hospital. Your life matters."
        ]
        import random
        return random.choice(responses)

# Test the detector
def test_crisis_detector():
    detector = CrisisDetector()
    
    test_cases = [
        ("i dont want to live anymore", True),
        ("feeling sad today", False),
        ("want to kill myself", True),
        ("life is great", False),
        ("i wish i was dead", True),
        ("worried about work", False)
    ]
    
    print("🧪 Testing Crisis Detector:")
    all_correct = True
    
    for text, expected in test_cases:
        result = detector.detect_crisis(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{text}' -> Crisis: {result} (expected: {expected})")
        if result != expected:
            all_correct = False
    
    if all_correct:
        print("✅ All tests passed! Crisis detection working correctly.")
    else:
        print("❌ Some tests failed. Review detection logic.")
    
    return all_correct

if __name__ == "__main__":
    test_crisis_detector()
