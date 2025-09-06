from typing import Dict, List, Any
from datetime import datetime

class ConversationHistory:
    """Manages conversation history and sentiment tracking."""
    
    def __init__(self):
        self.conversations = {}  # Store conversations by session ID
        
    def add_message(self, session_id: str, message: str, sentiment: str, confidence: float):
        """Add a message to the conversation history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            
        self.conversations[session_id].append({
            'message': message,
            'sentiment': sentiment,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
    
    def get_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the conversation history for a session."""
        return self.conversations.get(session_id, [])
    
    def analyze_trend(self, session_id: str) -> Dict[str, Any]:
        """Analyze sentiment trend for a session."""
        history = self.conversations.get(session_id, [])
        if not history:
            return {'trend': 'unknown', 'messages_count': 0}
        
        sentiment_scores = {
            'normal': 1,
            'moderate': 2,
            'high': 3
        }
        
        recent_messages = history[-3:]  # Look at last 3 messages
        scores = [sentiment_scores.get(msg['sentiment'], 0) for msg in recent_messages]
        
        if len(scores) >= 2:
            if scores[-1] < scores[-2]:  # Most recent better than previous
                trend = 'improving'
            elif scores[-1] > scores[-2]:  # Most recent worse than previous
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
            
        return {
            'trend': trend,
            'messages_count': len(history)
        }
