from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class RecommendationSystem:
    """System for providing mental health recommendations based on risk level."""
    
    def __init__(self):
        """Initialize the recommendation system with predefined resources."""
        self.resources = {
            "normal": [
                {
                    "type": "wellness",
                    "title": "Mindfulness Meditation",
                    "description": "Daily meditation practice for mental wellness",
                    "link": "https://www.mindful.org"
                },
                {
                    "type": "exercise",
                    "title": "Physical Activity",
                    "description": "Regular exercise routines for mental health",
                    "link": "https://www.health.gov/fitness"
                }
            ],
            "moderate": [
                {
                    "type": "crisis",
                    "title": "24/7 Crisis Helpline",
                    "description": "Immediate support available at 988 (US) or your local crisis hotline",
                    "link": "https://988lifeline.org"
                },
                {
                    "type": "professional_help",
                    "title": "Talk to a Mental Health Professional",
                    "description": "Connect with a counselor or therapist",
                    "link": "https://www.psychologytoday.com/find-therapist"
                },
                {
                    "type": "support_group",
                    "title": "Support Communities",
                    "description": "Connect with others who understand",
                    "link": "https://www.7cups.com"
                }
            ],
            "high": [
                {
                    "type": "emergency",
                    "title": "Emergency Help - Call 911 (US)",
                    "description": "If you're having thoughts of suicide or self-harm, please get immediate help",
                    "link": "tel:911"
                },
                {
                    "type": "crisis",
                    "title": "988 Suicide & Crisis Lifeline",
                    "description": "24/7 support - Call or Text 988",
                    "link": "https://988lifeline.org"
                },
                {
                    "type": "immediate_support",
                    "title": "Crisis Text Line",
                    "description": "Text HOME to 741741 to connect with a Crisis Counselor",
                    "link": "https://www.crisistextline.org"
                }
            ]
        }
    
    def get_recommendations(self, risk_level: str) -> List[Dict]:
        """
        Get personalized recommendations based on risk level.
        
        Args:
            risk_level (str): Detected risk level ('normal', 'moderate', or 'high')
            
        Returns:
            List[Dict]: List of recommended resources
        """
        try:
            recommendations = self.resources.get(risk_level.lower(), [])
            
            if not recommendations:
                logger.warning(f"No recommendations found for risk level: {risk_level}")
                return []
            
            logger.info(f"Found {len(recommendations)} recommendations for risk level: {risk_level}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def add_resource(self, risk_level: str, resource: Dict) -> bool:
        """
        Add a new resource to the recommendation system.
        
        Args:
            risk_level (str): Risk level for the resource
            resource (Dict): Resource information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            required_fields = ['type', 'title', 'description', 'link']
            if not all(field in resource for field in required_fields):
                logger.error("Missing required fields in resource")
                return False
            
            if risk_level not in self.resources:
                self.resources[risk_level] = []
                
            self.resources[risk_level].append(resource)
            logger.info(f"Added new resource for risk level: {risk_level}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding resource: {str(e)}")
            return False
