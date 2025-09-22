import json
from datetime import datetime
from typing import Dict, Any, List
import random

class ContextEnhancer:
    """
    Enhance personality analysis with cultural and historical context
    
    Academic Purpose: Demonstrate context-aware NLP systems and
    cultural adaptation of AI outputs
    """
    
    def __init__(self):
        """Initialize context enhancer with cultural databases"""
        self.cultural_contexts = self._load_cultural_contexts()
        self.historical_periods = self._extract_historical_periods()
        
    def _load_cultural_contexts(self) -> Dict[str, Any]:
        """Load cultural context database"""
        try:
            with open('data/cultural_contexts.json', 'r') as f:
                return json.load(f)['cultural_contexts']
        except FileNotFoundError:
            # Fallback for development
            return {
                "ancient_greece": {
                    "time_period": "800-146 BCE",
                    "personality_fit": {
                        "high_openness": ["philosopher", "playwright"],
                        "high_extraversion": ["politician", "orator"],
                    },
                    "cultural_values": ["wisdom", "honor", "civic_duty"],
                    "historical_context": "Golden age of philosophy and democracy."
                },
                "renaissance_italy": {
                    "time_period": "1300-1600 CE",
                    "personality_fit": {
                        "high_openness": ["artist", "inventor"],
                        "high_conscientiousness": ["banker", "engineer"],
                    },
                    "cultural_values": ["innovation", "artistic_beauty", "humanism"],
                    "historical_context": "Rebirth of classical learning and artistic innovation."
                }
            }
    
    def _extract_historical_periods(self) -> List[str]:
        """Extract list of available historical periods"""
        return list(self.cultural_contexts.keys())
    
    def enhance_with_cultural_context(self, personality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance personality analysis with appropriate cultural contexts
        
        Academic Value: Demonstrates cultural adaptation and context-aware
        system design in NLP applications
        
        Args:
            personality_analysis: Output from PersonalityAnalyzer
            
        Returns:
            Enhanced analysis with cultural context recommendations
        """
        personality_scores = personality_analysis['personality_scores']
        dominant_traits = personality_analysis['personality_summary']['dominant_traits']
        
        # Find suitable cultural contexts
        suitable_contexts = self._find_suitable_contexts(personality_scores)
        
        # Generate context recommendations
        context_recommendations = self._generate_context_recommendations(
            suitable_contexts, personality_scores
        )
        
        # Create enhanced analysis
        enhanced_analysis = personality_analysis.copy()
        enhanced_analysis['cultural_context'] = {
            'suitable_periods': suitable_contexts,
            'recommendations': context_recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return enhanced_analysis
    
    def _find_suitable_contexts(self, personality_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find cultural contexts that match personality profile"""
        suitable_contexts = []
        
        for context_name, context_data in self.cultural_contexts.items():
            suitability_score = self._calculate_context_suitability(
                personality_scores, context_data
            )
            
            if suitability_score > 0.3:  # Threshold for suitability
                suitable_contexts.append({
                    'context_name': context_name,
                    'context_data': context_data,
                    'suitability_score': suitability_score,
                    'matching_roles': self._get_matching_roles(personality_scores, context_data)
                })
        
        # Sort by suitability score
        suitable_contexts.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        # Return top 3 contexts
        return suitable_contexts[:3]
    
    def _calculate_context_suitability(self, personality_scores: Dict[str, Any], 
                                     context_data: Dict[str, Any]) -> float:
        """Calculate how well personality profile fits cultural context"""
        personality_fit = context_data.get('personality_fit', {})
        total_score = 0
        total_traits = 0
        
        for trait, trait_data in personality_scores.items():
            score = trait_data['final_score']
            level = trait_data.get('level', self._categorize_score(score))
            
            # Check for high trait matches
            if level == "High":
                high_trait_key = f"high_{trait}"
                if high_trait_key in personality_fit:
                    roles = personality_fit[high_trait_key]
                    role_score = len(roles) / 4.0  # Normalize by typical role count
                    total_score += role_score * (score / 10.0)
                    total_traits += 1
            
            # Check for low trait matches (especially for neuroticism)
            elif level == "Low" and trait == "neuroticism":
                low_neuroticism_key = "low_neuroticism"
                if low_neuroticism_key in personality_fit:
                    roles = personality_fit[low_neuroticism_key]
                    role_score = len(roles) / 4.0
                    total_score += role_score * ((10 - score) / 10.0)
                    total_traits += 1
        
        return total_score / max(total_traits, 1)
    
    def _get_matching_roles(self, personality_scores: Dict[str, Any], 
                          context_data: Dict[str, Any]) -> List[str]:
        """Get roles that match the personality profile in this context"""
        matching_roles = []
        personality_fit = context_data.get('personality_fit', {})
        
        for trait, trait_data in personality_scores.items():
            score = trait_data['final_score']
            level = trait_data.get('level', self._categorize_score(score))
            
            if level == "High":
                high_trait_key = f"high_{trait}"
                if high_trait_key in personality_fit:
                    matching_roles.extend(personality_fit[high_trait_key])
            elif level == "Low" and trait == "neuroticism":
                low_neuroticism_key = "low_neuroticism"
                if low_neuroticism_key in personality_fit:
                    matching_roles.extend(personality_fit[low_neuroticism_key])
        
        # Remove duplicates and return top matches
        unique_roles = list(set(matching_roles))
        return unique_roles[:5]
    
    def _generate_context_recommendations(self, suitable_contexts: List[Dict], 
                                        personality_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed recommendations for each suitable context"""
        recommendations = []
        
        for context in suitable_contexts:
            context_name = context['context_name']
            context_data = context['context_data']
            matching_roles = context['matching_roles']
            
            # Select best role match
            primary_role = matching_roles[0] if matching_roles else "citizen"
            
            # Create recommendation
            recommendation = {
                'period_name': context_name.replace('_', ' ').title(),
                'time_period': context_data.get('time_period', 'Unknown'),
                'suitability_score': context['suitability_score'],
                'primary_role': primary_role,
                'alternative_roles': matching_roles[1:4],
                'cultural_values': context_data.get('cultural_values', []),
                'historical_context': context_data.get('historical_context', ''),
                'personality_explanation': self._generate_personality_explanation(
                    personality_scores, context_data, primary_role
                )
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_personality_explanation(self, personality_scores: Dict[str, Any],
                                        context_data: Dict[str, Any], 
                                        primary_role: str) -> str:
        """Generate explanation of why this personality fits this context"""
        explanations = []
        
        # Find dominant traits that match this context
        for trait, trait_data in personality_scores.items():
            score = trait_data['final_score']
            level = trait_data.get('level', self._categorize_score(score))
            if level == "High":
                if trait == "openness" and score > 7:
                    explanations.append(f"Your high openness ({score:.1f}/10) aligns with this period's emphasis on innovation and art")
                elif trait == "conscientiousness" and score > 7:
                    explanations.append(f"Your conscientiousness ({score:.1f}/10) matches the structured, duty-oriented nature of roles like {primary_role}")
        
        # Default explanation if no specific matches
        if not explanations:
            explanations.append(f"Your personality profile shows strong compatibility with the values of this period")
        
        return ". ".join(explanations[:2])  # Return top 2 explanations
    
    def create_llama_context_prompt(self, enhanced_analysis: Dict[str, Any]) -> str:
        """
        Create a prompt for LLAMA based on enhanced personality analysis
        
        Args:
            enhanced_analysis: Output from enhance_with_cultural_context
            
        Returns:
            Formatted prompt string for LLAMA
        """
        personality_scores = enhanced_analysis.get('personality_scores', {})
        recommendations = enhanced_analysis.get('cultural_context', {}).get('recommendations', [])
        
        # Select primary recommendation
        primary_rec = recommendations[0] if recommendations else None
        
        if not primary_rec:
            return "Generate a general past life based on the personality analysis."
            
        prompt = f"""Based on comprehensive personality analysis, predict a past life for this individual:

PERSONALITY PROFILE:
"""
        
        # Add personality scores
        for trait, data in personality_scores.items():
            level = data.get('level', self._categorize_score(data['final_score']))
            explanation = data.get('explanation', '')[:50]
            prompt += f"- {trait.title()}: {data['final_score']:.1f}/10 ({level}) - {explanation}...\n"
        
        prompt += f"""
RECOMMENDED HISTORICAL CONTEXT:
Period: {primary_rec['period_name']} ({primary_rec['time_period']})
Historical Context: {primary_rec['historical_context']}
Cultural Values: {', '.join(primary_rec['cultural_values'])}
Primary Role Match: {primary_rec['primary_role']}
Alternative Roles: {', '.join(primary_rec['alternative_roles'])}

PERSONALITY-CONTEXT ALIGNMENT:
{primary_rec['personality_explanation']}

TASK: Based on this analysis, predict a specific past life that includes:
1. A specific name and identity
2. The exact role/occupation from the recommended period
3. Key personality traits manifested in that historical context
4. Specific challenges and achievements that align with the personality profile
5. Cultural and historical details that make the story authentic

Please provide a structured response with the past life details that will be used for story generation.

Format your response as:
NAME: [historical name]
PERIOD: [specific time and place]
OCCUPATION: [primary role]
PERSONALITY_MANIFESTATION: [how traits showed in that era]
KEY_LIFE_EVENTS: [major events that reflect personality]
CULTURAL_INTEGRATION: [how they fit into their society]
"""
        
        return prompt

    def _categorize_score(self, score: float) -> str:
        """Categorize personality score into levels"""
        if score <= 3.5:
            return "Low"
        elif score <= 7.5:
            return "Medium"
        else:
            return "High"