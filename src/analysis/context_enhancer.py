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
                return json.load(f)
        except FileNotFoundError:
            return {}
    
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
            'timestamp': datetime.now().isoformat()
        }
        
        return enhanced_analysis
    
    def _find_suitable_contexts(self, personality_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find cultural contexts that match personality profile"""
        suitable_contexts = []
        
        for context_name, context_data in self.cultural_contexts.items():
            suitability_score = self._calculate_context_suitability(personality_scores, context_data)
            if suitability_score > 0.5:  # Arbitrary threshold for suitability
                suitable_contexts.append({
                    'context_name': context_name,
                    'suitability_score': suitability_score,
                    'roles': self._get_matching_roles(personality_scores, context_data)
                })
        
        return suitable_contexts[:3]
    
    def _calculate_context_suitability(self, personality_scores: Dict[str, Any], 
                                     context_data: Dict[str, Any]) -> float:
        total_score = sum(personality_scores[trait]['final_score'] for trait in context_data['typical_roles'])
        total_traits = len(context_data['typical_roles'])
        return total_score / max(total_traits, 1)
    
    def _get_matching_roles(self, personality_scores: Dict[str, Any], 
                          context_data: Dict[str, Any]) -> List[str]:
        unique_roles = list(set(context_data['typical_roles']))
        return unique_roles[:5]
    
    def _generate_context_recommendations(self, suitable_contexts: List[Dict], 
                                        personality_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        recommendations = []
        for context in suitable_contexts:
            recommendations.append({
                'context_name': context['context_name'],
                'explanation': self._generate_personality_explanation(personality_scores, self.cultural_contexts[context['context_name']])
            })
        return recommendations
    
    def _generate_personality_explanation(self, personality_scores: Dict[str, Any],
                                        context_data: Dict[str, Any]) -> str:
        explanations = []
        for trait, data in personality_scores.items():
            explanations.append(f"{trait.title()} is reflected in the typical roles of {', '.join(context_data['typical_roles'])}.")
        return ". ".join(explanations[:2])  # Return top 2 explanations
    
    def create_llama_context_prompt(self, enhanced_analysis: Dict[str, Any]) -> str:
        """
        Create a prompt for LLAMA based on enhanced personality analysis
        
        Args:
            enhanced_analysis: Output from enhance_with_cultural_context
            
        Returns:
            Formatted prompt string for LLAMA
        """
        personality_scores = enhanced_analysis['personality_scores']
        prompt = f"""Based on comprehensive personality analysis, predict a past life for this individual:

PERSONALITY PROFILE:
"""
        
        # Add personality scores
        for trait, data in personality_scores.items():
            prompt += f"- {trait.title()}: {data['final_score']:.1f}/10 ({data['level']}) - {data['explanation'][:50]}...\n"
        
        prompt += f"""
RECOMMENDED HISTORICAL CONTEXT:
Suitable Periods: {', '.join([context['context_name'] for context in enhanced_analysis['cultural_context']['suitable_periods']])}

TASK: Based on this analysis, predict a specific past life that includes:
1. A specific name and identity
2. The exact role/occupation from the recommended period
3. Key personality traits manifested in that historical context
4. Specific challenges and achievements that align with the personality profile
5. Cultural and historical details that make the story authentic

Please provide a structured response with the past life details that will be used for story generation.
"""
        
        return prompt