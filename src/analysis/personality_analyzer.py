import json
import requests
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

class PersonalityAnalyzer:
    """
    Analyze personality traits using local LLAMA model
    
    Academic Purpose: Demonstrate integration of traditional NLP features
    with modern LLM analysis for psychological assessment
    """
    
    def __init__(self, llama_endpoint: str = "http://localhost:11434/api/generate"):
        """
        Initialize personality analyzer
        
        Args:
            llama_endpoint: Ollama API endpoint for local LLAMA model
        """
        self.llama_endpoint = llama_endpoint
        self.model_name = "llama2:7b"  # Can be configured
        
        # Big Five personality traits
        self.personality_traits = [
            "openness", "conscientiousness", "extraversion", 
            "agreeableness", "neuroticism"
        ]
        
        # Load trait descriptions for context
        self.trait_descriptions = self._load_trait_descriptions()
    
    def _load_trait_descriptions(self) -> Dict[str, str]:
        """Load detailed trait descriptions for LLAMA context"""
        return {
            "openness": """
            Openness to Experience: Reflects degree of intellectual curiosity, creativity, 
            and preference for novelty. High: creative, imaginative, curious, artistic, 
            values variety. Low: practical, conventional, prefers routine, traditional.
            """,
            "conscientiousness": """
            Conscientiousness: Reflects tendency to be organized, responsible, and 
            self-disciplined. High: organized, disciplined, careful, thorough, reliable. 
            Low: spontaneous, flexible, disorganized, casual, careless.
            """,
            "extraversion": """
            Extraversion: Reflects energy direction and social behavior. High: outgoing, 
            energetic, talkative, assertive, seeks social stimulation. Low: reserved, 
            quiet, independent, prefers solitude, thoughtful.
            """,
            "agreeableness": """
            Agreeableness: Reflects interpersonal orientation. High: trusting, helpful, 
            forgiving, straightforward, compassionate. Low: skeptical, competitive, 
            challenging, self-interested, critical.
            """,
            "neuroticism": """
            Neuroticism: Reflects emotional stability. High: anxious, moody, worrying, 
            sensitive to stress, emotionally reactive. Low: calm, relaxed, secure, 
            hardy, emotionally stable.
            """
        }
    
    def _call_llama_api(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Make API call to local LLAMA model
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Generated text response from LLAMA
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",  # Keep the model in memory for 5 minutes
            "options": {
                "temperature": temperature,
                "num_predict": 500,
                "top_p": 0.9
            }
        }
        
        try:
            # Increase timeout to allow for model loading on first run
            response = requests.post(self.llama_endpoint, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLAMA API: {e}")
            return ""
    
    def analyze_personality_from_features(self, comprehensive_features: Dict) -> Dict[str, Any]:
        """
        Analyze personality traits using NLP features and LLAMA model
        
        Academic Value: Demonstrates integration of quantitative NLP analysis
        with qualitative LLM assessment for comprehensive personality evaluation
        
        Args:
            comprehensive_features: Output from FeatureExtractor
            
        Returns:
            Detailed personality analysis with scores and explanations
        """
        # Extract key information from features
        trait_similarities = comprehensive_features['embeddings']['trait_similarities']
        linguistic_patterns = comprehensive_features['linguistic_patterns']['trait_indicators']
        text_stats = comprehensive_features['text_statistics']
        
        # Create analysis prompt for LLAMA
        analysis_prompt = self._create_personality_analysis_prompt(
            trait_similarities, linguistic_patterns, text_stats
        )
        
        # Get LLAMA analysis
        llama_response = self._call_llama_api(analysis_prompt)
        
        # Parse LLAMA response and create structured analysis
        personality_scores = self._parse_llama_personality_response(llama_response)
        
        # Combine NLP features with LLAMA analysis
        comprehensive_analysis = self._integrate_analyses(
            trait_similarities, linguistic_patterns, personality_scores, llama_response
        )
        
        return comprehensive_analysis
    
    def _create_personality_analysis_prompt(self, trait_similarities: Dict, 
                                          linguistic_patterns: Dict, 
                                          text_stats: Dict) -> str:
        """Create comprehensive prompt for LLAMA personality analysis"""
        
        prompt = f"""You are a personality psychology expert. Analyze the following data from a person's text responses and provide Big Five personality trait scores.

TEXT ANALYSIS DATA:

Semantic Similarity Scores (how similar the text is to trait descriptions):
"""
        
        for trait, scores in trait_similarities.items():
            prompt += f"\n{trait.title()}: {scores['trait_score']:.3f} (confidence: {scores['confidence']:.3f})"
        
        prompt += "\n\nLinguistic Pattern Analysis:"
        for trait, patterns in linguistic_patterns.items():
            prompt += f"\n{trait.title()}: {patterns['trait_lean']} lean (confidence: {patterns['confidence']:.3f})"
        
        prompt += f"\n\nText Statistics:"
        prompt += f"\nVocabulary richness: {text_stats.get('unique_words', 0)} unique words"
        prompt += f"\nAverage sentence length: {text_stats.get('avg_sentence_length', 0):.2f} words"
        prompt += f"\nTotal words: {text_stats.get('word_count', 0)}"
        
        prompt += f"""

PERSONALITY TRAIT DESCRIPTIONS:
{self.trait_descriptions['openness']}
{self.trait_descriptions['conscientiousness']}
{self.trait_descriptions['extraversion']}
{self.trait_descriptions['agreeableness']}
{self.trait_descriptions['neuroticism']}

TASK: Based on the analysis data above, provide personality scores for each Big Five trait on a scale of 1-10, along with brief explanations.

Please format your response as follows:
OPENNESS: [score 1-10] - [brief explanation]
CONSCIENTIOUSNESS: [score 1-10] - [brief explanation]
EXTRAVERSION: [score 1-10] - [brief explanation]
AGREEABLENESS: [score 1-10] - [brief explanation]
NEUROTICISM: [score 1-10] - [brief explanation]

OVERALL PERSONALITY TYPE: [2-3 sentence summary]
"""
        
        return prompt
    
    def _parse_llama_personality_response(self, llama_response: str) -> Dict[str, Any]:
        """Parse LLAMA response to extract personality scores"""
        scores = {}
        explanations = {}
        overall_summary = ""
        
        lines = llama_response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for trait scores
            for trait in self.personality_traits:
                trait_upper = trait.upper()
                if line.startswith(trait_upper + ":"):
                    try:
                        # Extract score and explanation
                        content = line[len(trait_upper + ":"):]
                        parts = content.split(" - ", 1)
                        
                        # Extract score
                        score_part = parts[0].strip()
                        score = int(''.join(filter(str.isdigit, score_part)))
                        scores[trait] = min(max(score, 1), 10)  # Ensure 1-10 range
                        
                        # Extract explanation
                        if len(parts) > 1:
                            explanations[trait] = parts[1].strip()
                        else:
                            explanations[trait] = "Analysis based on text patterns"
                            
                    except (ValueError, IndexError):
                        scores[trait] = 5  # Default middle score
                        explanations[trait] = "Unable to parse detailed analysis"
            
            # Look for overall summary
            if line.startswith("OVERALL PERSONALITY TYPE:"):
                overall_summary = line[len("OVERALL PERSONALITY TYPE:"):].strip()
        
        # Ensure all traits have scores to prevent KeyErrors
        for trait in self.personality_traits:
            if trait not in scores:
                scores[trait] = 5
                explanations[trait] = "Default score assigned due to parsing or API error."
        
        return {
            'scores': scores,
            'explanations': explanations,
            'overall_summary': overall_summary
        }
    
    def _integrate_analyses(self, trait_similarities: Dict, linguistic_patterns: Dict,
                          llama_scores: Dict, llama_response: str) -> Dict[str, Any]:
        """Integrate NLP analysis with LLAMA analysis for comprehensive results"""
        
        integrated_analysis = {
            'personality_scores': {},
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'analysis_method': 'hybrid_nlp_llm'
            },
            'detailed_analysis': {
                'llama_raw_response': llama_response,
                'nlp_features': {
                    'semantic_similarities': trait_similarities,
                    'linguistic_patterns': linguistic_patterns
                }
            }
        }
        
        # Combine scores from different methods
        for trait in self.personality_traits:
            # Get scores from different methods
            llama_score = llama_scores['scores'].get(trait, 5)
            semantic_score = self._convert_similarity_to_score(
                trait_similarities.get(trait, {}).get('trait_score', 0)
            )
            
            # Calculate weighted combined score
            combined_score = (0.7 * llama_score + 0.3 * semantic_score)
            
            # Calculate confidence based on agreement between methods
            score_agreement = 1.0 - abs(llama_score - semantic_score) / 10.0
            base_confidence = trait_similarities.get(trait, {}).get('confidence', 0.5)
            final_confidence = (score_agreement + base_confidence) / 2
            
            integrated_analysis['personality_scores'][trait] = {
                'final_score': round(combined_score, 2),
                'llama_score': llama_score,
                'semantic_score': round(semantic_score, 2),
                'confidence': round(final_confidence, 3),
                'explanation': llama_scores['explanations'].get(trait, ''),
                'level': self._categorize_score(combined_score)
            }
        
        # Add overall personality summary
        integrated_analysis['personality_summary'] = {
            'dominant_traits': self._identify_dominant_traits(integrated_analysis['personality_scores']),
            'personality_type': llama_scores.get('overall_summary', ''),
            'key_characteristics': self._generate_key_characteristics(integrated_analysis['personality_scores'])
        }
        
        return integrated_analysis
    
    def _convert_similarity_to_score(self, similarity: float) -> float:
        """Convert semantic similarity (-1 to 1) to personality score (1 to 10)"""
        # Map similarity score to 1-10 range
        # -1 to 1 becomes 1 to 10
        normalized = (similarity + 1) / 2  # 0 to 1
        score = 1 + (normalized * 9)  # 1 to 10
        return max(1, min(10, score))
    
    def _categorize_score(self, score: float) -> str:
        """Categorize personality score into levels"""
        if score <= 3:
            return "Low"
        elif score <= 7:
            return "Medium"
        else:
            return "High"

    def _identify_dominant_traits(self, personality_scores: Dict) -> List[str]:
        """Identify the most prominent personality traits"""
        sorted_traits = sorted(
            personality_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )
        
        # Return top 2-3 traits above average (5.5)
        dominant = [trait for trait, data in sorted_traits[:3] 
                   if data['final_score'] > 5.5]
        
        return dominant if dominant else [sorted_traits[0][0]]
    
    def _generate_key_characteristics(self, personality_scores: Dict) -> List[str]:
        """Generate key personality characteristics based on scores"""
        characteristics = []
        
        for trait, data in personality_scores.items():
            score = data['final_score']
            level = data['level']
            
            if level == "High":
                if trait == "openness":
                    characteristics.append("Creative and open to new experiences")
                elif trait == "conscientiousness":
                    characteristics.append("Organized and goal-oriented")
                elif trait == "extraversion":
                    characteristics.append("Outgoing and socially energetic")
                elif trait == "agreeableness":
                    characteristics.append("Cooperative and trusting")
                elif trait == "neuroticism":
                    characteristics.append("Emotionally sensitive")
            elif level == "Low":
                if trait == "neuroticism":
                    characteristics.append("Emotionally stable and calm")
                elif trait == "extraversion":
                    characteristics.append("Independent and reflective")
        
        return characteristics[:4]  # Return top 4 characteristics