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
            Openness reflects the degree of intellectual curiosity, creativity, and a preference for novelty and variety.
            """,
            "conscientiousness": """
            Conscientiousness denotes a person's degree of self-discipline, act dutifully, and aim for achievement.
            """,
            "extraversion": """
            Extraversion is characterized by sociability, talkativeness, assertiveness, and high amounts of emotional expressiveness.
            """,
            "agreeableness": """
            Agreeableness reflects individual differences in general concern for social harmony.
            """,
            "neuroticism": """
            Neuroticism refers to the tendency to experience negative emotions, such as anxiety, anger, or depression.
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
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.llama_endpoint, json=payload, timeout=60)
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
            if line.startswith("OPENNESS:"):
                scores['openness'] = float(line.split(":")[1].split("-")[0].strip())
                explanations['openness'] = line.split("-")[1].strip()
            elif line.startswith("CONSCIENTIOUSNESS:"):
                scores['conscientiousness'] = float(line.split(":")[1].split("-")[0].strip())
                explanations['conscientiousness'] = line.split("-")[1].strip()
            elif line.startswith("EXTRAVERSION:"):
                scores['extraversion'] = float(line.split(":")[1].split("-")[0].strip())
                explanations['extraversion'] = line.split("-")[1].strip()
            elif line.startswith("AGREEABLENESS:"):
                scores['agreeableness'] = float(line.split(":")[1].split("-")[0].strip())
                explanations['agreeableness'] = line.split("-")[1].strip()
            elif line.startswith("NEUROTICISM:"):
                scores['neuroticism'] = float(line.split(":")[1].split("-")[0].strip())
                explanations['neuroticism'] = line.split("-")[1].strip()
            elif line.startswith("OVERALL PERSONALITY TYPE:"):
                overall_summary = line.split(":")[1].strip()
        
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
            'overall_summary': llama_scores['overall_summary']
        }
        
        # Combine scores from different methods
        for trait in self.personality_traits:
            # Get scores from different methods
            integrated_analysis['personality_scores'][trait] = {
                'final_score': (trait_similarities[trait]['trait_score'] + llama_scores['scores'][trait]) / 2,
                'explanation': llama_scores['explanations'][trait]
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