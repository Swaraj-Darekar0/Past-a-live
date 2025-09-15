class StoryGenerator:
    """
    Generate personalized past life narratives using Google Gemini
    
    Academic Purpose: Demonstrate integration of NLP analysis with
    modern generative AI for creative content production
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the StoryGenerator with API key and model name
        
        Args:
            api_key: API key for Google Gemini
            model_name: Name of the model to use for generation
        """
        self.api_key = api_key
        self.model_name = model_name
        self.story_templates = self._load_story_templates()
    
    def _load_story_templates(self) -> Dict[str, str]:
        """Load narrative templates for different story styles"""
        return {
            "immersive_narrative": "Once upon a time in a land far away...",
            "historical_fiction": "In the year 1500, a young adventurer named...",
        }
    
    def generate_past_life_story(self, llama_prediction: str, 
                                  enhanced_analysis: Dict[str, Any]) -> str:
        """
        Generate a past life story based on LLAMA prediction and enhanced analysis
        
        Args:
            llama_prediction: Prediction from the LLAMA model
            enhanced_analysis: Enhanced personality analysis with context
        
        Returns:
            Generated past life story
        """
        prediction_data = self._parse_llama_prediction(llama_prediction)
        prompt = self._create_story_prompt(prediction_data, enhanced_analysis)
        story = self._call_gemini_api(prompt)
        return story
    
    def _parse_llama_prediction(self, llama_output: str) -> Dict[str, str]:
        """Parse structured output from LLAMA past life prediction"""
        prediction_data = {}
        lines = llama_output.split('\n')
        for line in lines:
            if line.strip():
                key, value = line.split(':', 1)
                prediction_data[key.strip()] = value.strip()
        return prediction_data
    
    def _create_story_prompt(self, prediction_data: Dict[str, str],
                             enhanced_analysis: Dict[str, Any]) -> str:
        """
        Create a prompt for the story generation model
        
        Args:
            prediction_data: Parsed prediction data from LLAMA
            enhanced_analysis: Enhanced analysis with cultural context
        
        Returns:
            Formatted prompt for story generation
        """
        prompt = f"""You are a master storyteller specializing in historical fiction. Create a compelling past life story based on detailed personality analysis and historical research.

CHARACTER PROFILE:
Name: {prediction_data['name']}
Historical Period: {prediction_data['period']}
Occupation/Role: {prediction_data['occupation']}
Cultural Context: {enhanced_analysis['cultural_context']['suitable_periods']}

PERSONALITY TRAITS (Big Five Analysis):
"""
        for trait, score in prediction_data['personality_scores'].items():
            prompt += f"- {trait.title()}: {score}\n"
        
        prompt += f"""
NARRATIVE REQUIREMENTS:
{self.story_templates['immersive_narrative']}

Create a complete past life story that feels authentic, emotionally engaging, and true to both the personality analysis and historical period."""
        
        return prompt
    
    def _call_gemini_api(self, prompt: str, max_retries: int = 3) -> str:
        """Make API call to Gemini with error handling and retries"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.gemini.com/generate",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"prompt": prompt, "model": self.model_name}
                )
                response.raise_for_status()
                return response.json().get("story", "")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return "Error generating story."
    
    def _structure_story_output(self, story_content: str, 
                                 prediction_data: Dict[str, str]) -> Dict[str, str]:
        """Structure the generated story output for better organization"""
        sections = {
            "introduction": "",
            "body": "",
            "conclusion": ""
        }
        # Logic to split story_content into sections
        return sections
    
    def _assess_personality_alignment(self, story_content: str, 
                                      personality_scores: Dict[str, Any]) -> float:
        """Assess the alignment of the story with personality traits"""
        alignment_score = 0.0
        # Logic to calculate alignment score
        return round(alignment_score, 3)
    
    def _assess_historical_elements(self, story_content: str) -> float:
        """Assess the presence of historical elements in the story"""
        authenticity_score = 0.0
        # Logic to calculate authenticity score
        return round(authenticity_score, 3)
    
    def _assess_narrative_quality(self, story_content: str) -> float:
        """Assess the overall narrative quality of the story"""
        quality_score = 0.0
        # Logic to calculate narrative quality score
        return round(min(quality_score, 1.0), 3)