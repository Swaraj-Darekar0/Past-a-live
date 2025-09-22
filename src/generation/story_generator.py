import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime
import re

class StoryGenerator:
    """
    Generate personalized past life narratives using Google Gemini
    
    Academic Purpose: Demonstrate integration of NLP analysis with
    modern generative AI for creative content production
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize story generator with Gemini API
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use for generation
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini API
        if api_key and api_key != "your-gemini-api-key-here":
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(model_name)
        
        # Story generation parameters
        self.generation_config = {
            'temperature': 0.8,  # Higher creativity for storytelling
            'max_output_tokens': 1024, # Reduced for faster response and shorter stories
            'top_p': 0.9,
            'top_k': 40
        }
        
        # Story templates for different narrative styles
        self.story_templates = self._load_story_templates()
    
    def _load_story_templates(self) -> Dict[str, str]:
        """Load narrative templates for different story styles"""
        return {
            "immersive_narrative": """
            Write an engaging, immersive past life story in second person ("You were...") 
            that feels like a vivid memory. Include:
            - Sensory details (what they saw, heard, felt)
            - Emotional moments and internal thoughts
            - Specific historical details and cultural context
            - Key life events that shaped their character
            - How their personality traits manifested in that era
            """,
            
            "biographical_chronicle": """
            Write a biographical chronicle of their past life, structured like a 
            historical account. Include:
            - Birth and early life circumstances
            - Major achievements and challenges
            - Relationships and social connections
            - Legacy and impact on their community
            - How they embodied the cultural values of their time
            """,
            
            "dramatic_journey": """
            Create a dramatic narrative focusing on a pivotal moment or journey 
            in their past life. Include:
            - A central conflict or challenge they faced
            - How they used their personality strengths to overcome obstacles
            - Character development and growth throughout the story
            - Rich dialogue and scene descriptions
            - A meaningful resolution that reflects their true nature
            """
        }
    
    def generate_past_life_story(self, llama_prediction: str, 
                                  enhanced_analysis: Dict[str, Any],
                                  narrative_style: str = "immersive_narrative") -> Dict[str, Any]:
        """
        Generate complete past life story using Gemini based on LLAMA prediction
        
        Academic Value: Demonstrates advanced prompt engineering and
        integration of structured analysis with creative generation
        
        Args:
            llama_prediction: Output from LLAMA personality analysis
            enhanced_analysis: Enhanced personality analysis with context
            narrative_style: Type of narrative to generate
            
        Returns:
            Complete story with metadata and analysis
        """
        # Parse LLAMA prediction
        prediction_data = self._parse_llama_prediction(llama_prediction)
        
        # Create comprehensive story prompt
        story_prompt = self._create_story_prompt(
            prediction_data, enhanced_analysis, narrative_style
        )
        
        # Generate story using Gemini
        story_content = self._call_gemini_api(story_prompt)
        
        # Post-process and structure the story
        structured_story = self._structure_story_output(
            story_content, prediction_data, enhanced_analysis
        )
        
        return structured_story
    
    def _parse_llama_prediction(self, llama_output: str) -> Dict[str, str]:
        """Parse structured output from LLAMA past life prediction"""
        prediction_data = {
            'name': 'Unknown',
            'period': 'Ancient Times',
            'occupation': 'Citizen',
            'personality_manifestation': 'Lived according to their nature',
            'key_life_events': 'Led a meaningful life',
            'cultural_integration': 'Was part of their community'
        }
        
        # Extract structured information from LLAMA output
        patterns = {
            'name': r'NAME:\s*(.+)',
            'period': r'PERIOD:\s*(.+)',
            'occupation': r'OCCUPATION:\s*(.+)',
            'personality_manifestation': r'PERSONALITY_MANIFESTATION:\s*(.+)',
            'key_life_events': r'KEY_LIFE_EVENTS:\s*(.+)',
            'cultural_integration': r'CULTURAL_INTEGRATION:\s*(.+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, llama_output, re.IGNORECASE | re.DOTALL)
            if match:
                prediction_data[key] = match.group(1).strip()
        
        return prediction_data
    
    def _create_story_prompt(self, prediction_data: Dict[str, str],
                             enhanced_analysis: Dict[str, Any],
                             narrative_style: str) -> str:
        """Create comprehensive prompt for Gemini story generation"""
        
        personality_scores = enhanced_analysis['personality_scores']
        recommendations = enhanced_analysis.get('cultural_context', {}).get('recommendations', [])
        
        if recommendations:
            cultural_context = recommendations[0]
        else:
            # Create a fallback context if no suitable one is found
            cultural_context = {
                'period_name': 'A Timeless Era',
                'historical_context': 'A period of personal growth and discovery, unbound by a specific time or place.',
                'time_period': 'N/A',
                'cultural_values': ['self-discovery', 'resilience', 'growth']
            }
        
        # Get narrative template
        narrative_template = self.story_templates.get(narrative_style, 
                                                    self.story_templates['immersive_narrative'])
        
        prompt = f"""You are a master storyteller like "Jenny Han" author of the book 'The summer i turned pretty' . Create a compelling past life story based on detailed personality analysis and historical research.

CHARACTER PROFILE:
Name: {prediction_data['name']}
Historical Period: {prediction_data.get('period', 'Unknown')}
Occupation/Role: {prediction_data.get('occupation', 'Unknown')}
Cultural Context: {cultural_context['period_name']} - {cultural_context['historical_context']}

PERSONALITY TRAITS (Big Five Analysis):
"""
        
        for trait, data in personality_scores.items():
            prompt += f"- {trait.title()}: {data['final_score']:.1f}/10 ({data['level']}) - {data['explanation']}\n"
        
        prompt += f"""
PERSONALITY IN HISTORICAL CONTEXT:
{prediction_data.get('personality_manifestation', 'N/A')}

KEY LIFE EVENTS TO INCORPORATE:
{prediction_data.get('key_life_events', 'N/A')}

CULTURAL INTEGRATION:
{prediction_data.get('cultural_integration', 'N/A')}

HISTORICAL DETAILS TO INCLUDE:
- Time Period: {cultural_context['time_period']}
- Cultural Values: {', '.join(cultural_context['cultural_values'])}
- Typical Social Structure: Include period-appropriate social dynamics
- Historical Context: {cultural_context['historical_context']}

NARRATIVE REQUIREMENTS:
{narrative_template}

WRITING STYLE:
- Engaging and immersive prose
- Historically accurate details
- Emotionally resonant moments
- Rich sensory descriptions
- A concise story of 3-4 paragraphs (around 300-400 words).

Create a complete past life story that feels authentic, emotionally engaging, and true to both the personality analysis and historical period.
"""
        
        return prompt
    
    def _call_gemini_api(self, prompt: str, max_retries: int = 3) -> str:
        """Make API call to Gemini with error handling and retries"""
        if not self.api_key or self.api_key == "your-gemini-api-key-here":
            return self._generate_fallback_story(prompt, "API key not configured.")
            
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        **self.generation_config
                    )
                )
                
                if response.text:
                    return response.text.strip()
                else:
                    raise Exception("Empty response from Gemini API")
                    
            except Exception as e:
                print(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return self._generate_fallback_story(prompt, str(e))
        
        return self._generate_fallback_story(prompt, "Max retries exceeded.")

    def _generate_fallback_story(self, prompt: str, reason: str) -> str:
        """Generate a basic fallback story if API fails"""
        return f"""(Fallback Story: Generation failed due to '{reason}')
        In a time long past, you lived a life that reflected the unique aspects of your personality. 
        Though the specific details of that existence are shrouded in the mists of time, the essence of 
        who you were then resonates with who you are today. Your past life was shaped by the same traits 
        that define you now, manifested through the customs, challenges, and opportunities of that era."""
    
    def _structure_story_output(self, story_content: str, 
                                 prediction_data: Dict[str, str],
                                 enhanced_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the generated story with metadata and analysis"""
        
        # Extract story sections if possible
        story_sections = self._extract_story_sections(story_content)
        
        # Calculate story statistics
        word_count = len(story_content.split())
        
        # Create comprehensive output
        structured_output = {
            'story': {
                'title': f"The Past Life of {prediction_data['name']}",
                'full_narrative': story_content,
                'sections': story_sections,
                'word_count': word_count
            },
            'character_profile': {
                'name': prediction_data['name'],
                'historical_period': prediction_data['period'],
                'occupation': prediction_data['occupation'],
                'personality_summary': enhanced_analysis['personality_summary']
            },
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name
            },
            'quality_metrics': {
                'personality_alignment': self._assess_personality_alignment(
                    story_content, enhanced_analysis['personality_scores']
                ),
                'historical_authenticity': self._assess_historical_elements(story_content),
                'narrative_engagement': self._assess_narrative_quality(story_content)
            }
        }
        
        return structured_output

    def _extract_story_sections(self, story_content: str) -> Dict[str, str]:
        """Extract different sections of the story for better organization"""
        paragraphs = [p.strip() for p in story_content.split('\n\n') if p.strip()]
        
        return {
            'opening': paragraphs[0] if paragraphs else "",
            'main_narrative': '\n\n'.join(paragraphs[1:-1]) if len(paragraphs) > 2 else '\n\n'.join(paragraphs),
            'conclusion': paragraphs[-1] if len(paragraphs) > 1 else ""
        }
    
    def _assess_personality_alignment(self, story_content: str, 
                                      personality_scores: Dict[str, Any]) -> float:
        """Assess the alignment of the story with personality traits"""
        alignment_score = 0.5  # Base score
        story_lower = story_content.lower()
        matches = 0
        
        for trait, data in personality_scores.items():
            if data['level'] == "High" and any(kw in story_lower for kw in self._get_trait_keywords(trait, 'high')):
                matches += 1
            elif data['level'] == "Low" and any(kw in story_lower for kw in self._get_trait_keywords(trait, 'low')):
                matches += 1
        
        alignment_score += (matches / len(personality_scores)) * 0.5
        return round(alignment_score, 3)
    
    def _assess_historical_elements(self, story_content: str) -> float:
        """Assess the presence of historical elements in the story"""
        historical_indicators = ['century', 'kingdom', 'village', 'castle', 'merchant', 'guild', 'noble']
        found_indicators = sum(1 for indicator in historical_indicators if indicator in story_content.lower())
        authenticity_score = min(found_indicators / 4.0, 1.0)
        return round(authenticity_score, 3)
    
    def _assess_narrative_quality(self, story_content: str) -> float:
        """Assess the overall narrative quality of the story"""
        quality_score = 0.5
        if 200 < len(story_content.split()) < 600: quality_score += 0.2 # Reward stories in the target length
        if '"' in story_content: quality_score += 0.1
        sensory_words = ['saw', 'heard', 'felt', 'smelled', 'tasted']
        if any(word in story_content.lower() for word in sensory_words): quality_score += 0.2
        return round(min(quality_score, 1.0), 3)

    def _get_trait_keywords(self, trait: str, level: str) -> List[str]:
        """Helper to get keywords for trait assessment"""
        keywords = {
            'openness': {'high': ['creative', 'curious', 'artistic'], 'low': ['practical', 'routine']},
            'conscientiousness': {'high': ['organized', 'disciplined', 'responsible'], 'low': ['spontaneous', 'flexible']},
            'extraversion': {'high': ['outgoing', 'social', 'energetic'], 'low': ['reserved', 'quiet']},
            'agreeableness': {'high': ['cooperative', 'trusting', 'helpful'], 'low': ['skeptical', 'competitive']},
            'neuroticism': {'high': ['anxious', 'worried', 'sensitive'], 'low': ['calm', 'stable']}
        }
        return keywords.get(trait, {}).get(level, [])