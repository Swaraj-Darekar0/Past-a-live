from typing import Dict, List, Any, Optional
import yaml
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

# Import all modules
from preprocessing.text_cleaner import TextPreprocessor
from preprocessing.feature_extractor import FeatureExtractor
from analysis.personality_analyzer import PersonalityAnalyzer
from analysis.context_enhancer import ContextEnhancer
from generation.story_generator import StoryGenerator

class PastLifePredictorPipeline:
    """
    Main orchestrator for the complete past life prediction pipeline
    
    Academic Purpose: Demonstrates end-to-end NLP system integration
    and pipeline orchestration for complex multi-stage processing
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the complete pipeline with configuration"""
        load_dotenv() # Load environment variables from .env
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.text_preprocessor = TextPreprocessor(remove_stopwords=self.config['preprocessing']['remove_stopwords'])
        self.feature_extractor = FeatureExtractor(embedding_model=self.config['models']['embeddings']['model'])
        self.personality_analyzer = PersonalityAnalyzer()
        self.context_enhancer = ContextEnhancer()
        # Load API key from environment variables for security
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.story_generator = StoryGenerator(api_key=gemini_api_key)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_logging(self):
        """Setup logging for the pipeline"""
        logging.basicConfig(level=self.config['logging']['level'], format=self.config['logging']['format'])

    def predict_past_life(self, user_responses: List[str], 
                         narrative_style: str = "immersive_narrative") -> Dict[str, Any]:
        """
        Complete pipeline: from user input to past life story
        
        Args:
            user_responses: List of user answers to personality questions
            narrative_style: Style of story to generate
            
        Returns:
            Complete past life prediction with story and analysis
        """
        start_time = datetime.now()
        self.logger.info("Starting past life prediction pipeline")
        
        try:
            # Step 1: Text Preprocessing - Combine all responses for holistic analysis
            self.logger.info("Step 1: Text preprocessing")
            combined_text = " ".join(user_responses)
            preprocessed_data = self.text_preprocessor.preprocess(combined_text)
            
            # Step 2: Feature Extraction
            self.logger.info("Step 2: NLP feature extraction")
            comprehensive_features = self.feature_extractor.extract_comprehensive_features(
                preprocessed_data
            )
            
            # Step 3: Personality Analysis with Local LLM
            self.logger.info("Step 3: Personality analysis with LLAMA")
            personality_analysis = self.personality_analyzer.analyze_personality_from_features(
                comprehensive_features
            )
            
            # Step 4: Context Enhancement
            self.logger.info("Step 4: Cultural context enhancement")
            enhanced_analysis = self.context_enhancer.enhance_with_cultural_context(
                personality_analysis
            )
            
            # Generate a specialized prompt for the local LLM to predict past life details
            context_prompt = self.context_enhancer.create_llama_context_prompt(enhanced_analysis)
            llama_prediction = self.personality_analyzer._call_llama_api(context_prompt)
            
            # Step 5: Story Generation with Cloud LLM
            self.logger.info("Step 5: Story generation with Gemini")
            final_story = self.story_generator.generate_past_life_story(
                llama_prediction, enhanced_analysis, narrative_style
            )
            
            # Step 6: Package Final Result
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                'past_life_story': final_story,
                'personality_analysis': personality_analysis,
                'processing_metadata': {
                    'pipeline_version': self.config.get('app', {}).get('version', '1.0.0'),
                    'processing_time_seconds': processing_time,
                    'input_text_length': len(combined_text),
                    'narrative_style': narrative_style
                }
            }
            
            self.logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response if pipeline fails"""
        return {
            "error": True,
            "message": f"An error occurred during prediction: {error_message}"
        }