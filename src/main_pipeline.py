from typing import Dict, List, Any, Optional
import yaml
import logging
from datetime import datetime

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
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.text_preprocessor = TextPreprocessor(remove_stopwords=self.config['preprocessing']['remove_stopwords'])
        self.feature_extractor = FeatureExtractor(embedding_model=self.config['models']['embeddings']['model'])
        self.personality_analyzer = PersonalityAnalyzer()
        self.context_enhancer = ContextEnhancer()
        self.story_generator = StoryGenerator(api_key=self.config['api_keys']['gemini'])

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_logging(self):
        """Setup logging for the pipeline"""
        logging.basicConfig(level=self.config['logging']['level'], format=self.config['logging']['format'])

    def predict_past_life(self, user_responses: List[str], 
                         narrative_style: str = "immersive_narrative") -> Dict[str, Any]:
        """Main method to predict past life based on user responses"""
        self.logger.info("Starting past life prediction pipeline.")
        
        # Step 1: Preprocess user responses
        preprocessed_data = [self.text_preprocessor.preprocess(response) for response in user_responses]
        
        # Step 2: Extract features
        comprehensive_features = [self.feature_extractor.extract_comprehensive_features(data) for data in preprocessed_data]
        
        # Step 3: Analyze personality
        personality_analysis = [self.personality_analyzer.analyze_personality_from_features(features) for features in comprehensive_features]
        
        # Step 4: Enhance with cultural context
        enhanced_analysis = [self.context_enhancer.enhance_with_cultural_context(analysis) for analysis in personality_analysis]
        
        # Step 5: Generate past life story
        stories = [self.story_generator.generate_past_life_story(analysis) for analysis in enhanced_analysis]
        
        self.logger.info("Past life prediction completed.")
        return stories

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response if pipeline fails"""
        return {"error": error_message}