import unittest
from src.analysis.personality_analyzer import PersonalityAnalyzer
from src.analysis.context_enhancer import ContextEnhancer

class TestPersonalityAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = PersonalityAnalyzer()
    
    def test_analyze_personality_from_features(self):
        # Mock comprehensive features for testing
        comprehensive_features = {
            'embeddings': {
                'trait_similarities': {
                    'openness': {'trait_score': 0.8, 'confidence': 0.9},
                    'conscientiousness': {'trait_score': 0.6, 'confidence': 0.85},
                    'extraversion': {'trait_score': 0.7, 'confidence': 0.8},
                    'agreeableness': {'trait_score': 0.9, 'confidence': 0.95},
                    'neuroticism': {'trait_score': 0.4, 'confidence': 0.7}
                }
            },
            'linguistic_patterns': {
                'trait_indicators': {
                    'openness': {'trait_lean': 'high', 'confidence': 0.9},
                    'conscientiousness': {'trait_lean': 'medium', 'confidence': 0.85},
                    'extraversion': {'trait_lean': 'high', 'confidence': 0.8},
                    'agreeableness': {'trait_lean': 'high', 'confidence': 0.95},
                    'neuroticism': {'trait_lean': 'low', 'confidence': 0.7}
                }
            },
            'text_statistics': {
                'avg_sentence_length': 15.0,
                'total_words': 100
            }
        }
        
        analysis_result = self.analyzer.analyze_personality_from_features(comprehensive_features)
        
        self.assertIn('personality_scores', analysis_result)
        self.assertIn('overall_summary', analysis_result)

class TestContextEnhancer(unittest.TestCase):
    
    def setUp(self):
        self.enhancer = ContextEnhancer()
    
    def test_enhance_with_cultural_context(self):
        # Mock personality analysis for testing
        personality_analysis = {
            'personality_scores': {
                'openness': {'final_score': 8.0},
                'conscientiousness': {'final_score': 6.0},
                'extraversion': {'final_score': 7.0},
                'agreeableness': {'final_score': 9.0},
                'neuroticism': {'final_score': 4.0}
            },
            'personality_summary': {
                'dominant_traits': ['openness', 'agreeableness']
            }
        }
        
        enhanced_analysis = self.enhancer.enhance_with_cultural_context(personality_analysis)
        
        self.assertIn('cultural_context', enhanced_analysis)
        self.assertIn('suitable_periods', enhanced_analysis['cultural_context'])

if __name__ == '__main__':
    unittest.main()