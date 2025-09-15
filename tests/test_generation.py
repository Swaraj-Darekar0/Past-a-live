import unittest
from src.generation.story_generator import StoryGenerator

class TestStoryGenerator(unittest.TestCase):

    def setUp(self):
        self.api_key = "your-gemini-api-key-here"  # Replace with a valid API key for testing
        self.story_generator = StoryGenerator(self.api_key)

    def test_generate_past_life_story(self):
        llama_prediction = "Sample prediction from LLAMA model."
        enhanced_analysis = {
            'personality_scores': {
                'openness': {'final_score': 8, 'level': 'High', 'explanation': 'Highly open to new experiences.'},
                'conscientiousness': {'final_score': 6, 'level': 'Medium', 'explanation': 'Generally organized.'},
                'extraversion': {'final_score': 7, 'level': 'High', 'explanation': 'Very sociable and energetic.'},
                'agreeableness': {'final_score': 5, 'level': 'Medium', 'explanation': 'Somewhat cooperative.'},
                'neuroticism': {'final_score': 4, 'level': 'Low', 'explanation': 'Rarely experiences negative emotions.'}
            },
            'cultural_context': {
                'period_name': 'Renaissance Italy',
                'historical_context': 'A time of great cultural change and artistic achievement.',
                'cultural_values': ['Humanism', 'Individualism', 'Secularism']
            }
        }

        story = self.story_generator.generate_past_life_story(llama_prediction, enhanced_analysis)
        self.assertIsInstance(story, str)
        self.assertGreater(len(story), 0)

    def test_parse_llama_prediction(self):
        llama_output = """
        NAME: Leonardo da Vinci
        PERIOD: 1452-1519, Renaissance Italy
        OCCUPATION: Artist and Inventor
        PERSONALITY_MANIFESTATION: Creativity and curiosity were evident in his works.
        KEY_LIFE_EVENTS: Created masterpieces like the Mona Lisa and The Last Supper.
        CULTURAL_INTEGRATION: Influenced the art and science of his time.
        """
        parsed_data = self.story_generator._parse_llama_prediction(llama_output)
        self.assertIn('name', parsed_data)
        self.assertIn('period', parsed_data)
        self.assertIn('occupation', parsed_data)

if __name__ == '__main__':
    unittest.main()