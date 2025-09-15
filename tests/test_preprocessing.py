import unittest
from src.preprocessing.text_cleaner import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor(remove_stopwords=True)

    def test_clean_text(self):
        raw_text = "This is a test!  Let's see how it cleans up."
        cleaned_text = self.preprocessor.clean_text(raw_text)
        self.assertEqual(cleaned_text, 'This is a test! Let\'s see how it cleans up.')

    def test_tokenize_text(self):
        raw_text = "Tokenize this sentence. And this one too!"
        sentences, words = self.preprocessor.tokenize_text(raw_text)
        self.assertEqual(len(sentences), 2)
        self.assertIn('tokenize', words)

    def test_lemmatize_tokens(self):
        tokens = ['running', 'ran', 'better', 'best']
        lemmatized = self.preprocessor.lemmatize_tokens(tokens)
        self.assertIn('run', lemmatized)
        self.assertIn('good', lemmatized)

    def test_preprocess(self):
        raw_text = "I love exploring new ideas and thinking creatively."
        result = self.preprocessor.preprocess(raw_text)
        self.assertIn('original_text', result)
        self.assertIn('statistics', result)

if __name__ == '__main__':
    unittest.main()