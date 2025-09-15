import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple

class TextPreprocessor:
    """
    Handles basic text preprocessing: tokenization and lemmatization
    
    Academic Purpose: Demonstrate fundamental NLP preprocessing techniques
    that are essential for any text analysis pipeline.
    """
    
    def __init__(self, remove_stopwords: bool = False):
        """
        Initialize preprocessor with configuration options
        
        Args:
            remove_stopwords: Whether to remove common English stopwords
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.remove_stopwords = remove_stopwords
        
        # Ensure NLTK data is available
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data if not present"""
        required_data = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords']
        for data in required_data:
            try:
                nltk.download(data)
            except Exception as e:
                print(f"Error downloading NLTK data: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        # Normalize quotes and apostrophes
        text = re.sub(r'[""''`]', '"', text)
        text = re.sub(r'[''`]', "'", text)
        
        return text
    
    def tokenize_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Tokenize text into sentences and words
        
        Academic Value: Demonstrates understanding of tokenization,
        which is foundational for all NLP tasks
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Tuple of (sentences, words)
        """
        # Clean text first
        clean_text = self.clean_text(text)
        
        # Sentence tokenization
        sentences = sent_tokenize(clean_text)
        
        # Word tokenization
        words = word_tokenize(clean_text.lower())
        
        # Remove punctuation-only tokens
        words = [word for word in words if word.isalnum()]
        
        return sentences, words
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their root forms
        
        Academic Value: Shows understanding of morphological analysis
        and text normalization
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        lemmatized = []
        for token in tokens:
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            lemmatized_token = self.lemmatizer.lemmatize(token)
            lemmatized.append(lemmatized_token)
        
        return lemmatized
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert treebank POS tags to WordNet format"""
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun
    
    def preprocess(self, text: str) -> Dict[str, any]:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Raw input text
            
        Returns:
            Dictionary containing processed text data
        """
        # Tokenization
        sentences, words = self.tokenize_text(text)
        
        # Lemmatization
        lemmatized_words = self.lemmatize_tokens(words)
        
        # Basic statistics
        stats = {
            'original_length': len(text),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
        
        return {
            'original_text': text,
            'sentences': sentences,
            'words': words,
            'lemmatized_words': lemmatized_words,
            'statistics': stats
        }

# Usage Example and Testing
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True)
    
    sample_text = """
    I really love exploring new ideas and thinking creatively about complex problems. 
    When facing challenges, I usually try to find innovative solutions rather than 
    following traditional approaches. I'm always curious about different perspectives!
    """
    
    result = preprocessor.preprocess(sample_text)
    
    print("=== Text Preprocessing Results ===")
    print(f"Original: {result['original_text'][:100]}...")
    print(f"Sentences: {len(result['sentences'])}")
    print(f"Words: {result['words'][:10]}...")
    print(f"Lemmatized: {result['lemmatized_words'][:10]}...")
    print(f"Statistics: {result['statistics']}")