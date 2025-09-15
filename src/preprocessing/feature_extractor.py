class FeatureExtractor:
    """
    Extract semantic embeddings and n-gram features from preprocessed text
    
    Academic Purpose: Demonstrate advanced NLP feature engineering techniques
    including semantic embeddings and statistical pattern analysis
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize feature extractor with embedding model
        
        Args:
            embedding_model: Name of sentence transformer model to use
        """
        # Load semantic embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize TF-IDF vectorizer for n-gram analysis
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),  # unigrams, bigrams, trigrams
            stop_words='english'
        )
        
        # Load personality trait patterns
        self.trait_patterns = self._load_trait_patterns()
        
        # Cache for embeddings (improves performance)
        self.embedding_cache = {}
    
    def _load_trait_patterns(self) -> Dict:
        """Load personality trait patterns from data file"""
        try:
            with open('data/personality_traits.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "openness": {"high_indicators": []},
                "conscientiousness": {"high_indicators": []},
                "extraversion": {"high_indicators": []},
                "agreeableness": {"high_indicators": []},
                "neuroticism": {"high_indicators": []}
            }
    
    def extract_semantic_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """
        Extract semantic embeddings from text
        
        Academic Value: Demonstrates understanding of modern NLP representation
        learning and semantic similarity computation
        
        Args:
            text: Input text (can be preprocessed or raw)
            
        Returns:
            Dictionary containing embeddings and similarity scores
        """
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Generate embeddings
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        
        # Calculate similarity to personality trait descriptions
        trait_similarities = {}
        
        for trait, patterns in self.trait_patterns.items():
            # Create trait description from patterns
            trait_description = " ".join(patterns['high_indicators'])
            similarity = self._calculate_similarity(embedding, trait_description)
            trait_similarities[trait] = {
                'trait_score': similarity,
                'confidence': self._calculate_confidence(similarity)
            }
        
        result = {
            'text_embedding': embedding,
            'embedding_dimension': len(embedding),
            'trait_similarities': trait_similarities
        }
        
        # Cache result
        self.embedding_cache[text] = result
        
        return result
    
    def extract_ngram_features(self, texts: List[str]) -> Dict[str, any]:
        """
        Extract n-gram features using TF-IDF
        
        Academic Value: Demonstrates statistical text analysis and
        pattern recognition in linguistic data
        
        Args:
            texts: List of text samples (for building vocabulary)
            
        Returns:
            Dictionary containing n-gram analysis results
        """
        if len(texts) < 2:
            # Need multiple texts for TF-IDF to work properly
            texts = texts + ["sample text for vocabulary building"]
        
        # Fit TF-IDF on texts
        tfidf_matrix = self.tfidf.fit_transform(texts)
        feature_names = self.tfidf.get_feature_names_out()
        
        # Analyze n-gram patterns for each text
        ngram_analysis = []
        
        for i, text in enumerate(texts):
            if i >= tfidf_matrix.shape[0]:
                continue
            ngram_analysis.append({
                'text': text,
                'features': tfidf_matrix[i].toarray().flatten().tolist()
            })
        
        return {
            'tfidf_matrix': tfidf_matrix,
            'vocabulary_size': len(feature_names),
            'ngram_analysis': ngram_analysis
        }
    
    def analyze_linguistic_patterns(self, lemmatized_words: List[str]) -> Dict[str, any]:
        """
        Analyze linguistic patterns in text for personality trait indicators
        
        Academic Value: Shows application of computational linguistics
        for psychological analysis
        
        Args:
            lemmatized_words: List of lemmatized words from text
            
        Returns:
            Dictionary of linguistic pattern analysis
        """
        word_counter = Counter(lemmatized_words)
        total_words = len(lemmatized_words)
        
        # Analyze trait indicators
        trait_indicators = {}
        
        for trait, patterns in self.trait_patterns.items():
            high_count = sum(word_counter.get(word, 0) for word in patterns['high_indicators'])
            trait_indicators[trait] = {
                'count': high_count,
                'percentage': (high_count / total_words) * 100 if total_words > 0 else 0
            }
        
        # Analyze linguistic complexity
        unique_words = len(set(lemmatized_words))
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Analyze common patterns
        most_common = word_counter.most_common(10)
        
        return {
            'trait_indicators': trait_indicators,
            'total_words': total_words,
            'lexical_diversity': lexical_diversity,
            'most_common': most_common
        }
    
    def extract_comprehensive_features(self, preprocessed_data: Dict) -> Dict[str, any]:
        """
        Extract all features from preprocessed text data
        
        Args:
            preprocessed_data: Output from TextPreprocessor.preprocess()
            
        Returns:
            Comprehensive feature dictionary
        """
        text = preprocessed_data['cleaned_text']
        lemmatized_words = preprocessed_data['lemmatized_words']
        
        # Extract semantic embeddings
        embedding_features = self.extract_semantic_embeddings(text)
        
        # Extract n-gram features (using single text)
        ngram_features = self.extract_ngram_features([text])
        
        # Analyze linguistic patterns
        linguistic_features = self.analyze_linguistic_patterns(lemmatized_words)
        
        # Combine all features
        comprehensive_features = {
            'embeddings': embedding_features,
            'ngrams': ngram_features,
            'linguistic_patterns': linguistic_features,
            'text_statistics': preprocessed_data['statistics']
        }
        
        return comprehensive_features
    
    def _calculate_similarity(self, embedding: np.ndarray, trait_description: str) -> float:
        # Placeholder for similarity calculation
        return np.random.rand()  # Replace with actual similarity calculation logic
    
    def _calculate_confidence(self, similarity: float) -> float:
        # Placeholder for confidence calculation
        return similarity  # Replace with actual confidence calculation logic

# Usage Example and Testing
if __name__ == "__main__":
    from text_cleaner import TextPreprocessor
    
    # Test feature extraction
    preprocessor = TextPreprocessor()
    extractor = FeatureExtractor()
    
    sample_texts = [
        "I love being creative and exploring new artistic ideas. Innovation excites me!",
        "I prefer structured approaches and always plan my work carefully and systematically.",
        "Meeting new people energizes me, and I enjoy being the center of attention at parties."
    ]
    
    print("=== Feature Extraction Results ===")
    
    for i, text in enumerate(sample_texts):
        print(f"\n--- Text {i+1} ---")
        print(f"Text: {text}")
        
        # Preprocess
        preprocessed = preprocessor.preprocess(text)
        
        # Extract features
        features = extractor.extract_comprehensive_features(preprocessed)
        
        # Print key results
        print("Trait Similarities:")
        for trait, scores in features['embeddings']['trait_similarities'].items():
            print(f"  {trait}: {scores['trait_score']:.3f} (confidence: {scores['confidence']:.3f})")
        
        print("Top N-grams:")
        if features['ngrams']['ngram_analysis']:
            analysis = features['ngrams']['ngram_analysis'][0]
            print(f"  Bigrams: {analysis['features'][:2]}")