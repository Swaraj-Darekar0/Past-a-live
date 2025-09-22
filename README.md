# Past Life Predictor with Advanced NLP Pipeline

## Project Overview
The Past Life Predictor is an advanced NLP application that analyzes personality traits from user text responses and generates personalized historical narratives. This project demonstrates a sophisticated multi-stage pipeline combining traditional NLP techniques with modern language models.

## Academic Significance
This project showcases:
- **Traditional NLP Integration**: Tokenization, lemmatization, semantic embeddings, and n-gram analysis
- **Hybrid AI Architecture**: Strategic use of local LLAMA models for analysis and cloud-based Gemini for generation  
- **Multi-Stage Processing**: Six-step pipeline from raw text to personalized narratives
- **Context-Aware Generation**: Cultural and historical context integration
- **Big Five Personality Assessment**: Computational psychology through text analysis

## Technical Architecture

### Core Components
1. **Text Preprocessing** - Tokenization, lemmatization, and cleaning using NLTK/spaCy
2. **Feature Extraction** - Semantic embeddings and n-gram analysis with sentence-transformers
3. **Personality Analysis** - Big Five trait prediction using local LLAMA model
4. **Context Enhancement** - Cultural/historical period matching algorithms
5. **Story Generation** - Creative narrative creation with Google Gemini
6. **Pipeline Orchestration** - End-to-end system coordination and error handling

### Technology Stack
- **Python 3.9+** - Core programming language
- **NLTK & spaCy** - Traditional NLP preprocessing
- **sentence-transformers** - Semantic embeddings (all-MiniLM-L6-v2)
- **LLAMA 2** - Local personality analysis via Ollama
- **Google Gemini 1.5 Flash** - Creative story generation
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning utilities

## Project Structure
```
past_life_predictor/
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py          # Step 1: Tokenization & lemmatization
│   │   └── feature_extractor.py     # Step 2: Embeddings & n-grams
│   ├── analysis/
│   │   ├── personality_analyzer.py  # Step 3: LLAMA personality analysis
│   │   └── context_enhancer.py      # Step 4: Cultural context matching
│   ├── generation/
│   │   └── story_generator.py       # Step 5: Gemini story generation
│   └── main_pipeline.py             # Step 6: Pipeline orchestration
├── data/
│   ├── personality_traits.json      # Big Five trait patterns
│   ├── cultural_contexts.json       # Historical period database
│   └── question_templates.json      # Personality assessment questions
├── tests/
│   └── test_*.py                    # Unit tests for all components
├── app.py                           # Streamlit web interface
├── run_app.py                       # Application launcher
├── config.yaml                      # Configuration settings
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
└── README.md                        # This file
```

## System Requirements

### Hardware Requirements
- **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and data
- **Internet**: Stable connection for Gemini API calls

### Software Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.9 or higher
- **Git**: For repository management

## Complete Installation Guide

### Step 1: Environment Setup

#### Create Project Directory
```bash
mkdir past_life_predictor
cd past_life_predictor
```

#### Setup Python Virtual Environment
```bash
# Create virtual environment
python -m venv nlp
# or
conda create -n nlp python=3.9

# Activate environment
# Windows:
nlp\Scripts\activate
# macOS/Linux:
source nlp/bin/activate
# Conda:
conda activate nlp
```

### Step 2: Install Dependencies

#### Option A: Using pip (Standard)
```bash
pip install -r requirements.txt
```

#### Option B: Using conda (Recommended for Windows)
```bash
# Install compiled packages via conda (avoids compilation issues)
conda install -c conda-forge spacy nltk scikit-learn numpy pandas pytorch sentence-transformers

# Install remaining packages via pip
pip install streamlit google-generativeai pyyaml python-dotenv requests transformers
```

#### Download NLP Models
```bash
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 3: Install and Configure Ollama

#### Install Ollama
1. **Download Ollama**: Visit [https://ollama.ai/download](https://ollama.ai/download)
2. **Install** following platform-specific instructions
3. **Verify installation**:
   ```bash
   ollama --version
   ```

#### Pull LLAMA Model
```bash
# For systems with 8GB+ RAM:
ollama pull llama2:7b

# For systems with 4-8GB RAM (lighter alternative):
ollama pull llama2:3b

# For very limited resources:
ollama pull tinyllama
```

#### Start Ollama Service
```bash
# Start Ollama server (keep this terminal open)
ollama serve

# Test the model in a new terminal:
ollama run llama2:7b
# Type "Hello" and press Enter to test
# Type "/bye" to exit
```

### Step 4: Configure API Keys

#### Get Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the generated key

#### Setup Environment Variables
```bash
# Copy the template
cp .env.example .env

# Edit .env file with your API key:
```

**.env file contents:**
```
# Google Gemini API Configuration
GEMINI_API_KEY=your-actual-api-key-here

# Ollama Configuration
OLLAMA_ENDPOINT=http://localhost:11434/api/generate
OLLAMA_MODEL=llama2:7b

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
```

### Step 5: Create Required Data Files

#### Create `data/personality_traits.json`:
```json
{
  "trait_patterns": {
    "openness": {
      "high_indicators": ["creative", "imaginative", "curious", "artistic", "innovative", "abstract", "philosophical"],
      "low_indicators": ["practical", "conventional", "traditional", "concrete", "routine", "simple"],
      "linguistic_patterns": {
        "high": ["I wonder", "what if", "imagine", "create", "explore", "new ideas"],
        "low": ["always do", "tried and true", "stick to", "reliable", "proven methods"]
      }
    },
    "conscientiousness": {
      "high_indicators": ["organized", "disciplined", "responsible", "punctual", "thorough", "systematic"],
      "low_indicators": ["spontaneous", "flexible", "casual", "relaxed", "impulsive", "disorganized"],
      "linguistic_patterns": {
        "high": ["plan ahead", "make sure", "always", "schedule", "organize", "systematic"],
        "low": ["go with flow", "wing it", "whatever", "spontaneous", "flexible"]
      }
    },
    "extraversion": {
      "high_indicators": ["outgoing", "energetic", "talkative", "assertive", "social", "enthusiastic"],
      "low_indicators": ["reserved", "quiet", "independent", "thoughtful", "introspective"],
      "linguistic_patterns": {
        "high": ["love meeting", "enjoy parties", "energizes me", "social", "outgoing"],
        "low": ["prefer solitude", "quiet time", "think before", "independent", "reflective"]
      }
    },
    "agreeableness": {
      "high_indicators": ["trusting", "helpful", "forgiving", "compassionate", "cooperative"],
      "low_indicators": ["skeptical", "competitive", "challenging", "critical", "suspicious"],
      "linguistic_patterns": {
        "high": ["help others", "work together", "understand", "support", "cooperate"],
        "low": ["question", "compete", "challenge", "critical", "skeptical"]
      }
    },
    "neuroticism": {
      "high_indicators": ["anxious", "moody", "worrying", "sensitive", "stressed", "emotional"],
      "low_indicators": ["calm", "relaxed", "secure", "stable", "confident", "resilient"],
      "linguistic_patterns": {
        "high": ["worry about", "stress", "anxious", "concern", "nervous", "overwhelmed"],
        "low": ["stay calm", "relaxed", "handle pressure", "confident", "stable"]
      }
    }
  }
}
```

#### Create `data/cultural_contexts.json`:
```json
{
  "cultural_contexts": {
    "ancient_greece": {
      "time_period": "800-146 BCE",
      "personality_fit": {
        "high_openness": ["philosopher", "playwright", "inventor", "explorer"],
        "high_extraversion": ["politician", "orator", "merchant", "athlete"],
        "high_conscientiousness": ["architect", "mathematician", "physician", "judge"],
        "high_agreeableness": ["teacher", "healer", "priest", "mediator"],
        "low_neuroticism": ["warrior", "leader", "adventurer", "athlete"]
      },
      "cultural_values": ["wisdom", "honor", "civic_duty", "artistic_beauty", "philosophical_inquiry"],
      "historical_context": "Golden age of philosophy, democracy, and artistic achievement",
      "notable_figures": ["Socrates", "Plato", "Aristotle", "Pericles"]
    },
    "renaissance_italy": {
      "time_period": "1300-1600 CE",
      "personality_fit": {
        "high_openness": ["artist", "inventor", "explorer", "humanist_scholar"],
        "high_extraversion": ["patron", "diplomat", "merchant_prince", "courtier"],
        "high_conscientiousness": ["banker", "engineer", "architect", "scholar"],
        "high_agreeableness": ["teacher", "physician", "diplomat"],
        "low_neuroticism": ["explorer", "entrepreneur", "leader"]
      },
      "cultural_values": ["innovation", "artistic_beauty", "humanism", "learning", "individual_achievement"],
      "historical_context": "Rebirth of classical learning, artistic innovation, and scientific discovery",
      "notable_figures": ["Leonardo da Vinci", "Michelangelo", "Galileo", "Medici family"]
    },
    "medieval_europe": {
      "time_period": "500-1500 CE",
      "personality_fit": {
        "high_conscientiousness": ["monk", "craftsman", "knight", "scribe"],
        "high_agreeableness": ["healer", "caregiver", "village_elder", "nun"],
        "high_neuroticism": ["hermit", "mystic", "scholar"],
        "high_openness": ["illuminator", "troubadour", "alchemist"],
        "high_extraversion": ["lord", "merchant", "storyteller"]
      },
      "cultural_values": ["faith", "loyalty", "honor", "community", "tradition"],
      "historical_context": "Age of feudalism, religious devotion, and emerging craftsmanship",
      "notable_figures": ["Charlemagne", "Thomas Aquinas", "Joan of Arc"]
    }
  }
}
```

#### Create `config.yaml`:
```yaml
app:
  name: "Past Life Predictor"
  version: "1.0.0"
  debug: true

models:
  local_llm:
    name: "llama2:7b"
    endpoint: "http://localhost:11434/api/generate"
    temperature: 0.3
    max_tokens: 512
  
  embeddings:
    model: "all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
  
  cloud_llm:
    provider: "gemini"
    model: "gemini-1.5-flash"
    temperature: 0.7
    max_tokens: 2048

preprocessing:
  min_text_length: 10
  max_text_length: 2000
  remove_stopwords: true
  use_lemmatization: true

analysis:
  personality_traits: ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
  confidence_threshold: 0.6

api_keys:
  gemini: "${GEMINI_API_KEY}"
```

### Step 6: Verification and Testing

#### Test Individual Components
```bash
# Test text preprocessing
python -c "from src.preprocessing.text_cleaner import TextPreprocessor; print('✅ Text preprocessing works')"

# Test feature extraction
python -c "from src.preprocessing.feature_extractor import FeatureExtractor; print('✅ Feature extraction works')"

# Test Ollama connection
curl http://localhost:11434/api/tags

# Test full pipeline (requires API key)
python -c "from src.main_pipeline import PastLifePredictorPipeline; pipeline = PastLifePredictorPipeline(); print('✅ Pipeline initialized')"
```

#### Run Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_preprocessing.py -v
```

## Running the Application

### Method 1: Direct Launch
```bash
streamlit run app.py
```

### Method 2: Using Runner Script
```bash
python run_app.py
```

### Method 3: Development Mode
```bash
# With debugging enabled
DEBUG=True streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### Basic Workflow
1. **Start Services**: Ensure Ollama is running (`ollama serve`)
2. **Launch App**: Run `streamlit run app.py`
3. **Answer Questions**: Provide thoughtful 2-3 sentence responses
4. **Choose Style**: Select your preferred narrative style
5. **Generate Story**: Click "Predict My Past Life"
6. **Review Results**: Explore personality analysis and generated story

### Performance Expectations
- **Processing Time**: 30-60 seconds total
- **LLAMA Analysis**: 10-20 seconds (CPU dependent)
- **Gemini Generation**: 5-15 seconds
- **Story Length**: 1500-2000 words

## Troubleshooting

### Common Issues

#### Ollama Connection Error
```bash
# Ensure Ollama is running
ollama serve

# Check if model is available
ollama list

# Test model
ollama run llama2:7b
```

#### Memory Issues
```bash
# Use smaller model
ollama pull llama2:3b

# Update config.yaml to use smaller model
```

#### API Key Errors
```bash
# Verify environment variables
echo $GEMINI_API_KEY

# Check .env file exists and has correct key
cat .env
```

#### Import Errors
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check virtual environment is active
which python  # Should show path to venv
```

### Getting Help

If you encounter issues:
1. Check the **Troubleshooting** section above
2. Review logs in the terminal/console
3. Test individual components using verification commands
4. Check GitHub Issues for similar problems

## Development

### Project Structure Explanation
- `src/preprocessing/` - Traditional NLP techniques (tokenization, embeddings)
- `src/analysis/` - Personality assessment and cultural context
- `src/generation/` - Story creation using Gemini
- `data/` - Knowledge bases for personality and cultural patterns
- `tests/` - Unit tests for all components

### Adding New Features
1. Create new modules in appropriate `src/` subdirectories
2. Add configuration options to `config.yaml`
3. Write unit tests in `tests/`
4. Update this README with new installation requirements

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for all functions
- Include docstrings with academic purpose explanations
- Add logging for debugging and monitoring

## Academic Applications

This project demonstrates:
- **NLP Pipeline Design**: Multi-stage processing architecture
- **Personality Psychology**: Computational Big Five assessment
- **Cultural Computing**: Historical context integration
- **Hybrid AI Systems**: Traditional ML + Modern LLMs
- **Software Engineering**: Production-ready system design

Perfect for:
- NLP coursework and research
- Psychology and computational linguistics
- AI system architecture learning
- Portfolio projects for data science/ML roles

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

## Acknowledgments

- **spaCy** and **NLTK** teams for NLP tools
- **Hugging Face** for sentence transformers
- **Ollama** team for local LLM infrastructure  
- **Google** for Gemini API access
- **Streamlit** for the web framework
