# Past Life Predictor with NLP Pipeline

## Project Overview
The Past Life Predictor is an innovative application that utilizes Natural Language Processing (NLP) techniques to analyze user input and generate personalized historical narratives based on inferred personality traits. The project demonstrates a multi-stage NLP pipeline that includes text preprocessing, personality analysis, context enhancement, and story generation.

## Features
- **Text Preprocessing**: Cleans and prepares user input for analysis.
- **Personality Analysis**: Utilizes local and cloud-based language models to assess personality traits based on user responses.
- **Cultural Context Enhancement**: Integrates cultural and historical contexts to enrich the generated narratives.
- **Story Generation**: Produces engaging past life stories that reflect the user's personality and historical context.

## Technical Architecture
The application is structured into several components:
1. **Preprocessing**: Handles text cleaning and feature extraction.
2. **Analysis**: Analyzes personality traits and enhances context.
3. **Generation**: Generates narratives based on the analysis.
4. **Utilities**: Contains configuration and helper functions.
5. **Main Pipeline**: Orchestrates the entire process from input to output.

## Project Structure
```
past_life_predictor/
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py
│   │   └── feature_extractor.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── personality_analyzer.py
│   │   └── context_enhancer.py
│   ├── generation/
│   │   ├── __init__.py
│   │   └── story_generator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── helpers.py
│   └── main_pipeline.py
├── data/
│   ├── personality_traits.json
│   ├── cultural_contexts.json
│   └── question_templates.json
├── models/
│   └── embeddings/
├── tests/
│   ├── test_preprocessing.py
│   ├── test_analysis.py
│   └── test_generation.py
├── app.py
├── run_app.py
├── requirements.txt
├── config.yaml
├── .env.example
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone <your-repo-url>
   cd past_life_predictor
   ```
2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Download necessary NLTK and spaCy models:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   python -m spacy download en_core_web_sm
   ```

## Usage
To run the application, execute:
```
streamlit run app.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.