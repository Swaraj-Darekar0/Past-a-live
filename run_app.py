"""
Main application runner for Past Life Predictor

Usage:
    python run_app.py

Make sure to:
1. Install all requirements: pip install -r requirements.txt
2. Download NLTK data: python -c "import nltk; nltk.download('all')"
3. Install spaCy model: python -m spacy download en_core_web_sm
4. Setup Ollama with LLAMA model: ollama pull llama2:7b
5. Set your Gemini API key in .env file
"""

import streamlit as st
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the app
from app import PastLifePredictorApp

def main():
    """Main application entry point"""
    app = PastLifePredictorApp()
    app.render_main_interface()

if __name__ == "__main__":
    main()
