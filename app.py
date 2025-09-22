import streamlit as st
import json
import time
import sys
import os
from datetime import datetime

# Add src directory to path to allow for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main_pipeline import PastLifePredictorPipeline

# Configure Streamlit page
st.set_page_config(
    page_title="Past Life Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .personality-score {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .story-container {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class PastLifePredictorApp:
    """Streamlit web application for Past Life Predictor"""
    
    def __init__(self):
        """Initialize the Streamlit app"""
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = None
            st.session_state.prediction_result = None
            st.session_state.user_responses = []

    def setup_pipeline(self):
        """Initialize the prediction pipeline"""
        try:
            if st.session_state.pipeline is None:
                with st.spinner("Initializing NLP pipeline..."):
                    st.session_state.pipeline = PastLifePredictorPipeline()
            return True
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {str(e)}")
            return False

    def render_main_interface(self):
        """Render the main application interface"""
        # Header
        st.markdown('<h1 class="main-header">🔮 Past Life Predictor</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        **Discover your past life through advanced NLP and AI analysis!**
        
        This application uses cutting-edge natural language processing to analyze your personality 
        and predict what your past life might have been like. The system combines:
        - Traditional NLP preprocessing (tokenization, lemmatization)
        - Semantic embedding analysis
        - Big Five personality prediction with LLAMA
        - Cultural context enhancement
        - Creative story generation with Gemini AI
        """)
        
        # Sidebar for navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.selectbox("Choose a page:", [
                "📝 Personality Assessment", 
                "📊 Technical Details", 
                "ℹ️ About the Project"
            ])
        
        if page == "📝 Personality Assessment":
            self.render_assessment_page()
        elif page == "📊 Technical Details":
            self.render_technical_page()
        else:
            self.render_about_page()
    
    def render_assessment_page(self):
        """Render the personality assessment interface"""
        st.header("Personality Assessment")
        
        # Personality questions
        questions = [
            "How do you typically handle unexpected challenges in your life?",
            "Describe your ideal social gathering or way to spend time with others.",
            "What type of activities or pursuits make you feel most fulfilled and energized?",
            "How do you approach making important decisions in your life?",
            "What are your biggest concerns or fears, and how do they affect your daily life?"
        ]
        
        # Collect user responses
        user_responses = []
        st.subheader("Please answer these questions thoughtfully (2-3 sentences each):")
        
        for i, question in enumerate(questions, 1):
            response = st.text_area(
                f"Question {i}: {question}",
                key=f"question_{i}",
                help="Please provide a thoughtful response of 2-3 sentences.",
                height=100
            )
            user_responses.append(response)
        
        # Narrative style selection
        st.subheader("Choose your story style:")
        narrative_style = st.selectbox(
            "Select the type of past life story you'd prefer:",
            [
                ("immersive_narrative", "Immersive Memory (You were...)"),
                ("biographical_chronicle", "Historical Biography"),
                ("dramatic_journey", "Dramatic Adventure")
            ],
            format_func=lambda x: x[1]
        )[0]
        
        # Process responses
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Predict My Past Life", type="primary", use_container_width=True):
                # Validate responses
                if not all(response.strip() for response in user_responses):
                    st.error("Please answer all questions before proceeding.")
                    return
                
                if not self.setup_pipeline():
                    return
                
                # Run prediction
                with st.spinner("Analyzing your personality and predicting your past life..."):
                    try:
                        result = st.session_state.pipeline.predict_past_life(
                            user_responses, narrative_style
                        )
                        st.session_state.prediction_result = result
                        st.session_state.user_responses = user_responses
                        
                        if 'error' in result:
                            st.error(result['message'])
                        else:
                            st.success("Past life prediction complete!")
                            
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
        
        # Display results if available
        if st.session_state.prediction_result and 'error' not in st.session_state.prediction_result:
            self.display_results()

    def display_results(self):
        """Display the prediction results"""
        result = st.session_state.prediction_result
        
        st.markdown("---")
        st.header("🌟 Your Past Life Prediction")
        
        # Story display
        story_data = result['past_life_story']
        st.markdown(f'<div class="story-container">', unsafe_allow_html=True)
        st.subheader(story_data['story']['title'])
        st.markdown(story_data['story']['full_narrative'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Character profile
        character = story_data['character_profile']
        st.subheader("📜 Character Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Name:** {character['name']}")
        with col2:
            st.markdown(f"**Period:** {character['historical_period']}")
        with col3:
            st.markdown(f"**Role:** {character['occupation']}")
        
        # Personality analysis
        st.subheader("Personality Analysis")
        personality_scores = result['personality_analysis']['personality_scores']
        
        for trait, data in personality_scores.items():
            col1, col2, col3 = st.columns([2, 1, 3])
            with col1:
                st.markdown(f"**{trait.title()}**")
            with col2:
                st.metric("Score", f"{data['final_score']:.1f}/10", data['level'])
            with col3:
                st.markdown(f"*{data['explanation']}*")
        
        # Quality metrics
        st.subheader("📊 Analysis Quality")
        metrics = story_data['quality_metrics']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Personality Alignment", f"{metrics['personality_alignment']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Historical Authenticity", f"{metrics['historical_authenticity']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Narrative Quality", f"{metrics['narrative_engagement']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing details
        with st.expander("🔧 Processing Details"):
            processing = result['processing_metadata']
            st.json({
                "Processing Time": f"{processing['processing_time_seconds']:.2f} seconds",
                "Pipeline Version": processing['pipeline_version'],
                "Narrative Style": processing['narrative_style'],
                "Input Length": f"{processing['input_text_length']} characters"
            })

    def render_technical_page(self):
        """Render technical implementation details"""
        st.header("📊 Technical Implementation")
        
        st.markdown("""
        ### NLP Pipeline Architecture
        
        This project demonstrates advanced NLP techniques through a multi-stage pipeline.
        """)

    def render_about_page(self):
        """Render about page with project information"""
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ### Past Life Predictor: Advanced NLP Pipeline
        
        This project demonstrates the integration of traditional NLP techniques with modern 
        large language models to create a comprehensive text analysis and generation system.
        """)

if __name__ == "__main__":
    app = PastLifePredictorApp()
    app.render_main_interface()