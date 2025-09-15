import streamlit as st
import json
import time
from datetime import datetime
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
        margin-bottom: 2rem;
    }
    .personality-score {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        margin: 0.5rem 0;
    }
    .story-container {
        background: #f8fafc;
        border-left: 5px solid #3b82f6;
    }
    .metric-box {
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class PastLifePredictorApp:
    """Streamlit web application for Past Life Predictor"""
    
    def __init__(self):
        """Initialize the Streamlit app"""
        self.pipeline = PastLifePredictorPipeline()

    def setup_pipeline(self):
        """Initialize the prediction pipeline"""
        self.pipeline.setup_logging()

    def render_main_interface(self):
        """Render the main application interface"""
        st.title("Past Life Predictor")
        st.write("Discover your past life through personality analysis!")

        user_input = st.text_area("Describe yourself in a few sentences:")
        if st.button("Predict Past Life"):
            if user_input:
                with st.spinner("Analyzing..."):
                    result = self.pipeline.predict_past_life([user_input])
                    self.display_results(result)
            else:
                st.warning("Please enter a description.")

    def display_results(self, result):
        """Display the prediction results"""
        st.subheader("Personality Analysis")
        for trait, score in result['personality_scores'].items():
            st.markdown(f"**{trait.title()}:** {score['final_score']}/10 - {score['explanation']}")

        st.subheader("Predicted Past Life")
        st.write(result['predicted_past_life'])

    def render_technical_page(self):
        """Render technical implementation details"""
        st.subheader("Technical Implementation")
        st.write("Details about the implementation will go here.")

    def render_about_page(self):
        """Render about page with project information"""
        st.subheader("About This Project")
        st.write("This project aims to analyze personality traits and predict past lives.")

if __name__ == "__main__":
    app = PastLifePredictorApp()
    app.render_main_interface()