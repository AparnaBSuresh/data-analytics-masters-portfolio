"""
Healthcare Chatbot - Main Application
Streamlined, styled Streamlit UI for a Healthcare Chatbot with RAG and pluggable backends

Author: ChatGPT (for Aparna)

Run:
  pip install -r requirements.txt
  streamlit run app.py

Backends supported:
  1) OpenAI chat completions (e.g., your fine-tuned model on OpenAI)
  2) Together API models
  3) Local HuggingFace Transformers pipeline (for your own fine-tuned model path)

NOTES:
- Replace the call_llm() backend settings with your fine-tuned model info.
- This UI is not a medical device; add your own compliance guardrails before deployment.
"""

import streamlit as st

# Import our modular components
from src.config.constants import APP_TITLE, DEFAULT_SYSTEM_PROMPT
from src.ui.styles import CSS
from src.ui.components import (
    render_medical_header, render_medical_disclaimer, render_sidebar, 
    render_chat_tab
)
from src.ui.privacy_notice import render_consent_checkbox


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render medical-grade header
    render_medical_header()

    # Check if consent has been given previously (persists across reruns)
    if "consent_accepted" not in st.session_state:
        st.session_state.consent_accepted = False
    
    # Only show consent checkbox at top if not yet accepted
    if not st.session_state.consent_accepted:
        consent = render_consent_checkbox()
        
        if not consent:
            st.warning("Please accept the privacy policy to continue using the medical AI assistant.")
            st.stop()
        else:
            # Store consent in session state (persists for entire session)
            st.session_state.consent_accepted = True
            st.rerun()  # Rerun to move checkbox to bottom
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Tabs
    tab_chat, = st.tabs(["💬 Medical Chat"])

    
    with tab_chat:
        render_chat_tab(settings)

    # Render prominent medical disclaimer
    render_medical_disclaimer()
    

if __name__ == "__main__":
    main()