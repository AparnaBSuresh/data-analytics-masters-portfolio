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
from src.ui.components import render_sidebar, render_knowledge_tab, render_sources_tab, render_chat_tab


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    if "rag_index" not in st.session_state:
        st.session_state.rag_index = None
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = []  # [{'name','text','size'}]
    if "last_snippets" not in st.session_state:
        st.session_state.last_snippets = []


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Main title
    st.title("💊 Healthcare Copilot")
    st.write("A friendly, retrieval‑augmented chat UI with fine‑tuned model support.")
    
    # Tabs
    tab_chat, tab_knowledge, tab_sources = st.tabs(["💬 Chat", "📚 Knowledge", "🔍 Sources"])
    
    # Render tabs
    with tab_knowledge:
        render_knowledge_tab()
    
    with tab_sources:
        render_sources_tab()
    
    with tab_chat:
        render_chat_tab(settings)


if __name__ == "__main__":
    main()