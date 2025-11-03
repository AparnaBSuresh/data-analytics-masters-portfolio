"""
Privacy and data protection components for medical-grade healthcare chatbot.
CEO perspective: HIPAA compliance and user trust are critical for medical applications.
"""

import streamlit as st


def render_privacy_notice():
    """Render HIPAA-compliant privacy notice."""
    with st.expander("🔒 Privacy & Data Protection", expanded=False):
        st.markdown("""
        **Your Privacy is Protected**
        
        🏥 **HIPAA Compliance**: This platform follows HIPAA guidelines for health information protection.
        
        🔐 **Data Security**: 
        - All conversations are encrypted in transit and at rest
        - No personal health information is stored permanently
        - Data is processed securely and deleted after session
        
        🛡️ **Your Rights**:
        - You control your data
        - No information is shared with third parties
        - You can delete your conversation history at any time
        
        📋 **Medical Disclaimer**: This AI provides general health information only and is not a substitute for professional medical advice.
        
        **For questions about privacy, contact: privacy@meditechsolutions.com**
        """)


def render_consent_checkbox():
    """Render consent checkbox for data processing."""
    consent = st.checkbox(
        "I understand this is for educational purposes only and agree to the privacy policy",
        value=False,
        help="Required to use the medical AI assistant"
    )
    return consent
