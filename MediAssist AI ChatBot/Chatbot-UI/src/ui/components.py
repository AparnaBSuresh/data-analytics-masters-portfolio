"""
UI components for the Healthcare Chatbot application.
"""

import streamlit as st
import os
import time
from typing import List, Dict, Optional
from ..config.constants import (
    QUICK_PROMPTS, DISCLAIMER, DEFAULT_SYSTEM_PROMPT, 
    APP_TITLE, APP_SUBTITLE, COMPANY_INFO, FINETUNED_MODELS, MEDICAL_DEPARTMENTS,
    QUICK_PROMPT_ANSWERS, DEPARTMENT_QUICK_ANSWERS
)
from ..utils.safety_utils import detect_emergency, generate_emergency_response, add_safety_disclaimer
from ..llm.backends import call_llm
from .privacy_notice import render_privacy_notice, render_consent_checkbox


def render_medical_header():
    """Render the professional medical header with company credentials."""
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🏥 {APP_TITLE}</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">{APP_SUBTITLE}</p>
        <div style="margin-top: 1rem; display: flex; gap: 1rem; flex-wrap: wrap;">
            <span class="trust-badge">HIPAA-Compliant</span>
            <span class="trust-badge">Physician-Validated</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_medical_disclaimer():
    """Render the prominent medical disclaimer."""
    st.markdown(f"""
    <div class="medical-disclaimer">
        {DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with settings and controls."""
    with st.sidebar:
        st.header("⚙️ Medical AI Settings")
        
        # Company credentials
        st.markdown(f"""
        <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #6b7280;">
                {COMPANY_INFO['credentials']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # System prompt (hidden from user)
        system_prompt = DEFAULT_SYSTEM_PROMPT

        # Medical Department Selection
        st.subheader("🏥 Medical Department")
        department = st.selectbox(
            "Choose Department", 
            list(MEDICAL_DEPARTMENTS.keys()), 
            index=0,
            help="Select the medical department for specialized responses"
        )
        
        # Show department info
        if department in MEDICAL_DEPARTMENTS:
            dept_info = MEDICAL_DEPARTMENTS[department]
            st.caption(f"**{dept_info['description']}**")
            with st.expander("Common Conditions", expanded=False):
                for condition in dept_info['common_conditions']:
                    st.write(f"• {condition}")

        # Model backend - use GGUF for fastest CPU inference
        provider = "Fine-tuned Models"
        model_name = "MedLlama-GGUF"  # Use GGUF format (fastest on CPU)
        
        # HuggingFace token (read from environment or hardcoded)
        # Try environment first, then fallback to hardcoded (not recommended for production)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") 
        
        # Not used for Fine-tuned Models
        api_key = ""
        api_base = ""
        hf_model_path = ""

        # Parameters
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

        return {
            "system_prompt": system_prompt,
            "department": department,
            "provider": provider,
            "model_name": model_name,
            "api_key": api_key,
            "api_base": api_base,
            "hf_model_path": hf_model_path,
            "hf_token": hf_token,
            "temperature": temperature
        }



def render_chat_tab(settings: Dict):
    """Render the main chat interface."""
    # Initialize clear counter if not exists (used to reset chat input widget)
    if "clear_counter" not in st.session_state:
        st.session_state.clear_counter = 0
    
    # Show chat history at the top (above chat input)
    for m in st.session_state.messages:
        if m["role"] == "system":
            continue
        bubble_class = "user-bubble" if m["role"] == "user" else "assist-bubble"
        st.markdown(f'<div class="chat-bubble {bubble_class}">{m["content"]}</div>', unsafe_allow_html=True)
        
        # Show performance metrics if stored with assistant message
        if m["role"] == "assistant" and "metrics" in m:
            st.markdown(m["metrics"], unsafe_allow_html=True)
    
    st.markdown("---")  # Separator between history and input
    
    # Input - use key with clear_counter to force widget reset when cleared
    user_msg = st.chat_input("Ask a healthcare question...", key=f"chat_input_{st.session_state.clear_counter}")
    
    # Quick prompts below the chat input - department specific
    st.subheader("💡 Quick Prompts")
    
    # Get department-specific prompts
    department = settings.get("department", "General")
    if department in MEDICAL_DEPARTMENTS:
        dept_info = MEDICAL_DEPARTMENTS[department]
        # Create department-specific prompts
        dept_prompts = [
            f"What are common symptoms of {dept_info['common_conditions'][0].lower()}?",
            f"How is {dept_info['common_conditions'][1].lower()} diagnosed?",
            f"What are the treatment options for {dept_info['common_conditions'][2].lower()}?",
            f"When should I seek emergency care for {department.lower()} issues?"
        ]
        qp = st.pills(f"Try these {department} questions:", dept_prompts, selection_mode="single", key=f"dept_pills_{st.session_state.clear_counter}")
    else:
        qp = st.pills("Try these common questions:", QUICK_PROMPTS, selection_mode="single", key=f"quick_pills_{st.session_state.clear_counter}")
    
    # Handle quick prompt selection - use it immediately if clicked
    if user_msg is None and qp:
        user_msg = qp

    if user_msg:
        # Check if we've already processed this exact user message with a response
        # Look for the pattern: [..., user_msg, assistant_response] in recent messages
        already_processed = False
        if len(st.session_state.messages) >= 2:
            last_msg = st.session_state.messages[-1]
            second_last_msg = st.session_state.messages[-2]
            # If last two messages are [user_msg, assistant_response], we've already processed
            if (second_last_msg.get("role") == "user" and 
                second_last_msg.get("content") == user_msg and
                last_msg.get("role") == "assistant"):
                already_processed = True
        
        # If already processed, skip all processing and just render controls
        if already_processed:
            pass  # Continue to render controls at bottom
        else:
            # Add user message to history first - it will appear in history at top on next render
            st.session_state.messages.append({"role": "user", "content": user_msg})
            # Don't rerun here - continue processing so model can be loaded and called

            # CHECK FOR HARDCODED QUICK PROMPT ANSWERS - instant response
            hardcoded_answer = None
            
            # Check exact match with general quick prompts
            if user_msg in QUICK_PROMPT_ANSWERS:
                hardcoded_answer = QUICK_PROMPT_ANSWERS[user_msg]
            else:
                # Check if it matches department-specific prompt patterns
                department = settings.get("department", "General")
                user_msg_lower = user_msg.lower()
                
                # Check for department-specific patterns
                if "what are common symptoms" in user_msg_lower or "common symptoms" in user_msg_lower:
                    # Try to extract condition from the question
                    hardcoded_answer = DEPARTMENT_QUICK_ANSWERS.get("common_symptoms", "")
                    # Customize based on department
                    if department in MEDICAL_DEPARTMENTS:
                        dept_info = MEDICAL_DEPARTMENTS[department]
                        if dept_info.get('common_conditions'):
                            conditions = ", ".join(dept_info['common_conditions'][:3])
                            hardcoded_answer = f"""**Common Symptoms for {department}:**

**Typical symptoms vary by condition, but common {department.lower()} conditions include:**
- {conditions}

**General symptom patterns:**
- Primary symptoms: Pain, discomfort, or functional changes related to the affected system
- Secondary symptoms: Fatigue, changes in appetite, sleep disturbances
- Warning signs requiring immediate attention: Severe pain, difficulty breathing, loss of consciousness, chest pain

**When to Seek Help:**
- Severe or worsening symptoms
- Symptoms interfering with daily activities
- Signs of complications or systemic involvement
- Any emergency warning signs

⚠️ **This is general information. Always consult with a healthcare provider for proper evaluation and diagnosis.**"""
                elif "how is" in user_msg_lower and "diagnosed" in user_msg_lower:
                    hardcoded_answer = DEPARTMENT_QUICK_ANSWERS.get("diagnosis", "")
                elif "treatment options" in user_msg_lower or ("how is" in user_msg_lower and "treated" in user_msg_lower):
                    hardcoded_answer = DEPARTMENT_QUICK_ANSWERS.get("treatment_options", "")
                elif "emergency care" in user_msg_lower or "when should i seek" in user_msg_lower:
                    hardcoded_answer = DEPARTMENT_QUICK_ANSWERS.get("emergency_care", "")
            
            # If we found a hardcoded answer, return it immediately
            if hardcoded_answer:
                # Use hardcoded answer without disclaimer
                final_answer = hardcoded_answer
                
                # Check if we've already added this exact response (avoid duplicates on rerun)
                last_assistant = None
                for i in range(len(st.session_state.messages) - 1, -1, -1):
                    if st.session_state.messages[i]["role"] == "assistant":
                        last_assistant = st.session_state.messages[i]
                        break
                
                # If last assistant response is different from what we're about to add, or doesn't exist, add it
                if not last_assistant or last_assistant.get("content") != final_answer:
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                # Rerun to show answer in history section at top
                st.rerun()

            # EMERGENCY DETECTION - CEO priority: Patient safety first
            is_emergency, detected_keywords = detect_emergency(user_msg)
            
            if is_emergency:
                # Emergency response - immediate escalation
                emergency_response = generate_emergency_response(detected_keywords)
                st.session_state.messages.append({"role": "assistant", "content": emergency_response})
                # Rerun to show emergency response in history section at top
                st.rerun()

            # Compose messages (replace system with current prompt)
            history = [{"role": "system", "content": settings["system_prompt"]}]
            for m in st.session_state.messages:
                if m["role"] != "system":
                    history.append(m)

            # Show loading message at the bottom (below quick prompts)
            spinner_msg = "Analyzing your health concern..."
            if settings["model_name"] in FINETUNED_MODELS:
                model_config = FINETUNED_MODELS[settings["model_name"]]
                if model_config.get("use_api"):
                    spinner_msg = "🧠 Processing with Medical AI... (This may take 30-60 seconds on CPU)"
                elif model_config.get("use_gguf"):
                    spinner_msg = "Generating response with quantized model... (This may take 15-30 seconds)"
                else:
                    # Local model - check if it's likely first load
                    from ..llm.backends import _model_cache
                    model_path = model_config.get("model_path", "")
                    cache_key = f"{model_path}_{model_config.get('is_adapter', False)}"
                    if cache_key not in _model_cache:
                        spinner_msg = "⏳ Loading model from HuggingFace (first time - this may take 5-10 minutes)..."
                    else:
                        spinner_msg = "🧠 Generating response with local model... (This may take 30-60 seconds on CPU)"
            
            # Process the request (loading message will show in spinner)
            # Display single loading message at the bottom (below quick prompts)
            loading_container = st.empty()
            with loading_container:
                st.info(f"⚡ {spinner_msg}")
            
            with st.spinner(""):
                # Empty spinner - we're showing info message instead
                start_time = time.time()
                try:
                    output = call_llm(
                        history,
                        provider=("Debug/Echo" if settings["provider"] == "Debug/Echo" else settings["provider"]),
                        model_name=settings["model_name"],
                        temperature=settings["temperature"],
                        api_key=settings["api_key"],
                        api_base=settings["api_base"],
                        hf_model_path=settings["hf_model_path"],
                        hf_token=settings.get("hf_token", ""),
                        max_new_tokens=1024,  # Increased for comprehensive responses (300+ words)
                        department=settings["department"]
                    )
                    total_time = time.time() - start_time
                    
                    # Get detailed timing info if available (for GGUF)
                    from ..llm.backends import get_last_timing_info
                    timing_info = get_last_timing_info()
                    
                    # Store timing in session state for display
                    if timing_info:
                        st.session_state.last_timing = timing_info
                    else:
                        # Fallback timing for other backends
                        st.session_state.last_timing = {"total_time": total_time}
                except Exception as e:
                    error_msg = str(e)
                    if "timeout" in error_msg.lower():
                        output = "⏱️ **Request Timeout**\n\nThe model is taking longer than expected (likely running on CPU). This is normal for the first few requests.\n\n**Tips:**\n- Wait a bit longer and try again\n- The model may still be processing your previous request\n- Consider using a GPU for faster inference\n\n**Technical details:** " + error_msg
                    else:
                        output = f"❌ **Error:** {error_msg}"
            
            # Clear loading message after processing
            loading_container.empty()

            # Prepare performance metrics if available
            metrics_html = None
            if "last_timing" in st.session_state and st.session_state.last_timing:
                timing = st.session_state.last_timing
                
                if "generation_time" in timing:
                    # Detailed timing for GGUF - display as compact metrics
                    metrics_html = f"""
<div style="background: #f0f9ff; padding: 12px; border-radius: 8px; margin-top: 8px; border-left: 3px solid #3b82f6;">
<strong>⚡ Performance Metrics</strong><br>
<span style="font-size: 0.85em;">
Total: <strong>{timing.get('total_time', 0):.2f}s</strong> | 
Generation: <strong>{timing.get('generation_time', 0):.2f}s</strong> | 
Speed: <strong>{timing.get('tokens_per_sec', 0):.1f} tok/s</strong> | 
Tokens: <strong>{timing.get('tokens_generated', 0)}</strong>
</span>
</div>
"""
                else:
                    # Simple timing for other backends
                    metrics_html = f"<div style='font-size: 0.85em; color: #6b7280; margin-top: 8px;'>⏱️ Response time: {timing.get('total_time', 0):.2f} seconds</div>"
            
            # Add assistant response to messages with metrics
            assistant_msg = {"role": "assistant", "content": output}
            if metrics_html:
                assistant_msg["metrics"] = metrics_html
            st.session_state.messages.append(assistant_msg)
            
            # Clear processing flag if it exists
            if "_processing_msg" in st.session_state:
                del st.session_state._processing_msg
            
            # Rerun to show both question and answer in the history section at top
            st.rerun()

    # Controls
    render_chat_controls(settings["system_prompt"])


def render_chat_controls(system_prompt: str):
    """Render chat control buttons."""
    import json
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear", use_container_width=True):
            # Clear all chat-related session state
            st.session_state.messages = [{"role": "system", "content": system_prompt}]
            
            # Clear timing info if it exists
            if "last_timing" in st.session_state:
                del st.session_state.last_timing
            
            # Clear processing flag if it exists
            if "_processing_msg" in st.session_state:
                del st.session_state._processing_msg
            
            # Increment clear counter to reset chat input widget
            st.session_state.clear_counter = st.session_state.get("clear_counter", 0) + 1
            
            st.rerun()

    with col2:
        data = {"title": "Healthcare Copilot", "messages": st.session_state.messages}
        b = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("💾 Export chat", b, file_name="chat_export.json", mime="application/json", use_container_width=True)
    
    # Privacy and data protection at the bottom (consent checkbox moved here after acceptance)
    st.markdown("---")
    st.subheader("🔒 Privacy & Data Protection")
    render_privacy_notice()
    
    # Show consent checkbox below the privacy notice (for reference only - consent already given at app level)
    # Don't check or stop execution here, just display for transparency
    render_consent_checkbox()
