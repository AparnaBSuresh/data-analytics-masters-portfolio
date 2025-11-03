"""
Configuration constants for the Healthcare Chatbot application.
"""

# App metadata
APP_TITLE = "MediAssist AI - Healthcare Guidance Platform"
APP_SUBTITLE = "Powered by Advanced AI with Medical Expert Validation"

# Medical disclaimer - CEO perspective: Clear, prominent, legally compliant
DISCLAIMER = (
    "⚠️ **MEDICAL DISCLAIMER:** This AI assistant provides general health information only and is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult qualified healthcare providers for medical decisions. In emergencies, call 911 immediately. "
    "This platform is HIPAA-compliant and follows FDA guidelines for health information systems."
)

# Emergency protocols
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "severe bleeding", "unconscious", 
    "difficulty breathing", "suicide", "overdose", "severe allergic reaction",
    "emergency", "urgent", "critical", "life threatening"
]

# Trust and credibility elements
COMPANY_INFO = {
    "credentials": "HIPAA-Compliant • Physicain Validated"
}

# Enhanced system prompt for medical-grade AI
DEFAULT_SYSTEM_PROMPT = """You are MediAssist AI, a comprehensive medical information assistant developed by MediTech Solutions and validated by board-certified physicians.

CORE PRINCIPLES:
- Provide detailed, evidence-based health information
- Always emphasize this is NOT medical advice or diagnosis
- Give comprehensive explanations with multiple paragraphs
- Maintain professional, empathetic, and educational tone
- Explain medical concepts in clear, understandable language

RESPONSE FORMAT - Always include ALL sections:

1. **Understanding the Concern** (2-3 sentences)
   - Acknowledge their symptoms with empathy
   - Explain what might be happening physiologically

2. **Possible Causes** (4-6 bullet points)
   - List common causes for their symptoms
   - Explain each cause briefly

3. **Self-Care Measures** (4-5 recommendations)
   - Home remedies and lifestyle modifications
   - Over-the-counter options (mention generic names)
   - When to try each approach

4. **Warning Signs** (4-5 red flags)
   - Symptoms that require immediate medical attention
   - Signs of serious complications

5. **When to See a Doctor**
   - Timeline for seeking professional care
   - What type of healthcare provider to see
   - What to expect during the visit

6. **Important Disclaimer**
   - Remind that this is educational information only
   - Emphasize the need for professional medical evaluation

RESPONSE REQUIREMENTS:
- Minimum 300 words per response
- Use clear paragraphs and bullet points
- Explain medical terms in plain language
- Be thorough but accessible
- Never give specific prescriptions or medications to take

EMERGENCY PROTOCOLS:
- If emergency keywords detected, immediately recommend calling 911
- Never delay emergency care recommendations
- Always prioritize patient safety
"""

# Agent profiles for different healthcare contexts
AGENT_PROFILES = {
    "General Care": "Focus on general patient education and safe triage suggestions.",
    "Pre‑Operative": "Pre‑op instructions, med holds, fasting, consent, risk mitigation.",
    "Post‑Operative": "Wound care, pain control, warning signs, mobility, follow‑up.",
    "Chronic Care": "Long‑term condition tracking, adherence, lifestyle guidance.",
    "Assisted Living": "Daily support, fall risk, medication reminders, caregiver tips.",
}

# Fine-tuned medical models
FINETUNED_MODELS = {
    # Option 1: Use Hugging Face Inference API (cloud-based, fast)
    "MedLlama-HF-API": {
        "description": "MedLlama - Hugging Face Inference API",
        "specialization": "Clinical reasoning and medical decision making",
        "api_url": "https://router.huggingface.co/hf-inference/models/AparnaSuresh/MedLlama-3b?wait_for_model=true",
        "provider": "HuggingFace",
        "is_finetuned": True,
        "use_api": True,
        "use_hf_inference": True
    },
    # Option 2: Load model locally from Hugging Face (requires GPU or patience with CPU)
    "MedLlama-Local": {
        "description": "MedLlama - Local from HuggingFace",
        "specialization": "Clinical reasoning and medical decision making",
        "model_path": "AparnaSuresh/MedLlama-3b",  # HuggingFace repo ID
        "provider": "HuggingFace",
        "is_finetuned": True,
        "use_api": False,  # Load locally, not via API
        "use_hf_inference": False,
        "is_adapter": False  # Full model, not just adapter
    },
    # Option 3: GGUF format with llama.cpp (FASTEST on CPU!)
    "MedLlama-GGUF": {
        "description": "MedLlama - GGUF format (Fastest CPU inference)",
        "specialization": "Clinical reasoning and medical decision making",
        "model_path": "./models/medllama-3b-gguf/medllama-3b-q4_k_m.gguf",  # Path to .gguf file (Q4_K_M quantized format for faster inference)
        "provider": "llama.cpp",
        "is_finetuned": True,
        "use_api": False,
        "use_hf_inference": False,
        "use_gguf": True,  # Flag to use GGUF backend
        "is_adapter": False
    }
}

# Medical departments based on MIMIC-3, MIMIC-4, MedQA, and iCliniq datasets
MEDICAL_DEPARTMENTS = {
    "Cardiology": {
        "description": "Heart and cardiovascular system conditions",
        "common_conditions": ["Heart attack", "Arrhythmia", "Heart failure", "Hypertension", "Chest pain"],
        "specialization": "Cardiovascular diseases, heart conditions, blood pressure management"
    },
    "Emergency Medicine": {
        "description": "Acute medical conditions requiring immediate attention",
        "common_conditions": ["Trauma", "Severe pain", "Breathing difficulties", "Allergic reactions", "Poisoning"],
        "specialization": "Emergency care, trauma, acute conditions, triage"
    },
    "Internal Medicine": {
        "description": "Adult internal organ diseases and general medicine",
        "common_conditions": ["Diabetes", "Hypertension", "Infections", "Metabolic disorders", "Chronic diseases"],
        "specialization": "General internal medicine, chronic disease management, adult health"
    },
    "Neurology": {
        "description": "Nervous system and brain-related conditions",
        "common_conditions": ["Stroke", "Epilepsy", "Headaches", "Memory disorders", "Movement disorders"],
        "specialization": "Brain and nervous system disorders, neurological conditions"
    },
    "Oncology": {
        "description": "Cancer diagnosis, treatment, and care",
        "common_conditions": ["Breast cancer", "Lung cancer", "Blood cancers", "Tumor management", "Chemotherapy side effects"],
        "specialization": "Cancer treatment, tumor management, oncology care"
    },
    "Surgery": {
        "description": "Surgical procedures and perioperative care",
        "common_conditions": ["Appendicitis", "Gallbladder issues", "Hernias", "Trauma surgery", "Elective procedures"],
        "specialization": "Surgical procedures, pre/post-operative care, surgical complications"
    },
    "Psychiatry": {
        "description": "Mental health and behavioral disorders",
        "common_conditions": ["Depression", "Anxiety", "Bipolar disorder", "Schizophrenia", "Substance abuse"],
        "specialization": "Mental health, psychiatric disorders, behavioral medicine"
    },
    "Radiology": {
        "description": "Medical imaging and diagnostic procedures",
        "common_conditions": ["X-ray interpretation", "CT scans", "MRI analysis", "Ultrasound", "Nuclear medicine"],
        "specialization": "Medical imaging, diagnostic radiology, image interpretation"
    }
}

# Quick prompts for common healthcare questions
QUICK_PROMPTS = [
    "Pre‑op medication question",
    "Post‑op wound care", 
    "Diabetes foot care tips",
    "Fall prevention checklist"
]

# Hardcoded answers for quick prompts - instant responses without LLM
QUICK_PROMPT_ANSWERS = {
    "Pre‑op medication question": """**Pre-Operative Medication Guidelines:**

**General Rules:**
- **Continue taking:** Blood pressure medications, heart medications (with a small sip of water)
- **STOP 1 week before:** Aspirin, ibuprofen, blood thinners (warfarin, clopidogrel) - *only if your surgeon approves*
- **STOP morning of surgery:** Most medications, especially diabetes medications
- **Discuss with surgeon:** All medications, supplements, and herbal remedies

**Important:**
- Never stop medications without your doctor's approval
- Bring a complete medication list to pre-op visit
- Inform your surgical team about ALL medications you take

⚠️ **Always consult with your surgical team and primary care physician before making any medication changes.**""",

    "Post‑op wound care": """**Post-Operative Wound Care:**

**Keep Wound Clean & Dry:**
- Follow your surgeon's specific instructions
- Gently wash with mild soap and water (unless told otherwise)
- Pat dry with clean towel
- Keep dressing clean and dry
- Change dressing as instructed

**Watch for Red Flags - Call Doctor Immediately:**
- Increasing redness, swelling, or warmth around the wound
- Pus or foul-smelling drainage
- Fever (temperature >100.4°F)
- Increased pain that doesn't respond to medication
- Wound edges pulling apart
- Red streaks spreading from the wound

**Normal Healing:**
- Mild redness and swelling (should decrease over time)
- Clear or slightly yellow drainage
- Gradual decrease in pain
- Wound edges coming together

⚠️ **This is general guidance. Always follow your surgeon's specific instructions and call them with any concerns.**""",

    "Diabetes foot care tips": """**Diabetes Foot Care - Daily Checklist:**

**Daily Inspection:**
- Check feet every day (use mirror for bottom of feet)
- Look for cuts, blisters, redness, swelling, or nail problems
- Check for temperature changes (warm or cold spots)

**Proper Foot Hygiene:**
- Wash feet daily in lukewarm water (test temperature with elbow)
- Dry gently, especially between toes
- Apply moisturizer (avoid between toes)
- Keep toenails trimmed straight across, not too short

**Foot Protection:**
- Never walk barefoot (even indoors)
- Wear properly fitting shoes and socks
- Check inside shoes for foreign objects
- Break in new shoes gradually

**When to See Doctor:**
- Any cut, sore, or blister that doesn't start healing after 1-2 days
- Any signs of infection (redness, warmth, drainage)
- Numbness, tingling, or burning in feet
- Changes in foot shape or color

⚠️ **Diabetes increases risk of foot complications. Regular podiatry visits are essential for preventing serious problems.**""",

    "Fall prevention checklist": """**Fall Prevention Checklist:**

**Home Safety:**
- ✅ Remove clutter from walkways
- ✅ Secure loose rugs or remove them
- ✅ Install grab bars in bathroom
- ✅ Improve lighting (especially stairs and hallways)
- ✅ Use non-slip mats in bathroom and kitchen
- ✅ Keep frequently used items within easy reach
- ✅ Fix loose handrails on stairs

**Personal Safety:**
- ✅ Wear proper footwear (non-slip, well-fitting)
- ✅ Use assistive devices if recommended (cane, walker)
- ✅ Take time - don't rush
- ✅ Be cautious on wet or icy surfaces
- ✅ Regular exercise to maintain strength and balance
- ✅ Review medications with doctor (some can cause dizziness)

**Health Factors:**
- ✅ Regular vision checkups
- ✅ Review blood pressure management
- ✅ Discuss osteoporosis risk with doctor
- ✅ Stay hydrated

⚠️ **Falls are a leading cause of injury in older adults. Consult with your healthcare provider for personalized fall prevention strategies.**"""
}

# Department-specific quick prompt templates and hardcoded answers
DEPARTMENT_QUICK_ANSWERS = {
    "common_symptoms": """**Common Symptoms:**

**Typical symptoms include:**
- Primary symptoms: [varies by condition]
- Secondary symptoms: [varies by condition]
- Warning signs requiring immediate attention: [varies by condition]

**When to Seek Help:**
- Severe or worsening symptoms
- Symptoms interfering with daily activities
- Signs of complications

⚠️ **This is general information. Always consult with a healthcare provider for proper evaluation and diagnosis.**""",

    "diagnosis": """**Diagnosis Process:**

**Typical diagnostic steps:**
1. Medical history and physical examination
2. Laboratory tests (blood work, etc.)
3. Imaging studies if indicated
4. Specialized testing as needed

**What to Expect:**
- Diagnostic process varies by condition
- Some tests may require preparation
- Results may take time to process

⚠️ **Diagnosis should always be made by qualified healthcare professionals based on complete evaluation.**""",

    "treatment_options": """**Treatment Options:**

**Common approaches include:**
- Lifestyle modifications
- Medications as prescribed
- Physical therapy or rehabilitation
- Surgical options when appropriate
- Supportive care

**Important:**
- Treatment plans are individualized
- Discuss all options with your healthcare team
- Ask about benefits, risks, and alternatives

⚠️ **Treatment decisions should be made in consultation with qualified healthcare providers who understand your specific situation.**""",

    "emergency_care": """**When to Seek Emergency Care:**

**Seek IMMEDIATE emergency care (call 911) for:**
- Chest pain or pressure
- Severe difficulty breathing
- Signs of stroke (sudden weakness, confusion, speech problems)
- Severe allergic reactions
- Uncontrolled bleeding
- Loss of consciousness
- Severe trauma

**Seek prompt medical attention (same-day urgent care) for:**
- Persistent severe pain
- High fever with other symptoms
- Worsening condition despite treatment
- Concerning new symptoms

⚠️ **In medical emergencies, call 911 immediately. Do not delay seeking emergency care.**"""
}

# RAG configuration
DEFAULT_RAG_TOP_K = 4
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MAX_FEATURES = 40000

# LLM configuration
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_NEW_TOKENS = 128  # Reduced to 128 for F16 GGUF (VERY slow on CPU). Quantize to Q4_K_M for better speed!
DEFAULT_MODEL_NAME = "MedLlama-GGUF"  # Currently using GGUF format for fast CPU inference

# Supported file types for knowledge base
SUPPORTED_FILE_TYPES = ["pdf", "txt"]
