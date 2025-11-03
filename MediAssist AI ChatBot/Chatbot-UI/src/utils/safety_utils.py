"""
Safety and emergency detection utilities for medical-grade healthcare chatbot.
CEO perspective: Patient safety is paramount - detect emergencies and escalate appropriately.
"""

import re
from typing import List, Dict, Tuple
from ..config.constants import EMERGENCY_KEYWORDS


def detect_emergency(user_input: str) -> Tuple[bool, List[str]]:
    """
    Detect emergency situations in user input.
    
    Args:
        user_input: User's message text
        
    Returns:
        Tuple of (is_emergency, detected_keywords)
    """
    user_lower = user_input.lower()
    detected_keywords = []
    
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in user_lower:
            detected_keywords.append(keyword)
    
    # Additional pattern matching for emergency situations
    emergency_patterns = [
        r'\b(chest pain|heart attack|stroke|bleeding|unconscious|emergency)\b',
        r'\b(can\'t breathe|difficulty breathing|choking)\b',
        r'\b(severe pain|intense pain|excruciating)\b',
        r'\b(overdose|poison|suicide|self harm)\b',
        r'\b(911|emergency room|ER|ambulance)\b'
    ]
    
    for pattern in emergency_patterns:
        matches = re.findall(pattern, user_lower)
        detected_keywords.extend(matches)
    
    is_emergency = len(detected_keywords) > 0
    return is_emergency, list(set(detected_keywords))


def generate_emergency_response(detected_keywords: List[str]) -> str:
    """
    Generate appropriate emergency response based on detected keywords.
    
    Args:
        detected_keywords: List of detected emergency keywords
        
    Returns:
        Emergency response message
    """
    response = "🚨 **EMERGENCY DETECTED** 🚨\n\n"
    response += "**IMMEDIATE ACTION REQUIRED:**\n"
    response += "• Call 911 or your local emergency services immediately\n"
    response += "• Do not delay seeking emergency medical care\n"
    response += "• If you're alone, call emergency services now\n\n"
    
    response += f"**Detected concerns:** {', '.join(detected_keywords)}\n\n"
    
    response += "**This AI assistant cannot provide emergency medical care.**\n"
    response += "**Your safety is the top priority - please seek immediate professional medical attention.**"
    
    return response


def add_safety_disclaimer(response: str, is_emergency: bool = False) -> str:
    """
    Add appropriate safety disclaimer to AI responses.
    
    Args:
        response: Original AI response
        is_emergency: Whether this is an emergency situation
        
    Returns:
        Response with appropriate safety disclaimer
    """
    if is_emergency:
        return response  # Emergency response already includes disclaimer
    
    disclaimer = "\n\n---\n"
    disclaimer += "⚠️ **IMPORTANT:** This information is for educational purposes only and does not constitute medical advice. "
    disclaimer += "Always consult with qualified healthcare professionals for medical decisions. "
    disclaimer += "In emergencies, call 911 immediately."
    
    return response + disclaimer


def validate_medical_input(user_input: str) -> Dict[str, any]:
    """
    Validate user input for medical appropriateness and safety.
    
    Args:
        user_input: User's message text
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "is_valid": True,
        "warnings": [],
        "suggestions": []
    }
    
    # Check for inappropriate content
    inappropriate_patterns = [
        r'\b(drug|medication|prescription)\s+(name|dosage|amount)\b',
        r'\b(how to|how do i)\s+(hurt|harm|kill)\b',
        r'\b(suicide|self harm|end my life)\b'
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, user_input.lower()):
            validation_result["warnings"].append("Content may require professional intervention")
            validation_result["suggestions"].append("Consider consulting mental health professionals")
    
    # Check for medication-specific questions
    if re.search(r'\b(medication|drug|pill|tablet)\b', user_input.lower()):
        validation_result["suggestions"].append("For medication questions, consult your pharmacist or doctor")
    
    return validation_result


def get_escalation_guidance(concern_type: str) -> str:
    """
    Provide appropriate escalation guidance based on concern type.
    
    Args:
        concern_type: Type of medical concern
        
    Returns:
        Escalation guidance message
    """
    escalation_guides = {
        "emergency": "Call 911 immediately or go to the nearest emergency room",
        "urgent": "Contact your healthcare provider today or visit urgent care",
        "routine": "Schedule an appointment with your healthcare provider",
        "preventive": "Discuss with your healthcare provider at your next visit"
    }
    
    return escalation_guides.get(concern_type, "Consult with your healthcare provider")
