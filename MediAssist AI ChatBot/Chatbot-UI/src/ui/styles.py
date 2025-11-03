"""
Medical-grade CSS styles for the Healthcare Chatbot application.
CEO perspective: Professional, trustworthy, accessible, compliant.
"""

CSS = """
<style>
/* Medical-grade color scheme */
:root {
    --medical-blue: #1e40af;
    --medical-light-blue: #dbeafe;
    --medical-green: #059669;
    --medical-red: #dc2626;
    --medical-gray: #6b7280;
    --medical-light-gray: #f9fafb;
    --medical-border: #e5e7eb;
}

/* App container - medical professional layout */
.block-container { 
    padding-top: 1rem; 
    max-width: 1200px; 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, var(--medical-blue) 0%, #3b82f6 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Medical disclaimer - prominent placement */
.medical-disclaimer {
    background: #fef3c7;
    border: 2px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #92400e;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Emergency alert styling */
.emergency-alert {
    background: #fee2e2;
    border: 2px solid var(--medical-red);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #991b1b;
    font-weight: bold;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

/* Professional chat bubbles */
.chat-bubble { 
    border-radius: 12px; 
    padding: 16px 20px; 
    margin: 8px 0; 
    line-height: 1.6;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border: 1px solid var(--medical-border);
}

.user-bubble { 
    background: var(--medical-light-blue); 
    border-left: 4px solid var(--medical-blue);
    margin-left: 20%;
}

.assist-bubble { 
    background: white; 
    border-left: 4px solid var(--medical-green);
    margin-right: 20%;
}

/* Medical source tags */
.tag { 
    display: inline-block; 
    padding: 4px 12px; 
    border-radius: 20px; 
    font-size: 11px; 
    margin-right: 8px; 
    border: 1px solid var(--medical-border); 
    background: var(--medical-light-gray);
    color: var(--medical-gray);
    font-weight: 500;
}

/* Citations and sources */
.citation { 
    font-size: 12px; 
    color: var(--medical-gray);
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--medical-border);
}

/* Trust indicators */
.trust-badge {
    background: var(--medical-green);
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: bold;
    margin-left: 8px;
}

/* Professional sidebar */
.sidebar .stSelectbox > div > div {
    background-color: white;
    border: 1px solid var(--medical-border);
    border-radius: 6px;
}

/* Hide default footer */
footer { visibility: hidden; }

/* Accessibility improvements */
.chat-bubble:focus {
    outline: 2px solid var(--medical-blue);
    outline-offset: 2px;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .user-bubble { margin-left: 10%; }
    .assist-bubble { margin-right: 10%; }
    .block-container { padding: 0.5rem; }
}
</style>
"""
