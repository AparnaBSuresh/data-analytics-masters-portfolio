# Simple Test Questions for Medical Chatbot

These questions are NOT in the quick prompts list and will test the quantized model performance.

## General Health Questions (Simple)

1. **What causes headaches?**
   - Simple, common condition
   - Tests general medical knowledge

2. **How do I treat a cold?**
   - Common everyday question
   - Tests practical advice

3. **What is blood pressure?**
   - Basic medical concept
   - Tests explanation ability

4. **When should I see a doctor?**
   - General health guidance
   - Tests triage knowledge

## Cardiology Questions

5. **What is high blood pressure?**
   - Cardiology-specific but simple
   - Tests department knowledge

6. **What causes chest pain?**
   - Common symptom question
   - Tests symptom recognition

## Respiratory Questions

7. **How do I know if I have asthma?**
   - Condition identification
   - Tests diagnostic awareness

8. **What helps with shortness of breath?**
   - Symptom management
   - Tests practical advice

## Digestive Questions

9. **What causes stomach pain?**
   - Common symptom
   - Tests general knowledge

10. **When is nausea serious?**
    - Symptom severity assessment
    - Tests triage ability

## Mental Health Questions

11. **What are signs of stress?**
    - Common mental health question
    - Tests awareness

12. **How can I sleep better?**
    - Lifestyle question
    - Tests practical advice

## Simple Follow-up Questions

13. **Is fever dangerous?**
    - Yes/no with explanation
    - Tests judgment

14. **How much water should I drink?**
    - Basic health advice
    - Tests general knowledge

15. **What vitamins do I need?**
    - Nutritional question
    - Tests basic knowledge

---

## Expected Performance (Q4_K_M Model)

After restarting Streamlit with the quantized model, you should see:

- **Response time:** 15-30 seconds (much better than 273 seconds!)
- **Quality:** Medical explanations appropriate for general public
- **Token generation:** 5-10 tokens/second
- **Terminal message:** "✅ Using QUANTIZED model (Q4_K_M)"

## How to Test

1. Restart Streamlit (if not already done)
2. Type any question from above
3. Check terminal for model loading message
4. Note the response time in the performance metrics

---

**Note:** These questions are designed to be:
- Simple and clear
- Common medical scenarios
- Not covered by hardcoded quick prompts
- Suitable for testing model performance

