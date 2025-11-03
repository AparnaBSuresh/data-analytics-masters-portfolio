"""
Performance Testing Script for MediAssist AI
Automatically collects QoS metrics for documentation
"""

import time
import json
import statistics
from datetime import datetime
from src.llm.backends import call_llm, get_last_timing_info
from src.config.constants import DEFAULT_SYSTEM_PROMPT, MEDICAL_DEPARTMENTS

# Test queries for different scenarios
TEST_QUERIES = {
    "Short": [
        "What is diabetes?",
        "Define hypertension",
        "What causes fever?"
    ],
    "Medium": [
        "What are the common symptoms of diabetes and how is it diagnosed?",
        "How can I manage high blood pressure at home?",
        "What are the risk factors for heart disease?"
    ],
    "Long": [
        "I have been experiencing persistent headaches for the past week, especially in the morning. The pain is throbbing and sometimes accompanied by nausea. What could be causing this and what should I do?",
        "My elderly parent has been having trouble with balance and memory. They seem confused at times and have fallen twice in the past month. What conditions could cause these symptoms?",
        "I'm worried about my cardiovascular health. I have a family history of heart disease, I'm overweight, and I don't exercise much. What are the warning signs I should watch for?"
    ],
    "Emergency": [
        "I'm having severe chest pain and difficulty breathing",
        "Someone is unconscious and not breathing",
        "Severe allergic reaction with swelling"
    ]
}

CONSISTENCY_TEST_QUERY = "What are the common symptoms of diabetes?"
CONSISTENCY_TRIALS = 5

def test_performance():
    """Test response time performance for different query lengths."""
    print("=" * 80)
    print("PERFORMANCE TESTING")
    print("=" * 80)
    
    all_results = []
    
    for category, queries in TEST_QUERIES.items():
        print(f"\n### Testing {category} Queries ###\n")
        
        for i, query in enumerate(queries, 1):
            print(f"Test Case: {category}-{i}")
            print(f"Query: {query[:70]}...")
            
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
            
            # For emergency queries, measure detection time
            if category == "Emergency":
                from src.utils.safety_utils import detect_emergency
                start = time.time()
                is_emergency, keywords = detect_emergency(query)
                end = time.time()
                
                result = {
                    "test_id": f"TC-P-{category}-{i}",
                    "category": category,
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "query_length_words": len(query.split()),
                    "response_time": end - start,
                    "emergency_detected": is_emergency,
                    "keywords": keywords
                }
                
                print(f"Emergency Detected: {is_emergency}")
                print(f"Detection Time: {result['response_time']:.3f}s")
                print(f"Keywords: {keywords}")
                
            else:
                # Regular LLM call
                start = time.time()
                try:
                    response = call_llm(
                        messages,
                        provider="Fine-tuned Models",
                        model_name="MedLlama-GGUF",
                        temperature=0.2,
                        max_new_tokens=1024,
                        department="Internal Medicine"
                    )
                    end = time.time()
                    
                    # Get timing info from GGUF backend
                    timing_info = get_last_timing_info()
                    
                    result = {
                        "test_id": f"TC-P-{category}-{i}",
                        "category": category,
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "query_length_words": len(query.split()),
                        "response_time": end - start,
                        "response_length_words": len(response.split()),
                        "response_length_chars": len(response),
                    }
                    
                    if timing_info:
                        result.update({
                            "generation_time": timing_info.get("generation_time", 0),
                            "tokens_generated": timing_info.get("tokens_generated", 0),
                            "tokens_per_sec": timing_info.get("tokens_per_sec", 0)
                        })
                    
                    print(f"Response Time: {result['response_time']:.2f}s")
                    print(f"Response Length: {result['response_length_words']} words")
                    if timing_info:
                        print(f"Generation Speed: {result.get('tokens_per_sec', 0):.1f} tok/s")
                    
                except Exception as e:
                    result = {
                        "test_id": f"TC-P-{category}-{i}",
                        "category": category,
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "error": str(e)
                    }
                    print(f"ERROR: {e}")
            
            all_results.append(result)
            print("-" * 80)
            time.sleep(2)  # Brief pause between tests
    
    # Calculate summary statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    non_emergency_results = [r for r in all_results if r.get("category") != "Emergency" and "error" not in r]
    emergency_results = [r for r in all_results if r.get("category") == "Emergency"]
    
    if non_emergency_results:
        avg_response_time = statistics.mean([r["response_time"] for r in non_emergency_results])
        avg_words = statistics.mean([r["response_length_words"] for r in non_emergency_results])
        
        print(f"\nNon-Emergency Queries:")
        print(f"  Average Response Time: {avg_response_time:.2f}s")
        print(f"  Average Response Length: {avg_words:.0f} words")
        
        if any("tokens_per_sec" in r for r in non_emergency_results):
            avg_speed = statistics.mean([r.get("tokens_per_sec", 0) for r in non_emergency_results if "tokens_per_sec" in r])
            print(f"  Average Generation Speed: {avg_speed:.1f} tok/s")
    
    if emergency_results:
        avg_emergency_time = statistics.mean([r["response_time"] for r in emergency_results])
        print(f"\nEmergency Detection:")
        print(f"  Average Detection Time: {avg_emergency_time:.4f}s")
        print(f"  All Emergencies Detected: {all(r.get('emergency_detected', False) for r in emergency_results)}")
    
    # Save results to file
    output_file = f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "results": all_results,
            "summary": {
                "total_tests": len(all_results),
                "non_emergency_avg_time": avg_response_time if non_emergency_results else None,
                "emergency_avg_time": avg_emergency_time if emergency_results else None
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return all_results


def test_consistency():
    """Test response consistency for the same query multiple times."""
    print("\n" + "=" * 80)
    print("CONSISTENCY TESTING")
    print("=" * 80)
    print(f"\nQuery: {CONSISTENCY_TEST_QUERY}")
    print(f"Trials: {CONSISTENCY_TRIALS}")
    print(f"Temperature: 0.2\n")
    
    results = []
    responses = []
    
    for trial in range(1, CONSISTENCY_TRIALS + 1):
        print(f"Trial {trial}/{CONSISTENCY_TRIALS}...")
        
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": CONSISTENCY_TEST_QUERY}
        ]
        
        try:
            response = call_llm(
                messages,
                provider="Fine-tuned Models",
                model_name="MedLlama-GGUF",
                temperature=0.2,
                max_new_tokens=1024,
                department="Internal Medicine"
            )
            
            word_count = len(response.split())
            char_count = len(response)
            
            # Check for required sections
            has_sections = {
                "Understanding": "understanding" in response.lower() or "concern" in response.lower(),
                "Causes": "causes" in response.lower() or "possible causes" in response.lower(),
                "Self-Care": "self-care" in response.lower() or "home remedies" in response.lower(),
                "Warning": "warning" in response.lower() or "red flag" in response.lower(),
                "Doctor": "doctor" in response.lower() or "medical" in response.lower(),
                "Disclaimer": "disclaimer" in response.lower() or "not medical advice" in response.lower()
            }
            
            result = {
                "trial": trial,
                "word_count": word_count,
                "char_count": char_count,
                "sections_found": sum(has_sections.values()),
                "sections_detail": has_sections
            }
            
            results.append(result)
            responses.append(response)
            
            print(f"  Words: {word_count}")
            print(f"  Sections Found: {result['sections_found']}/6")
            
        except Exception as e:
            print(f"  ERROR: {e}")
        
        time.sleep(2)
    
    # Calculate consistency metrics
    print("\n" + "-" * 80)
    print("CONSISTENCY RESULTS")
    print("-" * 80)
    
    if results:
        word_counts = [r["word_count"] for r in results]
        section_counts = [r["sections_found"] for r in results]
        
        avg_words = statistics.mean(word_counts)
        std_words = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        avg_sections = statistics.mean(section_counts)
        
        print(f"\nResponse Length:")
        print(f"  Average: {avg_words:.0f} words")
        print(f"  Std Dev: {std_words:.0f} words")
        print(f"  Range: {min(word_counts)} - {max(word_counts)} words")
        print(f"  Variance: ±{(std_words/avg_words*100):.1f}%")
        
        print(f"\nStructure Consistency:")
        print(f"  Average Sections Found: {avg_sections:.1f}/6")
        print(f"  Complete Structure Rate: {sum(1 for s in section_counts if s == 6)/len(section_counts)*100:.0f}%")
        
        # Consistency score (lower variance = higher score)
        consistency_score = 100 - (std_words / avg_words * 100)
        print(f"\nOverall Consistency Score: {consistency_score:.1f}%")
        
        if consistency_score >= 90:
            print("  Status: ✅ EXCELLENT")
        elif consistency_score >= 80:
            print("  Status: ✅ GOOD")
        else:
            print("  Status: ⚠️ NEEDS IMPROVEMENT")
    
    # Save results
    output_file = f"consistency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "query": CONSISTENCY_TEST_QUERY,
            "trials": CONSISTENCY_TRIALS,
            "results": results,
            "summary": {
                "avg_word_count": avg_words if results else 0,
                "word_variance": std_words if results else 0,
                "consistency_score": consistency_score if results else 0
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return results


def test_departments():
    """Test all medical departments."""
    print("\n" + "=" * 80)
    print("DEPARTMENT-SPECIFIC TESTING")
    print("=" * 80)
    
    test_query = "What are common conditions in this department?"
    results = []
    
    for dept_name, dept_info in MEDICAL_DEPARTMENTS.items():
        print(f"\nTesting Department: {dept_name}")
        print(f"Specialization: {dept_info.get('specialization', 'N/A')}")
        
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": test_query}
        ]
        
        try:
            start = time.time()
            response = call_llm(
                messages,
                provider="Fine-tuned Models",
                model_name="MedLlama-GGUF",
                temperature=0.2,
                max_new_tokens=512,
                department=dept_name
            )
            end = time.time()
            
            # Check if department context is reflected in response
            dept_keywords = dept_name.lower() in response.lower()
            
            result = {
                "department": dept_name,
                "response_time": end - start,
                "response_length": len(response.split()),
                "department_context_present": dept_keywords,
                "success": True
            }
            
            print(f"  Response Time: {result['response_time']:.2f}s")
            print(f"  Context Applied: {'✅ Yes' if dept_keywords else '⚠️ Not detected'}")
            
        except Exception as e:
            result = {
                "department": dept_name,
                "error": str(e),
                "success": False
            }
            print(f"  ERROR: {e}")
        
        results.append(result)
        time.sleep(1)
    
    # Summary
    successful = sum(1 for r in results if r.get("success", False))
    print(f"\n{'=' * 80}")
    print(f"Department Testing Complete: {successful}/{len(results)} successful")
    print(f"Success Rate: {successful/len(results)*100:.0f}%")
    
    output_file = f"department_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "total_departments": len(results),
                "successful": successful,
                "success_rate": successful/len(results)*100
            }
        }, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}\n")
    
    return results


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" MEDIASSIST AI - COMPREHENSIVE SYSTEM TESTING")
    print("=" * 80)
    print(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: MedLlama-3b (GGUF Q4_K_M)")
    print("\n")
    
    try:
        # Run performance tests
        print("\n[1/3] Running Performance Tests...")
        perf_results = test_performance()
        
        # Run consistency tests
        print("\n[2/3] Running Consistency Tests...")
        consistency_results = test_consistency()
        
        # Run department tests
        print("\n[3/3] Running Department Tests...")
        dept_results = test_departments()
        
        print("\n" + "=" * 80)
        print(" ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nCheck the generated JSON files for detailed results.")
        print("Use these results to populate your testing documentation.\n")
        
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

