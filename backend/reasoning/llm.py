import os
import json
from google import genai
from google.genai import types

client = genai.Client()

def generate_reasoning(objective, environment, evidence, history):
    model_name = "gemini-3.5-flash-lite"
    
    # Format evidence
    evidence_text = ""
    for i, ev in enumerate(evidence):
        evidence_text += f"\n[Source {i+1}]: {ev.get('document_title', 'Unknown')} - {ev.get('text', '')}\n"
        
    history_text = json.dumps(history, indent=2)
    
    prompt = f"""You are the Darukaa.Earth AI Environmental Scientist and Biodiversity Strategist.
Your goal is to provide a comprehensive, multi-variable, and scientifically grounded recommendation based on the user's objective, structured environmental data, and retrieved scientific evidence.

USER OBJECTIVE:
{objective}

CONVERSATION HISTORY:
{history_text}

STRUCTURED ENVIRONMENTAL STATE:
{json.dumps(environment, indent=2)}

RETRIEVED SCIENTIFIC EVIDENCE:
{evidence_text}

CRITICAL INSTRUCTIONS:
1. Multi-Metric Reasoning: You MUST explicitly connect at least 3 environmental variables in your analysis (e.g., how Soil Organic Carbon ↔ Rainfall ↔ Land Use interact to affect biodiversity). NO single-variable answers!
2. Evidence-Backed Recommendation: Your recommendation MUST include:
   - What to do (actionable intervention).
   - Why it works (scientific reasoning).
   - Which environmental metrics will improve (and by what estimated magnitude, based on evidence).
   - References to the credible sources from the evidence block (e.g., FAO, IPCC).
3. The 'answer' field should be a rich, comprehensive narrative response (2-3 paragraphs) that behaves like a consultation from an AI scientist, summarizing the diagnosis and the proposed solution.
4. The 'recommendation' field should be a bulleted action plan summarizing the steps.
5. If data is unavailable, explicitly state how it limits the recommendation.
"""

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "answer": {
                "type": "STRING", 
                "description": "A comprehensive, highly detailed response (2-3 paragraphs) acting as an AI environmental scientist. Cross-reference at least 3 environmental variables (like soil, climate, and land cover) to diagnose the issue, explain the scientific mechanism behind the problem, and present your evidence-backed solution comprehensively. Use markdown for readability."
            },
            "recommendation": {
                "type": "STRING", 
                "description": "A focused, actionable step-by-step bulleted plan summarizing 'What to do'. Must be concise but highly actionable."
            },
            "scientific_reasoning": {
                "type": "STRING", 
                "description": "Explain exactly 'Why it works', including quantitative improvement estimates and citing specific references/sources provided in the evidence."
            },
            "impacted_metrics": {
                "type": "ARRAY", 
                "items": {"type": "STRING"},
                "description": "List of variables that will improve, e.g., 'Soil Organic Carbon', 'Microbial Diversity', 'Water Retention'"
            },
            "time_horizon": {"type": "STRING", "description": "e.g., 'Short Term (1-2 years)', 'Medium Term (3-5 years)', or 'Long Term'"},
            "confidence": {"type": "STRING", "description": "Must be 'High', 'Medium', or 'Low' based on evidence quality."},
            "data_limitations": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "required": ["answer", "recommendation", "scientific_reasoning", "impacted_metrics", "time_horizon", "confidence", "data_limitations"]
    }

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in LLM reasoning: {e}")
        return None

