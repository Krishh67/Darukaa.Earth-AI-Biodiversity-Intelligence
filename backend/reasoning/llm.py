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
    
    prompt = f"""You are the Darukaa.Earth AI Biodiversity Intelligence agent.
Your task is to provide an evidence-backed biodiversity recommendation based on structured environmental data and retrieved scientific evidence.

USER OBJECTIVE:
{objective}

CONVERSATION HISTORY:
{history_text}

STRUCTURED ENVIRONMENTAL STATE:
{json.dumps(environment, indent=2)}

RETRIEVED SCIENTIFIC EVIDENCE:
{evidence_text}

INSTRUCTIONS:
1. Reason across multiple environmental metrics (e.g., how soil pH, SOC, and rainfall interact).
2. Ground your recommendation entirely in the provided Scientific Evidence and Environmental State.
3. Explicitly state if data is unavailable and how it limits the recommendation.
4. Do not invent quantitative improvement values unless directly supported by evidence.
5. Provide a specific, actionable intervention.
"""

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "answer": {"type": "STRING", "description": "A conversational opening addressing the user's objective"},
            "recommendation": {"type": "STRING", "description": "Specific, actionable intervention"},
            "scientific_reasoning": {"type": "STRING", "description": "Explanation connecting environmental state + evidence"},
            "impacted_metrics": {
                "type": "ARRAY", 
                "items": {"type": "STRING"},
                "description": "List of metrics that will be impacted, e.g., 'Soil Organic Carbon'"
            },
            "time_horizon": {"type": "STRING", "description": "Short, medium, or long term"},
            "confidence": {"type": "STRING", "description": "High/Medium/Low with explanation"},
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

