import json
from google import genai
from google.genai import types

client = genai.Client()

def verify_claims(llm_output, environment, evidence):
    model_name = "gemini-3.5-flash-lite"
    
    evidence_text = ""
    for i, ev in enumerate(evidence):
        evidence_text += f"\n[Source {i+1}]: {ev.get('document_title', 'Unknown')} - {ev.get('text', '')}\n"
        
    prompt = f"""You are a strict, objective AI Verification Node checking a biodiversity recommendation.
Your ONLY job is to verify the claims against the provided evidence and environmental state.

ENVIRONMENTAL STATE:
{json.dumps(environment, indent=2)}

SCIENTIFIC EVIDENCE:
{evidence_text}

RECOMMENDATION TO VERIFY:
{json.dumps(llm_output, indent=2)}

INSTRUCTIONS:
1. Verify if the numbers, metrics, and claims in the recommendation are actually supported by the Scientific Evidence and Environmental State.
2. DO NOT output your internal thinking or chain-of-thought in the final JSON response.
3. If perfectly supported, set 'is_supported' to True and leave 'flags' empty.
4. If there is a contradiction or hallucination, set 'is_supported' to False.
5. In 'flags', provide ONLY 1 or 2 extremely brief, user-friendly bullet points explaining what was corrected (e.g., "Corrected the rainfall statistic to match local data." or "Removed unverified claims about species growth."). Do not dump raw analysis.
6. If it fails verification, output a 'revised_answer' fixing the errors.
"""

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "is_supported": {"type": "BOOLEAN", "description": "True if all claims are accurate and supported. False otherwise."},
            "flags": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "If False, an extremely short, user-friendly summary of what was corrected (max 1 sentence per flag). Empty if True."
            },
            "revised_answer": {"type": "STRING", "description": "The completely revised recommendation text if it failed verification. Null if passed."}
        },
        "required": ["is_supported", "flags"]
    }

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.0
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in Verifier: {e}")
        return {"is_supported": False, "flags": ["Verification node failed to parse."]}

