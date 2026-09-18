import json
from google import genai
from google.genai import types

client = genai.Client()

def verify_claims(llm_output, environment, evidence):
    model_name = "gemini-2.5-flash"
    
    evidence_text = ""
    for i, ev in enumerate(evidence):
        evidence_text += f"\n[Source {i+1}]: {ev.get('document_title', 'Unknown')} - {ev.get('text', '')}\n"
        
    prompt = f"""You are the Darukaa.Earth Verification Node.
Your job is to strictly verify the claims made in the LLM's biodiversity recommendation against the actual provided evidence and environmental data.

ENVIRONMENTAL STATE:
{json.dumps(environment, indent=2)}

SCIENTIFIC EVIDENCE:
{evidence_text}

RECOMMENDATION TO VERIFY:
{json.dumps(llm_output, indent=2)}

INSTRUCTIONS:
1. Are the environmental values cited in the reasoning actually present in the Environmental State?
2. Are the scientific claims actually supported by the Scientific Evidence?
3. Are there any fabricated statistics?
4. Is the recommendation consistent with the evidence?

If verification fails, output revised_answer revising the incorrect parts. Otherwise leave it null.
"""

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "is_supported": {"type": "BOOLEAN", "description": "True if all claims are supported"},
            "flags": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "List of unsupported claims, fabricated numbers, or contradictions. Empty if perfectly supported."
            },
            "revised_answer": {"type": "STRING", "description": "A corrected, safe version of the reasoning if it failed verification. Null if it passed."}
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

