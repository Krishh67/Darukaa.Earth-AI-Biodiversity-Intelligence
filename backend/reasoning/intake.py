import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

def run_intake(query: str, history: list, has_location: bool):
    """
    Use Groq to determine if we need more context or if we are ready.
    Returns:
        dict: {"status": "CLARIFY", "question": "..."} 
              OR 
              {"status": "READY", "queries": ["query1", "query2"]}
    """
    
    # Format history
    history_text = ""
    for msg in history:
        history_text += f"{msg['role'].upper()}: {msg['content']}\n"
        
    location_msg = "Location is handled via the UI dropdown. Do not ask about coordinates or basic region geography." if has_location else "Location is handled via the UI dropdown."

    system_prompt = f"""You are an expert AI Environmental Scientist and Biodiversity Strategist.
Your goal is to evaluate the user's query and decide if you have enough context to generate a scientifically grounded, multi-metric environmental recommendation.

RULES:
1. {location_msg} Do NOT ask about location.
2. We automatically fetch soil data (pH, SOC, Nitrogen, Bulk Density), Climate (Temp, Rainfall), and Land Cover. Do NOT ask for these.
3. If the user's goal is too vague (e.g., "how do I fix my land?") or lacks specific operational context (e.g., current crops, farming practices, budget, specific goals), ask a SINGLE, deep, scientific clarification question to narrow down the problem.
4. If the user's query has enough operational context to proceed with a rigorous scientific literature search, return status "READY".
5. When READY, generate 2-4 optimized search queries for a RAG database to find relevant ecological papers and restoration models.
6. You MUST reply ONLY with a valid JSON object matching exactly one of these two structures:

Clarification needed:
{{
  "status": "CLARIFY",
  "question": "Your single, professional clarification question here"
}}

Ready to proceed:
{{
  "status": "READY",
  "queries": ["search query 1", "search query 2"]
}}"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    if history_text:
        messages.append({"role": "user", "content": f"Previous conversation:\n{history_text}"})
        
    messages.append({"role": "user", "content": f"New Query: {query}"})

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b", 
            messages=messages,
            response_format={"type": "json_object"},
            extra_body={
                "reasoning_effort": "low"
            },
            temperature=0.1
        )
        
        response_text = completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        print(f"Intake failed, defaulting to READY: {e}")
        # Default fallback so the app doesn't break
        return {"status": "READY", "queries": [query]}
