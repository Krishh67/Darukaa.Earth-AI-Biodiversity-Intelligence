from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
import os
from dotenv import load_dotenv

load_dotenv(override=True)

from backend.models.schemas import ChatRequest, ChatResponse
from backend.rag.retriever import RAGRetriever
from backend.environmental.soil import get_soil_data
from backend.environmental.climate import get_climate_data
from backend.environmental.landcover import get_landcover_data
from backend.reasoning.llm import generate_reasoning
from backend.reasoning.verifier import verify_claims
from backend.memory.conversation import memory

app = FastAPI(title="Darukaa.Earth AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG retriever on startup (loads FAISS, BM25 into memory)
retriever = None

@app.on_event("startup")
def startup_event():
    global retriever
    print("Initializing RAG Retriever...")
    retriever = RAGRetriever(data_dir="data")
    print("RAG initialized.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/sessions")
def get_sessions():
    return memory.list_sessions()

@app.get("/sessions/{session_id}")
def get_session_history(session_id: str):
    history = memory.get_history(session_id)
    return {"id": session_id, "history": history}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    session_id = memory.get_or_create(req.conversation_id)
    history = memory.get_history(session_id)
    is_initialized = memory.is_initialized(session_id)
    
    # 0. Conversation Intake Manager (Grok)
    queries = [req.message]
    if not is_initialized:
        print("Running Grok Intake Manager...")
        from backend.reasoning.intake import run_intake
        has_loc = (req.latitude is not None and req.longitude is not None)
        intake_res = run_intake(req.message, history, has_loc)
        
        if intake_res.get("status") == "CLARIFY":
            memory.add_message(session_id, "user", req.message)
            memory.add_message(session_id, "assistant", intake_res.get("question", "Could you provide more details?"))
            return ChatResponse(
                answer=intake_res.get("question", "Could you provide more details?"),
                is_clarification=True,
                conversation_id=session_id
            )
        else:
            # READY
            queries = intake_res.get("queries", [req.message])
            if not queries:
                queries = [req.message]
            memory.set_initialized(session_id)
            print(f"Grok READY. Generated queries: {queries}")
    else:
        print("Conversation already initialized. Skipping Grok Intake.")
    
    # 1. Gather Environmental Data if coordinates provided
    environment = {}
    if req.latitude is not None and req.longitude is not None:
        print(f"Fetching environment for {req.latitude}, {req.longitude}")
        
        # Call API scripts directly
        try:
            from api.soilgrids_data import get_soil_data
            soil_result = get_soil_data(req.latitude, req.longitude)
            if soil_result and soil_result.get("soil"):
                s = soil_result["soil"]
                environment["soil"] = {
                    "ph": s.get("ph"),
                    "soc": s.get("soc_g_kg"),
                    "nitrogen": s.get("nitrogen_g_kg"),
                    "bulk_density": s.get("bulk_density_kg_dm3")
                }
        except Exception as e:
            print(f"Soil API failed: {e}")
            
        try:
            from api.planetary_computer_environment import get_landcover_data
            landcover = get_landcover_data(req.latitude, req.longitude)
            if landcover:
                environment["land_cover"] = landcover
        except Exception as e:
            print(f"Landcover API failed: {e}")
            
        try:
            from backend.environmental.climate import get_climate_data
            climate = get_climate_data(req.latitude, req.longitude)
            if climate:
                environment["climate"] = climate
        except Exception as e:
            print(f"Climate API failed: {e}")
            
        environment["location"] = {"latitude": req.latitude, "longitude": req.longitude}
        
    # Apply structural JSON override if provided
    if req.env_override:
        print("Applying JSON Environmental Override...")
        # Recursively update the dictionary or just blind merge at top level
        # For simplicity, we'll top-level merge
        for k, v in req.env_override.items():
            if isinstance(v, dict) and k in environment and isinstance(environment[k], dict):
                environment[k].update(v)
            else:
                environment[k] = v

    # 2. Retrieve Evidence from RAG using multiple queries
    print(f"Retrieving evidence using queries: {queries}")
    evidence = []
    seen_chunks = set()
    
    for q in queries:
        docs = retriever.retrieve(q, top_k=3)
        for doc in docs:
            if doc["chunk_id"] not in seen_chunks:
                seen_chunks.add(doc["chunk_id"])
                evidence.append(doc)
    
    # 3. LLM Reasoning
    print("Generating reasoning...")
    llm_out = generate_reasoning(
        objective=req.message,
        environment=environment,
        evidence=evidence[:6], # Cap at 6 evidence chunks to avoid context bloat
        history=history
    )
    
    if not llm_out:
        raise HTTPException(status_code=500, detail="Failed to generate reasoning.")
        
    # 4. Verify Claims (only if enabled)
    verification = {"is_supported": True, "flags": [], "disabled": not req.enable_verification}
    if req.enable_verification:
        print("Verifying claims...")
        verification = verify_claims(llm_out, environment, evidence[:6])
        # Check if verification failed and revised
        if not verification.get("is_supported", False) and verification.get("revised_answer"):
            llm_out["scientific_reasoning"] = verification["revised_answer"]
    else:
        print("Verification node is DISABLED.")
        
    # 5. Update Memory
    memory.add_message(session_id, "user", req.message)
    memory.add_message(session_id, "assistant", llm_out["recommendation"])
    
    # Clean up evidence for frontend (remove raw vectors if they somehow sneaked in)
    clean_evidence = []
    for ev in evidence[:6]:
        clean_evidence.append({
            "chunk_id": ev.get("chunk_id", ""),
            "document_title": ev.get("document_title", ""),
            "text": ev.get("text", "")
        })

    return ChatResponse(
        answer=llm_out.get("answer", "Here is what I found."),
        is_clarification=False,
        conversation_id=session_id,
        environment=environment,
        recommendation=llm_out.get("recommendation", ""),
        scientific_reasoning=llm_out.get("scientific_reasoning", ""),
        time_horizon=llm_out.get("time_horizon", ""),
        confidence=llm_out.get("confidence", ""),
        impacted_metrics=llm_out.get("impacted_metrics", []),
        verification=verification,
        evidence=clean_evidence
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
