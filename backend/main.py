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

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    session_id = memory.get_or_create(req.conversation_id)
    history = memory.get_history(session_id)
    
    # 1. Gather Environmental Data if coordinates provided
    environment = {}
    if req.latitude is not None and req.longitude is not None:
        print(f"Fetching environment for {req.latitude}, {req.longitude}")
        soil = get_soil_data(req.latitude, req.longitude)
        if soil:
            environment["soil"] = soil
            
        landcover = get_landcover_data(req.latitude, req.longitude)
        if landcover:
            environment["land_cover"] = landcover
            
        climate = get_climate_data(req.latitude, req.longitude)
        if climate:
            environment["climate"] = climate
            
        environment["location"] = {"latitude": req.latitude, "longitude": req.longitude}
        
    # 2. Retrieve Evidence from RAG
    print(f"Retrieving evidence for query: {req.message}")
    evidence = retriever.retrieve(req.message, top_k=4)
    
    # 3. LLM Reasoning
    print("Generating reasoning...")
    llm_out = generate_reasoning(
        objective=req.message,
        environment=environment,
        evidence=evidence,
        history=history
    )
    
    if not llm_out:
        raise HTTPException(status_code=500, detail="Failed to generate reasoning.")
        
    # 4. Verify Claims
    print("Verifying claims...")
    verification = verify_claims(llm_out, environment, evidence)
    
    # Check if verification failed and revised
    if not verification.get("is_supported", False) and verification.get("revised_answer"):
        llm_out["scientific_reasoning"] = verification["revised_answer"]
        
    # 5. Update Memory
    memory.add_message(session_id, "user", req.message)
    memory.add_message(session_id, "assistant", llm_out.get("recommendation", ""))
    
    # Clean up evidence for frontend (remove raw vectors if they somehow sneaked in)
    clean_evidence = []
    for ev in evidence:
        clean_evidence.append({
            "chunk_id": ev.get("chunk_id", ""),
            "document_title": ev.get("document_title", ""),
            "text": ev.get("text", "")
        })

    return ChatResponse(
        answer=llm_out.get("answer", "Here is what I found."),
        recommendation=llm_out.get("recommendation", ""),
        scientific_reasoning=llm_out.get("scientific_reasoning", ""),
        environment=environment,
        impacted_metrics=llm_out.get("impacted_metrics", []),
        time_horizon=llm_out.get("time_horizon", ""),
        confidence=llm_out.get("confidence", ""),
        evidence=clean_evidence,
        verification=verification,
        conversation_id=session_id
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
