from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    conversation_id: Optional[str] = None
    env_override: Optional[Dict[str, Any]] = None
    enable_verification: bool = True

class ChatResponse(BaseModel):
    answer: str
    is_clarification: bool = False
    conversation_id: str
    environment: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    scientific_reasoning: Optional[str] = None
    time_horizon: Optional[str] = None
    confidence: Optional[str] = None
    impacted_metrics: Optional[List[str]] = None
    verification: Optional[Dict[str, Any]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
