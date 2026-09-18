from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    recommendation: str
    scientific_reasoning: str
    environment: Dict[str, Any]
    impacted_metrics: List[str]
    time_horizon: str
    confidence: str
    evidence: List[Dict[str, Any]]
    verification: Dict[str, Any]
    conversation_id: str

