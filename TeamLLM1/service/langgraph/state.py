# state.py
from typing import List, Optional
from pydantic import BaseModel

class GraphState(BaseModel):
    client_id: str
    query: str
    retrieved_docs: Optional[List[str]] = None
    answer: Optional[str] = None


class ChatRequest(BaseModel):
    client_id: str
    query: str


class ChatResponse(BaseModel):
    client_id: str
    answer: str