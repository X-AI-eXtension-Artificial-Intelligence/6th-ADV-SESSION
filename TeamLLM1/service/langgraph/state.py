# state.py
from typing import List, Optional, Tuple
from pydantic import BaseModel

class GraphState(BaseModel):
    client_id: str
    query: str
    retrieved_docs: Optional[List[str]] = None
    answer: Optional[str] = None
    ce_inputs: Optional[List[List[str]]] = None      # [[query, passage], ...]
    candidates: Optional[List[Tuple[str, str]]] = None  # [(title, text), ...]


class ChatRequest(BaseModel):
    client_id: str
    query: str


class ChatResponse(BaseModel):
    client_id: str
    answer: str