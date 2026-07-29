from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    page: int
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []

class UploadResponse(BaseModel):
    filename: str
    chunks: int
    message: str