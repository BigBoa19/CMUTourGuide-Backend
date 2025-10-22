from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
	message: str
	imageBase64: Optional[str] = None

class ChatResponse(BaseModel):
	reply: str
