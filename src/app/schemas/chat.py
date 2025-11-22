from pydantic import BaseModel

class Message(BaseModel):
	id: str
	text: str
	isUser: bool
	timestamp: str

class ChatRequest(BaseModel):
	messages: list[Message]

class ChatResponse(BaseModel):
	reply: str
