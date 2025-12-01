from fastapi import APIRouter
import requests
import os
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

router = APIRouter(prefix="", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
	print(req)
	reply = generate_reply(req.messages)
	return ChatResponse(reply=reply)

def generate_reply(messages: list[Message]) -> str:
	system_prompt = """You are a CMU Tour Guide AI assistant.
 	Your task is to help visitors navigate Carnegie Mellon University by answering questions and providing helpful information about campus locations, buildings, landmarks, and directions. 
	Provide informative and friendly responses about CMU campus. Format your responses in markdown text for better readability.
	Keep your responses short and concise."""

	chatHistory = [{ "role": "system", "content": system_prompt }]

	for message in messages:
		if message.isUser:
			chatHistory.append({ "role": "user", "content": message.text})
		else:
			chatHistory.append({ "role": "assistant", "content": message.text})

	payload = {
		"model": "openai/gpt-4o",
		"messages": chatHistory
	}
	url = "https://openrouter.ai/api/v1/chat/completions"
	api_key = os.getenv("OPENROUTER_API_KEY")
	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json"
	}
	response = requests.post(url, headers=headers, json=payload)
	print(response.json())
	return response.json()["choices"][0]["message"]["content"]
