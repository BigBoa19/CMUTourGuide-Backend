from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.ai import generate_reply

router = APIRouter(prefix="", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
	print(req)
	reply = generate_reply(req.messages)
	return ChatResponse(reply=reply)
