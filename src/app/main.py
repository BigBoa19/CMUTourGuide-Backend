from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.health import router as health_router
from app.routers.chat import router as chat_router
from app.config import settings

app = FastAPI(title="CMU Tour Guide API", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=settings.cors_origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)

@app.get("/")
async def root():
	return {"service": "tourguide-api", "status": "ok"}
