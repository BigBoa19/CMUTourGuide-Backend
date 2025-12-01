from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from app.routers.health import router as health_router
from app.routers.chat import router as chat_router
from app.routers.vision import router as vision_router

app = FastAPI(title="CMU Tour Guide API", version="0.1.0")

origins: List[str] = [
	"http://localhost:8081",
	# TODO: Add production origins
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(vision_router)


@app.get("/")
async def root():
	return {"service": "tourguide-api", "status": "ok"}
