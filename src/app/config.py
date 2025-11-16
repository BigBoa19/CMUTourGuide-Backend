from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List
import os

class Settings(BaseSettings):
	env: str = "dev"
	port: int = 8000
	cors_origins: List[str] = [
		"http://localhost:19006",
		"http://localhost:8081",
	]
	modal_api_url: str = "https://ncdev1919--cmu-tour-guide-cv-recognize-building.modal.run"

	@field_validator("cors_origins", mode="before")
	@classmethod
	def split_cors(cls, v):
		if isinstance(v, str):
			return [s.strip() for s in v.split(",") if s.strip()]
		return v

	class Config:
		env_file = ".env"
		# env_file = os.getenv("ENV_FILE", ".env")

settings = Settings()
