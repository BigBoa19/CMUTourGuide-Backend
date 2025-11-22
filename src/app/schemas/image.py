from pydantic import BaseModel

class ImageRequest(BaseModel):
	imageBase64: str

class ImageResponse(BaseModel):
	reply: str
