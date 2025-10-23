from pydantic import BaseModel

class ImageRequest(BaseModel):
	message: str
	imageBase64: str = None

class ImageResponse(BaseModel):
	reply: str
