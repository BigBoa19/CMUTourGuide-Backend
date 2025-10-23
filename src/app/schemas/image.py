from pydantic import BaseModel

class ImageRequest(BaseModel):
	message: str
	imageBase64: str

class ImageResponse(BaseModel):
	reply: str
