from fastapi import APIRouter
from app.schemas.image import ImageRequest, ImageResponse
from app.services.ai import identify_image

router = APIRouter(prefix="", tags=["image"])

@router.post("/image", response_model=ImageResponse)
async def image(req: ImageRequest) -> ImageResponse:
	print(req)
	reply = await identify_image(req.imageBase64)
	return ImageResponse(reply=reply)
