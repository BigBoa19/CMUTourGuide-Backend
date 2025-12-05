from fastapi import APIRouter
import httpx
from pydantic import BaseModel
from typing import Optional

class VisionRequest(BaseModel):
	imageBase64: str

class VisionResponse(BaseModel):
	building_name: str
	confidence: float
	error: Optional[str] = None


router = APIRouter(prefix="", tags=["image"])

@router.post("/vision", response_model=VisionResponse)
async def image(req: VisionRequest):
	reply = await recognize_building(req.imageBase64)
	return reply

async def recognize_building(image_base64: str) -> VisionResponse:
	try:
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				"https://ncdev1919--cmu-tour-guide-cv-recognize-building-lp.modal.run",
				json={"imageBase64": image_base64},
				headers={"Content-Type": "application/json"}
			)
			response.raise_for_status()
			res_json =  response.json()
			print(res_json)
			building = res_json.get("building", "Unknown")
			confidence = float(res_json.get("confidence", 0.0))
			return VisionResponse(building_name=building, confidence=confidence, error= None)
	except httpx.TimeoutException:
		return _error_response("Request to Modal API timed out", "TIMEOUT")
	except httpx.HTTPStatusError as e:
		return _error_response(
			f"Modal API returned error: {e.response.status_code}",
			f"HTTP_{e.response.status_code}"
		)
	except Exception as e:
		return _error_response(f"Failed to call Modal API: {str(e)}", str(e))

def _error_response(description: str, error: str) -> VisionResponse:
	return VisionResponse(
		building_name="Error",
		confidence=0.0,
		error=error
	)