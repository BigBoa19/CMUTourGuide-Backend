import httpx
from app.config import settings

async def recognize_building(image_base64: str) -> dict:
	if not settings.modal_api_url:
		return {
			"building": "Error",
			"confidence": 0.0,
			"description": "Modal API URL not configured",
			"error": "MODAL_URL_NOT_SET"
		}
	
	try:
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				settings.modal_api_url,
				json={"imageBase64": image_base64},
				headers={"Content-Type": "application/json"}
			)
			response.raise_for_status()
			return response.json()
	except httpx.TimeoutException:
		return {
			"building": "Error",
			"confidence": 0.0,
			"description": "Request to Modal API timed out",
			"error": "TIMEOUT"
		}
	except httpx.HTTPStatusError as e:
		return {
			"building": "Error",
			"confidence": 0.0,
			"description": f"Modal API returned error: {e.response.status_code}",
			"error": f"HTTP_{e.response.status_code}"
		}
	except Exception as e:
		return {
			"building": "Error",
			"confidence": 0.0,
			"description": f"Failed to call Modal API: {str(e)}",
			"error": str(e)
		}