import httpx

async def recognize_building(image_base64: str) -> dict:
	try:
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				"https://ncdev1919--cmu-tour-guide-cv-recognize-building.modal.run",
				json={"imageBase64": image_base64},
				headers={"Content-Type": "application/json"}
			)
			response.raise_for_status()
			return response.json()
	except httpx.TimeoutException:
		return _error_response("Request to Modal API timed out", "TIMEOUT")
	except httpx.HTTPStatusError as e:
		return _error_response(
			f"Modal API returned error: {e.response.status_code}",
			f"HTTP_{e.response.status_code}"
		)
	except Exception as e:
		return _error_response(f"Failed to call Modal API: {str(e)}", str(e))

def _error_response(description: str, error: str) -> dict:
	return {
		"building": "Error",
		"error": error,
		"description": description,
		"confidence": 0.0,
	}