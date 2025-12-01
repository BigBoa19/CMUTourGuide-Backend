import os
import requests


# async def identify_image(base64_image: str) -> str:
# 	try:
# 		# First, try to recognize the building using Modal
# 		building_info = None
# 		if base64_image:
# 			building_result = await recognize_building(base64_image)
# 			if building_result.get("building") and building_result.get("building") != "Unknown" and building_result.get("building") != "Error":
# 				building_info = building_result
# 				print(f"🏢 Recognized building: {building_info.get('building')} (confidence: {building_info.get('confidence', 0)})")

# 		# Build system prompt with building info
# 		system_prompt = """You are a CMU Tour Guide AI assistant.
# 		Your task is to help visitors navigate Carnegie Mellon University by analyzing images they take and providing helpful information about campus locations, buildings, and landmarks. 
# 		Provide informative and friendly responses about CMU campus.
# 		When given the building/monument the building has been identified as, return a short summary of the building, keep your responses short and concise.
# 		Format your responses in markdown text for better readability."""
		
# 		if building_info:
# 			system_prompt += f"\n\nIMPORTANT: The image has been identified as **{building_info.get('building')}** (confidence: {building_info.get('confidence', 0):.2%})."
# 			if building_info.get('description'):
# 				system_prompt += f" Building description: {building_info.get('description')}"
# 			system_prompt += " Use this information to provide accurate and specific information about this building."

# 		api_key = os.getenv("OPENROUTER_API_KEY")
# 		url = "https://openrouter.ai/api/v1/chat/completions"
# 		headers = {
# 			"Authorization": f"Bearer {api_key}",
# 			"Content-Type": "application/json"
# 		}
# 		print(system_prompt)
# 		messages = [
# 			{
# 				"role": "user",
# 				"content": [
# 					{
# 						"type": "text",
# 						"text": system_prompt
# 					}
# 				]
# 			}
# 		]
# 		payload = {
# 			"model": "google/gemini-2.0-flash-001",
# 			"messages": messages
# 		}
# 		response = requests.post(url, headers=headers, json=payload)
# 		print(response.json())
# 		return response.json()["choices"][0]["message"]["content"]
# 	except Exception as e:
# 		print(f"Error in identify_image_async: {type(e).__name__}: {str(e)}")
# 		raise
