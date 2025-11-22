from openai import OpenAI, AsyncOpenAI
from app.config import settings
from app.schemas.chat import Message
from app.services.vision import recognize_building
import os
import requests

async def identify_image(message: str, base64_image: str) -> str:
	print(message)
	try:
		# First, try to recognize the building using Modal
		building_info = None
		if base64_image:
			building_result = await recognize_building(base64_image)
			if building_result.get("building") and building_result.get("building") != "Unknown" and building_result.get("building") != "Error":
				building_info = building_result
				print(f"🏢 Recognized building: {building_info.get('building')} (confidence: {building_info.get('confidence', 0)})")

		# Build system prompt with building info
		system_prompt = """You are a CMU Tour Guide AI assistant.
		Your task is to help visitors navigate Carnegie Mellon University by analyzing images they take and providing helpful information about campus locations, buildings, landmarks, and directions. 
		Provide informative and friendly responses about CMU campus. Always format your responses in markdown text for better readability.
		When given the building/monument the building has been identified as, return a short summary of the building."""
		
		if building_info:
			system_prompt += f"\n\nIMPORTANT: The image has been identified as **{building_info.get('building')}** (confidence: {building_info.get('confidence', 0):.2%})."
			if building_info.get('description'):
				system_prompt += f" Building description: {building_info.get('description')}"
			system_prompt += " Use this information to provide accurate and specific information about this building."

		api_key = os.getenv("OPENROUTER_API_KEY")
		url = "https://openrouter.ai/api/v1/chat/completions"
		headers = {
			"Authorization": f"Bearer {api_key}",
			"Content-Type": "application/json"
		}

		messages = [
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": system_prompt
					}
				]
			}
		]
		payload = {
			"model": "google/gemini-2.0-flash-001",
			"messages": messages
		}
		response = requests.post(url, headers=headers, json=payload)
		print(response.json())
	except Exception as e:
		print(f"Error in identify_image_async: {type(e).__name__}: {str(e)}")
		raise

def generate_reply(messages: list[Message]) -> str:
	api_key = os.getenv("OPENROUTER_API_KEY")
	if not api_key:
		raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please configure it in Railway.")
	
	client = OpenAI(
		base_url="https://openrouter.ai/api/v1",
		api_key=api_key
	)

	system_prompt = """You are a CMU Tour Guide AI assistant.
 	Your task is to help visitors navigate Carnegie Mellon University by analyzing images they take and providing helpful information about campus locations, buildings, landmarks, and directions. 
	When users ask questions or share images, provide informative and friendly responses about CMU campus. Always format your responses in markdown text for better readability."""

	chatHistory = [{ "role": "system", "content": system_prompt }]

	for message in messages:
		if message.isUser:
			chatHistory.append({ "role": "user", "content": message.text})
		else:
			chatHistory.append({ "role": "assistant", "content": message.text})


	response = client.chat.completions.create(
		model="openai/gpt-4o",
		messages=chatHistory
	)

	print("Reply generated:", response.choices[0].message.content)
	return response.choices[0].message.content
