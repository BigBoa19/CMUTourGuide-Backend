from openai import OpenAI, AsyncOpenAI
from app.config import settings
from app.schemas.chat import Message
from app.services.vision import recognize_building
import os

async def identify_image_async(message: str, base64_image: str) -> str:
	"""Async version of identify_image that uses building recognition."""
	client = AsyncOpenAI(
		base_url="https://openrouter.ai/api/v1",
		api_key=os.getenv("OPENROUTER_API_KEY")
	)

	# First, try to recognize the building using Modal
	building_info = None
	if base64_image:
		building_result = await recognize_building(base64_image)
		if building_result.get("building") and building_result.get("building") != "Unknown" and building_result.get("building") != "Error":
			building_info = building_result
			print(f"🏢 Recognized building: {building_info.get('building')} (confidence: {building_info.get('confidence', 0)})")

	# Build enhanced system prompt with building info
	system_prompt = """You are a CMU Tour Guide AI assistant.
 	Your task is to help visitors navigate Carnegie Mellon University by analyzing images they take and providing helpful information about campus locations, buildings, landmarks, and directions. 
	When users ask questions or share images, provide informative and friendly responses about CMU campus. Always format your responses in markdown text for better readability."""
	
	if building_info:
		system_prompt += f"\n\nIMPORTANT: The image has been identified as **{building_info.get('building')}** (confidence: {building_info.get('confidence', 0):.2%})."
		if building_info.get('description'):
			system_prompt += f" Building description: {building_info.get('description')}"
		system_prompt += " Use this information to provide accurate and specific information about this building."

	# Build content array for user message
	user_content = [{"type": "text", "text": message}]
	if base64_image:
		user_content.append({
			"type": "image_url",
			"image_url": {
				"url": f"data:image/jpeg;base64,{base64_image}"
			}
		})
	
	try:
		response = await client.chat.completions.create(
			model="openai/gpt-4o",
			messages=[
				{
					"role": "system",
					"content": system_prompt
				},
				{
					"role": "user",
					"content": user_content,
				}
			]
		)

		print("Reply generated:", response.choices[0].message.content)
		return response.choices[0].message.content
	except Exception as e:
		print(f"Error in identify_image_async: {type(e).__name__}: {str(e)}")
		raise

def generate_reply(messages: list[Message]) -> str:
	client = OpenAI(
		base_url="https://openrouter.ai/api/v1",
		api_key=os.getenv("OPENROUTER_API_KEY")
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
