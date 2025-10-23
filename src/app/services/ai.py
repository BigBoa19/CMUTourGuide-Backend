from openai import OpenAI
from app.config import settings

def generate_reply(message: str, base64_image: str) -> str:
	print("Generating reply...")
	client = OpenAI(api_key=settings.openai_api_key)

	system_prompt = """You are a CMU Tour Guide AI assistant.
 	Your task is to help visitors navigate Carnegie Mellon University by analyzing images they take and providing helpful information about campus locations, buildings, landmarks, and directions. 
	When users ask questions or share images, provide informative and friendly responses about CMU campus. Always format your responses in markdown text for better readability."""

	content = [{ "type": "input_text", "text": message }]
	if base64_image:
		content.append({
			"type": "input_image",
			"image_url": f"data:image/jpeg;base64,{base64_image}",
		})
	
	response = client.responses.create(
		model="gpt-4.1",
		input=[
			{
				"role": "system",
				"content": system_prompt
			},
			{
				"role": "user",
				"content": content,
			}
		]
	)

	print("Reply generated:", response.output_text)
	return response.output_text