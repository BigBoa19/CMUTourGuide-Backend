from openai import OpenAI
from app.config import settings

def generate_reply(message: str, base64_image: str) -> str:
	print("Generating reply...")
	client = OpenAI(api_key=settings.openai_api_key)

	if not base64_image or len(base64_image) == 0:
		return "Sorry, the image was invalid"


	response = client.responses.create(
		model="gpt-4.1",
		input=[
			{
				"role": "user",
				"content": [
					{ "type": "input_text", "text": message },
					{
						"type": "input_image",
						"image_url": f"data:image/jpeg;base64,{base64_image}",
					},
            	],
			}
		]
	)

	print("Reply generated:", response.output_text)
	return response.output_text