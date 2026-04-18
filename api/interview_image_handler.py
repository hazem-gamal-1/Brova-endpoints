import base64
import os
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_IMAGE_MODEL = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    temperature=0.8,
)


class InterviewImageHandler:
    def __init__(self):
        self.model = DEFAULT_IMAGE_MODEL

    def convert_image_to_text(self, image_path) -> str:
        image=self._encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    },
                },
            ]
        )
        response = self.model.invoke([message])
        return response.content
    

    def _encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            




image_handler = InterviewImageHandler() 
print(image_handler.convert_image_to_text(r"E:\Langchain-projects\Brova\im.png"))    