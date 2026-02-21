from typing import Dict, List
from openai import AsyncOpenAI

class OpenAIClient:
    def __init__(self, api_key: str = None):
        self.client = AsyncOpenAI(
            api_key=api_key
        )
    
    async def get_response(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.responses.create(
            model=model,
            input=messages,
            **kwargs
        )
        return response