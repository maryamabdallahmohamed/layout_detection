from typing import Literal
from pydantic import BaseModel, Field
from ollama import Client
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from src.config import Config

class PageInformation(BaseModel):
    """Information about the page layout."""
    page_orientation: Literal["single_column", "double_column"] = Field(
        description="Whether the page layout is formatted in a single column or double columns."
    )

class LayoutClassifier:
    def __init__(self):
        self.client = Client(
            host=Config.OLLAMA_HOST,
            headers={'Authorization': 'Bearer ' + Config.OLLAMA_API_KEY} if Config.OLLAMA_API_KEY else None
        )
        self.parser = JsonOutputParser(pydantic_object=PageInformation)
        self.prompt = self._create_prompt()
        self.chain = self._create_chain()

    def _create_prompt(self):
        return PromptTemplate(
            template="""
            Look at the document page image provided and determine the page layout.
            
            Analyze the image and classify whether the page has:
            - "single_column": Text/content is arranged in a regular line by line layout.
            - "double_column": Text/content is arranged in two columns (side by side).
            
            Respond with a JSON object matching this exact format:
            {format_instructions}
            """,
            input_variables=[],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def _call_ollama(self, input_dict: dict):
        prompt_text = self.prompt.format_prompt().to_string()
        messages = [{
            "role": "user",
            "content": prompt_text,
            "images": [input_dict["image"]]
        }]
        
        response = self.client.chat(Config.OLLAMA_MODEL, messages=messages, stream=False)
        return response.message.content

    def _create_chain(self):
        model_runnable = RunnableLambda(self._call_ollama)
        parser_runnable = RunnableLambda(lambda x: self.parser.parse(x))
        return model_runnable | parser_runnable

    def classify(self, image_base64: str) -> dict:
        return self.chain.invoke({"image": image_base64})