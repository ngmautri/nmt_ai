
import asyncio
import time
from typing import Dict

import pydantic_ai
from nmt_ai.vector_db5 import fetch_data


from pydantic_ai import Agent, RunContext, ModelSettings
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel, Field, ValidationError, field_validator
from nmt_ai.models.vendor_invoices import Invoice

INSTRUCTIONS ="""
Think carefully before answering.
You should use the tool get_vendor_number.
"""




model_settings = ModelSettings(
    temperature=0,  # Giảm sáng tạo, tăng tính logic cấu trúc
    top_p=1,
    top_k=0,
    max_new_tokens=32384,
    max_tokens=32384,
    normalize_tool_arguments=True,
    extra_body={
        "think": False,
    },
    # reasoning_effort="medium"
)

model = OpenAIModel(
    model_name="granite4.2",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# 3. Initialize the Agent
agent = Agent(
    model,
    # capabilities=[Thinking(effort='high')],
    output_type=str,
    # system_prompt=SYSTEM_PROMPT,
    instructions=INSTRUCTIONS,
    model_settings=model_settings,
)

@agent.tool
async def get_vendor_number(ctx: RunContext[str], question: str) -> Dict:
    """
    verify vendor name or VAT. And retrieve related details from the vendor database.

    This tool uses the `fetch_data` function to perform a semantic search against
    the ChromaDB `vendor_collection`. It is designed to be called by the agent
    when vendor identification is required during invoice extraction or validation.

    Args:
        ctx (RunContext[str]): The runtime context provided by the agent framework.
        question (str): A vendor-related query string, such as a name or identifier.

    Returns:
        dict: A dictionary containing vendor details with keys:
            - "vendor_name_sap": The vendor's name from SAP records.
            - "vendor_number": The vendor's unique identifier.
        Returns None if no matching vendor is found.
    """
    return await fetch_data(question)


async def main():
    result = await agent.run(
        "What's vendor number for Microsoft?"
    )

    print(result.output)
    print(result.all_messages())
    # print(result.model_dump_json(indent=2))



if __name__ == "__main__":
    asyncio.run(main())
