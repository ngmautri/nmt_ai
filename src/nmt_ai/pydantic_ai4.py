import asyncio
import time

import pydantic_ai
from pydantic_ai import Agent, RunContext, ModelSettings
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel, Field, ValidationError, field_validator
from nmt_ai.models.vendor_invoices import Invoice


from typing import List, Optional, Literal
from nmt_ai.vector_db5 import fetch_data

model_dict = {
    # 1: 'granite4.1:3b',
    # 2: 'granite4.2:3b',
    # 2: 'granite4:tiny-h',
    # 3: 'ibm/granite4:latest',
    #  4: 'gemma4:latest',
    # 5: 'gemma4:e2b',
    6: 'granite4.1:8b',
    # 7: 'ministral-3:3b',
    8: 'ministral-3:8b',
    # 9: 'qwen3.5:4b',
}

SYSTEM_PROMPT = """
You are a financial invoice extraction agent.
Output MUST be valid JSON matching the Invoice schema.
Do not include explanations, comments, or natural language.
If a value is missing, set it to null.

Do not include explanations, comments, labels, or Markdown.
The very first character must be '{'.
"""

INSTRUCTIONS = """
you are an expert for invoice extraction. please extract invoice from provided text!
Output MUST be valid JSON matching the Invoice schema.
Do not include explanations, comments, or natural language.
If a value is missing, set it to null.
Do not invent or create data not present in the text.

you MUST follow the rules strictly:
You MUST use the tool `get_vendor_number' to verify vendor name and VAT number

#Document scanning:
- Search from top to bottom AND bottom to top
- Vendor info may be located: at top header OR at bottom/footer (company/legal section, signature block)

#Customer:
- Customer  MUST contain "Mascot" 0r "mascot" or "MASCOT"
- if not, set Customer = undefined

#Vendor
- Vendor MUST not contain "Mascot" 0r "mascot" or "MASCOT
- if "mascot" found in vendor section, it is okey to inteprete it as customer name  


"""

INSTRUCTIONS1 = """
Role: You are an expert in invoice extraction. Extract invoice details from provided text.

Rules:
1. Vendor Verification:
   - ALWAYS use the tool `get_vendor_number` to verify vendor name and VAT number.
   - If vendor info is missing or invalid, set Vendor = undefined.

2. Document Scanning:
   - Scan from top to bottom AND bottom to top.
   - Vendor info may appear in header OR footer (legal section, signature block).

3. Customer Rules:
   - Customer MUST contain the word "Mascot" (case-insensitive).
   - If no match, set Customer = undefined.

4. Vendor Rules:
   - Vendor MUST NOT contain "Mascot" (case-insensitive).
   - If "Mascot" appears in vendor section, interpret it as Customer instead.

5. Error Handling:
   - If any required field cannot be extracted, set it to undefined.
"""


def run_agent(llm_model,invoice_text):
    try:
        # 2. Configure Pydantic AI to use a local Granite endpoint (e.g., via Ollama on port 11434)

        model_settings = ModelSettings(
            temperature=0,  # Giảm sáng tạo, tăng tính logic cấu trúc
            top_p =  1,
            top_k = 0,
            max_new_tokens=32384,
            max_tokens=32384,
            normalize_tool_arguments=True,
        )

        model = OllamaModel(
            llm_model,  # or your specific Granite tag
            provider=OllamaProvider(
                base_url='http://localhost:11434/v1/',
                # base_url='http://localhost:11434/v1/',
                api_key='ollama'  # Local endpoint doesn't strictly need a real key,
            ),
        )

        # 3. Initialize the Agent
        agent = Agent(
            model,
            # capabilities=[Thinking(effort='high')],
            output_type=Invoice,
            # system_prompt=SYSTEM_PROMPT,
            instructions=INSTRUCTIONS,
            model_settings=model_settings,

        )

        @agent.tool
        async def get_vendor_number(ctx: RunContext[str], question: str) -> str:
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

        # 4. Run the Agent

        result = agent.run_sync(f"{invoice_text}")
        print(result)
        print(result.all_messages())
        print(result.output)
        print (isinstance(result.output, Invoice))
        return result.output

    except ValidationError as exc:
        print (exc)
        missing_fields = [
            error["loc"][0]
            for error in exc.errors()
            if error["type"] == "missing"
        ]

        print("Missing fields:", missing_fields)
        # Output: Missing fields: ['name', 'email']


print(pydantic_ai.__version__)



def check_invoice (invoice_text,input_file):
    result_list = []
    for k, v in model_dict.items():
        print(f"Model: {v}")
        print(f"        >>>>>>> Running. Please wait.....")
        start = time.time()
        r = run_agent(v,invoice_text)
        end = time.time()
        print(f"        =>>>>>> Execution time: {end - start:.6f} seconds")
        print(f"---------------------------------------------------")

        r_dict = {"llm_model":v, "invoice":r,"input_file":input_file,"duration":f"{end - start:.6f} second"}
        result_list.append(r_dict)

        # Filter and extract the names of all missing fields

    return result_list