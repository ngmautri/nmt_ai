import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pydantic_ai
from pydantic_ai import Agent, RunContext, ModelSettings
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_ai.providers.openai import OpenAIProvider

from nmt_ai.models.invoice_draft import InvoiceDraft
from nmt_ai.models.vendor_invoices import Invoice


from typing import List, Optional, Literal, Any

from nmt_ai.rag.retrieval import retrieve_similar_invoices
from nmt_ai.settings import DATA_PATH
from nmt_ai.vector_db5 import fetch_data

model_dict = {
    # 1: 'granite4.1:3b',
    # 2: 'granite4.2:3b',
    # 2: 'granite4:tiny-h',
    # 3: 'ibm/granite4:latest',
    #  4: 'gemma4:latest',
    # 5: 'gemma4:e2b',
    # 6: 'granite4.1:8b',
    7: 'granite4.2:8b',
    # 8: 'ministral-3:8b',
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
Extract invoice data strictly from the provided text to match the `Invoice` schema.

Field Mapping & Grounding:
- Ground every value in the text; never guess or invent data.
- Set any missing, unreadable, or non-conforming field to `null` (`None`).
- Scan the full document (header, line items, footer, legal blocks, and signatures); vendor metadata often sits in footers or signature blocks.
- Map fields to exact `Invoice` attributes:
  * `invoice_number`, `issue_date`, `due_date`, `currency`, `total_amount`, `vat_amount`
  * `customer_name`: Must contain "mascot" (case-insensitive); if not, set to `null`.
  * `vendor_name` and `vendor_vat_number`: Must NOT contain "mascot" (case-insensitive). If "mascot" appears in a vendor-like block, treat it as `customer_name` instead and keep `vendor_name` / `vendor_vat_number` grounded in the actual supplier.

Tool Verification:
- Always call `get_vendor_number` with the candidate `vendor_name` or `vendor_vat_number` to look up and populate `vendor_number`, `vendor_name_sap` and confirm vendor identity.
"""

INSTRUCTIONS12 = """
Extract invoice data strictly from the provided text to match the `Invoice` schema.

Field Mapping & Grounding:
- Ground every value in the text; never guess or invent data.
- Set any missing, unreadable, or non-conforming field to `null` (`None`).
- Scan the full document (header, line items, footer, legal blocks, and signatures); vendor metadata often sits in footers or signature blocks.
- Map fields to exact `Invoice` attributes:
  * `invoice_number`, `issue_date`, `due_date`, `currency`, `total_amount`, `vat_amount`
  * `customer_name`: Must contain "mascot" (case-insensitive); if not, set to `null`.
  * `vendor_name` and `vendor_vat_number`: Must NOT contain "mascot" (case-insensitive). If "mascot" appears in a vendor-like block, treat it as `customer_name` instead and keep `vendor_name` / `vendor_vat_number` grounded in the actual supplier.

Tool Verification:
- Always call `get_vendor_number` with the candidate `vendor_name` or `vendor_vat_number` to look up and populate `vendor_number` and confirm vendor identity.
"""

INSTRUCTIONS5 = """
You are a precise financial invoice extraction specialist. Extract structured invoice data strictly from the provided text to match the Invoice schema.

Core Rules:
1. Grounding: Extract only facts explicitly present in the text. Never guess, extrapolate, or invent missing values.
2. Missing values: Set any missing, unverified, or non-matching field to null.
3. Vendor verification: Always call `get_vendor_number` to verify the vendor name and VAT/tax identifier against the vendor database before finalizing vendor fields.

Document Scanning:
- Inspect the entire document, including top headers, line-item tables, bottom footers, legal notices, and signature blocks.
- Vendor details frequently appear in footers or signature blocks rather than only at the top.

Customer vs. Vendor Rules:
- Customer: Must match "Mascot" (case-insensitive). If "Mascot" is not present for the customer, set the customer field to null.
- Vendor: Must NOT match "Mascot" (case-insensitive). If "Mascot" appears in a vendor-like block, treat it as the customer rather than the vendor.
"""


INSTRUCTIONS3 = """
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


async def run_agent(llm_model, invoice_text, input_file):
    try:

        @dataclass
        class InvoiceDeps:
            similar_samples: list[dict[str, Any]]

        samples = await retrieve_similar_invoices(invoice_text, top_k=1)
        print(samples)
        deps = InvoiceDeps(similar_samples=samples)

        # 2. Configure Pydantic AI to use a local Granite endpoint (e.g., via Ollama on port 11434)

        # 2. Point PydanticAI at your local vLLM or llama-server endpoint
        provider = OpenAIProvider(
            base_url='http://localhost:11434/v1/',
            api_key="ollama",
        )

        model = OpenAIChatModel(
            model_name=llm_model,
            provider=provider,
        )

        # 3. Configure Granite 4.2 recommended sampling & optional thinking control
        settings = OpenAIChatModelSettings(
            temperature=0.1,
            top_p=0.95,
            # max_tokens=8192 4096,16384
            max_tokens=16384,
            # Optional: toggle Granite 4.2 thinking per agent or per run:
            openai_reasoning_effort='none',

        )

        agent = Agent(
            model=model,
            deps_type=InvoiceDeps,
            output_type=Invoice,
            model_settings=settings,
            instructions=INSTRUCTIONS,

        )

        @agent.instructions
        def inject_rag_examples(ctx: RunContext[InvoiceDeps]) -> str:
            if not ctx.deps.similar_samples:
                return ""

            blocks: list[str] = []
            for idx, sample in enumerate(ctx.deps.similar_samples, start=1):
                excerpt = str(sample.get("raw_text", ""))[:1000]
                verified_json = json.dumps(
                    sample.get("extracted_invoice", {}),
                    ensure_ascii=False,
                )
                blocks.append(
                    f"Reference {idx} (layout/mapping cue only—do NOT copy values):\n"
                    f"Text:\n{excerpt}\n"
                    f"JSON:\n{verified_json}"
                )

            return (
                    "Verified historical invoice examples for layout reference only. "
                    "Never copy invoice numbers, dates, or amounts from examples:\n\n"
                    + "\n\n".join(blocks)
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

        result = await agent.run(f"{invoice_text}",deps=deps)
        print(result)
        print(result.all_messages())
        print(result.output)
        print (isinstance(result.output, Invoice))

        extracted_invoice = result.output



        # Step 1: After running the agent, save the draft to a review folder
        draft = InvoiceDraft(
            source_file=input_file,
            raw_pdf_text=invoice_text,
            invoice= extracted_invoice,
        )
        draft_file = draft.save_json(Path(DATA_PATH + "/review_queue") / f"{Path(input_file).stem}_draft.json")

        # Step 2: Open `draft_file` in VS Code / editor, fix any `invoice` fields,
        # and change `"reviewed": false` -> `"reviewed": true`.

        # Step 3: Reload and commit verified drafts to ChromaDB
        # doc_id = commit_draft_file(draft_file)
        # if doc_id:
        #     print(f"Committed verified invoice to ChromaDB: {doc_id}")

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


print(f"PydanticAI version: {pydantic_ai.__version__}")


async def check_invoice (invoice_text, input_file):
    result_list = []
    for k, v in model_dict.items():
        print(f"Model: {v}")
        print(f"        >>>>>>> Running. Please wait.....")
        start = time.time()
        r = await run_agent(v,invoice_text,input_file)
        end = time.time()
        print(f"        =>>>>>> Execution time: {end - start:.6f} seconds")
        print(f"---------------------------------------------------")

        r_dict = {"llm_model":v, "invoice":r,"input_file":input_file,"duration":f"{end - start:.6f} second"}
        result_list.append(r_dict)

        # Filter and extract the names of all missing fields

    return result_list