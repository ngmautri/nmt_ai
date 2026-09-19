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

from typing import List, Optional, Literal
from nmt_ai.vector_db5 import fetch_data

class LineItem(BaseModel):
    description: Optional[str]
    quantity: Optional[float]
    unit_price: Optional[float]
    amount: Optional[float]


class Invoice_1(BaseModel):
    invoice_number: Optional[str] = Field(description="invoice number")
    # invoice_date: Optional[str]
    invoice_date: Optional[str] = Field(description="Invoice Date")
    language: Optional[str] = Field(default=None, description="the language of invoice")

    # due_date: Optional[str]
    due_date: Optional[str] = Field(description="the due date of invoice")
    vendor_name: Optional[str] = Field(description="vendor name")

    # it does not work with vendor_vat_number
    vendor_vat: Optional[str] = Field(default=None, description="Vendor VAT number")

    # vendor_number: Optional[str]  =  Field(description="vendor number")

    customer_name: Optional[str] = Field(description="customer name. e.g Mascot")
    currency: Optional[str] = Field(description="currency")
    payment_terms: Optional[str] = Field(description="payment term")

    subtotal: Optional[float] = Field(description="Pre-tax invoice total — sum of line items before VAT")
    tax_rate: Optional[float] = Field(description="Tax rate")
    tax_amount: Optional[float] = Field(description="Total VAT charged on the invoice")
    total: Optional[float] = Field(description="Final invoice total — net_total plus vat_amount")
    # confidence: float = Field(default=0.1 ,description="Confidence score of the extraction between 0.0 and 1.0.", ge=0.0, le=1.0)
    confidence: float = Field(default=0.1, description="Confidence score of the extraction between 0.0 and 1.0.")


class Invoice2(BaseModel):
    invoice_number: Optional[str] = Field(default=None, description="invoice number")
    invoice_date: Optional[str] = Field(default=None, description="Invoice Date")
    language: Optional[str] = Field(default=None, description="the language of invoice")
    due_date: Optional[str] = Field(default=None, description="the due date of invoice")
    vendor_vat: Optional[str] = Field(default=None, description="Vendor VAT number")
    customer_name: Optional[str] = Field(default=None, description="customer name. e.g Mascot")
    currency: Optional[str] = Field(default=None, description="currency")
    payment_terms: Optional[str] = Field(default=None, description="payment term")

    subtotal: Optional[float] = Field(default=None, description="Pre-tax invoice total — sum of line items before VAT")
    tax_rate: Optional[float] = Field(default=None, description="Tax rate")
    tax_amount: Optional[float] = Field(default=None, description="Total VAT charged on the invoice")
    total: Optional[float] = Field(default=None, description="Final invoice total — net_total plus vat_amount")

    # XÓA ge=0.0 và le=1.0 ở đây
    confidence: float = Field(default=0.1, description="Confidence score of the extraction between 0.0 and 1.0.")

    # Thay thế bằng validator của Pydantic để không làm lỗi Ollama Grammar
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class Invoice_tmp1(BaseModel):
    """
      A structured representation of an invoice document.

      This Pydantic model defines the key attributes extracted from invoices,
      including metadata (invoice number, dates, language), vendor and customer
      details, financial amounts, and confidence scores. It is designed to support
      automated invoice parsing, validation, and integration with downstream
      systems such as Granite 4.2 for schema-guided generation.

      Fields:
          invoice_number (Optional[str]): Unique identifier for the invoice.
          invoice_date (Optional[str]): Date the invoice was issued.
          language (Optional[str]): Language of the invoice text.
          due_date (Optional[str]): Payment due date.
          vendor_name (Optional[str]): Name of the vendor issuing the invoice.
          vendor_vat (Optional[str]): Vendor's VAT number.
          customer_name (Optional[str]): Name of the customer (e.g., Mascot).
          currency (Optional[str]): Currency code (e.g., USD, EUR).
          payment_terms (Optional[str]): Payment terms specified on the invoice.
          subtotal (Optional[float]): Pre-tax total of line items.
          tax_rate (Optional[float]): Applied tax rate.
          tax_amount (Optional[float]): Total VAT charged.
          total (Optional[float]): Final invoice total (subtotal + tax).
          confidence (float): Confidence score of extraction (0.0–1.0).
      """
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    invoice_date: Optional[str] = Field(None, description="Invoice date")
    language: Optional[str] = Field(None, description="Language of invoice")
    due_date: Optional[str] = Field(None, description="Due date")
    vendor_name: Optional[str] = Field(description="Vendor name")
    vendor_vat: Optional[str] = Field(None, description="Vendor VAT number")
    vendor_number: Optional[str] = Field(None, description="Vendor number in SAP to be retrieved from Tool.")
    vendor_name_sap: Optional[str] = Field(None, description="Vendor number in SAP to be retrieved from Tool.")
    customer_name: Optional[str] = Field(None, description="Customer name (must contain Mascot)")
    currency: Optional[Literal["USD", "EUR", "VND", "DKK", "SEK", "NOK", "PLN"]] = Field(None,
                                                                                         description="Currency code")
    payment_terms: Optional[str] = Field(None, description="Payment terms")

    subtotal: Optional[float] = Field(None, description="Pre-tax total")
    tax_rate: Optional[float] = Field(None, description="Tax rate")
    tax_amount: Optional[float] = Field(None, description="VAT amount")
    total: Optional[float] = Field(None, description="Final total")

    confidence: float = Field(0.0, description="Confidence score between 0.0 and 1.0")

    @field_validator("confidence")
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @field_validator("customer_name")
    def validate_customer(cls, v: Optional[str]) -> Optional[str]:
        if v and "mascot" not in v.lower():
            return None
        return v

    # @field_validator("vendor_name")
    # def validate_vendor(cls, v: Optional[str]) -> Optional[str]:
    #     if v and "mascot" in v.lower():
    #         return None
    #     return

class Invoice(BaseModel):
    """
    A structured representation of an invoice document.
    """

    invoice_number: Optional[str] = Field(
        default=None,
        description="Unique identifier assigned to the invoice (e.g., INV-2026-001)."
    )
    invoice_date: Optional[str] = Field(
        default=None,
        description="Date the invoice was issued, ideally in ISO 8601 format (YYYY-MM-DD)."
    )
    language: Optional[str] = Field(
        default=None,
        description="Language in which the invoice is written (e.g., 'EN', 'FR')."
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Date by which payment must be made, in ISO 8601 format."
    )
    vendor_name: Optional[str] = Field(
        default=None,
        description="Legal or trading name of the vendor issuing the invoice."
    )
    vendor_vat: Optional[str] = Field(
        default=None,
        description="Vendor's VAT (Value Added Tax) registration number."
    )
    customer_name: Optional[str] = Field(
        default=None,
        description="Name of the customer or client receiving the invoice."
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency code for monetary values, following ISO 4217 (e.g., 'USD', 'EUR')."
    )
    payment_terms: Optional[str] = Field(
        default=None,
        description="Agreed payment terms (e.g., 'Net 30', 'Due on receipt')."
    )
    subtotal: Optional[float] = Field(
        default=None,
        description="Total amount before tax, calculated as the sum of all line items."
    )
    tax_rate: Optional[float] = Field(
        default=None,
        description="Applied tax rate as a percentage (e.g., 10.0 for 10%)."
    )
    tax_amount: Optional[float] = Field(
        default=0,
        description="Total tax charged on the invoice, based on subtotal and tax rate."
    )
    total: Optional[float] = Field(
        default=None,
        description="Final payable amount, equal to subtotal plus tax_amount."
    )
    confidence: float = Field(
        default=0.1,
        description="Confidence score (0.0–1.0) indicating reliability of extracted values.",
        ge=0.0,
        le=1.0
    )

model_dict = {
    # 1: 'granite4.1:3b',
    # 2: 'granite4.2:3b',
    # 2: 'granite4:tiny-h',
    # 3: 'ibm/granite4:latest',
    #  4: 'gemma4:latest',
    # 5: 'gemma4:e2b',
    6: 'granite4.1:8b',
    7: 'ministral-3:3b',
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
            temperature=0.1,  # Giảm sáng tạo, tăng tính logic cấu trúc
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
            Retrieve a vendor's number and related details from the vendor database.

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
        print(result.all_messages())
        print(result.output)
        return result.output

    except ValidationError as exc:
        missing_fields = [
            error["loc"][0]
            for error in exc.errors()
            if error["type"] == "missing"
        ]

        print("Missing fields:", missing_fields)
        # Output: Missing fields: ['name', 'email']


print(pydantic_ai.__version__)



def check_invoice (invoice_text):
    result_list = []
    for k, v in model_dict.items():
        print(f"Model: {v}")
        print(f">>>>>>> Running. Please wait.....")
        start = time.time()
        r = run_agent(v,invoice_text)
        result_list.append(r)
        end = time.time()
        print(f"=>>>>>> Execution time: {end - start:.6f} seconds")
        print(f"---------------------------------------------------")
        # Filter and extract the names of all missing fields

    return result_list