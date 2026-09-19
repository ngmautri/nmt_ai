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
from vector_db5 import fetch_data

u_prompt=""""
Page 1 / 1
Faktura nr.
1080001476
Mascot International A/S
Silkeborgvej 14
7442 ENGESVANG
Udstedelsesdato
16.06.2026
Kundenummer:
6407939446
Kunde momsnummer:
DK13863385
Kunde CVR-nummer:
Forfaldsdato:
26.06.2026
Betalingsbetingelser:
10 days, net
Betalingsreference
64079394461080001476
DSV Air & Sea A/S Hovedgaden 630 2640 Hedehusene Denmark Tax no. DK26366224 Reg. no. 26366313
Phone: +45 43 20 30 40 www.dsv.com/da-dk/
Payment address: NORDEA CPH EUR IBAN: DK0920005036392616 DKK IBAN: DK1720000724121284
Kunde ref.
Forsendelse
58644466
DSV Service
DSV XPress Economy
Tracking nummer
872212738047 /
Afhentningsdato
26.05.2026
ETA
04.06.2026 / 01.06.2026
Afhentningadresse
Mascot International A/S, Silkeborgvej 14, DK-7442 Engesvang
Leveringsadresse
BMS Heavy Cranes INC, 1346 Markum Ranch Rd, US-76126 Fort Worth
88548902
Brutto vægt
Volumen
Chargeable Weight
Packages
Godsbeskrivelse
Samlet Co2
37.8 KG
0.254 m3
63.5 KG
4 Pieces
PARCELS
Ydelse
Valuta
Beløb
Valutakurs
Fakturavaluta
Nettobeløb
Momskode
XP0055
Customs GST
DKK
5.159,74
YY
Total moms per momskode
Momssats
Momskode
Valuta
Momspligtigt beløb
Momsbeløb
Brutto beløb
0,00%
YY
DKK
5.159,74
0,00
5.159,74
Total
DKK
5.159,74
0,00
5.159,74
Moms i DKK
DKK
0,00
Kontakt:
XPRESS@DK.DSV.COM
"""
from datetime import date


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
    vendor_name: Optional[str]  =  Field(description="vendor name")

    # it does not work with vendor_vat_number
    vendor_vat: Optional[str] = Field(default=None, description="Vendor VAT number")

    # vendor_number: Optional[str]  =  Field(description="vendor number")

    customer_name: Optional[str] = Field(description="customer name. e.g Mascot")
    currency: Optional[str]  = Field(description="currency")
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
    vendor_name: Optional[str] = Field(default=None, description="vendor name")
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

class Invoice(BaseModel):
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    invoice_date: Optional[str] = Field(None, description="Invoice date")
    language: Optional[str] = Field(None, description="Language of invoice")
    due_date: Optional[str] = Field(None, description="Due date")
    vendor_name: Optional[str] = Field(None, description="Vendor name")
    vendor_vat: Optional[str] = Field(None, description="Vendor VAT number")
    # vendor_number: Optional[str] = Field(None, description="Vendor number to be retrieved from Tool.")
    customer_name: Optional[str] = Field(None, description="Customer name (must contain Mascot)")
    currency: Optional[Literal["USD", "EUR", "VND", "DKK", "SEK","NOK","PLN"]] = Field(None, description="Currency code")
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

    @field_validator("vendor_name")
    def validate_vendor(cls, v: Optional[str]) -> Optional[str]:
        if v and "mascot" in v.lower():
            return None
        return

model_dict = {
    1: 'granite4.1:3b',
    # 2: 'granite4.2:3b',
    # 2: 'granite4:tiny-h',
    # 3: 'ibm/granite4:latest',
    # 4: 'gemma3:latest',
    # 5: 'gemma4:e2b',
    # 6: 'granite4.1:8b',
    7: 'ministral-3:3b',
     # 8: 'ministral-3',
    # 9: 'qwen3.5:4b',
}


SYSTEM_PROMPT = """
You are a financial invoice extraction agent.
Follow the schema documentation exactly.

You MUST follow rules strictly
- Never guess or hallucinate
- If a value is missing, return null
- Output MUST be valid JSON matching schema
- Numbers must be numeric (no commas, no currency symbols)
- Customer party NAME MUST contain "Mascot" 0r "mascot" or "MASCOT"
- Vendor party NAME MUST NOT contain "Mascot" or "mascot" or "MASCOT"
"""


INSTRUCTIONS = """
you are an expert for invoice extraction. please extract invoice from provided text!

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


# llm_model = model_dict[1]
# print(llm_model)

# 1. Define the desired output structure (Pydantic validation)
class AgentResponse(BaseModel):
    summary: str
    action_taken: bool


def run_agent(llm_model):
    try:
        # 2. Configure Pydantic AI to use a local Granite endpoint (e.g., via Ollama on port 11434)

        model_settings = ModelSettings(
            temperature=0.1,  # Giảm sáng tạo, tăng tính logic cấu trúc
            max_tokens=4096,  # Đủ không gian cho cả chuỗi <think> và JSON kết quả
        )

        model = OllamaModel (
            llm_model, # or your specific Granite tag
            provider=OllamaProvider(
                base_url='http://localhost:11434/v1/',
                # base_url='http://localhost:11434/v1/',
                api_key='ollama' # Local endpoint doesn't strictly need a real key,

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
            """Get vendor number"""
            return await fetch_data(question)


        # 4. Run the Agent


        result = agent.run_sync(f"{u_prompt}")
        print(result.all_messages())
        print(result.output)

    except ValidationError as exc:
        missing_fields = [
            error["loc"][0]
            for error in exc.errors()
            if error["type"] == "missing"
        ]

        print("Missing fields:", missing_fields)
        # Output: Missing fields: ['name', 'email']


print(pydantic_ai.__version__)

for k,v in model_dict.items():

    print (f"Model: {v}")
    print(f">>>>>>> Running. Please wait.....")
    start = time.time()
    run_agent(v)
    end = time.time()
    print(f"=>>>>>> Execution time: {end - start:.6f} seconds")
    print(f"---------------------------------------------------")
    # Filter and extract the names of all missing fields
