import time

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel, Field

from typing import List, Optional
u_prompt=""""
Nettobeløb Fragtgebyr Momspl. beløb Moms % Momsbeløb Betalt Fakturatotal
Ordre nr.......:
Kunde nr......:
Varetur.........:
Afdeling.......:
Arbejdskort.:
Rekvisition
Deres ref.
Leveret pr.
Vores ref.
Betalingsbetingelse
Forfaldsdato
Side Fakturanr. Dato
Valuta
Lev.
ant.
Bestillingsnummer tekst Ordre Lev.Dato Bruttopris Nettopris Beløb
AD Danmark A/S
Stensgårdvej 1
5500 Middelfart
Tlf. 30 50 30 50
info@addanmark.dk
Alt til værkstedet
- samlet ét sted! addanmark.dk
CVR: 12 04 80 33
Danske Bank - konto:
4394 0002589540
Returnering: Kun produkter i ubrudt emballage, og som er købt igennem AD Danmark A/S. Skaffevarer samt EL-dele og elektroniske komponenter tages ikke retur · Returgebyr: Returneres varen inden for 14 dage
pålægges 0 % i gebyr, mellem 14-30 dage pålægges et gebyr på 15 %, senere end 30 dage pålægges der et gebyr på 25 % · Betaling efter forfaldstidspunkt: Der tilskrives rente med 1,6% per påbegyndt måned ·
Ejendomsforhold: AD Danmark A/S bevarer ejendomsretten til produkterne, indtil betaling er sket i sin helhed · Herudover henvises til salgs- og leveringsbetingelser på www.addanmark.dk
*109110307*
Mascot International A/S
Silkeborgvej 14
7442 Engesvang 87244700
Att.: Erik Christensen
115224730
896115
Leveringsadresse:
Mascot International A/S
Silkeborgvej 14
7442 Engesvang Silkeborg
FAKTURA
ec1 Varetur Løbende måned + 10 dage 1 6709753 29.05.26
ec1 (WEB) 10/06-2026 DKK
1 VA369494241 29/05-26
SONAX Profiline Foam Polerskive 277,00 171,74 171,74
Ref.nr.: 494241
1 VA369494200 29/05-26
SONAX Profiline Foam Polerskive 289,00 179,18 179,18
Ref.nr.: 494200
Ved elektronisk betaling kan følgende betalingsinformation
benyttes: +71< 000000067097535+85799665<
350,92 29,00 379,92 25,00 94,98 0,00 474,90
EDoc barcode: 2605290045
EDoc received: 2026-05-29 10:33:26
EDoc original filename:
Faktura_6709753_til_ordre_115224730_hYIU.pdf
2605290045
"""
from datetime import date


class LineItem(BaseModel):
    description: Optional[str]
    quantity: Optional[float]
    unit_price: Optional[float]
    amount: Optional[float]


class Invoice(BaseModel):
    invoice_number: Optional[str] = Field(description="invoice number")
    # invoice_date: Optional[str]
    invoice_date: Optional[str] = Field(description="Invoice Date")
    language: Optional[str] = Field(description="the language of invoice")

    # due_date: Optional[str]
    due_date: Optional[str] = Field(description="the due date of invoice")
    vendor_name: Optional[str]  =  Field(description="vendor name")
    customer_name: Optional[str] = Field(description="customer name. e.g Mascot")
    currency: Optional[str]  = Field(description="currency")
    payment_terms: Optional[str] = Field(description="payment term")

    subtotal: Optional[float] = Field(description="Pre-tax invoice total — sum of line items before VAT")
    tax_rate: Optional[float] = Field(description="Tax rate")
    tax_amount: Optional[float] = Field(description="Total VAT charged on the invoice")
    total: Optional[float] = Field(description="Final invoice total — net_total plus vat_amount")
    confidence: float = Field(description="Confidence score of the extraction between 0.0 and 1.0.", ge=0.0, le=1.0)


model_dict = {
    1: 'granite4.1:3b',
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


# llm_model = model_dict[1]
# print(llm_model)

# 1. Define the desired output structure (Pydantic validation)
class AgentResponse(BaseModel):
    summary: str
    action_taken: bool


def run_agent(llm_model):
    # 2. Configure Pydantic AI to use a local Granite endpoint (e.g., via Ollama on port 11434)
    model = OpenAIChatModel(
        llm_model, # or your specific Granite tag
        provider=OllamaProvider(
            base_url='http://localhost:11434/v1/',
            api_key='ollama' # Local endpoint doesn't strictly need a real key
        )
    )

    # 3. Initialize the Agent
    agent = Agent(
        model,
        output_type=Invoice,
        # system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS
    )


    # 4. Run the Agent
    result = agent.run_sync(f"<text>{u_prompt}</text>")
    print(result.all_messages())
    print(result.output)


for k,v in model_dict.items():
    print (f"Model: {v}")
    print(f">>>>>>> Running. Please wait.....")
    start = time.time()
    run_agent(v)
    end = time.time()
    print(f"=>>>>>> Execution time: {end - start:.6f} seconds")
    print(f"---------------------------------------------------")