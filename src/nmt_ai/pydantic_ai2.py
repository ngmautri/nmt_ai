import time
from enum import Enum

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel, Field
u_prompt=""""
Bankverbindungen: Krönlein GmbH + Co. KG PhG: Krönlein Verwaltungs-GmbH
Commerzbank Schweinfurt IBAN: DE66 7938 0051 0400 4917 00 Carl-Zeiss-Str. 15, 97424 Schweinfurt Carl-Zeiss-Str. 15, 97424 Schweinfurt
Flessabank Schweinfurt IBAN: DE85 7933 0111 0000 0003 21 Sitz + Reg.-Ger.: Schweinfurt HRA 970 Sitz + Reg.-Ger.: Schweinfurt HRB 4038
Sparkasse Schweinfurt IBAN: DE46 7935 0101 0000 0760 00 USt-IdNr.: DE 133 884 070 GF: Stefan Morsch, Frank Wilm, Helmut Ernst, Dr. Ralf von Briel
Krönlein GmbH + Co. KG • Carl-Zeiss-Str. 15 • 97424 Schweinfurt Ihr Ansprechpartner
Lisa Pabst
Carl-Zeiss-Str. 15
97424 Schweinfurt
Tel: 09721 7755 0
Fax: 09721 7755 -
- Werkzeuge: - 499
- Baubeschlag: - 399
info@kroenlein.de
Lochweg 15
97318 Kitzingen
Tel: 09321 370 70
Fax: 09321 370 715
kitzingen@kroenlein.de
Mascot International GmbH
Neustadt 10
24939 Flensburg
R E C H N U N G – N R. 11632
Werbekostenzuschuß
Datum 26.05.2026
(Leistungsdatum Mai 2026)
gemäß Vereinbarung mit Herrn Mögel dürfen wir Ihnen anteilige
Werbekosten für unseren „Lotter Bestseller 2026-1“ in Rechnung
stellen.
300,00 €
300,00 €
19% Umsatzsteuer 57,00 €
357,00 €
Zahlbar sofort, ohne Abzug.
EDoc barcode: 2605280020
EDoc received: 2026-05-28 04:19:32
EDoc original filename:
rg11632_WKZ.pdf
2605280020
"""


class Category(str, Enum):
    INV = "Invoice"
    CRE = "Credit Note"
    NOTICE = "Notice"
    REMINDER = "Reminder"
    OTHER = "Other"

class TextCategorization(BaseModel):
    category: Category = Field(description="The primary category of the text.")
    confidence: float = Field(description="Confidence score of the categorization between 0.0 and 1.0.", ge=0.0, le=1.0)
    reasoning: str = Field(description="A brief explanation for why this category was chosen.")


model_dict = {
    1: 'granite4.1:3b',
    2: 'granite4:tiny-h',
    # 3: 'ibm/granite4:latest',
    # 4: 'gemma3:latest',
    5: 'gemma4:e2b',
    6: 'granite4.1:8b',
    7: 'ministral-3:3b',
    # 8: 'ministral-3',
    # 9: 'qwen3.5:4b',
}

# llm_model = model_dict[1]
# print(llm_model)

# 1. Define the desired output structure (Pydantic validation)
class AgentResponse(BaseModel):
    summary: str
    action_taken: bool


def run(llm_model):
    # 2. Configure Pydantic AI to use a local Granite endpoint (e.g., via Ollama on port 11434)
    model = OpenAIChatModel(
        llm_model,  # or your specific Granite tag
        provider=OllamaProvider(
            base_url='http://localhost:11434/v1/',
            api_key='ollama'  # Local endpoint doesn't strictly need a real key
        )
    )

    # 3. Initialize the Agent
    agent = Agent(
        model,
        output_type=TextCategorization,
        instructions="""  "You are an expert text classification assistant. "
            "Analyze the provided text and categorize it accurately."
        """,

    )


    # 4. Run the Agent
    result = agent.run_sync(f"<text>{u_prompt}</text>")
    print(result.output)


for k,v in model_dict.items():
    print (f"Model: {v}")
    print(f">>>>>>> Running. Please wait.....")
    start = time.time()
    run(v)
    end = time.time()
    print(f"=>>>>>> Execution time: {end - start:.6f} seconds")
    print(f"--------------------------------------------------")