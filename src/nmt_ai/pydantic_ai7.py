import asyncio
import time
from typing import Any

from pydantic import ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from nmt_ai.models.vendor_invoices import Invoice
from nmt_ai.vector_db5 import fetch_data

MODEL_DICT: dict[int, str] = {
    7: "granite4.2:8b",
}

CLEAN_INSTRUCTIONS = """
Extract invoice data from the provided text and match the Invoice schema.
Do not invent values not present in the text. Use null/None when a field is missing or cannot be verified.

Rules:
1. Call `get_vendor_number` to verify vendor details and VAT number before finalizing the vendor.
2. Scan the full document (header, body, and footer/signature blocks).
3. Customer must match "Mascot" (case-insensitive); otherwise set customer to null/None.
4. Vendor must NOT match "Mascot" (case-insensitive). If "Mascot" appears in a vendor-like block, treat it as customer rather than vendor.
"""


def build_agent(model_name: str) -> Agent[None, Invoice]:
    provider = OpenAIProvider(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",
    )
    model = OpenAIChatModel(
        model_name=model_name,
        provider=provider,
    )
    settings = OpenAIChatModelSettings(
        temperature=0.1,
        top_p=0.95,
        max_tokens=8192,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    )

    agent: Agent[None, Invoice] = Agent(
        model=model,
        output_type=Invoice,
        model_settings=settings,
        instructions=CLEAN_INSTRUCTIONS,
    )

    @agent.tool
    async def get_vendor_number(ctx: RunContext[None], question: str) -> str:
        """Search the vendor database for vendor details or VAT number matching `question`."""
        result = await fetch_data(question)
        return str(result) if result is not None else "No matching vendor found."

    return agent


async def run_agent(
    agent: Agent[None, Invoice],
    invoice_text: str,
    semaphore: asyncio.Semaphore | None = None,
) -> Invoice | None:
    async def _execute() -> Invoice | None:
        try:
            result = await agent.run(invoice_text)
            return result.output
        except ValidationError as exc:
            missing_fields = [
                err["loc"][0] for err in exc.errors() if err["type"] == "missing"
            ]
            print("Validation error. Missing fields:", missing_fields)
            print(exc)
            return None
        except Exception as exc:
            print(f"Agent execution failed: {exc}")
            return None

    if semaphore is not None:
        async with semaphore:
            return await _execute()
    return await _execute()


async def _run_single_model(
    model_name: str,
    agent: Agent[None, Invoice],
    invoice_text: str,
    input_file: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    print(f"Model: {model_name} -> Queued for {input_file}")
    start = time.perf_counter()
    invoice = await run_agent(agent, invoice_text, semaphore=semaphore)
    elapsed = time.perf_counter() - start
    print(f"Model: {model_name} finished {input_file} in {elapsed:.6f}s")
    return {
        "llm_model": model_name,
        "invoice": invoice,
        "input_file": input_file,
        "duration_sec": round(elapsed, 6),
    }


async def check_invoice(
    invoice_text: str,
    input_file: str,
    models: dict[int, str] = MODEL_DICT,
    agents: dict[str, Agent[None, Invoice]] | None = None,
    max_concurrency: int = 2,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    agent_map = agents or {name: build_agent(name) for name in models.values()}

    tasks = [
        _run_single_model(
            model_name=model_name,
            agent=agent_map[model_name],
            invoice_text=invoice_text,
            input_file=input_file,
            semaphore=semaphore,
        )
        for model_name in models.values()
    ]
    return await asyncio.gather(*tasks)


async def check_invoices_batch(
    invoices: list[tuple[str, str]],  # [(invoice_text, input_file), ...]
    models: dict[int, str] = MODEL_DICT,
    max_concurrency: int = 2,
) -> list[dict[str, Any]]:
    # Build agents once for the whole batch
    agent_map = {name: build_agent(name) for name in models.values()}
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        _run_single_model(
            model_name=model_name,
            agent=agent_map[model_name],
            invoice_text=invoice_text,
            input_file=input_file,
            semaphore=semaphore,
        )
        for invoice_text, input_file in invoices
        for model_name in models.values()
    ]
    return await asyncio.gather(*tasks)


# Example entry point:
# results = asyncio.run(check_invoice(invoice_text, "inv_001.pdf"))