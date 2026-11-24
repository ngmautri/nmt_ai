from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from sympy import true


# 1. Define strongly-typed dependencies & final structured output
@dataclass
class FxDeps:
    api_source: str


class ConversionResult(BaseModel):
    base_currency: str
    target_currency: str
    exchange_rate: float
    converted_amount: float = Field(description="Total amount in target currency")
    summary: str


# 2. Point PydanticAI at your local vLLM or llama-server endpoint
provider = OpenAIProvider(
    base_url='http://localhost:11434/v1/',
    api_key="ollama",
)

model = OpenAIChatModel(
    model_name="granite4.2:8b",
    provider=provider,
)

# 3. Configure Granite 4.2 recommended sampling & optional thinking control
settings = OpenAIChatModelSettings(
    temperature=1.0,
    top_p=0.95,
    max_tokens=4096,
    # Optional: toggle Granite 4.2 thinking per agent or per run:
    openai_reasoning_effort = 'low',

)

agent = Agent(
    model=model,
    deps_type=FxDeps,
    output_type=ConversionResult,
    model_settings=settings,
    system_prompt=(
        "You are a financial assistant. Always call the exchange rate tool "
        "before calculating currency conversions."
    ),
)


# 4. Register Python functions as tools — PydanticAI generates the JSON schema from type hints
@agent.tool
async def get_exchange_rate(
    ctx: RunContext[FxDeps],
    base_currency: str,
    target_currency: str,
) -> dict:
    """Fetch the current exchange rate between two ISO 4217 currency codes."""
    rates = {("USD", "VND"): 25450.0, ("EUR", "USD"): 1.09}
    rate = rates.get((base_currency.upper(), target_currency.upper()), 1.0)
    return {
        "source": ctx.deps.api_source,
        "base": base_currency.upper(),
        "target": target_currency.upper(),
        "rate": rate,
    }


# 5. Run the agent (automatically executes tool calls, validates args, and parses output)
if __name__ == "__main__":
    result = agent.run_sync(
        "Convert $1,250 USD to VND.",
        deps=FxDeps(api_source="internal-fx-feed"),
    )
    print(result.output.model_dump_json(indent=2))
    print (result.all_messages())