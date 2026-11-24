from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


model = OllamaModel(
    'granite4.2:8b',
    provider=OllamaProvider(
        base_url='http://localhost:11434/v1/',
        api_key='ollama'  # Local endpoint doesn't strictly need a real key,
    ),
)

agent = Agent(
    model,
    output_type=str,
)

result = agent.run_sync("what is LLM")

print(result.output)