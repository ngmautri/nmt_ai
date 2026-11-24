from ollama import chat


def get_current_weather(city: str) -> str:
  """Get the current weather for a specified city."""
  return f"The weather in {city} is 26°C and sunny."


response = chat(
    model="granite4.2:8b",
    messages=[
        {"role": "user", "content": "What's the weather like in Berlin right now?"}
    ],
    tools=[get_current_weather],
)

print(response.message)

if response.message.tool_calls:
  for tool_call in response.message.tool_calls:
    print("Function:", tool_call.function.name)
    print("Arguments:", tool_call.function.arguments)
else:
  print(response.message.content)

