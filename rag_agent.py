from smolagents import (
    CodeAgent,
    tool,
    load_tool,
    LiteLLMModel,
    HfApiModel,
)

import requests

from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor


# defaults to endpoint="http://localhost:4317"
tracer_provider = register(
  project_name="my-llm-app", # Default is 'default'
  endpoint="http://localhost:4317",  # Sends traces using gRPC
)

SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """
    Convert an amount from one currency to another using fawazahmed0's free API.

    Args:
       amount: The amount to convert
       from_currency: The source currency code (e.g., 'USD', 'EUR', 'BTC')
       to_currency: The target currency code (e.g., 'USD', 'EUR', 'BTC')
    """
    url = f'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{from_currency.lower()}.json'

    response = requests.get(url)
    rates = response.json()[from_currency.lower()]
    converted = amount * rates[to_currency.lower()]

    return round(converted, 2)



# model = LiteLLMModel(
#     model_id="ollama_chat/phi4",
#     api_base="http://localhost:11434"
# )

model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")


agent = CodeAgent(
    tools=[convert_currency],
    model=model,
    add_base_tools=True,
    planning_interval=3,
    verbosity_level=0,
)

res = agent.run(
    "What is 100 great british pounds to dollars?",
)

print(res)


