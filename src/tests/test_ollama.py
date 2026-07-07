import chromadb
from _testcapi import awaitType
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai import Embedder
import asyncio

class CityLocation(BaseModel):
    city: str
    country: str


model_dict ={
    1: "ministral-3:3b",
    2: "granite4.1:3b",
}

import time

start_time = time.time()   # record start

llm_mode = model_dict[2]
print("llm_mode: ", llm_mode)

model = OllamaModel(
    model_name=llm_mode,
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
# agent = Agent(model, output_type=NativeOutput(CityLocation))
agent = Agent(model)

result = agent.run_sync('is the invoice due in 2 weeks?')
print(result.output)
print(result.usage)
print(result.all_messages())
end_time = time.time()     # record end
print("Runtime:", end_time - start_time, "seconds")




# Register the ChromaDB search tool with the agent
@agent.tool
def search_knowledge_base(ctx: RunContext, query: str) -> str:
    # 1. Generate an embedding for the query (using your model/embedding function)
    # 2. Query ChromaDB

    print("search_knowledge_base")

    emb_model = OpenAIEmbeddingModel(
        model_name="embeddinggemma:latest",
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    )
    embedder = Embedder(emb_model)


    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_store")
    col = client.get_or_create_collection(name="test_collection")

    q_emb= embedder.embed_query(query)
    res = col.query(query_embeddings=q_emb.embeddings, n_results=2)

    # 3. Format and return the documents for the LLM
    documents = "\n".join(res.get('documents', [[]])[0])
    return documents or "No relevant information found."


