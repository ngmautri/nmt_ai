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
emb_model = OpenAIEmbeddingModel(
    model_name="embeddinggemma:latest",
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
embedder = Embedder(emb_model)


async def fetch_data():
    # Embed a search query

    client = chromadb.PersistentClient(path="./chroma_store")

    result = await embedder.embed_query('What is machine learning?')
    print(f'Embedding dimensions: {result.embeddings[0]}')
    print(f'Tokens used: {result.provider_name}')

    # > Embedding dimensions: 1536
    # Embed multiple documents at once
    docs = ["invoice total is 200", "payment due tomorrow"]

    embbs = await embedder.embed_documents(docs)
    print(embbs.embeddings)
    col = client.get_or_create_collection("test_collection")
    col.add(ids=["a", "b"], embeddings=embbs.embeddings, documents=docs)



    query = "is it 150?"
    result = await embedder.embed_query(query)

    res = col.query(query_embeddings=result.embeddings, n_results=2, include=["documents", "distances"])
    print(res)


asyncio.run(fetch_data())


