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
import settings
from nmt_ai.settings import DATA_PATH, VENDOR_DB

emb_model = OpenAIEmbeddingModel(
    model_name="embeddinggemma:latest",
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
embedder = Embedder(emb_model)


async def fetch_data(q):
    # Embed a search query

    print(q)

    client = chromadb.PersistentClient(path=VENDOR_DB)
    col = client.get_or_create_collection("vendor_collection")


    result = await embedder.embed_query(q)

    res = col.query(query_embeddings=result.embeddings, n_results=1, include=["documents", "distances"])
    print(res)
    return res["ids"]

#
# asyncio.run(fetch_data("Immobilien"))


