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
from nmt_ai.settings import DATA_PATH, VENDOR_DB

emb_model = OpenAIEmbeddingModel(
    model_name="embeddinggemma:latest",
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
embedder = Embedder(emb_model)


async def fetch_data(q):
    # Embed a search query

    print(f"Query:{q}")

    client = chromadb.PersistentClient(path=VENDOR_DB)
    col = client.get_or_create_collection("vendor_collection")


    result = await embedder.embed_query(q)

    res = col.query(query_embeddings=result.embeddings, n_results=1, include=["documents", "distances"])
    print(res)
    if len(res) > 0:
        # print (res["documents"][0])

        raw = res["documents"][0]
        s = raw[0]
        parts = [p.strip() for p in s.split(";")]

        d = {}
        for p in parts:
            key, value = p.split(":", 1)
            d[key.strip()] = value.strip()

        # print(d)
        # print(d["Vendor Number"])
        return {"vendor_name_sap": d["Vendor Name"], "vendor_number": d["Vendor Number"]}
    else:
        return None

# asyncio.run(fetch_data("40824537"))


