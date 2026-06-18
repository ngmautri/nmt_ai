import chromadb
import pandas as pd

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

df = pd.read_excel(DATA_PATH + "\\3600.xlsx")

# 2. Combine columns into one text string per row
texts = (str(df["Vendor"]) + " " + df["Name 1"] + " " + df["VAT registration no."]).tolist()
ids = (df["Vendor"]).tolist()
print(ids)



async def build_db():
    # Embed a search query

    client = chromadb.PersistentClient(path=VENDOR_DB)

    # > Embedding dimensions: 1536
    # Embed multiple documents at once
    docs = texts

    embbs = await embedder.embed_documents(docs)
    # print(embbs.embeddings)
    col = client.get_or_create_collection("test_collection")
    col.add(ids=ids, embeddings=embbs.embeddings, documents=docs)


asyncio.run(build_db())


