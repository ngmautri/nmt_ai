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

df = pd.read_excel(DATA_PATH + "\\vendor_input.xlsx")

# 2. Combine columns into one text string per row
texts = df['Document'].tolist()
print(texts[0])
print(len(texts)    )
ids = (df["ID"]).tolist()

id_strings = [str(num) for num in ids]
# print(id_strings)



async def build_db():
    # Embed a search query

    client = chromadb.PersistentClient(path=VENDOR_DB)

    # > Embedding dimensions: 1536
    # Embed multiple documents at once
    docs = texts

    n=0
    for doc in docs:
        embbs = await embedder.embed_documents(doc)
        # print(embbs.embeddings)
        col = client.get_or_create_collection("vendor_collection")

        print(doc)
        col.add(ids=id_strings[n], embeddings=embbs.embeddings, documents=doc)
        n = n + 1


asyncio.run(build_db())


