import pandas as pd
from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.ollama import OllamaProvider

from settings import DATA_PATH,VENDOR_DB


import pandas as pd
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# 1. Load Excel
df = pd.read_excel(DATA_PATH + "\\3600.xlsx")

# 2. Combine columns into one text string per row
texts = (str(df["Vendor"]) + " " + df["Name 1"] + " " + df["VAT registration no."]).tolist()
print(texts)
print(texts[0])

# 3. Initialize ChromaDB
client = chromadb.Client(Settings(
    persist_directory=VENDOR_DB,
))
collection = client.create_collection(name="3600_collection")

emb_model = OpenAIEmbeddingModel(
    model_name="embeddinggemma:latest",
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
embedder = Embedder(emb_model)

# 4. Set up OpenAI embeddings
async def get_embedding(text):


    result = await embedder.embed_query(text)
    print(f'Embedding dimensions: {result.embeddings[0]}')
    print(f'Tokens used: {result.provider_name}')

    return result.data[0].embedding

# 5. Generate embeddings for combined text
embeddings = [get_embedding(t) for t in texts]

# 6. Insert into ChromaDB with metadata
collection.add(
    ids=[str(i) for i in range(len(texts))],
    documents=texts,
    embeddings=embeddings,
    metadatas=df.to_dict(orient="records")  # keep original columns
)

embbs = await embedder.embed_documents(texts)
print(embbs.embeddings)

# 7. Query ChromaDB
results = collection.query(
    query_texts=["renewable energy"],
    n_results=2
)
print(results)
