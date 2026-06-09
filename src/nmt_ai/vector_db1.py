import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()

col = client.create_collection("embeds")

docs = ["invoice total is 200", "payment due tomorrow"]
embs = model.encode(docs).tolist()

col.add(ids=["a", "b"], embeddings=embs, documents=docs)

query = "when is payment due"
q_emb = model.encode([query]).tolist()

res = col.query(query_embeddings=q_emb, n_results=2)
print(res)
