from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction


ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name="embeddinggemma:latest",
)

embeddings = ollama_ef(["This is my first text to embed",
                        "This is my second document"])

collection = client.create_collection(name="my_collection")