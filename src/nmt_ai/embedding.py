import chromadb
from chromadb.utils import embedding_functions
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.huggingface import HuggingFaceModel

# ---------------------------------------------------------
# 1. Setup ChromaDB with Local Hugging Face Embeddings
# ---------------------------------------------------------
# Using a popular local embedding model from the Hugging Face Hub
hf_embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Persistent DB instance
chroma_client = chromadb.PersistentClient(path="./knowledge_db")
collection = chroma_client.get_or_create_collection(
    name="company_docs",
    embedding_function=hf_embedding_func  # Ensures data auto-embeds on read/write
)

# Mock Data Injection (Only run once to populate your DB)
collection.add(
    documents=["Employees can work remotely up to 3 days per week with manager approval."],
    ids=["doc_001"]
)

# ---------------------------------------------------------
# 2. Setup Pydantic AI Agent with Hugging Face LLM
# ---------------------------------------------------------
# Initialize a powerful open-source model via Hugging Face Inference Providers
hf_model = HuggingFaceModel('Qwen/Qwen3-235B-A22B')
agent = Agent(model=hf_model)


# ---------------------------------------------------------
# 3. Create the RAG Search Tool
# ---------------------------------------------------------
@agent.tool
def search_knowledge_base(ctx: RunContext, query: str) -> str:
    """Search the company knowledge base for internal policy context."""
    # Chroma handles embedding extraction under the hood via sentence-transformers
    results = collection.query(
        query_texts=[query],
        n_results=2
    )

    # Extract text findings
    documents = "\n".join(results.get('documents', [[]])[0])
    return documents or "No relevant policy documents found."


# ---------------------------------------------------------
# 4. Execute the Query
# ---------------------------------------------------------
if __name__ == "__main__":
    user_prompt = "How many days am I allowed to work from home?"

    # Run the agent synchronously 
    response = agent.run_sync(user_prompt)
    print("Agent Response:\n", response.data)
