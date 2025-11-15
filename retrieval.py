# retrieval.py
import os
from supabase import create_client
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Supabase client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

def retrieve(query, top_k=5):
    q_embed = embed_model.get_text_embedding(query)

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": q_embed,
            "match_threshold": 0.6,  # Add a threshold for semantic relevance
            "match_count": top_k
        }
    ).execute()

    return response.data

def build_context(results):
    return "\n\n---\n\n".join([doc["content"] for doc in results])
