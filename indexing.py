import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

from llama_index.core import Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Load env variables
load_dotenv()

# ---- SUPABASE CLIENT ----
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# ---- LOCAL EMBEDDING MODEL ----
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"
)
embed_model = Settings.embed_model

# ---- CHUNKER ----
node_parser = SentenceSplitter(
    chunk_size=900,
    chunk_overlap=80
)

# ---- LOAD STRUCTURED DOCS ----
with open("output/structured_docs.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

# ---- CHUNK → EMBED → UPLOAD ----
all_chunks = []
# 1. Generate all chunks
for item in raw_docs:
    doc = Document(text=item["text"], metadata=item["metadata"])
    chunks = node_parser.get_nodes_from_documents([doc])
    all_chunks.extend(chunks)

print(f"📄 Generated {len(all_chunks)} chunks.")

# 2. Batch embed all chunks
chunk_contents = [chunk.get_content() for chunk in all_chunks]
print(f"🧠 Embedding {len(chunk_contents)} chunks in batches...")
embeddings = embed_model.get_text_embedding_batch(chunk_contents, show_progress=True)
print("✅ Embeddings generated.")

# 3. Prepare batch for Supabase
batch_to_upsert = []
for i, chunk in enumerate(all_chunks):
    batch_to_upsert.append(
        {
            "content": chunk.get_content(),
            "embedding": embeddings[i],
            "metadata": chunk.metadata
        }
    )

# 4. Upsert to Supabase in smaller batches
supabase_batch_size = 200
table_name = "documents"
print(f"📦 Uploading {len(batch_to_upsert)} entries to '{table_name}' in batches of {supabase_batch_size}...")

for i in range(0, len(batch_to_upsert), supabase_batch_size):
    batch = batch_to_upsert[i:i+supabase_batch_size]
    response = supabase.table(table_name).upsert(batch).execute()
    print(f"  -> Uploaded batch {i//supabase_batch_size + 1}")

print(f"🎉 Done! Vector index stored in Supabase table '{table_name}'.")
