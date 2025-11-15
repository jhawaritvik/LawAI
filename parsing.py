from dotenv import load_dotenv
import os
from llama_cloud_services import LlamaParse
import json

load_dotenv()
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not api_key:
    raise RuntimeError("Set LLAMA_CLOUD_API_KEY in .env")

parser = LlamaParse(
    api_key=api_key,  
    num_workers=1,       # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="en",       # optionally define a language, default=en
)

result = parser.parse("C:\\Python\\LawAI\\constitution.pdf")

# get the llama-index text documents
text_documents = result.get_text_documents(split_by_page=False)

# Save parsed output
os.makedirs("output", exist_ok=True)
with open("output/parsed.json", "w", encoding="utf-8") as f:
    json.dump([{"text": d.text} for d in text_documents], f, ensure_ascii=False, indent=2)

print(f"Parsing complete. Parsed sections: {len(text_documents)}")
print("Saved → output/parsed.json")