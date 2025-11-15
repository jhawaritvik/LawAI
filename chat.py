# chat.py
import os
import time
from dotenv import load_dotenv
from google import genai
from retrieval import retrieve, build_context

load_dotenv()

client = genai.Client(api_key=os.getenv("LAWAI_GEMINI_KEY"))

def ask(query, return_chunks=False, max_retries=5):
    # 1) Retrieve relevant chunks
    results = retrieve(query, top_k=10)
    contexts = [doc["content"] for doc in results]
    context_block = build_context(results)

    # 2) Construct prompt
    prompt = f"""
You are a legal assistant. Answer using only the context.

Context:
{context_block}

Question:
{query}

Answer in clear, concise language. Cite Article numbers if relevant.
"""

    # ------------------------------------------------------
    # 3) Retry logic for Gemini
    # ------------------------------------------------------
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            answer = response.text.strip()
            break  # success → exit retry loop

        except Exception as e:
            print(f"[Gemini Error] Attempt {attempt}/{max_retries}: {e}")

            if attempt == max_retries:
                print("❌ Gemini failed after max retries. Returning fallback answer.")
                answer = "The model is currently unavailable. Please try again later."
                break

            # Exponential backoff
            sleep_time = 2 ** attempt
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    # ------------------------------------------------------
    # 4) Return chunks for RAGAS if needed
    # ------------------------------------------------------
    if return_chunks:
        return {
            "answer": answer,
            "contexts": contexts
        }

    return answer


# -----------------------------
# Manual CLI Chat Mode
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\n⚖️ Ask a legal question: ")
        if q.lower() in ["exit", "quit"]:
            break

        answer = ask(q)
        print("\n🟢 Answer:\n")
        print(answer)
