# chat.py
import os
import time
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from retrieval import retrieve, build_context

load_dotenv()

client = Groq()

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
    # 3) Retry logic for Groq
    # ------------------------------------------------------
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal assistant. Answer using only the provided context.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            answer = completion.choices[0].message.content.strip()
            break  # success → exit retry loop

        except BadRequestError as e:
            # Permanent configuration error (e.g. model_decommissioned) – don't retry
            print(f"[Groq Error] Permanent error: {e}")
            answer = "The model configuration is invalid or deprecated. Please contact the system administrator."
            break

        except Exception as e:
            print(f"[Groq Error] Attempt {attempt}/{max_retries}: {e}")

            if attempt == max_retries:
                print("❌ Groq failed after max retries. Returning fallback answer.")
                answer = "The model is currently unavailable. Please try again later."
                break

            # Exponential backoff for transient errors
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
