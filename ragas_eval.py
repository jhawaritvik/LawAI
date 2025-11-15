# ragas_eval.py

import os
import pandas as pd
import asyncio

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import llm_factory
from google import genai

from retrieval import retrieve
from chat import ask     # your Gemini-powered answer generator


# -----------------------------
# Load dataset (CSV)
# -----------------------------
def load_dataset(path):
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError("CSV must contain a 'query' column.")
    # 'reference' is optional but very useful
    return df.to_dict(orient="records")


# -----------------------------
# Per-sample evaluation logic
# -----------------------------
async def process_row(row, evaluator_llm):
    query = row["query"]
    reference = row.get("reference", None)

    # ---- Retrieve contexts from Supabase ----
    retrieved = retrieve(query, top_k=8)
    contexts = [doc["content"] for doc in retrieved]

    # ---- Generate answer using Gemini ----
    answer = await asyncio.get_event_loop().run_in_executor(None, ask, query)

    # ---- Create RAGAS Sample ----
    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        reference=reference,
        retrieved_contexts=contexts
    )

    # ---- Evaluate metrics ----
    metrics_to_eval = [
        faithfulness,
        answer_relevancy,
    ]
    if sample.reference:
        metrics_to_eval.append(context_precision)
        metrics_to_eval.append(context_recall)

    # Create a Ragas Dataset from the single sample
    from datasets import Dataset
    ragas_dataset = Dataset.from_dict({
        "question": [sample.user_input],
        "answer": [sample.response],
        "contexts": [sample.retrieved_contexts],
        "ground_truths": [[sample.reference]] if sample.reference else [[]]
    })

    result = evaluate(ragas_dataset, metrics=metrics_to_eval, llm=evaluator_llm)
    result_scores = result.to_pandas().iloc[0].to_dict()

    return {
        **row,
        "answer": answer,
        **result_scores
    }


# -----------------------------
# Main
# -----------------------------
def main():
    # Load the queries
    data = load_dataset("queries.csv")

    # Wrap Gemini as evaluation LLM
    gemini_client = genai.Client(api_key=os.getenv("LAWAI_GEMINI_KEY"))
    evaluator_llm = llm_factory(
        "gemini-2.5-flash",
        client=gemini_client
    )

    async def runner():
        results = []
        for row in data:
            out = await process_row(row, evaluator_llm)
            results.append(out)

        # Save results to CSV
        out_df = pd.DataFrame(results)
        out_df.to_csv("ragas_results.csv", index=False)
        print("\n✔ Evaluation complete. Saved to ragas_results.csv\n")

    asyncio.run(runner())


if __name__ == "__main__":
    main()
