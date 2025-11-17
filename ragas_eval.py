# ragas_eval.py

import os
import pandas as pd

from dotenv import load_dotenv
from datasets import Dataset
from langchain_openai import ChatOpenAI

from ragas import evaluate

from chat import ask


# -----------------------------
# Load dataset (CSV)
# -----------------------------
def load_dataset(path):
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError("CSV must contain a 'query' column.")
    # 'reference' is optional but very useful
    return df


# -----------------------------
# Per-sample evaluation logic
# -----------------------------
# -----------------------------
# Main
# -----------------------------
def main():
    # Load .env so Ragas' default LLM can see OPENAI_API_KEY
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in environment or .env file.")

    df = load_dataset("queries.csv")

    questions = []
    answers = []
    contexts_list = []
    ground_truths_list = []
    references_list = []

    for _, row in df.iterrows():
        query = row["query"]
        reference = row.get("reference", None) if "reference" in df.columns else None

        qa = ask(query, return_chunks=True)
        questions.append(query)
        answers.append(qa["answer"])
        contexts_list.append(qa["contexts"])

        if isinstance(reference, str) and reference.strip():
            ground_truths_list.append([reference])
            references_list.append(reference)
        else:
            ground_truths_list.append([])
            references_list.append("")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "reference": references_list,
            "ground_truths": ground_truths_list,
        }
    )

    # Let Ragas use its default LLM (configured via OPENAI_API_KEY)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = evaluate(dataset, llm=llm)
    result_df = result.to_pandas()
    result_df.to_csv("ragas_results.csv", index=False)

    print("✔ Evaluation complete. Saved to ragas_results.csv")
    print("Final RAGAS scores:")
    print(result)


if __name__ == "__main__":
    main()
