# Law AI - RAG System for Legal Documents

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions about the Constitution of India. It uses LlamaParse for document parsing, a Hugging Face model for embeddings, Supabase (Postgres with pgvector) for vector storage, and Google's Gemini for generating answers.

## Project Workflow

The project is structured as a sequential pipeline. You must run the scripts in the following order:

1.  **Parsing (`parsing.py`)**:
    *   The `constitution.pdf` is parsed using LlamaParse.
    *   The raw extracted text is saved to `output/parsed.json`.

2.  **Metadata Extraction (`metadata_extraction.py`)**:
    *   The parsed text is loaded from `output/parsed.json`.
    *   The script identifies the boundaries of different "Parts" and "Articles" within the constitution text.
    *   It then chunks the document and intelligently attaches the correct metadata (Part number, Article number, etc.) to each text chunk.
    *   The result is a list of structured documents, which are saved to `output/structured_docs.json`.

3.  **Indexing (`indexing.py`)**:
    *   The structured documents from the previous step are loaded.
    *   Each chunk of text is converted into a vector embedding using the `BAAI/bge-large-en-v1.5` model.
    *   The text chunks, their corresponding metadata, and their vector embeddings are uploaded and stored in a Supabase PostgreSQL table named `documents`. This table functions as our vector store.

4.  **Chat / Q&A (`chat.py`)**:
    *   This is the main entry point for asking questions.
    *   When you ask a question, the script first embeds your query into a vector.
    *   It then queries the Supabase table to retrieve the most relevant document chunks (the "context").
    *   Finally, it constructs a prompt containing your question and the retrieved context, sends it to the Gemini Pro model, and prints the generated answer.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Create Environment File**:
    Create a `.env` file in the root directory and add the following keys. The `.gitignore` file is already configured to ignore this file.

    *   `LLAMA_CLOUD_API_KEY`: Your API key for LlamaParse, used for parsing PDF documents. Obtain it from [Llama Cloud](https://cloud.llamaindex.ai/).
    *   `SUPABASE_URL`: The URL of your Supabase project. You can find this in your Supabase project settings under "API".
    *   `SUPABASE_KEY`: Your Supabase "anon" key (public key). Also found in your Supabase project settings under "API".
    *   `LAWAI_GEMINI_KEY`: Your API key for Google Gemini, used for generating responses. Obtain it from [Google AI Studio](https://aistudio.google.com/app/apikey).
    ```
    LLAMA_CLOUD_API_KEY="your-llama-cloud-api-key"
    SUPABASE_URL="your-supabase-project-url"
    SUPABASE_KEY="your-supabase-api-key"
    LAWAI_GEMINI_KEY="your-gemini-api-key"
    ```

## How to Run

1.  **Run the Data Pipeline**:
    Execute the scripts in order to process the PDF and populate your vector database.
    ```bash
    python parsing.py
    python metadata_extraction.py
    python indexing.py
    ```

2.  **Start the Chat Interface**:
    Once the indexing is complete, you can start asking questions.
    ```bash
    python chat.py
    ```
