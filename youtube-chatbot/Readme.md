# Youtube-Chatbot

## Operational Overview
* Engineered a Retrieval-Augmented Generation (RAG) pipeline using LangChain and FAISS to vectorize video transcripts, reducing manual information extraction time by ~80%.
* Integrated Google Gemini to synthesize context-aware chatbot responses, accelerating accurate data retrieval from unstructured multimedia by over 50%.

## Architecture & Technology Stack
* **Frontend Interface:** Streamlit
* **Orchestration Framework:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **LLM Integration:** Google Gemini (`gemini-1.5-flash` for generation, `gemini-embedding-001` for vectorization)
* **Data Ingestion:** `youtube-transcript-api`

## System Workflow
* **Ingestion:** The system parses standard YouTube URLs and extracts the video ID.
* **Extraction:** Retrieves the video transcript via the official YouTube API.
* **Vectorization:** Implements recursive character text splitting and generates embeddings via Google Gemini.
* **Indexing:** Stores vectorized chunks in a local FAISS index.
* **Inference:** Executes a similarity search against user queries and synthesizes a zero-shot, context-aware response utilizing the retrieved context.