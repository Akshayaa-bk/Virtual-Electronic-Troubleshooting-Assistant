# Virtual-Electronic-Troubleshooting-Assistant
The Virtual Electronic Troubleshooting Assistant is an AI-powered Streamlit app that enables users to upload technical PDFs and query them via a natural language chat. Built with LangChain, Hugging Face, Pinecone and hybrid search (BM25 + embeddings), it provides accurate, context-aware responses from uploaded documents.

![image](https://github.com/user-attachments/assets/8cbb4b5f-1852-4aab-ab27-0a398a886963)



## Features

* Upload and process multiple PDFs containing technical documentation.
* Hybrid Search: Combines BM25 sparse retrieval and Dense embeddings for better document search.
* Uses Mistral-7B-Instruct via Hugging Face Inference API for LLM-based question answering.
* Retrieval-Augmented Generation (RAG) approach for precise and context-aware responses.
* Interactive chat interface powered by Streamlit.
* Chat History Management: Maintains conversation context during your session

## Requirements

- Python 3.8+
- Pinecone account with API key
- HuggingFace account with API key

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- Pinecone (Vector DB)
- BM25 Encoder (Sparse Retrieval)
- PyPDF2 (PDF text extraction)
- dotenv (Environment variables management)
