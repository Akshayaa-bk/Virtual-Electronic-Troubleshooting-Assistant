# Virtual-Electronic-Troubleshooting-Assistant
The Virtual Electronic Troubleshooting Assistant is an AI-powered Streamlit app that enables users to upload technical PDFs and query them via a natural language chat. Built with LangChain, Hugging Face, Pinecone and hybrid search (BM25 + embeddings), it provides accurate, context-aware responses from uploaded documents.


![image](https://github.com/user-attachments/assets/422b63cc-bf2e-42e7-a4e0-fa6a51bd932d)
![image](https://github.com/user-attachments/assets/d129f54d-74e9-4fab-a49e-2e5c115d5a17)


![image](https://github.com/user-attachments/assets/38b6cd6e-2f7d-4d70-b126-38bff67f1035)
![image](https://github.com/user-attachments/assets/8e420a47-2ff5-465e-a909-d6e45a2825af)





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
