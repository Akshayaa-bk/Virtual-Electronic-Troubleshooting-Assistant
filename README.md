# Virtual-Electronic-Troubleshooting-Assistant
The Virtual Electronic Troubleshooting Assistant is an AI-powered Streamlit app that enables users to upload technical PDFs and query them via a natural language chat. Built with LangChain, Hugging Face, Pinecone and hybrid search (BM25 + embeddings), it provides accurate, context-aware responses from uploaded documents.


![image](https://github.com/user-attachments/assets/422b63cc-bf2e-42e7-a4e0-fa6a51bd932d)
![image](https://github.com/user-attachments/assets/d129f54d-74e9-4fab-a49e-2e5c115d5a17)


![image](https://github.com/user-attachments/assets/38b6cd6e-2f7d-4d70-b126-38bff67f1035)
![image](https://github.com/user-attachments/assets/8e420a47-2ff5-465e-a909-d6e45a2825af)





## 🌟 Features

### 🔧 Technical Troubleshooting Chat

- **PDF Document Processing**
  - Upload and process multiple technical manuals and documentation in PDF format
  - Automatic text extraction, chunking, and semantic indexing
  - Persistent storage of document contents in a Pinecone vector database

- **Intelligent Search & Retrieval**
  - Hybrid search combining dense vector embeddings with sparse BM25 retrieval
  - Context-aware document retrieval prioritizing most relevant content
  - Source attribution for retrieved information

- **Conversational AI Assistant**
  - Natural language interface for technical support
  - Context-aware responses based on uploaded documentation
  - Persistent chat history with clear formatting

- **Knowledge Base Management**
  - On-demand document processing
  - Real-time indexing of new documents
  - Document database reset functionality

### 📦 Inventory Management System

- **Comprehensive Inventory Dashboard**
  - Real-time inventory statistics and metrics
  - Visual indicators for stock levels and alerts
  - Tabbed interface for different inventory views

- **Smart Inventory Filtering**
  - Automatic identification of low stock items (≤5 units)
  - Out of stock items tracking
  - Full-text search across product names, spare parts, and descriptions

- **Stock Management**
  - Easy quantity updates for inventory items
  - Bulk import from Excel spreadsheets
  - Complete inventory database reset with confirmation

- **Reporting Capabilities**
  - Downloadable inventory reports in CSV format
  - Low stock alerts with visual indicators
  - Stock level forecasting

- **User-Friendly Interface**
  - Clean, responsive design
  - Intuitive navigation with tabbed sections
  - Success/warning notifications and confirmation dialogs

### 🔗 Integration Features

- **Inventory-Aware Troubleshooting**
  - Automatic part identification in user queries
  - Replacement part suggestions based on inventory availability
  - Integration between troubleshooting chat and inventory system

- **Natural Language Inventory Queries**
  - Detect replacement and inventory questions in natural language
  - Extract part names from user queries
  - Provide inventory status within the chat interface


## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- Pinecone (Vector DB)
- BM25 Encoder (Sparse Retrieval)
- PyPDF2 (PDF text extraction)
- dotenv (Environment variables management)
