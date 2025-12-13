# 📄 PDF Chatbot API (FastAPI + LangChain + Hugging Face)

A session-based PDF chatbot backend built using FastAPI, LangChain, ChromaDB, and Hugging Face LLMs.
This API allows users to upload a PDF, ask questions about its content, and maintain conversational memory per session.

### 🚀 Features

- 📥 Upload PDF and generate embeddings

- 🧠 Context-aware question answering using vector search

- 💬 Conversation memory per session

- 🧾 Session-based isolation (multiple users supported)

- 🔚 Proper session cleanup

- ⚡ FastAPI-powered REST API


### 🛠 Tech Stack

- Backend: FastAPI

- LLM: Mistral-7B-Instruct (Hugging Face)

- Embeddings: sentence-transformers/all-MiniLM-L6-v2

- Vector Store: ChromaDB (in-memory)

- Framework: LangChain

- Memory: ConversationSummaryMemory

- Language: Python 3.10+

### Project Structure
```
├── app.py                 # Main FastAPI application
├── steps.py               # PDF loading, chunking, prompt template
├── pdfs/                  # Temporary PDF storage
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```

## How to run?

Clone the repo:

```bash
git clone https://github.com/coderkushonline/pdf-chatbot
cd pdf-chatbot
```

Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

Set API Secrets in the .env file:
```.env
HUGGINGFACEHUB_API_TOKEN=YOUR_API_KEY
```

Run the main.py file:
```bash
python main.py
```

Visit localhost:8000/docs and try out the API

## 🧠 How It Works

- PDF is uploaded and split into chunks

- Embeddings are generated and stored in ChromaDB

- Questions trigger:

- Vector similarity search

- Context + conversation memory injection

- LLM response generation

- Memory is summarized and stored per session

- Session cleanup removes all data

## ⚠️ Limitations

- Vector store is in-memory (data lost on restart)

- Not optimized for very large PDFs

- Hugging Face API latency depends on model availability

## 🔮 Future Improvements

- Persistent vector store (disk / cloud)

- Streaming responses

- Authentication & rate limiting

- Frontend UI (React / Next.js)

- Support for DOCX and TXT files

