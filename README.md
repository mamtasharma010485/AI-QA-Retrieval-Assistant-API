🤖AI QA Retrieval Assistant API 🚀
🌟 About This Project
This project is a Production-Ready RAG (Retrieval-Augmented Generation) API Server designed to transform local documents into a searchable, interactive knowledge base.

✨ Why use this?
Traditional LLMs (like GPT-4) can sometimes "hallucinate" or provide outdated information. By using RAG, this API:

✅ Grounds Responses: The AI only answers using the specific documents you provide.
✅ Zero Hallucination: If the answer isn't in your files, it won't make one up.
✅ Privacy-First: Your documents stay on your local machine; only relevant snippets are sent to the LLM.
✅ Dynamic Knowledge: Simply drop new .txt files into the data/docs folder, re-ingest, and your AI is immediately smarter.
This server is perfect for building Internal Q&A bots, Documentation assistants, or Knowledge Management systems.

🧩 What is RAG? (Retrieval-Augmented Generation)
RAG is a technique that gives a Large Language Model (LLM) access to data it wasn't trained on. It works through three essential steps:

🔍 Retrieval: When you ask a question, the system searches through your documents to find the most relevant "chunks" of text.
🚀 Augmentation: The system augments your original question by merging it with the retrieved text chunks. This creates a single, context-rich prompt for the AI.
🧠 Generation: The LLM receives this augmented prompt and generates a factual, grounded answer based only on the provided content.
Think of it as giving the AI an open-book exam. Instead of relying on its memory, it looks up the answer in your provided notes and synthesizes a response.

🏗️ Architecture Overview
This application follows a modern RAG pipeline:

📂 Document Loading: Reads text files from the local filesystem (data/docs/).
✂️ Chunking: Splits documents into manageable pieces using RecursiveCharacterTextSplitter.
🔢 Embedding: Uses OpenAI's text-embedding-3-small to convert chunks into vectors.
🗄️ Vector Store: Persistently saves vectors to local storage (./chroma_db) using ChromaDB.
🔍 Retrieval: Leverages similarity search to find the most relevant document chunks based on user questions.
🧠 Generation: Crafts a strict prompt to prevent hallucinations and generates answers using OpenAI's gpt-4o-mini.
📁 Project Structure
RAGAPISERVER/
├── 📂 app/               # Application Source Code
│   ├── ⚙️ config.py        # configuration & environment parsing
│   ├── 📄 loader.py        # Reads & splits .txt documents
│   ├── 🚀 ingest.py        # Orchestrates document embedding
│   ├── 🌐 main.py          # FastAPI application & HTTP endpoints
│   ├── 🏗️ prompts.py       # Generates context-grounded LLM prompts
│   ├── 🧠 rag.py           # Integrates LLM answer generation
│   ├── 🔍 retriever.py     # Handles searching vector DB
│   ├── 📜 schemas.py       # Pydantic data validation models
│   └── 🛠️ utils.py         # Helper functions / logging
├── 📂 data/docs/         # 📚 Knowledge Base (Sample Text Files)
├── 📂 postman/           # 📮 Postman Collections & Environments
├── 📂 tests/             # 🧪 Pytest Test Suite
├── 📄 .env.example       # 🔑 Environment variables template
├── 📄 requirements.txt   # 📦 Python dependencies
└── 📄 README.md          # 📖 This file
📋 Prerequisites
Before you begin, ensure you have the following installed:

🐍 Python 3.11+: Download here
🔑 OpenAI API Key: Get your key here
📮 Postman: For testing the API endpoints
📁 Git: To clone/manage the repository (optional)
🚀 Quick Setup Instructions
1. 🐍 Create Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
2. 📦 Install Dependencies
pip install -r requirements.txt
3. 🔑 Configure Environment Variables
Copy the example config:
cp .env.example .env
Open .env and add your OpenAI API Key:
OPENAI_API_KEY=sk-proj-your-api-key-here or you can use ollama locally
🏃 Running the Server
Run the API using Uvicorn:

uvicorn app.main:app --reload
The server will start at http://localhost:8000.

🏠 API Base: http://localhost:8000
📖 Swagger UI: http://localhost:8000/docs
🕹️ Usage Workflow (Must Follow)
The vector database starts empty. Follow these steps in order:

🛠️ Step 1: Ingest Documents
Process the .txt files in data/docs/ and build the vector index.

curl -X POST http://localhost:8000/ingest
💬 Step 2: Ask Questions
The AI will answer based only on the ingested context.

curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"question": "What is an AI Agent?", "top_k": 3}'
📮 Postman Integration — Step-by-Step
📥 Step 1 — Import the Collection
Open Postman → Click the Import button.
Select postman\rag_api_collection.json.
✅ You will see "RAG API Server" in the Collections sidebar.
🌍 Step 2 — Import the Environment
Click Import again → Select postman\rag_api_environment.json.
✅ The "RAG API Local" environment is now available.
⚡ Step 3 — Activate the Environment
Top-right dropdown → Select "RAG API Local".
✅ This sets {{base_url}} = http://localhost:8000.
🔄 Step 4 — Run the Requests in Order
Execute the requests in this exact order:

Order	Request Name	Method	Endpoint	Purpose
1️⃣	Health Check	GET	/health	Check if API is alive
2️⃣	Clear Store	POST	/reset	Wipe existing vector data
3️⃣	Ingest Docs	POST	/ingest	🚀 Load knowledge base
4️⃣	Ask AI	POST	/ask	Get grounded answers
5️⃣	Retrieve	POST	/retrieve	Debug raw search results
🎯 How to Test Each Demo Question in Postman
Go to "4. Ask Question" in Postman and use these JSON bodies:

Topic	Question	Expected Content
Agents	What is an AI Agent?	Reasoning, planning, memory, and tools.
RAG	How does RAG reduce hallucination?	Grounding answers in retrieved context.
MCP	What is Model Context Protocol?	Standardizing AI helper integrations.
Metrics	Key LLM evaluation metrics?	Faithfulness, Relevance, Ragas.
Testing	Cypress AI Testing?	Self-healing and intent-driven tests.
⚠️ Critical Startup Rules
🛑 Rule 1 — Always Call /ingest Before /ask
The vector store starts empty. If you don't ingest first, you'll get an "information not available" response.

🔄 Rule 2 — Restart Server After Pip Installs
Uvicorn reload doesn't always catch new package installs. Ctrl+C and restart after pip install.

🧪 Running Tests
python -m pytest tests/test_api.py -v
🐛 Troubleshooting
Problem	Cause	Solution
documents: []	No ingestion done.	Run POST /ingest in Postman.
500 Server Error	Missing API Key.	Check .env file for OPENAI_API_KEY.
ModuleNotFoundError	Broken venv.	Re-run pip install -r requirements.txt.
✨ Features Included
✅ Full RAG Pipeline (Ingest -> Retrieve -> Generate)
✅ ChromaDB Persistence
✅ Metadata Filtering by topic
✅ Hallucination Protection system prompt
✅ Postman Ready with collections & scripts
✅ Automated Tests included
Built with ❤️ for AI Engineers 🚀
