# 🚀 Simple RAG Document Q&A API

A beginner-friendly project for learning the basics of **Retrieval-Augmented Generation (RAG)** using FastAPI, LangChain, and ChromaDB.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** is like giving an AI an **"Open-Book Exam."**

Normally, AI models answer from their memory. However, AI can sometimes make things up (hallucinate) or have outdated information.

With **RAG**, the AI first **reads your documents**, finds the most relevant content, and then answers **only from that content**. This ensures the answer is grounded in your facts!

---

## 🎯 Why this project exists?

This project is built to showcase the **fundamental blocks** of a RAG pipeline without the complexity of production-level orchestration. It's meant to be clean, simple, and educational.

### 🏗️ Project Structure

```
simple-rag-document-api/
  app/
    main.py        # 🚦 FastAPI routes (the 'front door')
    config.py      # ⚙️  Loading environment settings
    loader.py      # 📂 Reading your local .txt files
    rag.py         # 🧠 The RAG engine (Retrieve + Ground + Generate)
    schemas.py     # 📋 Data structures (Pydantic models)
  tests/
    conftest.py    # 🔧 Shared fixtures & mocks (auto-loaded by pytest)
    test_health.py # ✅ Tests for GET /health
    test_ingest.py # ✅ Tests for POST /ingest
    test_ask.py    # ✅ Tests for POST /ask
    test_loader.py # ✅ Tests for the document loader
    test_schemas.py# ✅ Tests for all Pydantic schemas
  data/
    docs/          # 📑 Put your .txt files here!
      ai_agents.txt
      rag_basics.txt
  pytest.ini       # 🧪 Pytest configuration
  requirements.txt # 📦 Python dependencies
  README.md        # 📍 This guide
```

---

## 🛠️ How to Run

### 1. Prerequisites
- Python 3.8+ installed
- An **OpenAI API Key**

### 2. Setup
Create a `.env` file and add your key:
```bash
cp .env.example .env
```
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Application
Run from the `simple-rag-document-api/` directory:
```bash
uvicorn app.main:app --reload
```

The API will be available at: **http://localhost:8000**  
Interactive docs (Swagger UI): **http://localhost:8000/docs**

---

## 🧪 Running Tests

> **No OpenAI API key is required to run tests.** All external services (OpenAI, ChromaDB) are mocked.

### Step 1 — Install test dependencies
If you haven't already, install everything from `requirements.txt`:
```bash
pip install -r requirements.txt
```
This installs `pytest`, `pytest-asyncio`, and `httpx` alongside the app dependencies.

### Step 2 — Navigate to the project folder
Make sure you are inside the `simple-rag-document-api/` directory:
```bash
cd simple-rag-document-api
```

### Step 3 — Run all tests
```bash
pytest
```
Pytest will automatically discover every file matching `tests/test_*.py`.

---

### 🔍 Other Useful Test Commands

| Command | What it does |
|---|---|
| `pytest` | Run all tests |
| `pytest -v` | Verbose — shows each test name with PASS/FAIL |
| `pytest tests/test_health.py` | Run only the health endpoint tests |
| `pytest tests/test_ask.py` | Run only the /ask endpoint tests |
| `pytest tests/test_ingest.py` | Run only the /ingest endpoint tests |
| `pytest tests/test_loader.py` | Run only the document loader unit tests |
| `pytest tests/test_schemas.py` | Run only the Pydantic schema tests |
| `pytest -k "health"` | Run any test whose name contains "health" |
| `pytest --tb=long` | Show full tracebacks on failures |
| `pytest -x` | Stop immediately on the first failure |

---

### 🗺️ What Each Test File Covers

#### `tests/test_health.py` — Health Endpoint (5 tests)
Tests the `GET /health` endpoint:
- Returns HTTP 200
- Response has `status` and `message` fields
- `status` value is `"up"`
- `message` is a non-empty string
- Content-Type is `application/json`

#### `tests/test_ingest.py` — Ingest Endpoint (6 tests)
Tests the `POST /ingest` endpoint:
- Returns HTTP 200 when documents are found
- Response has `message`, `num_documents`, `num_chunks` fields
- Counts match what was ingested
- Returns HTTP 404 when `data/docs/` folder is empty
- 404 error has a descriptive `detail` message
- Returns HTTP 500 on unexpected errors

#### `tests/test_ask.py` — Ask Endpoint (9 tests)
Tests the `POST /ask` endpoint (core RAG flow):
- Returns HTTP 200 for a valid question
- Response has `question`, `answer`, `documents` fields
- `question` field echoes what was sent
- `documents` list is populated from ChromaDB results
- Each document has `file_name` and `page_content`
- Returns polite fallback message when no docs match
- Returns HTTP 422 for missing `question` field
- Handles empty string edge case gracefully
- Returns HTTP 500 when the LLM throws an exception

#### `tests/test_loader.py` — Document Loader (7 tests)
Tests the `load_local_text_docs()` function directly (no mocking — uses real temp directories):
- Returns `[]` if folder doesn't exist
- Returns `[]` if folder is empty
- Loads a single `.txt` file correctly
- Loads multiple `.txt` files
- Ignores non-`.txt` files (`.md`, `.csv`, etc.)
- Each document has a `source` metadata key
- `page_content` matches what was written to the file

#### `tests/test_schemas.py` — Pydantic Schemas (13 tests)
Tests all request/response models directly:
- `AskRequest`: valid question, missing question, type check
- `DocumentSource`: valid data, missing fields raise errors
- `AskResponse`: valid response, empty documents list, missing fields
- `IngestResponse`: valid response, type check, missing fields
- `HealthResponse`: valid response, missing required fields

---

### ✅ Expected Output

After running `pytest`, you should see output like this:

```
========================== test session starts ==========================
platform win32 -- Python 3.11.x, pytest-8.x.x
collected 40 items

tests/test_health.py::TestHealthEndpoint::test_health_returns_200          PASSED
tests/test_health.py::TestHealthEndpoint::test_health_response_structure   PASSED
tests/test_health.py::TestHealthEndpoint::test_health_status_value         PASSED
tests/test_health.py::TestHealthEndpoint::test_health_message_is_string    PASSED
tests/test_health.py::TestHealthEndpoint::test_health_content_type_is_json PASSED
tests/test_ingest.py::TestIngestEndpoint::test_ingest_success              PASSED
...
tests/test_schemas.py::TestHealthResponse::test_missing_message_raises_error PASSED

========================== 40 passed in 3.21s ==========================
```

---

## 📄 Example API Response

**Question:** *"What is RAG in simple steps?"*

```json
{
  "question": "What is RAG in simple steps?",
  "answer": "Retrieval-Augmented Generation (RAG) works in three simple steps: 1. Retrieval: Finding the most relevant snippets from documents. 2. Grounding: Adding these snippets as 'context' to your question. 3. Generation: The LLM reads the context and your question, then generates an answer ONLY from that context.",
  "documents": [
    {
      "file_name": "rag_basics.txt",
      "page_content": "Retrieval-Augmented Generation (RAG) is a technique for grounding AI answers..."
    }
  ]
}
```

---

## ⚖️ License
This is an open-source educational project. Feel free to use and learn from it!
