import os
from pathlib import Path
from typing import List, Dict, Any

# LangChain components for the RAG pipeline
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- Configuration & Setup ---

# Define where to store the vector database locally
PERSIST_DIRECTORY = "data/chroma_db"

def get_vector_store() -> Chroma:
    """
    Initializes or retrieves the local ChromaDB vector store.
    
    A Vector Store is like a specialized database that can search by 'meaning'
    rather than just keywords.
    """
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

def ingest_documents(documents: List[Document]) -> Dict[str, Any]:
    """
    Ingests raw documents: Splits them into chunks -> Embeds them -> Stores in Chroma.
    
    The Ingestion Process:
    1.  Chunking: If a document is too long, we break it into smaller pieces.
    2.  Embedding: Convert each piece of text into a list of numbers (a vector).
    3.  Storage: Save these vectors into a searchable database.
    """
    if not documents:
        return {"num_documents": 0, "num_chunks": 0}

    # Split documents into 1000-character chunks with a small overlap
    # Overlap helps preserve context between continuous chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Store specifically named documents in the vector store
    vectorstore = get_vector_store()
    vectorstore.add_documents(chunks)
    vectorstore.persist()  # Save changes to disk

    return {
        "num_documents": len(documents),
        "num_chunks": len(chunks)
    }

async def ask_question(question: str) -> Dict[str, Any]:
    """
    The main RAG query flow:
    1.  Retrieve: Find the top 3 most relevant chunks based on the question.
    2.  Ground: Insert these chunks into a strict prompt.
    3.  Generate: Ask the AI for an answer based ONLY on those chunks.
    """
    # 1. RETRIEVAL
    vectorstore = get_vector_store()
    retrieved_docs = vectorstore.similarity_search(question, k=3)

    # If no documents are found, tell the user politely
    if not retrieved_docs:
        return {
            "answer": "The answer is not available in the provided documents.",
            "documents": []
        }

    # 2. PROMPT BUILDING
    # We combine the retrieved chunks into a single context string
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # This prompt tells the AI EXACTLY what its boundaries are
    system_prompt = (
        "You are a helpful assistant. Use ONLY the provided context to answer the question. "
        "Strictly follow these rules:\n"
        "1. If the answer is not in the context, say: 'The answer is not available in the provided documents.'\n"
        "2. Do NOT use your own knowledge or memory.\n"
        "3. Provide a concise and direct answer based on the documents.\n\n"
        "CONTEXT:\n"
        "{context}"
    )

    # Use a chat prompt template for clear structure
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # 3. GENERATION
    # Initialize the LLM (OpenAI)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Format the prompt and send it to the AI
    final_prompt = prompt_template.format_messages(
        context=context_text,
        question=question
    )

    response = await llm.ainvoke(final_prompt)

    # 4. PREPARE RESPONSE
    # Map the retrieved documents to a cleaner format (source filename and content)
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "file_name": Path(doc.metadata.get("source", "unknown")).name,
            "page_content": doc.page_content[:200] + "..."  # Truncate content for brevity
        })

    return {
        "answer": response.content,
        "documents": sources
    }
