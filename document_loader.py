import os
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

def load_documents(data_dir: str) -> List[Document]:
    """Loads text documents from the specified directory and splits them."""
    docs = []
    
    if not os.path.exists(data_dir):
        return docs
        
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Use filename without extension as the topic
            topic = filename.replace(".txt", "").replace("_", " ").title()
            
            # Create a base document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "file_name": filename,
                    "topic": topic
                }
            )
            docs.append(doc)
            
    return docs

def split_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks suitable for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Assign unique chunk IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['file_name']}_chunk_{i}_{uuid.uuid4().hex[:8]}"
        
    return chunks
