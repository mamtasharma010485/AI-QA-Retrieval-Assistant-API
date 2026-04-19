import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from typing import List
from langchain_core.documents import Document

def load_local_text_docs(docs_dir: str) -> List[Document]:
    """
    Scans a local folder for .txt files and loads them.
    
    This is the first step in RAG: getting your source data.
    
    Args:
        docs_dir: Path to the folder containing your text files.
        
    Returns:
        A list of LangChain Document objects (text + source metadata).
    """
    documents = []
    
    # Path setup: look for the directory relative to the current working directory
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        print(f"Warning: Folder '{docs_dir}' not found. Please create it and add .txt files.")
        return []

    # Iterate through all .txt files in the given directory
    for file in docs_path.glob("*.txt"):
        try:
            print(f"Reading file: {file.name}")
            loader = TextLoader(str(file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file.name}: {e}")

    print(f"Total documents loaded: {len(documents)}")
    return documents
