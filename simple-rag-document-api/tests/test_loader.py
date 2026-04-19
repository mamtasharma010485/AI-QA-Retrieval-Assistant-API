"""
test_loader.py — Unit tests for the document loader (app/loader.py).

The loader is the very first step of the RAG pipeline: reading .txt files
from disk. These tests use Python's tmp_path fixture to create real temp
directories, so no mocking is needed here.
"""

import pytest
from pathlib import Path
from app.loader import load_local_text_docs


class TestLoadLocalTextDocs:
    """Tests for the load_local_text_docs() function."""

    def test_returns_empty_list_for_missing_directory(self, tmp_path):
        """If the docs folder doesn't exist, return [] instead of crashing."""
        non_existent = tmp_path / "ghost_dir"
        result = load_local_text_docs(str(non_existent))
        assert result == []

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        """An existing but empty folder should return an empty list."""
        empty_dir = tmp_path / "docs"
        empty_dir.mkdir()

        result = load_local_text_docs(str(empty_dir))
        assert result == []

    def test_loads_single_txt_file(self, tmp_path):
        """A single .txt file in the folder must be loaded as one Document."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        (docs_dir / "rag_basics.txt").write_text(
            "RAG stands for Retrieval-Augmented Generation.",
            encoding="utf-8"
        )

        result = load_local_text_docs(str(docs_dir))
        assert len(result) == 1

    def test_loads_multiple_txt_files(self, tmp_path):
        """All .txt files in the directory should each become a Document."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        (docs_dir / "file1.txt").write_text("Content of file 1.", encoding="utf-8")
        (docs_dir / "file2.txt").write_text("Content of file 2.", encoding="utf-8")
        (docs_dir / "file3.txt").write_text("Content of file 3.", encoding="utf-8")

        result = load_local_text_docs(str(docs_dir))
        assert len(result) == 3

    def test_ignores_non_txt_files(self, tmp_path):
        """Only .txt files should be loaded — .pdf, .md, .csv etc. are ignored."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        (docs_dir / "valid.txt").write_text("Valid content.", encoding="utf-8")
        (docs_dir / "readme.md").write_text("# This is a markdown file", encoding="utf-8")
        (docs_dir / "data.csv").write_text("col1,col2\nval1,val2", encoding="utf-8")

        result = load_local_text_docs(str(docs_dir))
        assert len(result) == 1  # Only the .txt file

    def test_document_has_source_metadata(self, tmp_path):
        """Each loaded Document must carry a 'source' metadata key (the file path)."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        (docs_dir / "sample.txt").write_text("Sample text content.", encoding="utf-8")

        result = load_local_text_docs(str(docs_dir))
        assert len(result) == 1
        assert "source" in result[0].metadata

    def test_document_page_content_matches_file(self, tmp_path):
        """The Document's page_content should contain the text written to the file."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        content = "Embeddings convert text into numerical vectors."
        (docs_dir / "embeddings.txt").write_text(content, encoding="utf-8")

        result = load_local_text_docs(str(docs_dir))
        assert content in result[0].page_content
