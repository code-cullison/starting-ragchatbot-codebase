"""Regression guards for the original "query failed" bug: MAX_RESULTS = 0.

Chroma rejects n_results=0, VectorStore.search turns that into an error string, and
CourseSearchTool hands it to Claude, so every content question fails.
"""
from config import Config
from models import CourseChunk
from search_tools import CourseSearchTool
from vector_store import VectorStore


def test_default_max_results_is_positive():
    assert Config().MAX_RESULTS > 0


def test_zero_max_results_makes_every_content_search_fail(tmp_path):
    """Documents why the config value matters: with 0, no search can ever succeed."""
    store = VectorStore(str(tmp_path), "all-MiniLM-L6-v2", max_results=0)
    store.add_course_content([
        CourseChunk(content="hello world", course_title="C", lesson_number=0, chunk_index=0)])

    results = store.search("hello")
    assert results.documents == []
    assert "Search error" in results.error

    assert CourseSearchTool(store).execute(query="hello").startswith("Search error")
