import os
import sys
from unittest.mock import MagicMock, patch

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from config import Config  # noqa: E402
from models import Course, CourseChunk, Lesson  # noqa: E402
from rag_system import RAGSystem  # noqa: E402
from vector_store import VectorStore  # noqa: E402

COURSE_TITLE = "MCP: Build Rich-Context AI Apps with Anthropic"
COURSE_LINK = "https://example.com/mcp"
LESSON_LINKS = {0: "https://example.com/mcp/0", 1: "https://example.com/mcp/1"}


@pytest.fixture(scope="module")
def seeded_store(tmp_path_factory):
    """Real Chroma store in a temp dir with one small course (lessons 0 and 1)."""
    path = str(tmp_path_factory.mktemp("chroma"))
    store = VectorStore(path, "all-MiniLM-L6-v2", max_results=5)
    store.add_course_metadata(Course(
        title=COURSE_TITLE,
        course_link=COURSE_LINK,
        instructor="Elie Schoppik",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link=LESSON_LINKS[0]),
            Lesson(lesson_number=1, title="Why MCP", lesson_link=LESSON_LINKS[1]),
        ],
    ))
    store.add_course_content([
        CourseChunk(content="Welcome to the course on the Model Context Protocol.",
                    course_title=COURSE_TITLE, lesson_number=0, chunk_index=0),
        CourseChunk(content="MCP standardizes how applications give context to LLMs.",
                    course_title=COURSE_TITLE, lesson_number=1, chunk_index=1),
    ])
    return store


@pytest.fixture
def rag(seeded_store):
    """Real RAGSystem wired to the seeded store, with the Anthropic client mocked.

    Tests script `rag.ai_generator.client.messages.create.side_effect`.
    """
    cfg = Config(ANTHROPIC_API_KEY="test-key")
    with patch("rag_system.VectorStore", return_value=seeded_store), \
         patch("ai_generator.anthropic.Anthropic") as anthropic_cls:
        anthropic_cls.return_value = MagicMock()
        system = RAGSystem(cfg)
    return system


# ---- API test infrastructure -------------------------------------------------
# backend/app.py mounts ../frontend and builds a real RAGSystem at import time,
# so API tests use a separate app that mirrors its endpoints around a mock.

from typing import List, Optional  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    text: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def create_test_app(rag_system) -> FastAPI:
    """Mirror of app.py's endpoints, minus static mount/startup/middleware."""
    app = FastAPI(title="Course Materials RAG System (test)")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(total_courses=analytics["total_courses"],
                               course_titles=analytics["course_titles"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        rag_system.session_manager.delete_session(session_id)
        return {"status": "ok"}

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}

    return app


@pytest.fixture
def sample_sources():
    return [{"text": f"{COURSE_TITLE} - Lesson 1", "url": LESSON_LINKS[1]},
            {"text": f"{COURSE_TITLE} - Lesson 0", "url": None}]


@pytest.fixture
def mock_rag_system(sample_sources):
    """MagicMock RAGSystem with sensible default return values."""
    mock = MagicMock()
    mock.session_manager.create_session.return_value = "session_1"
    mock.query.return_value = ("MCP standardizes context.", sample_sources)
    mock.get_course_analytics.return_value = {
        "total_courses": 2, "course_titles": [COURSE_TITLE, "Other Course"]}
    return mock


@pytest.fixture
def client(mock_rag_system):
    return TestClient(create_test_app(mock_rag_system))
