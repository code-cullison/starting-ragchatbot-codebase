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
    store.add_course_metadata(
        Course(
            title=COURSE_TITLE,
            course_link=COURSE_LINK,
            instructor="Elie Schoppik",
            lessons=[
                Lesson(
                    lesson_number=0, title="Introduction", lesson_link=LESSON_LINKS[0]
                ),
                Lesson(lesson_number=1, title="Why MCP", lesson_link=LESSON_LINKS[1]),
            ],
        )
    )
    store.add_course_content(
        [
            CourseChunk(
                content="Welcome to the course on the Model Context Protocol.",
                course_title=COURSE_TITLE,
                lesson_number=0,
                chunk_index=0,
            ),
            CourseChunk(
                content="MCP standardizes how applications give context to LLMs.",
                course_title=COURSE_TITLE,
                lesson_number=1,
                chunk_index=1,
            ),
        ]
    )
    return store


@pytest.fixture
def rag(seeded_store):
    """Real RAGSystem wired to the seeded store, with the Anthropic client mocked.

    Tests script `rag.ai_generator.client.messages.create.side_effect`.
    """
    cfg = Config(ANTHROPIC_API_KEY="test-key")
    with (
        patch("rag_system.VectorStore", return_value=seeded_store),
        patch("ai_generator.anthropic.Anthropic") as anthropic_cls,
    ):
        anthropic_cls.return_value = MagicMock()
        system = RAGSystem(cfg)
    return system
