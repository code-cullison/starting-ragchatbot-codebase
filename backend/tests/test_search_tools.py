"""Tests for CourseSearchTool.execute (and ToolManager plumbing)."""
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults

from conftest import COURSE_LINK, COURSE_TITLE, LESSON_LINKS


def mock_store(results):
    store = MagicMock()
    store.search.return_value = results
    store.get_lesson_link.return_value = None
    store.get_course_link.return_value = COURSE_LINK
    return store


def results_of(*docs_and_lessons, course="Course A"):
    return SearchResults(
        documents=[d for d, _ in docs_and_lessons],
        metadata=[{"course_title": course, "lesson_number": n} for _, n in docs_and_lessons],
        distances=[0.1] * len(docs_and_lessons),
    )


# --- formatting / branching, with a mocked store ---------------------------

def test_formats_results_with_course_and_lesson_header():
    tool = CourseSearchTool(mock_store(results_of(("alpha text", 3))))
    assert tool.execute(query="q") == "[Course A - Lesson 3]\nalpha text"


def test_header_omits_lesson_when_chunk_has_no_lesson_number():
    tool = CourseSearchTool(mock_store(results_of(("alpha text", None))))
    assert tool.execute(query="q") == "[Course A]\nalpha text"


def test_passes_filters_through_to_store():
    store = mock_store(results_of(("t", 1)))
    CourseSearchTool(store).execute(query="q", course_name="MCP", lesson_number=1)
    store.search.assert_called_once_with(query="q", course_name="MCP", lesson_number=1)


def test_store_error_is_returned_verbatim():
    tool = CourseSearchTool(mock_store(SearchResults.empty("No course found matching 'x'")))
    assert tool.execute(query="q", course_name="x") == "No course found matching 'x'"


def test_empty_results_message_without_filters():
    tool = CourseSearchTool(mock_store(SearchResults([], [], [])))
    assert tool.execute(query="q") == "No relevant content found."


def test_empty_results_message_mentions_course_and_lesson_filters():
    tool = CourseSearchTool(mock_store(SearchResults([], [], [])))
    msg = tool.execute(query="q", course_name="MCP", lesson_number=2)
    assert msg == "No relevant content found in course 'MCP' in lesson 2."


def test_empty_results_message_mentions_lesson_zero():
    tool = CourseSearchTool(mock_store(SearchResults([], [], [])))
    msg = tool.execute(query="q", lesson_number=0)
    assert "lesson 0" in msg


def test_sources_are_deduplicated_and_link_to_lesson():
    store = mock_store(results_of(("a", 1), ("b", 1), ("c", 2)))
    store.get_lesson_link.side_effect = lambda title, n: f"http://l/{n}"
    tool = CourseSearchTool(store)
    tool.execute(query="q")
    assert tool.last_sources == [
        {"text": "Course A - Lesson 1", "url": "http://l/1"},
        {"text": "Course A - Lesson 2", "url": "http://l/2"},
    ]


def test_source_falls_back_to_course_link_when_lesson_link_missing():
    tool = CourseSearchTool(mock_store(results_of(("a", 1))))
    tool.execute(query="q")
    assert tool.last_sources == [{"text": "Course A - Lesson 1", "url": COURSE_LINK}]


def test_tool_definition_matches_execute_signature():
    definition = CourseSearchTool(MagicMock()).get_tool_definition()
    assert definition["name"] == "search_course_content"
    assert definition["input_schema"]["required"] == ["query"]
    assert set(definition["input_schema"]["properties"]) == {"query", "course_name", "lesson_number"}


# --- ToolManager -----------------------------------------------------------

def test_tool_manager_dispatches_and_reports_sources_then_resets():
    tm = ToolManager()
    tool = CourseSearchTool(mock_store(results_of(("a", 1))))
    tm.register_tool(tool)

    assert tm.execute_tool("search_course_content", query="q").startswith("[Course A - Lesson 1]")
    assert tm.get_last_sources()
    tm.reset_sources()
    assert tm.get_last_sources() == []


def test_tool_manager_unknown_tool():
    assert ToolManager().execute_tool("nope") == "Tool 'nope' not found"


# --- real Chroma store -----------------------------------------------------

def test_real_store_search_returns_matching_chunk(seeded_store):
    out = CourseSearchTool(seeded_store).execute(query="context for LLMs")
    assert f"[{COURSE_TITLE} - Lesson 1]" in out
    assert "standardizes" in out


def test_real_store_fuzzy_course_name_resolves(seeded_store):
    out = CourseSearchTool(seeded_store).execute(query="welcome", course_name="MCP")
    assert COURSE_TITLE in out


def test_real_store_lesson_filter_including_lesson_zero(seeded_store):
    tool = CourseSearchTool(seeded_store)
    out = tool.execute(query="anything", lesson_number=0)
    assert "Lesson 0" in out and "Lesson 1" not in out
    assert tool.last_sources == [{"text": f"{COURSE_TITLE} - Lesson 0", "url": LESSON_LINKS[0]}]


def test_real_store_course_and_lesson_filter_combined(seeded_store):
    out = CourseSearchTool(seeded_store).execute(
        query="anything", course_name="MCP", lesson_number=1)
    assert "Lesson 1" in out and "Lesson 0" not in out
