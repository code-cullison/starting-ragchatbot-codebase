"""RAGSystem.query end to end (real vector store + tools, mocked Anthropic client),
plus the /api/query endpoint via TestClient."""
import importlib
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from conftest import BACKEND_DIR, COURSE_LINK, COURSE_TITLE, LESSON_LINKS
from fakes import response, text_block, tool_use_block


def script_claude(rag, *responses):
    create = rag.ai_generator.client.messages.create
    create.side_effect = list(responses)
    return create


def search_then_answer(rag, answer, **tool_input):
    return script_claude(
        rag,
        response(tool_use_block("search_course_content", tool_input), stop_reason="tool_use"),
        response(text_block(answer)),
    )


# --- RAGSystem.query -------------------------------------------------------

def test_content_question_runs_search_and_returns_answer_with_sources(rag):
    create = search_then_answer(rag, "MCP standardizes context.",
                                query="what is MCP", course_name="MCP", lesson_number=1)

    answer, sources = rag.query("What is MCP?")

    assert answer == "MCP standardizes context."
    assert sources == [{"text": f"{COURSE_TITLE} - Lesson 1", "url": LESSON_LINKS[1]}]
    # the tool result Claude saw contains the real retrieved chunk
    tool_result = create.call_args_list[1].kwargs["messages"][-1]["content"][0]["content"]
    assert "standardizes how applications give context" in tool_result


def test_both_tools_are_offered_to_claude(rag):
    create = script_claude(rag, response(text_block("hi")))
    rag.query("hello")
    names = {t["name"] for t in create.call_args.kwargs["tools"]}
    assert names == {"search_course_content", "get_course_outline"}


def test_outline_question_uses_outline_tool(rag):
    search = tool_use_block("get_course_outline", {"course_title": "MCP"})
    create = script_claude(rag, response(search, stop_reason="tool_use"),
                           response(text_block("outline")))
    answer, sources = rag.query("Outline of MCP course?")
    assert answer == "outline"
    assert sources == [{"text": COURSE_TITLE, "url": COURSE_LINK}]
    tool_result = create.call_args_list[1].kwargs["messages"][-1]["content"][0]["content"]
    assert "0. Introduction" in tool_result and "1. Why MCP" in tool_result


def test_general_question_has_no_sources(rag):
    script_claude(rag, response(text_block("4")))
    assert rag.query("2+2?") == ("4", [])


def test_sources_do_not_leak_into_next_query(rag):
    search_then_answer(rag, "a", query="context")
    _, first = rag.query("q1")
    script_claude(rag, response(text_block("b")))
    _, second = rag.query("q2")
    assert first and second == []


def test_session_history_is_saved_and_sent_on_next_query(rag):
    sid = rag.session_manager.create_session()
    script_claude(rag, response(text_block("first answer")))
    rag.query("first question", sid)

    create = script_claude(rag, response(text_block("second answer")))
    rag.query("second question", sid)

    system = create.call_args.kwargs["system"]
    assert "User: first question" in system and "Assistant: first answer" in system


def test_unknown_course_search_reports_no_match_to_claude(rag):
    create = search_then_answer(rag, "not found", query="x", course_name="zzzz-nonexistent")
    # resolution is semantic (nearest neighbour), so any course "matches"; the point is no crash
    answer, _ = rag.query("q")
    assert answer == "not found"
    assert create.call_count == 2


def test_api_failure_propagates_from_query(rag):
    script_claude(rag, RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        rag.query("q")


# --- POST /api/query -------------------------------------------------------

@pytest.fixture
def client(rag, monkeypatch):
    """TestClient over the real FastAPI app, with RAGSystem construction stubbed out."""
    monkeypatch.chdir(BACKEND_DIR)  # app mounts ../frontend relative to cwd
    with patch("rag_system.RAGSystem", return_value=rag):
        import app as app_module
        importlib.reload(app_module)
    return TestClient(app_module.app)  # no `with`: skips startup doc ingestion


def test_api_query_returns_answer_sources_and_new_session(client, rag):
    search_then_answer(rag, "MCP answer", query="what is MCP")
    r = client.post("/api/query", json={"query": "What is MCP?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "MCP answer"
    assert body["sources"] and body["sources"][0]["url"]
    assert body["session_id"]


def test_api_query_reuses_supplied_session(client, rag):
    script_claude(rag, response(text_block("x")))
    r = client.post("/api/query", json={"query": "hi", "session_id": "abc"})
    assert r.json()["session_id"] == "abc"


def test_api_query_returns_500_with_detail_when_generation_fails(client, rag):
    script_claude(rag, RuntimeError("model not found"))
    r = client.post("/api/query", json={"query": "What is MCP?"})
    assert r.status_code == 500
    assert "model not found" in r.json()["detail"]


def test_api_courses_lists_seeded_course(client):
    r = client.get("/api/courses")
    assert r.status_code == 200
    assert COURSE_TITLE in r.json()["course_titles"]
