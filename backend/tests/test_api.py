"""API endpoint tests against the static-file-free test app from conftest."""
import pytest

from conftest import COURSE_TITLE

pytestmark = pytest.mark.api


class TestQueryEndpoint:
    def test_creates_session_when_missing(self, client, mock_rag_system, sample_sources):
        r = client.post("/api/query", json={"query": "What is MCP?"})
        assert r.status_code == 200
        assert r.json() == {"answer": "MCP standardizes context.",
                            "sources": sample_sources, "session_id": "session_1"}
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "session_1")

    def test_uses_provided_session(self, client, mock_rag_system):
        r = client.post("/api/query", json={"query": "hi", "session_id": "abc"})
        assert r.json()["session_id"] == "abc"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("hi", "abc")

    def test_missing_query_is_422(self, client):
        assert client.post("/api/query", json={}).status_code == 422

    def test_invalid_json_is_422(self, client):
        r = client.post("/api/query", content="nope",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_wrong_method_is_405(self, client):
        assert client.get("/api/query").status_code == 405

    def test_rag_error_is_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("boom")
        r = client.post("/api/query", json={"query": "x"})
        assert r.status_code == 500
        assert r.json()["detail"] == "boom"

    def test_empty_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("General answer", [])
        assert client.post("/api/query", json={"query": "x"}).json()["sources"] == []


class TestCoursesEndpoint:
    def test_returns_stats(self, client):
        r = client.get("/api/courses")
        assert r.status_code == 200
        assert r.json() == {"total_courses": 2,
                            "course_titles": [COURSE_TITLE, "Other Course"]}

    def test_no_courses(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0, "course_titles": []}
        assert client.get("/api/courses").json() == {
            "total_courses": 0, "course_titles": []}

    def test_error_is_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("db down")
        r = client.get("/api/courses")
        assert r.status_code == 500
        assert r.json()["detail"] == "db down"

    def test_post_not_allowed(self, client):
        assert client.post("/api/courses").status_code == 405


class TestRootAndSession:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "message" in r.json()

    def test_delete_session(self, client, mock_rag_system):
        r = client.delete("/api/session/abc")
        assert r.json() == {"status": "ok"}
        mock_rag_system.session_manager.delete_session.assert_called_once_with("abc")

    def test_unknown_route_404(self, client):
        assert client.get("/api/nope").status_code == 404
