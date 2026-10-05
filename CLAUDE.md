# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Always use `uv` to run the server and all Python scripts in this project (e.g. `uv run python script.py`); do not use pip.
- All `git push` operations to remote `origin` must be explicitly approved by the user before pushing.
- Install deps: `uv sync` (Python >= 3.13)
- Run the app: `./run.sh` (or `cd backend && uv run uvicorn app:app --reload --port 8000`)
  - Web UI at http://localhost:8000, Swagger docs at http://localhost:8000/docs
- Requires `ANTHROPIC_API_KEY` in a root `.env` (see `.env.example`).
- There is no test suite, linter, or build step configured. `main.py` is a stub; the real entry point is `backend/app.py`.

The server must be started from `backend/` — paths like `../docs`, `../frontend` and `./chroma_db` (ChromaDB persistence) are relative to that cwd.

## Architecture

Full-stack RAG app over course transcripts: FastAPI backend serving a vanilla JS frontend (`frontend/`, mounted at `/` as static files), ChromaDB for vectors, Claude for generation.

**Query flow** (`POST /api/query` in `backend/app.py` → `RAGSystem.query` in `rag_system.py`):
1. `RAGSystem` is the orchestrator wiring together `DocumentProcessor`, `VectorStore`, `AIGenerator`, `SessionManager`, and `ToolManager`.
2. Retrieval is **tool-based, not pre-fetched**: `AIGenerator` gives Claude the `search_course_content` tool (`search_tools.py`, `CourseSearchTool`) and Claude decides whether to search. When it returns `tool_use`, the tool is executed through `ToolManager` and the result is sent back for a final response (see `ai_generator.py`).
3. Sources shown in the UI are side-channel state: `CourseSearchTool.last_sources` is read via `ToolManager.get_last_sources()` after generation and then reset. New tools that should surface sources need to follow this pattern.
4. Conversation history is kept in-memory per session by `SessionManager` (truncated to `MAX_HISTORY` messages) and passed into the system prompt.

**Vector store** (`vector_store.py`) uses two Chroma collections: `course_catalog` (course titles/instructors/lesson metadata, used to semantically resolve a fuzzy `course_name` to an exact title) and `course_content` (text chunks, filtered by resolved course title and/or lesson number).

**Ingestion**: on app startup, `add_course_folder("../docs")` loads `.pdf/.docx/.txt` files, skipping courses whose title already exists in Chroma (so edited docs are not re-ingested unless the DB is cleared via `clear_existing=True` or by deleting `backend/chroma_db`). Course docs must follow this format, parsed in `document_processor.py`:
```
Course Title: ...
Course Link: ...
Course Instructor: ...

Lesson 0: <title>
Lesson Link: ...
<lesson text>
```
Text is chunked sentence-wise (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`).

**Config**: all tunables (model, embedding model `all-MiniLM-L6-v2`, chunk sizes, `MAX_RESULTS`, `MAX_HISTORY`, Chroma path) live in the `Config` dataclass in `backend/config.py`. Data models are in `backend/models.py`.
