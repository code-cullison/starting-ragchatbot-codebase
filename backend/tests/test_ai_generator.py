"""Tests that AIGenerator calls the Anthropic API and the search tool correctly."""
from unittest.mock import MagicMock, patch

import pytest

from ai_generator import AIGenerator
from fakes import response, text_block, thinking_block, tool_use_block


@pytest.fixture
def gen():
    with patch("ai_generator.anthropic.Anthropic") as cls:
        cls.return_value = MagicMock()
        yield AIGenerator("key", "test-model")


@pytest.fixture
def tool_manager():
    tm = MagicMock()
    tm.execute_tool.return_value = "TOOL OUTPUT"
    return tm


TOOLS = [{"name": "search_course_content", "input_schema": {"type": "object"}}]
SEARCH_USE = tool_use_block("search_course_content",
                            {"query": "what is MCP", "course_name": "MCP", "lesson_number": 1},
                            id="toolu_42")


def test_direct_answer_without_tool_use(gen, tool_manager):
    gen.client.messages.create.return_value = response(text_block("4"))
    assert gen.generate_response("2+2?", tools=TOOLS, tool_manager=tool_manager) == "4"
    tool_manager.execute_tool.assert_not_called()
    assert gen.client.messages.create.call_count == 1


def test_first_call_offers_tools_and_model(gen):
    gen.client.messages.create.return_value = response(text_block("ok"))
    gen.generate_response("hi", tools=TOOLS)
    kwargs = gen.client.messages.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == {"type": "auto"}
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_no_tools_key_when_no_tools_given(gen):
    gen.client.messages.create.return_value = response(text_block("ok"))
    gen.generate_response("hi")
    assert "tools" not in gen.client.messages.create.call_args.kwargs


def test_conversation_history_goes_into_system_prompt(gen):
    gen.client.messages.create.return_value = response(text_block("ok"))
    gen.generate_response("hi", conversation_history="User: earlier\nAssistant: reply")
    system = gen.client.messages.create.call_args.kwargs["system"]
    assert "Previous conversation:\nUser: earlier" in system


def test_tool_use_executes_tool_with_model_supplied_arguments(gen, tool_manager):
    gen.client.messages.create.side_effect = [
        response(SEARCH_USE, stop_reason="tool_use"),
        response(text_block("final")),
    ]
    assert gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager) == "final"
    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="what is MCP", course_name="MCP", lesson_number=1)


def test_follow_up_call_carries_tool_result_and_keeps_tools_available(gen, tool_manager):
    gen.client.messages.create.side_effect = [
        response(SEARCH_USE, stop_reason="tool_use"),
        response(text_block("final")),
    ]
    gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager)

    follow_up = gen.client.messages.create.call_args_list[1].kwargs
    assert follow_up["tools"] == TOOLS
    user_q, assistant, tool_result = follow_up["messages"]
    assert user_q == {"role": "user", "content": "q"}
    assert assistant == {"role": "assistant", "content": [SEARCH_USE]}
    assert tool_result == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_42", "content": "TOOL OUTPUT"}]}


def test_skips_thinking_blocks_in_final_answer(gen):
    gen.client.messages.create.return_value = response(thinking_block(), text_block("answer"))
    assert gen.generate_response("q") == "answer"


def test_skips_thinking_blocks_before_tool_use(gen, tool_manager):
    gen.client.messages.create.side_effect = [
        response(thinking_block(), SEARCH_USE, stop_reason="tool_use"),
        response(thinking_block(), text_block("final")),
    ]
    assert gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager) == "final"


def test_response_without_text_block_raises_clear_error(gen):
    """An empty/no-text reply (e.g. cut off by max_tokens) must not surface as a bare StopIteration."""
    gen.client.messages.create.return_value = response(thinking_block(), stop_reason="max_tokens")
    with pytest.raises(RuntimeError):
        gen.generate_response("q")


def test_second_tool_request_in_follow_up_is_executed(gen, tool_manager):
    second = tool_use_block("search_course_content", {"query": "again"}, id="toolu_43")
    gen.client.messages.create.side_effect = [
        response(SEARCH_USE, stop_reason="tool_use"),
        response(second, stop_reason="tool_use"),
        response(text_block("final")),
    ]
    assert gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager) == "final"
    assert tool_manager.execute_tool.call_count == 2
    calls = gen.client.messages.create.call_args_list
    assert "tools" in calls[1].kwargs
    assert "tools" not in calls[2].kwargs  # cap reached: Claude must answer in text


def test_tool_rounds_are_capped_and_last_call_has_no_tools(gen, tool_manager):
    """Claude that keeps asking for tools is cut off after MAX_TOOL_ROUNDS executions."""
    def another(i):
        return response(tool_use_block("search_course_content", {"query": "q"}, id=f"t{i}"),
                        stop_reason="tool_use")
    gen.client.messages.create.side_effect = [another(0), another(1), response(text_block("done"))]
    assert gen.generate_response("q", tools=TOOLS, tool_manager=tool_manager) == "done"
    assert tool_manager.execute_tool.call_count == AIGenerator.MAX_TOOL_ROUNDS
    assert gen.client.messages.create.call_count == AIGenerator.MAX_TOOL_ROUNDS + 1
