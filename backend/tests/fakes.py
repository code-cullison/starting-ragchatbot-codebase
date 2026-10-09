"""Fake Anthropic response objects shared by the tests."""

from types import SimpleNamespace


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def thinking_block(thinking="hmm"):
    return SimpleNamespace(type="thinking", thinking=thinking)


def tool_use_block(name, input, id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input, id=id)


def response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)
