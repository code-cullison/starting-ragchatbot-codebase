import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to two tools for course information: a content search tool and a course outline tool.

Search Tool Usage:
- Use the content search tool **only** for questions about specific course content or detailed educational materials
- You may make up to **two sequential tool calls** per query, each in its own step, when the second call depends on the first call's result
- Chain calls only when needed (e.g. get a course outline to find a lesson's title, then search course content using that title, possibly in another course); use a single call when one is enough, and never repeat an identical call
- If a tool returns an error or nothing useful, stop and answer with what you have; do not retry
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Course Outline Tool Usage:
- Use the outline tool for questions about a course's outline, structure, syllabus or lesson list
- Your answer must include the **course title, the course link, and the number and title of every lesson** — do not omit or summarize lessons

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search (and chain a second call if required), then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Max sequential tool-execution rounds per query before forcing a text answer
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Return the first text block, skipping thinking or other non-text blocks"""
        for block in response.content:
            if block.type == "text":
                return block.text
        raise RuntimeError(
            f"Claude returned no text content (stop_reason={response.stop_reason!r})"
        )

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        response = initial_response

        for round_num in range(1, self.MAX_TOOL_ROUNDS + 1):
            # Add AI's tool use response
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls and collect results. Every tool_use block
            # needs a matching tool_result; a raised exception is reported back
            # to Claude as an error result instead of aborting the request.
            tool_results = []
            failed = False
            for content_block in response.content:
                if content_block.type == "tool_use":
                    result_block = {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                    }
                    try:
                        result_block["content"] = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )
                    except Exception as e:
                        failed = True
                        result_block["content"] = f"Tool error: {e}"
                        result_block["is_error"] = True
                    tool_results.append(result_block)

            # Add tool results as single message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Follow-up call; tools stay defined (history contains tool_use blocks),
            # but once the round cap is reached or a tool failed, tool_choice "none"
            # forces a text answer
            follow_up_params = {
                **self.base_params,
                "messages": list(messages),
                "system": base_params["system"],
            }
            if "tools" in base_params:
                follow_up_params["tools"] = base_params["tools"]
                keep_tools = round_num < self.MAX_TOOL_ROUNDS and not failed
                follow_up_params["tool_choice"] = {
                    "type": "auto" if keep_tools else "none"
                }

            response = self.client.messages.create(**follow_up_params)
            if response.stop_reason != "tool_use":
                break

        return self._extract_text(response)
