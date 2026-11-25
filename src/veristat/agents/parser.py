"""Action parser for VeriStat ReAct agent: JSON schema first with Regex fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from veristat.agents.contracts import AgentAction, ToolCall


def parse_agent_action(raw_text: str) -> AgentAction:
    """Parse LLM output into an AgentAction.

    Priority:
    1. Direct or embedded JSON object with "action" and "action_input".
    2. Markdown block parsing fallback (Thought: ... Action: ... Action Input: ...).
    """
    cleaned = raw_text.strip()

    # 1. Try JSON extraction
    json_candidate = _extract_json_candidate(cleaned)
    if json_candidate is not None:
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict) and "action" in data:
                action_name = str(data["action"]).strip()
                action_input = data.get("action_input", {})
                thought = str(data.get("thought", "")).strip()

                if action_name.lower() == "finish":
                    return AgentAction(
                        is_finish=True,
                        final_answer=action_input if isinstance(action_input, dict) else {"value": action_input},
                        raw_text=raw_text,
                    )
                else:
                    return AgentAction(
                        is_finish=False,
                        tool_call=ToolCall(
                            tool_name=action_name,
                            tool_input=action_input if isinstance(action_input, dict) else {"param": action_input},
                            raw_thought=thought,
                        ),
                        raw_text=raw_text,
                    )
        except json.JSONDecodeError:
            pass

    # 2. Fallback: Regex extraction for Thought: ... Action: ... Action Input: ...
    thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", cleaned, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", cleaned, re.IGNORECASE)
    input_match = re.search(r"Action Input:\s*(.*?)$", cleaned, re.DOTALL | re.IGNORECASE)

    if action_match:
        act_name = action_match.group(1).strip()
        thought_str = thought_match.group(1).strip() if thought_match else ""
        input_str = input_match.group(1).strip() if input_match else "{}"

        # Try parsing input_str as JSON
        try:
            parsed_input = json.loads(input_str)
        except Exception:
            # If plain string or code block
            parsed_input = {"raw_input": input_str}

        if act_name.lower() == "finish":
            return AgentAction(
                is_finish=True,
                final_answer=parsed_input if isinstance(parsed_input, dict) else {"value": parsed_input},
                raw_text=raw_text,
            )
        return AgentAction(
            is_finish=False,
            tool_call=ToolCall(
                tool_name=act_name,
                tool_input=parsed_input if isinstance(parsed_input, dict) else {"code": input_str},
                raw_thought=thought_str,
            ),
            raw_text=raw_text,
        )

    # 3. Default to malformed action
    return AgentAction(
        is_finish=False,
        tool_call=None,
        final_answer=None,
        raw_text=raw_text,
    )


def _extract_json_candidate(text: str) -> str | None:
    """Find the outermost JSON object within text, including within markdown code blocks."""
    # Strip markdown block ```json ... ```
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        return md_match.group(1)

    # Look for raw balanced curly brackets
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        char = text[i]
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and in_string:
            escape = not escape
            continue
        elif not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
        escape = False

    return None
