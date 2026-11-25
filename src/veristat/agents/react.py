"""VeriStat ReAct Agent implementation built from scratch without external frameworks."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from veristat.agents.contracts import AgentAction
from veristat.agents.parser import parse_agent_action
from veristat.agents.prompts import build_system_prompt
from veristat.agents.state import AgentState, StepRecord
from veristat.llm.base import LLMClient
from veristat.llm.cache import SQLiteLLMCache, compute_canonical_cache_key
from veristat.tools.base import ToolRegistry


class VeriStatReActAgent:
    """ReAct Agent operating over statistical tools with atomic step transitions."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        cache: SQLiteLLMCache | None = None,
        max_steps: int = 8,
    ) -> None:
        self.llm = llm_client
        self.tools = tool_registry
        self.cache = cache
        self.max_steps = max_steps
        self.system_prompt = build_system_prompt(self.tools.get_schemas())

    def step(self, state: AgentState) -> AgentState:
        """Perform a single atomic reasoning and action step, returning the new state."""
        if state.is_terminal:
            return state

        # Construct messages from state history
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": f"Câu hỏi thống kê: {state.question}"})

        # Append previous steps as assistant and tool observation messages
        for s in state.steps:
            if s.action_name:
                assistant_text = json.dumps({
                    "thought": s.thought,
                    "action": s.action_name,
                    "action_input": s.action_input or {},
                }, ensure_ascii=False)
            else:
                assistant_text = s.thought
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": f"Observation: {s.observation}"})

        # Generate response from LLM (with cache check)
        cache_key = None
        cached_resp = None
        if self.cache is not None:
            payload = {
                "model": "qwen2.5-7b",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 1024,
            }
            cache_key = compute_canonical_cache_key(payload)
            cached_resp = self.cache.get(cache_key)

        if cached_resp is not None:
            llm_response = cached_resp
        else:
            llm_response = self.llm.generate(messages=messages, temperature=0.0, max_tokens=1024)
            if self.cache is not None and cache_key is not None:
                self.cache.put(cache_key, llm_response, payload)

        # Parse action
        action: AgentAction = parse_agent_action(llm_response.text)

        # Handle terminal finish action
        if action.is_finish:
            state.is_terminal = True
            state.status = "SUCCESS"
            state.final_answer = action.final_answer
            # Extract cited tables from final answer if present
            if action.final_answer and isinstance(action.final_answer, dict):
                cited = action.final_answer.get("cited_tables", [])
                if isinstance(cited, list):
                    state.cited_tables.update(cited)

            state.steps.append(
                StepRecord(
                    step_idx=state.current_step + 1,
                    thought="Hoàn thành phân tích và đưa ra đáp số.",
                    action_name="finish",
                    action_input=action.final_answer,
                    observation="Kết thúc phân tích.",
                )
            )
            state.current_step += 1
            return state

        # Handle malformed output
        if action.tool_call is None:
            obs = "Observation: [FORMAT_ERROR] Output không chứa cấu trúc JSON hoặc Action hợp lệ. Vui lòng trả về JSON với 'thought', 'action', 'action_input'."
            state.steps.append(
                StepRecord(
                    step_idx=state.current_step + 1,
                    thought=llm_response.text,
                    action_name=None,
                    action_input=None,
                    observation=obs,
                    error_type="FORMAT_ERROR",
                )
            )
            state.current_step += 1
            if state.current_step >= self.max_steps:
                state.is_terminal = True
                state.status = "MAX_STEPS_REACHED"
            return state

        # Execute registered tool
        tool_call = action.tool_call
        tool = self.tools.get(tool_call.tool_name)
        if tool is None:
            obs = f"Observation: [TOOL_NOT_FOUND] Công cụ '{tool_call.tool_name}' không tồn tại. Các công cụ hợp lệ: {[t.name for t in self.tools.list_tools()]}"
            state.steps.append(
                StepRecord(
                    step_idx=state.current_step + 1,
                    thought=tool_call.raw_thought,
                    action_name=tool_call.tool_name,
                    action_input=tool_call.tool_input,
                    observation=obs,
                    error_type="TOOL_NOT_FOUND",
                )
            )
        else:
            result = tool.execute(**tool_call.tool_input)
            if result.success:
                obs = result.output
                err_msg = None
                err_type = None
            else:
                obs = f"Observation: [{result.error_type or 'ERROR'}] {result.error}"
                err_msg = result.error
                err_type = result.error_type

            # Enrich cited tables from tool metadata
            if result.metadata and "tables" in result.metadata:
                state.cited_tables.update(result.metadata["tables"])

            state.steps.append(
                StepRecord(
                    step_idx=state.current_step + 1,
                    thought=tool_call.raw_thought,
                    action_name=tool_call.tool_name,
                    action_input=tool_call.tool_input,
                    observation=obs,
                    tool_error=err_msg,
                    error_type=err_type,
                    execution_time=result.metadata.get("cpu_time", 0.0),
                )
            )

        state.current_step += 1
        if state.current_step >= self.max_steps:
            state.is_terminal = True
            state.status = "MAX_STEPS_REACHED"

        return state

    def run(self, question: str, question_id: str = "Q000") -> AgentState:
        """Run complete ReAct trajectory up to max_steps or terminal finish."""
        state = AgentState(question_id=question_id, question=question)
        while not state.is_terminal and state.current_step < self.max_steps:
            state = self.step(state)
        return state
