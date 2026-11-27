"""Concrete implementation of `StepEnvironment` integrating Khoa's ToolRegistry
and ReAct loop with Sơn's MCTS, Self-Consistency, and Best-of-n contracts.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from typing import Any

from veristat.agents.contracts import AgentAction
from veristat.agents.parser import parse_agent_action
from veristat.agents.prompts import build_system_prompt
from veristat.agents.types import (
    FinalAnswer,
    LLMClient,
    ReActStep,
    StepEnvironment,
    StepObservation,
    TerminatedReason,
    ToolAction,
    ToolName,
    Trajectory,
    _normalize_final_answer_value,
)
from veristat.data.schema import BenchmarkQuestion
from veristat.tools.base import ToolRegistry


class VeriStatStepEnvironment:
    """Production StepEnvironment wrapping ToolRegistry (QueryTable, SearchDocs, RunPython)
    and typed LLMClient for ReAct, Self-Consistency, Best-of-n, and MCTS."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt or build_system_prompt(self.tools.get_schemas())

    def _build_messages(
        self,
        question: BenchmarkQuestion,
        history: tuple[ReActStep, ...],
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        domain_val = getattr(question, "domain", getattr(question, "category", "thống kê"))
        user_msg = (
            f"Câu hỏi: {question.question}\n"
            f"[Siêu dữ liệu: ID={question.question_id}, Lĩnh vực={domain_val}, "
            f"Kiểu đáp số={question.answer_type}, Đơn vị kỳ vọng={question.unit}]"
        )
        messages.append({"role": "user", "content": user_msg})

        for step in history:
            assistant_content = json.dumps(
                {
                    "thought": step.action.thought,
                    "action": step.action.tool_name,
                    "action_input": step.action.tool_args,
                },
                ensure_ascii=False,
            )
            messages.append({"role": "assistant", "content": assistant_content})
            obs_text = step.observation.output if step.observation else ""
            messages.append({"role": "user", "content": f"Observation: {obs_text}"})

        return messages

    def propose_actions(
        self,
        question: BenchmarkQuestion,
        history: tuple[ReActStep, ...],
        *,
        k: int = 1,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> list[ToolAction]:
        """Propose k candidate actions from the current trajectory state."""
        messages = self._build_messages(question, history)
        actions: list[ToolAction] = []

        for i in range(max(1, k)):
            step_seed = (seed + i * 17) if seed is not None else None
            # When k=1 and temperature==0, deterministic greedy
            temp = temperature if k > 1 else 0.0
            resp = self.llm.generate(
                messages,
                temperature=temp,
                seed=step_seed,
                max_tokens=1024,
            )

            parsed: AgentAction = parse_agent_action(resp.text)
            if parsed.is_finish:
                tool_name: ToolName = "final_answer"
                tool_args = parsed.final_answer or {}
                thought = "Tổng hợp kết quả và đưa ra đáp số cuối cùng."
            elif parsed.tool_call is not None:
                raw_name = parsed.tool_call.tool_name
                # Normalize tool name
                if raw_name in ("search_docs", "query_table", "run_python", "final_answer"):
                    tool_name = raw_name  # type: ignore[assignment]
                else:
                    tool_name = "search_docs"
                tool_args = parsed.tool_call.tool_input
                thought = parsed.tool_call.raw_thought or "Thực thi công cụ phân tích."
            else:
                domain_val = getattr(question, "domain", getattr(question, "category", "thống kê"))
                tool_name = "search_docs"
                tool_args = {"query": str(domain_val)}
                thought = f"Định dạng trả về chưa chuẩn, tìm kiếm lại thông tin lĩnh vực {domain_val}."

            actions.append(
                ToolAction(
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    raw_text=resp.text,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    latency_s=resp.latency_s,
                )
            )

        return actions

    def execute_action(
        self,
        action: ToolAction,
        history: tuple[ReActStep, ...] = (),
    ) -> StepObservation:
        """Execute a proposed action in the underlying tools/sandbox and record observations."""
        if action.tool_name == "final_answer":
            return StepObservation(
                output="[FINISH] Hoàn thành phân tích.",
                ok=True,
                error=None,
                safety_violation=False,
                queried_tables=(),
                computed_values=(),
                latency_s=0.0,
            )

        tool = self.tools.get(action.tool_name)
        if tool is None:
            return StepObservation(
                output=f"[TOOL_NOT_FOUND] Công cụ '{action.tool_name}' không tồn tại.",
                ok=False,
                error="TOOL_NOT_FOUND",
                safety_violation=False,
                queried_tables=(),
                computed_values=(),
                latency_s=0.0,
            )

        t0 = time.perf_counter()
        result = tool.execute(**action.tool_args)
        elapsed = time.perf_counter() - t0

        safety_violation = (
            result.error_type in ("SECURITY_VIOLATION", "AST_FORBIDDEN")
            or bool(result.metadata.get("safety_violation", False))
        )

        queried_tables: list[str] = []
        computed_values: list[Any] = []

        # 1. Enrich queried tables
        if action.tool_name == "query_table":
            if "tables" in result.metadata:
                queried_tables.extend(result.metadata["tables"])
            elif "verified_tables" in result.metadata:
                queried_tables.extend(result.metadata["verified_tables"])
            # Also extract direct mentions in SQL query
            sql_text = str(action.tool_args.get("sql", ""))
            tid_matches = re.findall(r"\b(NG\d{4}_T\d+|T\d{2})\b", sql_text, re.IGNORECASE)
            for m in tid_matches:
                m_upper = m.upper()
                if m_upper not in queried_tables:
                    queried_tables.append(m_upper)

        # 2. Enrich computed values (numbers and strings)
        if result.success and result.output:
            if action.tool_name == "run_python":
                computed_values.extend(self._extract_python_values(result.output))
            elif action.tool_name == "query_table":
                computed_values.extend(self._extract_table_values(result.output))

        return StepObservation(
            output=result.output if result.success else f"[{result.error_type or 'ERROR'}] {result.error}",
            ok=result.success,
            error=result.error if not result.success else None,
            safety_violation=safety_violation,
            queried_tables=tuple(queried_tables),
            computed_values=tuple(computed_values),
            latency_s=elapsed,
        )

    def _extract_python_values(self, stdout: str) -> list[Any]:
        """Extract printed numbers, lists, and string entities from Python sandbox output."""
        values: list[Any] = []
        for line in stdout.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            # Try JSON parsing
            try:
                parsed = json.loads(line_str)
                if isinstance(parsed, (int, float, str, list)):
                    values.append(parsed)
            except Exception:
                pass
            # Try float/int
            try:
                val = float(line_str)
                values.append(val)
                continue
            except ValueError:
                pass
            # Regex extract numbers
            num_matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line_str)
            for nm in num_matches:
                try:
                    values.append(float(nm))
                except ValueError:
                    pass
            # Also append stripped line as string candidate
            values.append(line_str)

        return values

    def _extract_table_values(self, csv_text: str) -> list[Any]:
        """Extract cell values from CSV table output."""
        values: list[Any] = []
        try:
            reader = csv.reader(io.StringIO(csv_text))
            for row in reader:
                for cell in row:
                    cell_clean = cell.strip()
                    try:
                        values.append(float(cell_clean))
                    except ValueError:
                        if cell_clean:
                            values.append(cell_clean)
        except Exception:
            pass
        return values

    def extract_final_answer(self, action: ToolAction) -> FinalAnswer:
        """Extract structured FinalAnswer from a final_answer action."""
        if action.tool_name == "final_answer":
            args = action.tool_args
            raw_answer = args.get("answer") if "answer" in args else args.get("value")
            unit = str(args.get("unit", ""))
            raw_cited = args.get("cited_tables", [])
            cited_tables = tuple(str(t) for t in raw_cited) if isinstance(raw_cited, list) else ()
            computation_code = str(args.get("computation_code", ""))
            parse_ok = raw_answer is not None
            return FinalAnswer(
                answer=_normalize_final_answer_value(raw_answer),
                unit=unit,
                cited_tables=cited_tables,
                computation_code=computation_code,
                parse_ok=parse_ok,
            )

        # Fallback: attempt to parse action.raw_text
        try:
            parsed = parse_agent_action(action.raw_text)
            if parsed.is_finish and parsed.final_answer:
                return self.extract_final_answer(
                    ToolAction(
                        thought=action.thought,
                        tool_name="final_answer",
                        tool_args=parsed.final_answer,
                        raw_text=action.raw_text,
                        prompt_tokens=action.prompt_tokens,
                        completion_tokens=action.completion_tokens,
                        latency_s=action.latency_s,
                    )
                )
        except Exception:
            pass

        return FinalAnswer(
            answer=None,
            unit="",
            cited_tables=(),
            computation_code="",
            parse_ok=False,
        )

    def run_react(
        self,
        question: BenchmarkQuestion,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_steps: int = 8,
        max_seconds: float = 60.0,
        max_tokens: int | None = None,
    ) -> Trajectory:
        """Run single complete ReAct trajectory conforming to `veristat.agents.types.Trajectory`."""
        history: list[ReActStep] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        start_time = time.perf_counter()

        terminated_reason: TerminatedReason = "max_steps"
        final_answer: FinalAnswer | None = None

        for step_idx in range(1, max_steps + 1):
            elapsed = time.perf_counter() - start_time
            if elapsed >= max_seconds:
                terminated_reason = "timeout"
                break

            total_tokens = total_prompt_tokens + total_completion_tokens
            if max_tokens is not None and total_tokens >= max_tokens:
                terminated_reason = "token_budget"
                break

            # 1. Propose single action
            proposals = self.propose_actions(
                question,
                tuple(history),
                k=1,
                temperature=temperature,
                seed=seed,
            )
            if not proposals:
                terminated_reason = "error"
                break

            action = proposals[0]
            total_prompt_tokens += action.prompt_tokens
            total_completion_tokens += action.completion_tokens

            # 2. Check terminal final_answer
            if action.tool_name == "final_answer":
                final_answer = self.extract_final_answer(action)
                obs = self.execute_action(action, tuple(history))
                history.append(ReActStep(step_idx=step_idx, action=action, observation=obs))
                terminated_reason = "final_answer"
                break

            # 3. Execute action
            obs = self.execute_action(action, tuple(history))
            history.append(ReActStep(step_idx=step_idx, action=action, observation=obs))

            if obs.safety_violation:
                terminated_reason = "safety_violation"
                break

        total_latency = time.perf_counter() - start_time
        return Trajectory(
            question_id=question.question_id,
            steps=tuple(history),
            final=final_answer,
            terminated_reason=terminated_reason,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_latency_s=total_latency,
            seed=seed,
        )
