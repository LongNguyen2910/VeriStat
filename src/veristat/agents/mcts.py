"""Monte Carlo Tree Search with UCT selection (A.3) in pure Python + NumPy."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

from veristat.agents.types import (
    FinalAnswer,
    ReActStep,
    StepEnvironment,
    TerminatedReason,
    ToolAction,
    Trajectory,
)
from veristat.data.schema import BenchmarkQuestion
from veristat.verifier.hybrid import HybridVerifier, VerificationResult


def canonical_action_key(action: ToolAction) -> str:
    """Return a deterministic deduplication key for a proposed `ToolAction`."""
    args_json = json.dumps(
        action.tool_args, sort_keys=True, ensure_ascii=False, default=str
    )
    return f"{action.tool_name}:{args_json}"


@dataclass
class MCTSNode:
    """Tree node storing step transition, UCT statistics, and exhaustion state."""

    node_id: int
    parent: MCTSNode | None
    step: ReActStep | None
    depth: int
    action_key: str
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    is_expanded: bool = False
    is_terminal: bool = False
    exhausted: bool = False
    terminal_trajectory: Trajectory | None = None
    terminal_verification: VerificationResult | None = None
    terminal_score: float | None = None

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def uct_score(self, c: float = math.sqrt(2.0)) -> float:
        """Compute UCT = Q(s,a) + c * sqrt(ln N(s) / N(s,a)), +inf if unvisited."""
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent is not None else self.visits
        return self.q_value + c * math.sqrt(
            math.log(max(1, parent_visits)) / self.visits
        )

    def history(self) -> tuple[ReActStep, ...]:
        """Return the ReActStep path from root to this node."""
        rev: list[ReActStep] = []
        cur: MCTSNode | None = self
        while cur is not None:
            if cur.step is not None:
                rev.append(cur.step)
            cur = cur.parent
        rev.reverse()
        return tuple(rev)

    def refresh_exhausted(self) -> None:
        """Update `exhausted` from this node upward so finished subtrees aren't re-picked."""
        cur: MCTSNode | None = self
        while cur is not None:
            if cur.is_terminal:
                cur.exhausted = True
            elif cur.is_expanded:
                cur.exhausted = all(ch.exhausted for ch in cur.children)
            cur = cur.parent


@dataclass(frozen=True)
class MCTSResult:
    """Complete outcome of MCTS-UCT search with full tree token accounting."""

    best_trajectory: Trajectory
    answer: float | str | list[str] | None
    best_score: float
    best_verification: VerificationResult | None
    iterations_run: int
    nodes_created: int
    terminal_nodes_evaluated: int
    terminated_reason: TerminatedReason
    generation_prompt_tokens: int
    generation_completion_tokens: int
    verifier_prompt_tokens: int
    verifier_completion_tokens: int
    total_tokens: int
    total_latency_s: float
    root_visits: int


def _backpropagate(node: MCTSNode, value: float) -> None:
    cur: MCTSNode | None = node
    while cur is not None:
        cur.visits += 1
        cur.value_sum += float(value)
        cur = cur.parent


def _build_trajectory_from_steps(
    question_id: str,
    steps: tuple[ReActStep, ...],
    *,
    final: FinalAnswer | None,
    terminated_reason: TerminatedReason,
    seed: int | None,
) -> Trajectory:
    prompt_toks = sum(s.action.prompt_tokens for s in steps)
    comp_toks = sum(s.action.completion_tokens for s in steps)
    latency = sum(
        s.action.latency_s + (s.observation.latency_s if s.observation else 0.0)
        for s in steps
    )
    return Trajectory(
        question_id=question_id,
        steps=steps,
        final=final,
        terminated_reason=terminated_reason,
        total_prompt_tokens=prompt_toks,
        total_completion_tokens=comp_toks,
        total_latency_s=latency,
        seed=seed,
    )


def _select_node(root: MCTSNode, c_uct: float) -> MCTSNode:
    cur = root
    while cur.is_expanded and not cur.is_terminal:
        candidates = [ch for ch in cur.children if not ch.exhausted]
        if not candidates:
            break
        cur = max(candidates, key=lambda ch: (ch.uct_score(c_uct), -ch.node_id))
    return cur


def _has_valid_final(traj: Trajectory | None) -> bool:
    if traj is None or traj.has_safety_violation or traj.final is None:
        return False
    return bool(traj.final.parse_ok and traj.final.answer is not None)


def run_mcts(
    env: StepEnvironment,
    verifier: HybridVerifier,
    question: BenchmarkQuestion,
    *,
    max_iterations: int = 12,
    max_depth: int = 6,
    branching_factor: int = 2,
    c_uct: float = math.sqrt(2.0),
    temperature: float = 0.7,
    token_budget: int | None = None,
    max_seconds: float = 120.0,
    seed: int = 42,
) -> MCTSResult:
    """Run MCTS-UCT search over step-level ReAct trajectories with hard token budget."""
    t0 = time.monotonic()
    root = MCTSNode(node_id=0, parent=None, step=None, depth=0, action_key="__root__")
    next_node_id = 1

    gen_prompt_tokens = 0
    gen_completion_tokens = 0
    ver_prompt_tokens = 0
    ver_completion_tokens = 0
    total_latency_s = 0.0

    terminal_nodes: list[MCTSNode] = []
    visited_nonterminal_nodes: list[MCTSNode] = []
    terminal_nodes_evaluated = 0
    iterations_run = 0
    stop_reason: TerminatedReason = "max_steps"
    saw_safety_pruning = False

    def _tokens_used() -> int:
        return (
            gen_prompt_tokens
            + gen_completion_tokens
            + ver_prompt_tokens
            + ver_completion_tokens
        )

    for it in range(max_iterations):
        if root.exhausted:
            break
        if token_budget is not None and _tokens_used() >= token_budget:
            stop_reason = "token_budget"
            break
        if (time.monotonic() - t0) >= max_seconds:
            stop_reason = "timeout"
            break

        iterations_run += 1
        node = _select_node(root, c_uct)

        if node.depth >= max_depth:
            node.is_expanded = True
            node.is_terminal = True
            node.terminal_score = 0.0
            _backpropagate(node, 0.0)
            node.refresh_exhausted()
            continue

        node.is_expanded = True
        history = node.history()
        proposed = env.propose_actions(
            question,
            history,
            k=branching_factor,
            temperature=temperature,
            seed=seed + it * 100 + node.node_id,
        )
        for act in proposed:
            gen_prompt_tokens += int(act.prompt_tokens)
            gen_completion_tokens += int(act.completion_tokens)
            total_latency_s += float(act.latency_s)

        existing_keys = {ch.action_key for ch in node.children}
        unique_actions: list[tuple[ToolAction, str]] = []
        for act in proposed:
            key = canonical_action_key(act)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            unique_actions.append((act, key))

        if not unique_actions:
            _backpropagate(node, 0.0)
            node.refresh_exhausted()
            continue

        for act, key in unique_actions:
            if token_budget is not None and _tokens_used() >= token_budget:
                stop_reason = "token_budget"
                break
            if (time.monotonic() - t0) >= max_seconds:
                stop_reason = "timeout"
                break

            child_depth = node.depth + 1
            step_idx = len(history)

            if act.tool_name == "final_answer":
                step = ReActStep(step_idx=step_idx, action=act, observation=None)
                child = MCTSNode(
                    node_id=next_node_id,
                    parent=node,
                    step=step,
                    depth=child_depth,
                    action_key=key,
                    is_expanded=True,
                    is_terminal=True,
                    exhausted=True,
                )
                next_node_id += 1
                node.children.append(child)

                final_obj = env.extract_final_answer(act)
                traj = _build_trajectory_from_steps(
                    question.question_id,
                    child.history(),
                    final=final_obj,
                    terminated_reason="final_answer",
                    seed=seed + child.node_id,
                )
                ver = verifier.verify(question, traj, seed=seed + child.node_id)
                ver_prompt_tokens += int(ver.verifier_prompt_tokens)
                ver_completion_tokens += int(ver.verifier_completion_tokens)
                total_latency_s += float(ver.latency_s)

                child.terminal_trajectory = traj
                child.terminal_verification = ver
                child.terminal_score = float(ver.score)
                terminal_nodes.append(child)
                terminal_nodes_evaluated += 1
                _backpropagate(child, ver.score)
                continue

            obs = env.execute_action(act, history)
            total_latency_s += float(obs.latency_s)
            step = ReActStep(step_idx=step_idx, action=act, observation=obs)
            child = MCTSNode(
                node_id=next_node_id,
                parent=node,
                step=step,
                depth=child_depth,
                action_key=key,
            )
            next_node_id += 1
            node.children.append(child)

            if obs.safety_violation:
                saw_safety_pruning = True
                child.is_expanded = True
                child.is_terminal = True
                child.exhausted = True
                child.terminal_score = 0.0
                child.terminal_trajectory = _build_trajectory_from_steps(
                    question.question_id,
                    child.history(),
                    final=None,
                    terminated_reason="safety_violation",
                    seed=seed + child.node_id,
                )
                terminal_nodes.append(child)
                _backpropagate(child, 0.0)
            elif child_depth >= max_depth:
                child.is_expanded = True
                child.is_terminal = True
                child.exhausted = True
                child.terminal_score = 0.0
                child.terminal_trajectory = _build_trajectory_from_steps(
                    question.question_id,
                    child.history(),
                    final=None,
                    terminated_reason="max_steps",
                    seed=seed + child.node_id,
                )
                terminal_nodes.append(child)
                _backpropagate(child, 0.0)
            else:
                interm_score = verifier.score_intermediate(child.history())
                visited_nonterminal_nodes.append(child)
                _backpropagate(child, interm_score)

        node.refresh_exhausted()

    valid_terminals = [n for n in terminal_nodes if _has_valid_final(n.terminal_trajectory)]
    if valid_terminals:
        best_node = max(
            valid_terminals,
            key=lambda n: (float(n.terminal_score or 0.0), n.visits, -n.node_id),
        )
        assert best_node.terminal_trajectory is not None
        best_traj = best_node.terminal_trajectory
        best_ans = best_traj.final.answer if best_traj.final is not None else None
        best_score = float(best_node.terminal_score or 0.0)
        best_ver = best_node.terminal_verification
        final_reason: TerminatedReason = "final_answer"
    else:
        if token_budget is not None and _tokens_used() >= token_budget:
            final_reason = "token_budget"
        elif stop_reason == "timeout":
            final_reason = "timeout"
        elif saw_safety_pruning and not visited_nonterminal_nodes:
            final_reason = "safety_violation"
        else:
            final_reason = stop_reason

        if terminal_nodes:
            best_node = max(
                terminal_nodes,
                key=lambda n: (float(n.terminal_score or 0.0), n.visits, -n.node_id),
            )
            best_traj = best_node.terminal_trajectory or _build_trajectory_from_steps(
                question.question_id,
                best_node.history(),
                final=None,
                terminated_reason=final_reason,
                seed=seed,
            )
            best_ver = best_node.terminal_verification
            best_score = float(best_node.terminal_score or 0.0)
        elif visited_nonterminal_nodes:
            best_node = max(
                visited_nonterminal_nodes,
                key=lambda n: (n.q_value, n.visits, -n.node_id),
            )
            best_traj = _build_trajectory_from_steps(
                question.question_id,
                best_node.history(),
                final=None,
                terminated_reason=final_reason,
                seed=seed,
            )
            best_ver = None
            best_score = best_node.q_value
        else:
            best_traj = _build_trajectory_from_steps(
                question.question_id,
                (),
                final=None,
                terminated_reason=final_reason,
                seed=seed,
            )
            best_ver = None
            best_score = 0.0
        best_ans = None

    return MCTSResult(
        best_trajectory=best_traj,
        answer=best_ans,
        best_score=best_score,
        best_verification=best_ver,
        iterations_run=iterations_run,
        nodes_created=next_node_id,
        terminal_nodes_evaluated=terminal_nodes_evaluated,
        terminated_reason=final_reason,
        generation_prompt_tokens=gen_prompt_tokens,
        generation_completion_tokens=gen_completion_tokens,
        verifier_prompt_tokens=ver_prompt_tokens,
        verifier_completion_tokens=ver_completion_tokens,
        total_tokens=_tokens_used(),
        total_latency_s=total_latency_s,
        root_visits=root.visits,
    )
