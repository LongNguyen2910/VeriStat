import dataclasses
import json

import pytest

from veristat.agents.types import (
    FinalAnswer,
    LLMResponse,
    ReActStep,
    StepObservation,
    ToolAction,
    Trajectory,
)
from tests.agents.mocks import make_trajectory


def test_dataclasses_are_frozen():
    resp = LLMResponse(text="ok", prompt_tokens=10, completion_tokens=5, latency_s=0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        resp.text = "mutated"  # type: ignore[misc]

    traj = make_trajectory()
    with pytest.raises(dataclasses.FrozenInstanceError):
        traj.question_id = "other"  # type: ignore[misc]


def test_trajectory_derived_properties():
    s0 = ReActStep(
        step_idx=0,
        action=ToolAction("t0", "search_docs", {"query": "dân số"}, "raw0", 10, 5, 0.1),
        observation=StepObservation(output=[{"table_id": "NG2025_T017"}], ok=True),
    )
    s1 = ReActStep(
        step_idx=1,
        action=ToolAction("t1", "query_table", {"sql": "SELECT 1"}, "raw1", 20, 10, 0.2),
        observation=StepObservation(
            output=[1], ok=True, queried_tables=("NG2025_T017", "NG2025_T020")
        ),
    )
    s2 = ReActStep(
        step_idx=2,
        action=ToolAction("t2", "query_table", {"sql": "SELECT 2"}, "raw2", 20, 10, 0.2),
        observation=StepObservation(
            output=[2], ok=True, queried_tables=("NG2025_T020", "NG2025_T046")
        ),
    )
    s3 = ReActStep(
        step_idx=3,
        action=ToolAction("t3", "final_answer", {"answer": 5.0}, "raw3", 10, 5, 0.1),
        observation=None,
    )
    traj = Trajectory(
        question_id="Q1",
        steps=(s0, s1, s2, s3),
        final=FinalAnswer(5.0, "%", ("NG2025_T017",), "result = 5.0", True),
        terminated_reason="final_answer",
        total_prompt_tokens=60,
        total_completion_tokens=30,
        total_latency_s=0.6,
        seed=7,
    )

    assert traj.total_tokens == 90
    assert traj.tool_calls_count == 3  # excludes final_answer
    assert traj.has_safety_violation is False
    assert traj.all_queried_tables == ("NG2025_T017", "NG2025_T020", "NG2025_T046")

    unsafe_traj = make_trajectory(safety_violation=True)
    assert unsafe_traj.has_safety_violation is True


@pytest.mark.parametrize(
    "answer,unit",
    [
        (42.5, "%"),
        ("TP. Hồ Chí Minh", "tỉnh/thành"),
        (["Hà Nội", "TP. Hồ Chí Minh", "Hải Phòng"], "top-3"),
        (None, ""),
    ],
)
def test_trajectory_json_roundtrip_all_answer_types(answer, unit):
    traj = make_trajectory(
        answer=answer,
        unit=unit,
        py_values=(answer,) if answer is not None else (),
        parse_ok=answer is not None,
    )
    payload = json.dumps(traj.to_dict(), ensure_ascii=False)
    restored = Trajectory.from_dict(json.loads(payload))
    assert restored == traj
