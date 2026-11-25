"""Evaluation metrics, statistical reproducibility, and checkpointed runner."""

from veristat.eval.metrics import (
    EvalRecord,
    aggregate_seed_summaries,
    bootstrap_ci,
    build_eval_record,
    build_stratified_subset,
    compute_citation_metrics,
    evaluate_prediction,
    summarize_records,
)
from veristat.eval.runner import (
    append_jsonl_record,
    collect_mcts_runs,
    collect_trajectory_pool,
    evaluate_from_pool,
    export_c4_traces,
    load_checkpoint_ids,
    load_questions,
    load_table_domain_map,
    read_jsonl,
)

__all__ = [
    "EvalRecord",
    "aggregate_seed_summaries",
    "append_jsonl_record",
    "bootstrap_ci",
    "build_eval_record",
    "build_stratified_subset",
    "collect_mcts_runs",
    "collect_trajectory_pool",
    "compute_citation_metrics",
    "evaluate_from_pool",
    "evaluate_prediction",
    "export_c4_traces",
    "load_checkpoint_ids",
    "load_questions",
    "load_table_domain_map",
    "read_jsonl",
    "summarize_records",
]
