"""CLI runner for VeriStat Single Prompt Baseline (A.4 / D.3 Baseline).

Usage:
  # Local CPU test with mock:
  uv run python scripts/run_single_prompt.py --mock --questions data/benchmark/dev_30.jsonl --out results/single_dev.jsonl

  # Live Colab Pro run with vLLM server:
  uv run python scripts/run_single_prompt.py --vllm-url http://localhost:8000/v1 --questions data/benchmark/benchmark_400.jsonl --out results/single_test.jsonl
"""

from __future__ import annotations

from veristat.agents.single_prompt import main

if __name__ == "__main__":
    main()
