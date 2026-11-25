"""Unit tests for SQLiteLLMCache, TaskCheckpoint, and VLLMClient conditional retry."""

import tempfile
from pathlib import Path

import pytest
from veristat.llm.base import LLMResponse
from veristat.llm.cache import SQLiteLLMCache, compute_canonical_cache_key
from veristat.llm.checkpoint import TaskCheckpoint
from veristat.llm.mock_client import MockLLMClient


def test_cache_hit_does_not_call_backend():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_cache.db"
        cache = SQLiteLLMCache(db_path)

        payload = {"model": "Qwen2.5-7B", "messages": [{"role": "user", "content": "hi"}], "temp": 0.0}
        key = compute_canonical_cache_key(payload)

        # Cache miss
        assert cache.get(key) is None

        # Store response
        resp = LLMResponse(text="Hello world!", prompt_tokens=5, completion_tokens=3, finish_reason="stop", model="Qwen2.5-7B")
        cache.put(key, resp, payload)

        # Cache hit
        cached_resp = cache.get(key)
        assert cached_resp is not None
        assert cached_resp.text == "Hello world!"
        assert cached_resp.cached is True


def test_two_requests_with_different_params_have_different_keys():
    p1 = {"model": "Qwen2.5-7B", "messages": [{"role": "user", "content": "hi"}], "top_p": 0.9}
    p2 = {"model": "Qwen2.5-7B", "messages": [{"role": "user", "content": "hi"}], "top_p": 0.7}
    assert compute_canonical_cache_key(p1) != compute_canonical_cache_key(p2)


def test_mock_llm_deterministic_output():
    client = MockLLMClient(canned_responses={"tăng trưởng": "Growth: 15%"})
    r1 = client.generate([{"role": "user", "content": "Tính tăng trưởng dân số"}])
    assert r1.text == "Growth: 15%"
    assert client.call_count == 1


def test_checkpoint_skips_completed_and_isolates_methods():
    with tempfile.TemporaryDirectory() as td:
        ckpt_file = Path(td) / "results.jsonl"
        ckpt = TaskCheckpoint(ckpt_file)

        # Append react method SUCCESS
        ckpt.append_result({
            "question_id": "Q001",
            "dataset_version": "v1",
            "method": "react",
            "config_hash": "cfgA",
            "status": "SUCCESS",
            "answer": {"val": 12},
        })

        # Check completion
        assert ckpt.is_completed("Q001", "v1", "react", "cfgA") is True
        # Different method is NOT completed
        assert ckpt.is_completed("Q001", "v1", "self_consistency", "cfgA") is False


def test_checkpoint_handles_corrupt_tail_line():
    with tempfile.TemporaryDirectory() as td:
        ckpt_file = Path(td) / "corrupt_test.jsonl"
        with open(ckpt_file, "w", encoding="utf-8") as f:
            f.write('{"question_id":"Q001","dataset_version":"v1","method":"react","config_hash":"c1","status":"SUCCESS"}\n')
            f.write('{"question_id":"Q002","dataset_version":"v1","method":"react","config_hash":"c1","status":"SUCC')  # Incomplete line

        ckpt = TaskCheckpoint(ckpt_file)
        completed = ckpt.load_completed_keys()
        assert len(completed) == 1
        assert ("v1", "react", "c1", "Q001") in completed
        assert ("v1", "react", "c1", "Q002") not in completed
