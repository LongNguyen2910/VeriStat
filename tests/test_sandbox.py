"""Unit tests for PythonSandbox and RunPythonTool."""

import pytest
from veristat.tools.python_tool import RunPythonTool
from veristat.tools.sandbox.ast_guard import check_python_ast


def test_sandbox_executes_valid_math():
    tool = RunPythonTool()
    code = """
x = 100
y = 125
growth = (y - x) / x * 100
print(f"Growth: {growth:.2f}%")
"""
    res = tool.execute(code=code)
    assert res.success is True
    assert res.output == "Growth: 25.00%"
    assert res.error is None


def test_ast_blocks_forbidden_imports():
    tool = RunPythonTool()

    # Block socket
    res_socket = tool.execute(code="import socket\ns = socket.socket()")
    assert res_socket.success is False
    assert res_socket.error_type == "AST_FORBIDDEN"
    assert "socket" in res_socket.error

    # Block subprocess
    res_sub = tool.execute(code="import subprocess\nsubprocess.run('ls')")
    assert res_sub.success is False
    assert res_sub.error_type == "AST_FORBIDDEN"

    # Block os
    res_os = tool.execute(code="import os\nos.listdir('.')")
    assert res_os.success is False
    assert res_os.error_type == "AST_FORBIDDEN"


def test_ast_blocks_eval_exec():
    tool = RunPythonTool()
    res_eval = tool.execute(code="eval('1 + 1')")
    assert res_eval.success is False
    assert res_eval.error_type == "AST_FORBIDDEN"

    res_exec = tool.execute(code="exec('print(123)')")
    assert res_exec.success is False
    assert res_exec.error_type == "AST_FORBIDDEN"


def test_sandbox_handles_infinite_loop_timeout():
    # Use shorter timeout for quick test
    tool = RunPythonTool(timeout_seconds=2.0)
    code = """
while True:
    pass
"""
    res = tool.execute(code=code)
    assert res.success is False
    assert res.error_type == "RESOURCE_LIMIT"
    assert "timeout" in res.error.lower()


def test_sandbox_runtime_catches_network_tampering():
    # Attempting to bypass via runtime socket invocation
    tool = RunPythonTool()
    code = """
try:
    import socket
    s = socket.socket()
except Exception as e:
    print(f"Caught: {e}")
"""
    # ast_guard will catch 'import socket' first
    res = tool.execute(code=code)
    assert res.success is False
    assert res.error_type == "AST_FORBIDDEN"
