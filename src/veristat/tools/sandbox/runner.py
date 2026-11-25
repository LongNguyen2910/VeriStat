"""Subprocess-level isolated sandbox runner for executing Agent Python code."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from veristat.tools.base import ToolResult
from veristat.tools.sandbox.ast_guard import check_python_ast

# Runtime wrapper script template that injects network blocking and safe execution context
_WRAPPER_TEMPLATE = """\
# --- VeriStat Sandbox Runtime Isolation ---
import sys

# 1. Disable network creation at socket level
try:
    import socket
    def _blocked_socket(*args, **kwargs):
        raise RuntimeError("SecurityViolation: Network access is strictly prohibited in sandbox.")
    socket.socket = _blocked_socket
    socket.create_connection = _blocked_socket
except Exception:
    pass

# 2. Execute user code
__name__ = "__main__"
{user_code}
"""


class PythonSandbox:
    """Executes Python code in an isolated subprocess with resource limits and safety controls."""

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        cpu_time_limit: int = 5,
        memory_limit_mb: int = 512,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.cpu_time_limit = cpu_time_limit
        self.memory_limit_mb = memory_limit_mb
        self.is_linux = platform.system() == "Linux"

    def run(self, code: str) -> ToolResult:
        # Pillar 1: Static AST Linting
        ast_check = check_python_ast(code)
        if not ast_check.is_safe:
            return ToolResult(
                tool_name="run_python",
                success=False,
                output="",
                error=f"AST_FORBIDDEN: {ast_check.reason}",
                error_type="AST_FORBIDDEN",
                metadata={"reason": ast_check.reason},
            )

        # Pillar 2 & 3: Resource Limiting & OS-level Subprocess Isolation
        with tempfile.TemporaryDirectory(prefix="veristat_sandbox_") as temp_dir:
            temp_path = Path(temp_dir)
            script_path = temp_path / "sandbox_exec.py"

            wrapped_code = _WRAPPER_TEMPLATE.format(user_code=code)
            script_path.write_text(wrapped_code, encoding="utf-8")

            # Clean environment: preserve minimal system paths on Windows so Python DLLs load
            clean_env = {}
            if platform.system() == "Windows":
                for k in ("SYSTEMROOT", "WINDIR", "PATH"):
                    if k in os.environ:
                        clean_env[k] = os.environ[k]
            else:
                for k in ("PATH", "LANG", "LC_ALL"):
                    if k in os.environ:
                        clean_env[k] = os.environ[k]

            cmd = [sys.executable, "-I", "-E", "-s", str(script_path.name)]

            preexec_fn = None
            if self.is_linux:
                try:
                    import resource

                    def _set_linux_limits():
                        # Set CPU time limit
                        resource.setrlimit(
                            resource.RLIMIT_CPU,
                            (self.cpu_time_limit, self.cpu_time_limit),
                        )
                        # Set virtual memory limit (AS)
                        mem_bytes = self.memory_limit_mb * 1024 * 1024
                        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

                    preexec_fn = _set_linux_limits
                except ImportError:
                    preexec_fn = None

            start_time = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    env=clean_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                    preexec_fn=preexec_fn,
                )
                elapsed = time.perf_counter() - start_time
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start_time
                return ToolResult(
                    tool_name="run_python",
                    success=False,
                    output="",
                    error=f"RESOURCE_LIMIT: Execution exceeded timeout of {self.timeout_seconds} seconds.",
                    error_type="RESOURCE_LIMIT",
                    metadata={"cpu_time": elapsed, "timeout": True},
                )
            except Exception as e:
                return ToolResult(
                    tool_name="run_python",
                    success=False,
                    output="",
                    error=f"RUNTIME_ERROR: Failed to invoke sandbox subprocess: {e}",
                    error_type="RUNTIME_ERROR",
                )

            stdout = proc.stdout
            stderr = proc.stderr

            if proc.returncode != 0:
                # Check for our network violation or resource limits in stderr
                if "SecurityViolation" in stderr:
                    return ToolResult(
                        tool_name="run_python",
                        success=False,
                        output=stdout,
                        error=f"SECURITY_VIOLATION: {stderr.strip()}",
                        error_type="SECURITY_VIOLATION",
                        metadata={"cpu_time": elapsed},
                    )
                return ToolResult(
                    tool_name="run_python",
                    success=False,
                    output=stdout,
                    error=f"RUNTIME_ERROR: Python process exited with code {proc.returncode}:\n{stderr.strip()}",
                    error_type="RUNTIME_ERROR",
                    metadata={"cpu_time": elapsed, "returncode": proc.returncode},
                )

            return ToolResult(
                tool_name="run_python",
                success=True,
                output=stdout.strip(),
                metadata={"cpu_time": elapsed},
            )
