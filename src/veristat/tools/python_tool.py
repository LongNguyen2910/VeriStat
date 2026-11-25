"""RunPython tool implementation wrapping the multi-tier sandbox."""

from __future__ import annotations

from typing import Any

from veristat.tools.base import BaseTool, ToolResult
from veristat.tools.sandbox.runner import PythonSandbox


class RunPythonTool(BaseTool):
    """Executes agent-generated Python calculation code within an isolated sandbox."""

    name = "run_python"
    description = (
        "Thực thi đoạn mã Python để tính toán số học (tăng trưởng, tỷ trọng, tương quan, xếp hạng) "
        "trong môi trường tiến trình cô lập hoàn toàn (không mạng, giới hạn tài nguyên). "
        "In kết quả cuối cùng ra stdout qua print()."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Mã nguồn Python độc lập, in kết quả ra stdout qua hàm print().",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        cpu_time_limit: int = 5,
        memory_limit_mb: int = 512,
    ) -> None:
        self.sandbox = PythonSandbox(
            timeout_seconds=timeout_seconds,
            cpu_time_limit=cpu_time_limit,
            memory_limit_mb=memory_limit_mb,
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        code = str(kwargs.get("code", "")).strip()
        if not code:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error="Python code string cannot be empty.",
                error_type="RUNTIME_ERROR",
            )
        return self.sandbox.run(code)
