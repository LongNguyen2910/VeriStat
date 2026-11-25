"""VeriStat: Vietnamese Socio-Economic Statistical Agent with Verifiable Answers."""

from veristat.agents.contracts import AgentAction, ToolCall, VerificationResult
from veristat.agents.react import VeriStatReActAgent
from veristat.agents.state import AgentState, StepRecord
from veristat.tools.base import BaseTool, ToolRegistry, ToolResult
from veristat.tools.docs_tool import SearchDocsTool
from veristat.tools.python_tool import RunPythonTool
from veristat.tools.table_tool import QueryTableTool

__version__ = "0.1.0"

__all__ = [
    "AgentAction",
    "ToolCall",
    "VerificationResult",
    "AgentState",
    "StepRecord",
    "VeriStatReActAgent",
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    "QueryTableTool",
    "SearchDocsTool",
    "RunPythonTool",
]
