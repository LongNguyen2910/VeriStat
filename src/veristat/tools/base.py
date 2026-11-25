"""Base tool interfaces and registry for VeriStat Agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """Standardized output of any tool execution in the VeriStat agent system."""

    tool_name: str
    success: bool
    output: str
    error: str | None = None
    error_type: str | None = None  # SQL_SYNTAX, AST_FORBIDDEN, RESOURCE_LIMIT, SECURITY_VIOLATION, RUNTIME_ERROR
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all VeriStat tools."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given keyword arguments."""
        ...


class ToolRegistry:
    """Registry managing available tools for the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return JSON schemas of all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            for t in self._tools.values()
        ]
