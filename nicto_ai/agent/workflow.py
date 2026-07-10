"""
NICTO AI - Workflow Engine
Chain multiple tools together in sequences with data passing between steps.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from .tool_agent import ToolAgent

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    on_success: Optional[str] = None
    on_fail: Optional[str] = None

    def __post_init__(self):
        self._status = StepStatus.PENDING
        self._result: Optional[Dict] = None
        self._error: Optional[str] = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    @property
    def status(self) -> StepStatus:
        return self._status

    @property
    def elapsed_ms(self) -> float:
        if self._start_time > 0 and self._end_time > 0:
            return (self._end_time - self._start_time) * 1000
        return 0.0


@dataclass
class WorkflowResult:
    """Result of executing a workflow."""
    name: str
    success: bool
    steps: Dict[str, Dict] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class Workflow:
    """A sequence of steps to execute."""

    def __init__(self, name: str, description: str = "", steps: List[WorkflowStep] = None):
        self.name = name
        self.description = description
        self.steps = steps or []

    def add_step(self, step: WorkflowStep) -> "Workflow":
        self.steps.append(step)
        return self


class WorkflowEngine:
    """
    Executes multi-tool workflows with data passing between steps.

    Usage:
        engine = WorkflowEngine(tool_agent)

        wf = Workflow("search_and_save", [
            WorkflowStep(name="search", tool="web_search", params={"query": "{input}"}),
            WorkflowStep(name="save", tool="knowledge_add", params={
                "text": "{search.results[0].summary}"
            }),
        ])

        result = engine.run(wf, {"input": "Python programming"})
    """

    def __init__(self, tool_agent: ToolAgent):
        self._agent = tool_agent
        self._custom_handlers: Dict[str, Callable] = {}
        self._presets: Dict[str, Workflow] = self._build_presets()
        self._context: Dict[str, Any] = {}

    def _build_presets(self) -> Dict[str, Workflow]:
        """Build pre-defined workflows."""
        return {
            "search_and_summarize": Workflow(
                name="search_and_summarize",
                description="Search the web and summarize results",
                steps=[
                    WorkflowStep(
                        name="search",
                        tool="web_search",
                        params={"query": "{input}"},
                    ),
                ],
            ),
            "search_and_save": Workflow(
                name="search_and_save",
                description="Search the web and save results to knowledge base",
                steps=[
                    WorkflowStep(
                        name="search",
                        tool="web_search",
                        params={"query": "{input}"},
                    ),
                ],
            ),
            "knowledge_search": Workflow(
                name="knowledge_search",
                description="Search NICTO's stored knowledge",
                steps=[
                    WorkflowStep(
                        name="query",
                        tool="knowledge_query",
                        params={"query": "{input}", "top_k": 5},
                    ),
                ],
            ),
        }

    def _resolve_params(self, params: Dict[str, Any], step_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve template variables like {input} and {step_name.field}."""
        resolved = {}
        # Merge input_data and context for template resolution
        format_vars = {**input_data, **self._context}
        for k, v in params.items():
            if isinstance(v, str):
                resolved[k] = v.format(**format_vars)
            else:
                resolved[k] = v
        return resolved

    def run(self, workflow: Workflow, input_data: Dict[str, Any] = None) -> WorkflowResult:
        """
        Execute a workflow.

        Args:
            workflow: Workflow to execute
            input_data: Input data with template variables

        Returns:
            WorkflowResult with step results
        """
        input_data = input_data or {}
        start_time = time.time()
        step_results: Dict[str, Dict] = {}

        # Merge input into context
        self._context.update(input_data)

        for step in workflow.steps:
            step._status = StepStatus.RUNNING
            step._start_time = time.time()

            # Check dependencies - get results from previous steps
            step_input = {"input": input_data.get("input", ""), "context": self._context.copy()}
            for dep in step.depends_on:
                if dep in step_results:
                    step_input[dep] = step_results[dep]
                else:
                    step._status = StepStatus.FAILED
                    step._error = f"Dependency '{dep}' not found"
                    break

            if step.status == StepStatus.FAILED:
                step_results[step.name] = {
                    "status": "failed",
                    "error": step._error,
                }
                if step.on_fail:
                    continue
                break

            # Resolve and execute
            try:
                resolved_params = self._resolve_params(step.params, step.name, step_input)
                logger.info("Executing step '%s': %s %s", step.name, step.tool, resolved_params)

                if step.name in self._custom_handlers:
                    result = self._custom_handlers[step.name](resolved_params)
                else:
                    cmd = f"/{step.tool} " + " ".join(f"{k}={v}" for k, v in resolved_params.items())
                    result = self._agent.process(cmd)

                step._status = StepStatus.SUCCESS
                step._end_time = time.time()
                step._error = None

                # Store result
                step_result = {
                    "status": "success",
                    "tool": step.tool,
                    "result": result.text if hasattr(result, "text") else result,
                    "elapsed_ms": step.elapsed_ms,
                }
                step_results[step.name] = step_result
                self._context[step.name] = step_result

            except Exception as e:
                step._status = StepStatus.FAILED
                step._error = str(e)
                step._end_time = time.time()
                step_result = {
                    "status": "failed",
                    "tool": step.tool,
                    "error": str(e),
                }
                step_results[step.name] = step_result
                self._context[step.name] = step_result

                if step.on_fail:
                    continue
                break

        elapsed = (time.time() - start_time) * 1000
        all_success = all(s["status"] == "success" for s in step_results.values())

        return WorkflowResult(
            name=workflow.name,
            success=all_success,
            steps=step_results,
            elapsed_ms=elapsed,
        )

    def list_presets(self) -> List[Dict]:
        """List available pre-built workflows."""
        return [
            {"name": name, "description": wf.description}
            for name, wf in self._presets.items()
        ]

    def register_handler(self, name: str, handler: Callable):
        """Register a custom handler for a workflow step."""
        self._custom_handlers[name] = handler
