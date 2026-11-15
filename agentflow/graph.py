"""A small state-graph engine for multi-agent workflows.

Concepts (similar to LangGraph):
  * State      - a dict shared by every agent; each node returns a partial update.
  * Node       - an agent function: state -> dict update.
  * Edge       - fixed transition, or a router function that picks the next node.
  * Interrupt  - pause before a node (e.g. human approval) and persist a checkpoint.
  * Trace      - every step is recorded with timing and a short summary.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

END = "__end__"


@dataclass
class Step:
    node: str
    started: float
    ms: float
    summary: str
    status: str = "ok"          # ok | error | interrupted


@dataclass
class Run:
    run_id: str
    state: dict
    next_node: str | None
    status: str = "running"     # running | interrupted | done | failed
    trace: list[Step] = field(default_factory=list)


class StateGraph:
    def __init__(self, name: str, max_steps: int = 25):
        self.name = name
        self.max_steps = max_steps
        self.nodes: dict[str, Callable[[dict], dict]] = {}
        self.edges: dict[str, str | Callable[[dict], str]] = {}
        self.entry: str | None = None
        self.interrupt_before: set[str] = set()
        self.summaries: dict[str, Callable[[dict], str]] = {}

    # ---------------- building
    def add_node(self, name: str, fn: Callable[[dict], dict], summary: Callable[[dict], str] | None = None):
        self.nodes[name] = fn
        if summary:
            self.summaries[name] = summary
        return self

    def add_edge(self, src: str, dst: str):
        self.edges[src] = dst
        return self

    def add_router(self, src: str, router: Callable[[dict], str]):
        self.edges[src] = router
        return self

    def set_entry(self, name: str):
        self.entry = name
        return self

    def interrupt(self, *names: str):
        self.interrupt_before.update(names)
        return self

    def describe(self) -> dict:
        return {"nodes": list(self.nodes), "entry": self.entry,
                "interrupts": sorted(self.interrupt_before),
                "edges": {k: (v if isinstance(v, str) else f"router:{v.__name__}") for k, v in self.edges.items()}}

    # ---------------- running
    def _next(self, node: str, state: dict) -> str:
        edge = self.edges.get(node, END)
        return edge(state) if callable(edge) else edge

    def start(self, state: dict, checkpointer=None) -> Run:
        run = Run(run_id=uuid.uuid4().hex[:10], state=dict(state), next_node=self.entry)
        return self._loop(run, checkpointer, resume=False)

    def resume(self, run: Run, update: dict, checkpointer=None) -> Run:
        """Continue an interrupted run after a human supplies `update` (e.g. approval)."""
        run.state.update(update)
        run.status = "running"
        return self._loop(run, checkpointer, resume=True)

    def _loop(self, run: Run, checkpointer, resume: bool) -> Run:
        steps = 0
        skip_interrupt_once = resume
        while run.next_node and run.next_node != END:
            node = run.next_node
            if node in self.interrupt_before and not skip_interrupt_once:
                run.status = "interrupted"
                run.trace.append(Step(node, time.time(), 0.0, "waiting for human approval", "interrupted"))
                if checkpointer:
                    checkpointer.save(run)
                return run
            skip_interrupt_once = False
            steps += 1
            if steps > self.max_steps:
                run.status = "failed"
                run.trace.append(Step(node, time.time(), 0.0, "max steps exceeded", "error"))
                break
            t0 = time.perf_counter()
            try:
                update = self.nodes[node](run.state) or {}
            except Exception as exc:  # keep the run inspectable instead of crashing
                run.status = "failed"
                run.trace.append(Step(node, time.time(), (time.perf_counter() - t0) * 1000, f"{type(exc).__name__}: {exc}", "error"))
                break
            run.state.update(update)
            summary = self.summaries.get(node, lambda s: "")(run.state)
            run.trace.append(Step(node, time.time(), round((time.perf_counter() - t0) * 1000, 2), summary))
            run.next_node = self._next(node, run.state)
        if run.status == "running":
            run.status = "done"
            run.next_node = None
        if checkpointer:
            checkpointer.save(run)
        return run
