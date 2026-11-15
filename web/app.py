"""Workflow console: inbox of runs, agent trace, graph view, draft review and human approval.

Run:  python web/app.py  ->  http://localhost:8200
"""
import json
import sys
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentflow.agents.classifier import LABELS  # noqa: E402
from agentflow.workflow import build_workflow, checkpointer  # noqa: E402

app = Flask(__name__, template_folder="templates")
graph, cp = build_workflow(), checkpointer()

NODES = {  # x, y positions for the graph view
    "intake": (70, 95), "research": (230, 95), "qualify": (390, 95), "archive": (390, 200),
    "write": (550, 95), "review": (710, 95), "human_review": (870, 95), "act": (1030, 95),
}
EDGES = [("intake", "research"), ("research", "qualify"), ("qualify", "archive"), ("qualify", "write"),
         ("write", "review"), ("review", "human_review"), ("human_review", "act")]


def status_label(run) -> str:
    if run["status"] == "interrupted":
        return "Needs approval"
    acts = run["state"].get("actions", {})
    if acts.get("crm_status") == "archived":
        return "Archived"
    if acts.get("crm_status") == "replied":
        return "Sent"
    return "Rejected" if run["status"] == "done" else run["status"].title()


@app.get("/")
def inbox():
    runs = cp.list()
    for r in runs:
        r["label"] = status_label(r)
    counts = {k: sum(1 for r in runs if r["label"] == k) for k in ("Needs approval", "Sent", "Archived", "Rejected")}
    return render_template("inbox.html", runs=runs, counts=counts, labels=LABELS)


@app.post("/process")
def process():
    for lead in json.loads((ROOT / "data" / "inbox.json").read_text()):
        graph.start({"lead": lead}, cp)
    return redirect(url_for("inbox"))


@app.get("/run/<run_id>")
def run_view(run_id):
    run = cp.load(run_id)
    visited = [s.node for s in run.trace]
    revisions = visited.count("write") - 1
    return render_template("run.html", run=run, s=run.state, visited=set(visited), nodes=NODES, edges=EDGES,
                           revisions=max(revisions, 0), labels=LABELS, status=status_label({"status": run.status, "state": run.state}))


@app.post("/run/<run_id>/decide")
def decide(run_id):
    run = cp.load(run_id)
    decision = {"action": request.form["action"]}
    if request.form.get("edited_body") and request.form["edited_body"].strip() != run.state["draft"]["body"].strip():
        decision["edited_body"] = request.form["edited_body"]
    graph.resume(run, {"human_decision": decision}, cp)
    return redirect(url_for("run_view", run_id=run_id))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8200)
