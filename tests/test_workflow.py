import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentflow.agents.classifier import extract_budget, extract_timeline_weeks  # noqa: E402
from agentflow.agents.qualifier import score_lead  # noqa: E402
from agentflow.agents.reviewer import review_draft  # noqa: E402
from agentflow.checkpoint import SQLiteCheckpointer  # noqa: E402
from agentflow.graph import END, StateGraph  # noqa: E402
from agentflow.workflow import build_workflow  # noqa: E402

INBOX = {l["id"]: l for l in json.loads((ROOT / "data/inbox.json").read_text())}


# ------------------------------------------------------------ graph engine
def test_graph_runs_nodes_in_order():
    g = StateGraph("t").add_node("a", lambda s: {"x": 1}).add_node("b", lambda s: {"y": s["x"] + 1})
    g.set_entry("a").add_edge("a", "b").add_edge("b", END)
    run = g.start({})
    assert run.status == "done" and run.state["y"] == 2 and [s.node for s in run.trace] == ["a", "b"]


def test_router_loop_stops_at_max_steps():
    g = StateGraph("t", max_steps=5).add_node("a", lambda s: {"n": s.get("n", 0) + 1})
    g.set_entry("a").add_router("a", lambda s: "a")
    run = g.start({})
    assert run.status == "failed" and run.state["n"] == 5


def test_node_error_is_captured_not_raised():
    g = StateGraph("t").add_node("a", lambda s: 1 / 0).set_entry("a")
    run = g.start({})
    assert run.status == "failed" and "ZeroDivisionError" in run.trace[-1].summary


def test_interrupt_checkpoint_and_resume(tmp_path):
    cp = SQLiteCheckpointer(tmp_path / "runs.db")
    g = StateGraph("t").add_node("a", lambda s: {}).add_node("gate", lambda s: {"ok": s["decision"]})
    g.set_entry("a").add_edge("a", "gate").add_edge("gate", END).interrupt("gate")
    run = g.start({}, cp)
    assert run.status == "interrupted"
    restored = cp.load(run.run_id)                      # e.g. after a server restart
    done = g.resume(restored, {"decision": True}, cp)
    assert done.status == "done" and done.state["ok"] is True


# ------------------------------------------------------------ agents
@pytest.mark.parametrize("text,budget", [("Budget is around $7,500", 7500), ("about 5000 USD", 5000),
                                         ("~$15k, 6 weeks", 15000), ("Budget 4k, 2 weeks", 4000), ("no money talk", None)])
def test_budget_extraction(text, budget):
    assert extract_budget(text) == budget


@pytest.mark.parametrize("text,weeks", [("launch in 4 weeks", 4), ("two months", 8), ("next week", 1), ("soon", None)])
def test_timeline_extraction(text, weeks):
    assert extract_timeline_weeks(text) == weeks


def test_qualifier_routes():
    company = {"employees": 50}
    base = {"confidence": 0.9, "timeline_weeks": 4}
    assert score_lead({**base, "category": "not_fit", "budget": None}, company)["route"] == "archive"
    assert score_lead({**base, "category": "ai_chatbot", "budget": 800}, company)["route"] == "decline"
    hot = score_lead({**base, "category": "ai_chatbot", "budget": 9000}, company)
    assert hot["route"] == "proposal" and hot["tier"] == "hot"


# ------------------------------------------------------------ full workflow
def test_spam_is_archived_without_reply():
    run = build_workflow().start({"lead": INBOX["L-1044"]})
    assert run.status == "done" and run.state["actions"]["crm_status"] == "archived"


def test_budget_gap_triggers_revision_then_passes():
    run = build_workflow().start({"lead": INBOX["L-1047"]})
    versions = [h["passed"] for h in run.state["review_history"]]
    assert versions == [False, True] and "mvp" in run.state["draft"]["body"].lower()


def test_nothing_is_sent_before_approval_and_reject_sends_nothing():
    g = build_workflow()
    run = g.start({"lead": INBOX["L-1043"]})
    assert run.status == "interrupted" and "actions" not in run.state
    rejected = g.resume(run, {"human_decision": {"action": "reject"}})
    assert rejected.status == "done" and "actions" not in rejected.state


def test_reviewer_flags_invented_price():
    run = build_workflow().start({"lead": INBOX["L-1041"]})
    f, body = run.state["facts"], run.state["draft"]["body"]
    assert review_draft(body, f)["passed"]
    bad = body.replace(f"${f['price_to']:,}", "$99,000")
    assert any(i["code"] == "grounding" for i in review_draft(bad, f)["issues"])
