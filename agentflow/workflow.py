"""The lead-to-reply workflow: 6 agents + a human approval gate.

intake -> research -> qualify --(not a fit)--> archive
                          \\--> write <-> review (max 2 revisions) -> human_review [pause] -> act
"""
from pathlib import Path

from .agents.actions import action_agent, archive_agent, human_review
from .agents.classifier import intake_agent
from .agents.qualifier import qualifier_agent
from .agents.research import research_agent
from .agents.reviewer import reviewer_agent
from .agents.writer import writer_agent
from .checkpoint import SQLiteCheckpointer
from .graph import END, StateGraph

ROOT = Path(__file__).resolve().parents[1]
MAX_REVISIONS = 2


def route_after_qualify(state: dict) -> str:
    return "archive" if state["qualification"]["route"] == "archive" else "write"


def route_after_review(state: dict) -> str:
    if state["review"]["passed"] or state["draft"]["version"] > MAX_REVISIONS:
        return "human_review"
    return "write"


def route_after_human(state: dict) -> str:
    return "act" if state.get("approved") else END


def build_workflow() -> StateGraph:
    g = StateGraph("lead_to_reply")
    g.add_node("intake", intake_agent,
               lambda s: f"{s['intake']['category_name']} ({s['intake']['confidence']:.0%}) · budget "
                         f"{'$' + format(s['intake']['budget'], ',') if s['intake']['budget'] else 'n/a'} · "
                         f"{str(s['intake']['timeline_weeks']) + ' wks' if s['intake']['timeline_weeks'] else 'no timeline'}")
    g.add_node("research", research_agent,
               lambda s: (s['research']['company'].get('name', 'no company profile') +
                          (f" · {s['research']['company'].get('employees')} employees" if s['research']['company'].get('employees') else "") +
                          (f" · case study: {s['research']['case_study']['id']}" if s['research']['case_study'] else "")))
    g.add_node("qualify", qualifier_agent,
               lambda s: f"score {s['qualification']['score']} · {s['qualification']['tier']} · route: {s['qualification']['route']}")
    g.add_node("write", writer_agent, lambda s: f"draft v{s['draft']['version']} ({len(s['draft']['body'].split())} words, {s['draft']['mode']})")
    g.add_node("review", reviewer_agent,
               lambda s: "passed all checks" if s['review']['passed'] else
               "needs revision: " + ", ".join(sorted({i['code'] for i in s['review']['issues']})))
    g.add_node("human_review", human_review, lambda s: "approved" if s.get("approved") else "rejected")
    g.add_node("act", action_agent, lambda s: f"email saved · CRM: {s['actions']['crm_status']} · follow-up {s['actions']['follow_up']}")
    g.add_node("archive", archive_agent, lambda s: "archived, no reply sent")

    g.set_entry("intake")
    g.add_edge("intake", "research").add_edge("research", "qualify")
    g.add_router("qualify", route_after_qualify)
    g.add_edge("write", "review")
    g.add_router("review", route_after_review)
    g.add_router("human_review", route_after_human)
    g.add_edge("act", END).add_edge("archive", END)
    g.interrupt("human_review")
    return g


def checkpointer() -> SQLiteCheckpointer:
    return SQLiteCheckpointer(ROOT / "data" / "runs.db")
