"""Action agent and human-approval step: nothing is sent until a person approves."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTBOX = ROOT / "outbox"
CRM_DB = ROOT / "data" / "pipeline.db"


def _crm():
    conn = sqlite3.connect(CRM_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS leads(id TEXT PRIMARY KEY, name TEXT, email TEXT, company TEXT,
                    category TEXT, score INTEGER, tier TEXT, status TEXT, follow_up TEXT)""")
    return conn


def add_business_days(start: date, days: int) -> date:
    d = start
    while days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days -= 1
    return d


def human_review(state: dict) -> dict:
    """Applies the reviewer's decision that was supplied when the run was resumed."""
    decision = state.get("human_decision", {})
    update = {"approved": decision.get("action") == "approve"}
    if decision.get("edited_body"):
        update["draft"] = {**state["draft"], "body": decision["edited_body"], "edited_by_human": True}
    return update


def action_agent(state: dict) -> dict:
    lead, draft, qual = state["lead"], state["draft"], state["qualification"]
    OUTBOX.mkdir(exist_ok=True)
    msg = EmailMessage()
    msg["To"], msg["Subject"] = f"{lead['from_name']} <{lead['from_email']}>", draft["subject"]
    msg.set_content(draft["body"])
    path = OUTBOX / f"{lead['id']}.eml"
    path.write_bytes(bytes(msg))
    follow_up = add_business_days(date.today(), 3).isoformat()
    with _crm() as conn:
        conn.execute("INSERT OR REPLACE INTO leads VALUES (?,?,?,?,?,?,?,?,?)",
                     (lead["id"], lead["from_name"], lead["from_email"], state["facts"]["company"],
                      state["intake"]["category"], qual["score"], qual["tier"], "replied", follow_up))
    return {"actions": {"email_file": path.name, "crm_status": "replied", "follow_up": follow_up}}


def archive_agent(state: dict) -> dict:
    lead = state["lead"]
    with _crm() as conn:
        conn.execute("INSERT OR REPLACE INTO leads VALUES (?,?,?,?,?,?,?,?,?)",
                     (lead["id"], lead["from_name"], lead["from_email"], lead["from_email"].split("@")[-1],
                      "not_fit", 0, "not a fit", "archived", None))
    return {"actions": {"crm_status": "archived"}}
