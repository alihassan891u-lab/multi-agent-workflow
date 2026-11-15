"""Reviewer agent: quality and safety gate before a human sees the draft.

Checks every draft against the fact sheet:
  grounding   - every $ amount and week count in the draft must come from the facts
  claims      - no guarantees or unverifiable superlatives
  personal    - greets the contact by first name and mentions the company
  budget_gap  - if the price starts above the client's budget, the draft must offer an option
  structure   - proposal drafts include price, timeline and a call-to-action
  too_long    - 180 words maximum
"""
from __future__ import annotations

import re

BANNED = ["guarantee", "guaranteed", "100% accurate", "best in the world", "cheapest", "no risk"]
MAX_WORDS = 180


def _allowed_numbers(f: dict) -> tuple[set[int], set[int]]:
    money = {f["price_from"], f["price_to"]}
    if f["client_budget"]:
        money.add(f["client_budget"])
    weeks = {int(w) for w in str(f["weeks"]).split("-")}
    if f["client_weeks"]:
        weeks.add(f["client_weeks"])
    return money, weeks


def review_draft(body: str, f: dict) -> dict:
    issues = []
    money_ok, weeks_ok = _allowed_numbers(f)
    text = body.lower()

    for m in re.finditer(r"\$(\d[\d,]*)", body):
        value = int(m.group(1).replace(",", ""))
        if value not in money_ok:
            issues.append({"code": "grounding", "message": f"Amount ${value:,} is not in the price list or the client's message."})
    for m in re.finditer(r"(\d+)(?:-(\d+))?[\s-]weeks?", text):
        for g in m.groups():
            if g and int(g) not in weeks_ok:
                issues.append({"code": "grounding", "message": f"'{m.group(0)}' does not match the service timeline or the client's target."})
    for phrase in BANNED:
        if phrase in text:
            issues.append({"code": "claims", "message": f"Remove the unverifiable claim '{phrase}'."})
    if f["first_name"].lower() not in text[:60]:
        issues.append({"code": "personal", "message": f"Greet {f['first_name']} by name."})
    if f["route"] == "proposal":
        if f["company"].lower() not in text:
            issues.append({"code": "personal", "message": f"Mention {f['company']} to personalise the reply."})
        if f["client_budget"] and f["client_budget"] < f["price_from"] and "mvp" not in text and "phase" not in text:
            issues.append({"code": "budget_gap", "message": f"Price starts above the client's ${f['client_budget']:,} budget; offer a phased or MVP option."})
        if "$" not in body or "week" not in text:
            issues.append({"code": "structure", "message": "Include the price range and timeline."})
        closing = text.split("best,")[0][-220:]          # the paragraph before the sign-off
        if "?" not in closing or not any(w in closing for w in ("call", "meeting", "chat")):
            issues.append({"code": "structure", "message": "End with a clear call-to-action question."})
    words = len(body.split())
    if words > MAX_WORDS:
        issues.append({"code": "too_long", "message": f"{words} words; keep it under {MAX_WORDS}."})
    return {"passed": not issues, "issues": issues, "words": words}


def reviewer_agent(state: dict) -> dict:
    review = review_draft(state["draft"]["body"], state["facts"])
    history = state.get("review_history", []) + [{"version": state["draft"]["version"], **review}]
    return {"review": review, "review_history": history}
