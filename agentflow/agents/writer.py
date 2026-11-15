"""Writer agent: drafts the reply email from verified facts only.

* LLM mode (OPENAI_API_KEY set): GPT writes from a JSON fact sheet.
* Offline mode: deterministic composer, so the workflow runs free and reproducibly.
On a revision it reads the reviewer's issue codes and fixes them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
PRICING = json.loads((ROOT / "data" / "pricing.json").read_text())
SIGNATURE = "Best,\nAli\nAI/ML Engineer"


def fact_sheet(state: dict) -> dict:
    lead, intake, research, qual = state["lead"], state["intake"], state["research"], state["qualification"]
    service = PRICING[intake["category"]]
    return {
        "first_name": lead["from_name"].split()[0],
        "company": research["company"].get("name") or research["company"]["domain"],
        "industry": research["company"].get("industry"),
        "subject": lead["subject"],
        "service": service["service"], "pitch": service["pitch"], "short": service["short"],
        "deliverables": service["deliverables"],
        "price_from": service["price_from"], "price_to": service["price_to"], "weeks": service["weeks"],
        "client_budget": intake["budget"], "client_weeks": intake["timeline_weeks"],
        "case_study": research["case_study"], "route": qual["route"],
    }


def _money(n: int) -> str:
    return f"${n:,}"


def compose_offline(f: dict, issues: list[str]) -> str:
    compact = "too_long" in issues
    hello = f"Hi {f['first_name']},\n\n"
    if f["route"] == "decline":
        return (hello +
                f"Thanks for your message about \"{f['subject']}\". Our {f['short']} projects start at "
                f"{_money(f['price_from'])}, which is above the {_money(f['client_budget'])} you mentioned, so I don't "
                "want to over-promise. A ready-made tool for your platform may be the best first step. If the budget "
                f"changes later, I'd be glad to help {f['company']}.\n\n" + SIGNATURE)

    if f.get("industry"):
        ind = f["industry"].lower()
        context = f" For {'an' if ind[0] in 'aeiou' else 'a'} {ind} team like {f['company']}, "
    else:
        context = f" For {f['company']}, "
    deliverables = f["deliverables"][:2] if compact else f["deliverables"]
    body = hello + f"Thanks for your message about \"{f['subject']}\"." + context + \
        f"I'd build {f['pitch']} that includes:\n" + "".join(f"- {d}\n" for d in deliverables)

    fit = ""
    if f["client_budget"] and f["client_budget"] >= f["price_from"]:
        fit = f", which fits the {_money(f['client_budget'])} budget you mentioned"
    body += f"\nTypical investment is {_money(f['price_from'])}-{_money(f['price_to'])} over {f['weeks']} weeks{fit}."
    if f["client_weeks"]:
        body += f" Your {f['client_weeks']}-week target is noted."
    body += "\n"

    if "budget_gap" in issues and f["client_budget"]:
        body += (f"\nSince your budget is {_money(f['client_budget'])}, we could start with a focused MVP that covers "
                 "the most important use case first, then add the rest in a second phase.\n")
    if f["case_study"] and not compact:
        cs = f["case_study"]
        body += f"\nFor reference, a recent build of mine ({cs['title']}) {cs['proof']}.\n"
    body += "\nWould a 20-minute call this week work to confirm scope? I'll send a fixed quote right after.\n\n" + SIGNATURE
    return body


def compose_llm(f: dict, issues: list[str], feedback: list[str]) -> str:
    prompt = ("Write a short, friendly reply email (max 180 words) to this inbound lead. Use ONLY the facts in the "
              "JSON. Do not invent numbers, clients or results. End with a call-to-action for a 20-minute call and "
              "sign as 'Ali, AI/ML Engineer'.\n\nFacts:\n" + json.dumps(f, indent=2))
    if feedback:
        prompt += "\n\nFix these reviewer issues:\n- " + "\n- ".join(feedback)
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0.3,
              "messages": [{"role": "user", "content": prompt}]}, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def writer_agent(state: dict) -> dict:
    f = fact_sheet(state)
    review = state.get("review") or {}
    issues = [i["code"] for i in review.get("issues", [])]
    feedback = [i["message"] for i in review.get("issues", [])]
    version = state.get("draft", {}).get("version", 0) + 1
    body = compose_llm(f, issues, feedback) if os.getenv("OPENAI_API_KEY") else compose_offline(f, issues)
    return {"facts": f, "draft": {"version": version, "subject": f"Re: {f['subject']}", "body": body,
                                  "mode": "llm" if os.getenv("OPENAI_API_KEY") else "offline"}}
