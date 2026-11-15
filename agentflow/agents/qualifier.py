"""Qualifier agent: scores the lead (0-100) and decides the route."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRICING = json.loads((ROOT / "data" / "pricing.json").read_text())


def score_lead(intake: dict, company: dict) -> dict:
    if intake["category"] == "not_fit":
        return {"score": 0, "tier": "not a fit", "route": "archive",
                "reasons": ["Message is not a request for an AI service (spam, job request or unrelated)."]}

    service = PRICING[intake["category"]]
    reasons, score = [], 40
    reasons.append(f"+40 fits a core service: {service['service']}")

    budget = intake["budget"]
    if budget is None:
        score += 15; reasons.append("+15 budget not stated")
    elif budget >= service["price_from"]:
        score += 30; reasons.append(f"+30 budget ${budget:,} covers the ${service['price_from']:,} starting price")
    elif budget >= 0.5 * service["price_from"]:
        score += 15; reasons.append(f"+15 budget ${budget:,} is below the ${service['price_from']:,} starting price")
    else:
        reasons.append(f"+0 budget ${budget:,} is far below the ${service['price_from']:,} starting price")

    weeks = intake["timeline_weeks"]
    min_weeks = int(service["weeks"].split("-")[0])
    if weeks is None:
        score += 10; reasons.append("+10 timeline not stated")
    elif weeks >= min_weeks:
        score += 15; reasons.append(f"+15 {weeks}-week timeline is realistic")
    else:
        score += 5; reasons.append(f"+5 {weeks}-week timeline is tight")

    employees = company.get("employees")
    if employees is None:
        score += 5; reasons.append("+5 company size unknown")
    elif employees >= 10:
        score += 15; reasons.append(f"+15 established company ({employees} employees)")
    else:
        score += 5; reasons.append(f"+5 small company ({employees} employees)")

    if intake["confidence"] < 0.35:
        reasons.append("! low classification confidence, check the category")

    tier = "hot" if score >= 90 else "warm" if score >= 60 else "low"
    route = "decline" if (budget is not None and budget < 0.5 * service["price_from"]) else "proposal"
    return {"score": score, "tier": tier, "route": route, "reasons": reasons}


def qualifier_agent(state: dict) -> dict:
    return {"qualification": score_lead(state["intake"], state["research"]["company"])}
