"""Research agent: builds a company profile and picks the most relevant case study."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
COMPANIES = ROOT / "data" / "companies"
CASES = json.loads((ROOT / "data" / "case_studies.json").read_text())


def company_profile(email: str) -> dict:
    domain = email.split("@")[-1].lower()
    path = COMPANIES / f"{domain}.md"
    if not path.exists():
        return {"domain": domain, "found": False}
    profile = {"domain": domain, "found": True}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            profile[k.strip()] = v.strip()
    if "employees" in profile:
        profile["employees"] = int(profile["employees"])
    return profile


def best_case_study(text: str, category: str) -> dict | None:
    docs = [c["text"] + " " + c["title"] for c in CASES]
    vec = TfidfVectorizer(stop_words="english").fit(docs + [text])
    sims = cosine_similarity(vec.transform([text]), vec.transform(docs))[0]
    ranked = sorted(zip(sims, CASES), key=lambda x: (x[1]["service"] == category, x[0]), reverse=True)
    score, case = ranked[0]
    if case["service"] != category and score < 0.1:
        return None
    return {**case, "similarity": round(float(score), 3)}


def research_agent(state: dict) -> dict:
    lead, intake = state["lead"], state["intake"]
    return {"research": {
        "company": company_profile(lead["from_email"]),
        "case_study": best_case_study(lead["body"], intake["category"]),
    }}
