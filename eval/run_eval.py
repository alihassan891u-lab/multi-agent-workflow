"""Evaluate the agents on the held-out test leads (data/test_leads.json).

Usage: python eval/run_eval.py
Writes: eval/results.json
"""
import json
import statistics
import sys
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentflow.agents.classifier import LABELS, build_model, classify, extract_budget, extract_timeline_weeks, train  # noqa: E402
from agentflow.agents.reviewer import review_draft  # noqa: E402
from agentflow.workflow import build_workflow  # noqa: E402

train_rows = json.loads((ROOT / "data/train_leads.json").read_text())
test_rows = json.loads((ROOT / "data/test_leads.json").read_text())


def corruptions(body: str, f: dict) -> dict:
    """Realistic mistakes an LLM writer can make; the reviewer must catch each one."""
    wrong_price = f"${f['price_to'] + 1500:,}"
    return {
        "invented_price": body.replace(f"${f['price_to']:,}", wrong_price, 1),
        "invented_timeline": body.replace(f"over {f['weeks']} weeks", "in 1 week", 1),
        "guarantee_claim": body.replace("Would a 20-minute call", "Results are guaranteed. Would a 20-minute call", 1),
        "missing_name": body.replace(f"Hi {f['first_name']},", "Hi there,", 1),
        "missing_company": body.replace(f["company"], "your team"),
        "missing_cta": body.split("Would a 20-minute call")[0] + "Best,\nAli",
    }


def main():
    model = train()
    cv = cross_val_score(build_model(), [r["text"] for r in train_rows], [r["label"] for r in train_rows], cv=5)
    y_true = [r["label"] for r in test_rows]
    y_pred = [classify(r["text"])["category"] for r in test_rows]
    labels = list(LABELS)

    budget_ok = sum(extract_budget(r["text"]) == r["budget"] for r in test_rows)
    weeks_ok = sum(extract_timeline_weeks(r["text"]) == r["timeline_weeks"] for r in test_rows)

    g = build_workflow()
    timings, reached_human, first_pass, revisions, caught, total_corrupt = [], 0, 0, [], 0, 0
    per_type = {}
    for i, r in enumerate(test_rows):
        lead = {"id": f"T-{i}", "from_name": "Alex Morgan", "from_email": f"alex@testco{i}.com",
                "subject": "New inquiry", "body": r["text"]}
        t0 = time.perf_counter()
        run = g.start({"lead": lead})
        timings.append((time.perf_counter() - t0) * 1000)
        if run.status != "interrupted":
            continue
        reached_human += 1
        hist = run.state["review_history"]
        first_pass += hist[0]["passed"]
        revisions.append(len(hist) - 1)
        f = run.state["facts"]
        if f["route"] != "proposal":
            continue
        for kind, bad in corruptions(run.state["draft"]["body"], f).items():
            if bad == run.state["draft"]["body"]:
                continue
            total_corrupt += 1
            hit = not review_draft(bad, f)["passed"]
            caught += hit
            per_type.setdefault(kind, [0, 0])
            per_type[kind][0] += hit
            per_type[kind][1] += 1

    results = {
        "train_examples": len(train_rows), "test_examples": len(test_rows),
        "classifier": {
            "test_accuracy": round(accuracy_score(y_true, y_pred), 3),
            "test_macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
            "cv5_train_accuracy": round(float(cv.mean()), 3),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "errors": [{"text": r["text"][:90], "true": t, "pred": p} for r, t, p in zip(test_rows, y_true, y_pred) if t != p],
        },
        "extraction": {"budget_accuracy": round(budget_ok / len(test_rows), 3),
                       "timeline_accuracy": round(weeks_ok / len(test_rows), 3)},
        "workflow": {
            "leads_run": len(test_rows), "reached_human_review": reached_human,
            "archived": len(test_rows) - reached_human,
            "drafts_passing_review_first_time": round(first_pass / reached_human, 3),
            "avg_revisions": round(statistics.mean(revisions), 2),
            "avg_ms_per_lead": round(statistics.mean(timings), 2),
        },
        "reviewer": {"corrupted_drafts": total_corrupt, "caught": caught,
                     "catch_rate": round(caught / total_corrupt, 3),
                     "by_type": {k: f"{v[0]}/{v[1]}" for k, v in per_type.items()}},
    }
    (ROOT / "eval/results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
