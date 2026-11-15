"""Process every lead in data/inbox.json. Runs pause at the human approval step.

Usage:
  python scripts/run_inbox.py                     # process inbox
  python scripts/run_inbox.py approve <run_id>    # approve and send
  python scripts/run_inbox.py reject <run_id>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentflow.workflow import build_workflow, checkpointer  # noqa: E402


def show(run):
    print(f"\nrun {run.run_id} · lead {run.state['lead']['id']} · {run.status}")
    for s in run.trace:
        print(f"  {s.node:<13} {s.ms:>7.2f} ms  {s.summary}")


def main():
    g, cp = build_workflow(), checkpointer()
    if len(sys.argv) == 3:
        action, run_id = sys.argv[1], sys.argv[2]
        run = cp.load(run_id)
        run = g.resume(run, {"human_decision": {"action": action}}, cp)
        show(run)
        return
    for lead in json.loads((ROOT / "data" / "inbox.json").read_text()):
        show(g.start({"lead": lead}, cp))


if __name__ == "__main__":
    main()
