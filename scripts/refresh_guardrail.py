"""Ship/abort guardrail for the automated refresh.

Compares the freshly-evaluated ensemble against the currently-committed model
on the held-out test window. Exits 0 (ship) only if the new model does not
regress; exits 1 (abort) otherwise. This is the safety net that stopped the
squad-strength experiment from ever reaching production.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
LOGLOSS_TOL = 0.005   # allow tiny noise; block real regressions
ACC_TOL = 0.005


def _ensemble(metrics: dict) -> tuple[float, float]:
    e = metrics["classification"]["ENSEMBLE"]
    return e["log_loss"], e["accuracy"]


def main() -> int:
    new = json.loads((REPORTS_DIR / "metrics.json").read_text())
    new_ll, new_acc = _ensemble(new)

    try:
        old_raw = subprocess.check_output(
            ["git", "show", "HEAD:reports/metrics.json"], text=True)
        old_ll, old_acc = _ensemble(json.loads(old_raw))
    except Exception:
        print(f"[guardrail] no committed baseline; shipping "
              f"(logloss={new_ll:.4f} acc={new_acc:.4f})")
        return 0

    print(f"[guardrail] committed: logloss={old_ll:.4f} acc={old_acc:.4f}")
    print(f"[guardrail] new:       logloss={new_ll:.4f} acc={new_acc:.4f}")
    if new_ll > old_ll + LOGLOSS_TOL or new_acc < old_acc - ACC_TOL:
        print("[guardrail] REGRESSION — aborting ship.")
        return 1
    print("[guardrail] OK — clear to ship.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
