#!/usr/bin/env bash
# Re-pull latest results, retrain, and ship ONLY if accuracy holds.
# Run locally (cron) or from CI. Push remote defaults to "space"; override with
# PITCHSENSE_PUSH_REMOTE. Requires git auth for that remote (token/credential).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
REMOTE="${PITCHSENSE_PUSH_REMOTE:-space}"
ARTIFACTS=(models/artifacts/model_bundle.joblib models/artifacts/ensemble_weights.json
           data/processed/state_snapshot.joblib reports/metrics.json
           reports/worldcup2026_simulation.json reports/evaluation_report.md
           reports/feature_importance_top50.csv)

echo "[refresh] $(date -u +%FT%TZ) — downloading latest results + retraining…"
make data
make features
make train
make evaluate

if ! $PY scripts/refresh_guardrail.py; then
  echo "[refresh] accuracy regressed — reverting, nothing shipped."
  git checkout -- "${ARTIFACTS[@]}" 2>/dev/null || true
  exit 1
fi

make simulate
git add "${ARTIFACTS[@]}"
if git diff --cached --quiet; then
  echo "[refresh] no model changes since last run — nothing to ship."
  exit 0
fi
git -c user.email=spnittes@gmail.com -c user.name=Nittes \
    commit -q -m "Automated refresh: retrain on latest results ($(date -u +%F))"
git push "$REMOTE" main
echo "[refresh] shipped updated model to '$REMOTE'."
