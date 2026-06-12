"""Temporal model: GRU over each team's recent match sequence.

A separate chronological pass captures, for every match, the last-N match
records of both teams *before* kickoff (same no-leakage discipline as the
tabular builder). The GRU encodes each side's sequence; a small head combines
both encodings with fixture context to predict W/D/L.

This model complements the tabular ensemble; report its validation/test
log loss with:  python -m training.temporal
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data.ingestion import load_dataset
from features.builder import FeatureBuilder
from models.nn import TemporalNet, _device

logger = logging.getLogger(__name__)

SEQ_LEN = 10
SEQ_FEATS = 7      # gf, ga, points, clean_sheet, opp_elo_n, importance, days_since
CTX_FEATS = 5      # elo_h_n, elo_a_n, elo_diff_n, neutral, importance


def _team_sequence(builder: FeatureBuilder, team: str, date) -> np.ndarray:
    seq = np.zeros((SEQ_LEN, SEQ_FEATS), dtype=np.float32)
    hist = list(builder.teams[team].history)[-SEQ_LEN:]
    for k, m in enumerate(hist):          # oldest -> newest, right-aligned
        row = SEQ_LEN - len(hist) + k
        days = min((date - m.date).days, 365) / 365.0
        seq[row] = (m.gf / 5.0, m.ga / 5.0, m.points / 3.0, float(m.clean_sheet),
                    (m.opp_elo - 1500.0) / 200.0, m.importance / 5.0, days)
    return seq


def build_sequences() -> dict[str, np.ndarray]:
    """Chronological pass -> sequence tensors aligned with the match table."""
    ds = load_dataset()
    builder = FeatureBuilder(ds)
    matches = ds.matches
    n = len(matches)
    seqs = np.zeros((n, 2, SEQ_LEN, SEQ_FEATS), dtype=np.float32)
    ctx = np.zeros((n, CTX_FEATS), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    dates = matches["date"].values
    min_hist = np.zeros(n, dtype=np.int32)

    for i, row in enumerate(matches.itertuples(index=False)):
        eh = builder.elo.get(row.home_team)
        ea = builder.elo.get(row.away_team)
        seqs[i, 0] = _team_sequence(builder, row.home_team, row.date)
        seqs[i, 1] = _team_sequence(builder, row.away_team, row.date)
        ctx[i] = ((eh - 1500.0) / 200.0, (ea - 1500.0) / 200.0,
                  (eh - ea) / 200.0, float(row.neutral), row.importance / 5.0)
        y[i] = row.outcome
        min_hist[i] = min(builder.teams[row.home_team].matches_played,
                          builder.teams[row.away_team].matches_played)
        builder.observe(row.home_team, row.away_team, row.date,
                        int(row.home_score), int(row.away_score),
                        row.tournament, bool(row.neutral), int(row.importance))
    return {"seqs": seqs, "ctx": ctx, "y": y, "dates": dates, "min_hist": min_hist}


def train_temporal(epochs: int = 30, batch_size: int = 512, lr: float = 1e-3) -> dict:
    data = build_sequences()
    import pandas as pd
    dates = pd.to_datetime(data["dates"])
    usable = (data["min_hist"] >= config.MIN_TEAM_HISTORY) & \
             (dates.year >= config.TRAIN_START_YEAR)
    tr = usable & (dates <= pd.Timestamp(config.TRAIN_END))
    va = usable & (dates > pd.Timestamp(config.TRAIN_END)) & \
         (dates <= pd.Timestamp(config.VALID_END))
    te = usable & (dates > pd.Timestamp(config.VALID_END))

    dev = _device()
    torch.manual_seed(config.RANDOM_SEED)
    model = TemporalNet(SEQ_FEATS, CTX_FEATS).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def tensors(mask):
        return (torch.from_numpy(data["seqs"][mask]),
                torch.from_numpy(data["ctx"][mask]),
                torch.from_numpy(data["y"][mask]))

    Xs, Xc, yt = tensors(tr)
    dl = DataLoader(TensorDataset(Xs, Xc, yt), batch_size=batch_size, shuffle=True)
    Vs, Vc, Vy = (t.to(dev) for t in tensors(va))

    best, best_state, bad = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for xs, xc, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xs.to(dev), xc.to(dev)), yb.to(dev))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Vs, Vc), Vy).item()
        logger.info("epoch %d valid loss %.5f", epoch, vloss)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(best_state)

    Ts, Tc, Ty = (t.to(dev) for t in tensors(te))
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(Ts, Tc), 1).cpu().numpy()
    metrics = {
        "valid_logloss": best,
        "test_logloss": float(log_loss(Ty.cpu(), proba, labels=[0, 1, 2])),
        "test_accuracy": float(accuracy_score(Ty.cpu(), proba.argmax(1))),
    }
    torch.save({"state_dict": model.state_dict(), "metrics": metrics},
               config.ARTIFACTS_DIR / "temporal_net.pt")
    logger.info("temporal model: %s", metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(train_temporal())
