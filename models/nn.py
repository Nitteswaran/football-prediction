"""PyTorch models behind an sklearn-compatible interface.

`TorchMLPClassifier` is the feedforward network used in the ensemble.
`TemporalNet` is a GRU over each team's recent-match sequence; it consumes
sequence tensors produced by `training.sequences` and is trained separately.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...], n_out: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier:
    """Feedforward classifier with internal imputation + standardisation.

    sklearn-compatible: fit(X, y[, eval_set]) / predict_proba / predict.
    """

    def __init__(self, hidden=(256, 128, 64), dropout=0.3, lr=1e-3,
                 weight_decay=1e-4, batch_size=512, max_epochs=80,
                 patience=8, n_classes=3, seed=config.RANDOM_SEED):
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.n_classes = n_classes
        self.seed = seed
        self._model: _MLP | None = None
        self._median: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    # -- preprocessing ---------------------------------------------------
    def _fit_scaler(self, X: np.ndarray) -> None:
        self._median = np.nanmedian(X, axis=0)
        self._median = np.where(np.isfinite(self._median), self._median, 0.0)
        Xf = self._impute(X)
        self._mu = Xf.mean(axis=0)
        self._sigma = Xf.std(axis=0)
        self._sigma[self._sigma < 1e-8] = 1.0

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32).copy()
        mask = ~np.isfinite(X)
        if mask.any():
            X[mask] = np.take(self._median, np.nonzero(mask)[1])
        return X

    def _transform(self, X) -> torch.Tensor:
        X = (self._impute(np.asarray(X, dtype=np.float32)) - self._mu) / self._sigma
        return torch.from_numpy(X.astype(np.float32))

    # -- API ---------------------------------------------------------------
    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in
                ("hidden", "dropout", "lr", "weight_decay", "batch_size",
                 "max_epochs", "patience", "n_classes", "seed")}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y, eval_set: tuple | None = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = _device()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self._fit_scaler(X)
        Xt = self._transform(X)
        yt = torch.from_numpy(y)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self._model = _MLP(X.shape[1], self.hidden, self.n_classes, self.dropout).to(dev)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        loss_fn = nn.CrossEntropyLoss()

        Xv = yv = None
        if eval_set is not None:
            Xv = self._transform(eval_set[0]).to(dev)
            yv = torch.from_numpy(np.asarray(eval_set[1], dtype=np.int64)).to(dev)

        best_loss, best_state, bad = float("inf"), None, 0
        for _epoch in range(self.max_epochs):
            self._model.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = loss_fn(self._model(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
            if Xv is not None:
                self._model.eval()
                with torch.no_grad():
                    vloss = loss_fn(self._model(Xv), yv).item()
                if vloss < best_loss - 1e-4:
                    best_loss, bad = vloss, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in self._model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self._model.eval()
        dev = _device()
        out = []
        Xt = self._transform(X)
        for i in range(0, len(Xt), 4096):
            logits = self._model(Xt[i:i + 4096].to(dev))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


class TemporalNet(nn.Module):
    """GRU encoder over each team's last-N match vectors + fixture context.

    Input: (batch, 2, seq_len, seq_feats) team sequences and (batch, ctx) context.
    """

    def __init__(self, seq_feats: int, ctx_feats: int, hidden: int = 64,
                 n_out: int = 3, dropout: float = 0.25):
        super().__init__()
        self.gru = nn.GRU(seq_feats, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + ctx_feats, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_out),
        )

    def forward(self, seqs: torch.Tensor, ctx: torch.Tensor):
        # seqs: (B, 2, T, F)
        _, h_home = self.gru(seqs[:, 0])
        _, h_away = self.gru(seqs[:, 1])
        z = torch.cat([h_home[-1], h_away[-1], ctx], dim=1)
        return self.head(z)
