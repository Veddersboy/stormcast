"""Small 1D-CNN on the raw last 120 one-minute returns and volumes (plus time of day), in PyTorch.

Architecture and training are fixed in docs/plan.md: conv(16, k5) -> conv(32, k5) -> global average pooling ->
dropout 0.2 -> linear, Adam 1e-3, weight decay 1e-4, batch 256, up to 15 epochs, early stopping on the last 10% of the
training window (by time). CPU, fixed seeds.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, out: int = 1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(), nn.Conv1d(16, 32, 5, padding=2),
                                  nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32 + 1, out))

    def forward(self, seq, tod):
        h = self.conv(seq).mean(dim=2)
        return self.head(torch.cat([h, tod[:, None]], dim=1)).squeeze(1)


def _tensor(a):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)


class CNN:
    def __init__(self, task: str = "direction", epochs: int = 15, patience: int = 3, seed: int = 0):
        self.task, self.epochs, self.patience, self.seed = task, epochs, patience, seed

    def _loss(self):
        return nn.BCEWithLogitsLoss() if self.task == "direction" else nn.MSELoss()

    def fit(self, seq: np.ndarray, tod: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y, np.float32)
        self.y_mean_ = 0.0 if self.task == "direction" else float(y[: int(0.9 * len(y))].mean())
        y = y - self.y_mean_
        cut = int(0.9 * len(y))
        S, T, Y = _tensor(seq), _tensor(tod / 390.0), _tensor(y)
        net, loss_fn = Net(), self._loss()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_loss, bad = None, np.inf, 0
        for _ in range(self.epochs):
            net.train()
            for b in np.array_split(rng.permutation(cut), max(1, cut // 256)):
                b = torch.as_tensor(b)
                opt.zero_grad()
                loss_fn(net(S[b], T[b]), Y[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss_fn(net(S[cut:], T[cut:]), Y[cut:]))
            if v < best_loss - 1e-5:
                best, best_loss, bad = copy.deepcopy(net.state_dict()), v, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best)
        self.net_ = net.eval()
        return self

    def predict(self, seq: np.ndarray, tod: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.net_(_tensor(seq), _tensor(tod / 390.0)).numpy()
        return 1 / (1 + np.exp(-out)) if self.task == "direction" else out + self.y_mean_
