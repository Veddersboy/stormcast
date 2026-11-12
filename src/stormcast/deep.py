"""StormNet: a multi-task deep network that reads the last four hours of one-minute bars and the tabular features, and
forecasts the next 30 minutes' log realized variance (with its own uncertainty), range quantiles and direction.

Architecture
  bars (240 x 3: return, bar range, volume)  -> conv stem -> 4 dilated residual TCN blocks -> stride-4 conv to 60 tokens
                                             -> 2-layer Transformer encoder -> attention pooling (weights kept, so the
                                                demo can show which part of the window the model looked at)
  tabular (25 features)                      -> 2-layer MLP
  concat -> shared trunk -> heads:
      volatility   Gaussian mean and log-variance (heteroscedastic negative log-likelihood: aleatoric uncertainty)
      range        5% / 50% / 95% quantiles of log range (pinball loss; non-crossing by construction)
      direction    logit (binary cross-entropy)

A deep ensemble (several seeds) adds epistemic uncertainty: the spread of the members' means. Everything that needs
statistics (feature scalers, channel scalers, target scaling) is fit on the training window only, and early stopping
uses the last 10% of the training window by time.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from stormcast.features import ALL_FEATURES

SEQ = 240
QUANTILES = (0.05, 0.5, 0.95)
CHANNELS = ("ret_bp", "range_bp", "lvol")


def device() -> torch.device:
    import os
    if os.environ.get("STORMCAST_DEVICE"):
        return torch.device(os.environ["STORMCAST_DEVICE"])
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------- bar channels
def bar_channels(bars: pd.DataFrame) -> np.ndarray:
    """(n_bars, 3) float32 per-bar inputs: log return in bp (clipped, zeroed across gaps > 30 minutes), log1p of the
    bar's high-low range in bp, and log1p volume."""
    t = bars.index.asi8
    c = bars.close.to_numpy(float)
    lc = np.log(c)
    r = np.diff(lc, prepend=lc[0]) * 1e4
    r[np.diff(t, prepend=t[0]) / 6e10 > 30] = 0.0
    rng = np.log1p((bars.high.to_numpy(float) - bars.low.to_numpy(float)) / c * 1e4)
    lv = np.log1p(bars.volume.to_numpy(float))
    return np.column_stack([np.clip(r, -50, 50), rng, lv]).astype(np.float32)


def window_ends(bars: pd.DataFrame, decision_utc) -> np.ndarray:
    """Index of the first bar at or after each decision: the window is bars [end - SEQ, end), all closed before it."""
    return np.searchsorted(bars.index.asi8, pd.DatetimeIndex(decision_utc).asi8, side="left").astype(np.int64)


# ---------------------------------------------------------------- network
class _ContiguousGrad(torch.autograd.Function):
    """Identity whose backward makes the gradient contiguous (works around a torch 2.5 MPS convolution-backward bug
    when the gradient arrives through a transpose)."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.contiguous()


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, p: float):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.n = nn.GroupNorm(8, ch)
        self.d = nn.Dropout(p)

    def forward(self, x):
        h = self.d(F.gelu(self.c1(x)))
        return F.gelu(x + self.n(self.c2(h)))


class AttnBlock(nn.Module):
    """Pre-norm Transformer encoder block (multi-head self-attention + GELU feed-forward)."""

    def __init__(self, d: int, heads: int, p: float):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d))
        self.d = nn.Dropout(p)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D // self.h), dim=-1)
        x = x + self.d(self.o((a @ v).transpose(1, 2).reshape(B, T, D)))
        return x + self.d(self.ff(self.n2(x)))


class StormNet(nn.Module):
    def __init__(self, n_tab: int, ch: int = 64, d: int = 64, p: float = 0.1):
        super().__init__()
        self.stem = nn.Conv1d(len(CHANNELS), ch, 5, padding=2)
        self.tcn = nn.Sequential(*[TCNBlock(ch, dl, p) for dl in (1, 2, 4, 8)])
        self.down = nn.Conv1d(ch, d, 4, stride=4)
        self.pos = nn.Parameter(torch.randn(1, SEQ // 4, d) * 0.02)
        self.attn_enc = nn.Sequential(AttnBlock(d, 4, p), AttnBlock(d, 4, p))
        self.pool_q = nn.Parameter(torch.randn(d) * 0.02)
        self.tab = nn.Sequential(nn.Linear(n_tab, 128), nn.GELU(), nn.Dropout(p), nn.Linear(128, 64), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(d + 64, 128), nn.GELU(), nn.Dropout(p), nn.Linear(128, 64), nn.GELU())
        self.vol = nn.Linear(64, 2)
        self.rng = nn.Linear(64, len(QUANTILES))
        self.dir = nn.Linear(64, 1)

    def forward(self, seq, tab):
        h = self.tcn(self.stem(seq.transpose(1, 2).contiguous()))         # (B, ch, SEQ)
        z = self.attn_enc(_ContiguousGrad.apply(self.down(h)).transpose(1, 2).contiguous() + self.pos)  # (B, 60, d)
        w = torch.softmax(z @ self.pool_q / math.sqrt(z.shape[-1]), dim=1)  # (B, 60)
        s = (w.unsqueeze(-1) * z).sum(1)
        u = self.trunk(torch.cat([s, self.tab(tab)], dim=1))
        mu, logvar = self.vol(u).unbind(1)
        q = self.rng(u)
        q = torch.cat([q[:, :1], q[:, :1] + torch.cumsum(F.softplus(q[:, 1:]), dim=1)], dim=1)  # non-crossing
        return {"mu": mu, "logvar": logvar.clamp(-8, 4), "q": q, "logit": self.dir(u).squeeze(1), "attn": w}


def pinball(q, y):
    taus = torch.tensor(QUANTILES, device=q.device)
    e = y[:, None] - q
    return torch.maximum(taus * e, (taus - 1) * e).mean()


# ---------------------------------------------------------------- one member
class Member:
    """One StormNet trained on one training window. Inputs are standardized with training-window statistics; the
    volatility and range targets are standardized too, and mapped back in predict()."""

    def __init__(self, seed: int = 0, epochs: int = 20, patience: int = 4, batch: int = 512, lr: float = 2e-3,
                 dir_weight: float = 0.2, verbose: bool = False):
        self.seed, self.epochs, self.patience, self.batch, self.lr = seed, epochs, patience, batch, lr
        self.dir_weight, self.verbose = dir_weight, verbose

    # -- tensors
    def _prep(self, chan: np.ndarray, df: pd.DataFrame, ends: np.ndarray, dev):
        tab = (df[ALL_FEATURES].to_numpy(np.float32) - self.tab_mu_) / self.tab_sd_
        return (torch.as_tensor(np.nan_to_num(tab), device=dev), torch.as_tensor(ends, device=dev))

    def _seq(self, chan_t, ends_b):
        idx = ends_b[:, None] - SEQ + torch.arange(SEQ, device=ends_b.device)[None, :]
        return chan_t[idx.clamp(min=0)]

    def fit(self, chan: np.ndarray, df: pd.DataFrame, ends: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = device()
        n = len(df)
        cut = int(0.9 * n)
        X = df[ALL_FEATURES].to_numpy(np.float32)
        self.tab_mu_, self.tab_sd_ = X[:cut].mean(0), X[:cut].std(0) + 1e-6
        lo = max(0, int(ends[:cut].min()) - SEQ)
        used = chan[lo:int(ends[:cut].max())]
        self.ch_mu_, self.ch_sd_ = used.mean(0), used.std(0) + 1e-6
        self.y_mu_, self.y_sd_ = float(df.y_lrv.iloc[:cut].mean()), float(df.y_lrv.iloc[:cut].std())
        self.r_mu_, self.r_sd_ = float(df.y_lrange.iloc[:cut].mean()), float(df.y_lrange.iloc[:cut].std())
        chan_t = torch.as_tensor((chan - self.ch_mu_) / self.ch_sd_, device=dev)
        tab, ends_t = self._prep(chan, df, ends, dev)
        yv = torch.as_tensor(((df.y_lrv.to_numpy() - self.y_mu_) / self.y_sd_).astype(np.float32), device=dev)
        yr = torch.as_tensor(((df.y_lrange.to_numpy() - self.r_mu_) / self.r_sd_).astype(np.float32), device=dev)
        yd = torch.as_tensor(df.y_dir.to_numpy(np.float32), device=dev)

        net = StormNet(len(ALL_FEATURES)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * math.ceil(cut / self.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.15)
        g = torch.Generator().manual_seed(self.seed)

        def loss_on(b):
            o = net(self._seq(chan_t, ends_t[b]), tab[b])
            nll = 0.5 * (o["logvar"] + (yv[b] - o["mu"]) ** 2 / o["logvar"].exp()).mean()
            return nll + pinball(o["q"], yr[b]) + self.dir_weight * F.binary_cross_entropy_with_logits(o["logit"], yd[b])

        best, best_loss, bad, self.history_ = None, np.inf, 0, []
        val_idx = torch.arange(cut, n, device=dev)
        for ep in range(self.epochs):
            net.train()
            for b in torch.randperm(cut, generator=g).to(dev).split(self.batch):
                opt.zero_grad(set_to_none=True)
                loss_on(b).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
            net.eval()
            with torch.no_grad():
                v = float(np.mean([float(loss_on(b)) for b in val_idx.split(4096)]))
            self.history_.append(v)
            if self.verbose:
                print(f"    seed {self.seed} epoch {ep + 1}: val loss {v:.4f}", flush=True)
            if v < best_loss - 1e-4:
                best, best_loss, bad = copy.deepcopy(net.state_dict()), v, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best)
        self.net_ = net.eval()
        self.best_val_ = best_loss
        return self

    @torch.no_grad()
    def predict(self, chan: np.ndarray, df: pd.DataFrame, ends: np.ndarray) -> dict:
        dev = next(self.net_.parameters()).device
        chan_t = torch.as_tensor((chan - self.ch_mu_) / self.ch_sd_, device=dev)
        tab, ends_t = self._prep(chan, df, ends, dev)
        outs = []
        for b in torch.arange(len(df), device=dev).split(4096):
            o = self.net_(self._seq(chan_t, ends_t[b]), tab[b])
            outs.append({k: v.float().cpu().numpy() for k, v in o.items()})
        o = {k: np.concatenate([x[k] for x in outs]) for k in outs[0]}
        return {"mu": o["mu"] * self.y_sd_ + self.y_mu_,
                "var": np.exp(o["logvar"]) * self.y_sd_ ** 2,
                "q": o["q"] * self.r_sd_ + self.r_mu_,
                "p_up": 1 / (1 + np.exp(-o["logit"])),
                "attn": o["attn"]}

    # -- persistence
    def state(self) -> dict:
        keep = ("seed", "tab_mu_", "tab_sd_", "ch_mu_", "ch_sd_", "y_mu_", "y_sd_", "r_mu_", "r_sd_", "best_val_")
        return {"attrs": {k: getattr(self, k) for k in keep},
                "weights": {k: v.cpu() for k, v in self.net_.state_dict().items()}}

    @classmethod
    def from_state(cls, st: dict):
        m = cls(seed=st["attrs"]["seed"])
        for k, v in st["attrs"].items():
            setattr(m, k, v)
        net = StormNet(len(ALL_FEATURES))
        net.load_state_dict(st["weights"])
        m.net_ = net.to(device()).eval()
        return m


# ---------------------------------------------------------------- deep ensemble
class Ensemble:
    """Deep ensemble of StormNet members with different seeds (Lakshminarayanan et al., 2017).

    Predictive variance of log RV = mean aleatoric variance + variance of the members' means (epistemic)."""

    def __init__(self, n: int = 5, **kw):
        self.n, self.kw = n, kw

    def fit(self, chan, df, ends):
        self.members_ = [Member(seed=s, **self.kw).fit(chan, df, ends) for s in range(self.n)]
        return self

    def predict(self, chan, df, ends) -> pd.DataFrame:
        ps = [m.predict(chan, df, ends) for m in self.members_]
        mus = np.stack([p["mu"] for p in ps])
        out = pd.DataFrame(index=df.index)
        out["stormnet"] = mus.mean(0)
        out["stormnet_aleatoric_sd"] = np.sqrt(np.stack([p["var"] for p in ps]).mean(0))
        out["stormnet_epistemic_sd"] = mus.std(0)
        q = np.stack([p["q"] for p in ps]).mean(0)
        for i, t in enumerate(QUANTILES):
            out[f"stormnet_q{int(t * 100):02d}"] = q[:, i]
        out["stormnet_p_up"] = np.stack([p["p_up"] for p in ps]).mean(0)
        self.last_attn_ = np.stack([p["attn"] for p in ps]).mean(0)
        return out

    def state(self) -> dict:
        return {"n": self.n, "members": [m.state() for m in self.members_]}

    @classmethod
    def from_state(cls, st: dict):
        e = cls(n=st["n"])
        e.members_ = [Member.from_state(m) for m in st["members"]]
        return e
