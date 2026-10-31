"""A small synthetic stock market that really backtests FASTEXPR, for proving the repair engine without touching BRAIN.

`Market` builds daily prices, volumes, caps, sectors and data fields; the fields carry planted signals with the
pathologies real near misses have (fast and turnover-heavy, slow with thin returns, edge only in small caps, sparse,
fat-tailed, fading in recent years, sector-driven, crowded). `evaluate()` runs an expression through the app's own
parser, and `simulate()` applies BRAIN's documented steps: decay, neutralization, scaling, truncation, one-day delay,
then Sharpe, Fitness = Sharpe * sqrt(|Returns| / max(Turnover, 0.125)), sub-universe Sharpe and PnL correlation.

So decay cuts turnover and neutralization changes Sharpe through mechanics, not through a response curve written for
the tests. The numbers describe this toy market only: they show whether repair logic works, never what BRAIN's pass
rate will be.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from alphago.brain.simulations import normalize_settings
from alphago.fastexpr import Assign, BinOp, Bool, Call, Ident, Num, Program, Str, Ternary, Unary, as_program

SHARPE_LIMIT, FITNESS_LIMIT, T_LOW, T_HIGH, Y2_LIMIT, CORR_LIMIT = 1.25, 1.0, 0.01, 0.7, 0.6, 0.7
GROUPS = ("market", "sector", "industry", "subindustry")


def _ffill(a: np.ndarray, limit: int) -> np.ndarray:
    return pd.DataFrame(a).ffill(limit=limit).to_numpy()


def _cs(fn, a: np.ndarray) -> np.ndarray:  # noqa: ANN001
    return np.vstack([fn(row) for row in a])


def _rank(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()


def _zscore(a: np.ndarray) -> np.ndarray:
    mean, sd = np.nanmean(a, axis=1, keepdims=True), np.nanstd(a, axis=1, keepdims=True)
    return (a - mean) / np.where(sd > 0, sd, np.nan)


def _roll(a: np.ndarray, d: int, how: str) -> np.ndarray:
    """Rolling mean / sum / std from cumulative sums (NaN-aware); min and max fall back to pandas."""
    d = int(d)
    if how in ("min", "max"):
        return getattr(pd.DataFrame(a).rolling(d, min_periods=max(2, d // 2)), how)().to_numpy()
    ok = np.isfinite(a)
    x = np.where(ok, a, 0.0)

    def window(v: np.ndarray) -> np.ndarray:
        c = np.cumsum(v, axis=0)
        out = c.copy()
        out[d:] = c[d:] - c[:-d]
        return out

    n, total = window(ok.astype(float)), window(x)
    enough = n >= max(2, d // 2)
    mean = np.where(enough, total / np.where(n > 0, n, 1), np.nan)
    if how == "mean":
        return mean
    if how == "sum":
        return np.where(enough, total, np.nan)
    var = window(x * x) / np.where(n > 0, n, 1) - np.nan_to_num(mean) ** 2
    return np.where(enough, np.sqrt(np.maximum(var, 0.0) * n / np.where(n > 1, n - 1, 1)), np.nan)


def _ts_rank(a: np.ndarray, d: int) -> np.ndarray:
    """Share of the last d days at or below today's value, in row chunks to keep memory small."""
    d = int(d)
    out = np.full_like(a, np.nan, dtype=float)
    view = np.lib.stride_tricks.sliding_window_view(a, d, axis=0)
    for i in range(0, view.shape[0], 256):
        w = view[i:i + 256]
        today = w[..., -1:]
        seen = np.isfinite(w).sum(-1)
        rank = (w <= today).sum(-1) / np.where(seen > 0, seen, 1)
        out[d - 1 + i:d - 1 + i + w.shape[0]] = np.where(np.isfinite(today[..., 0]) & (seen >= d // 2), rank, np.nan)
    return out


def _linear_decay(a: np.ndarray, n: int) -> np.ndarray:
    """Weights n+1, n, ... 1 over the last n+1 days (BRAIN's decay setting and ts_decay_linear)."""
    if n <= 0:
        return a
    weights = np.arange(n + 1, 0, -1, dtype=float)
    total, norm = np.zeros_like(a), np.zeros_like(a)
    for k, w in enumerate(weights):
        shifted = np.full_like(a, np.nan)
        shifted[k:] = a[:a.shape[0] - k]
        ok = np.isfinite(shifted)
        total[ok] += w * shifted[ok]
        norm[ok] += w
    return np.where(norm > 0, total / np.where(norm > 0, norm, 1), np.nan)


def _group_apply(a: np.ndarray, groups: np.ndarray, how: str) -> np.ndarray:
    """Per-day, per-group statistics in one pass: each (day, group) pair gets one key, so bincount does the work."""
    T = a.shape[0]
    gid = np.nan_to_num(groups).astype(np.int64)
    gid = gid - gid.min()
    size = int(gid.max()) + 1
    keys = gid + np.arange(T, dtype=np.int64)[:, None] * size
    ok = np.isfinite(a)
    k, v = keys[ok], a[ok]
    out = np.full_like(a, np.nan, dtype=float)
    if how in ("rank", "median"):
        order = np.lexsort((v, k))
        sk = order.argsort()
        counts = np.bincount(k, minlength=T * size)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        pos = sk - starts[k]
        if how == "rank":
            out[ok] = (pos + 1.0) / counts[k]
            return out
        sorted_v = v[order]
        out[ok] = sorted_v[starts[k] + (counts[k] - 1) // 2]
        return out
    n = np.bincount(k, minlength=T * size).astype(float)
    mean = np.bincount(k, weights=v, minlength=T * size) / np.where(n > 0, n, 1)
    if how == "demean":
        out[ok] = v - mean[k]
    elif how == "mean":
        out[ok] = mean[k]
    else:
        var = np.bincount(k, weights=v * v, minlength=T * size) / np.where(n > 0, n, 1) - mean ** 2
        sd = np.sqrt(np.maximum(var, 0.0))
        out[ok] = sd[k] if how == "std" else (v - mean[k]) / np.where(sd[k] > 0, sd[k], np.nan)
    return out


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cross-sectional OLS residual of y on x, day by day."""
    ok = np.isfinite(y) & np.isfinite(x)
    n = ok.sum(1, keepdims=True)
    yy, xx = np.where(ok, y, 0.0), np.where(ok, x, 0.0)
    ym, xm = yy.sum(1, keepdims=True) / np.maximum(n, 1), xx.sum(1, keepdims=True) / np.maximum(n, 1)
    xc = np.where(ok, x - xm, 0.0)
    denom = (xc * xc).sum(1, keepdims=True)
    beta = (xc * (yy - np.where(ok, ym, 0.0))).sum(1, keepdims=True) / np.where(denom > 0, denom, 1.0)
    return np.where(ok & (n >= 8), y - beta * x, np.nan)


def _hump(a: np.ndarray, h: float) -> np.ndarray:
    """Hold yesterday's value unless the change beats a threshold set by `h` and today's scale."""
    out = a.copy()
    for t in range(1, a.shape[0]):
        scale = np.nanmean(np.abs(a[t]))
        keep = np.isfinite(out[t - 1]) & (np.abs(a[t] - out[t - 1]) < h * 100 * scale)
        out[t] = np.where(keep | ~np.isfinite(a[t]), out[t - 1], a[t])
    return out


def _trade_when(cond: np.ndarray, alpha: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    out = np.full_like(alpha, np.nan, dtype=float)
    for t in range(alpha.shape[0]):
        prev = out[t - 1] if t else np.full(alpha.shape[1], np.nan)
        out[t] = np.where(cond[t] > 0, alpha[t], prev)
        out[t] = np.where(exit_[t] > 0, np.nan, out[t])
    return out


@dataclass
class Result:
    sharpe: float
    fitness: float
    turnover: float
    returns: float
    drawdown: float
    margin: float
    long_count: int
    short_count: int
    sub_sharpe: float
    y2_sharpe: float
    max_weight: float
    coverage: float
    daily: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    dates: list[str] = field(repr=False, default_factory=list)


class Market:
    def __init__(self, seed: int = 11, n: int = 300, days: int = 1260, warm: int = 270) -> None:
        self.n, self.days, self.warm = n, days + warm, warm
        rng = self.rng = np.random.default_rng(seed)
        T, N = self.days, n
        self.subindustry = np.tile(rng.integers(0, 60, N), (T, 1)).astype(float)
        self.industry = self.subindustry // 3
        self.sector = self.subindustry // 12
        self.market = np.zeros((T, N))
        size = rng.normal(0, 1.1, N)
        size_rank = pd.Series(size).rank(pct=True).to_numpy()
        self.size_rank = size_rank
        vol = 0.022 * (1 + 0.8 * (1 - size_rank))
        # common risk: market, sector, sub-industry, size, plus latent factors no neutralization names
        market = rng.normal(0.0003, 0.008, T) * np.where((np.arange(T) > T * 0.55) & (np.arange(T) < T * 0.6), 2.5, 1.0)
        beta = rng.normal(1.0, 0.25, N)
        rets = np.outer(market, beta)
        for groups, sd in ((self.sector[0], 0.005), (self.subindustry[0], 0.004)):
            f = rng.normal(0, sd, (T, int(groups.max()) + 1))
            rets += f[:, groups.astype(int)]
        rets += np.outer(rng.normal(0, 0.004, T), (size - size.mean()) / size.std())
        self.loadings = rng.normal(0, 1, (N, 6))
        self.latent = rng.normal(0, 0.0035, (T, 6))
        rets += self.latent @ self.loadings.T
        rets += rng.standard_t(5, (T, N)) * vol / math.sqrt(5 / 3)

        self.fields: dict[str, np.ndarray] = {}
        self.pools: dict[str, list[str]] = {}
        drift = np.zeros((T, N))

        def latent(phi: float, resid: float = 1.0) -> np.ndarray:
            z = np.zeros((T, N))
            z[0] = rng.standard_normal(N)
            shock = rng.standard_normal((T, N))
            for t in range(1, T):
                z[t] = phi * z[t - 1] + math.sqrt(1 - phi * phi) * shock[t]
            return z * resid

        def signal(name: str, phi: float, strength: float, mask: Any = 1.0, copies: int = 3, noise: float = 0.9,
                   base: np.ndarray | None = None) -> np.ndarray:
            z = latent(phi) if base is None else base
            if base is None:
                pos = np.zeros((T, N))
                pos[1:] = z[:-1]                                    # today's value moves tomorrow's return
                drift_mask = np.ones((T, N)) * mask
                pos[1:] *= drift_mask[:-1]
                drift[:] += strength * (vol * pos)
            for k in range(copies):
                self.fields[f"{name}{k + 1}"] = z + noise * latent(phi)
            return z

        S = 0.0072
        signal("fast", 0.45, S * 2.0)                                          # strong, turns over too fast
        signal("mid", 0.85, S * 1.15)                                          # decent, needs turnover trimmed
        signal("slow", 0.992, S * 1.25)                                        # low turnover, thin returns
        small = (1.0 - size_rank) ** 2.5 * 3.0
        signal("small", 0.9, S * 1.6, mask=small)                              # edge only in the smallest names
        recent = np.ones((T, N))
        recent[int(T * 0.72):] = 0.0
        signal("fading", 0.9, S * 1.7, mask=recent)                            # stops working in recent years
        crowd = signal("crowd", 0.93, S * 1.1, noise=0.5)                      # everyone trades this one
        self.crowd = crowd
        self.pools["self"] = ["ts_mean(rank(crowd1), 3)", "group_rank(rank(mid1), subindustry)"]
        self.pools["prod"] = ["rank(crowd2)", "rank(ts_mean(crowd3, 5))"]

        z = latent(0.9)                                                        # sector-driven: noisy until neutralized
        pos = np.zeros((T, N))
        pos[1:] = z[:-1]
        drift += S * 1.3 * vol * pos
        sector_tilt = latent(0.995)[:, self.sector[0].astype(int)]
        for k in range(3):
            self.fields[f"sect{k + 1}"] = 0.8 * z + 0.5 * latent(0.9) + 1.5 * sector_tilt

        z = latent(0.97)                                                       # fat-tailed: dominated by a few names
        pos = np.zeros((T, N))
        pos[1:] = z[:-1]
        drift += S * 1.15 * vol * pos
        tail = np.exp(rng.normal(0, 1.1, (N,)))
        for k in range(3):
            self.fields[f"fat{k + 1}"] = (z + 0.6 * latent(0.97)) * tail ** (1 + k * 0.2)

        z = latent(0.985)                                                      # sparse, reported quarterly
        pos = np.zeros((T, N))
        pos[1:] = z[:-1]
        drift += S * 1.6 * vol * pos
        for k in range(3):
            raw = z + 0.5 * latent(0.985)
            reported = np.full((T, N), np.nan)
            phase = rng.integers(0, 63, N)
            days_idx = np.arange(T)
            for i in range(N):
                if i % 5 < 2:                                                  # 40% of names report this field at all
                    idx = days_idx[(days_idx + phase[i]) % 63 == 0]
                    reported[idx, i] = raw[idx, i]
            self.fields[f"sparse{k + 1}"] = reported

        for k in range(6):
            self.fields[f"noise{k + 1}"] = latent(0.9)

        rets = rets + drift
        self.returns = np.clip(rets, -0.5, 0.5)
        price = 40.0 * np.exp(np.cumsum(self.returns, axis=0))
        cap = np.exp(size + 7.0)[None, :] * price / 40.0
        volume = cap / price * np.exp(rng.normal(-5.0, 0.3, (T, N)) + 0.6 * np.abs(self.returns) / vol * 0.1)
        self.fields.update(close=price, open=price * np.exp(rng.normal(0, 0.004, (T, N))), returns=self.returns,
                           volume=volume, cap=cap, adv20=_roll(volume * price, 20, "mean"),
                           high=price * 1.01, low=price * 0.99, vwap=price * np.exp(rng.normal(0, 0.002, (T, N))))
        self.dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2019-01-02", periods=T)]
        self.sub_mask = np.tile(size_rank >= 2 / 3, (T, 1))
        self._memo: dict[str, np.ndarray] = {}
        self._pool_pnl: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------- expressions

    def data_fields(self) -> list[str]:
        return [k for k in self.fields if k not in ("close", "open", "returns", "volume", "cap", "adv20", "high",
                                                    "low", "vwap")]

    def evaluate(self, code: str) -> np.ndarray:
        env: dict[str, np.ndarray] = {}
        program = as_program(code)
        assert isinstance(program, Program)
        value = None
        for stmt in program.statements:
            if isinstance(stmt, Assign):
                env[stmt.name] = self._eval(stmt.value, env)
            else:
                value = self._eval(stmt, env)
        return np.asarray(value, dtype=float)

    def _scalar(self, v: Any) -> np.ndarray:
        return np.full((self.days, self.n), float(v)) if np.isscalar(v) else v

    def _eval(self, node: Any, env: dict[str, np.ndarray]) -> Any:
        key = node.to_source() if hasattr(node, "to_source") else str(node)
        memo = not env
        if memo and key in self._memo:
            return self._memo[key]
        out = self._eval_inner(node, env)
        if memo and isinstance(out, np.ndarray):
            if len(self._memo) > 400:
                self._memo.clear()
            self._memo[key] = out
        return out

    def _eval_inner(self, node: Any, env: dict[str, np.ndarray]) -> Any:  # noqa: PLR0911, PLR0912
        ev = lambda n: self._eval(n, env)  # noqa: E731
        if isinstance(node, Num):
            return float(node.value)
        if isinstance(node, Bool):
            return 1.0 if node.value else 0.0
        if isinstance(node, Str):
            return node.value
        if isinstance(node, Ident):
            if node.name in env:
                return env[node.name]
            if node.name in GROUPS:
                return getattr(self, node.name)
            if node.name == "nan":
                return float("nan")
            return self.fields[node.name]
        if isinstance(node, Unary):
            v = ev(node.operand)
            return -v if node.op == "-" else (np.nan_to_num(v) == 0).astype(float) if node.op == "!" else v
        if isinstance(node, Ternary):
            c = self._scalar(ev(node.cond))
            return np.where(c > 0, self._scalar(ev(node.then)), self._scalar(ev(node.other)))
        if isinstance(node, BinOp):
            a, b = ev(node.left), ev(node.right)
            with np.errstate(all="ignore"):
                return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
                        "/": lambda: a / np.where(np.asarray(b) == 0, np.nan, b),
                        "<": lambda: (a < b) * 1.0, "<=": lambda: (a <= b) * 1.0, ">": lambda: (a > b) * 1.0,
                        ">=": lambda: (a >= b) * 1.0, "==": lambda: (a == b) * 1.0, "!=": lambda: (a != b) * 1.0,
                        "&&": lambda: ((a > 0) & (b > 0)) * 1.0, "||": lambda: ((a > 0) | (b > 0)) * 1.0}[node.op]()
        if isinstance(node, Call):
            return self._call(node, env)
        raise ValueError(f"cannot evaluate {node!r}")

    def _call(self, c: Call, env: dict[str, np.ndarray]) -> Any:  # noqa: PLR0911, PLR0912
        name = c.name
        args = [self._eval(a, env) for a in c.args]
        kw = {k: self._eval(v, env) for k, v in c.kwargs}
        x = args[0] if args else None
        d = int(args[1]) if len(args) > 1 and np.isscalar(args[1]) else None
        with np.errstate(all="ignore"):
            if name == "rank":
                return _rank(x)
            if name == "zscore":
                return _zscore(x)
            if name in ("quantile",):
                return _zscore(_rank(x))
            if name == "scale":
                return x / np.nansum(np.abs(x), axis=1, keepdims=True)
            if name == "winsorize":
                s = float(kw.get("std", args[1] if len(args) > 1 else 4))
                mean, sd = np.nanmean(x, axis=1, keepdims=True), np.nanstd(x, axis=1, keepdims=True)
                return np.clip(x, mean - s * sd, mean + s * sd)
            if name == "ts_backfill":
                return _ffill(x, d or 60)
            if name == "group_backfill":
                g, days = args[1], int(args[2])
                filled = _ffill(x, days)
                return np.where(np.isfinite(filled), filled, _group_apply(filled, g, "mean"))
            if name == "ts_rank":
                return _ts_rank(x, d)
            if name == "ts_mean":
                return _roll(x, d, "mean")
            if name == "ts_sum":
                return _roll(x, d, "sum")
            if name == "ts_std_dev":
                return _roll(x, d, "std")
            if name == "ts_delay":
                return np.vstack([np.full((d, x.shape[1]), np.nan), x[:-d]])
            if name == "ts_delta":
                return x - np.vstack([np.full((d, x.shape[1]), np.nan), x[:-d]])
            if name == "ts_zscore":
                m, s = _roll(x, d, "mean"), _roll(x, d, "std")
                return (x - m) / np.where(s > 0, s, np.nan)
            if name in ("ts_av_diff",):
                return x - _roll(x, d, "mean")
            if name == "ts_scale":
                lo, hi = _roll(x, d, "min"), _roll(x, d, "max")
                return (x - lo) / np.where(hi > lo, hi - lo, np.nan)
            if name == "ts_decay_linear":
                return _linear_decay(x, d - 1)
            if name == "ts_decay_exp_window":
                factor = float(kw.get("factor", 0.5))
                return pd.DataFrame(x).ewm(alpha=1 - factor, min_periods=2).mean().to_numpy()
            if name == "ts_step":
                return np.tile(np.arange(self.days, dtype=float)[:, None], (1, self.n))
            if name == "ts_regression":
                w = int(args[2])
                y, xs = pd.DataFrame(x), pd.DataFrame(args[1])
                slope = y.rolling(w, min_periods=5).cov(xs) / xs.rolling(w, min_periods=5).var()
                return (y - slope * xs).to_numpy() if not kw.get("rettype") else slope.to_numpy()
            if name == "ts_arg_max":
                return pd.DataFrame(x).rolling(d, min_periods=2).apply(lambda v: len(v) - 1 - np.argmax(v),
                                                                       raw=True).to_numpy()
            if name == "ts_corr":
                out = pd.DataFrame(x).rolling(int(args[2]), min_periods=5).corr(pd.DataFrame(args[1]))
                return out.to_numpy()
            if name == "group_rank":
                return _group_apply(x, args[1], "rank")
            if name == "group_neutralize":
                return _group_apply(x, args[1], "demean")
            if name == "group_zscore":
                return _group_apply(x, args[1], "zscore")
            if name == "group_median":
                return _group_apply(x, args[1], "median")
            if name == "group_std_dev":
                return _group_apply(x, args[1], "std")
            if name == "group_mean":
                return _group_apply(x, args[-1], "mean")
            if name == "bucket":
                return np.ceil(np.nan_to_num(x) * 10)
            if name == "hump":
                return _hump(x, float(kw.get("hump", args[1] if len(args) > 1 else 0.01)))
            if name == "trade_when":
                return _trade_when(self._scalar(args[0]), self._scalar(args[1]), self._scalar(args[2]))
            if name == "if_else":
                return np.where(self._scalar(args[0]) > 0, self._scalar(args[1]), self._scalar(args[2]))
            if name == "signed_power":
                return np.sign(x) * np.abs(x) ** float(args[1])
            if name == "power":
                return x ** float(args[1])
            if name == "abs":
                return np.abs(x)
            if name == "sign":
                return np.sign(x)
            if name == "log":
                return np.log(np.where(x > 0, x, np.nan))
            if name == "reverse":
                return -x
            if name == "divide":
                return x / np.where(args[1] == 0, np.nan, args[1])
            if name == "multiply":
                return x * args[1]
            if name == "add":
                return x + args[1]
            if name == "subtract":
                return x - args[1]
            if name in ("regression_neut", "vector_neut"):
                return _residual(x, args[1])
            if name == "truncate":
                cap = float(kw.get("maxPercent", 0.01)) * np.nansum(np.abs(x), axis=1, keepdims=True)
                return np.clip(x, -cap, cap)
            if name == "pasteurize":
                return np.where(np.isfinite(x), x, np.nan)
        raise ValueError(f"marketsim has no operator {name}")

    # ---------------------------------------------------------- backtest

    def _neutralize(self, a: np.ndarray, how: str) -> np.ndarray:
        how = (how or "NONE").upper()
        if how in ("SUBINDUSTRY", "INDUSTRY", "SECTOR", "MARKET"):
            return _group_apply(a, getattr(self, how.lower()), "demean")
        a = a - np.nanmean(a, axis=1, keepdims=True)
        if how in ("STATISTICAL", "SLOW", "FAST", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM"):
            for j in range(self.loadings.shape[1]):
                a = _residual(a, np.tile(self.loadings[:, j], (self.days, 1)))
        elif how == "CROWDING":
            a = _residual(a, self.crowd)
        return a

    def weights(self, code: str, settings: dict[str, Any]) -> np.ndarray:
        st = normalize_settings(settings)
        a = self.evaluate(code).copy()
        a[~np.isfinite(a)] = np.nan
        a = _linear_decay(a, int(st["decay"]))
        a = self._neutralize(a, st["neutralization"])
        w = a / np.nansum(np.abs(a), axis=1, keepdims=True)
        trunc = float(st["truncation"] or 0)
        if trunc > 0:
            for _ in range(3):
                w = np.clip(w, -trunc, trunc)
                w = w / np.nansum(np.abs(w), axis=1, keepdims=True)
        return np.nan_to_num(w)

    def simulate(self, code: str, settings: dict[str, Any]) -> Result:
        st = normalize_settings(settings)
        w = self.weights(code, st)
        if str(st.get("universe")) == "TOP1000":
            w = np.where(self.sub_mask, w, 0.0)
            w = w / np.maximum(np.abs(w).sum(1, keepdims=True), 1e-12)
        pnl = (w[:-1] * self.returns[1:]).sum(1)[self.warm:]
        traded = np.abs(np.diff(w, axis=0)).sum(1)[self.warm:]
        sd = pnl.std()
        sharpe = float(pnl.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0
        returns = float(pnl.mean() * 252 * 2)
        turnover = float(traded.mean())
        fitness = sharpe * math.sqrt(abs(returns) / max(turnover, 0.125))
        curve = np.cumsum(pnl) * 2
        drawdown = float((np.maximum.accumulate(curve) - curve).max())
        margin = float(pnl.sum() / max(traded.sum(), 1e-9))
        ws = np.where(self.sub_mask, w, 0.0)
        sub = (ws[:-1] * self.returns[1:]).sum(1)[self.warm:]
        sub_sharpe = float(sub.mean() / sub.std() * math.sqrt(252)) if sub.std() > 0 else 0.0
        tail = pnl[-504:]
        y2 = float(tail.mean() / tail.std() * math.sqrt(252)) if tail.std() > 0 else 0.0
        held = (w[self.warm:] != 0)
        return Result(sharpe, fitness, turnover, returns, drawdown, margin, int((w[self.warm:] > 0).sum(1).mean()),
                      int((w[self.warm:] < 0).sum(1).mean()), sub_sharpe, y2, float(np.abs(w[self.warm:]).max()),
                      float(held.sum(1).mean() / self.n), pnl, self.dates[self.warm + 1:])

    def is_checks(self, r: Result, *, universe: str = "TOP3000") -> list[dict[str, Any]]:
        sub_limit = 0.75 * math.sqrt(1 / 3) * abs(r.sharpe)

        def row(name: str, ok: bool, value: float, limit: float) -> dict[str, Any]:
            return {"name": name, "result": "PASS" if ok else "FAIL", "value": round(value, 4), "limit": round(limit, 4)}

        return [row("LOW_SHARPE", r.sharpe >= SHARPE_LIMIT, r.sharpe, SHARPE_LIMIT),
                row("LOW_FITNESS", r.fitness >= FITNESS_LIMIT, r.fitness, FITNESS_LIMIT),
                row("LOW_TURNOVER", r.turnover >= T_LOW, r.turnover, T_LOW),
                row("HIGH_TURNOVER", r.turnover <= T_HIGH, r.turnover, T_HIGH),
                {"name": "CONCENTRATED_WEIGHT", "result": "PASS" if r.coverage >= 0.2 and r.max_weight <= 0.1
                 else "FAIL"},
                row("LOW_SUB_UNIVERSE_SHARPE", r.sub_sharpe >= sub_limit, r.sub_sharpe, sub_limit),
                row("LOW_2Y_SHARPE", r.y2_sharpe >= Y2_LIMIT, r.y2_sharpe, Y2_LIMIT),
                {"name": "SELF_CORRELATION", "result": "PENDING"}, {"name": "PROD_CORRELATION", "result": "PENDING"},
                {"name": "MATCHES_COMPETITION", "result": "PASS"}]

    def pool_pnl(self, code: str) -> np.ndarray:
        if code not in self._pool_pnl:
            self._pool_pnl[code] = self.simulate(code, {"neutralization": "SUBINDUSTRY", "decay": 2}).daily
        return self._pool_pnl[code]

    def correlation(self, pnl: np.ndarray, pool: str) -> tuple[float, str, float]:
        """Highest PnL correlation with a pool: (correlation, expression, that alpha's Sharpe)."""
        best = (-1.0, "", 0.0)
        for code in self.pools[pool]:
            other = self.pool_pnl(code)
            c = float(np.corrcoef(pnl, other)[0, 1])
            if c > best[0]:
                best = (c, code, float(other.mean() / other.std() * math.sqrt(252)))
        return best

    def full_checks(self, r: Result) -> list[dict[str, Any]]:
        checks = [c for c in self.is_checks(r) if c["name"] not in ("SELF_CORRELATION", "PROD_CORRELATION")]
        self_corr, _, other_sharpe = self.correlation(r.daily, "self")
        prod_corr, _, _ = self.correlation(r.daily, "prod")
        beat = r.sharpe >= 1.1 * other_sharpe                  # community-reported rule: 10% better may pass
        checks.append({"name": "SELF_CORRELATION", "value": round(self_corr, 4), "limit": CORR_LIMIT,
                       "result": "PASS" if self_corr < CORR_LIMIT or beat else "FAIL"})
        checks.append({"name": "PROD_CORRELATION", "value": round(prod_corr, 4), "limit": CORR_LIMIT,
                       "result": "PASS" if prod_corr < CORR_LIMIT else "FAIL"})
        return checks


def alpha_id(code: str, settings: dict[str, Any]) -> str:
    st = normalize_settings(settings)
    key = f"{code}|{st['decay']}|{st['neutralization']}|{st['truncation']}|{st['universe']}"
    return "M" + hashlib.sha1(key.encode()).hexdigest()[:9]


def alpha_json(market: Market, code: str, settings: dict[str, Any]) -> dict[str, Any]:
    r = market.simulate(code, settings)
    st = normalize_settings(settings)
    return {"id": alpha_id(code, st), "type": "REGULAR", "settings": st, "regular": {"code": code},
            "status": "UNSUBMITTED", "dateCreated": "2026-10-20T10:00:00-04:00",
            "is": {"sharpe": round(r.sharpe, 3), "fitness": round(r.fitness, 3), "turnover": round(r.turnover, 4),
                   "returns": round(r.returns, 4), "drawdown": round(r.drawdown, 4), "margin": round(r.margin, 6),
                   # the small market stands in for TOP3000, so holdings are scaled to match that label
                   "longCount": r.long_count * 3000 // market.n, "shortCount": r.short_count * 3000 // market.n,
                   "checks": market.is_checks(r)}}


__all__ = ["Market", "Result", "alpha_id", "alpha_json", "Str", "Bool", "Call"]
