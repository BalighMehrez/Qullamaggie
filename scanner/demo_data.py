"""Synthetic market data with planted setups, for previewing the site offline.
Tickers and companies are fictional on purpose so demo output is never mistaken
for a real signal."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N = 252


def _index():
    end = pd.Timestamp.today().normalize()
    return pd.bdate_range(end=end - pd.tseries.offsets.BDay(1), periods=N)


def _ohlc(closes, rng_pct=0.028, vol=None, gaps=None):
    closes = np.asarray(closes, float)
    n = len(closes)
    prev = np.r_[closes[0], closes[:-1]]
    opens = prev * (1 + RNG.normal(0, 0.004, n))
    if gaps:
        for i, g in gaps.items():
            opens[i] = prev[i] * (1 + g)
    hi_base = np.maximum(opens, closes)
    lo_base = np.minimum(opens, closes)
    spread = np.abs(RNG.normal(rng_pct / 2, rng_pct / 5, n))
    highs = hi_base * (1 + spread)
    lows = lo_base * (1 - spread)
    if vol is None:
        vol = RNG.lognormal(14.5, 0.25, n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vol}, index=_index())


def _walk(start, drift, sd, n):
    return start * np.exp(np.cumsum(RNG.normal(drift, sd, n)))


def noise(start=None):
    start = start or RNG.uniform(20, 400)
    return _ohlc(_walk(start, RNG.normal(0.0003, 0.0006), 0.016, N))


def breakout(broke_today=False):
    a = _walk(50, 0.0002, 0.012, 170)
    leg = a[-1] * np.exp(np.linspace(0, np.log(1.55), 35))
    peak = leg[-1]
    pull = peak * np.linspace(0.97, 0.88, 12)
    rebuild = np.linspace(0.89, 0.955, 22) * peak
    tight = peak * (0.955 + RNG.normal(0, 0.004, 13))
    closes = np.r_[a, leg, pull, rebuild, tight][:N]
    vol = RNG.lognormal(14.5, 0.2, N)
    vol[170:205] *= 1.8
    vol[-13:] *= 0.55
    if broke_today:
        closes[-1] = peak * 0.975
        vol[-1] = vol[-60:-10].mean() * 2.6
    df = _ohlc(closes, 0.022, vol)
    df.iloc[-6:-1, df.columns.get_loc("High")] = np.minimum(df.High.iloc[-6:-1], peak * 0.965)
    if broke_today:
        df.iloc[-1, df.columns.get_loc("High")] = peak * 0.98
        df.iloc[-1, df.columns.get_loc("Low")] = peak * 0.952
    return df


def ep(age=1):
    closes = _walk(80, 0.0, 0.012, N)
    i = N - 1 - age
    closes[i:] = closes[i - 1] * 1.14 * np.exp(np.cumsum(RNG.normal(0.003, 0.01, N - i)))
    vol = RNG.lognormal(14.5, 0.2, N)
    vol[i] *= 6
    vol[i + 1:] *= 2
    return _ohlc(closes, 0.025, vol, gaps={i: 0.12})


def parabolic():
    base = _walk(30, 0.0005, 0.012, N - 15)
    run = base[-1] * np.exp(np.cumsum(np.r_[RNG.normal(0.02, 0.02, 10), [0.05, 0.06, 0.07, 0.08, 0.09]]))
    vol = RNG.lognormal(14.5, 0.2, N)
    vol[-15:] *= np.linspace(1.5, 4, 15)
    return _ohlc(np.r_[base, run], 0.03, vol)


NAMES = [
    ("ACMR", "Acmera Robotics", "Industrials"), ("BLTZ", "Blitz Networks", "Information Technology"),
    ("CRVX", "Corvex Therapeutics", "Health Care"), ("DUNE", "Dune Solar", "Utilities"),
    ("EMBR", "Ember Foods", "Consumer Staples"), ("FLUX", "Flux Semiconductor", "Information Technology"),
    ("GRVT", "Gravity Payments Co", "Financials"), ("HALO", "Halo Aerospace", "Industrials"),
]


def demo_universes():
    frames, ndx, spx = {}, [], []
    planted = {
        "ACMR": breakout(), "FLUX": breakout(), "BLTZ": breakout(broke_today=True),
        "CRVX": ep(age=0), "GRVT": ep(age=2), "HALO": parabolic(),
    }
    for t, name, sector in NAMES:
        frames[t] = planted[t] if t in planted else noise()
        m = {"ticker": t, "name": name, "sector": sector}
        spx.append(m)
        if t in ("ACMR", "BLTZ", "CRVX", "FLUX", "HALO", "DUNE"):
            ndx.append(m)
    for k in range(492):
        t = f"Z{k:03d}"
        frames[t] = noise()
        m = {"ticker": t, "name": f"Filler Co {k}", "sector": ""}
        spx.append(m)
        if k < 94:
            ndx.append(m)
    frames["QQQ"] = _ohlc(_walk(500, 0.0008, 0.009, N), 0.012)
    frames["SPY"] = _ohlc(_walk(560, 0.0006, 0.008, N), 0.01)
    return {"ndx": ndx, "spx": spx}, frames
