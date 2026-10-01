#!/usr/bin/env python3
"""
Qullamaggie setup scanner.

Scans the Nasdaq-100 and S&P 500 on daily bars for Kristjan Kullamaggie's three
setups and writes docs/data/setups.json for the static site:

  breakout   big prior move, tight orderly base on rising MAs, volume drying up
  ep         episodic pivot: big gap on heavy volume, still holding the gap-day low
  parabolic  parabolic short: extended multi-day run far above the 10-day MA

Run:  python scanner/scan.py            (real data via yfinance)
      python scanner/scan.py --demo     (synthetic data, for previewing the site)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "docs" / "data" / "setups.json"
UNIVERSE_CACHE = ROOT / "scanner" / "universe_cache.json"

NY = ZoneInfo("America/New_York")
BAR_FINAL = dt.time(16, 20)  # Yahoo's daily bar settles a few minutes after the 4 pm close
MIN_COVERAGE = 0.90          # refuse to publish if fewer index members than this have fresh data

# --------------------------------------------------------------------------- #
# Settings. Tuned for large caps: Qullamaggie's own numbers are for high-ADR
# small/mid caps, so ADR, gap and parabolic thresholds are relaxed here.
# --------------------------------------------------------------------------- #
CFG = {
    # liquidity / volatility floor
    "min_price": 5.0,
    "min_dollar_vol": 20e6,     # 20-day average dollar volume
    "min_adr": 2.5,             # 20-day average daily range, %

    # breakouts
    "bo_min_rs": 70,            # relative-strength percentile inside the universe
    "bo_prior_gain": 0.30,      # prior leg: low -> peak within 63 bars
    "bo_base_min": 5,           # base length in bars (after the peak)
    "bo_base_max": 45,
    "bo_max_depth": 0.25,       # base high -> base low
    "bo_max_retrace": 0.50,     # of the prior leg
    "bo_tight_days": 5,
    "bo_tight_adr_mult": 2.2,   # last-N range must be <= this many ADRs
    "bo_max_dist_adr": 1.5,     # close must be within this many ADRs of trigger
    "bo_min_breakout_vol": 1.3, # breakout-day volume vs 50-day average

    # episodic pivots
    "ep_min_gap": 0.08,
    "ep_min_vol_mult": 3.0,
    "ep_max_age": 5,            # bars since the gap

    # parabolic shorts
    "ps_lookback": 20,
    "ps_min_gain": 0.50,        # low of lookback -> close
    "ps_min_ext_adr": 5.0,      # distance above 10-day MA, in ADRs
    "ps_min_up_days": 3,

    "chart_bars": 90,
}

UNIVERSES = {
    "ndx": {
        "name": "Nasdaq-100",
        "urls": [  # tried in order; the member table moved off the main article
            "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
            "https://en.wikipedia.org/wiki/List_of_Nasdaq-100_companies",
            "https://en.wikipedia.org/wiki/Nasdaq-100",
        ],
        "benchmark": "QQQ",
        "size": (90, 115),      # plausible member count, to recognise the right table
    },
    "spx": {
        "name": "S&P 500",
        "urls": ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"],
        "benchmark": "SPY",
        "size": (480, 520),
    },
}


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def _cell(v) -> str:
    """Cell or header text without Wikipedia footnote markers: 'Ticker[12]' -> 'Ticker'."""
    return "" if pd.isna(v) else re.sub(r"\[[^\]]*\]", "", str(v)).strip()


def _label(c) -> str:
    # two-row headers come back as tuples; the lower row names the column
    return _cell(c[-1] if isinstance(c, tuple) else c)


def parse_constituents(html: str, size: tuple[int, int]) -> list[dict]:
    """Find the constituents table on a Wikipedia index page."""
    lo, hi = size
    seen_tables = []
    for table in pd.read_html(StringIO(html)):
        labels = [_label(c) for c in table.columns]
        seen_tables.append(f"{len(table)} rows {labels[:6]}")
        cols: dict[str, list] = {}
        for label, c in zip(labels, table.columns):
            cols.setdefault(label, []).append(c)
        # a label used twice (the S&P "changes" table has Added/Ticker and Removed/Ticker) is ambiguous
        pick = lambda *names: next((cols[n][0] for n in names if len(cols.get(n, ())) == 1), None)
        sym_col = pick("Symbol", "Ticker", "Ticker symbol", "Stock symbol")
        if sym_col is None or not lo <= len(table) <= hi:
            continue
        name_col = pick("Security", "Company", "Company name", "Name")
        sector_col = pick("GICS Sector", "ICB Industry", "Sector", "Industry")
        members, seen = [], set()
        for _, r in table.iterrows():
            t = _cell(r[sym_col]).split(":")[-1].strip().upper().replace(".", "-")  # "NASDAQ: AAPL"
            if not re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", t) or t in seen:
                continue
            seen.add(t)
            members.append({
                "ticker": t,
                "name": _cell(r[name_col]) if name_col is not None else "",
                "sector": _cell(r[sector_col]) if sector_col is not None else "",
            })
        if lo <= len(members) <= hi:
            return members
        seen_tables[-1] += f" -> {len(members)} usable tickers"
    raise ValueError("constituent table not found; tables on the page: " + " | ".join(seen_tables))


def fetch_constituents(key: str) -> list[dict]:
    """Current index members from Wikipedia, cached in the repo as a fallback."""
    import requests

    cache = json.loads(UNIVERSE_CACHE.read_text()) if UNIVERSE_CACHE.exists() else {}
    repo = os.environ.get("GITHUB_REPOSITORY", "qullamaggie-scanner")
    errors = []
    for url in UNIVERSES[key]["urls"]:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": f"qullamaggie-scanner/1.0 (https://github.com/{repo})"},
                timeout=30,
            )
            resp.raise_for_status()
            members = parse_constituents(resp.text, UNIVERSES[key]["size"])
        except Exception as exc:  # network hiccup, moved page or layout change
            errors.append(f"{url}: {exc}")
            continue
        print(f"{UNIVERSES[key]['name']}: {len(members)} members from {url}", file=sys.stderr)
        cache[key] = members
        UNIVERSE_CACHE.write_text(json.dumps(cache, indent=1))
        return members
    print(f"[warn] {key}: constituents fetch failed; using cache.\n  " + "\n  ".join(errors), file=sys.stderr)
    if key not in cache:
        raise RuntimeError(f"no member list for {UNIVERSES[key]['name']} and no cached copy")
    return cache[key]


def _bars(raw: pd.DataFrame | None, t: str) -> pd.DataFrame | None:
    """One ticker's clean OHLCV out of a yf.download result (any column layout)."""
    if raw is None or raw.empty:
        return None
    df = raw
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if t in raw.columns.get_level_values(level):
                df = raw.xs(t, axis=1, level=level)
                break
        else:
            return None
    try:
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna()
    except KeyError:
        return None
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    return df[~df.index.duplicated(keep="last")].sort_index()


def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """A year of daily bars. Yahoo drops symbols under load, so the ones that
    come back empty are retried in smaller batches."""
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    pending = list(tickers)
    short_tail: dict[str, list[str]] = {}  # last date Yahoo sent -> tickers whose data stops earlier
    for attempt, batch in enumerate((len(pending), 50, 10)):
        if not pending:
            break
        if attempt:
            print(f"Retrying {len(pending)} tickers...", file=sys.stderr)
            time.sleep(10 * attempt)
        failed = []
        for i in range(0, len(pending), batch):
            chunk = pending[i:i + batch]
            try:
                raw = yf.download(
                    chunk, period="1y", interval="1d", auto_adjust=True,
                    group_by="ticker", threads=True, progress=False,
                )
            except Exception as exc:
                print(f"[warn] download of {len(chunk)} tickers failed: {exc}", file=sys.stderr)
                raw = None
            for t in chunk:
                df = _bars(raw, t)
                if df is None or df.empty:
                    failed.append(t)
                    continue
                if df.index[-1] < raw.index[-1]:  # final row incomplete (NaN) or missing
                    short_tail.setdefault(f"{raw.index[-1]:%Y-%m-%d}", []).append(t)
                if len(df) >= 150:  # shorter histories (recent IPOs) are skipped, not retried
                    out[t] = df
        pending = failed
    for day, ts in short_tail.items():
        print(f"[info] {len(ts)} tickers have no complete bar for {day}: {' '.join(ts[:20])}", file=sys.stderr)
    if pending:
        print(f"[warn] no data for {len(pending)} tickers: {' '.join(pending[:40])}", file=sys.stderr)
    return out


def drop_live_bar(frames: dict[str, pd.DataFrame], now: dt.datetime | None = None) -> dict[str, pd.DataFrame]:
    """Before the close, Yahoo's bar for today is still moving. Drop it so every
    level on the site comes from a finished session, even on a manual daytime run."""
    now = (now or dt.datetime.now(NY)).astimezone(NY)
    if now.time() >= BAR_FINAL:
        return frames
    today = now.date()
    return {t: d.iloc[:-1] if len(d) and d.index[-1].date() == today else d for t, d in frames.items()}


def session_date(frames: dict[str, pd.DataFrame]) -> dt.date | None:
    """The session most tickers end on: the close this scan describes."""
    last = pd.Series([d.index[-1].date() for d in frames.values() if len(d)])
    return last.value_counts().idxmax() if len(last) else None


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["sma10"] = d.Close.rolling(10).mean()
    d["sma20"] = d.Close.rolling(20).mean()
    d["sma50"] = d.Close.rolling(50).mean()
    d["adr"] = ((d.High / d.Low - 1) * 100).rolling(20).mean()
    d["vol50"] = d.Volume.rolling(50).mean()
    d["dvol20"] = (d.Close * d.Volume).rolling(20).mean()
    return d


def rs_ranks(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Average percentile of 1/3/6-month returns: how Qullamaggie finds leaders."""
    rets = pd.DataFrame({
        t: {p: d.Close.iloc[-1] / d.Close.iloc[-p - 1] - 1 for p in (21, 63, 126)}
        for t, d in frames.items() if len(d) > 127
    }).T
    if rets.empty:
        return {}
    return (rets.rank(pct=True).mean(axis=1) * 100).round(0).to_dict()


def regime(d: pd.DataFrame) -> dict:
    last = d.iloc[-1]
    above10, above20 = last.Close > last.sma10, last.Close > last.sma20
    rising10 = d.sma10.iloc[-1] > d.sma10.iloc[-6]
    rising20 = d.sma20.iloc[-1] > d.sma20.iloc[-6]
    if above10 and above20 and rising10 and rising20:
        state, text = "favorable", "above its rising 10- and 20-day averages. Breakouts tend to work in this tape."
    elif not above10 and not above20:
        state, text = "unfavorable", "below its 10- and 20-day averages. Breakouts fail more often here, so size down or wait."
    else:
        state, text = "mixed", "between its 10- and 20-day averages. Be selective and keep size moderate."
    return {
        "state": state,
        "text": text,
        "close": round(float(last.Close), 2),
        "sma10": round(float(last.sma10), 2),
        "sma20": round(float(last.sma20), 2),
    }


# --------------------------------------------------------------------------- #
# Setup detectors. Each returns a dict or None.
# --------------------------------------------------------------------------- #
def _day(ts) -> str:
    # notes name the session, since visitors read them on later days ("Sep 29", not "today")
    return f"{ts:%b} {ts.day}"


def _breakout_pattern(d: pd.DataFrame) -> dict | None:
    """Pattern test on the last bar of d (no status logic)."""
    c, h, l, v = (d[k].to_numpy() for k in ("Close", "High", "Low", "Volume"))
    n = len(d)
    last = d.iloc[-1]
    adr = float(last.adr)
    bmax, bmin, tn = CFG["bo_base_max"], CFG["bo_base_min"], CFG["bo_tight_days"]
    if n < bmax + 70:
        return None

    peak = n - bmax + int(np.argmax(h[-bmax:]))
    base_len = n - 1 - peak
    if base_len < bmin:
        return None

    # the prior leg into the peak
    leg_low = float(l[peak - 63:peak + 1].min())
    base_high = float(h[peak])
    prior_gain = base_high / leg_low - 1
    if prior_gain < CFG["bo_prior_gain"]:
        return None

    base_lows = l[peak:]
    base_low = float(base_lows.min())
    depth = 1 - base_low / base_high
    if depth > CFG["bo_max_depth"]:
        return None
    if (base_high - base_low) / (base_high - leg_low) > CFG["bo_max_retrace"]:
        return None

    # lows holding: the base low was not set in the last 3 bars
    if int(np.argmin(base_lows)) >= len(base_lows) - 3:
        return None

    # tight recent range
    tight_high = float(h[-tn:].max())
    tight_low = float(l[-tn:].min())
    tight_pct = (tight_high / tight_low - 1) * 100
    if tight_pct > CFG["bo_tight_adr_mult"] * adr:
        return None

    # surfing rising moving averages
    if not (last.Close > last.sma50 and last.Close >= last.sma20 * 0.98):
        return None
    if not d.sma20.iloc[-1] > d.sma20.iloc[-6]:
        return None

    # volume drying up
    vol_ratio = float(v[-tn:].mean() / last.vol50)
    if vol_ratio > 1.0:
        return None

    return {
        "peak_idx": peak,
        "base_len": base_len,
        "base_high": base_high,
        "base_low": base_low,
        "prior_gain": prior_gain,
        "depth": depth,
        "tight_high": tight_high,
        "tight_low": tight_low,
        "tight_pct": tight_pct,
        "vol_ratio": vol_ratio,
    }


def detect_breakout(d: pd.DataFrame) -> dict | None:
    last, prev = d.iloc[-1], d.iloc[-2]
    adr = float(last.adr)

    # 1) Did a base that was valid yesterday break out today?
    y = _breakout_pattern(d.iloc[:-1])
    if y and last.High > y["tight_high"] and last.Volume >= CFG["bo_min_breakout_vol"] * prev.vol50:
        trigger = y["tight_high"]
        stop = max(float(last.Low), trigger * (1 - float(prev.adr) / 100))
        if last.Close > stop:
            return _bo_result("triggered", trigger, stop, y, d, adr,
                              extra=f"Broke out {_day(d.index[-1])} on {last.Volume / prev.vol50:.1f}x average volume")

    # 2) Valid base right now, trigger not yet taken
    p = _breakout_pattern(d)
    if not p:
        return None
    trigger = p["tight_high"]
    if (trigger / last.Close - 1) * 100 > CFG["bo_max_dist_adr"] * adr:
        return None
    stop = max(float(last.Low), trigger * (1 - adr / 100))
    return _bo_result("watch", trigger, stop, p, d, adr)


def _bo_result(status, trigger, stop, p, d, adr, extra=None):
    notes = [
        f"Prior move +{p['prior_gain'] * 100:.0f}%",
        f"base {p['base_len']} days, {p['depth'] * 100:.0f}% deep",
        f"last {CFG['bo_tight_days']} days {p['tight_pct']:.1f}% range",
        f"volume at {p['vol_ratio']:.2f}x the 50-day average",
    ]
    if p["tight_high"] < p["base_high"] * 0.995:
        notes.append(f"pivot sits below the base high of {p['base_high']:.2f}")
    if extra:
        notes.insert(0, extra)
    n = len(d)
    return {
        "type": "breakout",
        "side": "long",
        "status": status,
        "trigger": trigger,
        "stop": stop,
        "notes": notes,
        "zone": {
            "from": p["peak_idx"] - n,  # negative index from end
            "high": p["base_high"],
            "low": p["base_low"],
        },
    }


def detect_ep(d: pd.DataFrame) -> dict | None:
    o, h, l, c, v = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    vol50 = d.vol50.to_numpy()
    n = len(d)
    for age in range(CFG["ep_max_age"]):
        i = n - 1 - age
        gap = o[i] / c[i - 1] - 1
        vmult = v[i] / vol50[i - 1]
        if gap < CFG["ep_min_gap"] or vmult < CFG["ep_min_vol_mult"]:
            continue
        ep_low = float(l[i])
        if c[i:].min() < ep_low:       # lost the gap-day low: setup failed
            return None
        run_high = float(h[i:].max())
        last_close = float(c[-1])
        prior_3m = c[i - 1] / c[max(0, i - 64)] - 1
        close_pos = (c[i] - l[i]) / max(h[i] - l[i], 1e-9)
        notes = [
            f"Gapped +{gap * 100:.1f}% on {vmult:.1f}x average volume ({_day(d.index[i])}"
            + ("" if age == 0 else f", {age} session{'s' if age > 1 else ''} earlier") + ")",
            f"gap day closed in the {'upper' if close_pos >= 0.5 else 'lower'} half of its range",
        ]
        if prior_3m < 0.10:
            notes.append(f"neglected before the gap ({prior_3m * 100:+.0f}% over 3 months)")
        if age == 0:
            status, trigger = "watch", run_high  # buy tomorrow above the gap-day high (or its opening range)
        elif last_close > float(h[i]):
            status, trigger = "triggered", float(h[i])
        else:
            status, trigger = "watch", run_high
        return {
            "type": "ep",
            "side": "long",
            "status": status,
            "trigger": trigger,
            "stop": ep_low,
            "notes": notes,
            "zone": {"from": i - n, "high": float(h[i]), "low": ep_low},
        }
    return None


def _parabolic_run(d: pd.DataFrame) -> dict | None:
    c, l = d.Close.to_numpy(), d.Low.to_numpy()
    last = d.iloc[-1]
    k = CFG["ps_min_up_days"]
    up_days = 0
    for j in range(len(c) - 1, 0, -1):
        if c[j] > c[j - 1]:
            up_days += 1
        else:
            break
    if up_days < k:
        return None
    run_low = float(l[-CFG["ps_lookback"]:].min())
    gain = last.Close / run_low - 1
    ext_adr = (last.Close / last.sma10 - 1) * 100 / last.adr
    if gain < CFG["ps_min_gain"] and ext_adr < CFG["ps_min_ext_adr"]:
        return None
    return {"up_days": up_days, "gain": gain, "ext_adr": ext_adr, "run_low": run_low}


def detect_parabolic(d: pd.DataFrame) -> dict | None:
    last, n = d.iloc[-1], len(d)
    # 1) First crack today: the run qualified yesterday and today broke yesterday's low
    y = _parabolic_run(d.iloc[:-1])
    if y and last.Low < d.Low.iloc[-2]:
        trigger = float(d.Low.iloc[-2])
        stop = float(max(d.High.iloc[-2], last.High))
        if last.Close < stop:
            return _ps_result("triggered", trigger, stop, y, n,
                              f"First crack {_day(d.index[-1])}: broke the prior day's low")
    # 2) Still running: short a break of today's low next session
    p = _parabolic_run(d)
    if not p:
        return None
    return _ps_result("watch", float(last.Low), float(last.High), p, n)


def _ps_result(status, trigger, stop, p, n, extra=None):
    notes = [
        f"+{p['gain'] * 100:.0f}% in {CFG['ps_lookback']} days",
        f"{p['up_days']} straight up closes",
        f"{p['ext_adr']:.1f} ADRs above the 10-day average",
    ]
    if extra:
        notes.insert(0, extra)
    return {
        "type": "parabolic",
        "side": "short",
        "status": status,
        "trigger": trigger,
        "stop": stop,
        "notes": notes,
        "zone": {"from": -CFG["ps_lookback"], "high": stop, "low": p["run_low"]},
    }


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def chart_payload(d: pd.DataFrame) -> dict:
    t = d.tail(CFG["chart_bars"])
    r = lambda s: [round(float(x), 2) if math.isfinite(x) else None for x in s]
    return {
        "d": [x.strftime("%Y-%m-%d") for x in t.index],
        "o": r(t.Open), "h": r(t.High), "l": r(t.Low), "c": r(t.Close),
        "v": [int(x) for x in t.Volume],
        "sma10": r(t.sma10), "sma20": r(t.sma20),
    }


def scan_universe(key: str, members: list[dict], frames: dict[str, pd.DataFrame]) -> dict:
    info = {m["ticker"]: m for m in members}
    uni = {t: prep(frames[t]) for t in info if t in frames}
    ranks = rs_ranks(uni)
    setups = []
    for t, d in uni.items():
        last = d.iloc[-1]
        need = last[["Close", "sma10", "sma20", "sma50", "adr", "vol50", "dvol20"]].to_numpy(dtype=float)
        if not np.isfinite(need).all() or last.adr <= 0:
            continue  # bad or too-short data; never let NaN reach the site
        if last.Close < CFG["min_price"] or last.dvol20 < CFG["min_dollar_vol"] or last.adr < CFG["min_adr"]:
            # parabolic shorts can come from calmer names that suddenly go vertical
            candidates = [detect_parabolic]
        else:
            candidates = [detect_breakout, detect_ep, detect_parabolic]
        rs = ranks.get(t, 0)
        for fn in candidates:
            if fn is detect_breakout and rs < CFG["bo_min_rs"]:
                continue
            try:
                s = fn(d)
            except Exception as exc:
                print(f"[warn] {t} {fn.__name__}: {exc}", file=sys.stderr)
                continue
            if not s:
                continue
            close = float(last.Close)
            trig, stop = float(s["trigger"]), float(s["stop"])
            if s["side"] == "long":
                dist = (trig / close - 1) * 100
                risk = (trig - stop) / trig * 100
            else:
                dist = (close / trig - 1) * 100
                risk = (stop - trig) / trig * 100
            if not (math.isfinite(dist) and math.isfinite(risk) and risk > 0):
                print(f"[warn] {t} {s['type']}: skipped, levels trigger={trig} stop={stop}", file=sys.stderr)
                continue
            setups.append({
                "ticker": t,
                "name": info[t].get("name", ""),
                "sector": info[t].get("sector", ""),
                **s,
                "trigger": round(trig, 2),
                "stop": round(stop, 2),
                "close": round(close, 2),
                "dist_pct": round(dist, 2),
                "risk_pct": round(risk, 2),
                "risk_adr": round(risk / float(last.adr), 2),
                "adr": round(float(last.adr), 2),
                "rs": int(rs),
                "zone": {**s["zone"], "high": round(s["zone"]["high"], 2), "low": round(s["zone"]["low"], 2)},
                "chart": chart_payload(d),
            })
    order = {"triggered": 0, "watch": 1}
    setups.sort(key=lambda s: (order[s["status"]], abs(s["dist_pct"])))
    return {"name": UNIVERSES[key]["name"], "scanned": len(uni), "setups": setups}


def run(demo: bool = False) -> dict:
    if demo:
        from demo_data import demo_universes
        members_by_key, frames = demo_universes()
    else:
        members_by_key, errors = {}, []
        for k in UNIVERSES:  # try every index before giving up, so one log shows every problem
            try:
                members_by_key[k] = fetch_constituents(k)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise SystemExit("Could not load index members: " + "; ".join(errors))
        tickers = sorted({m["ticker"] for ms in members_by_key.values() for m in ms}
                         | {u["benchmark"] for u in UNIVERSES.values()})
        print(f"Downloading {len(tickers)} tickers...", file=sys.stderr)
        frames = drop_live_bar(download(tickers))

    # Only scan tickers whose data reaches the latest session, so a symbol Yahoo
    # stopped updating can't show up with last week's levels.
    session = session_date(frames)
    if session is None:
        raise SystemExit("No price data came back; keeping the previous scan.")
    print(f"Latest session in the data: {session}", file=sys.stderr)
    stale = sorted(t for t, d in frames.items() if d.index[-1].date() != session)
    if stale:
        print(f"[warn] {len(stale)} tickers have no bar for {session}: {' '.join(stale[:40])}", file=sys.stderr)
    frames = {t: d for t, d in frames.items() if t not in stale}

    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "demo": demo,
        "settings": CFG,
        "universes": {},
    }
    for key, u in UNIVERSES.items():
        members = members_by_key[key]
        have = sum(m["ticker"] in frames for m in members)
        if have < MIN_COVERAGE * len(members):
            # A bad data day would otherwise publish "no setups" as if the market were quiet.
            raise SystemExit(f"{u['name']}: fresh data for only {have} of {len(members)} members; "
                             "keeping the previous scan.")
        res = scan_universe(key, members, frames)
        bench = frames.get(u["benchmark"])
        res["benchmark"] = u["benchmark"]
        res["regime"] = regime(prep(bench)) if bench is not None else None
        res["as_of"] = session.isoformat()
        result["universes"][key] = res
        print(f"{u['name']}: {res['scanned']} scanned, {len(res['setups'])} setups", file=sys.stderr)
    return result


def write_summary(data: dict, path: str) -> None:
    """Markdown list of the setups for the GitHub Actions run page."""
    lines = []
    for u in data["universes"].values():
        lines += [f"### {u['name']}: {len(u['setups'])} setups at the close of {u['as_of']} "
                  f"({u['scanned']} stocks scanned)", ""]
        if u["setups"]:
            lines += ["| Ticker | Setup | Status | Trigger | Stop | Risk % |",
                      "|---|---|---|---:|---:|---:|"]
            lines += [f"| {s['ticker']} | {s['type']} | {s['status']} | {s['trigger']:.2f} "
                      f"| {s['stop']:.2f} | {s['risk_pct']:.2f} |" for s in u["setups"]]
            lines.append("")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def published_as_of(path: Path) -> str | None:
    """Session of the real scan already on the site, if any."""
    try:
        old = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if old.get("demo"):
        return None
    return max((u.get("as_of") or "" for u in old.get("universes", {}).values()), default="") or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args(argv)
    data = run(demo=args.demo)
    out = Path(args.out)
    new = max(u["as_of"] for u in data["universes"].values())
    old = None if args.demo else published_as_of(out)
    if old and new < old:
        # Some evenings Yahoo serves the previous session for hours after the close.
        print(f"::warning::Yahoo's data only reaches {new}, older than the published scan of {old}; "
              "keeping the published scan.")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: browsers reject NaN in JSON, so fail here rather than ship a broken page
    out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        write_summary(data, os.environ["GITHUB_STEP_SUMMARY"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
