"""Scanner tests on hand-built price series (no network)."""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
import scan  # noqa: E402

END = "2026-09-29"


def bars(closes, vol=None, spread=0.03, end=END):
    """Daily bars that open at the prior close, with a fixed intraday range."""
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({
        "Open": o,
        "High": np.maximum(o, c) * (1 + spread / 2),
        "Low": np.minimum(o, c) * (1 - spread / 2),
        "Close": c,
        "Volume": np.full(len(c), 1e6) if vol is None else np.asarray(vol, float),
    }, index=pd.bdate_range(end=end, periods=len(c)))
    return df


def add_bar(df, o, h, l, c, v):
    idx = df.index[-1] + pd.offsets.BDay(1)
    row = pd.DataFrame({"Open": [o], "High": [h], "Low": [l], "Close": [c], "Volume": [v]}, index=[idx])
    return pd.concat([df, row])


def flat(n, price):
    return price * (1 + 0.002 * (-1) ** np.arange(n))


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
@pytest.fixture
def base_df():
    """Flat, a 30-bar leg from 50 to 80, then a 30-bar base ending in a tight,
    quiet range just under the high."""
    closes = np.r_[
        flat(140, 50),
        np.geomspace(50, 80, 30),
        np.linspace(79, 72, 10),
        np.linspace(72.5, 77, 12),
        [77.6, 77.8, 77.5, 77.9, 77.7, 77.8, 77.6, 77.7],
    ]
    vol = np.full(len(closes), 1e6)
    vol[140:170] = 2e6
    vol[-8:] = 0.5e6
    return bars(closes, vol)


def test_breakout_watch_levels(base_df):
    d = scan.prep(base_df)
    s = scan.detect_breakout(d)
    assert s and s["type"] == "breakout" and s["status"] == "watch" and s["side"] == "long"
    last = d.iloc[-1]
    assert s["trigger"] == pytest.approx(d.High.iloc[-5:].max())
    assert s["trigger"] >= last.Close
    # stop: today's low, but never more than one ADR under the trigger
    assert s["stop"] == pytest.approx(max(last.Low, s["trigger"] * (1 - last.adr / 100)))
    assert s["stop"] < s["trigger"]


def test_breakout_triggered_on_volume(base_df):
    pivot = scan.prep(base_df).High.iloc[-5:].max()
    df = add_bar(base_df, 77.7, pivot * 1.03, 77.4, pivot * 1.02, 3e6)
    s = scan.detect_breakout(scan.prep(df))
    assert s and s["status"] == "triggered"
    assert s["trigger"] == pytest.approx(pivot)
    assert s["stop"] == pytest.approx(77.4)  # day's low, inside the one-ADR cap
    assert "Broke out today" in s["notes"][0]


def test_breakout_stop_capped_at_one_adr(base_df):
    pivot = scan.prep(base_df).High.iloc[-5:].max()
    df = add_bar(base_df, 77.7, pivot * 1.03, 74.0, pivot * 1.02, 3e6)  # wide day: low 6% under
    d = scan.prep(df)
    s = scan.detect_breakout(d)
    assert s and s["status"] == "triggered"
    assert s["stop"] == pytest.approx(pivot * (1 - d.adr.iloc[-2] / 100))
    assert s["stop"] > 74.0


def test_breakout_needs_volume(base_df):
    pivot = scan.prep(base_df).High.iloc[-5:].max()
    df = add_bar(base_df, 77.7, pivot * 1.03, 77.4, pivot * 1.02, 0.6e6)
    s = scan.detect_breakout(scan.prep(df))
    assert not s or s["status"] != "triggered"


def test_no_breakout_without_prior_leg():
    assert scan.detect_breakout(scan.prep(bars(flat(200, 50)))) is None


@pytest.fixture
def ep_df():
    df = bars(flat(199, 40))
    return add_bar(df, 46.0, 48.0, 45.5, 47.5, 6e6)  # +15% gap on 6x volume


def test_ep_on_gap_day(ep_df):
    s = scan.detect_ep(scan.prep(ep_df))
    assert s and s["type"] == "ep" and s["status"] == "watch"
    assert s["trigger"] == pytest.approx(48.0)
    assert s["stop"] == pytest.approx(45.5)
    assert any("neglected" in n for n in s["notes"])


def test_ep_triggered_after_follow_through(ep_df):
    df = add_bar(ep_df, 47.6, 49.2, 47.3, 48.8, 2e6)
    df = add_bar(df, 48.8, 50.0, 48.6, 49.5, 2e6)
    s = scan.detect_ep(scan.prep(df))
    assert s and s["status"] == "triggered"
    assert s["trigger"] == pytest.approx(48.0)
    assert s["stop"] == pytest.approx(45.5)
    assert "2 days ago" in s["notes"][0]


def test_ep_dropped_when_gap_low_lost(ep_df):
    df = add_bar(ep_df, 47.0, 47.2, 44.8, 45.0, 3e6)
    assert scan.detect_ep(scan.prep(df)) is None


@pytest.fixture
def para_df():
    closes = np.r_[flat(190, 30), 30 * 1.06 ** np.arange(1, 11)]  # +79% in 10 straight up days
    return bars(closes, np.r_[np.full(190, 1e6), np.linspace(2e6, 5e6, 10)])


def test_parabolic_watch(para_df):
    d = scan.prep(para_df)
    s = scan.detect_parabolic(d)
    assert s and s["side"] == "short" and s["status"] == "watch"
    assert s["trigger"] == pytest.approx(d.Low.iloc[-1])
    assert s["stop"] == pytest.approx(d.High.iloc[-1])


def test_parabolic_first_crack(para_df):
    prev = para_df.iloc[-1]
    df = add_bar(para_df, prev.Close * 1.01, prev.High * 1.01, prev.Low * 0.98, prev.Low * 1.01, 6e6)
    s = scan.detect_parabolic(scan.prep(df))
    assert s and s["status"] == "triggered"
    assert s["trigger"] == pytest.approx(prev.Low)
    assert s["stop"] == pytest.approx(prev.High * 1.01)


# --------------------------------------------------------------------------- #
# Data handling
# --------------------------------------------------------------------------- #
def test_drop_live_bar_only_during_session():
    frames = {"A": bars(flat(160, 10), end="2026-09-29"), "B": bars(flat(160, 10), end="2026-09-28")}
    during = dt.datetime(2026, 9, 29, 11, 0, tzinfo=scan.NY)
    after = dt.datetime(2026, 9, 29, 17, 0, tzinfo=scan.NY)
    trimmed = scan.drop_live_bar(frames, during)
    assert trimmed["A"].index[-1].date() == dt.date(2026, 9, 28)
    assert len(trimmed["B"]) == 160  # yesterday's bar is final: untouched
    assert len(scan.drop_live_bar(frames, after)["A"]) == 160


def test_session_date_is_majority():
    frames = {t: bars(flat(160, 10)) for t in "ABC"}
    frames["D"] = bars(flat(160, 10), end="2026-09-25")
    assert scan.session_date(frames) == dt.date(2026, 9, 29)


def test_bars_extracts_any_layout():
    one = bars(flat(160, 10))
    by_ticker = pd.concat({"AAA": one, "BBB": one * np.nan}, axis=1)
    assert len(scan._bars(by_ticker, "AAA")) == 160
    assert scan._bars(by_ticker, "BBB").empty          # failed symbol comes back all-NaN
    assert scan._bars(by_ticker, "CCC") is None
    by_field = by_ticker.swaplevel(axis=1)
    assert len(scan._bars(by_field, "AAA")) == 160
    assert len(scan._bars(one, "AAA")) == 160           # single-level columns


def test_download_retries_dropped_symbols(monkeypatch):
    import yfinance as yf

    calls = []

    def fake_download(tickers, **kw):
        calls.append(list(tickers))
        ok = [t for t in tickers if t != "FLAKY" or len(calls) > 1]
        frames = {t: bars(flat(200, 20)) for t in ok}
        frames.update({t: bars(flat(200, 20)) * np.nan for t in tickers if t not in ok})
        return pd.concat(frames, axis=1)

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(scan.time, "sleep", lambda s: None)
    out = scan.download(["AAA", "FLAKY", "BBB"])
    assert set(out) == {"AAA", "FLAKY", "BBB"}
    assert calls[1] == ["FLAKY"]


def test_parse_constituents_handles_footnotes():
    rows = "".join(f"<tr><td>T{i:03d}</td><td>Co {i}</td><td>Tech</td></tr>" for i in range(99))
    html = f"""
      <table><tr><th>Ticker</th><th>Note</th></tr><tr><td>X</td><td>y</td></tr></table>
      <table><tr><th>Company</th><th>Ticker[12]</th><th>GICS Sector[13]</th></tr>
        <tr><td>Berkshire</td><td>BRK.B</td><td>Financials</td></tr>
        {"".join(f"<tr><td>Co {i}</td><td>T{i:03d}</td><td>Tech</td></tr>" for i in range(99))}
      </table>"""
    members = scan.parse_constituents(html, (90, 115))
    assert len(members) == 100
    assert members[0] == {"ticker": "BRK-B", "name": "Berkshire", "sector": "Financials"}
    with pytest.raises(ValueError):
        scan.parse_constituents(f"<table><tr><th>Symbol</th></tr>{rows}</table>", (480, 520))


def test_parse_constituents_two_row_header_and_exchange_prefix():
    body = "".join(f"<tr><td>NASDAQ: T{i:03d}</td><td>Co {i}[a]</td></tr>" for i in range(100))
    html = f"""<table>
      <tr><th colspan="2">Current components</th></tr>
      <tr><th>Ticker</th><th>Company</th></tr>{body}</table>"""
    members = scan.parse_constituents(html, (90, 115))
    assert len(members) == 100
    assert members[0] == {"ticker": "T000", "name": "Co 0", "sector": ""}


def test_parse_constituents_skips_ambiguous_changes_table():
    body = "".join(f"<tr><td>2020</td><td>A{i:03d}</td><td>R{i:03d}</td></tr>" for i in range(100))
    html = f"""<table>
      <tr><th rowspan="2">Date</th><th>Added</th><th>Removed</th></tr>
      <tr><th>Ticker</th><th>Ticker</th></tr>{body}</table>"""
    with pytest.raises(ValueError, match="tables on the page"):
        scan.parse_constituents(html, (90, 115))


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_demo_scan_writes_valid_json():
    data = scan.run(demo=True)
    json.dumps(data, allow_nan=False)
    assert set(data["universes"]) == {"ndx", "spx"}
    for u in data["universes"].values():
        assert u["as_of"] and u["setups"]
        for s in u["setups"]:
            if s["side"] == "long":
                assert s["stop"] < s["trigger"]
            else:
                assert s["stop"] > s["trigger"]
            assert s["risk_pct"] > 0
            assert len(s["chart"]["c"]) == scan.CFG["chart_bars"]


def test_partial_download_is_not_published(monkeypatch):
    members = [{"ticker": f"T{i}", "name": "", "sector": ""} for i in range(100)]
    monkeypatch.setattr(scan, "fetch_constituents", lambda key: members)
    monkeypatch.setattr(scan, "download", lambda tickers: {f"T{i}": bars(flat(200, 20)) for i in range(50)})
    with pytest.raises(SystemExit, match="only 50 of 100"):
        scan.run()
