# Qullamaggie setups

A static site that lists active Kristjan Kullamägi (Qullamaggie) setups in the
**Nasdaq-100** and **S&P 500** with entry triggers and stops. Visitors switch
between the two indexes with the toggle at the top of the page (or link straight
to one with `#ndx` / `#spx`).

- **Breakouts**: strong prior leg, tight base on rising MAs, volume drying up
- **Episodic pivots**: big gap on heavy volume, still holding the gap-day low
- **Parabolic shorts**: extended multi-day run, short the first crack

A GitHub Action scans daily bars (Yahoo Finance via `yfinance`) after every US
close and commits `docs/data/setups.json`. GitHub Pages serves `docs/`.

Site: **https://balighmehrez.github.io/Qullamaggie/** (once Pages is on, below).

## Publishing (one time)

1. **Settings → Pages** → *Build and deployment* → Source: **Deploy from a branch**
   → pick the repo's default branch and the **`/docs`** folder → **Save**.
   The site is live about a minute later.
2. Optional: rename the default branch to `main` under **Settings → General →
   Default branch**. GitHub moves the Pages source and the nightly schedule
   with it.

After that it runs by itself:

- **Nightly scan** runs at 22:30 UTC Monday to Friday (6:30 pm New York in
  summer, 5:30 pm in winter), with a backup run at 12:15 UTC the next morning,
  before the open. GitHub can start scheduled runs hours late and Yahoo
  sometimes lags behind the close, so a scan never replaces a newer one.
  Use **Actions → Nightly scan → Run workflow** for a fresh scan any time;
  during market hours it ignores the unfinished bar, so the levels are always
  from a completed session.
- If Yahoo or Wikipedia has a bad day (under 90% of an index downloads), the
  scan fails and the previous results stay up rather than a half-empty list.
  The page warns visitors when the data is more than five days old.
- GitHub pauses scheduled workflows in public repos with no activity for 60
  days; the nightly commit keeps it active.

## Local use

```bash
pip install -r scanner/requirements.txt
python scanner/scan.py          # real scan
python scanner/scan.py --demo   # synthetic data for UI work
cd docs && python -m http.server  # open http://localhost:8000

pip install pytest && python -m pytest tests   # detector and data-handling tests
```

## Tuning

All thresholds live in `CFG` at the top of `scanner/scan.py`. They are relaxed
for large caps (ADR ≥ 2.5%, gaps ≥ 8%, parabolic ≥ 50% or ≥ 5 ADRs above the
10-day MA). Qullamaggie's originals target small/mid caps with ADR ≥ 5%.

## Notes on levels

- End-of-day scan: the trigger is where to act next session. Qullamaggie enters
  on the break of the 1-/5-minute opening-range high and stops at the low of the
  day, so tighten the listed stop to the actual day low after entry.
- Breakout stops are capped at 1 ADR below the trigger.
- Screening tool only, not investment advice.
