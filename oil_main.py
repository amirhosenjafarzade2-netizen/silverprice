"""
Oil Macro Environment Analyzer (Streamlit), v1.0

Which macro environments have historically been favorable / unfavorable for oil (WTI), what are we in now, and would following it have worked?
Same method as the silver app: point-in-time indicators -> Low / Mid / High states -> rules fitted on a LEARN period, chosen on VALIDATION,
graded once on a hidden FINAL TEST, with HAC t-stats, FDR q-values and expanding-window (fully out-of-sample) rules.

Files:
  app.py                       asset chooser (starts this file)
  oil_main.py                  config, data download, sidebar, calculations (this file)
  oil_indicators.py            indicator list, data sources, publication lags, feature builder
  macro_engine.py              asset-independent statistics / rules / backtest helpers (no Streamlit)
  oil_tabs_overview.py         Dashboard, Guide, Environments, Indicator ranking, Out-of-sample
  oil_tabs_analysis.py         Backtest, Explorer, Data & coverage
  oil_tabs_expectations.py     Expectations tab (rates, inflation, growth, dollar, recession expectations and their link to oil)
  silver_expectations.py       expectation engine, shared with the silver app

Optional secrets (Streamlit secrets or environment): FRED_API_KEY (more reliable FRED), EIA_API_KEY (free; unlocks OPEC / world supply-demand,
spare capacity and the true futures curve).
Educational only, not financial advice.
"""
import hashlib
import json
import os
import runpy
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # noqa: F401  (used by the tab files)
import streamlit as st
import yfinance as yf

_HERE0 = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
if _HERE0 not in sys.path:
    sys.path.insert(0, _HERE0)
from macro_engine import (EXPO, FAV, NEU, NO_SIGNAL, STRATEGIES, UNF, avail, bh_q, cot_fetch, eia_fetch, exp_pctl,  # noqa: E402,F401
                          fmt_summary, fred_fetch, fwd_drawdown, month_end, nw_t, pc, perf, rank_ic, run_backtest, run_rules,
                          split3, strategy_weights, summarize, walk_forward_rules)
from oil_indicators import (COT_CODE, CORE, FRED, FUT_IDS, LAGS, MANUAL_FILE, META_ALL, OIL_EXP_SIGN, PILLARS, SIGN,  # noqa: E402,F401
                            SOURCES, STEO_IDS, YF, build_features)
from silver_expectations import EXP_FRED, EXP_INFO, EXP_KEYS, EXP_H  # noqa: E402,F401

HORIZONS = (1, 2, 3, 6, 9, 12)
TARGETS = {"Oil's return (WTI)": "ret", "Oil, volatility-scaled forward return": "vol",
           "Oil avoids a deep drawdown (yes / no)": "ddb", "Oil minus cash (T-bills)": "cash"}
VOL_TARGET, DD_FLOOR, RULE_STEP, DATA_START = 0.40, 0.20, 6, "1990-01-01"
TGT_TXT = {"ret": "oil is higher", "vol": "oil's volatility-scaled forward return is positive",
           "ddb": f"oil avoids a {DD_FLOOR:.0%}+ drawdown", "cash": "oil beats cash"}
ICON = {FAV: "🟢", NEU: "⚪", UNF: "🔴"}


def show_df(box, d):
    try:
        box.dataframe(d, hide_index=True, width="stretch")
    except Exception:  # noqa: BLE001
        box.dataframe(d, hide_index=True, use_container_width=True)


def show_plot(box, f):
    try:
        box.plotly_chart(f, width="stretch")
    except Exception:  # noqa: BLE001
        box.plotly_chart(f, use_container_width=True)


def card(box):
    try:
        return box.container(border=True)
    except TypeError:
        return box.container()


def fmt_val(c, v):
    if pd.isna(v):
        return "-"
    return f"{v:+.1%}" if META_ALL[c][3] else f"{v:.2f}"


def state_label(c, state):
    return {"Low": META_ALL[c][1], "Mid": "Neutral", "High": META_ALL[c][2]}[state]


def get_key(name):
    key = None
    try:
        key = st.secrets.get(name)
    except Exception:  # noqa: BLE001
        pass
    return key or os.environ.get(name)


PEEK_FILE = os.path.join(_HERE0, ".final_test_peeks_oil.json")


def register_peek(sig):
    """Counts how many DISTINCT sets of settings have been shown the final test."""
    h_ = hashlib.md5(sig.encode()).hexdigest()
    try:
        with open(PEEK_FILE) as f:
            seen = json.load(f)
    except Exception:  # noqa: BLE001
        seen = list(st.session_state.get("peeks_oil", []))
    new = h_ not in seen
    if new:
        seen = seen + [h_]
        st.session_state["peeks_oil"] = seen
        try:
            with open(PEEK_FILE, "w") as f:
                json.dump(seen, f)
        except Exception:  # noqa: BLE001
            pass
    return len(seen), new


# ------------------------------------------------------------------ data
def _yahoo_fetch(start):
    px = yf.download(list(YF.values()), start=start, auto_adjust=True, progress=False, threads=True)["Close"]
    return px.rename(columns={v: k for k, v in YF.items()})


def _gpr_fetch():
    """Caldara-Iacoviello geopolitical risk index (monthly Excel file). Needs xlrd; failure is non-fatal."""
    import io
    import requests
    r = requests.get("https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls", timeout=(5, 30), headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    d = pd.read_excel(io.BytesIO(r.content))
    mc = next(c for c in d.columns if str(c).lower().startswith("month"))
    gc = next(c for c in d.columns if str(c).upper() == "GPR")
    s = pd.Series(pd.to_numeric(d[gc], errors="coerce").values, index=pd.to_datetime(d[mc]) + pd.offsets.MonthEnd(0)).dropna()
    return s[~s.index.duplicated(keep="last")]


def _manual_series():
    """Optional oil_manual_series.csv next to the app: date (observation month), rigs, opecplus_prod. Missing file is fine."""
    p = os.path.join(_HERE0, MANUAL_FILE)
    if not os.path.exists(p):
        return {}
    d = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    return {c: pd.to_numeric(d[c], errors="coerce").dropna() for c in ("rigs", "opecplus_prod") if c in d}


def _prepare_monthly(px, fr):
    """Re-date every series by its publication date, join with prices, forward-fill, take month-ends."""
    out = {}
    for k in ("m2", "cpi", "indpro"):
        if k in fr:
            out[k + "_yoy"] = avail(fr[k].pct_change(12) * 100, LAGS[k]).dropna()
            if k == "cpi":
                out["cpi_level"] = avail(fr[k], LAGS[k])
    if "gdp" in fr:
        out["gdp_yoy"] = avail(fr["gdp"].pct_change(4) * 100, LAGS["gdp"]).dropna()
    if "claims" in fr:
        out["claims"] = avail(fr["claims"].rolling(4).mean(), LAGS["claims"]).dropna()
    if "unrate" in fr:
        ur = fr["unrate"]
        out["unrate_lvl"] = avail(ur, LAGS["unrate"]).dropna()
        ma3 = ur.rolling(3).mean()
        out["sahm"] = avail(ma3 - ma3.shift(1).rolling(12).min(), LAGS["unrate"]).dropna()
    for k in ("prod_supplied", "gas_supplied"):  # weekly, noisy: 4-week average
        if k in fr:
            out[k] = avail(fr[k].rolling(4).mean(), LAGS[k]).dropna()
    handled = {"m2", "cpi", "indpro", "gdp", "claims", "unrate", "prod_supplied", "gas_supplied"}
    for k, s in fr.items():
        if k not in handled:
            out[k] = avail(s, LAGS[k]).dropna() if k in LAGS else s
    daily = px.join(pd.concat(out, axis=1), how="outer") if out else px
    daily["oil"] = daily["oil_wti"].combine_first(daily["oil_fut"]) if "oil_wti" in daily else daily["oil_fut"]
    last = daily["oil"].last_valid_index()
    daily = daily.sort_index().loc[:last].ffill()  # never run past the last real price date (lagged series carry future dates)
    return month_end(daily).dropna(subset=["oil"])


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_dataset(fred_key, eia_key):
    """Download everything once (Yahoo, FRED, EIA, CFTC, GPR in parallel), build monthly data, features and point-in-time percentiles."""
    with ThreadPoolExecutor(max_workers=len(FRED) + 6) as ex:
        fy = ex.submit(_yahoo_fetch, DATA_START)
        ff = {k: ex.submit(fred_fetch, v, DATA_START, fred_key) for k, v in FRED.items()}
        fo = {"cot": ex.submit(cot_fetch, DATA_START, COT_CODE), "gpr": ex.submit(_gpr_fetch)}
        if eia_key:
            fo["steo"] = ex.submit(eia_fetch, "steo", "seriesId", STEO_IDS, eia_key, "1995-01", "monthly")
            fo["fut"] = ex.submit(eia_fetch, "petroleum/pri/fut", "series", FUT_IDS, eia_key, DATA_START, "daily")
        px = fy.result()
        fr, failed = {}, []
        for k, f in ff.items():
            try:
                fr[k] = f.result()
            except Exception:  # noqa: BLE001
                failed.append(k)
        for k, f in fo.items():
            try:
                res = f.result()
                if k in ("steo", "fut"):
                    for c in res.columns:
                        fr[c] = res[c].dropna()
                else:
                    fr[k] = res
            except Exception:  # noqa: BLE001
                failed.append(k)
    if not eia_key:
        failed.append("eia_key")
    fr.update(_manual_series())
    if "oil_wti" in failed and "oil_fut" not in px:
        raise RuntimeError("no oil price could be downloaded")
    m = _prepare_monthly(px, fr)
    F = build_features(m)
    return m, F, F.apply(exp_pctl), sorted(set(failed)), time.time()


def target_returns(m, idx, h, kind):
    """ret = WTI forward return; vol = scaled to constant volatility using trailing vol only; ddb = 1 if the forward max drawdown stays shallower than DD_FLOOR;
    cash = WTI minus T-bills."""
    if kind == "ddb":
        d = fwd_drawdown(m["oil"], h)
        return (d > -DD_FLOOR).astype(float).where(d.notna()).reindex(idx)
    s = m["oil"].shift(-h) / m["oil"] - 1
    if kind == "vol":
        vol = m["oil"].pct_change().rolling(12).std() * np.sqrt(12)
        s = s * (VOL_TARGET / vol).clip(0.25, 2.0).fillna(1.0)
    elif kind == "cash":
        s = s - (((1 + m["irx"].fillna(0) / 100) ** (h / 12) - 1) if "irx" in m else 0)
    return s.reindex(idx)


@st.cache_data(show_spinner=False)
def _rules_c(F, P, fwd, learn_vals, h, cfg):
    return run_rules(F, P, fwd, pd.DatetimeIndex(learn_vals), h, cfg)


@st.cache_data(show_spinner=False)
def _wf_c(F, P, fwd, h, cut, step, cfg):
    return walk_forward_rules(F, P, fwd, h, cut, step, cfg)


# ------------------------------------------------------------------ sidebar (part 1: before data)
with st.sidebar:
    st.header("Settings")
    st.subheader("Evaluation mode")
    eval_mode = st.radio("Mode", ["🔬 Research (final test hidden)", "🔒 Locked evaluation (reveal final test)"], label_visibility="collapsed",
                         help="Research: experiment freely, everything is graded on VALIDATION only. Locked: freeze your choices, then reveal the final test. "
                              "Each distinct set of settings it is shown for is counted.")
    REVEAL = eval_mode.startswith("🔒") and st.checkbox("I have finished research: reveal the final test for these exact settings", False)
    LK = REVEAL
    if LK:
        st.caption("🔒 Settings are frozen while the final test is revealed.")
    if st.button("🔄 Refresh data", help="Data is downloaded once and reused for 6 hours."):
        get_dataset.clear()
    start = st.text_input("Data start date", "2000-01-01", disabled=LK)
    target_name = st.selectbox("What to predict", list(TARGETS), disabled=LK)
    target = TARGETS[target_name]
    h = st.select_slider("Look-ahead period (months)", options=list(HORIZONS), value=6, disabled=LK)
    st.subheader("Data split")
    train_frac = st.slider("LEARN share (rules are fitted here)", 0.3, 0.6, 0.5, 0.05, disabled=LK)
    val_frac = st.slider("VALIDATION share (choices are made here)", 0.1, 0.3, 0.2, 0.05, disabled=LK)
    st.caption(f"Final test = the remaining {max(0.0, 1 - train_frac - val_frac):.0%}.")
    st.subheader("Rules")
    pit = st.selectbox("How to define Low / Mid / High", ["Fixed terciles from the learn period", "Point-in-time percentile (expanding)"],
                       disabled=LK).startswith("Point")
    sel_fdr = st.selectbox("Select environments by", ["HAC t-stat", "FDR q-value (multiple-testing adjusted)"], disabled=LK).startswith("FDR")
    t_thr = st.slider("Minimum strength (HAC t-stat)", 0.5, 3.0, 1.5, 0.1, disabled=sel_fdr or LK)
    q_thr = st.slider("Maximum FDR q-value", 0.05, 0.50, 0.20, 0.05, disabled=(not sel_fdr) or LK)
    need_cons = st.checkbox("Require the effect in both halves of the learn period", True, disabled=LK)
    wmode = "equal" if st.selectbox("Indicator weighting", ["Equal (+1 / -1)", "By strength (shrunk: |t| − 1, capped)"], disabled=LK).startswith("Equal") else "strength"
    use_exp = st.checkbox("Add the expectation indicators to the rules", False, disabled=LK,
                          help="Adds the ten forward-looking indicators (expected Fed path, inflation, growth, dollar...). They are correlated with the other monetary indicators and shorten the history.")
    st.subheader("Backtest")
    strat = st.selectbox("Strategy", list(STRATEGIES), disabled=LK)
    vehicle = st.selectbox("Vehicle", ["USO (oil ETF, roll costs are inside the price)", "CL=F (Yahoo front-month futures: roll yield is NOT captured)"], disabled=LK)
    bps = st.slider("Trading cost (bps per 100% traded, incl. slippage)", 0, 100, 20, 5, disabled=LK)

st.title("🛢️ Oil Macro Environment Analyzer")
st.caption("Historically favorable or unfavorable macro environments for oil (WTI). Not a buy/sell signal, not financial advice.")
if train_frac + val_frac > 0.85:
    st.error("Learn + validation share must leave at least 15% for the final test.")
    st.stop()

try:
    with st.spinner("Loading data (downloaded once, then reused)..."):
        m, F_full, P_full, failed, data_ts = get_dataset(get_key("FRED_API_KEY"), get_key("EIA_API_KEY"))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not download market data: {e}")
    st.stop()
try:
    F_all = F_full.loc[start:]
except Exception:  # noqa: BLE001
    F_all = F_full.iloc[0:0]
if F_all.empty:
    st.error("The start date is invalid or after the last available data. Use a format like 2000-01-01.")
    st.stop()

# ------------------------------------------------------------------ sidebar (part 2: needs the data)
available = [c for c in F_all.columns if F_all[c].notna().any()]
late = F_all.index[0] + pd.DateOffset(years=4)  # an indicator that starts later than this would cut the usable history, so it is off by default
default = [c for c in available if (c in CORE or (use_exp and c in EXP_KEYS)) and (F_all[c].first_valid_index() or pd.Timestamp.max) <= late] or available
with st.sidebar:
    st.caption(f"Data through {m.index[-1]:%b %Y}, downloaded {(time.time() - data_ts) / 60:.0f} min ago.")
    st.subheader("Indicators")
    chosen = st.multiselect("Indicators to use", available, default=default, format_func=lambda c: META_ALL[c][0], disabled=LK,
                            help="Core set with a long history is on by default. Every extra indicator is another chance for a fluke, and "
                                 "indicators that start late shorten the usable history (see Data & coverage).")
if len(chosen) < 3:
    st.error("Choose at least 3 indicators.")
    st.stop()
F_ok = F_all[chosen].dropna()
if len(F_ok) < 90:
    st.error(f"Only {len(F_ok)} months have all chosen indicators. Move the start date earlier or drop the indicators with the latest start "
             f"({', '.join(F_all[chosen].apply(lambda s: s.first_valid_index()).sort_values().index[-2:])}).")
    st.stop()
P_ok = P_full[chosen].reindex(F_ok.index) if pit else None
cfg = ("q" if sel_fdr else "t", q_thr if sel_fdr else t_thr, need_cons, wmode)
META = {c: META_ALL[c] for c in chosen}

if "oil_wti" in failed:
    st.warning("FRED did not return WTI spot, so the Yahoo CL=F futures series is the price (roll gaps are inside it).")
miss = [k for k in failed if k not in ("eia_key",)]
if "eia_key" in failed:
    st.info("No EIA_API_KEY found: OPEC production, spare capacity, world supply / demand and the true futures curve are unavailable "
            "(the curve falls back to a USL/USO proxy). A free key from eia.gov/opendata adds them. Add it to Streamlit secrets or the environment.")
if miss:
    st.caption("Series that failed to download and were skipped: " + ", ".join(miss) + ". See the Data & coverage tab.")

# ------------------------------------------------------------------ calculations
fwd = target_returns(m, F_ok.index, h, target)
fdd = fwd_drawdown(m["oil"], h).reindex(F_ok.index)
sp = split3(F_ok, fwd, h, train_frac, val_frac)
ev, learn_idx, val_idx, test_idx = sp["ev"], sp["learn"], sp["val"], sp["test"]
unseen_idx = sp["unseen"] if REVEAL else val_idx
ev_r = ev if REVEAL else ev.intersection(learn_idx.union(val_idx))
grade_idx = test_idx if REVEAL else val_idx
grade_name = "Final test" if REVEAL else "Validation"
unseen_lbl = "validation + final test" if REVEAL else "validation"
if len(learn_idx) < 36 or len(val_idx) < 8 or len(test_idx) < 12:
    st.error("Not enough history for a learn / validation / test split. Move the start date earlier, shrink the learn or validation share, or use fewer indicators.")
    st.stop()

peek = register_peek(repr((start, target, h, sorted(chosen), train_frac, val_frac, pit, cfg, strat, vehicle, bps, use_exp))) if REVEAL else None

with st.spinner("Fitting rules (cached after the first run)..."):
    S, stats, good, bad, score, verdict = _rules_c(F_ok, P_ok, fwd, learn_idx.values, h, cfg)
    wfr = _wf_c(F_ok, P_ok, fwd, h, sp["cut"], RULE_STEP, cfg)

now = F_ok.index[-1]
v_now, sc_now = verdict.loc[now], float(score.loc[now])
tgt_txt = TGT_TXT[target]
st.info(f"Predicting: **{target_name}** over **{h} months**  |  Indicators: **{len(chosen)}**  |  Learn {learn_idx[0]:%b %Y}–{learn_idx[-1]:%b %Y}, "
        f"validation {val_idx[0]:%b %Y}–{val_idx[-1]:%b %Y}, "
        + (f"final test {test_idx[0]:%b %Y}–{test_idx[-1]:%b %Y}" if REVEAL else "final test hidden (research mode)"))
if REVEAL:
    n_pk, new_pk = peek
    (st.success if n_pk == 1 else st.warning)(
        "🔒 Final test revealed. " + ("First set of settings it has ever been shown for: a clean grade." if n_pk == 1 else
        f"Shown under {n_pk} different sets of settings ({'this one is new' if new_pk else 'already counted'}). Only the first reveal is clean; treat results as optimistic."))
else:
    st.info("🔬 Research mode: the final test is hidden everywhere and all grading uses the validation period.")

tabs_def = [("dash", "📊 Dashboard"), ("guide", "📖 Guide"), ("env", "🗺 Environments"), ("rank", "🏆 Indicator ranking"),
            ("test", "🧪 Out-of-sample"), ("bt", "💰 Backtest"), ("exp", "🔎 Explorer"), ("xpt", "📈 Expectations"), ("data", "🗂 Data & coverage")]
T = dict(zip([k for k, _ in tabs_def], st.tabs([n for _, n in tabs_def])))

# Each tab file runs with everything defined above available as globals; what it defines is passed on to the next file.
_ns = {k: v for k, v in globals().items() if not (k.startswith("__") and k.endswith("__"))}
for _fname in ("oil_tabs_overview.py", "oil_tabs_analysis.py", "oil_tabs_expectations.py"):
    _out = runpy.run_path(os.path.join(_HERE0, _fname), init_globals=_ns)
    _ns.update({k: v for k, v in _out.items() if not (k.startswith("__") and k.endswith("__"))})
