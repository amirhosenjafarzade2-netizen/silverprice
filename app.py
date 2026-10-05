"""
Silver Macro Environment Analyzer (Streamlit)

Question answered: under which macro conditions has silver historically done well
(good time to buy) and badly (avoid)? And what is the environment today?

Run locally:  streamlit run app.py
requirements.txt: streamlit, yfinance, pandas, numpy, plotly, requests
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Silver Macro Environment Analyzer", page_icon="🥈", layout="wide")

FRED = {"real_yield": "DFII10", "breakeven": "T10YIE", "fed_funds": "DFF", "curve": "T10Y2Y",
        "m2": "M2SL", "cpi": "CPIAUCSL", "indpro": "INDPRO"}
HORIZONS = (1, 2, 3, 6, 9, 12)
ML_MODELS = ["Logistic regression", "Random Forest", "Gradient Boosting"]
YF = {"silver": "SI=F", "gold": "GC=F", "dollar": "DX-Y.NYB", "oil": "CL=F",
      "copper": "HG=F", "vix": "^VIX", "spx": "^GSPC",
      "tnx": "^TNX", "irx": "^IRX", "pl": "PL=F"}  # tnx/irx = Yahoo rate proxies, used if FRED is unreachable

# indicator -> (plain title, label when LOW, label when HIGH, is it a % change?)
META_ALL = {
    "real_yield":     ("Real 10y interest rate (level)", "Low / negative real rates", "High real rates", False),
    "real_yield_chg": ("Real rate, 3-month change", "Real rates falling", "Real rates rising", False),
    "breakeven":      ("Inflation expectations (10y)", "Low inflation expectations", "High inflation expectations", False),
    "fed_chg":        ("Fed funds rate, 6-month change", "Fed cutting", "Fed hiking", False),
    "curve":          ("Yield curve (10y minus 2y)", "Flat / inverted curve", "Steep curve", False),
    "dollar_mom":     ("US Dollar, 3-month trend", "Dollar weakening", "Dollar strengthening", True),
    "oil_mom":        ("Oil, 3-month trend", "Oil falling", "Oil rising", True),
    "copper_mom":     ("Copper, 3-month trend (growth proxy)", "Copper falling", "Copper rising", True),
    "vix":            ("Market fear (VIX)", "Calm markets", "Fearful markets", False),
    "spx_mom":        ("Stocks (S&P 500), 3-month trend", "Stocks falling", "Stocks rising", True),
    "gs_ratio":       ("Gold/silver ratio", "Silver expensive vs gold", "Silver cheap vs gold", False),
    "cpi_yoy":        ("Inflation (CPI, year-on-year %)", "Low inflation", "High inflation", False),
    "real_fed":       ("Real policy rate (Fed funds minus CPI)", "Low / negative real policy rate", "High real policy rate", False),
    "m2_yoy":         ("Money supply (M2) growth, YoY %", "Slow money growth", "Fast money growth", False),
    "indpro_yoy":     ("Industrial production, YoY %", "Weak industry", "Strong industry", False),
    "cu_au":          ("Copper/gold ratio, 3-month change", "Falling (fear / slowdown)", "Rising (growth)", True),
    "plat_mom":       ("Platinum, 3-month trend", "Platinum falling", "Platinum rising", True),
    # fallback-only (used when FRED real-rate data can't be downloaded)
    "nom_yield":      ("10y Treasury yield (level)", "Low yields", "High yields", False),
    "nom_yield_chg":  ("10y Treasury yield, 3-month change", "Yields falling", "Yields rising", False),
}
META = dict(META_ALL)  # narrowed to the available indicators after data loads
STATES = ["Low", "Mid", "High"]


def state_label(col, state):
    return {"Low": META[col][1], "Mid": "Neutral", "High": META[col][2]}[state]


# ------------------------------------------------------------------ data
def fred_series(series_id: str, start: str) -> pd.Series:
    """FRED via optional API key (st.secrets / env FRED_API_KEY), else the public CSV, with retries."""
    key = None
    try:
        key = st.secrets.get("FRED_API_KEY")
    except Exception:
        pass
    key = key or os.environ.get("FRED_API_KEY")
    last = None
    for _ in range(2):
        try:
            if key:
                r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                                 params=dict(series_id=series_id, api_key=key, file_type="json",
                                             observation_start=start), timeout=(10, 30))
                r.raise_for_status()
                obs = r.json()["observations"]
                d = pd.DataFrame(obs)[["date", "value"]]
                d["value"] = pd.to_numeric(d["value"], errors="coerce")
            else:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
                r = requests.get(url, timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                d = pd.read_csv(io.StringIO(r.text), na_values=[".", ""])
                d.columns = ["date", "value"]
            d["date"] = pd.to_datetime(d["date"])
            return d.set_index("date")["value"].astype(float)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def month_end(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.resample("ME").last()
    except ValueError:  # older pandas
        return df.resample("M").last()


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def yahoo_daily(start: str) -> pd.DataFrame:
    px = yf.download(list(YF.values()), start=start, auto_adjust=True, progress=False)["Close"]
    return px.rename(columns={v: k for k, v in YF.items()})


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fred_cached(series_id: str, start: str) -> pd.Series:
    return fred_series(series_id, start)  # exceptions are not cached, so failures retry next run


def load_monthly(start: str, cb=None):
    cb = cb or (lambda f, t: None)
    cb(0.02, "Downloading market data (Yahoo Finance)...")
    px = yahoo_daily(start)
    cb(0.15, "Market data ready. Downloading macro data (FRED)...")
    fr, failed = {}, []
    items = list(FRED.items())
    for i, (k, v) in enumerate(items):
        cb(0.15 + 0.2 * i / len(items), f"Downloading macro data: {v}...")
        try:
            fr[k] = fred_cached(v, start)
        except Exception:  # noqa: BLE001
            failed.append(k)
            if k == "real_yield":  # host is down: don't wait for the rest
                failed += [x for x in FRED if x != k]
                break
    for k in ("m2", "cpi", "indpro"):  # monthly series: YoY growth, available only after publication
        if k in fr:
            y = fr.pop(k).pct_change(12) * 100
            y.index = y.index + pd.DateOffset(months=1) + pd.Timedelta(days=20)
            fr[k + "_yoy"] = y.dropna()
    cb(0.35, "Preparing monthly data...")
    daily = px.join(pd.concat(fr, axis=1), how="outer") if fr else px
    daily = daily.sort_index().ffill().dropna(subset=["silver"])
    return month_end(daily).dropna(subset=["silver"]), sorted(set(failed))


def build_features(m: pd.DataFrame) -> pd.DataFrame:
    def g(k):
        return m[k] if k in m else pd.Series(np.nan, index=m.index)

    F = pd.DataFrame(index=m.index)
    F["real_yield"] = g("real_yield")
    F["real_yield_chg"] = g("real_yield").diff(3)
    F["breakeven"] = g("breakeven")
    F["fed_chg"] = g("fed_funds").diff(6)
    F["curve"] = g("curve")
    F["nom_yield"] = g("tnx")
    F["nom_yield_chg"] = g("tnx").diff(3)
    if F["fed_chg"].isna().all():            # FRED fallback: 3-month T-bill yield as the policy-rate proxy
        F["fed_chg"] = g("irx").diff(6)
    if F["curve"].isna().all():              # FRED fallback: 10y minus 3-month
        F["curve"] = g("tnx") - g("irx")
    F["dollar_mom"] = g("dollar").pct_change(3)
    F["oil_mom"] = g("oil").pct_change(3).clip(-0.8, 1.5)  # 2020 negative oil price glitch
    F["copper_mom"] = g("copper").pct_change(3)
    F["vix"] = g("vix")
    F["spx_mom"] = g("spx").pct_change(3)
    F["gs_ratio"] = g("gold") / g("silver")
    F["cpi_yoy"] = g("cpi_yoy")
    F["real_fed"] = g("fed_funds") - g("cpi_yoy")
    F["m2_yoy"] = g("m2_yoy")
    F["indpro_yoy"] = g("indpro_yoy")
    F["cu_au"] = (g("copper") / g("gold")).pct_change(3)
    F["plat_mom"] = g("pl").pct_change(3)
    keep = [c for c in META_ALL if F[c].notna().any()]
    if "real_yield" in keep:                 # real-rate data available: the Treasury-yield stand-ins aren't needed
        keep = [c for c in keep if not c.startswith("nom_")]
    return F[keep]


# ------------------------------------------------------------------ analysis
def assign_states(F: pd.DataFrame, learn_idx) -> pd.DataFrame:
    """Low/Mid/High using thresholds learned ONLY on the learn period."""
    S = pd.DataFrame(index=F.index, columns=F.columns, dtype=object)
    for c in F.columns:
        lo, hi = F.loc[learn_idx, c].quantile([1 / 3, 2 / 3])
        col = F[c]
        lab = np.where(col <= lo, "Low", np.where(col <= hi, "Mid", "High")).astype(object)
        S[c] = pd.Series(lab, index=F.index).where(col.notna())
    return S


def half_edge(fwd, S, col, state, idx):
    mask = S.loc[idx, col] == state
    r = fwd.loc[idx][mask]
    return r.mean() - fwd.loc[idx].mean() if len(r) >= 4 else np.nan


def bucket_stats(S, fwd, idx, h):
    base = fwd.loc[idx]
    mid = len(idx) // 2
    first, second = idx[:mid], idx[mid:]
    rows = []
    for c in S.columns:
        for s in STATES:
            r = base[S.loc[idx, c] == s]
            n = len(r)
            if n < 8:
                continue
            sd = r.std()
            eff_n = max(n / h, 1.0)  # months overlap, so fewer independent observations
            t = (r.mean() - base.mean()) / (sd / np.sqrt(eff_n)) if sd > 0 else 0.0
            e1, e2 = half_edge(fwd, S, c, s, first), half_edge(fwd, S, c, s, second)
            edge = r.mean() - base.mean()
            consistent = bool(np.sign(e1) == np.sign(e2) == np.sign(edge)) if not (np.isnan(e1) or np.isnan(e2)) else False
            rows.append(dict(col=c, state=s, n=n, avg=r.mean(), median=r.median(),
                             win=(r > 0).mean(), edge=edge, t=t, consistent=consistent))
    return pd.DataFrame(rows)


def build_rules(stats, t_thr, need_consistent):
    ok = stats[(stats["t"].abs() >= t_thr)]
    if need_consistent:
        ok = ok[ok["consistent"]]
    good = {(r.col, r.state) for r in ok.itertuples() if r.edge > 0}
    bad = {(r.col, r.state) for r in ok.itertuples() if r.edge < 0}
    return good, bad


def score_series(S, good, bad):
    score = pd.Series(0, index=S.index, dtype=float)
    for c in S.columns:
        for s in STATES:
            if (c, s) in good:
                score += (S[c] == s).astype(float)
            if (c, s) in bad:
                score -= (S[c] == s).astype(float)
    return score


def to_verdict(score, lo_thr, hi_thr):
    v = pd.Series("Neutral", index=score.index)
    v[(score >= hi_thr) & (score > 0)] = "Buy-friendly"
    v[(score <= lo_thr) & (score < 0)] = "Avoid"
    return v


def summarize(fwd, verdict, idx):
    rows = []
    for g in ["Buy-friendly", "Neutral", "Avoid", "All months"]:
        r = fwd.loc[idx] if g == "All months" else fwd.loc[idx].where(verdict.loc[idx] == g).dropna()
        if len(r) == 0:
            rows.append(dict(Environment=g, Months=0, Avg=np.nan, Median=np.nan, Win=np.nan))
        else:
            rows.append(dict(Environment=g, Months=len(r), Avg=r.mean(), Median=r.median(), Win=(r > 0).mean()))
    return pd.DataFrame(rows)


def fmt_summary(d):
    o = d.copy()
    o["Avg"] = o["Avg"].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}")
    o["Median"] = o["Median"].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}")
    o["Win"] = o["Win"].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
    return o.rename(columns={"Avg": "Avg silver return after", "Median": "Median return after",
                             "Win": "% of times silver rose"})


# ------------------------------------------------------------------ helpers: horizon, ML, scan
def fwd_returns(m, F_ok, h):
    return (m["silver"].shift(-h) / m["silver"] - 1).reindex(F_ok.index)


def split_idx(F_ok, fwd, h, learn_frac):
    ev = F_ok.index.intersection(fwd.dropna().index)
    cut = int(len(ev) * learn_frac)
    return ev, cut, ev[: max(cut - h, 0)], ev[cut:]  # purge h months so learn/test never overlap


def run_rules(F_ok, fwd, learn_idx, h, t_thr, need_consistent):
    S = assign_states(F_ok, learn_idx)
    stats = bucket_stats(S, fwd, learn_idx, h)
    good, bad = build_rules(stats, t_thr, need_consistent)
    score = score_series(S, good, bad)
    lo, hi = np.quantile(score.loc[learn_idx], 0.25), np.quantile(score.loc[learn_idx], 0.75)
    return S, stats, good, bad, score, to_verdict(score, lo, hi)


def rank_ic(a, b):
    """Rank correlation between a prediction and what happened. 0 = no skill; ~0.1 is decent for markets."""
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 8 or d.iloc[:, 0].nunique() < 2:
        return np.nan
    return d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())


def make_model(name):
    if name.startswith("Logistic"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    if name.startswith("Random"):
        return RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=10, n_jobs=-1, random_state=0)
    return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)


@st.cache_data(show_spinner=False)
def walk_forward(Fev, fwdev, h, cut, step, model_name, _cb=None):
    """Retrain every `step` months on past data only (minus h months), predict the next block."""
    y, X = (fwdev > 0).astype(int).values, Fev.values
    starts = list(range(cut, len(Fev), step))
    out = []
    for i, s0 in enumerate(starts):
        tr_end, te = s0 - h, slice(s0, min(s0 + step, len(Fev)))
        if tr_end < 36:
            continue
        ytr = y[:tr_end]
        if len(np.unique(ytr)) < 2:
            p = np.full(te.stop - te.start, ytr.mean())
        else:
            p = make_model(model_name).fit(X[:tr_end], ytr).predict_proba(X[te])[:, 1]
        out.append(pd.DataFrame({"p": p, "base": ytr.mean()}, index=Fev.index[te]))
        if _cb:
            _cb((i + 1) / len(starts), model_name)
    return pd.concat(out) if out else pd.DataFrame(columns=["p", "base"])


@st.cache_data(show_spinner=False)
def fit_today(Fev, fwdev, x_now, model_name):
    y = (fwdev > 0).astype(int).values
    mdl = make_model(model_name).fit(Fev.values, y)
    p = float(mdl.predict_proba(x_now.values.reshape(1, -1))[0, 1])
    est = mdl[-1] if hasattr(mdl, "steps") else mdl
    imp = est.coef_[0] if hasattr(est, "coef_") else est.feature_importances_
    return p, float(y.mean()), pd.Series(imp, index=Fev.columns)


def ml_verdict(wf, margin):
    d = wf["p"] - wf["base"]
    v = pd.Series("Neutral", index=wf.index)
    v[d >= margin] = "Buy-friendly"
    v[d <= -margin] = "Avoid"
    return v


@st.cache_data(show_spinner=False)
def scan_horizons(m, F_ok, model_name, learn_frac, t_thr, need_consistent, horizons, _cb=None):
    """Pick the look-ahead using ONLY the learn period: fit on its first 70%, score its last 30%."""
    rows = []
    for i, h_ in enumerate(horizons):
        fwd_ = fwd_returns(m, F_ok, h_)
        _, _, learn_, _ = split_idx(F_ok, fwd_, h_, learn_frac)
        k = int(len(learn_) * 0.7)
        A, V = learn_[: max(k - h_, 0)], learn_[k:]
        row = {"h": h_, "rules_ic": np.nan, "ml_ic": np.nan, "n_val": len(V)}
        if len(A) >= 36 and len(V) >= 10:
            S = assign_states(F_ok, A)
            g_, b_ = build_rules(bucket_stats(S, fwd_, A, h_), t_thr, need_consistent)
            row["rules_ic"] = rank_ic(score_series(S, g_, b_).loc[V], fwd_.loc[V])
            yA = (fwd_.loc[A] > 0).astype(int)
            if model_name and yA.nunique() > 1:
                mdl = make_model(model_name).fit(F_ok.loc[A].values, yA.values)
                row["ml_ic"] = rank_ic(pd.Series(mdl.predict_proba(F_ok.loc[V].values)[:, 1], index=V), fwd_.loc[V])
        rows.append(row)
        if _cb:
            _cb((i + 1) / len(horizons), f"Testing the {h_}-month look-ahead...")
    d = pd.DataFrame(rows)
    d["combined"] = d[["rules_ic", "ml_ic"]].mean(axis=1, skipna=True)
    return d


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    start = st.text_input("Data start date", "2004-01-01")
    auto_h = st.checkbox("🔍 Auto-find best look-ahead", False,
                         help="Tests 1-12 month look-aheads using only the learning period, then picks the one with the most signal.")
    h_manual = st.select_slider("Look-ahead period (months)", options=list(HORIZONS), value=6, disabled=auto_h)
    learn_frac = st.slider("Share of history used to LEARN the rules", 0.4, 0.8, 0.6, 0.05)
    t_thr = st.slider("Minimum strength to count an environment", 0.5, 2.5, 1.0, 0.1,
                      help="Higher = fewer, but more reliable, rules.")
    need_consistent = st.checkbox("Require the effect in both halves of the learn period", True)
    st.subheader("Machine learning")
    run_ml = st.checkbox("Also run machine-learning models", True)
    ml_choice = st.selectbox("Model shown in detail", ML_MODELS, disabled=not run_ml)
    margin = st.slider("Confidence margin (points above/below normal odds)", 0.02, 0.20, 0.05, 0.01, disabled=not run_ml)
    step = st.select_slider("Retrain every (months)", options=[3, 6, 12], value=6, disabled=not run_ml)

st.title("🥈 Silver Macro Environment Analyzer")
st.caption("Which macro conditions have been good or bad for buying silver, and where are we now?")

bar = st.progress(0.0, text="Starting...")


def cb(frac, text):
    bar.progress(float(min(max(frac, 0.0), 1.0)), text=text)


try:
    m, failed = load_monthly(start, cb)
except Exception as e:
    bar.empty()
    st.error(f"Could not download market data: {e}")
    st.stop()

F_all = build_features(m)
available = list(F_all.columns)
with st.sidebar:
    st.subheader("Indicators")
    chosen = st.multiselect("Indicators to use", available, default=available, format_func=lambda c: META_ALL[c][0],
                            help="More indicators means more chances for a fluke. Switch some off and see if results change.")
if len(chosen) < 3:
    bar.empty()
    st.error("Pick at least 3 indicators.")
    st.stop()
META = {c: META_ALL[c] for c in chosen}  # only indicators in use
F = F_all[chosen]
F_ok = F.dropna()

if "real_yield" in failed:
    st.warning(
        "FRED (the source for real rates, inflation expectations, CPI, M2 and industrial production) did not respond, "
        "so the app is using Yahoo Finance stand-ins (10y Treasury yield and the 3-month T-bill). Reload later to retry, "
        "or add a free FRED API key (Settings > Secrets: FRED_API_KEY = \"your_key\").")
elif failed:
    st.info("Some FRED series were unavailable and were skipped: " + ", ".join(failed))

# ---- look-ahead: manual or auto
scan = None
if auto_h:
    scan = scan_horizons(m, F_ok, ml_choice if run_ml else None, learn_frac, t_thr, need_consistent, HORIZONS,
                         lambda f, t: cb(0.35 + 0.15 * f, t))
    h = int(scan.loc[scan["combined"].idxmax(), "h"]) if scan["combined"].notna().any() else 6
else:
    h = h_manual

cb(0.5, "Analysing macro environments...")
fwd = fwd_returns(m, F_ok, h)
ev, cut, learn_idx, test_idx = split_idx(F_ok, fwd, h, learn_frac)
if len(learn_idx) < 36 or len(test_idx) < 12:
    bar.empty()
    st.error("Not enough history. Move the start date earlier, change the learn share, or use fewer indicators.")
    st.stop()

S, stats, good, bad, score, verdict = run_rules(F_ok, fwd, learn_idx, h, t_thr, need_consistent)

ml = {}
if run_ml:
    Fev, fev = F_ok.loc[ev], fwd.loc[ev]
    for i, name in enumerate(ML_MODELS):
        ml[name] = walk_forward(Fev, fev, h, cut, step, name,
                                lambda f, t, i=i: cb(0.55 + 0.45 * (i + f) / len(ML_MODELS), f"Training {t} ({h}-month look-ahead)..."))
    ml_p_now, ml_base_now, ml_imp = fit_today(Fev, fev, F_ok.iloc[-1], ml_choice)
cb(1.0, "Done")
bar.empty()

st.info(f"Look-ahead used: **{h} months**" + (" (auto-selected using the learning period only)" if auto_h else "")
        + f"  |  Indicators: **{len(META)}**")

tabs_def = [("guide", "📖 Guide"), ("map", "🗺 Environment map"), ("rank", "🏆 Best vs worst"),
            ("test", "🧪 Does it hold up?"), ("today", "📍 Today")]
if run_ml:
    tabs_def.append(("ml", "🤖 Machine learning"))
if auto_h:
    tabs_def.append(("scan", "🔍 Look-ahead scan"))
T = dict(zip([k for k, _ in tabs_def], st.tabs([n for _, n in tabs_def])))

# ------------------------------------------------------------------ guide
with T["guide"]:
    st.markdown(f"""
### How this works (plain English)
**Method 1: environment rules**
1. Each month since {start[:4]} is described by **{len(META)} macro conditions** (rates, inflation, money supply, the Fed, the dollar, oil, copper, platinum, stocks, fear, gold/silver ratio).
2. Each condition is split into **Low / Neutral / High**, and we measure what silver did over the **next {h} months**.
3. Conditions where silver clearly beat the average become **buy-friendly signals**; the clearly worse ones become **avoid signals**. Adding them up gives a **macro score**.

**Method 2: machine learning** (tab 🤖)
Three models estimate the chance that silver is higher in {h} months. They are retrained over time, always using only past data (**walk-forward**), the way you would have used them in real life.

**Honesty built in:** everything is learned on the first {learn_frac:.0%} of history and judged on the later months the models never saw.
Look at **🧪 Does it hold up?** and the ML comparison before trusting anything.

**🔍 Auto-find look-ahead:** tries 1, 2, 3, 6, 9 and 12 months using only the learning period and picks the one with the most signal.
Because it is a "best of six" choice, treat it as a hint, then confirm on the untouched test period.

⚠️ Educational only, not financial advice. Macro conditions shift the odds slightly; they do not predict silver.
""")

# ------------------------------------------------------------------ map
with T["map"]:
    st.subheader(f"Average silver return over the next {h} months, by environment")
    st.caption("Learn period only. Green = silver tended to rise afterwards, red = tended to fall.")
    z, txt = [], []
    for c in META:
        zr, tr = [], []
        for s_ in STATES:
            r = stats[(stats["col"] == c) & (stats["state"] == s_)]
            if r.empty:
                zr.append(np.nan); tr.append("")
            else:
                zr.append(r["avg"].iloc[0] * 100)
                tr.append(f"{state_label(c, s_)}<br><b>{r['avg'].iloc[0]:+.1%}</b> (n={int(r['n'].iloc[0])})")
        z.append(zr); txt.append(tr)
    zmax = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 10
    fig = go.Figure(go.Heatmap(z=z, x=["Low", "Neutral", "High"], y=[META[c][0] for c in META],
                               text=txt, texttemplate="%{text}", colorscale="RdYlGn",
                               zmid=0, zmin=-zmax, zmax=zmax, showscale=False, xgap=3, ygap=3))
    fig.update_layout(height=max(420, 52 * len(META)), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    fig.update_traces(textfont_size=11)
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ ranking
def rank_table(df):
    return pd.DataFrame({
        "Environment": [state_label(r.col, r.state) for r in df.itertuples()],
        "Indicator": [META[r.col][0] for r in df.itertuples()],
        "Months": df["n"].values,
        "Avg return after": [f"{v:+.1%}" for v in df["avg"]],
        "vs. average": [f"{v * 100:+.1f} pts" for v in df["edge"]],
        "% rose": [f"{v:.0%}" for v in df["win"]],
        "Strength": [f"{v:+.1f}" for v in df["t"]],
        "Held in both halves?": ["✅" if v else "❌" for v in df["consistent"]],
    })


with T["rank"]:
    cL, cR = st.columns(2)
    with cL:
        st.subheader("🟢 Best environments to buy")
        st.dataframe(rank_table(stats.sort_values("edge", ascending=False).head(6)), hide_index=True, use_container_width=True)
    with cR:
        st.subheader("🔴 Environments to avoid")
        st.dataframe(rank_table(stats.sort_values("edge").head(6)), hide_index=True, use_container_width=True)
    st.caption(f"'vs. average' = extra silver return compared with the average month ({fwd.loc[learn_idx].mean():+.1%} over {h} months). "
               "'Strength' is roughly a t-statistic (above ±2 is fairly strong), adjusted for overlapping periods. "
               f"With {len(stats)} environments tested, a few will look good purely by luck. That is why the next tab matters.")
    st.markdown(f"**Rules in use** (strength ≥ {t_thr}{', held in both halves' if need_consistent else ''}): "
                f"{len(good)} buy-friendly, {len(bad)} avoid.")
    if good or bad:
        c1, c2 = st.columns(2)
        c1.markdown("**Buy-friendly:**\n" + "\n".join(f"- {state_label(c, s_)}" for c, s_ in sorted(good)) if good else "None")
        c2.markdown("**Avoid:**\n" + "\n".join(f"- {state_label(c, s_)}" for c, s_ in sorted(bad)) if bad else "None")

# ------------------------------------------------------------------ out of sample
with T["test"]:
    if not (good or bad):
        st.warning("No environment passed the strength filter. Lower the minimum strength in the sidebar.")
    else:
        a, b = st.columns(2)
        a.subheader("Learn period (in-sample)")
        a.caption(f"{learn_idx[0]:%b %Y} to {learn_idx[-1]:%b %Y}. Rules were built here, so this looks flattering.")
        a.dataframe(fmt_summary(summarize(fwd, verdict, learn_idx)), hide_index=True, use_container_width=True)
        b.subheader("Test period (out-of-sample) ✅")
        b.caption(f"{test_idx[0]:%b %Y} to {test_idx[-1]:%b %Y}. The rules never saw this data. This is the honest result.")
        tsum = summarize(fwd, verdict, test_idx)
        b.dataframe(fmt_summary(tsum), hide_index=True, use_container_width=True)
        bar_fig = go.Figure(go.Bar(x=tsum["Environment"], y=tsum["Avg"] * 100,
                                   marker_color=["#2e9e5b", "#9aa0a6", "#d64545", "#4a6fa5"],
                                   text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in tsum["Avg"] * 100], textposition="outside"))
        bar_fig.update_layout(title=f"Test period: average silver return over the next {h} months", yaxis_title="%", height=360)
        st.plotly_chart(bar_fig, use_container_width=True)
        tb = tsum.set_index("Environment")
        if tb.loc["Buy-friendly", "Months"] > 0 and tb.loc["Avoid", "Months"] > 0:
            spread = (tb.loc["Buy-friendly", "Avg"] - tb.loc["Avoid", "Avg"]) * 100
            if spread > 2:
                st.success(f"Out of sample, 'Buy-friendly' months beat 'Avoid' months by {spread:.1f} points. The signal has some support, "
                           "but the test period is short, so stay cautious.")
            else:
                st.warning(f"Out of sample, the gap between 'Buy-friendly' and 'Avoid' is only {spread:.1f} points. "
                           "The pattern did not hold up well, so don't lean on these rules.")
        st.caption(f"Months overlap (each looks {h} months ahead), so the number of truly independent observations is much smaller than the month count.")
        pl = go.Figure()
        pl.add_scatter(x=m.index, y=m["silver"], name="Silver", line=dict(color="#888"))
        for g_, col in [("Buy-friendly", "#2e9e5b"), ("Avoid", "#d64545")]:
            ix = verdict[verdict == g_].index
            pl.add_scatter(x=ix, y=m["silver"].reindex(ix), mode="markers", name=g_, marker=dict(color=col, size=6))
        pl.add_vline(x=test_idx[0], line_dash="dash", annotation_text="test period starts")
        pl.update_layout(title="Silver with macro verdicts", yaxis_type="log", height=380)
        st.plotly_chart(pl, use_container_width=True)

# ------------------------------------------------------------------ today
with T["today"]:
    now = F_ok.index[-1]
    cur, S_now = F_ok.loc[now], S.loc[now]
    v_now, sc_now = verdict.loc[now], int(score.loc[now])
    icon = {"Buy-friendly": "🟢", "Neutral": "⚪", "Avoid": "🔴"}[v_now]
    st.subheader(f"{icon} Current macro environment ({now:%b %Y}): {v_now}")
    c1, c2 = st.columns(2)
    c1.metric("Macro score (rules)", f"{sc_now:+d}", help="Buy-friendly conditions minus avoid conditions.")
    if run_ml:
        dm = ml_p_now - ml_base_now
        mv = "Buy-friendly" if dm >= margin else ("Avoid" if dm <= -margin else "Neutral")
        c2.metric(f"ML chance silver is higher in {h} months ({ml_choice})", f"{ml_p_now:.0%}",
                  f"{dm * 100:+.1f} pts vs normal odds ({ml_base_now:.0%}) → {mv}")
    rows = []
    for c in META:
        s_ = S_now[c]
        tag = "🟢 buy-friendly" if (c, s_) in good else ("🔴 avoid" if (c, s_) in bad else "⚪ neutral")
        val = f"{cur[c]:+.1%}" if META[c][3] else f"{cur[c]:.2f}"
        rows.append({"Indicator": META[c][0], "Reading": val, "Condition": state_label(c, s_), "Effect on silver": tag})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown(
        f"**How to use this:** a {v_now.lower()} reading tilts the odds only slightly. Use it as a *background filter*: "
        "be more willing to add silver when conditions are buy-friendly, and more patient when they say avoid. "
        "Check the out-of-sample results first, rebuild the analysis every few months, and size positions so being wrong is affordable.")
    st.caption("Latest values can lag a few days (FRED publishes with a delay). Not financial advice.")

# ------------------------------------------------------------------ machine learning
if run_ml:
    with T["ml"]:
        st.markdown(f"""
Each model estimates the **probability that silver is higher in {h} months** from the macro indicators.
Models are retrained every {step} months using past data only. **AUC** measures skill: 0.50 = coin flip, 0.55-0.60 = modest real skill,
above 0.65 would be suspicious in markets. A call is **Buy-friendly** when the probability is at least {margin * 100:.0f} points above normal odds, **Avoid** when that far below.
""")
        rows = []
        for name, wf in ml.items():
            if wf.empty:
                continue
            yy = (fwd.loc[wf.index] > 0).astype(int)
            auc = roc_auc_score(yy, wf["p"]) if yy.nunique() > 1 else np.nan
            acc, naive = ((wf["p"] > 0.5) == yy).mean(), ((wf["base"] > 0.5) == yy).mean()
            sm = summarize(fwd, ml_verdict(wf, margin), wf.index).set_index("Environment")
            ok = sm.loc["Buy-friendly", "Months"] >= 3 and sm.loc["Avoid", "Months"] >= 3
            sp = (sm.loc["Buy-friendly", "Avg"] - sm.loc["Avoid", "Avg"]) * 100 if ok else np.nan
            rows.append({"Model": name, "AUC": "-" if np.isnan(auc) else f"{auc:.2f}", "Direction accuracy": f"{acc:.0%}",
                         "Naive guess": f"{naive:.0%}", "Buy minus Avoid (return pts)": "-" if np.isnan(sp) else f"{sp:+.1f}"})
        st.subheader("Out-of-sample comparison")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        wf = ml[ml_choice]
        if wf.empty:
            st.warning("Not enough history to train this model. Move the start date earlier.")
        else:
            yy = (fwd.loc[wf.index] > 0).astype(int)
            auc = roc_auc_score(yy, wf["p"]) if yy.nunique() > 1 else np.nan
            if not np.isnan(auc):
                (st.success if auc >= 0.55 else st.warning)(
                    f"{ml_choice}: AUC {auc:.2f}. " + ("Some out-of-sample skill, but the test period is short." if auc >= 0.55
                                                     else "Barely better than a coin flip on unseen data. Don't rely on it."))
            mv_ser = ml_verdict(wf, margin)
            msum = summarize(fwd, mv_ser, wf.index)
            st.subheader(f"{ml_choice}: what silver did after each call (test period)")
            st.dataframe(fmt_summary(msum), hide_index=True, use_container_width=True)
            pf = go.Figure(go.Scatter(x=wf.index, y=wf["p"], mode="lines", name="P(silver up)"))
            pf.add_scatter(x=wf.index, y=wf["base"], mode="lines", name="Normal odds", line=dict(dash="dash"))
            pf.update_layout(title="Out-of-sample probability that silver is higher", yaxis_tickformat=".0%", height=340)
            st.plotly_chart(pf, use_container_width=True)
            st.subheader("What the model pays attention to (latest fit)")
            lab = ml_imp.rename(index={c: META[c][0] for c in ml_imp.index}).sort_values()
            st.bar_chart(lab)
            st.caption("Logistic regression: positive = pushes toward 'silver up', negative = toward 'down'. Tree models: bigger = more used.")

# ------------------------------------------------------------------ look-ahead scan
if auto_h:
    with T["scan"]:
        st.markdown(f"""
For each look-ahead, rules (and the ML model if enabled) are fitted on the **first 70% of the learning period** and scored on its **last 30%**.
**IC** is the rank correlation between the prediction and what silver actually did: 0 = no skill, about 0.10 = decent for markets.
The best average IC wins (**{h} months** here). The final test period was never used for this choice.
""")
        d = scan.copy()
        d["Look-ahead"] = d["h"].map(lambda x: f"{x} months" + (" ✅ chosen" if x == h else ""))
        for c in ["rules_ic", "ml_ic", "combined"]:
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else f"{v:+.2f}")
        d["Independent obs. (approx)"] = (scan["n_val"] / scan["h"]).round(0).astype(int)
        st.dataframe(d[["Look-ahead", "rules_ic", "ml_ic", "combined", "Independent obs. (approx)"]].rename(
            columns={"rules_ic": "Rules IC", "ml_ic": "ML IC", "combined": "Average IC"}), hide_index=True, use_container_width=True)
        st.caption("Longer look-aheads have far fewer independent observations, so their scores are noisier. "
                   "If every IC is near zero, there is no reliable horizon, and that is a valid answer.")
