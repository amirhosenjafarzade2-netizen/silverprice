"""
Silver Macro Environment Analyzer (Streamlit), v3.2

Which macro environments have been historically favorable / unfavorable for silver (and gold), what regime are we in,
what happened in comparable periods, and would following it have worked (with costs)?

requirements.txt: streamlit, yfinance, pandas, numpy, plotly, requests, scikit-learn
Optional: FRED_API_KEY in Streamlit secrets for reliable FRED access.
Educational only, not financial advice.

v3.2 changes:
- New tab: indicator ranking (which indicators predict best, in %, learn vs unseen data, by horizon).
- Backtest: 9 strategies (incl. gold rotation and trend filters), 3 signal sources, and a
  "find the best strategy" leaderboard with a first-half / second-half honesty check.
- New final tab: silver, gold and silver-vs-gold verdict for the current environment.
v3.1: parallel downloads; no progress callbacks inside cached functions.
"""
import io
import os
from concurrent.futures import ThreadPoolExecutor

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

FAV, NEU, UNF = "Favorable", "Neutral", "Unfavorable"
EXPO = {FAV: 1.0, NEU: 0.5, UNF: 0.0}
FRED = {"real_yield": "DFII10", "breakeven": "T10YIE", "fed_funds": "DFF", "curve": "T10Y2Y",
        "m2": "M2SL", "cpi": "CPIAUCSL", "indpro": "INDPRO", "nfci": "NFCI"}
YF = {"silver": "SI=F", "gold": "GC=F", "dollar": "DX-Y.NYB", "oil": "CL=F", "copper": "HG=F",
      "vix": "^VIX", "spx": "^GSPC", "tnx": "^TNX", "irx": "^IRX", "pl": "PL=F"}
HORIZONS = (1, 2, 3, 6, 9, 12)
ML_MODELS = ["Logistic regression", "Random Forest", "Gradient Boosting"]
TARGETS = {"Silver's return": "ret", "Silver minus gold (relative)": "gold", "Silver minus cash (T-bills)": "cash"}
PILLARS = ["Monetary", "Dollar", "Industrial", "Liquidity & risk", "Gold & valuation"]
STATES = ["Low", "Mid", "High"]

STRATEGIES = [
    "Scaled: 100% / 50% / 0%",
    "Defensive: 100% / 25% / 0%",
    "Aggressive: 100% / 75% / 25%",
    "Only when Favorable",
    "Hold unless Unfavorable",
    "Rotate: silver if Favorable, else gold",
    "Silver / gold / cash: Fav / Neutral / Unfav",
    "Trend only: silver above its 10-month average",
    "Macro + trend: not Unfavorable and above 10-month average",
]
NO_SIGNAL = {STRATEGIES[7]}  # strategies that ignore the macro signal
RANK_METRICS = ["Sharpe", "Sortino", "CAGR", "Calmar", "Max drawdown"]
NOW_KINDS = ("ret", "goldabs", "gold")
NOW_NAMES = {"ret": "Silver", "goldabs": "Gold", "gold": "Silver vs gold"}
NOW_PHRASE = {"ret": "silver is higher", "goldabs": "gold is higher", "gold": "silver beats gold"}
REL_TXT = {"Favorable": "Favors silver over gold", "Leaning favorable": "Leans toward silver",
           "Neutral": "No clear preference", "Leaning unfavorable": "Leans toward gold",
           "Unfavorable": "Favors gold over silver"}

# indicator -> (title, label when LOW, label when HIGH, is a % change?, pillar)
META_ALL = {
    "real_yield":     ("Real 10y interest rate (level)", "Low / negative real rates", "High real rates", False, "Monetary"),
    "real_yield_chg": ("Real rate, 3-month change", "Real rates falling", "Real rates rising", False, "Monetary"),
    "breakeven":      ("Inflation expectations (10y)", "Low inflation expectations", "High inflation expectations", False, "Monetary"),
    "fed_chg":        ("Fed funds rate, 6-month change", "Fed cutting", "Fed hiking", False, "Monetary"),
    "curve":          ("Yield curve (10y minus 2y)", "Flat / inverted curve", "Steep curve", False, "Monetary"),
    "cpi_yoy":        ("Inflation (CPI, year-on-year %)", "Low inflation", "High inflation", False, "Monetary"),
    "real_fed":       ("Real policy rate (Fed funds minus CPI)", "Low real policy rate", "High real policy rate", False, "Monetary"),
    "dollar_mom":     ("US Dollar, 3-month trend", "Dollar weakening", "Dollar strengthening", True, "Dollar"),
    "oil_mom":        ("Oil, 3-month trend", "Oil falling", "Oil rising", True, "Industrial"),
    "copper_mom":     ("Copper, 3-month trend", "Copper falling", "Copper rising", True, "Industrial"),
    "indpro_yoy":     ("Industrial production, YoY %", "Weak industry", "Strong industry", False, "Industrial"),
    "cu_au":          ("Copper/gold ratio, 3-month change", "Falling (fear / slowdown)", "Rising (growth)", True, "Industrial"),
    "plat_mom":       ("Platinum, 3-month trend", "Platinum falling", "Platinum rising", True, "Industrial"),
    "m2_yoy":         ("Money supply (M2) growth, YoY %", "Slow money growth", "Fast money growth", False, "Liquidity & risk"),
    "nfci":           ("Financial conditions (NFCI)", "Loose conditions", "Tight conditions", False, "Liquidity & risk"),
    "vix":            ("Market fear (VIX)", "Calm markets", "Fearful markets", False, "Liquidity & risk"),
    "spx_mom":        ("Stocks (S&P 500), 3-month trend", "Stocks falling", "Stocks rising", True, "Liquidity & risk"),
    "gs_ratio":       ("Gold/silver ratio", "Silver expensive vs gold", "Silver cheap vs gold", False, "Gold & valuation"),
    "gold_mom":       ("Gold, 3-month trend", "Gold falling", "Gold rising", True, "Gold & valuation"),
    "sg_mom":         ("Silver vs gold, 3-month relative strength", "Silver lagging gold", "Silver outperforming gold", True, "Gold & valuation"),
    "val_real":       ("Silver real-price percentile (vs own history)", "Cheap vs history", "Expensive vs history", False, "Gold & valuation"),
    "nom_yield":      ("10y Treasury yield (level)", "Low yields", "High yields", False, "Monetary"),
    "nom_yield_chg":  ("10y Treasury yield, 3-month change", "Yields falling", "Yields rising", False, "Monetary"),
}
CORE = ["real_yield", "real_yield_chg", "breakeven", "fed_chg", "curve", "dollar_mom", "oil_mom", "copper_mom",
        "vix", "spx_mom", "gs_ratio", "m2_yoy", "nfci", "sg_mom", "val_real"]
META = dict(META_ALL)  # narrowed to indicators in use once data loads
GROWTH, INFL = ["copper_mom", "indpro_yoy", "spx_mom"], ["cpi_yoy", "breakeven", "oil_mom"]
REGIMES = ["Goldilocks (growth↑ inflation↓)", "Reflation (growth↑ inflation↑)",
           "Stagflation (growth↓ inflation↑)", "Slowdown / deflation (growth↓ inflation↓)"]


def show_df(box, d):
    """Works on both old and new Streamlit (use_container_width was replaced by width='stretch')."""
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
    """Bordered container on new Streamlit, plain container on old versions."""
    try:
        return box.container(border=True)
    except TypeError:
        return box.container()


def state_label(col, state):
    return {"Low": META_ALL[col][1], "Mid": "Neutral", "High": META_ALL[col][2]}[state]


# ------------------------------------------------------------------ data
def get_fred_key():
    key = None
    try:
        key = st.secrets.get("FRED_API_KEY")
    except Exception:  # noqa: BLE001
        pass
    return key or os.environ.get("FRED_API_KEY")


def _fred_fetch(series_id, start, key):
    """Plain function (no Streamlit calls) so it is safe inside worker threads."""
    last = None
    for _ in range(2):
        try:
            if key:
                r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                                 params=dict(series_id=series_id, api_key=key, file_type="json",
                                             observation_start=start), timeout=(5, 20))
                r.raise_for_status()
                d = pd.DataFrame(r.json()["observations"])[["date", "value"]]
                d["value"] = pd.to_numeric(d["value"], errors="coerce")
            else:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
                r = requests.get(url, timeout=(5, 20), headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                d = pd.read_csv(io.StringIO(r.text), na_values=[".", ""])
                d.columns = ["date", "value"]
            d["date"] = pd.to_datetime(d["date"])
            return d.set_index("date")["value"].astype(float)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def _yahoo_fetch(start):
    px = yf.download(list(YF.values()), start=start, auto_adjust=True, progress=False, threads=True)["Close"]
    return px.rename(columns={v: k for k, v in YF.items()})


def month_end(df):
    try:
        return df.resample("ME").last()
    except ValueError:
        return df.resample("M").last()


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_raw(pre, key):
    """Yahoo + all FRED series downloaded in parallel. Exceptions from Yahoo are not cached."""
    with ThreadPoolExecutor(max_workers=len(FRED) + 1) as ex:
        fy = ex.submit(_yahoo_fetch, pre)
        ff = {k: ex.submit(_fred_fetch, v, pre, key) for k, v in FRED.items()}
        px = fy.result()
        fr, failed = {}, []
        for k, f in ff.items():
            try:
                fr[k] = f.result()
            except Exception:  # noqa: BLE001
                failed.append(k)
    return px, fr, failed


def load_monthly(start, cb=None):
    cb = cb or (lambda f, t: None)
    pre = (pd.Timestamp(start) - pd.DateOffset(years=10)).strftime("%Y-%m-%d")  # warm-up history for rolling stats
    cb(0.05, "Downloading market + macro data in parallel (first run only, then cached)...")
    px, fr, failed = load_raw(pre, get_fred_key())
    if failed or px.empty:
        load_raw.clear()  # don't keep a partial result for 6 hours; retry next run
    if "real_yield" in failed:  # FRED host is down: fall back to Yahoo stand-ins
        fr, failed = {}, list(FRED)

    def lag(s, days):
        return s.set_axis(s.index + pd.DateOffset(months=1) + pd.Timedelta(days=days))

    for k in ("m2", "cpi", "indpro"):  # monthly series: only usable after publication
        if k in fr:
            raw = fr.pop(k)
            fr[k + "_yoy"] = lag(raw.pct_change(12) * 100, 20).dropna()
            if k == "cpi":
                fr["cpi_level"] = lag(raw, 20)
    if "nfci" in fr:  # weekly, published with a short delay
        fr["nfci"] = fr["nfci"].set_axis(fr["nfci"].index + pd.Timedelta(days=7))
    cb(0.35, "Preparing monthly data...")
    daily = px.join(pd.concat(fr, axis=1), how="outer") if fr else px
    daily = daily.sort_index().ffill().dropna(subset=["silver"])
    return month_end(daily).dropna(subset=["silver"]), sorted(set(failed))


def exp_pctl(s, minp=36):
    return s.expanding(min_periods=minp).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan, raw=True)


def build_features(m):
    def g(k):
        return m[k] if k in m else pd.Series(np.nan, index=m.index)

    F = pd.DataFrame(index=m.index)
    F["real_yield"] = g("real_yield")
    F["real_yield_chg"] = g("real_yield").diff(3)
    F["breakeven"] = g("breakeven")
    F["fed_chg"] = g("fed_funds").diff(6)
    F["curve"] = g("curve")
    F["cpi_yoy"] = g("cpi_yoy")
    F["real_fed"] = g("fed_funds") - g("cpi_yoy")
    F["nom_yield"] = g("tnx")
    F["nom_yield_chg"] = g("tnx").diff(3)
    if F["fed_chg"].isna().all():
        F["fed_chg"] = g("irx").diff(6)
    if F["curve"].isna().all():
        F["curve"] = g("tnx") - g("irx")
    F["dollar_mom"] = g("dollar").pct_change(3)
    F["oil_mom"] = g("oil").pct_change(3).clip(-0.8, 1.5)
    F["copper_mom"] = g("copper").pct_change(3)
    F["indpro_yoy"] = g("indpro_yoy")
    F["cu_au"] = (g("copper") / g("gold")).pct_change(3)
    F["plat_mom"] = g("pl").pct_change(3)
    F["m2_yoy"] = g("m2_yoy")
    F["nfci"] = g("nfci")
    F["vix"] = g("vix")
    F["spx_mom"] = g("spx").pct_change(3)
    F["gs_ratio"] = g("gold") / g("silver")
    F["gold_mom"] = g("gold").pct_change(3)
    F["sg_mom"] = (g("silver") / g("gold")).pct_change(3)
    cpi = g("cpi_level")
    F["val_real"] = exp_pctl(g("silver") / cpi if cpi.notna().any() else g("silver"))
    keep = [c for c in META_ALL if F[c].notna().any()]
    if "real_yield" in keep:
        keep = [c for c in keep if not c.startswith("nom_")]
    return F[keep]


# ------------------------------------------------------------------ statistics
def assign_states(F, learn_idx):
    S = pd.DataFrame(index=F.index, columns=F.columns, dtype=object)
    for c in F.columns:
        lo, hi = F.loc[learn_idx, c].quantile([1 / 3, 2 / 3])
        col = F[c]
        lab = np.where(col <= lo, "Low", np.where(col <= hi, "Mid", "High")).astype(object)
        S[c] = pd.Series(lab, index=F.index).where(col.notna())
    return S


def nw_t(y, d, lags):
    """Newey-West (HAC) t-stat of b in y = a + b*d + e. Valid for overlapping returns."""
    y, d = np.asarray(y, float), np.asarray(d, float)
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    XtXi = np.linalg.pinv(X.T @ X)
    e = y - X @ (XtXi @ X.T @ y)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, min(lags, n - 1) + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (lags + 1)) * (G + G.T)
    se = np.sqrt((XtXi @ S @ XtXi)[1, 1])
    return float((XtXi @ X.T @ y)[1] / se) if se > 0 else 0.0


def block_ci(x, block, B=1000, seed=0):
    """Approximate 95% CI of the mean via moving-block bootstrap."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def half_edge(fwd, S, col, state, idx):
    r = fwd.loc[idx][S.loc[idx, col] == state]
    return r.mean() - fwd.loc[idx].mean() if len(r) >= 4 else np.nan


def bucket_stats(S, fwd, idx, h):
    base = fwd.loc[idx]
    yv = base.values
    mid = len(idx) // 2
    first, second = idx[:mid], idx[mid:]
    rows = []
    for c in S.columns:
        sv = S.loc[idx, c].values
        for s_ in STATES:
            mask = sv == s_
            n = int(mask.sum())
            if n < 8:
                continue
            r = base[mask]
            edge = r.mean() - base.mean()
            t = nw_t(yv, mask.astype(float), h)  # HAC: robust to overlapping windows
            e1, e2 = half_edge(fwd, S, c, s_, first), half_edge(fwd, S, c, s_, second)
            cons = bool(np.sign(e1) == np.sign(e2) == np.sign(edge)) if not (np.isnan(e1) or np.isnan(e2)) else False
            rows.append(dict(col=c, state=s_, n=n, avg=r.mean(), median=r.median(), win=(r > 0).mean(),
                             edge=edge, t=t, consistent=cons))
    return pd.DataFrame(rows)


def build_rules(stats, t_thr, need_consistent):
    ok = stats[stats["t"].abs() >= t_thr]
    if need_consistent:
        ok = ok[ok["consistent"]]
    return ({(r.col, r.state) for r in ok.itertuples() if r.edge > 0},
            {(r.col, r.state) for r in ok.itertuples() if r.edge < 0})


def score_series(S, good, bad):
    score = pd.Series(0.0, index=S.index)
    for c in S.columns:
        for s_ in STATES:
            if (c, s_) in good:
                score += (S[c] == s_).astype(float)
            if (c, s_) in bad:
                score -= (S[c] == s_).astype(float)
    return score


def to_verdict(score, lo_thr, hi_thr):
    v = pd.Series(NEU, index=score.index)
    v[(score >= hi_thr) & (score > 0)] = FAV
    v[(score <= lo_thr) & (score < 0)] = UNF
    return v


def summarize(fwd, groups, idx, h, fdd=None, order=(FAV, NEU, UNF)):
    rows = []
    for g_ in list(order) + ["All months"]:
        r = fwd.loc[idx] if g_ == "All months" else fwd.loc[idx].where(groups.reindex(idx) == g_).dropna()
        row = dict(Group=g_, Months=len(r), Avg=np.nan, Median=np.nan, Win=np.nan, Worst=np.nan, Best=np.nan,
                   lo=np.nan, hi=np.nan, DD=np.nan)
        if len(r):
            row.update(Avg=r.mean(), Median=r.median(), Win=(r > 0).mean(), Worst=r.min(), Best=r.max())
            row["lo"], row["hi"] = block_ci(r.values, h)
            if fdd is not None:
                row["DD"] = fdd.reindex(r.index).median()
        rows.append(row)
    return pd.DataFrame(rows)


def pc(v, plus=True):
    return "-" if pd.isna(v) else (f"{v:+.1%}" if plus else f"{v:.0%}")


def fm(metric, v):
    """Format a performance metric by name."""
    if pd.isna(v):
        return "-"
    return f"{v:.2f}" if metric in ("Sharpe", "Sortino", "Calmar") else pc(v)


def fmt_summary(d, dd=False):
    o = pd.DataFrame({"Environment": d["Group"], "Months": d["Months"], "Avg after": d["Avg"].map(pc),
                      "95% CI": [("-" if pd.isna(a) else f"{a:+.0%} to {b:+.0%}") for a, b in zip(d["lo"], d["hi"])],
                      "Median": d["Median"].map(pc), "% positive": d["Win"].map(lambda v: pc(v, False)),
                      "Worst": d["Worst"].map(pc), "Best": d["Best"].map(pc)})
    if dd:
        o["Typical max drawdown"] = d["DD"].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
    return o


# ------------------------------------------------------------------ targets, splits, rules
def cash_rate(m):
    return (m["irx"] / 100 / 12).fillna(0) if "irx" in m else pd.Series(0.0, index=m.index)


def target_returns(m, idx, h, kind):
    """kind: ret = silver return, gold = silver minus gold, cash = silver minus T-bills, goldabs = gold return."""
    if kind == "goldabs":
        return (m["gold"].shift(-h) / m["gold"] - 1).reindex(idx)
    s = m["silver"].shift(-h) / m["silver"] - 1
    if kind == "gold":
        s = s - (m["gold"].shift(-h) / m["gold"] - 1)
    elif kind == "cash":
        s = s - (m["irx"].fillna(0) / 100 * h / 12 if "irx" in m else 0)
    return s.reindex(idx)


def fwd_drawdown(s, h):
    v, out = s.values, np.full(len(s), np.nan)
    for i in range(len(v) - h):
        w = v[i: i + h + 1]
        out[i] = (w / np.maximum.accumulate(w) - 1).min()
    return pd.Series(out, index=s.index)


def split_idx(F_ok, fwd, h, learn_frac):
    ev = F_ok.index.intersection(fwd.dropna().index)
    cut = int(len(ev) * learn_frac)
    return ev, cut, ev[: max(cut - h, 0)], ev[cut:]  # purge h months: no learn/test overlap


def run_rules(F_ok, fwd, learn_idx, h, t_thr, need_consistent):
    S = assign_states(F_ok, learn_idx)
    stats = bucket_stats(S, fwd, learn_idx, h)
    good, bad = build_rules(stats, t_thr, need_consistent)
    score = score_series(S, good, bad)
    lo, hi = np.quantile(score.loc[learn_idx], 0.25), np.quantile(score.loc[learn_idx], 0.75)
    return S, stats, good, bad, score, to_verdict(score, lo, hi)


def rank_ic(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 8 or d.iloc[:, 0].nunique() < 2:
        return np.nan
    return d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())


# ------------------------------------------------------------------ indicator ranking
def hl_spread(fwd, S, c, idx):
    """Average outcome when the indicator is HIGH minus when it is LOW, in percentage points."""
    r, s = fwd.loc[idx], S.loc[idx, c]
    hi, lo = r[s == "High"], r[s == "Low"]
    return (hi.mean() - lo.mean()) * 100 if len(hi) >= 4 and len(lo) >= 4 else np.nan


def indicator_ranking(F_ok, fwd, S, learn_idx, test_idx, h):
    """Rank-correlation of each indicator with the future outcome, on learn and on unseen (test) data."""
    rows = []
    for c in F_ok.columns:
        xl, yl = F_ok.loc[learn_idx, c], fwd.loc[learn_idx]
        xt, yt = F_ok.loc[test_idx, c], fwd.loc[test_idx]
        ic_l, ic_t = rank_ic(xl, yl), rank_ic(xt, yt)
        t_l = nw_t(yl.rank().values, xl.rank().values, h) if len(xl) > 8 else np.nan
        hit = np.nan
        if pd.notna(ic_l) and ic_l != 0 and len(xt) >= 8:
            sx = np.sign(xt - xl.median()) * np.sign(ic_l)
            sy = np.sign(yt - yl.median())
            ok = (sx != 0) & (sy != 0)
            hit = float((sx[ok] == sy[ok]).mean()) if ok.sum() >= 8 else np.nan
        same = bool(pd.notna(ic_l) and pd.notna(ic_t) and np.sign(ic_l) == np.sign(ic_t))
        reliab = min(abs(ic_l), abs(ic_t)) * 100 if same else 0.0
        if pd.isna(ic_l) or pd.isna(ic_t):
            status = "-"
        elif not same:
            status = "❌ Flips on unseen data"
        elif abs(t_l) >= 2:
            status = "✅ Reliable"
        else:
            status = "🟡 Weak but consistent"
        rows.append(dict(col=c, ic_l=ic_l, ic_t=ic_t, t_l=t_l, hit=hit, reliab=reliab, status=status,
                         hl_l=hl_spread(fwd, S, c, learn_idx), hl_t=hl_spread(fwd, S, c, test_idx)))
    d = pd.DataFrame(rows)
    d["absl"] = d["ic_l"].abs()
    return d.sort_values(["reliab", "absl"], ascending=False, na_position="last").drop(columns="absl").reset_index(drop=True)


def ic_by_horizon(m, F_ok, target, learn_frac):
    """Rank correlation (learn period only) of each indicator with the outcome at every look-ahead."""
    out = {}
    for h_ in HORIZONS:
        fwd_ = target_returns(m, F_ok.index, h_, target)
        _, _, learn_, _ = split_idx(F_ok, fwd_, h_, learn_frac)
        out[h_] = {c: rank_ic(F_ok.loc[learn_, c], fwd_.loc[learn_]) for c in F_ok.columns}
    return pd.DataFrame(out)


# ------------------------------------------------------------------ machine learning
def make_model(name):
    if name.startswith("Logistic"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    if name.startswith("Random"):
        return RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=10, n_jobs=-1, random_state=0)
    return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)


@st.cache_data(show_spinner=False)
def walk_forward(Fev, fwdev, h, cut, step, model_name):
    """No Streamlit calls in here (progress is reported by the caller)."""
    y, X = (fwdev > 0).astype(int).values, Fev.values
    starts = list(range(cut, len(Fev), step))
    out = []
    for s0 in starts:
        tr_end, te = s0 - h, slice(s0, min(s0 + step, len(Fev)))
        if tr_end < 36:
            continue
        ytr = y[:tr_end]
        if len(np.unique(ytr)) < 2:
            p = np.full(te.stop - te.start, ytr.mean())
        else:
            p = make_model(model_name).fit(X[:tr_end], ytr).predict_proba(X[te])[:, 1]
        out.append(pd.DataFrame({"p": p, "base": ytr.mean()}, index=Fev.index[te]))
    return pd.concat(out) if out else pd.DataFrame(columns=["p", "base"])


@st.cache_data(show_spinner=False)
def fit_today(Fev, fwdev, x_now, model_name):
    y = (fwdev > 0).astype(int).values
    mdl = make_model(model_name).fit(Fev.values, y)
    p = float(mdl.predict_proba(x_now.values.reshape(1, -1))[0, 1])
    contrib = None
    if hasattr(mdl, "steps"):  # logistic: exact additive contributions in log-odds (what SHAP gives for linear models)
        sc, lr = mdl[0], mdl[-1]
        imp = pd.Series(lr.coef_[0], index=Fev.columns)
        contrib = pd.Series(lr.coef_[0] * (x_now.values - sc.mean_) / sc.scale_, index=Fev.columns)
    else:
        imp = pd.Series(mdl.feature_importances_, index=Fev.columns)
    return p, float(y.mean()), imp, contrib


def ml_verdict(wf, margin):
    d = wf["p"] - wf["base"]
    v = pd.Series(NEU, index=wf.index)
    v[d >= margin] = FAV
    v[d <= -margin] = UNF
    return v


def prob_metrics(wf, y):
    p, b = wf["p"].clip(0.01, 0.99), wf["base"].clip(0.01, 0.99)
    brier, brier_b = ((p - y) ** 2).mean(), ((b - y) ** 2).mean()
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    return brier, 1 - brier / brier_b, ll


@st.cache_data(show_spinner=False)
def scan_horizons(m, F_ok, target, model_name, learn_frac, t_thr, need_consistent, horizons):
    """Choose the look-ahead using ONLY the learn period (fit on first 70%, score last 30%).
    No Streamlit calls in here (progress is reported by the caller)."""
    def ics(pred, ret, V):
        half = len(V) // 2
        full, a, b = rank_ic(pred.loc[V], ret.loc[V]), rank_ic(pred.loc[V[:half]], ret.loc[V[:half]]), rank_ic(pred.loc[V[half:]], ret.loc[V[half:]])
        return full, (min(a, b) if not (np.isnan(a) or np.isnan(b)) else np.nan)

    rows = []
    for h_ in horizons:
        fwd_ = target_returns(m, F_ok.index, h_, target)
        _, _, learn_, _ = split_idx(F_ok, fwd_, h_, learn_frac)
        k = int(len(learn_) * 0.7)
        A, V = learn_[: max(k - h_, 0)], learn_[k:]
        row = dict(h=h_, rules_ic=np.nan, rules_stab=np.nan, ml_ic=np.nan, ml_stab=np.nan, n_val=len(V))
        if len(A) >= 36 and len(V) >= 10:
            S = assign_states(F_ok, A)
            g_, b_ = build_rules(bucket_stats(S, fwd_, A, h_), t_thr, need_consistent)
            row["rules_ic"], row["rules_stab"] = ics(score_series(S, g_, b_), fwd_, V)
            yA = (fwd_.loc[A] > 0).astype(int)
            if model_name and yA.nunique() > 1:
                mdl = make_model(model_name).fit(F_ok.loc[A].values, yA.values)
                p = pd.Series(mdl.predict_proba(F_ok.loc[V].values)[:, 1], index=V)
                row["ml_ic"], row["ml_stab"] = ics(p, fwd_, V)
        rows.append(row)
    d = pd.DataFrame(rows)
    d["combined"] = d[["rules_ic", "ml_ic"]].mean(axis=1, skipna=True)
    d["stable"] = d[["rules_stab", "ml_stab"]].mean(axis=1, skipna=True)
    return d


# ------------------------------------------------------------------ regimes, analogues, persistence
def classify_regime(Fa, idx, learn_idx):
    def zmean(cols):
        cols = [c for c in cols if c in Fa and Fa.loc[learn_idx, c].notna().sum() > 20]
        if not cols:
            return None
        return pd.concat([(Fa.loc[idx, c] - Fa.loc[learn_idx, c].mean()) / Fa.loc[learn_idx, c].std() for c in cols], axis=1).mean(axis=1)

    g, i = zmean(GROWTH), zmean(INFL)
    if g is None or i is None:
        return None
    names = np.where(g > 0, np.where(i > 0, REGIMES[1], REGIMES[0]), np.where(i > 0, REGIMES[2], REGIMES[3]))
    return pd.Series(names, index=idx).where(g.notna() & i.notna())


def analogues(F_ok, fwd, ev, x_now, h, k=10):
    Fe = F_ok.loc[ev]
    mu, sd = Fe.mean(), Fe.std().replace(0, 1)
    d = np.sqrt((((Fe - mu) / sd - (x_now - mu) / sd) ** 2).sum(axis=1))
    chosen = []
    for t in d.sort_values().index:  # keep analogues at least h months apart (avoid near-duplicates)
        if all(abs((t.year - c.year) * 12 + t.month - c.month) >= max(h, 3) for c in chosen):
            chosen.append(t)
        if len(chosen) == k:
            break
    out = pd.DataFrame({"Date": [c.strftime("%b %Y") for c in chosen],
                        "Closeness": [100 * np.exp(-d[c] / d.median()) for c in chosen],
                        "After": [fwd.loc[c] for c in chosen]})
    return out


def streak_info(v, fwd, ev):
    grp = (v != v.shift()).cumsum()
    run = v.groupby(grp).cumcount() + 1
    cur = v.iloc[-1]
    lens = pd.DataFrame({"v": v, "g": grp}).groupby("g").agg(v=("v", "first"), n=("v", "size"))
    med = lens[lens["v"] == cur]["n"].median()
    mask = (v == cur) & (run >= 3) & v.index.isin(ev)
    r = fwd.reindex(v.index)[mask].dropna()
    return cur, int(run.iloc[-1]), med, len(r), (r.mean() if len(r) else np.nan)


# ------------------------------------------------------------------ current verdict for silver / gold
def exposure_label(e):
    """Average of the rules and ML views (each 0 / 0.5 / 1) -> five-step label."""
    if e >= 0.99:
        return "Favorable", "🟢"
    if e >= 0.7:
        return "Leaning favorable", "🟡"
    if e > 0.3:
        return "Neutral", "⚪"
    if e > 0.01:
        return "Leaning unfavorable", "🟠"
    return "Unfavorable", "🔴"


@st.cache_data(show_spinner=False)
def analyse_target(kind, m, F_ok, h, learn_frac, t_thr, need_consistent, run_ml, ml_choice, step, margin):
    """Full rules + ML read of the CURRENT month for one target. No Streamlit calls inside."""
    fwd_ = target_returns(m, F_ok.index, h, kind)
    ev_, cut_, learn_, test_ = split_idx(F_ok, fwd_, h, learn_frac)
    S_, stats_, good_, bad_, score_, verdict_ = run_rules(F_ok, fwd_, learn_, h, t_thr, need_consistent)
    now_ = F_ok.index[-1]
    v_r, S_now = verdict_.loc[now_], S_.loc[now_]
    res = dict(rules=v_r, score=int(score_.loc[now_]),
               why_good=[state_label(c, s_) for c, s_ in sorted(good_) if S_now[c] == s_],
               why_bad=[state_label(c, s_) for c, s_ in sorted(bad_) if S_now[c] == s_])

    # out-of-sample evidence for the rules
    sp, ev_level = np.nan, "Unknown"
    ev_text = "Not enough Favorable and Unfavorable months in the unseen period to judge."
    if not (good_ or bad_):
        ev_level, ev_text = "None", "No environment passed the strength filter, so the rules have no opinion."
    else:
        ts = summarize(fwd_, verdict_, test_, h).set_index("Group")
        if ts.loc[FAV, "Months"] >= 3 and ts.loc[UNF, "Months"] >= 3:
            sp = (ts.loc[FAV, "Avg"] - ts.loc[UNF, "Avg"]) * 100
            lo_f, hi_u = ts.loc[FAV, "lo"], ts.loc[UNF, "hi"]
            overlap = bool(pd.isna(lo_f) or pd.isna(hi_u) or lo_f <= hi_u)
            if sp > 2 and not overlap:
                ev_level, ev_text = "Strong", f"Unseen data: Favorable beat Unfavorable by {sp:.1f} pts, confidence ranges do not overlap."
            elif sp > 0:
                ev_level, ev_text = "Weak", f"Unseen data: Favorable beat Unfavorable by {sp:.1f} pts, but the ranges overlap (could be noise)."
            else:
                ev_level, ev_text = "None", f"Unseen data: Favorable did NOT beat Unfavorable ({sp:+.1f} pts)."
    res.update(ev_level=ev_level, ev_text=ev_text, spread=sp)

    # machine learning
    ml_info = None
    if run_ml:
        Fev_, fev_ = F_ok.loc[ev_], fwd_.loc[ev_]
        wf_ = walk_forward(Fev_, fev_, h, cut_, step, ml_choice)
        p_, base_, _, _ = fit_today(Fev_, fev_, F_ok.iloc[-1], ml_choice)
        dm = p_ - base_
        auc = bss = np.nan
        if not wf_.empty:
            yy = (fwd_.loc[wf_.index] > 0).astype(int)
            if yy.nunique() > 1:
                auc = roc_auc_score(yy, wf_["p"])
                _, bss, _ = prob_metrics(wf_, yy)
        ml_info = dict(p=p_, base=base_, verdict=FAV if dm >= margin else (UNF if dm <= -margin else NEU), auc=auc, bss=bss)
    res["ml"] = ml_info

    e = float(np.mean([EXPO[v_r]] + ([EXPO[ml_info["verdict"]]] if ml_info else [])))
    res["e"] = e
    res["label"], res["icon"] = exposure_label(e)
    a = analogues(F_ok, fwd_, ev_, F_ok.iloc[-1], h)["After"]
    res["ana_med"], res["ana_pos"] = float(a.median()), float((a > 0).mean())
    return res


# ------------------------------------------------------------------ backtest
def strategy_weights(name, v, trend):
    """Weights (silver, gold, cash) held during the month AFTER each signal month."""
    z = pd.Series(0.0, index=v.index)
    s, g = z.copy(), z.copy()
    fav, unf = v == FAV, v == UNF
    if name.startswith("Scaled"):
        s = v.map(EXPO).astype(float)
    elif name.startswith("Defensive"):
        s = v.map({FAV: 1.0, NEU: 0.25, UNF: 0.0}).astype(float)
    elif name.startswith("Aggressive"):
        s = v.map({FAV: 1.0, NEU: 0.75, UNF: 0.25}).astype(float)
    elif name.startswith("Only when"):
        s = fav.astype(float)
    elif name.startswith("Hold unless"):
        s = (~unf).astype(float)
    elif name.startswith("Rotate"):
        s, g = fav.astype(float), (~fav).astype(float)
    elif name.startswith("Silver / gold"):
        s, g = fav.astype(float), (v == NEU).astype(float)
    elif name.startswith("Trend only"):
        s = trend.astype(float)
    elif name.startswith("Macro + trend"):
        s = ((~unf) & trend).astype(float)
    return pd.DataFrame({"silver": s, "gold": g, "cash": 1.0 - s - g})


def _contrib(w, r):
    return pd.Series(np.where(w.values == 0, 0.0, w.values * r.values), index=w.index)


def run_backtest_w(m, W, bps):
    """Signal at month-end t sets the position held during month t -> t+1. Costs charged on turnover."""
    rs = m["silver"].pct_change().shift(-1).reindex(W.index)
    rg = m["gold"].pct_change().shift(-1).reindex(W.index)
    rf = cash_rate(m).reindex(W.index)
    turn = W["silver"].diff().abs() + W["gold"].diff().abs()
    turn.iloc[0] = W["silver"].iloc[0] + W["gold"].iloc[0]
    ret = _contrib(W["silver"], rs) + _contrib(W["gold"], rg) + _contrib(W["cash"], rf) - turn * bps / 1e4
    return ret.dropna(), turn


def perf(r, rf, invested=np.nan, trades=np.nan):
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "Max drawdown"]
    n = len(r)
    if n < 2:
        out = dict.fromkeys(keys, np.nan)
        out.update({"% invested": invested, "Trades": trades})
        return out
    eq = (1 + r).cumprod()
    ex = r - rf.reindex(r.index).fillna(0)
    vol = r.std() * np.sqrt(12)
    dn = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(12)
    cagr = eq.iloc[-1] ** (12 / n) - 1
    mdd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Volatility": vol,
            "Sharpe": ex.mean() * 12 / vol if vol > 0 else np.nan,
            "Sortino": ex.mean() * 12 / dn if dn > 0 else np.nan,
            "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "Max drawdown": mdd, "% invested": invested, "Trades": trades}


def bench_curves(m, idx):
    out = {}
    for name, ser in {"Silver buy & hold": m["silver"], "Gold buy & hold": m["gold"], "S&P 500": m["spx"]}.items():
        out[name] = ser.pct_change().shift(-1).reindex(idx).dropna()
    out["Cash (T-bills)"] = cash_rate(m).reindex(idx)
    return out


def perf_table(rows):
    return pd.DataFrame([{"Strategy": n, "CAGR": pc(p["CAGR"]), "Volatility": pc(p["Volatility"], False),
                          "Sharpe": fm("Sharpe", p["Sharpe"]), "Sortino": fm("Sortino", p["Sortino"]),
                          "Calmar": fm("Calmar", p["Calmar"]), "Max drawdown": pc(p["Max drawdown"]),
                          "% invested": pc(p["% invested"], False), "Trades": p["Trades"]} for n, p in rows])


def growth_chart(curves, common, title):
    f = go.Figure()
    for name, r in curves.items():
        r = r.reindex(common).dropna()
        f.add_scatter(x=r.index, y=(1 + r).cumprod(), name=name)
    f.update_layout(title=title, height=400)
    return f


# ================================================================== UI
with st.sidebar:
    st.header("Settings")
    start = st.text_input("Data start date", "2004-01-01")
    target_name = st.selectbox("What to predict", list(TARGETS), help="Relative targets ask whether silver BEATS gold / cash.")
    target = TARGETS[target_name]
    auto_h = st.checkbox("🔍 Auto-find best look-ahead", False,
                         help="Tests 1-12 month look-aheads on the learning period only and prefers stable ones.")
    h_manual = st.select_slider("Look-ahead period (months)", options=list(HORIZONS), value=6, disabled=auto_h)
    learn_frac = st.slider("Share of history used to LEARN the rules", 0.4, 0.8, 0.6, 0.05)
    t_thr = st.slider("Minimum strength (HAC t-stat)", 0.5, 2.5, 1.0, 0.1)
    need_consistent = st.checkbox("Require the effect in both halves of the learn period", True)
    st.subheader("Machine learning")
    run_ml = st.checkbox("Also run machine-learning models", True)
    ml_choice = st.selectbox("Model shown in detail", ML_MODELS, disabled=not run_ml)
    margin = st.slider("Confidence margin (points vs normal odds)", 0.02, 0.20, 0.05, 0.01, disabled=not run_ml)
    step = st.select_slider("Retrain every (months)", options=[3, 6, 12], value=6, disabled=not run_ml)
    st.subheader("Backtest")
    mode = st.selectbox("Strategy", STRATEGIES)
    bps = st.slider("Trading cost (basis points per 100% traded)", 0, 100, 15, 5)
    find_best = st.checkbox("🏁 Find the best strategy", False,
                            help="Tests every strategy on every signal over the unseen test period and ranks them.")
    rank_by = st.selectbox("Rank strategies by", RANK_METRICS, disabled=not find_best)

st.title("🥈 Silver Macro Environment Analyzer")
st.caption("Historically favorable or unfavorable macro environments for silver and gold. Not a buy/sell signal, not financial advice.")
bar = st.progress(0.0, text="Starting...")


def cb(frac, text):
    bar.progress(float(min(max(frac, 0.0), 1.0)), text=text)


try:
    m, failed = load_monthly(start, cb)
except Exception as e:
    bar.empty()
    st.error(f"Could not download market data: {e}")
    st.stop()

F_all = build_features(m).loc[start:]
available = list(F_all.columns)
default = [c for c in available if c in CORE or c.startswith("nom_")] or available
with st.sidebar:
    st.subheader("Indicators")
    chosen = st.multiselect("Indicators to use", available, default=default, format_func=lambda c: META_ALL[c][0],
                            help="Core set is on by default. More indicators means more chances for a fluke.")
if len(chosen) < 3:
    bar.empty()
    st.error("Pick at least 3 indicators.")
    st.stop()
META = {c: META_ALL[c] for c in chosen}
F_ok = F_all[chosen].dropna()

if "real_yield" in failed:
    st.warning("FRED did not respond, so the app is using Yahoo stand-ins (10y yield, 3-month T-bill). Real rates, inflation, M2, "
               "industrial production and financial conditions are missing. Reload later, or add a free FRED_API_KEY in Secrets.")
elif failed:
    st.info("Some FRED series were unavailable and skipped: " + ", ".join(failed))

scan = None
if auto_h:
    cb(0.35, "Scanning look-ahead periods (cached after the first run)...")
    scan = scan_horizons(m, F_ok, target, ml_choice if run_ml else None, learn_frac, t_thr, need_consistent, HORIZONS)
    key = "stable" if scan["stable"].notna().any() else "combined"
    h = int(scan.loc[scan[key].idxmax(), "h"]) if scan[key].notna().any() else 6
else:
    h = h_manual

cb(0.5, "Analysing macro environments...")
fwd = target_returns(m, F_ok.index, h, target)
fdd = fwd_drawdown(m["silver"], h).reindex(F_ok.index)
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
        cb(0.55 + 0.45 * i / len(ML_MODELS), f"Training {name} ({h}-month look-ahead)...")
        ml[name] = walk_forward(Fev, fev, h, cut, step, name)
    ml_p, ml_base, ml_imp, ml_contrib = fit_today(Fev, fev, F_ok.iloc[-1], ml_choice)
cb(1.0, "Done")
bar.empty()

now = F_ok.index[-1]
v_now, sc_now = verdict.loc[now], int(score.loc[now])
ana = analogues(F_ok, fwd, ev, F_ok.iloc[-1], h)
reg = classify_regime(F_all.reindex(F_ok.index), F_ok.index, learn_idx)
streak = streak_info(verdict, fwd, ev)
expo_list = [EXPO[v_now]]
if run_ml:
    dm = ml_p - ml_base
    ml_v_now = FAV if dm >= margin else (UNF if dm <= -margin else NEU)
    expo_list.append(EXPO[ml_v_now])
expo_now = float(np.mean(expo_list))
ICON = {FAV: "🟢", NEU: "⚪", UNF: "🔴"}
tgt_txt = {"ret": "silver is higher", "gold": "silver beats gold", "cash": "silver beats cash"}[target]

st.info(f"Predicting: **{target_name}** over **{h} months**" + (" (auto-selected on the learning period only, then locked)" if auto_h else "")
        + f"  |  Indicators: **{len(META)}**")

tabs_def = [("dash", "📊 Dashboard"), ("guide", "📖 Guide"), ("env", "🗺 Environments"), ("rank", "🏆 Indicator ranking"),
            ("test", "🧪 Out-of-sample"), ("reg", "🧭 Regimes & analogues"), ("bt", "💰 Backtest")]
if run_ml:
    tabs_def.append(("ml", "🤖 Machine learning"))
if auto_h:
    tabs_def.append(("scan", "🔍 Look-ahead scan"))
tabs_def.append(("now", "✅ Silver & gold now"))
T = dict(zip([k for k, _ in tabs_def], st.tabs([n for _, n in tabs_def])))

# ------------------------------------------------------------------ dashboard
with T["dash"]:
    st.subheader(f"{ICON[v_now]} Historically {v_now.lower()} environment for silver ({now:%b %Y})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro score (rules)", f"{sc_now:+d}")
    if run_ml:
        c2.metric(f"ML: chance {tgt_txt} in {h}m", f"{ml_p:.0%}", f"{(ml_p - ml_base) * 100:+.1f} pts vs normal ({ml_base:.0%})")
    c3.metric("Suggested exposure", f"{expo_now:.0%}", help="Share of your INTENDED silver allocation, not of your portfolio.")
    if "val_real" in F_all and pd.notna(F_all["val_real"].get(now, np.nan)):
        c4.metric("Silver price vs own history (real)", f"{F_all['val_real'][now]:.0%} percentile")
    st.caption("Exposure = average of the rules and ML views (Favorable 100%, Neutral 50%, Unfavorable 0%) as a share of the silver allocation you already intended. "
               "See the last tab for a silver and gold read side by side.")

    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Pillar scores** (−100 = all unfavorable, +100 = all favorable)")
        S_now = S.loc[now]
        rows = []
        for pl in PILLARS:
            cols = [c for c in META if META[c][4] == pl]
            if cols:
                sc = 100 * (sum((c, S_now[c]) in good for c in cols) - sum((c, S_now[c]) in bad for c in cols)) / len(cols)
                rows.append({"Pillar": pl, "Score": f"{'🟢' if sc > 15 else ('🔴' if sc < -15 else '🟡')} {sc:+.0f}", "Indicators": len(cols)})
        show_df(st, pd.DataFrame(rows))
        if reg is not None:
            st.markdown(f"**Macro regime:** {reg.iloc[-1]}")
        cur, n_cur, med, n_h, avg_h = streak
        st.markdown(f"**Persistence:** {cur.lower()} for **{n_cur}** month(s) (typical run: {med:.0f}). "
                    + (f"After 3+ months in this state, the average outcome was {avg_h:+.1%} ({n_h} months)." if n_h >= 5 else ""))
    with cr:
        st.markdown(f"**10 most similar past environments** → outcome after {h} months ({target_name.lower()})")
        a = ana["After"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Median", pc(a.median())); k2.metric("Average", pc(a.mean())); k3.metric("Positive", f"{(a > 0).mean():.0%}")
        k4.metric("Worst", pc(a.min())); k5.metric("Best", pc(a.max()))
        st.caption("Only 10 observations: indicative, not statistical proof.")

    st.warning("**This is not a buy or sell signal.** It describes how silver behaved after similar macro conditions. "
               "It says nothing about whether silver is cheap or expensive today, and past patterns can stop working.")

# ------------------------------------------------------------------ guide
with T["guide"]:
    st.markdown(f"""
### How this works
- **Environments (rules):** each month is described by {len(META)} macro indicators, each split into Low/Neutral/High using the learning period.
  We measure silver's {h}-month outcome in each bucket and keep only environments that are statistically clear (HAC t-stat, valid for overlapping windows) and hold in both halves of history.
- **Macro score / pillars:** favorable minus unfavorable environments, overall and per theme (monetary, dollar, industrial, liquidity & risk, gold & valuation).
- **Indicator ranking:** how strongly each single indicator relates to the future outcome, in %, on the learning period AND on unseen data, plus by look-ahead.
- **Machine learning:** three models estimate the chance that *{tgt_txt}*, retrained walk-forward on past data only. Judged by AUC, Brier skill, log loss and calibration.
- **Regimes & analogues:** a growth × inflation regime, plus the 10 most similar historical months and what silver did next.
- **Backtest:** nine strategies (scaled, defensive, aggressive, silver-only-when-favorable, gold rotation, trend filters) on the unseen test period with trading costs.
  Tick *Find the best strategy* in the sidebar to rank them all, with a first-half / second-half check on whether the winner was luck.
- **Silver & gold now:** a separate read for silver, gold and silver-versus-gold in today's environment, with how well the rules worked on unseen data.
- **Honesty built in:** everything is learned on the first {learn_frac:.0%} of history and judged on later months. The look-ahead is selected on the learning period and then locked.
- **What to predict:** relative targets ask whether silver *beats* gold or cash, which is a different question from "does silver go up".

Educational only, not financial advice.
""")

# ------------------------------------------------------------------ environments
with T["env"]:
    st.subheader(f"Average outcome after {h} months, by environment (learn period)")
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
    fig = go.Figure(go.Heatmap(z=z, x=["Low", "Neutral", "High"], y=[META[c][0] for c in META], text=txt, texttemplate="%{text}",
                               colorscale="RdYlGn", zmid=0, zmin=-zmax, zmax=zmax, showscale=False, xgap=3, ygap=3))
    fig.update_layout(height=max(420, 52 * len(META)), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    fig.update_traces(textfont_size=11)
    show_plot(st, fig)

    def rank_table(df):
        return pd.DataFrame({
            "Environment": [state_label(r.col, r.state) for r in df.itertuples()],
            "Indicator": [META[r.col][0] for r in df.itertuples()], "Months": df["n"].values,
            "Avg after": [pc(v) for v in df["avg"]], "vs. average": [f"{v * 100:+.1f} pts" for v in df["edge"]],
            "% positive": [pc(v, False) for v in df["win"]], "HAC t": [f"{v:+.1f}" for v in df["t"]],
            "Both halves?": ["✅" if v else "❌" for v in df["consistent"]]})

    cL, cR = st.columns(2)
    cL.subheader("🟢 Most favorable"); show_df(cL, rank_table(stats.sort_values("edge", ascending=False).head(6)))
    cR.subheader("🔴 Most unfavorable"); show_df(cR, rank_table(stats.sort_values("edge").head(6)))
    st.caption(f"HAC t above ±2 is fairly strong. {len(stats)} environments are tested, so some look good by luck. That is why the out-of-sample tab matters. "
               f"Rules in use: {len(good)} favorable, {len(bad)} unfavorable.")
    if good or bad:
        c1, c2 = st.columns(2)
        c1.markdown("**Favorable:**\n" + "\n".join(f"- {state_label(c, s_)}" for c, s_ in sorted(good)) if good else "None")
        c2.markdown("**Unfavorable:**\n" + "\n".join(f"- {state_label(c, s_)}" for c, s_ in sorted(bad)) if bad else "None")

# ------------------------------------------------------------------ indicator ranking
with T["rank"]:
    st.subheader("Which indicators predict best?")
    st.caption(f"Target: **{target_name}** over **{h} months**. Each indicator is compared with the outcome that followed, "
               f"first on the learning period ({learn_idx[0]:%b %Y} to {learn_idx[-1]:%b %Y}), then on data it has never seen "
               f"({test_idx[0]:%b %Y} to {test_idx[-1]:%b %Y}). An indicator only ranks high if it works in BOTH.")
    rk = indicator_ranking(F_ok, fwd, S, learn_idx, test_idx, h)
    top = rk[rk["reliab"] > 0].head(3)
    if len(top):
        st.success("Most reliable so far: " + ", ".join(f"**{META[r.col][0]}** ({r.reliab:.0f}%)" for r in top.itertuples()))
    else:
        st.warning("No indicator kept the same direction on unseen data. Treat every single indicator here as unreliable for this target and look-ahead.")

    st.markdown("**Ranking** (sorted by reliable strength)")
    show_df(st, pd.DataFrame({
        "#": range(1, len(rk) + 1),
        "Indicator": [META[c][0] for c in rk["col"]],
        "Pillar": [META[c][4] for c in rk["col"]],
        "Direction": ["-" if pd.isna(v) else ("Higher → better outcome" if v > 0 else "Higher → worse outcome") for v in rk["ic_l"]],
        "Learn correlation": [pc(v) for v in rk["ic_l"]],
        "Unseen correlation": [pc(v) for v in rk["ic_t"]],
        "Reliable strength": [f"{v:.0f}%" for v in rk["reliab"]],
        "HAC t (learn)": ["-" if pd.isna(v) else f"{v:+.1f}" for v in rk["t_l"]],
        "Hit rate (unseen)": [pc(v, False) for v in rk["hit"]],
        "High minus Low (learn)": ["-" if pd.isna(v) else f"{v:+.1f} pts" for v in rk["hl_l"]],
        "High minus Low (unseen)": ["-" if pd.isna(v) else f"{v:+.1f} pts" for v in rk["hl_t"]],
        "Verdict": rk["status"]}))
    st.caption("**Correlation** = rank correlation between the indicator and the outcome (±10% is already useful for macro data, below ±5% is hard to tell from noise). "
               "**Reliable strength** = the smaller of the learn and unseen correlations, and 0% if the direction flips. "
               "**Hit rate** = how often the learn-period direction called the above/below-median outcome on unseen data (50% = coin flip). "
               "**High minus Low** = average outcome when the indicator was in its top third minus its bottom third. "
               "With this many indicators tested, a few look good by luck, which is why the unseen column matters.")

    d_ = rk.dropna(subset=["ic_l"])
    if len(d_):
        nm = [META[c][0] for c in d_["col"]]
        bf_ = go.Figure()
        bf_.add_bar(y=nm, x=d_["ic_l"] * 100, name="Learn period", orientation="h")
        bf_.add_bar(y=nm, x=d_["ic_t"] * 100, name="Unseen (test) period", orientation="h")
        bf_.update_layout(barmode="group", yaxis=dict(autorange="reversed"), height=max(420, 44 * len(d_)),
                          xaxis_title="Correlation with the outcome (%)", title="Learn vs unseen correlation")
        show_plot(st, bf_)

    pill = rk.assign(Pillar=[META[c][4] for c in rk["col"]]).groupby("Pillar")["reliab"].agg(["mean", "max", "count"]).sort_values("mean", ascending=False)
    st.markdown("**Which themes carry the most reliable signal**")
    show_df(st, pd.DataFrame({"Pillar": pill.index, "Average reliable strength": [f"{v:.1f}%" for v in pill["mean"]],
                              "Best indicator in pillar": [f"{v:.1f}%" for v in pill["max"]], "Indicators": pill["count"].values}))

    st.markdown("**Which look-ahead works best for each indicator** (learn period only)")
    ich = ic_by_horizon(m, F_ok, target, learn_frac).reindex(rk["col"])
    zz = ich.values * 100
    hm = go.Figure(go.Heatmap(z=zz, x=[f"{x}m" for x in ich.columns], y=[META[c][0] for c in ich.index],
                              text=[[("" if np.isnan(v) else f"{v:+.0f}%") for v in row] for row in zz], texttemplate="%{text}",
                              colorscale="RdYlGn", zmid=0, zmin=-max(10, np.nanmax(np.abs(zz))) if np.isfinite(zz).any() else -10,
                              zmax=max(10, np.nanmax(np.abs(zz))) if np.isfinite(zz).any() else 10, showscale=False, xgap=3, ygap=3))
    hm.update_layout(height=max(420, 36 * len(ich)), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    show_plot(st, hm)
    st.caption("Green = higher indicator values were followed by better outcomes, red = worse. Longer look-aheads have far fewer independent observations, "
               "so strong-looking numbers on the right are less trustworthy. This grid uses the learn period only.")

# ------------------------------------------------------------------ out of sample
with T["test"]:
    if not (good or bad):
        st.warning("No environment passed the strength filter. Lower the minimum strength in the sidebar.")
    else:
        a, b = st.columns(2)
        a.subheader("Learn period (in-sample)")
        a.caption(f"{learn_idx[0]:%b %Y} to {learn_idx[-1]:%b %Y}. Flattering by construction.")
        show_df(a, fmt_summary(summarize(fwd, verdict, learn_idx, h, fdd), True))
        b.subheader("Test period (out-of-sample) ✅")
        b.caption(f"{test_idx[0]:%b %Y} to {test_idx[-1]:%b %Y}. Never seen by the rules.")
        tsum = summarize(fwd, verdict, test_idx, h, fdd)
        show_df(b, fmt_summary(tsum, True))
        bf = go.Figure(go.Bar(x=tsum["Group"], y=tsum["Avg"] * 100, marker_color=["#2e9e5b", "#9aa0a6", "#d64545", "#4a6fa5"],
                              error_y=dict(type="data", symmetric=False, array=(tsum["hi"] - tsum["Avg"]) * 100,
                                           arrayminus=(tsum["Avg"] - tsum["lo"]) * 100),
                              text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in tsum["Avg"] * 100], textposition="outside"))
        bf.update_layout(title="Test period: average outcome with 95% block-bootstrap CI", yaxis_title="%", height=380)
        show_plot(st, bf)
        tb = tsum.set_index("Group")
        if tb.loc[FAV, "Months"] > 0 and tb.loc[UNF, "Months"] > 0:
            sp = (tb.loc[FAV, "Avg"] - tb.loc[UNF, "Avg"]) * 100
            overlap = tb.loc[FAV, "lo"] <= tb.loc[UNF, "hi"]
            (st.success if sp > 2 and not overlap else st.warning)(
                f"Favorable minus Unfavorable: {sp:.1f} points. " + ("The confidence intervals do not overlap, which is encouraging."
                                                                    if not overlap else "The confidence intervals overlap, so this could easily be noise."))
        pl = go.Figure()
        pl.add_scatter(x=m.index, y=m["silver"], name="Silver", line=dict(color="#888"))
        for g_, col in [(FAV, "#2e9e5b"), (UNF, "#d64545")]:
            ix = verdict[verdict == g_].index
            pl.add_scatter(x=ix, y=m["silver"].reindex(ix), mode="markers", name=g_, marker=dict(color=col, size=6))
        pl.add_vline(x=test_idx[0], line_dash="dash", annotation_text="test period starts")
        pl.update_layout(title="Silver with macro verdicts", yaxis_type="log", height=380)
        show_plot(st, pl)

# ------------------------------------------------------------------ regimes & analogues
with T["reg"]:
    if reg is None:
        st.info("Regime classification needs at least one growth indicator (copper, industrial production, stocks) and one inflation indicator (CPI, breakevens, oil).")
    else:
        st.subheader("Growth × inflation regimes")
        st.caption("Growth = average z-score of copper, industrial production and stocks; inflation = CPI, breakevens and oil (z-scores from the learn period). "
                   "Descriptive over all months with known outcomes.")
        rs = summarize(fwd, reg, ev, h, fdd, order=REGIMES)
        show_df(st, fmt_summary(rs, True))
        st.markdown(f"**Today:** {reg.iloc[-1]}")
    st.subheader("10 most similar historical environments")
    st.caption("Closest months by standardized distance across all selected indicators (at least h months apart). Closeness is relative to the typical distance in history.")
    show_df(st, pd.DataFrame({"Date": ana["Date"], "Closeness": ana["Closeness"].map(lambda v: f"{v:.0f}%"),
                              f"Outcome after {h}m": ana["After"].map(pc)}))

# ------------------------------------------------------------------ backtest
with T["bt"]:
    st.subheader("What if you had followed the signals? (test period only)")
    rf_all = cash_rate(m)

    # signal sources, all restricted to the same months so everything is comparable
    src = {"Rules": verdict.loc[test_idx]}
    if run_ml and not ml[ml_choice].empty:
        mv = ml_verdict(ml[ml_choice], margin)
        src[f"ML ({ml_choice})"] = mv
        both = src["Rules"].index.intersection(mv.index)
        e_ = (src["Rules"].loc[both].map(EXPO) + mv.loc[both].map(EXPO)) / 2
        comb = pd.Series(NEU, index=both)
        comb[e_ >= 0.75] = FAV
        comb[e_ <= 0.25] = UNF
        src["Rules + ML combined"] = comb
    bt_idx = None
    for v_ in src.values():
        bt_idx = v_.index if bt_idx is None else bt_idx.intersection(v_.index)
    nxt = m["silver"].pct_change().shift(-1).notna() & m["gold"].pct_change().shift(-1).notna()
    bt_idx = bt_idx[nxt.reindex(bt_idx).fillna(False).values.astype(bool)]
    src = {k: v_.loc[bt_idx] for k, v_ in src.items()}
    trend = (m["silver"] > m["silver"].rolling(10).mean()).reindex(bt_idx).fillna(False).astype(bool)

    if len(bt_idx) < 12:
        st.warning("The test period is too short for a backtest. Move the start date earlier or lower the learn share.")
    else:
        st.caption(f"Strategy shown: **{mode}**. Signal at each month-end sets the position for the next month. Idle money earns the T-bill rate. "
                   f"Cost: {bps} bps per 100% traded (switching silver to gold trades both). {len(bt_idx)} months, {bt_idx[0]:%b %Y} to {bt_idx[-1]:%b %Y}.")
        curves, rows = {}, []
        pairs = [("Trend only (no macro signal)", next(iter(src.values())))] if mode in NO_SIGNAL else list(src.items())
        for sname, v_ in pairs:
            W = strategy_weights(mode, v_, trend)
            ret, turn = run_backtest_w(m, W, bps)
            curves[sname] = ret
            rows.append((sname, perf(ret, rf_all, (W["silver"] + W["gold"]).reindex(ret.index).mean(), int((turn > 0).sum()))))
        for bname, bret in bench_curves(m, bt_idx).items():
            curves[bname] = bret
            is_cash = bname.startswith("Cash")
            rows.append((bname, perf(bret, rf_all, 0.0 if is_cash else 1.0, 0 if is_cash else 1)))
        show_plot(st, growth_chart(curves, bt_idx, "Growth of 1 unit"))
        show_df(st, perf_table(rows))
        st.caption("A short test period and a handful of trades make these numbers noisy. If the strategy only wins by sitting in cash during drawdowns, check max drawdown and Sharpe, not just CAGR.")

        if find_best:
            st.markdown("---")
            st.subheader(f"🏁 Strategy leaderboard (ranked by {rank_by})")
            res, rets = [], {}
            first_src = next(iter(src))
            for sname, v_ in src.items():
                for strat in STRATEGIES:
                    if strat in NO_SIGNAL and sname != first_src:
                        continue
                    W = strategy_weights(strat, v_, trend)
                    ret, turn = run_backtest_w(m, W, bps)
                    if len(ret) < 12:
                        continue
                    half = len(ret) // 2
                    inv = (W["silver"] + W["gold"]).reindex(ret.index).mean()
                    p_all = perf(ret, rf_all, inv, int((turn > 0).sum()))
                    p1, p2 = perf(ret.iloc[:half], rf_all), perf(ret.iloc[half:], rf_all)
                    lab = "No macro signal" if strat in NO_SIGNAL else sname
                    key_ = f"{lab} · {strat}"
                    rets[key_] = ret
                    res.append(dict(key=key_, signal=lab, strat=strat, p=p_all,
                                    val=p_all[rank_by], v1=p1[rank_by], v2=p2[rank_by]))
            bs = bench_curves(m, bt_idx)["Silver buy & hold"]
            hb = len(bs) // 2
            b1, b2 = perf(bs.iloc[:hb], rf_all)[rank_by], perf(bs.iloc[hb:], rf_all)[rank_by]
            if not res:
                st.info("No strategy had enough months to rank.")
            else:
                res.sort(key=lambda r_: -np.inf if pd.isna(r_["val"]) else r_["val"], reverse=True)
                medals = ["🥇", "🥈", "🥉"]
                lb = pd.DataFrame([{
                    "#": medals[i] if i < 3 else str(i + 1), "Signal": r_["signal"], "Strategy": r_["strat"],
                    "CAGR": pc(r_["p"]["CAGR"]), "Volatility": pc(r_["p"]["Volatility"], False),
                    "Sharpe": fm("Sharpe", r_["p"]["Sharpe"]), "Sortino": fm("Sortino", r_["p"]["Sortino"]),
                    "Calmar": fm("Calmar", r_["p"]["Calmar"]), "Max drawdown": pc(r_["p"]["Max drawdown"]),
                    "% invested": pc(r_["p"]["% invested"], False), "Trades": r_["p"]["Trades"],
                    f"{rank_by}, 1st half": fm(rank_by, r_["v1"]), f"{rank_by}, 2nd half": fm(rank_by, r_["v2"]),
                    "Beats silver in both halves?": "✅" if (pd.notna(r_["v1"]) and pd.notna(r_["v2"]) and r_["v1"] > b1 and r_["v2"] > b2) else "❌"}
                    for i, r_ in enumerate(res)])
                show_df(st, lb)
                st.caption(f"Silver buy & hold for reference: {rank_by} {fm(rank_by, perf(bs, rf_all)[rank_by])} overall "
                           f"({fm(rank_by, b1)} first half, {fm(rank_by, b2)} second half). "
                           "'Beats silver in both halves' means the strategy had a better ranking metric than silver buy & hold in each half of the test period.")
                best = res[0]
                st.success(f"Best on {rank_by} over the whole test period: **{best['strat']}** with **{best['signal']}** "
                           f"({rank_by} {fm(rank_by, best['val'])}, CAGR {pc(best['p']['CAGR'])}, max drawdown {pc(best['p']['Max drawdown'])}).")

                top_curves = {r_["key"]: rets[r_["key"]] for r_ in res[:3]}
                top_curves.update({k_: v_ for k_, v_ in bench_curves(m, bt_idx).items() if k_ in ("Silver buy & hold", "Gold buy & hold")})
                show_plot(st, growth_chart(top_curves, bt_idx, "Top 3 strategies vs buy & hold"))

                valid = [r_ for r_ in res if pd.notna(r_["v1"]) and pd.notna(r_["v2"])]
                if len(valid) >= 3:
                    pick = max(valid, key=lambda r_: r_["v1"])
                    v2s = sorted([r_["v2"] for r_ in valid], reverse=True)
                    rank2 = v2s.index(pick["v2"]) + 1
                    msg = (f"**Honesty check.** Picking the best strategy using only the FIRST half of the test period gives "
                           f"**{pick['strat']}** with **{pick['signal']}**. In the SECOND half it scored {rank_by} {fm(rank_by, pick['v2'])}, "
                           f"ranking **{rank2} of {len(valid)}** (median strategy {fm(rank_by, float(np.median(v2s)))}, silver buy & hold {fm(rank_by, b2)}).")
                    (st.success if rank2 <= max(1, len(valid) // 3) and pick["v2"] > b2 else st.warning)(msg)
                st.warning(f"Choosing the best of {len(res)} strategies on the same data used to rank them flatters the winner. "
                           "Trust a strategy more when it ranks well in both halves, beats silver buy & hold in both halves, and the honesty check above holds up.")

# ------------------------------------------------------------------ machine learning
if run_ml:
    with T["ml"]:
        st.markdown(f"""
Each model estimates the **probability that {tgt_txt}** in {h} months, retrained every {step} months on past data only.
**AUC** 0.50 = coin flip, 0.55-0.60 = modest skill, above 0.65 would be suspicious. **Brier skill** > 0 means the probabilities beat just quoting the normal odds.
**Log loss** punishes confident mistakes (lower is better).
""")
        rows = []
        for name, wf in ml.items():
            if wf.empty:
                continue
            yy = (fwd.loc[wf.index] > 0).astype(int)
            auc = roc_auc_score(yy, wf["p"]) if yy.nunique() > 1 else np.nan
            br, bss, ll = prob_metrics(wf, yy)
            sm = summarize(fwd, ml_verdict(wf, margin), wf.index, h).set_index("Group")
            ok = sm.loc[FAV, "Months"] >= 3 and sm.loc[UNF, "Months"] >= 3
            sp = (sm.loc[FAV, "Avg"] - sm.loc[UNF, "Avg"]) * 100 if ok else np.nan
            rows.append({"Model": name, "AUC": f"{auc:.2f}", "Brier skill": f"{bss:+.3f}", "Log loss": f"{ll:.3f}",
                         "Accuracy": f"{((wf['p'] > 0.5) == yy).mean():.0%}", "Naive": f"{((wf['base'] > 0.5) == yy).mean():.0%}",
                         "Favorable minus Unfavorable": "-" if np.isnan(sp) else f"{sp:+.1f} pts"})
        st.subheader("Out-of-sample comparison")
        show_df(st, pd.DataFrame(rows))
        wf = ml[ml_choice]
        if wf.empty:
            st.warning("Not enough history to train. Move the start date earlier.")
        else:
            yy = (fwd.loc[wf.index] > 0).astype(int)
            auc = roc_auc_score(yy, wf["p"]) if yy.nunique() > 1 else np.nan
            _, bss, _ = prob_metrics(wf, yy)
            (st.success if (auc >= 0.55 and bss > 0) else st.warning)(
                f"{ml_choice}: AUC {auc:.2f}, Brier skill {bss:+.3f}. " + ("Some out-of-sample skill, but the test period is short."
                                                                       if (auc >= 0.55 and bss > 0) else "Little or no reliable skill on unseen data. Don't rely on it."))
            msum = summarize(fwd, ml_verdict(wf, margin), wf.index, h, fdd)
            st.subheader(f"{ml_choice}: outcome after each call (test period)")
            show_df(st, fmt_summary(msum, True))
            c1, c2 = st.columns(2)
            pf = go.Figure(go.Scatter(x=wf.index, y=wf["p"], name="P(positive)"))
            pf.add_scatter(x=wf.index, y=wf["base"], name="Normal odds", line=dict(dash="dash"))
            pf.update_layout(title="Out-of-sample probability", yaxis_tickformat=".0%", height=340)
            show_plot(c1, pf)
            try:
                bins = pd.qcut(wf["p"], 5, duplicates="drop")
                cal = pd.DataFrame({"pred": wf["p"].groupby(bins, observed=True).mean(), "act": yy.groupby(bins, observed=True).mean()})
                cf = go.Figure(go.Scatter(x=cal["pred"], y=cal["act"], mode="lines+markers", name="Model"))
                cf.add_scatter(x=[0, 1], y=[0, 1], name="Perfect", line=dict(dash="dash"))
                cf.update_layout(title="Calibration: predicted vs actual", xaxis_title="Predicted", yaxis_title="Actual", height=340)
                show_plot(c2, cf)
            except ValueError:
                c2.info("Not enough spread in predictions for a calibration plot.")
            if ml_contrib is not None:
                st.subheader("Why today's prediction? (log-odds contribution of each indicator)")
                st.bar_chart(ml_contrib.rename(index={c: META[c][0] for c in ml_contrib.index}).sort_values())
                st.caption("Positive pushes toward the target being positive, negative pushes away. Exact for logistic regression.")
            else:
                st.subheader("What the model uses most")
                st.bar_chart(ml_imp.rename(index={c: META[c][0] for c in ml_imp.index}).sort_values())
                st.caption("Importance is not causality, and correlated indicators share credit unpredictably. Switch to logistic regression for exact contributions.")

# ------------------------------------------------------------------ scan
if auto_h:
    with T["scan"]:
        st.markdown(f"""
Each look-ahead is fitted on the **first 70% of the learning period** and scored on its **last 30%**. **IC** = rank correlation between prediction and outcome (0 = no skill, ~0.10 = decent).
**Stable IC** is the worse of the two validation halves, so a horizon only wins if it works in both. Selected: **{h} months**, then locked before the final test.
""")
        d = scan.copy()
        d["Look-ahead"] = d["h"].map(lambda x: f"{x} months" + (" ✅" if x == h else ""))
        for c in ["rules_ic", "ml_ic", "combined", "stable"]:
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else f"{v:+.2f}")
        d["Independent obs."] = (scan["n_val"] / scan["h"]).round(0).astype(int)
        show_df(st, d[["Look-ahead", "rules_ic", "ml_ic", "combined", "stable", "Independent obs."]].rename(
            columns={"rules_ic": "Rules IC", "ml_ic": "ML IC", "combined": "Average IC", "stable": "Stable IC"}))
        st.caption("Longer look-aheads have far fewer independent observations. If every IC is near zero, there is no reliable horizon, and that is a valid answer.")

# ------------------------------------------------------------------ silver & gold now
with T["now"]:
    st.subheader(f"✅ Silver & gold: where the environment stands now ({now:%b %Y})")
    st.caption(f"Each market gets its own rules and ML read for the next {h} months, using the same indicators and settings. "
               "Favorable / Unfavorable describes how similar environments played out historically. It is not a buy or sell instruction.")
    with st.spinner("Analysing silver, gold and silver vs gold..."):
        NOW = {k: analyse_target(k, m, F_ok, h, learn_frac, t_thr, need_consistent, run_ml, ml_choice, step, margin) for k in NOW_KINDS}

    for col, k in zip(st.columns(3), NOW_KINDS):
        r = NOW[k]
        box = card(col)
        box.markdown(f"### {r['icon']} {NOW_NAMES[k]}")
        box.markdown(f"**{REL_TXT[r['label']] if k == 'gold' else r['label']}**")
        box.markdown(f"Rules: {ICON[r['rules']]} {r['rules']} (score {r['score']:+d})")
        if r["ml"]:
            box.markdown(f"ML: {r['ml']['p']:.0%} chance {NOW_PHRASE[k]} in {h}m (normal {r['ml']['base']:.0%}) → {ICON[r['ml']['verdict']]} {r['ml']['verdict']}")
        box.markdown(f"Similar past periods: median {pc(r['ana_med'])}, {r['ana_pos']:.0%} positive")
        box.markdown(f"Evidence on unseen data: **{r['ev_level']}**")
        box.caption(r["ev_text"])
        if r["ml"] and pd.notna(r["ml"]["auc"]):
            box.caption(f"ML on unseen data: AUC {r['ml']['auc']:.2f}, Brier skill {r['ml']['bss']:+.3f}.")
        if r["why_good"]:
            box.markdown("**Helping now:** " + "; ".join(r["why_good"][:4]))
        if r["why_bad"]:
            box.markdown("**Hurting now:** " + "; ".join(r["why_bad"][:4]))

    es, eg, er = NOW["ret"]["e"], NOW["goldabs"]["e"], NOW["gold"]["e"]
    sl, gl = NOW["ret"]["label"].lower(), NOW["goldabs"]["label"].lower()
    if es >= 0.7 and eg >= 0.7:
        msg = "Both silver and gold sit in historically favorable environments."
    elif es <= 0.3 and eg <= 0.3:
        msg = "Both silver and gold sit in historically unfavorable environments."
    elif es >= 0.7:
        msg = f"Silver's environment is historically favorable ({sl}), while gold's is {gl}."
    elif eg >= 0.7:
        msg = f"Gold's environment is historically favorable ({gl}), while silver's is {sl}."
    elif es <= 0.3:
        msg = f"Silver's environment is historically unfavorable ({sl}), while gold's is {gl}."
    elif eg <= 0.3:
        msg = f"Gold's environment is historically unfavorable ({gl}), while silver's is {sl}."
    else:
        msg = "Neither metal is in a clearly favorable or unfavorable environment: mostly neutral."
    if er >= 0.7:
        msg += " Similar conditions have historically favored silver over gold."
    elif er <= 0.3:
        msg += " Similar conditions have historically favored gold over silver."
    else:
        msg += " There is no clear silver-versus-gold preference."
    (st.success if max(es, eg) >= 0.7 else (st.warning if min(es, eg) <= 0.3 else st.info))(msg)

    weak = [NOW_NAMES[k] for k in NOW_KINDS if NOW[k]["ev_level"] != "Strong"]
    if weak:
        st.caption("Evidence is Weak, None or Unknown for: " + ", ".join(weak) + ". For those, the label describes today's conditions "
                   "but the rules did not reliably predict outcomes on data they had never seen, so treat it as context, not a signal.")

    show_df(st, pd.DataFrame([{
        "Market": NOW_NAMES[k],
        "Overall": f"{NOW[k]['icon']} {REL_TXT[NOW[k]['label']] if k == 'gold' else NOW[k]['label']}",
        "Rules": f"{ICON[NOW[k]['rules']]} {NOW[k]['rules']}",
        "ML chance": "-" if not NOW[k]["ml"] else f"{NOW[k]['ml']['p']:.0%} (normal {NOW[k]['ml']['base']:.0%})",
        "Similar periods, median": pc(NOW[k]["ana_med"]),
        "Evidence": NOW[k]["ev_level"]} for k in NOW_KINDS]))
    st.warning("**Educational only, not financial advice.** This reads macro conditions only. It does not know about valuation, news, taxes, "
               "your goals or your time horizon, and relationships that held in the past can stop working.")
