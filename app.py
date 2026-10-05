"""
Silver Macro Environment Analyzer (Streamlit), v3.1

Which macro environments have been historically favorable / unfavorable for silver, what regime are we in,
what happened in comparable periods, and would following it have worked (with costs)?

requirements.txt: streamlit, yfinance, pandas, numpy, plotly, requests, scikit-learn
Optional: FRED_API_KEY in Streamlit secrets for reliable FRED access.
Educational only, not financial advice.

v3.1 changes:
- All downloads (Yahoo + 8 FRED series) run in parallel in one cached function.
- Progress callbacks removed from @st.cache_data functions (fixes CacheReplayClosureError).
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


def state_label(col, state):
    return {"Low": META[col][1], "Mid": "Neutral", "High": META[col][2]}[state]


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


# ------------------------------------------------------------------ backtest
def exposure_from(v, mode):
    if mode.startswith("Only"):
        return (v == FAV).astype(float)
    if mode.startswith("Hold"):
        return (v != UNF).astype(float)
    return v.map(EXPO).astype(float)


def run_backtest(m, expo, bps):
    """Signal at month-end t sets the position held during month t -> t+1. Costs charged on turnover."""
    r = m["silver"].pct_change().shift(-1).reindex(expo.index)
    rf = cash_rate(m).reindex(expo.index)
    turn = expo.diff().abs()
    turn.iloc[0] = expo.iloc[0]
    return (expo * r + (1 - expo) * rf - turn * bps / 1e4).dropna(), turn


def perf(r, rf, invested=np.nan, trades=np.nan):
    n = len(r)
    eq = (1 + r).cumprod()
    ex = r - rf.reindex(r.index).fillna(0)
    vol = r.std() * np.sqrt(12)
    dn = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(12)
    return {"CAGR": eq.iloc[-1] ** (12 / n) - 1, "Volatility": vol,
            "Sharpe": ex.mean() * 12 / vol if vol > 0 else np.nan,
            "Sortino": ex.mean() * 12 / dn if dn > 0 else np.nan,
            "Max drawdown": (eq / eq.cummax() - 1).min(), "% invested": invested, "Trades": trades}


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
    mode = st.selectbox("Strategy", ["Scaled: 100% / 50% / 0%", "Only when Favorable", "Hold unless Unfavorable"])
    bps = st.slider("Trading cost (basis points per 100% traded)", 0, 100, 15, 5)

st.title("🥈 Silver Macro Environment Analyzer")
st.caption("Historically favorable or unfavorable macro environments for silver. Not a buy/sell signal, not financial advice.")
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

tabs_def = [("dash", "📊 Dashboard"), ("guide", "📖 Guide"), ("env", "🗺 Environments"), ("test", "🧪 Out-of-sample"),
            ("reg", "🧭 Regimes & analogues"), ("bt", "💰 Backtest")]
if run_ml:
    tabs_def.append(("ml", "🤖 Machine learning"))
if auto_h:
    tabs_def.append(("scan", "🔍 Look-ahead scan"))
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
    st.caption("Exposure = average of the rules and ML views (Favorable 100%, Neutral 50%, Unfavorable 0%) as a share of the silver allocation you already intended.")

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
- **Machine learning:** three models estimate the chance that *{tgt_txt}*, retrained walk-forward on past data only. Judged by AUC, Brier skill, log loss and calibration.
- **Regimes & analogues:** a growth × inflation regime, plus the 10 most similar historical months and what silver did next.
- **Backtest:** follow the signals on the unseen test period with trading costs, versus silver, gold, stocks and cash.
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
    st.caption(f"Strategy: {mode}. Signal at each month-end sets the position for the next month. Idle money earns the T-bill rate. Cost: {bps} bps per 100% traded.")
    curves, rows = {}, []
    rf_all = cash_rate(m)
    sig = {"Rules strategy": verdict.loc[test_idx]}
    if run_ml and not ml[ml_choice].empty:
        sig[f"ML strategy ({ml_choice})"] = ml_verdict(ml[ml_choice], margin)
    common = None
    for name, v in sig.items():
        ex = exposure_from(v, mode)
        ret, turn = run_backtest(m, ex, bps)
        curves[name] = ret
        rows.append((name, perf(ret, rf_all, ex.reindex(ret.index).mean(), int((turn > 0).sum()))))
        common = ret.index if common is None else common.intersection(ret.index)
    for name, ser in {"Silver buy & hold": m["silver"], "Gold buy & hold": m["gold"], "S&P 500": m["spx"]}.items():
        ret = ser.pct_change().shift(-1).reindex(common).dropna()
        curves[name] = ret
        rows.append((name, perf(ret, rf_all, 1.0, 1)))
    curves["Cash (T-bills)"] = rf_all.reindex(common)
    rows.append(("Cash (T-bills)", perf(curves["Cash (T-bills)"], rf_all, 0.0, 0)))
    ef = go.Figure()
    for name, r in curves.items():
        r = r.reindex(common).dropna()
        ef.add_scatter(x=r.index, y=(1 + r).cumprod(), name=name)
    ef.update_layout(title="Growth of 1 unit", height=400)
    show_plot(st, ef)
    tbl = pd.DataFrame([{"Strategy": n, "CAGR": pc(p["CAGR"]), "Volatility": pc(p["Volatility"], False),
                         "Sharpe": f"{p['Sharpe']:.2f}", "Sortino": f"{p['Sortino']:.2f}" if pd.notna(p["Sortino"]) else "-",
                         "Max drawdown": pc(p["Max drawdown"]), "% invested": pc(p["% invested"], False),
                         "Trades": p["Trades"]} for n, p in rows])
    show_df(st, tbl)
    st.caption("A short test period and a handful of trades make these numbers noisy. If the strategy only wins by sitting in cash during drawdowns, check max drawdown and Sharpe, not just CAGR.")

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
