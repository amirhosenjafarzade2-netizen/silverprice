"""
macro_engine.py: asset-independent engine shared by the asset modules (oil_main.py now, silver could adopt it later).
NO Streamlit calls in here, so every function is safe inside worker threads and easy to test.

Contents: data plumbing (FRED / EIA / CFTC download, availability dating, month-end), point-in-time percentiles,
HAC statistics, rule building, expanding-window rules, summaries, simple backtest helpers.
The statistics are the same ones silver_main.py uses (copied, not changed), so results are comparable across assets.
"""
import io
import math

import numpy as np
import pandas as pd
import requests

FAV, NEU, UNF = "Favorable", "Neutral", "Unfavorable"
EXPO = {FAV: 1.0, NEU: 0.5, UNF: 0.0}
STATES = ["Low", "Mid", "High"]


# ------------------------------------------------------------------ data plumbing
def month_end(df):
    try:
        return df.resample("ME").last()
    except ValueError:  # older pandas
        return df.resample("M").last()


def fred_fetch(series_id, start, key=None):
    """One FRED series by observation date. Uses the API when a key is given, else the public CSV endpoint."""
    last = None
    for _ in range(2):
        try:
            if key:
                r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=(5, 20),
                                 params=dict(series_id=series_id, api_key=key, file_type="json", observation_start=start))
                r.raise_for_status()
                d = pd.DataFrame(r.json()["observations"])[["date", "value"]]
                d["value"] = pd.to_numeric(d["value"], errors="coerce")
            else:
                r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}",
                                 timeout=(5, 20), headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                d = pd.read_csv(io.StringIO(r.text), na_values=[".", ""])
                d.columns = ["date", "value"]
            d["date"] = pd.to_datetime(d["date"])
            return d.set_index("date")["value"].astype(float).dropna()
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def eia_fetch(route, id_col, ids, key, start, freq):
    """EIA API v2 (free key: eia.gov/opendata). ids = {eia_series_id: our_name}. Returns a wide frame (one column per series).
    Monthly periods are stamped at month END so they line up with the rest of the app."""
    rows, offset = [], 0
    while True:
        params = {"api_key": key, "frequency": freq, "data[0]": "value", "start": start, "offset": offset, "length": 5000,
                  "sort[0][column]": "period", "sort[0][direction]": "asc", f"facets[{id_col}][]": list(ids)}
        r = requests.get(f"https://api.eia.gov/v2/{route}/data/", params=params, timeout=(5, 60))
        r.raise_for_status()
        d = r.json()["response"]["data"]
        rows += d
        if len(d) < 5000:
            break
        offset += 5000
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"empty EIA response for {route}")
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    w = df.pivot_table(index="period", columns=id_col, values="value", aggfunc="last").sort_index()
    if freq == "monthly":
        w.index = w.index + pd.offsets.MonthEnd(0)
    return w.rename(columns=ids)


COT_DATASETS = ("72hh-3qpy", "kh3c-gbw2")


def cot_fetch(start, code):
    """Managed-money net position as % of open interest (weekly) for one CFTC contract code, indexed by a CONSERVATIVE release date
    (as-of Tuesday + 4 days, i.e. never earlier than the real Friday release). Audit against the CFTC calendar before relying on it."""
    last = None
    for ds in COT_DATASETS:
        try:
            r = requests.get(f"https://publicreporting.cftc.gov/resource/{ds}.json", timeout=(5, 30),
                             params={"$where": f"cftc_contract_market_code='{code}' AND report_date_as_yyyy_mm_dd >= '{start}T00:00:00.000'",
                                     "$select": "report_date_as_yyyy_mm_dd,m_money_positions_long_all,m_money_positions_short_all,open_interest_all",
                                     "$order": "report_date_as_yyyy_mm_dd", "$limit": 50000})
            r.raise_for_status()
            d = pd.DataFrame(r.json())
            if d.empty:
                raise ValueError("empty COT response")
            d["date"] = pd.to_datetime(d["report_date_as_yyyy_mm_dd"])
            for c in ("m_money_positions_long_all", "m_money_positions_short_all", "open_interest_all"):
                d[c] = pd.to_numeric(d[c], errors="coerce")
            s = (d["m_money_positions_long_all"] - d["m_money_positions_short_all"]) / d["open_interest_all"] * 100
            s.index = pd.DatetimeIndex(d["date"]) + pd.Timedelta(days=4)
            s = s.dropna()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if len(s) < 100:
                raise ValueError("too few COT observations")
            return s.astype(float)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def avail(s, lag):
    """Re-index an observation-dated series by the date each value became public. lag = (months, days) after the observation date."""
    months, days = lag
    idx = s.index + pd.DateOffset(months=months) + pd.Timedelta(days=days)
    out = pd.Series(s.values, index=pd.DatetimeIndex(idx))
    return out[~out.index.duplicated(keep="last")].sort_index()


def exp_pctl(s, minp=36):
    """Point-in-time percentile: where today's value ranks among all PREVIOUS values."""
    d = s.dropna()
    p = d.expanding(min_periods=minp).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan, raw=True)
    return p.reindex(s.index)


def seas_dev(s, years=5, rel=True):
    """Deviation from the same calendar month's average over the previous `years` years (needs >= 3 of them). Point-in-time by construction."""
    lag = pd.concat([s.shift(12 * i) for i in range(1, years + 1)], axis=1)
    base = lag.mean(axis=1).where(lag.notna().sum(axis=1) >= 3)
    return (s / base - 1) if rel else (s - base)


# ------------------------------------------------------------------ statistics
def assign_states(F, learn_idx, P=None):
    """Low / Mid / High per indicator: fixed terciles of the learn period, or (if P is given) point-in-time percentiles."""
    cols = {}
    for c in F.columns:
        if P is not None:
            p = P[c].reindex(F.index)
            lab = np.where(p <= 1 / 3, "Low", np.where(p <= 2 / 3, "Mid", "High")).astype(object)
            cols[c] = pd.Series(lab, index=F.index).where(p.notna())
        else:
            lo, hi = F.loc[learn_idx, c].quantile([1 / 3, 2 / 3])
            col = F[c]
            lab = np.where(col <= lo, "Low", np.where(col <= hi, "Mid", "High")).astype(object)
            cols[c] = pd.Series(lab, index=F.index).where(col.notna())
    return pd.DataFrame(cols)


def nw_t(y, d, lags):
    """Newey-West (HAC) t-stat of b in y = a + b*d + e. Valid for overlapping forward returns."""
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


def bh_q(p):
    """Benjamini-Hochberg false-discovery-rate q-values."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    ranked = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def block_ci(x, block, B=1000, seed=0):
    """Approximate 95% CI of the mean via moving-block bootstrap."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan, np.nan
    block = max(1, min(int(block), len(x)))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(x) / block))
    starts = rng.integers(0, len(x) - block + 1, size=(B, nb))
    ix = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, : len(x)]
    lo, hi = np.percentile(x[ix].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def rank_ic(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 8 or d.iloc[:, 0].nunique() < 2:
        return np.nan
    return d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())


def _half_edge(fwd, S, col, state, idx):
    r = fwd.loc[idx][S.loc[idx, col] == state]
    return r.mean() - fwd.loc[idx].mean() if len(r) >= 4 else np.nan


STAT_COLS = ["col", "state", "n", "avg", "median", "win", "edge", "t", "consistent", "p", "q"]


def bucket_stats(S, fwd, idx, h):
    """For every indicator x state (Low / Mid / High): average outcome, edge vs all months, HAC t-stat, stability across the two halves, FDR q."""
    base = fwd.loc[idx]
    yv = base.values
    mid = len(idx) // 2
    first, second = idx[:mid], idx[mid:]
    rows = []
    for c in S.columns:
        sv = S.loc[idx, c].astype(object).values
        for s_ in STATES:
            mask = np.asarray(sv == s_, dtype=bool)
            n = int(mask.sum())
            if n < 8:
                continue
            r = base[mask]
            edge = r.mean() - base.mean()
            t = nw_t(yv, mask.astype(float), h)
            e1, e2 = _half_edge(fwd, S, c, s_, first), _half_edge(fwd, S, c, s_, second)
            cons = bool(np.sign(e1) == np.sign(e2) == np.sign(edge)) if not (np.isnan(e1) or np.isnan(e2)) else False
            rows.append(dict(col=c, state=s_, n=n, avg=r.mean(), median=r.median(), win=(r > 0).mean(), edge=edge, t=t, consistent=cons))
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame(columns=STAT_COLS)
    d["p"] = [math.erfc(abs(t) / math.sqrt(2)) for t in d["t"]]
    d["q"] = bh_q(d["p"].values)
    return d


def build_rules(stats, cfg):
    """cfg = (selection 't' or 'q', threshold, need_consistent, weighting 'equal' or 'strength'). Returns ({(col, state): weight} good, bad)."""
    sel, thr, need_cons, wmode = cfg
    if stats.empty:
        return {}, {}
    ok = stats[stats["t"].abs() >= thr] if sel == "t" else stats[stats["q"] <= thr]
    if need_cons:
        ok = ok[ok["consistent"]]

    def w(t):
        return 1.0 if wmode == "equal" else float(min(max(abs(t) - 1.0, 0.0), 3.0))

    return ({(r.col, r.state): w(r.t) for r in ok.itertuples() if r.edge > 0},
            {(r.col, r.state): w(r.t) for r in ok.itertuples() if r.edge < 0})


def score_series(S, good, bad):
    score = pd.Series(0.0, index=S.index)
    for (c, s_), w in good.items():
        score += w * (S[c] == s_).fillna(False).astype(float)
    for (c, s_), w in bad.items():
        score -= w * (S[c] == s_).fillna(False).astype(float)
    return score


def to_verdict(score, lo_thr, hi_thr):
    v = pd.Series(NEU, index=score.index, dtype=object)
    v[(score >= hi_thr) & (score > 0)] = FAV
    v[(score <= lo_thr) & (score < 0)] = UNF
    return v


def run_rules(F, P, fwd, learn_idx, h, cfg):
    """Fit rules on learn_idx. Verdict thresholds = quartiles of the learn-period score."""
    S = assign_states(F, learn_idx, P)
    stats = bucket_stats(S, fwd, learn_idx, h)
    good, bad = build_rules(stats, cfg)
    score = score_series(S, good, bad)
    lo, hi = np.quantile(score.loc[learn_idx], 0.25), np.quantile(score.loc[learn_idx], 0.75)
    return S, stats, good, bad, score, to_verdict(score, lo, hi)


def walk_forward_rules(F, P, fwd, h, cut, step, cfg):
    """Expanding-window rules. Every `step` months from row `cut` the rules are refitted ONLY on rows whose h-month outcome was already
    known (row i is known at row s0 if i + h <= s0), then used for the next `step` months. Fully out-of-sample."""
    n = len(F)
    ev = F.index.intersection(fwd.dropna().index)
    evpos = np.array([F.index.get_loc(d) for d in ev])
    score = pd.Series(np.nan, index=F.index)
    verd = pd.Series(np.nan, index=F.index, dtype=object)
    for s0 in range(cut, n, step):
        learn = F.index[evpos[evpos <= s0 - h]]
        if len(learn) < 36:
            continue
        S = assign_states(F, learn, P)
        good, bad = build_rules(bucket_stats(S, fwd, learn, h), cfg)
        sc = score_series(S, good, bad)
        lo, hi = np.quantile(sc.loc[learn], 0.25), np.quantile(sc.loc[learn], 0.75)
        te = F.index[s0: s0 + step]
        score.loc[te] = sc.loc[te]
        verd.loc[te] = to_verdict(sc.loc[te], lo, hi)
    return pd.DataFrame({"score": score, "verdict": verd})


def summarize(fwd, groups, idx, h, fdd=None, order=(FAV, NEU, UNF)):
    rows = []
    for g_ in list(order) + ["All months"]:
        r = fwd.loc[idx] if g_ == "All months" else fwd.loc[idx].where(groups.reindex(idx) == g_).dropna()
        row = dict(Group=g_, Months=len(r), Avg=np.nan, Median=np.nan, Win=np.nan, Worst=np.nan, Best=np.nan, lo=np.nan, hi=np.nan, DD=np.nan)
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


def fwd_drawdown(s, h):
    v, out = s.values, np.full(len(s), np.nan)
    for i in range(len(v) - h):
        w = v[i: i + h + 1]
        out[i] = (w / np.maximum.accumulate(w) - 1).min()
    return pd.Series(out, index=s.index)


def split3(F_ok, fwd, h, train_frac, val_frac):
    """learn (rules fitted) -> validation (choices made) -> final test (grading only); months whose outcome windows would overlap are purged."""
    ev = F_ok.index.intersection(fwd.dropna().index)
    n = len(ev)
    c1, c2 = int(n * train_frac), int(n * (train_frac + val_frac))
    learn, val, test = ev[: max(c1 - h, 0)], ev[c1: max(c2 - h, c1)], ev[c2:]
    return dict(ev=ev, c1=c1, cut=int(F_ok.index.get_loc(ev[c1])) if c1 < n else len(F_ok),
                learn=learn, val=val, test=test, unseen=val.union(test))


# ------------------------------------------------------------------ backtest helpers
STRATEGIES = {
    "Scaled: 100% / 50% / 0%": ("map", (1.0, 0.5, 0.0)),
    "Defensive: 100% / 25% / 0%": ("map", (1.0, 0.25, 0.0)),
    "Aggressive: 100% / 75% / 25%": ("map", (1.0, 0.75, 0.25)),
    "Only when Favorable": ("map", (1.0, 0.0, 0.0)),
    "Hold unless Unfavorable": ("map", (1.0, 1.0, 0.0)),
    "Trend only: above 10-month average": ("trend", None),
    "Macro + trend: not Unfavorable and above 10-month average": ("both", None),
}
NO_SIGNAL = {"Trend only: above 10-month average"}


def strategy_weights(name, verdict, trend_up):
    kind, mp = STRATEGIES[name]
    if kind == "map":
        return verdict.map({FAV: mp[0], NEU: mp[1], UNF: mp[2]}).astype(float)
    if kind == "trend":
        return trend_up.astype(float)
    return ((verdict != UNF) & verdict.notna() & trend_up).astype(float)


def run_backtest(px, rf_m, w, idx, bps):
    """Weight w[t] is decided at month-end t and earns the return t -> t+1 (no look-ahead). Cost = bps per 100% traded.
    Unfilled weight earns T-bills."""
    rn = px.pct_change().shift(-1)
    rfn = rf_m.reindex(px.index).shift(-1).fillna(0.0)
    w = w.reindex(px.index).fillna(0.0)
    turn = w.diff().abs().fillna(w.abs())
    r = w * rn + (1 - w) * rfn - turn * bps / 1e4
    ok = rn.reindex(idx).notna()
    return r.reindex(idx)[ok], w.reindex(idx)[ok]


def perf(r, rf=None):
    r = r.dropna()
    n = len(r)
    if n < 6:
        return dict(CAGR=np.nan, Vol=np.nan, Sharpe=np.nan, MaxDD=np.nan, Months=n)
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(12)
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    return dict(CAGR=eq.iloc[-1] ** (12 / n) - 1, Vol=vol, Sharpe=ex.mean() * 12 / vol if vol > 0 else np.nan,
                MaxDD=(eq / eq.cummax() - 1).min(), Months=n)
