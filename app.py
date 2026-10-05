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

st.set_page_config(page_title="Silver Macro Environment Analyzer", page_icon="🥈", layout="wide")

FRED = {"real_yield": "DFII10", "breakeven": "T10YIE", "fed_funds": "DFF", "curve": "T10Y2Y"}
YF = {"silver": "SI=F", "gold": "GC=F", "dollar": "DX-Y.NYB", "oil": "CL=F",
      "copper": "HG=F", "vix": "^VIX", "spx": "^GSPC",
      "tnx": "^TNX", "irx": "^IRX"}  # tnx/irx = Yahoo rate proxies, used if FRED is unreachable

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


def load_monthly(start: str):
    px = yahoo_daily(start)
    fr, failed = {}, []
    for k, v in FRED.items():
        try:
            fr[k] = fred_cached(v, start)
        except Exception:  # noqa: BLE001
            failed.append(k)
            if len(failed) == 1 and k == "real_yield":
                failed += [x for x in FRED if x != k]  # host is down: don't wait for the rest
                break
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


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    start = st.text_input("Data start date", "2004-01-01")
    h = st.select_slider("Look-ahead period (months)", options=[1, 3, 6, 12], value=6)
    learn_frac = st.slider("Share of history used to LEARN the rules", 0.4, 0.8, 0.6, 0.05)
    t_thr = st.slider("Minimum strength to count an environment", 0.5, 2.5, 1.0, 0.1,
                      help="Higher = fewer, but more reliable, rules.")
    need_consistent = st.checkbox("Require the effect in both halves of the learn period", True)

st.title("🥈 Silver Macro Environment Analyzer")
st.caption("Which macro conditions have been good or bad for buying silver, and where are we now?")

try:
    with st.spinner("Downloading market + FRED macro data..."):
        m, failed = load_monthly(start)
except Exception as e:
    st.error(f"Could not download market data: {e}")
    st.stop()

F = build_features(m)
META = {c: META_ALL[c] for c in F.columns}  # only indicators we actually have
if failed:
    st.warning(
        "FRED (the source for real rates and inflation expectations) did not respond, so the app is using "
        "Yahoo Finance stand-ins: 10y Treasury yield and the 3-month T-bill. Real rates and inflation "
        "expectations are missing from this run. Reload later to retry, or add a free FRED API key "
        "(see Settings > Secrets: FRED_API_KEY = \"your_key\") for reliable access.")
F_ok = F.dropna()
fwd = (m["silver"].shift(-h) / m["silver"] - 1).reindex(F_ok.index)
ev = F_ok.index.intersection(fwd.dropna().index)
cut = int(len(ev) * learn_frac)
learn_idx, test_idx = ev[: max(cut - h, 0)], ev[cut:]  # purge h months so no overlap

if len(learn_idx) < 36 or len(test_idx) < 12:
    st.error("Not enough history. Move the start date earlier or change the learn share.")
    st.stop()

S = assign_states(F_ok, learn_idx)
stats = bucket_stats(S, fwd, learn_idx, h)
good, bad = build_rules(stats, t_thr, need_consistent)
score = score_series(S, good, bad)
lo_thr, hi_thr = np.quantile(score.loc[learn_idx], 0.25), np.quantile(score.loc[learn_idx], 0.75)
verdict = to_verdict(score, lo_thr, hi_thr)

tab_guide, tab_map, tab_rank, tab_test, tab_now = st.tabs(
    ["📖 Guide", "🗺 Environment map", "🏆 Best vs worst", "🧪 Does it hold up?", "📍 Today"])

# ------------------------------------------------------------------ guide
with tab_guide:
    st.markdown(f"""
### How this works (plain English)
1. Each month since {start[:4]} is described by **{len(META)} macro conditions**: interest rates, inflation expectations,
   the Fed, the dollar, oil, copper, stocks, fear, and the gold/silver ratio (where data is available).
2. Every condition is split into **Low / Neutral / High** (for example "dollar weakening / flat / strengthening").
3. For each one, we measure what silver did over the **next {h} months**.
4. Conditions where silver did clearly better than average become **buy-friendly signals**; the clearly worse ones become **avoid signals**.
5. Adding these up gives a **macro score** for any month, including today.
6. To stay honest, the rules are learned on the **first {learn_frac:.0%} of history only**, then checked on the
   remaining months that the rules never saw. That's the **"Does it hold up?"** tab, so look there before trusting anything.

**Tabs:** Environment map = everything at a glance. Best vs worst = the clearest environments.
Does it hold up? = the out-of-sample test. Today = the current verdict.

⚠️ Educational only, not financial advice. Macro conditions shift the odds slightly; they do not predict silver.
""")

# ------------------------------------------------------------------ map
with tab_map:
    st.subheader(f"Average silver return over the next {h} months, by environment")
    st.caption("Learn period only. Green = silver tended to rise afterwards, red = tended to fall.")
    z, txt = [], []
    for c in META:
        zr, tr = [], []
        for s in STATES:
            r = stats[(stats["col"] == c) & (stats["state"] == s)]
            if r.empty:
                zr.append(np.nan); tr.append("")
            else:
                zr.append(r["avg"].iloc[0] * 100)
                tr.append(f"{state_label(c, s)}<br><b>{r['avg'].iloc[0]:+.1%}</b> (n={int(r['n'].iloc[0])})")
        z.append(zr); txt.append(tr)
    zmax = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 10
    fig = go.Figure(go.Heatmap(z=z, x=["Low", "Neutral", "High"], y=[META[c][0] for c in META],
                               text=txt, texttemplate="%{text}", colorscale="RdYlGn",
                               zmid=0, zmin=-zmax, zmax=zmax, showscale=False, xgap=3, ygap=3))
    fig.update_layout(height=640, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    fig.update_traces(textfont_size=11)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ ranking
def rank_table(df):
    o = pd.DataFrame({
        "Environment": [state_label(r.col, r.state) for r in df.itertuples()],
        "Indicator": [META[r.col][0] for r in df.itertuples()],
        "Months": df["n"].values,
        "Avg return after": [f"{v:+.1%}" for v in df["avg"]],
        "vs. average": [f"{v * 100:+.1f} pts" for v in df["edge"]],
        "% rose": [f"{v:.0%}" for v in df["win"]],
        "Strength": [f"{v:+.1f}" for v in df["t"]],
        "Held in both halves?": ["✅" if v else "❌" for v in df["consistent"]],
    })
    return o


with tab_rank:
    cL, cR = st.columns(2)
    with cL:
        st.subheader("🟢 Best environments to buy")
        st.dataframe(rank_table(stats.sort_values("edge", ascending=False).head(6)), hide_index=True, use_container_width=True)
    with cR:
        st.subheader("🔴 Environments to avoid")
        st.dataframe(rank_table(stats.sort_values("edge").head(6)), hide_index=True, use_container_width=True)
    st.caption(f"'vs. average' = extra silver return compared with the average month ({fwd.loc[learn_idx].mean():+.1%} over {h} months). "
               "'Strength' ≈ t-statistic; above ±2 is fairly strong, and it is adjusted for overlapping periods. "
               f"With {len(stats)} environments tested, a few will look good purely by luck. That is why the next tab matters.")
    st.markdown(f"**Rules in use** (strength ≥ {t_thr}{', held in both halves' if need_consistent else ''}): "
                f"{len(good)} buy-friendly, {len(bad)} avoid.")
    if good or bad:
        c1, c2 = st.columns(2)
        c1.markdown("**Buy-friendly:**\n" + "\n".join(f"- {state_label(c, s)}" for c, s in sorted(good)) if good else "None")
        c2.markdown("**Avoid:**\n" + "\n".join(f"- {state_label(c, s)}" for c, s in sorted(bad)) if bad else "None")

# ------------------------------------------------------------------ out of sample
with tab_test:
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

        bar = go.Figure(go.Bar(x=tsum["Environment"], y=tsum["Avg"] * 100,
                               marker_color=["#2e9e5b", "#9aa0a6", "#d64545", "#4a6fa5"],
                               text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in tsum["Avg"] * 100], textposition="outside"))
        bar.update_layout(title=f"Test period: average silver return over the next {h} months", yaxis_title="%", height=360)
        st.plotly_chart(bar, use_container_width=True)

        tb = tsum.set_index("Environment")
        if tb.loc["Buy-friendly", "Months"] > 0 and tb.loc["Avoid", "Months"] > 0:
            spread = (tb.loc["Buy-friendly", "Avg"] - tb.loc["Avoid", "Avg"]) * 100
            if spread > 2:
                st.success(f"Out of sample, 'Buy-friendly' months beat 'Avoid' months by {spread:.1f} points. The signal has some support, "
                           "but the test period is short, so stay cautious.")
            else:
                st.warning(f"Out of sample, the gap between 'Buy-friendly' and 'Avoid' is only {spread:.1f} points. "
                           "The pattern did not hold up well, so don't lean on these rules.")
        st.caption("Months overlap (each looks {0} months ahead), so the number of truly independent observations is much smaller than the month count.".format(h))

        pl = go.Figure()
        pl.add_scatter(x=m.index, y=m["silver"], name="Silver", line=dict(color="#888"))
        for g, col in [("Buy-friendly", "#2e9e5b"), ("Avoid", "#d64545")]:
            ix = verdict[verdict == g].index
            pl.add_scatter(x=ix, y=m["silver"].reindex(ix), mode="markers", name=g, marker=dict(color=col, size=6))
        pl.add_vline(x=test_idx[0], line_dash="dash", annotation_text="test period starts")
        pl.update_layout(title="Silver with macro verdicts", yaxis_type="log", height=380)
        st.plotly_chart(pl, use_container_width=True)

# ------------------------------------------------------------------ today
with tab_now:
    now = F.dropna().index[-1]
    cur = F_ok.loc[now]
    S_now = S.loc[now]
    v_now, sc_now = verdict.loc[now], int(score.loc[now])
    icon = {"Buy-friendly": "🟢", "Neutral": "⚪", "Avoid": "🔴"}[v_now]
    st.subheader(f"{icon} Current macro environment ({now:%b %Y}): {v_now}")
    st.metric("Macro score", f"{sc_now:+d}", help="Buy-friendly conditions minus avoid conditions.")
    rows = []
    for c in META:
        s = S_now[c]
        tag = "🟢 buy-friendly" if (c, s) in good else ("🔴 avoid" if (c, s) in bad else "⚪ neutral")
        val = f"{cur[c]:+.1%}" if META[c][3] else f"{cur[c]:.2f}"
        rows.append({"Indicator": META[c][0], "Reading": val, "Condition": state_label(c, s), "Effect on silver": tag})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown(
        f"**How to use this:** a {v_now.lower()} reading tilts the odds only slightly. Use it as a *background filter*: "
        "be more willing to add silver when conditions are buy-friendly, and more patient when they say avoid. "
        "Check the 'Does it hold up?' tab first, rebuild the analysis every few months, and size positions so being wrong is affordable.")
    st.caption("Latest values can lag a few days (FRED publishes with a delay). Not financial advice.")
