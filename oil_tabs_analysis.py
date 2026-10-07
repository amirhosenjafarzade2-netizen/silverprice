"""
Oil tabs: Backtest, Explorer, Data & coverage.
Executed by oil_main.py (all config, helpers and results are available as globals).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ====================================================================== Backtest
with T["bt"]:
    st.subheader("What if you had followed the signal? (expanding rules only, so no look-ahead)")
    veh = "uso" if vehicle.startswith("USO") else "oil_fut"
    veh_name = "USO" if veh == "uso" else "CL=F"
    if veh not in m or not m[veh].notna().any():
        st.warning(f"{veh_name} prices are unavailable, so the backtest cannot run.")
    else:
        px = m[veh]
        rf = ((1 + m["irx"].fillna(0) / 100) ** (1 / 12) - 1) if "irx" in m else pd.Series(0.0, index=m.index)
        trend_up = px > px.rolling(10).mean()
        sig = wfr["verdict"]
        idx = grade_idx.intersection(px.dropna().index)
        if veh == "oil_fut":
            st.warning("CL=F is Yahoo's stitched front-month series. Its returns follow the SPOT-like price path and ignore the cost of rolling "
                       "(contango) or the gain from it (backwardation), so results are not what a futures holder earned. USO is the more realistic vehicle.")
        if len(idx) < 12:
            st.info(f"Only {len(idx)} graded months have {veh_name} prices (USO starts in 2006). Move the split or the start date.")
        else:
            w = strategy_weights(strat, sig, trend_up)
            r_s, w_s = run_backtest(px, rf, w, idx, bps)
            r_b, _ = run_backtest(px, rf, pd.Series(1.0, index=px.index), idx, 0)
            r_h, _ = run_backtest(px, rf, pd.Series(0.5, index=px.index), idx, 0)
            common = r_s.index.intersection(r_b.index)
            rows = []
            for nm, r_ in ((f"Strategy: {strat}", r_s), (f"Buy & hold {veh_name}", r_b), (f"50% {veh_name} / 50% cash", r_h)):
                p_ = perf(r_.reindex(common), rf)
                rows.append({"Portfolio": nm, "Months": p_["Months"], "CAGR": pc(p_["CAGR"]), "Volatility": pc(p_["Vol"], False),
                             "Sharpe": "-" if pd.isna(p_["Sharpe"]) else f"{p_['Sharpe']:.2f}", "Max drawdown": pc(p_["MaxDD"])})
            show_df(st, pd.DataFrame(rows))
            f = go.Figure()
            for nm, r_ in (("Strategy", r_s), (f"Buy & hold {veh_name}", r_b), ("50 / 50", r_h)):
                f.add_scatter(x=common, y=(1 + r_.reindex(common)).cumprod().values, name=nm)
            f.update_layout(title=f"Growth of 1 on the {grade_name.lower()} months (after {bps} bps cost per 100% traded)", height=380, legend=dict(orientation="h", y=-0.2))
            show_plot(st, f)
            inv = w_s.reindex(common).mean()
            st.caption(f"Average exposure {inv:.0%}. The idle part earns T-bill returns. The strategy uses the EXPANDING-rules verdict (refitted every {RULE_STEP} months on known outcomes). "
                       f"Graded months: {common[0]:%b %Y} to {common[-1]:%b %Y}, about {max(1, len(common) // 12)} years: a short, single-regime sample. "
                       "A strategy that beats buy & hold here has still passed only one test; compare against the trend-only strategy to see whether the macro rules add anything.")

# ====================================================================== Explorer
with T["exp"]:
    st.subheader("History explorer")
    pick = st.selectbox("Indicator", chosen, format_func=lambda c: META_ALL[c][0], key="oil_pick")
    s_ = F_all[pick].dropna()
    k_ = 100.0 if META_ALL[pick][3] else 1.0
    f = go.Figure()
    f.add_scatter(x=s_.index, y=s_.values * k_, name=META_ALL[pick][0], line=dict(color="#1f5fbf", width=2))
    o_ = m["oil"].reindex(s_.index)
    f.add_scatter(x=o_.index, y=o_.values, name="WTI, USD/bbl (right axis, log)", yaxis="y2", line=dict(color="#999", width=1.5))
    f.update_layout(title=META_ALL[pick][0] + (" (%)" if META_ALL[pick][3] else ""), height=400,
                    yaxis2=dict(overlaying="y", side="right", showgrid=False, type="log"), legend=dict(orientation="h", y=-0.2))
    show_plot(st, f)
    st.markdown(f"**Outcome after {h} months by state (learn period, {target_name.lower()})**")
    sub = stats[stats["col"] == pick]
    if sub.empty:
        st.info("Too few months in each state to report (each state needs at least 8 learn months).")
    else:
        show_df(st, pd.DataFrame({"State": [state_label(pick, s) for s in sub["state"]], "Months": sub["n"], "Avg after": sub["avg"].map(pc),
                                  "Median": sub["median"].map(pc), "% positive": sub["win"].map(lambda v: pc(v, False)), "Edge vs all": sub["edge"].map(pc),
                                  "HAC t": sub["t"].map(lambda v: f"{v:+.1f}"), "FDR q": sub["q"].map(lambda v: f"{v:.2f}"),
                                  "Same sign in both halves": sub["consistent"].map(lambda v: "✔" if v else "✘")}))
    ctx = st.toggle("Show Covid / 2008 / 2014-16 / 2022 context", False, key="oil_ctx")
    if ctx:
        st.caption("2008: demand collapse after the price spike. 2014-16: OPEC's market-share war and the US shale surge. Apr 2020: Covid demand collapse, front-month WTI briefly negative. "
                   "2022: Russia's invasion of Ukraine, supply-disruption premium and SPR releases. Check whether an indicator's edge survives with these windows removed before trusting it.")

# ====================================================================== Data & coverage
with T["data"]:
    st.subheader("Data & coverage")
    st.caption("What downloaded, from when to when, and which indicators it feeds. A missing source only removes the indicators that need it.")
    rows = []
    for k, (desc, src) in SOURCES.items():
        col = "oil" if k == "oil" else k
        ok = col in m and m[col].notna().any()
        s_ = m[col].dropna() if ok else None
        rows.append({"Series": desc, "Source": src, "Status": "✅" if ok else "❌", "First month": f"{s_.index[0]:%b %Y}" if ok else "-",
                     "Last month": f"{s_.index[-1]:%b %Y}" if ok else "-"})
    show_df(st, pd.DataFrame(rows))
    st.markdown("**Indicators and where their history starts**")
    ir = [{"Pillar": META_ALL[c][4], "Indicator": META_ALL[c][0], "First month": f"{F_full[c].first_valid_index():%b %Y}",
           "In use": "✔" if c in chosen else "", "Textbook sign for oil": {1: "Positive", -1: "Negative", 0: "Mixed"}[SIGN[c]]}
          for c in F_full.columns]
    show_df(st, pd.DataFrame(ir).sort_values(["Pillar", "Indicator"]))
    st.markdown(f"""
**Manual series (optional).** Rig count (Baker Hughes) and OPEC+ production have no reliable free API. Put a file named `{MANUAL_FILE}` next to the app:

```
date,rigs,opecplus_prod
2023-01-31,168,41.2
2023-02-28,170,41.0
```
`date` is the observation month-end, `rigs` the oil rig count, `opecplus_prod` crude production in mb/d. Each value is treated as public 1 month + 15 days after its date.
Either column can be left out. The indicators `rigs_chg` and `opecplus_chg` then appear automatically (they start when your file starts, so they shorten the usable history if you include them).

**Notes.**
- Weekly EIA series (stocks, production, refinery utilisation, demand) are used from the Wednesday release, about 6 days after the week ends. Seasonal norms use only the previous 5 years.
- EIA STEO history, OECD leading indicators, GPR and the macro series are the current vintage, so revisions give a mild look-ahead. Treat their results with a little extra caution.
- The CFTC release date is a conservative estimate (as-of Tuesday + 4 days).
- Airline activity: BTS air passenger miles (about 2-month lag) and airline stocks are included. Daily TSA throughput only starts in 2019, too short for the rules, but it can be added through the manual file idea if you extend the code.
- Free freight data is limited: the BTS freight index is monthly and lagged, tanker equities are a market proxy. Baltic / container rate indices have no free reliable feed.
""")
