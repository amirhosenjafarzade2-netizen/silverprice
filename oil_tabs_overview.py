"""
Oil tabs: Dashboard, Guide, Environments, Indicator ranking, Out-of-sample.
Executed by oil_main.py (all config, helpers and results are available as globals).
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_VCOL = {FAV: "#2e8b57", NEU: "#999999", UNF: "#c0392b"}

# ====================================================================== Dashboard
with T["dash"]:
    st.subheader(f"{ICON[v_now]} Historically {v_now.lower()} environment for oil ({now:%b %Y})")
    wv = wfr["verdict"].dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Single-fit rules (in-sample)", v_now, f"score {sc_now:+.1f}", delta_color="off")
    c2.metric("Expanding rules (out-of-sample)", wv.iloc[-1] if len(wv) else "-", help="Refitted every 6 months on past outcomes only. This is the honest signal.")
    c3.metric("WTI, latest month-end", f"${m['oil'].iloc[-1]:.1f}")
    c4.metric("Rules in force", f"{len(good)} good / {len(bad)} bad")
    st.caption(f"'Favorable' means: in months with this kind of macro picture, the {h}-month-forward outcome ({target_name.lower()}) was historically better than usual. "
               "A description of the past, not a forecast. The single-fit verdict is graded in-sample on the learn months, so judge the signal by the Out-of-sample tab.")

    # price chart with the single-fit verdict as colored dots
    f = go.Figure()
    f.add_scatter(x=m["oil"].loc[start:].index, y=m["oil"].loc[start:].values, name="WTI, USD/bbl", line=dict(color="#555", width=1.5))
    for v_, col_ in _VCOL.items():
        ix = verdict.index[verdict == v_]
        f.add_scatter(x=ix, y=m["oil"].reindex(ix).values, mode="markers", name=v_, marker=dict(color=col_, size=5))
    f.update_layout(title="WTI with the environment verdict (single-fit rules)", yaxis_type="log", height=380, legend=dict(orientation="h", y=-0.2))
    show_plot(st, f)

    # which pillars push the current verdict
    contrib, fired = {}, []
    for rules, sgn in ((good, 1), (bad, -1)):
        for (c, s_), w in rules.items():
            if S.loc[now, c] == s_:
                pl = META_ALL[c][4]
                contrib[pl] = contrib.get(pl, 0.0) + sgn * w
                fired.append({"Pillar": pl, "Indicator": META_ALL[c][0], "Now": fmt_val(c, F_ok.loc[now, c]), "State": state_label(c, s_),
                              "Effect on oil": "🟢 Favorable" if sgn > 0 else "🔴 Unfavorable", "Weight": f"{w:.1f}"})
    left, right = st.columns([1, 1])
    with left:
        if contrib:
            cs = pd.Series(contrib).sort_values()
            fb = go.Figure(go.Bar(x=cs.values, y=cs.index, orientation="h", marker_color=[_VCOL[FAV] if v > 0 else _VCOL[UNF] for v in cs.values]))
            fb.update_layout(title="Net score by pillar (rules firing today)", height=320, margin=dict(l=10, r=10, t=40, b=10))
            show_plot(st, fb)
        else:
            st.info("No rule is firing today (all rules sit in states the indicators are not in).")
    with right:
        rows = []
        for c in chosen:
            p_ = P_full[c].dropna().iloc[-1] if P_full[c].notna().any() else np.nan
            rows.append({"Pillar": META_ALL[c][4], "Indicator": META_ALL[c][0], "Now": fmt_val(c, F_ok.loc[now, c]),
                         "Percentile vs past": "-" if pd.isna(p_) else f"{p_:.0%}", "Reads as": state_label(c, S.loc[now, c]) if pd.notna(S.loc[now, c]) else "-"})
        show_df(st, pd.DataFrame(rows).sort_values(["Pillar", "Indicator"]))
    if fired:
        st.markdown("**Rules firing today**")
        show_df(st, pd.DataFrame(fired))

# ====================================================================== Guide
with T["guide"]:
    st.subheader("How to read this app")
    st.markdown(f"""
**Question.** Which macro environments were historically good or bad for WTI over the next {h} months, and does that survive out-of-sample?

**Pillars and what they capture**
- **Demand & activity:** US / global industrial surveys, copper, EM and China stocks, US product demand, driving, EIA world demand and the world supply-minus-demand balance.
- **Supply (OPEC & US):** OPEC production and spare capacity, US crude output, drilling activity (rig-count proxy), optional OPEC+ and Baker Hughes rigs from `oil_manual_series.csv`.
- **Inventories & refining:** crude, gasoline, distillate and Cushing stocks versus the seasonal norm of the previous 5 years, SPR, refinery utilisation versus season, the 3-2-1 crack spread.
- **Market structure & positioning:** futures curve slope (backwardation / contango), momentum, price versus CPI-adjusted history, Brent-WTI, speculative positioning, oil volatility, energy equities.
- **Dollar & monetary:** dollar trend, real yields, breakevens, Fed path, curve, inflation, M2, financial conditions, credit.
- **Risk & geopolitics:** VIX, stocks, Sahm gauge, the Caldara-Iacoviello geopolitical risk index, global policy uncertainty.
- **Transport, freight & air travel:** freight index, tanker stocks, air passenger miles, airline stocks (the Covid-type demand shock shows up here first).
- **Expectations:** expected Fed funds, real rate, inflation, growth, unemployment, dollar and recession probability (shared engine with the silver app).

**Method (same as the silver app).** Every indicator is re-dated to when it was public. Each month is labelled Low / Mid / High. Rules are fitted on the LEARN months,
tested with HAC t-stats (overlapping windows) and FDR q-values, and only kept if the effect appears in both halves. Rules are chosen on VALIDATION and graded once on the hidden FINAL TEST.
The expanding rules are refitted every {RULE_STEP} months using only outcomes already known at that time.

**Oil-specific cautions**
- Oil is driven by supply shocks that no macro indicator sees coming (wars, OPEC decisions). Expect weaker, less stable signals than for a purely macro-driven asset.
- Inventories are compared with a seasonal norm, otherwise the calendar masquerades as a signal.
- A spot price return is not what an investor earns: futures earn or pay the roll (contango costs money). The backtest therefore defaults to USO; the Yahoo CL=F series does NOT capture roll yield.
- 2020 (negative prices, Covid) and 2008 are single episodes that can dominate any statistic. Look at the stability columns, not just the average.
- With ~25 years of monthly data and 6-month windows, independent observations number only a few dozen. Treat every result as a hypothesis.
""")

# ====================================================================== Environments
with T["env"]:
    st.subheader(f"Average outcome after {h} months, by environment")
    st.caption("Learn months are IN-sample (the rules were built on them, so they look good by construction). The grading months are the real test.")
    fdd_ = fdd if target in ("ret", "vol") else None
    tabs_ = [("Learn period (in-sample)", learn_idx, verdict), (f"{grade_name} (single-fit rules, unseen)", grade_idx, verdict),
             (f"{grade_name} (expanding rules, unseen)", grade_idx, wfr["verdict"])]
    bars = {}
    for nm, idx, vv in tabs_:
        d = summarize(fwd, vv, idx, h, fdd_)
        st.markdown(f"**{nm}**")
        show_df(st, fmt_summary(d, dd=fdd_ is not None))
        bars[nm] = d.set_index("Group")["Avg"]
    fb = go.Figure()
    for nm, s_ in bars.items():
        fb.add_bar(x=[FAV, NEU, UNF], y=[s_.get(k, np.nan) * 100 for k in (FAV, NEU, UNF)], name=nm)
    fb.update_layout(barmode="group", height=340, yaxis_title="Average outcome, %", legend=dict(orientation="h", y=-0.25))
    show_plot(st, fb)

# ====================================================================== Indicator ranking
with T["rank"]:
    st.subheader("Which indicators line up best with oil's outcome?")
    rows = []
    for c in chosen:
        xl, yl, xt, yt = F_ok[c].reindex(learn_idx), fwd.reindex(learn_idx), F_ok[c].reindex(unseen_idx), fwd.reindex(unseen_idx)
        ic_l, ic_t = rank_ic(xl, yl), rank_ic(xt, yt)
        dl = pd.concat([xl, yl], axis=1).dropna()
        t_l = nw_t(dl.iloc[:, 1].rank().values, dl.iloc[:, 0].rank().values, h) if len(dl) > 8 else np.nan
        rows.append(dict(c=c, ic_l=ic_l, ic_t=ic_t, t=t_l, p=math.erfc(abs(t_l) / math.sqrt(2)) if pd.notna(t_l) else np.nan))
    R = pd.DataFrame(rows)
    R["q"] = bh_q(R["p"].fillna(1.0).values)
    out = []
    for r in R.sort_values("t", key=lambda s: -s.abs()).itertuples():
        same = bool(pd.notna(r.ic_l) and pd.notna(r.ic_t) and np.sign(r.ic_l) == np.sign(r.ic_t))
        status = "-" if (pd.isna(r.ic_l) or pd.isna(r.ic_t)) else ("❌ Flips on unseen data" if not same else
                                                                  ("✅ Consistent and significant" if abs(r.t) >= 2 else "🟡 Consistent but weak"))
        sg = SIGN[r.c]
        out.append({"Pillar": META_ALL[r.c][4], "Indicator": META_ALL[r.c][0], "Rank IC (learn)": f"{r.ic_l:+.2f}" if pd.notna(r.ic_l) else "-",
                    f"Rank IC ({unseen_lbl})": f"{r.ic_t:+.2f}" if pd.notna(r.ic_t) else "-", "HAC t (learn)": f"{r.t:+.1f}" if pd.notna(r.t) else "-",
                    "FDR q": f"{r.q:.2f}", "Status": status,
                    "Textbook sign": {1: "Positive", -1: "Negative", 0: "Mixed"}[sg],
                    "Matches textbook?": "n/a" if sg == 0 or pd.isna(r.ic_l) else ("✔" if np.sign(r.ic_l) == sg else "✘")})
    show_df(st, pd.DataFrame(out))
    n_ok = sum(1 for o in out if o["Status"].startswith("✅"))
    st.caption(f"Rank IC = rank correlation between the indicator and oil's outcome (positive: higher indicator, better outcome). {len(out)} indicators are tested, "
               f"so about {max(1, round(len(out) * 0.05))} 'significant' ones are expected by luck alone; {n_ok} reached ✅ here. Prefer indicators that are ✅, "
               "have FDR q below 0.2, keep their sign on unseen data AND match an economic story.")

# ====================================================================== Out-of-sample
with T["test"]:
    st.subheader(f"Does the signal work on unseen months? ({grade_name.lower()})")
    res = {}
    for nm, vv in (("Single-fit rules", verdict), ("Expanding rules (refitted every %d months)" % RULE_STEP, wfr["verdict"])):
        d = summarize(fwd, vv, grade_idx, h).set_index("Group")
        st.markdown(f"**{nm}**")
        show_df(st, fmt_summary(d.reset_index()))
        if d.loc[FAV, "Months"] >= 3 and d.loc[UNF, "Months"] >= 3:
            res[nm] = (d.loc[FAV, "Avg"] - d.loc[UNF, "Avg"]) * 100
    for nm, sp_ in res.items():
        (st.success if sp_ > 0 else st.warning)(f"{nm}: Favorable minus Unfavorable = **{sp_:+.1f} pts** over {h} months on the {grade_name.lower()} period.")
    if not res:
        st.info("Too few Favorable or Unfavorable months in the grading period to compare.")
    st.caption(f"The {grade_name.lower()} period holds about {len(grade_idx)} months, but {h}-month windows overlap, so independent observations are roughly {max(1, len(grade_idx) // h)}. "
               "A positive spread on so few observations is encouraging, not proof. In Research mode nothing here touches the final test.")
