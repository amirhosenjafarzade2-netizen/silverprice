"""
Oil tab: Expectations (market-implied, published-model and model-implied forward-looking indicators), their link to oil, and a grade table.
Executed by oil_main.py. The indicators come from silver_expectations.py (shared engine); only the textbook signs differ for oil.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_XC = [c for c in EXP_KEYS if c in F_full.columns]
_ICON = {"Market-implied": "🟦", "Published model": "🟨", "Model-implied": "🟪"}
_STXT = {1: "Positive", -1: "Negative", 0: "Mixed"}
_ACTUAL = {"ex_ff12": ("fed_funds", "Actual Fed funds (today)"), "ex_y10": ("dgs10", "Actual 10Y yield (today)"),
           "ex_growth": ("gdp_yoy", "Actual GDP growth (latest)"), "ex_unrate": ("unrate_lvl", "Actual unemployment (latest)"),
           "ex_ind": ("indpro_yoy", "Actual industrial production YoY (latest)"), "ex_infl": ("cpi_yoy", "Actual CPI inflation YoY (latest)")}


def _fx(c, v, delta=False):
    if pd.isna(v):
        return "-"
    f = EXP_INFO[c]["fmt"]
    if f == "usd":
        return f"{v * 100:+.1f} pts" if delta else f"{v:+.1%}"
    if f == "prob":
        return f"{v:+.0f} pts" if delta else f"{v:.0f}%"
    if f == "chg":
        return f"{v:+.2f} pts"
    return f"{v:+.2f} pts" if delta else f"{v:.2f}%"


def _grade(m, E, idx, h=EXP_H):
    """Score each model/market expectation against what happened h months later versus 'nothing changes'. All values are point-in-time, so grading is out-of-sample."""
    def g(k):
        return m[k] if k in m else pd.Series(np.nan, index=m.index)
    ff = g("fed_funds").fillna(g("dgs3m"))
    usd = g("dollar")
    spec = {"ex_ff12": (ff, ff.shift(-h)), "ex_y10": (g("dgs10"), g("dgs10").shift(-h)), "ex_growth": (g("gdp_yoy"), g("gdp_yoy").shift(-h)),
            "ex_unrate": (g("unrate_lvl"), g("unrate_lvl").shift(-h)), "ex_usd": (pd.Series(0.0, index=m.index), usd.shift(-h) / usd - 1),
            "ex_ind": (g("indpro_yoy"), g("indpro_yoy").shift(-h))}
    rows = []
    for k, (anchor, real) in spec.items():
        if k not in E.columns:
            continue
        d = pd.concat([E[k], real, anchor], axis=1, keys=["p", "r", "a"]).reindex(idx).dropna()
        if len(d) < 24:
            continue
        mae_m, mae_n = (d["p"] - d["r"]).abs().mean(), (d["a"] - d["r"]).abs().mean()
        rows.append(dict(k=k, n=len(d), mae_m=mae_m, mae_n=mae_n, skill=1 - mae_m / mae_n if mae_n > 0 else np.nan,
                         ic=rank_ic(d["p"] - d["a"], d["r"] - d["a"])))
    return pd.DataFrame(rows)


with T["xpt"]:
    st.subheader("📈 Expectations: what rates, inflation, growth, jobs and the dollar are priced / projected to do over the next 12 months")
    st.caption("Ten forward-looking indicators, all **point-in-time**. 🟦 **Market-implied** = from today's Treasury curve, nothing fitted. 🟨 **Published model** = Cleveland Fed. "
               "🟪 **Model-implied** = walk-forward ridge estimate (no free point-in-time survey history exists). Oil lens: expected rates and inflation matter through the dollar and real yields, "
               "expected growth and industrial output through demand.")
    if not _XC:
        st.error("No expectation indicator could be built (the Treasury / FRED series did not download). Click 'Refresh data' or add a FRED_API_KEY.")
    else:
        last = F_full[_XC].dropna(how="all").index[-1]
        st.markdown(f"**Latest read: {last:%b %Y}.** Change shown versus 3 months earlier.")
        for row_cols in (_XC[0:4], _XC[4:7], _XC[7:10]):
            for col_, c in zip(st.columns(len(row_cols)), row_cols):
                s_ = F_full[c].dropna()
                box = card(col_)
                if s_.empty:
                    box.markdown(f"**{META_ALL[c][0]}**")
                    box.caption("not available")
                    continue
                prev = F_full[c].shift(3).loc[s_.index[-1]]
                box.metric(META_ALL[c][0], _fx(c, s_.iloc[-1]), None if pd.isna(prev) else _fx(c, s_.iloc[-1] - prev, True), delta_color="off")
                box.caption(f"{_ICON[EXP_INFO[c]['method']]} {EXP_INFO[c]['method']}")

        def val(c):
            s_ = F_full[c].dropna() if c in F_full else pd.Series(dtype=float)
            return s_.iloc[-1] if len(s_) else np.nan

        lines = []
        if pd.notna(val("ex_ff_chg")):
            d_ff = val("ex_ff_chg")
            lines.append(f"**Fed:** the curve prices {'cuts' if d_ff <= -0.25 else ('hikes' if d_ff >= 0.25 else 'roughly no change')} ({d_ff:+.2f} pts over 12 months).")
        if pd.notna(val("ex_real")) and pd.notna(val("ex_infl")):
            lines.append(f"**Real policy rate in 12 months:** about {val('ex_real'):.2f}% (expected inflation {val('ex_infl'):.2f}%).")
        if pd.notna(val("ex_recess")):
            rp = val("ex_recess")
            lines.append(f"**Recession risk (yield-curve model):** {rp:.0f}% ({'low' if rp < 20 else ('moderate' if rp < 40 else 'elevated')}). For oil, a recession means demand loss.")
        if pd.notna(val("ex_ind")):
            lines.append(f"**Industrial demand (model-implied):** production growth about {val('ex_ind'):.1f}% in 12 months.")
        if pd.notna(val("ex_usd")):
            u_ = val("ex_usd")
            lines.append(f"**Dollar (model-implied):** {'stronger' if u_ >= 0.01 else ('weaker' if u_ <= -0.01 else 'about flat')} ({u_:+.1%}; a low-confidence kind of forecast).")
        if lines:
            st.markdown("\n".join(f"- {x}" for x in lines))

        sup = uns = neu = mix = 0
        for c in _XC:
            sg = OIL_EXP_SIGN[c]
            p_ = P_full[c].dropna().iloc[-1] if c in P_full and P_full[c].notna().any() else np.nan
            if sg == 0:
                mix += 1
            elif pd.isna(p_):
                continue
            elif (p_ >= 2 / 3 and sg > 0) or (p_ <= 1 / 3 and sg < 0):
                sup += 1
            elif (p_ >= 2 / 3 and sg < 0) or (p_ <= 1 / 3 and sg > 0):
                uns += 1
            else:
                neu += 1
        st.markdown(f"**Textbook read for oil:** {sup} expectation(s) point in oil's favor, {uns} against, {neu} in the middle of their range, {mix} without a clear sign. "
                    "'Textbook' = lower real rates, a weaker dollar, higher inflation expectations, stronger growth and industrial demand and lower recession risk help oil. "
                    "A checklist, not evidence: the table below shows what THIS sample says.")

        st.markdown("#### History")
        pk = st.selectbox("Indicator", _XC, format_func=lambda c: META_ALL[c][0], key="oil_xpt_pick")
        s_ = F_full[pk].loc[start:].dropna()
        if s_.empty:
            st.info("No history for this indicator after the chosen start date.")
        else:
            k_ = 100.0 if EXP_INFO[pk]["fmt"] == "usd" else 1.0
            f = go.Figure()
            f.add_scatter(x=s_.index, y=s_.values * k_, name=META_ALL[pk][0], line=dict(color="#1f5fbf", width=2.5))
            if pk in _ACTUAL and _ACTUAL[pk][0] in m:
                a_ = m[_ACTUAL[pk][0]].reindex(s_.index)
                f.add_scatter(x=a_.index, y=a_.values, name=_ACTUAL[pk][1], line=dict(color="#c0392b", dash="dot", width=1.5))
            o_ = m["oil"].reindex(s_.index)
            f.add_scatter(x=o_.index, y=o_.values, name="WTI, USD/bbl (right axis, log)", yaxis="y2", line=dict(color="#999", width=1.5))
            f.update_layout(title=META_ALL[pk][0], height=400, yaxis2=dict(overlaying="y", side="right", showgrid=False, type="log"),
                            legend=dict(orientation="h", y=-0.2))
            show_plot(st, f)
            st.caption(f"{_ICON[EXP_INFO[pk]['method']]} **{EXP_INFO[pk]['method']}.** {EXP_INFO[pk]['how']}")

        st.markdown(f"#### Do these expectations line up with oil's {h}-month outcome? ({target_name.lower()})")
        rows = []
        for c in _XC:
            x_ = F_full[c].reindex(fwd.index)
            xl, yl, xt, yt = x_.loc[learn_idx], fwd.loc[learn_idx], x_.loc[unseen_idx], fwd.loc[unseen_idx]
            ic_l, ic_t = rank_ic(xl, yl), rank_ic(xt, yt)
            dl = pd.concat([xl, yl], axis=1).dropna()
            t_l = nw_t(dl.iloc[:, 1].rank().values, dl.iloc[:, 0].rank().values, h) if len(dl) > 8 else np.nan
            same = bool(pd.notna(ic_l) and pd.notna(ic_t) and np.sign(ic_l) == np.sign(ic_t))
            status = "-" if (pd.isna(ic_l) or pd.isna(ic_t)) else ("❌ Flips on unseen data" if not same else ("✅ Consistent and significant" if abs(t_l) >= 2 else "🟡 Consistent but weak"))
            sg = OIL_EXP_SIGN[c]
            rows.append({"Indicator": META_ALL[c][0], "Method": f"{_ICON[EXP_INFO[c]['method']]} {EXP_INFO[c]['method']}", "Learn months": len(dl),
                         "Rank IC (learn)": "-" if pd.isna(ic_l) else f"{ic_l:+.2f}", f"Rank IC ({unseen_lbl})": "-" if pd.isna(ic_t) else f"{ic_t:+.2f}",
                         "HAC t (learn)": "-" if pd.isna(t_l) else f"{t_l:+.1f}", "Status": status, "Textbook sign": _STXT[sg],
                         "Matches textbook?": "n/a" if sg == 0 or pd.isna(ic_l) else ("✔" if np.sign(ic_l) == sg else "✘")})
        show_df(st, pd.DataFrame(rows))
        st.caption("Expectations built from the same curve (Fed path, real rate, 10Y, recession) are strongly correlated with each other and with the dollar / real-yield indicators, "
                   "so they are not independent votes. Ten indicators are tested here, so a lone ✅ can be luck.")

        st.markdown("#### Are the expectations any good? (what actually happened 12 months later)")
        gr = _grade(m, F_full[_XC], ev_r)
        if gr.empty:
            st.info("Not enough overlapping months to grade the expectations.")
        else:
            nm = {"ex_ff12": "Fed funds, 12M ahead", "ex_y10": "10Y yield, 12M ahead", "ex_growth": "GDP growth", "ex_unrate": "Unemployment rate",
                  "ex_usd": "Dollar 12M change", "ex_ind": "Industrial production growth"}
            show_df(st, pd.DataFrame({
                "Expectation": [f"{nm[k]} {_ICON[EXP_INFO[k]['method']]}" for k in gr["k"]], "Months graded": gr["n"],
                "Avg error, expectation": gr["mae_m"].map(lambda v: f"{v:.3f}"), "Avg error, 'nothing changes'": gr["mae_n"].map(lambda v: f"{v:.3f}"),
                "Skill vs 'nothing changes'": gr["skill"].map(pc), "IC of expected vs realised change": gr["ic"].map(lambda v: "-" if pd.isna(v) else f"{v:+.2f}")}))
            beat = int((gr["skill"] > 0).sum())
            (st.success if beat >= len(gr) * 0.6 else st.warning)(
                f"{beat} of {len(gr)} expectations had a smaller average error than 'nothing changes'. Windows overlap, so independent observations are roughly months ÷ 12. "
                + ("Where skill is negative, treat the indicator as a description of the model's view, not as a forecast." if beat < len(gr) else ""))
            st.caption("Recession probability and expected inflation are not graded here (they need official recession dates / are published series).")
