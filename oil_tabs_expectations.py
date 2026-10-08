"""
Oil tabs: Expectations (market-implied, published-model and model-implied forward-looking indicators).
Executed by oil_main.py (all config, helpers and results are available as globals).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_XC = [c for c in EXP_KEYS if c in F_full.columns]
_METH_ICON = {"Market-implied": "🟦", "Published model": "🟨", "Model-implied": "🟪"}
_SIGN_TXT = {1: "Positive", -1: "Negative", 0: "Mixed"}
# reference line shown next to each indicator's history: what the same quantity actually is today
_ACTUAL = {"ex_ff12": ("fed_funds", "Actual Fed funds (today)"), "ex_y10": ("dgs10", "Actual 10Y yield (today)"),
           "ex_growth": ("gdp_yoy", "Actual GDP growth (latest)"), "ex_unrate": ("unrate_lvl", "Actual unemployment (latest)"),
           "ex_ind": ("indpro_yoy", "Actual industrial production YoY (latest)"), "ex_infl": ("cpi_yoy", "Actual CPI inflation YoY (latest)"),
           "ex_demand": ("us_dem_yoy", "Actual US oil demand growth (latest)")}


def _fmt(c, v, delta=False):
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


def grade_expectations(m, E, idx, h=EXP_H):
    """Score each expectation against what actually happened h months later, versus a naive 'nothing changes' anchor.
    Every expectation is point-in-time (model ones are walk-forward), so grading on all months in idx is out-of-sample.
    Skill > 0 = smaller average error than the naive anchor. IC = rank correlation of the EXPECTED change with the REALISED change."""
    ff = (m["fed_funds"] if "fed_funds" in m else pd.Series(np.nan, index=m.index)).fillna(m["dgs3m"] if "dgs3m" in m else np.nan)

    def g(k):
        return m[k] if k in m else pd.Series(np.nan, index=m.index)

    usd = g("dollar")
    spec = {"ex_ff12": (ff, ff.shift(-h)), "ex_y10": (g("dgs10"), g("dgs10").shift(-h)),
            "ex_growth": (g("gdp_yoy"), g("gdp_yoy").shift(-h)), "ex_unrate": (g("unrate_lvl"), g("unrate_lvl").shift(-h)),
            "ex_usd": (pd.Series(0.0, index=m.index), usd.shift(-h) / usd - 1), "ex_ind": (g("indpro_yoy"), g("indpro_yoy").shift(-h)),
            "ex_demand": (g("us_dem_yoy").clip(-25, 25), g("us_dem_yoy").clip(-25, 25).shift(-h)),
            # curve-implied change is a 3-month quantity (contract 4 vs contract 1), so it is graded against oil's next 3 months, anchor = no change
            "ex_curve": (pd.Series(0.0, index=m.index), (g("oil").shift(-3) / g("oil") - 1) * 100)}
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
    st.subheader("📈 Expectations: what rates, inflation, growth, jobs, the dollar, oil demand and the oil futures curve are priced / projected to do")
    st.caption("Twelve forward-looking indicators, all **point-in-time** (each month uses only data that was public then). "
               "🟦 **Market-implied** = computed from today's Treasury curve and WTI futures curve, nothing fitted. 🟨 **Published model** = Cleveland Fed. "
               "🟪 **Model-implied** = a walk-forward ridge estimate, because no free point-in-time survey history exists for growth, unemployment, the dollar, industrial demand or oil demand. "
               "Model-implied values are compressions of the other macro indicators, so they add convenience, not new information. The grade table below tells you whether they beat 'nothing changes'.")
    miss = [k for k in EXP_FRED if k in failed]
    if miss:
        st.warning("Some inputs failed to download (" + ", ".join(EXP_FRED[k] for k in miss) + "), so the indicators that need them are missing or approximated.")
    if not _XC:
        st.error("No expectation indicator could be built. The Treasury / FRED series did not download. Click 'Refresh data' in the sidebar or add a FRED_API_KEY.")
    else:
        if "expinf1y" not in m or not m["expinf1y"].notna().any():
            st.info("Cleveland Fed expected inflation (EXPINF1YR) was unavailable, so expected inflation uses the 10Y breakeven instead (a longer horizon).")
        last = F_full[_XC].dropna(how="all").index[-1]
        st.markdown(f"**Latest read: {last:%b %Y}.** Change shown versus 3 months earlier.")

        # ---------------- headline cards
        for row_cols in [_XC[i:i + 4] for i in range(0, len(_XC), 4)]:
            cols_ = st.columns(len(row_cols))
            for col_, c in zip(cols_, row_cols):
                s_ = F_full[c].dropna()
                box = card(col_)
                if s_.empty:
                    box.markdown(f"**{META_ALL[c][0]}**")
                    box.caption("not available")
                    continue
                now_v = s_.iloc[-1]
                prev = F_full[c].shift(3).loc[s_.index[-1]]
                box.metric(META_ALL[c][0], _fmt(c, now_v), None if pd.isna(prev) else _fmt(c, now_v - prev, True), delta_color="off")
                box.caption(f"{_METH_ICON[EXP_INFO[c]['method']]} {EXP_INFO[c]['method']}")

        # ---------------- plain-English read
        def val(c):
            s_ = F_full[c].dropna() if c in F_full else pd.Series(dtype=float)
            return s_.iloc[-1] if len(s_) else np.nan

        lines = []
        d_ff = val("ex_ff_chg")
        if pd.notna(d_ff):
            lines.append(f"**Fed:** the curve prices {'cuts' if d_ff <= -0.25 else ('hikes' if d_ff >= 0.25 else 'roughly no change')} "
                         f"({d_ff:+.2f} pts over 12 months, to about {val('ex_ff12'):.2f}%).")
        if pd.notna(val("ex_real")):
            lines.append(f"**Real policy rate in 12 months:** about {val('ex_real'):.2f}% (expected inflation {val('ex_infl'):.2f}%).")
        rp = val("ex_recess")
        if pd.notna(rp):
            lines.append(f"**Recession risk (yield-curve model):** {rp:.0f}% ({'low' if rp < 20 else ('moderate' if rp < 40 else 'elevated')}).")
        if pd.notna(val("ex_growth")) and pd.notna(val("ex_unrate")):
            lines.append(f"**Economy (model-implied):** growth about {val('ex_growth'):.1f}%, unemployment about {val('ex_unrate'):.1f}% in 12 months.")
        u_ = val("ex_usd")
        if pd.notna(u_):
            lines.append(f"**Dollar (model-implied):** {'stronger' if u_ >= 0.01 else ('weaker' if u_ <= -0.01 else 'about flat')} ({u_:+.1%} over 12 months; low-confidence kind of forecast).")
        if lines:
            st.markdown("\n".join(f"- {x}" for x in lines))

        # textbook tally
        sup = uns = neu = mix = 0
        for c in _XC:
            sg = EXP_INFO[c]["oil"]
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
        st.markdown(f"**Textbook read for oil:** {sup} expectation(s) point in oil's favor, {uns} against, {neu} in the middle of their range, {mix} have no clear textbook sign. "
                    "'Textbook' = the usual story (stronger growth and fuel demand, a weaker dollar, lower real rates, higher inflation expectations and a backwardated curve help oil). "
                    "It is a checklist, not evidence: the table further down shows what the data in THIS sample says.")

        # ---------------- history chart
        st.markdown("#### History")
        pick = st.selectbox("Indicator", _XC, format_func=lambda c: META_ALL[c][0], key="xpt_pick")
        s_ = F_full[pick].loc[start:].dropna()
        if s_.empty:
            st.info("No history for this indicator after the chosen start date.")
        else:
            k_ = 100.0 if EXP_INFO[pick]["fmt"] == "usd" else 1.0
            f = go.Figure()
            f.add_scatter(x=s_.index, y=s_.values * k_, name=META_ALL[pick][0], line=dict(color="#1f5fbf", width=2.5))
            if pick in _ACTUAL and _ACTUAL[pick][0] in m:
                a_ = m[_ACTUAL[pick][0]].reindex(s_.index)
                f.add_scatter(x=a_.index, y=a_.values, name=_ACTUAL[pick][1], line=dict(color="#c0392b", dash="dot", width=1.5))
            sv = m["oil"].reindex(s_.index)
            f.add_scatter(x=sv.index, y=sv.values, name="Oil, USD/bbl (right axis, log)", yaxis="y2", line=dict(color="#999", width=1.5))
            f.update_layout(title=META_ALL[pick][0] + (" (%)" if EXP_INFO[pick]["fmt"] == "usd" else ""), height=420,
                            yaxis2=dict(overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=-0.2))
            show_plot(st, f)
            st.caption(f"{_METH_ICON[EXP_INFO[pick]['method']]} **{EXP_INFO[pick]['method']}.** {EXP_INFO[pick]['how']}")

        # ---------------- link to oil
        st.markdown(f"#### Do these expectations line up with oil's {h}-month outcome? ({target_name.lower()})")
        rows = []
        for c in _XC:
            x_ = F_full[c].reindex(fwd.index)
            xl, yl, xt, yt = x_.loc[learn_idx], fwd.loc[learn_idx], x_.loc[unseen_idx], fwd.loc[unseen_idx]
            ic_l, ic_t = rank_ic(xl, yl), rank_ic(xt, yt)
            dl = pd.concat([xl, yl], axis=1).dropna()
            t_l = nw_t(dl.iloc[:, 1].rank().values, dl.iloc[:, 0].rank().values, h) if len(dl) > 8 else np.nan
            same = bool(pd.notna(ic_l) and pd.notna(ic_t) and np.sign(ic_l) == np.sign(ic_t))
            if pd.isna(ic_l) or pd.isna(ic_t):
                status = "-"
            elif not same:
                status = "❌ Flips on unseen data"
            elif abs(t_l) >= 2:
                status = "✅ Consistent and significant"
            else:
                status = "🟡 Consistent but weak"
            sg = EXP_INFO[c]["oil"]
            rows.append({"Indicator": META_ALL[c][0], "Method": f"{_METH_ICON[EXP_INFO[c]['method']]} {EXP_INFO[c]['method']}", "Learn months": len(dl),
                         "Rank IC (learn)": "-" if pd.isna(ic_l) else f"{ic_l:+.2f}", f"Rank IC ({unseen_lbl})": "-" if pd.isna(ic_t) else f"{ic_t:+.2f}",
                         "HAC t (learn)": "-" if pd.isna(t_l) else f"{t_l:+.1f}", "Status": status, "Textbook sign": _SIGN_TXT[sg],
                         "Matches textbook?": "n/a" if sg == 0 or pd.isna(ic_l) else ("✔" if np.sign(ic_l) == sg else "✘")})
        show_df(st, pd.DataFrame(rows))
        st.caption("Rank IC = rank correlation between the indicator and oil's outcome (positive: higher indicator, better outcome). 'Unseen' follows the sidebar mode "
                   "(validation only in Research mode, validation + final test when revealed). Twelve indicators are tested here, so a lone ✅ can be luck; the rules and ML tabs apply the full multiple-testing discipline. "
                   "Expectations that are formed from the same curve (Fed path, real rate, 10Y, recession) are strongly correlated with each other and with the existing curve and real-yield indicators.")

        # ---------------- grade the expectations themselves
        st.markdown("#### Are the expectations any good? (what actually happened 12 months later)")
        gr = grade_expectations(m, F_full[_XC], ev_r)
        if gr.empty:
            st.info("Not enough overlapping months to grade the expectations.")
        else:
            nm = {"ex_ff12": "Fed funds, 12M ahead", "ex_y10": "10Y yield, 12M ahead", "ex_growth": "GDP growth", "ex_unrate": "Unemployment rate",
                  "ex_usd": "Dollar 12M change", "ex_ind": "Industrial production growth",
                  "ex_demand": "US oil demand growth", "ex_curve": "Oil change implied by curve (3M, % pts)"}
            show_df(st, pd.DataFrame({
                "Expectation": [f"{nm[k]} {_METH_ICON[EXP_INFO[k]['method']]}" for k in gr["k"]], "Months graded": gr["n"],
                "Avg error, expectation": gr["mae_m"].map(lambda v: f"{v:.3f}"), "Avg error, 'nothing changes'": gr["mae_n"].map(lambda v: f"{v:.3f}"),
                "Skill vs 'nothing changes'": gr["skill"].map(lambda v: pc(v)), "IC of expected vs realised change": gr["ic"].map(lambda v: "-" if pd.isna(v) else f"{v:+.2f}")}))
            beat = int((gr["skill"] > 0).sum())
            (st.success if beat >= len(gr) * 0.6 else st.warning)(
                f"{beat} of {len(gr)} expectations had a smaller average error than 'nothing changes' (graded on months up to the end of {unseen_lbl if not REVEAL else 'the final test'}, "
                f"outcomes realised 12 months later; windows overlap, so independent observations are roughly months ÷ 12). "
                + ("Where skill is negative, treat the indicator as a description of the model's view, not as a forecast." if beat < len(gr) else ""))
            st.caption("Error units: % points for rates, growth and unemployment, a fraction for the dollar. The anchor is today's value (or zero change for the dollar). "
                       "The recession probability is not graded here because it needs official recession dates; expected inflation is a published series and is not graded either. The curve-implied change is graded over 3 months (contract 4 vs contract 1), the rest over 12.")

        # ---------------- definitions
        with st.expander("📖 Definitions, sources and limits"):
            show_df(st, pd.DataFrame([{"Indicator": META_ALL[c][0], "Method": f"{_METH_ICON[EXP_INFO[c]['method']]} {EXP_INFO[c]['method']}",
                                       "How it is built": EXP_INFO[c]["how"], "Source series": EXP_INFO[c]["src"],
                                       "First month": F_full[c].first_valid_index().strftime("%b %Y") if F_full[c].notna().any() else "-"} for c in _XC]))
            st.markdown("""
- **Why not surveys?** The Survey of Professional Forecasters, Blue Chip, and the IEA / OPEC / EIA monthly outlooks are the real 'expected growth / unemployment / oil demand' sources, but their history is not available as a free, clean, point-in-time feed. The model-implied versions are an honest substitute, not the same thing.
- **Term premium.** Forward rates mix expectations with a term premium, so 'expected 10Y yield' and 'expected Fed funds' are market-implied, not pure forecasts. The same is true of the oil futures curve: it mixes expected spot moves with storage cost and risk premia, and futures prices have historically been poor forecasts of spot oil.
- **Publication lags.** Cleveland Fed inflation expectations are used from the 15th of the following month, GDP about 3 months + 30 days after the quarter starts, claims 5 days after the week, Treasury yields the same day.
- **Zero lower bound.** When rates are near zero the forward-implied Fed path is floored at 0 and carries little information.
- **Overlap with other indicators.** The expectation indicators are correlated with the curve, real-yield and dollar indicators already in the app. Turn on factor compression (sidebar) if you add them to the rules, so they do not count as extra independent votes.
- **Selection bias.** Twelve more indicators mean more chances for a fluke. Judge them by the unseen-data columns and the final test, not by how good one row looks.
""")


# ------------------------------------------------------------------ options market (historical OVX indicators + live USO snapshot)
import os  # noqa: E402
import yfinance as yf  # noqa: E402

OPT_LOG = os.path.join(_HERE0, "oil_options_log.csv")


@st.cache_data(show_spinner=False, ttl=3600)
def options_snapshot(symbol="USO"):
    """Live USO option-chain readings. A SINGLE snapshot: Yahoo has no option history, so this cannot be graded or used in the rules.
    atm_near / atm_far = at-the-money implied vol (%) for a ~1-2 month and a ~3-6 month expiry; term = far - near (negative = near-term stress);
    skew = IV of a put ~10% out of the money minus IV of a call ~10% out (positive = downside hedging, negative = upside / supply-scare demand);
    pc_vol / pc_oi = put-to-call ratios of volume and open interest in the near expiry."""
    t = yf.Ticker(symbol)
    spot = float(t.history(period="5d")["Close"].dropna().iloc[-1])
    exps = list(t.options)
    if not exps:
        raise ValueError("no option expiries returned")
    today = pd.Timestamp.today().normalize()
    dte = {e: (pd.Timestamp(e) - today).days for e in exps}
    near = next((e for e in exps if 14 <= dte[e] <= 60), exps[0])
    far = next((e for e in exps if 75 <= dte[e] <= 200), None)

    def iv_at(df, k):
        d = df[(df["impliedVolatility"] > 0.01) & (df["openInterest"].fillna(0) + df["volume"].fillna(0) > 0)]
        if d.empty:
            return np.nan
        return float(d.iloc[(d["strike"] - k).abs().argsort().iloc[0]]["impliedVolatility"]) * 100

    oc = t.option_chain(near)
    c, p = oc.calls, oc.puts
    out = dict(symbol=symbol, spot=spot, near=near, near_dte=dte[near], far=far, far_dte=dte.get(far),
               atm_near=np.nanmean([iv_at(c, spot), iv_at(p, spot)]), skew=iv_at(p, 0.9 * spot) - iv_at(c, 1.1 * spot),
               pc_vol=float(p["volume"].fillna(0).sum() / max(c["volume"].fillna(0).sum(), 1)),
               pc_oi=float(p["openInterest"].fillna(0).sum() / max(c["openInterest"].fillna(0).sum(), 1)), atm_far=np.nan)
    if far:
        oc2 = t.option_chain(far)
        out["atm_far"] = np.nanmean([iv_at(oc2.calls, spot), iv_at(oc2.puts, spot)])
    out["term"] = out["atm_far"] - out["atm_near"] if pd.notna(out["atm_far"]) else np.nan
    return out


def _opt_read(o):
    """Plain-English reading of a snapshot."""
    r = []
    if pd.notna(o["term"]):
        r.append("Implied vol is **higher in the near expiry than the far one** (inverted): the market prices a near-term shock, typical of disruption scares." if o["term"] < -1 else
                 "Implied vol rises with expiry (normal shape): no sign of an acute near-term scare." if o["term"] > 1 else "The implied-vol term structure is flat.")
    if pd.notna(o["skew"]):
        r.append("**Calls are richer than puts** (negative skew): demand for upside protection, the pattern seen when supply disruption is feared." if o["skew"] < -1 else
                 "**Puts are richer than calls**: investors are paying for downside protection (demand / growth worries)." if o["skew"] > 1 else "Put and call wings are priced about equally.")
    r.append(f"Put/call ratio {o['pc_vol']:.2f} by volume and {o['pc_oi']:.2f} by open interest (above 1 = more puts).")
    return r


with T["xpt"]:
    st.markdown("---")
    st.markdown("#### 🎯 Options market: what traders are paying for (geopolitics, demand shocks)")
    st.caption("**Historical** part (indicators in the Risk & geopolitics pillar): OVX, the options-implied volatility of oil, its gap to realized volatility, and its 3-month change. "
               "**Live** part: a snapshot of USO options. Yahoo keeps NO option history, so the snapshot cannot be backtested or graded. Save it with the button and, after a year or two, "
               "the log becomes a usable indicator.")
    ov = F_full["ovx"].dropna() if "ovx" in F_full else pd.Series(dtype=float)
    if len(ov):
        pov = P_full["ovx"].dropna().iloc[-1] if "ovx" in P_full and P_full["ovx"].notna().any() else np.nan
        a_, b_, c_ = st.columns(3)
        a_.metric("OVX (implied vol of oil)", f"{ov.iloc[-1]:.1f}", None if pd.isna(pov) else f"{pov:.0%} percentile vs past", delta_color="off")
        vr = F_full["ovx_vrp"].dropna() if "ovx_vrp" in F_full else pd.Series(dtype=float)
        if len(vr):
            b_.metric("Implied minus realized vol", f"{vr.iloc[-1]:+.1f} pts", help="Positive = options price more movement than recently happened: a fear premium.")
        gp = F_full["gpr"].dropna() if "gpr" in F_full else pd.Series(dtype=float)
        if len(gp):
            c_.metric("Geopolitical risk index", f"{gp.iloc[-1]:.0f}")
    else:
        st.info("OVX history is unavailable.")
    try:
        snap = options_snapshot("USO")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(f"USO ATM implied vol ({snap['near_dte']}d)", f"{snap['atm_near']:.1f}%")
        s2.metric("Term (far minus near)", "-" if pd.isna(snap["term"]) else f"{snap['term']:+.1f} pts")
        s3.metric("Put skew minus call skew", "-" if pd.isna(snap["skew"]) else f"{snap['skew']:+.1f} pts")
        s4.metric("Put / call (volume)", f"{snap['pc_vol']:.2f}")
        st.markdown("\n".join(f"- {x}" for x in _opt_read(snap)))
        if st.button("💾 Save today's snapshot to the log", key="oil_opt_save"):
            row = pd.DataFrame([dict(date=pd.Timestamp.today().normalize(), spot=snap["spot"], atm_near=snap["atm_near"], atm_far=snap["atm_far"],
                                     skew=snap["skew"], pc_vol=snap["pc_vol"], pc_oi=snap["pc_oi"])])
            old = pd.read_csv(OPT_LOG, parse_dates=["date"]) if os.path.exists(OPT_LOG) else pd.DataFrame()
            pd.concat([old, row]).drop_duplicates("date", keep="last").to_csv(OPT_LOG, index=False)
            st.success("Saved.")
        if os.path.exists(OPT_LOG):
            st.caption(f"Log: {len(pd.read_csv(OPT_LOG))} snapshot(s) in `oil_options_log.csv`. About 24 month-end rows are needed before it can feed the rules.")
    except Exception as e:  # noqa: BLE001
        snap = None
        st.info(f"Live USO option data could not be loaded ({type(e).__name__}). The historical OVX-based indicators still work.")
    st.caption("Limits: USO options are options on an ETF (they carry roll and ETF effects), not on WTI futures. Yahoo implied vols can be stale for illiquid strikes. "
               "CME's WTI futures options (the real source of skew history) are not free. Treat the snapshot as context, not a signal.")
