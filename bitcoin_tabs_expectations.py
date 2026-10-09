"""Bitcoin expectations tab: market-implied, published-model and model-implied forecasts.

This module is executed by ``bitcoin_main.py`` and intentionally consumes the
configuration, data frames, metadata and helper functions defined there.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_XC = [c for c in EXP_KEYS if c in F_full.columns]
_METH_ICON = {"Market-implied": "🟦", "Published model": "🟨", "Model-implied": "🟪"}
_SIGN_TXT = {1: "Positive", -1: "Negative", 0: "Mixed"}
_ACTUAL = {
    "ex_ff12": ("fed_funds", "Actual Fed funds (latest)"),
    "ex_y10": ("dgs10", "Actual 10Y yield (latest)"),
    "ex_growth": ("gdp_yoy", "Actual GDP growth (latest)"),
    "ex_unrate": ("unrate_lvl", "Actual unemployment (latest)"),
    "ex_m2": ("m2_yoy", "Actual M2 growth YoY (latest)"),
    "ex_liq": ("netliq_yoy", "Actual Fed net-liquidity YoY (latest)"),
    "ex_infl": ("cpi_yoy", "Actual CPI inflation YoY (latest)"),
}


def _safe_series(frame, key, index=None):
    """Return a numeric series aligned to index, or an all-NaN series if absent."""
    idx = frame.index if index is None else index
    if key not in frame.columns:
        return pd.Series(np.nan, index=idx, dtype=float)
    return pd.to_numeric(frame[key], errors="coerce").reindex(idx)


def _fmt(c, v, delta=False):
    """Format an expectation using the units declared in EXP_INFO."""
    if v is None or pd.isna(v):
        return "-"
    info = EXP_INFO.get(c, {})
    fmt = info.get("fmt", "pct")
    try:
        value = float(v)
    except (TypeError, ValueError):
        return "-"
    if fmt == "usd":
        return f"{value * 100:+.1f} pts" if delta else f"{value:+.1%}"
    if fmt == "prob":
        return f"{value:+.0f} pts" if delta else f"{value:.0f}%"
    if fmt == "chg":
        return f"{value:+.2f} pts"
    return f"{value:+.2f} pts" if delta else f"{value:.2f}%"


def grade_expectations(m, E, idx, h=EXP_H):
    """Grade expectation forecasts against realised values and a naive anchor.

    Skill is 1 - MAE(model)/MAE(anchor); positive values mean the forecast had
    lower mean absolute error than the anchor. ``ic`` is the rank correlation
    between the forecasted change and the realised change. Results are
    descriptive unless the supplied ``idx`` is strictly out of sample.
    """
    if not isinstance(m, pd.DataFrame) or m.empty or not isinstance(E, pd.DataFrame):
        return pd.DataFrame(columns=["k", "n", "mae_m", "mae_n", "skill", "ic"])

    idx = pd.Index(idx).intersection(m.index).intersection(E.index)
    if len(idx) == 0:
        return pd.DataFrame(columns=["k", "n", "mae_m", "mae_n", "skill", "ic"])

    ff = _safe_series(m, "fed_funds")
    if "dgs3m" in m:
        ff = ff.fillna(_safe_series(m, "dgs3m"))

    def g(key):
        return _safe_series(m, key)

    usd = g("dollar")
    future_usd = usd.shift(-h)
    # Avoid infinite percentage changes when the denominator is zero or invalid.
    usd_change = future_usd.div(usd.where(usd.abs() > 1e-12)).sub(1.0)
    spec = {
        "ex_ff12": (ff, ff.shift(-h)),
        "ex_y10": (g("dgs10"), g("dgs10").shift(-h)),
        "ex_growth": (g("gdp_yoy"), g("gdp_yoy").shift(-h)),
        "ex_unrate": (g("unrate_lvl"), g("unrate_lvl").shift(-h)),
        "ex_usd": (pd.Series(0.0, index=m.index), usd_change),
        "ex_m2": (g("m2_yoy"), g("m2_yoy").shift(-h)),
        "ex_liq": (g("netliq_yoy"), g("netliq_yoy").shift(-h)),
    }
    rows = []
    for key, (anchor, realised) in spec.items():
        if key not in E.columns:
            continue
        d = pd.concat(
            [pd.to_numeric(E[key], errors="coerce"), realised, anchor],
            axis=1,
            keys=["forecast", "realised", "anchor"],
        ).reindex(idx)
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 24:
            continue
        mae_model = float((d["forecast"] - d["realised"]).abs().mean())
        mae_anchor = float((d["anchor"] - d["realised"]).abs().mean())
        skill = 1.0 - mae_model / mae_anchor if mae_anchor > 1e-12 else np.nan
        forecast_change = d["forecast"] - d["anchor"]
        realised_change = d["realised"] - d["anchor"]
        ic = rank_ic(forecast_change, realised_change)
        rows.append({"k": key, "n": len(d), "mae_m": mae_model, "mae_n": mae_anchor, "skill": skill, "ic": ic})
    return pd.DataFrame(rows, columns=["k", "n", "mae_m", "mae_n", "skill", "ic"])


with T["xpt"]:
    st.subheader("📈 Expectations: what rates, inflation, growth, jobs and the dollar are priced / projected to do over the next 12 months")
    st.caption(
        "Eleven forward-looking indicators, all **point-in-time** (each month uses only data that was public then). "
        "🟦 **Market-implied** = computed from today's Treasury curve, nothing fitted. 🟨 **Published model** = Cleveland Fed. "
        "🟪 **Model-implied** = a walk-forward ridge estimate, because no free point-in-time survey history exists for growth, unemployment, the dollar, M2 growth or Fed liquidity. "
        "Model-implied values compress other macro indicators, so they add convenience, not new information. The grade table below tests whether they beat 'nothing changes'."
    )

    miss = [key for key in EXP_FRED if key in failed]
    if miss:
        st.warning(
            "Some inputs failed to download (" + ", ".join(EXP_FRED[key] for key in miss)
            + "), so indicators that need them may be missing or approximated."
        )

    _has_any_data = bool(_XC) and F_full[_XC].notna().any().any()
    if not _has_any_data:
        st.error(
            "No expectation indicator has usable data. Check Treasury/FRED downloads, then click 'Refresh data' "
            "in the sidebar or configure a FRED_API_KEY."
        )
    else:
        if "expinf1y" not in m.columns or not m["expinf1y"].notna().any():
            st.info("Cleveland Fed expected inflation (EXPINF1YR) was unavailable, so expected inflation uses the 10Y breakeven instead (a longer horizon).")

        _valid_rows = F_full[_XC].dropna(how="all")
        last = _valid_rows.index[-1]
        st.markdown(f"**Latest read: {last:%b %Y}.** Change shown versus 3 months earlier.")

        # Headline cards
        for row_cols in [_XC[i:i + 4] for i in range(0, len(_XC), 4)]:
            cols_ = st.columns(len(row_cols))
            for col_, c in zip(cols_, row_cols):
                s_ = pd.to_numeric(F_full[c], errors="coerce").dropna()
                box = card(col_)
                if s_.empty:
                    box.markdown(f"**{META_ALL.get(c, (c,))[0]}**")
                    box.caption("Not available")
                    continue
                now_v = float(s_.iloc[-1])
                latest_date = s_.index[-1]
                prior_date = latest_date - pd.DateOffset(months=3)
                prev = F_full[c].reindex([prior_date]).iloc[0] if prior_date in F_full.index else np.nan
                # If the exact calendar month is absent, use the last available value at/before that month.
                if pd.isna(prev):
                    hist = F_full.loc[F_full.index <= prior_date, c].dropna()
                    prev = hist.iloc[-1] if not hist.empty else np.nan
                delta = now_v - float(prev) if pd.notna(prev) else np.nan
                label = META_ALL.get(c, (c,))[0]
                box.metric(label, _fmt(c, now_v), None if pd.isna(delta) else _fmt(c, delta, True), delta_color="off")
                method = EXP_INFO.get(c, {}).get("method", "Model-implied")
                box.caption(f"{_METH_ICON.get(method, '⬜')} {method}")

        def val(c):
            if c not in F_full.columns:
                return np.nan
            s_ = pd.to_numeric(F_full[c], errors="coerce").dropna()
            return float(s_.iloc[-1]) if not s_.empty else np.nan

        # Plain-English summary
        lines = []
        d_ff = val("ex_ff_chg")
        if pd.notna(d_ff):
            direction = "cuts" if d_ff <= -0.25 else ("hikes" if d_ff >= 0.25 else "roughly no change")
            end_rate = val("ex_ff12")
            suffix = f", to about {end_rate:.2f}%" if pd.notna(end_rate) else ""
            lines.append(f"**Fed:** the curve prices {direction} ({d_ff:+.2f} pts over 12 months{suffix}).")
        if pd.notna(val("ex_real")):
            expected_infl = val("ex_infl")
            inflation_text = f" (expected inflation {expected_infl:.2f}%)" if pd.notna(expected_infl) else ""
            lines.append(f"**Real policy rate in 12 months:** about {val('ex_real'):.2f}%{inflation_text}.")
        recession_prob = val("ex_recess")
        if pd.notna(recession_prob):
            level = "low" if recession_prob < 20 else ("moderate" if recession_prob < 40 else "elevated")
            lines.append(f"**Recession risk (yield-curve model):** {recession_prob:.0f}% ({level}).")
        if pd.notna(val("ex_growth")) and pd.notna(val("ex_unrate")):
            lines.append(f"**Economy (model-implied):** growth about {val('ex_growth'):.1f}%, unemployment about {val('ex_unrate'):.1f}% in 12 months.")
        dollar_change = val("ex_usd")
        if pd.notna(dollar_change):
            dollar_view = "stronger" if dollar_change >= 0.01 else ("weaker" if dollar_change <= -0.01 else "about flat")
            lines.append(f"**Dollar (model-implied):** {dollar_view} ({dollar_change:+.1%} over 12 months; low-confidence forecast).")
        if lines:
            st.markdown("\n".join(f"- {line}" for line in lines))

        # Textbook tally
        sup = uns = neu = mix = 0
        for c in _XC:
            sign = EXP_INFO.get(c, {}).get("btc", 0)
            p_ = P_full[c].dropna().iloc[-1] if c in P_full.columns and P_full[c].notna().any() else np.nan
            if sign == 0:
                mix += 1
            elif pd.isna(p_):
                continue
            elif (p_ >= 2 / 3 and sign > 0) or (p_ <= 1 / 3 and sign < 0):
                sup += 1
            elif (p_ >= 2 / 3 and sign < 0) or (p_ <= 1 / 3 and sign > 0):
                uns += 1
            else:
                neu += 1
        st.markdown(
            f"**Textbook read for bitcoin:** {sup} expectation(s) point in bitcoin's favor, {uns} against, "
            f"{neu} in the middle of their range, {mix} have no clear textbook sign. "
            "'Textbook' = the usual story (lower real rates, a weaker dollar, faster money and liquidity growth and stronger growth help bitcoin; "
            "inflation expectations have no clear sign because bitcoin has traded like a liquidity asset as often as an inflation hedge). "
            "It is a checklist, not evidence: the table further down shows what the data in THIS sample says."
        )

        # History chart
        st.markdown("#### History")
        pick = st.selectbox("Indicator", _XC, format_func=lambda c: META_ALL.get(c, (c,))[0], key="xpt_pick")
        start_date = pd.Timestamp(start) if start is not None else F_full.index.min()
        s_ = pd.to_numeric(F_full[pick], errors="coerce").loc[lambda x: x.index >= start_date].dropna()
        if s_.empty:
            st.info("No history for this indicator after the chosen start date.")
        else:
            k_ = 100.0 if EXP_INFO.get(pick, {}).get("fmt") == "usd" else 1.0
            fig = go.Figure()
            fig.add_scatter(x=s_.index, y=s_.values * k_, name=META_ALL.get(pick, (pick,))[0], line=dict(color="#1f5fbf", width=2.5))
            if pick in _ACTUAL and _ACTUAL[pick][0] in m.columns:
                actual = _safe_series(m, _ACTUAL[pick][0], s_.index)
                fig.add_scatter(x=actual.index, y=actual.values, name=_ACTUAL[pick][1], line=dict(color="#c0392b", dash="dot", width=1.5))
            btc = _safe_series(m, "btc", s_.index)
            btc = btc.where(btc > 0)
            fig.add_scatter(x=btc.index, y=btc.values, name="Bitcoin, USD (right axis, log scale)", yaxis="y2", line=dict(color="#999", width=1.5))
            fig.update_layout(
                title=META_ALL.get(pick, (pick,))[0] + (" (%)" if EXP_INFO.get(pick, {}).get("fmt") == "usd" else ""),
                height=420,
                yaxis2=dict(overlaying="y", side="right", showgrid=False, type="log"),
                legend=dict(orientation="h", y=-0.2),
            )
            show_plot(st, fig)
            method = EXP_INFO.get(pick, {}).get("method", "Model-implied")
            how = EXP_INFO.get(pick, {}).get("how", "Definition unavailable.")
            st.caption(f"{_METH_ICON.get(method, '⬜')} **{method}.** {how}")

        # Relationship with Bitcoin outcome
        st.markdown(f"#### Do these expectations line up with bitcoin's {h}-month outcome? ({target_name.lower()})")
        rows = []
        for c in _XC:
            x_ = pd.to_numeric(F_full[c], errors="coerce").reindex(fwd.index)
            xl, yl = x_.reindex(learn_idx), fwd.reindex(learn_idx)
            xt, yt = x_.reindex(unseen_idx), fwd.reindex(unseen_idx)
            ic_l, ic_t = rank_ic(xl, yl), rank_ic(xt, yt)
            learn_pair = pd.concat([xl.rename("x"), yl.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            t_l = nw_t(learn_pair["y"].rank().values, learn_pair["x"].rank().values, h) if len(learn_pair) > 8 else np.nan
            same = bool(pd.notna(ic_l) and pd.notna(ic_t) and np.sign(ic_l) == np.sign(ic_t))
            if pd.isna(ic_l) or pd.isna(ic_t):
                status = "-"
            elif not same:
                status = "❌ Flips on unseen data"
            elif pd.notna(t_l) and abs(t_l) >= 2:
                status = "✅ Consistent and significant"
            else:
                status = "🟡 Consistent but weak"
            sign = EXP_INFO.get(c, {}).get("btc", 0)
            rows.append({
                "Indicator": META_ALL.get(c, (c,))[0],
                "Method": f"{_METH_ICON.get(EXP_INFO.get(c, {}).get('method', ''), '⬜')} {EXP_INFO.get(c, {}).get('method', 'Unknown')}",
                "Learn months": len(learn_pair),
                "Rank IC (learn)": "-" if pd.isna(ic_l) else f"{ic_l:+.2f}",
                f"Rank IC ({unseen_lbl})": "-" if pd.isna(ic_t) else f"{ic_t:+.2f}",
                "HAC t (learn)": "-" if pd.isna(t_l) else f"{t_l:+.1f}",
                "Status": status,
                "Textbook sign": _SIGN_TXT.get(sign, "Mixed"),
                "Matches textbook?": "n/a" if sign == 0 or pd.isna(ic_l) else ("✔" if np.sign(ic_l) == sign else "✘"),
            })
        show_df(st, pd.DataFrame(rows))
        st.caption(
            "Rank IC = rank correlation between the indicator and bitcoin's outcome (positive: higher indicator, better outcome). 'Unseen' follows the sidebar mode "
            "(validation only in Research mode, validation + final test when revealed). Eleven indicators are tested here, so a lone ✅ can be luck; the rules and ML tabs apply the full multiple-testing discipline. "
            "Expectations formed from the same curve (Fed path, real rate, 10Y, recession) are strongly correlated with each other and with existing curve and real-yield indicators."
        )

        # Grade expectations themselves
        st.markdown("#### Are the expectations any good? (what actually happened 12 months later)")
        gr = grade_expectations(m, F_full[_XC], ev_r)
        if gr.empty:
            st.info("Not enough overlapping months to grade the expectations.")
        else:
            names = {
                "ex_ff12": "Fed funds, 12M ahead", "ex_y10": "10Y yield, 12M ahead",
                "ex_growth": "GDP growth", "ex_unrate": "Unemployment rate",
                "ex_usd": "Dollar 12M change", "ex_m2": "M2 growth", "ex_liq": "Fed net-liquidity growth",
            }
            show_df(st, pd.DataFrame({
                "Expectation": [f"{names.get(k, k)} {_METH_ICON.get(EXP_INFO.get(k, {}).get('method', ''), '⬜')}" for k in gr["k"]],
                "Months graded": gr["n"],
                "Avg error, expectation": gr["mae_m"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-"),
                "Avg error, 'nothing changes'": gr["mae_n"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-"),
                "Skill vs 'nothing changes'": gr["skill"].map(lambda v: pc(v) if pd.notna(v) else "-"),
                "IC of expected vs realised change": gr["ic"].map(lambda v: "-" if pd.isna(v) else f"{v:+.2f}"),
            }))
            valid_skill = gr["skill"].dropna()
            beat = int((valid_skill > 0).sum())
            total = len(valid_skill)
            if total:
                (st.success if beat >= total * 0.6 else st.warning)(
                    f"{beat} of {total} expectations with defined skill had a smaller average error than 'nothing changes' "
                    f"(graded on months up to the end of {unseen_lbl if not REVEAL else 'the final test'}, outcomes realised 12 months later; "
                    "windows overlap, so independent observations are roughly months ÷ 12). "
                    + ("Where skill is negative, treat the indicator as a description of the model's view, not as a forecast." if beat < total else "")
                )
            else:
                st.info("The naive anchor has zero error or the sample does not permit a defined skill score for these expectations.")
            st.caption(
                "Error units: percentage points for rates, growth and unemployment, a fraction for the dollar. The anchor is today's value (or zero change for the dollar). "
                "The recession probability is not graded here because it needs official recession dates; expected inflation is a published series and is not graded either."
            )

        # Definitions, sources and limitations
        with st.expander("📖 Definitions, sources and limits"):
            definition_rows = []
            for c in _XC:
                s = pd.to_numeric(F_full[c], errors="coerce").dropna()
                first_month = s.index[0].strftime("%b %Y") if not s.empty else "-"
                info = EXP_INFO.get(c, {})
                definition_rows.append({
                    "Indicator": META_ALL.get(c, (c,))[0],
                    "Method": f"{_METH_ICON.get(info.get('method', ''), '⬜')} {info.get('method', 'Unknown')}",
                    "How it is built": info.get("how", "Definition unavailable."),
                    "Source series": info.get("src", "Not specified"),
                    "First month": first_month,
                })
            show_df(st, pd.DataFrame(definition_rows))
            st.markdown("""
- **Why not surveys?** The Survey of Professional Forecasters and Blue Chip are the real expected-growth / unemployment sources, but their history is not available as a free, clean, point-in-time feed. The model-implied versions are a substitute, not the same thing.
- **Term premium.** Forward rates mix expectations with a term premium, so expected 10Y yield and expected Fed funds are market-implied, not pure forecasts.
- **Publication lags.** Cleveland Fed inflation expectations are used from the 15th of the following month, GDP about 3 months + 30 days after the quarter starts, claims 5 days after the week, Treasury yields the same day.
- **Zero lower bound.** When rates are near zero the forward-implied Fed path is floored at 0 and carries little information.
- **Overlap with other indicators.** Expectations are correlated with the curve, real-yield and dollar indicators already in the app. Turn on factor compression if you add them to the rules, so they do not count as extra independent votes.
- **Selection bias.** Eleven more indicators mean more chances for a fluke. Judge them by unseen-data columns and the final test, not by how good one row looks.
""")
