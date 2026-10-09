"""
Bitcoin tabs: On-chain (valuation, network, liquidity-in-crypto and sentiment indicators).
Executed by bitcoin_main.py (all config, helpers and results are available as globals).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_CK = [c for c in CHAIN_KEYS if c in F_full.columns and F_full[c].notna().any()]
_SIGN_TXT_C = {1: "Positive", -1: "Negative", 0: "Mixed"}
_PIL_ORDER = [VAL, NET, SENT]


def _state_txt(c, p):
    if pd.isna(p):
        return "-"
    return ("🟢 " if p <= 1 / 3 else ("⚪ " if p <= 2 / 3 else "🟠 ")) + (META_ALL[c][1] if p <= 1 / 3 else ("Middle of its range" if p <= 2 / 3 else META_ALL[c][2]))


with T["chain"]:
    st.subheader("⛓ On-chain: what the Bitcoin network, its holders and its miners are doing")
    st.caption("Valuation, network and crypto-market indicators built from free public sources (Coin Metrics Community, blockchain.com, DefiLlama, alternative.me, CFTC). "
               "Every value is **point-in-time**: delayed by a publication lag and built from trailing windows only. "
               "Percentiles compare today's value with PAST months only. They describe where bitcoin stands in its own history; they are not forecasts.")
    miss = [k for k in failed if k.startswith("chain:") or k in ("onchain", "stablecoins", "fng", "cot")]
    if "onchain" in miss:
        st.error("The Coin Metrics / blockchain.com on-chain download failed, so no on-chain valuation or network indicator is available. Click 'Refresh data' later.")
    elif miss:
        st.warning("Some on-chain inputs did not download (" + ", ".join(miss) + "), so the indicators that need them are missing.")
    if not _CK:
        st.info("No on-chain indicator could be built (only the halving cycle would need no download; it needs at least the Yahoo price history).")
    else:
        last = F_full[_CK].dropna(how="all").index[-1]
        st.markdown(f"**Latest read: {last:%b %Y}** (values are the last known at that month-end).")

        # ---------------- headline cards
        head = [c for c in ("mvrv", "mvrv_z", "mayer", "puell", "hash_ribbon", "adr_mom", "stable_mom", "fng") if c in _CK][:8]
        for row_cols in [head[i:i + 4] for i in range(0, len(head), 4)]:
            cols_ = st.columns(len(row_cols))
            for col_, c in zip(cols_, row_cols):
                s_ = F_full[c].dropna()
                box = card(col_)
                now_v = s_.iloc[-1]
                prev = F_full[c].shift(3).loc[s_.index[-1]]
                p_ = P_full[c].dropna().iloc[-1] if c in P_full and P_full[c].notna().any() else np.nan
                box.metric(META_ALL[c][0], fmt_val(c, now_v), None if pd.isna(prev) else (f"{(now_v - prev) * 100:+.1f} pts vs 3m ago" if META_ALL[c][3] else f"{now_v - prev:+.2f} vs 3m ago"),
                           delta_color="off")
                box.caption(f"{'-' if pd.isna(p_) else f'{p_:.0%} of its own past months were lower'}")

        # ---------------- readings table
        st.markdown("#### All readings")
        rows = []
        for pil in _PIL_ORDER:
            for c in [x for x in _CK if META_ALL[x][4] == pil]:
                s_ = F_full[c].dropna()
                p_ = P_full[c].dropna().iloc[-1] if c in P_full and P_full[c].notna().any() else np.nan
                rows.append({"Pillar": pil, "Indicator": META_ALL[c][0], "Latest": fmt_val(c, s_.iloc[-1]),
                             "Percentile vs past": "-" if pd.isna(p_) else f"{p_:.0%}", "Reading": _state_txt(c, p_),
                             "First month": s_.index[0].strftime("%b %Y"), "Textbook sign when HIGH": _SIGN_TXT_C[ONCHAIN_INFO[c]["sign"]]})
        show_df(st, pd.DataFrame(rows))

        # ---------------- textbook tally
        sup = uns = neu = mix = 0
        for c in _CK:
            sg = ONCHAIN_INFO[c]["sign"]
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
        st.markdown(f"**Textbook read:** {sup} on-chain indicator(s) point in bitcoin's favor, {uns} against, {neu} in the middle of their range, {mix} have no clear textbook sign. "
                    "'Textbook' = the usual cycle story (valuation low vs holders' cost basis, miners recovering and activity rising are good; overheated valuation is bad). "
                    "It is a checklist, not evidence: the table below shows what the data in THIS sample says. Bitcoin has only about three full cycles, so be skeptical of any cycle rule.")

        # ---------------- history chart
        st.markdown("#### History")
        pick = st.selectbox("Indicator", _CK, format_func=lambda c: META_ALL[c][0], key="chain_pick")
        s_ = F_full[pick].loc[start:].dropna()
        if s_.empty:
            st.info("No history for this indicator after the chosen start date.")
        else:
            k_ = 100.0 if META_ALL[pick][3] else 1.0
            f = go.Figure()
            f.add_scatter(x=s_.index, y=s_.values * k_, name=META_ALL[pick][0] + (" (%)" if META_ALL[pick][3] else ""), line=dict(color="#1f5fbf", width=2.5))
            pv = P_full[pick].reindex(s_.index) if pick in P_full else None
            if pick == "mvrv":
                f.add_hline(y=1.0, line_dash="dot", line_color="#2e8b57", annotation_text="1.0: price = holders' average cost")
            sv = m["btc"].reindex(s_.index)
            f.add_scatter(x=sv.index, y=sv.values, name="Bitcoin, USD (right axis, log)", yaxis="y2", line=dict(color="#999", width=1.5))
            f.update_layout(title=META_ALL[pick][0], height=420, yaxis2=dict(overlaying="y", side="right", showgrid=False, type="log"), legend=dict(orientation="h", y=-0.2))
            show_plot(st, f)
            st.caption(f"**How it is built.** {ONCHAIN_INFO[pick]['how']}  \n*Source:* {ONCHAIN_INFO[pick]['src']}.")

        # ---------------- link to bitcoin's outcome
        st.markdown(f"#### Do these indicators line up with bitcoin's {h}-month outcome? ({target_name.lower()})")
        rows = []
        for c in _CK:
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
            sg = ONCHAIN_INFO[c]["sign"]
            rows.append({"Indicator": META_ALL[c][0], "Pillar": META_ALL[c][4], "Learn months": len(dl),
                         "Rank IC (learn)": "-" if pd.isna(ic_l) else f"{ic_l:+.2f}", f"Rank IC ({unseen_lbl})": "-" if pd.isna(ic_t) else f"{ic_t:+.2f}",
                         "HAC t (learn)": "-" if pd.isna(t_l) else f"{t_l:+.1f}", "Status": status, "Textbook sign": _SIGN_TXT_C[sg],
                         "Matches textbook?": "n/a" if sg == 0 or pd.isna(ic_l) else ("✔" if np.sign(ic_l) == sg else "✘")})
        show_df(st, pd.DataFrame(rows))
        st.caption("Rank IC = rank correlation between the indicator and bitcoin's outcome (positive: higher indicator, better outcome). 'Unseen' follows the sidebar mode "
                   "(validation only in Research mode, validation + final test when revealed). Overlapping windows mean far fewer independent observations than months, "
                   f"and {len(_CK)} indicators are tested here, so a lone ✅ can be luck; the rules and ML tabs apply the full multiple-testing discipline. "
                   "MVRV, its Z-score and the Mayer multiple are three views of one idea (price vs cost basis / trend) and are strongly correlated: do not count them as three votes (use factor compression in the sidebar). "
                   "Indicators that start in 2017-18 (stablecoins, Fear & Greed, CME positioning) have too few months for a meaningful learn / unseen split.")

        # ---------------- definitions
        with st.expander("📖 Definitions, sources and limits"):
            show_df(st, pd.DataFrame([{"Indicator": META_ALL[c][0], "Pillar": META_ALL[c][4], "How it is built": ONCHAIN_INFO[c]["how"], "Source": ONCHAIN_INFO[c]["src"],
                                       "First month": F_full[c].first_valid_index().strftime("%b %Y")} for c in _CK]))
            st.markdown("""
- **Publication lags.** On-chain values are used 2 days after the day they describe, stablecoin supply 2 days, Fear & Greed 1 day; CFTC positions from a constructed, conservative release date. Month-end values only contain what was known then.
- **Revisions.** Coin Metrics Community data can be revised when address clustering improves. The downloaded history is the revised one, so on-chain results are somewhat flattering compared with what traders could see live. There is no free first-release vintage.
- **Realized cap and MVRV** depend on the provider's clustering of addresses (exchange wallets, change outputs). Other providers publish different numbers.
- **Not included.** Exchange net flows, spot-ETF flows, funding rates, open interest and long-term-holder metrics need paid feeds or have no clean free history.
- **Regime change.** Spot ETFs (2024), corporate treasuries and futures have changed who holds bitcoin; cost-basis and cycle indicators calibrated on 2011-2021 may behave differently now.
- **Halving cycle.** Only three completed cycles exist (2012, 2016, 2020) plus the one starting April 2024. Treat the indicator as a descriptive calendar, not as evidence.
""")
