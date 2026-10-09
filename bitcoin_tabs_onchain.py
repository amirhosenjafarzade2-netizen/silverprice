"""
Bitcoin tabs: On-chain (valuation, network, liquidity-in-crypto and sentiment indicators).
Executed by bitcoin_main.py; shared configuration, helpers and data are expected as globals.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Keep this module compatible with the shared-global execution model used by bitcoin_main.py.
_CHAIN_KEYS = list(globals().get("CHAIN_KEYS", []))
_F_FULL = globals().get("F_full")
_P_FULL = globals().get("P_full")
_META = globals().get("META_ALL", {})
_INFO = globals().get("ONCHAIN_INFO", {})
_VAL = globals().get("VAL", "Valuation")
_NET = globals().get("NET", "Network")
_SENT = globals().get("SENT", "Sentiment")
_PIL_ORDER = [_VAL, _NET, _SENT]
_SIGN_TXT_C = {1: "Positive", -1: "Negative", 0: "Mixed"}


def _is_num(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _meta(c, pos, default="Unknown"):
    """Read metadata safely while preserving the main module's tuple format."""
    item = _META.get(c, ())
    try:
        value = item[pos]
        return default if value is None else value
    except (IndexError, KeyError, TypeError):
        return default


def _info(c, key, default="Not documented in metadata."):
    item = _INFO.get(c, {})
    value = item.get(key, default) if isinstance(item, dict) else default
    return default if value is None else value


def _sign(c):
    try:
        sign = int(_info(c, "sign", 0))
        return sign if sign in (-1, 0, 1) else 0
    except (TypeError, ValueError, OverflowError):
        return 0


def _percentile(c):
    """Return the latest available historical percentile, or NaN."""
    if not isinstance(_P_FULL, pd.DataFrame) or c not in _P_FULL.columns:
        return np.nan
    s = pd.to_numeric(_P_FULL[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan
    value = s.iloc[-1]
    return float(value) if _is_num(value) else np.nan


def _state_txt(c, p):
    if not _is_num(p):
        return "-"
    p = float(p)
    if p <= 1 / 3:
        return "🟢 " + str(_meta(c, 1, "Low vs its own history"))
    if p <= 2 / 3:
        return "⚪ Middle of its range"
    return "🟠 " + str(_meta(c, 2, "High vs its own history"))


def _safe_fmt(c, value):
    fn = globals().get("fmt_val")
    if callable(fn):
        try:
            return fn(c, value)
        except Exception:
            pass
    if not _is_num(value):
        return "—"
    return f"{float(value):,.3g}"


def _safe_show_df(frame):
    fn = globals().get("show_df")
    if callable(fn):
        fn(st, frame)
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def _safe_show_plot(fig):
    fn = globals().get("show_plot")
    if callable(fn):
        fn(st, fig)
    else:
        st.plotly_chart(fig, use_container_width=True)


def _valid_frame():
    return isinstance(_F_FULL, pd.DataFrame) and not _F_FULL.empty


def _safe_first_month(series):
    idx = series.first_valid_index()
    if idx is None:
        return "—"
    try:
        return pd.Timestamp(idx).strftime("%b %Y")
    except (TypeError, ValueError):
        return str(idx)


def _safe_rank_ic(x, y):
    fn = globals().get("rank_ic")
    if not callable(fn):
        return np.nan
    try:
        result = fn(x, y)
        return float(result) if _is_num(result) else np.nan
    except Exception:
        return np.nan


# Build the list only after validating the shared data frame and indicator columns.
if _valid_frame():
    _CK = [
        c for c in _CHAIN_KEYS
        if c in _F_FULL.columns
        and pd.to_numeric(_F_FULL[c], errors="coerce").notna().any()
    ]
else:
    _CK = []


with T["chain"]:
    st.subheader("⛓ On-chain: what the Bitcoin network, its holders and its miners are doing")
    st.caption(
        "Valuation, network and crypto-market indicators built from free public sources "
        "(Coin Metrics Community, blockchain.com, DefiLlama, alternative.me, CFTC). "
        "Every value is point-in-time: delayed by a publication lag and built from trailing windows only. "
        "Percentiles compare today's value with past months only. They describe where bitcoin stands "
        "in its own history; they are not forecasts."
    )

    _failed = list(globals().get("failed", []))
    miss = [k for k in _failed if str(k).startswith("chain:") or k in ("onchain", "stablecoins", "fng", "cot")]
    if "onchain" in miss:
        st.error(
            "The Coin Metrics / blockchain.com on-chain download failed, so no on-chain valuation "
            "or network indicator is available. Click 'Refresh data' later."
        )
    elif miss:
        st.warning("Some on-chain inputs did not download (" + ", ".join(map(str, miss)) + "); indicators that need them may be missing.")

    if not _CK:
        st.info("No on-chain indicator could be built from the available data. Check the downloads and refresh the data.")
    else:
        _latest_rows = _F_FULL[_CK].dropna(how="all")
        if _latest_rows.empty:
            st.info("No usable on-chain observations are available.")
        else:
            last = _latest_rows.index[-1]
            try:
                st.markdown(f"**Latest read: {pd.Timestamp(last):%b %Y}** (last known observation at that month-end).")
            except (TypeError, ValueError):
                st.markdown(f"**Latest read: {last}** (last known observation at that month-end).")

            # ---------------- headline cards
            _head_candidates = ("mvrv", "mvrv_z", "mayer", "puell", "hash_ribbon", "adr_mom", "stable_mom", "fng")
            head = [c for c in _head_candidates if c in _CK][:8]
            for row_start in range(0, len(head), 4):
                row_cols = head[row_start:row_start + 4]
                cols_ = st.columns(len(row_cols))
                for col_, c in zip(cols_, row_cols):
                    s_ = pd.to_numeric(_F_FULL[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if s_.empty:
                        continue
                    now_v = s_.iloc[-1]
                    # Compare to the value three rows/months before the latest observation,
                    # not to a potentially absent date in a sparse indicator series.
                    prev_series = pd.to_numeric(_F_FULL[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    prev_pos = _F_FULL.index.get_indexer([s_.index[-1]])
                    prev = np.nan
                    if len(prev_pos) and prev_pos[0] >= 3:
                        prev = prev_series.iloc[prev_pos[0] - 3]
                    p_ = _percentile(c)
                    card_fn = globals().get("card")
                    box = card_fn(col_) if callable(card_fn) else col_
                    is_pct = bool(_meta(c, 3, False))
                    if _is_num(prev):
                        delta = f"{(float(now_v) - float(prev)) * 100:+.1f} pts vs 3m ago" if is_pct else f"{float(now_v) - float(prev):+.2f} vs 3m ago"
                    else:
                        delta = None
                    box.metric(str(_meta(c, 0, c)), _safe_fmt(c, now_v), delta, delta_color="off")
                    box.caption("—" if not _is_num(p_) else f"{p_:.0%} of its own past months were lower")

            # ---------------- readings table
            st.markdown("#### All readings")
            rows = []
            for pil in _PIL_ORDER:
                for c in [x for x in _CK if _meta(x, 4, None) == pil]:
                    s_ = pd.to_numeric(_F_FULL[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if s_.empty:
                        continue
                    p_ = _percentile(c)
                    rows.append({
                        "Pillar": pil,
                        "Indicator": _meta(c, 0, c),
                        "Latest": _safe_fmt(c, s_.iloc[-1]),
                        "Percentile vs past": "—" if not _is_num(p_) else f"{p_:.0%}",
                        "Reading": _state_txt(c, p_),
                        "First month": _safe_first_month(s_),
                        "Textbook sign when HIGH": _SIGN_TXT_C[_sign(c)],
                    })
            if rows:
                _safe_show_df(pd.DataFrame(rows))
            else:
                st.info("Indicator metadata is incomplete, so the readings table could not be assembled.")

            # ---------------- textbook tally
            sup = uns = neu = mix = 0
            for c in _CK:
                sg = _sign(c)
                p_ = _percentile(c)
                if sg == 0:
                    mix += 1
                elif not _is_num(p_):
                    continue
                elif (p_ >= 2 / 3 and sg > 0) or (p_ <= 1 / 3 and sg < 0):
                    sup += 1
                elif (p_ >= 2 / 3 and sg < 0) or (p_ <= 1 / 3 and sg > 0):
                    uns += 1
                else:
                    neu += 1
            st.markdown(
                f"**Textbook read:** {sup} on-chain indicator(s) point in bitcoin's favor, {uns} against, "
                f"{neu} in the middle of their range, {mix} have no clear textbook sign. "
                "'Textbook' means the usual cycle story (low valuation vs holders' cost basis, miners recovering "
                "and activity rising are often viewed as favorable; overheated valuation is unfavorable). "
                "It is a checklist, not evidence: the table below shows what this sample says. Bitcoin has only "
                "about three full historical cycles, so be skeptical of cycle rules."
            )

            # ---------------- history chart
            st.markdown("#### History")
            _start = globals().get("start", None)
            pick = st.selectbox("Indicator", _CK, format_func=lambda c: str(_meta(c, 0, c)), key="chain_pick")
            _series = pd.to_numeric(_F_FULL[pick], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if _start is not None:
                try:
                    _series = _series.loc[_start:]
                except (TypeError, KeyError, ValueError):
                    pass
            s_ = _series.dropna()
            if s_.empty:
                st.info("No history for this indicator after the chosen start date.")
            else:
                is_pct = bool(_meta(pick, 3, False))
                k_ = 100.0 if is_pct else 1.0
                f = go.Figure()
                f.add_scatter(
                    x=s_.index, y=s_.values * k_, mode="lines",
                    name=str(_meta(pick, 0, pick)) + (" (%)" if is_pct else ""),
                    line=dict(color="#1f5fbf", width=2.5),
                )
                if pick == "mvrv":
                    f.add_hline(y=1.0, line_dash="dot", line_color="#2e8b57", annotation_text="1.0: price = holders' average cost")
                _m = globals().get("m")
                if isinstance(_m, dict) and isinstance(_m.get("btc"), pd.Series):
                    sv = pd.to_numeric(_m["btc"], errors="coerce").reindex(s_.index)
                    sv = sv.where(sv > 0).dropna()
                    if not sv.empty:
                        f.add_scatter(x=sv.index, y=sv.values, mode="lines", name="Bitcoin, USD (right axis, log)", yaxis="y2", line=dict(color="#999", width=1.5))
                        f.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, type="log"))
                f.update_layout(title=str(_meta(pick, 0, pick)), height=420, legend=dict(orientation="h", y=-0.2), margin=dict(t=55, b=70))
                _safe_show_plot(f)
                st.caption(f"**How it is built.** {_info(pick, 'how')}  \n*Source:* {_info(pick, 'src')}.")

            # ---------------- link to bitcoin's outcome
            _h = globals().get("h", "?")
            _target_name = str(globals().get("target_name", "forward return"))
            st.markdown(f"#### Do these indicators line up with bitcoin's {_h}-month outcome? ({_target_name.lower()})")
            _fwd = globals().get("fwd")
            _learn_idx = globals().get("learn_idx")
            _unseen_idx = globals().get("unseen_idx")
            _unseen_lbl = globals().get("unseen_lbl", "unseen")
            _nw_t = globals().get("nw_t")
            rows = []
            if isinstance(_fwd, pd.Series) and _learn_idx is not None and _unseen_idx is not None:
                for c in _CK:
                    x_ = pd.to_numeric(_F_FULL[c], errors="coerce").reindex(_fwd.index)
                    try:
                        xl, yl = x_.loc[_learn_idx], _fwd.loc[_learn_idx]
                        xt, yt = x_.loc[_unseen_idx], _fwd.loc[_unseen_idx]
                    except (KeyError, TypeError, IndexError):
                        continue
                    ic_l, ic_t = _safe_rank_ic(xl, yl), _safe_rank_ic(xt, yt)
                    dl = pd.concat([xl.rename("x"), yl.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
                    t_l = np.nan
                    if len(dl) > 8 and callable(_nw_t):
                        try:
                            t_result = _nw_t(dl["y"].rank().values, dl["x"].rank().values, _h)
                            t_l = float(t_result) if _is_num(t_result) else np.nan
                        except Exception:
                            t_l = np.nan
                    same = bool(_is_num(ic_l) and _is_num(ic_t) and np.sign(ic_l) == np.sign(ic_t))
                    if not _is_num(ic_l) or not _is_num(ic_t):
                        status = "—"
                    elif not same:
                        status = "❌ Flips on unseen data"
                    elif _is_num(t_l) and abs(t_l) >= 2:
                        status = "✅ Consistent and significant"
                    else:
                        status = "🟡 Consistent but weak"
                    sg = _sign(c)
                    rows.append({
                        "Indicator": _meta(c, 0, c),
                        "Pillar": _meta(c, 4, "Unknown"),
                        "Learn months": len(dl),
                        "Rank IC (learn)": "—" if not _is_num(ic_l) else f"{ic_l:+.2f}",
                        f"Rank IC ({_unseen_lbl})": "—" if not _is_num(ic_t) else f"{ic_t:+.2f}",
                        "HAC t (learn)": "—" if not _is_num(t_l) else f"{t_l:+.1f}",
                        "Status": status,
                        "Textbook sign": _SIGN_TXT_C[sg],
                        "Matches textbook?": "n/a" if sg == 0 or not _is_num(ic_l) else ("✔" if np.sign(ic_l) == sg else "✘"),
                    })
            if rows:
                _safe_show_df(pd.DataFrame(rows))
            else:
                st.info("The forward-outcome sample or train/unseen split is unavailable, so this comparison cannot be calculated.")
            st.caption(
                "Rank IC is the rank correlation between an indicator and bitcoin's outcome (positive means higher indicator values "
                "are associated with better outcomes in the tested sample). 'Unseen' follows the sidebar mode. Overlapping windows "
                "mean there are far fewer independent observations than months, and multiple indicators are tested, so a lone positive "
                "result may be luck. MVRV, its Z-score and the Mayer multiple are correlated views of valuation/trend and should not be "
                "counted as three independent votes; use factor compression. Indicators with short histories (such as stablecoins, Fear & "
                "Greed and CME positioning) may not support a meaningful train/unseen split."
            )

            # ---------------- definitions
            with st.expander("📖 Definitions, sources and limits"):
                _defs = []
                for c in _CK:
                    _defs.append({
                        "Indicator": _meta(c, 0, c),
                        "Pillar": _meta(c, 4, "Unknown"),
                        "How it is built": _info(c, "how"),
                        "Source": _info(c, "src"),
                        "First month": _safe_first_month(_F_FULL[c]),
                    })
                _safe_show_df(pd.DataFrame(_defs))
                st.markdown("""
- **Publication lags.** On-chain values are used after a publication lag; stablecoin supply, Fear & Greed and CFTC positions have their own delays. Month-end values should only contain information assumed available by then.
- **Revisions.** Coin Metrics Community data may be revised as address clustering improves. Downloaded history may therefore differ from the data traders saw in real time; free first-release vintages are generally unavailable.
- **Realized cap and MVRV.** These depend on provider address clustering (including exchange wallets and change outputs); providers can report different values.
- **Not included.** Exchange net flows, spot-ETF flows, funding rates, open interest and long-term-holder metrics may require paid feeds or may lack clean free history.
- **Regime change.** Spot ETFs (2024), corporate treasuries and futures have changed bitcoin ownership and market structure. Cost-basis and cycle indicators calibrated on earlier periods may behave differently now.
- **Halving cycle.** Only a few completed cycles are available. Treat the cycle as a descriptive calendar, not as statistical evidence.
""")
