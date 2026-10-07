"""
Silver tabs, part 2: Regimes & analogues, Backtest, Robustness, Explorer.
Executed by silver_main.py (all config, helpers and results are available as globals).
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ regimes & analogues
with T["reg"]:
    if reg is None:
        st.info("Regime classification needs at least one growth indicator (copper, industrial production, Philly Fed, Sahm) and one inflation indicator (CPI, breakevens, oil).")
    else:
        st.subheader("Growth × inflation regimes")
        st.caption("Growth = average z-score of copper, industrial production, Philly Fed survey and (inverted) Sahm gauge; inflation = CPI, breakevens and oil "
                   "(z-scores from the learn period). Stocks are not used: they measure risk appetite, not growth. Descriptive over all months with known outcomes.")
        rs = summarize(fwd, reg, ev_r, h, fdd, order=REGIMES)
        show_df(st, fmt_summary(rs, True))
        st.markdown(f"**Today:** {reg.iloc[-1]}")
        tr_df, tr_cur, _ = regime_transitions(reg, fwd, ev_r)
        st.markdown("**Regime transitions** (regime 3 months ago → regime now, only months where it changed; overlapping windows, so treat n as optimistic)")
        if tr_cur:
            st.markdown(f"Current transition: **{tr_cur}**")
        else:
            st.markdown("No regime change in the last 3 months.")
        if tr_df.empty:
            st.caption("Not enough months for any transition (needs 6+).")
        else:
            show_df(st, pd.DataFrame({"Transition": tr_df["Transition"], "Months": tr_df["Months"], f"Avg after {h}m": tr_df["Avg"].map(pc),
                                      "Median": tr_df["Median"].map(pc), "% positive": tr_df["Win"].map(lambda v: pc(v, False))}))
    st.subheader("10 most similar historical environments (descriptive, not tradable)")
    st.caption("Closest months by standardized distance across all selected indicators (at least h months apart). Closeness is relative to the typical distance in history. "
               "This searches the whole available history, so it describes the past; it is NOT a backtest of the method and NOT a tradable signal (see the predictive test below).")
    show_df(st, pd.DataFrame({"Date": ana["Date"], "Closeness": ana["Closeness"].map(lambda v: f"{v:.0f}%"),
                              f"Outcome after {h}m": ana["After"].map(pc)}))
    ser_path = (m["silver"] / m["gold"]) if target == "gold" else m["silver"]
    paths = analogue_paths(ser_path, ana["dt"], h)
    if not paths.empty:
        pf_ = go.Figure()
        for c in paths.columns:
            pf_.add_scatter(x=paths.index, y=paths[c], name=c, line=dict(width=1), opacity=0.55)
        pf_.add_scatter(x=paths.index, y=paths.median(axis=1), name="Median", line=dict(width=4, color="black"))
        pf_.update_layout(title=f"{'Silver / gold ratio' if target == 'gold' else 'Silver'} path after each analogue (% from start)",
                          xaxis_title="Months after", yaxis_title="%", height=400)
        show_plot(st, pf_)

    st.subheader("Predictive analogues (tradable signal, expanding database)")
    st.caption(f"For every {unseen_lbl} month the method looks for the {ANA_K} closest past months among months whose outcome was already known at that time "
               f"(standardised with that database only). Favorable = median outcome of the analogues is positive and at least 60% were positive; "
               f"Unfavorable = median not positive and at most 40% positive. This is the fair, tradable version of the descriptive table above, and it is available as a backtest signal.")
    if wa.empty:
        st.info("Not enough history for the predictive analogue test.")
    else:
        ev_a = unseen_idx.intersection(wa.index)
        sa = summarize(fwd, ana_ver, ev_a, h, fdd)
        show_df(st, fmt_summary(sa, True))
        ta = sa.set_index("Group")
        ic_a = rank_ic(wa["med"].reindex(ev_a), fwd.reindex(ev_a))
        if ta.loc[FAV, "Months"] >= 3 and ta.loc[UNF, "Months"] >= 3:
            spa = (ta.loc[FAV, "Avg"] - ta.loc[UNF, "Avg"]) * 100
            ova = ta.loc[FAV, "lo"] <= ta.loc[UNF, "hi"]
            (st.success if spa > 2 and not ova else st.warning)(
                f"Predictive analogues, {unseen_lbl}: Favorable minus Unfavorable = {spa:.1f} points; rank IC of the analogue median = {ic_a:+.2f}. "
                + ("Intervals do not overlap." if not ova else "Intervals overlap, so this could be noise."))
        else:
            st.caption(f"Too few Favorable or Unfavorable analogue calls to compare (rank IC of the analogue median: {pc(ic_a)}).")

# ------------------------------------------------------------------ backtest
with T["bt"]:
    st.subheader("What if you had followed the signals?")
    rf_all = cash_rate(m)
    sk, gk = px_names(impl_key)
    if impl_key == "Futures":
        st.warning("Futures mode uses Yahoo's continuous front-month series, so roll gaps and roll yield are baked into the returns and the result is not a real roll simulation. "
                   "Treat it as a rough check; the ETF implementation is the more defensible one.")

    # signal sources (full history of verdicts; the backtest slices them to validation (+ final test in locked mode))
    src = {"Rules": verdict}
    if not wfr.empty:
        src["Rules (expanding refit)"] = wfr["v"]
    if not ana_ver.empty:
        src["Analogues (expanding)"] = ana_ver
    if run_ml and not ml[ml_choice].empty:
        mv = ml_verdict(ml[ml_choice], margin)
        src[f"ML ({ml_choice})"] = mv
        both = verdict.index.intersection(mv.index)
        e_ = (verdict.loc[both].map(EXPO) + mv.loc[both].map(EXPO)) / 2
        comb = pd.Series(NEU, index=both)
        comb[e_ >= 0.75] = FAV
        comb[e_ <= 0.25] = UNF
        src["Rules + ML combined"] = comb
    bt_all = None
    for v_ in src.values():
        bt_all = v_.index if bt_all is None else bt_all.intersection(v_.index)
    okn = (nxt(m[sk]).notna() & nxt(m[gk]).notna()).reindex(bt_all).fillna(False).values.astype(bool)
    bt_all = bt_all[(bt_all >= val_idx[0]) & okn]
    if not REVEAL:
        bt_all = bt_all[bt_all < test_idx[0]]  # research mode: the final test period is not even simulated
    val_bt, test_bt = bt_all[bt_all < test_idx[0]], bt_all[bt_all >= test_idx[0]]
    src = {k: v_.loc[bt_all] for k, v_ in src.items()}
    px_s = m["silver"]
    aux = pd.DataFrame({"trend": (px_s > px_s.rolling(10).mean()), "vol": px_s.pct_change().rolling(12).std() * np.sqrt(12)}).reindex(bt_all)
    aux["trend"] = aux["trend"].fillna(False).astype(bool)
    grade_bt = test_bt if REVEAL else val_bt

    if len(val_bt) < 12 or (REVEAL and len(test_bt) < 12):
        st.warning("The validation or final-test period is too short for a backtest. Move the start date earlier or lower the learn / validation share.")
    else:
        if target in RISK_TARGETS:
            st.caption("Note: the signal here was learned for a risk-aware target, so Favorable means 'calmer / better risk-adjusted', not necessarily 'higher return'. "
                       "The backtest still reports ordinary returns.")
        opts = ["Final test (untouched)", "Validation (used for choices)", "Both"] if REVEAL else ["Validation (used for choices)"]
        per = st.radio("Period shown", opts, horizontal=True)
        idx_show = test_bt if per.startswith("Final") else (val_bt if per.startswith("Valid") else bt_all)
        st.caption(f"Strategy shown: **{mode}**. Signal at each month-end sets the position for the next month. Idle money earns the T-bill rate"
                   + (" (and so does the invested part of a futures position, as margin collateral)" if impl_key == "Futures" else "") + ". "
                   f"Cost: {bps} bps per 100% traded (switching silver to gold trades both). Implementation: **{impl_key}**"
                   + (" (SLV / GLD prices already include the funds' expense ratios)" if impl_key == "ETF" else "") + ". "
                   f"Validation {val_bt[0]:%b %Y}–{val_bt[-1]:%b %Y} ({len(val_bt)} months)"
                   + (f", final test {test_bt[0]:%b %Y}–{test_bt[-1]:%b %Y} ({len(test_bt)} months)." if REVEAL else ". Final test hidden (research mode)."))
        curves, rows, full = {}, [], {}
        pairs = [("Trend only (no macro signal)", next(iter(src.values())))] if mode in NO_SIGNAL else list(src.items())
        for sname, v_ in pairs:
            ret, turn, W = bt_one(m, v_, aux, mode, bps, impl_key)
            full[sname] = (ret, turn, W)
            r_, ntr, inv = seg(ret, turn, W, idx_show)
            curves[sname] = r_
            rows.append((sname, perf(r_, rf_all, inv, ntr)))
        first = pairs[0][0]
        ret0, turn0, W0 = full[first]
        ws = float(W0["silver"].reindex(idx_show).mean())
        wg = float(W0["gold"].reindex(idx_show).mean())
        sm_ = static_mix(m, ws, wg, idx_show, impl_key)
        curves["No-timing mix (same avg exposure)"] = sm_
        rows.append(("No-timing mix (same avg exposure)", perf(sm_, rf_all, ws + wg, 1)))
        for bname, bret in bench_curves(m, idx_show, impl_key).items():
            curves[bname] = bret
            is_cash = bname.startswith("Cash")
            rows.append((bname, perf(bret, rf_all, 0.0 if is_cash else 1.0, 0 if is_cash else 1)))
        show_plot(st, growth_chart(curves, idx_show, "Growth of 1 unit", mark=test_bt[0] if (REVEAL and per == "Both") else None))
        show_df(st, perf_table(rows))
        st.caption("The no-timing mix holds the same average silver / gold / cash split as the strategy, every month. If the strategy cannot beat it, the signal added nothing: "
                   "the result came from being less invested, not from timing. A short test period and a handful of trades make all numbers noisy.")

        pm = rank_by if find_best else "Sharpe"
        if run_perm:
            st.markdown(f"**Randomization test on the {grade_name.lower()} ({pm})** — how often does the same strategy run on a randomly time-shifted signal do as well?")
            prow = []
            for sname, v_ in pairs:
                ret, turn, W = full[sname]
                obs = perf(seg(ret, turn, W, grade_bt)[0], rf_all)[pm]
                p_, med_, p95_ = perm_p(shift_null(m, v_, aux, mode, bps, impl_key, grade_bt.values, pm), obs)
                prow.append({"Signal": sname, f"Observed {pm}": fm(pm, obs), "Random median": fm(pm, med_), "Random 95th pct": fm(pm, p95_),
                             "p-value": "-" if pd.isna(p_) else f"{p_:.2f}", "Verdict": "✅ beats luck" if (pd.notna(p_) and p_ <= 0.10) else "❌ not distinguishable from luck"})
            show_df(st, pd.DataFrame(prow))
            st.caption("p-value = share of shifted signals that scored at least as well (p ≤ 0.10 is suggestive, ≤ 0.05 is better). "
                       + ("This is a fair test only for a strategy chosen BEFORE seeing the final test." if REVEAL else
                          "In research mode this is run on the validation period, which you may have used for choices, so it is optimistic."))

        st.markdown(f"**Cost stress** (first signal, {grade_name.lower()})")
        crow = []
        for b_ in sorted({0, bps, 50, 100}):
            rr, tt, WW = bt_one(m, pairs[0][1], aux, mode, b_, impl_key)
            pp = perf(seg(rr, tt, WW, grade_bt)[0], rf_all)
            crow.append({"Cost (bps per 100% traded)": b_, "CAGR": pc(pp["CAGR"]), "Sharpe": fm("Sharpe", pp["Sharpe"]), "Max drawdown": pc(pp["Max drawdown"])})
        show_df(st, pd.DataFrame(crow))

        st.markdown(f"**Implementation check** (first signal, {grade_name.lower()})")
        irow = []
        for ik in ("ETF", "Futures"):
            if ik == "ETF" and not (m["slv"].notna().sum() > 36 and m["gld"].notna().sum() > 36):
                continue
            rr, tt, WW = bt_one(m, pairs[0][1], aux, mode, bps, ik)
            pp = perf(seg(rr, tt, WW, grade_bt)[0], rf_all)
            irow.append({"Implementation": ik, "CAGR": pc(pp["CAGR"]), "Sharpe": fm("Sharpe", pp["Sharpe"]),
                         "Max drawdown": pc(pp["Max drawdown"])})
        show_df(st, pd.DataFrame(irow))
        st.caption("If the two differ a lot, the result depends on tracking differences or roll gaps, not on the macro signal. "
                   "(Months where ETF prices do not exist yet are excluded from the ETF row only through missing returns, so the two rows may cover slightly different months.)")

        if len(bt_all) >= 40:
            roll = go.Figure()
            for nm_, r_full in (("Strategy: " + first, ret0.reindex(bt_all).dropna()),
                                ("Silver buy & hold", bench_curves(m, bt_all, impl_key)["Silver buy & hold"])):
                ex = r_full - rf_all.reindex(r_full.index).fillna(0)
                rs_ = ex.rolling(36).mean() * 12 / (r_full.rolling(36).std() * np.sqrt(12))
                roll.add_scatter(x=rs_.index, y=rs_, name=nm_)
            if REVEAL:
                vmark(roll, test_bt[0], "final test starts")
            roll.update_layout(title="Rolling 36-month Sharpe (does it work in every era, or only in some?)", height=340)
            show_plot(st, roll)

        if find_best:
            st.markdown("---")
            st.subheader(f"🏁 Strategy leaderboard (ranked by {rank_by} on VALIDATION only)")
            res = []
            for sname, v_ in src.items():
                for strat in STRATEGIES:
                    if strat in NO_SIGNAL and sname != next(iter(src)):
                        continue
                    ret, turn, W = bt_one(m, v_, aux, strat, bps, impl_key)
                    rv, nv, iv = seg(ret, turn, W, val_bt)
                    if len(rv) < 12:
                        continue
                    pv = perf(rv, rf_all, iv, nv)
                    if REVEAL:
                        rt, nt, it = seg(ret, turn, W, test_bt)
                        if len(rt) < 12:
                            continue
                        pt = perf(rt, rf_all, it, nt)
                    else:
                        pt = perf(rv.iloc[:0], rf_all)  # all NaN: the final test is hidden
                    lab = "No macro signal" if strat in NO_SIGNAL else sname
                    res.append(dict(key=f"{lab} · {strat}", signal=lab, strat=strat, v_src=v_, pv=pv, pt=pt, v=pv[rank_by], t=pt[rank_by], ret=ret))
            bsv = perf(bench_curves(m, val_bt, impl_key)["Silver buy & hold"], rf_all)[rank_by]
            bst = perf(bench_curves(m, test_bt, impl_key)["Silver buy & hold"], rf_all)[rank_by] if REVEAL else np.nan
            if not res:
                st.info("No strategy had enough months to rank.")
            else:
                res.sort(key=lambda r_: -np.inf if pd.isna(r_["v"]) else r_["v"], reverse=True)
                medals = ["🥇", "🥈", "🥉"]
                lrows = []
                for i, r_ in enumerate(res):
                    row = {"#": medals[i] if i < 3 else str(i + 1), "Signal": r_["signal"], "Strategy": r_["strat"],
                           f"{rank_by} (validation)": fm(rank_by, r_["v"]), "CAGR (validation)": pc(r_["pv"]["CAGR"])}
                    if REVEAL:
                        row.update({f"{rank_by} (final test)": fm(rank_by, r_["t"]), "CAGR (final test)": pc(r_["pt"]["CAGR"]),
                                    "Max DD (final test)": pc(r_["pt"]["Max drawdown"]), "% invested (test)": pc(r_["pt"]["% invested"], False),
                                    "Trades (test)": r_["pt"]["Trades"],
                                    "Beats silver on both?": "✅" if (pd.notna(r_["v"]) and pd.notna(r_["t"]) and r_["v"] > bsv and r_["t"] > bst) else "❌"})
                    else:
                        row.update({"Max DD (validation)": pc(r_["pv"]["Max drawdown"]), "% invested (validation)": pc(r_["pv"]["% invested"], False),
                                    "Trades (validation)": r_["pv"]["Trades"], "Beats silver (validation)?": "✅" if (pd.notna(r_["v"]) and r_["v"] > bsv) else "❌"})
                    lrows.append(row)
                show_df(st, pd.DataFrame(lrows))
                pick = res[0]
                if REVEAL:
                    st.caption(f"Silver buy & hold for reference: {rank_by} {fm(rank_by, bsv)} on validation, {fm(rank_by, bst)} on the final test. "
                               "Only the validation column was used for ranking; the final-test column is a grade, not a selection criterion.")
                    t_sorted = sorted([r_["t"] for r_ in res if pd.notna(r_["t"])], reverse=True)
                    rank_t = t_sorted.index(pick["t"]) + 1 if pd.notna(pick["t"]) else len(res)
                    msg = (f"**Locked pick (best on validation): {pick['strat']}** with **{pick['signal']}**. "
                           f"Validation {rank_by} {fm(rank_by, pick['v'])} → final test {fm(rank_by, pick['t'])} "
                           f"(silver buy & hold {fm(rank_by, bst)}). On the final test it ranks **{rank_t} of {len(res)}**.")
                    good_pick = pd.notna(pick["t"]) and pick["t"] > bst and rank_t <= max(1, len(res) // 3)
                    if run_perm:
                        ptest, med_, p95_ = perm_p(shift_null(m, pick["v_src"], aux, pick["strat"], bps, impl_key, test_bt.values, rank_by), pick["t"])
                        if pd.notna(ptest):
                            msg += f" Randomization test on the final test: p = {ptest:.2f} (random median {fm(rank_by, med_)}, 95th percentile {fm(rank_by, p95_)})."
                            good_pick = good_pick and ptest <= 0.10
                    (st.success if good_pick else st.warning)(msg)
                else:
                    st.caption(f"Silver buy & hold for reference: {rank_by} {fm(rank_by, bsv)} on validation. The final test is hidden in research mode.")
                    st.info(f"Current validation leader: **{pick['strat']}** with **{pick['signal']}** ({rank_by} {fm(rank_by, pick['v'])}). "
                            "Freeze your settings, switch to Locked evaluation and reveal the final test to grade it once.")
                top_curves = {r_["key"]: r_["ret"].reindex(bt_all).dropna() for r_ in res[:3]}
                top_curves.update({k_: v_ for k_, v_ in bench_curves(m, bt_all, impl_key).items() if k_ in ("Silver buy & hold", "Gold buy & hold")})
                show_plot(st, growth_chart(top_curves, bt_all, "Top 3 on validation" + (", followed through the final test" if REVEAL else ""), mark=test_bt[0] if REVEAL else None))
                st.warning(f"{len(res)} candidates were compared on the validation period, so the validation winner is flattered by selection. "
                           "Trust a strategy only if it also beats silver buy & hold on the final test, ranks high there, and survives the randomization test. "
                           "Once you have looked at the final test and changed settings because of it, it stops being untouched (the counter above tracks this).")

# ------------------------------------------------------------------ robustness
with T["rob"]:
    st.subheader("Multiple testing: how many 'findings' could be luck?")
    n_t = len(stats)
    if n_t:
        thr_t = t_thr if cfg[0] == "t" else 1.5
        p_thr = math.erfc(thr_t / math.sqrt(2))
        n_raw = int((stats["t"].abs() >= thr_t).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Environments tested", n_t)
        c2.metric(f"Pass |t| ≥ {thr_t:.1f}", n_raw)
        c3.metric("Expected to pass by pure luck", f"{n_t * p_thr:.1f}")
        c4.metric("Survive FDR q ≤ 0.10 / 0.25", f"{int((stats['q'] <= 0.10).sum())} / {int((stats['q'] <= 0.25).sum())}")
        st.caption("If the number passing is close to the number expected by luck, the rules are mostly noise. Switch 'Select environments by' to the FDR q-value in the sidebar to keep only environments that survive the adjustment. "
                   "(The luck estimate treats tests as independent; indicators overlap, so it is rough. Factor compression in the sidebar reduces that overlap.)")
        best_q = stats.sort_values("q").head(8)
        show_df(st, pd.DataFrame({"Environment": [state_label(r.col, r.state) for r in best_q.itertuples()],
                                  "Indicator": [META[r.col][0] for r in best_q.itertuples()], "HAC t": [f"{v:+.1f}" for v in best_q["t"]],
                                  "Raw p": [f"{v:.3f}" for v in best_q["p"]], "FDR q": [f"{v:.2f}" for v in best_q["q"]],
                                  "Both halves?": ["✅" if v else "❌" for v in best_q["consistent"]]}))

    if fac_mode:
        st.subheader("Factor composition")
        if fac_members:
            show_df(st, pd.DataFrame([{"Factor": META_ALL[k][0], "Pillar": META_ALL[k][4], "Members": ", ".join(META_ALL[c][0] for c in v)}
                                      for k, v in fac_members.items()]))
            st.caption(f"Indicators were merged when their average |Spearman correlation| on the early learn period was at least {fac_thr:.2f}. "
                       "Each factor is the average of the members' z-scores (signs aligned to the lead indicator); 'factor high' means high in the lead indicator's direction. "
                       "Indicators not listed here had no close relatives and stay as they are. Equal weights are deliberate: learned weights would use outcomes.")
        else:
            st.info("No indicators were correlated enough to merge at this threshold. Lower the merge threshold in the sidebar.")

    st.subheader("Ablations: does each add-on help out of sample?")
    st.caption(f"Each variant refits the rules on the learn period and is graded on the {grade_name.lower()}. "
               "These are diagnostics, not a menu to pick from: choosing the best row would be selection on the grading period. "
               "Differences of a few points between rows are within noise for samples this short.")

    def _P(Fx):
        return pit_percentiles(Fx).reindex(F_ok.index) if pit else None

    def _row(name, Fx, Px):
        sp_a, nr = spread_for(Fx, Px, fwd, learn_idx, grade_idx, h, cfg)
        return {"Variant": name, "Rules kept": nr,
                f"{grade_name}: Fav − Unfav (pts)": "-" if pd.isna(sp_a) else f"{sp_a:+.1f}"}

    abl = [_row("Current: factors on" if fac_mode else "Current: factors off", F_ok, P_ok)]
    if fac_mode:
        abl.append(_row("Factors off (raw indicators)", F_raw.loc[F_ok.index], _P(F_raw)))
    for thr_a in (0.5, 0.6, 0.7, 0.8):
        for al in (True, False):
            Ff, _, _ = build_factors(F_raw, struct_idx, thr_a, al)
            abl.append(_row(f"Factors, threshold {thr_a:.1f}, signs {'aligned' if al else 'not aligned'}",
                            Ff.loc[F_ok.index], _P(Ff)))
    if _key_ok and not fac_mode:
        run_vint_abl = st.checkbox("Also run the vintage-data ablation (downloads the other data version once, then cached)", False, key="vint_abl")
        if run_vint_abl:
            try:
                _, F_alt, P_alt, _, _ = get_dataset(get_fred_key(), not use_vintage)
                Fa_ = F_alt.reindex(index=F_ok.index, columns=base_cols)
                if Fa_.notna().all().all():
                    abl.append(_row("Vintage data OFF (latest-revised)" if use_vintage else "Vintage data ON (first-release)",
                                    Fa_, P_alt[base_cols].reindex(F_ok.index) if pit else None))
                else:
                    st.info("The other data version has gaps in the selected months, so the vintage ablation was skipped.")
            except Exception:  # noqa: BLE001
                st.info("Could not load the other data version, so the vintage ablation was skipped.")
    show_df(st, pd.DataFrame(abl))

    st.markdown("**Is the factor structure stable?** (merged pairs rebuilt on each half of the structure window)")
    stab = []
    for thr_a in (0.5, 0.6, 0.7, 0.8):
        j_ = group_stability(F_raw, struct_idx, thr_a)
        stab.append({"Merge threshold": f"{thr_a:.1f}", "Pair overlap between halves": "no merges" if pd.isna(j_) else f"{j_:.0%}"})
    show_df(st, pd.DataFrame(stab))
    st.caption("Overlap near 100% means the same indicators get merged in both halves of the structure window (stable). Low overlap means the factor structure depends on the sample.")

    st.markdown("**Signal comparison (graded period)**")
    srows = []
    for nm_, ser_ in (("Rules (single fit)", verdict),
                      ("Rules (expanding refit)", None if wfr.empty else wfr["v"]),
                      ("Analogues (expanding, tradable)", None if wa.empty else ana_ver)):
        if ser_ is None:
            continue
        ix_ = grade_idx.intersection(ser_.index)
        t_ = summarize(fwd, ser_, ix_, h).set_index("Group")
        ok_ = t_.loc[FAV, "Months"] >= 3 and t_.loc[UNF, "Months"] >= 3
        srows.append({"Signal": nm_, "Fav months": int(t_.loc[FAV, "Months"]), "Unfav months": int(t_.loc[UNF, "Months"]),
                      "Fav − Unfav (pts)": f"{(t_.loc[FAV, 'Avg'] - t_.loc[UNF, 'Avg']) * 100:+.1f}" if ok_ else "-"})
    show_df(st, pd.DataFrame(srows))

    st.subheader("Parameter sensitivity")
    fr_grid = tuple(f for f in (0.4, 0.5, 0.6, 0.7) if f <= train_frac + val_frac + 1e-9) if REVEAL else tuple(f for f in (0.3, 0.4, 0.5, 0.6) if f <= train_frac + 1e-9)
    if fr_grid:
        sens = sensitivity(F_ok, P_ok, fwd, h, fr_grid, (0.5, 1.0, 1.5, 2.0, 2.5), need_consistent, cfg[3], grade_idx.values)
        zs = sens.values
        lim = np.nanmax(np.abs(zs)) if np.isfinite(zs).any() else 5
        sf = go.Figure(go.Heatmap(z=zs, x=list(sens.columns), y=list(sens.index), colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim,
                                  text=[[("" if np.isnan(v) else f"{v:+.1f}") for v in row] for row in zs], texttemplate="%{text}",
                                  showscale=False, xgap=3, ygap=3))
        sf.update_layout(title=f"{grade_name}: Favorable minus Unfavorable (pts) for different learn shares and t thresholds",
                         xaxis_title="Minimum HAC t-stat", yaxis_title="Share of history used to learn", height=320)
        show_plot(st, sf)
        nz = int(np.isfinite(zs).sum())
        npos = int((zs[np.isfinite(zs)] > 0).sum())
        (st.success if nz and npos / nz >= 0.8 else st.warning)(
            f"{npos} of {nz} parameter settings keep Favorable ahead of Unfavorable on the {grade_name.lower()}. "
            + ("A robust finding survives most settings." if nz and npos / nz >= 0.8 else "If the result only works at one setting, it is fragile."))
    else:
        st.info("Increase the learn + validation share to enable the sensitivity grid.")

    st.subheader("Is an indicator's strength stable through time?")
    rk_top = indicator_ranking(F_ok, fwd, S, learn_idx, unseen_idx, h, fdd)
    top_cols = list(rk_top["col"].head(5))
    ric = rolling_ic(F_ok, fwd.where(fwd.index.isin(ev_r)), tuple(top_cols), 60)
    if not ric.dropna(how="all").empty:
        rf_ = go.Figure()
        for c in top_cols:
            rf_.add_scatter(x=ric.index, y=ric[c] * 100, name=META[c][0])
        rf_.add_shape(type="line", x0=ric.index[0], x1=ric.index[-1], y0=0, y1=0, line=dict(color="#888", dash="dot"))
        rf_.update_layout(title="Rolling 60-month rank correlation with the outcome (top 5 indicators)", yaxis_title="%", height=380)
        show_plot(st, rf_)
        st.caption("Lines that sit on one side of zero for the whole period are stable. Lines that cross zero repeatedly (or used to be strong and faded) describe relationships that come and go. "
                   "Windows overlap heavily, so wiggles are smoother than the true uncertainty.")

    st.subheader("Redundancy between indicators" + (" (after factor compression)" if fac_mode else ""))
    cm = F_ok.corr(method="spearman")
    cf_ = go.Figure(go.Heatmap(z=cm.values, x=[META[c][0] for c in cm.columns], y=[META[c][0] for c in cm.index], zmin=-1, zmax=1,
                               colorscale="RdBu", reversescale=True, showscale=True))
    cf_.update_layout(height=max(480, 30 * len(cm)), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    show_plot(st, cf_)
    pairs_hi = [(cm.index[i], cm.columns[j], cm.iloc[i, j]) for i in range(len(cm)) for j in range(i + 1, len(cm)) if abs(cm.iloc[i, j]) >= 0.8]
    if pairs_hi:
        st.markdown("**Highly overlapping pairs (|rank correlation| ≥ 0.8)** — they vote twice for the same story (tick factor compression in the sidebar to merge them):")
        st.markdown("\n".join(f"- {META[a][0]} ↔ {META[b][0]} ({r:+.2f})" for a, b, r in sorted(pairs_hi, key=lambda x: -abs(x[2]))))
    else:
        st.caption("No pair of selected indicators has |rank correlation| ≥ 0.8.")

# ------------------------------------------------------------------ explorer
with T["exp"]:
    st.subheader("History explorer")
    st.caption("Pick an indicator and a state to see every past month in that state and what silver, gold and silver-minus-gold did next. States use the current Low/Mid/High definition.")
    e1, e2 = st.columns(2)
    ind_x = e1.selectbox("Indicator", list(META), format_func=lambda c: META[c][0], key="exp_ind")
    st_x = e2.selectbox("State", STATES, format_func=lambda s_: state_label(ind_x, s_), key="exp_state")
    allowed = S.index if REVEAL else S.index[S.index <= test_idx[0] - pd.DateOffset(months=12)]
    sel_idx = S.index[(S[ind_x] == st_x).fillna(False).values].intersection(allowed)
    st.markdown(f"**{len(sel_idx)} months** in this state ({len(sel_idx) / len(S):.0%} of history). "
                f"Now: **{state_label(ind_x, S.loc[now, ind_x])}** (value {fmt_val(ind_x, F_ok.loc[now, ind_x])}).")
    rows = []
    for h_ in (1, 3, 6, 12):
        row = {"Horizon": f"{h_} months"}
        for nm_, kd in (("Silver", "ret"), ("Gold", "goldabs"), ("Silver − gold", "gold")):
            r = target_returns(m, F_ok.index, h_, kd).reindex(sel_idx).dropna()
            base_r = target_returns(m, F_ok.index, h_, kd).reindex(allowed).dropna()
            row[nm_] = "-" if r.empty else f"{r.mean():+.1%} ({(r > 0).mean():.0%} up, n={len(r)}; all months {base_r.mean():+.1%})"
        d_dd = fwd_drawdown(m["silver"], h_).reindex(sel_idx).dropna()
        row["Silver typical max drawdown"] = "-" if d_dd.empty else f"{d_dd.median():.0%}"
        rows.append(row)
    show_df(st, pd.DataFrame(rows))
    with st.expander("List the months"):
        st.write(", ".join(d.strftime("%b %Y") for d in sel_idx))