"""
Bitcoin tabs, part 1: Dashboard, Guide, Environments, Indicator ranking, Out-of-sample.
Executed by bitcoin_main.py (all config, helpers and results are available as globals).
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ dashboard
with T["dash"]:
    st.subheader(f"{ICON[v_now]} Historically {v_now.lower()} environment for bitcoin ({now:%b %Y})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro score (rules)", f"{sc_now:+.1f}" if cfg[3] != "equal" else f"{sc_now:+.0f}")
    if run_ml:
        c2.metric(f"ML: chance {tgt_txt} in {h}m", f"{ml_p:.0%}", f"{(ml_p - ml_base) * 100:+.1f} pts vs normal ({ml_base:.0%})")
    c3.metric("Suggested exposure", f"{expo_now:.0%}", help="Share of your INTENDED bitcoin allocation, not of your portfolio.")
    if "mvrv" in F_all and pd.notna(F_all["mvrv"].get(now, np.nan)):
        c4.metric("MVRV (market cap / realized cap)", f"{F_all['mvrv'][now]:.2f}",
                  help="On-chain valuation: the price relative to the average price at which coins last moved. Not an intrinsic valuation, and not a timing signal by itself.")

    r_ = NOW["ret"]
    st.markdown(f"**Bitcoin overall:** {r_['icon']} {r_['label']} | **Confidence: {r_['conf']}** "
                f"({r_['agree']} of {r_['n_votes']} evidence channels agree; evidence on unseen data: {r_['ev_level']}). "
                + " · ".join(f"{k} {VICON[v]}" for k, v in r_["votes"].items())
                + "  (macro models = " + ", ".join(f"{k} {VICON[v]}" for k, v in r_["comps"].items()) + ", all reading the same indicators)")
    st.caption("Exposure = average of the rules and ML views (Favorable 100%, Neutral 50%, Unfavorable 0%) as a share of the bitcoin allocation you already intended. "
               "The last tab compares bitcoin, gold and bitcoin-vs-gold side by side.")

    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Pillar scores** (−100 = all unfavorable, +100 = all favorable)")
        S_now = S.loc[now]
        rows, psc = [], {}
        for pl in PILLARS:
            cols = [c for c in META if META[c][4] == pl]
            if cols:
                sc = 100 * (sum((c, S_now[c]) in good for c in cols) - sum((c, S_now[c]) in bad for c in cols)) / len(cols)
                psc[pl] = sc
                rows.append({"Pillar": pl, "Score": f"{'🟢' if sc > 15 else ('🔴' if sc < -15 else '🟡')} {sc:+.0f}", "Indicators": len(cols)})
        show_df(st, pd.DataFrame(rows))
        mon = [psc[p] for p in MONETARY_PILLARS if p in psc]
        ind = [psc[p] for p in INDUSTRIAL_PILLARS if p in psc]
        if mon and ind:
            st.markdown(f"**Bitcoin's two backdrops:** macro and liquidity **{np.mean(mon):+.0f}** (rates, dollar, M2, Fed liquidity, growth and risk appetite) · "
                        f"on-chain **{np.mean(ind):+.0f}** (valuation vs holders' cost basis, miners, network activity).")
        if reg is not None:
            st.markdown(f"**Macro regime:** {reg.iloc[-1]}")
        cur, n_cur, med, n_h, avg_h = streak
        st.markdown(f"**Persistence:** {cur.lower()} for **{n_cur}** month(s) (typical run: {med:.0f}). "
                    + (f"After 3+ months in this state, the average outcome was {avg_h:+.1%} ({n_h} months)." if n_h >= 5 else ""))
    with cr:
        st.markdown(f"**10 most similar past environments (descriptive, NOT tradable)** → outcome after {h} months ({target_name.lower()})")
        a = ana["After"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Median", pc(a.median())); k2.metric("Average", pc(a.mean())); k3.metric("Positive", f"{(a > 0).mean():.0%}")
        k4.metric("Worst", pc(a.min())); k5.metric("Best", pc(a.max()))
        st.caption("Only 10 observations: indicative, not statistical proof. Descriptive: the search uses all history available" + ("" if REVEAL else " (final test excluded in research mode)")
                   + ". It is not a trading signal. The Regimes & analogues tab has a fair, expanding-database (tradable) version of this test.")

    with st.expander("What would change the rules' verdict?"):
        th = state_thresholds(F_ok, learn_idx, pit)
        rows = []
        for dct, eff in ((good, "helps"), (bad, "hurts")):
            for c, s_ in sorted(dct):
                if S_now[c] != s_:
                    continue
                lo_, hi_ = th.loc[c].iloc[0], th.loc[c].iloc[1]
                flips = {"Low": f"rises above {fmt_val(c, lo_)}", "High": f"falls below {fmt_val(c, hi_)}",
                         "Mid": f"leaves {fmt_val(c, lo_)} to {fmt_val(c, hi_)}"}[s_]
                rows.append({"Indicator": META[c][0], "Now": state_label(c, s_), "Effect on verdict": eff,
                             "Current value": fmt_val(c, F_ok.loc[now, c]), "Stops counting if it": flips})
        if rows:
            show_df(st, pd.DataFrame(rows))
            st.caption("Thresholds are the Low/High cut-offs used by the rules" + (" (approximated by the full history to date in point-in-time mode)." if pit else " (learn-period terciles).")
                       + (" Factor values are in z-score units." if fac_mode else ""))
        else:
            st.write("No active rule is currently contributing to the verdict.")

    st.warning("**This is not a buy or sell signal.** It describes how bitcoin behaved after similar macro conditions. "
               "It says nothing about whether bitcoin is cheap or expensive today, and past patterns can stop working.")

# ------------------------------------------------------------------ guide
with T["guide"]:
    st.markdown(f"""
### How this works
- **Three periods.** History is split into LEARN ({learn_idx[0]:%b %Y}–{learn_idx[-1]:%b %Y}), VALIDATION ({val_idx[0]:%b %Y}–{val_idx[-1]:%b %Y}) and FINAL TEST ({test_idx[0]:%b %Y}–{test_idx[-1]:%b %Y}).
  The selection hierarchy is explicit: indicators, factors, rules and the look-ahead are chosen on LEARN only; strategy and signal are chosen on VALIDATION; the FINAL TEST only grades the locked choice.
  Outcome windows are purged so periods never overlap.
- **What is predicted.** Raw bitcoin return, a **volatility-scaled forward return** (each outcome scaled by trailing volatility, so crisis years do not dominate), a **drawdown** target in two forms
  (yes / no: the worst fall over the look-ahead stays under {DD_FLOOR:.0%}; or the continuous drawdown, which keeps the size of the fall), or bitcoin relative to gold / cash.
  Macro, liquidity and stress variables often predict risk better than direction, so try the drawdown targets.
- **Environments (rules):** each month is described by {len(META)} {'factors / indicators' if fac_mode else 'macro indicators'}, each split into Low/Neutral/High (fixed terciles from the learn period, or point-in-time percentiles).
  We measure the {h}-month outcome in each bucket and keep environments that are statistically clear (HAC t-stat, or FDR q-value to account for testing dozens of buckets) and hold in both halves of the learn period.
- **Factor compression (optional).** FDR handles many tests but not correlated predictors. With factor mode on, indicators that move together (average-linkage |Spearman| ≥ the chosen threshold, measured on the early learn period only,
  no outcomes involved) become one factor: the average of their sign-aligned z-scores. Indicators with no close relatives stay as they are. The equal-weight average is a deliberate choice: learned weights would use outcomes and break the anti-leakage design.
  The Robustness tab shows the composition, how stable it is across the two halves of the structure window, and ablations over the merge threshold and sign alignment.
- **Indicator ranking:** correlation with the outcome on learn and unseen data, a drop-one test, plus two risk-aware descriptive columns: the High-minus-Low gap in standard deviations of the outcome,
  and the typical drawdown after High versus Low states.
- **Machine learning:** three models estimate the chance that *{tgt_txt}*, retrained walk-forward on data whose outcomes were known at the time. AUC comes with a bootstrap confidence interval.
- **Analogues come in two kinds.** The 10-closest-months table is **descriptive and not tradable** (it searches all history). The **predictive analogue test** is the tradable version: for every unseen month it searches only months whose outcome was already known then, and is scored on unseen months.
- **Backtest:** ten strategies on spot / ETF-style (default) or futures-style implementations with trading costs. Besides benchmarks it compares against a **no-timing mix with the same average exposure**,
  runs a **randomization test** (signal shifted in time) so you can see how often luck would have done as well, and shows an **implementation check** (spot / ETF-style vs futures-style side by side).
- **Robustness:** multiple-testing summary, factor composition and stability, **ablations** (factors, merge threshold, sign alignment, vintage data), signal comparison, parameter-sensitivity grid, rolling indicator strength, indicator redundancy.
  The ablations are diagnostics, not a menu: picking the best row would be selection on the grading period.
- **Projection:** a machine-learning **fan of possible bitcoin price routes** for the next 6-24 months. Quantile models (ridge + residual quantiles, gradient boosting, or both) predict the 10 / 25 / 50 / 75 / 90% levels of bitcoin's forward return from the macro indicators and, optionally, bitcoin's own momentum;
  the trust slider shrinks them toward plain history. Sample routes match the predicted distribution at every month. The tab grades the model walk-forward on unseen months against a history-only fan and says plainly when it adds nothing. Two feature sets add the expectation indicators.
- **Bitcoin & gold now:** separate reads for bitcoin, gold and bitcoin-versus-gold, with a confidence level that combines how many evidence channels agree and whether the rules worked on unseen data.

### On-chain, liquidity and crypto-market pillars (tab "⛓ On-chain")
- **Valuation:** MVRV (market cap / realized cap), its expanding Z-score, the Mayer multiple (price / 200-day average), the Puell multiple (miner issuance revenue / 1-year average) and NVT (network value / transfer volume).
- **Network:** hash ribbon and hash-rate trend (miner stress), active addresses and transaction trends, fee share of miner revenue, and the share of supply that moved in the last year (dormancy).
- **Liquidity:** M2 growth, financial conditions (NFCI), the Fed balance sheet, and **Fed net liquidity** (balance sheet minus the Treasury General Account minus reverse repo), the macro series most often linked to bitcoin.
- **Crypto market & sentiment (optional):** stablecoin supply (from 2017), the Fear & Greed index (from 2018), CME futures positioning of leveraged funds and asset managers (from Dec 2017) and the halving cycle (months since the last halving).
- **Free sources only:** Coin Metrics Community API (with blockchain.com as a fallback for some series), DefiLlama, alternative.me and the CFTC. Every on-chain value is delayed by a publication lag before it is used. Coin Metrics may **revise** its history when address clustering improves, so today's downloaded history can differ from what traders saw then; there is no free first-release vintage for on-chain data.
- **Not included (paid or no clean free history):** exchange net flows, spot-ETF flows, funding rates, open interest, long-term-holder supply and realized-profit metrics.
- **Short history.** Bitcoin has had only about three full halving cycles and roughly 10 years of the current market structure. The learn / validation / final-test split is therefore short, the halving-cycle indicator rests on three cycles, and any "environment" result deserves extra skepticism.
- The CFTC API gives only the Tuesday as-of date, so the release date is a **constructed, conservative estimate**: the Friday of that week, pushed one business day later around US federal holidays and rolled past weekends (never earlier than the real release, so no look-ahead).
  It is still an estimate: audit it against the CFTC's published release calendar before relying on it.

### Expectation indicators (new pillar, tab "📈 Expectations")
- Eleven forward-looking indicators: **expected Fed funds rate in 12 months, its change, expected real rate, expected inflation, expected 10Y yield, expected growth, recession probability, expected unemployment, expected USD direction, expected M2 money growth, expected Fed net-liquidity growth.**
- Three kinds, labelled everywhere: 🟦 **market-implied** (forward rates from the Treasury curve; the NY Fed yield-curve recession probit), 🟨 **published model** (Cleveland Fed 1-year expected inflation) and
  🟪 **model-implied** (growth, unemployment, dollar, M2 growth, Fed net liquidity: a walk-forward ridge forecast refitted only on outcomes already known, because no free point-in-time survey history exists).
  Model-implied values are compressions of the other macro indicators, so they add convenience rather than new information. The Expectations tab grades them against a 'nothing changes' baseline.
- They are **off by default** for the rules / ML / backtest (sidebar switch "Add expectation indicators to every model") because they overlap with the curve, real-yield and dollar indicators and add eleven more chances for a fluke; factor compression helps with the overlap.
  The Projection tab can use them regardless, through two extra feature sets, and has a with / without comparison graded on the same unseen months.

### Evaluation discipline
- **Research vs Locked.** In Research mode the final test is hidden everywhere and every grade uses validation only. In Locked mode you reveal it and **every setting is frozen**; the app counts how many different sets of settings the final test has been shown for.
  The counter is stored in `.final_test_peeks.json` next to the app (on hosts with an ephemeral disk it falls back to the session).
- **Expanding-window rules.** Rules are refitted every {RULE_STEP} months on outcomes already known, then applied to the next {RULE_STEP} months. This gives a verdict for every unseen month that never saw its own future.
- **Evidence channels.** Rules, ML and analogues share the same indicators, so they count as one macro channel. Price trend is the second. Confidence is High only if both agree and the rules worked on unseen data.
- **First-release data (optional, needs a FRED key, on by default when a key exists).** M2, CPI, industrial production, unemployment, the Philly Fed survey and NFCI use first-release values and release dates.
  Observations older than FRED's first archived vintage carry that earliest archived value, so very early history is only partly point-in-time.
- Still not done: nested hyperparameter selection inside every walk-forward fold, contract-level futures rolls, and PCA or other learned factor weights.

### Known limits (read these)
- **Publication lags are approximations** (see PUB_LAG in the code) wherever first-release data is switched off or unavailable.
- **Implementations are simplified.** Spot / ETF-style uses BTC-USD and GLD prices with no fee deducted (a spot bitcoin ETF charges roughly 0.2-0.25% a year, ETFs only exist since 2024; custody and exchange spreads are inside the trading-cost slider).
  Futures-style adds T-bill collateral income to spot returns but does NOT model the CME basis (roll cost or gain), which has been several percent a year in some periods. The Backtest tab shows both side by side.
- **Bitcoin trades 24/7**, so month-end means the last calendar day; the Yahoo price before Sept 2014 is replaced by the Coin Metrics reference price.
- **Credit spread** is Moody's Baa minus 10y (full history on FRED). ICE BofA high-yield spreads are limited to recent years on FRED, and the ISM PMI is no longer on FRED, so the Philly Fed factory survey stands in.
- **Selection bias remains** whenever you click through many settings (and several targets). Judge settings by the final-test and randomization results, not by the one that looks best.
- Educational only, not financial advice.
""")

# ------------------------------------------------------------------ environments
with T["env"]:
    st.subheader(f"Average outcome after {h} months, by environment (learn period)")
    z, txt = [], []
    for c in META:
        zr, tr = [], []
        for s_ in STATES:
            r = stats[(stats["col"] == c) & (stats["state"] == s_)]
            if r.empty:
                zr.append(np.nan); tr.append("")
            else:
                zr.append(r["avg"].iloc[0] * 100)
                tr.append(f"{state_label(c, s_)}<br><b>{r['avg'].iloc[0]:+.1%}</b> (n={int(r['n'].iloc[0])})")
        z.append(zr); txt.append(tr)
    zmax = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 10
    fig = go.Figure(go.Heatmap(z=z, x=["Low", "Neutral", "High"], y=[META[c][0] for c in META], text=txt, texttemplate="%{text}",
                               colorscale="RdYlGn", zmid=0, zmin=-zmax, zmax=zmax, showscale=False, xgap=3, ygap=3))
    fig.update_layout(height=max(420, 52 * len(META)), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    fig.update_traces(textfont_size=11)
    show_plot(st, fig)

    def rank_table(df):
        return pd.DataFrame({
            "Environment": [state_label(r.col, r.state) for r in df.itertuples()],
            "Indicator": [META[r.col][0] for r in df.itertuples()], "Months": df["n"].values,
            "Avg after": [pc(v) for v in df["avg"]], "vs. average": [f"{v * 100:+.1f} pts" for v in df["edge"]],
            "% positive": [pc(v, False) for v in df["win"]], "HAC t": [f"{v:+.1f}" for v in df["t"]],
            "FDR q": [f"{v:.2f}" for v in df["q"]], "Both halves?": ["✅" if v else "❌" for v in df["consistent"]]})

    cL, cR = st.columns(2)
    cL.subheader("🟢 Most favorable"); show_df(cL, rank_table(stats.sort_values("edge", ascending=False).head(6)))
    cR.subheader("🔴 Most unfavorable"); show_df(cR, rank_table(stats.sort_values("edge").head(6)))
    st.caption(f"HAC t above ±2 is fairly strong. {len(stats)} environments are tested, so some look good by luck: the FDR q column adjusts for that "
               f"(q below 0.10 means roughly 10% of such findings would be false). Rules in use: {len(good)} favorable, {len(bad)} unfavorable.")
    if good or bad:
        c1, c2 = st.columns(2)
        c1.markdown("**Favorable:**\n" + "\n".join(f"- {state_label(c, s_)}" for c, s_ in sorted(good)) if good else "None")
        c2.markdown("**Unfavorable:**\n" + "\n".join(f"- {state_label(c, s_)}" for c, s_ in sorted(bad)) if bad else "None")

# ------------------------------------------------------------------ indicator ranking
with T["rank"]:
    st.subheader("Which indicators predict best?")
    st.caption(f"Target: **{target_name}** over **{h} months**. Each indicator is compared with the outcome that followed, "
               f"first on the learning period ({learn_idx[0]:%b %Y} to {learn_idx[-1]:%b %Y}), then on data it has never seen "
               f"({unseen_idx[0]:%b %Y} to {unseen_idx[-1]:%b %Y}, {unseen_lbl}). Ranking is by the smaller of the two correlations, and 0 if the direction flips.")
    rk = indicator_ranking(F_ok, fwd, S, learn_idx, unseen_idx, h, fdd)
    ab = ablation(F_ok, fwd, h, sp["cut"], step, unseen_idx.values).set_index("col")
    top = rk[rk["minic"] > 0].head(3)
    if len(top):
        st.success("Most consistent so far: " + ", ".join(f"**{META[r.col][0]}** ({r.minic:.0f}%)" for r in top.itertuples()))
    else:
        st.warning("No indicator kept the same direction on unseen data. Treat every single indicator here as unreliable for this target and look-ahead.")

    show_df(st, pd.DataFrame({
        "#": range(1, len(rk) + 1),
        "Indicator": [META[c][0] for c in rk["col"]],
        "Pillar": [META[c][4] for c in rk["col"]],
        "Direction": ["-" if pd.isna(v) else ("Higher → better outcome" if v > 0 else "Higher → worse outcome") for v in rk["ic_l"]],
        "Learn corr": [pc(v) for v in rk["ic_l"]],
        "Unseen corr": [pc(v) for v in rk["ic_t"]],
        "Same direction?": ["✅" if v else "❌" for v in rk["same"]],
        "Min |corr|": [f"{v:.0f}%" for v in rk["minic"]],
        "HAC t (learn)": ["-" if pd.isna(v) else f"{v:+.1f}" for v in rk["t_l"]],
        "Hit rate (unseen) ±95%": ["-" if pd.isna(a) else f"{a:.0%} ± {b * 100:.0f}" for a, b in zip(rk["hit"], rk["hit_ci"])],
        "High − Low (learn)": ["-" if pd.isna(v) else f"{v:+.1f} pts" for v in rk["hl_l"]],
        "High − Low (unseen)": ["-" if pd.isna(v) else f"{v:+.1f} pts" for v in rk["hl_t"]],
        "High − Low in σ (learn)": ["-" if pd.isna(v) else f"{v:+.2f}σ" for v in rk["eff_l"]],
        "High − Low in σ (unseen)": ["-" if pd.isna(v) else f"{v:+.2f}σ" for v in rk["eff_t"]],
        "Drawdown after High − Low (unseen)": ["-" if pd.isna(v) else f"{v:+.0f} pts" for v in rk["dd_t"]],
        "ΔAUC if removed": [("-" if pd.isna(ab["dauc"].get(c, np.nan)) else f"{ab['dauc'][c]:+.3f}") for c in rk["col"]],
        "Verdict": rk["status"]}))
    st.caption("**Corr** = rank correlation with the outcome (±10% is already useful for macro data, below ±5% is hard to tell from noise). "
               "**Min |corr|** = the smaller of the learn and unseen correlations (0% if the sign flips). It is a consistency score, not a statistical test. "
               "**Hit rate** = how often the learn-period direction called the above/below-median outcome on unseen data (50% = coin flip), with a 95% range that allows for overlapping windows. "
               "**High − Low in σ** = the same High-minus-Low gap measured in standard deviations of the outcome (a risk-adjusted effect size; about 0.3σ or more is notable). "
               "**Drawdown after High − Low** = typical worst fall over the look-ahead after High states minus after Low states (positive = milder drawdowns when the indicator is High). "
               "**ΔAUC if removed** = how much a logistic walk-forward model loses without this indicator (positive = it adds information; differences under about 0.02 are noise). "
               "The σ and drawdown columns are descriptive and are not used to pick rules. With this many indicators tested, a few look good by luck.")

    d_ = rk.dropna(subset=["ic_l"])
    if len(d_):
        nm = [META[c][0] for c in d_["col"]]
        bf_ = go.Figure()
        bf_.add_bar(y=nm, x=d_["ic_l"] * 100, name="Learn period", orientation="h")
        bf_.add_bar(y=nm, x=d_["ic_t"] * 100, name="Unseen period", orientation="h")
        bf_.update_layout(barmode="group", yaxis=dict(autorange="reversed"), height=max(420, 44 * len(d_)),
                          xaxis_title="Correlation with the outcome (%)", title="Learn vs unseen correlation")
        show_plot(st, bf_)

    pill = rk.assign(Pillar=[META[c][4] for c in rk["col"]]).groupby("Pillar")["minic"].agg(["mean", "max", "count"]).sort_values("mean", ascending=False)
    st.markdown("**Which themes carry the most consistent signal**")
    show_df(st, pd.DataFrame({"Pillar": pill.index, "Average Min |corr|": [f"{v:.1f}%" for v in pill["mean"]],
                              "Best indicator in pillar": [f"{v:.1f}%" for v in pill["max"]], "Indicators": pill["count"].values}))

    st.markdown("**Signal decay: which look-ahead works best for each indicator** (learn period only)")
    ich = ic_by_horizon(m, F_ok, target, train_frac).reindex(rk["col"])
    zz = ich.values * 100
    lim = max(10, np.nanmax(np.abs(zz))) if np.isfinite(zz).any() else 10
    hm = go.Figure(go.Heatmap(z=zz, x=[f"{x}m" for x in ich.columns], y=[META[c][0] for c in ich.index],
                              text=[[("" if np.isnan(v) else f"{v:+.0f}%") for v in row] for row in zz], texttemplate="%{text}",
                              colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim, showscale=False, xgap=3, ygap=3))
    hm.update_layout(height=max(420, 36 * len(ich)), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    show_plot(st, hm)
    st.caption("Green = higher indicator values were followed by better outcomes, red = worse. Longer look-aheads have far fewer independent observations, "
               "so strong-looking numbers on the right are less trustworthy.")

# ------------------------------------------------------------------ out of sample
with T["test"]:
    if not (good or bad):
        st.warning("No environment passed the strength filter. Lower the minimum strength (or raise the FDR limit) in the sidebar.")
    else:
        a, b = st.columns(2)
        a.subheader("Learn period (in-sample)")
        a.caption(f"{learn_idx[0]:%b %Y} to {learn_idx[-1]:%b %Y}. Flattering by construction.")
        show_df(a, fmt_summary(summarize(fwd, verdict, learn_idx, h, fdd), True))
        b.subheader("Validation period (unseen by the rules)")
        b.caption(f"{val_idx[0]:%b %Y} to {val_idx[-1]:%b %Y}. Choices such as strategy and signal are made here.")
        show_df(b, fmt_summary(summarize(fwd, verdict, val_idx, h, fdd), True))
        tb = None
        if REVEAL:
            st.subheader("Final test ✅ (never used for any choice, if this is your first reveal)")
            st.caption(f"{test_idx[0]:%b %Y} to {test_idx[-1]:%b %Y}.")
            tsum = summarize(fwd, verdict, test_idx, h, fdd)
            show_df(st, fmt_summary(tsum, True))
            bf = go.Figure(go.Bar(x=tsum["Group"], y=tsum["Avg"] * 100, marker_color=["#2e9e5b", "#9aa0a6", "#d64545", "#4a6fa5"],
                                  error_y=dict(type="data", symmetric=False, array=(tsum["hi"] - tsum["Avg"]) * 100,
                                               arrayminus=(tsum["Avg"] - tsum["lo"]) * 100),
                                  text=[f"{v:+.1f}%" if pd.notna(v) else "" for v in tsum["Avg"] * 100], textposition="outside"))
            bf.update_layout(title="Final test: average outcome with 95% block-bootstrap CI", yaxis_title="%", height=380)
            show_plot(st, bf)
            tb = tsum.set_index("Group")
        else:
            st.subheader("Final test 🔒 hidden")
            st.caption("Research mode keeps the final test out of every table and chart. Use the validation results above while you experiment, then switch to Locked evaluation.")
            tb = summarize(fwd, verdict, val_idx, h, fdd).set_index("Group")
        if tb.loc[FAV, "Months"] > 0 and tb.loc[UNF, "Months"] > 0:
            sp_ = (tb.loc[FAV, "Avg"] - tb.loc[UNF, "Avg"]) * 100
            overlap = tb.loc[FAV, "lo"] <= tb.loc[UNF, "hi"]
            (st.success if sp_ > 2 and not overlap else st.warning)(
                f"{grade_name}: Favorable minus Unfavorable = {sp_:.1f} points. " + ("The confidence intervals do not overlap, which is encouraging."
                                                                                   if not overlap else "The confidence intervals overlap, so this could easily be noise."))
        m_show = m if REVEAL else m.loc[: val_idx[-1]]
        pl = go.Figure()
        pl.add_scatter(x=m_show.index, y=m_show["btc"], name="Bitcoin", line=dict(color="#888"))
        for g_, col in [(FAV, "#2e9e5b"), (UNF, "#d64545")]:
            ix = verdict[verdict == g_].index
            ix = ix[ix <= m_show.index[-1]]
            pl.add_scatter(x=ix, y=m_show["btc"].reindex(ix), mode="markers", name=g_, marker=dict(color=col, size=6))
        vmark(pl, val_idx[0], "validation starts")
        if REVEAL:
            vmark(pl, test_idx[0], "final test starts")
        pl.update_layout(title="Bitcoin with macro verdicts", yaxis_type="log", height=380)
        show_plot(st, pl)

    st.subheader(f"Expanding-window rules (refitted every {RULE_STEP} months, fully out-of-sample)")
    st.caption(f"Every {RULE_STEP} months the rules are re-estimated on all months whose outcome was already known at that time, then applied to the next {RULE_STEP} months. "
               f"Unlike the single fit above, this gives a verdict for every {unseen_lbl} month that never saw its own future, and shows whether the rule set stays useful as history accumulates.")
    if wfr.empty:
        st.info("Not enough history to refit the rules walk-forward.")
    else:
        ev_w = unseen_idx.intersection(wfr.index)
        sw = summarize(fwd, wfr["v"], ev_w, h, fdd)
        show_df(st, fmt_summary(sw, True))
        tw = sw.set_index("Group")
        if tw.loc[FAV, "Months"] >= 3 and tw.loc[UNF, "Months"] >= 3:
            spw = (tw.loc[FAV, "Avg"] - tw.loc[UNF, "Avg"]) * 100
            ovw = tw.loc[FAV, "lo"] <= tw.loc[UNF, "hi"]
            (st.success if spw > 2 and not ovw else st.warning)(
                f"Expanding-window rules, {unseen_lbl}: Favorable minus Unfavorable = {spw:.1f} points. "
                + ("Intervals do not overlap." if not ovw else "Intervals overlap, so this could be noise."))
        else:
            st.caption("Too few Favorable or Unfavorable months to compare.")
