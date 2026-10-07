"""
Silver tabs, part 3: Projection (ML route fan), Machine learning, Look-ahead scan, Silver & gold now.
Executed by silver_main.py (all config, helpers and results are available as globals).
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import ndtri
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ projection (ML route fan)
def proj_rank_ic(a, b):
    d = pd.concat([pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))], axis=1).dropna()
    if len(d) < 8 or d.iloc[:, 0].nunique() < 2:
        return np.nan
    return d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())


def pinball(y, Q, qs=PROJ_Q):
    e = np.asarray(y)[:, None] - Q
    qa = np.asarray(qs)
    return float(np.maximum(qa * e, (qa - 1) * e).mean())


def proj_price_features(m, idx):
    s = m["silver"]
    f = pd.DataFrame(index=m.index)
    f["px_mom1"], f["px_mom3"] = s.pct_change(1), s.pct_change(3)
    f["px_mom6"], f["px_mom12"] = s.pct_change(6), s.pct_change(12)
    f["px_vol12"] = s.pct_change().rolling(12).std() * np.sqrt(12)
    f["px_trend"] = s / s.rolling(10).mean() - 1
    f["px_dd12"] = s / s.rolling(12).max() - 1
    return f.reindex(idx)


def proj_inputs(m, F_ok, feats, length):
    """Features known at each month-end and forward log returns for every grid horizon up to `length`."""
    pf = proj_price_features(m, F_ok.index)
    X = F_ok if feats == PROJ_FEATS[1] else (pf if feats == PROJ_FEATS[2] else pd.concat([F_ok, pf], axis=1))
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    ks = [k for k in PROJ_GRID if k <= length]
    s = m["silver"]
    Y = pd.DataFrame({k: np.log(s.shift(-k) / s) for k in ks}).reindex(X.index)
    return X, Y, ks


def _fit_quantiles(model_name, Xtr, ytr, qs, Xte):
    """Predicted quantiles (rows = Xte, columns = qs)."""
    if model_name == PJ_RIDGE:
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=PROJ_ALPHA)).fit(Xtr, ytr)
        rq = np.quantile(ytr - mdl.predict(Xtr), qs)
        return mdl.predict(Xte)[:, None] + rq[None, :]
    cols = [GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=40, max_depth=2, learning_rate=0.05, subsample=0.8,
                                      min_samples_leaf=10, random_state=0).fit(Xtr, ytr).predict(Xte) for q in qs]
    return np.sort(np.column_stack(cols), axis=1)


@st.cache_data(show_spinner=False)
def proj_walk_forward(X, Y, qs, cut, end, step, model_name):
    """Every `step` months refit using only rows whose k-month outcome was already known, predict the next `step` months.
    Returns {k: (dates, model quantiles, history-only quantiles, realised log return)}. No Streamlit calls inside."""
    Xv, idx, out = X.values.astype(float), X.index, {}
    for k in Y.columns:
        yv = Y[k].values.astype(float)
        dts, M, B, yy = [], [], [], []
        for s0 in range(cut, end, step):
            te = np.arange(s0, min(s0 + step, end))
            tr = np.arange(0, max(s0 - int(k) + 1, 0))
            tr = tr[~np.isnan(yv[tr])]
            if len(tr) < PROJ_MINTR or len(te) == 0:
                continue
            M.append(_fit_quantiles(model_name, Xv[tr], yv[tr], qs, Xv[te]))
            B.append(np.tile(np.quantile(yv[tr], qs), (len(te), 1)))
            dts.extend(idx[te])
            yy.extend(yv[te])
        if M:
            out[int(k)] = (pd.DatetimeIndex(dts), np.vstack(M), np.vstack(B), np.array(yy, float))
    return out


@st.cache_data(show_spinner=False)
def proj_fit_today(X, Y, qs, model_name):
    """Fit on every row whose outcome is known and predict from the latest month. Returns (horizons, model quantiles, history quantiles)."""
    Xv = X.values.astype(float)
    ks, Qm, Qb = [], [], []
    for k in Y.columns:
        yv = Y[k].values.astype(float)
        tr = np.where(~np.isnan(yv))[0]
        if len(tr) < PROJ_MINTR:
            continue
        Qm.append(_fit_quantiles(model_name, Xv[tr], yv[tr], qs, Xv[-1:])[0])
        Qb.append(np.quantile(yv[tr], qs))
        ks.append(int(k))
    return ks, np.array(Qm), np.array(Qb)


def _proj_members(model_name):
    return [PJ_RIDGE, PJ_GBR] if model_name == PJ_AVG else [model_name]


def _combine_wf(parts):
    if len(parts) == 1:
        return parts[0]
    out = {}
    for k in parts[0]:
        if all(k in p for p in parts):
            idx, _, B, y = parts[0][k]
            out[k] = (idx, np.mean([p[k][1] for p in parts], axis=0), B, y)
    return out


def to_curve(ks, Q, L):
    """Interpolate per-horizon quantiles to every month 0..L (0 at month 0)."""
    mo = np.arange(0, L + 1)
    return np.column_stack([np.interp(mo, [0] + list(ks), [0.0] + list(Q[:, j])) for j in range(Q.shape[1])])


def proj_paths(Qc, qs, S0, n=PROJ_PATHS, seed=7):
    """Routes whose month-j value follows the predicted quantiles of month j exactly. A standardised random walk (W_j / sqrt(j) ~ N(0,1))
    supplies the month-to-month dependence; each month's z-score is mapped through that month's quantile curve (linear tails)."""
    L = Qc.shape[0] - 1
    zq = ndtri(np.asarray(qs))
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n // 2, L))
    e = np.vstack([e, -e])  # antithetic pairs
    Z = np.cumsum(e, axis=1) / np.sqrt(np.arange(1, L + 1))
    lr = np.zeros((len(e), L + 1))
    for j in range(1, L + 1):
        v = np.maximum.accumulate(Qc[j])
        slo = max((v[1] - v[0]) / (zq[1] - zq[0]), 1e-6)
        shi = max((v[-1] - v[-2]) / (zq[-1] - zq[-2]), 1e-6)
        z = Z[:, j - 1]
        y = np.interp(z, zq, v)
        y = np.where(z < zq[0], v[0] + slo * (z - zq[0]), y)
        y = np.where(z > zq[-1], v[-1] + shi * (z - zq[-1]), y)
        lr[:, j] = y
    return S0 * np.exp(lr)


def proj_score(wfd, lam, eval_idx, ks):
    """Walk-forward grade on unseen months: model fan (shrunk by lam toward history) versus the history-only fan."""
    rows, mid = [], len(PROJ_Q) // 2
    for k in ks:
        if k not in wfd:
            continue
        idx, M, B, y = wfd[k]
        ok = np.asarray(idx.isin(eval_idx)) & ~np.isnan(y)
        if ok.sum() < 8:
            continue
        Qb = B[ok]
        Qm = np.sort(Qb + lam * (M[ok] - Qb), axis=1)
        yy = y[ok]
        pm, pb = pinball(yy, Qm), pinball(yy, Qb)
        call, real = np.sign(Qm[:, mid] - Qb[:, mid]), np.sign(yy - Qb[:, mid])
        nz = call != 0
        rows.append(dict(k=k, n=int(ok.sum()), ic=proj_rank_ic(Qm[:, mid], yy),
                         hit=float((call[nz] == real[nz]).mean()) if nz.any() else np.nan,
                         skill=1 - pm / pb if pb > 0 else np.nan,
                         c80=float(((yy >= Qm[:, 0]) & (yy <= Qm[:, -1])).mean()),
                         c50=float(((yy >= Qm[:, 1]) & (yy <= Qm[:, -2])).mean())))
    return pd.DataFrame(rows)


def render_projection_tab(m, F_ok, val_idx, unseen_idx, unseen_lbl, reveal, model_name, feats, length, trust, label):
    """label(col) -> display name of a feature column."""
    lam = trust / 100.0
    st.subheader(f"🔮 Silver price projection: the next {length} months")
    st.caption("A machine-learning FAN of possible routes, not a single forecast line. Each month's spread is the model's predicted distribution of silver's return "
               "(10 / 25 / 50 / 75 / 90% levels), shrunk toward the unconditional history by the trust setting; sample routes move like a random walk but "
               "match that distribution at every month. The grading table below checks, on months the model never saw, whether it beats a history-only fan. "
               "Today's fit uses every outcome known so far (like the other 'now' reads); grading uses only unseen months.")

    X, Y, ks_all = proj_inputs(m, F_ok, feats, length)
    if len(X) < 120 or not ks_all:
        st.info("Not enough history for the projection with these settings. Move the start date earlier or use more indicators.")
        return
    members = _proj_members(model_name)
    cut = int(X.index.searchsorted(val_idx[0]))
    end = len(X) if reveal else int(X.index.searchsorted(val_idx[-1], side="right"))
    with st.spinner("Fitting projection models (cached after the first run)..."):
        wf = _combine_wf([proj_walk_forward(X, Y, PROJ_Q, cut, end, PROJ_STEP, nm) for nm in members])
        fits = [proj_fit_today(X, Y, PROJ_Q, nm) for nm in members]
    ks, Qm, Qb = fits[0][0], np.mean([f[1] for f in fits], axis=0), fits[0][2]
    if not ks:
        st.info("Not enough training rows to fit any horizon.")
        return
    L = min(length, max(ks))
    Qs = np.sort(Qb + lam * (Qm - Qb), axis=1)
    Qc, Qbc = to_curve(ks, Qs, L), to_curve(ks, Qb, L)
    S0 = float(m["silver"].loc[X.index[-1]])
    P = proj_paths(Qc, PROJ_Q, S0)
    fut = pd.DatetimeIndex([X.index[-1] + pd.offsets.MonthEnd(j) for j in range(L + 1)])
    mid = len(PROJ_Q) // 2

    # --- headline numbers
    run_max = np.maximum.accumulate(P, axis=1)
    mdd = (P / run_max - 1).min(axis=1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Median price in {L}m", f"${S0 * np.exp(Qc[L, mid]):,.1f}", f"{np.exp(Qc[L, mid]) - 1:+.1%} vs ${S0:,.1f} now")
    c2.metric(f"80% range in {L}m", f"${S0 * np.exp(Qc[L, 0]):,.0f} – ${S0 * np.exp(Qc[L, -1]):,.0f}")
    c3.metric(f"Chance higher in {L}m", f"{(P[:, L] > S0).mean():.0%}")
    c4.metric(f"Chance of a {PROJ_DD:.0%}+ drawdown on the way", f"{(mdd <= -PROJ_DD).mean():.0%}",
              f"typical worst fall {np.median(mdd):.0%}", delta_color="off")

    # --- fan chart
    hist = m["silver"].loc[: X.index[-1]].iloc[-60:]
    f = go.Figure()
    f.add_scatter(x=hist.index, y=hist.values, name="Silver (history)", line=dict(color="#555"))

    def band(lo, hi, color, name):
        f.add_scatter(x=list(fut) + list(fut[::-1]), y=list(S0 * np.exp(Qc[:, hi])) + list(S0 * np.exp(Qc[::-1, lo])),
                      fill="toself", fillcolor=color, line=dict(width=0), name=name, hoverinfo="skip")

    band(0, len(PROJ_Q) - 1, "rgba(74,111,165,0.15)", "10–90% range")
    band(1, len(PROJ_Q) - 2, "rgba(74,111,165,0.30)", "25–75% range")
    for i in range(25):
        f.add_scatter(x=fut, y=P[i], line=dict(width=1, color="rgba(70,110,170,0.22)"), showlegend=False, hoverinfo="skip")
    f.add_scatter(x=fut, y=S0 * np.exp(Qbc[:, mid]), name="History-only median", line=dict(color="#999", dash="dash", width=2))
    f.add_scatter(x=fut, y=S0 * np.exp(Qc[:, mid]), name="ML median route", line=dict(color="#1f5fbf", width=3))
    vmark(f, fut[0], "today")
    f.update_layout(title=f"Silver route fan: {model_name}, trust {trust}%", yaxis_title="USD / oz", height=500)
    show_plot(st, f)
    st.caption("Shaded bands are the model's predicted distribution at each month; the thin lines are 25 of 1,000 sample routes; "
               "the dashed grey line is what plain history alone would say. The gap between the blue and grey medians is everything the model adds.")

    # --- table by horizon
    rows = []
    for k in sorted({k for k in (1, 3, 6, 12, 18, 24) if k <= L} | {L}):
        rows.append({"Horizon": f"{k} months", "Median price": f"${S0 * np.exp(Qc[k, mid]):,.1f}",
                     "Median change": pc(np.exp(Qc[k, mid]) - 1), "History-only median": pc(np.exp(Qbc[k, mid]) - 1),
                     "25–75% range": f"{np.exp(Qc[k, 1]) - 1:+.0%} to {np.exp(Qc[k, -2]) - 1:+.0%}",
                     "10–90% range": f"{np.exp(Qc[k, 0]) - 1:+.0%} to {np.exp(Qc[k, -1]) - 1:+.0%}",
                     "Chance higher": f"{(P[:, k] > S0).mean():.0%}"})
    show_df(st, pd.DataFrame(rows))

    # --- grading
    st.subheader("Does the model beat plain history? (walk-forward, unseen months only)")
    ev_ = pd.DatetimeIndex(unseen_idx)
    sc = proj_score(wf, lam, ev_, ks)
    st.caption(f"Graded on {unseen_lbl} months. Each refit uses only rows whose outcome was already known. "
               "**Skill** = reduction in pinball loss (the quantile-forecast error) versus the history-only fan: positive means the model's fan was better. "
               "**Coverage** should be near its nominal level (80% / 50%). **Rank IC** = rank correlation of the model's median with the realised return. "
               "**Direction hit** = how often the model's tilt above / below the history median matched the outcome (50% = coin flip). "
               "Windows overlap, so the number of independent observations is roughly months ÷ horizon.")
    if sc.empty:
        st.info("Not enough unseen months to grade the projection.")
    else:
        show_df(st, pd.DataFrame({
            "Horizon": [f"{k} months" for k in sc["k"]], "Unseen months": sc["n"],
            "Skill vs history": sc["skill"].map(lambda v: pc(v)), "Rank IC": sc["ic"].map(lambda v: "-" if pd.isna(v) else f"{v:+.2f}"),
            "Direction hit": sc["hit"].map(lambda v: pc(v, False)),
            "80% range coverage": sc["c80"].map(lambda v: pc(v, False)), "50% range coverage": sc["c50"].map(lambda v: pc(v, False))}))
        good_ = (sc["skill"] > 0) & (sc["ic"] > 0)
        share, avg = float(good_.mean()), float(sc["skill"].mean())
        cov = float(sc["c80"].mean())
        if share >= 0.7 and avg > 0.02:
            st.success(f"The model beat history-only on {good_.sum()} of {len(sc)} horizons (average skill {avg:+.1%}). Some real information, but the test is short "
                       "and windows overlap: read the median tilt as a mild lean, not a target.")
        else:
            st.warning(f"The model did NOT reliably beat history-only ({good_.sum()} of {len(sc)} horizons better, average skill {avg:+.1%}). "
                       "Treat the fan as a risk range (how wide routes can be) and ignore any tilt in the median. Lowering the trust setting moves the fan toward history.")
        if cov < 0.70 or cov > 0.90:
            st.warning(f"The 80% range contained the outcome {cov:.0%} of the time on unseen months, so the fan is {'too narrow' if cov < 0.70 else 'too wide'}.")

        tr_rows = []
        for l_ in (0.0, 0.25, 0.5, 0.75, 1.0):
            s_ = proj_score(wf, l_, ev_, ks)
            if not s_.empty:
                tr_rows.append({"Trust in model": f"{l_:.0%}", "Average skill vs history": pc(s_["skill"].mean()),
                                "Average rank IC": "-" if s_["ic"].isna().all() else f"{s_['ic'].mean():+.2f}",
                                "Average 80% coverage": pc(s_["c80"].mean(), False)})
        st.markdown("**Skill by trust level** (a diagnostic, not a menu: choosing the best row would select on the grading months)")
        show_df(st, pd.DataFrame(tr_rows))

    # --- drivers
    kd = min(ks, key=lambda k: abs(k - 6))
    ok_ = Y[kd].notna().values
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=PROJ_ALPHA)).fit(X.values[ok_], Y[kd].values[ok_])
    scl, rg = mdl[0], mdl[-1]
    contrib = pd.Series(rg.coef_ * (X.iloc[-1].values - scl.mean_) / scl.scale_, index=X.columns) * 100
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(12)
    st.subheader(f"What is pushing the {kd}-month projection today?")
    st.bar_chart(contrib.rename(index={c: label(c) for c in contrib.index}).sort_values())
    st.caption(f"Contribution of each feature to the {kd}-month expected log return, in percentage points relative to an average month (ridge view, shown for every model). "
               "Correlated indicators share credit unpredictably, so read this as a story about the inputs, not as causes.")
    st.warning("**Not a price target and not financial advice.** The fan describes how wide silver's routes have been after similar conditions, "
               "adjusted by a model that may have no skill. Shocks, news and regime changes are not in the data.")


# ------------------------------------------------------------------ projection tab
with T["proj"]:
    render_projection_tab(m, F_ok, val_idx, unseen_idx, unseen_lbl, REVEAL, proj_model, proj_feats, proj_len, proj_trust,
                          lambda c: META_ALL[c][0] if c in META_ALL else PROJ_NAMES.get(c, c))

# ------------------------------------------------------------------ machine learning
if run_ml:
    with T["ml"]:
        st.markdown(f"""
Each model estimates the **probability that {tgt_txt}** in {h} months, retrained every {step} months on data whose outcome was already known at that time.
Scores below use only months whose outcome is known now. **AUC** 0.50 = coin flip, 0.55-0.60 = modest skill, above 0.65 would be suspicious; the bracket is a 95% block-bootstrap range.
**Brier skill** > 0 means the probabilities beat just quoting the normal odds. **Log loss** punishes confident mistakes (lower is better). **Rank IC** = rank correlation between the probability and the actual outcome.
""")
        rows = []
        for name, wf in ml.items():
            wk = known(wf, fwd, unseen_idx)
            if wk.empty:
                continue
            yy = (fwd.loc[wk.index] > 0).astype(int)
            auc = roc_auc_score(yy, wk["p"]) if yy.nunique() > 1 else np.nan
            lo_a, hi_a = auc_ci(yy.values, wk["p"].values, h)
            br, bss, ll = prob_metrics(wk, yy)
            sm = summarize(fwd, ml_verdict(wk, margin), wk.index, h).set_index("Group")
            ok = sm.loc[FAV, "Months"] >= 3 and sm.loc[UNF, "Months"] >= 3
            sp_ = (sm.loc[FAV, "Avg"] - sm.loc[UNF, "Avg"]) * 100 if ok else np.nan
            rows.append({"Model": name, "AUC": f"{auc:.2f} [{lo_a:.2f}–{hi_a:.2f}]" if pd.notna(lo_a) else f"{auc:.2f}",
                         "Brier skill": f"{bss:+.3f}", "Log loss": f"{ll:.3f}", "Rank IC": f"{rank_ic(wk['p'], fwd.loc[wk.index]):+.2f}",
                         "Accuracy": f"{((wk['p'] > 0.5) == yy).mean():.0%}", "Naive": f"{((wk['base'] > 0.5) == yy).mean():.0%}",
                         "Favorable minus Unfavorable": "-" if np.isnan(sp_) else f"{sp_:+.1f} pts"})
        st.subheader(f"Out-of-sample comparison ({unseen_lbl} months)")
        show_df(st, pd.DataFrame(rows))
        st.caption("Simple vs complex: if logistic regression scores about the same as the forests, the extra complexity is not buying anything.")
        wf = ml[ml_choice]
        wk = known(wf, fwd, unseen_idx)
        if wk.empty:
            st.warning("Not enough history to train. Move the start date earlier.")
        else:
            yy = (fwd.loc[wk.index] > 0).astype(int)
            auc = roc_auc_score(yy, wk["p"]) if yy.nunique() > 1 else np.nan
            lo_a, hi_a = auc_ci(yy.values, wk["p"].values, h)
            _, bss, _ = prob_metrics(wk, yy)
            skill = pd.notna(lo_a) and lo_a > 0.5 and bss > 0
            (st.success if skill else st.warning)(
                f"{ml_choice}: AUC {auc:.2f}" + (f" (95% range {lo_a:.2f}–{hi_a:.2f})" if pd.notna(lo_a) else "") + f", Brier skill {bss:+.3f}. "
                + ("The AUC range sits above 0.50, so there is some skill, but the test period is short." if skill
                   else "The AUC range includes 0.50 or the probabilities do not beat the normal odds: little reliable skill on unseen data. Don't rely on it."))
            msum = summarize(fwd, ml_verdict(wk, margin), wk.index, h, fdd)
            st.subheader(f"{ml_choice}: outcome after each call")
            show_df(st, fmt_summary(msum, True))
            c1, c2 = st.columns(2)
            pf = go.Figure(go.Scatter(x=wf.index, y=wf["p"], name="P(positive)"))
            pf.add_scatter(x=wf.index, y=wf["base"], name="Normal odds", line=dict(dash="dash"))
            pf.update_layout(title="Out-of-sample probability", yaxis_tickformat=".0%", height=340)
            show_plot(c1, pf)
            try:
                bins = pd.qcut(wk["p"], 5, duplicates="drop")
                cal = pd.DataFrame({"pred": wk["p"].groupby(bins, observed=True).mean(), "act": yy.groupby(bins, observed=True).mean()})
                cf = go.Figure(go.Scatter(x=cal["pred"], y=cal["act"], mode="lines+markers", name="Model"))
                cf.add_scatter(x=[0, 1], y=[0, 1], name="Perfect", line=dict(dash="dash"))
                cf.update_layout(title="Calibration: predicted vs actual", xaxis_title="Predicted", yaxis_title="Actual", height=340)
                show_plot(c2, cf)
            except ValueError:
                c2.info("Not enough spread in predictions for a calibration plot.")
            if ml_contrib is not None:
                st.subheader("Why today's prediction? (log-odds contribution of each indicator)")
                st.bar_chart(ml_contrib.rename(index={c: META[c][0] for c in ml_contrib.index}).sort_values())
                st.caption("Positive pushes toward the target being positive, negative pushes away. Exact for logistic regression.")
            else:
                st.subheader("What the model uses most")
                st.bar_chart(ml_imp.rename(index={c: META[c][0] for c in ml_imp.index}).sort_values())
                st.caption("Importance is not causality, and correlated indicators share credit unpredictably. Switch to logistic regression for exact contributions.")

# ------------------------------------------------------------------ scan
if auto_h:
    with T["scan"]:
        st.markdown(f"""
Each look-ahead is fitted on the **first 70% of the learning period** and scored on its **last 30%**. **IC** = rank correlation between prediction and outcome (0 = no skill, ~0.10 = decent).
**Stable IC** is the worse of the two validation halves, so a horizon only wins if it works in both. Selected: **{h} months**, then locked. Validation and final test are never used here.
""")
        d = scan.copy()
        d["Look-ahead"] = d["h"].map(lambda x: f"{x} months" + (" ✅" if x == h else ""))
        for c in ["rules_ic", "ml_ic", "combined", "stable"]:
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else f"{v:+.2f}")
        d["Independent obs."] = (scan["n_val"] / scan["h"]).round(0).astype(int)
        show_df(st, d[["Look-ahead", "rules_ic", "ml_ic", "combined", "stable", "Independent obs."]].rename(
            columns={"rules_ic": "Rules IC", "ml_ic": "ML IC", "combined": "Average IC", "stable": "Stable IC"}))
        st.caption("Longer look-aheads have far fewer independent observations. If every IC is near zero, there is no reliable horizon, and that is a valid answer.")

# ------------------------------------------------------------------ silver & gold now
with T["now"]:
    st.subheader(f"✅ Silver & gold: where the environment stands now ({now:%b %Y})")
    st.caption(f"Each market gets its own rules and ML read for the next {h} months, using the same indicators and settings. "
               "Favorable / Unfavorable describes how similar environments played out historically. It is not a buy or sell instruction. "
               "These three reads always use plain return targets (silver, gold, silver minus gold), whatever target is selected in the sidebar.")

    for col, k in zip(st.columns(3), NOW_KINDS):
        r = NOW[k]
        box = card(col)
        box.markdown(f"### {r['icon']} {NOW_NAMES[k]}")
        box.markdown(f"**{REL_TXT[r['label']] if k == 'gold' else r['label']}**")
        box.markdown(f"**Confidence: {r['conf']}** ({r['agree']} of {r['n_votes']} evidence channels agree)")
        box.markdown(" · ".join(f"{n_} {VICON[v]}" for n_, v in r["votes"].items()))
        box.caption("Macro models = " + ", ".join(f"{n_} {VICON[v]}" for n_, v in r["comps"].items()))
        box.markdown(f"Rules: {ICON[r['rules']]} {r['rules']} (score {r['score']:+.1f})")
        if r["ml"]:
            box.markdown(f"ML: {r['ml']['p']:.0%} chance {NOW_PHRASE[k]} in {h}m (normal {r['ml']['base']:.0%}) → {ICON[r['ml']['verdict']]} {r['ml']['verdict']}")
        box.markdown(f"Similar past periods (descriptive): median {pc(r['ana_med'])}, {r['ana_pos']:.0%} positive")
        box.markdown(f"Evidence on unseen data: **{r['ev_level']}**")
        box.caption(r["ev_text"])
        if r["ml"] and pd.notna(r["ml"]["auc"]):
            box.caption(f"ML on unseen data: AUC {r['ml']['auc']:.2f}, Brier skill {r['ml']['bss']:+.3f}.")
        if r["why_good"]:
            box.markdown("**Helping now:** " + "; ".join(r["why_good"][:4]))
        if r["why_bad"]:
            box.markdown("**Hurting now:** " + "; ".join(r["why_bad"][:4]))

    es, eg, er = NOW["ret"]["e"], NOW["goldabs"]["e"], NOW["gold"]["e"]
    sl, gl = NOW["ret"]["label"].lower(), NOW["goldabs"]["label"].lower()
    if es >= 0.7 and eg >= 0.7:
        msg = "Both silver and gold sit in historically favorable environments."
    elif es <= 0.3 and eg <= 0.3:
        msg = "Both silver and gold sit in historically unfavorable environments."
    elif es >= 0.7:
        msg = f"Silver's environment is historically favorable ({sl}), while gold's is {gl}."
    elif eg >= 0.7:
        msg = f"Gold's environment is historically favorable ({gl}), while silver's is {sl}."
    elif es <= 0.3:
        msg = f"Silver's environment is historically unfavorable ({sl}), while gold's is {gl}."
    elif eg <= 0.3:
        msg = f"Gold's environment is historically unfavorable ({gl}), while silver's is {sl}."
    else:
        msg = "Neither metal is in a clearly favorable or unfavorable environment: mostly neutral."
    if er >= 0.7:
        msg += " Similar conditions have historically favored silver over gold."
    elif er <= 0.3:
        msg += " Similar conditions have historically favored gold over silver."
    else:
        msg += " There is no clear silver-versus-gold preference."
    (st.success if max(es, eg) >= 0.7 else (st.warning if min(es, eg) <= 0.3 else st.info))(msg)

    weak = [NOW_NAMES[k] for k in NOW_KINDS if NOW[k]["ev_level"] != "Strong"]
    if weak:
        st.caption("Evidence is Weak, None or Unknown for: " + ", ".join(weak) + ". For those, the label describes today's conditions "
                   "but the rules did not reliably predict outcomes on data they had never seen, so treat it as context, not a signal.")

    show_df(st, pd.DataFrame([{
        "Market": NOW_NAMES[k],
        "Overall": f"{NOW[k]['icon']} {REL_TXT[NOW[k]['label']] if k == 'gold' else NOW[k]['label']}",
        "Confidence": f"{NOW[k]['conf']} ({NOW[k]['agree']}/{NOW[k]['n_votes']} agree)",
        "Rules": f"{ICON[NOW[k]['rules']]} {NOW[k]['rules']}",
        "ML chance": "-" if not NOW[k]["ml"] else f"{NOW[k]['ml']['p']:.0%} (normal {NOW[k]['ml']['base']:.0%})",
        "Similar periods, median": pc(NOW[k]["ana_med"]),
        "Evidence": NOW[k]["ev_level"]} for k in NOW_KINDS]))
    st.caption("Confidence combines two things: whether the two evidence channels (macro models, which share the same indicators, and price trend) point the same way, "
               "and whether the rules beat chance on unseen data. It is a rough guide, not a probability.")
    st.warning("**Educational only, not financial advice.** This reads macro conditions only. It does not know about valuation, news, taxes, "
               "your goals or your time horizon, and relationships that held in the past can stop working.")