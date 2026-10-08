"""
Oil Macro Analyzer: expectation indicators (engine, no Streamlit calls).

Builds twelve forward-looking indicators from the monthly, availability-dated frame `m` created in oil_main.py.
Every value is POINT-IN-TIME: it uses only data that was public at that month-end.

Three kinds of "expectation" are used, and the app labels each one so they are never confused:
  MARKET-IMPLIED  computed from today's Treasury yield curve (forward rates, yield-curve recession probit). No fitting, no outcomes.
  PUBLISHED MODEL the Cleveland Fed's 1-year expected inflation (FRED EXPINF1YR), a model blending market and survey data.
  MODEL-IMPLIED   no free, point-in-time survey history exists for these, so a walk-forward ridge regression estimates them:
                  every few months it is refitted ONLY on rows whose 12-month outcome was already known, then predicts the next months.
                  These are compressions of the other macro indicators (convenient, not new information), and the Expectations tab grades them.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXP_H = 12          # expectation horizon in months
EXP_STEP = 3        # months between refits of the model-implied expectations
EXP_MINTR = 60      # minimum training rows before a model-implied expectation is produced
EXP_ALPHA = 50.0    # ridge strength (strong on purpose: macro data is short, overlapping and noisy)
EXP_PILLAR = "Expectations"
# Estrella-Mishkin / NY Fed yield-curve probit: P(recession starting within 12 months) = Phi(a + b * (10y minus 3m spread, in % points))
PROBIT_A, PROBIT_B = -0.5333, -0.6330

MKT, PUB, MOD = "Market-implied", "Published model", "Model-implied"
EXP_KEYS = ["ex_ff12", "ex_ff_chg", "ex_real", "ex_infl", "ex_y10", "ex_growth", "ex_recess", "ex_unrate", "ex_usd", "ex_ind", "ex_curve", "ex_demand"]

# indicator -> (title, label when LOW, label when HIGH, is a % change?, pillar)   (same layout as META_ALL in oil_main.py)
EXP_META = {
    "ex_ff12":   ("Expected Fed funds rate, 12M ahead", "Low expected policy rate", "High expected policy rate", False, EXP_PILLAR),
    "ex_ff_chg": ("Expected change in Fed funds, next 12M", "Market expects cuts", "Market expects hikes", False, EXP_PILLAR),
    "ex_real":   ("Expected real policy rate, 12M ahead", "Low expected real rate", "High expected real rate", False, EXP_PILLAR),
    "ex_infl":   ("Expected inflation, next 12M", "Low expected inflation", "High expected inflation", False, EXP_PILLAR),
    "ex_y10":    ("Expected 10Y Treasury yield, 12M ahead", "Low expected 10Y yield", "High expected 10Y yield", False, EXP_PILLAR),
    "ex_growth": ("Expected economic growth (real GDP, next 12M)", "Weak expected growth", "Strong expected growth", False, EXP_PILLAR),
    "ex_recess": ("Recession probability, next 12M (%)", "Low recession risk", "High recession risk", False, EXP_PILLAR),
    "ex_unrate": ("Expected unemployment rate, 12M ahead", "Low expected unemployment", "High expected unemployment", False, EXP_PILLAR),
    "ex_usd":    ("Expected USD direction (12M change)", "Dollar expected weaker", "Dollar expected stronger", True, EXP_PILLAR),
    "ex_ind":    ("Expected industrial demand (US industrial production, 12M)", "Weak expected demand", "Strong expected demand", False, EXP_PILLAR),
    # oil-specific
    "ex_curve":  ("Futures-curve implied oil price change, front to 4th contract (%)", "Curve prices lower oil (backwardation)", "Curve prices higher oil (contango)", False, EXP_PILLAR),
    "ex_demand": ("Expected US oil demand growth (product supplied, next 12M)", "Weak expected oil demand", "Strong expected oil demand", False, EXP_PILLAR),
}

# fmt: pp = a level in % (4.25%), chg = signed % points, prob = 0-100 %, usd = signed fraction shown as %.   oil: textbook sign when HIGH (+1 / -1 / 0 = mixed)
EXP_INFO = {
    "ex_ff12": dict(method=MKT, fmt="pp", oil=-1, src="DFF, DGS3MO, DGS1, DGS2",
                    how="Today's Fed funds rate plus the market-implied change in the 3-month rate over 12 months. The forward 3-month rate in 12 months is 1.25 x (2y yield) - 0.25 x (1y yield) "
                        "(straight-line interpolation of the curve to 15 months); today's 3-month bill yield is subtracted so the bill / Fed funds gap cancels. Includes a small term premium. Floored at 0."),
    "ex_ff_chg": dict(method=MKT, fmt="chg", oil=-1, src="same as above",
                      how="Expected 12M Fed funds rate minus today's Fed funds rate. Negative = the curve prices cuts, positive = hikes."),
    "ex_real": dict(method=MKT, fmt="pp", oil=-1, src="expected Fed funds, expected inflation",
                    how="Expected 12M Fed funds rate minus expected 12M inflation: the policy rate in real terms a year from now."),
    "ex_infl": dict(method=PUB, fmt="pp", oil=1, src="EXPINF1YR (Cleveland Fed); falls back to the 10Y breakeven if missing",
                    how="Cleveland Fed 1-year expected inflation (a model that blends inflation swaps, Treasury yields and surveys), available from the month after it is estimated."),
    "ex_y10": dict(method=MKT, fmt="pp", oil=-1, src="DGS10, DGS1",
                   how="One-year-forward 10-year yield: 10y yield + (10y yield - 1y yield) / 10 (the standard forward-rate approximation)."),
    "ex_growth": dict(method=MOD, fmt="pp", oil=1, src="GDPC1 (target) + curve, credit, NFCI, VIX, labor, claims, industry, M2, CPI, dollar",
                      how="Walk-forward ridge forecast of real GDP growth (YoY %) as it will be reported 12 months later. No free point-in-time survey history exists, so this is a model, not a survey."),
    "ex_recess": dict(method=MKT, fmt="prob", oil=-1, src="DGS10, DGS3MO",
                      how="NY Fed / Estrella-Mishkin yield-curve probit: probability of a recession starting within 12 months = Phi(-0.5333 - 0.6330 x (10y minus 3m spread)). Famous but imperfect: it flagged 2022-24 for a recession that did not arrive."),
    "ex_unrate": dict(method=MOD, fmt="pp", oil=-1, src="UNRATE (target) + curve, credit, NFCI, VIX, claims, Sahm gauge, industry, M2, CPI, dollar",
                      how="Today's unemployment rate plus a walk-forward ridge forecast of its 12-month change."),
    "ex_usd": dict(method=MOD, fmt="usd", oil=-1, src="Yahoo DX-Y.NYB (target) + expected Fed path, curve, credit, VIX, dollar trend, inflation",
                   how="Walk-forward ridge forecast of the dollar index's 12-month % change. Currency moves are notoriously hard to forecast, so check the grade table before leaning on it."),
    "ex_ind": dict(method=MOD, fmt="pp", oil=1, src="INDPRO (target) + curve, credit, NFCI, VIX, Philly Fed, claims, M2, dollar",
                   how="Walk-forward ridge forecast of US industrial production growth (YoY %) 12 months later: factory and freight activity is a large part of oil demand. "
                       "A global industrial-production forecast would be better, but no free point-in-time global series exists, so US output stands in."),
    "ex_curve": dict(method=MKT, fmt="chg", oil=-1, src="EIA daily WTI futures, contract 1 and contract 4 (needs EIA_API_KEY)",
                     how="Price of the 4th WTI futures contract (about 3 months further out than the front contract) divided by the front contract, minus 1. "
                         "Negative = backwardation (the market pays more for oil now than later: tight supply, usually bullish context), positive = contango (loose supply, storage is being paid for). "
                         "This is a market price of expected change plus storage / risk premia, not a pure forecast, and it covers 3 months, not 12."),
    "ex_demand": dict(method=MOD, fmt="pp", oil=1, src="WRPUPUS2 (target, 4-week average YoY) + curve, credit, NFCI, VIX, labor, claims, industry, CFNAI, M2, CPI, dollar",
                      how="Walk-forward ridge forecast of US petroleum product supplied (the standard 'demand' proxy), YoY %, 12 months later. Covid-era swings are clipped to +-25%. "
                          "No free point-in-time survey (IEA / OPEC / EIA STEO vintages) exists, so this is a model, not a survey."),
}
EXP_FRED = {"dgs1": "DGS1", "dgs2": "DGS2", "dgs10": "DGS10", "dgs3m": "DGS3MO", "expinf1y": "EXPINF1YR", "gdp": "GDPC1", "claims": "ICSA"}
EXP_LAGS = {"expinf1y": (1, 15), "gdp": (3, 30), "claims": (0, 5)}  # (months, days) after the observation date


def _g(m, k):
    return m[k] if k in m else pd.Series(np.nan, index=m.index)


def expanding_forecast(Z, y, h=EXP_H, step=EXP_STEP, minrows=EXP_MINTR, alpha=EXP_ALPHA):
    """Predict y (the value realised h months after each row) from Z using ONLY rows whose outcome was already known at that time.
    Row i's outcome is known at month s0 only if i + h <= s0, so every refit at s0 trains on rows <= s0 - h and predicts rows s0 .. s0 + step - 1."""
    Zv, yv = Z.values.astype(float), np.asarray(y, float)
    n = len(Zv)
    okz = np.isfinite(Zv).all(axis=1)
    out = np.full(n, np.nan)
    if not okz.any():
        return pd.Series(out, index=Z.index)
    first = int(np.argmax(okz))
    for s0 in range(first + minrows + h, n, step):
        te = np.arange(s0, min(s0 + step, n))
        lim = s0 - h + 1
        tr = np.where(okz[:lim] & np.isfinite(yv[:lim]))[0]
        if len(tr) < minrows:
            continue
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Zv[tr], yv[tr])
        te = te[okz[te]]
        if len(te):
            out[te] = mdl.predict(Zv[te])
    return pd.Series(out, index=Z.index)


def build_expectations(m):
    """Returns a frame (index = m.index) with the twelve ex_* columns. Columns that cannot be built (missing data) are all-NaN and get dropped upstream."""
    y1, y2, y10, y3m = _g(m, "dgs1"), _g(m, "dgs2"), _g(m, "dgs10"), _g(m, "dgs3m")
    ff = _g(m, "fed_funds").fillna(y3m)
    E = pd.DataFrame(index=m.index)

    # ---- market-implied (no fitting)
    fwd3m = 1.25 * y2 - 0.25 * y1                      # 3-month rate, 12 months ahead
    E["ex_ff12"] = (ff + (fwd3m - y3m)).clip(lower=0.0)
    E["ex_ff_chg"] = E["ex_ff12"] - ff
    infl = _g(m, "expinf1y")
    if not infl.notna().any():
        infl = _g(m, "breakeven")
    E["ex_infl"] = infl
    E["ex_real"] = E["ex_ff12"] - infl
    E["ex_y10"] = y10 + (y10 - y1) / 10.0
    E["ex_recess"] = 100.0 * ndtr(PROBIT_A + PROBIT_B * (y10 - y3m))
    E.loc[(y10 - y3m).isna(), "ex_recess"] = np.nan
    E["ex_curve"] = -_g(m, "backwd4")                   # (contract 4 - contract 1) / contract 1, in %  (needs the EIA futures curve)

    # ---- model-implied (walk-forward ridge, point-in-time)
    dollar, ur = _g(m, "dollar"), _g(m, "unrate_lvl")
    gy, ip = _g(m, "gdp_yoy"), _g(m, "indpro_yoy")
    claims = _g(m, "claims")
    Z = pd.DataFrame({"slope_10_3m": y10 - y3m, "slope_10_2": y10 - y2, "credit": _g(m, "credit"), "nfci": _g(m, "nfci"),
                      "vix": _g(m, "vix"), "sahm": _g(m, "sahm"), "indpro": ip, "philly": _g(m, "philly"), "m2": _g(m, "m2_yoy"),
                      "cpi": _g(m, "cpi_yoy"), "cfnai": _g(m, "cfnai"), "usd_3m": dollar.pct_change(3), "ff_6m": ff.diff(6),
                      "claims_rel": claims / claims.rolling(12).mean() - 1, "ex_ff_chg": E["ex_ff_chg"]}, index=m.index)
    Z = Z.replace([np.inf, -np.inf], np.nan)
    Z = Z.loc[:, Z.notna().any()]
    cutoff = Z.index[0] + pd.DateOffset(years=6)  # drop inputs that start so late they would cut the usable history
    Z = Z[[c for c in Z if Z[c].first_valid_index() is not None and Z[c].first_valid_index() <= cutoff]]
    h = EXP_H
    if Z.shape[1] >= 3:
        if gy.notna().any():
            E["ex_growth"] = expanding_forecast(Z.assign(own=gy), gy.shift(-h))
        if ur.notna().any():
            E["ex_unrate"] = ur + expanding_forecast(Z.assign(own=ur), ur.shift(-h) - ur)
        if dollar.notna().any():
            E["ex_usd"] = expanding_forecast(Z.assign(own=dollar.pct_change(12)), dollar.shift(-h) / dollar - 1)
        if ip.notna().any():
            E["ex_ind"] = expanding_forecast(Z, ip.shift(-h))
        dem = _g(m, "us_dem_yoy").clip(-25, 25)
        if dem.notna().any():
            E["ex_demand"] = expanding_forecast(Z.assign(own=dem), dem.shift(-h))
    return E.reindex(columns=EXP_KEYS)
