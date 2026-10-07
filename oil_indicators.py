"""
Oil Macro Analyzer: indicator definitions, data sources and feature builder (no Streamlit calls).

Everything is POINT-IN-TIME: every series is re-dated to the day it became public (LAGS), seasonal norms use only PAST years,
percentiles use only past months. Series that cannot be downloaded simply drop their indicators (see the "Data & coverage" tab).

Free-data limits, stated honestly:
  * OPEC / world supply and demand, OPEC spare capacity and the futures curve need a free EIA key (EIA_API_KEY). Without it the curve falls back to a USL/USO proxy.
  * Rig count (Baker Hughes) and OPEC+ production have no reliable free API. Use the optional file oil_manual_series.csv (see the Data tab)
    or the drilling-activity proxy that is downloaded automatically.
  * EIA STEO history, OECD leading indicators and some macro series are the CURRENT (revised) vintage, a mild look-ahead; first-release data is not used here.
"""
import numpy as np
import pandas as pd

from macro_engine import exp_pctl, seas_dev
from silver_expectations import EXP_FRED, EXP_INFO, EXP_KEYS, EXP_LAGS, EXP_META, EXP_PILLAR, build_expectations  # noqa: F401  (shared engine)

DEM, SUP, INV, MKT = "Demand & activity", "Supply: OPEC & US", "Inventories & refining", "Market structure & positioning"
USD, RISK, SHIP = "Dollar & monetary", "Risk & geopolitics", "Transport, freight & air travel"
PILLARS = [DEM, SUP, INV, MKT, USD, RISK, SHIP, EXP_PILLAR]

# key, title, label when LOW, label when HIGH, is a % change (shown as %), pillar, textbook sign for oil when HIGH (+1 / -1 / 0 mixed), in default set
IND = [
    # ---- demand & activity (global / US industrial activity, oil demand, travel)
    ("indpro_yoy", "US industrial production, YoY %", "Weak industry", "Strong industry", False, DEM, 1, True),
    ("philly", "Philly Fed manufacturing survey", "Weak factory activity", "Strong factory activity", False, DEM, 1, True),
    ("empire", "NY Fed Empire State manufacturing survey", "Weak factory activity", "Strong factory activity", False, DEM, 1, False),
    ("cli_us", "OECD leading indicator, US", "Weak outlook", "Strong outlook", False, DEM, 1, False),
    ("cli_cn", "OECD leading indicator, China", "Weak outlook", "Strong outlook", False, DEM, 1, False),
    ("copper_mom", "Copper, 3-month trend (global activity)", "Copper falling", "Copper rising", True, DEM, 1, True),
    ("em_mom", "Emerging-market stocks (EEM), 3-month trend", "EM falling", "EM rising", True, DEM, 1, False),
    ("china_mom", "China stocks (FXI), 3-month trend", "China falling", "China rising", True, DEM, 1, False),
    ("us_demand", "US petroleum product demand, YoY (4-week avg)", "Weak US demand", "Strong US demand", True, DEM, 1, True),
    ("gas_demand", "US gasoline demand, YoY (4-week avg)", "Weak gasoline demand", "Strong gasoline demand", True, DEM, 1, False),
    ("vmt_yoy", "US vehicle miles travelled, YoY", "Less driving", "More driving", True, DEM, 1, False),
    ("world_dem", "World oil demand (EIA), YoY", "Weak world demand", "Strong world demand", True, DEM, 1, True),
    ("world_bal", "World supply minus demand, mb/d (EIA)", "Market in deficit", "Market in surplus", False, DEM, -1, True),
    # ---- supply
    ("opec_chg", "OPEC crude production, 6-month change (mb/d)", "OPEC cutting", "OPEC raising", False, SUP, -1, True),
    ("opec_spare", "OPEC spare production capacity (mb/d)", "Tight spare capacity", "Ample spare capacity", False, SUP, -1, True),
    ("opecplus_chg", "OPEC+ production, 6-month change (manual file)", "OPEC+ cutting", "OPEC+ raising", False, SUP, -1, False),
    ("us_prod_yoy", "US crude production, YoY", "US output falling", "US output rising", True, SUP, -1, True),
    ("drill_yoy", "Oil & gas well drilling activity, YoY (rig-count proxy)", "Drilling falling", "Drilling rising", True, SUP, 0, False),
    ("rigs_chg", "Baker Hughes oil rig count, 6-month change (manual file)", "Rigs falling", "Rigs rising", True, SUP, 0, False),
    # ---- inventories & refining
    ("crude_dev", "US commercial crude stocks vs 5-year seasonal norm", "Stocks below norm (tight)", "Stocks above norm (glut)", True, INV, -1, True),
    ("gas_dev", "US gasoline stocks vs 5-year seasonal norm", "Gasoline stocks low", "Gasoline stocks high", True, INV, -1, False),
    ("dist_dev", "US distillate stocks vs 5-year seasonal norm", "Distillate stocks low", "Distillate stocks high", True, INV, -1, True),
    ("cushing_dev", "Cushing crude stocks vs 5-year seasonal norm", "Cushing stocks low", "Cushing stocks high", True, INV, -1, False),
    ("spr_chg", "Strategic Petroleum Reserve, 6-month change", "SPR draining (supply added)", "SPR refilling (demand added)", True, INV, 0, False),
    ("refutil_dev", "Refinery utilisation vs seasonal norm (pts)", "Low utilisation", "High utilisation", False, INV, 1, True),
    ("crack", "3-2-1 crack spread ($/bbl)", "Thin refining margins", "Fat refining margins", False, INV, 1, True),
    # ---- market structure & positioning
    ("term_slope", "Futures curve slope (4th vs 1st contract; EIA, or USL/USO proxy)", "Backwardation (tight)", "Contango (loose)", False, MKT, -1, True),
    ("oil_mom", "WTI, 3-month trend", "Oil falling", "Oil rising", True, MKT, 1, True),
    ("oil_mom12", "WTI, 12-month trend", "Oil down over a year", "Oil up over a year", True, MKT, 1, False),
    ("oil_val", "WTI vs CPI-adjusted history (percentile)", "Cheap vs its history", "Expensive vs its history", False, MKT, -1, True),
    ("brent_wti", "Brent minus WTI ($/bbl)", "Narrow spread", "Wide spread", False, MKT, 0, False),
    ("cot_net", "Speculative positioning (CFTC WTI managed-money net, % of open interest)", "Light speculative positioning", "Crowded speculative longs", False, MKT, 0, False),
    ("ovx", "Oil volatility (OVX)", "Calm oil market", "Nervous oil market", False, MKT, 0, False),
    ("xle_rel", "Energy stocks (XLE) vs S&P 500, 3-month relative", "Energy lagging", "Energy outperforming", True, MKT, 1, False),
    # ---- dollar & monetary
    ("dollar_mom", "US Dollar, 3-month trend", "Dollar weakening", "Dollar strengthening", True, USD, -1, True),
    ("real_yield", "Real 10y interest rate (level)", "Low / negative real rates", "High real rates", False, USD, -1, True),
    ("real_yield_chg", "Real rate, 3-month change", "Real rates falling", "Real rates rising", False, USD, -1, False),
    ("breakeven", "Inflation expectations (10y breakeven)", "Low inflation expectations", "High inflation expectations", False, USD, 1, True),
    ("fed_chg", "Fed funds rate, 6-month change", "Fed cutting", "Fed hiking", False, USD, -1, True),
    ("curve", "Yield curve (10y minus 2y)", "Flat / inverted curve", "Steep curve", False, USD, 0, False),
    ("cpi_yoy", "Inflation (CPI, year-on-year %)", "Low inflation", "High inflation", False, USD, 0, False),
    ("cpi_accel", "Inflation momentum (CPI YoY, 3-month change)", "Inflation decelerating", "Inflation accelerating", False, USD, 0, False),
    ("real_fed", "Real policy rate (Fed funds minus CPI)", "Low real policy rate", "High real policy rate", False, USD, -1, False),
    ("m2_yoy", "Money supply (M2) growth, YoY %", "Slow money growth", "Fast money growth", False, USD, 0, False),
    ("nfci", "Financial conditions (NFCI)", "Loose conditions", "Tight conditions", False, USD, -1, True),
    ("credit", "Credit spread (Moody's Baa minus 10y)", "Tight spreads (calm credit)", "Wide spreads (credit stress)", False, USD, -1, True),
    # ---- risk & geopolitics
    ("vix", "Market fear (VIX)", "Calm markets", "Fearful markets", False, RISK, -1, True),
    ("spx_mom", "Stocks (S&P 500), 3-month trend", "Stocks falling", "Stocks rising", True, RISK, 1, True),
    ("sahm", "Sahm gauge (unemployment rise)", "Labor market stable", "Labor market weakening", False, RISK, -1, True),
    ("gpr", "Geopolitical risk index (Caldara-Iacoviello)", "Low geopolitical risk", "High geopolitical risk", False, RISK, 1, False),
    ("gpr_chg", "Geopolitical risk, 3-month change", "Tensions easing", "Tensions rising", False, RISK, 1, False),
    ("gepu", "Global economic policy uncertainty", "Low uncertainty", "High uncertainty", False, RISK, 0, False),
    # ---- transport, freight & air travel
    ("freight_yoy", "US freight transportation index, YoY", "Freight falling", "Freight rising", True, SHIP, 1, False),
    ("tanker_mom", "Tanker shipping stocks (STNG, FRO), 3-month trend", "Tankers falling", "Tankers rising", True, SHIP, 1, False),
    ("air_rpm_yoy", "Air passenger miles (BTS), YoY", "Air travel falling", "Air travel rising", True, SHIP, 1, False),
    ("air_mom", "Airline stocks (JETS, DAL, UAL), 3-month trend", "Airlines falling", "Airlines rising", True, SHIP, 1, False),
]
META_ALL = {k: (t, lo, hi, pct, pl) for k, t, lo, hi, pct, pl, _, _ in IND}
SIGN = {k: s for k, _, _, _, _, _, s, _ in IND}
CORE = [k for k, *_, core in IND if core]
# textbook sign for oil of each expectation indicator when HIGH (silver_expectations carries the silver signs, which differ for recession risk)
OIL_EXP_SIGN = {"ex_ff12": -1, "ex_ff_chg": -1, "ex_real": -1, "ex_infl": 1, "ex_y10": -1, "ex_growth": 1, "ex_recess": -1,
                "ex_unrate": -1, "ex_usd": -1, "ex_ind": 1}
META_ALL.update(EXP_META)
SIGN.update(OIL_EXP_SIGN)

# ------------------------------------------------------------------ sources
FRED = {
    "oil_wti": "DCOILWTICO", "oil_brent": "DCOILBRENTEU",
    # weekly EIA data mirrored on FRED
    "crude_stk": "WCESTUS1", "gas_stk": "WGTSTUS1", "dist_stk": "WDISTUS1", "cushing": "WCESTP11", "spr": "WCSSTUS1",
    "us_prod": "WCRFPUS2", "refutil": "WPULEUS3", "prod_supplied": "WRPUPUS2", "gas_supplied": "WGFUPUS2",
    # monthly activity / transport
    "vmt": "TRFVOLUSM227NFWA", "freight": "TSIFRGHT", "air_rpm": "AIRRPMTSI", "drill_ip": "IPN213111N",
    "empire": "GACDISA066MSFRBNY", "cli_us": "USALOLITOAASTSAM", "cli_cn": "CHNLOLITOAASTSAM", "gepu": "GEPUCURRENT",
    # macro (same series as the silver app)
    "real_yield": "DFII10", "breakeven": "T10YIE", "fed_funds": "DFF", "curve": "T10Y2Y", "m2": "M2SL", "cpi": "CPIAUCSL",
    "indpro": "INDPRO", "nfci": "NFCI", "credit": "BAA10Y", "unrate": "UNRATE", "philly": "GACDFSA066MSFRBPHI", **EXP_FRED}
YF = {"dollar": "DX-Y.NYB", "oil_fut": "CL=F", "copper": "HG=F", "vix": "^VIX", "ovx": "^OVX", "spx": "^GSPC", "irx": "^IRX",
      "rb": "RB=F", "ho": "HO=F", "uso": "USO", "usl": "USL", "xle": "XLE", "eem": "EEM", "fxi": "FXI",
      "jets": "JETS", "dal": "DAL", "ual": "UAL", "stng": "STNG", "fro": "FRO"}
STEO_IDS = {"COPR_OPEC": "opec_prod", "COPS_OPEC": "opec_spare", "PAPR_WORLD": "world_prod", "PATC_WORLD": "world_cons"}
FUT_IDS = {"RCLC1": "c1", "RCLC4": "c4"}   # NYMEX WTI Cushing futures, contract 1 and 4 (EIA, daily)
COT_CODE = "067651"                        # CFTC: WTI crude oil, NYMEX
MANUAL_FILE = "oil_manual_series.csv"      # optional: date, rigs, opecplus_prod (see Data tab)

# (months, days) after the observation date; a key that is not listed is a daily series available the same day
LAGS = {"m2": (1, 20), "cpi": (1, 20), "indpro": (1, 20), "unrate": (1, 10), "philly": (0, 21), "nfci": (0, 7), "empire": (0, 17),
        "crude_stk": (0, 6), "gas_stk": (0, 6), "dist_stk": (0, 6), "cushing": (0, 6), "spr": (0, 6), "us_prod": (0, 6),
        "refutil": (0, 6), "prod_supplied": (0, 6), "gas_supplied": (0, 6),
        "vmt": (2, 15), "freight": (2, 10), "air_rpm": (2, 15), "drill_ip": (1, 20),
        "cli_us": (1, 10), "cli_cn": (1, 10), "gepu": (1, 10), "gpr": (1, 5),
        "opec_prod": (2, 15), "opec_spare": (2, 15), "world_prod": (2, 15), "world_cons": (2, 15),
        "rigs": (1, 15), "opecplus_prod": (1, 15), **EXP_LAGS}

# what each downloaded column is, for the coverage table: column -> (description, source)
SOURCES = {
    "oil": ("WTI spot price (target)", "FRED DCOILWTICO, else Yahoo CL=F"), "oil_brent": ("Brent spot", "FRED DCOILBRENTEU"),
    "crude_stk": ("US commercial crude stocks", "FRED / EIA weekly"), "gas_stk": ("US gasoline stocks", "FRED / EIA weekly"),
    "dist_stk": ("US distillate stocks", "FRED / EIA weekly"), "cushing": ("Cushing crude stocks", "FRED / EIA weekly"),
    "spr": ("Strategic Petroleum Reserve", "FRED / EIA weekly"), "us_prod": ("US crude production", "FRED / EIA weekly"),
    "refutil": ("Refinery utilisation", "FRED / EIA weekly"), "prod_supplied": ("US product supplied (demand)", "FRED / EIA weekly"),
    "gas_supplied": ("US gasoline product supplied", "FRED / EIA weekly"), "vmt": ("Vehicle miles travelled", "FRED / FHWA"),
    "freight": ("Freight transportation index", "FRED / BTS"), "air_rpm": ("Air revenue passenger miles", "FRED / BTS"),
    "drill_ip": ("Industrial production: oil & gas drilling", "FRED / Fed"), "empire": ("Empire State survey", "FRED / NY Fed"),
    "cli_us": ("OECD leading indicator, US", "FRED / OECD"), "cli_cn": ("OECD leading indicator, China", "FRED / OECD"),
    "gepu": ("Global economic policy uncertainty", "FRED"), "gpr": ("Geopolitical risk index", "matteoiacoviello.com (needs xlrd)"),
    "opec_prod": ("OPEC crude production", "EIA STEO (needs EIA key)"), "opec_spare": ("OPEC spare capacity", "EIA STEO (needs EIA key)"),
    "world_prod": ("World liquids production", "EIA STEO (needs EIA key)"), "world_cons": ("World liquids consumption", "EIA STEO (needs EIA key)"),
    "c1": ("WTI futures, contract 1", "EIA (needs EIA key)"), "c4": ("WTI futures, contract 4", "EIA (needs EIA key)"),
    "cot": ("CFTC WTI managed-money net, % of open interest", "CFTC"), "rigs": ("Baker Hughes oil rigs", "manual file"),
    "opecplus_prod": ("OPEC+ production", "manual file"), "uso": ("USO ETF (backtest vehicle)", "Yahoo"), "usl": ("USL ETF (curve proxy)", "Yahoo"),
    "jets": ("JETS airline ETF", "Yahoo"), "ovx": ("Oil volatility index", "Yahoo"), "rb": ("RBOB gasoline futures", "Yahoo"), "ho": ("Heating oil futures", "Yahoo"),
}


def _mean_mom(m, keys, n=3):
    cols = [m[k].pct_change(n) for k in keys if k in m]
    return pd.concat(cols, axis=1).mean(axis=1) if cols else pd.Series(np.nan, index=m.index)


def build_features(m):
    """Monthly availability-dated frame m -> indicator frame F (one column per indicator in META_ALL that has data)."""
    def g(k):
        return m[k] if k in m else pd.Series(np.nan, index=m.index)

    F = pd.DataFrame(index=m.index)
    oil = g("oil")
    # demand & activity
    for k in ("indpro_yoy", "philly", "empire", "cli_us", "cli_cn"):
        F[k] = g(k)
    F["copper_mom"], F["em_mom"], F["china_mom"] = g("copper").pct_change(3), g("eem").pct_change(3), g("fxi").pct_change(3)
    F["us_demand"], F["gas_demand"], F["vmt_yoy"] = g("prod_supplied").pct_change(12), g("gas_supplied").pct_change(12), g("vmt").pct_change(12)
    F["world_dem"] = g("world_cons").pct_change(12)
    F["world_bal"] = g("world_prod") - g("world_cons")
    # supply
    F["opec_chg"], F["opec_spare"], F["opecplus_chg"] = g("opec_prod").diff(6), g("opec_spare"), g("opecplus_prod").diff(6)
    F["us_prod_yoy"], F["drill_yoy"], F["rigs_chg"] = g("us_prod").pct_change(12), g("drill_ip").pct_change(12), g("rigs").pct_change(6)
    # inventories & refining (seasonal norm = same month, previous 5 years)
    F["crude_dev"], F["gas_dev"] = seas_dev(g("crude_stk")), seas_dev(g("gas_stk"))
    F["dist_dev"], F["cushing_dev"] = seas_dev(g("dist_stk")), seas_dev(g("cushing"))
    F["spr_chg"] = g("spr").pct_change(6)
    F["refutil_dev"] = seas_dev(g("refutil"), rel=False)
    F["crack"] = ((2 * g("rb") * 42 + g("ho") * 42) - 3 * g("oil_fut")) / 3
    # market structure
    if "c1" in m and "c4" in m:
        F["term_slope"] = ((g("c4") / g("c1") - 1) * 100).where(g("c1") > 5).clip(-15, 30)
    else:  # proxy: the 12-month-contract ETF (USL) beating the front-month ETF (USO) means the curve is in contango
        F["term_slope"] = (g("usl") / g("uso")).pct_change(3) * 100
    F["oil_mom"], F["oil_mom12"] = oil.pct_change(3).clip(-0.8, 1.5), oil.pct_change(12).clip(-0.9, 3)
    cpi = g("cpi_level")
    F["oil_val"] = exp_pctl(oil / cpi if cpi.notna().any() else oil)
    F["brent_wti"] = g("oil_brent") - g("oil_wti")
    F["cot_net"], F["ovx"] = g("cot"), g("ovx")
    F["xle_rel"] = (g("xle") / g("spx")).pct_change(3)
    # dollar & monetary
    F["dollar_mom"] = g("dollar").pct_change(3)
    F["real_yield"], F["real_yield_chg"], F["breakeven"] = g("real_yield"), g("real_yield").diff(3), g("breakeven")
    F["fed_chg"], F["curve"] = g("fed_funds").diff(6), g("curve")
    if F["fed_chg"].isna().all():
        F["fed_chg"] = g("irx").diff(6)
    F["cpi_yoy"], F["cpi_accel"], F["real_fed"] = g("cpi_yoy"), g("cpi_yoy").diff(3), g("fed_funds") - g("cpi_yoy")
    F["m2_yoy"], F["nfci"], F["credit"] = g("m2_yoy"), g("nfci"), g("credit")
    # risk & geopolitics
    F["vix"], F["spx_mom"], F["sahm"] = g("vix"), g("spx").pct_change(3), g("sahm")
    F["gpr"], F["gpr_chg"], F["gepu"] = g("gpr"), g("gpr").diff(3), g("gepu")
    # transport
    F["freight_yoy"], F["air_rpm_yoy"] = g("freight").pct_change(12), g("air_rpm").pct_change(12)
    F["tanker_mom"], F["air_mom"] = _mean_mom(m, ("stng", "fro")), _mean_mom(m, ("jets", "dal", "ual"))
    # expectations (shared engine: market-implied curve, Cleveland Fed inflation, walk-forward model estimates)
    E = build_expectations(m)
    for c in E.columns:
        F[c] = E[c]
    F = F.replace([np.inf, -np.inf], np.nan)
    keep = [c for c in META_ALL if c in F and F[c].notna().any()]
    return F[keep]
