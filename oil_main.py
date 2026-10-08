"""
Oil Macro Environment Analyzer (Streamlit), v1.0 (built on the oil analyzer v4.6)

Which macro environments have been historically favorable / unfavorable for the oil price (WTI), what regime are we in,
what happened in comparable periods, and would following it have worked (with costs and futures roll costs)?

requirements.txt: streamlit, yfinance, pandas, numpy, plotly, requests, scikit-learn, scipy, python-docx, kaleido, xlrd
Optional keys (Streamlit secrets or environment variables):
  FRED_API_KEY  reliable FRED access and first-release (ALFRED) data
  EIA_API_KEY   free key from eia.gov/opendata. Switches on OPEC production, OPEC spare capacity, world oil demand (EIA STEO),
                the real WTI futures curve (backwardation / contango, curve-implied expectation) and roll-adjusted futures returns.
Optional local files in the folder  oil_data/  next to the app (CSV, first column = date, second = value):
  rig_count.csv       Baker Hughes US oil rig count, weekly (no free API exists)
  bdi.csv             Baltic Dry Index, daily or weekly
  opecplus_prod.csv   OPEC+ crude production, monthly, million barrels / day
Educational only, not financial advice.

Files:
  app.py                      asset chooser (this file is started from there)
  oil_main.py                 config, data, helper functions, sidebar UI, calculations (this file)
  oil_tabs_overview.py        Dashboard, Guide, Environments, Indicator ranking, Out-of-sample
  oil_tabs_analysis.py        Regimes & analogues, Backtest, Robustness, Explorer
  oil_tabs_expectations.py    Expectations tab (market / macro / curve expectation indicators, grading, link to oil)
  oil_tabs_projection.py      Projection, Machine learning, Look-ahead scan, Oil & stocks now, Report (Word file of all tabs)
  oil_expectations.py         expectation-indicator engine (imported by this file)

What is different from the oil analyzer:
- INDICATORS. ~60 oil-specific indicators in eight pillars: Demand & growth (incl. fuel demand and air travel), Supply (US output, drilling, OPEC),
  Inventories & refining, Market structure & positioning (futures curve, CFTC), Shipping & freight, Dollar & monetary, Risk & geopolitics, Expectations.
- THE PRICE. Targets use the WTI SPOT price (FRED / EIA), not Yahoo's continuous futures, because continuous futures contain contango roll gaps that are not returns.
- THE SECOND ASSET. Gold is replaced by the S&P 500 (SPY): oil versus stocks, rotation strategies, and the 'Oil & stocks now' tab.
- THE BACKTEST. ETF = USO (fees and roll costs inside the price). Futures = roll-adjusted WTI futures excess return built from the EIA contract-1 / contract-2 prices
  (needs EIA_API_KEY), plus T-bill collateral. Without the key the futures option falls back to Yahoo's continuous series and says so.
- WEEKLY EIA DATA. Inventories, production, refinery utilization and fuel demand are weekly, usable from the Wednesday release, and are compared with the
  average of the same week in the previous five years (seasonality removed, point-in-time).
"""
import hashlib
import io
import json
import math
import os
import runpy
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.special import ndtri
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# NOTE: st.set_page_config is called in app.py (it must be the first Streamlit call).
_HERE0 = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
if _HERE0 not in sys.path:
    sys.path.insert(0, _HERE0)
from oil_expectations import (EXP_FRED, EXP_H, EXP_INFO, EXP_KEYS, EXP_LAGS, EXP_META, EXP_PILLAR, build_expectations)  # noqa: E402

FAV, NEU, UNF = "Favorable", "Neutral", "Unfavorable"
EXPO = {FAV: 1.0, NEU: 0.5, UNF: 0.0}
SGN = {FAV: 1, NEU: 0, UNF: -1}
VICON = {1: "🟢", 0: "⚪", -1: "🔴"}

# ---- FRED series (key -> series id). A series that fails to download only removes the indicators that need it.
FRED = {
    # monetary / macro (same as the oil app)
    "real_yield": "DFII10", "breakeven": "T10YIE", "be5": "T5YIE", "fed_funds": "DFF", "curve": "T10Y2Y",
    "m2": "M2SL", "cpi": "CPIAUCSL", "indpro": "INDPRO", "nfci": "NFCI", "credit": "BAA10Y", "unrate": "UNRATE",
    "philly": "GACDFSA066MSFRBPHI", "cfnai": "CFNAI",
    # US weekly petroleum statistics (EIA). Observation date = week-ending Friday, released the following Wednesday.
    "crude_stk": "WCESTUS1",   # commercial crude stocks excl. SPR, thousand barrels
    "gas_stk": "WGTSTUS1",     # motor gasoline stocks
    "dist_stk": "WDISTUS1",    # distillate fuel oil stocks
    "us_prod": "WCRFPUS2",     # US field production of crude, thousand b/d
    "refutil": "WPULEUS3",     # refinery utilization, % of operable capacity
    "dem_tot": "WRPUPUS2",     # product supplied, all petroleum products (the standard US demand proxy), thousand b/d
    "dem_gas": "WGFUPUS2",     # finished motor gasoline product supplied
    "dem_jet": "WKJUPUS2",     # kerosene-type jet fuel product supplied (air-travel activity)
    # monthly activity, travel, freight and uncertainty
    "drill_ip": "IPN213111S",  # industrial production, drilling oil and gas wells (rig-activity proxy)
    "airpax": "AIRRPMTSI",     # US airline revenue passenger miles (BTS)
    "freight": "TSIFRGHT",     # freight transportation services index (BTS)
    "gepu": "GEPUCURRENT",     # global economic policy uncertainty
    # WTI spot price (EIA daily)
    "wti_spot": "DCOILWTICO", "brent_spot": "DCOILBRENTEU",
    # more oil-specific series
    "cushing": "WCESTP11",       # Cushing, OK crude stocks (weekly)
    "spr": "WCSSTUS1",           # crude stocks including the Strategic Petroleum Reserve (weekly); SPR change = change of this series
    "vmt": "TRFVOLUSM227NFWA",   # vehicle miles travelled (FHWA, monthly)
    "empire": "GACDISA066MSFRBNY", "cli_us": "USALOLITOAASTSAM", "cli_cn": "CHNLOLITOAASTSAM",  # NY Fed survey, OECD leading indicators
    **EXP_FRED}
# Series that get revised and therefore can use first-release (ALFRED) values when a FRED key is present.
VINT_KEYS = ("m2", "cpi", "indpro", "unrate", "philly", "nfci", "cfnai", "drill_ip", "freight", "empire")
_WK = (0, 5)  # weekly EIA data: week ends Friday, released the following Wednesday
# Monthly series are only usable after publication: (months, days) added to the observation date.
# With 'First-release (ALFRED) data' on (needs a FRED key) the real release date is used when it looks sane; otherwise these approximate lags apply.
PUB_LAG = {"m2": (1, 20), "cpi": (1, 20), "indpro": (1, 20), "unrate": (1, 10), "philly": (0, 21), "nfci": (0, 7), "cfnai": (1, 25),
           "crude_stk": _WK, "gas_stk": _WK, "dist_stk": _WK, "us_prod": _WK, "refutil": _WK, "dem_tot": _WK, "dem_gas": _WK, "dem_jet": _WK,
           "drill_ip": (1, 20), "airpax": (2, 10), "freight": (2, 10), "gepu": (1, 10), "gpr": (1, 5), "wti_spot": (0, 1),
           "cushing": _WK, "spr": _WK, "vmt": (2, 15), "empire": (0, 17), "cli_us": (1, 10), "cli_cn": (1, 10), "brent_spot": (0, 1),
           "steo": (2, 10), "rig": _WK, "local_m": (1, 15), "bdi": (0, 1), "curve_fut": (0, 1), **EXP_LAGS}
YF = {"oil_fut": "CL=F", "brent": "BZ=F", "gasoline": "RB=F", "heating": "HO=F", "natgas": "NG=F", "dollar": "DX-Y.NYB", "copper": "HG=F",
      "vix": "^VIX", "ovx": "^OVX", "spx": "^GSPC", "tnx": "^TNX", "irx": "^IRX", "uso": "USO", "spy": "SPY",
      "xle": "XLE", "xop": "XOP", "oih": "OIH", "eem": "EEM", "fxi": "FXI",
      "fro": "FRO", "dht": "DHT", "stng": "STNG", "luv": "LUV", "dal": "DAL", "ual": "UAL", "jets": "JETS"}
# EIA open data (v2). Needs EIA_API_KEY. STEO = Short-Term Energy Outlook history; futures = daily NYMEX WTI contracts 1-4.
EIA_STEO = {"opec_prod": "COPR_OPEC", "opec_spare": "COPS_OPEC", "world_cons": "PATC_WORLD", "world_prod": "PAPR_WORLD"}
EIA_FUT = {"c1": "RCLC1", "c2": "RCLC2", "c3": "RCLC3", "c4": "RCLC4"}
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"  # Caldara-Iacoviello geopolitical risk index (monthly)
LOCAL_DIR = os.path.join(_HERE0, "oil_data")
LOCAL_CSV = {"rig_count": "rig_count.csv", "bdi": "bdi.csv", "opecplus_prod": "opecplus_prod.csv"}
# CFTC Commitments of Traders (Socrata API). WTI crude, NYMEX = contract code 067651. Tried in order; failure is non-fatal.
COT_DATASETS = ("72hh-3qpy", "kh3c-gbw2")
COT_CODE = "067651"
COT_LAG_DAYS = 3  # positions are as of Tuesday, published Friday (pushed later around US federal holidays)
HORIZONS = (1, 2, 3, 6, 9, 12)
DECAY_HORIZONS = (1, 2, 3, 6, 9, 12, 18, 24)
ML_MODELS = ["Logistic regression", "Random Forest", "Gradient Boosting"]
VOL_TARGET = 0.35
DD_FLOOR = 0.20  # "dd" / "ddb" targets: good when the forward max drawdown is shallower than this (oil falls further than oil's 15% floor in a normal year)
TARGETS = {"Oil's return": "ret",
           "Oil, volatility-scaled forward return": "vol",
           "Oil avoids a deep drawdown (yes / no)": "ddb",
           f"Oil's forward max drawdown (continuous, vs {DD_FLOOR:.0%} floor)": "dd",
           "Oil minus S&P 500 (relative)": "rel",
           "Oil minus cash (T-bills)": "cash"}
RISK_TARGETS = ("vol", "dd", "ddb")
DEM, SUP, INV, STR, SHP, MON, RSK = ("Demand & growth", "Supply", "Inventories & refining", "Market structure & positioning",
                                     "Shipping & freight", "Dollar & monetary", "Risk & geopolitics")
PILLARS = [DEM, SUP, INV, STR, SHP, MON, RSK, EXP_PILLAR]
STATES = ["Low", "Mid", "High"]
RULE_STEP = 6  # months between refits of the expanding-window rules
ANA_K = 10  # analogues used per month in the predictive analogue test
FUT_COL = "oil_fut"  # price column used for the 'Futures' backtest; becomes 'oil_tr' (roll-adjusted) when the EIA futures curve is available

# --- projection tab constants
PROJ_Q = (0.10, 0.25, 0.50, 0.75, 0.90)
PROJ_GRID = (1, 2, 3, 4, 6, 9, 12, 18, 24)
PROJ_LENGTHS = [6, 12, 18, 24]
PROJ_STEP = 12        # months between refits in the walk-forward grading
PROJ_ALPHA = 150.0    # ridge strength (strong on purpose: macro data is short and noisy)
PROJ_MINTR = 48       # minimum training rows
PROJ_PATHS = 1000     # simulated routes (even number: antithetic pairs)
PROJ_DD = 0.20
PJ_RIDGE, PJ_GBR, PJ_AVG = "Ridge + residual quantiles", "Gradient boosting (quantile)", "Average of both"
PROJ_MODELS = [PJ_RIDGE, PJ_GBR, PJ_AVG]
PROJ_FEATS = ["Macro + oil price momentum", "Macro indicators only", "Oil price momentum only",
              "Macro + oil momentum + expectations", "Expectations + oil momentum only"]
PROJ_NAMES = {"px_mom1": "Oil, 1-month return", "px_mom3": "Oil, 3-month return", "px_mom6": "Oil, 6-month return",
              "px_mom12": "Oil, 12-month return", "px_vol12": "Oil, 12-month volatility",
              "px_trend": "Oil vs its 10-month average", "px_dd12": "Oil, drop from 12-month high"}

STRATEGIES = [
    "Scaled: 100% / 50% / 0%",
    "Defensive: 100% / 25% / 0%",
    "Aggressive: 100% / 75% / 25%",
    "Only when Favorable",
    "Hold unless Unfavorable",
    "Rotate: oil if Favorable, else S&P 500",
    "Oil / S&P 500 / cash: Fav / Neutral / Unfav",
    "Trend only: oil above its 10-month average",
    "Macro + trend: not Unfavorable and above 10-month average",
    "Vol-capped scaled: 100% / 50% / 0%, 35% vol target",
]
NO_SIGNAL = {STRATEGIES[7]}  # strategies that ignore the macro signal
RANK_METRICS = ["Sharpe", "Sortino", "CAGR", "Calmar", "Max drawdown"]
NOW_KINDS = ("ret", "eqabs", "rel")
NOW_NAMES = {"ret": "Oil", "eqabs": "S&P 500", "rel": "Oil vs S&P 500"}
NOW_PHRASE = {"ret": "oil is higher", "eqabs": "the S&P 500 is higher", "rel": "oil beats the S&P 500"}
REL_TXT = {"Favorable": "Favors oil over stocks", "Leaning favorable": "Leans toward oil",
           "Neutral": "No clear preference", "Leaning unfavorable": "Leans toward stocks",
           "Unfavorable": "Favors stocks over oil"}
TGT_TXT = {"ret": "oil is higher", "vol": "oil's volatility-scaled forward return is positive",
           "dd": f"oil avoids a {DD_FLOOR:.0%}+ drawdown", "ddb": f"oil avoids a {DD_FLOOR:.0%}+ drawdown",
           "rel": "oil beats the S&P 500", "cash": "oil beats cash"}

# indicator -> (title, label when LOW, label when HIGH, is a % change?, pillar)
META_ALL = {
    # ---- Demand & growth
    "indpro_yoy":    ("US industrial production, YoY %", "Weak industry", "Strong industry", False, DEM),
    "philly":        ("Philly Fed manufacturing survey", "Weak factory activity", "Strong factory activity", False, DEM),
    "cfnai":         ("Chicago Fed National Activity Index", "Below-trend US activity", "Above-trend US activity", False, DEM),
    "sahm":          ("Sahm gauge (unemployment rise)", "Labor market stable", "Labor market weakening", False, DEM),
    "copper_mom":    ("Copper, 3-month trend (global-growth proxy)", "Copper falling", "Copper rising", True, DEM),
    "em_mom":        ("Emerging-market stocks (EEM), 3-month trend", "EM stocks falling", "EM stocks rising", True, DEM),
    "china_mom":     ("China stocks (FXI), 3-month trend", "China stocks falling", "China stocks rising", True, DEM),
    "us_dem_yoy":    ("US oil demand (product supplied), YoY %, 4-week average", "Weak US oil demand", "Strong US oil demand", False, DEM),
    "gas_dem_yoy":   ("US gasoline demand, YoY %", "Weak gasoline demand", "Strong gasoline demand", False, DEM),
    "jet_dem_yoy":   ("US jet-fuel demand, YoY % (air-travel activity)", "Weak air-travel fuel demand", "Strong air-travel fuel demand", False, DEM),
    "airpax_yoy":    ("US airline passenger miles, YoY %", "Weak air travel", "Strong air travel", False, DEM),
    "airlines_mom":  ("Airline stocks (LUV, DAL), 3-month trend", "Airline stocks falling", "Airline stocks rising", True, DEM),
    "empire":        ("NY Fed Empire State manufacturing survey", "Weak factory activity", "Strong factory activity", False, DEM),
    "cli_us":        ("OECD leading indicator, US", "Weak outlook", "Strong outlook", False, DEM),
    "cli_cn":        ("OECD leading indicator, China", "Weak outlook", "Strong outlook", False, DEM),
    "vmt_yoy":       ("US vehicle miles travelled, YoY %", "Less driving", "More driving", False, DEM),
    "world_dem_yoy": ("World oil consumption, YoY % (EIA STEO, needs EIA key)", "Weak global oil demand", "Strong global oil demand", False, DEM),
    # ---- Supply
    "us_prod_yoy":   ("US crude production, YoY %", "Falling / flat US output", "Rising US output", False, SUP),
    "drill_yoy":     ("Oil & gas drilling activity (industrial production), YoY %", "Drilling falling", "Drilling rising", False, SUP),
    "rig_yoy":       ("Baker Hughes rig count, YoY % (your CSV)", "Rig count falling", "Rig count rising", False, SUP),
    "oilsvc_mom":    ("Oilfield services (OIH), 3-month trend (rig-activity proxy)", "Services stocks falling", "Services stocks rising", True, SUP),
    "ep_rel":        ("E&P stocks (XOP) vs oil, 3-month relative", "E&P lagging oil", "E&P outperforming oil", True, SUP),
    "opec_chg":      ("OPEC crude production, 3-month change, mb/d (EIA STEO, needs EIA key)", "OPEC cutting", "OPEC raising", False, SUP),
    "opec_spare":    ("OPEC spare production capacity, mb/d (EIA STEO, needs EIA key)", "Little spare capacity", "Ample spare capacity", False, SUP),
    "opecplus_chg":  ("OPEC+ crude production, 3-month change, mb/d (your CSV)", "OPEC+ cutting", "OPEC+ raising", False, SUP),
    # ---- Inventories & refining
    "crude_stk_5y":  ("US crude stocks vs 5-year seasonal average, %", "Stocks below normal (tight)", "Stocks above normal (glut)", False, INV),
    "gas_stk_5y":    ("US gasoline stocks vs 5-year seasonal average, %", "Gasoline stocks low", "Gasoline stocks high", False, INV),
    "dist_stk_5y":   ("US distillate stocks vs 5-year seasonal average, %", "Distillate stocks low", "Distillate stocks high", False, INV),
    "cover_days":    ("US crude stock cover (days of demand)", "Thin cover", "Ample cover", False, INV),
    "cushing_dev":   ("Cushing crude stocks vs 5-year seasonal average, %", "Cushing stocks low", "Cushing stocks high", False, INV),
    "spr_chg":       ("Crude stocks incl. SPR, 6-month change, % (SPR draw / refill)", "Stocks (SPR) draining", "Stocks (SPR) refilling", False, INV),
    "world_bal":     ("World oil supply minus demand, mb/d (EIA STEO, needs EIA key)", "Market in deficit", "Market in surplus", False, INV),
    "refutil_dev":   ("US refinery utilization vs 5-year seasonal norm, pts", "Below-seasonal utilization", "Above-seasonal utilization", False, INV),
    "refutil":       ("US refinery utilization, %", "Low utilization", "High utilization", False, INV),
    "crack":         ("3-2-1 crack spread, $/bbl (refining margin)", "Thin refining margins", "Fat refining margins", False, INV),
    # ---- Market structure & positioning
    "backwd":        ("Futures curve: front minus 2nd contract, % of price (+ = backwardation; EIA key)", "Contango (loose market)", "Backwardation (tight market)", False, STR),
    "backwd4":       ("Futures curve: front minus 4th contract, % of price (EIA key)", "Steep contango", "Steep backwardation", False, STR),
    "roll_proxy":    ("Roll-yield proxy: oil ETF (USO) minus front futures, 3-month", "Contango drag", "Backwardation / no drag", True, STR),
    "cot_net":       ("Speculative positioning (CFTC managed-money net WTI, % of open interest)", "Light speculative positioning", "Crowded speculative longs", False, STR),
    "xle_rel":       ("Energy stocks (XLE) vs S&P 500, 3-month relative", "Energy lagging stocks", "Energy leading stocks", True, STR),
    "val_real":      ("Oil price vs CPI-adjusted history (percentile)", "Low vs its own history", "High vs its own history", False, STR),
    "oil_mom":       ("Oil, 3-month trend", "Oil falling", "Oil rising", True, STR),
    "oil_mom12":     ("Oil, 12-month trend", "Oil down over a year", "Oil up over a year", True, STR),
    "natgas_mom":    ("Natural gas, 3-month trend", "Gas falling", "Gas rising", True, STR),
    # ---- Shipping & freight
    "freight_yoy":   ("US freight transportation index, YoY %", "Weak freight volumes", "Strong freight volumes", False, SHP),
    "tanker_mom":    ("Tanker stocks (FRO, DHT), 3-month trend (tanker-market proxy)", "Tanker stocks falling", "Tanker stocks rising", True, SHP),
    "bdi_mom":       ("Baltic Dry Index, 3-month change (your CSV)", "Dry-bulk freight falling", "Dry-bulk freight rising", True, SHP),
    "brent_wti":     ("Brent minus WTI spread, $/bbl", "Narrow spread", "Wide spread (US crude cheap: export pull)", False, SHP),
    # ---- Dollar & monetary
    "real_yield":     ("Real 10y interest rate (level)", "Low / negative real rates", "High real rates", False, MON),
    "real_yield_chg": ("Real rate, 3-month change", "Real rates falling", "Real rates rising", False, MON),
    "breakeven":      ("Inflation expectations (10y breakeven)", "Low inflation expectations", "High inflation expectations", False, MON),
    "be5":            ("Inflation expectations (5y breakeven)", "Low 5y inflation expectations", "High 5y inflation expectations", False, MON),
    "fed_chg":        ("Fed funds rate, 6-month change", "Fed cutting", "Fed hiking", False, MON),
    "curve":          ("Yield curve (10y minus 2y)", "Flat / inverted curve", "Steep curve", False, MON),
    "cpi_yoy":        ("Inflation (CPI, year-on-year %)", "Low inflation", "High inflation", False, MON),
    "cpi_accel":      ("Inflation momentum (CPI YoY, 3-month change)", "Inflation decelerating", "Inflation accelerating", False, MON),
    "real_fed":       ("Real policy rate (Fed funds minus CPI)", "Low real policy rate", "High real policy rate", False, MON),
    "dollar_mom":     ("US Dollar, 3-month trend", "Dollar weakening", "Dollar strengthening", True, MON),
    "nom_yield":      ("10y Treasury yield (level)", "Low yields", "High yields", False, MON),
    "nom_yield_chg":  ("10y Treasury yield, 3-month change", "Yields falling", "Yields rising", False, MON),
    "m2_yoy":         ("Money supply (M2) growth, YoY %", "Slow money growth", "Fast money growth", False, MON),
    # ---- Risk & geopolitics
    "vix":           ("Market fear (VIX)", "Calm markets", "Fearful markets", False, RSK),
    "ovx":           ("Oil-market fear (OVX, crude oil volatility index; starts 2007)", "Calm oil market", "Fearful oil market", False, RSK),
    "ovx_vrp":       ("Oil options: implied (OVX) minus realized volatility, pts", "Options cheap vs realized", "Options rich (fear premium)", False, RSK),
    "ovx_chg":       ("Oil options: OVX, 3-month change", "Implied vol falling", "Implied vol rising", False, RSK),
    "gpr":           ("Geopolitical risk index (Caldara-Iacoviello)", "Low geopolitical risk", "High geopolitical risk", False, RSK),
    "gpr_chg":       ("Geopolitical risk, 3-month change", "Tensions easing", "Tensions rising", False, RSK),
    "gepu":          ("Global economic policy uncertainty", "Low policy uncertainty", "High policy uncertainty", False, RSK),
    "nfci":          ("Financial conditions (NFCI)", "Loose conditions", "Tight conditions", False, RSK),
    "credit":        ("Credit spread (Moody's Baa minus 10y)", "Tight spreads (calm credit)", "Wide spreads (credit stress)", False, RSK),
    "credit_chg":    ("Credit spread, 3-month change", "Spreads tightening", "Spreads widening", False, RSK),
    "spx_mom":       ("Stocks (S&P 500), 3-month trend", "Stocks falling", "Stocks rising", True, RSK),
}
META_ALL.update(EXP_META)  # expectation indicators (oil_expectations.py); off by default, see the sidebar switch
# Default selection: the topics the analysis is built around, without late-starting series (OVX 2007, Brent 2007, CFTC 2006, USO 2006, EEM 2003, airlines 2007)
CORE = ["real_yield", "real_yield_chg", "breakeven", "fed_chg", "curve", "dollar_mom",
        "indpro_yoy", "philly", "copper_mom", "sahm", "us_dem_yoy", "jet_dem_yoy", "world_dem_yoy",
        "us_prod_yoy", "drill_yoy", "opec_chg", "opec_spare",
        "crude_stk_5y", "gas_stk_5y", "dist_stk_5y", "refutil_dev", "crack",
        "backwd", "freight_yoy", "gpr", "vix", "nfci", "credit", "spx_mom"]
# Textbook sign for the oil price when each indicator is HIGH (+1 / -1 / 0 = mixed or no clear story). A checklist, not evidence: the ranking tab shows what the data says.
OIL_SIGN = {"indpro_yoy": 1, "philly": 1, "cfnai": 1, "sahm": -1, "copper_mom": 1, "em_mom": 1, "china_mom": 1, "us_dem_yoy": 1, "gas_dem_yoy": 1,
            "jet_dem_yoy": 1, "airpax_yoy": 1, "airlines_mom": 1, "world_dem_yoy": 1, "empire": 1, "cli_us": 1, "cli_cn": 1, "vmt_yoy": 1,
            "us_prod_yoy": -1, "drill_yoy": 0, "rig_yoy": 0, "oilsvc_mom": 1, "ep_rel": 0, "opec_chg": -1, "opec_spare": -1, "opecplus_chg": -1,
            "crude_stk_5y": -1, "gas_stk_5y": -1, "dist_stk_5y": -1, "cushing_dev": -1, "cover_days": -1, "refutil": 1, "refutil_dev": 1, "crack": 1,
            "spr_chg": 0, "world_bal": -1, "backwd": 1, "backwd4": 1, "roll_proxy": 1, "cot_net": 0, "xle_rel": 1, "val_real": -1, "oil_mom": 1,
            "oil_mom12": 1, "natgas_mom": 0, "freight_yoy": 1, "tanker_mom": 1, "bdi_mom": 1, "brent_wti": 0,
            "real_yield": -1, "real_yield_chg": -1, "breakeven": 1, "be5": 1, "fed_chg": -1, "curve": 0, "cpi_yoy": 0, "cpi_accel": 0, "real_fed": -1,
            "dollar_mom": -1, "nom_yield": -1, "nom_yield_chg": -1, "m2_yoy": 0, "vix": -1, "ovx": 0, "ovx_vrp": 0, "ovx_chg": 0, "gpr": 1, "gpr_chg": 1,
            "gepu": 0, "nfci": -1, "credit": -1, "credit_chg": -1, "spx_mom": 1}
OIL_SIGN.update({k: v["oil"] for k, v in EXP_INFO.items()})
SIGN_TXT = {1: "Positive", -1: "Negative", 0: "Mixed"}
# column of the monthly frame -> (description, source), for the Data & coverage tab
SOURCES = {
    "oil": ("WTI spot price (the target)", "FRED DCOILWTICO (Yahoo CL=F as fallback)"), "oil_fut": ("WTI front-month futures (continuous)", "Yahoo CL=F"),
    "oil_tr": ("WTI roll-adjusted futures excess return index", "built from EIA contracts 1 / 2 (needs EIA key)"), "brent_spot": ("Brent spot", "FRED DCOILBRENTEU"),
    "crude_stk_5y": ("US crude stocks vs 5-year norm", "FRED / EIA weekly"), "gas_stk_5y": ("US gasoline stocks vs norm", "FRED / EIA weekly"),
    "dist_stk_5y": ("US distillate stocks vs norm", "FRED / EIA weekly"), "cushing_dev": ("Cushing stocks vs norm", "FRED / EIA weekly"),
    "spr_chg": ("Crude stocks incl. SPR, change", "FRED / EIA weekly"), "us_prod_yoy": ("US crude production", "FRED / EIA weekly"),
    "refutil_dev": ("Refinery utilization vs norm", "FRED / EIA weekly"), "us_dem_yoy": ("US product supplied (demand)", "FRED / EIA weekly"),
    "gas_dem_yoy": ("US gasoline demand", "FRED / EIA weekly"), "jet_dem_yoy": ("US jet-fuel demand", "FRED / EIA weekly"),
    "vmt_yoy": ("Vehicle miles travelled", "FRED / FHWA"), "freight_yoy": ("Freight transportation index", "FRED / BTS"),
    "airpax_yoy": ("Air revenue passenger miles", "FRED / BTS"), "drill_yoy": ("Industrial production: oil & gas drilling", "FRED / Fed"),
    "empire": ("Empire State survey", "FRED / NY Fed"), "cli_us": ("OECD leading indicator, US", "FRED / OECD"), "cli_cn": ("OECD leading indicator, China", "FRED / OECD"),
    "gepu": ("Global economic policy uncertainty", "FRED"), "gpr": ("Geopolitical risk index", "matteoiacoviello.com (needs xlrd)"),
    "opec_chg": ("OPEC crude production", "EIA STEO (needs EIA key)"), "opec_spare": ("OPEC spare capacity", "EIA STEO (needs EIA key)"),
    "world_dem_yoy": ("World liquids consumption", "EIA STEO (needs EIA key)"), "world_bal": ("World supply minus demand", "EIA STEO (needs EIA key)"),
    "backwd": ("WTI curve: contract 1 vs 2", "EIA (needs EIA key)"), "backwd4": ("WTI curve: contract 1 vs 4", "EIA (needs EIA key)"),
    "cot": ("CFTC WTI managed-money net, % of open interest", "CFTC"), "rig_yoy": ("Baker Hughes oil rigs", "your CSV (oil_data/)"),
    "opecplus_chg": ("OPEC+ production", "your CSV (oil_data/)"), "bdi": ("Baltic Dry Index", "your CSV (oil_data/)"),
    "uso": ("USO ETF (backtest vehicle)", "Yahoo"), "spy": ("SPY (the alternative asset)", "Yahoo"), "ovx": ("Oil volatility index (options-implied)", "Yahoo ^OVX"),
    "oil_rv": ("WTI realized volatility, 21 days", "computed from the price"), "jets": ("JETS airline ETF", "Yahoo"),
}
META = dict(META_ALL)  # narrowed to indicators in use once data loads (factor entries are added to META_ALL at run time)
# (indicator, sign): growth is NOT measured with stocks (stocks are risk appetite, not growth)
GROWTH = [("copper_mom", 1), ("indpro_yoy", 1), ("philly", 1), ("us_dem_yoy", 1), ("sahm", -1)]
INFL = [("cpi_yoy", 1), ("breakeven", 1), ("be5", 1)]
REGIMES = ["Goldilocks (growth↑ inflation↓)", "Reflation (growth↑ inflation↑)",
           "Stagflation (growth↓ inflation↑)", "Slowdown / deflation (growth↓ inflation↓)"]
DEMAND_PILLARS, SUPPLY_PILLARS = [DEM, MON, SHP], [SUP, INV, STR]


def show_df(box, d):
    """Works on both old and new Streamlit (use_container_width was replaced by width='stretch')."""
    try:
        box.dataframe(d, hide_index=True, width="stretch")
    except Exception:  # noqa: BLE001
        box.dataframe(d, hide_index=True, use_container_width=True)


def show_plot(box, f):
    try:
        box.plotly_chart(f, width="stretch")
    except Exception:  # noqa: BLE001
        box.plotly_chart(f, use_container_width=True)


PEEK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", ".final_test_peeks_oil.json")


def register_peek(sig):
    """Count how many DISTINCT sets of settings have been shown the final test (persisted next to the app, session fallback)."""
    h_ = hashlib.md5(sig.encode()).hexdigest()
    try:
        with open(PEEK_FILE) as f:
            seen = json.load(f)
    except Exception:  # noqa: BLE001
        seen = list(st.session_state.get("peeks", []))
    new = h_ not in seen
    if new:
        seen = seen + [h_]
        st.session_state["peeks"] = seen
        try:
            with open(PEEK_FILE, "w") as f:
                json.dump(seen, f)
        except Exception:  # noqa: BLE001
            pass
    return len(seen), new


def card(box):
    """Bordered container on new Streamlit, plain container on old versions."""
    try:
        return box.container(border=True)
    except TypeError:
        return box.container()


def vmark(f, x, text):
    """Vertical dashed marker (add_vline breaks on datetime axes in some plotly versions)."""
    f.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(dash="dash", color="#888"))
    f.add_annotation(x=x, y=1, yref="paper", text=text, showarrow=False, yanchor="bottom")


def state_label(col, state):
    return {"Low": META_ALL[col][1], "Mid": "Neutral", "High": META_ALL[col][2]}[state]


def fmt_val(c, v):
    if pd.isna(v):
        return "-"
    return f"{v:+.1%}" if META_ALL[c][3] else f"{v:.2f}"


# ------------------------------------------------------------------ data
DATA_START = "1990-01-01"  # always download the full history once; the sidebar start date only slices it

def get_fred_key():
    key = None
    try:
        key = st.secrets.get("FRED_API_KEY")
    except Exception:  # noqa: BLE001
        pass
    return key or os.environ.get("FRED_API_KEY")

def _fred_fetch(series_id, start, key):
    """Plain function (no Streamlit calls) so it is safe inside worker threads."""
    last = None
    for _ in range(2):
        try:
            if key:
                r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                                 params=dict(series_id=series_id, api_key=key, file_type="json",
                                             observation_start=start), timeout=(5, 20))
                r.raise_for_status()
                d = pd.DataFrame(r.json()["observations"])[["date", "value"]]
                d["value"] = pd.to_numeric(d["value"], errors="coerce")
            else:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
                r = requests.get(url, timeout=(5, 20), headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                d = pd.read_csv(io.StringIO(r.text), na_values=[".", ""])
                d.columns = ["date", "value"]
            d["date"] = pd.to_datetime(d["date"])
            return d.set_index("date")["value"].astype(float)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last

def get_eia_key():
    key = None
    try:
        key = st.secrets.get("EIA_API_KEY")
    except Exception:  # noqa: BLE001
        pass
    return key or os.environ.get("EIA_API_KEY")

def _cot_release_dates(dates):
    """Estimated release date of each COT report. The API gives only the Tuesday as-of date, so the release is CONSTRUCTED: Friday of that week,
    one business day later if a US federal holiday falls between the Monday of the report week and that Friday (conservative), and rolled
    forward past weekends / holidays. Conservative means never earlier than the real release, so no look-ahead.
    It is still an estimate: audit it against the CFTC's published release calendar before relying on it."""
    try:
        hol = USFederalHolidayCalendar().holidays(start=dates.min() - pd.Timedelta(days=10),
                                                  end=dates.max() + pd.Timedelta(days=20)).values.astype("datetime64[D]")
        d0 = dates.values.astype("datetime64[D]")
        fri = d0 + np.timedelta64(COT_LAG_DAYS, "D")
        mon = d0 - np.timedelta64(1, "D")
        delay = np.array([bool(((hol >= a) & (hol <= b)).any()) for a, b in zip(mon, fri)]).astype("int64").astype("timedelta64[D]")
        return pd.DatetimeIndex(np.busday_offset(fri + delay, 0, roll="forward", holidays=hol))
    except Exception:  # noqa: BLE001
        return pd.DatetimeIndex(dates) + pd.Timedelta(days=COT_LAG_DAYS)

def _cot_fetch(start):
    """Managed-money net position in NYMEX WTI crude as % of open interest (weekly), indexed by an estimated release date.
    Plain function (no Streamlit calls). Tries each known CFTC dataset id; raises if none works."""
    last = None
    for ds in COT_DATASETS:
        try:
            r = requests.get(f"https://publicreporting.cftc.gov/resource/{ds}.json", timeout=(5, 30),
                             params={"$where": f"cftc_contract_market_code='{COT_CODE}' AND report_date_as_yyyy_mm_dd >= '{start}T00:00:00.000'",
                                     "$select": "report_date_as_yyyy_mm_dd,m_money_positions_long_all,m_money_positions_short_all,open_interest_all",
                                     "$order": "report_date_as_yyyy_mm_dd", "$limit": 50000})
            r.raise_for_status()
            d = pd.DataFrame(r.json())
            if d.empty:
                raise ValueError("empty COT response")
            d["date"] = pd.to_datetime(d["report_date_as_yyyy_mm_dd"])
            for c in ("m_money_positions_long_all", "m_money_positions_short_all", "open_interest_all"):
                d[c] = pd.to_numeric(d[c], errors="coerce")
            s = ((d["m_money_positions_long_all"] - d["m_money_positions_short_all"]) / d["open_interest_all"] * 100)
            s.index = _cot_release_dates(pd.DatetimeIndex(d["date"]))
            s = s.dropna()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if len(s) < 100:
                raise ValueError("too few COT observations")
            return s.astype(float)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last

def month_end(df):
    try:
        return df.resample("ME").last()
    except ValueError:
        return df.resample("M").last()

def _eia_pages(url, params, page=5000):
    """All pages of an EIA open-data (v2) request. Plain function (no Streamlit calls) so it is safe inside worker threads."""
    rows, off = [], 0
    while True:
        r = requests.get(url, params=params + [("offset", off), ("length", page)], timeout=(5, 60))
        r.raise_for_status()
        resp = r.json()["response"]
        got = resp.get("data", [])
        rows += got
        off += len(got)
        if not got or off >= int(resp.get("total", 0) or 0) or off > 300000:
            break
    return rows


def _eia_steo(key, ids, start):
    """EIA Short-Term Energy Outlook history (monthly): OPEC production, OPEC spare capacity, world consumption.
    LATEST-REVISED values, not first releases, so this is only approximately point-in-time (a 2 month + 10 day lag is applied).
    STEO also contains forecasts: months after the previous calendar month are dropped."""
    params = [("api_key", key), ("frequency", "monthly"), ("data[0]", "value"), ("start", start[:7]),
              ("sort[0][column]", "period"), ("sort[0][direction]", "asc")] + [("facets[seriesId][]", i) for i in ids.values()]
    d = pd.DataFrame(_eia_pages("https://api.eia.gov/v2/steo/data/", params))
    if d.empty:
        raise ValueError("empty STEO response")
    d["period"], d["value"] = pd.to_datetime(d["period"]), pd.to_numeric(d["value"], errors="coerce")
    inv = {v: k for k, v in ids.items()}
    cutoff = pd.Timestamp.today().to_period("M").to_timestamp() - pd.DateOffset(months=1)
    out = {}
    for sid, g_ in d.groupby("seriesId"):
        s = g_.set_index("period")["value"].dropna().sort_index()
        out[inv[sid]] = s[s.index <= cutoff].astype(float)
    if not out:
        raise ValueError("no STEO series")
    return out


def _eia_futures(key, start):
    """Daily NYMEX WTI futures prices for contracts 1-4 (EIA). Returns a frame with columns c1..c4."""
    params = [("api_key", key), ("frequency", "daily"), ("data[0]", "value"), ("start", start),
              ("sort[0][column]", "period"), ("sort[0][direction]", "asc")] + [("facets[series][]", s) for s in EIA_FUT.values()]
    d = pd.DataFrame(_eia_pages("https://api.eia.gov/v2/petroleum/pri/fut/data/", params))
    if d.empty:
        raise ValueError("empty futures response")
    d["period"], d["value"] = pd.to_datetime(d["period"]), pd.to_numeric(d["value"], errors="coerce")
    w = d.pivot_table(index="period", columns="series", values="value").sort_index()
    w = w.rename(columns={v: k for k, v in EIA_FUT.items()})
    if "c1" not in w or w["c1"].notna().sum() < 500:
        raise ValueError("too few futures observations")
    return w


def _wti_expiries(start, end):
    """Last trading day of each WTI contract: 3 business days before the 25th calendar day of the month before delivery
    (if the 25th is not a business day, 3 business days before the business day just before it). Uses the US federal holiday calendar,
    which differs from NYMEX's in a day or two a year; a one-day error only moves a roll adjustment by one day (it nets out within the month)."""
    cbd = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    out = []
    for p in pd.period_range(pd.Timestamp(start) - pd.DateOffset(months=1), pd.Timestamp(end) + pd.DateOffset(months=1), freq="M"):
        d25 = pd.Timestamp(p.year, p.month, 25)
        base = d25 if cbd.is_on_offset(d25) else cbd.rollback(d25)
        out.append(base - 3 * cbd)
    return pd.DatetimeIndex(out)


def _oil_tr(c1, c2):
    """Roll-adjusted excess return index of a rolling long position in the front WTI contract (monthly compounding, 100 at the start).
    Daily P&L in DOLLARS (so the negative prices of April 2020 cause no trouble): on a roll day the P&L is today's new front price minus
    yesterday's SECOND contract price (the contract that became the front), otherwise today's minus yesterday's front price.
    Within a month the P&L is summed and divided by the previous month-end front price; months are then compounded. The value at every
    month-end therefore equals the compounded monthly return, and values in between are month-to-date."""
    d = pd.concat([c1, c2], axis=1, keys=["c1", "c2"]).dropna(subset=["c1"])
    idx = d.index
    roll = np.zeros(len(idx), bool)
    pos = idx.searchsorted(_wti_expiries(idx[0], idx[-1]), side="right")  # first trading day after each expiry
    roll[pos[(pos > 0) & (pos < len(idx))]] = True
    prev1, prev2 = d["c1"].shift(1), d["c2"].shift(1).fillna(d["c1"].shift(1))
    pnl = (d["c1"] - np.where(roll, prev2, prev1)).fillna(0.0)
    ym = idx.to_period("M")
    mtd = pnl.groupby(ym).cumsum()
    month_end_px = d["c1"].groupby(ym).last()
    base = pd.Series(ym.map(month_end_px.shift(1)).astype(float), index=idx)
    frac = (mtd / base).where(base > 0)
    month_ret = frac.groupby(ym).last()  # full-month (or month-to-date) return
    prev_level = (1 + month_ret.fillna(0.0)).cumprod().shift(1).fillna(1.0)
    level = pd.Series(ym.map(prev_level).astype(float), index=idx) * (1 + frac)
    return (100 * level).dropna()


def _gpr_fetch():
    """Caldara-Iacoviello geopolitical risk index (monthly, from newspaper counts). Needs the xlrd package for the .xls file."""
    r = requests.get(GPR_URL, timeout=(5, 30), headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    d = pd.read_excel(io.BytesIO(r.content))
    cols = {str(c).strip().lower(): c for c in d.columns}
    dc, vc = cols.get("month") or cols.get("date"), cols.get("gpr")
    if dc is None or vc is None:
        raise ValueError("unexpected GPR file layout")
    s = pd.Series(pd.to_numeric(d[vc], errors="coerce").values, index=pd.to_datetime(d[dc], errors="coerce")).dropna()
    s = s[s.index >= "1985-01-01"]
    if len(s) < 100:
        raise ValueError("too few GPR observations")
    return s.astype(float)


def _local_csv(name):
    """Optional user-supplied series in oil_data/<name>: first column date, second column value. Returns None when the file does not exist."""
    p = os.path.join(LOCAL_DIR, name)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    s = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(d.iloc[:, 0], errors="coerce")).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < 20:
        raise ValueError(f"{name}: too few rows")
    return s.astype(float)

def _yahoo_fetch(start):
    px = yf.download(list(YF.values()), start=start, auto_adjust=True, progress=False, threads=True)["Close"]
    return px.rename(columns={v: k for k, v in YF.items()})

def _avail(s, k, rel):
    """Re-index an observation-dated series by the date each value became available.
    With first-release data the real release date is used when it looks sane (0-150 days after the observation); otherwise the PUB_LAG rule."""
    months, days = PUB_LAG[k]
    lagged = (s.index + pd.DateOffset(months=months) + pd.Timedelta(days=days)).values
    idx = lagged
    if k in rel:
        rv = rel[k].reindex(s.index).values
        ok = (~pd.isna(rv)) & (rv >= s.index.values) & (rv <= (s.index + pd.Timedelta(days=150)).values)
        idx = np.where(ok, rv, lagged)
    out = pd.Series(s.values, index=pd.DatetimeIndex(idx))
    return out[~out.index.duplicated(keep="last")].sort_index()

def _seasonal_dev(s, years=5, rel=True):
    """Weekly level versus the average of the same week in each of the previous `years` years, in % (point-in-time: only past years are used).
    Needs at least 3 of the 5 previous years. Shifts of 52 weeks per year: the calendar drifts by about a day a year, which is irrelevant here."""
    s = s.dropna()
    ref = pd.concat([s.shift(52 * k) for k in range(1, years + 1)], axis=1)
    avg = ref.mean(axis=1).where(ref.notna().sum(axis=1) >= 3)
    return (s / avg - 1) * 100 if rel else (s - avg)


def _prepare_monthly(px, fr, failed, rel=None):
    rel = rel or {}
    for k in ("m2", "cpi", "indpro"):  # monthly series: only usable after publication
        if k in fr:
            raw = fr.pop(k)
            fr[k + "_yoy"] = _avail(raw.pct_change(12) * 100, k, rel).dropna()
            if k == "cpi":
                fr["cpi_level"] = _avail(raw, k, rel)
    if "gdp" in fr:  # quarterly real GDP: YoY growth, usable only after the advance estimate
        fr["gdp_yoy"] = _avail(fr.pop("gdp").pct_change(4) * 100, "gdp", rel).dropna()
    if "claims" in fr:  # weekly initial claims: 4-week average, usable from the Thursday release
        fr["claims"] = _avail(fr["claims"].rolling(4).mean(), "claims", rel).dropna()
    if "expinf1y" in fr:  # Cleveland Fed 1-year expected inflation (monthly)
        fr["expinf1y"] = _avail(fr["expinf1y"], "expinf1y", rel).dropna()
    if "unrate" in fr:  # Sahm gauge: 3-month average unemployment minus its minimum over the previous 12 months
        ur = fr.pop("unrate")
        fr["unrate_lvl"] = _avail(ur, "unrate", rel).dropna()
        ma3 = ur.rolling(3).mean()
        fr["sahm"] = _avail(ma3 - ma3.shift(1).rolling(12).min(), "unrate", rel).dropna()
    for k in ("philly", "nfci", "cfnai", "gepu", "empire", "cli_us", "cli_cn"):
        if k in fr:
            fr[k] = _avail(fr[k], k, rel).dropna()

    # ---- oil: weekly EIA statistics (4-week averages for flows, seasonal comparison for stocks)
    dem = fr.pop("dem_tot", None)
    dem4 = dem.rolling(4).mean() if dem is not None else None
    if dem4 is not None:
        fr["us_dem_yoy"] = _avail((dem4.pct_change(52) * 100).clip(-40, 40), "dem_tot", rel).dropna()
    if "dem_gas" in fr:
        fr["gas_dem_yoy"] = _avail((fr.pop("dem_gas").rolling(4).mean().pct_change(52) * 100).clip(-40, 40), "dem_gas", rel).dropna()
    if "dem_jet" in fr:
        fr["jet_dem_yoy"] = _avail((fr.pop("dem_jet").rolling(4).mean().pct_change(52) * 100).clip(-60, 60), "dem_jet", rel).dropna()
    if "crude_stk" in fr:
        stk = fr.pop("crude_stk")
        fr["crude_stk_5y"] = _avail(_seasonal_dev(stk), "crude_stk", rel).dropna()
        if dem4 is not None:
            fr["cover_days"] = _avail((stk / dem4.reindex(stk.index)).replace([np.inf, -np.inf], np.nan), "crude_stk", rel).dropna()
    if "gas_stk" in fr:
        fr["gas_stk_5y"] = _avail(_seasonal_dev(fr.pop("gas_stk")), "gas_stk", rel).dropna()
    if "dist_stk" in fr:
        fr["dist_stk_5y"] = _avail(_seasonal_dev(fr.pop("dist_stk")), "dist_stk", rel).dropna()
    if "us_prod" in fr:
        fr["us_prod_yoy"] = _avail(fr.pop("us_prod").rolling(4).mean().pct_change(52) * 100, "us_prod", rel).dropna()
    if "refutil" in fr:
        ru = fr["refutil"].rolling(4).mean()
        fr["refutil"] = _avail(ru, "refutil", rel).dropna()
        fr["refutil_dev"] = _avail(_seasonal_dev(ru, rel=False), "refutil", rel).dropna()  # percentage points above / below the same week of the previous 5 years
    if "cushing" in fr:
        fr["cushing_dev"] = _avail(_seasonal_dev(fr.pop("cushing")), "cushing", rel).dropna()
    if "spr" in fr:  # crude stocks incl. SPR: the change over 26 weeks isolates SPR policy and total stock building
        fr["spr_chg"] = _avail(fr.pop("spr").pct_change(26) * 100, "spr", rel).dropna()
    if "vmt" in fr:
        fr["vmt_yoy"] = _avail((fr.pop("vmt").pct_change(12) * 100).clip(-50, 60), "vmt", rel).dropna()
    if "brent_spot" in fr:
        fr["brent_spot"] = _avail(fr["brent_spot"], "brent_spot", rel).dropna()

    # ---- oil: monthly activity, travel, freight, geopolitics
    if "drill_ip" in fr:
        fr["drill_yoy"] = _avail(fr.pop("drill_ip").pct_change(12) * 100, "drill_ip", rel).dropna()
    if "airpax" in fr:  # Covid: -95% then +300% base effects are clipped
        fr["airpax_yoy"] = _avail((fr.pop("airpax").pct_change(12) * 100).clip(-60, 80), "airpax", rel).dropna()
    if "freight" in fr:
        fr["freight_yoy"] = _avail(fr.pop("freight").pct_change(12) * 100, "freight", rel).dropna()
    if "gpr" in fr:
        g_ = fr["gpr"]
        fr["gpr"], fr["gpr_chg"] = _avail(g_, "gpr", rel).dropna(), _avail(g_.diff(3), "gpr", rel).dropna()

    if "wti_spot" in fr:
        fr["wti_spot"] = _avail(fr["wti_spot"], "wti_spot", rel).dropna()

    # ---- oil: EIA STEO (OPEC, world demand) and optional local CSV series
    if "opec_prod" in fr:
        fr["opec_chg"] = _avail(fr.pop("opec_prod").diff(3), "steo", rel).dropna()
    if "opec_spare" in fr:
        fr["opec_spare"] = _avail(fr["opec_spare"], "steo", rel).dropna()
    if "world_prod" in fr and "world_cons" in fr:
        fr["world_bal"] = _avail(fr.pop("world_prod") - fr["world_cons"], "steo", rel).dropna()
    if "world_cons" in fr:
        fr["world_dem_yoy"] = _avail((fr.pop("world_cons").pct_change(12) * 100).clip(-25, 25), "steo", rel).dropna()
    if "rig_count" in fr:
        fr["rig_yoy"] = _avail(fr.pop("rig_count").rolling(4).mean().pct_change(52) * 100, "rig", rel).dropna()
    if "bdi" in fr:
        fr["bdi"] = _avail(fr["bdi"], "bdi", rel).dropna()
    if "opecplus_prod" in fr:
        fr["opecplus_chg"] = _avail(fr.pop("opecplus_prod").diff(3), "local_m", rel).dropna()

    # ---- oil: futures curve (EIA contracts) -> backwardation indicators and the roll-adjusted return index
    if "curve_fut" in fr:
        w = fr.pop("curve_fut")
        c1, c2, c4 = w["c1"], w.get("c2"), w.get("c4")
        ok1 = c1 > 5  # near-zero / negative prices (April 2020) make a % of price meaningless
        if c2 is not None:
            fr["backwd"] = _avail(((c1 - c2) / c1 * 100).where(ok1).clip(-15, 15), "curve_fut", rel).dropna()
            fr["oil_tr"] = _avail(_oil_tr(c1, c2), "curve_fut", rel)
        if c4 is not None:
            fr["backwd4"] = _avail(((c1 - c4) / c1 * 100).where(ok1).clip(-30, 30), "curve_fut", rel).dropna()

    daily = px.join(pd.concat(fr, axis=1), how="outer").sort_index() if fr else px.copy()
    daily = daily.loc[: px.index[-1]]  # availability dates after the last price are not known yet
    fut = daily["oil_fut"].ffill() if "oil_fut" in daily else None
    if "wti_spot" in daily and daily["wti_spot"].notna().any():  # the target price is the SPOT price: futures roll gaps are not returns
        sp = daily["wti_spot"]
        last = sp.last_valid_index()
        if fut is not None and fut.loc[last] > 0 and sp.loc[last] > 0:
            oil = sp.ffill().where(daily.index <= last, fut * (sp.loc[last] / fut.loc[last]))  # days after the last spot print: futures, scaled to the spot level
        else:
            oil = sp.ffill()
        daily["oil"] = oil
    else:
        daily["oil"] = fut
    daily = daily.ffill().dropna(subset=["oil"])
    daily = daily[daily["oil"] > 0]
    o_ = daily["oil"].reindex(px.index.intersection(daily.index)).where(lambda x: x > 5)  # business days only; drops the Apr-2020 near-zero prints
    daily["oil_rv"] = (o_.pct_change().rolling(21, min_periods=15).std() * np.sqrt(252) * 100).reindex(daily.index).ffill()
    return month_end(daily).dropna(subset=["oil"]), sorted(set(failed))

def exp_pctl(s, minp=36):
    """Point-in-time percentile: where today's value ranks among all PREVIOUS values."""
    d = s.dropna()
    p = d.expanding(min_periods=minp).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan, raw=True)
    return p.reindex(s.index)

@st.cache_data(show_spinner=False)
def pit_percentiles(F_src):
    return F_src.apply(exp_pctl)

def _fred_vintage(series_id, start, key):
    """First-release values and release dates (FRED output_type=4: initial release only). Returns (values by observation date, release dates).
    Observations older than the first archived vintage carry that earliest archived value (already revised); the sanity check in _avail then falls back to PUB_LAG."""
    r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=(5, 60),
                     params=dict(series_id=series_id, api_key=key, file_type="json", observation_start=start,
                                 realtime_start="1776-07-04", realtime_end="9999-12-31", output_type=4))
    r.raise_for_status()
    d = pd.DataFrame(r.json()["observations"])
    d["date"] = pd.to_datetime(d["date"])
    d["rel"] = pd.to_datetime(d["realtime_start"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value"]).drop_duplicates("date", keep="first").set_index("date").sort_index()
    return d["value"].astype(float), d["rel"]

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_dataset(key, eia_key=None, vintage=False):
    """Download EVERYTHING once (Yahoo + all FRED series + CFTC + GPR + EIA in parallel), build monthly data and features, and cache it.
    vintage=True (needs a FRED key) uses first-release values and release dates for M2, CPI, industrial production, unemployment, Philly Fed, NFCI,
    CFNAI, drilling activity and the freight index. Exceptions are not cached, so a total failure is retried on the next run."""
    vint_keys = VINT_KEYS if (vintage and key) else ()
    extra, xfail = {}, []
    with ThreadPoolExecutor(max_workers=len(FRED) + 8) as ex:
        fy = ex.submit(_yahoo_fetch, DATA_START)
        ff = {k: ex.submit(_fred_vintage if k in vint_keys else _fred_fetch, v, DATA_START, key) for k, v in FRED.items()}
        fc = ex.submit(_cot_fetch, DATA_START)
        fg = ex.submit(_gpr_fetch)
        fs = ex.submit(_eia_steo, eia_key, EIA_STEO, DATA_START) if eia_key else None
        fu = ex.submit(_eia_futures, eia_key, DATA_START) if eia_key else None
        px = fy.result()
        fr, rel, failed = {}, {}, []
        for k, f in ff.items():
            try:
                res_ = f.result()
                if k in vint_keys:
                    fr[k], rel[k] = res_
                else:
                    fr[k] = res_
            except Exception:  # noqa: BLE001
                if k in vint_keys:  # fall back to latest-revised data for this series
                    try:
                        fr[k] = _fred_fetch(FRED[k], DATA_START, key)
                    except Exception:  # noqa: BLE001
                        failed.append(k)
                else:
                    failed.append(k)
        for name, fut_, store in (("cot", fc, "cot"), ("gpr", fg, "gpr")):
            try:
                extra[store] = fut_.result()
            except Exception:  # noqa: BLE001
                xfail.append(name)
        if fs is not None:
            try:
                extra.update(fs.result())
            except Exception:  # noqa: BLE001
                xfail.append("eia_steo")
        if fu is not None:
            try:
                extra["curve_fut"] = fu.result()
            except Exception:  # noqa: BLE001
                xfail.append("eia_futures")
    for k, fn in LOCAL_CSV.items():  # optional local files: a missing file is silent, a broken one is reported
        try:
            s_ = _local_csv(fn)
            if s_ is not None:
                extra[k] = s_
        except Exception:  # noqa: BLE001
            xfail.append(k)
    if "real_yield" in failed:  # FRED host is down: fall back to Yahoo stand-ins
        fr, failed = {}, list(FRED)
    fr.update(extra)
    failed = failed + xfail
    m, failed = _prepare_monthly(px, fr, failed, rel)
    F = build_features(m)
    P = F.apply(exp_pctl)
    return m, F, P, failed, time.time()


def build_features(m):
    def g(k):
        return m[k] if k in m else pd.Series(np.nan, index=m.index)

    def basket(*ks):  # equal-weight average of the 3-month trends of the stocks that exist
        return pd.concat([g(k).pct_change(3) for k in ks], axis=1).mean(axis=1, skipna=True)

    F = pd.DataFrame(index=m.index)
    # ---- dollar & monetary
    F["real_yield"] = g("real_yield")
    F["real_yield_chg"] = g("real_yield").diff(3)
    F["breakeven"] = g("breakeven")
    F["be5"] = g("be5")
    F["fed_chg"] = g("fed_funds").diff(6)
    F["curve"] = g("curve")
    F["cpi_yoy"] = g("cpi_yoy")
    F["cpi_accel"] = g("cpi_yoy").diff(3)
    F["real_fed"] = g("fed_funds") - g("cpi_yoy")
    F["nom_yield"] = g("tnx")
    F["nom_yield_chg"] = g("tnx").diff(3)
    if F["fed_chg"].isna().all():
        F["fed_chg"] = g("irx").diff(6)
    if F["curve"].isna().all():
        F["curve"] = g("tnx") - g("irx")
    F["dollar_mom"] = g("dollar").pct_change(3)
    F["m2_yoy"] = g("m2_yoy")
    # ---- demand & growth
    F["indpro_yoy"] = g("indpro_yoy")
    F["philly"] = g("philly")
    F["cfnai"] = g("cfnai")
    F["sahm"] = g("sahm")
    F["copper_mom"] = g("copper").pct_change(3)
    F["em_mom"] = g("eem").pct_change(3)
    F["china_mom"] = g("fxi").pct_change(3)
    F["us_dem_yoy"] = g("us_dem_yoy")
    F["gas_dem_yoy"] = g("gas_dem_yoy")
    F["jet_dem_yoy"] = g("jet_dem_yoy")
    F["airpax_yoy"] = g("airpax_yoy")
    F["airlines_mom"] = basket("luv", "dal", "ual", "jets")
    F["empire"] = g("empire")
    F["cli_us"] = g("cli_us")
    F["cli_cn"] = g("cli_cn")
    F["vmt_yoy"] = g("vmt_yoy")
    F["world_dem_yoy"] = g("world_dem_yoy")
    # ---- supply
    F["us_prod_yoy"] = g("us_prod_yoy")
    F["drill_yoy"] = g("drill_yoy")
    F["rig_yoy"] = g("rig_yoy")
    F["oilsvc_mom"] = g("oih").pct_change(3)
    F["ep_rel"] = (g("xop") / g("oil")).pct_change(3)
    F["opec_chg"] = g("opec_chg")
    F["opec_spare"] = g("opec_spare")
    F["opecplus_chg"] = g("opecplus_chg")
    # ---- inventories & refining
    F["crude_stk_5y"] = g("crude_stk_5y")
    F["gas_stk_5y"] = g("gas_stk_5y")
    F["dist_stk_5y"] = g("dist_stk_5y")
    F["cover_days"] = g("cover_days")
    F["refutil"] = g("refutil")
    F["refutil_dev"] = g("refutil_dev")
    F["cushing_dev"] = g("cushing_dev")
    F["spr_chg"] = g("spr_chg")
    F["world_bal"] = g("world_bal")
    F["crack"] = (2 * g("gasoline") * 42 + g("heating") * 42) / 3 - g("oil_fut")  # $/bbl; RBOB and heating oil are quoted per gallon
    # ---- market structure & positioning
    F["backwd"] = g("backwd")
    F["backwd4"] = g("backwd4")
    F["roll_proxy"] = g("uso").pct_change(3) - g("oil_fut").pct_change(3)  # fallback when the real curve is unavailable (dropped below when it exists)
    F["cot_net"] = g("cot")
    F["xle_rel"] = (g("xle") / g("spx")).pct_change(3)
    cpi = g("cpi_level")
    F["val_real"] = exp_pctl(g("oil") / cpi if cpi.notna().any() else g("oil"))
    F["oil_mom"] = g("oil").pct_change(3).clip(-0.8, 1.5)
    F["oil_mom12"] = g("oil").pct_change(12).clip(-0.9, 3)
    F["natgas_mom"] = g("natgas").pct_change(3).clip(-0.8, 1.5)
    # ---- shipping & freight
    F["freight_yoy"] = g("freight_yoy")
    F["tanker_mom"] = basket("fro", "dht", "stng")
    F["bdi_mom"] = g("bdi").pct_change(3)
    F["brent_wti"] = ((g("brent_spot") - g("oil")) if g("brent_spot").notna().any() else (g("brent") - g("oil_fut"))).clip(-30, 30)
    # ---- risk & geopolitics
    F["vix"] = g("vix")
    F["ovx"] = g("ovx")
    F["ovx_vrp"] = g("ovx") - g("oil_rv")
    F["ovx_chg"] = g("ovx").diff(3)
    F["gpr"] = g("gpr")
    F["gpr_chg"] = g("gpr_chg")
    F["gepu"] = g("gepu")
    F["nfci"] = g("nfci")
    F["credit"] = g("credit")
    F["credit_chg"] = g("credit").diff(3)
    F["spx_mom"] = g("spx").pct_change(3)
    E = build_expectations(m)  # point-in-time expectation indicators (oil_expectations.py)
    for c in E.columns:
        F[c] = E[c]
    F = F.replace([np.inf, -np.inf], np.nan)
    if F["backwd"].notna().sum() > 36:  # the real futures curve exists: the ETF-based proxy would only duplicate it
        F["roll_proxy"] = np.nan
    keep = [c for c in META_ALL if c in F and F[c].notna().any()]
    if "real_yield" in keep:
        keep = [c for c in keep if not c.startswith("nom_")]
    return F[keep]

# ------------------------------------------------------------------ factor compression
def build_factors(F_src, learn_idx, thr, align=True):
    """Merge correlated indicators into factors WITHOUT using any outcome.
    Groups are built by average-linkage on |Spearman| measured on learn_idx only (the early part of the learn period). A group with 2+ members becomes
    one factor = average of z-scores (z-score stats from learn_idx, clipped at +-4), sign-aligned to the group's lead indicator when align=True.
    Singletons keep their original column.
    Returns (factor frame, META-style dict for the new factor columns, {factor column: member list})."""
    cols = list(F_src.columns)
    L = F_src.loc[learn_idx, cols]
    sp = L.corr(method="spearman")
    C = sp.abs().fillna(0).values
    cl = [[i] for i in range(len(cols))]
    while len(cl) > 1:
        best, bi, bj = -1.0, -1, -1
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a = float(C[np.ix_(cl[i], cl[j])].mean())
                if a > best:
                    best, bi, bj = a, i, j
        if best < thr:
            break
        cl[bi] = cl[bi] + cl[bj]
        del cl[bj]
    mu, sd = L.mean(), L.std().replace(0, 1).fillna(1)
    out, meta, members = {}, {}, {}
    for grp in sorted(cl, key=min):
        names = [cols[i] for i in sorted(grp)]
        if len(names) == 1:
            out[names[0]] = F_src[names[0]]
            continue
        idx_ = sorted(grp)
        sub = C[np.ix_(idx_, idx_)]
        lead = names[int(np.argmax(sub.sum(axis=1)))]
        z = []
        for n in names:
            c_ = sp.loc[n, lead]
            sgn = 1.0 if (not align or n == lead or not (c_ < 0)) else -1.0
            z.append(sgn * ((F_src[n] - mu[n]) / sd[n]).clip(-4, 4))
        key = "fac_" + "+".join(names)
        out[key] = pd.concat(z, axis=1).mean(axis=1)
        lt = META_ALL[lead]
        meta[key] = (f"Factor: {lt[0]} + {len(names) - 1} related", f"{lt[1]} (factor low)", f"{lt[2]} (factor high)", False, lt[4])
        members[key] = names
    return pd.DataFrame(out, index=F_src.index), meta, members


def group_stability(F_src, idx, thr):
    """Overlap (Jaccard) of merged indicator pairs when the factor structure is rebuilt on each half of the structure window.
    Near 100% = the same indicators get merged in both halves (stable). Low = the factor structure depends on the sample."""
    half = len(idx) // 2

    def pairs(ix):
        _, _, mem = build_factors(F_src, ix, thr)
        s = set()
        for v in mem.values():
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    s.add((v[i], v[j]))
        return s

    a, b = pairs(idx[:half]), pairs(idx[half:])
    return len(a & b) / len(a | b) if (a | b) else np.nan


# ------------------------------------------------------------------ statistics
def assign_states(F, learn_idx, P=None):
    """Low / Mid / High per indicator. Fixed terciles from the learn period, or (if P is given) point-in-time percentiles."""
    cols = {}
    for c in F.columns:
        if P is not None:
            p = P[c].reindex(F.index)
            lab = np.where(p <= 1 / 3, "Low", np.where(p <= 2 / 3, "Mid", "High")).astype(object)
            cols[c] = pd.Series(lab, index=F.index).where(p.notna())
        else:
            lo, hi = F.loc[learn_idx, c].quantile([1 / 3, 2 / 3])
            col = F[c]
            lab = np.where(col <= lo, "Low", np.where(col <= hi, "Mid", "High")).astype(object)
            cols[c] = pd.Series(lab, index=F.index).where(col.notna())
    return pd.DataFrame(cols)


def nw_t(y, d, lags):
    """Newey-West (HAC) t-stat of b in y = a + b*d + e. Valid for overlapping returns."""
    y, d = np.asarray(y, float), np.asarray(d, float)
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    XtXi = np.linalg.pinv(X.T @ X)
    e = y - X @ (XtXi @ X.T @ y)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, min(lags, n - 1) + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (lags + 1)) * (G + G.T)
    se = np.sqrt((XtXi @ S @ XtXi)[1, 1])
    return float((XtXi @ X.T @ y)[1] / se) if se > 0 else 0.0


def bh_q(p):
    """Benjamini-Hochberg false-discovery-rate q-values."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    ranked = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def _boot_idx(n, block, B, seed):
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    return (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]


def block_ci(x, block, B=1000, seed=0):
    """Approximate 95% CI of the mean via moving-block bootstrap."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan, np.nan
    idx = _boot_idx(len(x), block, B, seed)
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def auc_ci(y, p, block, B=300, seed=0):
    """Approximate 95% CI of AUC via moving-block bootstrap of (outcome, probability) pairs."""
    y, p = np.asarray(y), np.asarray(p, float)
    if len(y) < 20:
        return np.nan, np.nan
    vals = []
    for ix in _boot_idx(len(y), block, B, seed):
        yy = y[ix]
        if yy.min() != yy.max():
            vals.append(roc_auc_score(yy, p[ix]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 20 else (np.nan, np.nan)


def half_edge(fwd, S, col, state, idx):
    r = fwd.loc[idx][S.loc[idx, col] == state]
    return r.mean() - fwd.loc[idx].mean() if len(r) >= 4 else np.nan


STAT_COLS = ["col", "state", "n", "avg", "median", "win", "edge", "t", "consistent", "p", "q"]


def bucket_stats(S, fwd, idx, h):
    base = fwd.loc[idx]
    yv = base.values
    mid = len(idx) // 2
    first, second = idx[:mid], idx[mid:]
    rows = []
    for c in S.columns:
        sv = S.loc[idx, c].astype(object).values
        for s_ in STATES:
            mask = np.asarray(sv == s_, dtype=bool)
            n = int(mask.sum())
            if n < 8:
                continue
            r = base[mask]
            edge = r.mean() - base.mean()
            t = nw_t(yv, mask.astype(float), h)  # HAC: robust to overlapping windows
            e1, e2 = half_edge(fwd, S, c, s_, first), half_edge(fwd, S, c, s_, second)
            cons = bool(np.sign(e1) == np.sign(e2) == np.sign(edge)) if not (np.isnan(e1) or np.isnan(e2)) else False
            rows.append(dict(col=c, state=s_, n=n, avg=r.mean(), median=r.median(), win=(r > 0).mean(),
                             edge=edge, t=t, consistent=cons))
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame(columns=STAT_COLS)
    d["p"] = [math.erfc(abs(t) / math.sqrt(2)) for t in d["t"]]
    d["q"] = bh_q(d["p"].values)
    return d


def build_rules(stats, cfg):
    """cfg = (selection, threshold, need_consistent, weighting). Returns {(col, state): weight} for good and bad."""
    sel, thr, need_cons, wmode = cfg
    if stats.empty:
        return {}, {}
    ok = stats[stats["t"].abs() >= thr] if sel == "t" else stats[stats["q"] <= thr]
    if need_cons:
        ok = ok[ok["consistent"]]

    def w(t):
        return 1.0 if wmode == "equal" else float(min(max(abs(t) - 1.0, 0.0), 3.0))

    return ({(r.col, r.state): w(r.t) for r in ok.itertuples() if r.edge > 0},
            {(r.col, r.state): w(r.t) for r in ok.itertuples() if r.edge < 0})


def score_series(S, good, bad):
    score = pd.Series(0.0, index=S.index)
    for (c, s_), w in good.items():
        score += w * (S[c] == s_).fillna(False).astype(float)
    for (c, s_), w in bad.items():
        score -= w * (S[c] == s_).fillna(False).astype(float)
    return score


def to_verdict(score, lo_thr, hi_thr):
    v = pd.Series(NEU, index=score.index)
    v[(score >= hi_thr) & (score > 0)] = FAV
    v[(score <= lo_thr) & (score < 0)] = UNF
    return v


def _summarize(fwd, groups, idx, h, fdd=None, order=(FAV, NEU, UNF)):
    rows = []
    for g_ in list(order) + ["All months"]:
        r = fwd.loc[idx] if g_ == "All months" else fwd.loc[idx].where(groups.reindex(idx) == g_).dropna()
        row = dict(Group=g_, Months=len(r), Avg=np.nan, Median=np.nan, Win=np.nan, Worst=np.nan, Best=np.nan,
                   lo=np.nan, hi=np.nan, DD=np.nan)
        if len(r):
            row.update(Avg=r.mean(), Median=r.median(), Win=(r > 0).mean(), Worst=r.min(), Best=r.max())
            row["lo"], row["hi"] = block_ci(r.values, h)
            if fdd is not None:
                row["DD"] = fdd.reindex(r.index).median()
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _summarize_c(fwd, groups, idx_vals, h, fdd, order):
    return _summarize(fwd, groups, pd.DatetimeIndex(idx_vals), h, fdd, order)


def summarize(fwd, groups, idx, h, fdd=None, order=(FAV, NEU, UNF)):
    """Cached (index passed as a plain array because Streamlit cannot hash a pandas Index)."""
    return _summarize_c(fwd, groups, pd.DatetimeIndex(idx).values, h, fdd, tuple(order))


def pc(v, plus=True):
    return "-" if pd.isna(v) else (f"{v:+.1%}" if plus else f"{v:.0%}")


def fm(metric, v):
    """Format a performance metric by name."""
    if pd.isna(v):
        return "-"
    return f"{v:.2f}" if metric in ("Sharpe", "Sortino", "Calmar") else pc(v)


def fmt_summary(d, dd=False):
    o = pd.DataFrame({"Environment": d["Group"], "Months": d["Months"], "Avg after": d["Avg"].map(pc),
                      "95% CI": [("-" if pd.isna(a) else f"{a:+.0%} to {b:+.0%}") for a, b in zip(d["lo"], d["hi"])],
                      "Median": d["Median"].map(pc), "% positive": d["Win"].map(lambda v: pc(v, False)),
                      "Worst": d["Worst"].map(pc), "Best": d["Best"].map(pc)})
    if dd:
        o["Typical max drawdown"] = d["DD"].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
    return o


# ------------------------------------------------------------------ targets, splits, rules
def cash_rate(m):
    return ((1 + m["irx"] / 100) ** (1 / 12) - 1).fillna(0) if "irx" in m else pd.Series(0.0, index=m.index)


def fwd_drawdown(s, h):
    v, out = s.values, np.full(len(s), np.nan)
    for i in range(len(v) - h):
        w = v[i: i + h + 1]
        out[i] = (w / np.maximum.accumulate(w) - 1).min()
    return pd.Series(out, index=s.index)


def target_returns(m, idx, h, kind):
    """kind: ret = oil return (WTI SPOT price: futures roll gaps are not returns), vol = oil return scaled to a constant-volatility position (trailing vol only),
    dd = forward max drawdown of oil plus DD_FLOOR (continuous; positive = shallower than the floor),
    ddb = 1 if the forward max drawdown is shallower than DD_FLOOR else 0 (classification),
    rel = oil minus the S&P 500, cash = oil minus T-bills, eqabs = S&P 500 return."""
    if kind == "eqabs":
        return (m["spy"].shift(-h) / m["spy"] - 1).reindex(idx)
    if kind in ("dd", "ddb"):
        d = fwd_drawdown(m["oil"], h)
        out = (d + DD_FLOOR) if kind == "dd" else (d > -DD_FLOOR).astype(float).where(d.notna())
        return out.reindex(idx)
    s = m["oil"].shift(-h) / m["oil"] - 1
    if kind == "vol":
        vol = m["oil"].pct_change().rolling(12).std() * np.sqrt(12)  # known at the signal date
        scale = (VOL_TARGET / vol).clip(0.25, 2.0).fillna(1.0)
        s = s * scale
    elif kind == "rel":
        s = s - (m["spy"].shift(-h) / m["spy"] - 1)
    elif kind == "cash":
        s = s - (((1 + m["irx"].fillna(0) / 100) ** (h / 12) - 1) if "irx" in m else 0)
    return s.reindex(idx)


def split_idx(F_ok, fwd, h, frac):
    ev = F_ok.index.intersection(fwd.dropna().index)
    cut = int(len(ev) * frac)
    return ev, cut, ev[: max(cut - h, 0)], ev[cut:]  # purge h months: no learn/test overlap


def split3(F_ok, fwd, h, train_frac, val_frac):
    """learn (rules are fitted here) -> validation (all choices are made here) -> final test (grading only).
    Months whose outcome windows would overlap the next block are purged."""
    ev = F_ok.index.intersection(fwd.dropna().index)
    n = len(ev)
    c1, c2 = int(n * train_frac), int(n * (train_frac + val_frac))
    learn, val, test = ev[: max(c1 - h, 0)], ev[c1: max(c2 - h, c1)], ev[c2:]
    return dict(ev=ev, c1=c1, cut=int(F_ok.index.get_loc(ev[c1])) if c1 < n else len(F_ok),
                learn=learn, val=val, test=test, unseen=val.union(test))


def _run_rules(F_ok, P, fwd, learn_idx, h, cfg):
    S = assign_states(F_ok, learn_idx, P)
    stats = bucket_stats(S, fwd, learn_idx, h)
    good, bad = build_rules(stats, cfg)
    score = score_series(S, good, bad)
    lo, hi = np.quantile(score.loc[learn_idx], 0.25), np.quantile(score.loc[learn_idx], 0.75)
    return S, stats, good, bad, score, to_verdict(score, lo, hi)


@st.cache_data(show_spinner=False)
def _run_rules_c(F_ok, P, fwd, learn_vals, h, cfg):
    return _run_rules(F_ok, P, fwd, pd.DatetimeIndex(learn_vals), h, cfg)


def run_rules(F_ok, P, fwd, learn_idx, h, cfg):
    """Cached. (Streamlit cannot hash a pandas Index, so it is passed as a plain array.)"""
    return _run_rules_c(F_ok, P, fwd, pd.DatetimeIndex(learn_idx).values, h, cfg)


def spread_for(F_x, P_x, fwd, learn_idx, grade_idx, h, cfg):
    """Favorable minus Unfavorable (pts) on the grading period for rules fitted on learn_idx, plus the number of rules kept."""
    _, _, g_, b_, _, v_ = run_rules(F_x, P_x, fwd, learn_idx, h, cfg)
    ts = summarize(fwd, v_, grade_idx, h).set_index("Group")
    if ts.loc[FAV, "Months"] < 3 or ts.loc[UNF, "Months"] < 3:
        return np.nan, len(g_) + len(b_)
    return (ts.loc[FAV, "Avg"] - ts.loc[UNF, "Avg"]) * 100, len(g_) + len(b_)


def rank_ic(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 8 or d.iloc[:, 0].nunique() < 2:
        return np.nan
    return d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())


def state_thresholds(F_ok, learn_idx, pit):
    ref = F_ok if pit else F_ok.loc[learn_idx]
    return ref.quantile([1 / 3, 2 / 3]).T


# ------------------------------------------------------------------ indicator ranking
def hl_spread(fwd, S, c, idx):
    """Average outcome when the indicator is HIGH minus when it is LOW, in percentage points."""
    r, s = fwd.loc[idx], S.loc[idx, c]
    hi, lo = r[s == "High"], r[s == "Low"]
    return (hi.mean() - lo.mean()) * 100 if len(hi) >= 4 and len(lo) >= 4 else np.nan


def hl_eff(fwd, S, c, idx):
    """Risk-adjusted High minus Low spread: the same gap measured in standard deviations of the outcome."""
    r, s = fwd.loc[idx], S.loc[idx, c]
    hi, lo = r[s == "High"], r[s == "Low"]
    sd = r.std()
    return (hi.mean() - lo.mean()) / sd if len(hi) >= 4 and len(lo) >= 4 and sd > 0 else np.nan


def hl_dd(fdd, S, c, idx):
    """Typical forward max drawdown after High minus after Low, in points (positive = milder drawdowns after High)."""
    d, s = fdd.reindex(idx), S.loc[idx, c]
    hi, lo = d[s == "High"].dropna(), d[s == "Low"].dropna()
    return (hi.median() - lo.median()) * 100 if len(hi) >= 4 and len(lo) >= 4 else np.nan


def _indicator_ranking(F_ok, fwd, S, learn_idx, unseen_idx, h, fdd):
    rows = []
    for c in F_ok.columns:
        xl, yl = F_ok.loc[learn_idx, c], fwd.loc[learn_idx]
        xt, yt = F_ok.loc[unseen_idx, c], fwd.loc[unseen_idx]
        ic_l, ic_t = rank_ic(xl, yl), rank_ic(xt, yt)
        t_l = nw_t(yl.rank().values, xl.rank().values, h) if len(xl) > 8 else np.nan
        hit, hit_ci = np.nan, np.nan
        if pd.notna(ic_l) and ic_l != 0 and len(xt) >= 8:
            sx = np.sign(xt - xl.median()) * np.sign(ic_l)
            sy = np.sign(yt - yl.median())
            ok = (sx != 0) & (sy != 0)
            if ok.sum() >= 8:
                hit = float((sx[ok] == sy[ok]).mean())
                hit_ci = 1.96 * math.sqrt(0.25 / max(ok.sum() / h, 1.0))  # overlapping windows: fewer independent obs
        same = bool(pd.notna(ic_l) and pd.notna(ic_t) and np.sign(ic_l) == np.sign(ic_t))
        minic = min(abs(ic_l), abs(ic_t)) * 100 if same else 0.0
        if pd.isna(ic_l) or pd.isna(ic_t):
            status = "-"
        elif not same:
            status = "❌ Flips on unseen data"
        elif abs(t_l) >= 2:
            status = "✅ Consistent and significant"
        else:
            status = "🟡 Consistent but weak"
        rows.append(dict(col=c, ic_l=ic_l, ic_t=ic_t, t_l=t_l, hit=hit, hit_ci=hit_ci, same=same, minic=minic,
                         status=status, hl_l=hl_spread(fwd, S, c, learn_idx), hl_t=hl_spread(fwd, S, c, unseen_idx),
                         eff_l=hl_eff(fwd, S, c, learn_idx), eff_t=hl_eff(fwd, S, c, unseen_idx),
                         dd_t=hl_dd(fdd, S, c, unseen_idx)))
    d = pd.DataFrame(rows)
    d["absl"] = d["ic_l"].abs()
    return d.sort_values(["minic", "absl"], ascending=False, na_position="last").drop(columns="absl").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _indicator_ranking_c(F_ok, fwd, S, learn_vals, unseen_vals, h, fdd):
    return _indicator_ranking(F_ok, fwd, S, pd.DatetimeIndex(learn_vals), pd.DatetimeIndex(unseen_vals), h, fdd)


def indicator_ranking(F_ok, fwd, S, learn_idx, unseen_idx, h, fdd):
    return _indicator_ranking_c(F_ok, fwd, S, pd.DatetimeIndex(learn_idx).values, pd.DatetimeIndex(unseen_idx).values, h, fdd)


@st.cache_data(show_spinner=False)
def ic_by_horizon(m, F_ok, target, train_frac):
    """Rank correlation (learn period only) of each indicator with the outcome at every look-ahead (signal decay)."""
    out = {}
    for h_ in DECAY_HORIZONS:
        fwd_ = target_returns(m, F_ok.index, h_, target)
        _, _, learn_, _ = split_idx(F_ok, fwd_, h_, train_frac)
        out[h_] = {c: rank_ic(F_ok.loc[learn_, c], fwd_.loc[learn_]) for c in F_ok.columns}
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def rolling_ic(F_ok, fwd, cols, win=60):
    """Rolling rank correlation of indicator vs outcome (descriptive; overlapping outcomes mean few independent points)."""
    ev = fwd.dropna().index.intersection(F_ok.index)
    y = fwd.loc[ev].values
    out = {}
    for c in cols:
        x = F_ok.loc[ev, c].values
        r = np.full(len(ev), np.nan)
        for i in range(win, len(ev) + 1):
            r[i - 1] = pd.Series(x[i - win:i]).rank().corr(pd.Series(y[i - win:i]).rank())
        out[c] = pd.Series(r, index=ev)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ machine learning
def make_model(name):
    if name.startswith("Logistic"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    if name.startswith("Random"):
        return RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=10, n_jobs=-1, random_state=0)
    return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)


@st.cache_data(show_spinner=False)
def walk_forward(F, fwd, h, cut, step, model_name):
    """Predicts EVERY month from `cut` on (including the latest months whose outcome is not known yet).
    Training only ever uses rows whose outcome was already known at that time. No Streamlit calls in here."""
    X, yraw = F.values, fwd.values
    known_ = ~np.isnan(yraw)
    y = (yraw > 0).astype(int)
    out = []
    for s0 in range(cut, len(F), step):
        te = slice(s0, min(s0 + step, len(F)))
        tr = np.where(known_[: s0 - h])[0]
        if len(tr) < 36:
            continue
        ytr = y[tr]
        if len(np.unique(ytr)) < 2:
            p = np.full(te.stop - te.start, ytr.mean())
        else:
            p = make_model(model_name).fit(X[tr], ytr).predict_proba(X[te])[:, 1]
        out.append(pd.DataFrame({"p": p, "base": ytr.mean()}, index=F.index[te]))
    return pd.concat(out) if out else pd.DataFrame(columns=["p", "base"])


def known(wf, fwd, idx=None):
    """Walk-forward rows whose outcome is already known (for scoring; the backtest can use all rows).
    idx restricts scoring to a period (research mode scores on validation only)."""
    k = wf.index.intersection(fwd.dropna().index)
    if idx is not None:
        k = k.intersection(idx)
    return wf.loc[k]


@st.cache_data(show_spinner=False)
def fit_today(F, fwd, x_now, model_name):
    k = fwd.notna().values
    y = (fwd.values[k] > 0).astype(int)
    Fk = F.loc[k]
    if len(np.unique(y)) < 2:  # the outcome never varied in the sample (e.g. a drawdown target that is always 0 or always 1): no model can be fitted
        return float(y.mean()), float(y.mean()), pd.Series(0.0, index=F.columns), None
    mdl = make_model(model_name).fit(Fk.values, y)
    p = float(mdl.predict_proba(x_now.values.reshape(1, -1))[0, 1])
    contrib = None
    if hasattr(mdl, "steps"):  # logistic: exact additive contributions in log-odds
        sc, lr = mdl[0], mdl[-1]
        imp = pd.Series(lr.coef_[0], index=F.columns)
        contrib = pd.Series(lr.coef_[0] * (x_now.values - sc.mean_) / sc.scale_, index=F.columns)
    else:
        imp = pd.Series(mdl.feature_importances_, index=F.columns)
    return p, float(y.mean()), imp, contrib


@st.cache_data(show_spinner=False)
def ablation(F_ok, fwd, h, cut, step, eval_vals):
    """Drop-one test with a logistic walk-forward model: how much AUC / rank IC is lost when an indicator is removed?"""
    def run(cols):
        wk = known(walk_forward(F_ok[cols], fwd, h, cut, step, ML_MODELS[0]), fwd, pd.DatetimeIndex(eval_vals))
        if wk.empty:
            return np.nan, np.nan
        y = (fwd.loc[wk.index] > 0).astype(int)
        return (roc_auc_score(y, wk["p"]) if y.nunique() > 1 else np.nan), rank_ic(wk["p"], fwd.loc[wk.index])

    cols = list(F_ok.columns)
    b_auc, b_ic = run(cols)
    rows = []
    for c in cols:
        a, i = run([x for x in cols if x != c])
        rows.append(dict(col=c, dauc=b_auc - a, dic=b_ic - i))
    return pd.DataFrame(rows)


def ml_verdict(wf, margin):
    d = wf["p"] - wf["base"]
    v = pd.Series(NEU, index=wf.index)
    v[d >= margin] = FAV
    v[d <= -margin] = UNF
    return v


def prob_metrics(wf, y):
    p, b = wf["p"].clip(0.01, 0.99), wf["base"].clip(0.01, 0.99)
    brier, brier_b = ((p - y) ** 2).mean(), ((b - y) ** 2).mean()
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    return brier, 1 - brier / brier_b, ll


@st.cache_data(show_spinner=False)
def scan_horizons(m, F_ok, P, target, model_name, train_frac, cfg, horizons):
    """Choose the look-ahead using ONLY the learn period (fit on first 70%, score last 30%)."""
    def ics(pred, ret, V):
        half = len(V) // 2
        full, a, b = rank_ic(pred.loc[V], ret.loc[V]), rank_ic(pred.loc[V[:half]], ret.loc[V[:half]]), rank_ic(pred.loc[V[half:]], ret.loc[V[half:]])
        return full, (min(a, b) if not (np.isnan(a) or np.isnan(b)) else np.nan)

    rows = []
    for h_ in horizons:
        fwd_ = target_returns(m, F_ok.index, h_, target)
        _, _, learn_, _ = split_idx(F_ok, fwd_, h_, train_frac)
        k = int(len(learn_) * 0.7)
        A, V = learn_[: max(k - h_, 0)], learn_[k:]
        row = dict(h=h_, rules_ic=np.nan, rules_stab=np.nan, ml_ic=np.nan, ml_stab=np.nan, n_val=len(V))
        if len(A) >= 36 and len(V) >= 10:
            S = assign_states(F_ok, A, P)
            g_, b_ = build_rules(bucket_stats(S, fwd_, A, h_), cfg)
            row["rules_ic"], row["rules_stab"] = ics(score_series(S, g_, b_), fwd_, V)
            yA = (fwd_.loc[A] > 0).astype(int)
            if model_name and yA.nunique() > 1:
                mdl = make_model(model_name).fit(F_ok.loc[A].values, yA.values)
                p = pd.Series(mdl.predict_proba(F_ok.loc[V].values)[:, 1], index=V)
                row["ml_ic"], row["ml_stab"] = ics(p, fwd_, V)
        rows.append(row)
    d = pd.DataFrame(rows)
    d["combined"] = d[["rules_ic", "ml_ic"]].mean(axis=1, skipna=True)
    d["stable"] = d[["rules_stab", "ml_stab"]].mean(axis=1, skipna=True)
    return d


# ------------------------------------------------------------------ robustness
@st.cache_data(show_spinner=False)
def walk_forward_rules(F_ok, P, fwd, h, cut, step, cfg):
    """Nested walk-forward: every `step` months the rules are re-estimated on months whose outcome was already known at that time,
    then applied to the NEXT `step` months. Returns a fully out-of-sample verdict for every month from `cut` on."""
    idx = F_ok.index
    known_ = ~np.isnan(fwd.reindex(idx).values)
    out = []
    for s0 in range(cut, len(idx), step):
        te = idx[s0: min(s0 + step, len(idx))]
        cand = np.arange(max(s0 - h, 0))
        learn = idx[cand[known_[cand]]]
        if len(learn) < 36:
            continue
        S_ = assign_states(F_ok, learn, P)
        g_, b_ = build_rules(bucket_stats(S_, fwd, learn, h), cfg)
        sc = score_series(S_, g_, b_)
        lo, hi = np.quantile(sc.loc[learn], 0.25), np.quantile(sc.loc[learn], 0.75)
        out.append(pd.DataFrame({"v": to_verdict(sc, lo, hi).loc[te], "score": sc.loc[te]}))
    return pd.concat(out) if out else pd.DataFrame(columns=["v", "score"])


@st.cache_data(show_spinner=False)
def analogue_walk_forward(F_ok, fwd, h, cut, k=ANA_K):
    """Predictive analogue test. For every month from `cut` on, find the k closest past months among months whose outcome was ALREADY KNOWN
    at that time (position i with i + h <= t), standardised with that database only, and report the median / share-positive outcome.
    This is the fair, walk-forward version of the descriptive analogue table."""
    X = F_ok.values.astype(float)
    y = fwd.reindex(F_ok.index).values.astype(float)
    n = len(X)
    rows, idx = [], []
    gap = max(h, 3)
    for t in range(cut, n):
        db = np.arange(0, t - h + 1)
        db = db[~np.isnan(y[db])]
        if len(db) < 36:
            continue
        mu, sd = X[db].mean(axis=0), X[db].std(axis=0, ddof=1)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        d = np.sqrt((((X[db] - mu) / sd - (X[t] - mu) / sd) ** 2).sum(axis=1))
        chosen = []
        for j in np.argsort(d):
            p_ = db[j]
            if all(abs(p_ - c) >= gap for c in chosen):
                chosen.append(p_)
            if len(chosen) == k:
                break
        a = y[chosen]
        rows.append((float(np.median(a)), float((a > 0).mean())))
        idx.append(F_ok.index[t])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["med", "pos"])


def analogue_verdict(wa):
    v = pd.Series(NEU, index=wa.index, dtype=object)
    if wa.empty:
        return v
    v[(wa["med"] > 0) & (wa["pos"] >= 0.6)] = FAV
    v[(wa["med"] <= 0) & (wa["pos"] <= 0.4)] = UNF
    return v


@st.cache_data(show_spinner=False)
def sensitivity(F_ok, P, fwd, h, fracs, thrs, need_cons, wmode, test_vals):
    """Rules learned with different learn shares and strength thresholds, ALL graded on the same months.
    Cell = average outcome after Favorable minus after Unfavorable, in percentage points."""
    test_idx = pd.DatetimeIndex(test_vals)
    ev = F_ok.index.intersection(fwd.dropna().index)
    grid = np.full((len(fracs), len(thrs)), np.nan)
    for i, f in enumerate(fracs):
        learn = ev[: max(int(len(ev) * f) - h, 0)]
        if len(learn) < 36:
            continue
        S = assign_states(F_ok, learn, P)
        stats = bucket_stats(S, fwd, learn, h)
        for j, t in enumerate(thrs):
            good, bad = build_rules(stats, ("t", t, need_cons, wmode))
            sc = score_series(S, good, bad)
            v = to_verdict(sc, np.quantile(sc.loc[learn], 0.25), np.quantile(sc.loc[learn], 0.75)).loc[test_idx]
            r = fwd.loc[test_idx]
            fa, un = r[v == FAV], r[v == UNF]
            if len(fa) >= 3 and len(un) >= 3:
                grid[i, j] = (fa.mean() - un.mean()) * 100
    return pd.DataFrame(grid, index=[f"{int(f * 100)}%" for f in fracs], columns=[f"{t:.1f}" for t in thrs])


# ------------------------------------------------------------------ regimes, analogues, persistence
def classify_regime(Fa, idx, learn_idx):
    def zmean(spec):
        spec = [(c, s) for c, s in spec if c in Fa and Fa.loc[learn_idx, c].notna().sum() > 20]
        if not spec:
            return None
        return pd.concat([s * (Fa.loc[idx, c] - Fa.loc[learn_idx, c].mean()) / Fa.loc[learn_idx, c].std() for c, s in spec], axis=1).mean(axis=1)

    g, i = zmean(GROWTH), zmean(INFL)
    if g is None or i is None:
        return None
    names = np.where(g > 0, np.where(i > 0, REGIMES[1], REGIMES[0]), np.where(i > 0, REGIMES[2], REGIMES[3]))
    return pd.Series(names, index=idx).where(g.notna() & i.notna())


def regime_transitions(reg, fwd, ev, lookback=3):
    """Outcome by 'regime 3 months ago -> regime now' (only months where the regime changed)."""
    short = reg.map(lambda s: s.split(" (")[0] if isinstance(s, str) else np.nan)
    prev = short.shift(lookback)
    trans = (prev + " → " + short).where(prev.notna() & short.notna() & (prev != short))
    cur = trans.iloc[-1] if pd.notna(trans.iloc[-1]) else None
    rows = []
    for t_, grp in fwd.reindex(ev).groupby(trans.reindex(ev)):
        grp = grp.dropna()
        if len(grp) >= 6:
            rows.append(dict(Transition=t_, Months=len(grp), Avg=grp.mean(), Median=grp.median(), Win=(grp > 0).mean()))
    return pd.DataFrame(rows), cur, short.iloc[-1]


def analogues(F_ok, fwd, ev, x_now, h, k=10):
    """DESCRIPTIVE analogues: the whole available history is searched. For a fair predictive test see analogue_walk_forward."""
    Fe = F_ok.loc[ev]
    mu, sd = Fe.mean(), Fe.std().replace(0, 1)
    d = np.sqrt((((Fe - mu) / sd - (x_now - mu) / sd) ** 2).sum(axis=1))
    chosen = []
    for t in d.sort_values().index:  # keep analogues at least h months apart (avoid near-duplicates)
        if all(abs((t.year - c.year) * 12 + t.month - c.month) >= max(h, 3) for c in chosen):
            chosen.append(t)
        if len(chosen) == k:
            break
    return pd.DataFrame({"dt": chosen, "Date": [c.strftime("%b %Y") for c in chosen],
                         "Closeness": [100 * np.exp(-d[c] / d.median()) for c in chosen],
                         "After": [fwd.loc[c] for c in chosen]})


def analogue_paths(ser, dts, h):
    out, pos = {}, {d: i for i, d in enumerate(ser.index)}
    for d in dts:
        i = pos.get(d)
        if i is None or i + h >= len(ser):
            continue
        seg_ = ser.iloc[i: i + h + 1].values
        out[d.strftime("%b %Y")] = (seg_ / seg_[0] - 1) * 100
    return pd.DataFrame(out, index=range(h + 1))


def streak_info(v, fwd, ev):
    grp = (v != v.shift()).cumsum()
    run = v.groupby(grp).cumcount() + 1
    cur = v.iloc[-1]
    lens = pd.DataFrame({"v": v, "g": grp}).groupby("g").agg(v=("v", "first"), n=("v", "size"))
    med = lens[lens["v"] == cur]["n"].median()
    mask = (v == cur) & (run >= 3) & v.index.isin(ev)
    r = fwd.reindex(v.index)[mask].dropna()
    return cur, int(run.iloc[-1]), med, len(r), (r.mean() if len(r) else np.nan)


# ------------------------------------------------------------------ current verdict for oil / stocks
def exposure_label(e):
    """Average of the rules and ML views (each 0 / 0.5 / 1) -> five-step label."""
    if e >= 0.99:
        return "Favorable", "🟢"
    if e >= 0.7:
        return "Leaning favorable", "🟡"
    if e > 0.3:
        return "Neutral", "⚪"
    if e > 0.01:
        return "Leaning unfavorable", "🟠"
    return "Unfavorable", "🔴"


@st.cache_data(show_spinner=False)
def analyse_target(kind, m, F_ok, P, h, train_frac, val_frac, cfg, run_ml, ml_choice, step, margin, reveal):
    """Full rules + ML + analogues + momentum read of the CURRENT month for one target. No Streamlit calls inside."""
    fwd_ = target_returns(m, F_ok.index, h, kind)
    sp_ = split3(F_ok, fwd_, h, train_frac, val_frac)
    S_, stats_, good_, bad_, score_, verdict_ = run_rules(F_ok, P, fwd_, sp_["learn"], h, cfg)
    now_ = F_ok.index[-1]
    v_r, S_now = verdict_.loc[now_], S_.loc[now_]
    res = dict(rules=v_r, score=float(score_.loc[now_]),
               why_good=[state_label(c, s_) for c, s_ in sorted(good_) if S_now[c] == s_],
               why_bad=[state_label(c, s_) for c, s_ in sorted(bad_) if S_now[c] == s_])

    ev_u = sp_["unseen"] if reveal else sp_["val"]  # research mode never looks at the final test
    ev_r_ = sp_["ev"] if reveal else sp_["ev"].intersection(sp_["learn"].union(sp_["val"]))
    sp, ev_level = np.nan, "Unknown"
    ev_text = "Not enough Favorable and Unfavorable months in the unseen period to judge."
    if not (good_ or bad_):
        ev_level, ev_text = "None", "No environment passed the strength filter, so the rules have no opinion."
    else:
        ts = summarize(fwd_, verdict_, ev_u, h).set_index("Group")
        if ts.loc[FAV, "Months"] >= 3 and ts.loc[UNF, "Months"] >= 3:
            sp = (ts.loc[FAV, "Avg"] - ts.loc[UNF, "Avg"]) * 100
            lo_f, hi_u = ts.loc[FAV, "lo"], ts.loc[UNF, "hi"]
            overlap = bool(pd.isna(lo_f) or pd.isna(hi_u) or lo_f <= hi_u)
            if sp > 2 and not overlap:
                ev_level, ev_text = "Strong", f"Unseen data: Favorable beat Unfavorable by {sp:.1f} pts, confidence ranges do not overlap."
            elif sp > 0:
                ev_level, ev_text = "Weak", f"Unseen data: Favorable beat Unfavorable by {sp:.1f} pts, but the ranges overlap (could be noise)."
            else:
                ev_level, ev_text = "None", f"Unseen data: Favorable did NOT beat Unfavorable ({sp:+.1f} pts)."
    res.update(ev_level=ev_level, ev_text=ev_text, spread=sp)

    ml_info = None
    if run_ml:
        wf_ = walk_forward(F_ok, fwd_, h, sp_["cut"], step, ml_choice)
        p_, base_, _, _ = fit_today(F_ok, fwd_, F_ok.iloc[-1], ml_choice)
        dm = p_ - base_
        auc = bss = np.nan
        wk = known(wf_, fwd_, ev_u)
        if not wk.empty:
            yy = (fwd_.loc[wk.index] > 0).astype(int)
            if yy.nunique() > 1:
                auc = roc_auc_score(yy, wk["p"])
                _, bss, _ = prob_metrics(wk, yy)
        ml_info = dict(p=p_, base=base_, verdict=FAV if dm >= margin else (UNF if dm <= -margin else NEU), auc=auc, bss=bss)
    res["ml"] = ml_info

    e = float(np.mean([EXPO[v_r]] + ([EXPO[ml_info["verdict"]]] if ml_info else [])))
    res["e"] = e
    res["label"], res["icon"] = exposure_label(e)
    a = analogues(F_ok, fwd_, ev_r_, F_ok.iloc[-1], h)["After"]
    res["ana_med"], res["ana_pos"] = float(a.median()), float((a > 0).mean())

    # confidence: do the evidence channels agree, and did the rules work on unseen data?
    ser = {"ret": m["oil"], "eqabs": m["spy"], "rel": m["oil"] / m["spy"]}.get(kind, m["oil"])
    up = bool(ser.iloc[-1] > ser.rolling(10).mean().iloc[-1])
    comps = {"Rules": SGN[v_r]}
    if ml_info:
        comps["ML"] = SGN[ml_info["verdict"]]
    comps["Analogues"] = 1 if (res["ana_med"] > 0 and res["ana_pos"] >= 0.55) else (-1 if (res["ana_med"] < 0 and res["ana_pos"] <= 0.45) else 0)
    # Rules, ML and analogues all read the SAME indicators, so together they are ONE evidence channel, not three independent votes.
    cs = sum(comps.values())
    votes = {"Macro models": 1 if cs >= 2 else (-1 if cs <= -2 else 0), "Price trend": 1 if up else -1}
    npos, nneg = sum(v > 0 for v in votes.values()), sum(v < 0 for v in votes.values())
    agree, mixed = max(npos, nneg), npos == nneg
    if mixed or agree < 2:
        conf = "Low"
    elif ev_level == "Strong":
        conf = "High"
    elif ev_level == "Weak":
        conf = "Medium"
    else:
        conf = "Low"
    res.update(votes=votes, comps=comps, agree=agree, n_votes=len(votes), conf=conf)
    return res


# ------------------------------------------------------------------ backtest
def px_names(impl):
    """(oil, alternative) price columns for the chosen implementation. ETF: USO (fees and roll costs are inside the price) and SPY.
    Futures: the roll-adjusted WTI excess-return index when the EIA curve is available (FUT_COL = 'oil_tr'), otherwise Yahoo's continuous series."""
    return ("uso", "spy") if impl.startswith("ETF") else (FUT_COL, "spy")


def nxt(s):
    """Return earned from this month-end to the next one."""
    return s.shift(-1) / s - 1


def strategy_weights(name, v, aux):
    """Weights (oil, S&P 500, cash) held during the month AFTER each signal month."""
    z = pd.Series(0.0, index=v.index)
    s, g = z.copy(), z.copy()
    trend = aux["trend"].astype(bool)
    fav, unf = v == FAV, v == UNF
    if name.startswith("Scaled"):
        s = v.map(EXPO).astype(float)
    elif name.startswith("Defensive"):
        s = v.map({FAV: 1.0, NEU: 0.25, UNF: 0.0}).astype(float)
    elif name.startswith("Aggressive"):
        s = v.map({FAV: 1.0, NEU: 0.75, UNF: 0.25}).astype(float)
    elif name.startswith("Only when"):
        s = fav.astype(float)
    elif name.startswith("Hold unless"):
        s = (~unf).astype(float)
    elif name.startswith("Rotate"):
        s, g = fav.astype(float), (~fav).astype(float)
    elif name.startswith("Oil / S&P"):
        s, g = fav.astype(float), (v == NEU).astype(float)
    elif name.startswith("Trend only"):
        s = trend.astype(float)
    elif name.startswith("Macro + trend"):
        s = ((~unf) & trend).astype(float)
    elif name.startswith("Vol-capped"):
        cap = (VOL_TARGET / aux["vol"].astype(float)).clip(0, 1).fillna(1.0)  # de-risk when realised vol is high, never lever up
        s = v.map(EXPO).astype(float) * cap
    return pd.DataFrame({"oil": s, "alt": g, "cash": 1.0 - s - g})


def _contrib(w, r):
    return pd.Series(np.where(w.values == 0, 0.0, w.values * r.values), index=w.index)


def run_backtest_w(m, W, bps, impl="Futures"):
    """Signal at month-end t sets the position held during month t -> t+1. Costs charged on turnover."""
    ok_, ak_ = px_names(impl)
    ro, ra = nxt(m[ok_]).reindex(W.index), nxt(m[ak_]).reindex(W.index)
    rf = cash_rate(m).reindex(W.index)
    turn = W["oil"].diff().abs() + W["alt"].diff().abs()
    turn.iloc[0] = W["oil"].iloc[0] + W["alt"].iloc[0]
    coll = 0.0 if impl.startswith("ETF") else W["oil"] * rf  # futures are margin-based: the invested part earns T-bills too (the S&P leg is a fund, no collateral)
    ret = _contrib(W["oil"], ro) + _contrib(W["alt"], ra) + _contrib(W["cash"], rf) + coll - turn * bps / 1e4
    return ret.dropna(), turn


@st.cache_data(show_spinner=False)
def bt_one(m, v, aux, strat, bps, impl):
    W = strategy_weights(strat, v, aux)
    ret, turn = run_backtest_w(m, W, bps, impl)
    return ret, turn, W


def seg(ret, turn, W, idx):
    """Slice a full backtest to a period: (returns, number of trades, average % invested)."""
    ix = ret.index.intersection(idx)
    return ret.loc[ix], int((turn.reindex(ix) > 0).sum()), (W["oil"] + W["alt"]).reindex(ix).mean()


def perf(r, rf, invested=np.nan, trades=np.nan):
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "Max drawdown"]
    n = len(r)
    if n < 2:
        out = dict.fromkeys(keys, np.nan)
        out.update({"% invested": invested, "Trades": trades})
        return out
    eq = (1 + r).cumprod()
    ex = r - rf.reindex(r.index).fillna(0)
    vol = r.std() * np.sqrt(12)
    dn = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(12)
    cagr = eq.iloc[-1] ** (12 / n) - 1
    mdd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Volatility": vol,
            "Sharpe": ex.mean() * 12 / vol if vol > 0 else np.nan,
            "Sortino": ex.mean() * 12 / dn if dn > 0 else np.nan,
            "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "Max drawdown": mdd, "% invested": invested, "Trades": trades}


def bench_curves(m, idx, impl):
    ok_, ak_ = px_names(impl)
    ro, ra = nxt(m[ok_]).reindex(idx), nxt(m[ak_]).reindex(idx)
    if not impl.startswith("ETF"):
        ro = ro + cash_rate(m).reindex(idx)  # futures excess return plus T-bill collateral
    return {"Oil buy & hold": ro.dropna(), "S&P 500 buy & hold": ra.dropna(), "20/80 oil / S&P 500": (0.2 * ro + 0.8 * ra).dropna(),
            "Cash (T-bills)": cash_rate(m).reindex(idx)}


def static_mix(m, wo, wa, idx, impl):
    """Same AVERAGE exposure as a strategy but no timing at all: the fair 'did the signal add anything?' baseline."""
    W = pd.DataFrame({"oil": wo, "alt": wa, "cash": 1 - wo - wa}, index=idx)
    return run_backtest_w(m, W, 0, impl)[0]


@st.cache_data(show_spinner=False)
def shift_null(m, v, aux, strat, bps, impl, eval_vals, metric, n=300, seed=0):
    """Randomization test: circularly shift the signal by a random number of months (keeps its persistence and mix of
    Favorable / Neutral / Unfavorable, destroys its timing) and re-score the strategy on the evaluation months."""
    eval_idx, rf, L = pd.DatetimeIndex(eval_vals), cash_rate(m), len(v)
    rng = np.random.default_rng(seed)
    lo = max(6, L // 10)
    vals = []
    for k in rng.integers(lo, max(L - lo, lo + 1), size=n):
        vv = pd.Series(np.roll(v.values, int(k)), index=v.index)
        aa = pd.DataFrame({c: np.roll(aux[c].values, int(k)) for c in aux.columns}, index=aux.index)
        ret, _ = run_backtest_w(m, strategy_weights(strat, vv, aa), bps, impl)
        vals.append(perf(ret.loc[ret.index.intersection(eval_idx)], rf)[metric])
    return np.array(vals, float)


def perm_p(null, obs):
    null = null[~np.isnan(null)]
    if len(null) < 20 or pd.isna(obs):
        return np.nan, np.nan, np.nan
    return (np.sum(null >= obs) + 1) / (len(null) + 1), float(np.median(null)), float(np.percentile(null, 95))


def perf_table(rows):
    return pd.DataFrame([{"Strategy": n, "CAGR": pc(p["CAGR"]), "Volatility": pc(p["Volatility"], False),
                          "Sharpe": fm("Sharpe", p["Sharpe"]), "Sortino": fm("Sortino", p["Sortino"]),
                          "Calmar": fm("Calmar", p["Calmar"]), "Max drawdown": pc(p["Max drawdown"]),
                          "% invested": pc(p["% invested"], False), "Trades": p["Trades"]} for n, p in rows])


def growth_chart(curves, common, title, mark=None):
    f = go.Figure()
    for name, r in curves.items():
        r = r.reindex(common).dropna()
        f.add_scatter(x=r.index, y=(1 + r).cumprod(), name=name)
    f.update_layout(title=title, height=400)
    if mark is not None:
        vmark(f, mark, "final test starts")
    return f


# ================================================================== UI
with st.sidebar:
    st.header("Settings")
    st.subheader("Evaluation mode")
    eval_mode = st.radio("Mode", ["🔬 Research (final test hidden)", "🔒 Locked evaluation (reveal final test)"], label_visibility="collapsed",
                         help="Research: experiment freely, everything is graded on VALIDATION only and the final test is hidden everywhere. "
                              "Locked: freeze your choices, then reveal the final test. Each distinct set of settings it is shown for is counted.")
    REVEAL = False
    if eval_mode.startswith("🔒"):
        REVEAL = st.checkbox("I have finished research: reveal the final test for these exact settings", False)
    LK = REVEAL  # while the final test is revealed, every setting below is frozen
    if LK:
        st.caption("🔒 Settings are frozen while the final test is revealed. Untick the box above to change anything (the final test is hidden again; "
                   "each new set of settings you reveal is counted).")
    _key_ok = bool(get_fred_key())
    use_vintage = st.checkbox("📅 First-release (ALFRED) data for revised series", _key_ok, disabled=LK,
                              help="Uses first-release values and real release dates for M2, CPI, industrial production, unemployment, Philly Fed, NFCI, CFNAI, drilling activity and the freight index. "
                                   "Needs a FRED_API_KEY (on by default when one is found).")
    if use_vintage and not _key_ok:
        st.warning("No FRED_API_KEY found, so first-release data is off.")
        use_vintage = False
    if st.button("🔄 Refresh data", help="Data is downloaded once and reused for 6 hours. Click to download again now."):
        get_dataset.clear()
    start = st.text_input("Data start date", "2004-01-01", disabled=LK)
    target_name = st.selectbox("What to predict", list(TARGETS), disabled=LK,
                               help="Raw return is dominated by a few wild years. 'Volatility-scaled forward return' scales each outcome by trailing volatility. "
                                    "The drawdown targets predict risk instead of direction: the yes / no version is a classification, the continuous version keeps the size of the drawdown. "
                                    "Relative targets ask whether oil BEATS the S&P 500 / cash.")
    target = TARGETS[target_name]
    auto_h = st.checkbox("🔍 Auto-find best look-ahead", False, disabled=LK,
                         help="Tests 1-12 month look-aheads on the learning period only and prefers stable ones.")
    h_manual = st.select_slider("Look-ahead period (months)", options=list(HORIZONS), value=6, disabled=auto_h or LK)
    st.subheader("Data split")
    train_frac = st.slider("LEARN share (rules are fitted here)", 0.3, 0.6, 0.5, 0.05, disabled=LK)
    val_frac = st.slider("VALIDATION share (strategy choices are made here)", 0.1, 0.3, 0.2, 0.05, disabled=LK)
    st.caption(f"Final test = the remaining {max(0.0, 1 - train_frac - val_frac):.0%}. It only grades the locked choice.")
    st.subheader("Rules")
    scheme = st.selectbox("How to define Low / Mid / High", ["Fixed terciles from the learn period", "Point-in-time percentile (expanding)"], disabled=LK,
                          help="Point-in-time ranks each month against past months only, so it adapts to drifting levels but needs ~3 more years of history.")
    pit = scheme.startswith("Point")
    sel_mode = st.selectbox("Select environments by", ["HAC t-stat", "FDR q-value (multiple-testing adjusted)"], disabled=LK)
    t_thr = st.slider("Minimum strength (HAC t-stat)", 0.5, 3.0, 1.5, 0.1, disabled=sel_mode.startswith("FDR") or LK)
    q_thr = st.slider("Maximum FDR q-value", 0.05, 0.50, 0.20, 0.05, disabled=(not sel_mode.startswith("FDR")) or LK)
    need_consistent = st.checkbox("Require the effect in both halves of the learn period", True, disabled=LK)
    wlabel = st.selectbox("Indicator weighting", ["Equal (+1 / -1)", "By strength (shrunk: |t| − 1, capped)"], disabled=LK)
    fac_mode = st.checkbox("🧬 Compress correlated indicators into factors", False, disabled=LK,
                           help="Merges indicators that move together (measured on the early learn period only, no outcomes used) into one factor each, "
                                "so five versions of the same story no longer count as five votes.")
    fac_thr = st.slider("Merge when average |correlation| ≥", 0.4, 0.9, 0.6, 0.05, disabled=(not fac_mode) or LK)
    st.subheader("Machine learning")
    run_ml = st.checkbox("Also run machine-learning models", True, disabled=LK)
    ml_choice = st.selectbox("Model shown in detail", ML_MODELS, disabled=(not run_ml) or LK)
    margin = st.slider("Confidence margin (points vs normal odds)", 0.02, 0.20, 0.05, 0.01, disabled=(not run_ml) or LK)
    step = st.select_slider("Retrain every (months)", options=[3, 6, 12], value=6, disabled=(not run_ml) or LK)
    st.subheader("Backtest")
    mode = st.selectbox("Strategy", STRATEGIES, disabled=LK)
    impl = st.selectbox("Implementation", ["ETFs (USO / SPY; fund fees and roll costs are already in the prices)", "Futures (WTI contract 1 rolled monthly, T-bill collateral added; stocks via SPY)"], disabled=LK,
                        help="ETF is the simplest default. USO prices are NAV-based, so its expense ratio AND the roll cost of holding front-month futures are already inside the price. "
                             "Futures uses a roll-adjusted WTI excess-return series built from the EIA contract prices (needs EIA_API_KEY); without the key it falls back to Yahoo's continuous series, whose roll gaps are NOT real returns.")
    bps = st.slider("Trading cost (bps per 100% traded, incl. slippage)", 0, 100, 15, 5, disabled=LK)
    run_perm = st.checkbox("🎲 Randomization test", True, disabled=LK, help="Shifts the signal in time 300 times to see how often luck beats the strategy.")
    find_best = st.checkbox("🏁 Find the best strategy", False, disabled=LK,
                            help="Ranks every strategy on every signal using the VALIDATION period, then grades the winner on the untouched final test.")
    rank_by = st.selectbox("Rank strategies by", RANK_METRICS, disabled=(not find_best) or LK)
    st.subheader("Projection")
    proj_model = st.selectbox("Projection model", PROJ_MODELS, disabled=LK,
                              help="Ridge is fast. Gradient boosting (and the average of both) takes a while on the first run, then it is cached.")
    proj_feats = st.selectbox("Projection features", PROJ_FEATS, disabled=LK,
                              help="Oil's own momentum, volatility and trend can be added to the macro indicators, or used alone. The two 'expectations' options add the "
                                   "forward-looking indicators even if the sidebar switch for the other models is off.")
    proj_len = st.select_slider("Projection length (months)", options=PROJ_LENGTHS, value=12, disabled=LK)
    proj_trust = st.slider("Trust in the model (0% = history only)", 0, 100, 50, 10, disabled=LK,
                           help="Shrinks the model's fan toward the unconditional historical distribution of oil's returns.")

st.title("🛢️ Oil Macro Environment Analyzer")
st.caption("Historically favorable or unfavorable macro environments for crude oil (WTI): demand, supply, inventories, market structure, shipping, dollar, rates and geopolitics. Not a buy/sell signal, not financial advice.")
bar = st.progress(0.0, text="Starting...")


def cb(frac, text):
    bar.progress(float(min(max(frac, 0.0), 1.0)), text=text)


if train_frac + val_frac > 0.85:
    bar.empty()
    st.error("Learn + validation share must leave at least 15% for the final test.")
    st.stop()

try:
    cb(0.05, "Loading data (downloaded once, then reused)...")
    m, F_full, P_full, failed, data_ts = get_dataset(get_fred_key(), get_eia_key(), use_vintage)
    if failed and time.time() - data_ts > 300:  # partial download: retry, but not on every click
        get_dataset.clear()
        m, F_full, P_full, failed, data_ts = get_dataset(get_fred_key(), get_eia_key(), use_vintage)
except Exception as e:
    bar.empty()
    st.error(f"Could not download market data: {e}")
    st.stop()

try:
    F_all = F_full.loc[start:]  # the start date only slices the cached data
except Exception:  # noqa: BLE001
    F_all = F_full.iloc[0:0]
if F_all.empty:
    bar.empty()
    st.error("The start date is invalid or after the last available data. Use a format like 2004-01-01.")
    st.stop()
available = list(F_all.columns)
_late = F_all.index[0] + pd.DateOffset(years=4)  # an indicator that starts later than this would cut the usable history, so it is off by default
default = [c for c in available if (c in CORE or c.startswith("nom_")) and (F_all[c].first_valid_index() or pd.Timestamp.max) <= _late] or available
if "backwd" not in available and "roll_proxy" in available:  # no EIA futures curve: use the ETF-based roll-drag proxy for market structure
    default.append("roll_proxy")
with st.sidebar:
    st.caption(f"Data through {m.index[-1]:%b %Y}, downloaded {(time.time() - data_ts) / 60:.0f} min ago. Changing settings reuses it.")
    st.subheader("Indicators")
    chosen = st.multiselect("Indicators to use", available, default=default, format_func=lambda c: META_ALL[c][0], disabled=LK,
                            help="Core set is on by default. More indicators means more chances for a fluke. "
                                 "Indicators that start late (CFTC 2006, OVX 2007, Brent 2007, USO 2006, EEM 2003) or need an EIA key / your own CSV file are optional because they shorten the usable history or may be missing.")
    use_exp = st.checkbox("📈 Add expectation indicators to every model", False, disabled=LK,
                          help="Adds the twelve forward-looking indicators (expected Fed path, real rate, inflation, 10Y yield, growth, recession probability, unemployment, USD, "
                               "industrial demand, futures-curve expectation, oil demand) to the rules, ML, analogues and backtest. They are always shown in the Expectations tab and are available to the "
                               "Projection tab through its own feature setting, whatever this box says.")
if len(chosen) < 3:
    bar.empty()
    st.error("Pick at least 3 indicators.")
    st.stop()

base_cols = list(chosen) + ([c for c in EXP_KEYS if c in available and c not in chosen] if use_exp else [])
F_ok = F_all[base_cols].dropna()
F_src, fac_members = F_full[base_cols].dropna(), {}
F_raw = F_src  # raw (uncompressed) indicators, kept for the ablations
n_l = max(int(len(F_ok) * train_frac) - max(HORIZONS), 36)
struct_idx = F_ok.index[:n_l]  # structure window for factors: early learn period only, no outcomes involved
if fac_mode:
    F_src, fmeta, fac_members = build_factors(F_src, struct_idx, fac_thr)
    META_ALL.update(fmeta)
    F_ok = F_src.loc[F_ok.index]
P_ok = None
if pit:
    P_src = pit_percentiles(F_src) if fac_mode else P_full[base_cols]
    P_ok = P_src.reindex(F_ok.index)
    F_ok = F_ok.loc[P_ok.notna().all(axis=1)]
    P_ok = P_ok.loc[F_ok.index]
META = {c: META_ALL[c] for c in F_ok.columns}
cfg = ("q", q_thr, need_consistent, "t") if sel_mode.startswith("FDR") else ("t", t_thr, need_consistent, "t")
cfg = (cfg[0], cfg[1], cfg[2], "equal" if wlabel.startswith("Equal") else "t")
impl_key = "ETF" if impl.startswith("ETF") else "Futures"
FUT_COL = "oil_tr" if ("oil_tr" in m and m["oil_tr"].notna().sum() > 36) else "oil_fut"  # roll-adjusted futures when the EIA curve exists
if impl_key == "ETF" and not (m["uso"].notna().sum() > 36 and m["spy"].notna().sum() > 36):
    st.sidebar.warning("ETF prices unavailable, using futures.")
    impl_key = "Futures"

if "real_yield" in failed:
    st.warning("FRED did not respond, so the app is using Yahoo stand-ins (10y yield, 3-month T-bill). Real rates, inflation, M2, "
               "industrial production, credit spreads, unemployment and financial conditions are missing. Reload later, or add a free FRED_API_KEY in Secrets.")
elif failed:
    st.info("Some data series were unavailable and skipped: " + ", ".join(failed)
            + (" (the expectation indicators that need them are left out)" if any(k in EXP_FRED for k in failed) else "")
            + ". (cot = CFTC positioning, gpr = geopolitical risk file (needs the xlrd package), eia_steo / eia_futures = EIA API; the rest of the app works without them.)")
if not get_eia_key():
    st.info("No EIA_API_KEY found, so OPEC production, OPEC spare capacity, world oil demand, the WTI futures curve (backwardation / contango, curve-implied expectation) and the "
            "roll-adjusted futures return are switched off. A free key from eia.gov/opendata adds them. Weekly US inventories, production, refinery utilization and fuel demand "
            "need no EIA key (they come through FRED).")

scan = None
if auto_h:
    cb(0.25, "Scanning look-ahead periods (cached after the first run)...")
    scan = scan_horizons(m, F_ok, P_ok, target, ml_choice if run_ml else None, train_frac, cfg, HORIZONS)
    key = "stable" if scan["stable"].notna().any() else "combined"
    h = int(scan.loc[scan[key].idxmax(), "h"]) if scan[key].notna().any() else 6
else:
    h = h_manual

cb(0.35, "Analysing macro environments...")
fwd = target_returns(m, F_ok.index, h, target)
fdd = fwd_drawdown(m["oil"], h).reindex(F_ok.index)
if len(F_ok) < 60:
    bar.empty()
    _lt = F_all[base_cols].apply(lambda s_: s_.first_valid_index()).dropna().sort_values().index[-3:]
    st.error(f"Only {len(F_ok)} months have all the chosen indicators. Move the start date earlier or drop the indicators with the latest start "
             f"({', '.join(META_ALL[c][0] for c in _lt)}). See the Data & coverage tab.")
    st.stop()
sp = split3(F_ok, fwd, h, train_frac, val_frac)
ev, learn_idx, val_idx, test_idx = sp["ev"], sp["learn"], sp["val"], sp["test"]
unseen_idx = sp["unseen"] if REVEAL else val_idx  # research mode never grades on the final test
ev_r = ev if REVEAL else ev.intersection(learn_idx.union(val_idx))
grade_idx = test_idx if REVEAL else val_idx
grade_name = "Final test" if REVEAL else "Validation"
unseen_lbl = "validation + final test" if REVEAL else "validation"
if len(learn_idx) < 36 or len(val_idx) < 8 or len(test_idx) < 12:
    bar.empty()
    st.error("Not enough history for a learn / validation / test split. Move the start date earlier, shrink the learn or validation share, or use fewer indicators.")
    st.stop()

peek = None
if REVEAL:
    peek = register_peek(repr((start, target, h, auto_h, sorted(base_cols), fac_mode, fac_thr, use_vintage, train_frac, val_frac, pit, cfg,
                               run_ml, ml_choice, margin, step, mode, impl_key, bps, find_best, rank_by, use_exp,
                               proj_model, proj_feats, proj_len, proj_trust)))

S, stats, good, bad, score, verdict = run_rules(F_ok, P_ok, fwd, learn_idx, h, cfg)
wfr = walk_forward_rules(F_ok, P_ok, fwd, h, sp["cut"], RULE_STEP, cfg)
wa = analogue_walk_forward(F_ok, fwd, h, sp["cut"])
ana_ver = analogue_verdict(wa)

ml = {}
if run_ml:
    for i, name in enumerate(ML_MODELS):
        cb(0.4 + 0.25 * i / len(ML_MODELS), f"Training {name} ({h}-month look-ahead)...")
        ml[name] = walk_forward(F_ok, fwd, h, sp["cut"], step, name)
    ml_p, ml_base, ml_imp, ml_contrib = fit_today(F_ok, fwd, F_ok.iloc[-1], ml_choice)
cb(0.7, "Reading oil, stocks and oil vs stocks...")
NOW = {k: analyse_target(k, m, F_ok, P_ok, h, train_frac, val_frac, cfg, run_ml, ml_choice, step, margin, REVEAL) for k in NOW_KINDS}
cb(1.0, "Done")
bar.empty()

now = F_ok.index[-1]
v_now, sc_now = verdict.loc[now], float(score.loc[now])
ana = analogues(F_ok, fwd, ev_r, F_ok.iloc[-1], h)
reg = classify_regime(F_all.reindex(F_ok.index), F_ok.index, learn_idx)
streak = streak_info(verdict, fwd, ev_r)
expo_list = [EXPO[v_now]]
if run_ml:
    dm = ml_p - ml_base
    ml_v_now = FAV if dm >= margin else (UNF if dm <= -margin else NEU)
    expo_list.append(EXPO[ml_v_now])
expo_now = float(np.mean(expo_list))
ICON = {FAV: "🟢", NEU: "⚪", UNF: "🔴"}
tgt_txt = TGT_TXT[target]
tgt_note = {"vol": f" Outcomes are oil's forward return times a position-size multiplier ({VOL_TARGET:.0%} target / trailing volatility, known at the signal date), so calm and wild periods count more equally. It is not a realised vol-targeted portfolio.",
            "dd": f" Outcome shown = worst drawdown over the look-ahead plus {DD_FLOOR:.0%} points, so positive means the drawdown stayed shallower than {DD_FLOOR:.0%}. "
                  "Favorable = historically calm / shallow-drawdown environment.",
            "ddb": f" Outcome = 1 if oil's worst drawdown over the look-ahead stayed shallower than {DD_FLOOR:.0%}, else 0, so 'Avg after' is the share of cases that avoided it. "
                   "Favorable = historically calm / shallow-drawdown environment."}.get(target, "")

st.info(f"Predicting: **{target_name}** over **{h} months**" + (" (auto-selected on the learning period only, then locked)" if auto_h else "")
        + f"  |  {'Factors / indicators' if fac_mode else 'Indicators'}: **{len(META)}**  |  Learn {learn_idx[0]:%b %Y}–{learn_idx[-1]:%b %Y}, validation {val_idx[0]:%b %Y}–{val_idx[-1]:%b %Y}, "
        + (f"final test {test_idx[0]:%b %Y}–{test_idx[-1]:%b %Y}" if REVEAL else "final test hidden (research mode)") + tgt_note)

if REVEAL:
    n_pk, new_pk = peek
    (st.success if n_pk == 1 else st.warning)(
        "🔒 Final test revealed. " + ("This is the first set of settings it has ever been shown for, so it is a clean grade." if n_pk == 1 else
        f"It has now been shown under {n_pk} different sets of settings ({'this one is new' if new_pk else 'this one was already counted'}). "
        "Only the first reveal is a clean test; every extra look turns it into a selection tool, so treat results as optimistic."))
else:
    st.info("🔬 Research mode: the final test period is hidden everywhere and all grading uses the validation period. "
            "Switch to Locked evaluation in the sidebar once your settings are frozen.")

tabs_def = [("dash", "📊 Dashboard"), ("guide", "📖 Guide"), ("env", "🗺 Environments"), ("rank", "🏆 Indicator ranking"),
            ("test", "🧪 Out-of-sample"), ("reg", "🧭 Regimes & analogues"), ("bt", "💰 Backtest"), ("rob", "🛡 Robustness"),
            ("exp", "🔎 Explorer"), ("xpt", "📈 Expectations"), ("proj", "🔮 Projection")]
if run_ml:
    tabs_def.append(("ml", "🤖 Machine learning"))
if auto_h:
    tabs_def.append(("scan", "🔍 Look-ahead scan"))
tabs_def.append(("now", "✅ Oil & stocks now"))
tabs_def.append(("data", "🗂 Data & coverage"))
tabs_def.append(("report", "📄 Report (Word)"))
T = dict(zip([k for k, _ in tabs_def], st.tabs([n for _, n in tabs_def])))

# ------------------------------------------------------------------ tab modules
# Each tab file is executed with everything defined above (config, data, helpers, settings, results) available as globals,
# and whatever it defines is passed on to the next file.
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
_ns = {k: v for k, v in globals().items() if not (k.startswith("__") and k.endswith("__"))}
for _fname in ("oil_tabs_overview.py", "oil_tabs_analysis.py", "oil_tabs_expectations.py", "oil_tabs_projection.py"):
    _out = runpy.run_path(os.path.join(_HERE, _fname), init_globals=_ns)
    _ns.update({k: v for k, v in _out.items() if not (k.startswith("__") and k.endswith("__"))})
