"""
Bitcoin Macro Analyzer: on-chain and crypto-market data engine (NO Streamlit calls, so it is safe inside worker threads).

Free, key-less sources (every one is optional: a failure removes only the indicators that need it, never the app):
  Coin Metrics Community API  (community-api.coinmetrics.io)  daily BTC on-chain series: MVRV, realized cap, market cap, supply,
                                                               supply active in the last year, active addresses, transactions,
                                                               hash rate, fees, issuance, adjusted NVT, reference price (from 2010).
  blockchain.com charts API   (api.blockchain.info)            fallback for hash rate, active addresses, transactions, fees, miner revenue.
  DefiLlama stablecoins API   (stablecoins.llama.fi)           total stablecoin supply (crypto "dry powder"), from 2017.
  alternative.me              (api.alternative.me/fng)         Crypto Fear & Greed index, from 2018.

POINT-IN-TIME: every daily series is moved forward by a publication lag (ONCHAIN_LAG_DAYS) before it is used, so a month-end value only
contains data that was public at that month-end. Derived indicators use trailing windows only (no centred or future windows).

Honest limits (also shown in the app):
  * Coin Metrics Community values can be revised when address clustering improves. The history you download today is the REVISED history,
    not what traders saw at the time (there is no free first-release vintage for on-chain data).
  * Exchange net flows, ETF flows, funding rates, open interest and long-term-holder metrics are paid or have no clean free history, so they are not included.
"""
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
# our name -> Coin Metrics community metric id.  Each metric is requested on its own so one unsupported id cannot break the others.
CM_METRICS = {"price": "PriceUSD", "mvrv": "CapMVRVCur", "cap_real": "CapRealUSD", "cap_mkt": "CapMrktCurUSD", "sply": "SplyCur",
              "sply_act1y": "SplyAct1yr", "adr_act": "AdrActCnt", "tx_cnt": "TxCnt", "hashrate": "HashRate",
              "fee_usd": "FeeTotUSD", "iss_usd": "IssTotUSD", "nvt": "NVTAdj"}
BC_URL = "https://api.blockchain.info/charts/{name}"
# fallback chart names (blockchain.com) for the series that matter most
BC_CHARTS = {"hashrate": "hash-rate", "adr_act": "n-unique-addresses", "tx_cnt": "n-transactions",
             "fee_usd": "transaction-fees-usd", "rev_usd": "miners-revenue"}
LLAMA_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
ONCHAIN_LAG_DAYS = 2       # on-chain values for day d are used from d + 2 days (conservative: data providers publish with a delay)
STABLE_LAG_DAYS = 2
FNG_LAG_DAYS = 1
UA = {"User-Agent": "Mozilla/5.0"}
# Block-reward halvings (genesis, then every 210,000 blocks). Dates are history, not a forecast; the next one is expected around April 2028.
HALVINGS = pd.to_datetime(["2009-01-03", "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])

# indicator -> (title, label when LOW, label when HIGH, is a % change / fraction?, pillar)   (same layout as META_ALL in bitcoin_main.py)
VAL, NET, SENT = "On-chain valuation", "On-chain network", "Crypto market & sentiment"
ONCHAIN_META = {
    "mvrv":        ("MVRV (market cap / realized cap)", "Price near or below holders' cost basis", "Price far above holders' cost basis", False, VAL),
    "mvrv_z":      ("MVRV Z-score (market minus realized cap, in std devs)", "Undervalued vs cost basis", "Overvalued vs cost basis", False, VAL),
    "mayer":       ("Mayer multiple (price / 200-day average)", "Price below long-term trend", "Price stretched above long-term trend", False, VAL),
    "puell":       ("Puell multiple (miner issuance revenue / 1-year average)", "Miner revenue depressed", "Miner revenue elevated", False, VAL),
    "nvt":         ("NVT (network value / adjusted transfer volume, 90-day avg)", "Network used heavily for its value", "Network value high vs usage", False, VAL),
    "hash_ribbon": ("Hash ribbon (30d / 60d hash rate, %)", "Miner stress / capitulation", "Hash rate expanding", False, NET),
    "hash_mom":    ("Hash rate, 3-month change", "Hash rate falling", "Hash rate rising", True, NET),
    "adr_mom":     ("Active addresses, 3-month change (30d average)", "Fewer active users", "More active users", True, NET),
    "tx_mom":      ("Transaction count, 3-month change (30d average)", "Less on-chain activity", "More on-chain activity", True, NET),
    "fee_share":   ("Fees as % of miner revenue (30d)", "Fees a small part of revenue", "Fees a large part of revenue (congestion)", False, NET),
    "act_supply":  ("Share of supply that moved in the last year (%)", "Coins dormant (holders holding)", "Coins actively moving", False, NET),
    "stable_mom":  ("Stablecoin supply, 3-month change", "Stablecoin supply shrinking", "Stablecoin supply growing", True, SENT),
    "fng":         ("Crypto Fear & Greed index (7d average)", "Extreme fear", "Extreme greed", False, SENT),
    "halving_m":   ("Months since the last halving", "Early in the halving cycle", "Late in the halving cycle", False, SENT),
    "cot_lev":     ("CME futures: leveraged funds net position (% of open interest)", "Leveraged funds net short (often the basis trade)", "Leveraged funds net long", False, SENT),
    "cot_am":      ("CME futures: asset managers net position (% of open interest)", "Institutions light / short", "Institutions long", False, SENT),
}
CHAIN_KEYS = list(ONCHAIN_META)
# Textbook sign when the indicator is HIGH for bitcoin (+1 good / -1 bad / 0 mixed) and how each one is built (shown in the On-chain tab)
ONCHAIN_INFO = {
    "mvrv": dict(sign=-1, src="Coin Metrics CapMVRVCur", how="Market capitalisation divided by realized capitalisation (every coin valued at the price it last moved). Above ~3 has marked past cycle tops, near or below 1 has marked bottoms. Textbook: high = overheated."),
    "mvrv_z": dict(sign=-1, src="Coin Metrics CapMrktCurUSD, CapRealUSD", how="(Market cap - realized cap) divided by the EXPANDING standard deviation of market cap (only past data, at least one year). A standardised version of MVRV."),
    "mayer": dict(sign=-1, src="BTC price (Yahoo / Coin Metrics)", how="Price divided by its 200-day moving average. Textbook: very high = stretched, below 1 = below trend."),
    "puell": dict(sign=-1, src="Coin Metrics IssTotUSD", how="Daily USD value of newly issued coins divided by its 365-day average. Low = miners under pressure (historically near bottoms), high = miners paid richly (near tops)."),
    "nvt": dict(sign=-1, src="Coin Metrics NVTAdj", how="Adjusted network value to transfer-volume ratio (a 'P/E' for the network), smoothed over 90 days. High = value is high relative to on-chain usage."),
    "hash_ribbon": dict(sign=1, src="Coin Metrics HashRate", how="30-day average hash rate divided by its 60-day average, minus 1. Negative = miners switching machines off (capitulation), which has preceded several bottoms; positive = expansion."),
    "hash_mom": dict(sign=1, src="Coin Metrics HashRate", how="Change in the 30-day average hash rate over 90 days: network security investment trend."),
    "adr_mom": dict(sign=1, src="Coin Metrics AdrActCnt", how="Change in the 30-day average of daily active addresses over 90 days: user adoption trend (addresses are not people; exchanges and batching distort it)."),
    "tx_mom": dict(sign=1, src="Coin Metrics TxCnt", how="Change in the 30-day average daily transaction count over 90 days (inscription / spam waves distort it in 2023-24)."),
    "fee_share": dict(sign=0, src="Coin Metrics FeeTotUSD, IssTotUSD", how="Fees divided by fees plus new issuance, 30-day sums. High means a congested chain, which can mean euphoria or a spam wave: no clear textbook sign."),
    "act_supply": dict(sign=0, src="Coin Metrics SplyAct1yr, SplyCur", how="Percent of all coins that moved at least once in the last year. Low = long-term holding; it rises in bull markets as old coins are sold. Mixed sign."),
    "stable_mom": dict(sign=1, src="DefiLlama stablecoin supply", how="Change in total stablecoin supply over 90 days (7-day average): fresh dollars waiting inside crypto. Starts 2017."),
    "fng": dict(sign=0, src="alternative.me Fear & Greed", how="Index of volatility, momentum, social and survey inputs (partly price-derived), 7-day average. Contrarian or trend-following? The data decides. Starts 2018."),
    "halving_m": dict(sign=0, src="known halving dates", how="Months since the last block-reward halving (known history; the next is expected around April 2028). Only three full cycles exist, so any pattern here is thin evidence."),
    "cot_lev": dict(sign=0, src="CFTC Traders in Financial Futures, CME Bitcoin (133741)", how="Leveraged funds net position as a % of open interest, weekly. Mostly the futures leg of the cash-and-carry basis trade, so net SHORT can be a sign of institutional spot buying. Starts Dec 2017."),
    "cot_am": dict(sign=1, src="CFTC Traders in Financial Futures, CME Bitcoin (133741)", how="Asset managers net position as a % of open interest, weekly. Starts Dec 2017."),
}


# ------------------------------------------------------------------ fetchers (plain functions, no Streamlit)
def _cm_one(metric, start):
    url, params, rows = CM_URL, dict(assets="btc", metrics=metric, frequency="1d", start_time=start, page_size=10000), []
    for _ in range(12):
        r = requests.get(url, params=params, timeout=(5, 45), headers=UA)
        r.raise_for_status()
        j = r.json()
        rows += j.get("data", [])
        url = j.get("next_page_url")
        if not url:
            break
        params = None
    d = pd.DataFrame(rows)
    if d.empty or metric not in d:
        raise ValueError(f"no data for {metric}")
    s = pd.Series(pd.to_numeric(d[metric], errors="coerce").values, index=pd.to_datetime(d["time"], utc=True).dt.tz_localize(None).dt.normalize())
    s = s.dropna()
    return s[~s.index.duplicated(keep="last")].sort_index().astype(float)


def _bc_one(chart):
    r = requests.get(BC_URL.format(name=chart), timeout=(5, 45), headers=UA,
                     params=dict(timespan="all", sampled="true", metadata="false", cors="true", format="json"))
    r.raise_for_status()
    v = r.json()["values"]
    s = pd.Series([float(x["y"]) for x in v], index=pd.to_datetime([x["x"] for x in v], unit="s").normalize())
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq="D")).interpolate(limit=10)  # 'sampled' history is every few days


def fetch_onchain(start="2010-01-01"):
    """Returns (raw daily frame with the Coin Metrics-style columns, list of metric names that failed). Never raises."""
    out, failed = {}, []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {k: ex.submit(_cm_one, v, start) for k, v in CM_METRICS.items()}
        for k, f in futs.items():
            try:
                out[k] = f.result()
            except Exception:  # noqa: BLE001
                failed.append(k)
    need_bc = [k for k in ("hashrate", "adr_act", "tx_cnt", "fee_usd") if k not in out] + (["iss_usd"] if "iss_usd" not in out else [])
    if need_bc:  # blockchain.com fallback
        bc = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {k: ex.submit(_bc_one, ch) for k, ch in BC_CHARTS.items()
                    if k in need_bc or (k == "rev_usd" and "iss_usd" in need_bc) or (k == "fee_usd" and "iss_usd" in need_bc)}
            for k, f in futs.items():
                try:
                    bc[k] = f.result()
                except Exception:  # noqa: BLE001
                    pass
        for k in ("hashrate", "adr_act", "tx_cnt", "fee_usd"):
            if k not in out and k in bc:
                out[k] = bc[k]
                failed = [x for x in failed if x != k]
        if "iss_usd" not in out and "rev_usd" in bc and "fee_usd" in out:  # issuance = miner revenue - fees
            out["iss_usd"] = (bc["rev_usd"] - out["fee_usd"].reindex(bc["rev_usd"].index)).clip(lower=0).dropna()
            failed = [x for x in failed if x != "iss_usd"]
    return (pd.DataFrame(out).sort_index() if out else pd.DataFrame()), sorted(set(failed))


def fetch_stablecoins():
    """Total stablecoin supply in USD (daily, from 2017)."""
    r = requests.get(LLAMA_URL, timeout=(5, 45), headers=UA)
    r.raise_for_status()
    idx, val = [], []
    for e in r.json():
        v = (e.get("totalCirculatingUSD") or e.get("totalCirculating") or {}).get("peggedUSD")
        if v is None:
            continue
        idx.append(pd.to_datetime(int(e["date"]), unit="s").normalize())
        val.append(float(v))
    s = pd.Series(val, index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < 200:
        raise ValueError("too few stablecoin observations")
    return s


def fetch_fng():
    """Crypto Fear & Greed index (0 = extreme fear, 100 = extreme greed), daily from 2018."""
    r = requests.get(FNG_URL, timeout=(5, 45), headers=UA)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["data"])
    s = pd.Series(pd.to_numeric(d["value"], errors="coerce").values, index=pd.to_datetime(d["timestamp"].astype(int), unit="s").dt.normalize())
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < 200:
        raise ValueError("too few Fear & Greed observations")
    return s


# ------------------------------------------------------------------ derived, point-in-time features (daily)
def halving_months(index):
    """Months since the most recent halving at each date (NaN before the genesis block)."""
    idx = pd.DatetimeIndex(index)
    pos = HALVINGS.searchsorted(idx, side="right") - 1
    last = pd.Series(HALVINGS[np.clip(pos, 0, len(HALVINGS) - 1)], index=idx)
    out = (idx - pd.DatetimeIndex(last.values)).days / 30.4375
    return pd.Series(np.where(pos >= 0, out, np.nan), index=idx)


def _shift_avail(s, days):
    """Value observed on day d is usable from d + days (publication lag)."""
    s = s.dropna().copy()
    s.index = s.index + pd.Timedelta(days=days)
    return s[~s.index.duplicated(keep="last")]


def build_onchain(price, raw, stable=None, fng=None):
    """Daily DataFrame of the on-chain indicators, already lagged to when they were public. price = daily BTC price (spliced). Trailing windows only."""
    f = {}

    def has(k):
        return raw is not None and k in raw and raw[k].notna().sum() > 400

    if has("mvrv"):
        f["mvrv"] = raw["mvrv"]
    mc = raw["cap_mkt"] if has("cap_mkt") else (price * raw["sply"] if has("sply") else None)
    if mc is not None and has("cap_real"):
        rc = raw["cap_real"].reindex(mc.index)
        sd = mc.expanding(min_periods=365).std()
        f["mvrv_z"] = (mc - rc) / sd
        if "mvrv" not in f:
            f["mvrv"] = mc / rc
    p = price.dropna()
    f["mayer"] = p / p.rolling(200, min_periods=200).mean()
    if has("iss_usd"):
        iss = raw["iss_usd"]
        f["puell"] = iss / iss.rolling(365, min_periods=300).mean()
    if has("nvt"):
        f["nvt"] = raw["nvt"].rolling(90, min_periods=60).mean()
    if has("hashrate"):
        h30 = raw["hashrate"].rolling(30, min_periods=20).mean()
        f["hash_ribbon"] = (h30 / raw["hashrate"].rolling(60, min_periods=40).mean() - 1) * 100
        f["hash_mom"] = h30.pct_change(90)
    if has("adr_act"):
        f["adr_mom"] = raw["adr_act"].rolling(30, min_periods=20).mean().pct_change(90)
    if has("tx_cnt"):
        f["tx_mom"] = raw["tx_cnt"].rolling(30, min_periods=20).mean().pct_change(90)
    if has("fee_usd") and has("iss_usd"):
        fe, iss = raw["fee_usd"].rolling(30, min_periods=20).sum(), raw["iss_usd"].rolling(30, min_periods=20).sum()
        f["fee_share"] = fe / (fe + iss) * 100
    if has("sply_act1y") and has("sply"):
        f["act_supply"] = raw["sply_act1y"] / raw["sply"].reindex(raw["sply_act1y"].index) * 100
    out = {k: _shift_avail(v.replace([np.inf, -np.inf], np.nan), ONCHAIN_LAG_DAYS) for k, v in f.items()}
    if stable is not None and len(stable):
        out["stable_mom"] = _shift_avail(stable.rolling(7, min_periods=4).mean().pct_change(90), STABLE_LAG_DAYS)
    if fng is not None and len(fng):
        out["fng"] = _shift_avail(fng.rolling(7, min_periods=4).mean(), FNG_LAG_DAYS)
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()
