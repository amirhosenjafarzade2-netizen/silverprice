"""Bitcoin Macro Analyzer — on-chain and crypto-market data engine.

This module intentionally contains no Streamlit calls, so its functions can be run
inside a worker thread and tested independently.

Sources (all optional; a source failure should not stop the whole app):
* Coin Metrics Community API: BTC price and on-chain metrics.
* Blockchain.com Charts API: fallback network activity, hash rate, fees and miner revenue.
* DefiLlama: aggregate stablecoin supply.
* Alternative.me: Crypto Fear & Greed index.

Point-in-time handling
----------------------
Daily observations are shifted forward by a configurable publication lag before
being returned as features. Rolling calculations are trailing-only. This reduces
look-ahead bias, but does NOT make the history a true vintage dataset: providers can
revise historical on-chain observations, and daily close/publication timestamps can
vary. See ONCHAIN_INFO for source/methodology notes.

Important limitations
---------------------
* Coin Metrics Community history is the history available today and can be revised.
* Blockchain.com sampled history is interpolated only across short gaps; it is not
  equivalent to complete daily observations.
* Stablecoin supply is a broad proxy for crypto liquidity, not a measure of cash
  immediately available to buy BTC.
* Addresses are not people; batching, exchange wallets, inscriptions and spam can
  distort activity metrics.
* Exchange flows, ETF flows, funding, open interest and CFTC positioning are not
  fetched here. CFTC fields remain in the metadata because the main app loads them.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_METRICS = {
    "price": "PriceUSD",
    "mvrv": "CapMVRVCur",
    "cap_real": "CapRealUSD",
    "cap_mkt": "CapMrktCurUSD",
    "sply": "SplyCur",
    "sply_act1y": "SplyAct1yr",
    "adr_act": "AdrActCnt",
    "tx_cnt": "TxCnt",
    "hashrate": "HashRate",
    "fee_usd": "FeeTotUSD",
    "iss_usd": "IssTotUSD",
    "nvt": "NVTAdj",
}
BC_URL = "https://api.blockchain.info/charts/{name}"
BC_CHARTS = {
    "hashrate": "hash-rate",
    "adr_act": "n-unique-addresses",
    "tx_cnt": "n-transactions",
    "fee_usd": "transaction-fees-usd",
    "rev_usd": "miners-revenue",
}
LLAMA_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"

ONCHAIN_LAG_DAYS = 2
STABLE_LAG_DAYS = 2
FNG_LAG_DAYS = 1
REQUEST_TIMEOUT = (6, 45)
MAX_CM_PAGES = 20
MAX_WORKERS = 6
UA = {"User-Agent": "BitcoinMacroAnalyzer/1.1 (+public-data research; contact: none)"}

# Known historical halvings. The next halving is not hard-coded as a forecast.
HALVINGS = pd.to_datetime(
    ["2009-01-03", "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]
)

VAL, NET, SENT = "On-chain valuation", "On-chain network", "Crypto market & sentiment"
ONCHAIN_META = {
    "mvrv": ("MVRV (market cap / realized cap)", "Price near or below holders' cost basis", "Price far above holders' cost basis", False, VAL),
    "mvrv_z": ("MVRV Z-score (market minus realized cap, in std devs)", "Undervalued vs cost basis", "Overvalued vs cost basis", False, VAL),
    "mayer": ("Mayer multiple (price / 200-day average)", "Price below long-term trend", "Price stretched above long-term trend", False, VAL),
    "puell": ("Puell multiple (miner issuance revenue / 1-year average)", "Miner revenue depressed", "Miner revenue elevated", False, VAL),
    "nvt": ("NVT (network value / adjusted transfer volume, 90-day avg)", "Network used heavily for its value", "Network value high vs usage", False, VAL),
    "hash_ribbon": ("Hash ribbon (30d / 60d hash rate, %)", "Miner stress / capitulation", "Hash rate expanding", False, NET),
    "hash_mom": ("Hash rate, 3-month change", "Hash rate falling", "Hash rate rising", True, NET),
    "adr_mom": ("Active addresses, 3-month change (30d average)", "Fewer active users", "More active users", True, NET),
    "tx_mom": ("Transaction count, 3-month change (30d average)", "Less on-chain activity", "More on-chain activity", True, NET),
    "fee_share": ("Fees as % of miner revenue (30d)", "Fees a small part of revenue", "Fees a large part of revenue (congestion)", False, NET),
    "act_supply": ("Share of supply that moved in the last year (%)", "Coins dormant (holders holding)", "Coins actively moving", False, NET),
    "stable_mom": ("Stablecoin supply, 3-month change", "Stablecoin supply shrinking", "Stablecoin supply growing", True, SENT),
    "fng": ("Crypto Fear & Greed index (7d average)", "Extreme fear", "Extreme greed", False, SENT),
    "halving_m": ("Months since the last halving", "Early in the halving cycle", "Late in the halving cycle", False, SENT),
    # CFTC data is supplied by bitcoin_main.py, not by this module.
    "cot_lev": ("CME futures: leveraged funds net position (% of open interest)", "Leveraged funds net short (often the basis trade)", "Leveraged funds net long", False, SENT),
    "cot_am": ("CME futures: asset managers net position (% of open interest)", "Institutions light / short", "Institutions long", False, SENT),
}
CHAIN_KEYS = list(ONCHAIN_META)
ONCHAIN_INFO = {
    "mvrv": dict(sign=-1, src="Coin Metrics CapMVRVCur", how="Market capitalisation divided by realized capitalisation (coins valued at the price they last moved). Historical thresholds are regime-dependent; high values have coincided with overheated markets."),
    "mvrv_z": dict(sign=-1, src="Coin Metrics CapMrktCurUSD, CapRealUSD", how="(Market cap - realized cap) divided by the expanding standard deviation of market cap, using past observations only. This is one common MVRV-Z construction; implementation/data differences can change values."),
    "mayer": dict(sign=-1, src="BTC price (Yahoo / Coin Metrics)", how="Price divided by its 200-day moving average. Very high values indicate price stretched above trend; values below 1 indicate price below trend."),
    "puell": dict(sign=-1, src="Coin Metrics IssTotUSD", how="Daily USD value of newly issued coins divided by its trailing 365-day average. Low values can indicate miner revenue pressure; high values indicate elevated issuance revenue."),
    "nvt": dict(sign=-1, src="Coin Metrics NVTAdj", how="Adjusted network-value-to-transfer-volume metric, smoothed over 90 days. High values indicate network value is high relative to adjusted transfer activity."),
    "hash_ribbon": dict(sign=1, src="Coin Metrics HashRate", how="(30-day average hash rate / 60-day average hash rate - 1) × 100. Negative values indicate the shorter average is below the longer average; this is not a guaranteed bottom signal."),
    "hash_mom": dict(sign=1, src="Coin Metrics HashRate", how="Percentage change over 90 days in the 30-day average hash rate."),
    "adr_mom": dict(sign=1, src="Coin Metrics AdrActCnt", how="Percentage change over 90 days in the 30-day average of daily active addresses. Addresses are not unique people."),
    "tx_mom": dict(sign=1, src="Coin Metrics TxCnt", how="Percentage change over 90 days in the 30-day average daily transaction count. Inscription and spam waves can distort it."),
    "fee_share": dict(sign=0, src="Coin Metrics FeeTotUSD, IssTotUSD", how="30-day fees divided by 30-day fees plus issuance revenue, as a percentage. High values can reflect congestion or speculative activity; the signal is mixed."),
    "act_supply": dict(sign=0, src="Coin Metrics SplyAct1yr, SplyCur", how="Supply active within the last year divided by current supply, as a percentage. Interpret cautiously; it is not a pure long-term-holder measure."),
    "stable_mom": dict(sign=1, src="DefiLlama stablecoin supply", how="Percentage change over 90 days in the 7-day average of aggregate stablecoin supply. A broad crypto-liquidity proxy, not direct BTC buying power."),
    "fng": dict(sign=0, src="alternative.me Fear & Greed", how="7-day average of the 0–100 index. It includes market/momentum inputs and may be used contrarian or trend-following; the sign is intentionally unspecified."),
    "halving_m": dict(sign=0, src="known halving dates", how="Approximate months since the most recent known halving. There are few independent cycles, so historical patterns are weak evidence."),
    "cot_lev": dict(sign=0, src="CFTC Traders in Financial Futures, CME Bitcoin", how="Leveraged-funds net position as a percentage of open interest. Shorts can include cash-and-carry basis trades; do not interpret as a standalone bearish signal."),
    "cot_am": dict(sign=1, src="CFTC Traders in Financial Futures, CME Bitcoin", how="Asset-manager net position as a percentage of open interest. The main app loads this series separately."),
}


def _new_session() -> requests.Session:
    """Create a per-task session with bounded retries for transient HTTP failures."""
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(UA)
    return session


def _clean_series(values: Iterable, index: Iterable, *, positive: bool = False) -> pd.Series:
    """Normalize provider output into a unique, sorted, numeric daily Series."""
    idx = pd.to_datetime(index, utc=True, errors="coerce")
    idx = pd.DatetimeIndex(idx).tz_convert(None).normalize()
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy()
    n = min(len(idx), len(vals))
    s = pd.Series(vals[:n], index=idx[:n], dtype=float)
    s = s[~s.index.isna()].replace([np.inf, -np.inf], np.nan).dropna()
    if positive:
        s = s.where(s > 0).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = None
    return s


def _cm_one(metric: str, start: str = "2010-01-01") -> pd.Series:
    """Fetch one Coin Metrics daily series, following the provider's next_page_url."""
    rows = []
    url = CM_URL
    params = {"assets": "btc", "metrics": metric, "frequency": "1d", "start_time": start, "page_size": 10000}
    with _new_session() as session:
        for page in range(MAX_CM_PAGES):
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", [])
            if data:
                rows.extend(data)
            next_url = payload.get("next_page_url")
            if not next_url:
                break
            url, params = next_url, None
        else:
            raise RuntimeError(f"Coin Metrics pagination limit reached for {metric}")
    if not rows:
        raise ValueError(f"Coin Metrics returned no observations for {metric}")
    frame = pd.DataFrame(rows)
    if "time" not in frame or metric not in frame:
        raise ValueError(f"Coin Metrics response missing time or {metric}")
    s = _clean_series(frame[metric], frame["time"])
    if s.empty:
        raise ValueError(f"Coin Metrics returned no numeric observations for {metric}")
    return s


def _bc_one(chart: str) -> pd.Series:
    """Fetch a Blockchain.com chart and cautiously fill short gaps in sampled data."""
    url = BC_URL.format(name=chart)
    params = {"timespan": "all", "sampled": "true", "metadata": "false", "cors": "true", "format": "json"}
    with _new_session() as session:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    values = payload.get("values", [])
    if not values:
        raise ValueError(f"Blockchain.com returned no observations for {chart}")
    s = _clean_series([v.get("y") for v in values], pd.to_datetime([v.get("x") for v in values], unit="s", utc=True, errors="coerce"))
    if s.empty:
        raise ValueError(f"Blockchain.com returned no numeric observations for {chart}")
    # Daily reindexing supports rolling indicators. Interpolate at most 10 consecutive
    # missing days and do not extrapolate before/after the source's actual history.
    daily_index = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(daily_index).interpolate(method="time", limit=10, limit_area="inside").rename(chart)


def fetch_onchain(start: str = "2010-01-01") -> tuple[pd.DataFrame, list[str]]:
    """Return ``(raw_daily_frame, failed_metric_names)``; failures never escape.

    Each Coin Metrics metric is fetched independently, because the Community API can
    support some requested metrics but not others. Fallbacks are only used for metrics
    that did not load successfully. A missing metric is left missing rather than filled
    with a misleading zero.
    """
    out: dict[str, pd.Series] = {}
    failed: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(CM_METRICS))) as pool:
        futures = {pool.submit(_cm_one, metric, start): key for key, metric in CM_METRICS.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                out[key] = future.result().rename(key)
            except Exception:  # noqa: BLE001 - one failed provider metric must not stop other metrics
                failed.add(key)

    # Need issuance fallback inputs if issuance itself failed; miner revenue minus fees
    # is a proxy, not identical to Coin Metrics' issuance-revenue definition.
    need_bc = {k for k in ("hashrate", "adr_act", "tx_cnt", "fee_usd", "iss_usd") if k not in out}
    bc_needed = set(need_bc)
    if "iss_usd" in need_bc:
        bc_needed.add("rev_usd")
        bc_needed.add("fee_usd")
    bc_results: dict[str, pd.Series] = {}
    chart_futures = {}
    with ThreadPoolExecutor(max_workers=min(5, len(BC_CHARTS))) as pool:
        for key, chart in BC_CHARTS.items():
            if key in bc_needed:
                chart_futures[pool.submit(_bc_one, chart)] = key
        for future in as_completed(chart_futures):
            key = chart_futures[future]
            try:
                bc_results[key] = future.result().rename(key)
            except Exception:  # noqa: BLE001
                continue

    for key in ("hashrate", "adr_act", "tx_cnt", "fee_usd"):
        if key not in out and key in bc_results:
            out[key] = bc_results[key]
            failed.discard(key)

    if "iss_usd" not in out and "rev_usd" in bc_results:
        fees = out.get("fee_usd", bc_results.get("fee_usd"))
        if fees is not None:
            # Align both inputs to the revenue series; do not assume identical indexes.
            fee_aligned = fees.reindex(bc_results["rev_usd"].index)
            issuance_proxy = (bc_results["rev_usd"] - fee_aligned).clip(lower=0)
            issuance_proxy = issuance_proxy.where(fee_aligned.notna()).dropna()
            if not issuance_proxy.empty:
                out["iss_usd"] = issuance_proxy.rename("iss_usd")
                failed.discard("iss_usd")

    if not out:
        return pd.DataFrame(), sorted(failed or set(CM_METRICS))
    frame = pd.concat(out, axis=1).sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    # Do not silently forward-fill raw provider observations: downstream rolling
    # metrics should reflect actual missingness.
    return frame, sorted(failed)


def fetch_stablecoins() -> pd.Series:
    """Return daily aggregate stablecoin supply in USD (typically available from 2017)."""
    with _new_session() as session:
        response = session.get(LLAMA_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected DefiLlama stablecoin response format")

    dates, values = [], []
    for entry in payload:
        if not isinstance(entry, dict) or "date" not in entry:
            continue
        total = entry.get("totalCirculatingUSD")
        if total is None:
            total = entry.get("totalCirculating")
        if isinstance(total, dict):
            value = total.get("peggedUSD")
        else:
            value = total
        try:
            value = float(value)
            stamp = pd.to_datetime(int(entry["date"]), unit="s", utc=True).tz_localize(None).normalize()
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(value) and value > 0:
            dates.append(stamp)
            values.append(value)

    s = _clean_series(values, dates, positive=True).rename("stablecoin_supply")
    if len(s) < 200:
        raise ValueError(f"Too few stablecoin observations ({len(s)})")
    return s


def fetch_fng() -> pd.Series:
    """Return the daily Alternative.me Fear & Greed index (0–100), usually from 2018."""
    with _new_session() as session:
        response = session.get(FNG_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not rows:
        raise ValueError("Alternative.me returned no Fear & Greed observations")
    frame = pd.DataFrame(rows)
    if not {"timestamp", "value"}.issubset(frame.columns):
        raise ValueError("Unexpected Fear & Greed response format")
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    dates = pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce")
    s = _clean_series(frame["value"], dates).rename("fng")
    s = s.where(s.between(0, 100)).dropna()
    if len(s) < 200:
        raise ValueError(f"Too few Fear & Greed observations ({len(s)})")
    return s


def halving_months(index: Iterable) -> pd.Series:
    """Approximate months since the latest known halving; NaN before the first date."""
    idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    idx = idx.normalize()
    pos = HALVINGS.searchsorted(idx, side="right") - 1
    valid_pos = np.clip(pos, 0, len(HALVINGS) - 1)
    last = HALVINGS.take(valid_pos)
    values = (idx - last).days / 30.4375
    values = np.where((pos >= 0) & ~idx.isna(), values, np.nan)
    return pd.Series(values, index=idx, name="halving_m", dtype=float)


def _shift_avail(series: pd.Series, days: int) -> pd.Series:
    """A value dated day *d* becomes usable at *d + days*; normalize index safely."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float, name=getattr(series, "name", None))
    s = pd.to_numeric(series, errors="coerce").copy()
    idx = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True, errors="coerce"))
    idx = idx.tz_convert(None).normalize() + pd.Timedelta(days=int(days))
    s.index = idx
    s = s[~s.index.isna()].replace([np.inf, -np.inf], np.nan).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _raw_series(raw: pd.DataFrame | None, key: str) -> pd.Series | None:
    """Get a cleaned raw series with a normalized, sorted daily DatetimeIndex."""
    if raw is None or key not in raw.columns:
        return None
    s = pd.to_numeric(raw[key], errors="coerce").copy()
    idx = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True, errors="coerce"))
    s.index = idx.tz_convert(None).normalize()
    s = s[~s.index.isna()].replace([np.inf, -np.inf], np.nan)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _enough(series: pd.Series | None, minimum: int = 365) -> bool:
    return series is not None and int(series.notna().sum()) >= minimum


def _safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Percent change with zero/negative denominators converted to missing."""
    s = pd.to_numeric(series, errors="coerce")
    previous = s.shift(periods)
    valid = previous.abs() > 1e-12
    return (s / previous - 1).where(valid).replace([np.inf, -np.inf], np.nan)


def build_onchain(
    price: pd.Series,
    raw: pd.DataFrame | None,
    stable: pd.Series | None = None,
    fng: pd.Series | None = None,
) -> pd.DataFrame:
    """Build daily, lagged on-chain features from BTC price and raw provider data.

    Parameters match the existing ``bitcoin_main.py`` integration. Returns an empty
    DataFrame if no feature can be calculated. No future/centered rolling windows are
    used. Output is indexed by the date the value is considered available, not the
    original observation date.
    """
    if price is None or len(price) == 0:
        return pd.DataFrame()
    p = pd.to_numeric(price, errors="coerce").copy()
    pidx = pd.DatetimeIndex(pd.to_datetime(p.index, utc=True, errors="coerce"))
    p.index = pidx.tz_convert(None).normalize()
    p = p[~p.index.isna()].replace([np.inf, -np.inf], np.nan)
    p = p.where(p > 0).dropna()
    p = p[~p.index.duplicated(keep="last")].sort_index()
    if p.empty:
        return pd.DataFrame()

    raw_s = {key: _raw_series(raw, key) for key in CM_METRICS}
    features: dict[str, pd.Series] = {}

    mvrv = raw_s.get("mvrv")
    if _enough(mvrv):
        features["mvrv"] = mvrv

    market_cap = raw_s.get("cap_mkt")
    supply = raw_s.get("sply")
    if not _enough(market_cap) and _enough(supply):
        market_cap = p.mul(supply.reindex(p.index), axis=0).dropna()
    realized_cap = raw_s.get("cap_real")
    if _enough(market_cap) and _enough(realized_cap):
        common = market_cap.index.union(realized_cap.index)
        mc = market_cap.reindex(common)
        rc = realized_cap.reindex(common)
        # Expanding denominator is deliberately past-only. At least one year of data
        # is required to reduce unstable early-history values.
        std = mc.expanding(min_periods=365).std(ddof=1)
        features["mvrv_z"] = ((mc - rc) / std.where(std > 0)).replace([np.inf, -np.inf], np.nan)
        if "mvrv" not in features:
            features["mvrv"] = (mc / rc.where(rc > 0)).replace([np.inf, -np.inf], np.nan)

    features["mayer"] = p / p.rolling(200, min_periods=200).mean()

    issuance = raw_s.get("iss_usd")
    if _enough(issuance):
        issuance = issuance.where(issuance > 0)
        avg_issuance = issuance.rolling(365, min_periods=300).mean()
        features["puell"] = (issuance / avg_issuance.where(avg_issuance > 0)).replace([np.inf, -np.inf], np.nan)

    nvt = raw_s.get("nvt")
    if _enough(nvt):
        features["nvt"] = nvt.rolling(90, min_periods=60).mean()

    hashrate = raw_s.get("hashrate")
    if _enough(hashrate):
        h = hashrate.where(hashrate > 0)
        h30 = h.rolling(30, min_periods=20).mean()
        h60 = h.rolling(60, min_periods=40).mean()
        features["hash_ribbon"] = ((h30 / h60.where(h60 > 0)) - 1) * 100
        features["hash_mom"] = _safe_pct_change(h30, 90)

    addresses = raw_s.get("adr_act")
    if _enough(addresses):
        features["adr_mom"] = _safe_pct_change(addresses.rolling(30, min_periods=20).mean(), 90)

    transactions = raw_s.get("tx_cnt")
    if _enough(transactions):
        features["tx_mom"] = _safe_pct_change(transactions.rolling(30, min_periods=20).mean(), 90)

    fees = raw_s.get("fee_usd")
    if _enough(fees) and _enough(issuance):
        fee_30 = fees.clip(lower=0).rolling(30, min_periods=20).sum()
        issuance_30 = issuance.clip(lower=0).rolling(30, min_periods=20).sum()
        total_revenue = fee_30 + issuance_30
        features["fee_share"] = (100 * fee_30 / total_revenue.where(total_revenue > 0)).replace([np.inf, -np.inf], np.nan)

    active_supply = raw_s.get("sply_act1y")
    if _enough(active_supply) and _enough(supply):
        total_supply = supply.reindex(active_supply.index)
        features["act_supply"] = (100 * active_supply / total_supply.where(total_supply > 0)).replace([np.inf, -np.inf], np.nan)

    # Apply publication lags only after calculating each indicator, so rolling windows
    # represent observation dates. Every output feature is lagged exactly once.
    output: dict[str, pd.Series] = {}
    for key, values in features.items():
        output[key] = _shift_avail(values, ONCHAIN_LAG_DAYS).rename(key)

    if stable is not None and len(stable):
        stable_series = pd.to_numeric(stable, errors="coerce").copy()
        sidx = pd.DatetimeIndex(pd.to_datetime(stable_series.index, utc=True, errors="coerce"))
        stable_series.index = sidx.tz_convert(None).normalize()
        stable_series = stable_series[~stable_series.index.isna()]
        stable_series = stable_series[~stable_series.index.duplicated(keep="last")].sort_index()
        stable_series = stable_series.where(stable_series > 0)
        stable_avg = stable_series.rolling(7, min_periods=4).mean()
        stable_mom = _safe_pct_change(stable_avg, 90)
        output["stable_mom"] = _shift_avail(stable_mom, STABLE_LAG_DAYS).rename("stable_mom")

    if fng is not None and len(fng):
        fng_series = pd.to_numeric(fng, errors="coerce").copy()
        fidx = pd.DatetimeIndex(pd.to_datetime(fng_series.index, utc=True, errors="coerce"))
        fng_series.index = fidx.tz_convert(None).normalize()
        fng_series = fng_series[~fng_series.index.isna()]
        fng_series = fng_series[~fng_series.index.duplicated(keep="last")].sort_index()
        fng_series = fng_series.where(fng_series.between(0, 100))
        output["fng"] = _shift_avail(fng_series.rolling(7, min_periods=4).mean(), FNG_LAG_DAYS).rename("fng")

    if not output:
        return pd.DataFrame()
    frame = pd.concat(output, axis=1).sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame
