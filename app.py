"""
Silver Price Direction Predictor (Streamlit)

Run:
    pip install streamlit yfinance pandas numpy scikit-learn plotly
    # optional, for the deep learning model:
    pip install tensorflow
    streamlit run silver_app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Silver Direction Predictor", page_icon="🥈", layout="wide")

# Assets the documents say move together with silver
TICKERS = {
    "Silver": "SI=F",
    "Gold": "GC=F",
    "Platinum": "PL=F",
    "Copper": "HG=F",
    "Oil": "CL=F",
    "Dollar": "DX-Y.NYB",
    "SP500": "^GSPC",
}


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(start: str) -> pd.DataFrame:
    raw = yf.download(list(TICKERS.values()), start=start, auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns={v: k for k, v in TICKERS.items()})
    raw = raw[[c for c in TICKERS if c in raw.columns]]
    return raw.ffill().dropna()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)


def make_features(df: pd.DataFrame, horizon: int):
    X = pd.DataFrame(index=df.index)
    for c in df.columns:
        for k in (1, 5, 20):
            X[f"{c}_ret{k}d"] = df[c].pct_change(k)
    s = df["Silver"]
    X["rsi14"] = rsi(s)
    X["gap_ma20"] = s / s.rolling(20).mean() - 1
    X["gap_ma50"] = s / s.rolling(50).mean() - 1
    X["vol20"] = s.pct_change().rolling(20).std()
    X["gold_silver_ratio"] = df["Gold"] / s if "Gold" in df else np.nan
    X["weekday"] = df.index.dayofweek
    future = s.shift(-horizon)
    y = (future > s).astype(float)
    y[future.isna()] = np.nan  # last `horizon` days have no answer yet
    return X, y


# ---------------------------------------------------------------- models
def train_trees(name, Xtr, ytr, Xte):
    if name == "Random Forest":
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=0)
    else:
        m = GradientBoostingClassifier(random_state=0)
    m.fit(Xtr, ytr)
    return m, m.predict_proba(Xte)[:, 1]


def make_windows(X: np.ndarray, y: np.ndarray, idx_positions, window: int):
    xs, ys = [], []
    for p in idx_positions:
        if p - window + 1 < 0:
            continue
        xs.append(X[p - window + 1 : p + 1])
        ys.append(y[p])
    return np.array(xs, dtype="float32"), np.array(ys, dtype="float32")


def build_tcn_lstm(window, n_feat):
    """TCN-style dilated causal convolutions followed by an LSTM."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(window, n_feat))
    x = inp
    for d in (1, 2, 4):  # dilated causal conv blocks with residual links
        h = layers.Conv1D(32, 3, padding="causal", dilation_rate=d, activation="relu")(x)
        h = layers.SpatialDropout1D(0.2)(h)
        if x.shape[-1] != 32:
            x = layers.Conv1D(32, 1)(x)
        x = layers.Add()([x, h])
    x = layers.LSTM(32, dropout=0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


# ---------------------------------------------------------------- UI
st.title("🥈 Silver Price Direction Predictor")
st.caption("Will silver be higher or lower in N trading days? A hands-on version of the research summary.")

with st.sidebar:
    st.header("Settings")
    start = st.text_input("Data start date", "2012-01-01")
    horizon = st.slider("Predict direction N trading days ahead", 1, 60, 20)
    model_name = st.selectbox(
        "Model",
        ["Random Forest", "Gradient Boosting", "TCN-LSTM (deep learning, needs tensorflow)"],
    )
    test_frac = st.slider("Share of data held back for testing", 0.1, 0.4, 0.2, 0.05)
    st.subheader("Trading filter")
    conf = st.slider("Only go long if P(up) is above", 0.50, 0.80, 0.55, 0.01)
    use_rsi = st.checkbox("Also require RSI below 70 (avoid overbought)", True)
    run = st.button("Train & predict", type="primary", use_container_width=True)

tab_guide, tab_res, tab_how = st.tabs(["📖 Plain-English guide", "📊 Results", "🛠 How to use it"])

with tab_guide:
    st.markdown(
        """
### What the documents are saying

**1. Silver's direction is influenced by a few things**
- **US Dollar**: stronger dollar → silver usually falls; weaker dollar → silver usually rises.
- **Interest rates**: Fed rate hikes push silver down at first.
- **Economic growth**: silver is also used in industry, so growth tends to lift it.
- **Gold, platinum, copper, oil** move together with silver.

**2. Models try to learn those patterns**
- *Tree models* (Random Forest / XGBoost): lots of "if this, then that" rules voted together. Simple and robust.
- *Deep learning (TCN-LSTM)*: looks at the last 60 days as a sequence and finds patterns over time. More complex, needs more tuning.
- *ARIMA*: the old-school baseline.

**3. A filter makes trading signals safer**
Only act when the model is confident, and skip trades when RSI says silver is already overbought.

### ⚠️ Honest reality check
The documents quote accuracy of 85-90%, correlations of 99.9%, and 115% returns. **Treat those numbers with
suspicion.** Such results often come from "leakage" (the model accidentally sees the future) or from
overlapping data. This app avoids that by testing only on later dates, with a gap of N days between training and
testing. **Expect something close to 50-60%.** If you see 55%, that is already meaningful. This is
an educational tool, not financial advice.
"""
    )

if run:
    try:
        with st.spinner("Downloading market data..."):
            df = load_data(start)
    except Exception as e:
        st.error(f"Could not download data: {e}")
        st.stop()

    X, y = make_features(df, horizon)
    valid = X.dropna().index.intersection(y.dropna().index)
    Xv, yv = X.loc[valid], y.loc[valid]
    cut = int(len(Xv) * (1 - test_frac))
    gap = horizon  # purge overlap so training labels never peek into the test period
    Xtr, ytr = Xv.iloc[: cut - gap], yv.iloc[: cut - gap]
    Xte, yte = Xv.iloc[cut:], yv.iloc[cut:]
    latest = X.iloc[[-1]].ffill()

    with st.spinner("Training model..."):
        if model_name.startswith("TCN"):
            try:
                from tensorflow import keras  # noqa: F401
            except ImportError:
                st.error("TensorFlow isn't installed. Run `pip install tensorflow` or pick a tree model.")
                st.stop()
            window = 60
            full = X.ffill().bfill()
            mu, sd = full.iloc[: cut - gap].mean(), full.iloc[: cut - gap].std().replace(0, 1)
            Z = ((full - mu) / sd).values
            yfull = y.reindex(full.index).fillna(0).values
            pos_all = {d: i for i, d in enumerate(full.index)}
            tr_pos = [pos_all[d] for d in Xtr.index]
            te_pos = [pos_all[d] for d in Xte.index]
            Wtr, Ltr = make_windows(Z, yfull, tr_pos, window)
            Wte, _ = make_windows(Z, yfull, te_pos, window)
            net = build_tcn_lstm(window, Z.shape[1])
            net.fit(Wtr, Ltr, epochs=15, batch_size=64, validation_split=0.1, verbose=0)
            proba = net.predict(Wte, verbose=0).ravel()
            Xte, yte = Xte.iloc[-len(proba):], yte.iloc[-len(proba):]
            p_now = float(net.predict(Z[-window:][None], verbose=0)[0, 0])
            importances = None
        else:
            model, proba = train_trees(model_name, Xtr, ytr, Xte)
            p_now = float(model.predict_proba(latest)[0, 1])
            importances = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False)

    pred = (proba > 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    base = max(yte.mean(), 1 - yte.mean())  # accuracy of always guessing the most common outcome

    # Simple backtest: each day, hold silver if the filter says "go", else stay in cash
    s = df["Silver"]
    next_ret = s.pct_change().shift(-1).reindex(Xte.index).fillna(0)
    go_long = pd.Series(proba > conf, index=Xte.index)
    if use_rsi:
        go_long &= Xte["rsi14"] < 70
    strat = (1 + next_ret * go_long.astype(float)).cumprod()
    hold = (1 + next_ret).cumprod()

    with tab_res:
        st.subheader(f"Latest signal ({df.index[-1].date()})")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"P(silver higher in {horizon} days)", f"{p_now:.0%}")
        rsi_now = float(latest["rsi14"].iloc[0])
        action = "🟢 Lean UP" if p_now > conf and (not use_rsi or rsi_now < 70) else (
            "🔴 Lean DOWN / stay out" if p_now < 1 - conf + 0.0 or p_now < 0.5 else "⚪ No clear signal")
        c2.metric("Signal", action)
        c3.metric("RSI (14)", f"{rsi_now:.0f}")

        st.subheader("How well did it do on data it had never seen?")
        a, b, c = st.columns(3)
        a.metric("Model direction accuracy", f"{acc:.1%}")
        b.metric("Naive guess (always same answer)", f"{base:.1%}")
        c.metric("Edge over naive guess", f"{(acc - base) * 100:+.1f} pts")
        if acc - base < 0.02:
            st.info("The model barely beats a naive guess. That's normal for markets, and it's the honest result.")

        fig = go.Figure()
        fig.add_scatter(x=hold.index, y=hold, name="Buy & hold silver")
        fig.add_scatter(x=strat.index, y=strat, name="Model strategy (with filter)")
        fig.update_layout(title="Test-period growth of 1 unit (no fees or slippage)", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Strategy was in the market {go_long.mean():.0%} of days.")

        st.subheader("Silver price")
        st.line_chart(s)

        if importances is not None:
            st.subheader("What the model paid most attention to")
            st.bar_chart(importances.head(12))

    with tab_how:
        st.markdown(
            f"""
### How to use this model, step by step
1. **Pick a horizon** (default 20 days, as in the research) and click **Train & predict**.
2. **Check the accuracy first.** If it isn't above the "naive guess", don't trust the signal.
3. **Read the latest signal.** Currently P(up in {horizon} days) = **{p_now:.0%}**.
   Above {conf:.0%} with RSI < 70 → lean up. Weak or mixed → do nothing.
4. **Retrain regularly** (weekly or monthly) because market behaviour changes.
5. **Never use it alone.** Combine it with news, the dollar trend and rate decisions, and
   size positions so that a wrong call is affordable.
6. To improve it: add more features (inflation expectations, real interest rates), tune hyperparameters, and
   compare Random Forest vs TCN-LSTM on the same test period.
"""
        )
else:
    with tab_res:
        st.info("Set your options in the sidebar and click **Train & predict**.")
    with tab_how:
        st.info("Run the model first. This tab will then show how to read the result.")
