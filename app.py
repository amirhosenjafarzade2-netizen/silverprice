"""
Macro Environment Analyzer: entry point.
Choose an asset; each asset has its own <asset>_main.py (silver and oil exist so far).
Run with:  streamlit run app.py
"""
import os
import runpy
import sys

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# st.set_page_config must be the first Streamlit call, so it lives here.
# (Silver title / icon for now; the oil page sets its own title inside oil_main.py.)
st.set_page_config(page_title="Silver Macro Environment Analyzer", page_icon="🥈", layout="wide")

ASSETS = {
    "Gold": "gold_main.py",
    "Silver": "silver_main.py",
    "Oil": "oil_main.py",
    "Dollar": "dollar_main.py",
    "Bitcoin": "bitcoin_main.py",
    "S&P 500": "sp500_main.py",
}

with st.sidebar:
    asset = st.selectbox("Choose the asset to analyze", list(ASSETS), index=list(ASSETS).index("Silver"),
                         key="asset_choice")
    st.divider()

path = os.path.join(HERE, ASSETS[asset])
if not os.path.exists(path):
    st.title(f"{asset} Macro Environment Analyzer")
    st.info(f"The {asset} analyzer is not built yet (it will load from `{ASSETS[asset]}`). Silver and Oil are available right now.")
    st.stop()

runpy.run_path(path, run_name="__main__")
