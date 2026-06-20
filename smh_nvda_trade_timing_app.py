# smh_nvda_trade_timing_app.py
# Streamlit Cloud app: Thomas Action Today for SMH + NVDA
# Educational decision support only. Not financial advice.

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Thomas SMH + NVDA Action Today",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background-color:#f7f9fb;}
    .block-container {padding-top:1.0rem; padding-bottom:2rem; max-width:1200px;}
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid #e8eef5;
        padding: 10px 12px;
        border-radius: 14px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.035);
    }
    .action-card {
        background: white;
        border: 1px solid #e5edf5;
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        margin: 10px 0 16px 0;
    }
    .action-title {font-size:1.25rem; font-weight:800; margin-bottom:8px;}
    .action-line {font-size:1.05rem; line-height:1.55;}
    .big-green {background:#eaf7ef; border-left:7px solid #29a35a;}
    .big-yellow {background:#fff8e6; border-left:7px solid #d6a300;}
    .big-red {background:#fdecec; border-left:7px solid #d94848;}
    .big-blue {background:#edf2ff; border-left:7px solid #4b6fff;}
    .small-note {font-size:0.90rem; color:#5b6775;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=300)
def load_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["EMA200"] = out["Close"].ewm(span=200, adjust=False).mean()
    out["RSI14"] = rsi(out["Close"])
    out["ATR14"] = atr(out)
    out["ATR_PCT"] = out["ATR14"] / out["Close"] * 100
    out["HIGH_6M"] = out["Close"].rolling(126, min_periods=30).max()
    out["LOW_3M"] = out["Close"].rolling(63, min_periods=20).min()
    out["DIST_FROM_6M_HIGH"] = (out["Close"] / out["HIGH_6M"] - 1) * 100
    out["DIST_FROM_3M_LOW"] = (out["Close"] / out["LOW_3M"] - 1) * 100
    out["VOL_AVG20"] = out["Volume"].rolling(20).mean()
    out["VOL_RATIO"] = out["Volume"] / out["VOL_AVG20"]
    return out

def dollar_to_shares(dollars: float, price: float) -> int:
    return 0 if price <= 0 or dollars <= 0 else math.floor(dollars / price)

def money(x: float) -> str:
    return f"${x:,.0f}"

def price_fmt(x: float) -> str:
    return f"${x:,.2f}"

@dataclass
class PositionInput:
    ticker: str
    shares: int
    avg_cost: float

@dataclass
class MarketState:
    ticker: str
    price: float
    ema20: float
    ema50: float
    ema200: float
    rsi: float
    atr_pct: float
    dist_high: float
    dist_low: float
    vol_ratio: float


def get_market_state(ticker: str, period: str = "1y"):
    df = add_indicators(load_data(ticker, period=period))
    latest = df.iloc[-1]
    ms = MarketState(
        ticker=ticker,
        price=float(latest["Close"]),
        ema20=float(latest["EMA20"]),
        ema50=float(latest["EMA50"]),
        ema200=float(latest["EMA200"]),
        rsi=float(latest["RSI14"]),
        atr_pct=float(latest["ATR_PCT"]),
        dist_high=float(latest["DIST_FROM_6M_HIGH"]),
        dist_low=float(latest["DIST_FROM_3M_LOW"]),
        vol_ratio=float(latest["VOL_RATIO"]) if not np.isnan(latest["VOL_RATIO"]) else 1.0,
    )
    return df, ms


def analyze_market_regime() -> dict:
    rows = []
    score = 0
    for ticker in ["SPY", "QQQ", "SMH"]:
        try:
            _, ms = get_market_state(ticker, "1y")
            above_50 = ms.price > ms.ema50
            above_200 = ms.price > ms.ema200
            ema50_up = ms.ema50 > ms.ema200
            rsi_ok = ms.rsi >= 45
            ticker_score = int(above_50) + int(above_200) + int(ema50_up) + int(rsi_ok)
            score += ticker_score
            rows.append({
                "Ticker": ticker,
                "Price": price_fmt(ms.price),
                "Above EMA50": "Yes" if above_50 else "No",
                "Above EMA200": "Yes" if above_200 else "No",
                "EMA50 > EMA200": "Yes" if ema50_up else "No",
                "RSI": f"{ms.rsi:.1f}",
                "Score": ticker_score,
            })
        except Exception:
            rows.append({"Ticker": ticker, "Price": "N/A", "Above EMA50": "N/A", "Above EMA200": "N/A", "EMA50 > EMA200": "N/A", "RSI": "N/A", "Score": 0})
    if score >= 10:
        return {"regime": "Bullish", "tone": "big-green", "score": score, "max_score": 12, "buy_modifier": 1.0, "reason": "SPY, QQQ, and SMH trends are broadly healthy. Normal buy rules are allowed.", "rows": rows}
    if score >= 7:
        return {"regime": "Neutral", "tone": "big-yellow", "score": score, "max_score": 12, "buy_modifier": 0.6, "reason": "Market trend is mixed. New buys should be smaller and more selective.", "rows": rows}
    return {"regime": "Defensive", "tone": "big-red", "score": score, "max_score": 12, "buy_modifier": 0.25, "reason": "Market trend is weak. Preserve cash; only tiny starter buys or no buy.", "rows": rows}

def determine_price_level(ms: MarketState) -> tuple[str, str]:
    near_high = ms.dist_high > -3
    mildly_high = -6 < ms.dist_high <= -3
    normal_pullback = -10 <= ms.dist_high <= -6
    good_pullback = -16 <= ms.dist_high < -10
    deep_pullback = ms.dist_high < -16
    weak_trend = ms.price < ms.ema50 or ms.ema20 < ms.ema50 * 0.985
    hot = ms.rsi >= 68
    if weak_trend:
        return "Weak / below trend", "red"
    if near_high and hot:
        return "High price level", "red"
    if near_high:
        return "Near high", "yellow"
    if mildly_high:
        return "Slightly extended", "yellow"
    if normal_pullback:
        return "Normal pullback", "green"
    if good_pullback:
        return "Good pullback", "green"
    if deep_pullback:
        return "Deep pullback", "blue"
    return "Neutral", "blue"

def base_targets(ticker: str, account_value: float, conservative_mode: bool) -> dict:
    if ticker == "SMH":
        return {
            "normal": account_value * 0.60,
            "max": account_value * 0.80,
            "starter_high": account_value * (0.10 if conservative_mode else 0.20),
            "starter_near": account_value * (0.15 if conservative_mode else 0.25),
        }
    return {
        "normal": account_value * 0.20,
        "max": account_value * 0.25,
        "starter_high": 0,
        "starter_near": account_value * (0.03 if conservative_mode else 0.05),
    }

def action_model(ms: MarketState, pos: PositionInput, account_value: float, cash_available: float, conservative_mode: bool, market_regime: dict) -> dict:
    current_value = pos.shares * ms.price
    has_position = pos.shares > 0
    level, tone = determine_price_level(ms)
    targets = base_targets(ms.ticker, account_value, conservative_mode)
    weak = level == "Weak / below trend"
    hot = ms.rsi >= 68

    action, reason, priority, buy_dollars = "WATCH", "No clear advantage now.", 3, 0.0

    if ms.ticker == "SMH":
        if not has_position:
            if level == "High price level":
                action, buy_dollars, priority = "SMALL STARTER ONLY", min(targets["starter_high"], cash_available), 2
                reason = "No SMH position, but price is high. Use tiny starter only; preserve cash."
            elif level in ["Near high", "Slightly extended"]:
                action, buy_dollars, priority = "START SMALL, DO NOT CHASE", min(targets["starter_near"], cash_available), 2
                reason = "Trend is positive, but price is not cheap. Keep cash for 5%-15% pullback."
            elif level == "Normal pullback":
                action, buy_dollars, priority = "RATIONAL FIRST BUY", min(account_value * 0.25, cash_available), 1
                reason = "SMH pulled back while trend remains acceptable. Better first-entry zone."
            elif level == "Good pullback":
                action, buy_dollars, priority = "STRONG FIRST BUY", min(account_value * 0.35, cash_available), 1
                reason = "Meaningful pullback with acceptable trend; better risk/reward."
            elif level == "Deep pullback":
                if ms.price > ms.ema20:
                    action, buy_dollars, priority = "BUY AFTER REBOUND", min(account_value * 0.20, cash_available), 2
                    reason = "Deep pullback stabilized above EMA20. Buy smaller and watch risk."
                else:
                    action, priority = "WAIT FOR REBOUND", 3
                    reason = "Deep pullback but no rebound confirmation."
            elif weak:
                action, reason, priority = "WAIT", "Trend is weak. Do not build first position.", 3
        else:
            target_gap = max(0, targets["normal"] - current_value)
            if weak:
                action, reason, priority = "DO NOT ADD", "SMH trend weak. Hold or reduce if risk tolerance is low.", 3
            elif level in ["Normal pullback", "Good pullback"]:
                action, buy_dollars, priority = "ADD TO CORE", min(account_value * 0.15, cash_available, target_gap), 1
                reason = "Add only on pullback while trend remains acceptable."
            elif level in ["High price level", "Near high"] and current_value > targets["normal"] * 1.15:
                action, reason, priority = "HOLD / CONSIDER TRIM", "Position is large and price is high. Avoid adding.", 2
            else:
                action, reason, priority = "HOLD", "Core position is okay. No need to trade.", 3
    else:
        if not has_position:
            if level == "High price level":
                action, reason, priority = "DO NOT CHASE", "NVDA is tactical and volatile. Avoid chasing high price.", 3
            elif level in ["Near high", "Slightly extended"]:
                action, buy_dollars, priority = "WATCH / TINY STARTER", min(targets["starter_near"], cash_available), 2
                reason = "Only tiny exposure is rational near highs."
            elif level == "Normal pullback":
                action, buy_dollars, priority = "TACTICAL BUY", min(account_value * 0.08, cash_available), 1
                reason = "Pullback with trend intact; use smaller size than SMH."
            elif level == "Good pullback":
                action, buy_dollars, priority = "TACTICAL BUY", min(account_value * 0.12, cash_available), 1
                reason = "Better tactical entry, but still smaller than SMH."
            elif level == "Deep pullback":
                if ms.price > ms.ema20:
                    action, buy_dollars, priority = "SMALL BUY AFTER REVERSAL", min(account_value * 0.05, cash_available), 2
                    reason = "Deep pullback with rebound confirmation. Keep size small."
                else:
                    action, reason, priority = "WAIT FOR REVERSAL", "No tactical buy until stabilization.", 3
            elif weak:
                action, reason, priority = "WAIT", "Trend is weak. Do not buy NVDA tactically.", 3
        else:
            target_gap = max(0, targets["normal"] - current_value)
            if weak:
                action, reason, priority = "STOP / REDUCE", "NVDA tactical trend weak. Respect stop-loss.", 1
            elif level == "High price level" and hot:
                action, reason, priority = "TAKE PROFIT / TRIM", "NVDA is hot near high. Lock tactical profit.", 1
            elif current_value < targets["normal"] and level in ["Normal pullback", "Good pullback"]:
                action, buy_dollars, priority = "ADD SMALL", min(account_value * 0.05, cash_available, target_gap), 2
                reason = "Small tactical add only."
            else:
                action, reason, priority = "HOLD", "No immediate NVDA action.", 3

    original_buy_dollars = buy_dollars
    buy_dollars = buy_dollars * market_regime["buy_modifier"]
    if market_regime["regime"] == "Defensive" and original_buy_dollars > 0:
        reason += " Market Regime is Defensive, so buy size is sharply reduced."
        if action not in ["WAIT", "DO NOT CHASE"]:
            action = "DEFENSIVE: SMALL ONLY"
    elif market_regime["regime"] == "Neutral" and original_buy_dollars > 0:
        reason += " Market Regime is Neutral, so buy size is reduced."

    buy_shares = dollar_to_shares(buy_dollars, ms.price)
    real_buy_dollars = buy_shares * ms.price

    trim_shares, trim_action, trim_reason = 0, "No trim", "No trim now."
    if has_position:
        if ms.ticker == "SMH" and current_value > targets["normal"] * 1.20 and level in ["High price level", "Near high"] and hot:
            trim_shares, trim_action, trim_reason = math.floor(pos.shares * 0.20), "Trim 20%", "SMH is overweight and price level is hot."
        elif ms.ticker == "NVDA":
            pnl_pct = 0 if pos.avg_cost <= 0 else (ms.price / pos.avg_cost - 1) * 100
            if pnl_pct >= 10:
                trim_shares, trim_action, trim_reason = math.ceil(pos.shares * 0.50), "Sell 50%", "NVDA tactical gain reached +10% or more."
            elif weak:
                trim_shares, trim_action, trim_reason = pos.shares, "Exit / stop", "NVDA tactical trend weakened."

    return {
        "level": level, "tone": tone, "action": action, "reason": reason, "priority": priority,
        "buy_dollars": real_buy_dollars, "buy_shares": buy_shares,
        "current_value": current_value, "normal_target": targets["normal"], "max_position": targets["max"],
        "trim_action": trim_action, "trim_shares": trim_shares, "trim_value": trim_shares * ms.price, "trim_reason": trim_reason,
    }

def levels_for_next_actions(ms: MarketState, pos: PositionInput):
    anchor = pos.avg_cost if pos.shares > 0 and pos.avg_cost > 0 else ms.price
    if ms.ticker == "SMH":
        buy_levels = [("Starter / no chase zone", anchor * 0.97, "Small buy only"), ("Add zone 1", anchor * 0.95, "First add"), ("Add zone 2", anchor * 0.90, "Second add"), ("Add zone 3", anchor * 0.85, "Third add / only if trend stabilizes")]
        sell_levels = [("Trim 20%", anchor * 1.15, 0.20), ("Trim 20%", anchor * 1.25, 0.20), ("Trim 20%", anchor * 1.35, 0.20)]
        stop = anchor * 0.88
    else:
        buy_levels = [("Tiny starter zone", anchor * 0.97, "Tiny only"), ("Tactical buy zone", anchor * 0.95, "Small tactical buy"), ("Better tactical zone", anchor * 0.90, "Better risk/reward")]
        sell_levels = [("Sell 50%", anchor * 1.10, 0.50), ("Sell remaining / trim", anchor * 1.20, 0.50)]
        stop = anchor * 0.93
    buy_df = pd.DataFrame([{"Next Buy Level": n, "Price": price_fmt(p), "Use": u} for n, p, u in buy_levels])
    sell_df = pd.DataFrame([{"Sell Target": n, "Price": price_fmt(p), "Sell Shares": math.floor(pos.shares * pct), "Approx Value": money(math.floor(pos.shares * pct) * p)} for n, p, pct in sell_levels])
    return buy_df, sell_df, stop

def chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", mode="lines"))
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=32, b=10), title=f"{ticker} price level", xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig

# Sidebar
st.sidebar.header("Account Settings")
account_value = st.sidebar.number_input("Total short-term account value ($)", min_value=1000.0, value=10000.0, step=500.0)
cash_available = st.sidebar.number_input("Current cash available ($)", min_value=0.0, value=10000.0, step=100.0)
conservative_mode = st.sidebar.checkbox("Conservative new-position mode", value=True, help="Use smaller starter positions when SMH/NVDA are near highs.")
st.sidebar.subheader("Position Mode")
no_smh = st.sidebar.checkbox("I do NOT own SMH", value=True)
no_nvda = st.sidebar.checkbox("I do NOT own NVDA", value=True)
st.sidebar.subheader("Current Positions")
smh_shares = 0 if no_smh else st.sidebar.number_input("SMH shares", min_value=0, value=0, step=1)
smh_avg_cost = 0.0 if no_smh else st.sidebar.number_input("SMH average cost ($)", min_value=0.0, value=0.0, step=1.0)
nvda_shares = 0 if no_nvda else st.sidebar.number_input("NVDA shares", min_value=0, value=0, step=1)
nvda_avg_cost = 0.0 if no_nvda else st.sidebar.number_input("NVDA average cost ($)", min_value=0.0, value=0.0, step=1.0)
period = st.sidebar.selectbox("Chart period", ["3mo", "6mo", "1y"], index=2)
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

st.title("📈 Thomas Action Today: SMH + NVDA")
st.caption("Current price level + Market Regime → action → buy/sell amount and shares.")

market_regime = analyze_market_regime()
st.markdown(f"""
<div class="action-card {market_regime['tone']}"><div class="action-title">Market Regime: {market_regime['regime']} ({market_regime['score']}/{market_regime['max_score']})</div>
<div class="action-line"><b>Effect on new buys:</b> {market_regime['buy_modifier']:.0%} of normal buy size<br>
<b>Reason:</b> {market_regime['reason']}</div></div>
""", unsafe_allow_html=True)
with st.expander("Market Regime details: SPY / QQQ / SMH", expanded=False):
    st.dataframe(pd.DataFrame(market_regime["rows"]), use_container_width=True, hide_index=True)

st.markdown("""
<div class="action-card big-blue"><div class="action-title">Default Philosophy</div>
<div class="action-line">Normal target is <b>SMH 60% / NVDA 20% / Cash 20%</b>, but the app does not blindly buy that allocation. Market Regime and current price level control actual buy size.</div></div>
""", unsafe_allow_html=True)

positions = {"SMH": PositionInput("SMH", int(smh_shares), float(smh_avg_cost)), "NVDA": PositionInput("NVDA", int(nvda_shares), float(nvda_avg_cost))}
summaries, detail_data = [], {}

for ticker, pos in positions.items():
    try:
        df, ms = get_market_state(ticker, period)
        plan = action_model(ms, pos, account_value, cash_available, conservative_mode, market_regime)
        buy_df, sell_df, stop_ref = levels_for_next_actions(ms, pos)
        summaries.append({"Ticker": ticker, "Price Level": plan["level"], "Action": plan["action"], "Buy Now": money(plan["buy_dollars"]), "Buy Shares": plan["buy_shares"], "Trim Action": plan["trim_action"], "Trim Shares": plan["trim_shares"], "Stop Ref": price_fmt(stop_ref), "Priority": plan["priority"]})
        detail_data[ticker] = (df, ms, pos, plan, buy_df, sell_df, stop_ref)
    except Exception as e:
        st.error(f"Could not analyze {ticker}: {e}")

if summaries:
    action_df = pd.DataFrame(summaries).sort_values(["Priority", "Ticker"])
    top = action_df.iloc[0]
    tone_class = "big-green"
    if any(w in str(top["Action"]) for w in ["WAIT", "CHASE", "STOP", "DEFENSIVE"]):
        tone_class = "big-red" if market_regime["regime"] == "Defensive" else "big-yellow"
    elif any(w in str(top["Action"]) for w in ["START", "WATCH"]):
        tone_class = "big-yellow"
    st.markdown(f"""
    <div class="action-card {tone_class}"><div class="action-title">Thomas Action Today</div>
    <div class="action-line"><b>Market Regime:</b> {market_regime['regime']}<br><b>First priority:</b> {top['Ticker']} — {top['Price Level']} → <b>{top['Action']}</b><br>
    <b>Buy now:</b> {top['Buy Now']} / {top['Buy Shares']} shares<br>
    <b>Trim now:</b> {top['Trim Action']} / {top['Trim Shares']} shares<br>
    <b>Stop reference:</b> {top['Stop Ref']}</div></div>
    """, unsafe_allow_html=True)
    st.subheader("Action Summary")
    st.dataframe(action_df.drop(columns=["Priority"]), use_container_width=True, hide_index=True)

for ticker in ["SMH", "NVDA"]:
    if ticker not in detail_data:
        continue
    df, ms, pos, plan, buy_df, sell_df, stop_ref = detail_data[ticker]
    css = {"green": "big-green", "yellow": "big-yellow", "red": "big-red", "blue": "big-blue"}.get(plan["tone"], "big-blue")
    st.markdown(f"## {ticker}")
    st.markdown(f"""
    <div class="action-card {css}"><div class="action-title">{ticker}: {plan['level']} → {plan['action']}</div>
    <div class="action-line"><b>Recommended buy now:</b> {money(plan['buy_dollars'])} / {plan['buy_shares']} shares<br>
    <b>Recommended trim now:</b> {plan['trim_action']} — {plan['trim_shares']} shares ≈ {money(plan['trim_value'])}<br>
    <b>Reason:</b> {plan['reason']}<br><span class="small-note">Trim reason: {plan['trim_reason']}</span></div></div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", price_fmt(ms.price)); c2.metric("RSI14", f"{ms.rsi:.1f}"); c3.metric("ATR %", f"{ms.atr_pct:.2f}%"); c4.metric("From 6M High", f"{ms.dist_high:.1f}%")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Current Value", money(plan["current_value"])); c6.metric("Normal Target", money(plan["normal_target"])); c7.metric("Max Position", money(plan["max_position"])); c8.metric("Stop Reference", price_fmt(stop_ref))
    with st.expander(f"{ticker} next buy/sell levels", expanded=True):
        st.markdown("### Next Buy Levels"); st.dataframe(buy_df, use_container_width=True, hide_index=True)
        st.markdown("### Sell / Trim Targets"); st.dataframe(sell_df, use_container_width=True, hide_index=True)
    st.plotly_chart(chart(df.tail(130), ticker), use_container_width=True, key=f"chart_{ticker}")

st.markdown("---")
st.markdown("""
### How to use this app on phone
1. Enter total short-term account value.
2. Enter actual cash available.
3. Check “I do NOT own SMH/NVDA” if you have no position.
4. Read **Market Regime** first.
5. Read **Thomas Action Today** second.
6. If Market Regime is Defensive, buy size is automatically reduced.
7. Then check SMH and NVDA details.

Educational decision-support only. It does not place trades and does not guarantee profit.
""")
