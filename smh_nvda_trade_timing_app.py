# smh_nvda_trade_timing_app.py
# Streamlit cloud app: SMH + NVDA current price level analysis and trade sizing
# Educational decision support only. Not financial advice.

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="SMH + NVDA Trade Timing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background-color: #f7f9fb;}
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid #e8eef5;
        padding: 14px 16px;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 10px;
    }
    .big-signal {
        font-size: 1.4rem;
        font-weight: 800;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 8px 0 12px 0;
    }
    .buy {background: #eaf7ef; border: 1px solid #b9e3c5;}
    .watch {background: #fff8e6; border: 1px solid #f3d37a;}
    .sell {background: #fdecec; border: 1px solid #edb6b6;}
    .hold {background: #edf2ff; border: 1px solid #bccbff;}
    .small-note {font-size: 0.88rem; color: #5b6775;}
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
    out["RSI14"] = rsi(out["Close"])
    out["ATR14"] = atr(out)
    out["ATR_PCT"] = out["ATR14"] / out["Close"] * 100
    out["HIGH_6M"] = out["Close"].rolling(126, min_periods=30).max()
    out["DIST_FROM_6M_HIGH"] = (out["Close"] / out["HIGH_6M"] - 1) * 100
    out["LOW_3M"] = out["Close"].rolling(63, min_periods=20).min()
    out["DIST_FROM_3M_LOW"] = (out["Close"] / out["LOW_3M"] - 1) * 100
    out["VOL_AVG20"] = out["Volume"].rolling(20).mean()
    return out


def dollar_to_shares(dollars: float, price: float) -> int:
    if price <= 0:
        return 0
    return max(0, math.floor(dollars / price))


def current_level_plan(ticker, price, ema20, ema50, rsi14, atr_pct, dist_high, current_value, account_value, cash):
    """
    Converts price level into rational action and position size.
    Main rule: do not blindly allocate 60/20 when price is near high.
    """
    has_position = current_value > 0
    near_high = dist_high > -3
    mild_pullback = -8 <= dist_high <= -3
    good_pullback = -15 <= dist_high < -8
    deep_pullback = dist_high < -15
    trend_ok = price > ema50 and ema20 >= ema50 * 0.995
    strong_trend = price > ema20 and ema20 > ema50
    hot = rsi14 >= 68
    weak = price < ema50 or ema20 < ema50 * 0.98

    if ticker == "SMH":
        max_position = account_value * 0.80
        normal_target = account_value * 0.60

        if weak:
            action = "WAIT"
            buy_pct = 0.00
            level = "Weak / below trend"
            reason = "SMH is below key trend area. Do not build a large new position."
        elif near_high and hot:
            action = "NO BIG BUY"
            buy_pct = 0.00 if has_position else 0.10
            level = "High price level"
            reason = "SMH is near recent high and RSI is hot. Only tiny starter if you have no position."
        elif near_high:
            action = "SMALL STARTER ONLY"
            buy_pct = 0.00 if has_position else 0.20
            level = "Near high"
            reason = "Trend is good, but price is not cheap. Keep cash for a 5%-15% pullback."
        elif mild_pullback and trend_ok:
            action = "BUY / ADD"
            buy_pct = 0.20
            level = "Normal pullback"
            reason = "This is a reasonable add zone after a mild pullback."
        elif good_pullback and trend_ok:
            action = "STRONGER BUY / ADD"
            buy_pct = 0.25
            level = "Good pullback"
            reason = "Better risk/reward: meaningful pullback while trend remains intact."
        elif deep_pullback:
            action = "BUY ONLY AFTER REBOUND"
            buy_pct = 0.10 if price > ema20 else 0.00
            level = "Deep pullback"
            reason = "Deep pullback can be opportunity, but wait for stabilization."
        else:
            action = "HOLD / WATCH"
            buy_pct = 0.00
            level = "Neutral"
            reason = "No clear advantage now."

    else:  # NVDA
        max_position = account_value * 0.25
        normal_target = account_value * 0.20

        if weak:
            action = "WAIT / STOP IF HELD"
            buy_pct = 0.00
            level = "Weak tactical setup"
            reason = "NVDA is tactical. Avoid adding when trend weakens."
        elif near_high and hot:
            action = "DO NOT CHASE"
            buy_pct = 0.00
            level = "High price level"
            reason = "NVDA near high with hot RSI has poor risk/reward."
        elif near_high:
            action = "WATCH ONLY"
            buy_pct = 0.00 if has_position else 0.05
            level = "Near high"
            reason = "Only tiny exposure is rational near highs."
        elif mild_pullback and trend_ok:
            action = "TACTICAL BUY"
            buy_pct = 0.10
            level = "Normal pullback"
            reason = "Pullback while trend intact. Use smaller size than SMH."
        elif good_pullback and trend_ok:
            action = "TACTICAL BUY / ADD"
            buy_pct = 0.15
            level = "Good pullback"
            reason = "Better entry zone, but keep NVDA smaller than SMH."
        elif deep_pullback:
            action = "ONLY AFTER REVERSAL"
            buy_pct = 0.05 if price > ema20 else 0.00
            level = "Deep pullback"
            reason = "NVDA deep pullbacks can continue. Wait for rebound."
        else:
            action = "HOLD / WATCH"
            buy_pct = 0.00
            level = "Neutral"
            reason = "No clear tactical advantage now."

    desired_buy = account_value * buy_pct
    target_gap = max(0, max_position - current_value)
    buy_dollars = min(desired_buy, cash, target_gap)
    buy_shares = dollar_to_shares(buy_dollars, price)

    # Sell / trim logic
    trim_action = "No trim"
    trim_shares = 0
    trim_value = 0
    if has_position:
        if ticker == "SMH" and current_value > normal_target * 1.25 and near_high and hot:
            trim_action = "Trim 20% because SMH is overweight and hot"
            trim_shares = math.floor((current_value / price) * 0.20)
        elif ticker == "NVDA" and (near_high and hot or current_value > normal_target * 1.25):
            trim_action = "Trim 50% because NVDA is tactical and hot/overweight"
            trim_shares = math.floor((current_value / price) * 0.50)
        trim_value = trim_shares * price

    return {
        "level": level,
        "action": action,
        "reason": reason,
        "buy_dollars": buy_dollars,
        "buy_shares": buy_shares,
        "max_position": max_position,
        "normal_target": normal_target,
        "trim_action": trim_action,
        "trim_shares": trim_shares,
        "trim_value": trim_value,
        "strong_trend": strong_trend,
    }


def stop_and_targets(ticker, price, avg_cost, shares):
    if avg_cost > 0 and shares > 0:
        base = avg_cost
    else:
        base = price

    if ticker == "SMH":
        stop = base * 0.88
        sells = [
            ("Trim 20%", base * 1.15, 0.20),
            ("Trim 20%", base * 1.25, 0.20),
            ("Trim 20%", base * 1.35, 0.20),
        ]
        pullback_buys = [
            ("Add zone 1", base * 0.95),
            ("Add zone 2", base * 0.90),
            ("Add zone 3", base * 0.85),
        ]
    else:
        stop = base * 0.93
        sells = [
            ("Sell 50%", base * 1.10, 0.50),
            ("Sell remaining", base * 1.20, 0.50),
        ]
        pullback_buys = [
            ("Tactical buy zone", base * 0.95),
            ("Better tactical zone", base * 0.90),
        ]

    return stop, sells, pullback_buys


def make_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", mode="lines"))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        title=f"{ticker}: Current price level with EMA20 / EMA50",
        xaxis_rangeslider_visible=False,
    )
    return fig


st.sidebar.header("Account Settings")
account_value = st.sidebar.number_input("Total short-term account value ($)", min_value=1000.0, value=10000.0, step=500.0)
cash_available = st.sidebar.number_input("Current cash available ($)", min_value=0.0, value=6000.0, step=100.0)

st.sidebar.subheader("Current Positions")
smh_shares = st.sidebar.number_input("SMH shares", min_value=0, value=0, step=1)
smh_avg_cost = st.sidebar.number_input("SMH average cost ($)", min_value=0.0, value=0.0, step=1.0)
nvda_shares = st.sidebar.number_input("NVDA shares", min_value=0, value=0, step=1)
nvda_avg_cost = st.sidebar.number_input("NVDA average cost ($)", min_value=0.0, value=0.0, step=1.0)

period = st.sidebar.selectbox("Chart period", ["3mo", "6mo", "1y"], index=2)
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

st.title("📈 SMH + NVDA Price-Level Trade Timing")
st.caption("Analyzes whether current price is high, normal pullback, good pullback, or weak trend. Then recommends action and trade size.")

positions = {
    "SMH": {"shares": smh_shares, "avg_cost": smh_avg_cost},
    "NVDA": {"shares": nvda_shares, "avg_cost": nvda_avg_cost},
}

summary = []

for ticker, pos in positions.items():
    try:
        df = add_indicators(load_data(ticker, period=period))
        latest = df.iloc[-1]
        price = float(latest["Close"])
        ema20 = float(latest["EMA20"])
        ema50 = float(latest["EMA50"])
        rsi14 = float(latest["RSI14"])
        atr_pct = float(latest["ATR_PCT"])
        dist_high = float(latest["DIST_FROM_6M_HIGH"])
        current_value = pos["shares"] * price

        plan = current_level_plan(
            ticker=ticker,
            price=price,
            ema20=ema20,
            ema50=ema50,
            rsi14=rsi14,
            atr_pct=atr_pct,
            dist_high=dist_high,
            current_value=current_value,
            account_value=account_value,
            cash=cash_available,
        )

        stop, sell_targets, pullback_buys = stop_and_targets(ticker, price, pos["avg_cost"], pos["shares"])

        css = "buy" if "BUY" in plan["action"] else "sell" if "WAIT" in plan["action"] or "CHASE" in plan["action"] else "watch"

        st.markdown(f"## {ticker}")
        st.markdown(
            f"<div class='big-signal {css}'>{plan['level']} → {plan['action']}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class='metric-card'>"
            f"<b>Recommended buy now:</b> ${plan['buy_dollars']:,.0f} / {plan['buy_shares']} shares<br>"
            f"<b>Recommended sell/trim now:</b> {plan['trim_action']} "
            f"({plan['trim_shares']} shares ≈ ${plan['trim_value']:,.0f})<br>"
            f"<span class='small-note'>{plan['reason']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${price:,.2f}")
        c2.metric("RSI14", f"{rsi14:.1f}")
        c3.metric("ATR %", f"{atr_pct:.2f}%")
        c4.metric("From 6M High", f"{dist_high:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Current Value", f"${current_value:,.0f}")
        c6.metric("Normal Target", f"${plan['normal_target']:,.0f}")
        c7.metric("Max Position", f"${plan['max_position']:,.0f}")
        c8.metric("Stop Reference", f"${stop:,.2f}")

        with st.expander(f"{ticker} detailed buy/sell levels", expanded=True):
            buy_rows = []
            for name, level_price in pullback_buys:
                buy_rows.append({
                    "Buy Zone": name,
                    "Trigger Price": f"${level_price:,.2f}",
                    "Use": "Wait for this level if current price is high"
                })
            st.markdown("### Pullback Buy Zones")
            st.dataframe(pd.DataFrame(buy_rows), use_container_width=True, hide_index=True)

            sell_rows = []
            for name, target_price, pct in sell_targets:
                shares_to_sell = math.floor(pos["shares"] * pct)
                sell_rows.append({
                    "Sell Plan": name,
                    "Target Price": f"${target_price:,.2f}",
                    "Sell Shares": shares_to_sell,
                    "Approx Value": f"${shares_to_sell * target_price:,.0f}",
                })
            st.markdown("### Sell / Trim Targets")
            st.dataframe(pd.DataFrame(sell_rows), use_container_width=True, hide_index=True)

        st.plotly_chart(make_chart(df.tail(130), ticker), use_container_width=True, key=f"chart_{ticker}")

        summary.append({
            "Ticker": ticker,
            "Current Price Level": plan["level"],
            "Action": plan["action"],
            "Buy Now $": round(plan["buy_dollars"], 0),
            "Buy Shares": plan["buy_shares"],
            "Trim Now": plan["trim_action"],
            "Trim Shares": plan["trim_shares"],
            "Stop": round(stop, 2),
        })

    except Exception as e:
        st.error(f"Could not analyze {ticker}: {e}")

st.markdown("---")
st.subheader("Action Summary")
if summary:
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

st.markdown(
    """
    ### Rule Philosophy

    Default long-term target is **SMH 60%, NVDA 20%, Cash 20%**, but the app does **not** blindly force that allocation.

    If current price is high:
    - SMH: small starter only or wait
    - NVDA: usually do not chase
    - Cash: preserved for better pullback entries

    If price pulls back while trend is still healthy:
    - SMH gets larger add recommendations
    - NVDA gets smaller tactical recommendations

    Educational tool only. It does not place trades and does not guarantee profit.
    """
)
