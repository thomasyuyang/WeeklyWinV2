# smh_nvda_trade_timing_app.py
# Thomas AI Entry Screener V5 - Positions Manager
# Educational decision support only. Not financial advice.

import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Thomas AI Entry Screener V5", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1250px;}
div[data-testid="metric-container"] {background: rgba(255,255,255,0.95); border: 1px solid #e8eef5; padding: 10px 12px; border-radius: 14px; box-shadow: 0 1px 6px rgba(0,0,0,0.035);}
.card {background: white; border: 1px solid #e5edf5; border-radius: 18px; padding: 16px 18px; box-shadow: 0 2px 12px rgba(0,0,0,0.05); margin: 10px 0 16px 0;}
.green {background:#eaf7ef; border-left:7px solid #29a35a;}
.yellow {background:#fff8e6; border-left:7px solid #d6a300;}
.red {background:#fdecec; border-left:7px solid #d94848;}
.blue {background:#edf2ff; border-left:7px solid #4b6fff;}
.title {font-size:1.25rem; font-weight:800; margin-bottom:8px;}
.line {font-size:1.02rem; line-height:1.55;}
.note {font-size:0.90rem; color:#5b6775;}
</style>
""", unsafe_allow_html=True)

DEFAULT_POSITIONS = pd.DataFrame([{"Ticker":"SMH","Shares":0,"Avg Cost":0.0},{"Ticker":"NVDA","Shares":0,"Avg Cost":0.0},{"Ticker":"MSFT","Shares":0,"Avg Cost":0.0}])

def clean_positions(df):
    for col in ["Ticker","Shares","Avg Cost"]:
        if col not in df.columns: df[col] = "" if col=="Ticker" else 0
    out = df[["Ticker","Shares","Avg Cost"]].copy()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["Shares"] = pd.to_numeric(out["Shares"], errors="coerce").fillna(0).astype(int)
    out["Avg Cost"] = pd.to_numeric(out["Avg Cost"], errors="coerce").fillna(0.0).astype(float)
    out = out[out["Ticker"] != ""]
    return out.drop_duplicates(subset=["Ticker"], keep="last").reset_index(drop=True)

def positions_to_csv(df):
    return clean_positions(df).to_csv(index=False)

@st.cache_data(ttl=300)
def load_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df.dropna()

def rsi(series, period=14):
    delta = series.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df, period=14):
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def add_indicators(df):
    out=df.copy()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean(); out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean(); out["EMA200"] = out["Close"].ewm(span=200, adjust=False).mean()
    out["RSI14"] = rsi(out["Close"]); out["ATR14"] = atr(out); out["ATR_PCT"] = out["ATR14"] / out["Close"] * 100
    out["HIGH_6M"] = out["Close"].rolling(126, min_periods=30).max(); out["LOW_3M"] = out["Close"].rolling(63, min_periods=20).min()
    out["DIST_HIGH"] = (out["Close"] / out["HIGH_6M"] - 1) * 100; out["DIST_LOW"] = (out["Close"] / out["LOW_3M"] - 1) * 100
    out["VOL_AVG20"] = out["Volume"].rolling(20).mean(); out["VOL_RATIO"] = out["Volume"] / out["VOL_AVG20"]
    return out

@dataclass
class State:
    ticker: str; price: float; ema20: float; ema50: float; ema200: float; rsi: float; atr: float; atr_pct: float; dist_high: float; dist_low: float; vol_ratio: float

def get_state(ticker, period="1y"):
    df = add_indicators(load_data(ticker, period)); latest = df.iloc[-1]
    return df, State(ticker, float(latest.Close), float(latest.EMA20), float(latest.EMA50), float(latest.EMA200), float(latest.RSI14), float(latest.ATR14), float(latest.ATR_PCT), float(latest.DIST_HIGH), float(latest.DIST_LOW), float(latest.VOL_RATIO) if not np.isnan(latest.VOL_RATIO) else 1.0)

def money(x): return f"${x:,.0f}"
def price(x): return f"${x:,.2f}"
def shares_for(dollars, px): return max(0, math.floor(dollars/px)) if px>0 else 0

def market_regime():
    rows=[]; total=0
    for t in ["SPY","QQQ","SMH"]:
        try:
            _,s=get_state(t); score=int(s.price>s.ema50)+int(s.price>s.ema200)+int(s.ema50>s.ema200)+int(s.rsi>=45); total += score
            rows.append({"Ticker":t,"Price":price(s.price),"Above EMA50":s.price>s.ema50,"Above EMA200":s.price>s.ema200,"EMA50>EMA200":s.ema50>s.ema200,"RSI":round(s.rsi,1),"Score":score})
        except Exception:
            rows.append({"Ticker":t,"Price":"N/A","Above EMA50":False,"Above EMA200":False,"EMA50>EMA200":False,"RSI":"N/A","Score":0})
    if total>=10: return {"name":"Bullish","score":total,"modifier":1.0,"css":"green","reason":"Broad market, tech, and semiconductor trends are healthy.","rows":rows}
    if total>=7: return {"name":"Neutral","score":total,"modifier":0.6,"css":"yellow","reason":"Trend is mixed. New buys should be smaller.","rows":rows}
    return {"name":"Defensive","score":total,"modifier":0.25,"css":"red","reason":"Trend is weak. Preserve cash and avoid aggressive buying.","rows":rows}

def price_level(s):
    weak = s.price < s.ema50 or s.ema20 < s.ema50*0.985; hot = s.rsi>=68
    if weak: return "Weak / below trend","red"
    if s.dist_high>-3 and hot: return "High price level","red"
    if s.dist_high>-3: return "Near high","yellow"
    if s.dist_high>-6: return "Slightly extended","yellow"
    if -10 <= s.dist_high <= -6: return "Normal pullback","green"
    if -16 <= s.dist_high < -10: return "Good pullback","green"
    if s.dist_high < -16: return "Deep pullback","blue"
    return "Neutral","blue"

def dynamic_stop(s, avg_cost, has_position):
    if s.ticker in ["NVDA","AVGO","TSM","AMD"]: ema_stop=s.ema50*0.97; atr_stop=s.price-2.0*s.atr
    else: ema_stop=s.ema50*0.98; atr_stop=s.price-1.8*s.atr
    stop=max(ema_stop, atr_stop)
    if has_position and avg_cost>0 and s.price>avg_cost*1.08:
        stop=max(stop, s.ema20*0.98, s.price-1.8*s.atr)
    return stop

def entry_score(s, regime_name):
    score=0; reasons=[]
    if s.price>s.ema200: score+=8; reasons.append("above EMA200")
    if s.price>s.ema50: score+=8; reasons.append("above EMA50")
    if s.ema20>s.ema50: score+=7; reasons.append("EMA20>EMA50")
    if s.ema50>s.ema200: score+=7; reasons.append("EMA50>EMA200")
    if -16<=s.dist_high<=-6: score+=30; reasons.append("good pullback zone")
    elif -6<s.dist_high<=-3: score+=14; reasons.append("slight pullback")
    elif s.dist_high>-3: score-=10; reasons.append("near high")
    elif s.dist_high<-16: score+=8; reasons.append("deep pullback; needs confirmation")
    if 42<=s.rsi<=60: score+=20; reasons.append("healthy RSI")
    elif 60<s.rsi<=68: score+=10; reasons.append("RSI a little warm")
    elif s.rsi>70: score-=15; reasons.append("overheated RSI")
    elif s.rsi<35: score-=8; reasons.append("too weak/oversold")
    score += 20 if s.atr_pct<=3 else 12 if s.atr_pct<=5 else 3
    if regime_name=="Defensive": score-=20
    elif regime_name=="Neutral": score-=8
    return max(0,min(100,score)), "; ".join(reasons)

def target_pct(ticker):
    if ticker=="SMH": return 0.60
    if ticker in ["VGT","QQQ"]: return 0.40
    if ticker in ["NVDA","MSFT","GOOGL","AVGO","TSM"]: return 0.20
    if ticker=="MO": return 0.15
    return 0.15

def max_pct(ticker):
    if ticker=="SMH": return 0.80
    if ticker in ["VGT","QQQ"]: return 0.50
    if ticker in ["NVDA","MSFT","GOOGL","AVGO","TSM"]: return 0.25
    if ticker=="MO": return 0.20
    return 0.20

def action_for_candidate(s, current_value, avg_cost, account_value, cash, regime, holding_count, max_holdings):
    level,tone=price_level(s); score,reason=entry_score(s, regime["name"]); has_position=current_value>0
    stop=dynamic_stop(s, avg_cost, has_position); risk_pct=(s.price/stop-1)*100 if stop>0 else np.nan
    base_buy_pct=0; action="WATCH"
    if level in ["Normal pullback","Good pullback"] and score>=65:
        base_buy_pct=0.12 if s.ticker not in ["SMH","VGT","QQQ"] else 0.18; action="BUY / ADD"
    elif level in ["Slightly extended","Near high"] and not has_position and score>=55:
        base_buy_pct=0.04 if s.ticker not in ["SMH","VGT","QQQ"] else 0.08; action="SMALL STARTER ONLY"
    elif level=="High price level": action="DO NOT CHASE"
    elif level=="Weak / below trend": action="WAIT / REDUCE IF HELD"
    elif level=="Deep pullback": action="WAIT FOR REVERSAL"
    if not has_position and holding_count>=max_holdings and base_buy_pct>0:
        action="MAX HOLDINGS REACHED"; base_buy_pct=0
    buy_dollars=account_value*base_buy_pct*regime["modifier"]
    buy_dollars=min(buy_dollars, cash, max(0, account_value*max_pct(s.ticker)-current_value))
    buy_shares=shares_for(buy_dollars, s.price); buy_dollars=buy_shares*s.price
    raw_adds=[s.price*0.97, s.price*0.95, s.price*0.90]; min_add=stop*1.03
    valid_adds=[x for x in raw_adds if x>min_add]
    trim_action="No trim"; trim_shares=0; shares_est=math.floor(current_value/s.price) if s.price>0 else 0
    if has_position:
        pnl=(s.price/avg_cost-1)*100 if avg_cost>0 else 0
        if s.price<stop: trim_action,trim_shares="STOP / EXIT",shares_est
        elif s.ticker!="SMH" and pnl>=10: trim_action,trim_shares="Trim 50%",math.ceil(shares_est*0.5)
        elif current_value>account_value*target_pct(s.ticker)*1.25 and level in ["High price level","Near high"]: trim_action,trim_shares="Trim 20%",math.ceil(shares_est*0.2)
    return {"Ticker":s.ticker,"Price":s.price,"Level":level,"Tone":tone,"Entry Score":score,"Action":action,"Buy $":buy_dollars,"Buy Shares":buy_shares,"Current Value":current_value,"Target %":target_pct(s.ticker),"Max %":max_pct(s.ticker),"Reference Stop":stop,"Risk %":risk_pct,"Valid Add Levels":valid_adds,"Invalid Adds Removed":len(raw_adds)-len(valid_adds),"Trim Action":trim_action,"Trim Shares":trim_shares,"Reason":reason}

def plot_chart(df,ticker):
    fig=go.Figure(); fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name=ticker))
    fig.add_trace(go.Scatter(x=df.index,y=df.EMA20,name="EMA20",mode="lines")); fig.add_trace(go.Scatter(x=df.index,y=df.EMA50,name="EMA50",mode="lines")); fig.add_trace(go.Scatter(x=df.index,y=df.EMA200,name="EMA200",mode="lines"))
    fig.update_layout(height=380,margin=dict(l=10,r=10,t=35,b=10),xaxis_rangeslider_visible=False,legend=dict(orientation="h")); return fig

st.title("📈 Thomas AI Entry Screener V5")
st.caption("Position tracking + persistent CSV manager + best entry candidates + dynamic stops.")

st.sidebar.header("Account")
account_value=st.sidebar.number_input("Total short-term account value ($)", min_value=1000.0, value=10000.0, step=500.0)
cash=st.sidebar.number_input("Current cash available ($)", min_value=0.0, value=10000.0, step=100.0)
max_holdings=st.sidebar.slider("Maximum holdings allowed",1,5,3)
min_cash_pct=st.sidebar.slider("Minimum cash reserve %",0,50,20)/100
st.sidebar.header("Candidate Universe")
candidate_text=st.sidebar.text_area("Candidates to screen","SMH,NVDA,MSFT,GOOGL,MO,VGT,QQQ",height=80)
candidates=[x.strip().upper() for x in candidate_text.replace(";",",").split(",") if x.strip()]
period=st.sidebar.selectbox("Chart period",["3mo","6mo","1y"],index=2)
if st.sidebar.button("Refresh market data"): st.cache_data.clear()

st.markdown('<div class="card blue"><div class="title">Positions Manager</div><div class="line">Streamlit Cloud does not permanently save manual entries by itself. Upload your saved positions CSV, edit it, then download the updated CSV after changes.</div></div>', unsafe_allow_html=True)
if "positions_df" not in st.session_state: st.session_state.positions_df=DEFAULT_POSITIONS.copy()
upload=st.file_uploader("Upload saved positions CSV", type=["csv"])
if upload is not None:
    try:
        st.session_state.positions_df=clean_positions(pd.read_csv(upload)); st.success("Positions loaded from CSV.")
    except Exception as e: st.error(f"Could not read CSV: {e}")
positions_edit=st.data_editor(st.session_state.positions_df, num_rows="dynamic", use_container_width=True, key="positions_editor", column_config={"Ticker":st.column_config.TextColumn("Ticker"),"Shares":st.column_config.NumberColumn("Shares",min_value=0,step=1),"Avg Cost":st.column_config.NumberColumn("Avg Cost",min_value=0.0,step=0.01,format="$%.2f")})
if st.button("Apply position changes"):
    st.session_state.positions_df=clean_positions(positions_edit); st.success("Position changes applied in this session. Download CSV to keep them permanently.")
st.download_button("Download updated positions CSV", data=positions_to_csv(positions_edit), file_name="thomas_positions.csv", mime="text/csv")

positions=clean_positions(positions_edit); active_positions=positions[positions.Shares>0].copy(); holding_count=len(active_positions.Ticker.unique()); current_holdings=set(active_positions.Ticker.unique())
reg=market_regime()
st.markdown(f'<div class="card {reg["css"]}"><div class="title">Market Regime: {reg["name"]} ({reg["score"]}/12)</div><div class="line"><b>Buy size modifier:</b> {reg["modifier"]:.0%}<br><b>Reason:</b> {reg["reason"]}</div></div>', unsafe_allow_html=True)
with st.expander("Market regime details", expanded=False): st.dataframe(pd.DataFrame(reg["rows"]), hide_index=True, use_container_width=True)
st.markdown(f'<div class="card blue"><div class="title">Portfolio Rules</div><div class="line"><b>Max holdings:</b> {max_holdings} | <b>Current holdings:</b> {holding_count} | <b>Minimum cash reserve:</b> {min_cash_pct:.0%}<br>New candidate buys are blocked if max holdings are reached. Add levels below Reference Stop × 1.03 are removed.</div></div>', unsafe_allow_html=True)
pos_map={r.Ticker:{"shares":int(r.Shares),"avg_cost":float(r["Avg Cost"])} for _,r in active_positions.iterrows()}
all_tickers=sorted(set(candidates)|set(pos_map.keys())); results=[]; details={}
for t in all_tickers:
    try:
        df,s=get_state(t,period); sh=pos_map.get(t,{}).get("shares",0); avg=pos_map.get(t,{}).get("avg_cost",0.0); current_value=sh*s.price
        result=action_for_candidate(s,current_value,avg,account_value,cash,reg,holding_count,max_holdings); results.append(result); details[t]=(df,s,result,sh,avg)
    except Exception as e: st.warning(f"Could not load {t}: {e}")
if results:
    dfres=pd.DataFrame(results)
    if max(0,cash-account_value*min_cash_pct)<=0:
        dfres["Buy $"]=0.0; dfres["Buy Shares"]=0; dfres["Action"]=dfres["Action"].where(dfres["Current Value"]>0,"CASH RESERVE LIMIT")
    top_buy=dfres[(dfres["Buy Shares"]>0)&(~dfres["Action"].str.contains("MAX|RESERVE|CHASE|WAIT",regex=True))].sort_values("Entry Score",ascending=False)
    if not top_buy.empty:
        tb=top_buy.iloc[0]; css={"green":"green","yellow":"yellow","red":"red","blue":"blue"}.get(tb.Tone,"blue")
        st.markdown(f'<div class="card {css}"><div class="title">Thomas Best Entry Candidate Today: {tb.Ticker}</div><div class="line"><b>Action:</b> {tb.Action}<br><b>Buy:</b> {money(tb["Buy $"])} / {int(tb["Buy Shares"])} shares<br><b>Price level:</b> {tb.Level} | <b>Entry score:</b> {tb["Entry Score"]}/100<br><b>Reference stop:</b> {price(tb["Reference Stop"])}<br><span class="note">{tb.Reason}</span></div></div>', unsafe_allow_html=True)
    else: st.markdown('<div class="card yellow"><div class="title">Thomas Best Entry Candidate Today: No strong new buy</div><div class="line">Reason: candidates may be near high, weak, max holdings reached, or cash reserve limit reached.</div></div>', unsafe_allow_html=True)
    display=dfres.copy(); display["Price"]=display["Price"].map(price); display["Buy $"]=display["Buy $"].map(money); display["Reference Stop"]=display["Reference Stop"].map(price); display["Risk %"]=display["Risk %"].map(lambda x:f"{x:.1f}%"); display["Current Value"]=display["Current Value"].map(money); display["Target %"]=display["Target %"].map(lambda x:f"{x:.0%}"); display["Max %"]=display["Max %"].map(lambda x:f"{x:.0%}")
    cols=["Ticker","Level","Action","Entry Score","Price","Buy $","Buy Shares","Current Value","Reference Stop","Risk %","Trim Action","Trim Shares","Invalid Adds Removed","Reason"]
    st.subheader("Candidate Screener Ranking"); st.dataframe(display.sort_values("Entry Score",ascending=False)[cols], hide_index=True, use_container_width=True)
    st.subheader("Current Holding Review")
    if holding_count==0: st.info("No current holdings entered.")
    else: st.dataframe(display[display.Ticker.isin(current_holdings)][cols], hide_index=True, use_container_width=True)
    selected=st.selectbox("View detail chart", all_tickers)
    if selected in details:
        df,s,res,sh,avg=details[selected]; st.markdown(f"## {selected} Detail")
        c1,c2,c3,c4=st.columns(4); c1.metric("Current Price",price(s.price)); c2.metric("Entry Score",f"{res['Entry Score']}/100"); c3.metric("Reference Stop",price(res["Reference Stop"])); c4.metric("Risk to Stop",f"{res['Risk %']:.1f}%")
        add_levels=res["Valid Add Levels"]; add_df=pd.DataFrame({"Valid Add Level":[price(x) for x in add_levels] if add_levels else ["None"],"Note":["Above stop × 1.03"]*len(add_levels) if add_levels else ["Lower add zones removed because they conflict with stop"]})
        st.markdown("### Valid Add Levels"); st.dataframe(add_df, hide_index=True, use_container_width=True); st.plotly_chart(plot_chart(df.tail(130),selected), use_container_width=True)
st.markdown("---")
st.markdown("""### How to keep holdings saved
After editing positions, click **Download updated positions CSV**. Next time you open the cloud app, upload that CSV first. This is the simplest reliable method for Streamlit Community Cloud without setting up a database or Google Sheets.
""")
