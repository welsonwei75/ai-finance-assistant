import streamlit as st
import httpx
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI 智慧量化金融交易終端",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI 智慧量化金融交易終端 & 情緒儀表板")
st.markdown("本系統整合 **FinBERT 財經情緒模型** 與 **量化技術指標特徵**，即時預測市場走勢。")

# ==========================================
# 左側控制面板：完美對齊後端要求的技術指標欄位
# ==========================================
st.sidebar.header("⚙️ 量化特徵與交易設定")

st.sidebar.markdown("### 📊 技術指標 (Technical Indicators)")
rsi_val = st.sidebar.slider("RSI (相對強弱指標)", 0.0, 100.0, 50.0, step=1.0)
macd_val = st.sidebar.slider("MACD 柱狀值", -10.0, 10.0, 0.0, step=0.1)
macd_sig = st.sidebar.slider("MACD Signal (訊號線)", -10.0, 10.0, 0.0, step=0.1)
volatility_val = st.sidebar.slider("Volatility (歷史波動度)", 0.0, 1.0, 0.15, step=0.01)
volume_chg = st.sidebar.slider("Volume Change (成交量變化率 %)", -1.0, 5.0, 0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 輸入今日焦點新聞 (每行一則)")
default_news = (
    "NVIDIA Advanced Memory Chip Demand Surges with HBM Supply Bottlenecks\n"
    "TSMC Reports Record-Breaking Q2 Revenue Driven by AI Server Shipments\n"
    "Global Manufacturing Slows Down Amid Macroeconomic Headwinds and Inflation"
)
news_input = st.sidebar.text_area("市場即時新聞", value=default_news, height=150)

BACKEND_URL = "https://ai-finance-assistant-v5zz37sr8-ai-agent99.vercel.app/api/v1/trade-assistant"

if st.sidebar.button("🚀 開始即時量化分析", type="primary"):
    news_list = [line.strip() for line in news_input.split("\n") if line.strip()]
    
    if not news_list:
        st.error("請至少輸入一則新聞進行分析！")
    else:
        with st.spinner("🔮 後端 AI 正在瘋狂結合【量化指標】與【新聞情緒】進行運算..."):
            try:
                # 【核心修正】完美對齊後端 Pydantic 期待的巢狀結構
                payload = {
                    "news": news_list,
                    "indicators": {
                        "rsi": rsi_val,
                        "macd": macd_val,
                        "macd_signal": macd_sig,
                        "volatility": volatility_val,
                        "volume_change": volume_chg
                    }
                }
                
                response = httpx.post(BACKEND_URL, json=payload, timeout=45.0)
                
                if response.status_code == 200:
                    data = response.json()
                    metrics = data.get("metrics", {})
                    sentiment_analysis = data.get("sentiment_analysis", [])
                    report_markdown = data.get("report", "")
                    
                    st.success("✨ 分析完成！")
                    
                    # 區塊一：核心指標看板
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="🔮 明日市場上漲機率預測", 
                            value=metrics.get("up_probability", "N/A")
                        )
                    with col2:
                        signal = metrics.get("signal", "HOLD")
                        if "BUY" in signal:
                            st.subheader(f"📊 策略行動訊號: :green[{signal}]")
                        elif "SHORT" in signal:
                            st.subheader(f"📊 策略行動訊號: :red[{signal}]")
                        else:
                            st.subheader(f"📊 策略行動訊號: :orange[{signal}]")
                            
                    st.markdown("---")
                    
                    # 區塊二：FinBERT 新聞情緒視覺化圖表
                    st.subheader("📊 今日新聞 FinBERT 情緒權重分佈")
                    chart_data = []
                    for idx, item in enumerate(sentiment_analysis):
                        news_text = item.get("news", "")
                        scores = item.get("scores", {})
                        short_news = f"新聞 {idx+1}: {news_text[:30]}..." 
                        
                        chart_data.append({"新聞": short_news, "情緒": "正面 (Positive)", "權重分數": scores.get("positive", 0)})
                        chart_data.append({"新聞": short_news, "情緒": "負面 (Negative)", "權重分數": scores.get("negative", 0)})
                        chart_data.append({"新聞": short_news, "情緒": "中立 (Neutral)", "權重分數": scores.get("neutral", 0)})
                    
                    df = pd.DataFrame(chart_data)
                    fig = px.bar(
                        df, x="權重分數", y="新聞", color="情緒", 
                        orientation='h',
                        color_discrete_map={"正面 (Positive)": "#2ecc71", "負面 (Negative)": "#e74c3c", "中立 (Neutral)": "#95a5a6"},
                        barmode="stack",
                        height=300 + (len(sentiment_analysis) * 40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("---")
                    
                    # 區塊三：大模型交易報告
                    st.subheader("📝 AI 交易助理投資報告 (繁體中文)")
                    st.markdown(report_markdown)
                    
                else:
                    st.error(f"後端 API 回報錯誤 (Status: {response.status_code})")
                    st.code(response.text)
            except Exception as e:
                st.error(f"連線至後端伺服器失敗: {str(e)}")
else:
    st.info("💡 請在左側面板調整 RSI、MACD 等量化參數，並點擊「開始即時量化分析」查看儀表板成果。")