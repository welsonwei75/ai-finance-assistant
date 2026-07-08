"""Independent Streamlit client for the Vercel trade-assistant API."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pandas as pd
import httpx
import streamlit as st

st.set_page_config(page_title="AI Market Desk", page_icon="📈", layout="wide")
st.markdown("""<style>
.stApp {background: linear-gradient(135deg,#07111f 0%,#101b2d 100%);}
[data-testid='stMetric'] {background:#142238;border:1px solid #29415f;padding:18px;border-radius:14px;}
</style>""", unsafe_allow_html=True)
st.title("📈 AI Market Desk")
st.caption("FinBERT 情緒 × XGBoost 趨勢 × LLM 交易備忘錄")

with st.sidebar:
    st.header("分析參數")
    api_url = st.text_input("後端 API", os.getenv("BACKEND_API_URL", "https://your-api.vercel.app"))
    market = st.selectbox("市場", ["US_STOCK", "CRYPTO", "FOREX", "TAIWAN_STOCK"])
    risk = st.select_slider("風險偏好", options=["conservative", "balanced", "aggressive"], value="balanced")
    st.subheader("技術指標")
    rsi = st.number_input("RSI", 0.0, 100.0, 50.0)
    macd = st.number_input("MACD", value=0.0, format="%.4f")
    macd_signal = st.number_input("MACD Signal", value=0.0, format="%.4f")
    volatility = st.number_input("波動率", 0.0, 10.0, 0.2)
    volume_change = st.number_input("成交量變化率", -1.0, 100.0, 0.0)

news_text = st.text_area("每行輸入一則核心新聞", height=200, placeholder="Company raises full-year guidance...\nCentral bank signals rates may remain elevated...")


async def request_analysis(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call the remote backend with finite timeouts and useful error details."""

    timeout = httpx.Timeout(90.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{url.rstrip('/')}/api/v1/trade-assistant", json=payload)
        response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("後端回傳格式無效")
    return result


if st.button("開始分析", type="primary", use_container_width=True):
    news = [line.strip() for line in news_text.splitlines() if line.strip()]
    if not news:
        st.warning("請至少輸入一則新聞。")
    else:
        payload = {
            "news": news,
            "market": market,
            "risk_preference": risk,
            "indicators": {
                "rsi": rsi, "macd": macd, "macd_signal": macd_signal,
                "volatility": volatility, "volume_change": volume_change,
            },
        }
        try:
            with st.spinner("正在整合市場訊號…"):
                result = asyncio.run(request_analysis(api_url, payload))
            left, right = st.columns(2)
            left.metric("交易訊號", result["prediction"]["signal"])
            right.metric("上漲機率", f"{result['prediction']['up_probability']:.1%}")
            chart = pd.DataFrame(
                {"情緒分數": [item["score"] for item in result["sentiments"]]},
                index=[f"新聞 {index + 1}" for index in range(len(result["sentiments"]))],
            )
            st.subheader("新聞情緒")
            st.bar_chart(chart, color="#36d399")
            st.subheader("每日交易建議報告")
            st.markdown(result["report_markdown"])
            st.download_button(
                "下載 Markdown 報告", result["report_markdown"], "daily-trading-report.md", "text/markdown"
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            st.error(f"後端服務錯誤：{detail}")
        except (httpx.RequestError, KeyError, TypeError, ValueError) as exc:
            st.error(f"無法完成分析：{exc}")
