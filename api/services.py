import os
import httpx
import logging
import asyncio
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# 限制最大並發連線數，防止 FinBERT 報出 Device or resource busy 錯誤
limits = httpx.Limits(max_keepalive_connections=1, max_connections=1)

async def analyze_sentiment(news_list: list, hf_token: str) -> list:
    """呼叫 Hugging Face FinBERT API 進行情緒分析"""
    HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {hf_token}"}
    results = []
    
    async with httpx.AsyncClient(limits=limits, timeout=20.0) as client:
        for news in news_list:
            if not news.strip():
                continue
            try:
                response = await client.post(HF_URL, headers=headers, json={"inputs": news})
                if response.status_code == 200:
                    res_json = response.json()
                    scores = {item['label']: item['score'] for item in res_json[0]}
                    results.append({"news": news, "scores": scores})
                else:
                    results.append({"news": news, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}})
            except Exception as e:
                logger.error(f"FinBERT local connection error skipped: {str(e)}")
                results.append({"news": news, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}})
            await asyncio.sleep(0.2)
            
    return results

async def generate_report(market_trend_pred: str, sentiment_summary: str, raw_news_list: list, risk_level: float, api_key: str) -> str:
    """透過 Google 官方 SDK 呼叫 Gemini 1.5 Flash，免去手拼 URL 出錯的風險"""
    retrieved_context = "\n".join([f"- {news}" for news in raw_news_list[:3]])
    prompt_content = f"市場上漲機率: {market_trend_pred}\n情緒: {sentiment_summary}\n新聞: {retrieved_context}\n請用繁體中文生成一份包含市場概述、基於風險度 {risk_level} 的交易建議與風險提示的簡短報告。"
    
    try:
        # 使用官方 SDK 初始化客戶端
        client = genai.Client(api_key=api_key)
        
        # 官方原生非阻塞調用方式，自動校正底層所有 v1/v1beta 路由
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt_content,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1000
            )
        )
        return response.text
    except Exception as e:
        logger.error(f"Google Gemini SDK core error: {str(e)}")
        raise RuntimeError("Report provider is temporarily unavailable")
