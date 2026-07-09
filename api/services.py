import os
import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)

# 嚴格限制連線池，彻底消滅 FinBERT [Errno 16] Device busy 錯誤
limits = httpx.Limits(max_keepalive_connections=1, max_connections=1)

async def analyze_sentiment(news_list: list, hf_token: str) -> list:
    """呼叫 Hugging Face FinBERT API 進行情緒分析 (安全降級版)"""
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
            await asyncio.sleep(0.3)
            
    return results

async def generate_report(market_trend_pred: str, sentiment_summary: str, raw_news_list: list, risk_level: float, api_key: str) -> str:
    """
    採用全球標準通用大模型 REST 端點，徹底根除 Google 的 404 路由詛咒
    """
    # 這裡我們使用相容性最高、最穩定的開放標準 API 網址 (你也可以換成 SiliconFlow、DeepSeek 或 OpenAI 官方端點)
    url = "https://api.deepseek.com/v1/chat/completions"
    
    retrieved_context = "\n".join([f"- {news}" for news in raw_news_list[:3]])
    prompt_content = f"市場上漲機率: {market_trend_pred}\n情緒: {sentiment_summary}\n新聞: {retrieved_context}\n請用繁體中文生成一份包含市場概述、基於風險度 {risk_level} 的交易建議與風險提示的簡短報告。"
    
    payload = {
        "model": "deepseek-chat", # 或是你所使用的通用模型名稱
        "messages": [
            {"role": "system", "content": "你是一位專業的 AI 交易員助理，請一律使用繁體中文回覆。"},
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    # 這裡會自動帶入我們在 Vercel 後台設定的最新 LLM_API_KEY
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        async with httpx.AsyncClient(limits=limits, timeout=40.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            res_json = response.json()
            return res_json['choices'][0]['message']['content']
    except Exception as e:
        if 'response' in locals() and response is not None:
            logger.error(f"LLM API Error Detail: {response.text}")
        logger.error(f"LLM API core error: {str(e)}")
        raise RuntimeError("Report provider is temporarily unavailable")
