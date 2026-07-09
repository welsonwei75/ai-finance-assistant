import os
import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)

# 強制將最大連線數限制為 1，徹底消滅 FinBERT 的 Errno 16 Device busy 錯誤
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
            await asyncio.sleep(0.3) # 拉長休眠時間，確保 Socket 完全釋放
            
    return results

async def generate_report(market_trend_pred: str, sentiment_summary: str, raw_news_list: list, risk_level: float, api_key: str) -> str:
    """
    使用 Google Gemini 最核心、最不可能 404 的標準 REST API 網址
    """
    # 【核心修正】這是 Google 針對 1.5-flash 規定的絕對標準全域端點
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    retrieved_context = "\n".join([f"- {news}" for news in raw_news_list[:3]])
    prompt_content = f"市場上漲機率: {market_trend_pred}\n情緒: {sentiment_summary}\n新聞: {retrieved_context}\n請用繁體中文生成一份包含市場概述、基於風險度 {risk_level} 的交易建議與風險提示的簡短報告。"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_content
            }]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1000
        }
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            # 如果發現還是 404，印出詳細的 Google 回傳內容，方便我們在 Logs 裡一目了然
            if response.status_code != 200:
                logger.error(f"Google Gemini response raw bytes: {response.text}")
                
            response.raise_for_status()
            res_json = response.json()
            return res_json['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        logger.error(f"Gemini API core error: {str(e)}")
        raise RuntimeError("Report provider is temporarily unavailable")
