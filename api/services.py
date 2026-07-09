import os
import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)

# 限制連線池，徹底防範 Errno 16 忙碌錯誤
limits = httpx.Limits(max_keepalive_connections=1, max_connections=1)

async def analyze_sentiment(news_list: list, hf_token: str) -> list:
    """呼叫 Hugging Face FinBERT API"""
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
    使用雙重模型相容機制（Flash-latest / Pro）繞過 Google 刁鑽的 404 路由限制
    """
    retrieved_context = "\n".join([f"- {news}" for news in raw_news_list[:3]])
    prompt_content = f"市場上漲機率: {market_trend_pred}\n情緒: {sentiment_summary}\n新聞: {retrieved_context}\n請用繁體中文生成一份包含市場概述、基於風險度 {risk_level} 的交易建議與風險提示的簡短報告。"
    
    payload = {
        "contents": [{"parts": [{"text": prompt_content}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000}
    }
    headers = {"Content-Type": "application/json"}
    
    # 備案模型清單：先嘗試最新的別名網址，失敗再切換到全域開放的舊版 Pro 端點
    models_to_try = ["gemini-1.5-pro", "gemini-1.5-flash"]
    
    async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
        for model in models_to_try:
            # 這是 Google 標準的 v1beta 呼叫網址結構
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                logger.info(f"Attempting to generate report using model: {model}")
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    res_json = response.json()
                    return res_json['candidates'][0]['content']['parts'][0]['text']
                else:
                    logger.warning(f"Model {model} failed with status {response.status_code}: {response.text}")
            except Exception as e:
                logger.error(f"Exception raised for model {model}: {str(e)}")
                
        # 如果所有模型都宣告失敗
        raise RuntimeError("All Gemini endpoints (1.5-flash-latest, gemini-pro) returned 404/Error.")
