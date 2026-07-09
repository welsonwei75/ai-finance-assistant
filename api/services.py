import os
import httpx
import logging

logger = logging.getLogger(__name__)

# 限制最大並發連線數
limits = httpx.Limits(max_keepalive_connections=2, max_connections=5)

async def analyze_sentiment(news_list: list, hf_token: str) -> list:
    """呼叫 Hugging Face FinBERT API 進行情緒分析"""
    HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {hf_token}"}
    results = []
    
    async with httpx.AsyncClient(limits=limits, timeout=20.0) as client:
        for news in news_list:
            try:
                response = await client.post(HF_URL, headers=headers, json={"inputs": news})
                if response.status_code == 200:
                    res_json = response.json()
                    scores = {item['label']: item['score'] for item in res_json[0]}
                    results.append({"news": news, "scores": scores})
                else:
                    results.append({"news": news, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}})
            except Exception as e:
                logger.error(f"FinBERT error: {str(e)}")
                results.append({"news": news, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}})
    return results

async def generate_report(market_trend_pred: str, sentiment_summary: str, raw_news_list: list, risk_level: float, api_key: str) -> str:
    """透過 Google Gemini 1.5 Flash 雲端 API 非同步生成繁體中文報告"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    retrieved_context = "\n".join([f"- {news}" for news in raw_news_list[:3]])
    
    prompt_content = f"""
    你是一位專業的 AI 交易員助理。請根據以下量化預測結果與檢索到的市場新聞，生成一份精簡的「每日交易建議報告」。
    
    【量化模型預測】
    - 明日市場上漲機率預測: {market_trend_pred}
    - 整體市場情緒 (FinBERT): {sentiment_summary}
    
    【今日焦點新聞摘要 (RAG 檢索結果)】
    {retrieved_context}
    
    請提供以下格式的報告（使用繁體中文）：
    1. 市場現況概述
    2. 交易行動建議 (基於風險偏好度: {risk_level})
    3. 風險提示
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt_content}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000}
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        async with httpx.AsyncClient(limits=limits, timeout=25.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            res_json = response.json()
            return res_json['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise RuntimeError("Report provider is temporarily unavailable")
