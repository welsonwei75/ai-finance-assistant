import os
import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)

# 建立全域信號量：嚴格限制整個環境同時只能有 1 個外發連線，徹底根除 Errno 16
sem = asyncio.Semaphore(1)
limits = httpx.Limits(max_keepalive_connections=1, max_connections=1)

async def _analyze_one(client: httpx.AsyncClient, news: str, headers: dict, HF_URL: str) -> dict:
    """單則新聞分析，加入全域信號量鎖定與完全錯誤捕捉"""
    async with sem: # 同一時間，只有一則新聞能使用 Socket 連線
        try:
            response = await client.post(HF_URL, headers=headers, json={"inputs": news})
            if response.status_code == 200:
                res_json = response.json()
                if res_json and isinstance(res_json, list) and len(res_json) > 0:
                    scores = {item['label'].lower(): item['score'] for item in res_json[0]}
                    # 相容大小寫與標籤名稱
                    pos = scores.get('positive', scores.get('pos', 0.33))
                    neg = scores.get('negative', scores.get('neg', 0.33))
                    neu = scores.get('neutral', scores.get('neu', 0.34))
                    return {"news": news, "scores": {"positive": pos, "negative": neg, "neutral": neu}}
            
            logger.warning(f"FinBERT API responded with status {response.status_code}, falling back to neutral.")
        except Exception as e:
            logger.error(f"FinBERT connection error [Errno 16 / Timeout] caught and skipped: {str(e)}")
        
        # 發生任何連線忙碌、逾時或 502，一律回傳中立分數，絕不讓後端崩潰
        return {"news": news, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}}

async def analyze_sentiment(news_list: list, hf_token: str) -> list:
    """串行安全版情緒分析"""
    HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {hf_token}"}
    results = []
    
    # 設置極短的連線保持，避免佔用 Vercel 資源
    async with httpx.AsyncClient(limits=limits, timeout=15.0) as client:
        # 使用標準的 for 迴圈串行執行，不使用 asyncio.gather 併發
        for news in news_list:
            if not news.strip():
                continue
            res = await _analyze_one(client, news, headers, HF_URL)
            results.append(res)
            await asyncio.sleep(0.5) # 強制休眠 0.5 秒，給予 Linux 核心釋放 Socket 的緩衝時間
            
    return results

async def generate_report(market_trend_pred: str, sentiment_summary: str, raw_news_list: list, risk_level: float, api_key: str) -> str:
    """通用通用大模型報告生成器 (完美捕捉 429 速率限制錯誤)"""
    # 如果你的 Key 是 OpenAI，請保留此端點；若是 DeepSeek，請改回 https://api.deepseek.com/v1/chat/completions
    url = "https://api.openai.com/v1/chat/completions"
    
    retrieved_context = "\n".join([f"- {news}" for news in raw_news_list[:3]])
    prompt_content = f"市場上漲機率: {market_trend_pred}\n情緒: {sentiment_summary}\n新聞: {retrieved_context}\n請用繁體中文生成一份包含市場概述、基於風險度 {risk_level} 的交易建議與風險提示的簡短報告。"
    
    payload = {
        "model": "gpt-3.5-turbo", # 或是你所使用的相容模型名稱 (例如 deepseek-chat)
        "messages": [
            {"role": "system", "content": "你是一位專業的 AI 交易員助理，請一律使用繁體中文回覆。"},
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
            logger.info("Sending request to LLM Provider...")
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            res_json = response.json()
            return res_json['choices'][0]['message']['content']
    except Exception as e:
        # 這裡會精準捕獲 429 Too Many Requests 等錯誤，並向上拋出由 index.py 的防爆機制接住
        logger.error(f"LLM API Failed: {str(e)}")
        raise RuntimeError("LLM rate limit or insufficient balance.")
