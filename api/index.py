import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 修正導入：只導入最新版 services.py 裡實際存在的函數
from api.services import analyze_sentiment, generate_report

app = FastAPI(
    title="AI Market Trade Assistant",
    version="1.0.0"
)

class TradePayload(BaseModel):
    news_list: List[str]
    risk_level: float = 0.5

@app.get("/")
async def root():
    return {"status": "ok", "message": "Welcome to AI Finance API. Go to /docs for Swagger UI."}

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/api/v1/trade-assistant")
async def trade_assistant_endpoint(payload: TradePayload):
    hf_token = os.getenv("HF_TOKEN")
    llm_key = os.getenv("LLM_API_KEY")
    
    if not hf_token or not llm_key:
        raise HTTPException(status_code=500, detail="Server configuration error: Missing API tokens")
        
    if not payload.news_list:
        raise HTTPException(status_code=400, detail="News list cannot be empty")
        
    try:
        # 1. 執行情緒分析
        analysis_res = await analyze_sentiment(payload.news_list, hf_token)
        
        # 2. 模擬量化特徵運算
        total_pos = sum([item['scores'].get('positive', 0) for item in analysis_res])
        total_neg = sum([item['scores'].get('negative', 0) for item in analysis_res])
        avg_pos = total_pos / len(payload.news_list) if payload.news_list else 0
        avg_neg = total_neg / len(payload.news_list) if payload.news_list else 0
        
        up_probability = min(max((avg_pos - avg_neg + 0.5) * 100, 10.0), 95.0)
        market_sentiment_text = "偏向樂觀 (Optimistic)" if avg_pos > avg_neg else "偏向悲觀 (Pessimistic)"
        
        # 3. 生成 Gemini 報告
        report_markdown = await generate_report(
            market_trend_pred=f"{up_probability:.2f}% 機率上漲",
            sentiment_summary=market_sentiment_text,
            raw_news_list=payload.news_list,
            risk_level=payload.risk_level,
            api_key=llm_key
        )
        
        return {
            "status": "success",
            "metrics": {
                "up_probability": f"{up_probability:.2f}%",
                "signal": "STRONG BUY" if up_probability > 70 else ("HOLD" if up_probability > 40 else "SHORT")
            },
            "sentiment_analysis": analysis_res,
            "report": report_markdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
