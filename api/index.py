import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 確保從正確的服務層導入
from api.services import analyze_sentiment, generate_report

app = FastAPI(
    title="AI Market Trade Assistant", 
    version="1.0.0"
)

# 1. 前端傳過來的是技術指標結構，定義好 Pydantic 結構
class IndicatorsPayload(BaseModel):
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    volatility: float = 0.15
    volume_change: float = 0.0

class TradePayload(BaseModel):
    news: List[str]
    indicators: IndicatorsPayload

@app.get("/")
async def root():
    return {"status": "ok", "message": "Go to /docs for Swagger UI"}

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/api/v1/trade-assistant")
async def trade_assistant_endpoint(payload: TradePayload):
    hf_token = os.getenv("HF_TOKEN")
    llm_key = os.getenv("LLM_API_KEY")
    
    if not hf_token:
        raise HTTPException(status_code=500, detail="Server configuration error: Missing HF_TOKEN")
        
    if not payload.news:
        raise HTTPException(status_code=400, detail="News list cannot be empty")
        
    try:
        # 1. 執行 FinBERT 情緒分析
        analysis_res = await analyze_sentiment(payload.news, hf_token)
        
        # 2. 模擬量化特徵與上漲機率運算
        total_pos = sum([item['scores'].get('positive', 0) for item in analysis_res])
        total_neg = sum([item['scores'].get('negative', 0) for item in analysis_res])
        avg_pos = total_pos / len(payload.news) if payload.news else 0
        avg_neg = total_neg / len(payload.news) if payload.news else 0
        
        up_probability = min(max((avg_pos - avg_neg + 0.5) * 100, 10.0), 95.0)
        market_sentiment_text = "偏向樂觀" if avg_pos > avg_neg else "偏向悲觀"
        
        # 3. 呼叫大模型（已修正屬性錯誤，真正實現免錢防爆安全降級）
        try:
            report_markdown = await generate_report(
                market_trend_pred=f"{up_probability:.2f}% 機率上漲",
                sentiment_summary=market_sentiment_text,
                raw_news_list=payload.news,
                risk_level=payload.indicators.rsi, 
                api_key=llm_key
            )
        except Exception as llm_error:
            # 即使 DeepSeek 報 402 沒錢，也會被這裡牢牢接住，回傳罐頭文字，不卡死 API
            report_markdown = f"### 📝 每日交易助理報告 (系統提示：大模型服務暫不可用)\n\n**當前量化訊號：** {market_sentiment_text}\n\n*提示：目前 LLM 報告生成管道餘額不足，但上方的 FinBERT 情緒指標與上漲機率預測（{up_probability:.2f}%）皆為即時運算之真實數據，前端可正常抓取顯示。*"

        # 4. 成功回傳 200 OK
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
        # 捕捉真正的非預期系統錯誤
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
