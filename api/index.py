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

# 【核心修正】嚴格對齊前端 Streamlit 傳過來的欄位名稱！
class TradePayload(BaseModel):
    news_list: List[str]
    risk_level: float = 0.5

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
        
    if not payload.news_list:
        raise HTTPException(status_code=400, detail="News list cannot be empty")
        
    try:
        # 1. 執行 FinBERT 情緒分析
        analysis_res = await analyze_sentiment(payload.news_list, hf_token)
        
        # 2. 模擬量化特徵與上漲機率運算
        total_pos = sum([item['scores'].get('positive', 0) for item in analysis_res])
        total_neg = sum([item['scores'].get('negative', 0) for item in analysis_res])
        avg_pos = total_pos / len(payload.news_list) if payload.news_list else 0
        avg_neg = total_neg / len(payload.news_list) if payload.news_list else 0
        
        up_probability = min(max((avg_pos - avg_neg + 0.5) * 100, 10.0), 95.0)
        market_sentiment_text = "偏向樂觀" if avg_pos > avg_neg else "偏向悲觀"
        
        # 3. 呼叫大模型（加入防爆安全降級：即使 DeepSeek 沒餘額，也絕不讓 API 報 500 崩潰）
        try:
            report_markdown = await generate_report(
                market_trend_pred=f"{up_probability:.2f}% 機率上漲",
                sentiment_summary=market_sentiment_text,
                raw_news_list=payload.news_list,
                risk_level=payload.risk_level,
                api_key=llm_key
            )
        except Exception as llm_error:
            report_markdown = f"### 📝 每日交易助理報告 (系統提示：大模型服務暫不可用)\n\n**當前量化訊號：** {market_sentiment_text}\n\n*提示：目前 LLM 報告生成管道餘額不足，但上方的 FinBERT 情緒指標與上漲機率預測（{up_probability:.2f}%）皆為即時運算之真實數據，前端可正常抓取顯示。*"

        # 4. 成功回傳 200 OK 與完整量化 JSON 數據
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
