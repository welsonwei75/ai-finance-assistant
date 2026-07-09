"""FastAPI entry point for Vercel Serverless Functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.config import get_settings
from api.services import LLMReportService, SentimentService, UpstreamServiceError, XGBoostPredictor

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="AI Market Trade Assistant", version="1.0.0")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Welcome to AI Finance API"}

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

class TechnicalIndicators(BaseModel):
    """Validated, bounded technical features expected by the trained model."""

    model_config = ConfigDict(extra="forbid")
    rsi: Annotated[float, Field(ge=0, le=100)]
    macd: Annotated[float, Field(ge=-1000, le=1000)]
    macd_signal: Annotated[float, Field(ge=-1000, le=1000)]
    volatility: Annotated[float, Field(ge=0, le=10)]
    volume_change: Annotated[float, Field(ge=-1, le=100)]


class NewsPayload(BaseModel):
    """One complete trade-assistant request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    news: Annotated[list[str], Field(min_length=1, max_length=20)]
    indicators: TechnicalIndicators
    market: Literal["US_STOCK", "CRYPTO", "FOREX", "TAIWAN_STOCK"] = "US_STOCK"
    risk_preference: Literal["conservative", "balanced", "aggressive"] = "balanced"

    @field_validator("news")
    @classmethod
    def validate_news(cls, values: list[str]) -> list[str]:
        """Reject blank or excessively long untrusted news inputs."""

        if any(not item.strip() or len(item) > 1000 for item in values):
            raise ValueError("Each news item must contain 1-1000 characters")
        return [item.strip() for item in values]


class SentimentResult(BaseModel):
    """Normalized FinBERT result for one headline."""

    text: str
    label: str
    confidence: float
    score: float
    degraded: bool


class PredictionResult(BaseModel):
    """Normalized model prediction."""

    signal: str
    up_probability: float


class AnalysisResponse(BaseModel):
    """Stable public response contract consumed by the frontend."""

    sentiments: list[SentimentResult]
    average_sentiment: float
    prediction: PredictionResult
    report_markdown: str


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Return liveness without calling paid upstream services."""

    return {"status": "ok", "version": app.version}


@app.post("/api/v1/trade-assistant", response_model=AnalysisResponse)
async def trade_assistant(payload: NewsPayload) -> AnalysisResponse:
    """Run sentiment, prediction, and report generation as one workflow."""

    try:
        sentiments = await SentimentService(settings).analyze_news_list(payload.news)
        average = sum(float(item["score"]) for item in sentiments) / len(sentiments)
        predictor = XGBoostPredictor(Path(__file__).with_name("models") / "market_xgb.json")
        prediction = predictor.predict(payload.indicators.model_dump(), average)
        report = await LLMReportService(settings).generate_report(
            sentiments, prediction, payload.risk_preference, payload.market
        )
        return AnalysisResponse(
            sentiments=sentiments,
            average_sentiment=round(average, 6),
            prediction=prediction,
            report_markdown=report,
        )
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        LOGGER.exception("Trade assistant failed")
        raise HTTPException(status_code=500, detail="Analysis pipeline is unavailable") from exc
