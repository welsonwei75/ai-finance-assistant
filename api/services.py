"""External AI integrations and lightweight market prediction services."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import httpx

from api.config import Settings

LOGGER = logging.getLogger(__name__)
HF_URL: Final[str] = "https://api-inference.huggingface.co/models/ProsusAI/finbert"


class UpstreamServiceError(RuntimeError):
    """Raised when a required external service cannot provide a valid response."""


class SentimentService:
    """Analyze financial headlines through the hosted FinBERT inference API."""

    def __init__(self, settings: Settings, max_concurrency: int = 5) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _analyze_one(self, client: httpx.AsyncClient, text: str) -> dict[str, Any]:
        """Analyze one headline, retrying transient loading and network failures."""

        headers = {"Authorization": f"Bearer {self._settings.hf_token.get_secret_value()}"}
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        async with self._semaphore:
            for attempt in range(3):
                try:
                    response = await client.post(HF_URL, headers=headers, json=payload)
                    if response.status_code == 503:
                        retry_after = min(float(response.headers.get("retry-after", 2**attempt)), 8.0)
                        LOGGER.warning("FinBERT is loading; retrying in %.1fs", retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    response.raise_for_status()
                    raw = response.json()
                    scores = raw[0] if raw and isinstance(raw[0], list) else raw
                    if not isinstance(scores, list) or not scores:
                        raise ValueError("Unexpected FinBERT response shape")
                    by_label = {str(item["label"]).lower(): float(item["score"]) for item in scores}
                    label, confidence = max(by_label.items(), key=lambda item: item[1])
                    signed_score = by_label.get("positive", 0.0) - by_label.get("negative", 0.0)
                    return {
                        "text": text,
                        "label": label,
                        "confidence": round(confidence, 6),
                        "score": round(signed_score, 6),
                        "degraded": False,
                    }
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    if attempt < 2 and isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
                        await asyncio.sleep(2**attempt)
                        continue
                    LOGGER.exception("FinBERT analysis failed for one headline")
                    break

        # A neutral result keeps the complete workflow usable while clearly flagging degraded data.
        return {"text": text, "label": "neutral", "confidence": 0.0, "score": 0.0, "degraded": True}

    async def analyze_news_list(self, news_list: list[str]) -> list[dict[str, Any]]:
        """Analyze headlines concurrently while retaining their original order."""

        timeout = httpx.Timeout(self._settings.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return list(await asyncio.gather(*(self._analyze_one(client, item) for item in news_list)))


class XGBoostPredictor:
    """Load a compact XGBoost JSON booster and emit normalized trading signals."""

    FEATURE_ORDER: Final[tuple[str, ...]] = (
        "rsi", "macd", "macd_signal", "volatility", "volume_change", "sentiment_score"
    )

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._booster: Any | None = None

    def _load(self) -> Any:
        """Lazily import native dependencies to reduce initialization cost."""

        if self._booster is None:
            if not self._model_path.is_file():
                raise RuntimeError(f"XGBoost model not found: {self._model_path}")
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(self._model_path)
            self._booster = booster
        return self._booster

    def predict(self, features: Mapping[str, float], sentiment_score: float) -> dict[str, Any]:
        """Predict upside probability and map it to an actionable standard signal."""

        import numpy as np
        import xgboost as xgb

        merged = {**features, "sentiment_score": sentiment_score}
        missing = [name for name in self.FEATURE_ORDER if name not in merged]
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")
        matrix = np.asarray([[float(merged[name]) for name in self.FEATURE_ORDER]], dtype=np.float32)
        probability = float(self._load().predict(xgb.DMatrix(matrix, feature_names=list(self.FEATURE_ORDER)))[0])
        probability = max(0.0, min(1.0, probability))
        signal = "🟢 STRONG BUY" if probability >= 0.65 else "🔴 SHORT" if probability <= 0.35 else "🟡 HOLD"
        return {"signal": signal, "up_probability": round(probability, 6)}


class LLMReportService:
    """Generate concise Markdown reports through an OpenAI-compatible Chat API."""

    SYSTEM_PROMPT: Final[str] = """You are a disciplined institutional market analyst.
Use only supplied evidence. Never invent prices, facts, or certainty. Clearly separate facts,
model output, interpretation, risks, invalidation conditions, and position-sizing guidance.
The report is decision support, not financial advice. Return Traditional Chinese Markdown with:
# 每日交易建議報告; ## 執行摘要; ## 新聞與情緒; ## 模型訊號;
## 交易計畫; ## 風險與失效條件; ## 免責聲明."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def generate_report(
        self,
        news: Sequence[Mapping[str, Any]],
        prediction: Mapping[str, Any],
        risk_preference: str,
        market: str,
    ) -> str:
        """Send bounded structured context to the configured LLM endpoint."""

        context = json.dumps(
            {"market": market, "risk_preference": risk_preference, "news": news, "prediction": prediction},
            ensure_ascii=False,
        )
        payload = {
            "model": self._settings.llm_model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"請依據以下 JSON 撰寫報告：\n{context}"},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        url = f"{str(self._settings.llm_base_url).rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("LLM returned an empty report")
                return content.strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            LOGGER.exception("LLM report generation failed")
            raise UpstreamServiceError("Report provider is temporarily unavailable") from exc
