# AI Market Trade Assistant

前後端分離：FastAPI 部署至 Vercel，Streamlit 獨立部署。FinBERT 與 LLM 均使用外部 API，後端不含本地 NLP 模型。

## 啟動

1. 安裝 `requirements.txt`，複製 `.env.example` 為 `.env` 並填入金鑰。
2. 執行 `python scripts/train_model.py` 產生模型 JSON。
3. 執行 `uvicorn api.index:app --reload`。
4. 前端另行安裝 `requirements-frontend.txt`、設定 `BACKEND_API_URL`，再執行 `streamlit run app.py`。

Vercel 需設定 `HF_TOKEN`、`LLM_API_KEY`、`LOG_LEVEL`、`LLM_BASE_URL`、`LLM_MODEL`。Streamlit 僅需公開後端 URL，不存放供應商金鑰。

> 訓練腳本的合成資料僅供部署流程驗證，不構成真實交易模型。正式上線前請使用受治理的歷史資料完成 walk-forward 驗證、校準、回測與版本管理。
