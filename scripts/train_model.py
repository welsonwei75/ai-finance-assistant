"""Build the compact demonstration XGBoost JSON artifact."""
from pathlib import Path
import numpy as np
import xgboost as xgb

FEATURES = ["rsi", "macd", "macd_signal", "volatility", "volume_change", "sentiment_score"]


def main() -> None:
    """Train deterministic demo data; use validated historical data in production."""
    rng = np.random.default_rng(42)
    x = np.column_stack((rng.uniform(0, 100, 4000), rng.normal(0, 2, (4000, 2)),
                         rng.uniform(.05, 1.2, 4000), rng.normal(0, .5, 4000),
                         rng.uniform(-1, 1, 4000))).astype(np.float32)
    score = -.025 * (x[:, 0] - 50) + .5 * (x[:, 1] - x[:, 2]) - x[:, 3] + 1.4 * x[:, 5]
    y = (score + rng.normal(0, .8, 4000) > 0).astype(np.float32)
    model = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": .08,
                       "subsample": .9, "seed": 42, "nthread": 1},
                      xgb.DMatrix(x, label=y, feature_names=FEATURES), num_boost_round=40)
    output = Path(__file__).parents[1] / "api" / "models" / "market_xgb.json"
    model.save_model(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
