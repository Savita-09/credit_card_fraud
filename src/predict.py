"""Offline inference uses the exact same verified pipeline as the HTTP service."""
import argparse
from pathlib import Path

from backend.config import Settings
from backend.csv_utils import results_csv
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService
from src.data_processing import read_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    settings = Settings()
    scorer = PredictionService(ModelService(settings.model_dir), settings, history=None)
    frame = read_csv(args.input)
    results = scorer.score(frame, args.threshold, record=False)
    args.output.write_text(results_csv(frame, results), encoding="utf-8")
    print(f"Wrote {len(results):,} scored transactions to {args.output}")
