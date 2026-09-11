"""Consistent upload validation and spreadsheet-safe CSV export."""
import io
import pandas as pd
from src.data_processing import read_csv, validate_features


def parse_upload(filename, content, schema, max_rows):
    if not filename or not filename.lower().endswith(".csv"):
        raise ValueError("Upload a .csv file.")
    frame = read_csv(content)
    if len(frame) > max_rows:
        raise ValueError(f"Batch exceeds {max_rows:,} rows. Split it into smaller files.")
    return validate_features(frame, schema)


def results_csv(frame, results):
    exported = frame.reset_index(drop=True).copy()
    # Neutralize spreadsheet formulas in any untrusted categorical cells (numeric negatives remain numeric).
    for name in exported.select_dtypes(include=["object", "string"]):
        exported[name] = exported[name].map(lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    for name in ["prediction", "fraud_probability", "risk_level", "threshold", "model_version"]:
        exported[name] = [result[name] for result in results]
    output = io.StringIO()
    exported.to_csv(output, index=False)
    return output.getvalue()
