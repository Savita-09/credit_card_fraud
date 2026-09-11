"""Convert raw Hugging Face rows into the nine feature columns accepted by the app."""
import argparse
from pathlib import Path
import pandas as pd
from src.dataset import HF_INPUTS, prepare_transactions


def prepare_file(input_path, output_path, limit=None):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    temporary = output_path.with_suffix('.preparing')
    try:
        for raw in pd.read_csv(input_path, usecols=HF_INPUTS, chunksize=10000, nrows=limit):
            prepared = prepare_transactions(raw)
            prepared.to_csv(temporary, mode='a' if total else 'w', header=not total, index=False)
            total += len(prepared)
        if not total:
            raise ValueError("No transaction rows found.")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Prepared {total:,} transactions: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/raw/hf_creditcard.csv'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/hf_transactions.csv'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    prepare_file(args.input, args.output, args.limit)
