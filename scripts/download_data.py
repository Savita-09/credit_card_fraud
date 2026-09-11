"""Download and verify the pinned Hugging Face dataset, or the older ULB source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile

DATASET_ID = "santosh3110/credit_card_fraud_transactions"
REVISION = "63e73d05a06e1b87ef7dd9c6092eabc46245fcac"
ARCHIVE_SHA256 = "0be20e0480cd79790f92e2c103acbd8b2c0500c7dd85fc4d12d67ac6b134548e"
URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/credit_card_fraud_transactions.zip?download=true"
ULB_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(output: Path, dataset="huggingface", archive_path: Path | None = None):
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".source.json")
    if output.exists():
        if not manifest.exists():
            raise ValueError(f"Existing {output} has no source manifest; choose another output path.")
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        expected = DATASET_ID if dataset == "huggingface" else "ulb-creditcard"
        identity = metadata.get("dataset_id", "ulb-creditcard" if metadata.get("url") == ULB_URL else None)
        if identity != expected or metadata.get("sha256") != sha256(output):
            raise ValueError("Existing dataset does not match the requested source or checksum.")
        if dataset == "huggingface" and metadata.get("revision") != REVISION:
            raise ValueError("Existing dataset has a different revision; choose another output path.")
        print(json.dumps({**metadata, "cached": True}), flush=True)
        return metadata
    temporary = output.with_suffix(".download")
    extracted = output.with_suffix(".extracting")
    try:
        url = URL if dataset == "huggingface" else ULB_URL
        if archive_path is not None:
            if dataset != "huggingface":
                raise ValueError("--archive is supported only for the pinned Hugging Face ZIP.")
            shutil.copyfile(archive_path, temporary)
        else:
            print(f"Downloading {url}", flush=True)
            with urlopen(url, timeout=120) as response, temporary.open("wb") as destination:
                shutil.copyfileobj(response, destination, length=1024 * 1024)
        if dataset == "huggingface":
            if sha256(temporary) != ARCHIVE_SHA256:
                raise ValueError("Downloaded archive failed its published Hugging Face SHA-256 check.")
            with zipfile.ZipFile(temporary) as archive:
                csv_members = [item for item in archive.infolist() if not item.is_dir() and item.filename.lower().endswith('.csv')]
                if len(csv_members) != 1:
                    raise ValueError("Expected exactly one CSV member in the pinned dataset archive.")
                member = csv_members[0]
                if member.file_size > 2 * 1024**3:
                    raise ValueError("Archive CSV exceeds the supported 2 GiB limit.")
                # Stream into a fixed path; archive member names never become filesystem paths.
                with archive.open(member) as source, extracted.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            extracted.replace(output)
            metadata = {"dataset_id": DATASET_ID, "name": "Hugging Face credit-card transactions", "revision": REVISION,
                        "source": f"https://huggingface.co/datasets/{DATASET_ID}", "url": url, "license": "apache-2.0",
                        "archive_sha256": ARCHIVE_SHA256, "archive_member": member.filename,
                        "provenance_note": "Public benchmark. The dataset card does not document its collection methodology.",
                        "currency": None, "amount_unit": "dataset units"}
        else:
            temporary.replace(output)
            metadata = {"dataset_id": "ulb-creditcard", "name": "ULB credit-card benchmark", "url": url,
                        "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", "currency": "EUR", "amount_unit": "EUR"}
        metadata.update(sha256=sha256(output), file=output.name, bytes=output.stat().st_size)
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)
        extracted.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["huggingface", "ulb"], default="huggingface")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded ZIP; its published checksum is still verified")
    args = parser.parse_args()
    download(args.output or Path("data/raw/hf_creditcard.csv" if args.dataset == "huggingface" else "data/raw/creditcard.csv"), args.dataset, args.archive)
