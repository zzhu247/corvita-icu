"""Step 1: get the source files, fingerprint them, and check they never change silently.

First run  -> downloads (if needed), extracts, writes data/manifest.json.
Later runs -> recomputes SHA-256 and fails if anything differs from the manifest.
Use --update only when you deliberately change the source data.

    python -m src.ingest            # download if missing, then create/verify manifest
    python -m src.ingest --no-download   # files were downloaded by hand into data/raw
"""
import argparse
import hashlib
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config, p


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(cfg: dict) -> None:
    raw = p(cfg["data"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    for name in cfg["data"]["source_files"]:
        dest = raw / name
        if dest.exists():
            print(f"[ingest] {name} already present, skip download")
            continue
        url = f"{cfg['data']['base_url']}/{name}"
        print(f"[ingest] downloading {url}")
        urllib.request.urlretrieve(url, dest)


def extract(cfg: dict) -> None:
    raw = p(cfg["data"]["raw_dir"])
    if p(cfg["data"]["records_dir"]).exists():
        return
    with tarfile.open(raw / "set-a.tar.gz") as tar:
        tar.extractall(raw, filter="data")   # "data" filter blocks path traversal
    print(f"[ingest] extracted to {cfg['data']['records_dir']}")


def build_manifest(cfg: dict) -> dict:
    raw = p(cfg["data"]["raw_dir"])
    files = {}
    for name in cfg["data"]["source_files"]:
        path = raw / name
        files[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    # Also fingerprint the extracted record files, so edits after extraction are caught.
    rec_files = sorted(p(cfg["data"]["records_dir"]).glob("*.txt"))
    rec_hash = hashlib.sha256()
    for f in rec_files:
        rec_hash.update(f.name.encode())
        rec_hash.update(sha256_file(f).encode())

    combined = hashlib.sha256(
        "".join(files[n]["sha256"] for n in sorted(files)).encode()
        + rec_hash.hexdigest().encode()
    ).hexdigest()

    return {
        "source": cfg["data"]["base_url"],
        "files": files,
        "n_record_files": len(rec_files),
        "records_sha256": rec_hash.hexdigest(),
        "data_version": combined[:12],   # short id used in metadata and API responses
    }


def verify_or_create(cfg: dict, update: bool = False) -> dict:
    manifest_path = p(cfg["data"]["manifest"])
    current = build_manifest(cfg)

    if manifest_path.exists() and not update:
        saved = json.loads(manifest_path.read_text())
        keys = ["files", "n_record_files", "records_sha256", "data_version"]
        diffs = [k for k in keys if saved.get(k) != current[k]]
        if diffs:
            raise RuntimeError(
                f"Source data does not match manifest ({diffs}). "
                "If the change is intended, rerun with --update."
            )
        print(f"[ingest] checksums OK, data_version={current['data_version']}")
        return saved

    current["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(current, indent=2))
    print(f"[ingest] manifest written, data_version={current['data_version']}")
    return current


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--update", action="store_true", help="overwrite the manifest on purpose")
    args = ap.parse_args()

    cfg = load_config()
    if not args.no_download:
        download(cfg)
    extract(cfg)
    verify_or_create(cfg, update=args.update)


if __name__ == "__main__":
    main()
