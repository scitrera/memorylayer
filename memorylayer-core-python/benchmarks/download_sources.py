#!/usr/bin/env python3
"""Download eval source datasets from HuggingFace into a local gitignored cache.

Run this once so converters and benchmarks work offline (no repeated HF pulls).
Files land under ``benchmarks/.cache/raw/<dataset>/``. Idempotent: existing
files of the expected size are skipped. Network only; run outside the sandbox.

Usage:
    python benchmarks/download_sources.py                 # all manifest datasets
    python benchmarks/download_sources.py --only longmemeval
    python benchmarks/download_sources.py --max-bytes 500_000_000   # skip giant files
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).parent / ".cache" / "raw"
_HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"

# Curated source manifest. Keep it modest — large "haystack" splits are opt-in.
# longmemeval_m (~2.7GB) is intentionally omitted; _s (~278MB) is the standard set.
MANIFEST: dict[str, dict] = {
    "gammacorpus": {
        "repo": "rubenroy/GammaCorpus-Fact-QA-450k",
        "files": ["gammacorpus-fact-qa-450k.jsonl"],
    },
    "longmemeval": {
        "repo": "xiaowu0162/longmemeval",
        "files": ["longmemeval_oracle", "longmemeval_s"],
    },
}


def _remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed HF host
            size = resp.headers.get("x-linked-size") or resp.headers.get("content-length")
            return int(size) if size else None
    except Exception:
        return None


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:  # noqa: S310 - fixed HF host
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.rename(dest)


def download(only: list[str] | None, max_bytes: int | None) -> None:
    names = only or list(MANIFEST)
    for name in names:
        spec = MANIFEST.get(name)
        if spec is None:
            print(f"!! unknown dataset: {name} (known: {list(MANIFEST)})")
            continue
        out_dir = RAW_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in spec["files"]:
            url = _HF.format(repo=spec["repo"], file=fname)
            dest = out_dir / Path(fname).name
            remote = _remote_size(url)
            if dest.exists() and remote is not None and dest.stat().st_size == remote:
                print(f"== {name}/{dest.name}: present ({remote:,} bytes), skipping")
                continue
            if max_bytes is not None and remote is not None and remote > max_bytes:
                print(f"!! {name}/{dest.name}: {remote:,} bytes > --max-bytes {max_bytes:,}, skipping")
                continue
            print(f">> {name}/{dest.name}: downloading {remote or '?':,} bytes ...")
            _download(url, dest)
            print(f"   done -> {dest} ({dest.stat().st_size:,} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download eval source datasets to local cache")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of datasets to fetch (default: all)")
    ap.add_argument("--max-bytes", type=int, default=None, help="Skip files larger than this many bytes")
    args = ap.parse_args()
    download(args.only, args.max_bytes)


if __name__ == "__main__":
    main()
