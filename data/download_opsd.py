"""Fetch the pinned OPSD snapshot that `prepare_dataset.py` derives from.

    python data/download_opsd.py

You almost certainly do not need this. `data/processed/representative_days.csv`
is committed, and everything in this repository runs from it. This script exists
so the committed file is **reproducible** rather than something that has to be
taken on trust.

The download is ~130 MB. The snapshot is pinned to a dated release rather than
`latest` because OPSD stopped updating the package, and because a dataset that
changes underneath a committed derivation is not a derivation.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

SNAPSHOT = "2020-10-06"
URL = (
    f"https://data.open-power-system-data.org/time_series/{SNAPSHOT}/"
    "time_series_60min_singleindex.csv"
)
EXPECTED_SHA256 = "6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8"
EXPECTED_BYTES = 130_339_665

OUT_PATH = Path(__file__).parent / "raw" / "time_series_60min_singleindex.csv"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path) -> None:
    """A snapshot that does not match its hash is not the snapshot."""
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise ValueError(
            f"{path} is {size:,} bytes, expected {EXPECTED_BYTES:,}. This is not "
            f"the pinned {SNAPSHOT} snapshot and the derived dataset would differ."
        )
    actual = sha256_of(path)
    if actual != EXPECTED_SHA256:
        raise ValueError(
            f"{path} has sha256 {actual}, expected {EXPECTED_SHA256}. Refusing to "
            "let a different file masquerade as the pinned snapshot."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check an existing file against the pinned hash and stop",
    )
    args = parser.parse_args(argv)

    if not args.verify_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}\n  -> {args.out}  (~130 MB, this takes a while)")
        urllib.request.urlretrieve(URL, args.out)  # noqa: S310 - pinned https URL

    if not args.out.exists():
        raise FileNotFoundError(args.out)

    verify(args.out)
    print(f"verified: {args.out} matches the pinned {SNAPSHOT} snapshot")
    print("now run: python data/prepare_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
