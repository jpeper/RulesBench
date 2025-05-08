#!/usr/bin/env python3
"""
Aggregate Kingdomino‑style JSON examples into a TSV.

New features
------------
* ``--resize N``  →  if N > 0, the PNG is resized so its longest side is N pixels,
  then re‑encoded **as PNG** (loss‑less compression; only resolution changes).
* ``--md5-out``   →  path to write the MD5 checksum of the TSV (default:
  <output_tsv>.md5).

TSV columns (tab‑separated):
    index   image   question    answer  split

`image` is a JSON list containing **one** Base‑64‑encoded PNG (original bytes if
``--resize`` not used, otherwise the resized PNG).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import List, Tuple

import requests           # pip install requests
from PIL import Image      # pip install pillow

###############################################################################
# Helpers
###############################################################################


def fetch_png_bytes(url: str) -> bytes:
    """Download *url* and return raw PNG bytes.  Warn if suffix ≠ '.png'."""
    if not url.lower().endswith(".png"):
        sys.stderr.write(f"[WARN] URL does not end with '.png': {url}\n")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def resize_png(png_bytes: bytes, longest_side: int) -> bytes:
    """Return PNG bytes resized so max(width, height) == *longest_side*."""
    img = Image.open(io.BytesIO(png_bytes))
    img.thumbnail((longest_side, longest_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def gather_examples(input_dir: Path) -> List[dict]:
    """Load each `*.json` (non‑recursive) into memory."""
    examples: List[dict] = []
    for file in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf‑8"))
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"[WARN] {file.name} invalid JSON – skipped ({exc}).\n")
            continue
        missing = {k for k in ("ID", "Question", "Answer", "game_state_url") if k not in data}
        if missing:
            sys.stderr.write(f"[WARN] {file.name} missing keys: {', '.join(missing)} – skipped.\n")
            continue
        examples.append(data)
    return examples


def write_tsv(
    examples: List[dict],
    out_path: Path,
    *,
    split: str,
    dump_dir: Path | None = None,
    resize: int = 0,
) -> None:
    """Write TSV; optionally dump each (re)‑encoded PNG to *dump_dir*."""
    fieldnames = ["index", "image", "question", "answer", "split"]
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf‑8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        wr.writeheader()

        for ex in examples:
            try:
                png_raw = fetch_png_bytes(ex["game_state_url"])
                if resize > 0:
                    png_raw = resize_png(png_raw, resize)

                if dump_dir:
                    (dump_dir / f"{ex['ID']}.png").write_bytes(png_raw)

                wr.writerow(
                    {
                        "index": ex["ID"],
                        "image": json.dumps([png_to_b64(png_raw)]),
                        "question": ex["Question"],
                        "answer": ex["Answer"],
                        "split": split,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"[ERROR] ID {ex['ID']} failed: {exc}\n")


def write_md5(file_path: Path, md5_out: Path) -> None:
    h = hashlib.md5()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    md5_out.write_text(f"{h.hexdigest()}  {file_path.name}\n", encoding="utf-8")


###############################################################################
# CLI
###############################################################################


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Aggregate JSON → PNG‑preserving TSV.")
    p.add_argument("input_dir", type=Path, help="Directory containing *.json examples")
    p.add_argument("output_tsv", type=Path, help="Where to write the TSV")
    p.add_argument("--split", default="train", help="Value for the 'split' column")
    p.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="Save each PNG to this directory (default: <output_tsv_stem>_images)",
    )
    p.add_argument(
        "--resize",
        type=int,
        default=0,
        metavar="PIX",
        help="If >0, resize PNG so the longest side is PIX pixels (kept in PNG format)",
    )
    p.add_argument(
        "--md5-out",
        type=Path,
        default=None,
        help="File to write MD5 of the TSV (default: <output_tsv>.md5)",
    )
    args = p.parse_args(argv)

    if not args.input_dir.is_dir():
        p.error(f"{args.input_dir} is not a directory.")

    dump_dir = args.dump_dir or args.output_tsv.with_suffix("").with_name(args.output_tsv.stem + "_images")
    md5_file = args.md5_out or args.output_tsv.with_suffix(".md5")

    examples = gather_examples(args.input_dir)
    if not examples:
        sys.stderr.write("No valid JSON files found – nothing to do.\n")
        return

    write_tsv(
        examples,
        args.output_tsv,
        split=args.split,
        dump_dir=dump_dir,
        resize=args.resize,
    )
    write_md5(args.output_tsv, md5_file)

    print(f"Wrote {len(examples)} example(s) to {args.output_tsv}")
    print(f"PNG dumps saved to {dump_dir}")
    print(f"MD5 hash written to {md5_file}")


if __name__ == "__main__":
    main()
