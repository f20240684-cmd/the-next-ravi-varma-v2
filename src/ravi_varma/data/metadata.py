"""metadata.jsonl helpers: one schema, one reader, one writer.

Every script that touches `data/metadata/metadata.jsonl` (download,
captioning, preparation) goes through these functions so field names and
ordering stay consistent and a partial run can never truncate the file
(writes go to a temp file that is atomically renamed into place).

Record schema (all fields except `image` optional):

    image          path relative to data/raw/
    caption        flat training caption (written by generate_captions.py)
    artist         "Raja Ravi Varma"
    title          human-readable artwork title, cleaned from the source
    commons_file   exact Wikimedia Commons file title ("File:...") -- lets
                   scripts/download_dataset.py --from-metadata re-fetch the
                   identical image
    license        license short name reported by Wikimedia Commons
    source         human-readable provenance string
    year           only if known from the source; never guessed
    source_width,  pixel size of the original file on Commons. Validation
    source_height  checks min_resolution against this, so a local copy that
                   was upscaled cannot hide a low-resolution original.
"""
from __future__ import annotations

import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional

FIELD_ORDER = [
    "image", "caption", "artist", "title", "commons_file", "license", "source", "year", "source_width", "source_height",
]

# Fragments that describe the *file* rather than the artwork.
_TITLE_NOISE = [
    r"^file:",
    r"\s*label\s+QS:.*$",                      # Wikidata QuickStatements residue
    r",?\s*source\s*-.*$",                     # "Source- ebay, Oct 2009"
    r",?\s*from the ravi varma (studio|press).*$",
    r",?\s*ravi varma press$",
    r",?\s*c\.\s*\d{4}'?s?$",
    r"^painting\s+(of\s+)?(\d+\s+)?",
    r"\(\s*\d{4}\s*-\s*\d+\s*\)",              # regnal years "(1652 - 80)"
    r"\.(jpe?g|png|webp|tiff?)$",
    r"raja\s*ravi\s*varma\s*\(1848-1906\)",
    r"^raja\s*ravi\s*varma\s*[,\-–—:]?\s*",
    r"^ravi\s*varma\s*[-,]\s*",
    r"^rajaravivarma\s+",
    r"\bby\s+(raja\s+)?ravi\s*varma\b",
    r"\bby\s+rrv\b",
    r"[-–—]\s*raja\s+ravi\s+varma$",
    r"\braja\s+ravi\s+varma$",
    r"\brrv\b",
    r"\s*-\s*google art project",
    r"\bwellcome\s+v\d+",
    r"\bva\s+ac\s+\w+",
    r"\brcin\s+\d+",
    r"-\s*royal collection",
    r"\((cropped|crop|duplicate|another version)\)",
    r"\bcrop\b",
    r"\([^)]*(oleograph|chromolithograph)[^)]*\)",
    r",\s*raja\s+ravi\s+varma\b",
    r"\(\s*\d+\s*\)$",
    r"\(\d{4}\)",
    r"\d+$",
]


def clean_title(raw: Optional[str]) -> Optional[str]:
    """Turn a Commons file name or an HTML ObjectName into a short artwork
    title, e.g. 'File:Raja Ravi Varma, Shakuntala lost in thoughts (1901).jpg'
    -> 'Shakuntala lost in thoughts'. Returns None if nothing meaningful is
    left. Purely string cleanup: it never adds information."""
    if not raw:
        return None
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = text.replace("_", " ").replace('"', "").strip()
    for _ in range(2):  # second pass catches noise exposed by the first (e.g. "Raja ravivarma painting 50 ...")
        for pattern in _TITLE_NOISE:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE).strip(" ,.-–—")
    text = re.sub(r"\s+", " ", text).strip(" ,.-–—")
    # Split CamelCase file names such as "AshtaSiddhi".
    if " " not in text and re.search(r"[a-z][A-Z]", text):
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text if len(text) >= 3 else None


def extract_year(raw: Optional[str]) -> Optional[int]:
    """Year only when the source states it explicitly as '(1891)' and it is
    within Ravi Varma's working life; otherwise None."""
    if not raw:
        return None
    match = re.search(r"\((1[89]\d{2})\)", raw)
    if match and 1860 <= int(match.group(1)) <= 1910:
        return int(match.group(1))
    return None


def load_metadata(path: "str | Path") -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ordered(record: dict) -> dict:
    ordered = {k: record.get(k) for k in FIELD_ORDER if k in record}
    ordered.update({k: v for k, v in record.items() if k not in ordered})
    return ordered


def save_metadata(path: "str | Path", records: Iterable[dict]) -> None:
    """Atomically rewrite the metadata file, sorted by image name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda r: r["image"])
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".metadata.", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(_ordered(r), ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
