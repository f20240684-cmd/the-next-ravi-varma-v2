#!/usr/bin/env python3
"""
Download Raja Ravi Varma paintings from Wikimedia Commons.

The top-level Raja Ravi Varma category contains subcategories, so this
script recursively searches those subcategories until it has collected
the requested number of images.

Usage:
    python scripts/download_dataset.py --limit 40
    python scripts/download_dataset.py --from-metadata   # re-fetch exactly the files listed in metadata.jsonl

`--from-metadata` is the reproducible path: data/metadata/metadata.jsonl is
committed to git and records the exact Commons file (`commons_file`) behind
every raw image, so a fresh checkout (or a Colab runtime) can rebuild the
identical data/raw/ directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "src"),
)

from ravi_varma.data.metadata import clean_title, extract_year, load_metadata
from ravi_varma.utils.logging import setup_logging


COMMONS_API = "https://commons.wikimedia.org/w/api.php"

ROOT_CATEGORY = "Category:Paintings by Raja Ravi Varma"

HEADERS = {
    "User-Agent": (
        "TheNextRaviVarma/0.1 "
        "(local research project; educational use)"
    )
}

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


def api_get(
    session: requests.Session,
    params: dict,
    logger,
    retries: int = 5,
):
    """
    Make a Wikimedia API request with retry handling for 429/5xx errors.
    """

    for attempt in range(retries):

        try:
            response = session.get(
                COMMONS_API,
                params=params,
                headers=HEADERS,
                timeout=60,
            )

            # Wikimedia rate limit
            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After",
                    "5",
                )

                try:
                    wait_time = int(retry_after)
                except ValueError:
                    wait_time = 5

                wait_time = max(wait_time, 5)

                logger.warning(
                    "Wikimedia rate limit (429). "
                    "Waiting %s seconds...",
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            # Temporary server errors
            if response.status_code in {
                500,
                502,
                503,
                504,
            }:

                wait_time = 5 * (attempt + 1)

                logger.warning(
                    "Wikimedia server error %s. "
                    "Retrying in %s seconds...",
                    response.status_code,
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:

            if attempt == retries - 1:
                raise

            wait_time = 5 * (attempt + 1)

            logger.warning(
                "Request failed: %s. "
                "Retrying in %s seconds...",
                e,
                wait_time,
            )

            time.sleep(wait_time)

    return None


def get_category_contents(
    session: requests.Session,
    category: str,
    logger,
):
    """
    Return all direct files and subcategories inside a Wikimedia category.
    """

    files = []
    subcategories = []

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file|subcat",
        "cmlimit": "500",
        "format": "json",
    }

    while True:

        data = api_get(
            session,
            params,
            logger,
        )

        if not data:
            break

        members = (
            data
            .get("query", {})
            .get("categorymembers", [])
        )

        for member in members:

            namespace = member.get("ns")
            title = member.get("title")

            # Namespace 6 = File
            if namespace == 6:
                files.append(title)

            # Namespace 14 = Category
            elif namespace == 14:
                subcategories.append(title)

        continuation = data.get("continue")

        if not continuation:
            break

        params.update(continuation)

        # Small delay between API requests
        time.sleep(0.5)

    return files, subcategories


def collect_paintings(
    session: requests.Session,
    root_category: str,
    limit: int,
    logger,
):
    """
    Recursively walk Wikimedia categories and collect file titles.
    """

    queue = [root_category]

    visited_categories = set()
    collected_files = []
    seen_files = set()

    while queue and len(collected_files) < limit:

        category = queue.pop(0)

        if category in visited_categories:
            continue

        visited_categories.add(category)

        logger.info(
            "Scanning category: %s",
            category,
        )

        try:
            files, subcategories = get_category_contents(
                session,
                category,
                logger,
            )

        except requests.RequestException as e:

            logger.warning(
                "Could not scan category '%s': %s",
                category,
                e,
            )

            continue

        # Add files
        for file_title in files:

            if file_title in seen_files:
                continue

            seen_files.add(file_title)
            collected_files.append(file_title)

            logger.info(
                "Found painting %d/%d: %s",
                len(collected_files),
                limit,
                file_title,
            )

            if len(collected_files) >= limit:
                break

        # Add subcategories
        for subcategory in subcategories:

            if subcategory not in visited_categories:
                queue.append(subcategory)

        # Don't hammer Wikimedia
        time.sleep(1)

    logger.info(
        "Collected %d painting files "
        "from %d categories.",
        len(collected_files),
        len(visited_categories),
    )

    return collected_files[:limit]


def get_image_info(
    session: requests.Session,
    titles: list[str],
    logger,
):
    """
    Get image URLs and metadata for multiple files in one API request.
    """

    if not titles:
        return []

    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        # A 2048-px rendition is plenty for 448-640 px training and avoids 100 MB+
        # TIFF originals; width/height below still report the ORIGINAL size.
        "iiurlwidth": 2048,
        "format": "json",
    }

    data = api_get(
        session,
        params,
        logger,
    )

    if not data:
        return []

    pages = (
        data
        .get("query", {})
        .get("pages", {})
    )

    results = []

    for page in pages.values():

        imageinfo_list = page.get(
            "imageinfo",
            [],
        )

        if not imageinfo_list:
            continue

        imageinfo = imageinfo_list[0]

        url = imageinfo.get("thumburl") or imageinfo.get("url")

        if not url:
            continue

        ext_meta = imageinfo.get(
            "extmetadata",
            {},
        )

        license_short = (
            ext_meta
            .get("LicenseShortName", {})
            .get("value", "unknown")
        )

        obj_name = (
            ext_meta
            .get("ObjectName", {})
            .get("value", page.get("title", ""))
        )

        results.append(
            {
                "title": page.get("title", ""),
                "url": url,
                "license": license_short,
                "object_name": obj_name,
                "width": imageinfo.get("width"),
                "height": imageinfo.get("height"),
            }
        )

    return results


def download_image(
    session: requests.Session,
    url: str,
    destination: Path,
    logger,
):
    """
    Download one image with retry handling.
    """

    for attempt in range(5):

        try:

            response = session.get(
                url,
                headers=HEADERS,
                timeout=120,
            )

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After",
                    "10",
                )

                try:
                    wait_time = int(retry_after)
                except ValueError:
                    wait_time = 10

                wait_time = max(wait_time, 10)

                logger.warning(
                    "Image server rate limited us. "
                    "Waiting %s seconds...",
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            if not response.content:
                raise RuntimeError(
                    "Downloaded file is empty."
                )

            destination.write_bytes(
                response.content
            )

            return True

        except (
            requests.RequestException,
            RuntimeError,
        ) as e:

            if attempt == 4:

                logger.warning(
                    "Failed to download %s: %s",
                    url,
                    e,
                )

                return False

            wait_time = 5 * (attempt + 1)

            logger.warning(
                "Download failed: %s. "
                "Retrying in %s seconds...",
                e,
                wait_time,
            )

            time.sleep(wait_time)

    return False


def download_from_metadata(session: requests.Session, metadata_path: Path, raw_dir: Path, logger) -> int:
    """Re-download every image listed in metadata.jsonl that is missing from
    raw_dir, using the recorded Commons file title and saving it under the
    recorded `image` name. Returns the number of files still missing."""
    all_records = load_metadata(metadata_path)
    failed = 0
    # Non-Commons records (e.g. Wellcome Collection) carry a direct `download_url`.
    for record in all_records:
        if record.get("download_url") and not (raw_dir / record["image"]).exists():
            logger.info("Downloading %s -> %s", record["download_url"], raw_dir / record["image"])
            if not download_image(session=session, url=record["download_url"], destination=raw_dir / record["image"], logger=logger):
                failed += 1
            time.sleep(1)

    records = [r for r in all_records if r.get("commons_file")]
    missing = [r for r in records if not (raw_dir / r["image"]).exists()]
    logger.info("%d Commons records in %s, %d images missing locally.", len(records), metadata_path, len(missing))

    for start in range(0, len(missing), 20):
        batch = missing[start:start + 20]
        infos = {i["title"]: i for i in get_image_info(session, [r["commons_file"] for r in batch], logger)}
        for record in batch:
            info = infos.get(record["commons_file"])
            if info is None:
                logger.warning("Commons file not found (renamed/deleted?): %s", record["commons_file"])
                failed += 1
                continue
            destination = raw_dir / record["image"]
            logger.info("Downloading %s -> %s", record["commons_file"], destination)
            if not download_image(session=session, url=info["url"], destination=destination, logger=logger):
                failed += 1
            time.sleep(2)  # stay well under Wikimedia's rate limits
        time.sleep(1)
    return failed


WELLCOME_API = "https://api.wellcomecollection.org/catalogue/v2/images"
WELLCOME_LICENSES = {"pdm": "Public Domain Mark", "cc-0": "CC0", "cc-by": "CC BY 4.0"}


def next_free_index(raw_dir: Path) -> int:
    # Names are not contiguous (0000-0239 for 200 files): continue after the highest number.
    numbers = [int(p.stem) for p in raw_dir.iterdir() if p.is_file() and p.stem.isdigit()]
    return max(numbers) + 1 if numbers else 0


def download_wellcome(session: requests.Session, metadata_path: Path, raw_dir: Path, limit: int, min_side: int, logger) -> int:
    """Ravi Varma chromolithographs from the Wellcome Collection (open IIIF API).
    Keeps only open licences and works attributed to Varma; skips scans that are
    already in the dataset via Wikimedia Commons ("... Wellcome V0045115.jpg").
    Every record is tagged medium=chromolithograph so prints can be excluded later."""
    known_text = " ".join(str(r.get("commons_file", "")) + " " + str(r.get("source", "")) for r in load_metadata(metadata_path)) if metadata_path.exists() else ""
    results, page = [], 1
    while True:
        r = session.get(WELLCOME_API, params={"query": "Ravi Varma", "pageSize": 100, "page": page}, timeout=60)
        r.raise_for_status()
        data = r.json()
        results += data.get("results", [])
        if page >= data.get("totalPages", 1):
            break
        page += 1
    index, downloaded = next_free_index(raw_dir), 0
    with metadata_path.open("a", encoding="utf-8") as meta_f:
        for item in results:
            if downloaded >= limit:
                break
            loc = (item.get("locations") or [{}])[0]
            lic = loc.get("license", {}).get("id")
            title = item.get("source", {}).get("title", "")
            image_id = loc.get("url", "").rstrip("/").split("/")[-2] if loc.get("url") else None
            if lic not in WELLCOME_LICENSES or "varma" not in title.lower() or not image_id:
                continue
            if image_id in known_text:
                logger.info("Skipping %s: already in the dataset (%s)", image_id, title[:60])
                continue
            info = session.get(loc["url"], timeout=60).json()
            width, height = info.get("width"), info.get("height")
            if min_side and min(width or 0, height or 0) < min_side:
                logger.info("Skipping %s: %sx%s is below --min-side %d", image_id, width, height, min_side)
                continue
            url = loc["url"].replace("info.json", "full/!2048,2048/0/default.jpg")
            filename = f"{index:04d}.jpg"
            if not download_image(session=session, url=url, destination=raw_dir / filename, logger=logger):
                continue
            work = item.get("source", {}).get("id")
            record = {
                "image": filename, "caption": None, "artist": "Raja Ravi Varma",
                "title": clean_title(title.split(". Chromolithograph")[0]), "license": WELLCOME_LICENSES[lic],
                "source": f"Wellcome Collection (https://wellcomecollection.org/works/{work}, image {image_id}), license: {WELLCOME_LICENSES[lic]}",
                "download_url": url, "medium": "chromolithograph", "year": extract_year(title),
                "source_width": width, "source_height": height,
            }
            meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            meta_f.flush()
            logger.info("Downloaded %s -> %s (%s)", title[:70], filename, WELLCOME_LICENSES[lic])
            index += 1
            downloaded += 1
            time.sleep(1)
    logger.info("Downloaded %d Wellcome images -> %s", downloaded, raw_dir)
    return downloaded


def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Number of images to download.",
    )

    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory for raw images.",
    )

    parser.add_argument(
        "--metadata-file",
        default="data/metadata/metadata.jsonl",
        help="Metadata JSONL file.",
    )

    parser.add_argument(
        "--category",
        default=ROOT_CATEGORY,
        help="Root Wikimedia category.",
    )

    parser.add_argument(
        "--source",
        choices=["commons", "wellcome"],
        default="commons",
        help="commons = Wikimedia Commons category tree; wellcome = Wellcome Collection chromolithographs.",
    )

    parser.add_argument(
        "--min-side",
        type=int,
        default=0,
        help="Skip files whose ORIGINAL shorter side is below this many pixels (e.g. 800).",
    )

    parser.add_argument(
        "--from-metadata",
        action="store_true",
        help="Re-download exactly the files recorded in --metadata-file (reproducible rebuild).",
    )
    args = parser.parse_args()

    logger = setup_logging(
        "download_dataset"
    )

    raw_dir = Path(
        args.raw_dir
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = Path(
        args.metadata_file
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update(
        HEADERS
    )

    if args.from_metadata:
        failed = download_from_metadata(session, metadata_path, raw_dir, logger)
        if failed:
            logger.error("%d images could not be downloaded.", failed)
            return 1
        logger.info("All images listed in %s are present in %s.", metadata_path, raw_dir)
        logger.info("Next: python scripts/validate_dataset.py")
        return 0

    if args.source == "wellcome":
        download_wellcome(session, metadata_path, raw_dir, args.limit, args.min_side, logger)
        logger.info("Next: python scripts/validate_dataset.py")
        return 0

    # ---------------------------------------------------------
    # STEP 1
    # Find paintings recursively
    # ---------------------------------------------------------

    logger.info(
        "Searching Wikimedia Commons recursively..."
    )

    # Files already in the metadata are skipped, so a second run adds NEW paintings
    # instead of re-downloading the first ones the category walk returns.
    known = {r.get("commons_file") for r in load_metadata(metadata_path)} if metadata_path.exists() else set()
    file_titles = collect_paintings(
        session=session,
        root_category=args.category,
        limit=100_000,  # whole category tree; filtered below
        logger=logger,
    )
    file_titles = [t for t in file_titles if t not in known]
    logger.info("%d candidates are not in %s yet.", len(file_titles), metadata_path)

    if not file_titles:

        logger.error(
            "No paintings were found."
        )

        return 1

    logger.info(
        "Found %d candidate paintings.",
        len(file_titles),
    )

    # ---------------------------------------------------------
    # STEP 2
    # Get metadata in batches
    # ---------------------------------------------------------

    image_infos = []

    # Wikimedia allows multiple titles per API call.
    # Use batches of 20 to stay safe.
    batch_size = 20

    for start in range(
        0,
        len(file_titles),
        batch_size,
    ):

        batch = file_titles[
            start:start + batch_size
        ]

        logger.info(
            "Getting metadata for paintings %d-%d...",
            start + 1,
            min(
                start + batch_size,
                len(file_titles),
            ),
        )

        infos = get_image_info(
            session,
            batch,
            logger,
        )

        image_infos.extend(infos)

        time.sleep(1)

    # ---------------------------------------------------------
    # STEP 3
    # Download images
    # ---------------------------------------------------------

    downloaded = 0

    next_index = next_free_index(raw_dir)

    # Open metadata in append mode.
    with metadata_path.open(
        "a",
        encoding="utf-8",
    ) as meta_f:

        for info in image_infos:

            if downloaded >= args.limit:
                break

            if args.min_side and min(info.get("width") or 0, info.get("height") or 0) < args.min_side:
                logger.info("Skipping %s: %sx%s is below --min-side %d", info["title"], info.get("width"), info.get("height"), args.min_side)
                continue

            url = info["url"]

            # -------------------------------------------------
            # Correctly extract extension from URL PATH.
            # Query parameters are NOT included.
            # -------------------------------------------------

            parsed_url = urlparse(url)

            suffix = Path(
                parsed_url.path
            ).suffix.lower()

            if suffix not in ALLOWED_EXTENSIONS:
                suffix = ".jpg"

            filename = (
                f"{next_index:04d}{suffix}"
            )

            destination = (
                raw_dir / filename
            )

            logger.info(
                "Downloading %d/%d: %s",
                downloaded + 1,
                args.limit,
                info["title"],
            )

            success = download_image(
                session=session,
                url=url,
                destination=destination,
                logger=logger,
            )

            if not success:
                continue

            # -------------------------------------------------
            # Save metadata
            # -------------------------------------------------

            record = {
                "image": filename,
                "caption": None,
                "artist": "Raja Ravi Varma",
                "title": clean_title(info["object_name"]) or clean_title(info["title"]),
                "commons_file": info["title"],
                "license": info["license"],
                "source": (
                    "Wikimedia Commons "
                    f"({info['title']}), "
                    f"license: {info['license']}"
                ),
                "year": extract_year(info["title"]),
                "source_width": info.get("width"),
                "source_height": info.get("height"),
            }

            meta_f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            meta_f.flush()

            logger.info(
                "Downloaded %s -> %s (%s)",
                info["title"],
                destination,
                info["license"],
            )

            downloaded += 1
            next_index += 1

            # Important: don't trigger Wikimedia's rate limit.
            time.sleep(2)

    # ---------------------------------------------------------
    # STEP 4
    # Summary
    # ---------------------------------------------------------

    logger.info(
        "Downloaded %d/%d images -> %s",
        downloaded,
        args.limit,
        raw_dir,
    )

    logger.info(
        "Metadata written to %s",
        metadata_path,
    )

    if downloaded > 0:

        logger.info(
            "Next: python scripts/validate_dataset.py"
        )

        return 0

    logger.error(
        "No images were downloaded."
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )