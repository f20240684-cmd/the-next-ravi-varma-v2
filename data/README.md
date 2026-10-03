# `data/` — dataset layout

```
data/
├── raw/          # source images (NOT in git; rebuild with download_dataset.py --from-metadata)
├── metadata/     # metadata.jsonl (provenance + captions) and validation_report.json   [in git]
├── captions/     # <stem>.json structured captions with every VQA question/answer       [in git]
├── processed/    # aspect-bucketed training JPEGs (not in git) + manifest.jsonl          [manifest in git]
└── reference/    # 12 held-out paintings for evaluation (not in git) + reference.jsonl   [list in git]
```

## Pipeline

```
download_dataset.py --from-metadata  ->  data/raw/
validate_dataset.py                  ->  data/metadata/validation_report.json  (accepted / excluded + reasons)
generate_captions.py                 ->  data/captions/*.json + caption field in metadata.jsonl
prepare_dataset.py                   ->  data/processed/*.jpg + manifest.jsonl, data/reference/
```

`prepare_dataset.py` re-runs validation and prepares exactly the accepted images, so
curation decisions always apply.

## Current corpus (Wikimedia Commons, `Category:Paintings by Raja Ravi Varma`)

| Stage | Images |
|---|---|
| Downloaded | 200 |
| Excluded — local copy upscaled from an original < 384 px | 26 |
| Excluded — curated non-paintings (stamps, gallery photos, multi-panel grids, printed text)¹ | 7 |
| Excluded — duplicate copy of the same artwork (kept the highest-resolution copy) | 33 |
| **Accepted** | **134** |
| Held out for evaluation (`data/reference/`) | 12 |
| **Training set** | **122** |

¹ 10 images are on the curated exclusion list; 3 of them were already excluded as low-resolution upscales.

Licenses: 193 of 200 are public domain; the rest are CC BY 4.0 / CC BY-SA (2.0, 4.0) /
GODL-India, recorded per image in `metadata.jsonl`.

Every exclusion and its reason is listed in `metadata/validation_report.json`; the manual
decisions (and why) live in `configs/dataset.yaml` under `curation:`.

## `metadata/metadata.jsonl` schema

```json
{"image": "0001.jpg", "caption": "...", "artist": "Raja Ravi Varma", "title": "Menaka and Shakuntala",
 "commons_file": "File:Raja Ravi Varma, Menaka and Sakunthala (1891).jpg", "license": "Public domain",
 "source": "Wikimedia Commons (...), license: Public domain", "year": 1891,
 "source_width": 2386, "source_height": 3417}
```

- `image` (required): path relative to `data/raw/`.
- `caption`: written by `generate_captions.py`.
- `title`: cleaned from the Commons object name / file name (string cleanup only, never invented).
- `commons_file`: exact Commons file — lets `download_dataset.py --from-metadata` re-fetch the identical image.
- `year`: only when the source states it explicitly as `(1891)`; otherwise `null`.
- `source_width/height`: size of the original on Commons. Validation checks the minimum
  resolution against this, so an upscaled local copy cannot hide a low-resolution original.

## Captions

Each `captions/<stem>.json` holds the BLIP sentence, the answers to a fixed list of BLIP-VQA
questions (people count, clothing, colour, jewelry, posture, action, expression, held object,
setting, day/night), pixel measurements (orientation, brightness, palette) and the final flat
caption. Captions describe *content*; the painterly style is left to the `<rvvarma>` trigger
token. Known VQA failure modes (setting answered as "painting"/"museum", anachronistic objects
such as "phone") are filtered; remaining errors can be corrected by hand in the sidecar and
re-assembled with `generate_captions.py --reflatten`.

## Adding images

Only use sources whose license you can verify and record. Drop the file in `data/raw/`, add a
metadata line with at least `image`, `source` and `license`, then re-run validate → captions → prepare.
