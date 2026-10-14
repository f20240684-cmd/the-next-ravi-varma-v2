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

## Current corpus (v3, 2026-10-14)

Two open-licence sources:
- **Wikimedia Commons**, `Category:Paintings by Raja Ravi Varma` (whole category tree, 261 files)
- **Wellcome Collection**, Ravi Varma chromolithographs via its open IIIF API (`download_dataset.py --source wellcome`, 39 files)

The second download (`--min-side 800`) skipped originals under 800 px, so the new images are much sharper:
median short side **2,400 px**, versus 605 px for the first 200.

| Stage | Images |
|---|---|
| Downloaded | 300 (200 first download + 61 Commons + 39 Wellcome) |
| Excluded — local copy upscaled from an original < 384 px | 26 |
| Excluded — curated non-paintings (stamps, gallery photos, a sculpture photo, multi-panel grids, printed text)¹ | 8 |
| Excluded — duplicate copy of the same artwork (kept the highest-resolution copy)² | 72 |
| **Accepted** | **194** (v2: 134) |
| Held out for evaluation (`data/reference/`, unchanged) | 12 |
| **Training set** | **182** (v2: 122) |

¹ 11 images are on the curated exclusion list; 3 of them were already excluded as low-resolution upscales.

² 3 were caught automatically by the image hash. The rest are curated groups in `configs/dataset.yaml`; the groups
added for v3 were found with CLIP image similarity (0.89–0.98) and confirmed by eye. Commons and Wellcome often hold
several scans or print runs of the same picture, e.g. 7 copies of the *Saraswati* oleograph.

**Prints.** The 29 accepted Wellcome images are chromolithographs (`"medium": "chromolithograph"` in
`metadata.jsonl`) with a printed title under the picture. Their captions name it ("printed title text in the bottom
margin") so the trigger word does not learn the text, the same fix that removed the gilt frames in v2.

**Captions.** Every training caption fits CLIP's window: at most 69 tokens plus the 6-token `<rvvarma>, ` prefix.
Descriptive Commons "titles" are cut to their first sentence, and Wikidata and Malayalam-script residue is removed.

Licenses: 250 public domain, 39 Public Domain Mark (Wellcome), the rest CC BY 4.0 / CC BY-SA (2.0, 4.0) /
GODL-India, recorded per image in `metadata.jsonl`.

The **v2 LoRA** in this repository was trained on the v2 set (122 images); v3 is prepared and packaged
(`dist/ravi_varma_dataset.zip`) for the next Colab training run.

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
