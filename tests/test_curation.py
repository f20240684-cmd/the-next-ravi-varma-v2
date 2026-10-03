"""Curation, duplicate resolution, upscale detection, hold-out split and
aspect-ratio bucketing -- the parts of the dataset pipeline that decide
what the LoRA is actually trained on."""
import json

import numpy as np
import pytest
from PIL import Image

from ravi_varma.config import DatasetConfig
from ravi_varma.data.dataset import BucketBatchSampler, RaviVarmaImageCaptionDataset, load_manifest, prepare_dataset
from ravi_varma.data.metadata import clean_title, extract_year, load_metadata, save_metadata
from ravi_varma.data.validation import DatasetValidator, check_manifest
from ravi_varma.utils.image import aspect_buckets, nearest_bucket, trim_uniform_border


def _textured(size, seed):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (size[1] // 16 + 1, size[0] // 16 + 1, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.BILINEAR)


@pytest.fixture
def corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _textured((600, 800), 1).save(raw / "a.jpg")
    _textured((600, 800), 2).save(raw / "b.jpg")
    _textured((800, 600), 3).save(raw / "c.jpg")
    # d is the same artwork as a, at higher resolution (curated duplicate group)
    _textured((900, 1200), 1).save(raw / "d.jpg")
    # e is a near-identical copy of b (automatic near-duplicate detection)
    _textured((600, 800), 2).resize((590, 790)).save(raw / "e.jpg")
    # f is a 500px local upscale of a 200px original
    _textured((500, 700), 6).save(raw / "f.jpg")
    _textured((600, 600), 7).save(raw / "stamp.jpg")

    meta = tmp_path / "metadata.jsonl"
    records = [{"image": n, "caption": f"caption for {n}", "source": "test"} for n in ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "stamp.jpg"]]
    records.append({"image": "f.jpg", "caption": "upscaled", "source": "test", "source_width": 200, "source_height": 280})
    save_metadata(meta, records)

    return DatasetConfig.load(
        {
            "raw_dir": str(raw),
            "processed_dir": str(tmp_path / "processed"),
            "captions_dir": str(tmp_path / "captions"),
            "metadata_file": str(meta),
            "manifest_file": str(tmp_path / "processed" / "manifest.jsonl"),
            "reference_dir": str(tmp_path / "reference"),
            "curation": {
                "exclude": {"stamp.jpg": "postage stamp", "missing.jpg": "not on disk"},
                "duplicate_groups": [["a.jpg", "d.jpg"]],
                "holdout": ["c.jpg"],
                "caption_notes": {"b.jpg": "inside a gilt frame"},
            },
        }
    )


class TestCuration:
    def test_accepted_and_excluded_sets(self, corpus):
        report = DatasetValidator(corpus).validate()
        assert sorted(report.accepted) == ["b.jpg", "c.jpg", "d.jpg"]
        assert report.excluded["stamp.jpg"].startswith("curated_exclusion")
        assert report.excluded["a.jpg"].startswith("curated_duplicate")  # lower-res copy of d
        assert report.excluded["e.jpg"].startswith("near_duplicate")
        assert report.excluded["f.jpg"].startswith("upscaled_low_resolution")
        assert any(i.category == "stale_curation_entry" and i.path == "missing.jpg" for i in report.issues)

    def test_report_json_is_auditable(self, corpus, tmp_path):
        validator = DatasetValidator(corpus)
        report = validator.validate()
        validator.save_report(report, tmp_path / "report.json")
        saved = json.loads((tmp_path / "report.json").read_text())
        assert saved["valid_samples"] == 3
        assert {d["kept"] for d in saved["duplicate_sets"]} == {"d.jpg", "b.jpg"}

    def test_prepare_uses_accepted_and_holds_out_reference(self, corpus):
        report = DatasetValidator(corpus).validate()
        manifest = prepare_dataset(corpus, accepted=report.accepted)
        assert sorted(r.source_image for r in manifest) == ["b.jpg", "d.jpg"]
        reference = [json.loads(line) for line in open(f"{corpus.reference_dir}/reference.jsonl")]
        assert [r["source_image"] for r in reference] == ["c.jpg"]
        assert next(r for r in manifest if r.source_image == "b.jpg").text.endswith(", inside a gilt frame")
        for rec in manifest:
            assert rec.text.startswith("<rvvarma>, ")
            assert (rec.width, rec.height) == Image.open(rec.image).size
            assert rec.width % 64 == 0 and rec.height % 64 == 0

    def test_prepare_clears_stale_processed_files(self, corpus):
        prepare_dataset(corpus, accepted=["b.jpg"])
        prepare_dataset(corpus, accepted=["d.jpg"])
        names = sorted(p.name for p in __import__("pathlib").Path(corpus.processed_dir).glob("*.jpg"))
        assert names == ["d.jpg"]

    def test_manifest_check_flags_placeholder_and_missing_trigger(self, corpus, tmp_path):
        manifest = prepare_dataset(corpus, accepted=["b.jpg", "d.jpg"])
        lines = [json.dumps({**manifest[0].to_dict(), "text": "no trigger here"}),
                 json.dumps({**manifest[1].to_dict(), "text": "<rvvarma>, unknown (no VLM available)"})]
        bad = tmp_path / "bad_manifest.jsonl"
        bad.write_text("\n".join(lines) + "\n")
        report = check_manifest(bad, "<rvvarma>")
        assert report.valid_samples == 0
        reasons = " ".join(report.excluded.values())
        assert "missing_trigger" in reasons and "placeholder_caption" in reasons
        assert check_manifest(corpus.manifest_file, "<rvvarma>").valid_samples == 2


class TestBuckets:
    def test_buckets_are_valid_sd_sizes(self):
        for w, h in aspect_buckets(512):
            assert w % 64 == 0 and h % 64 == 0
            assert 0.85 * 512 * 512 <= w * h <= 1.15 * 512 * 512

    def test_typical_portrait_canvas_maps_to_448x640(self):
        assert nearest_bucket(1000, 1420, aspect_buckets(512)) == (448, 640)
        assert nearest_bucket(1000, 1000, aspect_buckets(512)) == (512, 512)

    def test_bucket_sampler_never_mixes_sizes(self, corpus):
        prepare_dataset(corpus, accepted=["b.jpg", "c.jpg", "d.jpg"])
        manifest = load_manifest(corpus.manifest_file)
        ds = RaviVarmaImageCaptionDataset(manifest, keep_aspect=True)
        sampler = BucketBatchSampler(ds, batch_size=2, seed=0)
        seen = []
        for batch in sampler:
            assert len({ds.size_of(i) for i in batch}) == 1
            seen += batch
        assert sorted(seen) == list(range(len(ds)))


class TestImageHelpers:
    def test_trim_uniform_border_removes_white_margin(self):
        painting = _textured((200, 300), 9)
        framed = Image.new("RGB", (240, 340), "white")
        framed.paste(painting, (20, 20))
        trimmed = trim_uniform_border(framed)
        assert abs(trimmed.width - 200) <= 2 and abs(trimmed.height - 300) <= 2

    def test_trim_leaves_unbordered_image_alone(self):
        img = _textured((200, 300), 10)
        assert trim_uniform_border(img).size == img.size


class TestMetadata:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("File:Raja Ravi Varma, Shakuntala lost in thoughts (1901).jpg", "Shakuntala lost in thoughts"),
            ("File:Sri Krishna as Envoy (cropped).jpg", "Sri Krishna as Envoy"),
            ("File:Raja Ravi Varma, Vasantika (oleographic print).jpg", "Vasantika"),
            ("Hamsa Damayanti label QS:Len,Hamsa Damayanti label QS:Lsi,x", "Hamsa Damayanti"),
            ('<div class="fn">\n"Shakuntala looking back"</div>', "Shakuntala looking back"),
            ("File:AshtaSiddhi.jpg", "Ashta Siddhi"),
            (None, None),
        ],
    )
    def test_clean_title(self, raw, expected):
        assert clean_title(raw) == expected

    def test_extract_year_only_when_explicit(self):
        assert extract_year("File:Raja Ravi Varma, Birth of Krishna (1890).jpg") == 1890
        assert extract_year("File:Painting of Maharana Raj Singh - I (1652 - 80).jpg") is None
        assert extract_year("File:Shakuntala.jpg") is None

    def test_save_metadata_sorts_and_round_trips(self, tmp_path):
        path = tmp_path / "m.jsonl"
        save_metadata(path, [{"image": "b.jpg", "caption": "x"}, {"image": "a.jpg", "title": "t"}])
        rows = load_metadata(path)
        assert [r["image"] for r in rows] == ["a.jpg", "b.jpg"]
        assert list(rows[0].keys())[0] == "image"
