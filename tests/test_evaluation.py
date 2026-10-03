import json
from pathlib import Path

import pytest
from PIL import Image

from ravi_varma.evaluation.clip_score import clip_available, text_image_similarity
from ravi_varma.evaluation.report import batch_evaluate, evaluate_image
from ravi_varma.evaluation.style_similarity import compute_style_similarity


def _make_image(path, color=(80, 40, 20)):
    Image.new("RGB", (64, 64), color).save(path)


class TestMetricCalculationsWithoutCLIP:
    """In this sandbox, open_clip/torch are not installed, so these tests
    exercise the real 'CLIP unavailable' code path -- verifying we return
    None rather than a fabricated score, per the project spec."""

    def test_clip_reports_unavailable(self):
        availability = clip_available()
        # This sandbox has no torch/open_clip installed; if a future
        # environment *does* have them, this simply confirms the flag is
        # a bool with a reason when False.
        assert isinstance(availability.available, bool)
        if not availability.available:
            assert availability.reason

    def test_text_image_similarity_returns_none_without_clip(self, tmp_path):
        img_path = tmp_path / "img.jpg"
        _make_image(img_path)
        if clip_available().available:
            pytest.skip("CLIP is available in this environment; None-path not exercised.")
        assert text_image_similarity(img_path, "a painting") is None

    def test_style_similarity_returns_none_without_clip(self, tmp_path):
        img_path = tmp_path / "img.jpg"
        _make_image(img_path)
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        _make_image(ref_dir / "ref1.jpg")
        if clip_available().available:
            pytest.skip("CLIP is available in this environment; None-path not exercised.")
        result = compute_style_similarity(img_path, ref_dir)
        assert result.mean_similarity is None
        assert result.num_references_used == 0


class TestMissingImages:
    def test_evaluate_image_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            evaluate_image(tmp_path / "does_not_exist.png", tmp_path / "reference")

    def test_batch_evaluate_on_directory_with_no_images(self, tmp_path):
        empty_dir = tmp_path / "empty_generated"
        empty_dir.mkdir()
        results = batch_evaluate(empty_dir, tmp_path / "reference", tmp_path / "report.csv")
        assert results == []


class TestEmptyReferenceDirectory:
    def test_empty_reference_dir_returns_no_references_used(self, tmp_path):
        img_path = tmp_path / "gen.png"
        _make_image(img_path)
        ref_dir = tmp_path / "empty_reference"
        ref_dir.mkdir()
        result = compute_style_similarity(img_path, ref_dir)
        assert result.num_references_used == 0
        assert result.mean_similarity is None

    def test_nonexistent_reference_dir_handled_gracefully(self, tmp_path):
        img_path = tmp_path / "gen.png"
        _make_image(img_path)
        result = compute_style_similarity(img_path, tmp_path / "does_not_exist_dir")
        assert result.num_references_used == 0


class TestSidecarPromptLoading:
    def test_evaluate_image_reads_prompt_from_sidecar_json(self, tmp_path):
        img_path = tmp_path / "gen.png"
        _make_image(img_path)
        sidecar = tmp_path / "gen.json"
        sidecar.write_text(json.dumps({"prompt": "a royal portrait"}))
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()

        result = evaluate_image(img_path, ref_dir)
        assert result.prompt == "a royal portrait"

    def test_evaluate_image_no_prompt_no_sidecar_sets_none(self, tmp_path):
        img_path = tmp_path / "gen2.png"
        _make_image(img_path)
        ref_dir = tmp_path / "reference2"
        ref_dir.mkdir()

        result = evaluate_image(img_path, ref_dir)
        assert result.prompt is None
        assert result.text_alignment is None


class TestRunEvaluation:
    """Run-level evaluation with real CLIP (skipped if unavailable)."""

    def _gen_dir(self, root, name, lora, colors):
        import numpy as np

        d = root / name
        d.mkdir()
        rng = np.random.default_rng(len(name))
        for pid, color in colors.items():
            for seed in (1, 2):
                arr = np.clip(np.array(color) + rng.integers(-40, 40, (64, 64, 3)), 0, 255).astype("uint8")
                Image.fromarray(arr).save(d / f"{pid}_seed{seed}.png")
                (d / f"{pid}_seed{seed}.json").write_text(json.dumps(
                    {"prompt": f"<rvvarma>, {pid}", "prompt_id": pid, "seed": seed, "lora_loaded": lora, "lora_scale": 1.0}
                ))
        return d

    def test_metrics_paired_comparison_and_outputs(self, tmp_path):
        if not clip_available().available:
            pytest.skip("CLIP not available")
        from ravi_varma.evaluation.run_eval import evaluate_run

        lora = self._gen_dir(tmp_path, "lora", True, {"red_portrait": (180, 40, 30), "green_garden": (40, 160, 50)})
        base = self._gen_dir(tmp_path, "base", False, {"red_portrait": (90, 90, 90), "green_garden": (100, 100, 100)})
        ref = tmp_path / "reference"
        ref.mkdir()
        _make_image(ref / "r1.jpg", (170, 50, 40))
        _make_image(ref / "r2.jpg", (60, 150, 60))

        run = evaluate_run(lora, ref, tmp_path / "eval", baseline_dir=base)
        assert run.num_images == 4 and run.num_references == 2
        assert all(r.prompt_adherence is not None and r.style_ref_mean is not None for r in run.images)
        assert set(run.per_prompt) == {"red_portrait", "green_garden"}
        assert all(v["diversity"] is not None and v["diversity"] >= 0 for v in run.per_prompt.values())
        assert run.baseline_comparison["num_pairs"] == 4
        for name in ("metrics.csv", "summary.json", "qualitative_review.csv", "contact_sheet.jpg", "contact_sheet_baseline.jpg"):
            assert (tmp_path / "eval" / name).exists()

    def test_review_sheet_is_never_overwritten(self, tmp_path):
        if not clip_available().available:
            pytest.skip("CLIP not available")
        from ravi_varma.evaluation.run_eval import evaluate_run

        lora = self._gen_dir(tmp_path, "lora", True, {"p": (120, 60, 30)})
        evaluate_run(lora, tmp_path / "noref", tmp_path / "eval")
        sheet = tmp_path / "eval" / "qualitative_review.csv"
        sheet.write_text(sheet.read_text() + "reviewer,notes\n")
        evaluate_run(lora, tmp_path / "noref", tmp_path / "eval")
        assert sheet.read_text().endswith("reviewer,notes\n")

    def test_trigger_token_stripped_before_clip(self):
        from ravi_varma.evaluation.clip_score import strip_trigger_tokens

        assert strip_trigger_tokens("<rvvarma>, royal portrait") == "royal portrait"
        assert strip_trigger_tokens("royal portrait") == "royal portrait"
