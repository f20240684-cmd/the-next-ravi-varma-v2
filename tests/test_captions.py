"""Caption assembly rules: what makes it into a training caption, and that
captions fit CLIP's 77-token window. (The BLIP models themselves are not run
here.)"""
import pytest

from ravi_varma.data.captions import (
    HeuristicCaptioner,
    StructuredCaption,
    VLMCaptioner,
    _approx_token_count,
    _strip_medium_prefix,
    get_captioner,
)


def _caption(**kw):
    base = dict(
        subject="a woman sitting by a lake", composition="single-figure composition, vertical format",
        pose="sitting", clothing="wearing red sari", jewelry="necklace jewelry", expression="calm expression",
        objects="holding fan", environment="in a garden", lighting="dark low-key tones",
        visual_characteristics="warm red and ochre palette", raw_caption="a woman sitting by a lake",
        title="Lady in the Moonlight", source="vlm",
    )
    base.update(kw)
    return StructuredCaption(**base)


class TestFlatCaption:
    def test_priority_order_and_no_unknowns(self):
        flat = _caption(objects="unknown", jewelry="unknown").as_flat_caption()
        assert flat.startswith("Lady in the Moonlight, a woman sitting by a lake, wearing red sari")
        assert "unknown" not in flat

    def test_trims_lowest_priority_fields_to_fit_budget(self):
        cap = _caption()
        full = cap.as_flat_caption(max_tokens=1000)
        short = cap.as_flat_caption(max_tokens=20)
        assert _approx_token_count(short) <= 20
        assert full.startswith(short)  # trimming only drops trailing (low-priority) fields
        assert "palette" in full and "palette" not in short

    def test_round_trip_through_sidecar_dict(self):
        cap = _caption(qa={"clothing": "red sari"})
        again = StructuredCaption.from_dict({**cap.to_dict(), "flat_caption": "ignored"})
        assert again.as_flat_caption() == cap.as_flat_caption()
        assert again.qa == {"clothing": "red sari"}


class TestAnswerFiltering:
    @pytest.mark.parametrize("answer", ["", "unknown", "yes", "nothing", "in painting", "museum", "phone", "cell phone", "tie"])
    def test_uninformative_meta_and_anachronistic_answers_dropped(self, answer):
        assert not VLMCaptioner._informative(answer)

    @pytest.mark.parametrize("answer", ["sari", "sword", "fan", "veena", "palace", "bow and arrow"])
    def test_period_plausible_answers_kept(self, answer):
        assert VLMCaptioner._informative(answer)

    def test_period_rewrites(self):
        from ravi_varma.data.captions import _period_rewrite

        assert _period_rewrite("a woman with a guitar resting") == "a woman with a stringed instrument resting"
        assert _period_rewrite("a woman in a red sari holding a gun") == "a woman in a red sari"
        assert _period_rewrite("a man holding a sword") == "a man holding a sword"

    def test_medium_prefix_removed(self):
        assert _strip_medium_prefix("a painting of a man and a woman") == "a man and a woman"
        assert _strip_medium_prefix("painting of three women") == "three women"
        assert _strip_medium_prefix("a woman holding a painting") == "a woman holding a painting"


class TestFallbackPolicy:
    def test_heuristic_requires_explicit_opt_in(self, monkeypatch):
        monkeypatch.setattr(VLMCaptioner, "_try_load", lambda self: False)
        with pytest.raises(RuntimeError):
            get_captioner(prefer_vlm=True)
        assert isinstance(get_captioner(prefer_vlm=True, allow_heuristic_fallback=True), HeuristicCaptioner)
        assert isinstance(get_captioner(prefer_vlm=False), HeuristicCaptioner)

    def test_heuristic_caption_has_no_placeholder_text(self, tmp_path):
        from PIL import Image

        path = tmp_path / "x.jpg"
        Image.new("RGB", (300, 400), (150, 60, 30)).save(path)
        cap = HeuristicCaptioner().caption(path)
        assert cap.source == "heuristic"
        assert "no VLM" not in cap.as_flat_caption()
