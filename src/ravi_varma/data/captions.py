"""Dense, template-based caption generation.

Each painting gets a *structured* caption produced by two open-weight BLIP
models (Milestone 1 of the proposal: "a VLM ... following a rigid template"):

  * `Salesforce/blip-image-captioning-large` writes one free-form sentence
    describing the scene;
  * `Salesforce/blip-vqa-base` answers a fixed list of questions (people
    count, clothing, jewelry, pose, expression, held objects, setting,
    lighting). Every question and raw answer is stored in the per-image JSON
    sidecar so the caption is auditable.

Answers that carry no information ("unknown", "nothing", a bare "yes") are
dropped rather than guessed at, and person-specific questions are skipped
when the model reports no people in the image. Colour palette and
orientation are *measured* from pixels, not asked.

Captions deliberately describe *content*, not *style*: words like "oil
painting" or "Ravi Varma" are left out so the `<rvvarma>` trigger token is
what absorbs the painterly style during LoRA training. The flat caption is
assembled in priority order and trimmed to fit CLIP's 77-token window.

`torch`/`transformers` are imported lazily so validation and preparation
work without them. If the VLM cannot load, `get_captioner` raises unless the
caller explicitly asks for the `HeuristicCaptioner`, whose pixel-statistics
captions are far too weak to train on (an earlier run trained on exactly
such placeholders).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from ravi_varma.utils.image import safe_open_image
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

CAPTION_FIELDS = [
    "subject",
    "composition",
    "pose",
    "clothing",
    "jewelry",
    "expression",
    "objects",
    "environment",
    "lighting",
    "visual_characteristics",
]

# Order in which fields are dropped last -> first when the flat caption is
# too long for CLIP. Title and the free-form sentence carry the most signal.
FLAT_CAPTION_PRIORITY = [
    "title",
    "raw_caption",
    "clothing",
    "jewelry",
    "pose",
    "expression",
    "objects",
    "environment",
    "composition",
    "lighting",
    "visual_characteristics",
]

CLIP_MAX_TOKENS = 77  # including BOS/EOS
UNINFORMATIVE_ANSWERS = {
    "", "unknown", "none", "nothing", "no", "yes", "nobody", "no one", "not sure", "it", "unclear",
    "painting", "picture", "art", "drawing", "photo",
}
# Answers mentioning the artwork/medium describe the image as an object
# ("in painting", "museum"), not the scene; style belongs to the trigger token.
_META_WORDS = (
    "painting", "picture", "museum", "frame", "gallery", "art", "canvas", "photo", "drawing", "show", "exhibit", "exhibition",
)
# Settings that read naturally with an article ("in a palace"); anything else
# (proper nouns like "india", plurals) is used as "in <answer>".
_COMMON_PLACES = {
    "palace", "garden", "forest", "park", "room", "temple", "court", "courtyard", "field", "lake", "river", "village",
    "house", "jungle", "desert", "battlefield", "street", "hall", "bedroom", "balcony", "cave", "mountain", "beach",
    "hut", "throne room", "kitchen", "pond", "meadow", "clearing", "boat", "chariot", "market",
}
# Objects that cannot appear in a 19th-century painting: when VQA names one
# it is misreading the image (a fan, a flower, a scroll...), so the answer is
# dropped instead of teaching the model a wrong association.
_ANACHRONISMS = (
    "phone", "cell phone", "cellphone", "tuxedo", "gun", "rifle", "pistol", "smoking", "piano", "giraffe", "wii",
    "purse", "selfie", "television", "tv", "remote", "laptop", "computer", "camera", "tie", "bottle", "cup of coffee",
    "frisbee", "tennis", "racket", "skateboard", "surfboard", "kite", "baseball", "bat", "umbrella stand", "toothbrush",
    "scissors", "book bag", "backpack", "handbag", "suitcase", "wii", "controller", "microphone", "guitar", "headphones",
    "cigarette", "sunglasses", "glasses", "watch", "car", "bicycle", "skis", "snowboard", "hot dog", "pizza", "donut",
)
# Period-correct rewrites applied to the BLIP sentence and VQA answers: BLIP
# names Indian string instruments (veena, swarbat, sitar) "guitar", and
# occasionally invents a modern held object that is better removed than kept.
_REWRITES = [
    (r"\bguitars?\b", "stringed instrument"),
    (r"\bplaying (a |the )?violin\b", "playing a stringed instrument"),
    (r",?\s*(holding|with|carrying)\s+(a|an)\s+(gun|rifle|pistol|phone|cell phone|camera)\b", ""),
]


def _period_rewrite(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    for pattern, repl in _REWRITES:
        text = re.sub(pattern, repl, text)
    return text.strip()


_JEWELRY_WORDS = {"necklace", "necklaces", "bracelet", "bracelets", "earrings", "ring", "jewelry", "red necklace"}
_POSTURES = {"standing", "sitting", "lying down", "lying", "kneeling", "walking"}
_COLORS = {
    "red", "blue", "green", "yellow", "white", "black", "orange", "pink", "purple", "brown", "gold", "golden",
    "maroon", "gray", "grey", "silver", "beige", "cream",
}


@dataclass
class StructuredCaption:
    subject: str = "unknown"
    composition: str = "unknown"
    pose: str = "unknown"
    clothing: str = "unknown"
    jewelry: str = "unknown"
    expression: str = "unknown"
    objects: str = "unknown"
    environment: str = "unknown"
    lighting: str = "unknown"
    visual_characteristics: str = "unknown"
    raw_caption: Optional[str] = None  # free-form VLM sentence, if available
    title: Optional[str] = None  # artwork title from provenance metadata (never generated)
    source: str = "heuristic"  # "vlm" | "heuristic"
    qa: dict[str, str] = field(default_factory=dict)  # every VQA question -> raw answer
    measured: dict[str, str] = field(default_factory=dict)  # pixel measurements (lighting/palette/orientation)

    def to_dict(self) -> dict:
        d = {f: getattr(self, f) for f in CAPTION_FIELDS}
        d.update({"raw_caption": self.raw_caption, "title": self.title, "source": self.source, "qa": self.qa, "measured": self.measured})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StructuredCaption":
        """Rebuild from a caption sidecar (see `to_dict`)."""
        names = set(CAPTION_FIELDS) | {"raw_caption", "title", "source", "qa", "measured"}
        return cls(**{k: v for k, v in d.items() if k in names and v is not None})

    @classmethod
    def reassemble(cls, d: dict, title: Optional[str] = None) -> "StructuredCaption":
        """Re-derive a VLM caption from a sidecar's raw VQA answers with the
        current assembly rules. Sidecars written before pixel measurements
        were stored get them recovered from the formatted fields."""
        if d.get("source") != "vlm" or "qa" not in d:
            return cls.from_dict({**d, "title": title})
        measured = d.get("measured") or {
            "lighting": (d.get("lighting") or "").replace(", night scene", "") or "unknown",
            "palette": d.get("visual_characteristics") or "unknown",
            "orientation": (d.get("composition") or "").split(", ")[-1],
        }
        return assemble_caption(d["qa"], d.get("raw_caption"), measured, title)

    def as_text_block(self) -> str:
        """Human-readable structured caption (one heading per field)."""
        return "\n\n".join(f"{f.replace('_', ' ').capitalize()}:\n{getattr(self, f)}" for f in CAPTION_FIELDS) + "\n"

    def _parts(self) -> list[str]:
        parts = []
        for name in FLAT_CAPTION_PRIORITY:
            value = getattr(self, name)
            if value and value.strip() and not value.startswith("unknown"):
                parts.append(value.strip())
        return list(dict.fromkeys(parts))  # de-duplicate, keep order

    def as_flat_caption(self, token_counter: Optional[Callable[[str], int]] = None, max_tokens: int = 70) -> str:
        """Single-line training caption built from informative fields only,
        in priority order, trimmed so `token_counter(caption) <= max_tokens`.
        70 leaves room for BOS/EOS plus the "<rvvarma>, " prefix inside
        CLIP's 77-token window; anything past 77 would be silently cut by
        the tokenizer during training."""
        count = token_counter or _approx_token_count
        parts = self._parts()
        while len(parts) > 1 and count(", ".join(parts)) > max_tokens:
            parts.pop()
        return ", ".join(parts)


def _approx_token_count(text: str) -> int:
    """Rough CLIP BPE count (~1.3 tokens per word plus punctuation) for when
    the real tokenizer isn't available."""
    return int(len(re.findall(r"\w+", text)) * 1.3) + text.count(",")


def clip_token_counter(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5") -> Callable[[str], int]:
    """Token counter backed by the exact tokenizer SD1.5 trains with; falls
    back to an approximation if it can't be loaded."""
    try:
        from transformers import CLIPTokenizer

        tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        return lambda text: len(tok(text, add_special_tokens=False).input_ids)
    except Exception as e:  # offline / transformers missing
        logger.warning("CLIP tokenizer unavailable (%s); using an approximate token count.", e)
        return _approx_token_count


def measure_visual_properties(img: Image.Image) -> dict[str, str]:
    """Properties computed from pixels (never guessed): orientation, overall
    brightness, dominant colour family."""
    w, h = img.size
    thumb = img.convert("RGB").resize((64, 64))
    pixels = list(thumb.getdata())
    avg_r = sum(p[0] for p in pixels) / len(pixels)
    avg_g = sum(p[1] for p in pixels) / len(pixels)
    avg_b = sum(p[2] for p in pixels) / len(pixels)
    brightness = (avg_r + avg_g + avg_b) / 3

    if brightness < 85:
        lighting = "dark low-key tones"
    elif brightness > 170:
        lighting = "bright high-key tones"
    else:
        lighting = "mid-tone lighting"

    if avg_r > avg_g * 1.08 and avg_r > avg_b * 1.08:
        palette = "warm red and ochre palette"
    elif avg_b > avg_r and avg_b > avg_g:
        palette = "cool blue palette"
    elif avg_g > avg_r and avg_g > avg_b:
        palette = "green-dominant palette"
    else:
        palette = "balanced earthy palette"

    aspect = w / h
    if 0.9 <= aspect <= 1.1:
        orientation = "square format"
    elif aspect > 1.1:
        orientation = "landscape format"
    else:
        orientation = "vertical format"
    return {"lighting": lighting, "palette": palette, "orientation": orientation}


class HeuristicCaptioner:
    """Zero-dependency fallback captioner.

    Only reports what it can measure from pixels and leaves every content
    field "unknown". Useful for testing the pipeline offline; NOT suitable
    for training (the captions carry almost no conditioning signal).
    """

    available = True

    def caption(self, image_path: "str | Path", title: Optional[str] = None) -> StructuredCaption:
        img = safe_open_image(image_path, max_side=256)
        if img is None:
            raise ValueError(f"Unreadable image: {image_path}")
        props = measure_visual_properties(img)
        return StructuredCaption(
            composition=props["orientation"],
            lighting=props["lighting"],
            visual_characteristics=props["palette"],
            title=title,
            source="heuristic",
            measured=props,
        )


# Template questions. `None` in `requires_person` means "always ask".
_QUESTIONS = {
    "people_count": "how many people are in the picture?",
    "clothing": "what is the main person wearing?",
    "clothing_color": "what color is the main person's clothing?",
    "has_jewelry": "is the main person wearing jewelry?",
    "jewelry": "what jewelry is the main person wearing?",
    "pose": "is the main person standing, sitting or lying down?",
    "action": "what is the main person doing?",
    "expression": "what is the facial expression of the main person?",
    "has_object": "is the main person holding something?",
    "objects": "what is the main person holding?",
    "environment": "where is this scene taking place?",
    "time": "is it day or night in the picture?",
}
_PERSON_QUESTIONS = {"clothing", "clothing_color", "has_jewelry", "jewelry", "pose", "action", "expression", "has_object", "objects"}
# Follow-up questions only asked when the gating yes/no question says "yes";
# without the gate, VQA models name *some* object/jewel for every image.
_GATED = {"jewelry": "has_jewelry", "objects": "has_object"}
_COUNT_WORDS = {
    "0": 0, "none": 0, "no": 0, "zero": 0, "1": 1, "one": 1, "2": 2, "two": 2, "3": 3, "three": 3,
    "4": 4, "four": 4, "5": 5, "five": 5, "6": 6, "six": 6, "7": 7, "seven": 7, "8": 8, "eight": 8,
}
_NUMBER_NAMES = {2: "two", 3: "three", 4: "four", 5: "five"}


def _parse_count(answer: str) -> Optional[int]:
    if answer in _COUNT_WORDS:
        return _COUNT_WORDS[answer]
    return int(answer) if answer.isdigit() else None


def _strip_medium_prefix(text: str) -> str:
    """'a painting of a woman ...' -> 'a woman ...' (content, not medium)."""
    return re.sub(r"^(an?\s+)?(oil\s+)?(painting|picture|drawing|image|photo)\s+of\s+", "", text).strip()


class VLMCaptioner:
    """BLIP caption + BLIP-VQA template captioner. Loads lazily; `available`
    reports whether both models actually loaded."""

    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-large",
        vqa_model_id: Optional[str] = "Salesforce/blip-vqa-base",
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.vqa_model_id = vqa_model_id
        self.device = device
        self._processor = self._model = None
        self._vqa_processor = self._vqa_model = None
        self.available = self._try_load()

    def _try_load(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor
        except Exception as e:  # broad: covers ImportError and broken/partial installs
            logger.warning("torch/transformers not usable (%s); VLM captioning unavailable.", e)
            return False

        try:
            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device).eval()
            if self.vqa_model_id:
                self._vqa_processor = BlipProcessor.from_pretrained(self.vqa_model_id)
                self._vqa_model = BlipForQuestionAnswering.from_pretrained(self.vqa_model_id).to(self.device).eval()
            return True
        except Exception as e:  # network errors, disk space, etc.
            logger.warning("Could not load VLM captioner (%s / %s): %s", self.model_id, self.vqa_model_id, e)
            return False

    def _raw_caption(self, image: Image.Image) -> str:
        import torch

        inputs = self._processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=40, num_beams=3, repetition_penalty=1.3)
        text = self._processor.decode(out[0], skip_special_tokens=True).strip()
        # BLIP-large often prefixes "there is" / "arafed" artefacts.
        text = re.sub(r"^(there (is|are)|arafed|araffe)\s+", "", text)
        text = re.sub(r"\b(arafed|araffe)\s+", "", text)
        return text

    def _ask(self, image: Image.Image, question: str) -> str:
        import torch

        inputs = self._vqa_processor(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._vqa_model.generate(**inputs, max_new_tokens=12)
        return self._vqa_processor.decode(out[0], skip_special_tokens=True).strip().lower()

    @staticmethod
    def _informative(answer: Optional[str]) -> bool:
        if not answer or answer in UNINFORMATIVE_ANSWERS:
            return False
        if any(re.search(rf"\b{re.escape(w)}s?\b", answer) for w in _ANACHRONISMS):
            return False
        return not any(re.search(rf"\b{w}s?\b", answer) for w in _META_WORDS)

    def caption(self, image_path: "str | Path", title: Optional[str] = None) -> StructuredCaption:
        if not self.available:
            raise RuntimeError("VLMCaptioner is not available; see the warning logged at load time.")

        # BLIP works at 384px; decoding a 100MP scan at full size is wasteful.
        img = safe_open_image(image_path, max_side=768)
        if img is None:
            raise ValueError(f"Unreadable image: {image_path}")
        img.thumbnail((768, 768))

        props = measure_visual_properties(img)
        base = _strip_medium_prefix(self._raw_caption(img))
        qa: dict[str, str] = {}
        if self._vqa_model is not None:
            qa["people_count"] = self._ask(img, _QUESTIONS["people_count"])
            has_people = _parse_count(qa["people_count"]) != 0
            for key, question in _QUESTIONS.items():
                if key == "people_count" or (key in _PERSON_QUESTIONS and not has_people):
                    continue
                if key in _GATED and qa.get(_GATED[key]) != "yes":
                    continue
                qa[key] = self._ask(img, question)

        return assemble_caption(qa, base, props, title)


def assemble_caption(qa: dict[str, str], raw_caption: Optional[str], measured: dict[str, str], title: Optional[str]) -> StructuredCaption:
    """Turn raw VQA answers + the BLIP sentence + pixel measurements into a
    StructuredCaption. Kept separate from the models so captions can be
    re-assembled from saved sidecars (`generate_captions.py --reflatten`)
    whenever these rules change, without re-running the VLM."""

    def answer(key: str) -> Optional[str]:
        value = _period_rewrite(qa.get(key))
        return value if VLMCaptioner._informative(value) else None

    raw_caption = _period_rewrite(raw_caption)
    count = _parse_count(qa.get("people_count", ""))
    orientation = measured.get("orientation", "")
    if count == 1:
        composition = f"single-figure composition, {orientation}"
    elif count is not None and count > 1:
        composition = f"group of {_NUMBER_NAMES.get(count, 'many')} figures, {orientation}"
    else:
        composition = orientation
    composition = composition.strip(", ") or "unknown"

    clothing = answer("clothing")
    if clothing in _JEWELRY_WORDS:  # "what is she wearing?" -> "necklace" belongs to the jewelry field
        clothing = None
    color = answer("clothing_color")
    if clothing in _COLORS:  # "what is she wearing?" -> "blue"
        clothing = f"{clothing} clothing"
    if clothing and color in _COLORS and color not in clothing:
        clothing = f"{color} {clothing}"
    jewelry = answer("jewelry")
    posture, action = answer("pose"), answer("action")
    if action in _POSTURES:  # the action is just a (possibly conflicting) posture
        action = None
    pose = ", ".join(p for p in (posture, action) if p) or None
    expression = answer("expression")
    objects = answer("objects")
    environment = answer("environment")
    if environment and not environment.startswith(("in ", "on ", "at ", "by ", "near ")):
        article = "a " if environment in _COMMON_PLACES else ""
        environment = f"in {article}{environment}"
    lighting = measured.get("lighting", "unknown") + (", night scene" if qa.get("time") == "night" else "")

    return StructuredCaption(
        subject=raw_caption or "unknown",
        composition=composition,
        pose=pose or "unknown",
        clothing=f"wearing {clothing}" if clothing else "unknown",
        jewelry=f"{jewelry} jewelry" if jewelry and "jewel" not in jewelry else (jewelry or "unknown"),
        expression=f"{expression} expression" if expression and "expression" not in expression else (expression or "unknown"),
        objects=f"holding {objects}" if objects else "unknown",
        environment=environment or "unknown",
        lighting=lighting,
        visual_characteristics=measured.get("palette", "unknown"),
        raw_caption=raw_caption,
        title=title,
        source="vlm",
        qa=qa,
        measured=measured,
    )


def get_captioner(
    prefer_vlm: bool = True,
    model_id: str = "Salesforce/blip-image-captioning-large",
    vqa_model_id: Optional[str] = "Salesforce/blip-vqa-base",
    device: str = "cpu",
    allow_heuristic_fallback: bool = False,
):
    """Return the VLM captioner. If it cannot load, raise unless
    `allow_heuristic_fallback` (or `prefer_vlm=False`) explicitly opts in to
    the much weaker pixel-statistics captioner."""
    if prefer_vlm:
        vlm = VLMCaptioner(model_id=model_id, vqa_model_id=vqa_model_id, device=device)
        if vlm.available:
            return vlm
        if not allow_heuristic_fallback:
            raise RuntimeError(
                "VLM captioner could not be loaded (see warning above). Install torch/transformers and "
                "allow the BLIP weights to download, or pass --no-vlm to knowingly use heuristic captions."
            )
        logger.warning("Falling back to HeuristicCaptioner -- these captions are NOT suitable for training.")
    return HeuristicCaptioner()
