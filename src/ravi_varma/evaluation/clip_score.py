"""Text-image alignment via CLIP cosine similarity.

Uses `open_clip` if installed; if it (or torch) is unavailable, every
function returns `None` rather than fabricating a plausible-looking score --
per the project spec, we never invent evaluation metrics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: dict = {}


def _load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
    key = (model_name, pretrained, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        import open_clip
        import torch
    except Exception:
        logger.warning("open_clip/torch not installed or not usable; CLIP-based metrics are unavailable.")
        return None

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
        _MODEL_CACHE[key] = (model, preprocess, tokenizer, torch)
        return _MODEL_CACHE[key]
    except Exception as e:
        logger.warning("Could not load CLIP model '%s' (%s): %s", model_name, pretrained, e)
        return None


@dataclass
class ClipAvailability:
    available: bool
    reason: Optional[str] = None


def clip_available(device: str = "cpu") -> ClipAvailability:
    loaded = _load_clip(device=device)
    if loaded is None:
        return ClipAvailability(
            False,
            "open_clip/torch not installed, or the pretrained CLIP weights could not be "
            "downloaded (requires network access to the model hub the first time).",
        )
    return ClipAvailability(True)


TRIGGER_PATTERN = re.compile(r"<[^<>\s]+>,?\s*")


def strip_trigger_tokens(text: str) -> str:
    """Remove `<token>` style triggers before scoring: CLIP has never seen
    them, so they only add noise to the text embedding."""
    return TRIGGER_PATTERN.sub("", text).strip(" ,")


def text_image_similarity(image: "str | Path | Image.Image", text: str, device: str = "cpu") -> Optional[float]:
    """Cosine similarity between a CLIP text embedding of `text` (trigger
    tokens removed) and a CLIP image embedding of `image`, in [-1, 1] (in
    practice usually [0.15, 0.40] for CLIP ViT-B/32). Returns None if CLIP
    cannot be loaded."""
    loaded = _load_clip(device=device)
    if loaded is None:
        return None
    model, preprocess, tokenizer, torch = loaded

    img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    image_input = preprocess(img).unsqueeze(0).to(device)
    text_input = tokenizer([strip_trigger_tokens(text)]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    return float(similarity)


def image_embedding(image: "str | Path | Image.Image", device: str = "cpu"):
    """Return a normalized CLIP image embedding (numpy array), or None."""
    loaded = _load_clip(device=device)
    if loaded is None:
        return None
    model, preprocess, _tokenizer, torch = loaded

    img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    image_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).cpu().numpy()


def embed_images(images: "list[str | Path | Image.Image]", device: str = "cpu", batch_size: int = 16):
    """Normalized CLIP embeddings for many images as an (N, D) numpy array,
    or None if CLIP is unavailable."""
    loaded = _load_clip(device=device)
    if loaded is None:
        return None
    model, preprocess, _tokenizer, torch = loaded
    import numpy as np

    out = []
    for i in range(0, len(images), batch_size):
        batch = [im if isinstance(im, Image.Image) else Image.open(im).convert("RGB") for im in images[i:i + batch_size]]
        with torch.no_grad():
            feats = model.encode_image(torch.stack([preprocess(im) for im in batch]).to(device))
            feats /= feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), dtype="float32")


def embed_texts(texts: list[str], device: str = "cpu"):
    """Normalized CLIP text embeddings (trigger tokens removed), or None."""
    loaded = _load_clip(device=device)
    if loaded is None:
        return None
    model, _preprocess, tokenizer, torch = loaded
    with torch.no_grad():
        feats = model.encode_text(tokenizer([strip_trigger_tokens(t) for t in texts]).to(device))
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()
