"""Pose adherence for ControlNet (Milestone 4): does the generated image keep
the reference pose?

OpenPose is run on both images; each reference person is matched to the
generated person whose keypoints are closest, and we report PCK -- the
fraction of the reference's visible body keypoints that reappear within
`threshold` (in coordinates normalised to image width/height). 1.0 = every
joint in place, 0.0 = no person / no joint near where it should be.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

Person = list[Optional[tuple[float, float]]]  # 18 OpenPose body joints, None if not detected


def pose_keypoints(image: Image.Image) -> list[Person]:
    """Normalised body keypoints of every person OpenPose finds in `image`."""
    from ravi_varma.generation.controlnet import _openpose_detector

    detector = _openpose_detector()
    if not detector.available:
        raise RuntimeError("OpenPose detector unavailable; cannot score pose adherence.")
    poses = detector._detector.detect_poses(np.array(image.convert("RGB")))
    return [[(k.x, k.y) if k is not None else None for k in p.body.keypoints] for p in poses]


def _person_pck(ref: Person, gen: Person, threshold: float) -> float:
    visible = [(r, g) for r, g in zip(ref, gen) if r is not None]
    if not visible:
        return 0.0
    hits = sum(g is not None and np.hypot(r[0] - g[0], r[1] - g[1]) <= threshold for r, g in visible)
    return hits / len(visible)


def pose_pck(reference: list[Person], generated: list[Person], threshold: float = 0.1) -> Optional[float]:
    """Mean over reference people of their best-matching generated person's PCK.
    None when the reference has no detectable person (nothing to compare)."""
    if not reference:
        return None
    return float(np.mean([max((_person_pck(r, g, threshold) for g in generated), default=0.0) for r in reference]))
