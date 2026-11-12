"""Edge-side simple prefilter: runs on edge device to decide what to upload.

Design principle
----------------
The edge prefilter uses ONLY computationally cheap signals derived directly
from the student model's forward pass (absolute confidence, detection count)
to make a coarse upload decision.  No historical baseline is maintained on
the edge — that would defeat the purpose of keeping the edge simple.

The cloud-side BHSS then performs fine-grained selection from the uploaded
pool using multi-signal scoring with historical baseline comparison.

This two-tier design:
1. Simulates real deployment constraints (edge CPU/GPU limited)
2. Avoids uploading obviously-easy frames (bandwidth saving)
3. Leaves the difficult differentiation work to the cloud

Credibility note
----------------
The edge prefilter uses *absolute* avg_conf, while BHSS's conf_drop uses
*relative* confidence drop from a sliding baseline.  They are conceptually
orthogonal signals:

  - Edge:    "Is the model uncertain about this frame right now?"
  - BHSS:    "Did confidence degrade compared to recent history?"

A frame from a consistently dark scene may have low avg_conf (passes edge)
but conf_drop ≈ 0 (low BHSS score) — the edge filter alone cannot
differentiate it from a frame where confidence genuinely collapsed.
BHSS provides that differentiation.

All compared methods (random, confidence, behavior_drift, ours) operate on
the same edge-prefiltered candidate pool, so comparisons remain fair.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_edge_prefilter(
    scores: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Mark each frame as edge_pass=True/False based on simple criteria.

    Parameters
    ----------
    scores : DataFrame
        Must contain an 'avg_conf' column (output of compute_hard_scores).
    cfg : dict
        The ``sampling`` sub-dict from the experiment config, optionally
        containing an ``edge_prefilter`` block.

    Returns
    -------
    DataFrame
        A copy of *scores* with an added boolean column ``edge_pass``.
    """
    prefilter_cfg = cfg.get("edge_prefilter", {})
    enabled = bool(prefilter_cfg.get("enabled", False))

    scores = scores.copy()
    if not enabled:
        scores["edge_pass"] = True
        return scores

    ptype = str(prefilter_cfg.get("type", "confidence"))

    if ptype == "confidence":
        threshold = float(prefilter_cfg.get("conf_threshold", 0.70))
        scores["edge_pass"] = scores["avg_conf"] < threshold
    else:
        # Unknown type → pass everything (safe default).
        scores["edge_pass"] = True

    n_pass = int(scores["edge_pass"].sum())
    n_total = len(scores)
    print(
        f"[edge_prefilter] type={ptype}  conf_threshold={threshold:.2f}  "
        f"pass={n_pass}/{n_total} ({100 * n_pass / max(1, n_total):.1f}%)"
    )
    return scores


def save_edge_prefiltered_images(
    scores: pd.DataFrame,
    out_dir: Path,
    hard_link: bool = True,
) -> list[Path]:
    """Save edge-passing images so later experiments can reuse them.

    Prefers hard-links (same filesystem, no extra disk usage); falls back
    to a physical copy when the source and destination are on different
    volumes.

    Parameters
    ----------
    scores : DataFrame
        Must contain ``path`` and ``edge_pass`` columns.
    out_dir : Path
        Destination directory (created if needed).  Existing files with
        the same name are overwritten.
    hard_link : bool
        If True, attempt ``Path.hardlink_to`` before falling back to copy.

    Returns
    -------
    list[Path]
        Absolute paths of the saved images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    edge_pass = scores[scores.get("edge_pass", True)]

    for _, row in edge_pass.iterrows():
        src = Path(row["path"])
        if not src.exists():
            continue
        dst = out_dir / src.name
        try:
            dst.unlink(missing_ok=True)
            if hard_link:
                try:
                    dst.hardlink_to(src)
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
            saved.append(dst)
        except Exception:
            # last-resort fallback
            try:
                shutil.copy2(src, dst)
                saved.append(dst)
            except Exception:
                pass

    print(f"[edge_prefilter] saved {len(saved)} images → {out_dir}")
    return saved


def load_cached_edge_prefilter_images(
    cache_dir: Path,
) -> list[Path]:
    """Return sorted list of *.jpg paths in *cache_dir*.

    Returns an empty list when the directory does not exist or is empty.
    """
    if not cache_dir.exists():
        return []
    return sorted(p for p in cache_dir.glob("*.jpg") if p.is_file())


def cached_edge_prefilter_available(cache_dir: Path, min_images: int = 10) -> bool:
    """Check whether a usable edge-prefilter cache exists."""
    paths = load_cached_edge_prefilter_images(cache_dir)
    return len(paths) >= max(1, int(min_images))


def edge_prefilter_summary(scores: pd.DataFrame) -> dict:
    """Return a one-line summary dict for logging / decision JSON."""
    n_total = len(scores)
    n_pass = int(scores["edge_pass"].sum()) if "edge_pass" in scores.columns else n_total
    return {
        "edge_prefilter_enabled": bool(
            (scores.get("edge_pass", True) != True).any()
        ),
        "edge_candidate_frames": n_total,
        "edge_pass_frames": n_pass,
        "edge_pass_ratio": float(n_pass / max(1, n_total)),
    }
