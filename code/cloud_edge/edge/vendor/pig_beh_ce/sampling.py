from __future__ import annotations

from pathlib import Path
import ast
import gc
import shutil
import time

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from .files import list_images, list_videos, read_image, safe_clear_dir
from .yolo_ops import image_predict_behavior_stats, image_predict_behavior_stats_from_array


def _behavior_counts(behavior_hist, det_count: float) -> dict:
    """Extract per-class counts and non-lie ratio from normalized histogram.

    ``behavior_hist`` is a 3-element normalized histogram [p_stand, p_lie, p_eat].
    Multiplying by ``det_count`` recovers the raw per-class counts exactly.
    """
    total = max(int(det_count), 1)
    stand = int(round(float(behavior_hist[0]) * det_count))
    lie   = int(round(float(behavior_hist[1]) * det_count))
    eat   = int(round(float(behavior_hist[2]) * det_count))
    # Clamp in case of rounding
    diff = total - (stand + lie + eat)
    if diff != 0:
        lie = max(0, lie + diff)
    non_lie = stand + eat
    ratio = non_lie / float(total) if total > 0 else 0.0
    return {
        'stand_count': stand,
        'lie_count': lie,
        'eat_count': eat,
        'total_count': total,
        'non_lie_ratio': ratio,
    }


def dhash(image_bgr: np.ndarray, hash_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).astype(np.uint8).reshape(-1)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def robust_norm_abs(value: float, ref: float, eps: float = 1e-6) -> float:
    return float(abs(value - ref) / (abs(ref) + eps))


def _clip_score(value: float, limit: float | None) -> float:
    if limit is None or float(limit) <= 0:
        return float(value)
    return float(min(max(value, 0.0), float(limit)))


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(value)


def should_compute_tta_uncertainty(cfg: dict, weights: dict) -> bool:
    strategy = str(cfg.get('strategy', 'ours')).strip().lower()
    if strategy in ('uncertainty', 'tta_uncertainty'):
        return True
    enabled = _as_bool(cfg.get('enable_tta_uncertainty'), default=True)
    return enabled and float(weights.get('tta_uncertainty', 0.0)) > 0.0


def _hist_from_value(value, nc: int) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            value = []
    arr = np.asarray(value if value is not None else [], dtype=np.float32)
    if arr.size != nc:
        out = np.zeros(nc, dtype=np.float32)
        out[:min(nc, arr.size)] = arr[:min(nc, arr.size)]
        arr = out
    s = float(arr.sum())
    return arr / s if s > 0 else arr


def tta_behavior_uncertainty(model: YOLO, image_path: Path, imgsz: int, nc: int) -> float:
    img = read_image(image_path)
    variants = [
        img,
        cv2.convertScaleAbs(img, alpha=1.0, beta=18),
        cv2.convertScaleAbs(img, alpha=1.0, beta=-18),
        cv2.GaussianBlur(img, (5, 5), 0),
    ]
    hists, confs, det_counts = [], [], []
    for v in variants:
        res = model.predict(v, imgsz=imgsz, verbose=False)
        hist = np.zeros(nc, dtype=np.float32)
        frame_confs = []
        det_count = 0
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item()) if b.cls is not None else 0
                if 0 <= cls_id < nc:
                    hist[cls_id] += 1.0
                frame_confs.append(float(b.conf.item()))
                det_count += 1
        if hist.sum() > 0:
            hist = hist / hist.sum()
        hists.append(hist)
        confs.append(float(np.mean(frame_confs)) if frame_confs else 0.0)
        det_counts.append(det_count)
    hist_var = float(np.mean(np.var(np.stack(hists, axis=0), axis=0))) if hists else 0.0
    conf_var = float(np.var(confs))
    det_var = float(np.var(det_counts) / max(np.mean(det_counts) + 1e-6, 1.0))
    return hist_var + conf_var + 0.1 * det_var


def save_sampled_frames_from_videos(videos_dir: Path, out_dir: Path, sample_every: int, max_frames: int | None = None, progress_every: int = 200) -> pd.DataFrame:
    safe_clear_dir(out_dir)
    rows = []
    global_index = 0
    saved = 0
    sample_every = max(1, int(sample_every))
    if max_frames is not None and int(max_frames) <= 0:
        max_frames = None
    for video_path in list_videos(videos_dir):
        print(f'[INFO] sampling video: {video_path.name}, sample_every={sample_every}, max_frames={max_frames}')
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f'[WARN] failed to open video: {video_path}')
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        local_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            local_idx += 1
            if local_idx % sample_every != 0:
                continue
            out_name = f'{video_path.stem}_f{local_idx:08d}.jpg'
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), frame)
            rows.append({'path': str(out_path), 'frame_index': global_index, 'video_path': str(video_path), 'video_frame_index': local_idx, 'timestamp_sec': float(local_idx / fps) if fps > 0 else None})
            global_index += 1
            saved += 1
            if progress_every > 0 and saved % int(progress_every) == 0:
                print(f'[INFO] sampled frames: {saved} (latest={video_path.name}:{local_idx})')
            if max_frames is not None and saved >= int(max_frames):
                cap.release()
                return pd.DataFrame(rows)
        cap.release()
    return pd.DataFrame(rows)


def compute_hard_scores_for_paths(student_path: Path, frame_table: pd.DataFrame, imgsz: int, cfg: dict) -> pd.DataFrame:
    model = YOLO(str(student_path))
    try:
        nc = int(cfg.get('num_classes', 3))
        baseline_window = int(cfg.get('baseline_window', 100))
        weights = cfg.get('weights', {})
        compute_tta = should_compute_tta_uncertainty(cfg, weights)
        component_clip = cfg.get('score_component_clip', 1.0)
        rows = []
        history_confs, history_area, history_entropy, history_hists, history_counts = [], [], [], [], []
        warmup_min = max(5, baseline_window // 5)

        for _, item in frame_table.iterrows():
            p = Path(item['path'])
            stats = image_predict_behavior_stats(model, p, imgsz=imgsz, nc=nc)
            cur_hist = np.asarray(stats['behavior_hist'], dtype=np.float32)
            if len(history_confs) < warmup_min:
                ref_conf = stats['avg_conf'] if stats['avg_conf'] > 0 else 1.0
                ref_area = stats['area_mean'] if stats['area_mean'] > 0 else 1.0
                ref_count = stats['det_count'] if stats['det_count'] > 0 else 1.0
                ref_hist = cur_hist.copy()
                ref_entropy = stats['behavior_entropy']
            else:
                start = max(0, len(history_confs) - baseline_window)
                ref_conf = float(np.median(history_confs[start:]))
                ref_area = float(np.median(history_area[start:]))
                ref_count = float(np.median(history_counts[start:]))
                ref_entropy = float(np.median(history_entropy[start:]))
                ref_hist = np.mean(np.stack(history_hists[start:], axis=0), axis=0)
                if ref_hist.sum() > 0:
                    ref_hist = ref_hist / ref_hist.sum()

            conf_drop = _clip_score(max(0.0, (ref_conf - stats['avg_conf']) / max(ref_conf, 1e-6)), component_clip)
            area_drift = _clip_score(robust_norm_abs(stats['area_mean'], ref_area), component_clip)
            count_drift = _clip_score(robust_norm_abs(stats['det_count'], ref_count), component_clip)
            behavior_drift = _clip_score(float(np.abs(cur_hist - ref_hist).sum() / 2.0), component_clip)
            entropy_rise = _clip_score(max(0.0, float(stats['behavior_entropy'] - ref_entropy)), component_clip)
            tta_uncertainty = _clip_score(tta_behavior_uncertainty(model, p, imgsz=imgsz, nc=nc), component_clip) if compute_tta else 0.0
            # BHSS v2: 3-signal score (conf_drop + count_drift + behavior_drift).
            # entropy_rise and area_drift are still computed for analysis but excluded
            # from the selection score because:
            #   - entropy_rise is highly correlated with behavior_drift (redundant)
            #   - area_drift mainly reflects pig growth, not model difficulty
            hard_score = (
                float(weights.get('conf_drop', 0.50)) * conf_drop +
                float(weights.get('count_drift', 0.25)) * count_drift +
                float(weights.get('behavior_drift', 0.25)) * behavior_drift +
                float(weights.get('tta_uncertainty', 0.00)) * tta_uncertainty
            )
            row = item.to_dict()
            row.update({
                **{k: v for k, v in stats.items() if k != 'behavior_hist'},
                'behavior_hist': cur_hist.tolist(),
                **_behavior_counts(cur_hist, stats['det_count']),
                'ref_conf': ref_conf,
                'ref_area': ref_area,
                'ref_count': ref_count,
                'ref_behavior_hist': ref_hist.tolist(),
                'conf_drop': conf_drop,
                'area_drift': area_drift,
                'count_drift': count_drift,
                'behavior_drift': behavior_drift,
                'entropy_rise': entropy_rise,
                'tta_uncertainty': tta_uncertainty,
                'hard_score': hard_score,
            })
            rows.append(row)
            history_confs.append(stats['avg_conf'])
            history_area.append(stats['area_mean'])
            history_counts.append(stats['det_count'])
            history_entropy.append(stats['behavior_entropy'])
            history_hists.append(cur_hist)
        return pd.DataFrame(rows)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_hard_scores(student_path: Path, image_dir: Path, imgsz: int, cfg: dict) -> pd.DataFrame:
    images = list_images(image_dir)[::max(1, int(cfg.get('sample_every', 1)))]
    frame_table = pd.DataFrame([{'path': str(p), 'frame_index': i} for i, p in enumerate(images)])
    return compute_hard_scores_for_paths(student_path, frame_table, imgsz=imgsz, cfg=cfg)


def compute_hard_scores_from_videos(student_path: Path, videos_dir: Path, cache_dir: Path, imgsz: int, cfg: dict) -> pd.DataFrame:
    frame_table = save_sampled_frames_from_videos(videos_dir, cache_dir, sample_every=int(cfg.get('video_sample_every', cfg.get('sample_every', 30))), max_frames=cfg.get('max_video_frames_per_stage', None), progress_every=int(cfg.get('progress_every', 200)))
    if frame_table.empty:
        return frame_table
    return compute_hard_scores_for_paths(student_path, frame_table, imgsz=imgsz, cfg=cfg)


def _drift_upload_single(
    frame_row: dict,
    upload_dir: Path | None,
    selected_rows: list[dict],
    max_upload: int,
) -> int:
    """Upload a single extreme-confidence frame immediately.

    Not throttle-gated (deliberate safety valve); the caller still records
    its ``frame_index`` as the last-upload anchor for the batch gate.
    """
    if upload_dir is None:
        return 0
    remaining = max_upload - len(selected_rows) if max_upload > 0 else 1
    if remaining <= 0:
        return 0
    src = Path(str(frame_row.get('path', '')))
    if src.exists():
        dst = upload_dir / src.name
        try:
            dst.unlink(missing_ok=True)
            shutil.copy2(str(src), str(dst))
            frame_row['path'] = str(dst)
        except Exception:
            pass
    frame_row['selected'] = True
    selected_rows.append(frame_row)
    return 1


def _min_interval_frames(adaptive_cfg: dict, sample_every: int) -> int:
    """Resolve the drift-upload throttle in sampled-frame units.

    ``upload_min_interval_frames`` (explicit, deterministic) wins; when it is
    absent the value is derived from the legacy wall-clock
    ``upload_min_interval_s`` at the dataset's nominal 25 fps
    (15.0 s at sample_every=20 -> 19 sampled frames).
    """
    frames = int(adaptive_cfg.get('upload_min_interval_frames', 0) or 0)
    if frames > 0:
        return frames
    secs = float(adaptive_cfg.get('upload_min_interval_s', 15.0))
    return max(1, int(round(secs * 25.0 / max(1, int(sample_every)))))


def _drift_upload_batch(
    drift_frames: list[dict],
    upload_dir: Path | None,
    selected_rows: list[dict],
    max_upload: int,
    min_interval_frames: int,
    last_upload_idx: int | None,
    p10_threshold: float | None = None,
    h_med_threshold: float | None = None,
    min_frame_gap: int = 3,
    random_ratio: float = 0.0,
    seed: int = 42,
    non_lie_ratio_threshold: float = 0.0,
    dbds: DBDSSelector | None = None,
) -> int:
    """Upload drift frames that satisfy the selection condition.

    **Selection modes** (controlled by parameters):

    0. **DBDS** (``dbds`` given): accept frames whose class content pulls
       the upload-set histogram toward the reference distribution
       (linearized chi-square marginal gain > tau).  Per-class quotas
       emerge from saturation -- no class names involved.

    1. **Random sampling** (``random_ratio > 0``):
       Randomly sample ``random_ratio`` fraction of drift frames.

    2. **Dual-signal filtering — entropy** (``p10_threshold`` and ``h_med_threshold`` set,
       ``non_lie_ratio_threshold == 0``):

           conf < P_10  AND  behavior_entropy > H_med

    3. **Dual-signal filtering — activity** (``p10_threshold`` set,
       ``non_lie_ratio_threshold > 0``):

           conf < P_10  AND  non_lie_ratio > threshold

       Uses the proportion of non-lying pigs (stand + eat) as the second signal
       instead of behavior entropy.  ``non_lie_ratio > 0.3`` means >30% of
       detected pigs are standing or eating — a more discriminative signal
       than entropy when the lie class dominates.

    4. **Full upload** (both thresholds None, ``random_ratio == 0``):
       Every drift frame is uploaded (subject to gap/cap).

    Adjacent frames are skipped via ``min_frame_gap`` (minimum difference
    in ``frame_index`` between consecutive uploaded frames).
    The upload throttle is a frame-index gate: a window starting within
    ``min_interval_frames`` sampled frames of the previous upload event is
    skipped (deterministic replacement for the old wall-clock
    ``upload_min_interval_s`` gate, which made upload decisions depend on
    machine speed and broke cross-batch reproducibility).
    """
    import random as _random

    if not drift_frames or upload_dir is None:
        return 0

    # Frame-index gate (skip if too soon after last upload)
    if last_upload_idx is not None and min_interval_frames > 0:
        first_idx = int(drift_frames[0].get('frame_index', -1))
        if first_idx - last_upload_idx < min_interval_frames:
            return 0

    # -- Frame selection ------------------------------------------
    if dbds is not None:
        # DBDS: accept frames whose class content pulls the upload-set
        # histogram toward the reference distribution (saturation makes
        # per-class quotas emerge automatically -- no class names involved).
        filtered = [item for item in drift_frames if dbds.accept_row(item)]
    elif random_ratio > 0.0:
        # Random sampling within drift window (B_random baseline).
        # Shuffle deterministically with stage+batch seed, then take
        # the first random_ratio fraction.
        _rng = _random.Random(seed + len(selected_rows))
        shuffled = list(drift_frames)
        _rng.shuffle(shuffled)
        n_random = max(1, int(len(shuffled) * random_ratio))
        filtered = shuffled[:n_random]
        # Sort by frame_index for consistent temporal ordering
        filtered.sort(key=lambda x: int(x.get('frame_index', 0)))
    else:
        # Dual-signal filter (or full-upload passthrough)
        _p10 = float(p10_threshold) if p10_threshold is not None else None
        _h_med = float(h_med_threshold) if h_med_threshold is not None else None
        _nlr_thresh = float(non_lie_ratio_threshold) if non_lie_ratio_threshold > 0 else None
        filtered = []
        for item in drift_frames:
            conf = float(item.get('avg_conf', 1.0))
            # Signal 1 — model uncertainty: confidence below adaptive baseline
            conf_ok = conf < _p10 if _p10 is not None else True

            # Signal 2 — behavior / activity (mutually exclusive paths):
            if _nlr_thresh is not None:
                # NEW: non_lie_ratio > threshold (pig activity signal)
                nlr = float(item.get('non_lie_ratio', 0.0))
                sig2_ok = nlr > _nlr_thresh
            elif _h_med is not None:
                # ORIGINAL: behavior_entropy > H_med (behavior diversity signal)
                ent = float(item.get('behavior_entropy', 0.0))
                sig2_ok = ent > _h_med
            else:
                sig2_ok = True

            if conf_ok and sig2_ok:
                filtered.append(item)
    if not filtered:
        return 0

    # Global cap
    remaining = max_upload - len(selected_rows) if max_upload > 0 else len(filtered)
    if remaining <= 0:
        return 0

    # -- Upload with temporal gap -----------------------------
    last_uploaded_idx: int | None = None
    if selected_rows:
        last_uploaded_idx = int(selected_rows[-1].get('frame_index', -99999))

    uploaded = 0
    for item in filtered:
        if uploaded >= remaining:
            break
        # Skip if too close to previous uploaded frame
        idx = int(item.get('frame_index', -1))
        if last_uploaded_idx is not None and abs(idx - last_uploaded_idx) < min_frame_gap:
            continue
        src = Path(str(item.get('path', '')))
        if src.exists():
            dst = upload_dir / src.name
            try:
                dst.unlink(missing_ok=True)
                shutil.copy2(str(src), str(dst))
                item['path'] = str(dst)
            except Exception:
                pass
        item['selected'] = True
        selected_rows.append(item)
        if dbds is not None:
            dbds.update_row(item)
        last_uploaded_idx = idx
        uploaded += 1

    return uploaded


class DBDSSelector:
    """Distribution-Balanced Drift Screening (DBDS) online selection state.

    Maintains the running class histogram ``H`` of frames uploaded in the
    current stage and accepts a frame when its class content pulls ``H``
    toward the reference distribution ``p``.  The acceptance score is the
    linearized marginal gain of the chi-square divergence to ``p``::

        g(h) = sum_c h_c * (p_c - q_c) / p_c,   q = H / |H|  (q = 0 if |H| = 0)

    Accept when ``g > tau``.  The rule saturates automatically: once a
    class' upload share reaches ``p_c``, further frames of that class get
    zero/negative gain -- per-class quotas emerge without naming any class.
    All operations act on class-histogram indices, so the method is
    class-agnostic (any ``num_classes``).

    A small diversity reservoir (herding-style insert/evict against the
    same reference) is maintained on every scored frame and persisted
    across stages, so the minimum-upload fallback can be topped up with
    under-represented content even when the live stream is degenerate
    (e.g. night stages where ~95% of frames are one class).
    """

    def __init__(
        self,
        num_classes: int,
        reference: str = 'uniform',
        tau: float = 1.0,
        reservoir_size: int = 200,
        reservoir_dir: Path | None = None,
        tag: str = '',
    ):
        self.nc = max(1, int(num_classes))
        # Only 'uniform' supported for now (1/K each); source-prior option
        # can be added by passing an explicit vector here.
        self.p = np.full(self.nc, 1.0 / self.nc, dtype=np.float64)
        self.tau = float(tau)
        self.reservoir_size = max(0, int(reservoir_size))
        self.reservoir_dir = reservoir_dir
        self.tag = str(tag or '')
        # Running histogram of frames uploaded this stage
        self.H = np.zeros(self.nc, dtype=np.float64)
        self.N = 0.0
        # Diversity reservoir (persistent across stages)
        self._reservoir: list[dict] = []
        self._res_H = np.zeros(self.nc, dtype=np.float64)
        self._res_N = 0.0
        if self.reservoir_dir is not None:
            self.reservoir_dir.mkdir(parents=True, exist_ok=True)
            self._load_reservoir()

    # -- Per-frame scoring ------------------------------------------

    def _counts(self, row) -> np.ndarray:
        """Class detection counts of a frame (histogram index based)."""
        hist = _hist_from_value(row.get('behavior_hist'), self.nc)
        det = float(row.get('det_count', 0.0) or 0.0)
        return hist * max(det, 0.0)

    def _gain(self, h: np.ndarray, H: np.ndarray, N: float) -> float:
        """Linearized marginal gain of chi-square divergence to reference p."""
        q = H / N if N > 0 else np.zeros_like(H)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = (self.p - q) / self.p
            w[~np.isfinite(w)] = 0.0
        return float(np.sum(h * w))

    def accept_row(self, row: dict) -> bool:
        """True if this frame pulls the upload histogram toward reference."""
        h = self._counts(row)
        g = self._gain(h, self.H, self.N)
        row['dbds_gain'] = g
        return g > self.tau

    def update_row(self, row: dict) -> None:
        """Record an actually-uploaded frame into the running histogram."""
        h = self._counts(row)
        self.H += h
        self.N += float(h.sum())

    # -- Diversity reservoir (herding-maintained, persistent) -------

    def _reservoir_csv(self) -> Path:
        return self.reservoir_dir / 'reservoir_rows.csv'

    def _load_reservoir(self) -> None:
        csv_path = self._reservoir_csv()
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return
        for _, r in df.iterrows():
            p = Path(str(r.get('path', '')))
            if p.exists():
                row = r.to_dict()
                self._reservoir.append(row)
                h = self._counts(row)
                self._res_H += h
                self._res_N += float(h.sum())
        if self._reservoir:
            print(f'[dbds] reservoir loaded: {len(self._reservoir)} frames from {csv_path}')

    def save(self) -> None:
        if self.reservoir_dir is None or not self._reservoir:
            return
        cols = ['path', 'frame_index', 'behavior_hist', 'det_count', 'dbds_gain']
        pd.DataFrame([{c: r.get(c) for c in cols} for r in self._reservoir]).to_csv(
            self._reservoir_csv(), index=False, encoding='utf-8-sig')

    def reservoir_update(self, row: dict) -> None:
        """Offer a scored frame to the diversity reservoir (O(cap) worst case)."""
        if self.reservoir_size <= 0 or row.get('selected'):
            return
        h = self._counts(row)
        src = Path(str(row.get('path', '')))
        store = dict(row)
        if self.reservoir_dir is not None and src.exists():
            # Persistent copy: the reservoir survives stage frame-cache clears
            # and can be consumed by later stages' fallback.
            _name = f'{self.tag}_{src.name}' if self.tag else src.name
            dst = self.reservoir_dir / _name
            try:
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
                store['path'] = str(dst)
            except Exception:
                pass
        if len(self._reservoir) < self.reservoir_size:
            self._reservoir.append(store)
            self._res_H += h
            self._res_N += float(h.sum())
            return
        # Replace the least-useful member (lowest current gain vs reservoir dist)
        worst_i, worst_g = -1, float('inf')
        for i, r in enumerate(self._reservoir):
            g = self._gain(self._counts(r), self._res_H, self._res_N)
            if g < worst_g:
                worst_i, worst_g = i, g
        g_new = self._gain(h, self._res_H, self._res_N)
        if worst_i >= 0 and g_new > worst_g:
            old = self._reservoir[worst_i]
            h_old = self._counts(old)
            self._res_H -= h_old
            self._res_H += h
            self._res_N += float(h.sum()) - float(h_old.sum())
            try:
                Path(str(old.get('path', ''))).unlink(missing_ok=True)
            except Exception:
                pass
            self._reservoir[worst_i] = store

    def reservoir_ranked(self, exclude_paths: set[str] | None = None) -> list[dict]:
        """Reservoir frames ranked by current gain vs the upload histogram.

        Frames whose class content is under-represented in the *upload set*
        rank first -- the fallback quota consumes the top of this list.
        """
        exclude_paths = exclude_paths or set()
        scored: list[tuple[float, dict]] = []
        for r in self._reservoir:
            if str(r.get('path', '')) in exclude_paths:
                continue
            g = self._gain(self._counts(r), self.H, self.N)
            r['dbds_gain'] = g
            scored.append((g, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in scored]


def compute_hard_scores_from_videos_streaming(
    student_path: Path,
    videos_dir: Path,
    frame_cache_dir: Path,
    imgsz: int,
    cfg: dict,
    upload_dir: Path | None = None,
    seed: int = 42,
    videos: list[Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream-process videos with per-block BHSS selection for upload.

    Frames flow directly from ``cv2.VideoCapture.read()`` [U+FFFD]?YOLO (array,
    no disk read) [U+FFFD]?BHSS scoring [U+FFFD]?per-block ranking [U+FFFD]?upload only top-K.

    This is the full edge-side pipeline:

    1. Read frame from video (every *sample_every* frames)
    2. YOLO inference on in-memory numpy array [U+FFFD]?**no disk read**
    3. Update sliding-window baseline, compute BHSS hard_score
    4. Compute dhash from array (for dedup [U+FFFD]?no disk read)
    5. Write frame to *frame_cache_dir* (write-only, for later export)
    6. Buffer frame metadata in memory
    7. Every *block_size* frames: rank by hard_score, select top-K,
       copy selected frames from cache [U+FFFD]?*upload_dir*
    8. After all videos: flush remaining buffer

    **Adaptive streaming mode** (``selection_mode == 'adaptive_streaming'``):
    When ``strategy == 'adaptive'``, each frame is monitored via
    ``ConfidenceDriftDetector`` (binomial test on short-term confidence
    distribution).  When a statistically significant run of low-confidence
    frames is detected, frames within the drift window are filtered by
    the **dual-signal condition**:

        conf < P_10 (model prediction info) AND
        behavior_entropy > H_med (livestock behavior characteristics)

    Parameters
    ----------
    videos : list[Path] | None
        Optional subset of videos to process.  When None, all videos in
        ``videos_dir`` are processed.  Useful for splitting train/val
        within a stage for shadow deployment validation.

    Returns
    -------
    (all_scores, selected, drift_detected) : tuple of (DataFrame, DataFrame, bool)
        *all_scores* contains every scored frame (for analysis / CSV).
        *selected* contains only the frames chosen for upload/training.
        *drift_detected* is True when the ConfidenceDriftDetector opened a
        drift window at least once during this stream (always False on the
        non-adaptive path).  The drift-triggered update variant (reviewer
        E2-1) uses it to gate the stage-end model update.
    """
    nc = int(cfg.get('num_classes', 3))
    baseline_window = int(cfg.get('baseline_window', 100))
    weights = cfg.get('weights', {})
    component_clip = cfg.get('score_component_clip', 1.0)
    sample_every = max(1, int(cfg.get('video_sample_every', cfg.get('sample_every', 30))))
    max_frames = cfg.get('max_video_frames_per_stage', None)
    if max_frames is not None and int(max_frames) <= 0:
        max_frames = None

    # -- Selection parameters --------------------------------------
    strategy = str(cfg.get('strategy', 'ours'))
    upload_ratio = float(cfg.get('upload_ratio', 0.20))
    max_upload = int(cfg.get('max_upload_per_stage', 240))
    if strategy == 'full_upload':
        max_upload = 10 ** 9                     # no cap: upload all drift frames
    dbds_cfg = cfg.get('dbds', {})
    dbds_active = (strategy == 'dbds')
    if dbds_active:
        _dbds_cap = int(dbds_cfg.get('max_upload_per_stage', 300))
        if _dbds_cap > 0:
            max_upload = _dbds_cap               # DBDS: class-balanced stage cap (300)
    block_size = max(1, int(cfg.get('online_block_size', 120)))
    min_frame_gap = int(cfg.get('min_frame_gap', 5))
    dedup_hamming_threshold = int(cfg.get('dedup_hamming_threshold', 5))
    hard_score_threshold = cfg.get('hard_score_threshold', None)
    hard_score_quantile = cfg.get('hard_score_quantile', None)
    max_candidate_pool_size = int(cfg.get('max_candidate_pool_size', 0))
    prefilter_enabled = bool(cfg.get('edge_prefilter', {}).get('enabled', False))
    prefilter_threshold = float(cfg.get('edge_prefilter', {}).get('conf_threshold', 0.70))

    # -- Adaptive streaming config ----------------------------------
    selection_mode = str(cfg.get('selection_mode', 'online_blocks')).strip().lower()
    adaptive_online = (strategy in ('adaptive', 'full_upload', 'random', 'dbds') and selection_mode == 'adaptive_streaming')
    adaptive_cfg = cfg.get('adaptive', {})
    adaptive_long_window = int(adaptive_cfg.get('long_window', 300))
    adaptive_short_window = int(adaptive_cfg.get('short_window', 20))
    adaptive_p_low = float(adaptive_cfg.get('p_low', 0.10))
    adaptive_alpha_level = float(adaptive_cfg.get('alpha', 0.05))
    adaptive_min_interval_frames = _min_interval_frames(adaptive_cfg, sample_every)
    adaptive_min_frame_gap = int(cfg.get('min_frame_gap', 3))
    adaptive_min_upload_fallback = int(adaptive_cfg.get('min_upload_fallback', 0))
    # Dual-signal mode: when use_entropy=True, filter by conf<P_10 AND (H>H_med OR non_lie_ratio>threshold)
    # When False, use confidence-only (conf<P_10) for ablation comparison
    adaptive_use_entropy = bool(adaptive_cfg.get('use_entropy', True))
    # non_lie_ratio_threshold > 0 → use non_lie_ratio as signal 2 instead of behavior_entropy
    adaptive_nlr_threshold = float(adaptive_cfg.get('non_lie_ratio_threshold', 0.0))
    # use_conf=False → disable confidence signal (conf < P_10), for nlr-only single-signal ablation
    adaptive_use_conf = bool(adaptive_cfg.get('use_conf', True))
    # Trigger-mode ablation (reviewer E2-1): window-opening condition of
    # ConfidenceDriftDetector.  binomial = V1 statistical test (default);
    # fixed_interval / conf_threshold only swap the OPEN condition — the
    # per-stage single update and the conf<P_10 AND signal-2 upload filter
    # are shared by all three modes.
    adaptive_trigger_mode = str(adaptive_cfg.get('trigger_mode', 'binomial')).strip().lower()
    adaptive_fixed_interval_frames = int(adaptive_cfg.get('fixed_interval_frames', 300))
    adaptive_fixed_window_frames = int(adaptive_cfg.get('fixed_window_frames', 300))
    adaptive_threshold_conf = float(adaptive_cfg.get('threshold_conf', 0.55))
    adaptive_threshold_window = int(adaptive_cfg.get('threshold_window', 30))
    adaptive_threshold_sustain = int(adaptive_cfg.get('threshold_sustain', 15))
    # Effective min gap for drift-window uploads:
    #   full_upload → 0 (no gap, all frames pass)
    #   otherwise   → config min_frame_gap
    if strategy == 'full_upload':
        _effective_min_gap = 0
    else:
        _effective_min_gap = adaptive_min_frame_gap

    safe_clear_dir(frame_cache_dir)
    if upload_dir is not None:
        upload_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(student_path))
    try:
        all_rows: list[dict] = []
        history_confs: list[float] = []
        history_area: list[float] = []
        history_entropy: list[float] = []
        history_hists: list[np.ndarray] = []
        history_counts: list[float] = []
        warmup_min = max(5, baseline_window // 5)

        # -- Adaptive streaming state -------------------------------
        drift_detector: ConfidenceDriftDetector | None = None
        _drift_last_upload_idx: int | None = None  # frame-index anchor of the last upload (deterministic gate)
        if adaptive_online:
            drift_detector = ConfidenceDriftDetector(
                long_window=adaptive_long_window,
                short_window=adaptive_short_window,
                p_low=adaptive_p_low,
                alpha=adaptive_alpha_level,
                trigger_mode=adaptive_trigger_mode,
                fixed_interval_frames=adaptive_fixed_interval_frames,
                fixed_window_frames=adaptive_fixed_window_frames,
                threshold_conf=adaptive_threshold_conf,
                threshold_window=adaptive_threshold_window,
                threshold_sustain=adaptive_threshold_sustain,
            )

        # -- DBDS state (class-agnostic distribution-balanced selection) --
        dbds_selector: DBDSSelector | None = None
        if dbds_active:
            _res_dir = None
            if upload_dir is not None:
                # selected_samples/dbds_reservoir: shared across stages,
                # isolated per experiment group.
                _res_dir = upload_dir.parent.parent / 'dbds_reservoir'
            dbds_selector = DBDSSelector(
                num_classes=nc,
                reference=str(dbds_cfg.get('reference', 'uniform')),
                tau=float(dbds_cfg.get('tau', 1.0)),
                reservoir_size=int(dbds_cfg.get('reservoir_size', 200)),
                reservoir_dir=_res_dir,
                tag=(upload_dir.parent.name if upload_dir is not None else ''),
            )

        # Per-block selection state
        buffer: list[dict] = []          # frame rows for current block
        selected_rows: list[dict] = []   # selected frames across all blocks
        global_index = 0
        uploaded_total = 0

        video_paths = videos if videos is not None else list_videos(videos_dir)
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f'[WARN] failed to open video: {video_path}')
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            local_idx = 0
            print(f'[stream] video: {video_path.name}  '
                  f'sample_every={sample_every}  block_size={block_size}  '
                  f'max_frames={max_frames}')

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                local_idx += 1
                if local_idx % sample_every != 0:
                    continue

                # -- YOLO inference (array [U+FFFD]?stats, no disk read) ----
                stats = image_predict_behavior_stats_from_array(
                    model, frame_bgr, imgsz=imgsz, nc=nc,
                )

                # -- dhash from array (no disk needed for dedup) ----
                frame_dhash = dhash(frame_bgr)

                # -- Brightness (mean gray value, 0-255) for night-scene detection --
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                frame_brightness = float(np.mean(frame_gray))

                # -- Write frame to cache (write-only, for export) --
                out_name = f'{video_path.stem}_f{local_idx:08d}.jpg'
                cache_path = frame_cache_dir / out_name
                cv2.imwrite(str(cache_path), frame_bgr)
                frame_path = str(cache_path)

                # -- Sliding-window baseline (BHSS only, skipped for adaptive) --
                cur_hist = np.asarray(stats['behavior_hist'], dtype=np.float32)

                if not adaptive_online:
                    if len(history_confs) < warmup_min:
                        ref_conf = stats['avg_conf'] if stats['avg_conf'] > 0 else 1.0
                        ref_area = stats['area_mean'] if stats['area_mean'] > 0 else 1.0
                        ref_count = stats['det_count'] if stats['det_count'] > 0 else 1.0
                        ref_hist = cur_hist.copy()
                        ref_entropy = stats['behavior_entropy']
                    else:
                        start = max(0, len(history_confs) - baseline_window)
                        ref_conf = float(np.median(history_confs[start:]))
                        ref_area = float(np.median(history_area[start:]))
                        ref_count = float(np.median(history_counts[start:]))
                        ref_entropy = float(np.median(history_entropy[start:]))
                        ref_hist = np.mean(np.stack(history_hists[start:], axis=0), axis=0)
                        if ref_hist.sum() > 0:
                            ref_hist = ref_hist / ref_hist.sum()

                    # -- BHSS score components --------------------------
                    conf_drop = _clip_score(
                        max(0.0, (ref_conf - stats['avg_conf']) / max(ref_conf, 1e-6)),
                        component_clip,
                    )
                    area_drift = _clip_score(
                        robust_norm_abs(stats['area_mean'], ref_area), component_clip,
                    )
                    count_drift = _clip_score(
                        robust_norm_abs(stats['det_count'], ref_count), component_clip,
                    )
                    behavior_drift = _clip_score(
                        float(np.abs(cur_hist - ref_hist).sum() / 2.0), component_clip,
                    )
                    entropy_rise = _clip_score(
                        max(0.0, float(stats['behavior_entropy'] - ref_entropy)),
                        component_clip,
                    )
                    hard_score = (
                        float(weights.get('conf_drop', 0.50)) * conf_drop
                        + float(weights.get('count_drift', 0.25)) * count_drift
                        + float(weights.get('behavior_drift', 0.25)) * behavior_drift
                    )
                else:
                    # Adaptive streaming: BHSS not used; drift detector handles everything
                    ref_conf = 0.0
                    ref_area = 0.0
                    ref_count = 0.0
                    ref_hist = cur_hist.copy()
                    ref_entropy = 0.0
                    conf_drop = 0.0
                    area_drift = 0.0
                    count_drift = 0.0
                    behavior_drift = 0.0
                    entropy_rise = 0.0
                    hard_score = 0.0

                # -- Per-frame drift detection (statistical, no hard thresholds) --
                # Moved AFTER row construction because drift_detector.update()
                # stores frame_row for later upload (needs avg_conf, non_lie_ratio, etc.)
                drift_state = 'normal'  # default; updated after row is built

                # -- Edge prefilter check ---------------------------
                edge_pass = True
                if prefilter_enabled:
                    edge_pass = stats['avg_conf'] < prefilter_threshold

                row = {
                    'path': frame_path,
                    'frame_index': global_index,
                    'video_path': str(video_path),
                    'video_frame_index': local_idx,
                    'timestamp_sec': float(local_idx / fps) if fps > 0 else None,
                    **{k: v for k, v in stats.items() if k != 'behavior_hist'},
                    'behavior_hist': cur_hist.tolist(),
                    **_behavior_counts(cur_hist, stats['det_count']),
                    'ref_conf': ref_conf,
                    'ref_area': ref_area,
                    'ref_count': ref_count,
                    'ref_behavior_hist': ref_hist.tolist(),
                    'conf_drop': conf_drop,
                    'area_drift': area_drift,
                    'count_drift': count_drift,
                    'behavior_drift': behavior_drift,
                    'entropy_rise': entropy_rise,
                    'tta_uncertainty': 0.0,
                    'hard_score': hard_score,
                    'drift_state': drift_state,
                    'drift_window_id': 0,   # 该帧所属漂移窗编号（0 = 不在窗内）
                    'edge_pass': edge_pass,
                    'brightness': frame_brightness,
                    '_dhash_bytes': frame_dhash.tobytes() if frame_dhash is not None else None,
                }

                # -- Per-frame drift detection (runs after row is built) --
                if drift_detector is not None:
                    drift_state = drift_detector.update(
                        stats['avg_conf'], stats['behavior_entropy'], row,
                    )
                    row['drift_state'] = drift_state
                    row['drift_window_id'] = int(drift_detector.window_count)

                    # Extreme confidence drop [U+FFFD]?immediate single-frame upload
                    if drift_detector.is_extreme:
                        if dbds_selector is not None and not dbds_selector.accept_row(row):
                            pass  # DBDS gate: skip frames that don't balance the upload set
                        else:
                            _drift_upload_single(row, upload_dir, selected_rows,
                                                max_upload)
                            if dbds_selector is not None:
                                dbds_selector.update_row(row)
                            _drift_last_upload_idx = int(row.get('frame_index', -1))

                    # Drift episode ended [U+FFFD]?select from accumulated window
                    if drift_detector.was_drifting:
                        drift_frames = drift_detector.on_drift_end()
                        # -- Tag each drift frame with its stage's P_10 / H_med baselines --
                        # These are needed for one-shot ablation (V8) so the global
                        # candidate pool carries per-stage thresholds for C/D selection.
                        for f in (drift_frames or []):
                            f['P_10_stage'] = float(drift_detector.P_10)
                            f['H_med_stage'] = float(drift_detector.H_med)
                        if drift_frames:
                            n_selected = _drift_upload_batch(
                                drift_frames, upload_dir, selected_rows,
                                max_upload, adaptive_min_interval_frames,
                                _drift_last_upload_idx,
                                p10_threshold=None if strategy in ('full_upload', 'random') or not adaptive_use_conf else drift_detector.P_10,
                                h_med_threshold=None if strategy in ('full_upload', 'random') or adaptive_nlr_threshold > 0 else (drift_detector.H_med if adaptive_use_entropy else None),
                                non_lie_ratio_threshold=adaptive_nlr_threshold if strategy not in ('full_upload', 'random') and adaptive_use_entropy else 0.0,
                                min_frame_gap=_effective_min_gap,
                                random_ratio=upload_ratio if strategy == 'random' else 0.0,
                                seed=seed + len(selected_rows),
                                dbds=dbds_selector,
                            )
                            if n_selected > 0:
                                _drift_last_upload_idx = int(drift_frames[-1].get('frame_index', -1))
                                print(
                                    f'[drift] window={len(drift_frames)} frames, '
                                    f'selected={n_selected}, '
                                    f'total_uploaded={len(selected_rows)}, '
                                    f'state={drift_detector.get_state()}'
                                )
                if dbds_selector is not None:
                    # Maintain the diversity reservoir on every scored frame
                    # (skips frames already selected for upload).
                    dbds_selector.reservoir_update(row)
                all_rows.append(row)
                if not adaptive_online:
                    buffer.append(row)
                    history_confs.append(stats['avg_conf'])
                    history_area.append(stats['area_mean'])
                    history_counts.append(stats['det_count'])
                    history_entropy.append(stats['behavior_entropy'])
                    history_hists.append(cur_hist)

                global_index += 1

                # -- Per-block selection & upload (non-adaptive path) --
                if not adaptive_online and len(buffer) >= block_size:
                    _flush_streaming_block(
                        buffer, selected_rows, upload_dir,
                        strategy, upload_ratio, max_upload,
                        min_frame_gap, dedup_hamming_threshold,
                        seed, hard_score_threshold,
                        max_candidate_pool_size, hard_score_quantile,
                        prefilter_enabled,
                    )
                    uploaded_total = len(selected_rows)
                    buffer.clear()

                if max_frames is not None and global_index >= int(max_frames):
                    cap.release()
                    # Flush remaining (adaptive or block)
                    if adaptive_online and drift_detector is not None and drift_detector.is_drifting:
                        drift_frames = drift_detector.on_drift_end()
                        if drift_frames:
                            n_sel = _drift_upload_batch(
                                drift_frames, upload_dir, selected_rows,
                                max_upload, adaptive_min_interval_frames,
                                _drift_last_upload_idx,
                                p10_threshold=None if strategy in ('full_upload', 'random') or not adaptive_use_conf else drift_detector.P_10,
                                h_med_threshold=None if strategy in ('full_upload', 'random') or adaptive_nlr_threshold > 0 else (drift_detector.H_med if adaptive_use_entropy else None),
                                non_lie_ratio_threshold=adaptive_nlr_threshold if strategy not in ('full_upload', 'random') and adaptive_use_entropy else 0.0,
                                min_frame_gap=_effective_min_gap,
                                random_ratio=upload_ratio if strategy == 'random' else 0.0,
                                seed=seed + len(selected_rows),
                                dbds=dbds_selector,
                            )
                            uploaded_total = len(selected_rows)
                            print(
                                f'[drift] max_frames flush: {len(drift_frames)} frames, '
                                f'{n_sel} selected, {uploaded_total} total uploaded'
                            )
                    elif not adaptive_online and buffer:
                        _flush_streaming_block(
                            buffer, selected_rows, upload_dir,
                            strategy, upload_ratio, max_upload,
                            min_frame_gap, dedup_hamming_threshold,
                            seed, hard_score_threshold,
                            max_candidate_pool_size, hard_score_quantile,
                            prefilter_enabled,
                        )
                        uploaded_total = len(selected_rows)
                    if dbds_selector is not None:
                        dbds_selector.save()
                    all_df = pd.DataFrame(all_rows)
                    sel_df = pd.DataFrame(selected_rows)
                    print(
                        f'[stream] done (max_frames={max_frames}): '
                        f'{len(all_df)} scored, {uploaded_total} uploaded '
                        f'({100 * uploaded_total / max(1, len(all_df)):.1f}%)'
                    )
                    return all_df, sel_df, (
                        drift_detector.drift_opened
                        if (adaptive_online and drift_detector is not None) else False
                    )

            cap.release()

        # -- Flush final partial block / drift remaining ---------
        if adaptive_online and drift_detector is not None and drift_detector.is_drifting:
            drift_frames = drift_detector.on_drift_end()
            if drift_frames:
                n_sel = _drift_upload_batch(
                    drift_frames, upload_dir, selected_rows,
                    max_upload, adaptive_min_interval_frames,
                    _drift_last_upload_idx,
                    p10_threshold=None if strategy in ('full_upload', 'random') or not adaptive_use_conf else drift_detector.P_10,
                    h_med_threshold=None if strategy in ('full_upload', 'random') or adaptive_nlr_threshold > 0 else (drift_detector.H_med if adaptive_use_entropy else None),
                    non_lie_ratio_threshold=adaptive_nlr_threshold if strategy not in ('full_upload', 'random') and adaptive_use_entropy else 0.0,
                    min_frame_gap=_effective_min_gap,
                    random_ratio=upload_ratio if strategy == 'random' else 0.0,
                    seed=seed + len(selected_rows),
                    dbds=dbds_selector,
                )
                uploaded_total = len(selected_rows)
                if n_sel > 0:
                    print(
                        f'[drift] final flush: {len(drift_frames)} drift frames, '
                        f'{n_sel} selected, {uploaded_total} total uploaded, '
                        f'state={drift_detector.get_state()}'
                    )
        elif not adaptive_online and buffer:
            _flush_streaming_block(
                buffer, selected_rows, upload_dir,
                strategy, upload_ratio, max_upload,
                min_frame_gap, dedup_hamming_threshold,
                seed, hard_score_threshold,
                max_candidate_pool_size, hard_score_quantile,
                prefilter_enabled,
            )
            uploaded_total = len(selected_rows)

        # -- Minimum upload fallback ---------------------------------
        # When drift-based selection produces too few frames (or none at all),
        # fall back to behavior-entropy-based (or random) selection to
        # guarantee a minimum amount of training data per stage.
        # DBDS instead tops up from the diversity reservoir: frames whose
        # class content is under-represented in the current upload set.
        if adaptive_online and adaptive_min_upload_fallback > 0 and all_rows:
            drift_uploaded = len(selected_rows)
            if drift_uploaded < adaptive_min_upload_fallback:
                need = adaptive_min_upload_fallback - drift_uploaded
                if max_upload > 0:
                    need = min(need, max_upload - drift_uploaded)
                if need > 0 and upload_dir is not None:
                    import random as _random
                    selected_indices = {r['frame_index'] for r in selected_rows}
                    selected_paths = {str(r.get('path', '')) for r in selected_rows}
                    if dbds_selector is not None:
                        # DBDS: under-represented historical frames first.
                        # Exclude by path AND frame_index: reservoir rows from
                        # the current stage share frame_index with uploaded
                        # frames (their path copy predates the upload copy).
                        candidates = [
                            r for r in dbds_selector.reservoir_ranked(exclude_paths=selected_paths)
                            if r.get('frame_index') not in selected_indices
                        ]
                    else:
                        candidates = [
                            r for r in all_rows
                            if r['frame_index'] not in selected_indices
                        ]
                        if strategy == 'random':
                            # Random fallback for B_random baseline
                            _rng = _random.Random(seed + len(selected_rows))
                            _rng.shuffle(candidates)
                        else:
                            # Entropy-based: high behavior diversity first
                            candidates.sort(
                                key=lambda r: float(r.get('behavior_entropy', 0)),
                                reverse=True,
                            )
                    fallback_added = 0
                    for item in candidates:
                        if fallback_added >= need:
                            break
                        src = Path(str(item.get('path', '')))
                        if src.exists():
                            dst = upload_dir / src.name
                            try:
                                dst.unlink(missing_ok=True)
                                shutil.copy2(str(src), str(dst))
                                item['path'] = str(dst)
                            except Exception:
                                pass
                        item['selected'] = True
                        if dbds_selector is not None:
                            dbds_selector.update_row(item)
                        selected_rows.append(item)
                        fallback_added += 1
                    uploaded_total = len(selected_rows)
                    print(
                        f'[fallback] drift_uploads={drift_uploaded} < '
                        f'min={adaptive_min_upload_fallback}, '
                        f'{"reservoir" if dbds_selector is not None else "activity"}_fallback=+{fallback_added}, '
                        f'total={uploaded_total}'
                    )


        if dbds_selector is not None:
            dbds_selector.save()
        all_df = pd.DataFrame(all_rows)
        sel_df = pd.DataFrame(selected_rows)
        print(
            f'[stream] done: {len(all_df)} scored, {uploaded_total} uploaded '
            f'({100 * uploaded_total / max(1, len(all_df)):.1f}%)'
        )
        return all_df, sel_df, (
            drift_detector.drift_opened
            if (adaptive_online and drift_detector is not None) else False
        )

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def precompute_stage_scores(
    student_path: Path,
    videos_dir: Path,
    frame_cache_dir: Path,
    scores_csv_path: Path,
    imgsz: int,
    cfg: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """Phase 1: Stream-process videos [U+FFFD]?YOLO + BHSS, save frames + scores CSV.

    This runs the expensive inference ONCE.  The saved CSV and cached frames
    can then be reused by multiple experiments (Phase 2) with different
    selection strategies and upload ratios [U+FFFD]?eliminating repeated inference.

    Frames flow: cv2.VideoCapture [U+FFFD]?YOLO (numpy array) [U+FFFD]?BHSS scoring [U+FFFD]?    save frame to *frame_cache_dir*, accumulate row, NO per-block selection.

    Parameters
    ----------
    student_path : Path
        Edge student model for inference.
    videos_dir : Path
        Directory containing video files for one stage.
    frame_cache_dir : Path
        Persistent directory where sampled frames are saved (shared across
        experiments).  Created / overwritten.
    scores_csv_path : Path
        Where to write the full scores DataFrame (CSV with hex-encoded dhash).
    imgsz : int
        YOLO input size.
    cfg : dict
        Sampling config dict (same structure as ``sampling`` in yaml).
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        All scored frames (also written to *scores_csv_path*).
    """
    nc = int(cfg.get('num_classes', 3))
    baseline_window = int(cfg.get('baseline_window', 100))
    weights = cfg.get('weights', {})
    component_clip = cfg.get('score_component_clip', 1.0)
    sample_every = max(1, int(cfg.get('video_sample_every', cfg.get('sample_every', 30))))
    max_frames = cfg.get('max_video_frames_per_stage', None)
    if max_frames is not None and int(max_frames) <= 0:
        max_frames = None
    prefilter_enabled = bool(cfg.get('edge_prefilter', {}).get('enabled', False))
    prefilter_threshold = float(cfg.get('edge_prefilter', {}).get('conf_threshold', 0.70))

    safe_clear_dir(frame_cache_dir)
    scores_csv_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(student_path))
    try:
        all_rows: list[dict] = []
        history_confs: list[float] = []
        history_area: list[float] = []
        history_entropy: list[float] = []
        history_hists: list[np.ndarray] = []
        history_counts: list[float] = []
        warmup_min = max(5, baseline_window // 5)
        global_index = 0

        for video_path in list_videos(videos_dir):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f'[precompute] WARN failed to open video: {video_path}')
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            local_idx = 0
            print(f'[precompute] video: {video_path.name}  '
                  f'sample_every={sample_every}  max_frames={max_frames}')

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                local_idx += 1
                if local_idx % sample_every != 0:
                    continue

                # -- YOLO inference (array [U+FFFD]?stats, no disk read) ----
                stats = image_predict_behavior_stats_from_array(
                    model, frame_bgr, imgsz=imgsz, nc=nc,
                )

                # -- dhash from array --------------------------------
                frame_dhash = dhash(frame_bgr)

                # -- Write frame to persistent cache ------------------
                out_name = f'{video_path.stem}_f{local_idx:08d}.jpg'
                cache_path = frame_cache_dir / out_name
                cv2.imwrite(str(cache_path), frame_bgr)

                # -- Sliding-window baseline --------------------------
                cur_hist = np.asarray(stats['behavior_hist'], dtype=np.float32)
                if len(history_confs) < warmup_min:
                    ref_conf = stats['avg_conf'] if stats['avg_conf'] > 0 else 1.0
                    ref_area = stats['area_mean'] if stats['area_mean'] > 0 else 1.0
                    ref_count = stats['det_count'] if stats['det_count'] > 0 else 1.0
                    ref_hist = cur_hist.copy()
                    ref_entropy = stats['behavior_entropy']
                else:
                    start = max(0, len(history_confs) - baseline_window)
                    ref_conf = float(np.median(history_confs[start:]))
                    ref_area = float(np.median(history_area[start:]))
                    ref_count = float(np.median(history_counts[start:]))
                    ref_entropy = float(np.median(history_entropy[start:]))
                    ref_hist = np.mean(np.stack(history_hists[start:], axis=0), axis=0)
                    if ref_hist.sum() > 0:
                        ref_hist = ref_hist / ref_hist.sum()

                # -- BHSS score components ----------------------------
                conf_drop = _clip_score(
                    max(0.0, (ref_conf - stats['avg_conf']) / max(ref_conf, 1e-6)),
                    component_clip,
                )
                area_drift = _clip_score(
                    robust_norm_abs(stats['area_mean'], ref_area), component_clip,
                )
                count_drift = _clip_score(
                    robust_norm_abs(stats['det_count'], ref_count), component_clip,
                )
                behavior_drift = _clip_score(
                    float(np.abs(cur_hist - ref_hist).sum() / 2.0), component_clip,
                )
                entropy_rise = _clip_score(
                    max(0.0, float(stats['behavior_entropy'] - ref_entropy)),
                    component_clip,
                )
                hard_score = (
                    float(weights.get('conf_drop', 0.50)) * conf_drop
                    + float(weights.get('count_drift', 0.25)) * count_drift
                    + float(weights.get('behavior_drift', 0.25)) * behavior_drift
                )

                # -- Edge prefilter check -----------------------------
                edge_pass = True
                if prefilter_enabled:
                    edge_pass = stats['avg_conf'] < prefilter_threshold

                # -- dhash as hex (CSV-safe) --------------------------
                dhash_hex = None
                if frame_dhash is not None:
                    dhash_hex = frame_dhash.tobytes().hex()

                row = {
                    'path': str(cache_path),
                    'frame_index': global_index,
                    'video_path': str(video_path),
                    'video_frame_index': local_idx,
                    'timestamp_sec': float(local_idx / fps) if fps > 0 else None,
                    **{k: v for k, v in stats.items() if k != 'behavior_hist'},
                    'behavior_hist': cur_hist.tolist(),
                    **_behavior_counts(cur_hist, stats['det_count']),
                    'ref_conf': ref_conf,
                    'ref_area': ref_area,
                    'ref_count': ref_count,
                    'ref_behavior_hist': ref_hist.tolist(),
                    'conf_drop': conf_drop,
                    'area_drift': area_drift,
                    'count_drift': count_drift,
                    'behavior_drift': behavior_drift,
                    'entropy_rise': entropy_rise,
                    'tta_uncertainty': 0.0,
                    'hard_score': hard_score,
                    'edge_pass': edge_pass,
                    '_dhash_hex': dhash_hex,
                }
                all_rows.append(row)
                history_confs.append(stats['avg_conf'])
                history_area.append(stats['area_mean'])
                history_counts.append(stats['det_count'])
                history_entropy.append(stats['behavior_entropy'])
                history_hists.append(cur_hist)

                global_index += 1

                if global_index % 200 == 0:
                    print(f'[precompute] scored {global_index} frames ...')

                if max_frames is not None and global_index >= int(max_frames):
                    cap.release()
                    all_df = pd.DataFrame(all_rows)
                    all_df.to_csv(scores_csv_path, index=False, encoding='utf-8-sig')
                    print(
                        f'[precompute] done (max_frames={max_frames}): '
                        f'{len(all_df)} frames scored [U+FFFD]?{scores_csv_path}'
                    )
                    return all_df

            cap.release()

        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(scores_csv_path, index=False, encoding='utf-8-sig')
        print(
            f'[precompute] done: {len(all_df)} frames scored [U+FFFD]?{scores_csv_path}'
        )
        return all_df

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_cached_scores(scores_csv_path: Path) -> pd.DataFrame:
    """Phase 2: Load precomputed BHSS scores from CSV.

    Converts ``_dhash_hex`` back to ``_dhash_bytes`` for compatibility
    with ``_get_dhash`` during selection.
    """
    if not scores_csv_path.exists():
        raise FileNotFoundError(f'Scores cache not found: {scores_csv_path}')
    df = pd.read_csv(scores_csv_path)
    # Convert hex dhash back to bytes for _get_dhash compatibility
    if '_dhash_hex' in df.columns:
        def _hex_to_bytes(h):
            if isinstance(h, str) and h:
                try:
                    return bytes.fromhex(h)
                except Exception:
                    return None
            return None
        df['_dhash_bytes'] = df['_dhash_hex'].apply(_hex_to_bytes)
    print(f'[load_cached_scores] loaded {len(df)} scored frames from {scores_csv_path}')
    return df


def recompute_hard_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Recompute ``hard_score`` from individual BHSS components.

    The precomputed CSV stores the three raw score components
    (``conf_drop``, ``count_drift``, ``behavior_drift``) independently.
    This lets Phase 2 experiments vary the fusion weights **without**
    re-running inference [U+FFFD]?the components are simply recombined.

    Parameters
    ----------
    df : pd.DataFrame
        Precomputed scores (must contain conf_drop, count_drift, behavior_drift).
    weights : dict
        Mapping of component name [U+FFFD]?weight, e.g.
        ``{'conf_drop': 0.50, 'count_drift': 0.25, 'behavior_drift': 0.25}``.
        Weights are normalised to sum to 1.0.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``hard_score`` recomputed.
    """
    required = ['conf_drop', 'count_drift', 'behavior_drift']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'Cannot recompute hard_score [U+FFFD]?missing columns: {missing}')
    w_conf = float(weights.get('conf_drop', 0.50))
    w_count = float(weights.get('count_drift', 0.25))
    w_behav = float(weights.get('behavior_drift', 0.25))
    total = w_conf + w_count + w_behav
    if total <= 0:
        raise ValueError(f'Weight sum must be > 0, got {total}')
    w_conf /= total
    w_count /= total
    w_behav /= total
    out = df.copy()
    out['hard_score'] = (
        w_conf   * out['conf_drop'].fillna(0.0)
        + w_count * out['count_drift'].fillna(0.0)
        + w_behav * out['behavior_drift'].fillna(0.0)
    )
    return out


def _compute_adaptive_score(
    df: pd.DataFrame,
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """Compute dynamic variance-weighted fusion score for the ``adaptive`` strategy.

    For each block, two hardness signals are extracted:

        h_conf = 1 [U+FFFD]?norm(avg_conf)     (lower confidence [U+FFFD]?harder)
        h_act  = norm(non_lie_ratio)     (higher activity [U+FFFD]?harder)

    The fusion weight alpha is set by the *within-block* variance of each signal:
    the signal with more spread is more discriminative and gets more weight.

        alpha = Var(h_conf) / (Var(h_conf) + Var(h_act) + epsilon)

        S = alpha . h_conf + (1-[U+FFFD]? . h_act

    This is **parameter-free** [U+FFFD]?no hand-tuned weights, no fixed thresholds.
    When confidence varies a lot (e.g. lighting changes), alpha [U+FFFD]?1 and ranking
    follows confidence.  When pig activity varies a lot (e.g. feeding time),
    alpha [U+FFFD]?0 and ranking follows behaviour.  When both are stable the scores
    are near-uniform and selection reverts to proportional sampling.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``avg_conf`` and ``non_lie_ratio`` columns.
    epsilon : float
        Stability constant for divisions.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``adaptive_score`` column.
    """
    df = df.copy()
    conf = df['avg_conf'].values.astype(np.float64)
    act = df['non_lie_ratio'].values.astype(np.float64)

    # -- Min-max normalisation within the block ---------------------
    conf_min, conf_max = conf.min(), conf.max()
    act_min, act_max = act.min(), act.max()

    if conf_max - conf_min > epsilon:
        h_conf = 1.0 - (conf - conf_min) / (conf_max - conf_min)
    else:
        h_conf = np.full_like(conf, 0.5)

    if act_max - act_min > epsilon:
        h_act = (act - act_min) / (act_max - act_min)
    else:
        h_act = np.full_like(act, 0.5)

    # -- Variance-driven weight ------------------------------------
    var_conf = float(np.var(h_conf))
    var_act = float(np.var(h_act))
    alpha = var_conf / (var_conf + var_act + epsilon)

    df['adaptive_score'] = alpha * h_conf + (1.0 - alpha) * h_act
    df['_adaptive_alpha'] = alpha
    return df


# ==========================================================================[U+FFFD]?#  ConfidenceDriftDetector [U+FFFD]?statistical drift detection via consecutive
#  confidence drops.
#
#  Inspired by D:\project\system\edge_app\application\drift_detection_adaevo.py
#  (AdaEvo: sliding-window confidence monitoring [U+FFFD]?drift classification) and
#  D:\project\system\edge_app\application\visual_work.py
#  (UGSSelector: per-frame uncertainty scoring with quality gating).
#
#  Models "consecutive confidence drops" as a binomial hypothesis test on
#  the short-term window: is the proportion of low-confidence frames (conf
#  below the long-term P_10) significantly above the expected 10%?
#
#  All thresholds derived from data distribution [U+FFFD]?no hard-coded values:
#    P_10  = 10th percentile of long window  (robust lower decile)
#    alpha     = 0.01  (Neyman-Pearson significance level)
#    W_L   = 300  (~4 min [U+FFFD]?stable baseline)
#    W_S   = 30  (~24 s [U+FFFD]?CLT-compliant, adequate power)
#    Critical k from binomial distribution (analytical, not manual)


class ConfidenceDriftDetector:
    """Statistical drift detector based on consecutive confidence drops.

    Models "consecutive confidence drops" as a binomial hypothesis test:
    is the proportion of low-confidence frames in the recent short window
    significantly above the expected level defined by the long-term
    empirical distribution?

    Maintains a long-term window (W_L) to estimate the empirical confidence
    distribution and median behavior entropy, plus a short-term window (W_S)
    monitored for an anomalous proportion of low-confidence frames.  Drift
    is detected when the short-term count of low-confidence frames
    (conf < P_10) exceeds the critical value from a binomial test at
    significance level alpha.

    **Dual baselines** (P_10 frozen, H_med sliding after warmup):

        P_10  = 10th percentile of long-term confidence (frozen [U+FFFD]?stable drift reference)
        H_med = median behavior entropy (sliding [U+FFFD]?contextual behaviour baseline)

    All thresholds are derived from the data distribution [U+FFFD]?no hard-coded
    absolute values.  Each parameter has a statistical justification:

        P_10  = 10th percentile of long-term confidence (robust lower bound)
        H_med = 50th percentile of sliding long-term behavior entropy (contextual centre)
        alpha     = 0.05 (Neyman-Pearson significance level)
        p_low = 0.10 (by definition of P_10, exactly 10% of data)
        H0: P(conf < P_10) = p_low   vs   H1: P(conf < P_10) > p_low
        Drift when binomial P(X [U+FFFD]?k | p_low, n=W_S) < alpha

    Usage (per frame)::

        detector = ConfidenceDriftDetector(long_window=300, short_window=20)
        for each frame:
            state = detector.update(avg_conf, behavior_entropy, frame_row)
            if state == 'normal' and detector.was_drifting:
                drift_frames = detector.on_drift_end()
                # filter: conf < P_10 AND entropy > H_med, then upload

    Parameters
    ----------
    long_window : int
        Size of the long-term baseline window for computing percentiles
        (default 300 [U+FFFD]?~4 min at 0.8 s sampling).
    short_window : int
        Size of the short-term monitoring window (default 20 [U+FFFD]?~16 s,
        balances responsiveness against statistical power; n[U+FFFD]?0 provides
        adequate binomial test power for moderate-to-large effects).
    p_low : float
        Lower quantile defining "low confidence" (default 0.10 [U+FFFD]?standard
        robust lower decile).
    alpha : float
        Significance level for the one-sided binomial test
        (default 0.05 [U+FFFD]?Fisher's standard "significant" threshold).
    """

    def __init__(
        self,
        long_window: int = 300,
        short_window: int = 20,
        p_low: float = 0.10,
        alpha: float = 0.05,
        trigger_mode: str = 'binomial',
        fixed_interval_frames: int = 300,
        fixed_window_frames: int = 300,
        threshold_conf: float = 0.55,
        threshold_window: int = 30,
        threshold_sustain: int = 15,
    ):
        self.long_window = max(30, int(long_window))
        self.short_window = max(10, int(short_window))
        self.p_low = float(p_low)
        self.alpha = float(alpha)

        # Trigger mode (reviewer E2-1 ablation): the window-OPENING
        # condition of the drift state machine.  'binomial' = original
        # V1 statistical test; 'fixed_interval' opens windows on a
        # deterministic frame cadence; 'conf_threshold' opens when the
        # rolling mean confidence stays below a threshold.  Everything
        # downstream (warmup, baselines, drift-frames accumulation, the
        # conf<P_10 AND signal-2 upload filter) is shared by all modes —
        # only WHEN the window opens changes.
        self._trigger_mode = str(trigger_mode).strip().lower()
        if self._trigger_mode not in ('binomial', 'fixed_interval', 'conf_threshold'):
            print(f'[sampling] unknown trigger_mode {trigger_mode!r} '
                  f'-> falling back to "binomial"')
            self._trigger_mode = 'binomial'
        self._fixed_interval_frames = max(1, int(fixed_interval_frames))
        self._fixed_window_frames = max(1, int(fixed_window_frames))
        self._threshold_conf = float(threshold_conf)
        self._threshold_window = max(1, int(threshold_window))
        self._threshold_sustain = max(1, int(threshold_sustain))

        # Pre-compute binomial critical value k_crit:
        #   smallest k such that P(X [U+FFFD]?k | Binomial(n, p_low)) < alpha
        self._k_crit = self._compute_k_crit(
            self.short_window, self.p_low, self.alpha,
        )

        # -- State --------------------------------------------------
        self._long_confs: list[float] = []
        self._short_confs: list[float] = []      # conf values
        self._short_rows: list[dict] = []         # frame metadata

        # Dual adaptive baselines:
        #   P_10  [U+FFFD]?frozen after warmup  (stable confidence reference)
        #   H_med [U+FFFD]?sliding after warmup  (contextual behavior reference)
        self._P_10: float = 0.0                   # model prediction info
        self._P_90: float = 0.0
        self._long_entropies: list[float] = []     # behavior entropy history
        self._H_med: float = 0.0                   # livestock behavior baseline

        self._drift_state: str = 'normal'        # 'normal' | 'drifting'
        self._drift_frames: list[dict] = []       # accumulated drift frames
        self._was_drifting: bool = False          # latch for end-of-drift
        self._drift_opened: bool = False          # monotonic: a window opened at least once (drift-triggered update)
        # 开窗计数（＝触发次数）。论文 α 敏感性的核心指标：α 只通过
        # _k_crit 影响判据，触发次数是它最直接的观测量。纯计数，不参与
        # 任何判定（2026-09-30 新增）。
        self._drift_windows: int = 0
        self._frame_count: int = 0
        self._warmup_done: bool = False
        self._baseline_frozen: bool = False       # P_10 frozen after warmup (H_med keeps sliding)

        # Non-binomial trigger state (E2-1)
        self._mode_frame_count: int = 0            # post-warmup frames (fixed_interval cadence)
        self._mode_confs: list[float] = []         # rolling conf window (conf_threshold)
        self._low_streak: int = 0                  # consecutive low-mean updates (conf_threshold)

    # -- Public API ---------------------------------------------------

    def update(self, avg_conf: float, behavior_entropy: float, frame_row: dict) -> str:
        """Process one frame.  Returns 'normal' or 'drifting'.

        When the return value is 'normal' and :attr:`was_drifting` is True,
        the caller should call :meth:`on_drift_end` to retrieve the
        accumulated drift frames for selection and upload.

        Parameters
        ----------
        avg_conf : float
            Mean detection confidence for this frame.
        behavior_entropy : float
            Normalized behavior entropy (group-level behavioral diversity).
        frame_row : dict
            Frame metadata (path, stats, etc.).
        """
        prev_state = self._drift_state
        self._frame_count += 1

        # -- Long-term windows -------------------------------------
        self._long_confs.append(float(avg_conf))
        self._long_entropies.append(float(behavior_entropy))
        if len(self._long_confs) > self.long_window:
            self._long_confs.pop(0)
        if len(self._long_entropies) > self.long_window:
            self._long_entropies.pop(0)

        warmup_min = max(10, self.long_window // 3)
        if len(self._long_confs) < warmup_min:
            return 'normal'

        # -- Warmup complete: freeze P_10, initialise H_med ------------
        # P_10 is frozen after warmup to serve as a stable confidence
        # reference for drift detection.  H_med, by contrast, continues
        # to slide [U+FFFD]?the entropy filter should be contextual ("more
        # diverse than recent frames") because pig behaviour patterns
        # change systematically across day / night / eating stages.
        if not self._warmup_done:
            self._update_baselines()
            self._warmup_done = True
            self._baseline_frozen = True   # marks P_10 as frozen

        # -- Sliding H_med (contextual behaviour baseline) ----------
        # After warmup, P_10 stays frozen but H_med tracks the
        # long_window of recent behaviour entropies so that the filter
        # adapts to the current stage's behavioural regime.
        if self._warmup_done and len(self._long_entropies) >= 10:
            ent_arr = np.asarray(self._long_entropies, dtype=np.float64)
            self._H_med = float(np.percentile(ent_arr, 50.0))

        # -- Short-term window -------------------------------------
        self._short_confs.append(float(avg_conf))
        self._short_rows.append(frame_row)
        if len(self._short_confs) > self.short_window:
            self._short_confs.pop(0)
            self._short_rows.pop(0)

        # -- State machine -----------------------------------------
        # Window-opening condition depends on trigger_mode (reviewer
        # E2-1 ablation).  The binomial branch is the original V1 code
        # path and is unchanged when trigger_mode='binomial' (default).
        if self._trigger_mode == 'binomial':
            # -- Count low-confidence frames ------------------------
            k = sum(1 for c in self._short_confs if c < self._P_10)
            if self._drift_state == 'normal' and k >= self._k_crit:
                # Drift starts [U+FFFD]?collect from first low-conf frame
                self._drift_state = 'drifting'
                self._drift_opened = True
                self._drift_windows += 1
                for i, c in enumerate(self._short_confs):
                    if c < self._P_10:
                        self._drift_frames = list(self._short_rows[i:])
                        break
            elif self._drift_state == 'drifting':
                self._drift_frames.append(frame_row)
                # Drift ends when k returns to expected value
                expected = int(self.p_low * self.short_window)
                if k <= expected:
                    self._drift_state = 'normal'
        else:
            self._update_trigger_mode(avg_conf, frame_row)

        # Latch for external caller
        self._was_drifting = (prev_state == 'drifting' and self._drift_state == 'normal')
        return self._drift_state

    def on_drift_end(self) -> list[dict] | None:
        """Retrieve accumulated drift frames after drift ends.

        Returns the list of frame dicts from the drift window, or None
        if no drift was in progress.  The caller should filter by
        ``conf < P_10 AND behavior_entropy > H_med`` and upload.
        """
        if not self._drift_frames:
            return None
        frames = self._drift_frames
        self._drift_frames = []
        return frames

    # -- Properties ---------------------------------------------------

    @property
    def P_10(self) -> float:
        """Frozen 10th percentile of confidence (model uncertainty baseline)."""
        return self._P_10

    @property
    def H_med(self) -> float:
        """Sliding median behavior entropy (contextual livestock behavior baseline)."""
        return self._H_med

    @property
    def is_drifting(self) -> bool:
        """True while a drift episode is in progress."""
        return self._drift_state == 'drifting'

    @property
    def was_drifting(self) -> bool:
        """True on the frame where drift just ended (latch)."""
        return self._was_drifting

    @property
    def window_count(self) -> int:
        """Number of drift windows opened so far (= 触发次数).

        Monotonic counter, incremented on every 'normal' -> 'drifting'
        transition in all three trigger modes.  Logged per frame as
        ``drift_window_id`` and aggregated per stage by the runner.
        """
        return self._drift_windows

    @property
    def drift_opened(self) -> bool:
        """True if a drift window opened at least once (monotonic latch).

        The drift-triggered update variant (reviewer E2-1) uses this at
        stage end: the model-update state is entered only when the
        detector fired during the stage.
        """
        return self._drift_opened

    @property
    def is_extreme(self) -> bool:
        """Latest frame has extremely low confidence (far-left tail).

        Defined as conf < P_10 [U+FFFD]?2.(P_90 [U+FFFD]?P_10), i.e. well into the
        left tail of the empirical distribution.  Analogue of
        visual_work's ``upload_high_score``, but data-driven.
        """
        if not self._warmup_done or not self._short_confs:
            return False
        iqr_like = self._P_90 - self._P_10
        if iqr_like <= 0:
            return False
        return self._short_confs[-1] < self._P_10 - 2.0 * iqr_like

    def get_state(self) -> dict:
        """Return current detector state for logging / debugging."""
        k = sum(1 for c in self._short_confs if c < self._P_10) if self._short_confs else 0
        state = {
            'frames': self._frame_count,
            'warmup': self._warmup_done,
            'P_10': round(self._P_10, 4),
            'P_90': round(self._P_90, 4),
            'H_med': round(self._H_med, 4),
            'k_crit': self._k_crit,
            'k_current': k,
            'short_window_n': len(self._short_confs),
            'long_window_n': len(self._long_confs),
            'drift_state': self._drift_state,
            'drift_frames_accumulated': len(self._drift_frames),
            'drift_opened': self._drift_opened,
            'drift_windows': self._drift_windows,
            'trigger_mode': self._trigger_mode,
        }
        if self._trigger_mode == 'conf_threshold':
            state['conf_rolling_mean'] = (
                round(sum(self._mode_confs) / len(self._mode_confs), 4)
                if self._mode_confs else None)
            state['conf_low_streak'] = self._low_streak
        return state

    # -- Internal helpers ---------------------------------------------

    def _update_baselines(self) -> None:
        """Recompute P_10, P_90, and H_med from the long windows.

        P_10 and P_90 are computed once at warmup and frozen thereafter
        (stable confidence reference for the binomial drift test).
        H_med continues to slide after warmup [U+FFFD]?the behaviour-entropy
        baseline adapts to the current stage's behavioural regime.
        """
        if len(self._long_confs) >= 10:
            arr = np.asarray(self._long_confs, dtype=np.float64)
            self._P_10 = float(np.percentile(arr, self.p_low * 100.0))
            self._P_90 = float(np.percentile(arr, 90.0))
        if len(self._long_entropies) >= 10:
            ent_arr = np.asarray(self._long_entropies, dtype=np.float64)
            self._H_med = float(np.percentile(ent_arr, 50.0))

    def _update_trigger_mode(self, avg_conf: float, frame_row: dict) -> None:
        """Non-binomial window-opening conditions (reviewer E2-1).

        Only the OPEN condition changes per mode — the drift-frames
        accumulation contract, the conf<P_10 AND signal-2 upload filter,
        and the 'normal' + was_drifting latch are shared with the
        binomial state machine.

        fixed_interval: opens on the first post-warmup frame and then
        every ``fixed_interval_frames`` sampled frames (deterministic,
        confidence-independent); closes after ``fixed_window_frames``.
        conf_threshold: opens when the rolling mean of the last
        ``threshold_window`` frame confidences stays below
        ``threshold_conf`` for ``threshold_sustain`` consecutive
        updates; closes on the first update whose rolling mean recovers.
        """
        if self._trigger_mode == 'fixed_interval':
            self._mode_frame_count += 1
            if self._drift_state == 'normal':
                if (self._mode_frame_count - 1) % self._fixed_interval_frames == 0:
                    self._drift_state = 'drifting'
                    self._drift_opened = True
                    self._drift_windows += 1
                    self._drift_frames = [frame_row]
            else:
                self._drift_frames.append(frame_row)
                if len(self._drift_frames) >= self._fixed_window_frames:
                    self._drift_state = 'normal'
            return

        # conf_threshold
        self._mode_confs.append(float(avg_conf))
        if len(self._mode_confs) > self._threshold_window:
            self._mode_confs.pop(0)
        if len(self._mode_confs) < self._threshold_window:
            return  # rolling window not full yet
        rolling_mean = sum(self._mode_confs) / len(self._mode_confs)
        if self._drift_state == 'normal':
            if rolling_mean < self._threshold_conf:
                self._low_streak += 1
                if self._low_streak >= self._threshold_sustain:
                    self._drift_state = 'drifting'
                    self._drift_opened = True
                    self._drift_windows += 1
                    self._drift_frames = [frame_row]
                    self._low_streak = 0
            else:
                self._low_streak = 0
        else:
            self._drift_frames.append(frame_row)
            if rolling_mean >= self._threshold_conf:
                self._drift_state = 'normal'

    @staticmethod
    def _compute_k_crit(n: int, p: float, alpha: float) -> int:
        """Smallest k s.t. P(X [U+FFFD]?k | Binomial(n, p)) < alpha."""
        from math import comb
        for k in range(n + 1):
            sf = sum(
                comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
                for i in range(k, n + 1)
            )
            if sf < alpha:
                return k
        return n + 1  # unreachable (drift never fires)

# ==========================================================================[U+FFFD]?#  Streaming adaptive upload helpers (inspired by
#  D:\project\system\edge_app\application\visual_work.py)
# ==========================================================================[U+FFFD]?
class _AdaptiveUploadBuffer:
    """Accumulates triggered frames and launches async batch uploads.

    Mirrors the pattern from ``visual_work.detect_objects()``:
      - Each high-score frame is appended to a candidate list.
      - When the list reaches ``trigger_count`` OR a frame exceeds
        ``high_score``, the batch is uploaded asynchronously.
      - ``min_interval`` prevents upload storms.

    Parameters
    ----------
    upload_dir : Path | None
        Target directory for uploaded frames.
    upload_ratio : float
        Fraction of scored frames to upload (within the batch).
    trigger_count : int
        Min candidates before triggering an upload.
    high_score : float
        Immediate-upload threshold (0-1).
    min_interval_s : float
        Minimum seconds between consecutive uploads.
    max_upload_global : int
        Hard cap on total uploads across all batches.
    """

    def __init__(
        self,
        upload_dir: Path | None,
        upload_ratio: float = 0.20,
        trigger_count: int = 4,
        high_score: float = 0.75,
        min_interval_s: float = 15.0,
        max_upload_global: int = 240,
    ):
        self.upload_dir = upload_dir
        self.upload_ratio = float(upload_ratio)
        self.trigger_count = max(1, int(trigger_count))
        self.high_score = float(high_score)
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.max_upload_global = max(0, int(max_upload_global))

        self._candidates: list[dict] = []
        self._last_upload_time = 0.0
        self._uploaded_total = 0
        self._selected: list[dict] = []

    def add(self, frame_row: dict, score: float, alpha: float) -> bool:
        """Add a triggered frame to the candidate buffer.

        Returns True if an upload was launched (so the caller can log it).
        """
        import time

        frame_row['adaptive_score'] = float(score)
        frame_row['_adaptive_alpha'] = float(alpha)
        self._candidates.append(frame_row)

        now = time.time()
        # -- Trigger conditions --------------------------------------
        emergency = score >= self.high_score and len(self._candidates) >= 1
        batch_full = len(self._candidates) >= self.trigger_count
        can_upload = (now - self._last_upload_time) >= self.min_interval_s or emergency

        if (emergency or batch_full) and can_upload:
            self._flush(now)
            return True
        return False

    def flush_remaining(self) -> int:
        """Upload any remaining candidates (call at end of stream)."""
        import time
        if self._candidates:
            self._flush(time.time())
        return len(self._selected)

    def _flush(self, now: float) -> None:
        """Rank candidates by adaptive_score, apply dedup, copy top-K."""
        import shutil
        from pathlib import Path

        cand = self._candidates
        self._candidates = []
        self._last_upload_time = now

        if not cand:
            return

        # Rank by adaptive_score descending
        cand.sort(key=lambda x: float(x.get('adaptive_score', 0.0)), reverse=True)

        # Apply upload_ratio within this batch
        batch_quota = max(1, int(len(cand) * self.upload_ratio))
        # Enforce global cap
        remaining = self.max_upload_global - self._uploaded_total if self.max_upload_global > 0 else batch_quota
        n = min(batch_quota, remaining) if self.max_upload_global > 0 else batch_quota
        n = max(1, n)

        top = cand[:n]
        for item in top:
            src = Path(str(item.get('path', '')))
            if src.exists() and self.upload_dir is not None:
                dst = self.upload_dir / src.name
                try:
                    dst.unlink(missing_ok=True)
                    shutil.copy2(str(src), str(dst))
                    item['path'] = str(dst)
                except Exception:
                    pass
            item['selected'] = True
            self._selected.append(item)
            self._uploaded_total += 1

    @property
    def uploaded_count(self) -> int:
        return self._uploaded_total

    @property
    def selected_rows(self) -> list[dict]:
        return self._selected


def _flush_streaming_block(
    buffer: list[dict],
    selected_rows: list[dict],
    upload_dir: Path | None,
    strategy: str,
    upload_ratio: float,
    max_upload: int,
    min_frame_gap: int,
    dedup_hamming_threshold: int,
    seed: int,
    hard_score_threshold: float | None,
    max_candidate_pool_size: int,
    hard_score_quantile: float | None,
    prefilter_enabled: bool,
) -> None:
    """Run BHSS selection on a single streaming block and save uploads."""
    block_df = pd.DataFrame(buffer)

    # Edge prefilter: only edge-passing frames enter BHSS ranking
    if prefilter_enabled:
        cand = block_df[block_df['edge_pass']].copy()
    else:
        cand = block_df.copy()

    if cand.empty:
        return

    # BHSS candidate filtering
    # For strategies that rank by hard_score (ours/hardscore), the quantile
    # pre-filter is REDUNDANT [U+FFFD]?top-K selection already picks the best frames.
    # The quantile only shrinks the candidate pool and causes truncation at
    # higher upload_ratios. Keep it for random/confidence/behavior_drift so
    # those strategies still select from a hard-sample pool.
    if strategy in ('ours', 'hardscore', 'activity', 'adaptive'):
        # No quantile pre-filter: all frames are candidates; ranking+top-K does the work
        cand = cand.copy()
    else:
        threshold = _resolve_hard_score_threshold(cand, hard_score_threshold, hard_score_quantile)
        cand = _filter_candidates(cand, threshold, max_candidate_pool_size)
    if cand.empty:
        return

    # -- Adaptive scoring: per-block variance-weighted fusion ------
    if strategy == 'adaptive':
        cand = _compute_adaptive_score(cand)

    # Block quota
    block_quota = max(1, int(np.ceil(len(block_df) * float(upload_ratio))))
    # Don't exceed global max_upload (0 or negative = no cap)
    if max_upload is not None and int(max_upload) > 0:
        remaining_global = max(0, int(max_upload) - len(selected_rows))
        block_target = min(block_quota, remaining_global)
    else:
        block_target = block_quota
    if block_target <= 0:
        return

    # Rank within block
    ranked = _order_candidates(cand, strategy, seed + len(selected_rows))

    # Build dedup pool, then take top-K [U+FFFD]?ensures different ratios select
    # different numbers even when dedup shrinks the candidate pool
    block_idx = len(selected_rows) // max(1, block_quota)
    pool = _build_dedup_pool(ranked, min_frame_gap, dedup_hamming_threshold, block_index=block_idx)
    block_selected = pool[:block_target]

    # Copy selected frames from cache [U+FFFD]?upload directory
    for item in block_selected:
        src = Path(str(item.get('path', '')))
        if src.exists() and upload_dir is not None:
            dst = upload_dir / src.name
            try:
                dst.unlink(missing_ok=True)
                shutil.copy2(str(src), str(dst))
                item['path'] = str(dst)
            except Exception:
                pass  # keep original path if copy fails

    selected_rows.extend(block_selected)


def _resolve_hard_score_threshold(scores: pd.DataFrame, hard_score_threshold: float | None, hard_score_quantile: float | None = None) -> float | None:
    if hard_score_threshold is not None:
        return float(hard_score_threshold)
    if hard_score_quantile is None:
        return None
    q = float(hard_score_quantile)
    if q <= 0 or q >= 1 or 'hard_score' not in scores.columns:
        return None
    return float(scores['hard_score'].quantile(q))


def _filter_candidates(scores: pd.DataFrame, hard_score_threshold: float | None, max_candidate_pool_size: int | None) -> pd.DataFrame:
    cand = scores.copy()
    if hard_score_threshold is not None:
        cand = cand[cand['hard_score'] >= float(hard_score_threshold)].copy()
    if cand.empty:
        return cand
    if max_candidate_pool_size is not None and int(max_candidate_pool_size) > 0:
        cand = cand.sort_values('hard_score', ascending=False).head(int(max_candidate_pool_size)).copy()
    return cand


def _order_candidates(cand: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    if strategy == 'full_upload':
        return cand.sort_values('frame_index', ascending=True)
    if strategy == 'random':
        return cand.sample(frac=1.0, random_state=seed)
    if strategy == 'confidence':
        return cand.sort_values('avg_conf', ascending=True)
    if strategy in ('uncertainty', 'tta_uncertainty'):
        return cand.sort_values('tta_uncertainty', ascending=False)
    if strategy == 'behavior_drift':
        return cand.sort_values('behavior_drift', ascending=False)
    if strategy in ('ours', 'hardscore'):
        return cand.sort_values('hard_score', ascending=False)
    if strategy == 'activity':
        return cand.sort_values('non_lie_ratio', ascending=False)
    if strategy == 'adaptive':
        return cand.sort_values('adaptive_score', ascending=False)
    if strategy == 'source_only':
        return cand.iloc[0:0].copy()
    raise ValueError(f'Unknown sampling strategy: {strategy}')


def _get_dhash(row, dedup_hamming_threshold: int) -> np.ndarray | None:
    """Extract dhash from a row, preferring pre-computed streaming hash."""
    # Pre-computed during streaming (no disk read needed)
    if '_dhash_bytes' in row and row['_dhash_bytes'] is not None:
        try:
            return np.frombuffer(row['_dhash_bytes'], dtype=np.uint8)
        except Exception:
            pass
    # Fall back to disk read
    p = row.get('path', '')
    if p and Path(str(p)).exists():
        try:
            return dhash(read_image(Path(str(p))))
        except Exception:
            pass
    return None


def _build_dedup_pool(
    ordered: pd.DataFrame,
    min_frame_gap: int,
    dedup_hamming_threshold: int,
    block_index: int | None = None,
) -> list[dict]:
    """Build a deduplicated candidate pool from ranked frames.

    Iterates ranked frames in order (highest-scoring first), greedily
    keeping each frame that is NOT a near-duplicate of any already-kept
    frame.  Returns the FULL pool [U+FFFD]?no size limit.  The caller slices to
    the desired upload count so that different upload ratios always select
    different numbers of frames from the same dedup pool.

    Returns
    -------
    list[dict]
        Deduplicated candidate frames with ``selected=True`` and optional
        ``selection_block_index`` set.
    """
    pool: list[dict] = []
    pool_hashes: list[np.ndarray] = []
    pool_frames: list[int] = []
    for _, row in ordered.iterrows():
        frame_index = int(row['frame_index'])
        if any(abs(frame_index - f) < min_frame_gap for f in pool_frames):
            continue
        h = _get_dhash(row, dedup_hamming_threshold)
        if h is None:
            continue
        if dedup_hamming_threshold >= 0 and any(hamming(h, old) <= dedup_hamming_threshold for old in pool_hashes):
            continue
        item = row.to_dict()
        item['selected'] = True
        if block_index is not None:
            item['selection_block_index'] = block_index
        pool.append(item)
        pool_hashes.append(h)
        pool_frames.append(frame_index)
    return pool


def select_samples(scores: pd.DataFrame, strategy: str, upload_ratio: float, max_upload: int, min_frame_gap: int, dedup_hamming_threshold: int, seed: int = 42, hard_score_threshold: float | None = None, max_candidate_pool_size: int | None = None, hard_score_quantile: float | None = None) -> pd.DataFrame:
    if scores.empty:
        return scores.copy()
    # Skip quantile pre-filter for ours/hardscore/activity/adaptive [U+FFFD]?ranking+top-K is sufficient
    threshold = None
    if strategy in ('ours', 'hardscore', 'activity', 'adaptive'):
        cand = scores.copy()
    else:
        threshold = _resolve_hard_score_threshold(scores, hard_score_threshold, hard_score_quantile)
        cand = _filter_candidates(scores, threshold, max_candidate_pool_size)
    if cand.empty:
        return cand
    if strategy == 'full_upload':
        ordered = _order_candidates(cand, strategy, seed)
        n = len(ordered)
    elif strategy == 'source_only':
        return cand.iloc[0:0].copy()
    else:
        # -- Adaptive scoring: variance-weighted fusion ------------
        if strategy == 'adaptive':
            cand = _compute_adaptive_score(cand)
        ordered = _order_candidates(cand, strategy, seed)
        n = min(max_upload, max(1, int(len(cand) * upload_ratio))) if max_upload is not None and int(max_upload) > 0 else max(1, int(len(cand) * upload_ratio))

    pool = _build_dedup_pool(ordered, min_frame_gap, dedup_hamming_threshold)
    selected = pool[:n]
    out = pd.DataFrame(selected)
    if not out.empty:
        out['candidate_pool_size_after_threshold'] = len(cand)
        out['dedup_pool_size'] = len(pool)
        out['hard_score_threshold'] = threshold
        out['selection_mode'] = 'global'
    return out


def select_samples_online_blocks(scores: pd.DataFrame, strategy: str, upload_ratio: float, max_upload: int, min_frame_gap: int, dedup_hamming_threshold: int, seed: int = 42, hard_score_threshold: float | None = None, max_candidate_pool_size: int | None = None, hard_score_quantile: float | None = None, block_size: int = 120) -> pd.DataFrame:
    if scores.empty:
        return scores.copy()
    if strategy == 'full_upload':
        return select_samples(scores, strategy, 1.0, max_upload, min_frame_gap, dedup_hamming_threshold, seed, None, None)
    if strategy == 'source_only':
        return scores.iloc[0:0].copy()

    block_size = max(1, int(block_size))
    target_total = min(int(max_upload), max(1, int(len(scores) * float(upload_ratio)))) if max_upload is not None and int(max_upload) > 0 else max(1, int(len(scores) * float(upload_ratio)))
    ordered_stream = scores.sort_values('frame_index', ascending=True).reset_index(drop=True)
    selected: list[dict] = []
    candidate_pool_size = 0
    dedup_pool_sizes: list[int] = []

    for block_index, start in enumerate(range(0, len(ordered_stream), block_size)):
        if len(selected) >= target_total:
            break
        block = ordered_stream.iloc[start:start + block_size].copy()
        # Skip quantile pre-filter for ours/hardscore/activity/adaptive [U+FFFD]?ranking+top-K is sufficient
        if strategy in ('ours', 'hardscore', 'activity', 'adaptive'):
            cand = block.copy()
        else:
            block_threshold = _resolve_hard_score_threshold(block, hard_score_threshold, hard_score_quantile)
            cand = _filter_candidates(block, block_threshold, max_candidate_pool_size)
        candidate_pool_size += len(cand)
        if cand.empty:
            continue
        # -- Adaptive scoring: per-block variance-weighted fusion --
        if strategy == 'adaptive':
            cand = _compute_adaptive_score(cand)
        remaining = target_total - len(selected)
        block_quota = max(1, int(np.ceil(len(block) * float(upload_ratio))))
        n = min(remaining, block_quota)
        ranked = _order_candidates(cand, strategy, seed + block_index)
        pool = _build_dedup_pool(ranked, min_frame_gap, dedup_hamming_threshold, block_index)
        dedup_pool_sizes.append(len(pool))
        selected.extend(pool[:n])

    out = pd.DataFrame(selected)
    if not out.empty:
        out['candidate_pool_size_after_threshold'] = candidate_pool_size
        out['dedup_pool_size'] = int(sum(dedup_pool_sizes))
        out['hard_score_threshold'] = hard_score_threshold
        out['hard_score_quantile'] = hard_score_quantile
        out['selection_mode'] = 'online_blocks'
        out['selection_block_size'] = block_size
    return out
