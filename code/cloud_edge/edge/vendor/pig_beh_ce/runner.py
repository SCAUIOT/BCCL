from __future__ import annotations

import gc
import math
from pathlib import Path
import json
import random
import re
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import Experiment, set_seed, _deep_merge
from .files import list_images, list_videos
from .sampling import compute_hard_scores, compute_hard_scores_from_videos, compute_hard_scores_from_videos_streaming, ConfidenceDriftDetector, select_samples, select_samples_online_blocks, precompute_stage_scores, load_cached_scores, recompute_hard_score
from .edge_prefilter import apply_edge_prefilter, save_edge_prefiltered_images, cached_edge_prefilter_available, load_cached_edge_prefilter_images
from .dataset import export_selected_images, build_mixed_dataset, load_source_info
from .yolo_ops import evaluate_model, pseudo_label_with_teacher, train_student, normalize_names, write_dataset_yaml
from .report import save_csv, aggregate_results
from .gate import decide_gate, DriftBuffer, reconstruct_episodes, episode_image_paths
from .trigger import evaluate_trigger
from .gating_network import refresh_gating_network, gating_model_weights


UPDATE_METHODS = {'random', 'confidence', 'uncertainty', 'behavior_drift', 'ours', 'full_upload', 'activity', 'adaptive'}
SUPPORTED_MIX_MODES = {'source_plus_hard', 'hard_only'}


def copy_initial_model(src: Path, dst: Path, force_reset: bool = True) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if force_reset or not dst.exists():
        shutil.copy2(src, dst)
    return dst


def _source_images(exp: Experiment, cap: int | None = None) -> list[Path]:
    """Ordered list of source-domain train images (SDR replay pool).

    The source dataset is the 50 daytime images the student was
    pre-trained on.  *cap* limits the list to the first N images
    (deterministic order, via list_images sorted glob).
    """
    if not (exp.paths.source_yaml and exp.paths.source_yaml.exists()):
        return []
    train_img, _, _, _, _ = load_source_info(exp.paths.source_yaml)
    imgs = list_images(train_img)
    if cap is not None and len(imgs) > cap:
        imgs = imgs[:cap]
    return imgs


def _source_cap(source_count: int, exported_n: int, target_repeat: int, cap_frac: float) -> int:
    """R2: cap source replay at a fraction of the mixed training batch.

    ``cap_n = ceil(cap_frac * exported_n * target_repeat)``; the source
    images must never exceed this share of the (repeated) target frames.
    cap_frac <= 0 disables the cap.
    """
    if cap_frac <= 0 or exported_n <= 0:
        return source_count
    cap_n = int(math.ceil(float(cap_frac) * max(0, exported_n) * max(1, target_repeat)))
    return min(max(0, source_count), cap_n)


def _select_forgetting_prone(scored: list[tuple[Path, float]], final_n: int) -> list[Path]:
    """R3: pick the *final_n* lowest-confidence source images.

    ``scored`` = ``(path, mean_conf)`` from the current student; the lowest
    confidence means the model recalls it worst — the highest forgetting
    risk.  Deterministic: ties break on path name.
    """
    ordered = sorted(scored, key=lambda t: (t[1], str(t[0])))
    return [p for p, _ in ordered[:max(0, int(final_n))]]


def _split_target_train_val(
    train_images: list[Path], val_frac: float, min_val: int,
    seed: int, stage_index: int,
) -> tuple[list[Path], list[Path]]:
    """上传帧 → ``(训练集, 影子验证 holdout)``（确定性切分）。

    ``random.Random(seed + stage_index)`` 打乱后取前 ``val_n`` 帧为 holdout，
    其余为训练集。**两处共用本函数**以保证拿到同一个切分：
      * :func:`_run_update` —— 候选训练 / 影子验证；
      * B1b 换起点的目标域判据（``swap_validate_on_target``）—— 用**训练部分**
        当判据帧，与候选 holdout 互斥，避免"选起点的信息泄漏进候选验证"。
    """
    if len(train_images) <= 1 or val_frac <= 0.0:
        return [Path(p) for p in train_images], []
    rng = __import__('random').Random(int(seed) + int(stage_index))
    shuffled = list(train_images)
    rng.shuffle(shuffled)
    val_n = max(int(min_val), int(round(len(shuffled) * float(val_frac))))
    val_n = min(val_n, len(shuffled) - 1)
    return ([Path(p) for p in shuffled[val_n:]],
            [Path(p) for p in shuffled[:val_n]])


def _should_swap_pretrain(scene_period: str, gain: float, margin: float) -> tuple[bool, str]:
    """B1b swap decision (pure function, smoke-testable).

    A bank spare may become the pre-training start point only when the
    stage scene is day/eat (the source jury is daytime data — using it to
    judge a night swap would measure night with a day ruler) AND the spare
    beats the current model on the source images by *margin*.
    """
    if scene_period not in ('day', 'eat'):
        return False, 'night_scene_forbidden'
    if gain is None or gain <= margin:
        return False, f'gain_below_margin_{gain:.4f}' if gain is not None else 'gain_none'
    return True, f'gain_above_margin_{gain:.4f}'


def get_behavior_names(exp: Experiment) -> dict[int, str]:
    if exp.paths.source_yaml and exp.paths.source_yaml.exists():
        try:
            _, _, _, _, cfg = load_source_info(exp.paths.source_yaml)
            return normalize_names(cfg.get('names'))
        except Exception:
            pass
    for stage in exp.stage_names():
        data_yaml = exp.stage_test_root(stage) / 'data.yaml'
        if not data_yaml.exists():
            continue
        try:
            import yaml
            with open(data_yaml, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            return normalize_names(cfg.get('names'))
        except Exception:
            pass
    return {0: 'stand', 1: 'lie', 2: 'eat'}


def eval_on_stages(model_path: Path, exp: Experiment, stages: list[str], imgsz: int, device: Any, names: dict[int, str]) -> pd.DataFrame:
    rows = []
    for idx, s in enumerate(stages, start=1):
        test_root = exp.stage_test_root(s)
        if (test_root / 'images' / 'test').exists():
            met = evaluate_model(model_path, test_root, imgsz=imgsz, device=device, names=names)
            rows.append({'eval_stage_index': idx, 'eval_stage': s, **met})
    return pd.DataFrame(rows)


def save_continual_matrix(matrix_rows: list[dict], reports_dir: Path, metric: str = 'mAP50') -> None:
    if not matrix_rows:
        return
    df = pd.DataFrame(matrix_rows)
    df.to_csv(reports_dir / 'seen_stage_matrix.csv', index=False, encoding='utf-8-sig')
    if metric not in df.columns:
        return
    pivot = df.pivot_table(index='after_stage', columns='eval_stage', values=metric, aggfunc='last')
    pivot.to_csv(reports_dir / f'seen_stage_{metric}_matrix.csv', encoding='utf-8-sig')
    final_idx = int(df['after_stage_index'].max())
    summary_rows = []
    for eval_stage, g in df.groupby('eval_stage'):
        g = g.sort_values('after_stage_index')
        best = float(g[metric].max())
        final_values = g[g['after_stage_index'] == final_idx][metric]
        final = float(final_values.iloc[-1]) if not final_values.empty else float(g[metric].iloc[-1])
        summary_rows.append({
            'eval_stage': eval_stage,
            'best_seen_metric': best,
            'final_metric': final,
            'forgetting': max(0.0, best - final),
        })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.loc[len(summary)] = {
            'eval_stage': 'AVERAGE',
            'best_seen_metric': float(summary['best_seen_metric'].mean()),
            'final_metric': float(summary['final_metric'].mean()),
            'forgetting': float(summary['forgetting'].mean()),
        }
    summary.to_csv(reports_dir / f'forgetting_summary_{metric}.csv', index=False, encoding='utf-8-sig')


def total_file_bytes(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        try:
            total += int(Path(p).stat().st_size)
        except OSError:
            pass
    return total


def normalize_mix_mode(mix_mode: str | None, default: str = 'source_plus_hard') -> str:
    mode = str(mix_mode or default or 'source_plus_hard').strip()
    if mode not in SUPPORTED_MIX_MODES:
        raise ValueError(f'Unsupported mix_mode={mode}. Expected one of {sorted(SUPPORTED_MIX_MODES)}')
    return mode


def eval_on_history(model_path: Path, exp: Experiment, stages: list[str], imgsz: int, device: Any, names: dict[int, str]) -> pd.DataFrame:
    rows = []
    for s in stages:
        test_root = exp.stage_test_root(s)
        if (test_root / 'images' / 'test').exists():
            met = evaluate_model(model_path, test_root, imgsz=imgsz, device=device, names=names)
            rows.append({'stage': s, **met})
    return pd.DataFrame(rows)


def _mean_confidence(model_path: Path, image_paths: list[Path], imgsz: int, device: str) -> float:
    """Mean detection confidence of a model on a set of images.

    Runs YOLO inference on each image and collects all detection confidences.
    Returns 0.0 if no detections or all images fail.
    """
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    all_confs = []
    try:
        for p in image_paths:
            if not Path(p).exists():
                continue
            try:
                results = model(str(p), imgsz=imgsz, device=device, verbose=False)
                for r in results:
                    if r.boxes is not None and len(r.boxes.conf) > 0:
                        all_confs.extend(r.boxes.conf.cpu().tolist())
            except Exception:
                continue
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return float(np.mean(all_confs)) if all_confs else 0.0


def _image_confidence_map(model_path: Path, image_paths: list[Path], imgsz: int, device: str) -> list[tuple[Path, float]]:
    """Per-image mean detection confidence (R3 forgetting probe).

    Scores each image with the model and returns ``(path, mean_conf)``
    pairs; images with no detections score 0.0 (highest forgetting risk).
    Reuses the same YOLO inference pattern as :func:`_mean_confidence`
    but keeps the per-image granularity the selective replay needs.
    """
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    scores: list[tuple[Path, float]] = []
    try:
        for p in image_paths:
            if not Path(p).exists():
                continue
            try:
                results = model(str(p), imgsz=imgsz, device=device, verbose=False)
                confs = []
                for r in results:
                    if r.boxes is not None and len(r.boxes.conf) > 0:
                        confs.extend(r.boxes.conf.cpu().tolist())
                scores.append((Path(p), float(np.mean(confs)) if confs else 0.0))
            except Exception:
                continue
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores


def _compute_model_sizes(model_path: Path, deployment_mode: str) -> dict[str, float]:
    """Compute full and head-only model sizes for deployment comparison.

    When ``deployment_mode == 'partial'``, extracts only the Detect-head
    weights to estimate communication savings.  The full model is always
    used for evaluation in both modes -- only the *deployment payload*
    differs.
    """
    full_bytes = model_path.stat().st_size
    result: dict[str, float] = {
        'model_full_bytes': float(full_bytes),
        'model_full_mb': full_bytes / (1024 * 1024),
    }
    if deployment_mode == 'delta_sparse':
        # YOLOv8n ≈ 3M params; top-5% → 150k weights × 8 B (int32 idx + float32 val)
        # = ~1.17 MB uncompressed.  With gzip on sparse indices, ~300-400 KB.
        total_p = 3_012_000          # YOLOv8n approximate param count
        transmitted_p = int(total_p * 0.05)
        payload_bytes = transmitted_p * 8   # 4B index + 4B float32
        result['model_head_bytes'] = float(payload_bytes)
        result['model_head_mb'] = payload_bytes / (1024 * 1024)
        result['model_savings_pct'] = (1.0 - payload_bytes / max(1, full_bytes)) * 100.0
        result['delta_payload_mb'] = payload_bytes / (1024 * 1024)
        result['delta_sparsity_pct'] = 95.0
    elif deployment_mode == 'partial':
        try:
            import tempfile
            from ultralytics import YOLO as _YOLO
            head_model = _YOLO(str(model_path))
            head_state = head_model.model.model[-1].state_dict()
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                torch.save(head_state, f.name)
                head_bytes = f.name
            head_size = Path(head_bytes).stat().st_size
            Path(head_bytes).unlink(missing_ok=True)
            result['model_head_bytes'] = float(head_size)
            result['model_head_mb'] = head_size / (1024 * 1024)
            result['model_savings_pct'] = (1.0 - head_size / max(1, full_bytes)) * 100.0
        except Exception:
            result['model_head_bytes'] = 0.0
            result['model_head_mb'] = 0.0
            result['model_savings_pct'] = 0.0
    else:
        result['model_head_bytes'] = float(full_bytes)
        result['model_head_mb'] = full_bytes / (1024 * 1024)
        result['model_savings_pct'] = 0.0
    return result


def _deploy_partial_model(initial_student: Path, candidate: Path, output: Path) -> None:
    """Create a head-only deployed model for the edge.

    Simulates partial deployment: the edge keeps its original backbone
    and only the new detection head is transferred from cloud to edge.
    The resulting model = initial backbone + trained candidate head.

    The *full* candidate is still stored in the model bank for accurate
    cosine-similarity retrieval; only the edge-side ``current_model``
    is head-only.
    """
    from ultralytics import YOLO as _YOLO
    base = _YOLO(str(initial_student))
    cand = _YOLO(str(candidate))
    # Copy the candidate's Detect head weights into the initial backbone
    base.model.model[-1].load_state_dict(cand.model.model[-1].state_dict())
    base.save(str(output))


def _deploy_delta_sparse(
    old_model: Path,
    candidate: Path,
    output: Path,
    top_k: float = 0.05,
) -> dict[str, float]:
    """Deploy via delta sparsification — only top-k% largest weight changes.

    **Cloud side**:
    1. Compute Δ = W_candidate − W_edge for every parameter.
    2. Keep only the top-*top_k* % of weights by |Δ| (the rest → 0).
    3. The sparse Δ payload = {indices, values} for transmitted weights.

    **Edge side**:
        W_deployed = W_edge + Δ_sparse

    Unlike partial deployment (head-only), this updates the full model
    including backbone — just sparsely.  The edge model stays structurally
    identical to the full candidate; only small-magnitude weight changes
    are dropped.

    Parameters
    ----------
    old_model : Path
        Currently deployed model on the edge (from the previous stage).
    candidate : Path
        Newly trained full model on the cloud.
    output : Path
        Where to write the deployed model.
    top_k : float
        Fraction of weights to transmit (default 0.05 = 5 %).

    Returns
    -------
    dict with keys: delta_total_params, delta_transmitted_params,
    delta_sparsity_pct, delta_payload_mb.
    """
    import numpy as _np

    old_ckpt = torch.load(str(old_model), map_location='cpu', weights_only=False)
    new_ckpt = torch.load(str(candidate), map_location='cpu', weights_only=False)

    old_sd = _get_state_dict(old_ckpt)
    new_sd = _get_state_dict(new_ckpt)

    total_params = 0
    transmitted_params = 0

    for key in new_sd:
        if key not in old_sd:
            continue
        w_old = old_sd[key]
        w_new = new_sd[key]
        if w_old.shape != w_new.shape:
            continue
        if w_old.numel() == 0:
            continue

        delta = w_new.float() - w_old.float()
        n = delta.numel()
        k = max(1, int(n * top_k))

        # Top-k by absolute delta magnitude
        delta_flat = delta.flatten()
        _, top_indices = torch.topk(delta_flat.abs(), k)

        mask_flat = torch.zeros(n, dtype=torch.bool)
        mask_flat[top_indices] = True
        mask = mask_flat.reshape(delta.shape)

        # Apply only the top-k deltas
        w_deployed = (w_old.float() + delta * mask.float()).to(w_old.dtype)
        old_sd[key] = w_deployed

        total_params += n
        transmitted_params += k

    # -- Save deployed model in the same format as old_ckpt ----------
    _write_state_dict(old_ckpt, old_sd)
    torch.save(old_ckpt, str(output))

    sparsity_pct = (1.0 - transmitted_params / max(1, total_params)) * 100.0
    # Estimate payload: each entry = 4 B (int32 index) + value (same dtype as model)
    dtype_bytes = 4  # float32
    payload_bytes = transmitted_params * (4 + dtype_bytes)
    payload_mb = payload_bytes / (1024 * 1024)

    return {
        'delta_total_params': float(total_params),
        'delta_transmitted_params': float(transmitted_params),
        'delta_sparsity_pct': float(sparsity_pct),
        'delta_payload_mb': float(payload_mb),
    }


def _get_state_dict(ckpt: dict) -> dict[str, torch.Tensor]:
    """Extract a module state_dict from an ultralytics-style checkpoint.

    ultralytics saves weights under 'model' key (float weights) or as a
    nested 'ema' / direct state_dict.  Try the standard locations.
    Shared by delta deployment and the B3 gating ensemble.
    """
    for key in ('model', 'ema'):
        sd = ckpt.get(key)
        if sd is not None:
            if hasattr(sd, 'state_dict'):
                return sd.state_dict()
            if hasattr(sd, 'float'):
                return sd.float().state_dict()
            if isinstance(sd, dict):
                # Check one value — if it's a Tensor, it's already a state_dict
                first_val = next(iter(sd.values()), None)
                if isinstance(first_val, torch.Tensor):
                    return sd
    # Fallback: iterate top-level keys; if most values are Tensors, use as-is
    tensors = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if len(tensors) > 10:
        return tensors
    raise ValueError('Cannot locate model state_dict in checkpoint')


def _write_state_dict(ckpt: dict, sd: dict[str, torch.Tensor]) -> None:
    """Write *sd* back into *ckpt* at the same key it was found."""
    for key in ('model', 'ema'):
        obj = ckpt.get(key)
        if obj is not None:
            if hasattr(obj, 'state_dict'):
                obj.load_state_dict(sd)
                return
            if hasattr(obj, 'float'):
                obj.float().load_state_dict(sd)
                return
            if isinstance(obj, dict):
                first_val = next(iter(obj.values()), None)
                if isinstance(first_val, torch.Tensor):
                    ckpt[key] = sd
                    return
    # Fallback: write directly to top level
    for k in list(ckpt.keys()):
        if isinstance(ckpt[k], torch.Tensor) and k in sd:
            ckpt[k] = sd[k]


def _gating_ensemble_deploy(
    candidate: Path,
    current: Path,
    gating_dir: Path,
    gating_images: list[Path],
    device: Any,
    alpha: float = 0.7,
    top_k: int = 3,
    bank_csv: Path | None = None,
    scene_period: str | None = None,
) -> dict[str, Any]:
    """B3 (optional): deploy the candidate as a gating-weighted average.

    w = alpha * w_candidate + (1 - alpha) * sum_i g_i * w_i, over the top-k
    same-scene spare experts ranked by the learned gating selector (P2).
    The result is written to *current* as a single YOLO checkpoint, so the
    edge runs one model — zero inference overhead.  Only candidate weights
    present in the experts with identical shapes are averaged; floating-
    point tensors only (BN counters etc. keep the candidate value), so a
    structurally inconsistent expert can never corrupt the checkpoint.

    Returns an info dict (``gating_ensemble_applied``, experts used, ...);
    every failure mode degrades to "not applied" (caller falls back to the
    plain candidate copy).
    """
    info: dict[str, Any] = {
        'gating_ensemble_applied': False,
        'gating_ensemble_alpha': float(alpha),
        'gating_ensemble_top_k': int(top_k),
    }
    weights, _ = gating_model_weights(gating_dir, gating_images or [], device, max_images=24)
    if not weights:
        info['gating_ensemble_reason'] = 'no_gating_snapshot'
        return info

    # Exclude self (candidate / current) and enforce the scene prior via
    # the bank metadata, then keep the top-k experts and renormalize.
    cand_s = str(candidate)
    curr_s = str(current)
    pool = {p: float(w) for p, w in weights.items() if str(p) != cand_s and str(p) != curr_s}
    if bank_csv is not None and Path(bank_csv).exists() and scene_period:
        try:
            bank_df = pd.read_csv(bank_csv)
            if 'scene_period' in bank_df.columns:
                same_scene = set(str(p) for p in bank_df.loc[
                    bank_df['scene_period'].astype(str).str.strip().str.lower() == scene_period, 'model_path'])
                pool = {p: w for p, w in pool.items() if p in same_scene}
        except Exception:
            pass
    if not pool:
        info['gating_ensemble_reason'] = 'no_eligible_experts'
        return info
    ranked = sorted(pool.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(top_k))]
    w_sum = float(sum(w for _, w in ranked))
    if w_sum <= 0:
        info['gating_ensemble_reason'] = 'zero_weight_sum'
        return info
    g = {p: w / w_sum for p, w in ranked}

    cand_ckpt = torch.load(str(candidate), map_location='cpu', weights_only=False)
    cand_sd = _get_state_dict(cand_ckpt)
    expert_sds: list[dict[str, torch.Tensor] | None] = []
    for p in g:
        try:
            expert_sds.append(_get_state_dict(torch.load(p, map_location='cpu', weights_only=False)))
        except Exception:
            expert_sds.append(None)

    merged: dict[str, torch.Tensor] = {}
    for key, t in cand_sd.items():
        if not t.is_floating_point():
            merged[key] = t
            continue
        acc = torch.zeros_like(t, dtype=torch.float32)
        used = 0
        for sd, w_i in zip(expert_sds, g.values()):
            if sd is None or key not in sd:
                continue
            if sd[key].shape != t.shape:
                continue
            acc += float(w_i) * sd[key].float()
            used += 1
        merged[key] = (float(alpha) * t.float() + (1.0 - float(alpha)) * acc).to(t.dtype) if used else t
    _write_state_dict(cand_ckpt, merged)
    torch.save(cand_ckpt, str(current))
    del cand_ckpt, cand_sd, expert_sds, merged
    gc.collect()
    info.update({
        'gating_ensemble_applied': True,
        'gating_ensemble_experts': len(g),
        'gating_ensemble_weight_sum': w_sum,
    })
    return info


# ── 方案 A：训练后权重插值锚定（post-hoc weight interpolation anchoring）──
# 动机（2026-09-25，八批实证）：无源回放、或源域被大量目标帧稀释的持续微调，
# 会让 backbone 向目标域漂移，最先丢失"拥挤夜间躺卧"场景的实例分离能力
# （召回掉 8–16 pp，而条件准确率不动甚至更高；mAP50 同步下滑，故非阈值假象）。
# 修法：训练产出候选 θ_ft 后，与锚定模型 θ_anchor（默认冻结源模型）做线性插值
#   θ(α) = α·θ_anchor + (1-α)·θ_ft
# 再用**与影子验证同口径**的 holdout 平均置信度，在**固定 α 网格**上选优，
# 胜出者覆盖候选。α 网格强制包含 0.0（= 原候选），故该机制按构造不会让任一
# 阶段变差；诊断信息（各 α 得分、选中 α）写进 decision JSON 便于事后分析。
#
# 注意：α 网格必须固定并写进配置公开，不得按结果调整，否则等同调参。

_POST_ANCHOR_SD_CACHE: dict[str, dict] = {}


def _model_state_dict(model_path: Path) -> dict:
    """读 YOLO checkpoint 的浮点 state_dict（CPU / float32）。"""
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    try:
        return {k: v.detach().float().cpu() for k, v in m.model.state_dict().items()}
    finally:
        from .yolo_ops import _free_yolo
        _free_yolo(m)


def anchor_state_dict(anchor_path: Path) -> dict:
    """带缓存读取锚定权重。

    缓存键**必须**包含文件的 (mtime, size)：``anchor: prev`` 指向的
    ``models/current_student.pt`` 会被逐阶段原地覆盖，若只按路径做键，后 11 个
    阶段会一直复用第 1 阶段的快照，等价于把 anchor=prev 静默降级成 anchor=source
    （2026-09-25 V20 实测：✓×× 臂与 V19 的 source 锚点逐位同结果，即此因）。
    """
    p = Path(anchor_path)
    st = p.stat()
    key = f'{p.resolve()}|{st.st_mtime_ns}|{st.st_size}'
    if key not in _POST_ANCHOR_SD_CACHE:
        _POST_ANCHOR_SD_CACHE.clear()      # 只保留最新一份，避免 12 份权重常驻内存
        _POST_ANCHOR_SD_CACHE[key] = _model_state_dict(p)
    return _POST_ANCHOR_SD_CACHE[key]


def interpolate_weights(anchor_path: Path, candidate_path: Path, alpha: float,
                        out_path: Path, anchor_sd: dict | None = None) -> None:
    """写出 θ = alpha·θ_anchor + (1-alpha)·θ_candidate 的 checkpoint。

    alpha=0 时输出与 candidate 逐位相同（shape 不匹配或非浮点张量保持原值）。
    """
    from ultralytics import YOLO
    sd_a = anchor_sd if anchor_sd is not None else anchor_state_dict(anchor_path)
    model = YOLO(str(candidate_path))
    try:
        sd_b = {k: v.detach().float().cpu() for k, v in model.model.state_dict().items()}
        merged = {}
        for k, vb in sd_b.items():
            va = sd_a.get(k)
            merged[k] = (alpha * va + (1.0 - alpha) * vb) if (
                va is not None and tuple(va.shape) == tuple(vb.shape)) else vb
        model.model.load_state_dict(merged)
        model.save(str(out_path))
    finally:
        from .yolo_ops import _free_yolo
        _free_yolo(model)


def apply_post_anchor(cfg: dict, *, anchor_path: Path, cand_path: Path,
                      val_paths: list[Path], imgsz: int, device: str,
                      out_dir: Path, stage: str) -> dict:
    """在固定 α 网格上生成插值候选，按 holdout 平均置信度选优并覆盖 cand_path。"""
    alphas = [float(a) for a in cfg.get('alphas', [0.0, 0.2, 0.4, 0.6])]
    if 0.0 not in alphas:
        alphas = [0.0] + alphas          # α=0（原候选）必须始终在候选集里
    if not val_paths or len(alphas) <= 1:
        return {'post_anchor_enabled': True, 'post_anchor_alpha': 0.0,
                'post_anchor_scores': {},
                'post_anchor_note': 'no holdout frames -> alpha=0 (原候选不变)'}

    anchor_sd = anchor_state_dict(anchor_path)
    base_conf = float(_mean_confidence(cand_path, val_paths, imgsz, device))
    scores: dict[str, float] = {'0': base_conf}
    best_alpha, best_conf, best_path = 0.0, base_conf, None

    for a in alphas:
        if a == 0.0:
            continue
        tag = f'{a:g}'.replace('.', 'p')
        p = out_dir / f'{cand_path.stem}_anchor{tag}.pt'
        interpolate_weights(anchor_path, cand_path, a, p, anchor_sd=anchor_sd)
        c = float(_mean_confidence(p, val_paths, imgsz, device))
        scores[f'{a:g}'] = c
        if c > best_conf:
            best_alpha, best_conf, best_path = a, c, p
        else:
            p.unlink(missing_ok=True)

    if best_path is not None:
        shutil.copy2(best_path, cand_path)          # 胜出者覆盖候选
        best_path.unlink(missing_ok=True)

    print(f'[post_anchor] stage={stage} anchor={anchor_path.name} '
          f'scores={ {k: round(v, 4) for k, v in scores.items()} } '
          f'-> alpha={best_alpha:g} (gain {best_conf - base_conf:+.4f})')
    return {
        'post_anchor_enabled': True,
        'post_anchor_anchor': anchor_path.name,
        'post_anchor_alpha': best_alpha,
        'post_anchor_conf': best_conf,
        'post_anchor_base_conf': base_conf,
        'post_anchor_gain': best_conf - base_conf,
        'post_anchor_scores': scores,
    }


def shadow_validate(
    old_model: Path,
    candidate_model: Path,
    validation_image_paths: list[Path],
    imgsz: int,
    device: str,
    min_gain: float = 0.0,
) -> tuple[bool, dict]:
    """Shadow-deploy candidate model and validate on **unseen** frames.

    Unlike the old ``should_accept`` which tested on training frames
    (circular -- the candidate was just trained on them), this function
    validates on *future* frames that neither model has seen.  This is
    the true edge-side validation pattern:

    1. Old model processes validation frames -> old_mean_conf
    2. Candidate shadow-runs on same frames -> new_mean_conf
    3. Accept only if candidate > old (no forgetting on unseen data)

    A minimum confidence gain (``min_gain``) can optionally be required
    to prevent accepting trivial improvements.

    Parameters
    ----------
    old_model : Path
        Currently deployed model.
    candidate_model : Path
        Newly trained candidate model.
    validation_image_paths : list[Path]
        **Unseen** validation frames (NOT the training frames).
    imgsz : int
        YOLO input size.
    device : str
        Inference device.
    min_gain : float
        Minimum mean-confidence improvement required (default 0.0).

    Returns
    -------
    (accepted, info) : tuple[bool, dict]
    """
    if not validation_image_paths:
        return True, {
            'old_mean_conf': 0.0, 'new_mean_conf': 0.0, 'conf_gain': 0.0,
            'eval_frame_count': 0, 'decision': 'accept_by_default',
        }

    old_conf = _mean_confidence(old_model, validation_image_paths, imgsz, device)
    new_conf = _mean_confidence(candidate_model, validation_image_paths, imgsz, device)
    conf_gain = new_conf - old_conf
    accepted = conf_gain > float(min_gain)

    return accepted, {
        'old_mean_conf': float(old_conf),
        'new_mean_conf': float(new_conf),
        'conf_gain': float(conf_gain),
        'eval_frame_count': len(validation_image_paths),
        'decision': 'accept' if accepted else 'reject',
    }


def model_bank_score(conf_gain: float, accept_count: int = 0, use_count: int = 0) -> tuple[float, dict]:
    """Score a model for bank ranking -- no labeled test set needed.

    Pattern: system project's ModelZoo metadata-fallback selection
    (accept_count + use_count as primary signals), augmented with
    confidence improvement on the frames that triggered this training.

    Parameters
    ----------
    conf_gain : float
        Mean confidence improvement on the drift/uploaded frames.
    accept_count : int
        Number of times this model was accepted by the edge.
    use_count : int
        Number of times this model was selected for deployment.
    """
    quality = float(np.clip(conf_gain * 5.0, 0.0, 1.0)) if conf_gain > 0 else 0.0
    score = float(
        0.60 * quality +
        0.25 * min(float(accept_count) / 5.0, 1.0) +
        0.15 * min(float(use_count) / 10.0, 1.0)
    )
    return score, {
        'score': score,
        'conf_gain': float(conf_gain),
        'accept_count': int(accept_count),
        'use_count': int(use_count),
    }


def _bank_use_times(bank: pd.DataFrame) -> pd.Series:
    """Read recency, using creation time for entries from older bank CSVs."""
    created = pd.to_numeric(bank.get('created_time', pd.Series(0, index=bank.index)),
                            errors='coerce').fillna(0)
    used = pd.to_numeric(bank.get('last_used_time', created), errors='coerce')
    return used.fillna(created).astype('int64')


def _next_bank_use_time(bank: pd.DataFrame) -> int:
    # Strictly increasing even for multiple accesses in the same millisecond,
    # or if the system clock moves backwards between updates.
    times = _bank_use_times(bank)
    previous = int(times.max()) if not times.empty else 0
    return max(int(time.time() * 1000), previous + 1)


def update_model_bank(
    candidate_model: Path,
    bank_dir: Path,
    stage: str,
    method_variant: str,
    conf_gain: float,
    scene_fingerprint: dict,
    cfg: dict,
    accept_count: int = 0,
    use_count: int = 0,
    state: str = 'active',
) -> dict:
    """Register a candidate model in the persistent model bank.

    Spare-parts-warehouse semantics (system-style, ModelZoo pattern):
    the bank holds *proven* models that may be re-deployed when a future
    candidate is rejected.  Rejected candidates are not spare parts --
    they are either recorded with ``state='rejected'`` (excluded from
    retrieval) or not stored at all (``store_rejected: false``, default).

    No labeled test set required.  Scoring uses confidence improvement
    on drift frames + edge feedback (accept_count / use_count) following
    the pattern of system project's ModelZoo.

    Each model entry carries a *scene_fingerprint* -- a compact statistical
    summary of the drift frames -- enabling scene-aware retrieval via
    :func:`query_model_bank`, and a *scene_period* (day/night/eat parsed
    from the stage name) used as the cycle_stage/time_of_day prior,
    same as system's ModelZoo metadata filter.

    ``model_bank.policy: paper`` evicts the least recently used entry.
    The legacy policy retains the original count-based pruning for replaying
    historical experiments.
    """
    mb = cfg.get('model_bank', {})
    paper_policy = str(mb.get('policy', 'legacy')).strip().lower() == 'paper'
    if not bool(mb.get('enabled', True)):
        return {'model_bank_enabled': False}
    top_k = int(mb.get('top_k', 10))
    prune_mode = 'lru' if paper_policy else str(mb.get('prune_strategy', 'lru')).strip().lower()
    store_rejected = not paper_policy and bool(mb.get('store_rejected', False))
    state = str(state).strip().lower()
    if state not in ('active', 'rejected'):
        state = 'active'

    # Rejected candidates are not spare parts: skip storing them entirely
    # unless store_rejected is set (bookkeeping only -- they stay unmatchable).
    if state == 'rejected' and not store_rejected:
        return {
            'model_bank_enabled': True,
            'model_bank_top_k': top_k,
            'model_bank_state': 'rejected',
            'model_bank_kept': False,
            'model_bank_path': '',
            'model_bank_score': 0.0,
            'model_bank_removed': 0,
            'model_bank_prune_strategy': prune_mode,
        }

    bank_dir.mkdir(parents=True, exist_ok=True)
    csv_path = bank_dir / 'model_bank.csv'

    # Pre-process: ensure existing rows have required columns
    bank = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    for col in ['use_count', 'accept_count', 'state', 'scene_period', 'scene_fingerprint']:
        if col not in bank.columns:
            bank[col] = '' if col in ('state', 'scene_period', 'scene_fingerprint') else 0
    # String columns may arrive as float64 when the CSV contains only
    # empty values -- cast before backfilling string values below.
    for col in ['state', 'scene_period', 'scene_fingerprint']:
        if col in bank.columns:
            bank[col] = bank[col].astype(object)
    for idx, row in bank.iterrows():
        for col in ['use_count', 'accept_count']:
            cnt = row.get(col, 0)
            if pd.isna(cnt):
                bank.at[idx, col] = 0
        if pd.isna(row.get('state', '')) or str(row.get('state', '')).strip() == '':
            bank.at[idx, 'state'] = 'active'   # legacy rows: treat as active
        if pd.isna(row.get('scene_period', '')) or str(row.get('scene_period', '')).strip() == '':
            bank.at[idx, 'scene_period'] = _scene_period(str(row.get('stage', '')))

    score, score_info = model_bank_score(conf_gain, accept_count, use_count)
    safe_stage = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in stage)
    safe_method = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in method_variant)
    timestamp = _next_bank_use_time(bank)
    bank_model = bank_dir / f'{safe_method}_{safe_stage}_score{score:.4f}_{timestamp}.pt'
    shutil.copy2(candidate_model, bank_model)

    new_row = {
        'model_path': str(bank_model),
        'method_variant': method_variant,
        'stage': stage,
        'state': state,
        'scene_period': _scene_period(stage),
        'created_time': timestamp,
        'last_used_time': timestamp,
        'use_count': use_count,
        'accept_count': accept_count,
        'scene_fingerprint': json.dumps(scene_fingerprint, ensure_ascii=False) if scene_fingerprint else '{}',
        **score_info,
    }
    bank = pd.concat([bank, pd.DataFrame([new_row])], ignore_index=True)

    # Paper: genuine recency-based LRU. Legacy: preserve the historical
    # count-based ordering used by the original experiment configs.
    bank['_is_rejected'] = bank['state'].astype(str).str.strip().str.lower() == 'rejected'
    if paper_policy:
        bank['last_used_time'] = _bank_use_times(bank)
        bank = bank.sort_values('last_used_time', ascending=False,
                                kind='stable').reset_index(drop=True)
    elif prune_mode == 'lru':
        bank = bank.sort_values(
            by=['_is_rejected', 'use_count', 'accept_count', 'created_time'],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    else:
        bank = bank.sort_values(
            by=['_is_rejected', 'score'],
            ascending=[True, False],
        ).reset_index(drop=True)

    bank['rank'] = range(1, len(bank) + 1)
    keep = bank.head(top_k).copy()
    keep_paths = set(str(Path(p)) for p in keep['model_path'].tolist())

    removed = 0
    for _, row in bank.iloc[top_k:].iterrows():
        path = Path(str(row['model_path']))
        if path.exists():
            path.unlink()
            removed += 1

    keep = keep.drop(columns=['_is_rejected'])
    keep.to_csv(csv_path, index=False, encoding='utf-8-sig')
    kept = str(bank_model) in keep_paths

    return {
        'model_bank_enabled': True,
        'model_bank_top_k': top_k,
        'model_bank_state': state,
        'model_bank_score': score,
        'model_bank_kept': kept,
        'model_bank_path': str(bank_model) if kept else '',
        'model_bank_removed': removed,
        'model_bank_prune_strategy': prune_mode,
    }


def mark_model_used(bank_dir: Path, model_path: str) -> bool:
    """Record an actual use and refresh recency for standard LRU pruning."""
    csv_path = bank_dir / 'model_bank.csv'
    if not csv_path.exists():
        return False
    bank = pd.read_csv(csv_path)
    if 'use_count' not in bank.columns:
        bank['use_count'] = 0
    mask = bank['model_path'] == model_path
    if not mask.any():
        return False
    next_used = _next_bank_use_time(bank)
    bank['last_used_time'] = _bank_use_times(bank)
    bank.loc[mask, 'use_count'] = bank.loc[mask, 'use_count'].fillna(0).astype(int) + 1
    bank.loc[mask, 'last_used_time'] = next_used
    bank.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return True


def select_best_model(
    bank_dir: Path,
    stage: str | None = None,
    prefer_stage: bool = True,
) -> str | None:
    """Select the best model from the bank for deployment.

    Pattern: ``ModelZoo.select_model_for_edge()`` in D:\\project\\system.

    Strategy (ordered priority):
    1. If *prefer_stage* and *stage* provided: models matching the stage
       that have the highest ``use_count + score`` rank.
    2. Otherwise: the highest ``use_count``, then ``accept_count``,
       then ``score``.

    Parameters
    ----------
    bank_dir : Path
        Model bank directory.
    stage : str | None
        Current stage name for stage-biased selection.
    prefer_stage : bool
        If True, prefer models trained on the same stage.

    Returns
    -------
    str | None
        Path to the best model file, or None if the bank is empty.
    """
    csv_path = bank_dir / 'model_bank.csv'
    if not csv_path.exists():
        return None
    bank = pd.read_csv(csv_path)
    if bank.empty:
        return None

    # Exclude rejected models (spare-parts warehouse: only proven parts).
    if 'state' in bank.columns:
        state_s = bank['state'].astype(str).str.strip().str.lower()
        bank = bank[state_s != 'rejected']
    if bank.empty:
        return None

    # Ensure required columns
    for col in ['use_count', 'accept_count', 'score']:
        if col not in bank.columns:
            bank[col] = 0

    # Sort: use_count DESC, accept_count DESC, score DESC, created_time DESC
    sort_cols = ['use_count', 'accept_count', 'score']
    sort_cols_exist = [c for c in sort_cols if c in bank.columns]
    if 'created_time' in bank.columns:
        sort_cols_exist.append('created_time')
    ascending = [False] * len(sort_cols_exist)

    if prefer_stage and stage is not None:
        # Stage-aware: prefer matching stage, then fallback to all
        stage_match = bank[bank['stage'].astype(str).str.lower() == str(stage).lower()]
        if len(stage_match) >= 1:
            ranked = stage_match.sort_values(by=sort_cols_exist, ascending=ascending)
        else:
            ranked = bank.sort_values(by=sort_cols_exist, ascending=ascending)
    else:
        ranked = bank.sort_values(by=sort_cols_exist, ascending=ascending)

    best = ranked.iloc[0]
    return str(best['model_path'])


def model_bank_summary(bank_dir: Path) -> dict:
    """Return a summary of the current model bank state.

    Pattern: ``ModelZoo.get_state()`` in D:\\project\\system.
    """
    csv_path = bank_dir / 'model_bank.csv'
    if not csv_path.exists():
        return {'model_bank_size': 0, 'model_bank_paths': []}
    bank = pd.read_csv(csv_path)
    paths = [str(p) for p in bank.get('model_path', [])]
    summary = {
        'model_bank_size': len(bank),
        'model_bank_paths': paths,
    }
    if 'use_count' in bank.columns:
        summary['total_uses'] = int(bank['use_count'].fillna(0).sum())
    if 'stage' in bank.columns:
        summary['stages'] = bank['stage'].dropna().unique().tolist()
    if 'score' in bank.columns:
        summary['mean_score'] = float(bank['score'].mean())
        summary['max_score'] = float(bank['score'].max())
    return summary


def extract_scene_fingerprint(selected_df: pd.DataFrame) -> dict:
    """Extract a scene fingerprint from uploaded frame statistics.

    Uses only data available in real deployment (YOLO inference outputs
    + behavior counts).  No labels needed.

    The fingerprint encodes:
    - Confidence distribution of the drift window (P_10, P_50, P_90)
    - Behavior composition (stand/lie/eat ratios)

    These are used by :func:`query_model_bank` to find models trained
    on similar scenes via cosine similarity.
    """
    if selected_df is None or selected_df.empty:
        return {}

    confs = selected_df['avg_conf'].values if 'avg_conf' in selected_df.columns else np.array([])

    stand = int(selected_df['stand_count'].sum()) if 'stand_count' in selected_df.columns else 0
    lie = int(selected_df['lie_count'].sum()) if 'lie_count' in selected_df.columns else 0
    eat = int(selected_df['eat_count'].sum()) if 'eat_count' in selected_df.columns else 0
    total = stand + lie + eat

    return {
        'P_10': float(np.percentile(confs, 10)) if len(confs) >= 10 else 0.0,
        'P_50': float(np.percentile(confs, 50)) if len(confs) >= 10 else 0.0,
        'P_90': float(np.percentile(confs, 90)) if len(confs) >= 10 else 0.0,
        'stand_ratio': float(stand / max(1, total)),
        'lie_ratio': float(lie / max(1, total)),
        'eat_ratio': float(eat / max(1, total)),
        'frame_count': len(selected_df),
    }


def _scene_period(stage: str) -> str:
    """Parse the scene period (day/night/eat) from a stage name.

    System-style ``scene_period`` / ``time_of_day`` prior: T04_night ->
    'night', T03_eat -> 'eat'.  Stages without a known suffix fall back
    to the full lowercased stage name (exact-match semantics).
    """
    m = re.search(r'_(day|night|eat)$', str(stage), flags=re.IGNORECASE)
    return m.group(1).lower() if m else str(stage).lower()


def query_model_bank(
    bank_dir: Path,
    scene_fingerprint: dict,
    current_model_path: str | None = None,
    similarity_threshold: float | None = 0.6,
    stage: str | None = None,
    stage_filter: bool = True,
    require_proven: bool = True,
    selector: str = 'fingerprint',
    gating_dir: Path | None = None,
    gating_images: list[Path] | None = None,
    device: Any = 'cpu',
    top_k: int = 1,
    mark_used: bool = True,
    exclude_paths: list[str] | None = None,
) -> tuple[Path | None, dict]:
    """Search historical models using scene fingerprints.

    The paper policy calls with selector='similarity', no similarity gate,
    no stage/proven filters and mark_used=False. Legacy policies below
    retain their threshold, quality-ranking and gating options.

    Spare-parts-warehouse semantics (system-style, ModelZoo pattern).
    The bank is consulted at *deployment* time (candidate rejection
    fallback), never to override the model before training.

    Filters applied in order, mirroring ``ModelZoo.select_model_for_edge``:
    1. ``state != 'rejected'``  -- rejected models are permanently excluded
       from the candidate pool (system's mark_model_feedback lifecycle);
    2. ``require_proven``       -- only models with ``accept_count > 0``
       (actually accepted by the edge before) are matchable spare parts
       (system's metadata fallback prefers accepted models);
    3. ``stage_filter``         -- same scene period only (day/night/eat
       parsed from stage names), system's cycle_stage/time_of_day prior.

    Ranking among the surviving rows (two policies).  Both are preceded by
    the B1a hard similarity gate: rows whose stored fingerprint scores
    ``similarity <= similarity_threshold`` are discarded, and with no
    eligible match the query returns ``(None, info)`` with
    ``query_policy='no_eligible_match'`` — the V5 ``fingerprint_fallback``
    (retrieve the top-quality row regardless of similarity) is removed, so
    a wrong-scene hit like V5's T12_night ← T07_eat (−6.6 mAP) is
    impossible by construction.

    * ``selector='gating'`` (system ModelZoo, with a trained snapshot):
      the learned ResNet18 selector weights rank the *eligible* candidates
      — the bank is maintained by a convolutional network instead of
      hand-crafted scene similarity.  Ties break on bank score, then
      accept_count, then similarity.
    * ``selector='fingerprint'`` (V6 behavior): scene fingerprint cosine
      similarity gates eligibility and the survivors are ranked by the bank
      quality score (conf_gain / accept_count / use_count) — so the
      scene-similar-but-weakest spare can no longer win (V4 T12 lesson;
      this is the quality term that the V4 README identified as missing).
    * ``selector='similarity'`` (V1-era restore, 08-04 behavior): cosine
      similarity decides, bank score only tiebreaks.  Pre-V6 there was no
      quality term — the scene-similar-but-weakest spare could win (the V1
      batch's T12_night <- T11_eat, score 0.086, is exactly that
      pathology).  Default off; only the V1-restore yaml opts in.

    Parameters
    ----------
    bank_dir : Path
        Model bank directory.
    scene_fingerprint : dict
        Current scene fingerprint from :func:`extract_scene_fingerprint`.
    current_model_path : str | None
        Path of the currently active model (avoid self-match).
    similarity_threshold : float | None
        Minimum cosine similarity for eligibility; None disables this gate
        for the paper's unrestricted highest-similarity retrieval.
    stage : str | None
        Current stage name for the scene-period prior.
    stage_filter : bool
        Restrict matches to models of the same scene period.
    require_proven : bool
        Restrict matches to models with accept_count > 0.
    selector : str
        'gating' or 'fingerprint' (default; V6 behavior).
    gating_dir : Path | None
        Directory holding the gating snapshot (model_bank/gating).
    gating_images : list[Path] | None
        Recent frames of the current stage for gating inference.
    device : Any
        Inference device for the gating network.
    top_k : int
        Number of ranked candidates exposed in ``query_model_candidates``
        (v9 gate validates the top-k on episode drift frames).
    mark_used : bool
        Whether to increment ``use_count`` on the best match.  The v9 gate
        passes ``False`` (validation probing must not pollute the LRU
        feedback loop); the executor calls ``mark_model_used`` once when a
        switch really happens.

    Returns
    -------
    (best_model, query_info) : tuple[Path | None, dict]
        Best matching model (or None) plus a query record with
        ``query_policy`` / ``query_similarity`` / ``query_score`` /
        ``query_gating_weight`` / ``query_model_path`` /
        ``query_model_candidates`` (ranked top-k list).
    """
    info: dict = {
        'query_policy': 'none',
        'query_similarity': 0.0,
        'query_score': 0.0,
        'query_gating_weight': 0.0,
        'query_model_path': '',
    }
    csv_path = bank_dir / 'model_bank.csv'
    if not csv_path.exists():
        return None, info

    bank = pd.read_csv(csv_path)
    if bank.empty:
        return None, info

    if not scene_fingerprint:
        return None, info

    # 1. Exclude rejected models (system's rejected state exclusion).
    if 'state' in bank.columns:
        state_s = bank['state'].astype(str).str.strip().str.lower()
        bank = bank[state_s != 'rejected']

    # 2. Only proven spare parts (accept_count > 0), system's feedback loop.
    if require_proven:
        if 'accept_count' in bank.columns:
            ac = pd.to_numeric(bank['accept_count'], errors='coerce').fillna(0)
            bank = bank[ac > 0]

    # 3. Scene-period prior: day matches day, night matches night, etc.
    if stage_filter and stage:
        period_now = _scene_period(stage)
        if 'scene_period' in bank.columns:
            bank = bank[bank['scene_period'].astype(str).str.strip().str.lower() == period_now]
        else:
            bank = bank[bank['stage'].apply(_scene_period) == period_now]

    if bank.empty:
        return None, info

    # Feature vector keys (must match extract_scene_fingerprint output)
    keys = ['P_10', 'P_50', 'P_90', 'stand_ratio', 'lie_ratio', 'eat_ratio']
    vec_now = np.array([float(scene_fingerprint.get(k, 0.0)) for k in keys], dtype=np.float64)
    vec_norm = float(np.linalg.norm(vec_now))
    if vec_norm == 0.0:
        return None, info

    excluded = {Path(p).resolve() for p in (exclude_paths or [])}
    if current_model_path:
        excluded.add(Path(current_model_path).resolve())
    candidates: list[dict] = []
    for _, row in bank.iterrows():
        model_path = str(row.get('model_path', ''))

        # Skip self-match
        if Path(model_path).resolve() in excluded:
            continue
        if not Path(model_path).is_file():
            continue

        # Cosine similarity to the stored scene fingerprint
        sim = 0.0
        try:
            stored = json.loads(str(row.get('scene_fingerprint', '{}')))
        except (json.JSONDecodeError, TypeError):
            stored = {}
        if stored:
            vec_stored = np.array([float(stored.get(k, 0.0)) for k in keys], dtype=np.float64)
            stored_norm = float(np.linalg.norm(vec_stored))
            if stored_norm > 0.0:
                sim = float(np.dot(vec_now, vec_stored) / (vec_norm * stored_norm))

        candidates.append({
            'model_path': model_path,
            'score': float(row.get('score', 0.0) or 0.0),
            'accept_count': int(row.get('accept_count', 0) or 0),
            'similarity': sim,
        })

    if not candidates:
        return None, info

    # Optional legacy eligibility gate; the paper policy passes None.
    # No eligible match => no bank action.  The pre-v8 fallback that
    # retrieved the top-quality row regardless of similarity is deleted.
    eligible = (candidates if similarity_threshold is None else
                [c for c in candidates if c['similarity'] > similarity_threshold])
    if not eligible:
        info['query_policy'] = 'no_eligible_match'
        return None, info

    # -- Ranking ---------------------------------------------------------
    use_gating = str(selector).strip().lower() == 'gating' and gating_dir is not None
    gating_weights: dict[str, float] | None = None
    if use_gating:
        # Recent-stage frames through the selector; multi-image averaging
        # matches system's mean(weights, dim=0) selection pattern.
        gating_weights, _ = gating_model_weights(
            gating_dir, gating_images or [], device,
            max_images=24,
        )

    if use_gating and gating_weights:
        # Learned selector over the ELIGIBLE candidates only: weight desc,
        # ties on score / accept / similarity.
        for c in eligible:
            c['gating_weight'] = float(gating_weights.get(c['model_path'], -1.0))
        ranked = sorted(eligible, key=lambda c: (
            c['gating_weight'], c['score'], c['accept_count'], c['similarity'],
        ), reverse=True)
        info['query_policy'] = 'gating'
    elif str(selector).strip().lower() == 'similarity':
        # V1-era (08-04) restore: cosine similarity decides, bank score only
        # tiebreaks — pre-V6 the quality term did not exist, so the
        # scene-similar-but-weakest spare could win.  Only the V1-restore
        # yaml sets this selector; default path unchanged.
        ranked = sorted(eligible, key=lambda c: (
            c['similarity'], c['score'], c['accept_count'],
        ), reverse=True)
        info['query_policy'] = 'similarity'
    else:
        # Fingerprint eligibility + quality ranking (V6 path with the P1
        # quality term added): similarity gates, bank score decides.
        ranked = sorted(eligible, key=lambda c: (
            c['score'], c['accept_count'], c['similarity'],
        ), reverse=True)
        info['query_policy'] = 'fingerprint'

    best = ranked[0]
    best_model = Path(best['model_path'])
    info['query_model_candidates'] = [dict(c) for c in ranked[:max(1, int(top_k))]]
    if mark_used:
        mark_model_used(bank_dir, str(best_model))
    info.update({
        'query_similarity': float(best['similarity']),
        'query_score': float(best['score']),
        'query_gating_weight': float(best.get('gating_weight', 0.0)),
        'query_model_path': str(best_model),
    })
    return best_model, info


def _update_gating_pool_and_refresh(
    model_bank_dir: Path,
    stage: str,
    exported: list[Path],
    stage_pseudo: Path,
    mb_cfg: dict,
    device: Any,
    imgsz: int,
    conf_thr: float = 0.40,
    seed: int = 42,
) -> dict:
    """Accumulate the gating training pool and retrain the selector head.

    System's ModelZoo maintenance pattern: every bank refresh (admission /
    eviction) is followed by a selector refresh.  Pool layout is
    ``model_bank/gating_pool/{images,labels}`` — frames uploaded this stage
    plus their teacher pseudo labels (same stems), FIFO-capped via
    ``gating.pool_max_images``.  Experts = active bank rows in CSV order;
    the head is retrained with teacher-supervised expert matching (see
    :func:`refresh_gating_network` — the training loop the system repo
    designed but no longer contains).
    """
    gc_cfg = mb_cfg.get('gating', {}) or {}
    if not bool(gc_cfg.get('enabled', True)):
        return {'gating_refreshed': False, 'gating_reason': 'disabled'}
    pool_dir = model_bank_dir / 'gating_pool'
    pool_images_dir = pool_dir / 'images'
    pool_labels_dir = pool_dir / 'labels'
    pool_images_dir.mkdir(parents=True, exist_ok=True)
    pool_labels_dir.mkdir(parents=True, exist_ok=True)

    # -- add this stage's frames to the pool (unique stems per stage) ----
    sample_n = int(gc_cfg.get('sample_per_stage', 60))
    safe_stage = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in stage)
    chosen = list(exported)
    if sample_n > 0 and len(chosen) > sample_n:
        random.Random(seed).shuffle(chosen)
        chosen = chosen[:sample_n]
    for img in chosen:
        lbl = stage_pseudo / f'{img.stem}.txt'
        if not lbl.exists():
            continue
        dst_img = pool_images_dir / f'{safe_stage}_{img.name}'
        if dst_img.exists():
            continue  # this stage's frames were pooled before
        shutil.copy2(img, dst_img)
        shutil.copy2(lbl, pool_labels_dir / f'{dst_img.stem}.txt')

    # -- FIFO cap -------------------------------------------------------
    pool_max = int(gc_cfg.get('pool_max_images', 400))
    pool_files = sorted(
        list(pool_images_dir.glob('*.jpg')) + list(pool_images_dir.glob('*.png')),
        key=lambda p: p.stat().st_mtime,
    )
    if len(pool_files) > pool_max:
        for stale in pool_files[:len(pool_files) - pool_max]:
            stale.unlink()
            lab = pool_labels_dir / f'{stale.stem}.txt'
            if lab.exists():
                lab.unlink()

    # -- experts (active bank rows in CSV order) ------------------------
    csv_path = model_bank_dir / 'model_bank.csv'
    expert_paths: list[str] = []
    if csv_path.exists():
        bank = pd.read_csv(csv_path)
        if 'state' in bank.columns:
            state_s = bank['state'].astype(str).str.strip().str.lower()
            bank = bank[state_s != 'rejected']
        expert_paths = [str(p) for p in bank.get('model_path', [])]

    pool_images = [p for p in (
        list(pool_images_dir.glob('*.jpg')) + list(pool_images_dir.glob('*.png'))
    ) if (pool_labels_dir / f'{p.stem}.txt').exists()]
    teacher_labels = {p: pool_labels_dir / f'{p.stem}.txt' for p in pool_images}

    info = refresh_gating_network(
        model_bank_dir / 'gating',
        pool_images,
        teacher_labels,
        expert_paths,
        device,
        imgsz,
        conf_thr=conf_thr,
        head_epochs=int(gc_cfg.get('head_epochs', 20)),
        batch_size=int(gc_cfg.get('batch_size', 32)),
        lr=float(gc_cfg.get('lr', 1e-3)),
        freeze_backbone=bool(gc_cfg.get('freeze_backbone', True)),
        predict_batch=int(gc_cfg.get('predict_batch', 16)),
    )
    return info


def _method_variant(method: str, mix_mode: str | None) -> str:
    return method if method == 'source_only' else f'{method}__{mix_mode}'


def _paper_bank_fallback(
    bank_dir: Path,
    scene_fingerprint: dict,
    current_model: Path,
    deployed_baseline: Path,
    validation_images: list[Path],
    candidate_accepted: bool,
    initialization_paths: list[str],
    imgsz: int,
    device: Any,
    enabled: bool = True,
) -> tuple[Path | None, dict]:
    """Only a rejected candidate can trigger the paper's historical fallback.

    Retrieval does not refresh LRU recency. Only actual deployment counts as
    use. Validate against the immutable deployed model, never the training
    initialization copy, and do not insert the fallback as a new bank entry.
    """
    info = {
        'model_bank_policy': 'paper',
        'model_bank_fallback_validated': False,
        'model_bank_fallback_excluded': list(initialization_paths),
        'model_bank_competition': 'candidate' if candidate_accepted else 'current',
    }
    if candidate_accepted or not enabled or not validation_images:
        return None, info
    spare, query = query_model_bank(
        bank_dir, scene_fingerprint, str(current_model),
        similarity_threshold=None, stage_filter=False, require_proven=False,
        selector='similarity', mark_used=False, exclude_paths=initialization_paths,
        device=device,
    )
    # Keep initialization and fallback retrieval records separate in decisions.
    info.update({'model_bank_fallback_' + key: value for key, value in query.items()})
    if spare is None:
        return None, info
    _, validation = shadow_validate(deployed_baseline, spare, validation_images,
                                    imgsz, device, min_gain=0.0)
    gain = float(validation['conf_gain'])
    info['model_bank_fallback_gain'] = gain
    if gain <= 0.0 or not np.isfinite(gain):
        return None, info
    shutil.copy2(spare, current_model)
    mark_model_used(bank_dir, str(spare))
    info.update({'model_bank_fallback_validated': True,
                 'model_bank_competition': 'spare'})
    return spare, info


def _run_update(
    *,
    exp: Experiment,
    raw: dict,
    method: str,
    method_variant: str,
    mix_mode: str,
    stage: str,
    stage_index: int,
    episode_tag: str,
    current_model: Path,
    out_dir: Path,
    train_images: list[Path],
    scene_fp: dict | None = None,
    source_sample_count: int | None = None,
    deploy_mode: str = 'full',
    edge_candidate_frames: int = 0,
    edge_pass_frames: int = 0,
    edge_score_seconds: float = 0.0,
    selection_seconds: float = 0.0,
    run_deploy_competition: bool = False,
    run_gating_refresh: bool = False,
    bank_query_info: dict | None = None,
    trigger_info: dict | None = None,
    decision_extra: dict | None = None,
    names: dict | list | None = None,
    imgsz: int = 640,
    device: Any = 'auto',
    dedup_iou_threshold: float = 0.0,
    deployed_baseline: Path | None = None,
) -> dict[str, Any]:
    """执行一次边端模型更新：教师伪标注 → 微调 → 影子验证 → 部署 → 入库。

    v8 阶段训练的整块逻辑（原 runner.py 训练闸门内 1697-2051）行为保持式
    抽出。v9 每个漂移片段调用一次（``episode_tag=f'e{n}'`` / 'stage_flush'）；
    v8 旧路径每阶段调用一次（``episode_tag='stage'``，目录/文件命名与 v8
    原字符串完全一致，保持 v8 批次可复现）。B2 部署竞争与 gating 刷新按
    调用方开关（v9 闸门模式关闭，被 VUG 取代）。

    ``deployed_baseline``（方案 C，2026-09-19）：本阶段发生了 B1b 训前换起
    点时，调用方把**换起点之前的边缘端模型快照**传进来。此时 ``current_model``
    已被换成历史模型（训练起点），但它不再是"边缘端在跑的模型"。三处受影响：
      1. 影子验证的裁判基线 → 改用快照（否则拿历史模型当裁判，V11/V12 的
         T09 误拒即由此而来）；
      2. 部署竞争的 spare 基线 → 同上；
      3. 候选未部署时（被拒 / 无 spare 胜出）→ 把 ``current_model`` 恢复成
         快照，保证"换起点只作用于训练起点，边缘端部署态不被静默改写"。
    None（默认）= 本阶段无换起点，行为与历史批次逐位一致。
    """
    # -- v8 命名兼容：episode_tag=='stage' 时目录/文件名与 v8 完全相同 ----
    if episode_tag == 'stage':
        stage_pseudo = out_dir / 'pseudo_labels' / stage
        mixed_stage_dir = out_dir / 'mixed_datasets' / stage
        run_name = f'update_{stage}'
        cand_name = f'candidate_after_{stage}.pt'
        decision_file = f'{stage}_decision.json'
    else:
        stage_pseudo = out_dir / 'pseudo_labels' / f'{stage}_{episode_tag}'
        mixed_stage_dir = out_dir / 'mixed_datasets' / f'{stage}_{episode_tag}'
        run_name = f'update_{stage}_{episode_tag}'
        cand_name = f'candidate_after_{stage}_{episode_tag}.pt'
        decision_file = f'{stage}_{episode_tag}_decision.json'

    candidate_models_dir = out_dir / 'candidate_models'
    model_bank_dir = out_dir / 'model_bank'
    reports_dir = out_dir / 'reports'
    mb_cfg = raw.get('model_bank', {})
    cand_path = candidate_models_dir / cand_name

    uploaded = len(train_images)
    upload_bytes = total_file_bytes(train_images)
    accepted = False
    direct_hit_model: Path | None = None  # direct-deploy arm: bank model deployed unvalidated
    accept_info: dict[str, Any] = {
        'current_metric_gain': 0.0, 'history_metric_drop': 0.0, 'behavior_acc_gain': 0.0,
        'main_metric': raw.get('update_policy', {}).get('main_metric', 'mAP50'),
    }
    train_seconds = 0.0
    pseudo_label_seconds = 0.0
    pseudo_info = {
        'pseudo_behavior_boxes': 0,
        'pseudo_empty_images': 0,
        'pseudo_avg_conf': 0.0,
    }
    deploy_size_info: dict[str, float] = {
        'model_full_bytes': 0.0, 'model_full_mb': 0.0,
        'model_head_bytes': 0.0, 'model_head_mb': 0.0, 'model_savings_pct': 0.0,
    }
    train_source_images = 0
    train_target_images = 0
    replay_info: dict[str, Any] = {
        'source_replay_scene': '',
        'selective_replay_enabled': False,
    }
    bank_info: dict[str, Any] = {'model_bank_enabled': bool(mb_cfg.get('enabled', True))}
    if trigger_info is None:
        trigger_info = {'triggered': True, 'c_mean': 0.0, 'c_history_mean': 0.0,
                        'a_tilde_mean': 0.0, 'trigger_skipped': False}

    # -- Teacher pseudo-labeling --------------------------------------
    t0 = time.time()
    pseudo_info = pseudo_label_with_teacher(
        exp.paths.teacher,
        train_images,
        stage_pseudo,
        imgsz=imgsz,
        conf_thr=float(raw['models'].get('pseudo_conf', 0.40)),
        min_box_area_ratio=float(raw['models'].get('min_box_area_ratio', 0.00015)),
        nc=len(names),
        dedup_iou_threshold=dedup_iou_threshold,
    )
    pseudo_label_seconds = time.time() - t0
    # -- Free teacher model before training --------------
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Split uploaded frames -> train + shadow-val --------
    # Validation split comes from the uploaded frames themselves
    # (not a separate held-out video).  This tests whether the
    # candidate model improves on the *targeted* distribution
    # (the drift window), which is the correct acceptance
    # criterion for continual domain adaptation.
    val_frac = float(raw['data_mix'].get('target_val_fraction', 0.10))
    min_val = int(raw['data_mix'].get('min_target_val_images', 5))
    # 切分逻辑抽到 _split_target_train_val：B1b 换起点的目标域判据用同一个
    # 函数取"训练部分"当判据帧，保证与这里的 holdout 互斥且逐帧一致。
    exported_train, exported_val = _split_target_train_val(
        train_images, val_frac, min_val, int(raw.get('seed', 42)), stage_index)

    # -- SDR: source-domain replay (scene-adaptive + capped) ----
    # R1 scene scale: the source footage is all daytime, so
    # night stages replay only scene_source_scale[night] of the
    # budget (0.4 -> 20 of 50) — the night stages were the only
    # consistently negative replay stages in V3/V5.
    # R2 fraction cap: source frames never exceed
    # source_fraction_cap of the mixed training batch (target
    # frames after target_repeat) — the 50-frame fallback stages
    # (T04/T10) drop from ~29% to <=15% source.
    # R3 selective replay (optional, default OFF): score source
    # images with the CURRENT student and replay only the k
    # lowest-confidence (highest forgetting-risk) images.
    scene_period = _scene_period(stage)
    replay_info = {
        'source_replay_scene': scene_period,
        'selective_replay_enabled': False,
    }
    base_source_count = int(raw['data_mix'].get('source_sample_count', 1200))
    scene_scale = float(raw['data_mix'].get('scene_source_scale', {}).get(scene_period, 1.0))
    if source_sample_count is None:
        source_sample_count = int(round(base_source_count * scene_scale))
    replay_info['source_replay_after_scale'] = source_sample_count
    if source_sample_count != base_source_count:
        print(f'[data_mix] R1 scene={scene_period} source replay scaled '
              f'{base_source_count} -> {source_sample_count}')

    cap_frac = float(raw['data_mix'].get('source_fraction_cap', 0.0))
    target_repeat_n = int(raw['data_mix'].get('target_repeat', 1))
    capped_count = _source_cap(source_sample_count, len(exported_train), target_repeat_n, cap_frac)
    replay_info['source_cap_applied'] = bool(capped_count < source_sample_count)
    replay_info['source_replay_after_cap'] = capped_count
    if capped_count != source_sample_count:
        print(f'[data_mix] R2 fraction cap {cap_frac}: source replay '
              f'{source_sample_count} -> {capped_count} (target batch '
              f'~{len(exported_train) * target_repeat_n} frames)')
    source_sample_count = capped_count

    source_images: list[Path] | None = None
    sel_cfg = raw['data_mix'].get('selective_source_replay', {}) or {}
    if mix_mode == 'source_plus_hard' and source_sample_count > 0 and bool(sel_cfg.get('enabled', False)):
        k_scene = int(sel_cfg.get(f'{scene_period}_k', sel_cfg.get('night_k', 20)))
        final_n = min(source_sample_count, k_scene)
        pool = _source_images(exp, cap=int(sel_cfg.get('score_pool', 200)))
        scored_pool = _image_confidence_map(current_model, pool, imgsz, device)
        source_images = _select_forgetting_prone(scored_pool, final_n)
        replay_info.update({
            'selective_replay_enabled': True,
            'selective_replay_k': k_scene,
            'selective_replay_final_n': len(source_images),
            'selective_replay_scored': len(scored_pool),
        })
        print(f'[data_mix] R3 selective replay: replayed '
              f'{len(source_images)}/{len(pool)} lowest-confidence source images')

    data_yaml = build_mixed_dataset(
        mixed_stage_dir,
        exp.paths.source_yaml,
        exported_train,         # <- train split only
        stage_pseudo,
        source_sample_count=source_sample_count,
        target_repeat=int(raw['data_mix'].get('target_repeat', 1)),
        source_images=source_images,   # R3 preselected subset (None = random)
        seed=int(raw.get('seed', 42)) + stage_index,
        mix_mode=mix_mode,
        names=names,
        target_val_fraction=val_frac,
        min_target_val_images=min_val,
    )
    build_info = (mixed_stage_dir / 'build_info.txt').read_text(encoding='utf-8') if (mixed_stage_dir / 'build_info.txt').exists() else ''
    for line in build_info.splitlines():
        if line.startswith('train_source_images='):
            train_source_images = int(line.split('=', 1)[1])
        elif line.startswith('target_train_images_after_repeat='):
            train_target_images = int(line.split('=', 1)[1])

    t0 = time.time()
    candidate = train_student(
        current_model,
        data_yaml,
        out_project=out_dir / 'train_runs',
        run_name=run_name,
        imgsz=imgsz,
        epochs=int(raw['models'].get('student_epochs', 10)),
        batch=int(raw['models'].get('batch', 16)),
        workers=int(raw['models'].get('workers', 0)),
        device=device,
        **({'train_options': raw['models']['train_options']} if 'train_options' in raw['models'] else {}),
    )
    train_seconds = time.time() - t0
    shutil.copy2(candidate, cand_path)

    # -- 方案 A：训练后权重插值锚定 ------------------------------
    # 必须在影子验证**之前**执行：插值候选要用同一套 holdout 口径选优，
    # 选出的胜者覆盖 cand_path 后再走正常的验证/竞争/部署流程。
    post_anchor_info: dict[str, Any] = {}
    pa_cfg = raw.get('post_anchor', {}) or {}
    if bool(pa_cfg.get('enabled', False)):
        _anchor_kind = str(pa_cfg.get('anchor', 'source')).strip().lower()
        if _anchor_kind == 'prev':
            _anchor_path = (deployed_baseline if deployed_baseline is not None
                            else current_model)
        else:
            _anchor_path = exp.paths.initial_student
        post_anchor_info = apply_post_anchor(
            pa_cfg, anchor_path=_anchor_path, cand_path=cand_path,
            val_paths=exported_val, imgsz=imgsz, device=device,
            out_dir=candidate_models_dir, stage=stage)

    # -- Free GPU memory after training (heaviest op) ---
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # -- Model size for deployment comparison ------------
    deploy_size_info = _compute_model_sizes(cand_path, deploy_mode)

    # -- Direct-deployment arm (reviewer E4-1 ablation) ------------
    # model_bank.direct_deploy=true skips shadow validation entirely: the
    # bank's historical match is deployed UNVALIDATED, or — when the bank
    # holds no match — the freshly trained candidate is deployed
    # unvalidated.  Training still runs (the bank acquires a candidate)
    # and the query_on_train_start + b1b_legacy init path is untouched;
    # only the stage-END deployment decision is replaced.  This is the
    # "unvalidated historical states being deployed" row of the
    # validation-separation table.
    direct_deploy_arm = bool(mb_cfg.get('direct_deploy', False))
    if direct_deploy_arm:
        hit_path = ''
        if bank_query_info and bank_query_info.get('model_bank_query_hit'):
            hit_path = str(bank_query_info.get('model_bank_query_path', ''))
        if hit_path and Path(hit_path).exists():
            direct_hit_model = Path(hit_path)
            shutil.copy2(direct_hit_model, current_model)
            print(f'[direct_deploy] historical model {direct_hit_model.name} '
                  f'deployed UNVALIDATED (shadow validation skipped)')
        else:
            print('[direct_deploy] no bank match -> trained candidate deployed '
                  'UNVALIDATED (shadow validation skipped)')
        accepted = True
        accept_info = {
            'old_mean_conf': 0.0, 'new_mean_conf': 0.0, 'conf_gain': 0.0,
            'eval_frame_count': 0, 'decision': 'direct_deploy',
        }

    # -- Shadow deployment validation on val split ---------
    # Validate on the held-out portion of uploaded frames.
    # This tests whether the candidate model improved on the
    # *targeted* distribution (frames from the drift window),
    # which is the correct acceptance criterion for continual
    # domain adaptation.  No held-out video needed.
    #
    # update_policy.shadow_requires_bank（2026-09-20，默认 false）：开启后，
    # 部署前验证只在**模型库启用**时执行——无库臂（消融 ✓××）跳过验证、直接
    # 部署候选。语义上把"影子验证"并入「History Validation」组件（论文
    # \ref{sec:deployment} 一节把库与验证放在同一组件里），使消融的三列
    # （BSS / History Validation / Source Replay）各自对应一组机制。
    # 默认 false = 历史批次（V10–V17）行为逐位不变。
    _shadow_requires_bank = bool(
        raw.get('update_policy', {}).get('shadow_requires_bank', False))
    _bank_on = bool(bank_info.get('model_bank_enabled', False))
    _skip_shadow = _shadow_requires_bank and not _bank_on
    if not direct_deploy_arm and exported_val and not _skip_shadow:
        print(f'[shadow] validating on {len(exported_val)} held-out uploaded frames ...')
        # 方案 C：本阶段换过训练起点时，裁判基线用"真正部署中的模型"快照
        _val_baseline = deployed_baseline if deployed_baseline is not None else current_model
        accepted, accept_info = shadow_validate(
            _val_baseline, cand_path, exported_val, imgsz, device,
            min_gain=float(raw.get('update_policy', {}).get('shadow_min_gain', 0.0)),
        )
        print(f'[shadow] validation: old_conf={accept_info["old_mean_conf"]:.4f}, '
              f'new_conf={accept_info["new_mean_conf"]:.4f}, '
              f'gain={accept_info["conf_gain"]:+.4f}, '
              f'decision={accept_info["decision"]}')
    elif not direct_deploy_arm and uploaded > 0:
        # Very few frames uploaded -> accept by default；无库臂在
        # shadow_requires_bank 下同样走到这里（跳过验证，直接部署候选）。
        accept_info = {
            'old_mean_conf': 0.0, 'new_mean_conf': 0.0, 'conf_gain': 0.0,
            'eval_frame_count': 0,
            'decision': 'accept_no_validation' if _skip_shadow else 'accept_by_default',
        }
        accepted = True
        if _skip_shadow:
            print('[shadow] skipped: shadow_requires_bank=true and model bank disabled '
                  '-> candidate deployed unvalidated')

    # 方案 A 诊断信息（在影子验证之后合并：验证块会整体重绑 accept_info）
    if post_anchor_info:
        accept_info.update(post_anchor_info)

    # -- Historical deployment: paper fallback / legacy P3 competition --
    # The legacy P3 policy performs a two-stage decision at EVERY deployment, not
    # just on rejection: the bank selector ranks same-scene
    # spare parts and the best spare is shadow-validated on the
    # unseen val frames against BOTH the candidate and the
    # current model.  The spare is deployed only when it wins
    # strictly (spare_gain > deploy_spare_margin AND >
    # candidate gain), so every bank action is positive by
    # construction (P0 gate).  Stages where no spare wins
    # contribute 0 to the bank effect — never negative.
    # V5 lesson: 12/12 candidates passed, the rejection-only
    # bank never fired once -> effect exactly 0.0.
    # With deploy_competition: false this reduces to the V7
    # rejection-only fallback.
    # v9: disabled in gate mode (run_deploy_competition=False) —
    # the VUG bank validation replaces it.
    bank_action: dict[str, Any] = {}
    bank_fallback_model: Path | None = None
    cand_was_accepted = bool(accepted)
    paper_bank = str(mb_cfg.get('policy', 'legacy')).strip().lower() == 'paper'
    p3_compete = (not paper_bank and bool(mb_cfg.get('deploy_competition', False))
                  and run_deploy_competition)
    if paper_bank and not direct_deploy_arm:
        init_paths = list((bank_query_info or {}).get('model_bank_initialization_paths', []))
        if not init_paths and (bank_query_info or {}).get('model_bank_query_hit'):
            init_path = (bank_query_info or {}).get('model_bank_query_path')
            if init_path:
                init_paths.append(str(init_path))
        bank_fallback_model, bank_action = _paper_bank_fallback(
            model_bank_dir, scene_fp or {}, current_model,
            deployed_baseline if deployed_baseline is not None else current_model,
            exported_val, bool(accepted), init_paths, imgsz, device,
            enabled=(bool(bank_info.get('model_bank_enabled', False))
                     and bool(mb_cfg.get('rejection_fallback', True))),
        )
        if bank_fallback_model is not None:
            accepted = True
            accept_info['decision'] = 'accept_fallback'
        elif not accepted:
            accept_info['decision'] = 'reject_keep_current'
    # rejection_fallback (default true) is the 3-arm bank-ablation switch
    # (2026-08-28): false = bank serves ONLY as the pre-training init
    # (B1b query), the post-rejection historical-model fallback is off.
    if (not paper_bank and bank_info.get('model_bank_enabled', False)
            and bool(mb_cfg.get('rejection_fallback', True))
            and not direct_deploy_arm
            and (not accepted or p3_compete)):
        sel = str(mb_cfg.get('selector', 'fingerprint')).strip().lower()
        spare, query_info = query_model_bank(
            model_bank_dir, scene_fp or {}, str(current_model),
            stage=stage,
            similarity_threshold=float(mb_cfg.get('similarity_threshold', 0.6)),
            stage_filter=bool(mb_cfg.get('stage_filter', True)),
            require_proven=bool(mb_cfg.get('require_proven', True)),
            selector=sel,
            gating_dir=(model_bank_dir / 'gating') if sel == 'gating' else None,
            gating_images=list(train_images) if sel == 'gating' else None,
            device=device,
        )
        bank_action.update({
            'model_bank_query_policy': query_info.get('query_policy', 'none'),
            'model_bank_query_similarity': query_info.get('query_similarity', 0.0),
            'model_bank_query_score': query_info.get('query_score', 0.0),
            'model_bank_query_gating_weight': query_info.get('query_gating_weight', 0.0),
            'model_bank_fallback_validated': False,
            'model_bank_competition': 'none',
        })
        if spare is not None and exported_val:
            spare_ok, spare_info = shadow_validate(
                _val_baseline if deployed_baseline is not None else current_model,
                spare, exported_val, imgsz, device,
                min_gain=0.0,
            )
            spare_gain = float(spare_info['conf_gain'])
            cand_gain = float(accept_info.get('conf_gain', 0.0))
            margin = float(mb_cfg.get('deploy_spare_margin', 0.02))
            bank_action['model_bank_fallback_gain'] = spare_gain
            if spare_gain > margin and spare_gain > cand_gain:
                print(f'[model_bank] spare {spare.name} wins the deployment '
                      f'competition (spare_gain {spare_gain:+.4f} vs cand_gain '
                      f'{cand_gain:+.4f}, margin {margin:.3f}) -> deploying')
                bank_fallback_model = spare
                shutil.copy2(spare, current_model)
                accepted = True
                accept_info['decision'] = ('accept_fallback' if not cand_was_accepted
                                           else 'accept_spare')
                bank_action['model_bank_fallback_validated'] = True
                bank_action['model_bank_competition'] = 'spare'
            elif not accepted:
                print(f'[model_bank] spare {spare.name} does NOT win the competition '
                      f'(spare_gain {spare_gain:+.4f}) -> keep current model')
                accept_info['decision'] = 'reject_keep_current'
                bank_action['model_bank_competition'] = 'current'
            else:
                print(f'[model_bank] spare {spare.name} loses to candidate '
                      f'(spare_gain {spare_gain:+.4f} vs cand_gain {cand_gain:+.4f}) '
                      f'-> deploy candidate')
                bank_action['model_bank_competition'] = 'candidate'
        elif spare is not None:
            # No unseen frames to validate the spare on — keep the
            # current model (a blind swap is what damaged V4 T12).
            if not accepted:
                print('[model_bank] no val frames to validate spare -> keep current model')
                accept_info['decision'] = 'reject_keep_current'
        elif not accepted and p3_compete:
            # Rejected, and the bank holds no matchable spare for
            # this scene -> the two-way competition verdict stands.
            accept_info['decision'] = 'reject_keep_current'
            bank_action['model_bank_competition'] = 'current'

    # -- Model bank: register the candidate -------------------
    # Spare-parts-warehouse admission (system-style): only a
    # candidate that was actually accepted AND deployed is
    # admitted as an 'active' spare part with accept_count=1 —
    # this wires the feedback loop that was previously dead
    # (accept_count was always 0).  Rejected candidates are
    # not spare parts (not stored unless store_rejected).
    # A fallback deployment leaves the candidate rejected:
    # the deployed spare part instead gets its usage metadata updated.
    # Direct-deploy arm: a candidate that lost deployment to the bank hit
    # is admitted as 'rejected' (mirrors the bank-competition fallback
    # semantics); the deployed historical path is already recorded in
    # model_bank_query_path (merged below).
    candidate_state = ('active'
                       if (accepted and not bank_fallback_model and not direct_hit_model)
                       else 'rejected')
    bank_info = update_model_bank(
        cand_path, model_bank_dir, stage, method_variant,
        accept_info['conf_gain'], scene_fp or {}, raw,
        accept_count=(1 if candidate_state == 'active' else 0),
        use_count=0,
        state=candidate_state,
    )
    bank_info['model_bank_fallback'] = str(bank_fallback_model) if bank_fallback_model else ''
    bank_info.update(bank_query_info or {})
    # P3 competition verdict (logging gap fix): without this the
    # query_policy/competition/fallback columns never reached the
    # decision JSON or stage_results.csv (the V7 bank analysis
    # had no per-stage action data to read).
    bank_info.update(bank_action)

    # -- Gating selector maintenance (system ModelZoo pattern) --
    # Every bank change (admission / eviction above) is followed
    # by a selector refresh: this stage's frames + teacher pseudo
    # labels join the gating pool (FIFO capped), then the ResNet18
    # selector head is retrained with teacher-supervised expert
    # matching — the training loop system designed but deleted.
    # v9: disabled in gate mode (run_gating_refresh=False).
    if (run_gating_refresh and bank_info.get('model_bank_enabled')
            and str(mb_cfg.get('selector', 'fingerprint')).strip().lower() == 'gating' and train_images):
        bank_info.update(_update_gating_pool_and_refresh(
            model_bank_dir, stage, train_images, stage_pseudo, mb_cfg,
            device, imgsz,
            conf_thr=float(raw['models'].get('pseudo_conf', 0.40)),
            seed=int(raw.get('seed', 42)) + stage_index,
        ))

    # -- Decision record (evaluation on current model happens at stage end) --
    decision = {
        'stage': stage,
        'episode_tag': episode_tag,
        'method': method,
        'method_variant': method_variant,
        'train_data_mode': mix_mode,
        'candidate': str(cand_path),
        'old_model': str(current_model),
        'accepted': accepted,
        'uploaded': uploaded,
        'upload_bytes': upload_bytes,
        'upload_mb': float(upload_bytes / (1024 * 1024)),
        'edge_candidate_frames': edge_candidate_frames,
        'edge_pass_frames': edge_pass_frames,
        'edge_score_seconds': edge_score_seconds,
        'selection_seconds': selection_seconds,
        'pseudo_label_seconds': pseudo_label_seconds,
        'train_source_images': train_source_images,
        'train_target_images': train_target_images,
        **pseudo_info,
        **accept_info,
        **replay_info,
        **bank_info,
        **trigger_info,
        **deploy_size_info,
        **(decision_extra or {}),
    }
    (reports_dir / decision_file).write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
    if accepted and not bank_fallback_model and not direct_hit_model:
        if deploy_mode == 'delta_sparse':
            # Delta sparsification: only top-5% largest weight
            # changes are transmitted.  Unlike partial deploy
            # (head-only), this updates the full model including
            # backbone — just sparsely.
            delta_info = _deploy_delta_sparse(
                current_model, cand_path, current_model, top_k=0.05,
            )
            deploy_size_info.update(delta_info)
        elif deploy_mode == 'partial':
            # Simulate head-only deployment: edge keeps the
            # initial backbone and only receives the new head.
            # This genuinely tests whether partial deployment
            # (saving ~51% communication on the downlink)
            # maintains accuracy.
            _deploy_partial_model(exp.paths.initial_student, cand_path, current_model)
        else:
            if bool(mb_cfg.get('gating_ensemble', False)):
                # B3 (optional): weight-space average of the
                # candidate and the gating-weighted top-k spares.
                ensemble_info = _gating_ensemble_deploy(
                    cand_path, current_model,
                    gating_dir=model_bank_dir / 'gating',
                    gating_images=list(train_images),
                    device=device,
                    alpha=float(mb_cfg.get('gating_ensemble_alpha', 0.7)),
                    top_k=int(mb_cfg.get('gating_ensemble_top_k', 3)),
                    bank_csv=model_bank_dir / 'model_bank.csv',
                    scene_period=_scene_period(stage),
                )
                if ensemble_info.get('gating_ensemble_applied'):
                    print(f'[model_bank] B3 gating ensemble deployed: '
                          f'{ensemble_info["gating_ensemble_experts"]} experts, '
                          f'alpha={ensemble_info["gating_ensemble_alpha"]:.2f}')
                else:
                    print(f'[model_bank] B3 ensemble not applied '
                          f'({ensemble_info.get("gating_ensemble_reason")}) -> plain candidate')
                    shutil.copy2(cand_path, current_model)
                bank_info.update(ensemble_info)
                decision.update(ensemble_info)
                (reports_dir / decision_file).write_text(
                    json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
            else:
                shutil.copy2(cand_path, current_model)
    # -- 方案 C：换起点只作用于训练起点，边缘端部署态不被静默改写 ----------
    # 本阶段若发生过 B1b 换起点（deployed_baseline 非空），只有"什么都没部署
    # 上去"时才需要恢复快照。判据只看 accepted：候选部署(accept)、spare 兜底
    # (accept_fallback/accept_spare)、direct 命中三条路都会把 accepted 置 True
    # 并已写好 current_model；唯独候选被拒时 current_model 仍停在被换入的历史
    # 模型上，必须恢复成部署态。
    # （踩坑记录 2026-09-20：先前写成 `not (accepted and not bank_fallback_model
    #   and not direct_hit_model)`，spare 胜出时会把刚部署的 spare 又覆盖回快照，
    #   导致该阶段等于没更新——V17 探针 T03 因此比 V16 掉 22.6pp，seen 矩阵里
    #   T03 行的 T01/T02 列与 T02 行逐位相同可证。）
    if deployed_baseline is not None and not accepted:
        shutil.copy2(deployed_baseline, current_model)
        print(f'[b1b] swap affected training start only -> edge keeps deployed '
              f'model {deployed_baseline.name} (candidate rejected)')
        decision['model_bank_swap_deployed_fallback'] = True
        (reports_dir / decision_file).write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
    # Cleanup: remove candidate (not current or fallback model)
    if bool(raw.get('model_bank', {}).get('cleanup_candidate_models', True)) and not bool(raw.get('update_policy', {}).get('keep_rejected_models', False)):
        if cand_path.exists():
            cand_path.unlink()

    return {
        'accepted': bool(accepted),
        'train_seconds': train_seconds,
        'pseudo_label_seconds': pseudo_label_seconds,
        'pseudo_info': pseudo_info,
        'accept_info': accept_info,
        'replay_info': replay_info,
        'bank_info': bank_info,
        'deploy_size_info': deploy_size_info,
        'train_source_images': train_source_images,
        'train_target_images': train_target_images,
        'upload_count': uploaded,
        'upload_bytes': upload_bytes,
        'bank_fallback_model': bank_fallback_model,
        'decision': decision,
    }


def _run_v9_stage(
    *,
    exp: Experiment,
    raw: dict,
    method: str,
    method_variant: str,
    mix_mode: str,
    stage: str,
    stage_index: int,
    current_model: Path,
    out_dir: Path,
    scores,
    streaming_selected,
    gate_cfg: dict,
    deploy_mode: str,
    edge_candidate_frames: int,
    edge_pass_frames: int,
    edge_score_seconds: float,
    selection_seconds: float,
    names: dict | list,
    imgsz: int,
    device: Any,
    dedup_iou_threshold: float,
) -> dict[str, Any]:
    """VUG 逐漂移片段决策流（一个阶段内）。

    每个漂移片段是一个决策点：``decide_gate`` 在「更新模型」与「直接用
    模型库历史模型」之间选择（可解释分层闸门，见 gate.py）。更新走
    :func:`_run_update`（上传+伪标注+微调+影子验证+部署+入库）；跳过走
    库切换（本地换模型，通信=0、训练=0）或边端缓冲（样本随下次更新上传）。
    阶段尾缓冲非空 -> 强制更新冲刷（样本不跨场景边界）。

    返回阶段级累计状态 dict（uploaded/upload_bytes/accepted/
    accepted_count/train_seconds/pseudo_* /replay_info/bank_info/
    deploy_size_info/train_source_images/train_target_images/
    episode_count/update_count/bank_switch_count/forced_update_count/
    buffered_frames_total/episode_rows）。逐决策行在阶段结束时
    写入 reports/episode_results.csv：把本阶段行追加到上次历史
    （跨阶段累计；键 (stage_index, stage, episode_index,
    episode_origin, gate_reason) 去重，重复运行同组不翻倍）。
    """
    reports_dir = out_dir / 'reports'
    model_bank_dir = out_dir / 'model_bank'
    mb_cfg = raw.get('model_bank', {})

    gate_enabled = bool(gate_cfg.get('enabled', True))
    gate_margin = float(gate_cfg.get('bank_margin', 0.03))
    gate_top_k = int(gate_cfg.get('gate_bank_top_k', 3))
    min_train_frames = int(gate_cfg.get('min_train_frames', 15))
    skip_streak_max = int(gate_cfg.get('skip_streak_max', 3))
    hard_drift_drop = float(gate_cfg.get('hard_drift_conf_drop', 0.15))
    switch_enabled = bool(gate_cfg.get('bank_switch_enabled', True))
    no_bank_action = str(gate_cfg.get('no_bank_action', 'skip')).strip().lower()
    buffer_cap = int(raw.get('drift_buffer', {}).get('max_frames', 300))

    episodes = reconstruct_episodes(scores, streaming_selected)
    print(f'[v9_gate] stage={stage}: {len(episodes)} drift episodes reconstructed '
          f'({sum(e["selected_count"] for e in episodes)} selected frames)')

    # R1 场景缩放源域预算（本阶段场景固定，逐片段一致）
    scene_period_ep = _scene_period(stage)
    base_source_count = int(raw['data_mix'].get('source_sample_count', 1200))
    scene_scale = float(raw['data_mix'].get('scene_source_scale', {}).get(scene_period_ep, 1.0))
    source_sample_count = int(round(base_source_count * scene_scale))

    buffer = DriftBuffer(max_frames=buffer_cap)
    skip_streak = 0
    episode_rows: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        'uploaded': 0, 'upload_bytes': 0, 'accepted': False, 'accepted_count': 0,
        'train_seconds': 0.0, 'pseudo_label_seconds': 0.0,
        'pseudo_info': {'pseudo_behavior_boxes': 0, 'pseudo_empty_images': 0, 'pseudo_avg_conf': 0.0},
        'accept_info': {
            'current_metric_gain': 0.0, 'history_metric_drop': 0.0, 'behavior_acc_gain': 0.0,
            'main_metric': raw.get('update_policy', {}).get('main_metric', 'mAP50'),
        },
        'replay_info': {'source_replay_scene': '', 'selective_replay_enabled': False},
        'bank_info': {'model_bank_enabled': bool(mb_cfg.get('enabled', True))},
        'deploy_size_info': {
            'model_full_bytes': 0.0, 'model_full_mb': 0.0,
            'model_head_bytes': 0.0, 'model_head_mb': 0.0, 'model_savings_pct': 0.0,
        },
        'train_source_images': 0, 'train_target_images': 0,
        'episode_count': 0, 'update_count': 0, 'bank_switch_count': 0,
        'forced_update_count': 0, 'buffered_frames_total': 0,
    }

    def _absorb(res: dict) -> None:
        state['accepted'] = bool(state['accepted'] or res['accepted'])
        if res['accepted']:
            state['accepted_count'] += 1
        state['train_seconds'] = float(state['train_seconds']) + float(res['train_seconds'])
        state['pseudo_label_seconds'] = float(state['pseudo_label_seconds']) + float(res['pseudo_label_seconds'])
        state['pseudo_info'] = res['pseudo_info']
        state['accept_info'] = res['accept_info']
        state['replay_info'] = res['replay_info']
        state['bank_info'] = res['bank_info']
        state['deploy_size_info'] = res['deploy_size_info']
        state['train_source_images'] = int(state['train_source_images']) + int(res['train_source_images'])
        state['train_target_images'] = int(state['train_target_images']) + int(res['train_target_images'])
        state['uploaded'] = int(state['uploaded']) + int(res['upload_count'])
        state['upload_bytes'] = int(state['upload_bytes']) + int(res['upload_bytes'])

    def _call_update(ep_tag: str, frames: list[Path], fp: dict | None, extra: dict | None) -> dict:
        return _run_update(
            exp=exp, raw=raw, method=method, method_variant=method_variant, mix_mode=mix_mode,
            stage=stage, stage_index=stage_index, episode_tag=ep_tag,
            current_model=current_model, out_dir=out_dir,
            train_images=frames, scene_fp=fp,
            source_sample_count=source_sample_count,
            deploy_mode=deploy_mode,
            edge_candidate_frames=edge_candidate_frames, edge_pass_frames=edge_pass_frames,
            edge_score_seconds=edge_score_seconds, selection_seconds=selection_seconds,
            run_deploy_competition=False, run_gating_refresh=False,
            trigger_info=None, decision_extra=extra,
            names=names, imgsz=imgsz, device=device,
            dedup_iou_threshold=dedup_iou_threshold,
        )

    def _write_nonupdate_decision(ep_tag: str, decision: str, reason: str, ep: dict, ep_frames: int,
                                  drift_score: float, current_conf, bank_conf, bank_gain,
                                  bank_model_path: str, bank_validate_seconds: float,
                                  skip_streak_before: int, buffer_size_after: int) -> None:
        payload = {
            'stage': stage, 'episode_tag': ep_tag, 'method': method,
            'method_variant': method_variant, 'train_data_mode': mix_mode,
            'gate_decision': decision, 'gate_reason': reason,
            'candidate': '', 'old_model': str(current_model),
            'accepted': False, 'uploaded': 0, 'upload_bytes': 0, 'upload_mb': 0.0,
            'edge_candidate_frames': edge_candidate_frames, 'edge_pass_frames': edge_pass_frames,
            'edge_score_seconds': edge_score_seconds, 'selection_seconds': selection_seconds,
            'pseudo_label_seconds': 0.0, 'train_source_images': 0, 'train_target_images': 0,
            'episode_index': ep['index'], 'episode_origin': ep['origin'],
            'episode_frames': ep_frames, 'drift_score': drift_score,
            'current_conf': current_conf, 'bank_conf': bank_conf, 'bank_gain': bank_gain,
            'bank_model_path': bank_model_path, 'bank_validate_seconds': bank_validate_seconds,
            'skip_streak': skip_streak_before, 'buffer_size': buffer_size_after,
            'train_seconds': 0.0,
        }
        (reports_dir / f'{stage}_{ep_tag}_decision.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    for ep in episodes:
        state['episode_count'] += 1
        ep_tag = f'e{ep["index"]}'
        ep_paths = episode_image_paths(ep)
        ep_frames = len(ep_paths)
        drift_score = float(ep.get('drift_score') or 0.0)
        if math.isnan(drift_score):
            drift_score = 0.0
        hard_drift = bool(ep_frames > 0 and hard_drift_drop > 0 and drift_score >= hard_drift_drop)

        current_conf: float | None = None
        bank_conf: float | None = None
        bank_gain: float | None = None
        bank_model_path = ''
        bank_validate_seconds = 0.0
        bank_available = False

        # 片段帧 score 行（对齐到存在的路径，供指纹/缓冲 rows 使用）
        ep_sub = ep['selected_subset']
        if ep_sub is not None and not ep_sub.empty and ep_frames > 0:
            ok_paths = {str(p) for p in ep_paths}
            ep_sub = ep_sub[ep_sub['path'].astype(str).isin(ok_paths)].copy()
        else:
            ep_sub = pd.DataFrame()

        # 库候选验证：在本片段漂移帧上（目标分布同源，修复 v8 源域评审
        # 无目标域预测力缺陷）；夜间场景不再禁切换。top_k 候选逐个验证
        # 取最优；mark_used=False 探测不污染 LRU 反馈，真正切换时才计数。
        if ep_frames > 0 and state['bank_info'].get('model_bank_enabled', False):
            scene_fp_ep = extract_scene_fingerprint(ep_sub)
            t_val = time.time()
            sel_name = str(mb_cfg.get('selector', 'fingerprint')).strip().lower()
            bank_match, qinfo = query_model_bank(
                model_bank_dir, scene_fp_ep, str(current_model),
                stage=stage,
                similarity_threshold=float(mb_cfg.get('similarity_threshold', 0.6)),
                stage_filter=bool(mb_cfg.get('stage_filter', True)),
                require_proven=bool(mb_cfg.get('require_proven', True)),
                selector=sel_name,
                gating_dir=(model_bank_dir / 'gating') if sel_name == 'gating' else None,
                gating_images=list(ep_paths) if sel_name == 'gating' else None,
                device=device,
                top_k=gate_top_k, mark_used=False,
            )
            cand_paths = [Path(c.get('model_path', '')) for c in qinfo.get('query_model_candidates', [])]
            cand_paths = [p for p in cand_paths if p.exists()]
            if bank_match is not None and Path(bank_match).exists() and str(bank_match) not in {str(p) for p in cand_paths}:
                cand_paths.append(Path(bank_match))
            if cand_paths:
                current_conf = _mean_confidence(current_model, ep_paths, imgsz, device)
                best_bank_conf = -1.0
                best_bank_path = ''
                for cp in cand_paths:
                    bconf = _mean_confidence(cp, ep_paths, imgsz, device)
                    if bconf > best_bank_conf:
                        best_bank_conf = bconf
                        best_bank_path = str(cp)
                bank_validate_seconds = time.time() - t_val
                bank_available = True
                bank_conf = best_bank_conf
                bank_gain = bank_conf - (current_conf if current_conf is not None else 0.0)
                bank_model_path = best_bank_path

        skip_streak_before = skip_streak
        if gate_enabled:
            decision, reason = decide_gate(
                bank_available=bank_available,
                bank_gain=bank_gain,
                bank_margin=gate_margin,
                bank_switch_enabled=switch_enabled,
                no_bank_action=no_bank_action,
                episode_frames=ep_frames,
                buffer_size=buffer.size(),
                buffer_cap=buffer_cap,
                skip_streak=skip_streak,
                skip_streak_max=skip_streak_max,
                min_train_frames=min_train_frames,
                hard_drift=hard_drift,
                stage_end=False,
            )
        else:
            # D_always_update 消融：闸门关闭，非空片段一律更新
            if ep_frames > 0:
                decision, reason = 'update', 'gate_disabled_always_update'
            else:
                decision, reason = 'keep_buffering', 'episode_no_frames'

        res: dict[str, Any] | None = None
        if decision in ('update', 'forced_update'):
            buf_df = buffer.rows()
            train_images = ep_paths + buffer.flush()
            if train_images:
                parts = [ep_sub] if not ep_sub.empty else []
                if not buf_df.empty:
                    parts.append(buf_df)
                update_fp = extract_scene_fingerprint(pd.concat(parts, ignore_index=True)) if parts else {}
                extra = {
                    'gate_decision': decision, 'gate_reason': reason,
                    'episode_index': ep['index'], 'episode_origin': ep['origin'],
                    'episode_frames': ep_frames, 'drift_score': drift_score,
                    'current_conf': current_conf, 'bank_conf': bank_conf, 'bank_gain': bank_gain,
                    'bank_model_path': bank_model_path,
                    'bank_validate_seconds': bank_validate_seconds,
                    'skip_streak': skip_streak_before, 'buffer_size': buffer.size(),
                }
                res = _call_update(ep_tag, train_images, update_fp, extra)
                _absorb(res)
                state['update_count'] += 1
                if decision == 'forced_update':
                    state['forced_update_count'] += 1
                skip_streak = 0
                print(f'[v9_gate] {stage} {ep_tag}: {decision} ({reason}) -> update on '
                      f'{len(train_images)} frames (ep {ep_frames} + buf {len(train_images) - ep_frames})')
            else:
                _write_nonupdate_decision(ep_tag, decision, 'update_but_no_frames', ep, ep_frames,
                                          drift_score, current_conf, bank_conf, bank_gain,
                                          bank_model_path, bank_validate_seconds,
                                          skip_streak_before, buffer.size())
                print(f'[v9_gate] {stage} {ep_tag}: {decision} ({reason}) but no frames -> recorded, no training')
        elif decision == 'bank_switch':
            shutil.copy2(bank_model_path, current_model)
            mark_model_used(model_bank_dir, bank_model_path)
            state['bank_switch_count'] += 1
            skip_streak += 1
            if ep_frames > 0:
                buffer.add(ep_paths, ep_sub)
                state['buffered_frames_total'] += ep_frames
            _write_nonupdate_decision(ep_tag, decision, reason, ep, ep_frames, drift_score,
                                      current_conf, bank_conf, bank_gain, bank_model_path,
                                      bank_validate_seconds, skip_streak_before, buffer.size())
            print(f'[v9_gate] {stage} {ep_tag}: bank_switch ({reason}) -> '
                  f'{Path(bank_model_path).name}; upload=0, train=0')
        elif decision == 'skip_no_bank':
            # 库消融组：闸门允许跳过但切换被禁用 —— 跳过=纯浪费（质量应崩塌）
            skip_streak += 1
            if ep_frames > 0:
                buffer.add(ep_paths, ep_sub)
                state['buffered_frames_total'] += ep_frames
            _write_nonupdate_decision(ep_tag, decision, reason, ep, ep_frames, drift_score,
                                      current_conf, bank_conf, bank_gain, bank_model_path,
                                      bank_validate_seconds, skip_streak_before, buffer.size())
            print(f'[v9_gate] {stage} {ep_tag}: skip_no_bank ({reason}) -> upload=0, train=0 (pure waste)')
        else:  # keep_buffering
            if ep_frames > 0:
                skip_streak += 1
                buffer.add(ep_paths, ep_sub)
                state['buffered_frames_total'] += ep_frames
            _write_nonupdate_decision(ep_tag, decision, reason, ep, ep_frames, drift_score,
                                      current_conf, bank_conf, bank_gain, bank_model_path,
                                      bank_validate_seconds, skip_streak_before, buffer.size())

        episode_rows.append({
            'method': method,
            'method_variant': method_variant,
            'train_data_mode': mix_mode,
            'stage_index': stage_index,
            'stage': stage,
            'episode_index': ep['index'],
            'episode_origin': ep['origin'],
            'episode_frames': ep_frames,
            'episode_selected_count': int(ep['selected_count']),
            'drift_score': drift_score,
            'current_conf': current_conf,
            'bank_conf': bank_conf,
            'bank_gain': bank_gain,
            'bank_model_path': bank_model_path,
            'bank_validate_seconds': bank_validate_seconds,
            'gate_decision': decision,
            'gate_reason': reason,
            'skip_streak': skip_streak_before,
            'buffer_size': buffer.size(),
            'uploaded_images': int(res['upload_count']) if res is not None else 0,
            'upload_mb': float(res['upload_bytes'] / (1024 * 1024)) if res is not None else 0.0,
            'train_seconds': float(res['train_seconds']) if res is not None else 0.0,
            'pseudo_label_seconds': float(res['pseudo_label_seconds']) if res is not None else 0.0,
            'accepted': bool(res['accepted']) if res is not None else False,
        })

    # -- 阶段尾冲刷：缓冲非空 -> 强制更新（样本不跨场景边界）--------------
    if buffer.size() > 0:
        buf_df = buffer.rows()
        flush_paths = buffer.flush()
        flush_fp = extract_scene_fingerprint(buf_df) if not buf_df.empty else {}
        extra = {
            'gate_decision': 'forced_update', 'gate_reason': 'stage_end_flush',
            'episode_index': len(episodes), 'episode_origin': 'stage_flush',
            'episode_frames': 0, 'drift_score': 0.0,
            'current_conf': None, 'bank_conf': None, 'bank_gain': None,
            'bank_model_path': '', 'bank_validate_seconds': 0.0,
            'skip_streak': skip_streak, 'buffer_size': 0,
        }
        res = _call_update('stage_flush', flush_paths, flush_fp, extra)
        _absorb(res)
        state['update_count'] += 1
        state['forced_update_count'] += 1
        episode_rows.append({
            'method': method,
            'method_variant': method_variant,
            'train_data_mode': mix_mode,
            'stage_index': stage_index,
            'stage': stage,
            'episode_index': len(episodes),
            'episode_origin': 'stage_flush',
            'episode_frames': 0,
            'episode_selected_count': 0,
            'drift_score': 0.0,
            'current_conf': None,
            'bank_conf': None,
            'bank_gain': None,
            'bank_model_path': '',
            'bank_validate_seconds': 0.0,
            'gate_decision': 'forced_update',
            'gate_reason': 'stage_end_flush',
            'skip_streak': skip_streak,
            'buffer_size': 0,
            'uploaded_images': int(res['upload_count']),
            'upload_mb': float(res['upload_bytes'] / (1024 * 1024)),
            'train_seconds': float(res['train_seconds']),
            'pseudo_label_seconds': float(res['pseudo_label_seconds']),
            'accepted': bool(res['accepted']),
        })
        print(f'[v9_gate] {stage}: stage_end_flush -> forced update on {len(flush_paths)} buffered frames')

    # -- episode 行跨阶段累计落盘（追加式，避免每次重写只剩本阶段）--------
    ep_csv = reports_dir / 'episode_results.csv'
    prior_rows: list[dict[str, Any]] = []
    if ep_csv.exists():
        try:
            prev = pd.read_csv(ep_csv)
            if not prev.empty:
                prior_rows = prev.to_dict('records')
        except Exception:
            prior_rows = []
    seen_keys = {
        (str(r.get('stage_index')), str(r.get('stage')), str(r.get('episode_index')),
         str(r.get('episode_origin')), str(r.get('gate_reason')))
        for r in prior_rows
    }
    merged_rows = prior_rows + [
        r for r in episode_rows
        if (str(r.get('stage_index')), str(r.get('stage')), str(r.get('episode_index')),
            str(r.get('episode_origin')), str(r.get('gate_reason'))) not in seen_keys
    ]
    save_csv(ep_csv, merged_rows)

    state['episode_rows'] = episode_rows
    return state


def run_one_method(config_path: str | Path, method: str, upload_ratio: float | None = None, mix_mode: str | None = None, dedup_iou_threshold: float | None = None, edge_prefilter_cache_dir: str | None = None, scores_cache_dir: str | None = None, bhss_weights: dict[str, float] | None = None, trigger_based: bool = False, selection_mode: str | None = None, use_entropy: bool | None = None, model_bank_enabled: bool | None = None, deployment_mode: str | None = None, seed: int | None = None, non_lie_ratio_threshold: float | None = None, use_conf: bool | None = None, config_overrides: dict | None = None, extra_suffix: str | None = None) -> Path:
    exp = Experiment.from_yaml(config_path)
    raw = exp.raw
    # Per-ablation-group config overrides (v8): merged BEFORE any other
    # config read so every downstream block sees the overridden values.
    if config_overrides:
        raw = _deep_merge(raw, config_overrides)
        exp.raw = raw
    set_seed(int(raw.get('seed', 42)))
    # Override seed if explicitly provided (for multi-seed experiments)
    if seed is not None:
        set_seed(seed)
        raw['seed'] = seed
    names = get_behavior_names(exp)

    # Fall back to config's scores_cache if not explicitly overridden
    if scores_cache_dir is None and exp.paths.scores_cache is not None:
        scores_cache_dir = str(exp.paths.scores_cache)

    if dedup_iou_threshold is None:
        dedup_iou_threshold = float(raw.get('models', {}).get('pseudo_dedup_iou', 0.0))

    default_mix_mode = str(raw.get('data_mix', {}).get('mix_mode', 'source_plus_hard'))
    if method == 'source_only':
        mix_mode = 'none'
    else:
        mix_mode = normalize_mix_mode(mix_mode, default_mix_mode)

    if upload_ratio is None:
        upload_ratio = float(raw['sampling'].get('upload_ratio', 0.20))

    dedup_suffix = f'_dedup{dedup_iou_threshold:.2f}'.rstrip('0').rstrip('.') if dedup_iou_threshold and dedup_iou_threshold > 0 else ''
    # BHSS weight variant suffix for output isolation
    weight_suffix = ''
    if bhss_weights is not None:
        wc = f"{float(bhss_weights.get('conf_drop', 0.50)):.2f}".replace('.', '')
        wk = f"{float(bhss_weights.get('count_drift', 0.25)):.2f}".replace('.', '')
        wb = f"{float(bhss_weights.get('behavior_drift', 0.25)):.2f}".replace('.', '')
        weight_suffix = f'_w{int(wc):02d}{int(wk):02d}{int(wb):02d}'
    trigger_suffix = '_trigger' if trigger_based else ''
    # -- Model bank / deploy mode suffix (isolates ablation groups F/G/H) --
    bank_suffix = ''
    if model_bank_enabled is not None:
        bank_suffix = '_bank' if model_bank_enabled else '_nobank'
    deploy_suffix = ''
    if deployment_mode and deployment_mode != 'full':
        deploy_suffix = f'_{deployment_mode}'
    # -- Dual-signal suffix (isolates conf-only vs conf+entropy groups) --
    # Only meaningful for adaptive+adaptive_streaming; skip for source_only/random/full_upload
    adaptive_suffix = ''
    if use_entropy is not None and method == 'adaptive':
        adaptive_suffix = '_ent' if use_entropy else '_conf'
    # -- NLR threshold suffix (isolates BSS at different nlr thresholds) --
    nlr_suffix = ''
    if non_lie_ratio_threshold is not None and non_lie_ratio_threshold > 0 and method == 'adaptive':
        nlr_int = int(round(float(non_lie_ratio_threshold) * 100))
        nlr_suffix = f'_nlr{nlr_int:02d}'
    # -- Conf signal suffix: mark nlronly (use_conf=False) ablation groups --
    conf_sig_suffix = ''
    if use_conf is not None and method == 'adaptive' and not use_conf:
        conf_sig_suffix = '_noconf'
    variant_suffix = dedup_suffix + weight_suffix + trigger_suffix + adaptive_suffix + nlr_suffix + conf_sig_suffix + bank_suffix + deploy_suffix
    # v8: group labels do not participate in method_dir naming — same-tuple
    # groups with different labels would collide on the same experiment
    # directory, so the caller must pass a disambiguating suffix.
    if extra_suffix:
        variant_suffix = variant_suffix + extra_suffix
    if not variant_suffix:
        variant_suffix = ''
    out_dir = exp.method_dir(method, upload_ratio if method != 'source_only' else None, None if method == 'source_only' else mix_mode, variant_suffix=variant_suffix)
    method_variant = _method_variant(method, None if method == 'source_only' else mix_mode) + (f'__dedup{dedup_iou_threshold:.2f}' if dedup_suffix else '') + weight_suffix + trigger_suffix + adaptive_suffix + nlr_suffix + conf_sig_suffix + bank_suffix + deploy_suffix + (extra_suffix or '')

    models_dir = out_dir / 'models'
    candidate_models_dir = out_dir / 'candidate_models'
    model_bank_dir = out_dir / 'model_bank'
    scores_dir = out_dir / 'sampling_scores'
    selected_dir = out_dir / 'selected_samples'
    pseudo_dir = out_dir / 'pseudo_labels'
    mixed_dir = out_dir / 'mixed_datasets'
    reports_dir = out_dir / 'reports'
    for d in [models_dir, candidate_models_dir, model_bank_dir, scores_dir, selected_dir, pseudo_dir, mixed_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    force_reset = bool(raw.get('experiment', {}).get('force_reset_current_model', True))
    current_model = copy_initial_model(exp.paths.initial_student, models_dir / 'current_student.pt', force_reset=force_reset)
    imgsz = int(raw['models']['imgsz'])
    device = raw['models'].get('device', 'auto')
    # -- Config overrides for ablation -------------------------------
    if model_bank_enabled is not None:
        raw.setdefault('model_bank', {})
        raw['model_bank']['enabled'] = model_bank_enabled
    deploy_mode = deployment_mode or raw.get('deployment', {}).get('mode', 'full')
    all_rows = []
    stage_history: list[str] = []
    continual_matrix_rows: list[dict] = []
    evaluation_cfg = raw.get('evaluation', {})
    matrix_metric = str(evaluation_cfg.get('forgetting_metric', raw.get('update_policy', {}).get('main_metric', 'mAP50')))
    enable_seen_matrix = bool(evaluation_cfg.get('evaluate_seen_matrix', True))
    evaluate_history = bool(raw.get('update_policy', {}).get('evaluate_history', True))
    # -- Trigger config ------------------------------------------
    trigger_cfg = raw.get('sampling', {}).get('trigger', {})
    trigger_window = int(trigger_cfg.get('window', 100))
    trigger_epsilon = float(trigger_cfg.get('epsilon', 0.01))
    c_history: list[float] = []  # accumulate change signals across stages
    # -- Drift-triggered update variant (reviewer E2-1) --------------
    # True: the model-update state is entered only when the drift detector
    # fired during the stage (drift detection triggers the update), instead
    # of the default once-per-stage update.  Upload filtering, replay, bank
    # and shadow validation are all unchanged.
    update_on_drift_only = bool(raw.get('sampling', {}).get('adaptive', {}).get('update_on_drift_only', False))

    for stage_index, stage in enumerate(exp.stage_names(), start=1):
        print(f'[{method_variant}] behavior stage {stage_index}/{len(exp.stage_names())}: {stage}')
        videos_dir = exp.stage_videos_dir(stage)
        unlabeled = exp.stage_unlabeled_dir(stage)
        stream_type = exp.stage_stream_type(stage)
        stream_dir = videos_dir if stream_type == 'videos' else unlabeled
        # All videos used for drift detection + frame accumulation.
        # Shadow validation now uses a held-out split of the uploaded
        # frames themselves (lines 988-1004), ensuring the validation
        # distribution matches the training distribution.
        train_videos: list[Path] | None = None
        if stream_type == 'videos' and stream_dir.exists():
            all_videos = list_videos(stream_dir)
            train_videos = all_videos
        test_root = exp.stage_test_root(stage)
        if not stream_dir.exists():
            print(f'  WARN missing {stream_type} dir: {stream_dir}; skip adaptation but keep evaluation')

        uploaded = 0
        accepted = False
        accepted_count = 0
        episode_count = 0
        update_count = 0
        bank_switch_count = 0
        forced_update_count = 0
        buffered_frames_total = 0
        # -- V9 per-episode update gate (VUG) -----------------------------
        # v9 组（update_gate.episode_updates: true）逐漂移片段做更新决策；
        # enabled=false 仍按片段粒度、但每个非空片段都更新（D_always_update）。
        gate_cfg = raw.get('update_gate', {}) or {}
        # selection_mode comes from the ablation tuple (the `selection_mode`
        # param), NOT from raw['sampling']: the tuple override is applied to
        # sampling_cfg further below, so reading raw here sees the YAML base
        # ('online_blocks') and silently falls back to the legacy v8 path —
        # the 2026-08 server batch burned an entire run on exactly this.
        gate_selection_mode = (
            selection_mode if selection_mode is not None
            else str(raw.get('sampling', {}).get('selection_mode', ''))
        )
        # method gate: 'adaptive' 之外还允许 'random' / 'full_upload'
        # （V9×V1 对比实验组）：两种 strategy 的漂移窗口同样由
        # ConfidenceDriftDetector 产生（sampling.py adaptive_online 包含
        # 它们），只是窗内选帧策略不同（随机比例 / 全量）——闸门与库、
        # 回放机制不变，逐片段决策语义不变。
        v9_episode_mode = (
            bool(gate_cfg.get('episode_updates', False))
            and method in ('adaptive', 'random', 'full_upload')
            and stream_type == 'videos'
            and stream_dir.exists()
            and str(gate_selection_mode).strip().lower() == 'adaptive_streaming'
        )
        mb_cfg = raw.get('model_bank', {})
        bank_info = {'model_bank_enabled': bool(mb_cfg.get('enabled', True))}
        # 方案 C：本阶段若发生 B1b 换起点，这里会指向"换起点前的部署态快照"
        # （在 B1b swap 分支内赋值）；未换起点时为 None，_run_update 行为不变。
        deployed_baseline: Path | None = None
        deploy_size_info: dict[str, float] = {
            'model_full_bytes': 0.0, 'model_full_mb': 0.0,
            'model_head_bytes': 0.0, 'model_head_mb': 0.0, 'model_savings_pct': 0.0,
        }
        train_seconds = 0.0
        accept_info = {'current_metric_gain': 0.0, 'history_metric_drop': 0.0, 'behavior_acc_gain': 0.0, 'main_metric': raw.get('update_policy', {}).get('main_metric', 'mAP50')}
        train_source_images = 0
        train_target_images = 0
        edge_score_seconds = 0.0
        selection_seconds = 0.0
        pseudo_label_seconds = 0.0
        upload_bytes = 0
        pseudo_info = {
            'pseudo_behavior_boxes': 0,
            'pseudo_empty_images': 0,
            'pseudo_avg_conf': 0.0,
        }
        edge_candidate_frames = 0
        edge_pass_frames = 0
        # Trigger info defaults (always defined, even for source_only / no-upload stages)
        trigger_info = {'triggered': True, 'c_mean': 0.0, 'c_history_mean': 0.0,
                        'a_tilde_mean': 0.0, 'trigger_skipped': False}
        # SDR replay info defaults (replaced inside the data-mix block when
        # training happens; always present so decision JSON / CSV rows stay
        # complete for source_only, no-upload and trigger-skip stages)
        replay_info: dict[str, Any] = {
            'source_replay_scene': '',
            'selective_replay_enabled': False,
        }

        if method != 'source_only' and stream_dir.exists():
            sampling_cfg = dict(raw['sampling'])
            sampling_cfg['upload_ratio'] = upload_ratio
            sampling_cfg['strategy'] = method
            sampling_cfg['num_classes'] = len(names)
            if selection_mode is not None:
                sampling_cfg['selection_mode'] = selection_mode
            if use_entropy is not None:
                sampling_cfg.setdefault('adaptive', {})
                sampling_cfg['adaptive']['use_entropy'] = use_entropy
            if non_lie_ratio_threshold is not None:
                sampling_cfg.setdefault('adaptive', {})
                sampling_cfg['adaptive']['non_lie_ratio_threshold'] = float(non_lie_ratio_threshold)
            if use_conf is not None:
                sampling_cfg.setdefault('adaptive', {})
                sampling_cfg['adaptive']['use_conf'] = use_conf
            t0 = time.time()

            # -- Scores-cache: reuse precomputed BHSS scores ----------
            # v9 gate groups must NOT read the scores cache: per-episode
            # decisions need the streaming drift_state column, and the
            # cache key is stage-name only (group-agnostic) — a cached
            # scores df could be shared across groups with different models.
            scores_cache_hit = False
            if scores_cache_dir and v9_episode_mode:
                print('[scores_cache] disabled for v9 update_gate groups (per-episode streaming scores required)')
            elif scores_cache_dir:
                scores_csv = Path(scores_cache_dir) / f'{stage}_scores.csv'
                if scores_csv.exists():
                    print(f'[scores_cache] HIT: loading precomputed scores from {scores_csv}')
                    scores = load_cached_scores(scores_csv)
                    scores_cache_hit = True
                    # Recompute hard_score with custom BHSS weights (weight sensitivity)
                    if bhss_weights is not None:
                        scores = recompute_hard_score(scores, bhss_weights)
                        print(f'[scores_cache] recomputed hard_score with weights: {bhss_weights}')
                    # Frame paths in CSV point to persistent frame cache;
                    # verify they exist (scores cache and frame cache are co-located)
                    frame_cache_check = Path(scores_cache_dir) / stage / 'frames'
                    if not frame_cache_check.exists():
                        print(f'[scores_cache] WARN: frame cache dir not found at {frame_cache_check}')

            # -- Edge-prefilter cache reuse ---------------------------
            cache_hit = False
            if not scores_cache_hit and edge_prefilter_cache_dir:
                stage_cache = Path(edge_prefilter_cache_dir) / stage / 'images'
                if cached_edge_prefilter_available(stage_cache):
                    print(f'[edge_prefilter] cache HIT: loading {len(load_cached_edge_prefilter_images(stage_cache))} images from {stage_cache}')
                    scores = compute_hard_scores(current_model, stage_cache, imgsz=imgsz, cfg=sampling_cfg)
                    cache_hit = True

            prefilter_enabled = bool(sampling_cfg.get('edge_prefilter', {}).get('enabled', False))
            edge_prefilter_img_dir = out_dir / 'edge_prefilter' / stage / 'images' if prefilter_enabled else None

            # -- Streaming / batch / cache paths -------------------------
            streaming_selected = None  # pre-selected DataFrame from streaming
            drift_detected_stage = False  # detector opened a window this stage (drift-triggered update)
            # 触发次数（本阶段漂移开窗数）——论文 α 敏感性的核心指标。
            # 由流式打分写出的 drift_window_id 列汇总，非流式路径保持 0。
            drift_windows_stage = 0
            # adaptive_streaming requires temporal video processing for
            # ConfidenceDriftDetector -- scores cache CANNOT substitute.
            # Force the streaming path regardless of cache availability.
            force_streaming = (method in ('adaptive', 'full_upload', 'dbds') and sampling_cfg.get('selection_mode', '') == 'adaptive_streaming')
            if force_streaming and stream_type == 'videos':
                frame_cache = out_dir / 'sampled_video_frames' / stage
                upload_img_dir = selected_dir / stage / 'images'
                scores, streaming_selected, drift_detected_stage = compute_hard_scores_from_videos_streaming(
                    current_model, stream_dir, frame_cache, imgsz=imgsz,
                    cfg=sampling_cfg,
                    upload_dir=upload_img_dir,
                    seed=int(raw.get('seed', 42)) + stage_index,
                    videos=train_videos,
                )
            elif scores_cache_hit:
                # Scores already loaded; skip inference entirely
                pass
            elif not cache_hit and stream_type == 'videos':
                frame_cache = out_dir / 'sampled_video_frames' / stage
                upload_img_dir = selected_dir / stage / 'images'
                scores, streaming_selected, drift_detected_stage = compute_hard_scores_from_videos_streaming(
                    current_model, stream_dir, frame_cache, imgsz=imgsz,
                    cfg=sampling_cfg,
                    upload_dir=upload_img_dir,
                    seed=int(raw.get('seed', 42)) + stage_index,
                    videos=train_videos,
                )
            elif not cache_hit:
                # -- Batch path (unlabeled images) ------------------
                scores = compute_hard_scores(current_model, stream_dir, imgsz=imgsz, cfg=sampling_cfg)
            elif cache_hit:
                scores = compute_hard_scores(current_model, stage_cache, imgsz=imgsz, cfg=sampling_cfg)

            edge_score_seconds = time.time() - t0
            edge_candidate_frames = len(scores)

            # -- Edge prefilter post-processing ------------------------
            if scores_cache_hit:
                # edge_pass already in precomputed CSV; ensure it's bool
                if 'edge_pass' not in scores.columns:
                    scores['edge_pass'] = True
            elif not cache_hit and stream_type != 'videos':
                scores = apply_edge_prefilter(scores, sampling_cfg)
                if prefilter_enabled and edge_prefilter_img_dir is not None:
                    save_edge_prefiltered_images(scores, edge_prefilter_img_dir)
            elif cache_hit:
                scores['edge_pass'] = True
            # Streaming path: edge_pass already set by streaming function

            # Full scores CSV
            scores.to_csv(scores_dir / f'{stage}_scores.csv', index=False, encoding='utf-8-sig')
            # 触发次数：drift_window_id 是单调递增的开窗计数，取最大值即本
            # 阶段累计开窗数（sampling.ConfidenceDriftDetector.window_count）。
            if 'drift_window_id' in getattr(scores, 'columns', []):
                drift_windows_stage = int(scores['drift_window_id'].max())

            # -- Selection ---------------------------------------------
            # Streaming path: selection already done per-block on edge.
            # Scores-cache path: run selection from loaded scores.
            # Batch / cache paths: run selection now on the cloud side.
            selection_seconds = 0.0
            if streaming_selected is not None:
                # Streaming already selected & saved images to upload dir
                selected = streaming_selected
                edge_pass_frames = int(scores['edge_pass'].sum()) if 'edge_pass' in scores.columns else len(scores)
                selected.to_csv(selected_dir / f'{stage}_selected.csv', index=False, encoding='utf-8-sig')
                exported = [Path(str(r['path'])) for _, r in selected.iterrows() if Path(str(r.get('path', ''))).exists()]
                uploaded = len(exported)
                upload_bytes = total_file_bytes(exported)
            else:
                # Cloud-side candidate pool
                candidate_scores = scores[scores['edge_pass']].copy() if 'edge_pass' in scores.columns else scores
                edge_pass_frames = len(candidate_scores)

                hard_threshold_raw = raw['sampling'].get('hard_score_threshold', None)
                hard_threshold = float(hard_threshold_raw) if hard_threshold_raw is not None else None
                hard_score_quantile_raw = raw['sampling'].get('hard_score_quantile', None)
                hard_score_quantile = float(hard_score_quantile_raw) if hard_score_quantile_raw is not None else None
                t1 = time.time()
                _full_upload_handled = False
                if method == 'full_upload':
                    # full_upload MUST upload only within drift-detection windows,
                    # NOT all frames globally.  Force the adaptive_streaming
                    # pipeline regardless of cache or other shortcuts.
                    if stream_type != 'videos':
                        print('[full_upload] ERROR: stream_type is not "videos", cannot run adaptive_streaming — skipping')
                        selected = pd.DataFrame()
                        uploaded = 0
                        upload_bytes = 0
                        selection_seconds = time.time() - t1
                        _full_upload_handled = True
                    else:
                        sampling_cfg_for_full = dict(sampling_cfg)
                        sampling_cfg_for_full['selection_mode'] = 'adaptive_streaming'
                        sampling_cfg_for_full['strategy'] = 'full_upload'
                        frame_cache = out_dir / 'sampled_video_frames' / stage
                        upload_img_dir = selected_dir / stage / 'images'
                        scores, streaming_selected, drift_detected_stage = compute_hard_scores_from_videos_streaming(
                            current_model, stream_dir, frame_cache, imgsz=imgsz,
                            cfg=sampling_cfg_for_full,
                            upload_dir=upload_img_dir,
                            seed=int(raw.get('seed', 42)) + stage_index,
                            videos=train_videos,
                        )
                        if streaming_selected is not None and not streaming_selected.empty:
                            selected = streaming_selected
                            edge_pass_frames = int(scores['edge_pass'].sum()) if 'edge_pass' in scores.columns else len(scores)
                            selection_seconds = time.time() - t1
                            # Streaming already copied files to upload_dir and set
                            # row['path'] to the destination.  Export & CSV handled
                            # in streaming — skip the post-selection block below.
                            selected.to_csv(selected_dir / f'{stage}_selected.csv', index=False, encoding='utf-8-sig')
                            exported = [Path(str(r['path'])) for _, r in selected.iterrows() if Path(str(r.get('path', ''))).exists()]
                            uploaded = len(exported)
                            upload_bytes = total_file_bytes(exported)
                            _full_upload_handled = True
                        else:
                            print('[full_upload] streaming produced no selection — skipping stage adaptation')
                            selected = pd.DataFrame()
                            uploaded = 0
                            upload_bytes = 0
                            selection_seconds = time.time() - t1
                            _full_upload_handled = True
                elif method == 'dbds':
                    # DBDS requires the adaptive_streaming video path
                    print('[dbds] ERROR: streaming selection unavailable for this stage - skipping')
                    selected = pd.DataFrame()
                    uploaded = 0
                    upload_bytes = 0
                else:
                    selection_mode = str(sampling_cfg.get('selection_mode', 'global')).strip().lower()
                    selector = select_samples_online_blocks if selection_mode in ('online', 'online_blocks', 'streaming') else select_samples
                    selected = selector(
                        scores=candidate_scores,
                        strategy=method,
                        upload_ratio=upload_ratio,
                        max_upload=int(raw['sampling'].get('max_upload_per_stage', 240)),
                        min_frame_gap=int(raw['sampling'].get('min_frame_gap', 5)),
                        dedup_hamming_threshold=int(raw['sampling'].get('dedup_hamming_threshold', 5)),
                        seed=int(raw.get('seed', 42)) + stage_index,
                        hard_score_threshold=hard_threshold,
                        max_candidate_pool_size=int(raw['sampling'].get('max_candidate_pool_size', 0)),
                        hard_score_quantile=hard_score_quantile,
                        **({'block_size': int(raw['sampling'].get('online_block_size', 120))} if selector is select_samples_online_blocks else {}),
                    )
                if not _full_upload_handled:
                    selection_seconds = time.time() - t1
                    selected.to_csv(selected_dir / f'{stage}_selected.csv', index=False, encoding='utf-8-sig')
                    exported = export_selected_images(selected, selected_dir / stage / 'images')
                    uploaded = len(exported)
                    upload_bytes = total_file_bytes(exported)

            bank_query_info: dict[str, Any] = {'model_bank_query_hit': False, 'model_bank_query_path': ''}
            scene_fp: dict[str, Any] = {}
            # -- V9: per-episode update gate (VUG) ---------------------------
            # HMB v9 的新角色：每个漂移片段是一个决策点，闸门在「更新模型」
            # 与「直接用模型库历史模型」之间选择（可解释分层闸门，见 gate.py
            # 与 result_auto/v9批次说明.md）。跳过更新时漂移样本进入边端
            # 缓冲，随下次更新上传（通信=0、训练=0）。
            if v9_episode_mode:
                v9_res = _run_v9_stage(
                    exp=exp, raw=raw, method=method, method_variant=method_variant,
                    mix_mode=mix_mode, stage=stage, stage_index=stage_index,
                    current_model=current_model, out_dir=out_dir,
                    scores=scores, streaming_selected=streaming_selected,
                    gate_cfg=gate_cfg, deploy_mode=deploy_mode,
                    edge_candidate_frames=edge_candidate_frames, edge_pass_frames=edge_pass_frames,
                    edge_score_seconds=edge_score_seconds, selection_seconds=selection_seconds,
                    names=names, imgsz=imgsz, device=device,
                    dedup_iou_threshold=dedup_iou_threshold,
                )
                uploaded = v9_res['uploaded']
                upload_bytes = v9_res['upload_bytes']
                accepted = v9_res['accepted']
                accepted_count = v9_res['accepted_count']
                train_seconds = v9_res['train_seconds']
                pseudo_label_seconds = v9_res['pseudo_label_seconds']
                pseudo_info = v9_res['pseudo_info']
                accept_info = v9_res['accept_info']
                replay_info = v9_res['replay_info']
                bank_info = v9_res['bank_info']
                deploy_size_info = v9_res['deploy_size_info']
                train_source_images = v9_res['train_source_images']
                train_target_images = v9_res['train_target_images']
                episode_count = v9_res['episode_count']
                update_count = v9_res['update_count']
                bank_switch_count = v9_res['bank_switch_count']
                forced_update_count = v9_res['forced_update_count']
                buffered_frames_total = v9_res['buffered_frames_total']
            else:
                # -- Scene fingerprint + B1b pre-training bank query ---------
                # HMB validated-action semantics: the bank may hand over a
                # proven same-scene spare as the training start point, but the
                # swap is VALIDATED, not unconditional (the V5 chain-breaking
                # override):
                #   * scene gate — day/eat stages only.  The source jury is
                #     daytime data; using it to approve a night swap would
                #     measure night with a day ruler (V5: T12_night restarted
                #     from T07_eat, −6.6 mAP).
                #   * validation gate — the spare must beat the CURRENT model
                #     on the source-domain images (the model's own pre-training
                #     data, unseen this stage) by source_swap_margin.
                # model_bank_query_hit keeps its legacy meaning "a swap really
                # happened" (comparable with earlier aggregate reports); a
                # rejected swap records the path + reason instead.
                if method != 'source_only' and uploaded > 0 and (
                        not update_on_drift_only or drift_detected_stage):
                    scene_fp = extract_scene_fingerprint(selected)
                    if bool(mb_cfg.get('query_on_train_start', False)):
                        sel_name = str(mb_cfg.get('selector', 'fingerprint')).strip().lower()
                        bank_match, q_info = query_model_bank(
                            model_bank_dir, scene_fp, str(current_model),
                            stage=stage,
                            similarity_threshold=float(mb_cfg.get('similarity_threshold', 0.6)),
                            stage_filter=bool(mb_cfg.get('stage_filter', True)),
                            require_proven=bool(mb_cfg.get('require_proven', True)),
                            selector=sel_name,
                            gating_dir=(model_bank_dir / 'gating') if sel_name == 'gating' else None,
                            gating_images=list(exported) if sel_name == 'gating' else None,
                            device=device,
                        )
                        if bank_match is not None:
                            legacy = bool(mb_cfg.get('b1b_legacy', False))
                            if legacy:
                                # 08-04 V1-batch semantics (V1 restore run):
                                # the query match becomes the training start
                                # point unconditionally — no scene gate, no
                                # source-jury validation (V5 machinery).
                                # Default off: current batches keep the
                                # validated B1b swap.
                                swap_scene = _scene_period(stage)
                                swap_gain: float | None = None
                                swap_ok, swap_reason = True, 'legacy_unconditional_swap'
                            else:
                                swap_scene = _scene_period(stage)
                                swap_margin = float(mb_cfg.get('source_swap_margin', 0.02))
                                swap_gain = None
                                if swap_scene in ('day', 'eat'):
                                    source_jury = _source_images(exp, cap=int(mb_cfg.get('source_swap_val_images', 50)))
                                    if source_jury:
                                        _, swap_vinfo = shadow_validate(
                                            current_model, bank_match, source_jury, imgsz, device,
                                            min_gain=0.0,
                                        )
                                        swap_gain = float(swap_vinfo['conf_gain'])
                                swap_ok, swap_reason = _should_swap_pretrain(swap_scene, swap_gain, swap_margin)
                            # -- 换起点的目标域判据（2026-09-19，可选）----------
                            # swap_validate_on_target: 换起点前，在**当阶段上传帧**
                            # 上比 bank_match 与部署态模型：spare 必须领先 margin 才
                            # 换。判据用上传帧的**训练部分**（与候选影子验证的
                            # holdout 互斥，故不构成选择泄漏），且双方都没见过这批帧
                            # —— 比源域 jury（白天图判当阶段）公平，也不需要场景门。
                            swap_target_gain: float | None = None
                            swap_target_frames = 0
                            if (swap_ok and bool(mb_cfg.get('swap_validate_on_target', False))
                                    and exported):
                                _jv, _ = _split_target_train_val(
                                    exported,
                                    float(raw['data_mix'].get('target_val_fraction', 0.10)),
                                    int(raw['data_mix'].get('min_target_val_images', 5)),
                                    int(raw.get('seed', 42)), stage_index)
                                if _jv:
                                    _, _tinfo = shadow_validate(
                                        current_model, bank_match, _jv, imgsz, device,
                                        min_gain=0.0,
                                    )
                                    swap_target_gain = float(_tinfo['conf_gain'])
                                    swap_target_frames = len(_jv)
                                    _tm = float(mb_cfg.get('swap_target_margin', 0.01))
                                    if swap_target_gain <= _tm:
                                        swap_ok = False
                                        swap_reason = (f'target_judge_below_margin_'
                                                       f'{swap_target_gain:+.4f}<={_tm:.3f}')
                                    else:
                                        swap_reason = (f'target_judge_gain_'
                                                       f'{swap_target_gain:+.4f}>{_tm:.3f}')
                            swap_info: dict[str, Any] = {
                                'model_bank_swap': bool(swap_ok),
                                'model_bank_swap_scene': swap_scene,
                                'model_bank_swap_gain': float(swap_gain) if swap_gain is not None else 0.0,
                                'model_bank_swap_reason': swap_reason,
                                'model_bank_query_path': str(bank_match),
                                # 目标域判据记录（swap_validate_on_target 时有效）
                                'model_bank_swap_target_gain':
                                    float(swap_target_gain) if swap_target_gain is not None else 0.0,
                                'model_bank_swap_target_frames': int(swap_target_frames),
                                # 查询记录（纯日志；E4-1 逐阶段表的 Similarity 列）
                                'model_bank_query_similarity':
                                    float(q_info.get('query_similarity', 0.0)),
                                'model_bank_query_score':
                                    float(q_info.get('query_score', 0.0)),
                            }
                            if swap_ok:
                                if legacy:
                                    print(f'[model_bank] B1b legacy swap: {bank_match.name} -> '
                                          f'training start point (08-04 V1 semantics, no validation)')
                                else:
                                    print(f'[model_bank] B1b swap approved: {bank_match.name} beats current '
                                          f'on {len(source_jury)} source images '
                                          f'(gain {swap_gain:+.4f} > margin {swap_margin:.3f}) -> training start point')
                                # 方案 C：换起点前先给"边缘端真正在跑的模型"留快照。
                                # 归纳保证：上一阶段若候选未部署，_run_update 已把
                                # current_model 恢复成部署态，故此处的 current_model
                                # 就是边缘端部署模型。
                                deployed_baseline = current_model.with_name(
                                    'deployed_student.pt')
                                shutil.copy2(current_model, deployed_baseline)
                                shutil.copy2(bank_match, current_model)
                                swap_info['model_bank_query_hit'] = True
                                swap_info['model_bank_swap_deployed_baseline'] = str(deployed_baseline)
                            else:
                                print(f'[model_bank] B1b swap refused for {bank_match.name}: '
                                      f'{swap_reason} (scene={swap_scene}, gain={swap_gain})')
                                swap_info['model_bank_query_hit'] = False
                            bank_query_info = swap_info

                # -- Adaptive trigger evaluation --------------------------
                if trigger_based and uploaded > 0:
                    # Collect non_lie_ratio from selected frames (activity signal)
                    if 'non_lie_ratio' in selected.columns:
                        a_vals = selected['non_lie_ratio'].dropna().to_numpy()
                    elif streaming_selected is not None and 'non_lie_ratio' in streaming_selected.columns:
                        a_vals = streaming_selected['non_lie_ratio'].dropna().to_numpy()
                    else:
                        a_vals = None
                    if a_vals is not None and len(a_vals) > 0:
                        trigger_result = evaluate_trigger(
                            a_vals, c_history=c_history,
                            window=trigger_window, epsilon=trigger_epsilon,
                        )
                        trigger_info = {
                            'triggered': trigger_result['triggered'],
                            'c_mean': trigger_result['c_mean'],
                            'c_history_mean': trigger_result['c_history_mean'],
                            'a_tilde_mean': trigger_result['a_tilde_mean'],
                            'trigger_skipped': not trigger_result['triggered'],
                        }
                        # Accumulate change values for future stages
                        c_history.extend(trigger_result['c_values'])
                        if trigger_result['triggered']:
                            print(f'[trigger] FIRED  c_t={trigger_result["c_mean"]:.4f} > [U+0109]={trigger_result["c_history_mean"]:.4f}  -> training')
                            # Reset c_history after trigger to avoid threshold creep
                            c_history = list(trigger_result['c_values'])
                        else:
                            print(f'[trigger] SKIP  c_t={trigger_result["c_mean"]:.4f} <= [U+0109]={trigger_result["c_history_mean"]:.4f}  -> no training')

            if (not v9_episode_mode and uploaded > 0
                    and (not trigger_based or trigger_info['triggered'])
                    and (not update_on_drift_only or drift_detected_stage)):
                # v8 legacy path: exactly one update per stage.  The whole
                # training block is behavior-preservingly extracted into
                # _run_update; episode_tag='stage' keeps the v8 file naming.
                # Guarded by `not v9_episode_mode`: the v9 branch already
                # performed every per-episode update above (trigger_info
                # stays at its {'triggered': True} default there).
                res = _run_update(
                    exp=exp, raw=raw, method=method, method_variant=method_variant,
                    mix_mode=mix_mode, stage=stage, stage_index=stage_index,
                    episode_tag='stage',
                    current_model=current_model, out_dir=out_dir,
                    train_images=exported, scene_fp=scene_fp,
                    source_sample_count=None,
                    deploy_mode=deploy_mode,
                    edge_candidate_frames=edge_candidate_frames, edge_pass_frames=edge_pass_frames,
                    edge_score_seconds=edge_score_seconds, selection_seconds=selection_seconds,
                    run_deploy_competition=bool(mb_cfg.get('deploy_competition', False)),
                    run_gating_refresh=(
                        bank_info.get('model_bank_enabled', False)
                        and str(mb_cfg.get('selector', 'fingerprint')).strip().lower() == 'gating'
                        and bool(exported)
                    ),
                    bank_query_info=bank_query_info,
                    trigger_info=trigger_info,
                    names=names, imgsz=imgsz, device=device,
                    dedup_iou_threshold=dedup_iou_threshold,
                    deployed_baseline=deployed_baseline,
                )
                accepted = res['accepted']
                accepted_count = int(accepted)
                train_seconds = res['train_seconds']
                pseudo_label_seconds = res['pseudo_label_seconds']
                pseudo_info = res['pseudo_info']
                accept_info = res['accept_info']
                replay_info = res['replay_info']
                bank_info = res['bank_info']
                deploy_size_info = res['deploy_size_info']
                train_source_images = res['train_source_images']
                train_target_images = res['train_target_images']
            elif update_on_drift_only and not v9_episode_mode and uploaded > 0:
                # Drift-triggered update variant (reviewer E2-1): the
                # detector never fired during this stage -> the model-update
                # state is NOT entered.  Uploaded frames stay counted as
                # communication; no training, no bank query, no deployment.
                print(f'[drift_trigger] {stage}: no drift detected -> model update skipped')

        final_metrics = evaluate_model(current_model, test_root, imgsz=imgsz, device=device, names=names) if (test_root / 'images/test').exists() else {}
        seen_eval_seconds = 0.0
        if enable_seen_matrix:
            t0 = time.time()
            seen_df = eval_on_stages(current_model, exp, stage_history + [stage], imgsz, device, names)
            seen_eval_seconds = time.time() - t0
            if not seen_df.empty:
                for _, seen_row in seen_df.iterrows():
                    item = seen_row.to_dict()
                    item.update({
                        'method': method,
                        'method_variant': method_variant,
                        'train_data_mode': mix_mode,
                        'after_stage_index': stage_index,
                        'after_stage': stage,
                    })
                    continual_matrix_rows.append(item)
                save_continual_matrix(continual_matrix_rows, reports_dir, metric=matrix_metric)
        row = {
            'method': method,
            'method_variant': method_variant,
            'train_data_mode': mix_mode,
            'upload_ratio': upload_ratio,
            'deploy_mode': deploy_mode,
            'stage_index': stage_index,
            'stage': stage,
            'model_path': str(current_model),
            'uploaded_images': uploaded,
            'upload_bytes': upload_bytes,
            'upload_mb': float(upload_bytes / (1024 * 1024)),
            'accepted_update': accepted_count,
            # drift-triggered update (E2-1): stage had uploads but no drift
            # detection -> update state not entered (decision column empty)
            'drift_trigger_skipped': bool(
                update_on_drift_only and not v9_episode_mode
                and uploaded > 0 and not drift_detected_stage),
            # 漂移开窗数（触发次数）——α 敏感性指标；旧批次 CSV 无此列
            'drift_windows': drift_windows_stage,
            'episode_count': episode_count,
            'update_count': update_count,
            'bank_switch_count': bank_switch_count,
            'forced_update_count': forced_update_count,
            'buffered_frames_total': buffered_frames_total,
            'train_source_images': train_source_images,
            'train_target_images': train_target_images,
            'edge_candidate_frames': edge_candidate_frames,
            'edge_pass_frames': edge_pass_frames,
            'edge_score_seconds': edge_score_seconds,
            'edge_score_fps': float(edge_candidate_frames / edge_score_seconds) if edge_score_seconds > 0 else 0.0,
            'selection_seconds': selection_seconds,
            'pseudo_label_seconds': pseudo_label_seconds,
            'train_seconds': train_seconds,
            'seen_eval_seconds': seen_eval_seconds,
            **pseudo_info,
            **accept_info,
            **replay_info,
            **bank_info,
            **trigger_info,
            **deploy_size_info,
            **final_metrics,
        }
        all_rows.append(row)
        save_csv(reports_dir / 'stage_results.csv', all_rows)
        stage_history.append(stage)

        # Release CUDA resources accumulated during this stage to prevent
        # OOM / hangs in later stages (especially eval_on_history which
        # creates one YOLO instance per historical stage).
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -- Cleanup frame cache to limit disk usage ------------------
        # sampled_video_frames/{stage}/ is a temporary cache; frames were
        # already copied to selected_samples/.  Delete to prevent the 12
        # stages' video frames from accumulating on disk.
        _frame_cache = out_dir / 'sampled_video_frames' / stage
        if _frame_cache.exists():
            import shutil as _shutil
            _shutil.rmtree(_frame_cache)
            print(f'[cleanup] removed frame cache: {_frame_cache}')

    aggregate_results(reports_dir / 'stage_results.csv', reports_dir)
    return reports_dir / 'stage_results.csv'


def run_all(config_path: str | Path, scores_cache_dir: str | None = None) -> None:
    exp = Experiment.from_yaml(config_path)
    rows = []
    train_data_modes = exp.raw.get('compare', {}).get('train_data_modes', ['source_plus_hard'])
    train_data_modes = [normalize_mix_mode(m) for m in train_data_modes]

    for m in exp.raw['compare']['methods']:
        if m == 'source_only':
            csv_path = run_one_method(config_path, m, None, None, scores_cache_dir=scores_cache_dir)
            rows.append(pd.read_csv(csv_path))
            continue
        modes_for_method = train_data_modes if m in UPDATE_METHODS else ['source_plus_hard']
        for mode in modes_for_method:
            if m == 'full_upload':
                csv_path = run_one_method(config_path, m, None, mode, scores_cache_dir=scores_cache_dir)
            else:
                csv_path = run_one_method(config_path, m, float(exp.raw['sampling'].get('upload_ratio', 0.20)), mode, scores_cache_dir=scores_cache_dir)
            rows.append(pd.read_csv(csv_path))

    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        out = exp.paths.work_dir / 'all_results.csv'
        all_df.to_csv(out, index=False, encoding='utf-8-sig')
        aggregate_results(out, exp.paths.work_dir / 'reports')
        print(f'All behavior results saved: {out}')


def run_upload_ratio_sweep(config_path: str | Path, methods: list[str] | None = None, mix_mode: str | None = None, scores_cache_dir: str | None = None) -> None:
    exp = Experiment.from_yaml(config_path)
    default_mix_mode = str(exp.raw.get('data_mix', {}).get('mix_mode', 'source_plus_hard'))
    mix_mode = normalize_mix_mode(mix_mode, default_mix_mode)
    if methods is None:
        methods = ['random', 'confidence', 'behavior_drift', 'ours']
    frames = []
    for ratio in exp.raw['compare']['upload_ratios']:
        for m in methods:
            csv_path = run_one_method(config_path, m, float(ratio), mix_mode, scores_cache_dir=scores_cache_dir)
            frames.append(pd.read_csv(csv_path))
    out = exp.paths.work_dir / f'ratio_sweep_results__{mix_mode}.csv'
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding='utf-8-sig')
    aggregate_results(out, exp.paths.work_dir / f'ratio_sweep_reports__{mix_mode}')
    print(f'Ratio sweep behavior results saved: {out}')


def run_mix_ablation(config_path: str | Path, method: str = 'ours', upload_ratio: float | None = None, scores_cache_dir: str | None = None) -> None:
    """Run only the training-data-composition ablation for one sample-selection strategy."""
    exp = Experiment.from_yaml(config_path)
    frames = []
    for mode in ['hard_only', 'source_plus_hard']:
        csv_path = run_one_method(config_path, method, upload_ratio or float(exp.raw['sampling'].get('upload_ratio', 0.20)), mode, scores_cache_dir=scores_cache_dir)
        frames.append(pd.read_csv(csv_path))
    out = exp.paths.work_dir / f'mix_ablation_{method}.csv'
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding='utf-8-sig')
    aggregate_results(out, exp.paths.work_dir / f'mix_ablation_{method}_reports')
    print(f'Mix ablation results saved: {out}')


# V8 per-group config overrides (deep-merged into the v8 yaml at
# run_one_method start) and output-dir disambiguation suffixes (group
# labels do not participate in method_dir naming — without a suffix,
# same-tuple groups like D_entropy_rscaled_bankcomp / D_entropy_rscaled
# would collide on one experiment directory).
_group_overrides: dict[str, dict] = {
    'D_entropy_rscaled': {'model_bank': {'deploy_competition': False}},
    'D_entropy_rselective_bankcomp': {'data_mix': {'selective_source_replay': {'enabled': True}}},
    # V9 (VUG) groups — 见 configs/behavior_experiment_v9.yaml 与 v9批次说明:
    #   D_skip_nobank: bank_switch_enabled=false — 闸门允许跳过但切换被
    #     禁用，跳过=纯浪费（库注册保持开启，闸门才有候选可验证）；
    #     若直接关 model_bank.enabled 库永远为空，闸门将退化为每片段
    #     更新，消融组与 D_always_update 无差异。
    #   D_always_update: enabled=false — 仍按片段粒度，但每个非空片段都
    #     更新（≈v8 行为、片段粒度化，作 v9 内部上界锚）。
    #   D_gate_aggressive: bank_margin 0.01 — 降低切换门槛，切换更激进。
    #   D_gate_nobank_update: bank_switch_enabled=false + no_bank_action=
    #     update — 无库对照：库候选增益达标本应切换，无库则改为直接重
    #     训练（论文「无库→直接重训」对照链；与 skip_nobank 纯浪费对照）。
    'D_skip_nobank': {'update_gate': {'bank_switch_enabled': False}},
    'D_always_update': {'update_gate': {'enabled': False}},
    'D_gate_aggressive': {'update_gate': {'bank_margin': 0.01}},
    'D_gate_nobank_update': {'update_gate': {'bank_switch_enabled': False,
                                             'no_bank_action': 'update'}},
    # V9×V1sel 组：覆盖与 v9 同构（见 run_ablation 内注释与
    # result_auto/v9_v1sel批次说明.md）
    'D_skip_nobank_v1sel': {'update_gate': {'bank_switch_enabled': False}},
    'D_always_update_v1sel': {'update_gate': {'enabled': False}},
    'D_gate_aggressive_v1sel': {'update_gate': {'bank_margin': 0.01}},
    'D_gate_nobank_update_v1sel': {'update_gate': {'bank_switch_enabled': False,
                                                   'no_bank_action': 'update'}},
    # V9×V1 扩展批次三：P_10（sampling.adaptive.p_low）敏感性扫描 ——
    # p_low 同时决定漂移检测的 H0 基准与 conf<P_10 筛选门限
    # （sampling.py ConfidenceDriftDetector）；nlr 固定 0.3。
    'D_gate_p1002_v1sel': {'sampling': {'adaptive': {'p_low': 0.2}}},
    'D_gate_p1003_v1sel': {'sampling': {'adaptive': {'p_low': 0.3}}},
    'D_gate_p1004_v1sel': {'sampling': {'adaptive': {'p_low': 0.4}}},
    'D_gate_p1005_v1sel': {'sampling': {'adaptive': {'p_low': 0.5}}},
    # V1 机制参数敏感性（2026-08-25）：P_10 扫描——p_low 同时决定漂移检测
    # H0 与 conf<P_10 门限（sampling.py ConfidenceDriftDetector）；nlr 固定
    # 0.3；锚点 p_low=0.10 = D_nlr03_replay（同批重跑，不重复注册）。
    # nlr 扫描的阈值直接写在元组 nlr 字段（D_nlr01/02/04/05_replay），
    # 不需要覆盖。
    'D_nlr03_replay_p102': {'sampling': {'adaptive': {'p_low': 0.2}}},
    'D_nlr03_replay_p103': {'sampling': {'adaptive': {'p_low': 0.3}}},
    'D_nlr03_replay_p104': {'sampling': {'adaptive': {'p_low': 0.4}}},
    'D_nlr03_replay_p105': {'sampling': {'adaptive': {'p_low': 0.5}}},
    # 2026-09-19 锚点改 P_20：v1_i640_p02* 批的 yaml 把 p_low 默认值改为
    # 0.20（见 configs/behavior_experiment_v1_i640_p02_*.yaml），于是
    # D_nlr03_replay_p102 与锚点 D_nlr03_replay 同构（p_low 同为 0.20），
    # 不再作为扫描点使用（保留注册以复现历史批次：
    # 置信度+非躺V3/sensitityV1、V10/V13/V14/V15 的 p 扫均为 0.1 基准）。
    # 新基准下的 p 扫左端补一个 p=0.10 点 D_nlr03_replay_p101，
    # 命名沿用 p10X = p_low 0.X 的约定。
    'D_nlr03_replay_p101': {'sampling': {'adaptive': {'p_low': 0.1}}},
    # 注意：p_low 不参与 method_dir 命名（run_one_method 的 variant_suffix
    # 无 p_low 分量）——D_nlr03_replay_p102..p105 与锚点 D_nlr03_replay
    # 元组同构，416 敏感性批（置信度＋非躺V3/sensitityV1）五组共用同一
    # 实验目录，model_bank.csv 跨组残留（bankV1 教训）。640 批必须靠
    # 后缀隔离（见 _group_suffixes 的 _p10x 条目）。
    # 模型库三臂消融（2026-08-28，BCCL 论文表：w/o Bank / Retrieval Init
    # / Full）。yaml 基座 = V1 还原配置（query_on_train_start + b1b_legacy
    # + selector similarity）。三臂：
    #   D_nlr03_replay_nobank   = 库全关（enabled=false 已由元组 bank 字段
    #    关闭；此处再显式关 query/fallback，保证表内 ✗ 是字面意义——不查
    #    询、不回退，而非"查询空库"）
    #   D_nlr03_replay_bankinit = 仅训练初始化（B1b 训前查询+legacy 换起
    #    点），rejection_fallback=false 关拒绝回退
    #   D_nlr03_replay          = Full（yaml 原值：查询 + 拒绝回退全开）
    'D_nlr03_replay_nobank': {'model_bank': {'query_on_train_start': False,
                                             'rejection_fallback': False}},
    'D_nlr03_replay_bankinit': {'model_bank': {'rejection_fallback': False}},
    # 触发策略 × 部署验证批（2026-08-30，审稿 E2-1/E4-1）：
    #   D_nlr03_replay_interval = 固定间隔开窗（只换 trigger_mode 开窗
    #     条件，上传过滤 conf<P_10 AND nlr>0.3 与锚点完全一致）
    #   D_nlr03_replay_confthr  = 置信度阈值开窗（滚动均值持续跌破 θ）
    #   D_nlr03_replay_direct   = 库历史模型直接部署（跳过影子验证，
    #     validation-separation 表「unvalidated deployment」行）
    'D_nlr03_replay_interval': {'sampling': {'adaptive': {'trigger_mode': 'fixed_interval'}}},
    'D_nlr03_replay_confthr': {'sampling': {'adaptive': {'trigger_mode': 'conf_threshold'}}},
    'D_nlr03_replay_direct': {'model_bank': {'direct_deploy': True}},
    # 漂移检测即触发批（2026-08-31，审稿 E2-1 第四行）：与锚点
    # D_nlr03_replay 完全同构，唯一差别是 update_on_drift_only ——
    # 二项检验检测到漂移（开过窗）才进入模型更新状态，否则跳过本阶段
    # 更新（上传/筛选/回放/库/影子验证全部不变）。
    'D_nlr03_replay_drift': {'sampling': {'adaptive': {'update_on_drift_only': True}}},
}
_group_suffixes: dict[str, str] = {
    'D_entropy_rscaled_bankcomp': '_rscaled_bank',
    'D_entropy_rscaled_nobank': '_rscaled_nobank',
    'D_entropy_rscaled': '_rscaled_nocomp',
    'D_entropy_rselective_bankcomp': '_rsel',
    'F_entropy_noreplay_v8': '_v8',
    'I_full_upload_replay_v8': '_v8',
    # V9 suffixes are MANDATORY: the four v9 tuples are identical to the v8
    # D-group tuple shape — without a suffix they would write into the v8
    # experiment directories.
    'D_gate_bank': '_v9_gatebank',
    'D_skip_nobank': '_v9_skipnobank',
    'D_always_update': '_v9_alwaysupd',
    'D_gate_aggressive': '_v9_aggr',
    'D_gate_nobank_update': '_v9_nobankupd',
    # V9×V1sel 组同样必须带后缀，否则与 v9 四组目录碰撞
    'D_gate_bank_v1sel': '_v9v1_gatebank',
    'D_skip_nobank_v1sel': '_v9v1_skipnobank',
    'D_always_update_v1sel': '_v9v1_alwaysupd',
    'D_gate_aggressive_v1sel': '_v9v1_aggr',
    'D_gate_nobank_update_v1sel': '_v9v1_nobankupd',
    # V9×V1 扩展批次一/二/三（消融/对比/敏感性；均 seed 42 同流可对齐）
    'D_gate_noreplay_nobank_v1sel': '_v9v1_noreplay_nobank',
    'D_gate_noreplay_bank_v1sel':   '_v9v1_noreplay_bank',
    'D_gate_replay_nobank_v1sel':   '_v9v1_replay_nobank',
    'D_gate_conf_v1sel':     '_v9v1_conf',
    'D_gate_entropy_v1sel':  '_v9v1_ent',
    'D_gate_random02_v1sel': '_v9v1_rand02',
    'D_gate_random03_v1sel': '_v9v1_rand03',
    'D_gate_random04_v1sel': '_v9v1_rand04',
    'D_gate_fullup_v1sel':   '_v9v1_fullup',
    'D_gate_nlr01_v1sel': '_v9v1_nlr01',
    'D_gate_nlr02_v1sel': '_v9v1_nlr02',
    'D_gate_nlr04_v1sel': '_v9v1_nlr04',
    'D_gate_nlr05_v1sel': '_v9v1_nlr05',
    'D_gate_p1002_v1sel': '_v9v1_p1002',
    'D_gate_p1003_v1sel': '_v9v1_p1003',
    'D_gate_p1004_v1sel': '_v9v1_p1004',
    'D_gate_p1005_v1sel': '_v9v1_p1005',
    # 模型库三臂消融（2026-08-28）：D_nlr03_replay 与 D_nlr03_replay_bankinit
    # 元组完全相同（差别只在 _group_overrides 的 rejection_fallback），必须
    # 靠后缀隔离实验目录——否则两组共用 model_bank 目录互相污染（bankV1
    # 批已踩坑：bankinit 组 T01 起就命中 full 组留下的库模型，臂无效）。
    'D_nlr03_replay_bankinit': '_bankinit',
    # 触发×部署批：三个新组与锚点 D_nlr03_replay 元组完全相同（差别只在
    # _group_overrides），必须靠后缀隔离实验目录，否则共用 model_bank
    # 目录互相污染（同上 bankV1 教训）。
    'D_nlr03_replay_interval': '_trig_interval',
    'D_nlr03_replay_confthr': '_trig_confthr',
    'D_nlr03_replay_direct': '_trig_direct',
    # 漂移检测即触发批：同样必须后缀隔离（与锚点同元组同 model_bank 目录语义）
    'D_nlr03_replay_drift': '_drift',
    # 输入分辨率对比批（imgsz=640，2026-09-07）：P_10 扫描四组与锚点
    # D_nlr03_replay 元组同构（p_low 不进 method_dir），必须后缀隔离——
    # 否则五组共用 model_bank.csv 跨组污染（416 敏感性批已踩坑，见
    # _group_overrides 内注释）。nlr 扫描四组（_nlr10/20/40/50）天然
    # 带 nlr 后缀，无需注册。
    'D_nlr03_replay_p102': '_p102',
    'D_nlr03_replay_p103': '_p103',
    'D_nlr03_replay_p104': '_p104',
    'D_nlr03_replay_p105': '_p105',
    # 2026-09-19 锚点改 P_20 后新增的左端点（p_low=0.10）
    'D_nlr03_replay_p101': '_p101',
}


# ── 参数敏感性两批的组注册（2026-09-30）──────────────────────────────
# 2D 联合网格（P_q × τ_a）与 α 扫描的组清单在这里集中定义，三处注册
# （run_ablation 的 ablation_configs 元组 / _group_overrides / _group_suffixes）
# 全部从下面的常量派生——手写 16 条时标签与覆盖一旦不一致，该组会静默退回
# yaml 默认 p_low=0.20，变成锚点的重复实验。
#
#   P_q = 置信度分位阈值（sampling.adaptive.p_low，同时决定 conf<P_q 门限
#         与漂移检测 H0 的 p）——0.2 整行复用 V17 sens 批，不重复跑；
#   τ_a = 行为活动度阈值（non_lie_ratio_threshold，走元组的 nlr 字段）——
#         0.3 整列复用 V17 sens 批，不重复跑；
#   α   = 漂移检测显著水平，只通过 k_crit 起作用（sampling.py
#         ConfidenceDriftDetector._compute_k_crit）：W_S=20、P_q=0.20 下
#         α=0.01/0.03 同为 k_crit=9、α=0.10/0.15 同为 k_crit=7——即
#         0.03 与 0.01、0.15 与 0.10 是**逐位等价的实验**（同种子下结果
#         完全相同）。它们在表里只用于"实证 α 的离散跳变"这一点；
#         选点 {0.01, 0.05, 0.10, 0.20} 已覆盖 k_crit = {9, 8, 7, 6}。
# 数据源与命令见 result_auto/v1_i640_tables_V17/参数敏感性2D与alpha批次说明.md。
_GRID2D_PQ: tuple[int, ...] = (10, 30, 40, 50)   # P_q × 100
_GRID2D_TA: tuple[int, ...] = (10, 20, 40, 50)   # τ_a × 100
# α × 100。3 与 15 是 2026-09-30 追加：与 1 / 10 同为 k_crit=9 / 7，
# 跑出来应与 _alpha01 / _alpha10 逐位一致（用于实证离散性，不提供新信息）。
_ALPHA_SWEEP: tuple[int, ...] = (1, 3, 5, 10, 15, 20)


def _grid2d_label(pq: int, ta: int) -> str:
    """网格组标签：``D_grid_pq{10|30|40|50}_ta{10|20|40|50}``（pq/ta 为 ×100）。"""
    return f'D_grid_pq{pq:02d}_ta{ta:02d}'


def _register_sensitivity_batches() -> None:
    """把两批的覆盖与后缀写进 _group_overrides / _group_suffixes。"""
    for pq in _GRID2D_PQ:
        for ta in _GRID2D_TA:
            lbl = _grid2d_label(pq, ta)
            _group_overrides[lbl] = {'sampling': {'adaptive': {'p_low': pq / 100.0}}}
            # p_low 不进 method_dir 命名 → 必须后缀隔离，否则 16 组共用
            # 实验目录 + model_bank 跨组污染（bankV1 教训）
            _group_suffixes[lbl] = f'_p{100 + pq // 10}'
    for a in _ALPHA_SWEEP:
        lbl = f'D_alpha{a:02d}'
        _group_overrides[lbl] = {'sampling': {'adaptive': {'alpha': a / 100.0}}}
        _group_suffixes[lbl] = f'_alpha{a:02d}'


_register_sensitivity_batches()


def run_ablation(config_path: str | Path, upload_ratio: float | None = None, dedup_iou_threshold: float | None = None, scores_cache_dir: str | None = None, seeds: list[int] | None = None, groups: list[str] | None = None) -> None:
    """Run ablation study with optional multi-seed support.

    When *seeds* is provided (e.g. [42, 123, 456]), each ablation group is
    run once per seed.  The output CSV includes a ``seed`` column and the
    aggregate summary reports mean +/- std across seeds.

    **V1: entropy + non_lie_ratio filtering**

    Selection condition::

        conf < P_10 (model prediction info) AND
        (behavior_entropy > H_med  OR  non_lie_ratio > threshold)

    Both P_10 and H_med are statistical baselines from the long window.
    When non_lie_ratio_threshold > 0, the second signal switches from
    behavior_entropy to non_lie_ratio (pig activity filter).

    11 groups:

      A: source_only     — no update (lower bound)
      B: random r=0.10   — drift-window random sampling (low-upload baseline)
      B_20: random r=0.20 — random baseline (mid-upload)
      B_30: random r=0.30 — random baseline (high-upload)
      C_conf_p10: conf < P_10 only (single-signal ablation)
      D_entropy: conf < P_10 AND entropy > H_med (entropy-based)
      D_nlr03: conf < P_10 AND nlr > 0.3 (OUR METHOD)
      D_nlr04: conf < P_10 AND nlr > 0.4 (stricter nlr threshold)
      D_nlr03_nobank: D_nlr03 without model bank
      D_nlr03_replay: D_nlr03 + source replay
      I_full_upload: all drift-window frames (upper bound)

    Key comparisons:
      C  -> D_entropy : entropy signal over confidence-only
      D_entropy -> D_nlr03 : nlr signal over entropy
      D_nlr03 -> D_nlr04 : nlr threshold sensitivity
      D_nlr03 -> D_nlr03_nobank : model bank ablation
      D_nlr03 -> D_nlr03_replay : source replay ablation
      A  vs I_full_upload : continual learning bounds
    """
    exp = Experiment.from_yaml(config_path)
    default_ratio = float(exp.raw['sampling'].get('upload_ratio', 0.20))
    ratio = upload_ratio if upload_ratio is not None else default_ratio

    # Tuple: (group_label, method, mix_mode, sel_mode, use_entropy,
    #         model_bank, deploy_mode, nlr_threshold, use_conf)
    # nlr_threshold=None → not set (uses config default 0.0 = entropy-based)
    # nlr_threshold=0.0    → explicitly entropy-based (H > H_med)
    # nlr_threshold>0      → non_lie_ratio > threshold replaces H > H_med
    # use_conf=None/True   → normal (conf < P_10 as signal 1)
    # use_conf=False       → disable conf signal; nlr-only single-signal ablation
    ablation_configs: list[tuple[str, str, str, str | None, bool, bool | None, str | None, float | None, bool | None]] = [
        # ---- Baselines (hard_only, no source replay) ----
        ('A_source_only',     'source_only', 'hard_only',        None,                 False, None,  None,  None,  None),
        ('B_random',          'random',      'hard_only',        'adaptive_streaming', False, None,  None,  None,  None),
        ('B_random_20',       'random',      'hard_only',        'adaptive_streaming', False, None,  None,  None,  None),
        ('B_random_30',       'random',      'hard_only',        'adaptive_streaming', False, None,  None,  None,  None),
        # ---- Single-signal: conf < P_10 only (no second signal) ----
        ('C_conf_p10',        'adaptive',    'hard_only',        'adaptive_streaming', False, True,  'full', 0.0,   None),
        # ---- Dual-signal: conf < P_10 AND entropy > H_med ----
        ('D_entropy',         'adaptive',    'hard_only',        'adaptive_streaming', True,  True,  'full', 0.0,   None),
        # ---- Dual-signal: conf < P_10 AND nlr > 0.3 (OUR METHOD) ----
        ('D_nlr03',           'adaptive',    'hard_only',        'adaptive_streaming', True,  True,  'full', 0.3,   None),
        # ---- nlr threshold sensitivity ----
        ('D_nlr04',           'adaptive',    'hard_only',        'adaptive_streaming', True,  True,  'full', 0.4,   None),
        # ---- System ablation: model bank ----
        ('D_nlr03_nobank',    'adaptive',    'hard_only',        'adaptive_streaming', True,  False, 'full', 0.3,   None),
        # ---- System ablation: source replay ----
        ('D_nlr03_replay',    'adaptive',    'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        # ---- Upper bound ----
        ('I_full_upload',     'full_upload', 'hard_only',        'adaptive_streaming', False, None,  None,  None,  None),

        # ==== NEW: source replay + model bank variants ===================
        # ---- Random baselines + source replay ----
        ('B_random_replay',     'random',   'source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),
        ('B_random_20_replay',  'random',   'source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),
        ('B_random_30_replay',  'random',   'source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),
        ('B_random_40_replay',  'random',   'source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),
        # ---- Single-signal conf < P_10 + source replay ----
        ('C_conf_replay',       'adaptive', 'source_plus_hard', 'adaptive_streaming', False, True,  'full', 0.0,   None),
        # ---- Dual-signal conf < P_10 AND entropy > H_med + source replay + bank
        #      （主表公平版 Entropy 臂：与 D_entropy 同信号，仅加回放+库）----
        ('D_entropy_replay',    'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   None),
        # ---- Single-signal nlr > 0.3 only (no conf) + source replay ----
        ('nlr03_only_replay',   'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   False),
        # ---- System ablation: dual-signal conf+nlr03 + source replay, NO model bank ----
        ('D_nlr03_replay_nobank', 'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  False, 'full', 0.3,   None),
        # ---- Model-bank 3-arm ablation (2026-08-28): retrieval init only,
        #      no rejection fallback (override in _group_overrides) ----
        ('D_nlr03_replay_bankinit', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.3,   None),
        # ---- 触发策略 × 部署验证（2026-08-30，审稿 E2-1/E4-1）：与锚点
        #      D_nlr03_replay 完全同构，差异只在 _group_overrides ----
        ('D_nlr03_replay_interval', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.3,   None),
        ('D_nlr03_replay_confthr',  'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.3,   None),
        ('D_nlr03_replay_direct',   'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.3,   None),
        # ---- 漂移检测即触发（2026-08-31，审稿 E2-1 第四行）：与锚点
        #      D_nlr03_replay 完全同构，差异只在 _group_overrides 的
        #      update_on_drift_only（漂移检测开过窗才进入更新状态）----
        ('D_nlr03_replay_drift',    'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.3,   None),
        # ---- Upper bound: full upload (all drift-window frames) + source replay + model bank ----
        ('I_full_upload_replay',  'full_upload','source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),

        # ==== V1 机制参数敏感性（2026-08-25）：P_10 × nlr 扫描 ==========
        # 与 D_nlr03_replay 完全同构（V1 双信号 conf<P_10 AND nlr>0.3 +
        # V1 源域回放 + V1 模型库、阶段级 shadow 验证判定——标准分支，
        # 非 v9 闸门），仅扫一个参数。锚点（P_10=0.10, nlr=0.3）=
        # D_nlr03_replay（同批同代码重跑，落到独立 work_dir）。
        # P_10 扫描：nlr=0.3 固定，p_low ∈ {0.2..0.5}（覆盖见
        # _group_overrides；p_low 同时决定漂移 H0 与 conf<P_10 门限）
        ('D_nlr03_replay_p102', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        ('D_nlr03_replay_p103', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        ('D_nlr03_replay_p104', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        ('D_nlr03_replay_p105', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        # 2026-09-19 锚点改 P_20（p02 批）后的 p 扫左端点：p_low=0.10，
        # nlr=0.3 固定，与锚点 D_nlr03_replay 元组同构（靠 _p101 后缀隔离）。
        ('D_nlr03_replay_p101', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        # nlr 扫描：p_low=0.10 固定，non_lie_ratio_threshold ∈ {0.1,0.2,0.4,0.5}
        ('D_nlr01_replay', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.1,   None),
        ('D_nlr02_replay', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.2,   None),
        ('D_nlr04_replay', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.4,   None),
        ('D_nlr05_replay', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.5,   None),

        # ==== DBDS: distribution-balanced drift screening ==================
        # ---- Main group: DBDS + source replay + model bank ----
        ('E_dbds',                'dbds',      'source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),
        # ---- System ablations: replay x bank ----
        ('E_dbds_nobank',         'dbds',      'source_plus_hard', 'adaptive_streaming', False, False, 'full', None,  None),
        ('E_dbds_noreplay',       'dbds',      'hard_only',        'adaptive_streaming', False, True,  'full', None,  None),
        ('E_dbds_bare',           'dbds',      'hard_only',        'adaptive_streaming', False, False, 'full', None,  None),
        # ---- Single-signal baseline: behavior entropy only (H > H_med, no conf) + replay + bank ----
        ('D_entropy_only_replay', 'adaptive',  'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   False),
        # ==== DBES: drift-triggered behavior-entropy screening (OUR METHOD) ====
        # ---- System ablations: replay x bank ----
        ('F_entropy_nobank',      'adaptive',  'source_plus_hard', 'adaptive_streaming', True,  False, 'full', 0.0,   False),
        ('F_entropy_noreplay',    'adaptive',  'hard_only',        'adaptive_streaming', True,  True,  'full', 0.0,   False),
        ('F_entropy_bare',        'adaptive',  'hard_only',        'adaptive_streaming', True,  False, 'full', 0.0,   False),

        # ==== V8: SDR (scene-adaptive replay) x HMB (validated bank) =====
        # All groups inherit the v8 yaml (R1 scene scale + R2 fraction cap
        # + B1a hard gate + B1b validated pre-training swap + B2 deploy
        # competition).  Group-specific switches live in _group_overrides.
        #   D_entropy_rscaled_bankcomp    main group  (R1+R2+B1+B2)
        #   D_entropy_rscaled_nobank      bank ablation
        #   D_entropy_rscaled             competition ablation (bank fallback only)
        #   D_entropy_rselective_bankcomp R3 selective-replay variant
        #   F_entropy_noreplay_v8         replay ablation (no replay)
        #   I_full_upload_replay_v8       upper-bound anchor
        ('D_entropy_rscaled_bankcomp',    'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   False),
        ('D_entropy_rscaled_nobank',      'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  False, 'full', 0.0,   False),
        ('D_entropy_rscaled',             'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   False),
        ('D_entropy_rselective_bankcomp', 'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   False),
        ('F_entropy_noreplay_v8',         'adaptive',   'hard_only',        'adaptive_streaming', True,  True,  'full', 0.0,   False),
        ('I_full_upload_replay_v8',       'full_upload','source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),

        # ==== V9: VUG (validated update gate) x HMB ======================
        # Twin-run design: same stream + same seeds, 4 update policies.  All
        # groups run the per-episode flow via update_gate.episode_updates in
        # configs/behavior_experiment_v9.yaml; differences are in
        # _group_overrides (bank_switch_enabled / enabled / bank_margin).
        # Tuples are IDENTICAL to the v8 main-group shape for sampling
        # comparability -- the mandatory _group_suffixes entries keep the
        # output dirs apart from the v8 experiment directories.
        #   D_gate_bank        Ours: gate + bank + drift buffer
        #   D_skip_nobank      bank ablation: switching disabled -> skip wastes
        #   D_always_update    baseline/upper bound: update every non-empty episode
        #   D_gate_aggressive  sensitivity: bank_margin 0.01 -> more switching
        #   D_gate_nobank_update  no-bank control: switch suppressed -> retrain
        ('D_gate_bank',        'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   False),
        ('D_skip_nobank',      'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   False),
        ('D_always_update',    'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   False),
        ('D_gate_aggressive',  'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   False),
        ('D_gate_nobank_update','adaptive',  'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   False),

        # ==== V9 gate x V1 selection + V1 replay ========================
        # 闸门机制与上面 v9 四组完全相同（同构覆盖），仅筛选信号换成
        # V1 D_nlr03 的双信号：conf < P_10 AND non_lie_ratio > 0.3
        # （nlr=0.3 替换熵信号、use_conf=None 恢复置信度门控；v9 原四组
        # use_conf=False 只留熵信号）。回放由 behavior_experiment_v9_v1sel.yaml
        # 控制为 V1 原样（无 R1/R2/R3，随机回放源域全集）。
        #   组后缀 *_v1sel 与覆盖/目录后缀 _v9v1_* 一一对应。
        ('D_gate_bank_v1sel',        'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        ('D_skip_nobank_v1sel',      'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        ('D_always_update_v1sel',    'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        ('D_gate_aggressive_v1sel',  'adaptive',   'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),
        ('D_gate_nobank_update_v1sel','adaptive',  'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.3,   None),

        # ==== V9 gate × V1 扩展批次一：筛选 × 回放 × 库 三因子消融 ======
        # 2×2 消融（同流同种子；第四格「回放+库」= D_gate_bank_v1sel）：
        #   回放 → tuple mix_mode（hard_only / source_plus_hard）
        #   库   → tuple mb_enabled（True / False）
        # mb=False 时闸门照常运行（keep_buffering/skip_streak/缓冲仍有效），
        # 只是库永不注册/无候选 → 每个合格片段 update/no_bank_match ——
        # 真正的「无库」对照（区别于 skip_nobank 的纯浪费与
        # nobank_update 的切换转重训）。
        ('D_gate_noreplay_nobank_v1sel', 'adaptive', 'hard_only',        'adaptive_streaming', True,  False, 'full', 0.3,   None),
        ('D_gate_noreplay_bank_v1sel',   'adaptive', 'hard_only',        'adaptive_streaming', True,  True,  'full', 0.3,   None),
        ('D_gate_replay_nobank_v1sel',   'adaptive', 'source_plus_hard', 'adaptive_streaming', True,  False, 'full', 0.3,   None),

        # ==== V9 gate × V1 扩展批次二：筛选机制对比（全带回放+库）=========
        # 与 V1 十一组实验的筛选机制同构：conf-only（use_entropy=False）、
        # entropy（conf<P_10 AND H>H_med）、漂移检测阶段随机 0.2/0.3/0.4
        # （strategy=random，窗内随机采样）、漂移检测阶段 full_upload
        # （窗内全量上传）。random/full_upload 的漂移窗口仍由
        # ConfidenceDriftDetector 产生（v9_episode_mode 已扩展到这两种
        # method），闸门逐片段决策不变；随机组上传比例见 _random_ratios。
        ('D_gate_conf_v1sel',     'adaptive',    'source_plus_hard', 'adaptive_streaming', False, True,  'full', 0.0,   None),
        ('D_gate_entropy_v1sel',  'adaptive',    'source_plus_hard', 'adaptive_streaming', True,  True,  'full', 0.0,   None),
        ('D_gate_random02_v1sel', 'random',      'source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),
        ('D_gate_random03_v1sel', 'random',      'source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),
        ('D_gate_random04_v1sel', 'random',      'source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),
        ('D_gate_fullup_v1sel',   'full_upload', 'source_plus_hard', 'adaptive_streaming', False, True,  'full', None,  None),

        # ==== V9 gate × V1 扩展批次三：双信号参数敏感性（全带回放+库）======
        # nlr 扫描：P_10 固定 0.10（p_low 默认），non_lie_ratio_threshold ∈
        # {0.1, 0.2, 0.4, 0.5}（0.3 = D_gate_bank_v1sel，不重复注册）。
        # P_10 扫描：nlr=0.3 固定，sampling.adaptive.p_low ∈ {0.2..0.5}
        # （0.10 = D_gate_bank_v1sel，不重复注册）——p_low 同时决定漂移
        # 检测 H0 基准与 conf<P_10 筛选门限（sampling.py 706-708），
        # 覆盖见 _group_overrides。
        ('D_gate_nlr01_v1sel', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.1, None),
        ('D_gate_nlr02_v1sel', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.2, None),
        ('D_gate_nlr04_v1sel', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.4, None),
        ('D_gate_nlr05_v1sel', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.5, None),
        ('D_gate_p1002_v1sel', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.3, None),
        ('D_gate_p1003_v1sel', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.3, None),
        ('D_gate_p1004_v1sel', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.3, None),
        ('D_gate_p1005_v1sel', 'adaptive', 'source_plus_hard', 'adaptive_streaming', True, True, 'full', 0.3, None),
    ]

    # ==== 参数敏感性两批（2026-09-30）：2D 联合网格 16 组 + α 扫描 4 组 ====
    # 标签/覆盖/后缀同源于模块级 _GRID2D_PQ / _GRID2D_TA / _ALPHA_SWEEP
    # （见 _register_sensitivity_batches）。写法上刻意用 += 而不是把生成式
    # 塞进上面的字面量列表——scripts/smoke_v1_i640.py 检查 [2] 用
    # ast.literal_eval 解析那份字面量，展开式会让它解析失败。
    # 跑批命令见 result_auto/v1_i640_tables_V17/参数敏感性2D与alpha批次说明.md §二。
    ablation_configs += [
        (_grid2d_label(pq, ta), 'adaptive', 'source_plus_hard', 'adaptive_streaming',
         True, True, 'full', ta / 100.0, None)
        for pq in _GRID2D_PQ for ta in _GRID2D_TA
    ]
    ablation_configs += [
        (f'D_alpha{a:02d}', 'adaptive', 'source_plus_hard', 'adaptive_streaming',
         True, True, 'full', 0.3, None)
        for a in _ALPHA_SWEEP
    ]

    # NOTE: per-group config overrides and disambiguation suffixes live in
    # the module-level _group_overrides / _group_suffixes dicts below the
    # function (they are looked up by group label inside the loop).

    # Integrated 9-group batch (all drift-triggered via adaptive_streaming):
    #   replay + bank : B_random_replay(10%) / B_random_20_replay / B_random_30_replay /
    #                   C_conf_replay / nlr03_only_replay / D_nlr03_replay / I_full_upload_replay
    #   ablations     : D_nlr03_nobank (no replay, no bank) / D_nlr03_replay_nobank (replay, no bank)

    # -- Optional group filter ------------------------------------------
    if groups:
        ablation_configs = [c for c in ablation_configs if c[0] in set(groups)]
        if not ablation_configs:
            raise ValueError(f'No matching groups for --groups {groups}')
        print(f'[ablation] filtered to {len(ablation_configs)} groups: '
              f'{[c[0] for c in ablation_configs]}')

    do_dedup = dedup_iou_threshold is not None and dedup_iou_threshold > 0.0
    if do_dedup:
        dedup_tag = f'_dedup{dedup_iou_threshold:.2f}'.rstrip('0').rstrip('.')
    else:
        dedup_tag = ''

    if seeds is None:
        seeds = [42]
    print(f'[ablation] seeds={seeds} ({len(seeds)} run(s) per group)')

    frames = []
    for seed_val in seeds:
        for group_label, method, mix_mode, sel_mode, use_entropy, mb_enabled, dep_mode, nlr_threshold, use_conf in ablation_configs:
            dedup_val = dedup_iou_threshold if do_dedup else 0.0
            print(f'\n{"="*60}')
            print(f'Ablation [{group_label}] seed={seed_val}: method={method}, mix_mode={mix_mode}', end='')
            if sel_mode:
                print(f', selection_mode={sel_mode}', end='')
            print(f', use_entropy={use_entropy}', end='')
            if nlr_threshold is not None and nlr_threshold > 0:
                print(f', non_lie_ratio>{nlr_threshold}', end='')
            if use_conf is not None and not use_conf:
                print(f', use_conf={use_conf}', end='')
            if mb_enabled is not None:
                print(f', model_bank={mb_enabled}', end='')
            if dep_mode:
                print(f', deploy={dep_mode}', end='')
            if dedup_val > 0:
                print(f', dedup_iou={dedup_val:.2f}', end='')
            print(f'\n{"="*60}')
            # B_random_*: random sampling ratio within drift windows
            _random_ratios = {
                'B_random_20': 0.20, 'B_random_30': 0.30,
                'B_random_20_replay': 0.20, 'B_random_30_replay': 0.30,
                'B_random_40_replay': 0.40,
                # V9×V1 对比组：漂移检测阶段随机 0.2/0.3/0.4（窗内随机采样比例）
                'D_gate_random02_v1sel': 0.20,
                'D_gate_random03_v1sel': 0.30,
                'D_gate_random04_v1sel': 0.40,
            }
            group_ratio = _random_ratios.get(group_label, ratio)
            group_override = _group_overrides.get(group_label)
            if group_override:
                print(f'[ablation] config overrides for {group_label}: {group_override}')
            csv_path = run_one_method(config_path, method, group_ratio, mix_mode,
                                      dedup_iou_threshold=dedup_val if dedup_val > 0 else None,
                                      scores_cache_dir=scores_cache_dir,
                                      selection_mode=sel_mode,
                                      use_entropy=use_entropy,
                                      model_bank_enabled=mb_enabled,
                                      deployment_mode=dep_mode,
                                      seed=seed_val,
                                      non_lie_ratio_threshold=nlr_threshold,
                                      use_conf=use_conf,
                                      config_overrides=group_override,
                                      extra_suffix=_group_suffixes.get(group_label))
            df = pd.read_csv(csv_path)
            df['ablation_group'] = group_label
            df['seed'] = seed_val
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    seed_tag = f'_seeds{"-".join(str(s) for s in seeds)}'
    out = exp.paths.work_dir / f'ablation_results{dedup_tag}{seed_tag}.csv'
    all_df.to_csv(out, index=False, encoding='utf-8-sig')

    reports_dir = exp.paths.work_dir / f'ablation_reports{dedup_tag}{seed_tag}'
    aggregate_results(out, reports_dir)

    if len(seeds) > 1:
        metric_cols = ['mAP50', 'mAP50_95', 'Precision', 'Recall',
                       'behavior_acc_iou50', 'behavior_recall_iou50']
        available = [c for c in metric_cols if c in all_df.columns]
        if available:
            per_seed = all_df.groupby(['ablation_group', 'seed'])[available].mean().reset_index()
            summary = per_seed.groupby('ablation_group')[available].agg(['mean', 'std']).round(4)
            summary.columns = [f'{col}_{stat}' for col, stat in summary.columns]
            summary.to_csv(reports_dir / 'seed_summary_mean_std.csv', encoding='utf-8-sig')
            print(f'\nMulti-seed summary (mean +/- std across {len(seeds)} seeds):')
            print(summary.to_string())

    print(f'\nAblation results saved: {out}')
    print(f'Ablation reports saved: {reports_dir}')


# -- BHSS weight sensitivity configurations ---------------------------
BHSS_WEIGHT_CONFIGS: list[dict] = [
    # Single-component (prove each contributes, but none alone suffices)
    {'name': 'conf_only',     'weights': {'conf_drop': 1.00, 'count_drift': 0.00, 'behavior_drift': 0.00}},
    {'name': 'count_only',    'weights': {'conf_drop': 0.00, 'count_drift': 1.00, 'behavior_drift': 0.00}},
    {'name': 'behavior_only', 'weights': {'conf_drop': 0.00, 'count_drift': 0.00, 'behavior_drift': 1.00}},
    # Two-component (prove 3 > 2)
    {'name': 'no_behavior',   'weights': {'conf_drop': 0.50, 'count_drift': 0.50, 'behavior_drift': 0.00}},
    {'name': 'no_count',      'weights': {'conf_drop': 0.50, 'count_drift': 0.00, 'behavior_drift': 0.50}},
    {'name': 'no_conf',       'weights': {'conf_drop': 0.00, 'count_drift': 0.50, 'behavior_drift': 0.50}},
    # Three-component (prove 0.50/0.25/0.25 is optimal)
    {'name': 'ours',          'weights': {'conf_drop': 0.50, 'count_drift': 0.25, 'behavior_drift': 0.25}},
    {'name': 'equal',         'weights': {'conf_drop': 0.33, 'count_drift': 0.33, 'behavior_drift': 0.34}},
    {'name': 'conf_heavy',    'weights': {'conf_drop': 0.70, 'count_drift': 0.15, 'behavior_drift': 0.15}},
    {'name': 'balanced',      'weights': {'conf_drop': 0.40, 'count_drift': 0.30, 'behavior_drift': 0.30}},
]


def run_bhss_weight_sweep(config_path: str | Path, upload_ratio: float | None = None, mix_mode: str | None = None, scores_cache_dir: str | None = None) -> None:
    """BHSS weight sensitivity experiment.

    Runs the same continual-learning pipeline with different BHSS fusion
    weights -- recombining the precomputed ``conf_drop``, ``count_drift``,
    and ``behavior_drift`` components **without re-running inference**.

    Produces a weight-sensitivity summary comparing each configuration's
    average mAP50 at a fixed upload ratio.
    """
    exp = Experiment.from_yaml(config_path)
    default_mix_mode = str(exp.raw.get('data_mix', {}).get('mix_mode', 'source_plus_hard'))
    mix_mode = normalize_mix_mode(mix_mode, default_mix_mode)
    if upload_ratio is None:
        upload_ratio = float(exp.raw['sampling'].get('upload_ratio', 0.20))

    print(f'[weight_sweep] {len(BHSS_WEIGHT_CONFIGS)} weight configs x {len(exp.stage_names())} stages')
    print(f'[weight_sweep] upload_ratio={upload_ratio}, mix_mode={mix_mode}')
    for cfg in BHSS_WEIGHT_CONFIGS:
        w = cfg['weights']
        print(f'  {cfg["name"]:20s}  conf={w["conf_drop"]:.2f}  count={w["count_drift"]:.2f}  behav={w["behavior_drift"]:.2f}')

    frames: list[pd.DataFrame] = []
    for cfg in BHSS_WEIGHT_CONFIGS:
        print(f'\n{"="*60}')
        print(f'[weight_sweep] {cfg["name"]}: {cfg["weights"]}')
        print(f'{"="*60}')
        csv_path = run_one_method(
            config_path, 'ours', upload_ratio, mix_mode,
            scores_cache_dir=scores_cache_dir,
            bhss_weights=cfg['weights'],
        )
        df = pd.read_csv(csv_path)
        df['weight_config'] = cfg['name']
        df['w_conf_drop'] = cfg['weights']['conf_drop']
        df['w_count_drift'] = cfg['weights']['count_drift']
        df['w_behavior_drift'] = cfg['weights']['behavior_drift']
        frames.append(df)

    out = exp.paths.work_dir / 'bhss_weight_sweep.csv'
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out, index=False, encoding='utf-8-sig')

    reports_dir = exp.paths.work_dir / 'bhss_weight_sweep_reports'
    aggregate_results(out, reports_dir)

    # -- Weight-sensitivity summary ---------------------------------
    _summarize_weight_sweep(all_df, reports_dir)

    print(f'\n[weight_sweep] results: {out}')
    print(f'[weight_sweep] reports: {reports_dir}')


def _summarize_weight_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    """Create weight-sensitivity pivot tables and ranking."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty or 'weight_config' not in df.columns:
        return

    metrics = ['mAP50', 'mAP50_95', 'Precision', 'Recall', 'behavior_acc_iou50', 'behavior_recall_iou50']
    available = [m for m in metrics if m in df.columns]

    # Per-weight-config summary (mean across stages)
    agg = {'uploaded_images': 'sum', 'train_seconds': 'sum'}
    for m in available:
        agg[m] = 'mean'
    summary = df.groupby(['weight_config', 'w_conf_drop', 'w_count_drift', 'w_behavior_drift']).agg(agg).reset_index()

    # Sort by mAP50 descending
    if 'mAP50' in summary.columns:
        summary = summary.sort_values('mAP50', ascending=False)

    summary.to_csv(out_dir / 'weight_sweep_summary.csv', index=False, encoding='utf-8-sig')

    # -- Ranking table (markdown-friendly) --------------------------
    rank_cols = ['weight_config', 'w_conf_drop', 'w_count_drift', 'w_behavior_drift']
    rank_cols += [f'{m}' for m in available]
    rank_cols += ['uploaded_images', 'train_seconds']
    rank_cols_exist = [c for c in rank_cols if c in summary.columns]
    ranking = summary[rank_cols_exist].copy()
    if 'mAP50' in ranking.columns:
        ranking.insert(0, 'rank', range(1, len(ranking) + 1))
    ranking.to_csv(out_dir / 'weight_sweep_ranking.csv', index=False, encoding='utf-8-sig')

    # -- Stage-wise pivot -------------------------------------------
    if 'stage' in df.columns and 'mAP50' in df.columns:
        for m in available:
            pivot = df.pivot_table(index='weight_config', columns='stage', values=m, aggfunc='last')
            pivot.to_csv(out_dir / f'weight_sweep_{m}_by_stage.csv', encoding='utf-8-sig')

    import matplotlib.pyplot as plt
    if 'mAP50' in summary.columns:
        # Bar chart: mAP50 by weight config
        labels = summary['weight_config'].tolist()
        values = summary['mAP50'].tolist()
        ours_idx = labels.index('ours') if 'ours' in labels else None

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['#2ca02c' if l == 'ours' else '#1f77b4' for l in labels]
        bars = ax.bar(range(len(labels)), values, color=colors)
        if ours_idx is not None:
            ax.axhline(y=values[ours_idx], color='#2ca02c', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Mean mAP50')
        ax.set_title('BHSS Weight Sensitivity -- mAP50 by weight configuration')
        for i, v in enumerate(values):
            ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / 'weight_sweep_map50.png', dpi=200)
        plt.close()

    print(f'[weight_sweep] summary saved to {out_dir}')


def run_precompute(config_path: str | Path) -> Path:
    """Phase 1: Precompute BHSS scores for all stages.

    Stream-processes each stage's videos through YOLO + BHSS scoring,
    saves frames and scores CSV to the configured ``scores_cache`` directory.
    Subsequent experiments (ratio-sweep, ablation, etc.) can reuse these
    cached scores via ``--scores-cache``, avoiding repeated inference.
    """
    exp = Experiment.from_yaml(config_path)
    raw = exp.raw
    set_seed(int(raw.get('seed', 42)))

    scores_cache_root = exp.paths.scores_cache
    if scores_cache_root is None:
        scores_cache_root = exp.paths.work_dir / 'scores_cache'
    scores_cache_root.mkdir(parents=True, exist_ok=True)

    imgsz = int(raw['models']['imgsz'])
    sampling_cfg = dict(raw['sampling'])
    sampling_cfg['num_classes'] = len(get_behavior_names(exp))
    student_path = exp.paths.initial_student

    print(f'[precompute] student model: {student_path}')
    print(f'[precompute] scores cache root: {scores_cache_root}')
    print(f'[precompute] stages: {exp.stage_names()}')

    for stage_index, stage in enumerate(exp.stage_names(), start=1):
        videos_dir = exp.stage_videos_dir(stage)
        if not videos_dir.exists() or not any(videos_dir.glob('*')):
            print(f'[precompute] SKIP stage {stage_index}/{len(exp.stage_names())}: {stage} (no videos)')
            continue

        frame_cache_dir = scores_cache_root / stage / 'frames'
        scores_csv = scores_cache_root / f'{stage}_scores.csv'

        print(f'\n[precompute] stage {stage_index}/{len(exp.stage_names())}: {stage}')
        print(f'  videos: {videos_dir}')
        print(f'  frame cache: {frame_cache_dir}')
        print(f'  scores CSV: {scores_csv}')

        t0 = time.time()
        precompute_stage_scores(
            student_path=student_path,
            videos_dir=videos_dir,
            frame_cache_dir=frame_cache_dir,
            scores_csv_path=scores_csv,
            imgsz=imgsz,
            cfg=sampling_cfg,
            seed=int(raw.get('seed', 42)) + stage_index,
        )
        elapsed = time.time() - t0
        print(f'[precompute] stage {stage} done in {elapsed:.1f}s')

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f'\n[precompute] ALL DONE -> {scores_cache_root}')
    return scores_cache_root


# -- Activity ratio sweep ------------------------------------------------


def run_activity_ratio_sweep(
    config_path: str | Path,
    mix_mode: str | None = None,
    scores_cache_dir: str | None = None,
) -> Path:
    """Sweep upload ratios with the ``activity`` (non-lie ratio) strategy."""
    exp = Experiment.from_yaml(config_path)
    raw = exp.raw
    ratios = raw.get('compare', {}).get('upload_ratios', [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80])

    default_mix = str(raw.get('data_mix', {}).get('mix_mode', 'source_plus_hard'))
    mix_mode = normalize_mix_mode(mix_mode, default_mix)

    print(f'[activity_ratio_sweep] {len(ratios)} ratios x activity: {ratios}')

    frames: list[pd.DataFrame] = []
    for ratio in ratios:
        print(f'\n{"="*60}')
        print(f'[activity_ratio_sweep] upload_ratio={ratio:.2f}')
        csv_path = run_one_method(
            config_path, 'activity', ratio, mix_mode,
            scores_cache_dir=scores_cache_dir,
        )
        df = pd.read_csv(csv_path)
        df['sweep_ratio'] = ratio
        frames.append(df)

    out = exp.paths.work_dir / 'activity_ratio_sweep.csv'
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out, index=False, encoding='utf-8-sig')
    reports_dir = exp.paths.work_dir / 'activity_ratio_sweep_reports'
    aggregate_results(out, reports_dir)
    print(f'\n[activity_ratio_sweep] done -> {out}')
    return out


# -- Deployment comparison (full vs partial) -----------------------------


def run_deploy_compare(
    config_path: str | Path,
    upload_ratio: float | None = None,
    mix_mode: str | None = None,
    scores_cache_dir: str | None = None,
) -> Path:
    """Compare full vs partial (head-only) model deployment.

    Runs the same continual-learning pipeline twice -- once with full model
    deployment (default) and once with partial head-only deployment.
    Reports parameter count, file size, and mAP50 impact.
    """
    from .yolo_ops import measure_deployment_stats, apply_head_weights

    exp = Experiment.from_yaml(config_path)
    raw = exp.raw

    if upload_ratio is None:
        upload_ratio = float(raw['sampling'].get('upload_ratio', 0.20))
    default_mix = str(raw.get('data_mix', {}).get('mix_mode', 'source_plus_hard'))
    mix_mode = normalize_mix_mode(mix_mode, default_mix)

    rows: list[dict] = []
    reports_dir = exp.paths.work_dir / 'deploy_compare_reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    for deploy_mode in ('full', 'partial'):
        label = f'{deploy_mode}_deploy'
        print(f'\n{"="*60}')
        print(f'[deploy_compare] {label}')
        print(f'{"="*60}')

        if deploy_mode == 'full':
            csv_path = run_one_method(
                config_path, 'activity', upload_ratio, mix_mode,
                scores_cache_dir=scores_cache_dir,
            )
        else:
            # Partial: run the full pipeline, but the deployment step
            # measures head-only stats
            csv_path = run_one_method(
                config_path, 'activity', upload_ratio, mix_mode,
                scores_cache_dir=scores_cache_dir,
            )

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Per-stage deployment stats
        for _, row in df.iterrows():
            model_path = Path(str(row.get('model_path', '')))
            if model_path.exists():
                stats = measure_deployment_stats(
                    model_path, exp.paths.initial_student,
                )
            else:
                stats = {}
            item = row.to_dict()
            item['deploy_mode'] = deploy_mode
            item.update(stats)
            rows.append(item)

    out = exp.paths.work_dir / 'deploy_compare.csv'
    all_df = pd.DataFrame(rows)
    all_df.to_csv(out, index=False, encoding='utf-8-sig')

    # -- Summary plots --------------------------------------------
    import matplotlib.pyplot as plt

    if not all_df.empty and 'deploy_mode' in all_df.columns:
        # Parameter count bar
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) mAP50 comparison
        ax = axes[0]
        for mode, g in all_df.groupby('deploy_mode'):
            if 'mAP50' in g.columns:
                ax.plot(g['stage_index'], g['mAP50'], marker='o', label=mode)
        ax.set_xlabel('Stage')
        ax.set_ylabel('mAP50')
        ax.set_title('Full vs Partial Deployment: mAP50')
        ax.legend()

        # (b) Parameter count
        ax = axes[1]
        for mode, g in all_df.groupby('deploy_mode'):
            if 'full_params' in g.columns:
                vals = g['full_params'] if mode == 'full' else g['head_params']
                ax.bar(mode, vals.mean() / 1e6)
        ax.set_ylabel('Parameters (M)')
        ax.set_title('Deployment Parameter Count')

        # (c) Deployment size
        ax = axes[2]
        for mode, g in all_df.groupby('deploy_mode'):
            col = 'full_size_mb' if mode == 'full' else 'head_size_mb'
            if col in g.columns:
                ax.bar(mode, g[col].mean())
        ax.set_ylabel('Size (MB)')
        ax.set_title('Deployment File Size')

        plt.tight_layout()
        plt.savefig(reports_dir / 'deploy_comparison.png', dpi=200)
        plt.close()

    print(f'\n[deploy_compare] done -> {out}')
    return out


def run_trigger_compare(
    config_path: str | Path,
    upload_ratio: float | None = None,
    mix_mode: str | None = None,
    scores_cache_dir: str | None = None,
) -> Path:
    """Compare stage-based training vs adaptive-trigger-based training.

    Runs the activity pipeline twice -- once training every stage (baseline)
    and once only training when the adaptive trigger fires (c_t > c[U+0304]).
    Reports mAP50, number of training events, and compute savings.
    """
    exp = Experiment.from_yaml(config_path)
    raw = exp.raw

    if upload_ratio is None:
        upload_ratio = float(raw['sampling'].get('upload_ratio', 0.20))
    default_mix = str(raw.get('data_mix', {}).get('mix_mode', 'source_plus_hard'))
    mix_mode = normalize_mix_mode(mix_mode, default_mix)

    rows: list[dict] = []
    reports_dir = exp.paths.work_dir / 'trigger_compare_reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -- Group 1: stage-based (train every stage, no trigger) --------
    print(f'\n{"="*60}')
    print('[trigger_compare] Group 1: stage-based (train every stage)')
    print(f'{"="*60}')
    csv_stage = run_one_method(
        config_path, 'activity', upload_ratio, mix_mode,
        scores_cache_dir=scores_cache_dir,
        trigger_based=False,
    )
    df_stage = pd.read_csv(csv_stage)
    if not df_stage.empty:
        df_stage['trigger_mode'] = 'stage_based'
        rows.append(df_stage)

    # -- Group 2: trigger-based (train only when c_t > c[U+0304]) -----------
    print(f'\n{"="*60}')
    print('[trigger_compare] Group 2: trigger-based (adaptive trigger)')
    print(f'{"="*60}')
    csv_trigger = run_one_method(
        config_path, 'activity', upload_ratio, mix_mode,
        scores_cache_dir=scores_cache_dir,
        trigger_based=True,
    )
    df_trigger = pd.read_csv(csv_trigger)
    if not df_trigger.empty:
        df_trigger['trigger_mode'] = 'trigger_based'
        rows.append(df_trigger)

    # -- Concatenate & save -----------------------------------------
    out = exp.paths.work_dir / 'trigger_compare.csv'
    all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    all_df.to_csv(out, index=False, encoding='utf-8-sig')

    # -- Summary plots ----------------------------------------------
    if not all_df.empty and 'trigger_mode' in all_df.columns:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) mAP50 over stages
        ax = axes[0]
        for mode, g in all_df.groupby('trigger_mode'):
            if 'stage_index' in g.columns and 'mAP50' in g.columns:
                g = g.sort_values('stage_index')
                style = '-' if mode == 'trigger_based' else '--'
                ax.plot(g['stage_index'], g['mAP50'], marker='o', linestyle=style,
                        linewidth=2, label=f'{mode}')
        ax.set_xlabel('Stage')
        ax.set_ylabel('mAP50')
        ax.set_title('Stage-based vs Adaptive Trigger: mAP50')
        ax.legend()

        # (b) Training events per stage (accepted_update)
        ax = axes[1]
        for mode, g in all_df.groupby('trigger_mode'):
            if 'stage_index' in g.columns:
                g = g.sort_values('stage_index')
                trigger_count = g.get('triggered', g.get('accepted_update'))
                if not trigger_count.empty:
                    ax.plot(g['stage_index'], trigger_count, marker='s',
                            linewidth=2, label=f'{mode}')
        ax.set_xlabel('Stage')
        ax.set_ylabel('Training Events')
        ax.set_title('Training Triggers per Stage')
        ax.legend()

        # (c) Cumulative training time
        ax = axes[2]
        for mode, g in all_df.groupby('trigger_mode'):
            if 'stage_index' in g.columns and 'train_seconds' in g.columns:
                g = g.sort_values('stage_index')
                cum_time = g['train_seconds'].cumsum()
                ax.plot(g['stage_index'], cum_time / 3600, marker='^',
                        linewidth=2, label=f'{mode}')
        ax.set_xlabel('Stage')
        ax.set_ylabel('Cumulative Train Time (h)')
        ax.set_title('Cumulative Training Time')
        ax.legend()

        plt.tight_layout()
        plt.savefig(reports_dir / 'trigger_comparison.png', dpi=200)
        plt.close()

        # -- Summary CSV --------------------------------------------
        summary_rows = []
        for mode, g in all_df.groupby('trigger_mode'):
            info: dict = {'trigger_mode': mode}
            metric_cols = ['mAP50', 'mAP50_95', 'Precision', 'Recall',
                           'behavior_acc_iou50', 'behavior_recall_iou50']
            for m in metric_cols:
                if m in g.columns:
                    info[f'mean_{m}'] = g[m].mean()
            count_cols = ['uploaded_images', 'train_seconds', 'accepted_update']
            for c in count_cols:
                if c in g.columns:
                    info[f'total_{c}'] = g[c].sum()
            # Count triggered stages
            if 'triggered' in g.columns:
                info['triggered_stages'] = int(g['triggered'].sum())
            elif 'accepted_update' in g.columns:
                info['triggered_stages'] = int((g['accepted_update'] > 0).sum())
            summary_rows.append(info)
        if summary_rows:
            save_csv(reports_dir / 'trigger_compare_summary.csv', summary_rows)

    print(f'\n[trigger_compare] done -> {out}')
    return out


# -- Adaptive drift detection comparison ----------------------------------


def run_adaptive_compare(
    config_path: str | Path,
    upload_ratio: float | None = None,
    mix_mode: str | None = None,
    scores_cache_dir: str | None = None,
) -> Path:
    """Compare adaptive drift detection vs baselines.

    Runs the same continual-learning pipeline with five strategies:

      A: adaptive   -- ConfidenceDriftDetector (binomial test on conf drops)
      B: activity   -- non_lie_ratio ranking (baseline 1)
      C: ours       -- BHSS fixed-weight scoring (baseline 2)
      D: random     -- random selection (lower bound)
      E: full_upload -- all frames (upper bound)

    The adaptive strategy uses ``selection_mode='adaptive_streaming'``
    (per-frame statistical drift detection), while B--D use the standard
    ``online_blocks`` block-based pipeline.

    Key comparisons:
      A vs B: confidence-drift detection vs pure activity signal
      A vs C: data-driven statistical test vs fixed-weight BHSS
      A vs D: selection effectiveness
      A vs E: efficiency near the upper bound
    """
    exp = Experiment.from_yaml(config_path)
    raw = exp.raw

    if upload_ratio is None:
        upload_ratio = float(raw['sampling'].get('upload_ratio', 0.20))
    default_mix = str(raw.get('data_mix', {}).get('mix_mode', 'source_plus_hard'))
    mix_mode = normalize_mix_mode(mix_mode, default_mix)

    # -- Configs -----------------------------------------------------
    # adaptive uses adaptive_streaming (drift-based); others use online_blocks
    compare_configs = [
        ('adaptive',      'adaptive',      'adaptive_streaming'),
        ('activity',      'activity',      None),   # None = use config default
        ('ours',          'ours',          None),
        ('random',        'random',        None),
        ('full_upload',   'full_upload',   None),
    ]

    frames: list[pd.DataFrame] = []
    for label, method, sel_mode in compare_configs:
        print(f'\n{"="*60}')
        extra = f', selection_mode={sel_mode}' if sel_mode else ''
        print(f'[adaptive_compare] {label}: method={method}{extra}')
        print(f'{"="*60}')
        csv_path = run_one_method(
            config_path, method, upload_ratio, mix_mode,
            scores_cache_dir=scores_cache_dir,
            selection_mode=sel_mode,
        )
        df = pd.read_csv(csv_path)
        df['compare_group'] = label
        frames.append(df)

    out = exp.paths.work_dir / 'adaptive_compare.csv'
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out, index=False, encoding='utf-8-sig')
    reports_dir = exp.paths.work_dir / 'adaptive_compare_reports'
    aggregate_results(out, reports_dir)
    print(f'\n[adaptive_compare] done -> {out}')
    return out


# ==========================================================================
#  V8: One-shot batch upload ablation
# ==========================================================================
#  Tests whether dual-signal frame selection (conf < P_10 AND H > H_med)
#  is inherently better than single-confidence selection, by eliminating
#  continual-learning dynamics.  All drift-detected frames from all stages
#  are pooled, selection strategies are applied once, and a single model
#  is trained and evaluated — removing sequential effects (forgetting,
#  model-bank fallback cascades, rejection loops).


def _evaluate_on_all_stages(
    model_path: Path,
    exp: Experiment,
    names: dict[int, str],
) -> dict[str, float]:
    """Evaluate a model on all 12 stage test sets and return aggregated metrics.

    Each stage has a ``test_labeled/`` directory with 50 hand-labeled images.
    Metrics are averaged across stages (macro-average), and per-class APs
    are also averaged.
    """
    imgsz = int(exp.raw['models']['imgsz'])
    device = exp.raw['models'].get('device', 'auto')
    per_class_keys = ['AP50_95_stand', 'AP50_95_lie', 'AP50_95_eat']
    metrics_keys = ['mAP50', 'mAP50_95', 'Precision', 'Recall',
                    'behavior_acc_iou50', 'behavior_recall_iou50']
    all_keys = metrics_keys + per_class_keys + ['eval_seconds', 'num_eval_images']

    accum: dict[str, float] = {k: 0.0 for k in all_keys}
    n_stages = 0

    for stage in exp.stage_names():
        test_root = exp.stage_test_root(stage)
        test_images = test_root / 'images' / 'test'
        if not test_images.exists():
            continue
        met = evaluate_model(model_path, test_root, imgsz=imgsz, device=device, names=names)
        for k in all_keys:
            accum[k] += float(met.get(k, 0.0))
        n_stages += 1

    if n_stages == 0:
        return {k: 0.0 for k in all_keys}

    return {k: v / n_stages for k, v in accum.items()}


def run_one_shot_ablation(
    config_path: str | Path,
    seeds: list[int] | None = None,
) -> Path:
    """V8: One-shot batch upload ablation experiment.

    **Motivation**: V7 shows D_conf_ent < C_conf_p10 in continual learning.
    Is this because dual-signal selects worse frames, or because continual
    learning dynamics (model-bank fallback, forgetting, rejection cascades)
    hurt D more?  This experiment eliminates sequential effects by pooling
    all drift frames and training once.

    **Pipeline**::

        12 stages × source model → per-stage drift detection
            → global candidate pool (all drift frames)
            → 6 selection strategies
            → one-shot train (8:2 split) → evaluate on all stages

    **6 groups**:

        A_source_only   — no training  (lower bound)
        C_conf_p10      — conf < P_10  (single-signal ablation)
        D_conf_ent      — conf < P_10 AND H > H_med  (OUR METHOD)
        R_random_20     — random 20 % of pool
        R_random_30     — random 30 % of pool
        R_random_40     — random 40 % of pool
        I_full_upload   — all pool frames  (upper bound)
    """
    exp = Experiment.from_yaml(config_path)
    raw = exp.raw
    if seeds is None:
        seeds = [42]
    set_seed(int(raw.get('seed', 42)))

    imgsz = int(raw['models']['imgsz'])
    device = raw['models'].get('device', 'auto')
    names = get_behavior_names(exp)
    nc = len(names)
    sample_every = max(1, int(raw['sampling'].get('video_sample_every',
                       raw['sampling'].get('sample_every', 30))))
    adaptive_cfg = raw['sampling'].get('adaptive', {})
    long_window = int(adaptive_cfg.get('long_window', 300))
    short_window = int(adaptive_cfg.get('short_window', 20))
    p_low = float(adaptive_cfg.get('p_low', 0.10))
    alpha = float(adaptive_cfg.get('alpha', 0.05))

    # -- Output directory -------------------------------------------
    work_dir = exp.paths.work_dir / 'experiments' / 'one_shot_ablation'
    work_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    #  Phase 1: Build global candidate pool
    # ================================================================
    #  For each stage, run the SOURCE MODEL (never updated) with
    #  ConfidenceDriftDetector in full_upload mode to collect every
    #  frame within every detected drift window.
    #
    #  Each drift frame is tagged with:
    #    P_10_stage  — per-stage 10th percentile confidence baseline
    #    H_med_stage — per-stage median behavior entropy baseline
    #  These are used later for C_conf_p10 / D_conf_ent filtering.
    # ================================================================

    pool_dir = work_dir / 'candidate_pool'
    pool_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = pool_dir / 'all_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Use source model — never updated during pool construction
    source_model = exp.paths.initial_student

    pool_frames: list[pd.DataFrame] = []
    # Track largest P_10 across stages for optional global-threshold variant
    global_p10_values: dict[str, float] = {}
    global_hmed_values: dict[str, float] = {}

    for stage_idx, stage in enumerate(exp.stage_names()):
        videos_dir = exp.stage_videos_dir(stage)
        stream_type = exp.stage_stream_type(stage)
        if not videos_dir.exists() or stream_type != 'videos':
            print(f'[pool] SKIP {stage}: no videos')
            continue

        print(f'\n[pool] stage {stage_idx + 1}/{len(exp.stage_names())}: {stage}')
        video_paths = list_videos(videos_dir)
        print(f'  videos: {len(video_paths)}  long_window={long_window}  '
              f'short_window={short_window}')

        # -- Create per-stage drift detector -------------------------
        detector = ConfidenceDriftDetector(
            long_window=long_window,
            short_window=short_window,
            p_low=p_low,
            alpha=alpha,
        )

        # -- Stage frame cache & upload dir --------------------------
        stage_frame_cache = pool_dir / 'temp_frames' / stage
        stage_upload = pool_dir / 'temp_uploads' / stage / 'images'
        stage_frame_cache.mkdir(parents=True, exist_ok=True)
        stage_upload.mkdir(parents=True, exist_ok=True)

        # -- Build config override for full_upload streaming ---------
        stage_cfg = dict(raw['sampling'])
        stage_cfg['strategy'] = 'full_upload'
        stage_cfg['selection_mode'] = 'adaptive_streaming'
        stage_cfg['num_classes'] = nc
        stage_cfg['adaptive'] = dict(adaptive_cfg)
        stage_cfg['adaptive']['upload_min_interval_s'] = 0.0
        stage_cfg['adaptive']['min_upload_fallback'] = 0
        stage_cfg['adaptive']['drift_dedup_gap_frames'] = 150  # ~120s gap → kills 98% adjacent-frame redundancy

        try:
            scores, selected, _ = compute_hard_scores_from_videos_streaming(
                student_path=source_model,
                videos_dir=videos_dir,
                frame_cache_dir=stage_frame_cache,
                imgsz=imgsz,
                cfg=stage_cfg,
                upload_dir=stage_upload,
                seed=42,
                videos=video_paths,
            )
        except Exception:
            print(f'[pool] ERROR processing {stage}, skipping')
            import traceback
            traceback.print_exc()
            continue

        if selected is not None and not selected.empty:
            # -- Move frames to shared pool directory -----------------
            # (avoid scattered upload dirs; one canonical location)
            moved_paths: list[str] = []
            for _, row in selected.iterrows():
                src = Path(str(row['path']))
                if src.exists():
                    dst = frames_dir / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    moved_paths.append(str(dst))
                else:
                    moved_paths.append(str(row['path']))

            # Ensure P_10_stage / H_med_stage columns exist
            if 'P_10_stage' not in selected.columns:
                selected['P_10_stage'] = float(detector.P_10)
            else:
                selected['P_10_stage'] = selected['P_10_stage'].fillna(float(detector.P_10))
            if 'H_med_stage' not in selected.columns:
                selected['H_med_stage'] = float(detector.H_med)
            else:
                selected['H_med_stage'] = selected['H_med_stage'].fillna(float(detector.H_med))

            # Overwrite path column to point to shared pool dir
            selected = selected.copy()
            selected['path'] = moved_paths
            selected['stage'] = stage
            pool_frames.append(selected)

            global_p10_values[stage] = float(detector.P_10)
            global_hmed_values[stage] = float(detector.H_med)

            print(f'  drift frames: {len(selected)}  '
                  f'P_10={detector.P_10:.4f}  H_med={detector.H_med:.4f}')
        else:
            print(f'  WARN: no drift frames detected for {stage}')

        # -- Cleanup stage temp dirs ---------------------------------
        if stage_frame_cache.exists():
            shutil.rmtree(stage_frame_cache, ignore_errors=True)
        if stage_upload.exists():
            shutil.rmtree(stage_upload.parent, ignore_errors=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Remove temp dirs -------------------------------------------
    for _d in [pool_dir / 'temp_frames', pool_dir / 'temp_uploads']:
        if _d.exists():
            shutil.rmtree(_d, ignore_errors=True)

    if not pool_frames:
        raise RuntimeError('No drift frames collected — candidate pool is empty. '
                           'Check that videos exist and source model is valid.')

    candidate_pool = pd.concat(pool_frames, ignore_index=True)
    candidate_pool.to_csv(work_dir / 'candidate_pool.csv',
                          index=False, encoding='utf-8-sig')

    n_pool = len(candidate_pool)
    print(f'\n[pool] GLOBAL CANDIDATE POOL: {n_pool} frames '
          f'across {len(pool_frames)} stages')
    print(f'  P_10 range: {min(global_p10_values.values()):.4f} – '
          f'{max(global_p10_values.values()):.4f}')
    print(f'  H_med range: {min(global_hmed_values.values()):.4f} – '
          f'{max(global_hmed_values.values()):.4f}')

    # ================================================================
    #  Phase 2: Apply selection strategies
    # ================================================================

    selections_dir = work_dir / 'selections'
    selections_dir.mkdir(parents=True, exist_ok=True)

    # -- P_10 / H_med fallback: fill zero values with stage-wise percentiles --
    print('\n[pool] Applying P_10 / H_med fallback for zero-value frames ...')
    n_p10_fixed = 0
    n_hmed_fixed = 0
    for stage_name in candidate_pool['stage'].unique():
        stage_mask = candidate_pool['stage'] == stage_name
        stage_data = candidate_pool[stage_mask]
        # P_10 fallback
        p10_zero = stage_mask & (candidate_pool['P_10_stage'] == 0)
        if p10_zero.any():
            fallback_p10 = float(stage_data['avg_conf'].quantile(0.10))
            candidate_pool.loc[p10_zero, 'P_10_stage'] = fallback_p10
            n_p10_fixed += int(p10_zero.sum())
        # H_med fallback
        hmed_zero = stage_mask & (candidate_pool['H_med_stage'] == 0)
        if hmed_zero.any():
            fallback_hmed = float(stage_data['behavior_entropy'].median())
            candidate_pool.loc[hmed_zero, 'H_med_stage'] = fallback_hmed
            n_hmed_fixed += int(hmed_zero.sum())
    print(f'[pool] P_10 fixed: {n_p10_fixed} frames, H_med fixed: {n_hmed_fixed} frames')

    # All pool frame paths (verified exist)
    all_paths: list[Path] = []
    for p_str in candidate_pool['path']:
        p = Path(p_str)
        if p.exists():
            all_paths.append(p)

    # Ensure P_10_stage / H_med_stage columns exist
    has_p10 = 'P_10_stage' in candidate_pool.columns
    has_hmed = 'H_med_stage' in candidate_pool.columns
    has_nlr = 'non_lie_ratio' in candidate_pool.columns

    groups: dict[str, list[Path]] = {}

    # --- C_conf_p10: conf < P_10 (single signal, confidence-only) ----
    if has_p10:
        mask_c = candidate_pool['avg_conf'] < candidate_pool['P_10_stage']
        groups['C_conf_p10'] = [Path(p) for p, m in
                                zip(candidate_pool['path'], mask_c)
                                if m and Path(p).exists()]
    else:
        global_p10 = float(np.median(list(global_p10_values.values())))
        groups['C_conf_p10'] = [p for p in all_paths
                                if p.stat().st_size > 0]
        print('[WARN] P_10_stage column missing — using global fallback')

    # --- D_entropy: conf < P_10 AND H > H_med (original dual-signal) --
    if has_p10 and has_hmed:
        mask_de = ((candidate_pool['avg_conf'] < candidate_pool['P_10_stage']) &
                   (candidate_pool['behavior_entropy'] > candidate_pool['H_med_stage']))
        groups['D_entropy'] = [Path(p) for p, m in
                               zip(candidate_pool['path'], mask_de)
                               if m and Path(p).exists()]
    else:
        print('[WARN] Cannot apply entropy dual-signal — missing P_10/H_med columns')

    # --- D_nlr03: conf < P_10 AND non_lie_ratio > 0.3 (NEW) ----------
    if has_p10 and has_nlr:
        mask_nlr03 = ((candidate_pool['avg_conf'] < candidate_pool['P_10_stage']) &
                      (candidate_pool['non_lie_ratio'] > 0.3))
        groups['D_nlr03'] = [Path(p) for p, m in
                             zip(candidate_pool['path'], mask_nlr03)
                             if m and Path(p).exists()]
    else:
        print('[WARN] Cannot apply nlr03 dual-signal — missing P_10/non_lie_ratio columns')

    # --- D_nlr04: conf < P_10 AND non_lie_ratio > 0.4 (NEW) ----------
    if has_p10 and has_nlr:
        mask_nlr04 = ((candidate_pool['avg_conf'] < candidate_pool['P_10_stage']) &
                      (candidate_pool['non_lie_ratio'] > 0.4))
        groups['D_nlr04'] = [Path(p) for p, m in
                             zip(candidate_pool['path'], mask_nlr04)
                             if m and Path(p).exists()]
    else:
        print('[WARN] Cannot apply nlr04 dual-signal — missing P_10/non_lie_ratio columns')

    # --- Random X% ------------------------------------------------
    import random as _random
    rng = _random.Random(42)
    for frac, label in [(0.20, 'R_random_20'), (0.30, 'R_random_30'),
                         (0.40, 'R_random_40')]:
        n = max(1, int(len(all_paths) * frac))
        groups[label] = rng.sample(all_paths, min(n, len(all_paths)))

    # --- Full upload (upper bound) ----------------------------------
    groups['I_full_upload'] = list(all_paths)

    # Save & print
    for name, paths in groups.items():
        pd.DataFrame({'path': [str(p) for p in paths]}).to_csv(
            selections_dir / f'{name}_selected.csv',
            index=False, encoding='utf-8-sig')
        pct = len(paths) / max(1, n_pool) * 100
        total_mb = sum(p.stat().st_size for p in paths if p.exists()) / (1024 * 1024)
        print(f'[select] {name:18s}  {len(paths):5d} frames '
              f'({pct:5.1f}% of pool)  {total_mb:6.1f} MB')

    # ================================================================
    #  Phase 3: One-shot train + evaluate (per group, per seed)
    # ================================================================

    results: list[dict] = []
    models_dir = work_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = work_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 3a. Source only (no training)
    print('\n[one-shot] A_source_only: evaluating source model ...')
    src_metrics = _evaluate_on_all_stages(source_model, exp, names)
    src_row = {
        'ablation_group': 'A_source_only',
        'seed': int(raw.get('seed', 42)),
        'uploaded_images': 0,
        'upload_mb': 0.0,
        'train_seconds': 0.0,
        'pseudo_behavior_boxes': 0,
        'pseudo_empty_images': 0,
        'pseudo_avg_conf': 0.0,
        **src_metrics,
    }
    results.append(src_row)
    print(f'  mAP50={src_metrics["mAP50"]:.4f}  '
          f'mAP50_95={src_metrics["mAP50_95"]:.4f}')

    # 3b. Train each group with each seed
    for seed_val in seeds:
        set_seed(seed_val)
        for group_name, selected_paths in groups.items():
            n_frames = len(selected_paths)
            print(f'\n[one-shot] {group_name}  seed={seed_val}  '
                  f'frames={n_frames}')

            group_dir = work_dir / 'groups' / f'{group_name}_seed{seed_val}'
            group_dir.mkdir(parents=True, exist_ok=True)

            # -- Pseudo-label (teacher model) -------------------------
            pseudo_dir = group_dir / 'pseudo_labels'
            t0 = time.time()
            pseudo_info = pseudo_label_with_teacher(
                exp.paths.teacher,
                selected_paths,
                pseudo_dir,
                imgsz=imgsz,
                conf_thr=float(raw['models'].get('pseudo_conf', 0.40)),
                min_box_area_ratio=float(raw['models'].get('min_box_area_ratio', 0.00015)),
                nc=nc,
                dedup_iou_threshold=float(raw['models'].get('pseudo_dedup_iou', 0.0)),
            )
            pseudo_seconds = time.time() - t0
            print(f'  pseudo: {pseudo_info["pseudo_behavior_boxes"]} boxes, '
                  f'{pseudo_info["pseudo_empty_images"]} empty, '
                  f'avg_conf={pseudo_info["pseudo_avg_conf"]:.3f}, '
                  f'{pseudo_seconds:.1f}s')

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # -- 8:2 stratified split (by stage if available) ---------
            val_frac = float(raw['data_mix'].get('target_val_fraction', 0.20))
            min_val = int(raw['data_mix'].get('min_target_val_images', 5))

            # Collect (path, stage) pairs
            pool_stage_map: dict[str, str] = {}
            if 'stage' in candidate_pool.columns and 'path' in candidate_pool.columns:
                for _, row in candidate_pool.iterrows():
                    pool_stage_map[str(row['path'])] = str(row['stage'])

            # Group selected paths by stage for stratified split
            stage_to_paths: dict[str, list[Path]] = {}
            for p in selected_paths:
                s = pool_stage_map.get(str(p), 'unknown')
                stage_to_paths.setdefault(s, []).append(p)

            train_paths: list[Path] = []
            val_paths: list[Path] = []
            rng_split = _random.Random(seed_val)
            for s, paths in stage_to_paths.items():
                if len(paths) < 3:
                    train_paths.extend(paths)
                    continue
                shuffled = list(paths)
                rng_split.shuffle(shuffled)
                n_val = max(1, int(len(shuffled) * val_frac))
                n_val = min(n_val, len(shuffled) - 1)
                val_paths.extend(shuffled[:n_val])
                train_paths.extend(shuffled[n_val:])

            # Ensure minimum validation set
            if len(val_paths) < min_val and len(train_paths) > min_val:
                overflow = train_paths[:min_val - len(val_paths)]
                val_paths.extend(overflow)
                train_paths = train_paths[min_val - len(val_paths):]

            print(f'  split: train={len(train_paths)}  val={len(val_paths)}')

            # -- Build YOLO dataset ----------------------------------
            data_dir = group_dir / 'dataset'
            data_dir.mkdir(parents=True, exist_ok=True)

            for split, paths in [('train', train_paths), ('val', val_paths)]:
                img_dir = data_dir / 'images' / split
                lbl_dir = data_dir / 'labels' / split
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                for p in paths:
                    # Copy image
                    dst_img = img_dir / p.name
                    if not dst_img.exists():
                        try:
                            shutil.copy2(p, dst_img)
                        except OSError:
                            continue
                    # Copy pseudo label
                    lbl_src = pseudo_dir / f'{p.stem}.txt'
                    if lbl_src.exists():
                        shutil.copy2(lbl_src, lbl_dir / f'{p.stem}.txt')

            write_dataset_yaml(data_dir, names)

            # -- Train ------------------------------------------------
            t0 = time.time()
            trained_model = train_student(
                source_model,
                data_dir / 'data.yaml',
                out_project=group_dir / 'train_runs',
                run_name='one_shot',
                imgsz=imgsz,
                epochs=int(raw['models'].get('student_epochs', 10)),
                batch=int(raw['models'].get('batch', 4)),
                workers=int(raw['models'].get('workers', 0)),
                device=device,
            )
            train_seconds = time.time() - t0

            # Save trained model
            model_dst = models_dir / f'{group_name}_seed{seed_val}.pt'
            shutil.copy2(trained_model, model_dst)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # -- Evaluate on all stages --------------------------------
            eval_metrics = _evaluate_on_all_stages(trained_model, exp, names)

            upload_bytes = sum(p.stat().st_size for p in selected_paths
                               if p.exists())

            row: dict[str, Any] = {
                'ablation_group': group_name,
                'seed': seed_val,
                'model_path': str(model_dst),
                'uploaded_images': n_frames,
                'upload_mb': upload_bytes / (1024 * 1024),
                'train_seconds': train_seconds,
                'pseudo_label_seconds': pseudo_seconds,
                **pseudo_info,
                **eval_metrics,
            }
            results.append(row)

            # Per-group result
            print(f'  TRAINED in {train_seconds:.0f}s  '
                  f'mAP50={eval_metrics["mAP50"]:.4f}  '
                  f'mAP50_95={eval_metrics["mAP50_95"]:.4f}')

            # Save incremental results
            pd.DataFrame(results).to_csv(
                reports_dir / 'one_shot_results.csv',
                index=False, encoding='utf-8-sig')

    # ================================================================
    #  Phase 4: Aggregate & save
    # ================================================================

    results_df = pd.DataFrame(results)
    results_df.to_csv(reports_dir / 'one_shot_results.csv',
                      index=False, encoding='utf-8-sig')

    # -- Summary by ablation group (mean across seeds) ---------------
    metric_cols = ['mAP50', 'mAP50_95', 'Precision', 'Recall',
                   'behavior_acc_iou50', 'behavior_recall_iou50',
                   'AP50_95_stand', 'AP50_95_lie', 'AP50_95_eat']
    count_cols = ['uploaded_images', 'upload_mb', 'train_seconds',
                  'pseudo_behavior_boxes']
    available_metrics = [c for c in metric_cols if c in results_df.columns]
    available_counts = [c for c in count_cols if c in results_df.columns]

    # Per-seed summary
    group_order = ['A_source_only', 'C_conf_p10', 'D_entropy',
                   'D_nlr03', 'D_nlr04',
                   'R_random_20', 'R_random_30', 'R_random_40',
                   'I_full_upload']
    agg_dict: dict[str, Any] = {}
    for c in available_metrics:
        agg_dict[c] = 'mean'
    for c in available_counts:
        agg_dict[c] = 'sum' if c != 'train_seconds' else 'mean'

    summary = results_df.groupby('ablation_group').agg(agg_dict).reset_index()

    # Sort by group order
    summary['_order'] = summary['ablation_group'].apply(
        lambda x: group_order.index(x) if x in group_order else 99)
    summary = summary.sort_values('_order').drop(columns=['_order'])

    summary.to_csv(reports_dir / 'summary_by_method.csv',
                   index=False, encoding='utf-8-sig')

    # -- Multi-seed stats (if > 1 seed) ------------------------------
    if len(seeds) > 1 and len(available_metrics) > 0:
        per_seed = results_df.groupby(
            ['ablation_group', 'seed'])[available_metrics].mean().reset_index()
        seed_stats = per_seed.groupby('ablation_group')[available_metrics].agg(
            ['mean', 'std']).round(4)
        seed_stats.columns = [f'{col}_{stat}' for col, stat in seed_stats.columns]
        seed_stats.to_csv(reports_dir / 'seed_summary_mean_std.csv',
                          encoding='utf-8-sig')
        print(f'\nMulti-seed summary (mean ± std across {len(seeds)} seeds):')
        print(seed_stats.to_string())

    # -- Key comparisons ---------------------------------------------
    print(f'\n{"="*60}')
    print('KEY COMPARISONS (mean mAP50 across seeds)')
    print(f'{"="*60}')

    def _get_map50(label: str) -> float:
        sub = summary[summary['ablation_group'] == label]
        if sub.empty or 'mAP50' not in sub.columns:
            return float('nan')
        return float(sub['mAP50'].iloc[0])

    comparisons = [
        ('D_entropy - C_conf_p10', 'D_entropy', 'C_conf_p10',
         'Entropy dual-signal vs single-signal'),
        ('D_nlr03 - C_conf_p10', 'D_nlr03', 'C_conf_p10',
         'NLR03 dual-signal vs single-signal'),
        ('D_nlr04 - C_conf_p10', 'D_nlr04', 'C_conf_p10',
         'NLR04 dual-signal vs single-signal'),
        ('D_nlr03 - D_entropy', 'D_nlr03', 'D_entropy',
         'NLR03 vs entropy: + = NLR better'),
        ('D_nlr04 - D_entropy', 'D_nlr04', 'D_entropy',
         'NLR04 vs entropy: + = NLR better'),
        ('D_nlr03 - R_random_30', 'D_nlr03', 'R_random_30',
         'Smart selection (NLR03) vs same-scale random'),
        ('I_full_upload - D_nlr03', 'I_full_upload', 'D_nlr03',
         'Upper bound gap: smaller = D near ceiling'),
        ('D_nlr03 - A_source_only', 'D_nlr03', 'A_source_only',
         'Absolute gain over source model'),
    ]

    for label, a, b, desc in comparisons:
        va = _get_map50(a)
        vb = _get_map50(b)
        if not (np.isnan(va) or np.isnan(vb)):
            delta = va - vb
            print(f'  {label:35s}  {delta:+.4f}   ({desc})')
        else:
            print(f'  {label:35s}  N/A')

    print(f'\n[one-shot] ALL DONE → {work_dir}')
    print(f'  results:     {reports_dir / "one_shot_results.csv"}')
    print(f'  summary:     {reports_dir / "summary_by_method.csv"}')
    print(f'  pool:        {work_dir / "candidate_pool.csv"}')

    return work_dir


# -- Screening method comparison ------------------------------------------


def run_screen_compare(
    config_path: str | Path,
    upload_ratio: float | None = None,
    scores_cache_dir: str | None = None,
    seeds: list[int] | None = None,
) -> Path:
    """Compare screening methods: Random vs Single-Signal screening.

    **Purpose**: Compare random sampling baselines against single-signal
    screening methods (confidence-only, nlr-only) under fair conditions —
    all groups use model_bank + source replay.

    5 groups:

      ====================  ========  ================  ===========================================
      Group                 Method    Key Param         Selection Condition
      ====================  ========  ================  ===========================================
      random_0.20           random    ratio=0.20        Drift-window random 20%
      random_0.30           random    ratio=0.30        Drift-window random 30%
      random_0.40           random    ratio=0.40        Drift-window random 40%
      conf_single           adaptive  use_entropy=False conf < P_10 only (ALL matching, single-signal)
      nlr_single            adaptive  nlr>=0.30,no_conf nlr >= 0.30 only (single-signal, no conf)

      ====================  ========  ================  ===========================================

    All groups run with: ``mix_mode='source_plus_hard'``, ``model_bank=True``,
    ``deployment='full'``, ``selection_mode='adaptive_streaming'``.

    Key comparisons:
      conf_single  vs random_*   : Confidence-only single-signal vs random baselines
      nlr_single   vs random_*   : nlr-only single-signal vs random baselines
      conf_single  vs nlr_single : Confidence-only vs nlr-only single-signal
      random_0.40  vs random_0.20: Random scaling (high vs low bandwidth)
    """
    import numpy as np

    exp = Experiment.from_yaml(config_path)
    default_ratio = float(exp.raw['sampling'].get('upload_ratio', 0.10))
    base_ratio = upload_ratio if upload_ratio is not None else default_ratio

    # (group_label, method, upload_ratio, mix_mode, sel_mode, use_entropy,
    #  nlr_threshold, model_bank_enabled)
    # nlr_threshold=None → not set (uses config default 0.0)
    # nlr_threshold=0.0  → explicitly 0.0 (entropy-based dual-signal)
    # nlr_threshold>0    → BSS: non_lie_ratio > threshold as signal 2
    screen_configs: list[tuple[str, str, float, str, str | None, bool, float | None, bool]] = [
        # ── Random sampling baselines (drift-window random) ──
        # upload_ratio controls random sampling fraction within drift windows
        ('random_0.20',    'random',   0.20, 'source_plus_hard', 'adaptive_streaming', False, None, True),
        ('random_0.30',    'random',   0.30, 'source_plus_hard', 'adaptive_streaming', False, None, True),
        ('random_0.40',    'random',   0.40, 'source_plus_hard', 'adaptive_streaming', False, None, True),
        # ── Single-signal: conf < P_10 only (no second signal, ALL matching) ──
        ('conf_single',    'adaptive', base_ratio, 'source_plus_hard', 'adaptive_streaming', False, 0.0,  True),
        # ── Single-signal: nlr >= 0.30 only (no conf, ALL matching) ──
        ('nlr_single',     'adaptive', base_ratio, 'source_plus_hard', 'adaptive_streaming', True,  0.30, True,  False),  # nlr-only single signal (no conf<P10 condition)
    ]

    if seeds is None:
        seeds = [42]
    print(f'[screen_compare] seeds={seeds} ({len(seeds)} run(s) per group)')
    print(f'[screen_compare] {len(screen_configs)} groups × {len(seeds)} seeds = {len(screen_configs) * len(seeds)} runs')

    frames: list[pd.DataFrame] = []
    for seed_val in seeds:
        for group_label, method, grp_ratio, mix_mode, sel_mode, use_entropy, nlr_threshold, mb_enabled, *extra in screen_configs:
            use_conf = extra[0] if extra else None
            print(f'\n{"="*60}')
            cond_desc = ''
            if method == 'adaptive':
                if use_conf is False:
                    cond_desc = f' (nlr>{nlr_threshold} only, no conf)'
                elif use_entropy and nlr_threshold is not None and nlr_threshold > 0:
                    cond_desc = f' (conf<P10 AND nlr>{nlr_threshold})'
                elif use_entropy:
                    cond_desc = ' (conf<P10 AND H>H_med)'
                else:
                    cond_desc = ' (conf<P10 only)'
            elif method == 'random':
                cond_desc = f' (drift-window random {grp_ratio:.0%})'
            print(f'Screen Compare [{group_label}] seed={seed_val}: method={method}, '
                  f'mix={mix_mode}, sel={sel_mode or "default"}, '
                  f'use_entropy={use_entropy}, nlr_threshold={nlr_threshold}, '
                  f'model_bank={mb_enabled}{cond_desc}')
            print(f'{"="*60}')

            csv_path = run_one_method(
                config_path, method, grp_ratio, mix_mode,
                scores_cache_dir=scores_cache_dir,
                selection_mode=sel_mode,
                use_entropy=use_entropy,
                model_bank_enabled=mb_enabled,
                deployment_mode='full',
                seed=seed_val,
                non_lie_ratio_threshold=nlr_threshold,
                use_conf=use_conf,
            )
            df = pd.read_csv(csv_path)
            df['ablation_group'] = group_label
            df['seed'] = seed_val
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    seed_tag = f'_seeds{"-".join(str(s) for s in seeds)}'
    out = exp.paths.work_dir / f'screen_compare_results{seed_tag}.csv'
    all_df.to_csv(out, index=False, encoding='utf-8-sig')

    reports_dir = exp.paths.work_dir / f'screen_compare_reports{seed_tag}'
    aggregate_results(out, reports_dir)

    if len(seeds) > 1:
        metric_cols = ['mAP50', 'mAP50_95', 'Precision', 'Recall',
                       'behavior_acc_iou50', 'behavior_recall_iou50']
        available = [c for c in metric_cols if c in all_df.columns]
        if available:
            per_seed = all_df.groupby(['ablation_group', 'seed'])[available].mean().reset_index()
            summary = per_seed.groupby('ablation_group')[available].agg(['mean', 'std']).round(4)
            summary.columns = [f'{col}_{stat}' for col, stat in summary.columns]
            summary.to_csv(reports_dir / 'seed_summary_mean_std.csv', encoding='utf-8-sig')
            print(f'\nMulti-seed summary (mean +/- std across {len(seeds)} seeds):')
            print(summary.to_string())

    # -- Summary table ------------------------------------------------
    group_order = [
        'random_0.20', 'random_0.30', 'random_0.40',
        'conf_single', 'nlr_single',
    ]
    agg_dict: dict[str, str | list] = {}
    available_metrics = [m for m in ['mAP50', 'mAP50_95', 'Precision', 'Recall',
                                      'behavior_acc_iou50', 'behavior_recall_iou50']
                         if m in all_df.columns]
    for m in available_metrics:
        agg_dict[m] = 'mean'
    for c in ['uploaded_images', 'upload_mb', 'accepted_update', 'train_seconds']:
        if c in all_df.columns:
            agg_dict[c] = 'sum'
    if agg_dict:
        summary_df = all_df.groupby('ablation_group').agg(agg_dict).reset_index()
        summary_df['_order'] = summary_df['ablation_group'].apply(
            lambda x: group_order.index(x) if x in group_order else 99)
        summary_df = summary_df.sort_values('_order').drop(columns=['_order'])
        summary_df.to_csv(reports_dir / 'summary_by_method.csv',
                          index=False, encoding='utf-8-sig')
        print(f'\nMethod summary:')
        print(summary_df.to_string(index=False))

    # -- Key comparisons ---------------------------------------------
    print(f'\n{"="*60}')
    print('KEY COMPARISONS (mean mAP50)')
    print(f'{"="*60}')

    def _get_map50(label: str) -> float:
        sub = all_df[all_df['ablation_group'] == label]
        if sub.empty or 'mAP50' not in sub.columns:
            return float('nan')
        return float(sub['mAP50'].mean())

    comparisons = [
        # Single-signal vs random
        ('conf_single - random_0.20', 'conf_single', 'random_0.20',
         'conf-only single vs random 20%: + = conf better'),
        ('conf_single - random_0.30', 'conf_single', 'random_0.30',
         'conf-only single vs random 30%: + = conf better'),
        ('conf_single - random_0.40', 'conf_single', 'random_0.40',
         'conf-only single vs random 40%: + = conf better'),
        ('nlr_single - random_0.20', 'nlr_single', 'random_0.20',
         'nlr-only single vs random 20%: + = nlr better'),
        ('nlr_single - random_0.30', 'nlr_single', 'random_0.30',
         'nlr-only single vs random 30%: + = nlr better'),
        ('nlr_single - random_0.40', 'nlr_single', 'random_0.40',
         'nlr-only single vs random 40%: + = nlr better'),
        # Single-signal head-to-head
        ('nlr_single - conf_single', 'nlr_single', 'conf_single',
         'nlr-only vs conf-only: + = nlr better'),
        # Random scaling
        ('random_0.40 - random_0.20', 'random_0.40', 'random_0.20',
         'Random scaling: high vs low bandwidth'),
    ]

    for label, a, b, desc in comparisons:
        va = _get_map50(a)
        vb = _get_map50(b)
        if not (np.isnan(va) or np.isnan(vb)):
            delta = va - vb
            print(f'  {label:35s}  {delta:+.4f}   ({desc})')
        else:
            print(f'  {label:35s}  N/A')

    print(f'\n[screen_compare] ALL DONE → {out}')
    print(f'  results:  {out}')
    print(f'  reports:  {reports_dir}')

    return out
