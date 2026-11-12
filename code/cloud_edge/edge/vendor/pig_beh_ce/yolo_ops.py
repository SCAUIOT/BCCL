from __future__ import annotations

from pathlib import Path
import math
import time
from typing import Any

import gc

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from .files import list_images
from .config import resolve_device


def _free_yolo(instance, clear_cuda_cache: bool = True) -> None:
    """Explicitly release a YOLO model and its CUDA resources."""
    if instance is not None:
        try:
            del instance
        except Exception:
            pass
    gc.collect()
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_names(names: dict | list | None) -> dict[int, str]:
    if names is None:
        return {0: 'stand', 1: 'lie', 2: 'eat'}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def load_dataset_meta(data_yaml: Path) -> tuple[int, dict[int, str]]:
    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    names = normalize_names(cfg.get('names'))
    nc = int(cfg.get('nc', len(names)))
    return nc, names


def write_dataset_yaml(root: Path, names: dict | list | None = None) -> Path:
    names = normalize_names(names)
    payload = {
        'path': str(root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(names),
        'names': names,
    }
    out = root / 'data.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return out


def write_eval_yaml(dataset_root: Path, names: dict | list | None = None) -> Path:
    names = normalize_names(names)
    payload = {
        'path': str(dataset_root.resolve()),
        'train': 'images/test',
        'val': 'images/test',
        'test': 'images/test',
        'nc': len(names),
        'names': names,
    }
    out = dataset_root / 'data.yaml'
    with open(out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return out


def parse_yolo_label(label_path: Path) -> list[dict[str, float]]:
    items = []
    if not label_path.exists():
        return items
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, xc, yc, w, h = int(float(parts[0])), *[float(x) for x in parts[1:5]]
        items.append({'cls': cls, 'xc': xc, 'yc': yc, 'w': w, 'h': h})
    return items


def box_iou_xywhn(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1 = a['xc'] - a['w'] / 2, a['yc'] - a['h'] / 2
    ax2, ay2 = a['xc'] + a['w'] / 2, a['yc'] + a['h'] / 2
    bx1, by1 = b['xc'] - b['w'] / 2, b['yc'] - b['h'] / 2
    bx2, by2 = b['xc'] + b['w'] / 2, b['yc'] + b['h'] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return float(inter / union) if union > 0 else 0.0


def _extract_behavior_stats_from_results(results, nc: int, h: int, w: int) -> dict[str, Any]:
    """Shared stats extraction used by both file-based and array-based predict."""
    confs, areas, cls_ids = [], [], []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_id = int(b.cls.item()) if b.cls is not None else 0
            if 0 <= cls_id < nc:
                cls_ids.append(cls_id)
            confs.append(float(b.conf.item()))
            xyxy = b.xyxy.view(-1).detach().cpu().numpy()
            area = max(0.0, float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) / max(1, w * h)))
            areas.append(area)
    hist = np.zeros(nc, dtype=np.float32)
    for c in cls_ids:
        hist[c] += 1.0
    if hist.sum() > 0:
        hist = hist / hist.sum()
    entropy = 0.0
    if hist.sum() > 0:
        entropy = float(-(hist * np.log(hist + 1e-12)).sum() / max(math.log(max(nc, 2)), 1e-6))
    dominant_cls = int(np.argmax(hist)) if hist.sum() > 0 else -1
    return {
        'det_count': float(len(cls_ids)),
        'avg_conf': float(np.mean(confs)) if confs else 0.0,
        'area_mean': float(np.mean(areas)) if areas else 0.0,
        'area_var': float(np.var(areas)) if areas else 0.0,
        'behavior_entropy': entropy,
        'dominant_behavior': dominant_cls,
        'behavior_hist': hist.tolist(),
    }


def image_predict_behavior_stats_from_array(
    model: YOLO, image_bgr: np.ndarray, imgsz: int, nc: int, conf: float = 0.25,
) -> dict[str, Any]:
    """Same as image_predict_behavior_stats but takes an in-memory BGR array.

    This avoids the disk round-trip when frames come directly from
    ``cv2.VideoCapture.read()`` — essential for streaming video pipelines.
    """
    h, w = image_bgr.shape[:2]
    results = model.predict(image_bgr, imgsz=imgsz, conf=conf, verbose=False)
    return _extract_behavior_stats_from_results(results, nc, h, w)


def image_predict_behavior_stats(model: YOLO, image_path: Path, imgsz: int, nc: int, conf: float = 0.25) -> dict[str, Any]:
    results = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)
    h0, w0 = 640, 640
    for r in results:
        h0, w0 = r.orig_shape[:2]
        break
    return _extract_behavior_stats_from_results(results, nc, h0, w0)


def predict_boxes_xywhn(model: YOLO, image_path: Path, imgsz: int, conf: float = 0.25) -> list[dict[str, float]]:
    results = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            xywhn = b.xywhn.view(-1).detach().cpu().numpy().tolist()
            out.append({
                'cls': int(b.cls.item()) if b.cls is not None else 0,
                'conf': float(b.conf.item()),
                'xc': float(xywhn[0]), 'yc': float(xywhn[1]), 'w': float(xywhn[2]), 'h': float(xywhn[3]),
            })
    return out


def behavior_match_metrics(model: YOLO, images_dir: Path, labels_dir: Path, imgsz: int, conf: float, iou_thr: float = 0.5, names: dict[int, str] | None = None) -> dict[str, float]:
    total_gt = 0
    matched_gt = 0
    correct_cls = 0
    names = normalize_names(names)
    gt_by_cls = {i: 0 for i in names}
    matched_by_cls = {i: 0 for i in names}
    correct_by_cls = {i: 0 for i in names}
    for img in list_images(images_dir):
        gts = parse_yolo_label(labels_dir / f'{img.stem}.txt')
        preds = sorted(predict_boxes_xywhn(model, img, imgsz=imgsz, conf=conf), key=lambda x: x['conf'], reverse=True)
        used = set()
        total_gt += len(gts)
        for gt in gts:
            gt_cls = int(gt['cls'])
            if gt_cls in gt_by_cls:
                gt_by_cls[gt_cls] += 1
        for pred in preds:
            best_i, best_iou = -1, 0.0
            for i, gt in enumerate(gts):
                if i in used:
                    continue
                iou = box_iou_xywhn(pred, gt)
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i >= 0 and best_iou >= iou_thr:
                used.add(best_i)
                matched_gt += 1
                gt_cls = int(gts[best_i]['cls'])
                pred_cls = int(pred['cls'])
                if gt_cls in matched_by_cls:
                    matched_by_cls[gt_cls] += 1
                if pred_cls == gt_cls:
                    correct_cls += 1
                    if gt_cls in correct_by_cls:
                        correct_by_cls[gt_cls] += 1
    out = {
        'behavior_acc_iou50': float(correct_cls / matched_gt) if matched_gt else 0.0,
        'behavior_recall_iou50': float(matched_gt / total_gt) if total_gt else 0.0,
        'num_gt_behavior_boxes': float(total_gt),
        'num_matched_behavior_boxes': float(matched_gt),
    }
    for cls_id, name in names.items():
        safe_name = str(name).replace(' ', '_')
        gt_n = gt_by_cls.get(cls_id, 0)
        matched_n = matched_by_cls.get(cls_id, 0)
        correct_n = correct_by_cls.get(cls_id, 0)
        out[f'behavior_gt_{safe_name}'] = float(gt_n)
        out[f'behavior_matched_{safe_name}'] = float(matched_n)
        out[f'behavior_recall_iou50_{safe_name}'] = float(matched_n / gt_n) if gt_n else 0.0
        out[f'behavior_acc_iou50_{safe_name}'] = float(correct_n / matched_n) if matched_n else 0.0
    return out


def evaluate_model(model_path: Path, dataset_root: Path, imgsz: int, conf: float = 0.25, device: Any = 'auto', names: dict | list | None = None, *, output_project: Path | None = None, data_yaml_override: Path | None = None) -> dict[str, Any]:
    model = YOLO(str(model_path))
    try:
        data_yaml = data_yaml_override or dataset_root / 'data.yaml'
        if not data_yaml.exists():
            data_yaml = write_eval_yaml(dataset_root, names=names)
        nc, names_dict = load_dataset_meta(data_yaml)
        t0 = time.time()
        metrics = model.val(
            data=str(data_yaml),
            split='test',
            imgsz=imgsz,
            device=resolve_device(device),
            workers=0,           # avoid subprocess leak across repeated calls
            verbose=False,
            plots=False,
            save_json=False,
            project=str(output_project or Path.cwd() / '.ultralytics_eval_cache'),
            name='val',
            exist_ok=True,
        )
        elapsed = time.time() - t0
        try:
            out = {
                'mAP50': float(metrics.box.map50),
                'mAP50_95': float(metrics.box.map),
                'Precision': float(metrics.box.mp),
                'Recall': float(metrics.box.mr),
            }
            maps = getattr(metrics.box, 'maps', None)
            if maps is not None:
                for i, ap in enumerate(list(maps)):
                    out[f'AP50_95_{names_dict.get(i, str(i))}'] = float(ap)
        except Exception:
            out = {'mAP50': None, 'mAP50_95': None, 'Precision': None, 'Recall': None, 'raw': str(metrics)}
        images_dir = dataset_root / 'images' / 'test'
        labels_dir = dataset_root / 'labels' / 'test'
        out.update(behavior_match_metrics(model, images_dir, labels_dir, imgsz=imgsz, conf=conf, names=names_dict))
        out.update({'eval_seconds': elapsed, 'num_eval_images': len(list_images(images_dir))})
        return out
    finally:
        _free_yolo(model)


def dedup_boxes_nms(boxes: list[dict[str, float]], iou_threshold: float) -> list[dict[str, float]]:
    """Remove duplicate detection boxes via class-aware NMS.

    Boxes are expected as a list of dicts with keys: cls, conf, xc, yc, w, h (all xywhn).
    For each class, boxes are sorted by confidence descending. A box is removed if its
    IoU with any higher-confidence box of the same class >= iou_threshold.
    """
    if not boxes or iou_threshold <= 0.0 or iou_threshold >= 1.0:
        return boxes

    # Group by class
    by_class: dict[int, list[dict[str, float]]] = {}
    for b in boxes:
        cls = int(b['cls'])
        by_class.setdefault(cls, []).append(b)

    kept: list[dict[str, float]] = []
    for cls, cls_boxes in by_class.items():
        sorted_boxes = sorted(cls_boxes, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        cls_kept: list[dict[str, float]] = []
        for b in sorted_boxes:
            overlap = False
            for k in cls_kept:
                if box_iou_xywhn(b, k) >= iou_threshold:
                    overlap = True
                    break
            if not overlap:
                cls_kept.append(b)
        kept.extend(cls_kept)
    return kept


def pseudo_label_with_teacher(
    teacher_path: Path,
    images: list[Path],
    labels_dir: Path,
    imgsz: int,
    conf_thr: float,
    min_box_area_ratio: float,
    nc: int | None = None,
    dedup_iou_threshold: float = 0.0,
) -> dict[str, Any]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(teacher_path))
    try:
        total_boxes = 0
        empty_images = 0
        confs = []
        class_counts: dict[int, int] = {}
        dedup_removed_total = 0
        predict_iou = 1.0 if (dedup_iou_threshold > 0.0 and dedup_iou_threshold < 1.0) else 0.7
        for img_path in images:
            results = model.predict(str(img_path), imgsz=imgsz, conf=conf_thr, iou=predict_iou, verbose=False)
            boxes: list[dict[str, float]] = []
            for r in results:
                if r.boxes is None:
                    continue
                h, w = r.orig_shape[:2]
                for b in r.boxes:
                    xyxy = b.xyxy.view(-1).detach().cpu().numpy()
                    area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) / max(1, w * h))
                    if area < min_box_area_ratio:
                        continue
                    cls_id = int(b.cls.item()) if b.cls is not None else 0
                    xywhn = b.xywhn.view(-1).detach().cpu().numpy().tolist()
                    boxes.append({
                        'cls': cls_id,
                        'conf': float(b.conf.item()),
                        'xc': float(xywhn[0]), 'yc': float(xywhn[1]),
                        'w': float(xywhn[2]), 'h': float(xywhn[3]),
                    })

            before_dedup = len(boxes)
            if dedup_iou_threshold > 0.0 and dedup_iou_threshold < 1.0:
                boxes = dedup_boxes_nms(boxes, dedup_iou_threshold)
            dedup_removed = before_dedup - len(boxes)
            dedup_removed_total += dedup_removed

            lines = []
            for b in boxes:
                lines.append(str(int(b['cls'])) + ' ' + ' '.join(f'{b[k]:.6f}' for k in ['xc', 'yc', 'w', 'h']))
                confs.append(float(b['conf']))
                cls_id = int(b['cls'])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

            total_boxes += len(lines)
            if not lines:
                empty_images += 1
            with open(labels_dir / f'{img_path.stem}.txt', 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
        out: dict[str, Any] = {
            'pseudo_behavior_boxes': int(total_boxes),
            'pseudo_empty_images': int(empty_images),
            'pseudo_avg_conf': float(np.mean(confs)) if confs else 0.0,
            'pseudo_dedup_removed_boxes': int(dedup_removed_total),
        }
        class_ids = range(int(nc)) if nc is not None else sorted(class_counts)
        for cls_id in class_ids:
            out[f'pseudo_class_{cls_id}_boxes'] = int(class_counts.get(int(cls_id), 0))
        return out
    finally:
        _free_yolo(model)


def train_student(model_path: Path, data_yaml: Path, out_project: Path, run_name: str, imgsz: int, epochs: int, batch: int, workers: int, device: Any, train_options: dict | None = None) -> Path:
    model = YOLO(str(model_path))
    try:
        model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, workers=workers, device=resolve_device(device), project=str(out_project), name=run_name, exist_ok=True, verbose=False, **(train_options or {}))
        best = out_project / run_name / 'weights' / 'best.pt'
        if not best.exists():
            raise RuntimeError(f'Training finished but best.pt not found: {best}')
        return best
    finally:
        _free_yolo(model)


# ── Partial model deployment ────────────────────────────────────────────


def _head_key_patterns() -> list[str]:
    """Return state_dict key substrings that belong to the detection head.

    YOLOv8's Detect module sits at the end of the model.  Its state-dict
    keys typically contain ``cv2.``, ``cv3.``, or ``dfl.`` — the
    classification / regression convs and the distribution-focal-loss
    layer of the anchor-free head.

    Additionally, any key whose numeric prefix (e.g. ``model.22.``) is at
    or beyond ``head_start_index`` is treated as head-level.
    """
    return ['cv2.', 'cv3.', 'dfl.']


def _is_head_key(key: str, head_start_index: int = 21) -> bool:
    """Decide whether *key* belongs to the detection head."""
    # Check by module index prefix: "model.N."
    if key.startswith('model.'):
        parts = key.split('.')
        if len(parts) >= 2:
            try:
                idx = int(parts[1])
                if idx >= head_start_index:
                    return True
            except ValueError:
                pass
    # Check by sub-layer name patterns
    for pat in _head_key_patterns():
        if pat in key:
            return True
    return False


def count_model_params(model_path: Path) -> dict[str, int]:
    """Count total and head-only parameters.

    Returns
    -------
    dict with ``total``, ``head``, ``backbone_neck`` counts.
    """
    model = YOLO(str(model_path))
    try:
        state = model.model.state_dict()
        total = sum(v.numel() for v in state.values())
        head = sum(v.numel() for k, v in state.items() if _is_head_key(k))
        return {'total': total, 'head': head, 'backbone_neck': total - head}
    finally:
        _free_yolo(model)


def extract_head_weights(model_path: Path) -> dict[str, Any]:
    """Extract only the detection-head weights from a YOLO checkpoint.

    Returns a lightweight dict containing the head state-dict entries plus
    metadata needed for re-assembly.
    """
    model = YOLO(str(model_path))
    try:
        full_state = model.model.state_dict()
        head_state = {k: v.clone() for k, v in full_state.items() if _is_head_key(k)}
        return {'head_state': head_state, 'head_key_count': len(head_state)}
    finally:
        _free_yolo(model)


def apply_head_weights(base_model_path: Path, head_weights: dict[str, Any], output_path: Path) -> Path:
    """Apply extracted head weights onto a base model and save.

    Parameters
    ----------
    base_model_path : Path
        The base model (usually the initial student or current model).
    head_weights : dict
        Result from :func:`extract_head_weights`.
    output_path : Path
        Where to save the merged model.

    Returns
    -------
    Path
        *output_path*.
    """
    model = YOLO(str(base_model_path))
    try:
        current_state = model.model.state_dict()
        head_state = head_weights['head_state']
        # Apply head weights
        for k, v in head_state.items():
            if k in current_state and current_state[k].shape == v.shape:
                current_state[k] = v.clone()
        model.model.load_state_dict(current_state)
        model.save(str(output_path))
        return output_path
    finally:
        _free_yolo(model)


def measure_deployment_stats(
    model_path: Path,
    base_model_path: Path,
) -> dict:
    """Compare full vs partial deployment cost.

    Returns
    -------
    dict with keys:
        full_params, head_params, full_size_bytes, head_size_bytes,
        full_copy_seconds (estimated)
    """
    import tempfile
    import os

    full_params = count_model_params(model_path)
    head_weights = extract_head_weights(model_path)

    full_size = model_path.stat().st_size

    # Write head-only weights to temp file to measure size
    head_size = 0
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        try:
            torch.save(head_weights, tmp)
            head_size = os.path.getsize(tmp.name)
        finally:
            os.unlink(tmp.name)

    # Estimate copy time (very rough: ~100 MB/s for local disk)
    est_full_seconds = full_size / (100 * 1024 * 1024)
    est_head_seconds = head_size / (100 * 1024 * 1024)

    return {
        'full_params': full_params['total'],
        'head_params': full_params['head'],
        'backbone_neck_params': full_params['backbone_neck'],
        'full_size_bytes': full_size,
        'head_size_bytes': head_size,
        'full_size_mb': full_size / (1024 * 1024),
        'head_size_mb': head_size / (1024 * 1024),
        'est_full_deploy_seconds': est_full_seconds,
        'est_head_deploy_seconds': est_head_seconds,
        'param_reduction_ratio': full_params['head'] / max(full_params['total'], 1),
    }
