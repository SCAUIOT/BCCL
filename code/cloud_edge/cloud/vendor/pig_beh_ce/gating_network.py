"""System-style gating network for the spare-parts warehouse (ModelZoo pattern).

Pattern: ``cloud_app/application/gatingNetwork.py`` + ``ModelZoo.select_model_for_edge``
in ``D:\\project\\system`` — the bank there is *maintained* by a convolutional
network: a pretrained ResNet18 backbone with a ``Linear(512, num_models)`` head
and Softmax, which learns "which expert model fits these images" instead of
hand-crafted scene-fingerprint similarity.

The system repo implements the inference path (``/select_model``) but its
training loop (``train_selector``, ``GatingDataset``) was deleted; this module
implements the *complete* mechanism, following the training design documented
in the system's ``运行准备.md``:

* training images: frames uploaded by the edge (our drift-window frames);
* teacher supervision: the teacher model's pseudo labels are ground truth;
* per-image label: the expert (bank model) whose detection boxes best match
  the teacher's boxes on that image — argmin matching error;
* the head is (re)trained whenever the bank composition changes (admission /
  eviction), with weight transfer for surviving experts
  (system's ``use_weight_from_before``, defined there but never wired).

Usage in the runner:

* :func:`refresh_gating_network` — called after every ``update_model_bank``,
  retrains the selector head on the gating pool and saves a versioned
  snapshot (``gating_model.pth`` + ``metadata.json`` with the head->model
  mapping), same as the system's ``gating-models/<timestamp>/`` snapshots.
* :func:`gating_model_weights` — called by ``query_model_bank`` at deployment
  (rejection fallback) time: feeds the current stage's recent frames through
  the selector, averages the per-image weights, and returns
  ``{model_path: weight}`` for ranking.

The "keep current model" competition (P0) that guards fallback deployment
lives in the runner, mirroring the system's edge-side shadow evaluation
(``clip_online_eval.ClipShadowComparator``).
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

from .yolo_ops import box_iou_xywhn, parse_yolo_label

# ImageNet normalization (torchvision ResNet18 pretrained weights expect it).
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD),
])


def _torch_device(device: Any) -> torch.device:
    """Normalize the pipeline's device spec ('auto'/0/'cpu'/'cuda:0') to a torch.device."""
    if isinstance(device, torch.device):
        return device
    text = str(device).strip().lower()
    if text in ('cpu', 'none'):
        return torch.device('cpu')
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class GatingNetwork(nn.Module):
    """Pretrained ResNet18 backbone + Linear(num_models) head + Softmax.

    Pattern: ``GatingNetwork`` in D:\\project\\system\\cloud_app\\application\\gatingNetwork.py.
    """

    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.num_models = int(num_models)
        # ImageNet-pretrained backbone (system's ResNet18_Weights.DEFAULT);
        # degrade to random init if the weights cannot be downloaded —
        # the head is retrained from scratch on the gating pool anyway.
        try:
            self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.pretrained_backbone = True
        except Exception:
            self.pretrained_backbone = False
            try:
                self.resnet18 = models.resnet18(weights=None)
            except Exception:
                # very old torchvision: legacy pretrained kwarg API
                self.resnet18 = models.resnet18(pretrained=True)
        ft_cnt = self.resnet18.fc.in_features  # 512 — read before replacing
        self.resnet18.fc = nn.Identity()  # drop the ImageNet classification head
        self.fc = nn.Linear(ft_cnt, self.num_models)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet18(x)
        return self.softmax(self.fc(x))


# ---------------------------------------------------------------------------
# Snapshot save / load (system's gating-models/<timestamp>/ versioning)
# ---------------------------------------------------------------------------

def save_gating_snapshot(net: GatingNetwork, model_map: dict[str, str], gating_dir: Path) -> dict:
    """Save the selector weights + head->model mapping.

    ``model_map``: {'0': '<bank model path>', ...} — head index to expert
    model file, the same mapping the system persists in its
    ``gating-models/.../metadata.json``.
    """
    gating_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), gating_dir / 'gating_model.pth')
    meta = {
        'model_map': {str(k): str(v) for k, v in model_map.items()},
        'num_models': int(net.num_models),
        'created_time': int(time.time() * 1000),
    }
    (gating_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return meta


def load_gating_snapshot(gating_dir: Path, device: Any) -> tuple[GatingNetwork | None, dict | None]:
    """Load the selector snapshot if present (system's find_latest_gating_model)."""
    meta_path = gating_dir / 'metadata.json'
    weights_path = gating_dir / 'gating_model.pth'
    if not (meta_path.exists() and weights_path.exists()):
        return None, None
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        net = GatingNetwork(int(meta['num_models'])).to(_torch_device(device))
        net.load_state_dict(torch.load(str(weights_path), map_location=_torch_device(device)))
        net.eval()
        return net, meta
    except (json.JSONDecodeError, KeyError, RuntimeError):
        return None, None


def transfer_head_weights(
    net: GatingNetwork,
    prev_meta: dict,
    prev_sd: dict,
) -> None:
    """Weight transfer for the selector head (system's use_weight_from_before).

    Surviving experts keep their old head row; newly admitted experts are
    initialized with the mean of the old head rows, so the selector degrades
    gracefully when the bank composition changes instead of restarting from
    a random head.
    """
    try:
        old_map: dict[str, int] = {str(v): int(k) for k, v in prev_meta['model_map'].items()}
        old_w = prev_sd['fc.weight']
        old_b = prev_sd['fc.bias']
        mean_w = old_w.mean(dim=0, keepdim=True)
        mean_b = old_b.mean()
        with torch.no_grad():
            for i, path in enumerate(net.model_map_order):  # pragma: no cover - set by caller
                if path in old_map:
                    net.fc.weight[i].copy_(old_w[old_map[path]])
                    net.fc.bias[i].copy_(old_b[old_map[path]])
                else:
                    net.fc.weight[i].copy_(mean_w.squeeze(0))
                    net.fc.bias[i].fill_(float(mean_b.item()))
    except (KeyError, RuntimeError):
        pass  # keep the fresh random head on any mismatch


# ---------------------------------------------------------------------------
# Expert matching (teacher-supervised labels)
# ---------------------------------------------------------------------------

def _expert_boxes_batched(model: YOLO, image_paths: list[Path], imgsz: int, conf: float, batch: int) -> list[list[dict]]:
    """Run one bank expert over the pool images (batched) and return boxes.

    Box dicts use the same keys as :func:`predict_boxes_xywhn` / teacher
    labels: xc/yc/w/h normalized + cls.
    """
    out: list[list[dict]] = []
    for start in range(0, len(image_paths), batch):
        chunk = [str(p) for p in image_paths[start:start + batch]]
        results = model.predict(source=chunk, imgsz=imgsz, conf=conf, verbose=False)
        for r in results:
            boxes: list[dict] = []
            if r.boxes is not None and len(r.boxes):
                xywhn = r.boxes.xywhn.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                for j in range(len(xywhn)):
                    boxes.append({
                        'cls': int(cls[j]),
                        'xc': float(xywhn[j, 0]), 'yc': float(xywhn[j, 1]),
                        'w': float(xywhn[j, 2]), 'h': float(xywhn[j, 3]),
                    })
            out.append(boxes)
    return out


def _expert_match_error(teacher_boxes: list[dict], expert_boxes: list[dict], iou_thr: float = 0.5) -> float | None:
    """Fraction of teacher boxes not matched by the expert (None = no teacher supervision)."""
    if not teacher_boxes:
        return None
    matched = 0
    for t in teacher_boxes:
        best = 0.0
        for e in expert_boxes:
            if e['cls'] != t['cls']:
                continue
            best = max(best, box_iou_xywhn(t, e))
        if best >= iou_thr:
            matched += 1
    return 1.0 - matched / len(teacher_boxes)


def _load_image_tensors(image_paths: list[Path], device: torch.device) -> torch.Tensor | None:
    tensors = []
    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
        except (OSError, ValueError):
            continue
        tensors.append(_preprocess(img))
    if not tensors:
        return None
    return torch.stack(tensors).to(device)


# ---------------------------------------------------------------------------
# Refresh (train) + query
# ---------------------------------------------------------------------------

def refresh_gating_network(
    gating_dir: Path,
    pool_images: list[Path],
    teacher_labels: dict[Path, Path],
    expert_paths: list[str],
    device: Any,
    imgsz: int,
    conf_thr: float = 0.40,
    head_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
    predict_batch: int = 16,
    iou_thr: float = 0.5,
) -> dict:
    """Retrain the selector head on the gating pool (system's train_selector design).

    * experts = current bank models (active rows, in bank CSV order);
    * for every pool image with teacher labels, each expert predicts boxes
      and the expert with the smallest matching error vs the teacher becomes
      the image's label (argmin error — "which expert fits this scene best");
    * the head is (re)trained from scratch with weight transfer from the
      previous snapshot (surviving experts keep their rows), backbone frozen.

    Returns a dict for the per-stage decision record.
    """
    dev = _torch_device(device)
    experts = [str(p) for p in expert_paths if str(p)]
    train_images = [p for p in pool_images if p in teacher_labels]
    info = {
        'gating_refreshed': False,
        'gating_num_models': len(experts),
        'gating_train_images': len(train_images),
        'gating_final_loss': 0.0,
        'gating_reason': '',
        'gating_backbone_pretrained': True,
    }
    if len(experts) < 2:
        info['gating_reason'] = f'need >=2 experts, have {len(experts)}'
        return info
    if not train_images:
        info['gating_reason'] = 'no labeled pool images'
        return info

    # 1. Per-expert predictions over the pool (batched).
    per_expert_boxes: list[list[list[dict]]] = []
    for ep in experts:
        try:
            model = YOLO(ep)
        except Exception:
            per_expert_boxes.append([[] for _ in train_images])
            continue
        try:
            per_expert_boxes.append(_expert_boxes_batched(model, train_images, imgsz, conf_thr, predict_batch))
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 2. Per-image labels: argmin teacher-matching error over experts.
    labels: list[int] = []
    usable: list[Path] = []
    for i, p in enumerate(train_images):
        teacher_boxes = parse_yolo_label(teacher_labels[p])
        errs = [_expert_match_error(teacher_boxes, boxes[i], iou_thr) for boxes in per_expert_boxes]
        errs = [e if e is not None else 1.0 for e in errs]
        labels.append(int(np.argmin(errs)))
        usable.append(p)
    if not usable:
        info['gating_reason'] = 'no image had teacher boxes'
        return info

    # 3. Head init + weight transfer from the previous snapshot.
    net = GatingNetwork(len(experts)).to(dev)
    info['gating_backbone_pretrained'] = bool(getattr(net, 'pretrained_backbone', True))
    net.model_map_order = experts  # consumed by transfer_head_weights
    prev_net, prev_meta = load_gating_snapshot(gating_dir, device)
    if prev_net is not None and prev_meta is not None:
        prev_sd = prev_net.state_dict()
        transfer_head_weights(net, prev_meta, prev_sd)
        del prev_net
    if freeze_backbone:
        for param in net.resnet18.parameters():
            param.requires_grad = False
    net.train()

    # 4. Train the head (CrossEntropy over experts).
    x = _load_image_tensors(usable, dev)
    if x is None:
        info['gating_reason'] = 'could not load any pool image'
        return info
    y = torch.tensor(labels[:len(usable)], device=dev)
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    final_loss = 0.0
    for _epoch in range(int(head_epochs)):
        epoch_losses = []
        perm = torch.randperm(len(usable), device=dev)
        for start in range(0, len(usable), int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            optimizer.zero_grad()
            out = net(x[idx])
            loss = criterion(out, y[idx])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        final_loss = float(np.mean(epoch_losses))
    del x, y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Snapshot (system's gating-models/<timestamp> pattern).
    save_gating_snapshot(net, {str(i): ep for i, ep in enumerate(experts)}, gating_dir)
    info.update({
        'gating_refreshed': True,
        'gating_num_models': len(experts),
        'gating_train_images': len(usable),
        'gating_final_loss': final_loss,
        'gating_reason': '',
    })
    return info


def gating_model_weights(
    gating_dir: Path,
    image_paths: list[Path],
    device: Any,
    max_images: int = 24,
) -> tuple[dict[str, float] | None, dict | None]:
    """Rank bank models by selector weight for the current scene.

    Feeds up to ``max_images`` recent frames through the selector and
    averages the per-image softmax weights (system's
    ``mean(weights, dim=0)`` + argmax), returning ``{model_path: weight}``.
    Returns (None, None) if no trained snapshot exists.
    """
    net, meta = load_gating_snapshot(gating_dir, device)
    if net is None or meta is None or not image_paths:
        return None, None
    dev = _torch_device(device)
    net.to(dev)
    net.eval()
    x = _load_image_tensors(list(image_paths[:int(max_images)]), dev)
    if x is None:
        return None, None
    with torch.no_grad():
        weights = net(x)
        avg = weights.mean(dim=0).cpu().numpy()
    del x
    model_map = {str(k): str(v) for k, v in meta['model_map'].items()}
    return {model_map[str(i)]: float(avg[i]) for i in range(len(avg))}, meta