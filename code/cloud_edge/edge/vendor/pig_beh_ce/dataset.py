from __future__ import annotations

from pathlib import Path
import random

import yaml

from .files import list_images, safe_clear_dir, copy_file, _hardlink_or_copy
from .yolo_ops import write_dataset_yaml, normalize_names


SUPPORTED_MIX_MODES = {'source_plus_hard', 'hard_only'}


def label_for_image(label_dir: Path, image: Path) -> Path:
    return label_dir / f'{image.stem}.txt'


def _labels_dir_from_images_rel(images_rel: str) -> str:
    return images_rel.replace('images', 'labels', 1)


def load_source_info(source_yaml: Path) -> tuple[Path, Path, Path, Path, dict]:
    source_yaml = Path(source_yaml).resolve()
    with open(source_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get('path', '.'))
    if not root.is_absolute():
        root = (source_yaml.parent / root).resolve()
    train_rel = cfg.get('train', 'images/train')
    val_rel = cfg.get('val', train_rel)
    train_img = root / train_rel
    train_lbl = root / _labels_dir_from_images_rel(train_rel)
    val_img = root / val_rel
    val_lbl = root / _labels_dir_from_images_rel(val_rel)
    return train_img, train_lbl, val_img, val_lbl, cfg


def _copy_images_and_labels(
    images: list[Path],
    src_label_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    create_empty_label: bool = True,
) -> int:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for img in images:
        _hardlink_or_copy(img, dst_img_dir / img.name)
        lab = label_for_image(src_label_dir, img)
        out_lab = dst_lbl_dir / f'{img.stem}.txt'
        if lab.exists():
            _hardlink_or_copy(lab, out_lab)
        elif create_empty_label:
            out_lab.write_text('', encoding='utf-8')
        linked += 1
    return linked


def _resolve_optional_source_split(source_cfg: dict, source_yaml: Path, key: str) -> tuple[Path, Path] | None:
    if key not in source_cfg:
        return None
    root = Path(source_cfg.get('path', '.'))
    source_yaml = Path(source_yaml).resolve()
    if not root.is_absolute():
        root = (source_yaml.parent / root).resolve()
    img_rel = source_cfg[key]
    return root / img_rel, root / _labels_dir_from_images_rel(img_rel)


def _select_source_train_images(
    all_src_train_images: list[Path],
    source_sample_count: int | None,
    rng: random.Random,
) -> list[Path]:
    """Choose source replay images.

    Rules:
      source_sample_count > 0  -> random subset with that maximum size
      source_sample_count == 0 -> no source replay images
      source_sample_count < 0 or None -> all source train images

    This explicit behavior is important for the ablation:
      hard_only means no source replay in train split.
    """
    if source_sample_count is None or int(source_sample_count) < 0:
        return list(all_src_train_images)
    n = int(source_sample_count)
    if n == 0:
        return []
    if len(all_src_train_images) > n:
        return rng.sample(list(all_src_train_images), n)
    return list(all_src_train_images)


def build_mixed_dataset(
    out_root: Path,
    source_yaml: Path | None,
    selected_images: list[Path],
    pseudo_labels_dir: Path,
    source_sample_count: int,
    target_repeat: int = 1,
    source_images: list[Path] | None = None,
    seed: int = 42,
    mix_mode: str = 'source_plus_hard',
    names: dict | list | None = None,
    target_val_fraction: float = 0.10,
    min_target_val_images: int = 5,
) -> Path:
    """Build a YOLO dataset for cloud-side student update.

    mix_mode controls the training-data-composition ablation:

    1) source_plus_hard
       train = source replay images + current-stage selected hard images with teacher pseudo labels
       Purpose: adapt to current domain while reducing catastrophic forgetting.

    2) hard_only
       train = current-stage selected hard images with teacher pseudo labels only
       Purpose: test whether using only uploaded hard samples overfits current domain or forgets old stages.

    When *source_images* is given (SDR forgetting-aware replay, R3), it is
    used verbatim as the source replay subset instead of the random
    ``_select_source_train_images`` draw; nonexistent paths are dropped and
    at most ``source_sample_count`` images are taken.  Passing ``[]`` forces
    zero source replay.

    Validation/test split comes from source val/test/train fallback when a source dataset is configured.
    If no source dataset exists, a small held-out pseudo-labeled subset from the uploaded target images is used
    only for choosing best.pt during training. Formal model quality is evaluated later on each stage's
    test_labeled set, which is never copied into this training dataset.
    """
    mix_mode = str(mix_mode or 'source_plus_hard').strip()
    if mix_mode not in SUPPORTED_MIX_MODES:
        raise ValueError(f'Unsupported mix_mode={mix_mode}. Expected one of {sorted(SUPPORTED_MIX_MODES)}')

    safe_clear_dir(out_root)
    for p in ['images/train', 'labels/train', 'images/val', 'labels/val', 'images/test', 'labels/test']:
        (out_root / p).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    selected_images = list(selected_images)
    source_yaml_path = Path(source_yaml).resolve() if source_yaml else None
    source_available = bool(source_yaml_path and source_yaml_path.exists())

    src_img = src_lbl = val_img = val_lbl = None
    source_cfg = {'names': normalize_names(names)}
    all_src_train_images: list[Path] = []
    if source_available:
        src_img, src_lbl, val_img, val_lbl, source_cfg = load_source_info(source_yaml_path)
        all_src_train_images = list_images(src_img)

    if source_available and not all_src_train_images:
        raise FileNotFoundError(f'No source train images found. source_yaml={source_yaml}, train_dir={src_img}')

    # 1) Optional source replay for train split.
    source_selection = 'random'
    if source_available and mix_mode == 'source_plus_hard':
        if source_images is not None:
            # R3 forgetting-aware replay: caller pre-scored and preselected
            # the source images (verbatim order = forgetting-risk order).
            src_train_images = [p for p in source_images if p.exists()]
            n_cap = int(source_sample_count)
            if 0 <= n_cap < len(src_train_images):
                src_train_images = src_train_images[:n_cap]
            source_selection = 'preselected'
        else:
            src_train_images = _select_source_train_images(all_src_train_images, source_sample_count, rng)
    else:
        src_train_images = []

    if src_train_images:
        _copy_images_and_labels(src_train_images, src_lbl, out_root / 'images/train', out_root / 'labels/train')

    # 2) Current target hard samples with teacher pseudo labels.
    target_train_images = list(selected_images)
    target_val_images: list[Path] = []
    if not source_available and selected_images:
        shuffled = list(selected_images)
        rng.shuffle(shuffled)
        frac = max(0.0, min(float(target_val_fraction), 0.5))
        if len(shuffled) > 1 and frac > 0:
            min_val = max(1, int(min_target_val_images))
            val_n = max(min_val, int(round(len(shuffled) * frac)))
            val_n = min(val_n, len(shuffled) - 1)
            target_val_images = shuffled[:val_n]
            target_train_images = shuffled[val_n:]
        if not target_train_images:
            target_train_images = list(selected_images)
            target_val_images = []

    target_written = 0
    for rep in range(max(1, int(target_repeat))):
        for img in target_train_images:
            out_name = img.name if rep == 0 else f'{img.stem}_rep{rep}{img.suffix}'
            _hardlink_or_copy(img, out_root / 'images/train' / out_name)
            lab = label_for_image(pseudo_labels_dir, img)
            out_lab = out_root / 'labels/train' / f'{Path(out_name).stem}.txt'
            if lab.exists():
                _hardlink_or_copy(lab, out_lab)
            else:
                out_lab.write_text('', encoding='utf-8')
            target_written += 1

    if target_written == 0 and not src_train_images:
        raise RuntimeError('Mixed dataset train split is empty: no source images and no selected hard samples.')

    # 3) Validation split. Kept away from test_labeled to avoid evaluation leakage.
    val_images: list[Path] = []
    val_label_dir: Path | None = None
    val_source = 'target_pseudo_holdout'
    if source_available:
        val_images = list_images(val_img)
        val_label_dir = val_lbl
        val_source = 'source_val'
        if not val_images:
            optional_test = _resolve_optional_source_split(source_cfg, source_yaml_path, 'test')
            if optional_test is not None:
                test_img, test_lbl = optional_test
                test_images = list_images(test_img)
                if test_images:
                    val_images = test_images
                    val_label_dir = test_lbl
                    val_source = 'source_test_fallback'
        if not val_images:
            fallback_n = min(100, len(all_src_train_images))
            val_images = list(all_src_train_images[:fallback_n])
            val_label_dir = src_lbl
            val_source = 'source_train_fallback_first_100'
    else:
        val_images = target_val_images or target_train_images[:min(100, len(target_train_images))]
        val_label_dir = pseudo_labels_dir
        if not target_val_images:
            val_source = 'target_pseudo_train_fallback'

    if val_images and val_label_dir is not None:
        _copy_images_and_labels(val_images, val_label_dir, out_root / 'images/val', out_root / 'labels/val')
        _copy_images_and_labels(val_images, val_label_dir, out_root / 'images/test', out_root / 'labels/test')

    names = normalize_names(source_cfg.get('names'))
    (out_root / 'build_info.txt').write_text('\n'.join([
        f'mix_mode={mix_mode}',
        f'source_replay_available={source_available}',
        f'source_yaml={source_yaml_path or ""}',
        f'source_train_dir={src_img}',
        f'source_selection={source_selection}',
        f'train_source_images={len(src_train_images)}',
        f'target_selected_images={len(selected_images)}',
        f'target_train_images_before_repeat={len(target_train_images)}',
        f'target_train_images_after_repeat={target_written}',
        f'target_val_images={len(target_val_images)}',
        f'target_repeat={target_repeat}',
        f'val_images={len(val_images)}',
        f'val_source={val_source}',
        f'names={names}',
    ]) + '\n', encoding='utf-8')
    return write_dataset_yaml(out_root, names=names)


def export_selected_images(selected_df, out_dir: Path) -> list[Path]:
    safe_clear_dir(out_dir)
    result = []
    for _, row in selected_df.iterrows():
        src = Path(row['path'])
        dst = out_dir / src.name
        copy_file(src, dst)
        result.append(dst)
    return result
