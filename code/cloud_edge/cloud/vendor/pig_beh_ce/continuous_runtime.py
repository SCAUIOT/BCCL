"""YOLO adapter and command entry for the continuous V17 experiment."""
from __future__ import annotations

import gc
import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from dataclasses import asdict

from .continuous import (ContinuousRun, DiskBudget, VIDEO_READ_SUMMARY_FIELDS, atomic_json, remove_owned,
                         effective_inference_fps, normalize_method, output_lock, sample_frame, sha256_file,
                         validate_settings, video_manifest)


# Original V17 paper ablation rows: selection -> model bank -> source replay.
CONTINUOUS_ABLATION_GROUPS = {
    'A_source_only': ('source_only', False, 'hard_only'),
    'D_nlr03_nobank': ('bccl', False, 'hard_only'),
    'D_nlr03': ('bccl', True, 'hard_only'),
    'D_nlr03_replay': ('bccl', True, 'source_plus_hard'),
}
CONTINUOUS_JOINT_PARAMETERS = {
    f'D_grid_pq{pq:02d}_ta{ta:02d}': {
        'p_low': pq / 100., 'non_lie_ratio_threshold': ta / 100., 'alpha': .05}
    for pq in (10, 20, 30, 40, 50) for ta in (10, 20, 30, 40, 50)
    if (pq, ta) != (20, 30)  # reuse the current continuous BCCL anchor
}
CONTINUOUS_ALPHA_PARAMETERS = {
    f'D_alpha{alpha:02d}': {
        'p_low': .20, 'non_lie_ratio_threshold': .30, 'alpha': alpha / 100.}
    for alpha in (1, 10, 20)  # alpha=.05 is the same anchor
}
CONTINUOUS_SENSITIVITY_PARAMETERS = {
    **CONTINUOUS_JOINT_PARAMETERS, **CONTINUOUS_ALPHA_PARAMETERS}
CONTINUOUS_SENSITIVITY_BATCHES = {
    'joint': tuple(CONTINUOUS_JOINT_PARAMETERS),
    'alpha': tuple(CONTINUOUS_ALPHA_PARAMETERS),
}
CONTINUOUS_GROUPS = {
    **CONTINUOUS_ABLATION_GROUPS,
    **{group: ('bccl', True, 'source_plus_hard')
       for group in CONTINUOUS_SENSITIVITY_PARAMETERS},
}


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_configuration(config_path, videos_dir=None, work_dir=None, method=None, group=None):
    import yaml
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    base = config_path.parent.parent if config_path.parent.name == 'configs' else Path.cwd()
    if videos_dir is not None:
        raw['paths']['videos_dir'] = str(Path(videos_dir).resolve())
    if work_dir is not None:
        raw['paths']['work_dir'] = str(Path(work_dir).resolve())
    if method is not None:
        raw['continuous']['method'] = normalize_method(method)
    elif 'method' in raw['continuous']:
        raw['continuous']['method'] = normalize_method(raw['continuous']['method'])
    if group is None and raw['continuous'].get('method') == 'source_only':
        group = 'A_source_only'
    if group is not None:
        if group not in CONTINUOUS_GROUPS:
            raise ValueError(f'Unknown continuous ablation/sensitivity group: {group}')
        group_method, bank_enabled, mix_mode = CONTINUOUS_GROUPS[group]
        if method is not None and normalize_method(method) != group_method:
            raise ValueError('The method does not match the continuous ablation group')
        raw['continuous']['method'] = group_method
        raw['continuous_ablation_group'] = group
        raw['data_mix']['mix_mode'] = mix_mode
        if mix_mode == 'hard_only':
            raw['data_mix']['source_sample_count'] = 0
        bank = raw.setdefault('model_bank', {})
        bank.update(enabled=bank_enabled, query_on_train_start=bank_enabled,
                    rejection_fallback=bank_enabled)
        if not bank_enabled:
            bank['deploy_competition'] = False
        if group in CONTINUOUS_SENSITIVITY_PARAMETERS:
            raw['continuous'].update(CONTINUOUS_SENSITIVITY_PARAMETERS[group])
            raw['continuous_sensitivity_family'] = (
                'joint' if group in CONTINUOUS_JOINT_PARAMETERS else 'alpha')
    validate_settings(raw['continuous'])
    videos = resolve_path(raw['paths']['videos_dir'], base)
    output = resolve_path(raw['paths']['work_dir'], base)
    if videos == output or videos.is_relative_to(output):
        raise ValueError('The output directory must not contain the input video directory')
    if raw['data_mix']['mix_mode'] not in ('source_plus_hard', 'hard_only'):
        raise ValueError('Unsupported data_mix.mix_mode')
    if not 0 < float(raw['data_mix']['target_val_fraction']) <= .5:
        raise ValueError('A nonzero target holdout is required for deployment validation')
    if int(raw['data_mix']['min_target_val_images']) < 1:
        raise ValueError('min_target_val_images must be positive')
    if int(raw['continuous']['min_batch_images']) <= int(raw['data_mix']['min_target_val_images']):
        raise ValueError('Upload batches must leave both training and holdout frames')
    storage = raw['storage']
    if float(storage['max_work_dir_gb']) <= 0 or float(storage['min_free_disk_gb']) < 0:
        raise ValueError('Invalid disk budget')
    if int(raw.get('model_bank', {}).get('top_k', 10)) > 10:
        raise ValueError('The 20 GB profile supports at most 10 historical models')
    if raw['models'].get('train_options', {}).get('cache', False) is not False:
        raise ValueError('The bounded-storage profile requires models.train_options.cache=false')
    if raw['models'].get('workers', 0) != 0:
        raise ValueError('The synchronous profile requires models.workers=0')
    return raw, base, videos, output


def evaluation_datasets(raw: dict, base: Path) -> list[tuple[str, Path]]:
    return [(str(item['name']), resolve_path(item['root'], base))
            for item in raw.get('final_evaluation', {}).get('datasets', [])]


def experiment_parameters(raw: dict) -> dict:
    """Export effective values without changing configuration/checkpoint signatures."""
    settings, mix = raw.get('continuous', {}), raw.get('data_mix', {})
    return {
        'p_low': settings.get('p_low', .20),
        'non_lie_ratio_threshold': settings.get('non_lie_ratio_threshold', .30),
        'alpha': settings.get('alpha', .05), 'seed': raw.get('seed', 42),
        'sample_every_frames': settings.get('sample_every_frames'),
        'sample_fps': settings.get('sample_fps'),
        'short_window_seconds': settings.get('short_window_seconds'),
        'warmup_seconds': settings.get('warmup_seconds'),
        'min_batch_images': settings.get('min_batch_images'),
        'max_batch_images': settings.get('max_batch_images'),
        'min_samples_per_drift_window': settings.get('min_samples_per_drift_window', 0),
        'mix_mode': mix.get('mix_mode'), 'source_sample_count': mix.get('source_sample_count', 0),
        'model_bank_enabled': raw.get('model_bank', {}).get('enabled', True),
    }


class _VideoReadFailure(RuntimeError):
    """A video-only failure; inference, training and storage exceptions propagate."""


class YoloBackend:
    def __init__(self, config_path: Path, raw: dict, base: Path, root: Path, budget: DiskBudget):
        import cv2
        import pandas as pd
        import torch
        from ultralytics import YOLO
        from .config import Experiment, ExpPaths, resolve_device, set_seed
        from .runner import get_behavior_names

        self.cv2, self.pd, self.torch, self.YOLO = cv2, pd, torch, YOLO
        self.exp = Experiment(raw=raw, config_path=config_path.resolve(), paths=ExpPaths(
            dataset_root=resolve_path(raw['paths']['dataset_root'], base),
            source_yaml=resolve_path(raw['paths']['source_yaml'], base), work_dir=root,
            initial_student=resolve_path(raw['paths']['initial_student'], base),
            teacher=resolve_path(raw['paths']['teacher'], base)))
        self.raw, self.base, self.root, self.budget = raw, base, root, budget
        self.names = get_behavior_names(self.exp)
        if raw['data_mix']['mix_mode'] == 'source_plus_hard':
            from .dataset import load_source_info
            from .files import list_images
            source_images, source_labels, _, _, _ = load_source_info(self.exp.paths.source_yaml)
            images = list_images(source_images)
            if not images or not source_labels.is_dir():
                raise RuntimeError(f'Source replay dataset is missing images/labels: {source_images}')
        self.device = resolve_device(raw['models'].get('device', 'auto'))
        self.imgsz = int(raw['models']['imgsz'])
        self.model = None
        self.loaded_bundle = None
        set_seed(int(raw.get('seed', 42)))

    def bundle_path(self, bundle: str) -> Path:
        path = (self.root / bundle).resolve()
        if not path.is_relative_to((self.root / 'bundles').resolve()):
            raise ValueError(f'Invalid model bundle: {bundle}')
        return path

    def initialize(self) -> str:
        bundle = self.root / 'bundles' / 'b000000'
        (bundle / 'models').mkdir(parents=True, exist_ok=True)
        (bundle / 'model_bank').mkdir(exist_ok=True)
        shutil.copy2(self.exp.paths.initial_student, bundle / 'models/current_student.pt')
        return 'bundles/b000000'

    def frames(self, entry, start: int):
        """Yield sampled frames and video_event records, retrying each failed cursor once."""
        cv2 = self.cv2
        logger = logging.getLogger(__name__)
        cap = None
        fps = count = None
        index = start
        retry_index = None
        phase = 'open'
        read_errors = (_VideoReadFailure, getattr(cv2, 'error', _VideoReadFailure),
                       OSError, ValueError, OverflowError)

        def close_capture():
            nonlocal cap
            if cap is not None:
                cap.release()
                cap = None

        def open_capture():
            nonlocal cap, fps, count, phase
            phase = 'open'
            cap = cv2.VideoCapture(entry.path)
            if not cap.isOpened():
                raise _VideoReadFailure('Cannot open video')
            phase = 'header'
            new_fps = float(cap.get(cv2.CAP_PROP_FPS))
            new_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(new_fps) or new_fps <= 0:
                raise _VideoReadFailure('Invalid FPS in video header')
            if fps is not None and not math.isclose(fps, new_fps, abs_tol=1e-6):
                raise _VideoReadFailure('FPS changed when reopening the video')
            fps = new_fps
            count = int(new_count) if math.isfinite(new_count) and new_count > 0 else None
            if index:
                phase = 'seek'
                success = cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                if not success or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - index) > .5:
                    close_capture()
                    cap = cv2.VideoCapture(entry.path)
                    if not cap.isOpened():
                        raise _VideoReadFailure('Cannot reopen video for sequential cursor restore')
                    for cursor in range(index):
                        if not cap.grab():
                            raise _VideoReadFailure(
                                f'Cannot restore cursor {index}: sequential grab failed at {cursor}')

        def event(kind, reason, error):
            offset = index / fps if fps is not None else 0.
            shortfall = max(0, count - index) if count is not None else None
            from_header = shortfall is not None and fps is not None
            remaining = shortfall / fps if from_header else max(0., entry.end - entry.start - offset)
            return {'video_event': {
                'type': kind, 'video_path': entry.path, 'frame_index': index,
                'source_fps': fps, 'header_frame_count': count,
                'header_shortfall_frames': shortfall, 'recorded_time': entry.start + offset,
                'unprocessed_seconds_estimate': remaining,
                'estimate_source': 'video_header' if from_header else 'filename',
                'reason': reason, 'error': error}}

        try:
            while True:
                try:
                    if cap is None:
                        open_capture()
                    phase = 'read'
                    ok, image = cap.read()
                    ok = ok and image is not None
                    error = 'Decoder returned no frame'
                except read_errors as exc:
                    ok, image = False, None
                    error = f'{type(exc).__name__}: {exc}'
                else:
                    if not ok and index > 0 and (
                            (count is not None and index >= count) or
                            (count is None and index / fps >= entry.end - entry.start)):
                        return  # ordinary EOF, not an abnormal video event
                if not ok:
                    failed_phase = phase
                    close_capture()
                    if retry_index != index:
                        retry_index = index
                        report = event('video_decode_retry', failed_phase, error)
                        logger.warning('Video decode retry at %d/%s: %s; reopening once: %s',
                                       index, count, error, entry.path)
                        yield report
                        continue
                    small_tail = (failed_phase == 'read' and index > 0 and fps is not None and
                                  ((count is not None and
                                    0 < count - index <= max(2, math.ceil(fps))) or
                                   (count is None and
                                    0 <= entry.end - entry.start - index / fps <= 1.)))
                    kind = 'video_tail_shortfall' if small_tail else 'video_decode_skip'
                    report = event(kind, failed_phase, error)
                    logger.warning(
                        '%s at %d/%s (unprocessed estimate %.3f s): %s; '
                        'continuing to the next video without synthesizing frames: %s',
                        kind, index, count, report['video_event']['unprocessed_seconds_estimate'],
                        error, entry.path)
                    yield report
                    return
                if retry_index == index:
                    yield event('video_decode_recovered', 'read', '')
                retry_index = None
                settings = self.raw['continuous']
                if sample_frame(index, fps, settings.get('sample_fps'),
                                every_n_frames=settings.get('sample_every_frames')):
                    yield {'index': index, 'fps': fps, 'image': image}
                index += 1
        finally:
            close_capture()

    def predict(self, frame: dict, bundle: str) -> dict:
        from .sampling import _behavior_counts
        from .yolo_ops import image_predict_behavior_stats_from_array
        if self.model is None or self.loaded_bundle != bundle:
            self.release()
            self.model = self.YOLO(str(self.bundle_path(bundle) / 'models/current_student.pt'))
            self.model.to(self.device)
            self.loaded_bundle = bundle
        stats = image_predict_behavior_stats_from_array(
            self.model, frame['image'], self.imgsz, len(self.names))
        stats.update(_behavior_counts(stats['behavior_hist'], stats['det_count']))
        return stats

    def save_frame(self, frame: dict, sample_index: int) -> Path:
        ok, encoded = self.cv2.imencode('.jpg', frame['image'],
                                        [self.cv2.IMWRITE_JPEG_QUALITY,
                                         int(self.raw['storage'].get('jpeg_quality', 90))])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        self.budget.check(len(encoded))
        directory = self.root / 'pending'
        directory.mkdir(exist_ok=True)
        path = directory / f'frame_{sample_index:012d}.jpg'
        with path.open('wb') as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        self.budget.used += len(encoded)
        return path

    def release(self) -> None:
        self.model = None
        self.loaded_bundle = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _copy_bank(self, original: Path, destination: Path) -> None:
        shutil.copytree(original, destination)
        table = destination / 'model_bank.csv'
        if table.exists():
            bank = self.pd.read_csv(table)
            bank['model_path'] = bank['model_path'].map(lambda value: str(destination / Path(value).name))
            bank.to_csv(table, index=False, encoding='utf-8-sig')

    def update(self, update_id: int, rows: list[dict], old_bundle: str, model_version: int) -> dict:
        from .runner import (_run_update, _split_target_train_val, extract_scene_fingerprint,
                             mark_model_used, query_model_bank, shadow_validate)
        new_bundle = f'bundles/b{update_id:06d}'
        target = self.bundle_path(new_bundle)
        receipt = target / 'result.json'
        if receipt.exists():
            result = json.loads(receipt.read_text(encoding='utf-8'))
            if result['input_bundle'] != old_bundle:
                raise RuntimeError('Prepared update does not match the checkpoint deployment')
            if sha256_file(target / 'models/current_student.pt') != result['model_sha256']:
                raise RuntimeError('Prepared model checksum mismatch')
            return result
        if target.exists():
            remove_owned(target, self.root)  # failed, uncommitted transaction only
        target.mkdir(parents=True)
        for name in ('models', 'candidate_models', 'reports'):
            (target / name).mkdir()
        original = self.bundle_path(old_bundle)
        baseline = original / 'models/current_student.pt'
        current = target / 'models/current_student.pt'
        shutil.copy2(baseline, current)
        self._copy_bank(original / 'model_bank', target / 'model_bank')
        images = [Path(row['path']) for row in rows]
        fingerprint = extract_scene_fingerprint(self.pd.DataFrame(rows))
        cfg = self.raw.get('model_bank', {})
        paper_bank = str(cfg.get('policy', 'legacy')).strip().lower() == 'paper'
        query = {'model_bank_query_hit': False}
        started = time.monotonic()
        if cfg.get('enabled', True) and cfg.get('query_on_train_start', True):
            match, info = query_model_bank(
                target / 'model_bank', fingerprint, str(current), stage='stream',
                similarity_threshold=None if paper_bank else float(cfg.get('similarity_threshold', .8)),
                stage_filter=False,
                require_proven=False if paper_bank else bool(cfg.get('require_proven', False)),
                selector='similarity' if paper_bank else str(cfg.get('selector', 'similarity')),
                mark_used=not paper_bank, device=self.device)
            if match is not None:
                train_part, _ = _split_target_train_val(
                    images, float(self.raw['data_mix']['target_val_fraction']),
                    int(self.raw['data_mix']['min_target_val_images']),
                    int(self.raw.get('seed', 42)), update_id)
                _, check = shadow_validate(baseline, match, train_part, self.imgsz, self.device)
                gain = float(check['conf_gain'])
                approved = gain > (0.0 if paper_bank else float(cfg.get('swap_target_margin', .01)))
                query.update({'model_bank_query_hit': approved, 'model_bank_query_path': str(match),
                              'model_bank_query_similarity': info.get('query_similarity', 0),
                              'model_bank_swap_target_gain': gain})
                if approved:
                    shutil.copy2(match, current)
        if paper_bank:
            # The deployed model can itself be a copy of a bank model. Identify
            # the actual initialization by contents, including when retrieval
            # did not replace the deployment, so fallback excludes that model.
            bank_table = target / 'model_bank/model_bank.csv'
            init_hash = sha256_file(current)
            bank_paths = (self.pd.read_csv(bank_table)['model_path'].tolist()
                          if bank_table.exists() else [])
            query['model_bank_initialization_paths'] = [
                str(path) for path in map(Path, bank_paths)
                if path.is_file() and sha256_file(path) == init_hash]
            for path in query['model_bank_initialization_paths']:
                mark_model_used(target / 'model_bank', path)
        result = _run_update(
            exp=self.exp, raw=self.raw,
            method='full_upload' if self.raw.get('continuous', {}).get('method') == 'full_upload' else 'adaptive',
            method_variant='continuous_' + self.raw.get('continuous', {}).get('method', 'bccl'),
            mix_mode=self.raw['data_mix']['mix_mode'], stage='stream', stage_index=update_id,
            episode_tag='batch', current_model=current, out_dir=target,
            train_images=images, scene_fp=fingerprint, names=self.names, imgsz=self.imgsz,
            source_sample_count=int(self.raw['data_mix']['source_sample_count']),
            device=self.device, deploy_mode='full', deployed_baseline=baseline,
            run_deploy_competition=False if paper_bank else bool(cfg.get('deploy_competition', True)),
            bank_query_info=query,
            decision_extra={'update_id': update_id, 'input_model_version': model_version,
                            'window_floor_images': sum(row.get('selection_reason') == 'window_floor' for row in rows)})
        prepared = {'bundle': new_bundle, 'input_bundle': old_bundle,
                    'model_changed': sha256_file(current) != sha256_file(baseline),
                    'model_sha256': sha256_file(current),
                    'accepted': bool(result['accepted']), 'uploaded_images': len(rows),
                    'window_floor_images': sum(row.get('selection_reason') == 'window_floor' for row in rows),
                    'sample_indices': [row['sample_index'] for row in rows],
                    'upload_bytes': result['upload_bytes'],
                    'train_seconds': result['train_seconds'],
                    'pseudo_label_seconds': result['pseudo_label_seconds'],
                    'update_seconds': time.monotonic() - started,
                    'decision': result['decision']}
        for name in ('train_runs', 'pseudo_labels', 'mixed_datasets', 'candidate_models'):
            remove_owned(target / name, self.root)
        atomic_json(receipt, prepared)  # ready to commit; current deployment is untouched
        return prepared

    def cleanup(self, active: str, previous: str) -> None:
        keep = {self.bundle_path(active), self.bundle_path(previous)}
        for path in (self.root / 'bundles').iterdir():
            if path.is_dir() and path.resolve() not in keep:
                remove_owned(path, self.root)

    def evaluate(self, bundle: str, state: dict) -> None:
        import yaml
        from .yolo_ops import evaluate_model
        model = self.bundle_path(bundle) / 'models/current_student.pt'
        reports = self.root / 'reports'
        rows = []
        if self.raw.get('final_evaluation', {}).get('enabled', True):
            for name, dataset in evaluation_datasets(self.raw, self.base):
                # Ultralytics writes label caches next to labels. Stage links/copies so
                # even those caches stay inside the output budget, including external datasets.
                staged = self.root / 'eval_input'
                remove_owned(staged, self.root)
                (staged / 'images/test').mkdir(parents=True)
                (staged / 'labels/test').mkdir(parents=True)
                inputs = [path for path in (dataset / 'images/test').iterdir()
                          if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]
                self.budget.check(sum(path.stat().st_size for path in inputs), refresh=True)
                for image in inputs:
                    label = dataset / 'labels/test' / f'{image.stem}.txt'
                    for source, destination in [(image, staged / 'images/test' / image.name),
                                                (label, staged / 'labels/test' / label.name)]:
                        if not source.exists():
                            continue  # absent label is an empty/background image in YOLO
                        try:
                            os.link(source, destination)
                        except OSError:
                            shutil.copy2(source, destination)
                data_yaml = reports / 'final_eval_data.yaml'
                data_yaml.write_text(yaml.safe_dump(
                    {'path': str(staged), 'train': 'images/test', 'val': 'images/test',
                     'test': 'images/test', 'nc': len(self.names), 'names': self.names},
                    allow_unicode=True), encoding='utf-8')
                metrics = evaluate_model(
                    model, staged, self.imgsz, device=self.device, names=self.names,
                    output_project=self.root / 'eval_cache', data_yaml_override=data_yaml)
                rows.append({'dataset': name, **metrics})
                remove_owned(staged, self.root)
        if rows:
            self.pd.DataFrame(rows).to_csv(reports / 'final_evaluation.csv', index=False,
                                           encoding='utf-8-sig')
        exported = self.root / 'final_student.pt'
        temporary = exported.with_suffix('.tmp')
        shutil.copy2(model, temporary)
        os.replace(temporary, exported)
        metric_keys = ('mAP50', 'mAP50_95', 'Precision', 'Recall',
                       'behavior_acc_iou50', 'behavior_recall_iou50')
        atomic_json(reports / 'final_summary.json', {
            'method': self.raw.get('continuous', {}).get('method', 'bccl'),
            'ablation_group': self.raw.get('continuous_ablation_group'),
            'sensitivity_family': self.raw.get('continuous_sensitivity_family'),
            'parameters': experiment_parameters(self.raw),
            'model': str(exported), 'model_sha256': sha256_file(model),
            'model_version': state['model_version'], 'updates': state['updates'],
            'wall_seconds_before_final_evaluation': state.get('wall_seconds', 0.),
            'drift_events': state['drift_events'], 'samples': state['samples'],
            'video_read_summary': {key: state.get(key, 0) for key in VIDEO_READ_SUMMARY_FIELDS},
            'dataset_count': len(rows), 'aggregation': 'macro_mean_across_datasets',
            'metrics': {key: sum(float(row.get(key, 0)) for row in rows) / len(rows)
                        for key in metric_keys} if rows else {},
            'num_eval_images': sum(int(row.get('num_eval_images', 0)) for row in rows)})
        remove_owned(self.root / 'eval_cache', self.root)


def run_continuous(config_path, *, videos_dir=None, work_dir=None, resume=False,
                   dry_run=False, max_samples=None, method=None, budget_root=None,
                   expected_manifest=None, group=None):
    raw, base, videos, root = load_configuration(config_path, videos_dir, work_dir, method, group)
    manifest = video_manifest(videos)
    if expected_manifest is not None and [asdict(item) for item in manifest] != expected_manifest:
        raise ValueError('Video files changed during the comparison batch; all methods need the same inputs')
    hours = sum(item.end - item.start for item in manifest) / 3600
    source_only = raw['continuous'].get('method') == 'source_only'
    required = [resolve_path(raw['paths'][key], base) for key in
                (('initial_student',) if source_only else ('initial_student', 'teacher', 'source_yaml'))]
    missing = [str(path) for path in required if not path.is_file()]
    missing_modules = [name for name in ('cv2', 'torch', 'ultralytics', 'pandas')
                       if importlib.util.find_spec(name) is None]
    tests = evaluation_datasets(raw, base)
    if raw.get('final_evaluation', {}).get('enabled', True):
        if not tests:
            raise ValueError('Configure independent final_evaluation.datasets before running')
        for name, dataset in tests:
            if not (dataset / 'images/test').is_dir() or not (dataset / 'labels/test').is_dir():
                missing.append(f'evaluation {name}: {dataset}/images/test + labels/test')
            elif not any(path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                         for path in (dataset / 'images/test').iterdir()):
                missing.append(f'evaluation {name}: empty test images directory {dataset}')
    owner = Path(budget_root).resolve() if budget_root else root
    if not root.is_relative_to(owner):
        raise ValueError('Method output must be inside its shared disk budget directory')
    sampling = raw['continuous']
    nominal_rate = (effective_inference_fps(sampling) if 'sample_every_frames' in sampling
                    else float(sampling['sample_fps']))
    sampling_summary = ({'sampling_mode': 'frame_stride',
                         'sample_every_frames': sampling['sample_every_frames'],
                         'assumed_source_fps': 25.,
                         'nominal_inference_fps': nominal_rate}
                        if 'sample_every_frames' in sampling else
                        {'sampling_mode': 'inference_rate',
                         'target_inference_fps': sampling['sample_fps']})
    summary = {'method': raw['continuous'].get('method', 'bccl'),
               'ablation_group': raw.get('continuous_ablation_group'),
               'sensitivity_family': raw.get('continuous_sensitivity_family'),
               'parameters': experiment_parameters(raw),
               'model_bank_enabled': raw.get('model_bank', {}).get('enabled', True),
               'mix_mode': raw['data_mix']['mix_mode'],
               'source_sample_count': raw['data_mix']['source_sample_count'],
               'videos': len(manifest), 'hours_from_filenames': round(hours, 4),
               'input_gib': round(sum(item.size for item in manifest) / 1024**3, 3),
               **sampling_summary,
               'min_samples_per_drift_window': sampling.get('min_samples_per_drift_window', 0),
               'approx_samples_from_filenames': 0 if source_only else round(hours * 3600 * nominal_rate),
               'work_dir': str(root), 'max_work_dir_gb': raw['storage']['max_work_dir_gb'],
               'disk_budget_directory': str(owner),
               'min_free_disk_gb': raw['storage']['min_free_disk_gb'],
               'missing_inputs': missing, 'missing_modules': missing_modules,
               'recording_order': [Path(item.path).name for item in manifest]}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return summary
    if missing or missing_modules:
        raise RuntimeError('Runtime preflight failed. Supply missing weights/datasets/dependencies '
                           'listed above. --dry-run needs no GPU packages.')
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root):
        budget = DiskBudget(owner, int(raw['storage']['max_work_dir_gb'] * 1e9),
                            int(raw['storage']['min_free_disk_gb'] * 1e9))
        budget.check(int(raw['continuous'].get('update_scratch_gb', 2) * 1e9), refresh=True)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(root / 'yolo_settings'))
        backend = YoloBackend(Path(config_path), raw, base, root, budget)
        signature = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        run = ContinuousRun(root, raw['continuous'], manifest, backend, signature,
                            resume=resume, budget=budget)
        try:
            # Only write after the saved signature/manifest have been validated.
            atomic_json(root / 'run_parameters.json', {
                'signature': signature, 'method': raw['continuous'].get('method', 'bccl'),
                'group': raw.get('continuous_ablation_group'),
                'sensitivity_family': raw.get('continuous_sensitivity_family'),
                'parameters': experiment_parameters(raw)})
            state = run.run(max_samples=max_samples)
            print(f'[continuous] complete={state["complete"]} samples={state["samples"]} '
                  f'updates={state["updates"]} model_version={state["model_version"]} '
                  f'skipped_videos={state["skipped_videos"]} '
                  f'skipped_seconds_estimate={state["skipped_seconds_estimate"]:.3f}', flush=True)
            return state
        finally:
            backend.release()


def write_batch_summary(root: Path, methods: list[str]) -> Path:
    rows = []
    for method in methods:
        directory = root / method
        state_path = directory / 'state.json'
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text(encoding='utf-8'))
        final_path = directory / 'reports/final_summary.json'
        final = json.loads(final_path.read_text(encoding='utf-8')) if final_path.exists() else {}
        parameter_path = directory / 'run_parameters.json'
        parameter_info = (json.loads(parameter_path.read_text(encoding='utf-8'))
                          if parameter_path.exists() else final)
        totals = {'uploaded_images': 0, 'window_floor_images': 0, 'upload_bytes': 0, 'train_seconds': 0.,
                  'pseudo_label_seconds': 0.}
        events = directory / 'reports/events.jsonl'
        if events.exists():
            with events.open(encoding='utf-8') as stream:
                for line in stream:
                    event = json.loads(line)
                    if event['type'] == 'update':
                        for key in totals:
                            totals[key] += event.get(key, 0)
        rows.append({'method': 'BCCL' if method == 'bccl' else method,
                     **parameter_info.get('parameters', {}),
                     'complete': state['complete'], 'evaluated': state['evaluated'],
                     'samples': state['samples'], 'updates': state['updates'],
                     'model_version': state['model_version'],
                     'wall_seconds': state.get('wall_seconds', 0.),
                     **{key: state.get(key, 0) for key in VIDEO_READ_SUMMARY_FIELDS},
                     **totals, **final.get('metrics', {})})
    output = root / 'comparison.csv'
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        temporary = output.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    return output


def run_continuous_batch(config_path, *, methods, videos_dir=None, work_dir=None,
                         resume=False, dry_run=False, max_samples=None):
    """Independent trajectories, sequential execution, one aggregate output budget."""
    selected = [normalize_method(method) for method in methods]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Specify at least one method, without duplicates')
    return _run_continuous_arms(
        config_path, [(method, method, None) for method in selected],
        videos_dir=videos_dir, work_dir=work_dir, resume=resume,
        dry_run=dry_run, max_samples=max_samples)


def run_continuous_ablation_batch(config_path, *, groups, videos_dir=None, work_dir=None,
                                  resume=False, dry_run=False, max_samples=None):
    selected = list(groups)
    if (not selected or len(set(selected)) != len(selected) or
            any(group not in CONTINUOUS_GROUPS for group in selected)):
        raise ValueError(f'Choose distinct continuous experiment groups from {tuple(CONTINUOUS_GROUPS)}')
    return _run_continuous_arms(
        config_path, [(group, CONTINUOUS_GROUPS[group][0], group) for group in selected],
        videos_dir=videos_dir, work_dir=work_dir, resume=resume,
        dry_run=dry_run, max_samples=max_samples)


def run_continuous_sensitivity_batch(config_path, *, sensitivity, videos_dir=None, work_dir=None,
                                     resume=False, dry_run=False, max_samples=None):
    if sensitivity not in CONTINUOUS_SENSITIVITY_BATCHES:
        raise ValueError(f'Unknown continuous sensitivity preset: {sensitivity}')
    return run_continuous_ablation_batch(
        config_path, groups=CONTINUOUS_SENSITIVITY_BATCHES[sensitivity],
        videos_dir=videos_dir, work_dir=work_dir, resume=resume,
        dry_run=dry_run, max_samples=max_samples)


def _run_continuous_arms(config_path, arms, *, videos_dir=None, work_dir=None,
                          resume=False, dry_run=False, max_samples=None):
    _, _, videos, root = load_configuration(config_path, videos_dir, work_dir)
    frozen_manifest = [asdict(item) for item in video_manifest(videos)]

    def execute():
        results = []
        for label, method, group in arms:
            directory = root / label
            continuing = resume and (directory / 'state.json').exists()
            print(f'[continuous-batch] method={label} resume={continuing}', flush=True)
            result = run_continuous(
                config_path, videos_dir=videos_dir, work_dir=str(directory),
                method=method, budget_root=root, resume=continuing,
                dry_run=dry_run, max_samples=max_samples, expected_manifest=frozen_manifest,
                **({'group': group} if group is not None else {}))
            results.append(result)
            if not dry_run:
                write_batch_summary(root, [label for label, _, _ in arms])
        return results

    if dry_run:
        return execute()
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root):
        frozen_path = root / 'batch_manifest.json'
        if frozen_path.exists():
            if json.loads(frozen_path.read_text(encoding='utf-8')) != frozen_manifest:
                raise ValueError('Input recordings differ from the saved comparison manifest')
        else:
            atomic_json(frozen_path, frozen_manifest)
        return execute()
