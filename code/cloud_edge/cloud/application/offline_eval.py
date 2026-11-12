"""Re-evaluate saved models without opening videos or changing adaptation runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'vendor'))

from pig_beh_ce.continuous import DiskBudget, atomic_json, remove_owned, sha256_file
from pig_beh_ce.continuous_runtime import evaluation_datasets, resolve_path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
METRICS = ('mAP50', 'mAP50_95', 'Precision', 'Recall',
           'behavior_acc_iou50', 'behavior_recall_iou50')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', required=True, help='Reads final_evaluation.datasets and model image size')
    result.add_argument('--experiments-dir', nargs='+', default=[],
                        help='Recursively find final_student.pt in completed experiment directories')
    result.add_argument('--groups', nargs='+', help='Select experiment folder names, case insensitive')
    result.add_argument('--model', action='append', default=[], metavar='NAME=PATH',
                        help='Add a fixed model, e.g. A_source_only=./weights/student_yolov8n_behavior.pt')
    result.add_argument('--output-dir', required=True, help='A new directory, separate from training outputs')
    result.add_argument('--device', help='Override models.device, e.g. 0 or cpu')
    result.add_argument('--dry-run', action='store_true', help='Check model/data paths without importing GPU packages')
    return result


def discover_models(directories, explicit, groups=None):
    selected = {name.casefold() for name in groups} if groups else None
    models = {}

    def add(name, path):
        if selected is not None and name.casefold() not in selected:
            return
        if not re.fullmatch(r'[\w.-]+', name) or name in ('.', '..'):
            raise ValueError(f'Unsafe experiment name: {name}')
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Model not found: {path}')
        key = name.casefold()
        if key in models and models[key][1] != path:
            raise ValueError(f'Duplicate experiment name {name}; choose separate roots or explicit unique names')
        models[key] = (name, path)

    for directory in directories:
        root = Path(directory).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'Experiment directory not found: {root}')
        for path in sorted(root.rglob('final_student.pt')):
            if selected is not None and path.parent.name.casefold() not in selected:
                continue
            state_path = path.parent / 'state.json'
            if state_path.exists() and not json.loads(state_path.read_text(encoding='utf-8')).get('complete'):
                raise ValueError(f'Adaptation is not complete: {path.parent}')
            add(path.parent.name, path)
    for item in explicit:
        name, separator, value = item.partition('=')
        if not separator or not value:
            raise ValueError('--model must be NAME=PATH')
        add(name, Path(value))
    if selected is not None and selected - models.keys():
        raise ValueError('No saved final model for: ' + ', '.join(sorted(selected - models.keys())))
    if not models:
        raise ValueError('No models selected. Supply --experiments-dir or --model NAME=PATH.')
    return list(models.values())


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def stage_dataset(dataset, staged):
    for split in ('images/test', 'labels/test'):
        (staged / split).mkdir(parents=True)
    for image in sorted((dataset / 'images/test').iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        for source, destination in (
            (image, staged / 'images/test' / image.name),
            (dataset / 'labels/test' / f'{image.stem}.txt', staged / 'labels/test' / f'{image.stem}.txt'),
        ):
            if source.is_file():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)


def run_evaluation(args, evaluator=None):
    config = Path(args.config).resolve()
    raw = yaml.safe_load(config.read_text(encoding='utf-8'))
    base = config.parent.parent if config.parent.name == 'configs' else Path.cwd()
    datasets = evaluation_datasets(raw, base)
    if not datasets or len({name for name, _ in datasets}) != len(datasets):
        raise ValueError('Configure a nonempty final_evaluation.datasets list with unique names')
    for name, dataset in datasets:
        if not (dataset / 'images/test').is_dir() or not (dataset / 'labels/test').is_dir():
            raise FileNotFoundError(f'{name}: expected images/test and labels/test inside {dataset}')
        if not any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                   for path in (dataset / 'images/test').iterdir()):
            raise ValueError(f'No evaluation images: {dataset}')
    models = discover_models(args.experiments_dir, args.model, args.groups)
    output = Path(args.output_dir).resolve()
    protected = [path.parent for _, path in models] + [path for _, path in datasets]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError('Output must be separate from model and evaluation-data directories')
    if output.exists():
        raise ValueError(f'Output already exists; select a new --output-dir: {output}')
    names = {0: 'stand', 1: 'lie', 2: 'eat'}
    source_yaml = raw.get('paths', {}).get('source_yaml')
    if source_yaml and resolve_path(source_yaml, base).is_file():
        source = yaml.safe_load(resolve_path(source_yaml, base).read_text(encoding='utf-8'))
        names = source.get('names', names)
    names = ({i: str(value) for i, value in enumerate(names)} if isinstance(names, list)
             else {int(key): str(value) for key, value in names.items()})
    model_settings = raw.get('models', {})
    imgsz = int(model_settings.get('imgsz', 640))
    device = args.device if args.device is not None else model_settings.get('device', 'auto')
    plan = {'evaluation_only': True, 'config': str(config), 'output_dir': str(output),
            'imgsz': imgsz, 'device': device, 'behavior_conf': .25, 'names': names,
            'models': [{'name': name, 'path': str(path)} for name, path in models],
            'datasets': [{'name': name, 'root': str(path)} for name, path in datasets]}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return plan
    # The plan needs no video directory, teacher, training packages, or checkpoint signature.
    output.mkdir(parents=True, exist_ok=False)
    if evaluator is None:
        os.environ.setdefault('YOLO_CONFIG_DIR', str(output / 'yolo_settings'))
        # Import only after validating all inputs; --dry-run works without torch/YOLO.
        from pig_beh_ce.yolo_ops import evaluate_model
        evaluator = evaluate_model
    storage = raw.get('storage', {})
    budget = DiskBudget(output, int(float(storage.get('max_work_dir_gb', 18)) * 1e9),
                        int(float(storage.get('min_free_disk_gb', 2)) * 1e9))
    budget.check(refresh=True)
    atomic_json(output / 'evaluation_config.json', plan)
    comparison = []
    for method, model in models:
        started = time.monotonic()
        model_hash = sha256_file(model)
        reports = output / method / 'reports'
        reports.mkdir(parents=True)
        staged, cache = output / '_eval_input', output / '_eval_cache'
        rows = []
        for name, dataset in datasets:
            print(f'[offline-eval] model={method} dataset={name}', flush=True)
            inputs = [path for path in (dataset / 'images/test').iterdir()
                      if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
            labels = [dataset / 'labels/test' / f'{path.stem}.txt' for path in inputs]
            # Reserve the full copy size even though hard links normally avoid image copies.
            budget.check(sum(path.stat().st_size for path in inputs + labels if path.is_file()), refresh=True)
            try:
                stage_dataset(dataset, staged)
                data_yaml = staged / 'data.yaml'
                data_yaml.write_text(yaml.safe_dump(
                    {'path': str(staged), 'train': 'images/test', 'val': 'images/test',
                     'test': 'images/test', 'nc': len(names), 'names': names},
                    allow_unicode=True), encoding='utf-8')
                metrics = evaluator(model, staged, imgsz=imgsz, conf=.25, device=device,
                                    names=names, output_project=cache, data_yaml_override=data_yaml)
                rows.append({'dataset': name, **metrics})
                write_csv(reports / 'final_evaluation.csv', rows)
            finally:
                remove_owned(staged, output)
                remove_owned(cache, output)
        if sha256_file(model) != model_hash:
            raise RuntimeError(f'Model changed during offline evaluation: {model}')
        summary = {'evaluation_only': True, 'method': method, 'model': str(model),
                   'model_sha256': model_hash, 'dataset_count': len(rows),
                   'aggregation': 'macro_mean_across_datasets',
                   'metrics': {key: sum(float(row[key]) for row in rows) / len(rows) for key in METRICS},
                   'num_eval_images': sum(int(row['num_eval_images']) for row in rows),
                   'evaluation_wall_seconds': time.monotonic() - started}
        atomic_json(reports / 'final_summary.json', summary)
        comparison.append({key: value for key, value in summary.items() if key != 'metrics'}
                          | summary['metrics'])
        write_csv(output / 'comparison.csv', comparison)
    return comparison


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        run_evaluation(args)
    except (ValueError, FileNotFoundError) as error:
        command.error(str(error))


if __name__ == '__main__':
    main()
