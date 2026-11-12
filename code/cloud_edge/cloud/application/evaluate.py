"""Re-evaluate cloud final models with a new test configuration, without adaptation."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import yaml
from .config import algorithm, cloud_raw, read_yaml
from . import offline_eval
from pig_beh_ce.continuous import atomic_json


def main():
    parser = offline_eval.parser()
    parser.description = __doc__
    args = parser.parse_args()
    host, config_path = read_yaml(args.config)
    raw = cloud_raw(algorithm(host, config_path), host, config_path, Path(args.output_dir).resolve())
    with tempfile.TemporaryDirectory(prefix="cloud_edge_eval_") as directory:
        snapshot = Path(directory) / "evaluation.yaml"
        snapshot.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        args.config = str(snapshot)
        offline_eval.run_evaluation(args)
        if not args.dry_run:
            output = Path(args.output_dir).resolve()
            saved = output / "evaluation_inputs.yaml"
            saved.write_text(snapshot.read_text(encoding="utf-8"), encoding="utf-8")
            plan = json.loads((output / "evaluation_config.json").read_text(encoding="utf-8"))
            plan.update(config=str(saved), cloud_config=str(config_path))
            atomic_json(output / "evaluation_config.json", plan)


if __name__ == "__main__":
    main()
