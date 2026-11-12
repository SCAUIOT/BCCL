"""Video inference on the edge; synchronous adaptation over HTTP."""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from .config import (algorithm, algorithm_hash, digest, identifier, local_path,
                     read_yaml, token_from)
from .profiles import apply_profile, profile_hash, select_profiles
from .model_change import CloudClient
from pig_beh_ce.continuous import (ContinuousRun, DiskBudget, atomic_json, output_lock,
                                   remove_owned, sha256_file, video_manifest)
from pig_beh_ce.continuous_runtime import YoloBackend, experiment_parameters


class RemoteEdgeBackend(YoloBackend):
    """Reuse only decoding, student prediction and JPEG writing from YoloBackend."""
    def __init__(self, root, shared, profile, config, config_path, budget, client, session):
        # Do not call YoloBackend.__init__: no source data or teacher on the edge.
        self.root, self.raw, self.budget = root, apply_profile(shared, profile), budget
        self.config, self.config_path, self.client, self.session = config, config_path, client, session
        self.sid = session["session_id"]
        self.names = {int(k): v for k, v in session["class_names"].items()}
        self.imgsz = int(shared["models"]["imgsz"])
        self.model, self.loaded_bundle = None, None

    def _runtime(self):
        if hasattr(self, "torch"):
            return
        import cv2
        import torch
        from ultralytics import YOLO
        from pig_beh_ce.config import resolve_device, set_seed
        self.cv2, self.torch, self.YOLO = cv2, torch, YOLO
        self.device = resolve_device(self.config.get("device", "auto"))
        set_seed(int(self.raw.get("seed", 42)))

    def initialize(self):
        model = self.root / "bundles/b000000/models/current_student.pt"
        candidate = self.config.get("paths", {}).get("initial_student")
        candidate = local_path(candidate, self.config_path) if candidate else None
        if candidate and candidate.is_file():
            if sha256_file(candidate) != self.session["initial_sha"]:
                raise RuntimeError("Local edge student differs from the cloud student")
            model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, model)
        else:
            self.client.download(self.sid, "initial", model, self.session["initial_sha"])
        return "bundles/b000000"

    def reconcile(self, state):
        remote = self.client.state(self.sid)
        local_sha = sha256_file(self.bundle_path(state["bundle"]) / "models/current_student.pt")
        if remote["updates"] < state["updates"]:
            raise RuntimeError("Cloud state is older than edge state; restore the matching cloud work directory")
        if remote["updates"] > state["updates"] + int(bool(state["inflight"])):
            raise RuntimeError("Cloud is ahead of this edge checkpoint; use the matching edge work directory")
        if remote["updates"] == state["updates"]:
            if remote["model_sha256"] != local_sha:
                raise RuntimeError("Edge and cloud deployment checksums differ")
            self.client.acknowledge(self.sid, state["updates"], local_sha)

    def frames(self, entry, start):
        self._runtime()
        yield from super().frames(entry, start)

    def predict(self, frame, bundle):
        self._runtime()
        return super().predict(frame, bundle)

    def save_frame(self, frame, sample_index):
        self._runtime()
        return super().save_frame(frame, sample_index)

    def release(self):
        if hasattr(self, "torch"):
            super().release()

    def update(self, update_id, rows, old_bundle, model_version):
        target = self.bundle_path(f"bundles/b{update_id:06d}")
        receipt = target / "result.json"
        if receipt.exists():
            result = json.loads(receipt.read_text(encoding="utf-8"))
            if result["input_bundle"] != old_bundle or sha256_file(target / "models/current_student.pt") != result["model_sha256"]:
                raise RuntimeError("Local prepared deployment does not match this checkpoint")
            return result
        baseline = self.bundle_path(old_bundle) / "models/current_student.pt"
        self.client.acknowledge(self.sid, update_id - 1, sha256_file(baseline))
        target.mkdir(parents=True, exist_ok=True)
        job = self.client.submit_update(self.sid, rows, {
            "update_id": update_id, "model_version": model_version,
            "input_model_sha256": sha256_file(baseline),
            "algorithm_hash": self.session["algorithm_hash"]})
        result = self.client.wait(self.sid, job)
        if result["input_bundle"] != old_bundle or result["sample_indices"] != [r["sample_index"] for r in rows]:
            raise RuntimeError("Cloud receipt differs from the submitted update")
        expected_bytes = sum(Path(r["path"]).stat().st_size for r in rows)
        if result["upload_bytes"] != expected_bytes:
            raise RuntimeError("Cloud upload byte count differs from the edge batch")
        self.client.download(self.sid, f"update-{update_id}", target / "models/current_student.pt",
                             result["model_sha256"])
        result["bundle"] = f"bundles/b{update_id:06d}"
        result["remote_session_id"] = self.sid
        atomic_json(receipt, result)
        return result

    def cleanup(self, active, previous):
        keep = {self.bundle_path(active), self.bundle_path(previous)}
        for path in (self.root / "bundles").iterdir():
            if path.is_dir() and path.resolve() not in keep:
                remove_owned(path, self.root)
        update_id = int(Path(active).name[1:])
        model = self.bundle_path(active) / "models/current_student.pt"
        self.client.acknowledge(self.sid, update_id, sha256_file(model))

    def evaluate(self, bundle, state):
        model_sha = sha256_file(self.bundle_path(bundle) / "models/current_student.pt")
        self.client.acknowledge(self.sid, state["updates"], model_sha)
        job = self.client.finalize(self.sid, {**{k: v for k, v in state.items() if k not in
                                             ("manifest", "pending", "monitor", "window_selections")},
                                           "model_sha256": model_sha})
        result = self.client.wait(self.sid, job)
        if result["model_sha256"] != model_sha:
            raise RuntimeError("Final evaluation model differs from the deployed edge model")
        self.client.download(self.sid, "final_student.pt", self.root / "final_student.pt", model_sha)
        reports = self.root / "reports"
        if result["has_evaluation_csv"]:
            self.client.download(self.sid, "final_evaluation.csv", reports / "final_evaluation.csv")
        summary = dict(result["summary"])
        summary.update(cloud_model=summary["model"], model=str(self.root / "final_student.pt"),
                       remote_session_id=self.sid, deployment_location="edge", evaluation_location="cloud")
        remote = self.client.state(self.sid)
        summary["upload_totals"] = {k: remote[k] for k in ("uploaded_images", "upload_bytes", "wire_payload_bytes")}
        atomic_json(reports / "final_summary.json", summary)


def comparison(root):
    rows = []
    for state_path in sorted(root.glob("*/state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        summary_path = state_path.parent / "reports/final_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        profile = json.loads((state_path.parent / "profile.json").read_text(encoding="utf-8"))
        totals = {k: 0 for k in ("uploaded_images", "upload_bytes", "wire_payload_bytes", "train_seconds", "pseudo_label_seconds")}
        events = state_path.parent / "reports/events.jsonl"
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event["type"] == "update":
                    for key in totals:
                        totals[key] += event.get(key, 0)
        rows.append({**profile, "complete": state["complete"], "evaluated": state["evaluated"],
                     "samples": state["samples"], "updates": state["updates"],
                     "model_version": state["model_version"], **totals,
                     "upload_mib": totals["upload_bytes"] / 1024**2, **summary.get("metrics", {})})
    if rows:
        target = root / "comparison.csv"
        fields = list(dict.fromkeys(k for row in rows for k in row))
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)


def run(config_path, *, suite="methods", names=None, resume=False, plan=False,
        dry_run=False, max_samples=None, run_id=None):
    config, path = read_yaml(config_path)
    shared = algorithm(config, path)
    profiles = select_profiles(shared, suite, names)
    if plan:
        print(json.dumps({"suite": suite, "arm_count": len(profiles), "profiles": [p.to_dict() for p in profiles]},
                         ensure_ascii=False, indent=2))
        return
    node = identifier(config["edge"]["node_id"])
    run_id = identifier(run_id or config["edge"]["run_id"])
    videos = local_path(config["paths"]["videos_dir"], path)
    manifest = video_manifest(videos)
    data = [asdict(e) for e in manifest]
    stream_signature = digest(data)
    if dry_run:
        print(json.dumps({"videos": len(data), "video_hours": sum(e.end - e.start for e in manifest) / 3600,
                          "cloud_url": config["network"]["cloud_url"], "profiles": [p.to_dict() for p in profiles]},
                         ensure_ascii=False, indent=2))
        return
    token = token_from(config) if config["network"].get("require_token", False) else ""
    output_base = local_path(config["paths"]["work_dir"], path)
    if videos == output_base or videos.is_relative_to(output_base):
        raise ValueError("Output directory must not contain the input recordings")
    batch_root = output_base / run_id
    batch_root.mkdir(parents=True, exist_ok=True)
    budget = DiskBudget(output_base, int(shared["storage"]["max_work_dir_gb"] * 1e9),
                         int(shared["storage"]["min_free_disk_gb"] * 1e9))
    client = CloudClient(config["network"], token)
    try:
        with output_lock(batch_root):
            batch_record = {"node_id": node, "run_id": run_id, "manifest": data,
                            "algorithm_hash": algorithm_hash(shared)}
            saved = batch_root / "batch_manifest.json"
            if saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != batch_record:
                raise ValueError("Batch inputs/settings changed; choose a new run_id")
            atomic_json(saved, batch_record)
            for profile in profiles:
                root = batch_root / profile.arm
                root.mkdir(exist_ok=True)
                if (root / "state.json").exists() and not resume:
                    raise ValueError(f"Existing {profile.arm}; use --resume to reuse or continue it")
                initial = config["paths"].get("initial_student")
                initial = local_path(initial, path) if initial else None
                session = client.create_session({"node_id": node, "run_id": run_id, "arm": profile.arm,
                              "algorithm_hash": profile_hash(shared, profile),
                              "stream_signature": stream_signature, "manifest": data,
                              "edge_ip": config["edge"].get("ip", ""),
                              "initial_student_sha": sha256_file(initial) if initial and initial.is_file() else None})
                if not (root / "state.json").exists() and session["updates"]:
                    raise ValueError("Cloud already has updates but local checkpoint is missing; restore edge state or use a new run_id")
                backend = RemoteEdgeBackend(root, shared, profile, config, path, budget, client, session)
                atomic_json(root / "profile.json", profile.to_dict())
                atomic_json(root / "session.json", {"session_id": session["session_id"],
                            "cloud_url": config["network"]["cloud_url"], "algorithm_hash": session["algorithm_hash"]})
                raw = apply_profile(shared, profile)
                atomic_json(root / "run_parameters.json", {"profile": profile.to_dict(),
                            "parameters": experiment_parameters(raw)})
                signature = digest({"algorithm_hash": profile_hash(shared, profile), "stream": stream_signature,
                                    "initial_sha": session["initial_sha"], "session_id": session["session_id"]})
                run = ContinuousRun(root, raw["continuous"], manifest, backend, signature,
                                    resume=bool(resume and (root / "state.json").exists()), budget=budget)
                backend.reconcile(run.state)
                run.run(max_samples=max_samples)
                comparison(batch_root)
    finally:
        client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/edge.yaml")
    p.add_argument("--suite", choices=("methods", "ablation", "joint", "alpha", "all"), default="methods")
    p.add_argument("--arms", nargs="+", help="Specific method, ablation or sensitivity IDs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan", action="store_true", help="Print experiment parameters without network, videos or ML imports")
    p.add_argument("--dry-run", action="store_true", help="Also validate video names/manifest; no network or inference")
    p.add_argument("--max-samples", type=int, help="Per-arm pilot limit; use --resume to continue")
    p.add_argument("--run-id")
    args = p.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        p.error("--max-samples must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config, suite=args.suite, names=args.arms, resume=args.resume, plan=args.plan,
        dry_run=args.dry_run, max_samples=args.max_samples, run_id=args.run_id)


if __name__ == "__main__":
    main()
