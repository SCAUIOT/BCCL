"""Authenticated HTTP service with a durable, single-GPU job queue."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from .config import (algorithm, cloud_raw, digest, identifier, local_path, read_yaml,
                     token_from)
from .profiles import apply_profile, canonical, profile_hash
from .protocol import save_multipart
from pig_beh_ce.continuous import (DiskBudget, atomic_json, output_lock, remove_owned,
                                   sha256_file)

LOG = logging.getLogger(__name__)


class Conflict(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def real_backend(raw, root, budget, config_path):
    from pig_beh_ce.continuous_runtime import YoloBackend
    return YoloBackend(config_path, raw, root, root, budget)


def missing_inputs(config, path):
    required = [local_path(config["paths"][k], path) for k in ("initial_student", "teacher", "source_yaml")]
    missing = [str(p) for p in required if not p.is_file()]
    source = local_path(config["paths"]["source_yaml"], path)
    if source.is_file():
        source_cfg, _ = read_yaml(source)
        data_root = Path(source_cfg.get("path", "."))
        if not data_root.is_absolute():
            data_root = (source.parent / data_root).resolve()
        train = source_cfg.get("train", "images/train")
        if not isinstance(train, str):
            raise ValueError("source_yaml.train must be a directory path, as in the V17 source loader")
        for p in (data_root / train, data_root / train.replace("images", "labels", 1)):
            if not p.is_dir():
                missing.append(str(p))
    if config.get("evaluation", {}).get("enabled", True):
        raw = cloud_raw(algorithm(config, path), config, path, Path("unused"))
        for test in raw["final_evaluation"]["datasets"]:
            for sub in ("images/test", "labels/test"):
                p = Path(test["root"]) / sub
                if not p.is_dir():
                    missing.append(str(p))
    return missing


class CloudManager:
    def __init__(self, config, config_path, backend_factory=real_backend, start_worker=True):
        self.config, self.config_path = config, Path(config_path)
        self.shared = algorithm(config, config_path)
        self.root = local_path(config["paths"]["work_dir"], config_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.initial = local_path(config["paths"]["initial_student"], config_path)
        if not self.initial.is_file():
            raise FileNotFoundError(f"Initial student not found: {self.initial}")
        self.initial_sha = sha256_file(self.initial)
        storage = self.shared["storage"]
        self.budget = DiskBudget(self.root, int(storage["max_work_dir_gb"] * 1e9),
                                 int(storage["min_free_disk_gb"] * 1e9))
        self.backend_factory = backend_factory
        self.lock = threading.RLock()
        self.tasks = queue.Queue()
        self.scheduled = set()
        self.worker = None
        self.recover()
        if start_worker:
            self.worker = threading.Thread(target=self._worker, name="cloud-gpu-worker", daemon=True)
            self.worker.start()

    def close(self):
        self.tasks.put(None)
        if self.worker:
            self.worker.join(timeout=20)

    def session_root(self, session_id):
        identifier(session_id)
        index = self.root / "sessions" / (session_id + ".json")
        if not index.is_file():
            raise FileNotFoundError("Unknown session")
        path = (self.root / read_json(index)["relative_root"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Session root escapes the output directory")
        return path

    def state(self, session_id):
        with self.lock:
            return read_json(self.session_root(session_id) / "state.json")

    def create_session(self, message):
        node, run, arm = (identifier(message[k]) for k in ("node_id", "run_id", "arm"))
        profile = canonical(arm, self.shared)
        if profile.arm != arm:
            raise ValueError(f"Use canonical arm {profile.arm}")
        expected = profile_hash(self.shared, profile)
        if message.get("algorithm_hash") != expected:
            raise Conflict("Shared experiment settings differ between edge and cloud")
        stream_signature = message.get("stream_signature", "")
        if not isinstance(stream_signature, str) or len(stream_signature) != 64:
            raise ValueError("Missing video manifest signature")
        if message.get("initial_student_sha") not in (None, "", self.initial_sha):
            raise Conflict("Edge initial student differs from the cloud initial student")
        session_id = digest([node, run, arm])[:32]
        with self.lock:
            root = self.root / "runs" / node / run / arm
            state_path = root / "state.json"
            if state_path.exists():
                state = read_json(state_path)
                if (state["algorithm_hash"], state["stream_signature"], state["initial_sha"]) != (
                        expected, stream_signature, self.initial_sha):
                    raise Conflict("Existing run has different settings, videos or initial weights; use a new run_id")
            else:
                self.budget.check(self.initial.stat().st_size, refresh=True)
                model = root / "bundles/b000000/models/current_student.pt"
                model.parent.mkdir(parents=True, exist_ok=True)
                (root / "bundles/b000000/model_bank").mkdir(exist_ok=True)
                (root / "reports").mkdir(exist_ok=True)
                shutil.copy2(self.initial, model)
                state = {"session_id": session_id, "node_id": node, "run_id": run, "arm": arm,
                         "profile": profile.to_dict(), "algorithm_hash": expected,
                         "edge_ip": str(message.get("edge_ip", "")),
                         "stream_signature": stream_signature, "initial_sha": self.initial_sha,
                         "model_sha256": self.initial_sha, "bundle": "bundles/b000000",
                         "model_version": 0, "updates": 0, "acked_update": 0,
                         "uploaded_images": 0, "upload_bytes": 0, "wire_payload_bytes": 0,
                         "train_seconds": 0., "pseudo_label_seconds": 0., "finished": False,
                         "complete": False, "evaluated": False}
                atomic_json(state_path, state)
                atomic_json(root / "profile.json", profile.to_dict())
                atomic_json(root / "stream_manifest.json", message.get("manifest", []))
                snapshot = cloud_raw(apply_profile(self.shared, profile), self.config, self.config_path, root)
                atomic_json(root / "cloud_config_snapshot.json", snapshot)
                atomic_json(root / "provenance.json", {
                    "initial_student_sha256": self.initial_sha,
                    **{k: {"path": snapshot["paths"][k],
                            "sha256": sha256_file(Path(snapshot["paths"][k]))
                            if Path(snapshot["paths"][k]).is_file() else None}
                       for k in ("teacher", "source_yaml")}})
            atomic_json(self.root / "sessions" / (session_id + ".json"),
                        {"relative_root": str(root.relative_to(self.root))})
            return {**state, "class_names": self.shared.get("class_names", {0: "stand", 1: "lie", 2: "eat"})}

    def _schedule(self, session_id, job_id):
        key = (session_id, job_id)
        if key not in self.scheduled:
            self.scheduled.add(key)
            self.tasks.put(key)

    def recover(self):
        with self.lock:
            for path in self.root.glob("runs/*/*/*/jobs/*/job.json"):
                job = read_json(path)
                if job["status"] in ("queued", "running"):
                    job["status"] = "queued"
                    atomic_json(path, job)
                    self._schedule(job["session_id"], job["job_id"])

    def _job(self, session_id, job_id):
        identifier(job_id)
        return self.session_root(session_id) / "jobs" / job_id

    def job_status(self, session_id, job_id):
        with self.lock:
            root = self._job(session_id, job_id)
            job = read_json(root / "job.json")
            response = {k: job.get(k) for k in ("job_id", "kind", "status", "error")}
            if job["status"] == "done":
                response["result"] = read_json(root / "result.json")
            return response

    def submit_update(self, session_id, metadata, files, content_length=0):
        with self.lock:
            root = self.session_root(session_id)
            state = self.state(session_id)
            if state["profile"]["method"] == "source_only":
                raise Conflict("source_only does not accept adaptation uploads")
            self.budget.check(int(self.shared["continuous"]["update_scratch_gb"] * 1e9), refresh=True)
            incoming = self.root / "incoming" / uuid.uuid4().hex
            incoming.mkdir(parents=True)
            try:
                max_bytes = int(self.config["network"].get("max_upload_mib", 256) * 1024**2)
                manifest, rows, fingerprint = save_multipart(
                    metadata, files, incoming / "input",
                    int(self.shared["continuous"]["max_batch_images"]), max_bytes)
                update_id = manifest.get("update_id")
                if type(update_id) is not int or update_id < 1:
                    raise ValueError("Invalid update_id")
                if len(rows) < int(self.shared["continuous"]["min_batch_images"]):
                    raise ValueError("Upload batch is below min_batch_images")
                job_id = f"u{update_id:06d}"
                job_root = root / "jobs" / job_id
                if (job_root / "job.json").exists():
                    job = read_json(job_root / "job.json")
                    if job["fingerprint"] != fingerprint:
                        raise Conflict("An update_id cannot be reused with different samples")
                    if job["status"] == "failed":
                        job.update(status="queued", error=None)
                        atomic_json(job_root / "job.json", job)
                        self._schedule(session_id, job_id)
                    return self.job_status(session_id, job_id)
                if state["finished"] or update_id != state["updates"] + 1:
                    raise Conflict("Update is out of sequence or experiment is finished")
                if state["acked_update"] != state["updates"]:
                    raise Conflict("Acknowledge the previously deployed update before submitting another")
                if (manifest.get("input_model_sha256") != state["model_sha256"] or
                        manifest.get("model_version") != state["model_version"] or
                        manifest.get("algorithm_hash") != state["algorithm_hash"]):
                    raise Conflict("Update input model or experiment signature differs from cloud state")
                for path in (root / "jobs").glob("*/job.json") if (root / "jobs").exists() else []:
                    if read_json(path)["status"] in ("queued", "running"):
                        raise Conflict("Only one outstanding job per arm is permitted")
                job_root.mkdir(parents=True)
                os.replace(incoming / "input", job_root / "input")
                for row in rows:
                    row["path"] = str((job_root / "input" / Path(row["path"]).name).resolve())
                atomic_json(job_root / "rows.json", rows)
                job = {"session_id": session_id, "job_id": job_id, "kind": "update",
                       "status": "queued", "update_id": update_id, "fingerprint": fingerprint,
                       "input_bundle": state["bundle"], "input_model_sha256": state["model_sha256"],
                       "input_version": state["model_version"], "wire_payload_bytes": int(content_length)}
                atomic_json(job_root / "job.json", job)
                self._schedule(session_id, job_id)
                return self.job_status(session_id, job_id)
            finally:
                remove_owned(incoming, self.root)

    def finalize(self, session_id, message):
        with self.lock:
            state = self.state(session_id)
            if message.get("model_sha256") != state["model_sha256"] or message.get("updates") != state["updates"]:
                raise Conflict("Final evaluation must use the final deployed model")
            if not message.get("complete") or message.get("model_version") != state["model_version"]:
                raise Conflict("Finish the edge stream before final evaluation")
            job_root = self._job(session_id, "final")
            if (job_root / "job.json").exists():
                job = read_json(job_root / "job.json")
                if job["edge_state"]["model_sha256"] != message["model_sha256"]:
                    raise Conflict("Final evaluation model changed")
                if job["status"] == "failed":
                    job.update(status="queued", error=None)
                    atomic_json(job_root / "job.json", job)
                    self._schedule(session_id, "final")
                return self.job_status(session_id, "final")
            if state["acked_update"] != state["updates"]:
                raise Conflict("Acknowledge deployment before final evaluation")
            for path in (self.session_root(session_id) / "jobs").glob("*/job.json"):
                if read_json(path)["status"] in ("queued", "running"):
                    raise Conflict("Wait for the outstanding update before final evaluation")
            allowed = ("complete", "updates", "model_version", "model_sha256", "samples", "drift_events",
                       "wall_seconds", "video_decode_retries", "video_decode_recoveries", "video_tail_shortfalls",
                       "skipped_videos", "skipped_seconds_estimate")
            job = {"session_id": session_id, "job_id": "final", "kind": "evaluation", "status": "queued",
                   "edge_state": {k: message.get(k, 0) for k in allowed}}
            atomic_json(job_root / "job.json", job)
            self._schedule(session_id, "final")
            return self.job_status(session_id, "final")

    def _append_update(self, root, update_id, result):
        path = root / "reports/cloud_events.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if json.loads(line).get("update_id") == update_id:
                    return
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"type": "update", "update_id": update_id, **result}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def execute(self, session_id, job_id):
        root, job_root = self.session_root(session_id), self._job(session_id, job_id)
        job = read_json(job_root / "job.json")
        with self.lock:
            job.update(status="running", error=None)
            atomic_json(job_root / "job.json", job)
        backend = None
        try:
            state = self.state(session_id)
            if (job_root / "result.json").exists():
                result = read_json(job_root / "result.json")
            else:
                raw = read_json(root / "cloud_config_snapshot.json")
                provenance = read_json(root / "provenance.json")
                for key in ("teacher", "source_yaml"):
                    recorded = provenance[key]
                    if recorded["sha256"] and sha256_file(Path(recorded["path"])) != recorded["sha256"]:
                        raise RuntimeError(f"Cloud {key} changed since session creation; restore it or start a new run")
                backend = self.backend_factory(raw, root, self.budget, self.config_path)
                if job["kind"] == "update":
                    result = backend.update(job["update_id"], read_json(job_root / "rows.json"),
                                            job["input_bundle"], job["input_version"])
                    actual_bytes = sum(Path(row["path"]).stat().st_size for row in read_json(job_root / "rows.json"))
                    if result["uploaded_images"] != len(read_json(job_root / "rows.json")) or result["upload_bytes"] != actual_bytes:
                        raise RuntimeError("Training receipt and uploaded image totals differ")
                    model = root / result["bundle"] / "models/current_student.pt"
                    if sha256_file(model) != result["model_sha256"]:
                        raise RuntimeError("Training receipt model checksum mismatch")
                    result["wire_payload_bytes"] = job["wire_payload_bytes"]
                else:
                    edge_state = dict(job["edge_state"])
                    backend.evaluate(state["bundle"], edge_state)
                    summary = read_json(root / "reports/final_summary.json")
                    if summary["model_sha256"] != state["model_sha256"]:
                        raise RuntimeError("Final evaluation used a different model")
                    result = {"summary": summary, "model_sha256": state["model_sha256"],
                              "has_evaluation_csv": (root / "reports/final_evaluation.csv").exists()}
                atomic_json(job_root / "result.json", result)
            with self.lock:
                state = self.state(session_id)
                if job["kind"] == "update":
                    if state["updates"] == job["update_id"] - 1:
                        if state["model_sha256"] != job["input_model_sha256"]:
                            raise Conflict("Cloud deployment changed during update")
                        state.update(bundle=result["bundle"], updates=job["update_id"],
                                     model_sha256=result["model_sha256"],
                                     model_version=state["model_version"] + int(result["model_changed"]))
                        for key in ("uploaded_images", "upload_bytes", "wire_payload_bytes", "train_seconds", "pseudo_label_seconds"):
                            state[key] += result[key]
                        atomic_json(root / "state.json", state)
                    elif state["updates"] != job["update_id"]:
                        raise Conflict("Prepared update cannot be committed over newer state")
                    self._append_update(root, job["update_id"], result)
                else:
                    state.update(finished=True, complete=True, evaluated=True)
                    atomic_json(root / "state.json", state)
                job["status"] = "done"
                atomic_json(job_root / "job.json", job)
                self.write_comparison()
        except Exception as error:
            LOG.exception("Cloud job %s/%s failed", session_id, job_id)
            with self.lock:
                job.update(status="failed", error=f"{type(error).__name__}: {error}")
                atomic_json(job_root / "job.json", job)
        finally:
            try:
                if backend is not None:
                    backend.release()
            except Exception:
                LOG.exception("GPU cleanup failed for %s/%s", session_id, job_id)
            finally:
                with self.lock:
                    self.scheduled.discard((session_id, job_id))

    def _worker(self):
        while True:
            item = self.tasks.get()
            if item is None:
                return
            try:
                self.execute(*item)
            except Exception:
                LOG.exception("Unexpected task-store error for %s", item)
            finally:
                self.tasks.task_done()

    def acknowledge(self, session_id, message):
        with self.lock:
            state, root = self.state(session_id), self.session_root(session_id)
            update_id = message.get("update_id")
            if type(update_id) is not int or not 0 <= update_id <= state["updates"]:
                raise Conflict("Invalid deployment acknowledgement")
            if update_id:
                result = read_json(root / "jobs" / f"u{update_id:06d}" / "result.json")
                if message.get("model_sha256") != result["model_sha256"]:
                    raise Conflict("Edge acknowledged different weights")
            state["acked_update"] = max(state["acked_update"], update_id)
            atomic_json(root / "state.json", state)
            for job in (root / "jobs").glob("u*/job.json") if (root / "jobs").exists() else []:
                value = read_json(job)
                if value["status"] == "done" and value["update_id"] <= state["acked_update"]:
                    remove_owned(job.parent / "input", root)
            for bundle in (root / "bundles").iterdir():
                # An edge retry may acknowledge the old model while a newer
                # bundle is still training/prepared. Never reclaim that bundle.
                if (bundle.name.startswith("b") and bundle.name[1:].isdigit()
                        and int(bundle.name[1:]) < max(0, state["updates"] - 1)):
                    remove_owned(bundle, root)
            self.budget.check(refresh=True)
            return {"acked_update": state["acked_update"]}

    def artifact(self, session_id, name):
        root, state = self.session_root(session_id), self.state(session_id)
        if name == "initial":
            return self.initial
        if name == "current":
            return root / state["bundle"] / "models/current_student.pt"
        if name.startswith("update-") and name[7:].isdigit():
            result = read_json(root / "jobs" / f"u{int(name[7:]):06d}" / "result.json")
            path = root / result["bundle"] / "models/current_student.pt"
            if not path.is_file():
                raise FileNotFoundError("Old model was reclaimed after acknowledged deployment")
            return path
        allowed = {"final_student.pt": root / "final_student.pt",
                   "final_summary.json": root / "reports/final_summary.json",
                   "final_evaluation.csv": root / "reports/final_evaluation.csv"}
        if name not in allowed or not allowed[name].is_file():
            raise FileNotFoundError("Artifact not available")
        return allowed[name]

    def write_comparison(self):
        rows = []
        for index in (self.root / "sessions").glob("*.json"):
            state = self.state(index.stem)
            summary_path = self.session_root(index.stem) / "reports/final_summary.json"
            summary = read_json(summary_path) if summary_path.exists() else {}
            rows.append({"node_id": state["node_id"], "run_id": state["run_id"],
                         **state["profile"], "complete": state["finished"],
                         **{k: state[k] for k in ("updates", "model_version", "model_sha256", "uploaded_images",
                                                   "upload_bytes", "wire_payload_bytes", "train_seconds", "pseudo_label_seconds")},
                         "upload_mib": state["upload_bytes"] / 1024**2, **summary.get("metrics", {})})
        if rows:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            target = self.root / "comparison.csv"
            temp = target.with_suffix(".tmp")
            with temp.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cloud.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, path = read_yaml(args.config)
    raw = algorithm(config, path)
    missing = missing_inputs(config, path)
    if args.dry_run:
        print(json.dumps({"role": "cloud", "missing_inputs": missing,
                          "work_dir": str(local_path(config["paths"]["work_dir"], path)),
                          "experiment": raw}, indent=2, ensure_ascii=False))
        return 1 if missing else 0
    if missing:
        parser.error("Missing cloud inputs: " + ", ".join(missing))
    token = token_from(config) if config["network"].get("require_token", False) else ""
    from .api_v2 import create_app
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = local_path(config["paths"]["work_dir"], path)
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root):
        manager = CloudManager(config, path)
        try:
            create_app(manager, token).run(host=config["network"].get("bind_host", "0.0.0.0"),
                                            port=int(config["network"].get("port", 8800)),
                                            threaded=True, use_reloader=False)
        finally:
            manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
