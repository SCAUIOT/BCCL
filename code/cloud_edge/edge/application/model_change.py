"""Use the original system's /model_api JSON + multipart + polling protocol."""
from __future__ import annotations
from contextlib import ExitStack
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
import requests
from .config import digest
from pig_beh_ce.continuous import sha256_file


def update_metadata(sid, rows, experiment):
    samples = []
    for row in rows:
        image = Path(row["path"])
        clean = {k: v for k, v in row.items() if k != "path"}
        clean.update(filename=f"frame_{int(row['sample_index']):012d}.jpg",
                     size=image.stat().st_size, sha256=sha256_file(image))
        samples.append(clean)
    return {"task_type": "pig_count_behavior", "trigger": "confidence_drift",
            "activity": [r["non_lie_ratio"] for r in rows], "activity_mode": "non_lie_ratio",
            "experiment": {**experiment, "session_id": sid}, "samples": samples}


class CloudClient:
    def __init__(self, network, token=""):
        self.network = network
        self.base = network["cloud_url"].rstrip("/")
        if urlparse(self.base).scheme not in ("http", "https"):
            raise ValueError("cloud_url must use http:// or https://")
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = "Bearer " + token
        self.edges, self.artifacts = {}, {}

    def request(self, method, path, *, payload=None, params=None, upload=None, stream=False):
        attempts = int(self.network.get("retry_attempts", 6))
        last = None
        for attempt in range(attempts):
            try:
                with ExitStack() as opened:
                    kwargs = {"timeout": float(self.network.get("request_timeout_seconds", 60)), "stream": stream}
                    if payload is not None:
                        kwargs["json"] = payload
                    if params is not None:
                        kwargs["params"] = params
                    if upload is not None:
                        edge_name, rows, metadata = upload
                        kwargs["data"] = {"edge_name": edge_name,
                                          "metadata": json.dumps(metadata, ensure_ascii=False, allow_nan=False)}
                        kwargs["files"] = [("files", (sample["filename"], opened.enter_context(Path(row["path"]).open("rb")),
                                            "image/jpeg")) for row, sample in zip(rows, metadata["samples"])]
                    response = self.session.request(method, self.base + path, **kwargs)
                    if response.status_code in (429, 500, 502, 503, 504):
                        detail = response.text[:500]
                        response.close()
                        raise requests.ConnectionError(f"Transient cloud response: {detail}")
                    if not response.ok:
                        detail, code = response.text[:1500], response.status_code
                        response.close()
                        raise RuntimeError(f"Cloud HTTP {code}: {detail}")
                    if stream:
                        return response
                    try:
                        return response.json()
                    finally:
                        response.close()
            except (requests.ConnectionError, requests.Timeout) as error:
                last = error
                if attempt + 1 < attempts:
                    time.sleep(min(10., float(self.network.get("retry_backoff_seconds", 1)) * 2**attempt))
        raise RuntimeError(f"Cloud unavailable after {attempts} attempts; resume this run: {last}")

    def create_session(self, payload):
        # Each experiment arm must have a separate system edge_name/model binding.
        edge_name = payload["node_id"][:40] + "-" + digest([payload["node_id"], payload["run_id"], payload["arm"]])[:24]
        session = self.request("POST", "/model_api/register", payload={
            "edge_name": edge_name, "task_type": "pig_count_behavior", "experiment": payload})
        sid = session["session_id"]
        self.edges[sid] = edge_name
        self.artifacts[sid, "initial"] = session["model"]
        return session

    def state(self, sid):
        value = self.request("GET", "/model_api/edge_state", params={"edge_name": self.edges[sid]})
        self.artifacts[sid, "current"] = value["model_path"]
        if value["updates"]:
            self.artifacts[sid, f"update-{value['updates']}"] = value["model_path"]
        return value

    def _model_path(self, sid, artifact):
        if (sid, artifact) not in self.artifacts:
            if artifact.startswith("update-"):
                task = sid + f".u{int(artifact[7:]):06d}"
                job = self.request("GET", f"/model_api/status/{task}")
                if job["status"] != "Completed":
                    raise RuntimeError("Requested model task is not complete")
                self.artifacts[sid, artifact] = job["result"]["model_path"]
            elif artifact == "current":
                self.state(sid)
            else:
                raise RuntimeError(f"Artifact was not supplied by the cloud: {artifact}")
        return self.artifacts[sid, artifact]

    def acknowledge(self, sid, update_id, sha):
        return self.request("POST", "/model_api/model_feedback", payload={
            "edge_name": self.edges[sid],
            "model_path": self._model_path(sid, f"update-{update_id}" if update_id else "initial"),
            "accepted": True, "metrics": {"update_id": update_id, "model_sha256": sha}})

    def submit_update(self, sid, rows, experiment):
        metadata = update_metadata(sid, rows, experiment)
        metadata["edge_name"] = self.edges[sid]
        return self.request("POST", "/model_api/continual_update", upload=(self.edges[sid], rows, metadata))

    def finalize(self, sid, state):
        return self.request("POST", "/model_api/final_evaluate",
                            payload={"edge_name": self.edges[sid], "state": state})

    def wait(self, sid, job):
        started = time.monotonic()
        maximum = float(self.network.get("max_job_wait_seconds", 172800))
        while True:
            if job["status"] == "Completed":
                result = dict(job["result"])
                if "bundle" in result:
                    candidate = self.request("POST", "/model_api/candidate_model", payload={
                        "edge_name": self.edges[sid], "task_id": job["task_id"]})
                    if not candidate.get("available") or candidate["model_sha256"] != result["model_sha256"]:
                        raise RuntimeError("Candidate model differs from completed update")
                    result["model_path"] = candidate["model_path"]
                    number = int(Path(result["bundle"]).name[1:])
                    self.artifacts[sid, f"update-{number}"] = candidate["model_path"]
                for name, path in result.get("artifacts", {}).items():
                    self.artifacts[sid, name] = path
                return result
            if job["status"] == "Failed":
                raise RuntimeError("Cloud job failed; fix the cause and use --resume: " + str(job.get("error")))
            if time.monotonic() - started > maximum:
                raise TimeoutError("Cloud job is pending; resume later with the same run_id")
            time.sleep(float(self.network.get("poll_seconds", 2)))
            job = self.request("GET", f"/model_api/status/{job['task_id']}")

    def download(self, sid, artifact, destination, expected_sha=None):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if expected_sha and destination.is_file() and sha256_file(destination) == expected_sha:
            return
        path = self._model_path(sid, artifact)
        report = artifact in ("final_summary.json", "final_evaluation.csv")
        response = self.request("GET", "/model_api/download_report" if report else "/model_api/download_model",
            params={"edge_name": self.edges[sid], "report_path" if report else "model_path": path}, stream=True)
        temporary = destination.with_name(destination.name + ".part")
        maximum = int(self.network.get("max_model_mib", 256) * 1024**2)
        try:
            count = 0
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    count += len(chunk)
                    if count > maximum:
                        raise ValueError("Cloud artifact exceeds max_model_mib")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if expected_sha and sha256_file(temporary) != expected_sha:
                raise RuntimeError("Downloaded model checksum mismatch; existing deployment is unchanged")
            os.replace(temporary, destination)
        finally:
            response.close()
            if temporary.exists():
                temporary.unlink()

    def close(self):
        self.session.close()
