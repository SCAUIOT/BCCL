"""system/application/api_v2.py wire contract, backed by the V17 experiment engine."""
from __future__ import annotations
import hmac
import json
import logging
from pathlib import Path
from flask import Blueprint, Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from .config import digest, identifier
from .profiles import canonical, profile_hash
from .runtime import Conflict, read_json
from pig_beh_ce.continuous import atomic_json

LOG = logging.getLogger(__name__)
STATUS = {"queued": "Queued", "running": "Running", "done": "Completed", "failed": "Failed"}


def create_app(manager, token=""):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(manager.config["network"].get("max_upload_mib", 256) * 1024**2)
    model_api = Blueprint("model_api_v2", __name__)

    @app.before_request
    def authenticate():
        if request.path == "/health" or not token:
            return None
        if not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
            return jsonify(error="Unauthorized"), 401

    @app.errorhandler(Exception)
    def failure(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        code = (409 if isinstance(error, Conflict) else 404 if isinstance(error, FileNotFoundError)
                else 400 if isinstance(error, (ValueError, KeyError, TypeError)) else 500)
        if code == 500:
            LOG.exception("HTTP request failed")
        return jsonify(error=f"{type(error).__name__}: {error}"), code

    def body():
        value = request.get_json(silent=True) if request.is_json else dict(request.form)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object or form fields")
        return value

    def edge_index(edge_name):
        identifier(edge_name)
        return manager.root / "edges" / (digest(edge_name) + ".json")

    def binding(edge_name):
        record = read_json(edge_index(edge_name))
        if record["edge_name"] != edge_name:
            raise Conflict("Edge binding mismatch")
        return record["session_id"]

    def task_id(sid, job):
        return sid + "." + job

    def parse_task(value):
        sid, separator, job = value.partition(".")
        if not separator:
            raise ValueError("Invalid task_id")
        identifier(sid)
        identifier(job)
        return sid, job

    def job_response(sid, job_id):
        job = manager.job_status(sid, job_id)
        response = {"task_id": task_id(sid, job_id), "status": STATUS[job["status"]],
                    "message": "queued" if job["status"] == "queued" else job["status"]}
        if job.get("error"):
            response["error"] = job["error"]
        if job["status"] == "done":
            result = dict(job["result"])
            if job["kind"] == "update":
                model = manager.session_root(sid) / result["bundle"] / "models/current_student.pt"
                result.update(model_path=str(model), download_path="/model_api/download_model")
                response["model_path"] = str(model)
            else:
                root = manager.session_root(sid)
                result["artifacts"] = {"final_student.pt": str(root / "final_student.pt"),
                    "final_summary.json": str(root / "reports/final_summary.json")}
                if result["has_evaluation_csv"]:
                    result["artifacts"]["final_evaluation.csv"] = str(root / "reports/final_evaluation.csv")
            response["result"] = result
        return response

    @app.get("/health")
    def health():
        return jsonify(service="cloud_edgeV1", protocol="system.model_api", queue_size=manager.tasks.qsize())

    @model_api.post("/register")
    def register():
        data = body()
        edge_name = identifier(data["edge_name"])
        supplied = data.get("experiment")
        if supplied is None:
            # The original system registration requires only edge_name/task_type.
            profile = canonical("bccl", manager.shared)
            supplied = {"node_id": edge_name, "run_id": "system_protocol", "arm": profile.arm,
                        "algorithm_hash": profile_hash(manager.shared, profile),
                        "stream_signature": digest(["system", edge_name])}
        if not isinstance(supplied, dict):
            raise ValueError("experiment must be a JSON object")
        with manager.lock:
            session = manager.create_session(supplied)
            index = edge_index(edge_name)
            if index.exists() and read_json(index)["session_id"] != session["session_id"]:
                raise Conflict("edge_name is already bound to another experiment")
            atomic_json(index, {"edge_name": edge_name, "session_id": session["session_id"]})
        return jsonify(message=f"registered {edge_name}", task_type=data.get("task_type", "pig_count_behavior"),
                       model=str(manager.initial), download_path="/model_api/download_model",
                       edge_name=edge_name, **session)

    @model_api.post("/continual_update")
    def continual_update():
        if "files" not in request.files:
            raise ValueError("No files provided; expected repeated multipart field files")
        sid = binding(request.form.get("edge_name", ""))
        text = request.form.get("metadata", "{}")
        if len(text.encode("utf-8")) > 2 * 1024**2:
            raise ValueError("metadata exceeds 2 MiB")
        metadata = json.loads(text)
        if metadata.get("experiment", {}).get("session_id") != sid:
            raise Conflict("metadata experiment differs from registered edge_name")
        job = manager.submit_update(sid, metadata, request.files.getlist("files"), request.content_length or 0)
        return jsonify(job_response(sid, job["job_id"]))

    @model_api.get("/status/<task>")
    def status(task):
        sid, job = parse_task(task)
        return jsonify(job_response(sid, job))

    @model_api.post("/candidate_model")
    def candidate_model():
        data = body()
        sid = binding(data.get("edge_name", ""))
        if data.get("task_id"):
            task_sid, job = parse_task(data["task_id"])
            if task_sid != sid:
                raise Conflict("Candidate belongs to a different edge")
        else:
            number = manager.state(sid)["updates"]
            if not number:
                return jsonify(available=False, reason="no_completed_candidate")
            job = f"u{number:06d}"
        value = manager.job_status(sid, job)
        if value["status"] != "done" or value["kind"] != "update":
            return jsonify(available=False, reason="no_completed_candidate")
        result = job_response(sid, job)["result"]
        if data.get("current_model") == result["model_path"]:
            return jsonify(available=False, reason="already_current")
        return jsonify(available=True, task_id=task_id(sid, job), model_path=result["model_path"],
                       download_path="/model_api/download_model", model_sha256=result["model_sha256"],
                       candidate_info={"deployment_decision": result.get("decision", {}),
                                       "validation_location": "cloud_upload_holdout"})

    def allowed_artifact(path, report=False):
        if not path:
            raise ValueError("model_path is required" if not report else "report_path is required")
        path = Path(path).resolve()
        if report:
            valid = path.name in ("final_summary.json", "final_evaluation.csv") and path.parent.name == "reports"
        else:
            valid = path.suffix == ".pt"
        if not valid or not (path == manager.initial.resolve() or path.is_relative_to((manager.root / "runs").resolve())):
            raise ValueError("Path is not a downloadable experiment artifact")
        if not path.is_file():
            raise FileNotFoundError("Artifact not found or already reclaimed")
        edge_name = request.args.get("edge_name")
        if edge_name and path != manager.initial.resolve() and not path.is_relative_to(manager.session_root(binding(edge_name)).resolve()):
            raise Conflict("Artifact belongs to another edge experiment")
        return path

    @model_api.get("/download_model")
    def download_model():
        return send_file(allowed_artifact(request.args.get("model_path")), as_attachment=True, conditional=True)

    @model_api.post("/model_feedback")
    @model_api.post("/activate_model")
    def model_feedback():
        data = body()
        sid = binding(data.get("edge_name", ""))
        state = manager.state(sid)
        metrics = data.get("metrics") or {}
        update_id = metrics.get("update_id", state["updates"])
        if type(update_id) is not int or not 0 <= update_id <= state["updates"]:
            raise Conflict("Invalid feedback update_id")
        expected = manager.artifact(sid, f"update-{update_id}" if update_id else "initial")
        if Path(data["model_path"]).resolve() != expected.resolve():
            raise Conflict("Feedback model_path differs from the selected deployment")
        if bool(data.get("accepted", False)):
            ack = manager.acknowledge(sid, {"update_id": update_id,
                "model_sha256": metrics.get("model_sha256", state["model_sha256"])})
        else:
            ack = {"acked_update": state["acked_update"]}
        atomic_json(manager.session_root(sid) / "deployment_feedback.json", data)
        return jsonify(message="feedback recorded", **ack)

    # Extensions needed by the experiment; the original system has no final-test or checkpoint API.
    @model_api.get("/edge_state")
    def edge_state():
        sid = binding(request.args.get("edge_name", ""))
        with manager.lock:
            return jsonify(**manager.state(sid), model_path=str(manager.artifact(sid, "current")))

    @model_api.post("/final_evaluate")
    def final_evaluate():
        data = body()
        sid = binding(data.get("edge_name", ""))
        job = manager.finalize(sid, data["state"])
        return jsonify(job_response(sid, job["job_id"]))

    @model_api.get("/download_report")
    def download_report():
        return send_file(allowed_artifact(request.args.get("report_path"), report=True), as_attachment=True)

    app.register_blueprint(model_api, url_prefix="/model_api")
    return app
