# BCCL Cloud-Edge Experiments

Run `edge/` on the edge device and `cloud/` on the GPU server. They communicate through the HTTP `/model_api/` protocol on port 8800; shared storage is not required.

The edge processes recorded videos in chronological order and selects samples after drift detection. Each uploaded batch triggers cloud-side pseudo-labeling, training, and deployment validation. The edge pauses video reading during the update, downloads the selected model, and continues from the next frame. Final models are evaluated after all videos finish.

## Project layout

| Path | Purpose |
|---|---|
| `edge/run.py` | Video processing, experiment selection, and resume entry point |
| `cloud/run.py` | Cloud API and training service |
| `cloud/evaluate.py` | Independent evaluation of completed experiments |
| Each side's `application/` | Communication and experiment orchestration |
| Each side's `vendor/pig_beh_ce/` | Bundled detection, drift, training, and evaluation code |
| Each side's `configs/` | Host settings and experiment parameters |
| Each side's `requirements.txt` | Python dependencies |

## Setup

Copy `edge/` to the edge device and `cloud/` to the server. On each host, install a device-compatible PyTorch/CUDA build, then run inside its directory:

```bash
python -m pip install -r requirements.txt
```

Provide the videos on the edge, and student/teacher weights, labeled source data, and labeled test data on the cloud. Weights and datasets are not included. Edit these settings before running. Paths refer to files on the corresponding host; relative paths resolve against the configuration file's directory.

| Configuration | Main settings |
|---|---|
| `edge/configs/edge.yaml` | `network.cloud_url`: reachable server URL, e.g. `http://192.168.1.100:8800` |
| | `edge.ip`: edge identification only; `edge.node_id` and `edge.run_id`: experiment identity |
| | `paths.videos_dir`, `paths.work_dir`; leave `paths.initial_student` empty to download from the cloud |
| `cloud/configs/cloud.yaml` | `network.bind_host` and `network.port`: server listening address |
| | `paths.initial_student`, `paths.teacher`: model weights |
| | `paths.source_yaml`: source YOLO dataset YAML; `paths.dataset_root`: evaluation data root |
| | `paths.work_dir`, `device`, and `evaluation.names` |
| Both `configs/experiment.yaml` files | Keep identical; control drift, sample selection, training, replay, model bank, and storage |

The default test layout is `<dataset_root>/continual/<condition>/test_labeled/`, containing `images/test/` and `labels/test/`. Conditions are `T1`, `T2`, and `T3`, each with `Day`, `Night`, `Feeding`, and `Light` suffixes. Class IDs are `0: stand`, `1: lie`, and `2: eat`.

## Main parameters

Defaults in `configs/experiment.yaml`:

| Parameter | Meaning | Default |
|---|---|---|
| `continuous.sample_every_frames` | Infer once every N source frames | 25 |
| `continuous.short_window_seconds` | Drift statistics window | 300 s |
| `continuous.p_low` | Confidence quantile, Pq | 0.2 |
| `continuous.non_lie_ratio_threshold` | Non-lying pig ratio threshold, tau_a | 0.2 |
| `continuous.alpha` | Drift significance threshold | 0.05 |

At 25 FPS, a stride of 25 means about one inference per second. Storage defaults reserve 2 GB free and limit working outputs to 18 GB; input videos, datasets, environments, and initial weights require additional space.

## Run experiments

Start the cloud service first, from `cloud/` on the server:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/cloud.yaml
```

Then run one method from `edge/` on the edge device:

```bash
python run.py --config configs/edge.yaml --arms BCCL --resume
```

Use `--arms` to select one or more experiments. For example:

```bash
python run.py --config configs/edge.yaml --arms activity D_nlr03_nobank D_nlr03 --resume
```

`activity` selects by the non-lying pig ratio. `D_nlr03_nobank` disables both the model bank and source replay; `D_nlr03` keeps the bank but disables replay. These legacy names use the current configured thresholds. Full BCCL enables both features.

To run a predefined suite, replace `--arms ...` with `--suite NAME`:

| Suite | Experiments with the default configuration |
|---|---|
| `methods` | `full_upload`, `confidence`, `activity`, BCCL |
| `ablation` | `A_source_only`, `D_nlr03_nobank`, `D_nlr03`, BCCL |
| `joint` | 25 Pq/tau_a combinations from 0.1 to 0.5, including the shared BCCL baseline |
| `alpha` | alpha = 0.01, 0.05, 0.10, 0.20, including the shared baseline |
| `all` | All 34 unique experiments |

Add `--plan` to inspect the experiment list without running. Experiments execute sequentially. To resume, keep the same configuration, run ID, and work directories, restart the cloud service, and repeat the edge command with `--resume`. Use a new run ID for changed settings or inputs. Single-machine checkpoints are not compatible.

## Offline evaluation and results

Final evaluation runs automatically when enabled in `cloud.yaml`. To reevaluate completed experiments, run from `cloud/` on the server; the edge and cloud services do not need to be running:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --config configs/cloud.yaml \
  --experiments-dir /data/cloud_edgeV1/cloud_runs/runs/pig-edge-01/v17_tau02_seed42 \
  --output-dir /data/cloud_edgeV1/reevaluation_T1T3_v1 \
  --device 0
```

Adjust the experiment path to match `paths.work_dir`, `edge.node_id`, and `edge.run_id`. Use a new output directory. Add `--groups bccl` to evaluate only BCCL, or list multiple experiment names.

Results include `final_student.pt`, `reports/final_evaluation.csv`, `reports/final_summary.json`, and the aggregate `comparison.csv`. Metrics include mAP50 and behavior accuracy among class-independent box matches at IoU >= 0.5. Overall scores average the 12 conditions. Upload counts and JPEG bytes are cumulative for the complete video stream; divide bytes by 2^20 for MiB.
