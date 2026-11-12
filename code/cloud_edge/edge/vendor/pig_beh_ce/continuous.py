"""Bounded, resumable, synchronous video adaptation (no ML imports here)."""
from __future__ import annotations

import calendar
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
CONTINUOUS_METHODS = ('full_upload', 'confidence', 'activity', 'bccl', 'source_only')
FRAME_FIELDS = ['sample_index', 'video_index', 'frame_index', 'recorded_time',
                'source_fps', 'model_version', 'avg_conf', 'det_count',
                'non_lie_ratio', 'behavior_entropy', 'P_low', 'drift_state', 'selected',
                'selection_reason']
VIDEO_READ_COUNTERS = {'video_decode_retry': 'video_decode_retries',
                       'video_decode_recovered': 'video_decode_recoveries',
                       'video_tail_shortfall': 'video_tail_shortfalls',
                       'video_decode_skip': 'skipped_videos'}
VIDEO_READ_SUMMARY_FIELDS = (*VIDEO_READ_COUNTERS.values(), 'skipped_seconds_estimate')
VIDEO_ISSUE_FIELDS = ['type', 'method', 'video_index', 'video_path', 'frame_index',
                      'source_fps', 'header_frame_count', 'header_shortfall_frames',
                      'recorded_time', 'unprocessed_seconds_estimate', 'estimate_source',
                      'reason', 'error', 'model_version']


def normalize_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized in ('entropy', 'non_lie'):
        normalized = 'activity'  # compatibility with earlier four-method commands
    if normalized not in CONTINUOUS_METHODS:
        raise ValueError(f'Unknown continuous method: {method}; choose {CONTINUOUS_METHODS}')
    return normalized


def select_drift_frame(method: str, stats: dict, threshold: float,
                       nlr_threshold: float) -> bool:
    """Within drift windows: full / confidence only / non-lie only / dual-signal BCCL."""
    method = normalize_method(method)
    if method == 'source_only':
        return False
    if method == 'full_upload':
        return True
    if method == 'activity':
        return float(stats['non_lie_ratio']) > nlr_threshold
    if float(stats['avg_conf']) >= threshold:
        return False
    if method == 'confidence':
        return True
    return float(stats['non_lie_ratio']) > nlr_threshold


@contextmanager
def output_lock(root: Path):
    """OS-released lock prevents two controllers from resuming the same run."""
    with (root / 'run.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another process owns this output directory: {root}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def remove_owned(path: Path, root: Path) -> None:
    """Only remove a concrete child of the dedicated output directory."""
    resolved, owner = path.resolve(), root.resolve()
    if resolved == owner or not resolved.is_relative_to(owner):
        raise ValueError(f'Refusing cleanup outside output directory: {path}')
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@dataclass(frozen=True)
class VideoEntry:
    path: str
    start: float
    end: float
    size: int
    mtime_ns: int


def video_manifest(directory: Path) -> list[VideoEntry]:
    """Use recording timestamps, never NVR/T1/T2 filename prefixes or mtimes."""
    entries = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        stamps = re.findall(r'(?<!\d)(20\d{12})(?!\d)', path.stem)
        if len(stamps) < 2:
            raise ValueError(f'Video filename needs recording start/end YYYYmmddHHMMSS: {path.name}')
        start, end = [calendar.timegm(datetime.strptime(s, '%Y%m%d%H%M%S').timetuple())
                      for s in stamps[-2:]]
        if end <= start:
            raise ValueError(f'Invalid recording time range: {path.name}')
        stat = path.stat()
        entries.append(VideoEntry(str(path.resolve()), start, end, stat.st_size, stat.st_mtime_ns))
    entries.sort(key=lambda item: (item.start, item.path))
    if not entries:
        raise ValueError(f'No videos in {directory}')
    for before, after in zip(entries, entries[1:]):
        if after.start < before.end:
            raise ValueError(f'Overlapping recordings need an explicit input selection: '
                             f'{before.path}, {after.path}')
    return entries


def effective_inference_fps(settings: dict, native_fps: float = 25.) -> float:
    """Actual inference rate; 25 source FPS / stride 25 = one inference/second.

    Before reading a video header, 25 source FPS is a nominal rate only. The
    executor resets the monitor to the actual rate before processing its frames.
    """
    if 'sample_every_frames' in settings:
        return native_fps / int(settings['sample_every_frames'])
    return min(native_fps, float(settings['sample_fps']))


def sample_frame(frame_index: int, native_fps: float, target_fps: float | None = None,
                 *, every_n_frames: int | None = None) -> bool:
    """Sample zero-based source indices by frame stride or inference rate."""
    if native_fps <= 0:
        raise ValueError('FPS must be positive')
    if every_n_frames is not None:
        if int(every_n_frames) != every_n_frames or every_n_frames <= 0 or target_fps is not None:
            raise ValueError('Frame stride must be a positive integer and exclude target_fps')
        return frame_index % int(every_n_frames) == 0
    if target_fps is None or target_fps <= 0:
        raise ValueError('FPS must be positive')
    if target_fps >= native_fps:
        return True  # never duplicate/interpolate frames from slower source videos
    return (frame_index == 0 or
            math.floor(frame_index * target_fps / native_fps + 1e-8) >
            math.floor((frame_index - 1) * target_fps / native_fps + 1e-8))


@lru_cache(maxsize=32)
def binomial_critical(n: int, p: float, alpha: float) -> int:
    tail = 0.0
    critical = n + 1
    for k in range(n, -1, -1):
        log_probability = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                           + k * math.log(p) + (n - k) * math.log1p(-p))
        tail += math.exp(log_probability)
        if tail < alpha:
            critical = k
    return critical


class DriftMonitor:
    """Frozen per-model confidence baseline, bounded rolling binomial monitor.

    Each complete window is drifting iff P(Binomial(n, p) >= low_count)
    is strictly below alpha. Previous windows only determine transition labels;
    they never extend drift after the current window stops being significant.
    Time horizons are converted using the effective inference FPS. Per-frame
    upload signals are evaluated by the executor, separate from drift detection.
    """
    def __init__(self, settings: dict, fps: float):
        self.settings = settings
        self.fps = fps
        self.short_n = max(10, round(float(settings['short_window_seconds']) * fps))
        self.warmup_n = max(10, round(float(settings['warmup_seconds']) * fps))
        self.p = float(settings.get('p_low', .2))
        self.critical = binomial_critical(self.short_n, self.p, float(settings.get('alpha', .05)))
        self.warmup: list[float] = []
        self.threshold: float | None = None
        self.flags: deque[bool] = deque(maxlen=self.short_n)
        self.low_count = 0
        self.drifting = False

    def clear_short(self) -> None:
        self.flags.clear()
        self.low_count = 0
        self.drifting = False

    def update(self, confidence: float) -> str:
        if not math.isfinite(confidence):
            raise ValueError('Non-finite detection confidence')
        if self.threshold is None:
            self.warmup.append(confidence)
            if len(self.warmup) >= self.warmup_n:
                ordered = sorted(self.warmup)
                q = (len(ordered) - 1) * self.p
                lo = math.floor(q)
                self.threshold = ordered[lo] + (ordered[math.ceil(q)] - ordered[lo]) * (q - lo)
                self.warmup.clear()
            return 'warmup'
        if len(self.flags) == self.short_n:
            self.low_count -= self.flags[0]
        flag = confidence < self.threshold
        self.flags.append(flag)
        self.low_count += flag
        was_drifting = self.drifting
        # The precomputed critical count is equivalent to the strict tail test,
        # without summing a binomial tail on each of millions of input frames.
        self.drifting = len(self.flags) == self.short_n and self.low_count >= self.critical
        if self.drifting != was_drifting:
            return 'started' if self.drifting else 'ended'
        return 'drifting' if self.drifting else 'normal'

    def snapshot(self) -> dict:
        return {'fps': self.fps, 'warmup': self.warmup, 'threshold': self.threshold,
                'flags': list(self.flags), 'drifting': self.drifting}

    @classmethod
    def restore(cls, settings: dict, state: dict) -> 'DriftMonitor':
        monitor = cls(settings, state['fps'])
        monitor.warmup = list(state['warmup'])
        monitor.threshold = state['threshold']
        monitor.flags.extend(state['flags'])
        monitor.low_count = sum(monitor.flags)
        monitor.drifting = bool(state['drifting'])
        return monitor


class DiskBudget:
    def __init__(self, root: Path, max_bytes: int, reserve_bytes: int):
        self.root, self.max_bytes, self.reserve_bytes = root, max_bytes, reserve_bytes
        self.used = 0
        self.refresh()

    def refresh(self) -> None:
        self.used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def check(self, additional: int = 0, refresh: bool = False) -> None:
        if refresh:
            self.refresh()
        free = shutil.disk_usage(self.root).free
        if self.used + additional > self.max_bytes or free - additional < self.reserve_bytes:
            raise RuntimeError(f'Disk budget reached: output={self.used / 1e9:.2f} GB, '
                               f'free={free / 1e9:.2f} GB. Resume after freeing space; '
                               'input videos are never deleted.')


class Journals:
    """Concatenated gzip members allow truncation to the last atomic checkpoint."""
    def __init__(self, root: Path, offsets: dict | None = None):
        root.mkdir(parents=True, exist_ok=True)
        self.frames = (root / 'frames.csv.gz').open('a+b')
        self.events = (root / 'events.jsonl').open('a+b')
        self.video_issues = (root / 'video_decode_issues.csv').open('a+b')
        if offsets is not None:
            self.frames.truncate(offsets['frames'])
            self.events.truncate(offsets['events'])
            self.video_issues.truncate(offsets.get('video_issues', 0))
        self.frames.seek(0, 2)
        self.events.seek(0, 2)
        self.video_issues.seek(0, 2)
        if self.video_issues.tell() == 0:
            self._video_issue({}, header_only=True)
        self.text = None
        self.writer = None

    def frame(self, row: dict) -> None:
        if self.writer is None:
            empty = self.frames.tell() == 0
            self.text = io.TextIOWrapper(gzip.GzipFile(fileobj=self.frames, mode='wb',
                                                       compresslevel=1, mtime=0),
                                         encoding='utf-8', newline='')
            self.writer = csv.DictWriter(self.text, fieldnames=FRAME_FIELDS, extrasaction='ignore')
            if empty:
                self.writer.writeheader()
        self.writer.writerow(row)

    def event(self, row: dict) -> None:
        self.events.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
        if row.get('type') in VIDEO_READ_COUNTERS:
            self._video_issue(row)

    def _video_issue(self, row: dict, header_only: bool = False) -> None:
        text = io.StringIO(newline='')
        writer = csv.DictWriter(text, fieldnames=VIDEO_ISSUE_FIELDS, extrasaction='ignore')
        if header_only:
            writer.writeheader()
        else:
            writer.writerow(row)
        self.video_issues.write(text.getvalue().encode('utf-8'))

    def checkpoint(self) -> dict:
        if self.text is not None:
            self.text.close()  # GzipFile leaves its external raw file open
            self.text = self.writer = None
        for stream in (self.frames, self.events, self.video_issues):
            stream.flush()
            os.fsync(stream.fileno())
        return {'frames': self.frames.tell(), 'events': self.events.tell(),
                'video_issues': self.video_issues.tell()}

    def close(self) -> None:
        self.checkpoint()
        self.frames.close()
        self.events.close()
        self.video_issues.close()


def validate_settings(settings: dict) -> None:
    normalize_method(settings.get('method', 'bccl'))
    if settings.get('drift_rule', 'binomial_window') != 'binomial_window':
        raise ValueError('continuous.drift_rule must be binomial_window')
    if 'sample_every_frames' in settings:
        stride = float(settings['sample_every_frames'])
        if (not math.isfinite(stride) or stride <= 0 or not stride.is_integer()
                or 'sample_fps' in settings):
            raise ValueError('Use a positive integer sample_every_frames without sample_fps')
    elif not math.isfinite(float(settings.get('sample_fps', 0))) or float(settings.get('sample_fps', 0)) <= 0:
        raise ValueError('Configure sample_every_frames or a positive sample_fps')
    positive = ['short_window_seconds', 'warmup_seconds',
                'max_collect_seconds', 'pending_max_age_seconds', 'checkpoint_every_samples']
    for key in positive:
        if float(settings[key]) <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 < float(settings.get('p_low', .2)) < 1 or not 0 < float(settings.get('alpha', .05)) < 1:
        raise ValueError('p_low and alpha must be between zero and one')
    if not 0 <= float(settings.get('non_lie_ratio_threshold', .3)) <= 1:
        raise ValueError('non_lie_ratio_threshold must be between zero and one')
    if int(settings['min_batch_images']) < 6:
        raise ValueError('min_batch_images must leave at least 5 holdout frames and one train frame')
    if int(settings['max_batch_images']) < int(settings['min_batch_images']):
        raise ValueError('max_batch_images must be >= min_batch_images')
    floor = float(settings.get('min_samples_per_drift_window', 0))
    if (not math.isfinite(floor) or floor < 0 or not floor.is_integer()
            or floor > int(settings['max_batch_images'])):
        raise ValueError('min_samples_per_drift_window must be an integer between 0 and max_batch_images')
    if not 0 <= float(settings['short_gap_seconds']) <= float(settings['long_gap_seconds']):
        raise ValueError('Gap thresholds must satisfy 0 <= short <= long')
    if float(settings['min_frame_gap_seconds']) < 0:
        raise ValueError('min_frame_gap_seconds cannot be negative')


class ContinuousRun:
    """Backend supplies video decoding/prediction, bounded image writes and V17 updates.

    The checkpoint is the commit point. Updates are prepared in isolated bundles;
    a completed receipt can be committed after interruption without retraining.
    """
    def __init__(self, root: Path, settings: dict, manifest: list[VideoEntry], backend: Any,
                 signature: str, resume: bool = False, budget: DiskBudget | None = None):
        validate_settings(settings)
        self.root, self.settings, self.manifest, self.backend = root, settings, manifest, backend
        self.method = normalize_method(settings.get('method', 'bccl'))
        self.window_floor = int(settings.get('min_samples_per_drift_window', 0))
        root.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self._wall_mark = time.monotonic()
        self.state_path = root / 'state.json'
        manifest_data = [asdict(item) for item in manifest]
        if self.state_path.exists():
            if not resume:
                raise ValueError('Output already has state.json. Use --resume or a new work directory.')
            self.state = json.loads(self.state_path.read_text(encoding='utf-8'))
            if self.state['signature'] != signature or self.state['manifest'] != manifest_data:
                raise ValueError('Configuration or input manifest changed; use a separate work directory.')
            self.monitor = DriftMonitor.restore(settings, self.state['monitor'])
        else:
            if resume:
                raise ValueError('--resume requires an existing state.json')
            self.monitor = DriftMonitor(settings, effective_inference_fps(settings))
            self.state = {'signature': signature, 'manifest': manifest_data, 'video_index': 0,
                          'next_frame': 0, 'samples': 0, 'model_version': 0, 'updates': 0,
                          'drift_events': 0, 'bundle': backend.initialize(), 'pending': [],
                          'last_time': None, 'last_selected_time': None, 'collect_since': None,
                          'window_selections': [],
                          'inflight': None, 'complete': False, 'evaluated': False}
            atomic_json(root / 'manifest.json', manifest_data)
        self.state.setdefault('window_selections', [])
        for key in VIDEO_READ_SUMMARY_FIELDS:
            self.state.setdefault(key, 0)
        self.journals = Journals(root / 'reports', self.state.get('journal_offsets'))
        if not self.state_path.exists():
            self.checkpoint()
        self._remove_orphan_images()

    def checkpoint(self) -> None:
        now = time.monotonic()
        self.state['wall_seconds'] = self.state.get('wall_seconds', 0.) + now - self._wall_mark
        self._wall_mark = now
        self.state['monitor'] = self.monitor.snapshot()
        self.state['journal_offsets'] = self.journals.checkpoint()
        atomic_json(self.state_path, self.state)
        self._remove_orphan_images()

    def _remove_orphan_images(self) -> None:
        keep = {str(Path(row['path']).resolve()) for row in self.state['pending']}
        for path in (self.root / 'pending').glob('*.jpg'):
            if str(path.resolve()) not in keep:
                remove_owned(path, self.root)

    def _discard_pending(self, reason: str) -> None:
        if self.state['pending']:
            self.journals.event({'type': 'discard_small_batch', 'reason': reason,
                                 'count': len(self.state['pending']),
                                 'model_version': self.state['model_version']})
        discarded = {row['sample_index'] for row in self.state['pending']}
        self.state['window_selections'] = [
            row for row in self.state['window_selections'] if row['sample_index'] not in discarded]
        self.state['pending'] = []
        self.state['last_selected_time'] = None

    def _prune_window_selections(self, timestamp: float) -> None:
        """Track real selections in the rolling video-time window, not each overlapping query.

        Uploaded selections still count when the deployed weights are unchanged;
        discarded samples do not. No additional image cache is needed for fallback.
        """
        cutoff = timestamp - float(self.settings['short_window_seconds'])
        self.state['window_selections'] = [
            row for row in self.state['window_selections'] if row['recorded_time'] > cutoff + 1e-9]

    def _check_gap(self, timestamp: float, fps: float) -> None:
        last = self.state['last_time']
        gap = 0 if last is None else timestamp - last
        if gap < -1e-6:
            raise ValueError('Video timestamps run backwards; check overlapping recordings')
        full = (gap > float(self.settings['long_gap_seconds']) or
                not math.isclose(self.monitor.fps, fps, abs_tol=1e-6))
        if full or gap > float(self.settings['short_gap_seconds']):
            self._discard_pending('recording_gap_or_fps_change')
            self.state['window_selections'] = []
            if full:
                self.monitor = DriftMonitor(self.settings, fps)
            else:
                self.monitor.clear_short()
            self.state['collect_since'] = None
            self.journals.event({'type': 'baseline_reset' if full else 'short_window_reset',
                                 'gap_seconds': gap, 'recorded_time': timestamp,
                                 'model_version': self.state['model_version']})

    def _update(self, reason: str) -> None:
        if len(self.state['pending']) < int(self.settings['min_batch_images']):
            return
        update_id = self.state['updates'] + 1
        self.state['inflight'] = {'update_id': update_id, 'reason': reason}
        self.checkpoint()  # durable cursor + original deployment + batch BEFORE training
        if self.budget:
            self.budget.check(int(self.settings.get('update_scratch_gb', 2) * 1e9), refresh=True)
        self.backend.release()
        result = self.backend.update(update_id, self.state['pending'], self.state['bundle'],
                                     self.state['model_version'])
        previous = self.state['bundle']
        pending = self.state['pending']
        self.state['bundle'] = result['bundle']
        self.state['updates'] = update_id
        self.state['pending'] = []
        self.state['inflight'] = None
        if result['model_changed']:
            self.state['model_version'] += 1
            self.monitor = DriftMonitor(self.settings, self.monitor.fps)
            self.state['window_selections'] = []
            self.state['last_selected_time'] = None
            self.state['collect_since'] = None
        else:
            self.state['collect_since'] = self.state['last_time'] if self.monitor.drifting else None
        self.journals.event({'type': 'update', 'update_id': update_id, 'reason': reason,
                             'recorded_time': self.state['last_time'],
                             'model_version': self.state['model_version'], **result})
        self.checkpoint()  # deploy commit point; older bundle remains available until here
        for row in pending:
            remove_owned(Path(row['path']), self.root)
        self.backend.cleanup(self.state['bundle'], previous)
        if self.budget:
            self.budget.check(refresh=True)

    def _video_read_event(self, video_index: int, entry: VideoEntry, event: dict) -> None:
        event = {**event, 'video_index': video_index, 'video_path': entry.path,
                 'method': self.method, 'model_version': self.state['model_version']}
        counter = VIDEO_READ_COUNTERS[event['type']]
        self.state[counter] += 1
        self.journals.event(event)
        if event['type'] == 'video_decode_skip':
            self.state['skipped_seconds_estimate'] += event['unprocessed_seconds_estimate']
            # Include the unread remainder even when this is the final video.
            # Keep last_time at the last genuine observation so subsequent files
            # still use the full recording gap when deciding whether to rewarm.
            last = self.state['last_time']
            if last is not None:
                self._check_gap(max(entry.end, last), self.monitor.fps)

    def run(self, max_samples: int | None = None) -> dict:
        stop_at = self.state['samples'] + max_samples if max_samples else None
        try:
            if self.method == 'source_only' and not self.state['complete']:
                # The fixed initial model has no adaptation stream to replay.
                # Evaluate it once on the same independent final test groups.
                self.state['video_index'] = len(self.manifest)
                self.state['next_frame'] = 0
                self.state['complete'] = True
                self.journals.event({'type': 'source_only', 'method': self.method,
                                     'bundle': self.state['bundle'], 'updates': 0})
                self.checkpoint()
            if self.state['inflight']:
                self._update(self.state['inflight']['reason'])
            if not self.state['complete']:
                for video_index in range(self.state['video_index'], len(self.manifest)):
                    entry = self.manifest[video_index]
                    start = self.state['next_frame'] if video_index == self.state['video_index'] else 0
                    for frame in self.backend.frames(entry, start):
                        if 'video_event' in frame:
                            self._video_read_event(video_index, entry, frame['video_event'])
                            continue
                        effective_fps = effective_inference_fps(self.settings, float(frame['fps']))
                        timestamp = entry.start + frame['index'] / float(frame['fps'])
                        self._check_gap(timestamp, effective_fps)
                        stats = self.backend.predict(frame, self.state['bundle'])
                        status = self.monitor.update(float(stats['avg_conf']))
                        self.state['video_index'] = video_index
                        self.state['next_frame'] = frame['index'] + 1
                        self.state['samples'] += 1
                        self.state['last_time'] = timestamp
                        if status == 'started':
                            self.state['drift_events'] += 1
                            self.state['collect_since'] = timestamp
                            self.journals.event({'type': 'drift_start', 'recorded_time': timestamp,
                                                 'method': self.method,
                                                 'model_version': self.state['model_version']})
                        pending = self.state['pending']
                        if pending and timestamp - pending[0]['recorded_time'] > float(self.settings['pending_max_age_seconds']):
                            self._discard_pending('small_batch_expired')
                            pending = self.state['pending']
                        self._prune_window_selections(timestamp)
                        selected = False
                        selection_reason = ''
                        last_selected = self.state['last_selected_time']
                        minimum_gap = (0. if self.method == 'full_upload' else
                                       float(self.settings['min_frame_gap_seconds']))
                        if (self.monitor.drifting and
                                (last_selected is None or timestamp - last_selected >= minimum_gap - 1e-6)):
                            primary = select_drift_frame(
                                self.method, stats, self.monitor.threshold,
                                float(self.settings['non_lie_ratio_threshold']))
                            fallback = (self.window_floor > len(self.state['window_selections'])
                                        and float(stats['det_count']) > 0)
                            if primary or fallback:
                                selection_reason = 'primary' if primary else 'window_floor'
                        if selection_reason:
                            if self.budget:
                                self.budget.check(16 * 1024**2)
                            path = self.backend.save_frame(frame, self.state['samples'])
                            row = {**stats, 'path': str(path), 'recorded_time': timestamp,
                                   'sample_index': self.state['samples'],
                                   'video_index': video_index, 'frame_index': frame['index'],
                                   'model_version': self.state['model_version'],
                                   'selection_reason': selection_reason}
                            pending.append(row)
                            if self.window_floor:
                                self.state['window_selections'].append({
                                    'sample_index': self.state['samples'], 'recorded_time': timestamp})
                            self.state['last_selected_time'] = timestamp
                            selected = True
                            if selection_reason == 'window_floor':
                                self.journals.event({'type': 'window_floor_selection',
                                                     'sample_index': self.state['samples'],
                                                     'recorded_time': timestamp, 'method': self.method,
                                                     'model_version': self.state['model_version'],
                                                     'selected_in_window': len(self.state['window_selections']),
                                                     'window_minimum': self.window_floor})
                        self.journals.frame({'sample_index': self.state['samples'],
                                             'video_index': video_index, 'frame_index': frame['index'],
                                             'recorded_time': timestamp, 'source_fps': frame['fps'],
                                             'model_version': self.state['model_version'], **stats,
                                             'P_low': self.monitor.threshold, 'drift_state': status,
                                             'selected': selected, 'selection_reason': selection_reason})
                        since = self.state['collect_since']
                        limit = since is not None and timestamp - since >= float(self.settings['max_collect_seconds'])
                        if len(pending) >= int(self.settings['max_batch_images']):
                            self._update('batch_full')
                        elif status == 'ended' or limit:
                            if len(pending) < int(self.settings['min_batch_images']):
                                self.journals.event({'type': 'no_upload',
                                                     'reason': 'drift_end' if status == 'ended' else 'collection_timeout',
                                                     'buffered_images': len(pending),
                                                     'recorded_time': timestamp,
                                                     'model_version': self.state['model_version']})
                            self._update('drift_end' if status == 'ended' else 'collection_timeout')
                            self.state['collect_since'] = timestamp if self.monitor.drifting else None
                        if self.state['samples'] % int(self.settings['checkpoint_every_samples']) == 0:
                            self.checkpoint()
                            if self.budget:
                                self.budget.check(refresh=True)
                            print(f'[continuous] samples={self.state["samples"]} '
                                  f'updates={self.state["updates"]} version={self.state["model_version"]} '
                                  f'video={video_index + 1}/{len(self.manifest)}', flush=True)
                        if stop_at is not None and self.state['samples'] >= stop_at:
                            self.checkpoint()
                            return self.state  # pilot limit: no early final evaluation
                    self.state['video_index'] = video_index + 1
                    self.state['next_frame'] = 0
                    self.checkpoint()
                self._update('end_of_stream')
                self._discard_pending('end_of_stream_below_minimum')
                self.state['complete'] = True
                self.checkpoint()
            if not self.state['evaluated']:
                self.backend.release()
                self.backend.evaluate(self.state['bundle'], self.state)
                self.state['evaluated'] = True
                self.checkpoint()
            return self.state
        finally:
            self.backend.release()
            self.journals.close()
