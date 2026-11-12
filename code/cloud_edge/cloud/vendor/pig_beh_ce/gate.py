"""HMB v9 验证式更新闸门（VUG, Validated Update Gate）。

纯 Python 模块（仅 pathlib/typing/pandas/numpy），零 YOLO 依赖，可独立
smoke 测试。职责：

- ``decide_gate``          每个漂移片段（决策点）的边端三选一决策
                           （可解释分层闸门，纯函数）；
- ``DriftBuffer``          边端漂移样本 FIFO 缓冲（小片段跨决策点累积）；
- ``reconstruct_episodes`` 从流式打分 scores df 重建漂移片段（决策点）序列；
- ``episode_image_paths``  片段内已持久化样本路径（防御性过滤失效路径）。

决策优先级（从高到低，详见 result_auto/v9批次说明.md）：

1. 无帧可训（空片段 + 空缓冲）     -> keep_buffering / episode_no_frames
2. hard_drift（drift_score 超阈）  -> forced_update / hard_drift
3. 阶段结束且缓冲非空              -> forced_update / stage_end_flush
4. 缓冲满                          -> forced_update / buffer_cap
5. 连续非更新次数超限              -> forced_update / skip_streak_max
6. 总帧数低于最小训练量            -> keep_buffering / below_min_train_frames_{n}
7. 库候选验证增益 > margin         -> bank_switch / bank_gain_{x:.4f}
8. 同上但切换被禁用（消融组）      -> no_bank_action=skip   -> skip_no_bank / bank_switch_suppressed
                                      no_bank_action=update -> update / bank_switch_suppressed_retrain
9. 增益不足                        -> update / gain_below_margin_{x:.4f}
10. 库验证无有效增益               -> update / bank_gain_none
11. 无可切换库模型                 -> update / no_bank_match
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 闸门决策空间（写入 episode_results.csv 的 gate_decision 列）
GATE_DECISIONS = ('update', 'bank_switch', 'keep_buffering', 'forced_update', 'skip_no_bank')


def decide_gate(
    *,
    bank_available: bool = False,
    bank_gain: float | None = None,
    bank_margin: float = 0.03,
    bank_switch_enabled: bool = True,
    no_bank_action: str = 'skip',
    episode_frames: int = 0,
    buffer_size: int = 0,
    buffer_cap: int = 300,
    skip_streak: int = 0,
    skip_streak_max: int = 3,
    min_train_frames: int = 15,
    hard_drift: bool = False,
    stage_end: bool = False,
) -> tuple[str, str]:
    """每个漂移片段的更新决策（可解释分层闸门，纯函数）。

    返回 ``(decision, reason)``：
    decision ∈ {'update','bank_switch','keep_buffering','forced_update','skip_no_bank'}；
    reason 为人类可读原因码（写入 episode_results.csv 的 gate_reason 列）。

    安全强制（硬漂移 / 缓冲满 / 连续跳过超限 / 阶段结束）优先于效率选择
    （库切换 / 更新），样本不足时继续缓冲。参数语义见模块 docstring 决策表。

    库验证增益达标但切换被禁用（bank_switch_enabled=False）时：
    no_bank_action='skip' 返回 skip_no_bank（纯浪费消融 D_skip_nobank）；
    'update' 返回 update（无库直接重训对照 D_gate_nobank_update）。
    """
    total = int(episode_frames) + int(buffer_size)

    # 1) 空片段 + 空缓冲：无帧可训，绝不强制更新（避免空训练）
    if total <= 0:
        return 'keep_buffering', 'episode_no_frames'
    # 2) 硬漂移：片段置信度相对阶段基线骤降，必须更新
    if hard_drift:
        return 'forced_update', 'hard_drift'
    # 3) 阶段结束冲刷：样本不跨场景边界
    if stage_end:
        return 'forced_update', 'stage_end_flush'
    # 4) 缓冲满：边端存储上限触发上传
    if int(buffer_cap) > 0 and int(buffer_size) >= int(buffer_cap):
        return 'forced_update', 'buffer_cap'
    # 5) 连续非更新次数超限：防止跳过滑向持续漂移
    if int(skip_streak_max) > 0 and int(skip_streak) >= int(skip_streak_max):
        return 'forced_update', 'skip_streak_max'
    # 6) 总帧数不足最小训练量：继续缓冲，不单独训练
    if total < int(min_train_frames):
        return 'keep_buffering', f'below_min_train_frames_{total}'
    # 7/8) 库候选在本片段漂移帧上的验证增益超过 margin
    if bank_available and bank_gain is not None and float(bank_gain) > float(bank_margin):
        if bank_switch_enabled:
            return 'bank_switch', f'bank_gain_{float(bank_gain):.4f}'
        # 切换被禁用的消融组：'skip' = 跳过纯浪费（D_skip_nobank）；
        # 'update' = 无库对照，把这次本应切换的片段改为直接重训练
        # （D_gate_nobank_update：论文「无库→直接重训」对照链）
        if no_bank_action == 'update':
            return 'update', 'bank_switch_suppressed_retrain'
        return 'skip_no_bank', 'bank_switch_suppressed'
    # 9-11) 更新
    if bank_available:
        if bank_gain is None:
            return 'update', 'bank_gain_none'
        return 'update', f'gain_below_margin_{float(bank_gain):.4f}'
    return 'update', 'no_bank_match'


class DriftBuffer:
    """边端漂移样本 FIFO 缓冲。

    ``add(paths, rows)`` 中 paths 与 rows 必须一一对应（rows 为对应帧的
    score 行，用于指纹/统计；调用方负责对齐）。超出 ``max_frames`` 时
    逐出最旧路径并返回被逐出路径列表；``flush()`` 返回全部路径并清空；
    ``rows()`` 返回当前缓冲帧的 score 行 DataFrame。
    """

    def __init__(self, max_frames: int):
        self.max_frames = max(0, int(max_frames))
        self._paths: list[Path] = []
        self._rows: list[dict[str, Any]] = []

    def add(self, paths, rows: pd.DataFrame | list[dict[str, Any]] | None = None) -> list[Path]:
        paths = [Path(p) for p in (paths or [])]
        self._paths.extend(paths)
        if rows is not None:
            if isinstance(rows, pd.DataFrame) and not rows.empty:
                self._rows.extend(rows.to_dict('records'))
            elif rows:
                self._rows.extend(list(rows))
        evicted: list[Path] = []
        if self.max_frames > 0:
            while len(self._paths) > self.max_frames:
                evicted.append(self._paths.pop(0))
                if self._rows:
                    self._rows.pop(0)
        return evicted

    def size(self) -> int:
        return len(self._paths)

    def flush(self) -> list[Path]:
        paths, self._paths = self._paths, []
        self._rows = []
        return paths

    def rows(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows) if self._rows else pd.DataFrame()


def _episode_mean_conf(subset: pd.DataFrame) -> float:
    """片段内已选帧的平均检测置信度（无帧 -> NaN）。"""
    if subset is None or subset.empty or 'avg_conf' not in subset.columns:
        return float('nan')
    vals = pd.to_numeric(subset['avg_conf'], errors='coerce').dropna()
    return float(vals.mean()) if not vals.empty else float('nan')


def _stage_p10(subset: pd.DataFrame, scores: pd.DataFrame | None) -> float:
    """阶段基线 P_10：优先取片段行上的 P_10_stage 标签（detector 冻结值），
    否则用 scores 全阶段 avg_conf 的 10 分位数兜底。"""
    if subset is not None and not subset.empty and 'P_10_stage' in subset.columns:
        vals = pd.to_numeric(subset['P_10_stage'], errors='coerce').dropna()
        if not vals.empty:
            return float(vals.iloc[0])
    if scores is not None and not scores.empty and 'avg_conf' in scores.columns:
        vals = pd.to_numeric(scores['avg_conf'], errors='coerce').dropna()
        if len(vals) >= 10:
            return float(np.percentile(vals, 10))
    return 0.0


def reconstruct_episodes(scores, selected) -> list[dict]:
    """从流式打分 scores df 重建漂移片段（决策点）序列。

    片段语义与 ``sampling.ConfidenceDriftDetector`` 保持一致：

    - 片段 = scores['drift_state'] 中一段最大的 'drifting' 连续帧，加上
      紧随其后的 normal 终止帧（detector 在 k<=expected 时把该帧并入
      窗口后才复位；流尾没有终止帧则到段尾）；
    - detector 在漂移启动时以短窗首帧播种窗口（sampling.py 1667-1669），
      因此片段起点前至多 ~short_window 帧的低置信帧也属于该窗口：凡带
      P_10_stage 标签且 frame_index 落在 (上一片段终点, 本片段终点] 的
      selected 行都归入当前片段，片段起点扩展到其首个已选帧；
    - 无 P_10_stage 标签的 selected 行（极端单帧 + fallback 补足）合成
      一个 origin='fallback' 的末尾合成片段；
    - 被帧门（upload_min_interval_frames）跳过的窗口保留
      selected_count=0 的决策点；
    - drift_score = P_10_stage − mean(episode avg_conf)，描述片段内模型
      置信度相对阶段基线的下沉幅度（硬漂移信号）。

    每个片段 dict 字段：index / origin('drift'|'fallback') / start / end /
    selected_count / selected_subset(DataFrame) / mean_conf / P_10_stage /
    drift_score。
    """
    if selected is None or not isinstance(selected, pd.DataFrame) or selected.empty:
        selected = pd.DataFrame()

    episodes: list[dict] = []

    # -- 无漂移状态列（非流式打分）：所有已选帧合成一个 fallback 片段 ----
    if scores is None or not isinstance(scores, pd.DataFrame) or scores.empty or 'drift_state' not in scores.columns:
        if not selected.empty:
            p10 = _stage_p10(selected, scores)
            mean_conf = _episode_mean_conf(selected)
            episodes.append({
                'index': 1,
                'origin': 'fallback',
                'start': 0.0,
                'end': 0.0,
                'selected_count': int(len(selected)),
                'selected_subset': selected.copy(),
                'mean_conf': mean_conf,
                'P_10_stage': p10,
                'drift_score': p10 - mean_conf,
            })
        return episodes

    # -- 'drifting' 最大连续段 + 紧随其后的终止帧 -------------------------
    states = scores['drift_state'].fillna('').astype(str).str.strip().str.lower().tolist()
    fidx = pd.to_numeric(scores['frame_index'], errors='coerce')
    runs: list[tuple[float, float]] = []  # (start_frame, end_frame)
    i, n = 0, len(states)
    while i < n:
        if states[i] == 'drifting':
            if pd.isna(fidx.iloc[i]):
                i += 1
                continue
            s = float(fidx.iloc[i])
            j = i
            while j < n and states[j] == 'drifting':
                j += 1
            last = float(fidx.iloc[j - 1])
            # 终止帧（state=='normal' 且紧随段尾，detector 语义）并入
            e = float(fidx.iloc[j]) if (j < n and not pd.isna(fidx.iloc[j])) else last
            runs.append((s, e))
            i = j
        else:
            i += 1

    # -- 已选帧归位：带标签行按 span 归入漂移片段，其余进 fallback --------
    # 空 selected（被帧门全跳过的阶段）保留 selected_count=0 的决策点，
    # 不触碰列（空 DataFrame 无 frame_index 列）。
    if selected.empty or 'frame_index' not in selected.columns:
        sel_idx = pd.Series([], dtype='float64')
    else:
        sel_idx = pd.to_numeric(selected['frame_index'], errors='coerce')
    if 'P_10_stage' in selected.columns:
        has_tag = selected['P_10_stage'].notna() & (pd.to_numeric(selected['P_10_stage'], errors='coerce') > 0)
    else:
        has_tag = pd.Series(False, index=selected.index)
    taken = pd.Series(False, index=selected.index)

    for run_idx, (s, e) in enumerate(runs, start=1):
        in_span = (sel_idx <= e) & (~taken)
        subset = selected.loc[in_span & has_tag].copy()
        taken |= in_span & has_tag
        p10 = _stage_p10(subset, scores)
        mean_conf = _episode_mean_conf(subset)
        episodes.append({
            'index': run_idx,
            'origin': 'drift',
            'start': s,
            'end': e,
            'selected_count': int(len(subset)),
            'selected_subset': subset,
            'mean_conf': mean_conf,
            'P_10_stage': p10,
            'drift_score': p10 - mean_conf,
        })

    # 残留行（极端单帧 + fallback 补足 + 防御性未归位标签行）合成末片段
    fb_subset = selected.loc[~taken].copy()
    if not fb_subset.empty:
        p10 = _stage_p10(fb_subset, scores)
        mean_conf = _episode_mean_conf(fb_subset)
        fb_idx = pd.to_numeric(fb_subset['frame_index'], errors='coerce').dropna()
        episodes.append({
            'index': len(episodes) + 1,
            'origin': 'fallback',
            'start': float(fb_idx.min()) if not fb_idx.empty else 0.0,
            'end': float(fb_idx.max()) if not fb_idx.empty else 0.0,
            'selected_count': int(len(fb_subset)),
            'selected_subset': fb_subset,
            'mean_conf': mean_conf,
            'P_10_stage': p10,
            'drift_score': p10 - mean_conf,
        })
    return episodes


def episode_image_paths(episode: dict) -> list[Path]:
    """片段内已持久化的样本路径（防御性过滤已失效路径）。

    streaming 已把选中帧复制到 selected_samples/{stage}/images 并把
    row['path'] 改为目标路径；帧缓存目录在阶段尾被删除，但
    selected_samples 持久。返回顺序与 selected_subset 行序一致。
    """
    subset = episode.get('selected_subset')
    if subset is None or (isinstance(subset, pd.DataFrame) and subset.empty):
        return []
    if isinstance(subset, pd.DataFrame):
        rows = subset.to_dict('records')
    else:
        rows = list(subset)
    paths: list[Path] = []
    for r in rows:
        p = r.get('path') if isinstance(r, dict) else getattr(r, 'path', None)
        if p is None:
            continue
        pp = Path(str(p))
        if pp.exists():
            paths.append(pp)
    return paths
