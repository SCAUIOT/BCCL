from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def _group_col(df: pd.DataFrame) -> str:
    # Prefer ablation_group for ablation experiments (isolates D/E even when
    # method_variant is identical).  Fall back to method_variant → method.
    if 'ablation_group' in df.columns and df['ablation_group'].notna().any():
        return 'ablation_group'
    if 'method_variant' in df.columns:
        return 'method_variant'
    return 'method'


def aggregate_results(results_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    if df.empty:
        return

    group_col = _group_col(df)
    agg_spec = {
        'mAP50': 'mean',
        'mAP50_95': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'behavior_acc_iou50': 'mean',
        'behavior_recall_iou50': 'mean',
        'uploaded_images': 'sum',
        'upload_mb': 'sum',
        'accepted_update': 'sum',
        'episode_count': 'sum',
        'update_count': 'sum',
        'bank_switch_count': 'sum',
        'forced_update_count': 'sum',
        'buffered_frames_total': 'sum',
        'edge_candidate_frames': 'sum',
        'edge_score_seconds': 'sum',
        'selection_seconds': 'sum',
        'pseudo_label_seconds': 'sum',
        'pseudo_behavior_boxes': 'sum',
        'pseudo_empty_images': 'sum',
        'pseudo_avg_conf': 'mean',
        'train_source_images': 'sum',
        'train_target_images': 'sum',
        'train_seconds': 'sum',
        'seen_eval_seconds': 'sum',
    }
    usable = {k: v for k, v in agg_spec.items() if k in df.columns}
    if usable and group_col in df.columns:
        summary = df.groupby(group_col).agg(usable).reset_index()
        summary.to_csv(out_dir / 'summary_by_method.csv', index=False, encoding='utf-8-sig')

    if 'stage_index' in df.columns and 'mAP50' in df.columns:
        plt.figure(figsize=(10, 5))
        for name, g in df.groupby(group_col):
            g = g.sort_values('stage_index')
            plt.plot(g['stage_index'], g['mAP50'], marker='o', label=str(name))
        plt.xlabel('Continual stage index')
        plt.ylabel('mAP50')
        plt.title('Behavior-recognition continual performance')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / 'map50_curve.png', dpi=200)
        plt.close()

    if 'stage_index' in df.columns and 'behavior_acc_iou50' in df.columns:
        plt.figure(figsize=(10, 5))
        for name, g in df.groupby(group_col):
            g = g.sort_values('stage_index')
            plt.plot(g['stage_index'], g['behavior_acc_iou50'], marker='o', label=str(name))
        plt.xlabel('Continual stage index')
        plt.ylabel('Behavior Acc@IoU0.5')
        plt.title('Behavior classification accuracy on matched pigs')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / 'behavior_acc_curve.png', dpi=200)
        plt.close()

    metric = 'mAP50'
    if metric not in df.columns:
        metric = 'behavior_acc_iou50' if 'behavior_acc_iou50' in df.columns else None
    if metric and 'uploaded_images' in df.columns:
        plt.figure(figsize=(8, 5))
        by = df.groupby(group_col).agg({'uploaded_images': 'sum', metric: 'mean'}).reset_index()
        plt.scatter(by['uploaded_images'], by[metric])
        for _, r in by.iterrows():
            plt.annotate(str(r[group_col]), (r['uploaded_images'], r[metric]), fontsize=8)
        plt.xlabel('Total uploaded images')
        plt.ylabel(f'Mean {metric}')
        plt.title('Communication cost vs behavior performance')
        plt.tight_layout()
        plt.savefig(out_dir / f'upload_vs_{metric}.png', dpi=200)
        plt.close()


def plot_learning_comparison(
    with_df: pd.DataFrame,
    without_df: pd.DataFrame,
    out_dir: Path,
    metric: str = 'mAP50',
    with_label: str = 'With Method (activity + trigger)',
    without_label: str = 'Without Method (source_only)',
) -> None:
    """Plot teacher-student co-improvement curves: solid = with method, dotted = without.

    Parameters
    ----------
    with_df : DataFrame
        Stage results from the "with method" experiment.
    without_df : DataFrame
        Stage results from the baseline experiment.
    out_dir : Path
        Output directory for the saved PNG.
    metric : str
        Metric to plot on the y-axis.
    with_label, without_label : str
        Legend labels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Solid line: with method
    if not with_df.empty and 'stage_index' in with_df.columns and metric in with_df.columns:
        g = with_df.sort_values('stage_index')
        ax.plot(g['stage_index'], g[metric], marker='o', linestyle='-',
                linewidth=2, color='#2ca02c', label=with_label)

    # Dotted line: without method
    if not without_df.empty and 'stage_index' in without_df.columns and metric in without_df.columns:
        g = without_df.sort_values('stage_index')
        ax.plot(g['stage_index'], g[metric], marker='s', linestyle='--',
                linewidth=2, color='#d62728', label=without_label)

    ax.set_xlabel('Continual Stage Index', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title('Learning Curve: With vs Without Our Method', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f'learning_curve_{metric}.png', dpi=200)
    plt.close()


def plot_teacher_student_coimprove(
    student_df: pd.DataFrame,
    teacher_df: pd.DataFrame | None = None,
    out_dir: Path = Path('.'),
    metric: str = 'mAP50',
) -> None:
    """Plot teacher-student joint improvement over continual stages.

    Solid lines = models using our method.
    Dotted lines = baseline without our method.

    This produces a figure similar to VPA-style teacher-student co-improvement plots.

    Parameters
    ----------
    student_df : DataFrame
        Student model per-stage results. Must have 'stage_index' and *metric*.
        If it contains both with/without results, set 'method_variant' column.
    teacher_df : DataFrame or None
        Optional teacher model results (if teacher is also updated).
    out_dir : Path
        Output directory.
    metric : str
        Metric to plot.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Student curves
    if 'method_variant' in student_df.columns:
        # Separate with/without by variant
        with_df = student_df[student_df['method_variant'].str.contains('trigger|activity')]
        without_df = student_df[~student_df['method_variant'].str.contains('trigger|activity')]
    else:
        with_df = student_df
        without_df = pd.DataFrame()

    if not with_df.empty and 'stage_index' in with_df.columns and metric in with_df.columns:
        g = with_df.sort_values('stage_index')
        ax.plot(g['stage_index'], g[metric], marker='o', linestyle='-',
                linewidth=2.5, color='#1f77b4', label='Student (with method)')

    if not without_df.empty and 'stage_index' in without_df.columns and metric in without_df.columns:
        g = without_df.sort_values('stage_index')
        ax.plot(g['stage_index'], g[metric], marker='o', linestyle=':',
                linewidth=2.5, color='#1f77b4', alpha=0.5,
                label='Student (without method)')

    # Teacher curve (optional)
    if teacher_df is not None and not teacher_df.empty:
        if 'stage_index' in teacher_df.columns and metric in teacher_df.columns:
            g = teacher_df.sort_values('stage_index')
            ax.plot(g['stage_index'], g[metric], marker='^', linestyle='-',
                    linewidth=2.5, color='#ff7f0e', label='Teacher (with method)')

    ax.set_xlabel('Continual Stage Index', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title('Teacher-Student Co-Improvement in Continual Domain Adaptation', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'teacher_student_coimprove.png', dpi=200)
    plt.close()
