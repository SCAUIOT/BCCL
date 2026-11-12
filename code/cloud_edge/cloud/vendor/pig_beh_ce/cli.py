from __future__ import annotations

import argparse


METHODS = ['source_only', 'random', 'confidence', 'uncertainty', 'behavior_drift', 'ours', 'full_upload', 'activity', 'adaptive']
MIX_MODES = ['source_plus_hard', 'hard_only']


def main():
    from .continuous_runtime import CONTINUOUS_GROUPS, CONTINUOUS_SENSITIVITY_BATCHES

    parser = argparse.ArgumentParser('pig-behavior-ce-update')
    sub = parser.add_subparsers(dest='cmd', required=True)

    continuous = sub.add_parser('run-continuous', help='Chronological video stream with immediate drift updates')
    continuous.add_argument('--config', required=True)
    continuous.add_argument('--videos-dir', help='Override the recording directory')
    continuous.add_argument('--work-dir', help='Override the dedicated output directory')
    continuous.add_argument('--resume', action='store_true', help='Restore cursor, detector and deployed model')
    continuous.add_argument('--dry-run', action='store_true', help='Validate manifest/configuration without ML packages')
    continuous.add_argument('--max-samples', type=int, help='Pause after this many new sampled frames; resume later')
    continuous_arms = continuous.add_mutually_exclusive_group()
    continuous_arms.add_argument('--methods', nargs='+', choices=['full_upload', 'confidence', 'activity', 'non_lie', 'entropy', 'BCCL', 'bccl', 'source_only'],
                                 help='Run methods sequentially; entropy and non_lie are aliases for activity')
    continuous_arms.add_argument('--groups', nargs='+',
                                 choices=list(CONTINUOUS_GROUPS),
                                 help='Run specific continuous ablation or sensitivity groups')
    continuous_arms.add_argument('--sensitivity', choices=list(CONTINUOUS_SENSITIVITY_BATCHES),
                                 help='joint: 24 grid cells excluding the BCCL anchor; alpha: 3 other levels')

    p1 = sub.add_parser('run-method')
    p1.add_argument('--config', required=True)
    p1.add_argument('--method', required=True, choices=METHODS)
    p1.add_argument('--upload-ratio', type=float, default=None)
    p1.add_argument('--mix-mode', default=None, choices=MIX_MODES,
                    help='Training data composition for update methods: source_plus_hard or hard_only')
    p1.add_argument('--dedup-iou', type=float, default=None,
                    help='Pseudo-label dedup NMS IoU threshold (e.g. 0.9). 0 or omit to disable.')
    p1.add_argument('--edge-prefilter-cache', default=None,
                    help='Reuse previously saved edge-prefiltered images from this directory '
                         'instead of re-extracting frames from videos.')
    p1.add_argument('--scores-cache', default=None,
                    help='Reuse precomputed BHSS scores from this directory (Phase 2). '
                         'Skips YOLO inference entirely; runs selection + training only.')
    p1.add_argument('--bhss-weights', default=None,
                    help='Custom BHSS fusion weights as JSON, e.g. \'{"conf_drop":0.50,"count_drift":0.25,"behavior_drift":0.25}\'. '
                         'Recombines precomputed components without re-running inference.')
    p1.add_argument('--trigger', action='store_true', default=False,
                    help='Enable adaptive cloud trigger (activity change detection). '
                         'Only trains when c_t > c_bar (meaningful behavior shift).')

    p2 = sub.add_parser('run-all')
    p2.add_argument('--config', required=True)
    p2.add_argument('--scores-cache', default=None,
                    help='Reuse precomputed BHSS scores from this directory.')

    p3 = sub.add_parser('ratio-sweep')
    p3.add_argument('--config', required=True)
    p3.add_argument('--mix-mode', default=None, choices=MIX_MODES)
    p3.add_argument('--method', default=None, choices=METHODS,
                    help='Run ratio sweep for a single method only. Omit to run all methods.')
    p3.add_argument('--scores-cache', default=None,
                    help='Reuse precomputed BHSS scores from this directory.')

    p4 = sub.add_parser('mix-ablation')
    p4.add_argument('--config', required=True)
    p4.add_argument('--method', default='ours', choices=[m for m in METHODS if m != 'source_only'])
    p4.add_argument('--upload-ratio', type=float, default=None)
    p4.add_argument('--scores-cache', default=None,
                    help='Reuse precomputed BHSS scores from this directory.')

    p5 = sub.add_parser('ablation')
    p5.add_argument('--config', required=True)
    p5.add_argument('--upload-ratio', type=float, default=None)
    p5.add_argument('--dedup-iou', type=float, default=None,
                    help='Pseudo-label dedup NMS IoU threshold (e.g. 0.9). 0 or omit to disable.')
    p5.add_argument('--scores-cache', default=None,
                    help='Reuse precomputed BHSS scores from this directory.')
    p5.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='Multiple random seeds for statistical significance, e.g. --seeds 42 123 456')
    p5.add_argument('--groups', nargs='+', default=None,
                    help='Run only specific ablation groups, e.g. --groups B_random_replay C_conf_replay')

    p6 = sub.add_parser('precompute')
    p6.add_argument('--config', required=True,
                    help='Path to behavior_experiment.yaml')

    p7 = sub.add_parser('weight-sweep')
    p7.add_argument('--config', required=True)
    p7.add_argument('--upload-ratio', type=float, default=None,
                    help='Fixed upload ratio for all weight configs (default: from config)')
    p7.add_argument('--mix-mode', default=None, choices=MIX_MODES)
    p7.add_argument('--scores-cache', default=None,
                    help='Reuse precomputed BHSS scores from this directory.')

    p8 = sub.add_parser('activity-ratio-sweep')
    p8.add_argument('--config', required=True)
    p8.add_argument('--mix-mode', default=None, choices=MIX_MODES)
    p8.add_argument('--scores-cache', default=None,
                    help='Reuse precomputed scores from this directory.')

    p9 = sub.add_parser('deploy-compare')
    p9.add_argument('--config', required=True)
    p9.add_argument('--upload-ratio', type=float, default=None)
    p9.add_argument('--mix-mode', default=None, choices=MIX_MODES)
    p9.add_argument('--scores-cache', default=None,
                    help='Reuse precomputed scores from this directory.')

    p10 = sub.add_parser('trigger-compare')
    p10.add_argument('--config', required=True)
    p10.add_argument('--upload-ratio', type=float, default=None)
    p10.add_argument('--mix-mode', default=None, choices=MIX_MODES)
    p10.add_argument('--scores-cache', default=None,
                     help='Reuse precomputed scores from this directory.')

    p11 = sub.add_parser('adaptive-compare')
    p11.add_argument('--config', required=True)
    p11.add_argument('--upload-ratio', type=float, default=None)
    p11.add_argument('--mix-mode', default=None, choices=MIX_MODES)
    p11.add_argument('--scores-cache', default=None,
                     help='Reuse precomputed scores from this directory.')

    p12 = sub.add_parser('one-shot-ablation')
    p12.add_argument('--config', required=True)
    p12.add_argument('--seeds', type=int, nargs='+', default=None,
                     help='Multiple random seeds, e.g. --seeds 42 123 456')

    p13 = sub.add_parser('screen-compare',
                          help='Compare screening methods: Random vs BSS vs Single-Signal '
                               '(all with model_bank + source replay)')
    p13.add_argument('--config', required=True,
                     help='Path to behavior_experiment.yaml')
    p13.add_argument('--upload-ratio', type=float, default=None,
                     help='Base upload ratio for adaptive methods (BSS/conf/entropy, default: from config). '
                          'Random groups use hardcoded 0.25/0.50/0.75 upload ratios. '
                          'BSS groups use nlr thresholds 0.25/0.50/0.75 (not upload ratios).')
    p13.add_argument('--scores-cache', default=None,
                     help='Reuse precomputed BHSS scores from this directory.')
    p13.add_argument('--seeds', type=int, nargs='+', default=None,
                     help='Multiple random seeds, e.g. --seeds 42 123 456')

    args = parser.parse_args()
    if args.cmd == 'run-continuous':
        if args.max_samples is not None and args.max_samples <= 0:
            parser.error('--max-samples must be positive')
        from .continuous_runtime import (run_continuous, run_continuous_batch,
                                         run_continuous_ablation_batch,
                                         run_continuous_sensitivity_batch)
        options = dict(videos_dir=args.videos_dir, work_dir=args.work_dir,
                       resume=args.resume, dry_run=args.dry_run, max_samples=args.max_samples)
        if args.sensitivity:
            run_continuous_sensitivity_batch(args.config, sensitivity=args.sensitivity, **options)
        elif args.groups:
            run_continuous_ablation_batch(args.config, groups=args.groups, **options)
        elif args.methods:
            run_continuous_batch(args.config, methods=args.methods, **options)
        else:
            run_continuous(args.config, **options)
        return

    from .runner import (run_one_method, run_all, run_upload_ratio_sweep, run_mix_ablation,
                         run_ablation, run_precompute, run_bhss_weight_sweep,
                         run_activity_ratio_sweep, run_deploy_compare, run_trigger_compare,
                         run_adaptive_compare, run_one_shot_ablation, run_screen_compare)
    if args.cmd == 'run-method':
        bhss_weights = None
        if getattr(args, 'bhss_weights', None):
            import json
            bhss_weights = json.loads(args.bhss_weights)
        run_one_method(args.config, args.method, args.upload_ratio, args.mix_mode,
                       dedup_iou_threshold=args.dedup_iou,
                       edge_prefilter_cache_dir=args.edge_prefilter_cache,
                       scores_cache_dir=args.scores_cache,
                       bhss_weights=bhss_weights,
                       trigger_based=getattr(args, 'trigger', False))
    elif args.cmd == 'run-all':
        run_all(args.config, scores_cache_dir=args.scores_cache)
    elif args.cmd == 'ratio-sweep':
        methods = [args.method] if args.method else None
        run_upload_ratio_sweep(args.config, mix_mode=args.mix_mode, methods=methods,
                               scores_cache_dir=args.scores_cache)
    elif args.cmd == 'mix-ablation':
        run_mix_ablation(args.config, args.method, args.upload_ratio,
                         scores_cache_dir=args.scores_cache)
    elif args.cmd == 'ablation':
        run_ablation(args.config, args.upload_ratio, dedup_iou_threshold=args.dedup_iou,
                     scores_cache_dir=args.scores_cache, seeds=args.seeds,
                     groups=args.groups)
    elif args.cmd == 'precompute':
        run_precompute(args.config)
    elif args.cmd == 'weight-sweep':
        run_bhss_weight_sweep(args.config, upload_ratio=args.upload_ratio,
                              mix_mode=args.mix_mode,
                              scores_cache_dir=args.scores_cache)
    elif args.cmd == 'activity-ratio-sweep':
        run_activity_ratio_sweep(args.config, mix_mode=args.mix_mode,
                                 scores_cache_dir=args.scores_cache)
    elif args.cmd == 'deploy-compare':
        run_deploy_compare(args.config, upload_ratio=args.upload_ratio,
                           mix_mode=args.mix_mode,
                           scores_cache_dir=args.scores_cache)
    elif args.cmd == 'trigger-compare':
        run_trigger_compare(args.config, upload_ratio=args.upload_ratio,
                            mix_mode=args.mix_mode,
                            scores_cache_dir=args.scores_cache)
    elif args.cmd == 'adaptive-compare':
        run_adaptive_compare(args.config, upload_ratio=args.upload_ratio,
                             mix_mode=args.mix_mode,
                             scores_cache_dir=args.scores_cache)
    elif args.cmd == 'one-shot-ablation':
        run_one_shot_ablation(args.config, seeds=args.seeds)
    elif args.cmd == 'screen-compare':
        run_screen_compare(args.config, upload_ratio=args.upload_ratio,
                           scores_cache_dir=args.scores_cache,
                           seeds=args.seeds)


if __name__ == '__main__':
    main()
