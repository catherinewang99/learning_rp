from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml


def load_rows(path, budget):
    rows = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        step = record.get('step')
        if isinstance(step, (int, float)) and step <= budget:
            rows.setdefault(step, {}).update(record)
    return rows


def series(rows, key):
    return [(step, row[key]) for step, row in sorted(rows.items())
            if key in row and isinstance(row[key], (int, float)) and math.isfinite(row[key])]


def average(rows, key, start):
    values = [v for s, v in series(rows, key) if s >= start]
    return float(np.mean(values)) if values else ''


def summarize(directory, budget):
    cfg = yaml.safe_load((directory / 'config.yaml').read_text())
    rows = load_rows(directory / 'metrics.jsonl', budget)
    out = {'run': directory.name, 'seed': cfg['seed'], 'budget_vector_steps': budget,
           'env_transitions_per_agent': budget * cfg['arena']['num_envs']}
    success = series(rows, 'eval/behavior/audio_matched_success')
    if not success or success[-1][0] < budget:
        out['status'] = 'incomplete_budget'
        return out
    out['status'] = 'complete'
    out['audio_success_last'] = success[-1][1]
    out['audio_success_tail_mean'] = average(rows, 'eval/behavior/audio_matched_success', .8 * budget)
    x, y = map(np.asarray, zip(*success))
    # Do not invent an unmeasured time-zero evaluation.
    area = float(np.sum((y[1:] + y[:-1]) * .5 * np.diff(x)))
    out['audio_success_auc_observed'] = area / (x[-1] - x[0]) if len(x) > 1 else ''
    out['auc_start_step'] = int(x[0])
    for label, key in {
        'vision_success_tail': 'eval/behavior/vision_matched_success',
        'audio_progress_tail': 'audio/distance_progress_mean',
        'audio_episode_return_tail': 'audio/episode_return',
        'pi_similarity_tail': 'eval/pi_cka/mean',
        'pi_valid_fraction_tail': 'eval/pi_cka/valid_pair_fraction',
        'v_cosine_tail': 'eval/v_cosine/mean',
        'v_valid_fraction_tail': 'eval/v_cosine/valid_pair_fraction',
        'k_similarity_tail': 'eval/k_cka/mean',
        'align_ratio_applied_tail': 'audio/align_task_grad_ratio_applied',
        'nonfinite_alignment_fraction': 'audio/align_skipped_nonfinite',
    }.items():
        out[label] = average(rows, key, .8 * budget)
    runtime = series(rows, 'elapsed_seconds')
    out['elapsed_seconds'] = runtime[-1][1] if runtime else ''
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, required=True, help='Common vector-step budget, e.g. 100000.')
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error('--budget must be positive.')
    rows = [summarize(directory, args.budget) for directory in args.directories]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()
