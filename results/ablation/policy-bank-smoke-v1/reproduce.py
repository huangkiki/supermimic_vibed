"""Reproduce the declared 18-state bank qualification artifacts from raw receipts."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['svg.hashsalt'] = 'policy-bank-smoke-v1'

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo.resolve()))
from experiments.ablation.compare_fidelity import compare

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('raw', type=Path, help='Unpacked policy-bank-smoke-v1 raw directory')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
root, output = args.raw, args.output
output.mkdir(parents=True, exist_ok=True)
campaign = json.loads((root / 'campaign.json').read_text())
assert campaign['status'] == 'completed' and len(campaign['allocations']) == 4
assert all(r['returncode'] == 0 for r in campaign['allocations'])
summary = dict(status='native_bank_qualification_passed', cpu_tests=44,
               actual_gpu_hours=campaign['actual_gpu_hours'], samples_per_bank=18,
               checkpoint_iteration=100, training_seeds=[0], gradient_horizon=1,
               prediction_horizons=[1, 8, 16, 32], probes=4,
               scope='Engineering qualification with undertrained collector; different roll-ins induce different survivor-conditioned banks; no mature-policy or algorithm-efficiency conclusion',
               gpu_uuid=campaign['allocations'][0]['gpu_uuid'], banks={})
all_diagnostics, all_pairs = [], []
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout='constrained', sharey=True)
for ax, prefix in zip(axes, ('reference_pd', 'policy')):
    full, state = root / f'{prefix}-full', root / f'{prefix}-state_only'
    bank = json.loads((full / 'bank.json').read_text())
    assert hashlib.sha256((full / 'heldout-bank.pt').read_bytes()).hexdigest() == bank['bank_sha256']
    summary['banks'][prefix] = {k: bank[k] for k in (
        'bank_sha256', 'boundary_action_delta_l2_mean', 'following_action_delta_l2_mean',
        'boundary_action_replay_max_error', 'sampling_rounds')}
    for probe in (full, state):
        validation = json.loads((probe / 'validation.json').read_text())
        assert validation['bank_identity'] == 'sha256_verified'
        assert validation['prediction_rows'] == 72 and validation['gradient_rows'] == 144
    pairs = compare(full, state)
    assert all(row['bank_identity'] == 'sha256' for row in pairs)
    all_pairs.extend(pairs)
    with (full / 'bank-diagnostics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    all_diagnostics.extend(rows)
    values = [[float(r[key]) for r in rows] for key in
              ('boundary_action_delta_l2', 'following_action_delta_l2_mean')]
    ax.boxplot(values, tick_labels=['Probe boundary', 'Later steps\n(mean per start)'], showfliers=False)
    for x, y in enumerate(values, 1):
        ax.scatter(np.full(len(y), x) + np.linspace(-.09, .09, len(y)), y, s=15, alpha=.7)
    ax.set(title=prefix, yscale='log')
    ax.grid(axis='y', alpha=.2)
axes[0].set_ylabel('Action change L2 (log scale)')
fig.suptitle('Bank continuity qualification: undertrained collector, 18 starts each\n'
             'Different survivor-conditioned state distributions; not a policy-performance comparison', fontsize=11)
fig.savefig(output / 'bank-continuity.png', dpi=160)
svg = output / 'bank-continuity.svg'
fig.savefig(svg, metadata={'Date': None})
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
plt.close(fig)
for name, rows in [('bank-diagnostics.csv', all_diagnostics), ('paired-gradient-metrics.csv', all_pairs)]:
    with (output / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
(output / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
