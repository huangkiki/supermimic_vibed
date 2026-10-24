#!/usr/bin/env bash
# Run outside the immutable execution checkout; reuse the existing bounded runner.
set -euo pipefail
: "${ABLATION_EXECUTION_ROOT:?Set the immutable training checkout}"
: "${ABLATION_QUALIFIED_CAMPAIGN:?Set the accepted all40 Smoke campaign}"
: "${ABLATION_RESOURCE_OUTPUT:?Set a new absolute output directory}"
case "$ABLATION_RESOURCE_OUTPUT" in /*) ;; *) echo 'Output must be absolute' >&2; exit 2 ;; esac
abl_forecasts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/configs" && pwd)"
export ABLATION_FORECASTS="$abl_forecasts"
mkdir -- "$ABLATION_RESOURCE_OUTPUT"
trap 'printf "%s\n" "$?" > "$ABLATION_RESOURCE_OUTPUT/exit.txt"' EXIT
cd -- "$ABLATION_EXECUTION_ROOT"

for abl_method in ppo_sim ppo_wm_online supermimic; do
  export ABLATION_RESOURCE_METHOD="$abl_method"
  # Check the original qualification and current resources before every stage.
  python - <<'PY'
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from experiments.ablation.run_suite import check_forecast, source_snapshot, wait_for_idle_gpu

root = Path.cwd()
output = Path(os.environ['ABLATION_RESOURCE_OUTPUT'])
if not output.is_absolute():
    raise ValueError('Output must be absolute')
qualified = Path(os.environ['ABLATION_QUALIFIED_CAMPAIGN'])
campaign = json.loads((qualified / 'campaign.json').read_text())
validation = json.loads((qualified / 'validation.json').read_text())
assert campaign['status'] == 'completed' and campaign['args']['subset'] == 'all40'
assert len(validation['checks']) == 7 and validation['evaluation_processes'] == 12
assert all(check['training_and_full_physical_evaluation_valid'] for check in validation['checks'])
assert source_snapshot(root) == campaign['source_hashes'], 'Execution source changed since native Smoke'
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True)
files = sorted(Path(os.environ['ABLATION_FORECASTS']).glob('all40-resource-*.json'))
forecasts = [json.loads(path.read_text()) for path in files]
assert len(forecasts) == 3 and sum(f['hard_gpu_hours'] for f in forecasts) <= 3.2
assert all(f['execution_commit'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() for f in forecasts)
hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
identity = output / 'forecast-sha256.json'
if identity.exists():
    assert json.loads(identity.read_text()) == hashes, 'Forecasts changed during qualification'
else:
    # Reserve the entire recipe's storage before its first allocation.
    assert shutil.disk_usage(output).free / 2**30 > sum(f['projected_disk_gib'] for f in forecasts) + 24
    identity.write_text(json.dumps(hashes, indent=2) + '\n')
method = os.environ['ABLATION_RESOURCE_METHOD']
forecast = json.loads((Path(os.environ['ABLATION_FORECASTS']) / f'all40-resource-{method}.json').read_text())
resources, observations = wait_for_idle_gpu(2, forecast['gpu_uuid'])
free = shutil.disk_usage(output).free / 2**30
check_forecast(forecast, resources, 'all40', [0], forecast['hard_gpu_hours'], free, forecast['scope'])
(output / f'{method}-preflight.json').write_text(json.dumps(dict(
    verified_at=datetime.now().astimezone().isoformat(),
    resources=resources, observations=observations, free_disk_gib=free,
    forecast=forecast, qualification=str(qualified), source_hashes=campaign['source_hashes']), indent=2) + '\n')
PY
  case "$abl_method" in
    ppo_sim) abl_methods=(pretrain ppo_sim); abl_iterations=32; abl_cap=1.05 ;;
    ppo_wm_online) abl_methods=(ppo_wm_online); abl_iterations=532; abl_cap=1.0 ;;
    supermimic) abl_methods=(supermimic); abl_iterations=532; abl_cap=1.15 ;;
  esac
  abl_run="$ABLATION_RESOURCE_OUTPUT/$abl_method"
  python -m experiments.ablation.run_suite --phase formal --subset all40 \
    --methods "${abl_methods[@]}" --iterations "$abl_iterations" \
    --checkpoint-every "$abl_iterations" --ppo-actor-lr 0.00002 --evaluation-repeats 1 \
    --seeds 0 --gpu 2 --max-gpu-hours "$abl_cap" \
    --forecast "$abl_forecasts/all40-resource-$abl_method.json" --output "$abl_run"
  python -m experiments.ablation.validate_run "$abl_run"
  python -m experiments.ablation.analyze --evaluations "$abl_run"/*/ckpt_*/evaluation.json \
    --campaign "$abl_run/campaign.json" --output "$abl_run/analysis" \
    --title "All 40 motions: $abl_method resource qualification only; 32 policy iterations"
done
