"""Bounded native entrypoint for one fixed-policy parameter-gradient diagnostic.

Run through `python -m experiments.ablation.fidelity --probe-space policy --help`.
This entrypoint is separate from, and never retrains, any policy or world model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import math
from pathlib import Path
import shutil
import time

import torch

from .common import Config, append_csv, identity, motion_files, sha256, write_json
from .policy_fidelity import ClosedLoopPhysical, PolicyGradientProbe, scan_parameter_directions, summarize_direction_scan, validate_starts


def select_bank(bank, indices):
    """Select explicit held-out starts while retaining the legacy bank unchanged."""
    size = len(bank['frames'])
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= size:
        raise ValueError('Unique sample indices within the source bank required')
    result = deepcopy(bank)
    for key in ('state', 'legacy_state', 'frames', 'xi', 'previous_action', 'history_before_probe', 'reset_frames'):
        if bank.get(key) is not None:
            result[key] = bank[key][indices].clone()
    for key in ('actions', 'prefix_actions'):
        if key in bank:
            result[key] = bank[key][:, indices].clone()
    for key in ('prefix_contacts',):
        if key in bank:
            result[key] = [[step[i] for i in indices] for step in bank[key]]
    for key in ('motion_names', 'start_sampling_attempts'):
        if key in bank:
            result[key] = [bank[key][i] for i in indices]
    if 'replay_history' in bank:
        result['replay_history'] = {k: v[indices].clone() for k, v in bank['replay_history'].items()}
    source_indices = bank.get('source_sample_indices', list(range(size)))
    result['source_sample_indices'] = [source_indices[i] for i in indices]
    return result


@torch.no_grad()
def reconstruct_history(env, bank, hist, *, replay_endpoint=False, audit=None):
    """Replay the saved prefix once to recover raw histories missing in old banks.

    The recovered history is a new numerical reconstruction, NOT an archived
    historical training minibatch. The original endpoint is retained after a
    strict drift check. Explicit replay_endpoint creates a distinct derived bank
    from saved prefix actions; it must never be described as the legacy endpoint.
    The input file is never overwritten.
    """
    from supermimic.mdp import flatten_state

    env.reset(torch.arange(env.N, device=env.device), bank['reset_frames'])
    env.refresh()
    env.failed_eval.zero_()
    env.done_eval.zero_()
    samples, contacts = [], []
    for action in bank['prefix_actions']:
        samples.append(dict(s=flatten_state(env.state()).clone(), a_prev=env.a_prev.clone(),
                            xi=env.xi.clone(), f=env.f.clone()))
        env.step(action)
        if replay_endpoint:
            from .fidelity import contact_signature
            contacts.append(contact_signature(env))
        if env.failed_eval.any():
            raise ValueError('Replayed prefix failed; cannot reconstruct a valid raw history')
    drift = (flatten_state(env.state()) - bank['state']).abs().amax(-1)
    if audit is not None:
        audit.update(prefix_endpoint_drift=drift.cpu().tolist(), legacy_drift_tolerance=1e-5,
                     exact_legacy_endpoint_retained=not replay_endpoint)
    if not torch.isfinite(drift).all() or not torch.equal(env.f, bank['frames']):
        raise ValueError('Replayed prefix has nonfinite state or changed frames')
    if not replay_endpoint and float(drift.max()) > 1e-5:
        raise ValueError(f'Legacy prefix endpoint cannot be reproduced: max drift {float(drift.max())}')
    if not torch.equal(env.xi, bank['xi']) or not torch.equal(env.a_prev, bank['previous_action']):
        raise ValueError('Legacy prefix parameters/action differ')
    if replay_endpoint:
        bank['legacy_state'] = bank['state'].clone()
        bank['state'] = flatten_state(env.state()).clone()
        bank['prefix_contacts'] = contacts
        bank['parent_prefix_controller'] = bank['prefix_controller']
        bank['prefix_controller'] = 'saved_actions_replay'
        bank['bank_kind'] = 'derived_prefix_replay_endpoint'
        bank['prefix_rule'] = 'fixed parent reset frames/actions; newly recorded raw history and endpoint; no policy recollection'
        # These belong to the parent trajectory and are invalid at the new endpoint.
        bank.pop('actions', None)
        bank.pop('history_before_probe', None)
    starts = dict(s=bank['state'].clone(), a_prev=bank['previous_action'].clone(),
                  xi=bank['xi'].clone(), f=bank['frames'].clone())
    samples.append(starts)
    history = {key: torch.stack([samples[max(0, len(samples) - hist + i)][key] for i in range(hist)], 1)
               for key in starts}
    return history, {'history_origin': 'saved held-out prefix replay; not historical training replay',
                     'prefix_endpoint_drift': drift.cpu().tolist(),
                     'endpoint_convention': ('new replay endpoint; distinct from legacy bank' if replay_endpoint
                                             else 'retain exact legacy-bank endpoint after drift <= 1e-5'),
                     'historical_training_update_recovered': False}


def parser():
    value = argparse.ArgumentParser(__doc__)
    value.add_argument('--checkpoint', required=True, help='fixed SuperMimic pi/wm/critic/opt_pi checkpoint')
    value.add_argument('--bank', required=True, help='existing held-out bank; original file is read-only')
    value.add_argument('--output', required=True)
    value.add_argument('--sample-indices', type=int, nargs='+', default=[0, 1])
    value.add_argument('--prepare-history-only', action='store_true',
                       help='replay the entire source bank in original environment order; no gradient or policy update')
    value.add_argument('--replay-prefix-bank', action='store_true',
                       help='history preparation only: make a DISTINCT bank from saved prefix actions and its actual replay endpoint')
    value.add_argument('--record-contacts', action='store_true',
                       help='retain native geometry contacts after every physical transition')
    value.add_argument('--gradient-horizon', type=int, choices=(1, 8, 32), default=32)
    value.add_argument('--probe-seed', type=int, default=9001)
    value.add_argument('--directions', type=int, default=2, help='additional random unit L2 directions')
    value.add_argument('--epsilons', type=float, nargs='+', default=[.001, .003, .01])
    value.add_argument('--repeats', type=int, default=3)
    value.add_argument('--no-critic', action='store_true')
    value.add_argument('--detach-action-reward', action='store_true')
    value.add_argument('--precision', choices=('training', 'fp32'), default='training')
    value.add_argument('--estimator-seeds', type=int, nargs='+', help='auxiliary matched Gaussian BPTT/PPO comparison; independent gradient noise tapes')
    value.add_argument('--reference-seeds', type=int, nargs='+', help='held-out physical noise tapes, disjoint from estimator seeds')
    value.add_argument('--convergence-counts', type=int, nargs='+',
                       help='independent-direction convergence mode; nested estimator sample counts')
    value.add_argument('--direction-seed', type=int, default=50001,
                       help='independent Rademacher directions, generated before any gradient')
    value.add_argument('--action-std', type=float, default=.1, help='shared fixed Gaussian std for the auxiliary comparison only')
    value.add_argument('--checkpoint-audit',
                       help='fixed-policy WM checkpoint / physical PPO plan; separate diagnostic mode')
    value.add_argument('--gpu', type=int, required=True, help='physical GPU index, checked before CUDA allocation')
    value.add_argument('--gpu-uuid', required=True)
    value.add_argument('--max-seconds', type=float, default=600)
    return value


def main(argv=None):
    started = time.perf_counter()
    # Help and argument checking work without loading Isaac Lab or allocating a GPU.
    args, launcher = parser().parse_known_args(argv)
    time_limit = 3600 if args.convergence_counts is not None or args.checkpoint_audit else 1800
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.max_seconds > time_limit:
        raise ValueError(f'This diagnostic requires 0 < max-seconds <= {time_limit}')
    if args.convergence_counts is not None and args.estimator_seeds is None:
        raise ValueError('Convergence counts require the matched estimator mode')
    audit_plan = None
    if args.checkpoint_audit:
        from .checkpoint_audit import validate_plan
        audit_plan = json.loads(Path(args.checkpoint_audit).read_text())
        validate_plan(audit_plan)
        if (args.estimator_seeds is not None or args.prepare_history_only or args.no_critic
                or args.detach_action_reward or args.precision != 'fp32' or args.gradient_horizon != 32):
            raise ValueError('Checkpoint audit requires standalone full-objective FP32 H32 mode')
        if not math.isfinite(args.action_std) or args.action_std <= 0:
            raise ValueError('Invalid Gaussian action std')
    if args.replay_prefix_bank and not args.prepare_history_only:
        raise ValueError('A distinct replay-prefix bank requires history preparation only')
    limit = 64 if args.prepare_history_only else 16
    if len(args.sample_indices) > limit:
        raise ValueError(f'At most {limit} starts allowed for this diagnostic mode')
    if args.prepare_history_only and (args.estimator_seeds is not None or args.no_critic or args.detach_action_reward):
        raise ValueError('History preparation does not compute gradients or estimator comparisons')
    if args.repeats < 2 or args.directions < 1 or len(args.epsilons) < 2 or args.epsilons != sorted(set(args.epsilons)) or any(not math.isfinite(e) or e <= 0 for e in args.epsilons):
        raise ValueError('Invalid repeat/direction/epsilon design')
    if args.estimator_seeds is not None:
        from .compare_policy_estimators import validate_design
        if args.reference_seeds is None or args.detach_action_reward or args.precision != 'fp32':
            raise ValueError('Matched estimators require held-out reference seeds, full reward gradients and explicit fp32')
        if args.convergence_counts is None:
            validate_design(args.estimator_seeds, args.reference_seeds, args.epsilons)
        else:
            from .estimator_convergence import validate_design as validate_convergence
            validate_convergence(args.estimator_seeds, args.reference_seeds, args.convergence_counts,
                                 args.epsilons, args.directions, args.direction_seed)
        if not math.isfinite(args.action_std) or args.action_std <= 0:
            raise ValueError('Shared action std must be positive and finite')
    elif args.reference_seeds is not None:
        raise ValueError('Reference noise seeds require the matched-estimator mode')
    from .run_suite import wait_for_idle_gpu
    resources, observations = wait_for_idle_gpu(args.gpu, args.gpu_uuid, timeout=1.)
    if float(resources['memory_free_mib']) < 16384:
        raise RuntimeError('Need 16 GiB free for initial small-batch parameter-gradient qualification')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new output directory; preserve prior evidence')
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 8 * 2**30:
        raise RuntimeError('Need 8 GiB free including full vectors and historical optimizer states')
    if torch.cuda.is_initialized():
        raise RuntimeError('Select the diagnostic GPU before initializing CUDA')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    output.mkdir(parents=True)
    planned_rollouts = args.repeats + 2 * len(args.epsilons) * (3 + args.directions) + 1
    if args.estimator_seeds is not None:
        planned_rollouts = len(args.reference_seeds) * (1 + 4 * len(args.epsilons)) + args.repeats - 1
    if args.convergence_counts is not None:
        planned_rollouts = len(args.reference_seeds) * (1 + 2 * args.directions * len(args.epsilons)) + args.repeats - 1
    if args.prepare_history_only:
        planned_rollouts = 0
    if audit_plan:
        a = audit_plan
        planned_rollouts = (len(a['gradient_seeds']) + len(a['reference_seeds'])
                            * (1 + 2*a['directions']*len(a['epsilons'])
                               + (1 + 2*len(a['wm_checkpoints']))*len(a['epsilons'])) + a['repeats']-1)
        if a.get('update_only_parent'):
            planned_rollouts = 1 + len(a['reference_seeds']) * (1 + 3*len(a['target_action_rms']))
    receipt = dict(status='started', args=vars(args), resources=resources, preflight_observations=observations,
        requested_time_budget_seconds=args.max_seconds, planned_physical_rollouts=planned_rollouts,
        planned_physical_steps=planned_rollouts * args.gradient_horizon * len(args.sample_indices)
                               * (audit_plan['replicas'] if audit_plan else 1),
        requested_gpu_hour_budget=args.max_seconds / 3600,
        budget_enforcement="checked before capture and each rollout; initialization or one operation may overrun; actual time retained",
        memory_estimate='16 GiB initial conservative reservation; not a measured peak',
        claim='diagnostic source-path validation; no scientific result until native output is qualified')
    write_json(output / 'probe.json', receipt)
    try:
        _run(args, launcher, output, receipt, started)
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['seconds'] = time.perf_counter() - started
        receipt['allocated_gpu_hours'] = receipt['seconds'] / 3600
        write_json(output / 'probe.json', receipt)


def _run(args, launcher, output, receipt, started):
    from isaaclab.app import add_launcher_args, launch_simulation
    launch_parser = argparse.ArgumentParser()
    add_launcher_args(launch_parser)
    launch_args = launch_parser.parse_args(launcher)
    from supermimic.env import TrackingSim, make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, normalize_xi, unflatten_state
    from supermimic.motion import MotionLib
    from supermimic.networks import Critic, Policy, WorldModel
    from supermimic.robot import default_joint_pos_list
    import supermimic.networks as networks

    device = 'cuda:0'
    torch.set_num_threads(4)
    checkpoint_hash, bank_hash = sha256(args.checkpoint), sha256(args.bank)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = Config(**checkpoint['config'])
    if cfg.method not in ('supermimic', 'bptt_wm_frozen'):
        raise ValueError('A deterministic BPTT checkpoint with matching Adam history is required')
    manifest = identity(cfg, hash_bootstrap=False)
    for key in ('assets', 'motions', 'packages', 'torch', 'cuda'):
        if manifest[key] != checkpoint['manifest'][key]:
            raise ValueError(f'Checkpoint and diagnostic environment differ: {key}')
    for name, digest in manifest['source_hashes'].items():
        if name.startswith('supermimic/') and digest != checkpoint['manifest']['source_hashes'][name]:
            raise ValueError(f'Original training source differs: {name}')
    original_bank = torch.load(args.bank, map_location=device, weights_only=False)
    if args.prepare_history_only and args.sample_indices != list(range(len(original_bank['frames']))):
        raise ValueError('History preparation requires the complete source bank in its original order')
    bank = select_bank(original_bank, args.sample_indices)
    if bank['collector_sha256'] != checkpoint_hash or bank['motion_hashes'] != manifest['motions'] or bank['asset_hashes'] != manifest['assets']:
        raise ValueError('Held-out bank collector, motions or assets differ from the checkpoint')
    audit_plan = None
    if args.checkpoint_audit:
        from .checkpoint_audit import repeat_bank
        audit_plan = json.loads(Path(args.checkpoint_audit).read_text())
        if 'replay_history' not in bank:
            raise ValueError('Prepare original raw histories before replicating audit starts')
        bank = repeat_bank(bank, audit_plan['replicas'])
    torch.backends.cuda.matmul.allow_tf32 = args.precision == 'training'
    torch.backends.cudnn.allow_tf32 = args.precision == 'training'
    networks.AMP_DTYPE = torch.bfloat16 if args.precision == 'training' and cfg.amp == 'bf16' else None
    torch.manual_seed(args.probe_seed)
    kin = G1Kinematics(device)
    motion = MotionLib([str(p) for p in motion_files(cfg)], kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    policy = Policy(OBS_DIM, 73 * cfg.goal_frames, cfg.hist).to(device)
    wm, critic = WorldModel().to(device), Critic(INV_DIM + CRITIC_EXTRA + 73 * cfg.goal_frames).to(device)
    for key, module in (('pi', policy), ('wm', wm), ('critic', critic)):
        module.load_state_dict(checkpoint[key])
    sim_cfg = make_sim(device, launch_args)
    with launch_simulation(sim_cfg, launch_args):
        env = TrackingSim(sim_cfg, len(bank['frames']), motion, mdp, gamma=cfg.gamma, eval_mode=True)
        if args.replay_prefix_bank and 'replay_history' in bank:
            raise ValueError('A derived prefix bank requires a legacy parent, not an already prepared bank')
        if 'replay_history' not in bank:
            audit = receipt.setdefault('history_reconstruction', {})
            bank['replay_history'], bank['history_provenance'] = reconstruct_history(
                env, bank, cfg.hist, replay_endpoint=args.replay_prefix_bank, audit=audit)
            if args.replay_prefix_bank:
                bank['parent_bank_sha256'] = bank_hash
        if bank.get('history_provenance', {}).get('historical_training_update_recovered') is not False:
            raise ValueError('Explicit fixed-bank history provenance required')
        starts = dict(s=bank['state'], a_prev=bank['previous_action'], xi=bank['xi'], f=bank['frames'])
        validate_starts(starts, bank['replay_history'], env.N, cfg.hist)
        torch.save(bank, output / 'policy-bank.pt')
        if args.record_contacts or audit_plan:
            model = env.sim.backend_manager._solver.mj_model
            write_json(output / 'contact-map.json', dict(geom_body_ids=[int(i) for i in model.geom_bodyid],
                       body_names=[model.body(i).name for i in range(model.nbody)]))
        if args.prepare_history_only:
            torch.cuda.synchronize()
            if sha256(args.checkpoint) != checkpoint_hash or sha256(args.bank) != bank_hash:
                raise RuntimeError('Input checkpoint or bank changed during history preparation')
            receipt.update(status='complete', phase='history_preparation_only',
                checkpoint_sha256=checkpoint_hash, bank_sha256=bank_hash,
                prepared_bank_sha256=sha256(output / 'policy-bank.pt'), manifest=manifest,
                history_provenance=bank['history_provenance'],
                prefix_steps=len(bank['prefix_actions']), prefix_simulator_steps=len(bank['prefix_actions']) * env.N,
                gradient_updates=0, peak_torch_memory_bytes=torch.cuda.max_memory_allocated())
            return
        probe = PolicyGradientProbe(policy=policy, wm=wm, critic=critic, mdp=mdp, motion=motion,
            starts=starts, history=bank['replay_history'], unflatten_state=unflatten_state, normalize_xi=normalize_xi,
            horizon=args.gradient_horizon, gamma=cfg.gamma, goal_frames=cfg.goal_frames, seed=args.probe_seed,
            no_critic=args.no_critic, detach_action_reward=args.detach_action_reward)
        def check_budget():
            if time.perf_counter() - started > args.max_seconds:
                raise TimeoutError('Policy diagnostic time budget exhausted before next operation')
        check_budget()
        if audit_plan:
            from .checkpoint_audit import run_audit
            from .policy_estimators import stochastic_probe
            probe = stochastic_probe(probe, args.action_std)
            physical = ClosedLoopPhysical(probe, env, bank, record_contacts=True)
            receipt['checkpoint_audit'] = run_audit(probe, physical, output, audit_plan, check_budget,
                                                   expected_manifest=checkpoint['manifest'])
        elif args.estimator_seeds is not None:
            from .policy_estimators import stochastic_probe
            from .compare_policy_estimators import run_comparison
            probe = stochastic_probe(probe, args.action_std)
            physical = ClosedLoopPhysical(probe, env, bank, record_contacts=args.record_contacts)
            options = dict(gradient_seeds=args.estimator_seeds, reference_seeds=args.reference_seeds,
                           epsilons=args.epsilons, repeats=args.repeats, check_budget=check_budget)
            if args.convergence_counts is None:
                result = run_comparison(probe, physical, output, **options)
            else:
                from .estimator_convergence import run_convergence
                result = run_convergence(probe, physical, output, **options,
                    counts=args.convergence_counts, directions=args.directions, direction_seed=args.direction_seed)
            receipt['estimator_comparison'] = result
        else:
            captured = probe.capture(checkpoint['opt_pi'])
            torch.save(captured, output / 'gradients.pt')
            physical = ClosedLoopPhysical(probe, env, bank, record_contacts=args.record_contacts)
            def bounded_physical():
                check_budget()
                return physical()
            result = scan_parameter_directions(probe, captured, bounded_physical, epsilons=args.epsilons,
                                               random_directions=args.directions, repeats=args.repeats)
            torch.save(result, output / 'direction-rollouts.pt')
            write_json(output / 'summary.json', summarize_direction_scan(result))
            for row in result['rows']:
                append_csv(output / 'directions.csv', row)
            receipt['actual_adam_improvement'] = result['actual_adam_improvement']
        torch.cuda.synchronize()
        if sha256(args.checkpoint) != checkpoint_hash or sha256(args.bank) != bank_hash:
            raise RuntimeError('Input checkpoint or bank changed during the diagnostic')
        receipt.update(status='complete', checkpoint_sha256=checkpoint_hash, bank_sha256=bank_hash,
            prepared_bank_sha256=sha256(output / 'policy-bank.pt'), manifest=manifest, source=probe.provenance,
            history_provenance=bank['history_provenance'],
            training_checkpoint_path='experiments/ablation/train.py; critic norm timing differs from original',
            gradient_target='original supermimic/train.py actor prefix at fixed stored checkpoint; not historical update replay',
            precision=dict(tf32=torch.backends.cuda.matmul.allow_tf32, amp=str(networks.AMP_DTYPE), compiled=False),
            peak_torch_memory_bytes=torch.cuda.max_memory_allocated(),
            reset_protocol='native state restore, nominal xi, Newton solver reset flags=0',
            numerical_reference='central finite differences; not an absolute ground-truth Jacobian')


if __name__ == '__main__':
    main()
