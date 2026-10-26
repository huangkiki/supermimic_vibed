"""Fixed-policy WM checkpoint audit against native score and numerical references.

Independent directions live in the *declared output-bias subspace*. Complete
parameter gradients are also tested as selected update directions. Neither is a
claim about a full Jacobian or a production PPO optimization schedule.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import torch

from .analyze import save_csv
from .common import sha256, write_json
from .estimator_convergence import VectorMoments
from .policy_estimators import estimate_pair, score_gradient
from .policy_fidelity import displaced_parameters, parameter_layout, parameter_vector


def repeat_bank(bank, replicas):
    if not isinstance(replicas, int) or not 1 <= replicas <= 32:
        raise ValueError('Require 1..32 replicas of each fixed start')
    result = deepcopy(bank)
    count = len(bank['frames'])
    indices = torch.arange(count, device=bank['frames'].device).repeat(replicas)
    for key in ('state', 'legacy_state', 'frames', 'xi', 'previous_action', 'history_before_probe', 'reset_frames'):
        if bank.get(key) is not None:
            result[key] = bank[key][indices].clone()
    for key in ('actions', 'prefix_actions'):
        if key in bank:
            result[key] = bank[key][:, indices].clone()
    for key in ('prefix_contacts',):
        if key in bank:
            result[key] = [list(step) * replicas for step in bank[key]]
    for key in ('motion_names', 'start_sampling_attempts', 'source_sample_indices'):
        if key in bank:
            result[key] = list(bank[key]) * replicas
    result['replay_history'] = {key: value[indices].clone() for key, value in bank['replay_history'].items()}
    result['replicas'] = replicas
    result['original_start_count'] = count
    return result


def validate_plan(plan):
    from .estimator_convergence import validate_design
    validate_design(plan['gradient_seeds'], plan['reference_seeds'], [len(plan['gradient_seeds'])],
                    plan['epsilons'], plan['directions'], plan['direction_seed'])
    if len(plan['wm_checkpoints']) < 2 or len(set(plan['wm_checkpoints'])) != len(plan['wm_checkpoints']):
        raise ValueError('At least two distinct WM checkpoints required')
    if plan['primary_epsilon'] not in plan['epsilons']:
        raise ValueError('Primary epsilon must be preregistered in the scale grid')
    if not 1 <= plan['replicas'] <= 32 or plan['repeats'] < 2:
        raise ValueError('Invalid replicas/repeats')


def bias_slice(policy):
    """Locate the actual last decoder bias; never infer it from vector suffixes."""
    target = policy.mean_policy.dec[-1].bias
    offset = 0
    for parameter in policy.parameters():
        if parameter is target:
            return slice(offset, offset + parameter.numel())
        offset += parameter.numel()
    raise ValueError('Output bias missing from named policy parameters')


def run_audit(probe, physical, output, plan, check_budget, *, expected_manifest):
    validate_plan(plan)
    if plan.get('update_only_parent'):
        from .calibrated_updates import run_updates
        return run_updates(probe, physical, output, plan, check_budget)
    output = Path(output)
    device = probe.device
    layout = parameter_layout(probe.policy)
    total = parameter_vector(probe.policy).numel()
    section = bias_slice(probe.policy)
    size = section.stop - section.start
    rng = np.random.default_rng(plan['direction_seed'])
    directions = (2 * rng.integers(0, 2, (plan['directions'], size)) - 1) / math.sqrt(size)
    # Persist before computing any gradient; directions cannot depend on results.
    np.save(output / 'bias-directions.npy', directions)
    write_json(output / 'audit-plan.json', plan)
    write_json(output / 'parameter-layout.json', layout)
    hashes = {name: sha256(name) for name in plan['wm_checkpoints']}
    timings, draws, prediction_rows, fd_rows, update_rows, baseline_rows = [], [], [], [], [], []
    means, moments, references = {}, {}, []
    allocations = dict(simulator_steps=0, imagined_gradient_steps=0, imagined_reference_steps=0,
                       wm_training_seconds=0., policy_optimization_updates=0)
    started = time.perf_counter()

    def timed(label, seed, call):
        check_budget()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = call()
        torch.cuda.synchronize(device)
        timings.append(dict(operation=label, seed=seed, seconds=time.perf_counter() - start))
        save_csv(output / 'timings.csv', timings)
        return result

    def physical_call(score=False):
        result = physical.score_rollout() if score else physical()
        allocations['simulator_steps'] += probe.batch * probe.horizon
        return result

    def wm_call():
        result = probe.objective()
        allocations['imagined_reference_steps'] += probe.batch * probe.horizon
        return result

    def add_gradient(name, seed, vector, objective):
        if name not in moments:
            moments[name] = VectorMoments(vector)
        moments[name].add(vector)
        bias = vector[section].double().cpu().numpy()
        draws.append(dict(method=name, seed=seed, objective=float(objective.mean()),
                          gradient_l2=float(vector.norm()), bias=bias.tolist(),
                          projections=(directions @ bias).tolist()))
        write_json(output / 'gradient-draws.json', draws)

    physical.record_contacts = False
    for seed in plan['gradient_seeds']:
        probe.seed = seed
        def estimate_physical():
            rollout = physical_call(score=True)
            return score_gradient(probe.policy, rollout)['gradient'], rollout['objective']
        vector, objective = timed('ppo_sim', seed, estimate_physical)
        add_gradient('ppo_sim', seed, vector, objective)
    means['ppo_sim'] = moments['ppo_sim'].mean.float().clone()
    print('AUDIT physical score gradients complete', flush=True)

    # Reference tapes are disjoint from all gradient-estimation tapes. Preserve
    # every trajectory, including failures/contact switches, for all WM versions.
    geom_body = physical.env.sim.backend_manager._solver.mj_model.geom_bodyid.tolist()
    from .fidelity_math import body_contact_signatures, contact_groups, window_contact_groups
    prefix = physical.bank.get('prefix_contacts', [])
    physical.record_contacts = True
    for seed in plan['reference_seeds']:
        probe.seed = seed
        result = timed('physical_baseline', seed, lambda: physical_call(score=True))
        contacts = prefix + result['geometry_contacts']
        labels = contact_groups(body_contact_signatures(contacts, geom_body))
        groups = {h: window_contact_groups(labels, len(prefix), h) for h in (1, 8, 16, 32) if h <= probe.horizon}
        reference = dict(seed=seed, states=tuple(x.cpu() for x in result['states']),
                         actions=result['actions'].cpu(), proxy=result['proxy'].cpu(), hard=result['hard'].cpu(),
                         hard_termination=result['hard_termination'].cpu(), contact_groups=groups,
                         geometry_contacts=result['geometry_contacts'], drift=result['initial_state_drift'].cpu())
        references.append(reference)
        for n in range(probe.batch):
            baseline_rows.append(dict(seed=seed, sample=n, start=n % physical.bank['original_start_count'],
                proxy=float(result['proxy'][n]), hard=float(result['hard'][n]),
                failed=bool(result['hard_termination'][:, n].any()), contact_group=groups[probe.horizon][n]))
    torch.save(references, output / 'physical-tapes.pt')
    physical.record_contacts = False
    save_csv(output / 'physical-baselines.csv', baseline_rows)
    repeats = []
    probe.seed = plan['reference_seeds'][0]
    for repeat in range(plan['repeats'] - 1):
        result = timed('physical_repeat', probe.seed, physical_call)
        repeats.append(dict(proxy=result['proxy'].cpu().tolist(), hard=result['hard'].cpu().tolist()))
    write_json(output / 'baseline-repeats.json', repeats)

    def measure(direction, name, seeds, epsilons, model, call):
        before = parameter_vector(probe.policy)
        for seed in seeds:
            probe.seed = seed
            for epsilon in epsilons:
                for sign in (1, -1):
                    check_budget()
                    with displaced_parameters(probe.policy, sign * epsilon * direction):
                        actual = parameter_vector(probe.policy) - before
                        rounding = float((actual - sign * epsilon * direction).norm() / epsilon)
                        if rounding > .02:
                            raise ValueError('Perturbation rounding exceeds 2 percent')
                        result = call()
                    for n in range(probe.batch):
                        fd_rows.append(dict(model=model, direction=name, seed=seed, epsilon=epsilon, sign=sign,
                            sample=n, proxy=float(result['proxy'][n]),
                            hard=float(result['hard'][n]) if 'hard' in result else '',
                            relative_rounding=rounding))
        torch.testing.assert_close(parameter_vector(probe.policy), before, rtol=0, atol=0)
        save_csv(output / 'directional-returns.csv', fd_rows)

    for i, small in enumerate(directions):
        direction = torch.zeros(total, device=device)
        direction[section] = torch.as_tensor(small, device=device, dtype=direction.dtype)
        timed('physical_independent_direction', i, lambda: measure(direction, str(i), plan['reference_seeds'],
              plan['epsilons'], 'physical', physical_call))
        print(f'AUDIT physical direction {i + 1}/{len(directions)} complete', flush=True)

    # Swap dynamics + their normalization only. Policy, critic, starts, xi, noise,
    # original actor source and fixed prediction scales stay identical.
    scale = torch.tensor([.3]*3 + [.4]*3 + [1.]*3 + [3.14]*3 + [1.]*29 + [10.]*29, device=device)
    for path in plan['wm_checkpoints']:
        checkpoint = Path(path).stem
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('assets', 'motions', 'packages', 'torch', 'cuda'):
            if loaded['manifest'][key] != expected_manifest[key]:
                raise ValueError(f'WM checkpoint provenance differs: {key}')
        for name, digest in expected_manifest['source_hashes'].items():
            if name.startswith('supermimic/') and loaded['manifest']['source_hashes'][name] != digest:
                raise ValueError(f'WM checkpoint training source differs: {name}')
        probe.wm.load_state_dict(loaded['wm'])
        del loaded
        for seed in plan['gradient_seeds']:
            probe.seed = seed
            pair = timed(checkpoint + '_paired_gradients', seed, lambda: estimate_pair(probe))
            allocations['imagined_gradient_steps'] += 2 * probe.batch * probe.horizon
            for method in ('bptt', 'ppo'):
                name = checkpoint + '_' + method
                add_gradient(name, seed, pair[method], pair['rollout']['objective'])
        for method in ('bptt', 'ppo'):
            name = checkpoint + '_' + method
            means[name] = moments[name].mean.float().clone()
        for reference in references:
            with torch.no_grad():
                state = tuple(x[0].to(device) for x in reference['states'])
                xi = probe.normalize_xi(probe.starts['xi'])
                for h, action in enumerate(reference['actions'], 1):
                    check_budget()
                    state = probe.wm.step(state, action.to(device), xi)
                    allocations['imagined_reference_steps'] += probe.batch
                    if h not in reference['contact_groups']:
                        continue
                    truth = tuple(x[h].to(device) for x in reference['states'])
                    error = probe.wm.state_error(state, truth)
                    if not torch.isfinite(error).all():
                        raise FloatingPointError('Nonfinite open-loop prediction')
                    scaled = (error / scale).square().mean(-1)
                    for n in range(probe.batch):
                        prediction_rows.append(dict(checkpoint=checkpoint, sha256=hashes[path], seed=reference['seed'],
                            sample=n, horizon=h, scaled_mse=float(scaled[n]),
                            joint_position_mse=float(error[n, 12:41].square().mean()),
                            contact_group=reference['contact_groups'][h][n]))
        save_csv(output / 'prediction.csv', prediction_rows)
        for i, small in enumerate(directions):
            direction = torch.zeros(total, device=device)
            direction[section] = torch.as_tensor(small, device=device, dtype=direction.dtype)
            timed(checkpoint + '_wm_reference', i, lambda: measure(direction, str(i), plan['reference_seeds'],
                  [plan['primary_epsilon']], checkpoint, wm_call))
        print(f'AUDIT {checkpoint} prediction and gradients complete', flush=True)

    torch.save({name: vector.cpu() for name, vector in means.items()}, output / 'gradient-means.pt')
    for name, mean in means.items():
        norm = mean.norm()
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError(f'Invalid selected update direction: {name}')
        direction = mean / norm
        for seed, reference in zip(plan['reference_seeds'], references):
            probe.seed = seed
            for epsilon in plan['epsilons']:
                with displaced_parameters(probe.policy, epsilon * direction):
                    result = timed(name + '_physical_update', seed, physical_call)
                # Paired physical effects, never evidence of global optimizer superiority.
                action_rms = float((result['actions'][0].cpu() - reference['actions'][0]).square().mean().sqrt())
                for n in range(probe.batch):
                    update_rows.append(dict(method=name, seed=seed, sample=n, epsilon=epsilon,
                        proxy_improvement=float(result['proxy'][n].cpu() - reference['proxy'][n]),
                        hard_improvement=float(result['hard'][n].cpu() - reference['hard'][n]),
                        initial_action_rms=action_rms, contact_group=reference['contact_groups'][probe.horizon][n]))
        save_csv(output / 'physical-updates.csv', update_rows)
        print(f'AUDIT physical update {name} complete', flush=True)
    if any(sha256(name) != digest for name, digest in hashes.items()):
        raise RuntimeError('WM checkpoint changed during experiment')
    summary = dict(status='complete', seconds=time.perf_counter()-started, allocations=allocations,
                   checkpoint_hashes=hashes, bias_offset=section.start, bias_dimensions=size,
                   parameter_count=total, gradient_statistics={name: m.summary() for name, m in moments.items()},
                   physical_tapes_sha256=sha256(output / 'physical-tapes.pt'),
                   gradient_means_sha256=sha256(output / 'gradient-means.pt'),
                   independent_direction_scope='29 output-bias parameters; not full-policy accuracy',
                   scope='one policy/critic snapshot; noise replicas are not training seeds')
    write_json(output / 'audit-summary.json', summary)
    return summary
