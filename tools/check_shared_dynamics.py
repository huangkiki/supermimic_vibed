"""Compare world models on the same native transitions, separate from policy tracking."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoints', nargs='+', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--motion_dir', type=Path, default=REPO / 'data' / 'lafan1_g1')
parser.add_argument('--seed', type=int, default=7001)
parser.add_argument('--frames_per_clip', type=int, default=4)
parser.add_argument('--steps', type=int, default=8)
add_launcher_args(parser)
args = parser.parse_args()

import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.networks import WorldModel
from supermimic.robot import default_joint_pos_list
from supermimic.state import normalize_xi, unflatten_state

DEVICE = 'cuda:0'
COMPONENTS = ('position_m', 'rotation_rad', 'linear_velocity_m_s', 'angular_velocity_rad_s', 'joint_position_rad', 'joint_velocity_rad_s')
WIDTHS = (3, 3, 3, 3, 29, 29)


@torch.no_grad()
def collect(env, motion, starts, noise):
    env.reset(torch.arange(env.N, device=DEVICE), starts)
    env.done_eval.zero_()
    env.failed_eval.zero_()
    alive = torch.ones(env.N, dtype=torch.bool, device=DEVICE)
    records = {key: [] for key in ('s', 's1', 'a', 'xi', 'mask')}
    for _ in range(args.steps):
        action = (motion.q[env.f + 1] - env.default_q) / env.alpha
        action = (action + noise * torch.randn_like(action)).clamp(-20, 20)
        transition = env.step(action)
        for key in ('s', 's1', 'a', 'xi'):
            records[key].append(transition[key].clone())
        records['mask'].append(alive.clone())
        alive &= ~transition['done']
    return {key: torch.stack(values, dim=1) for key, values in records.items()}


def metrics(error, mask, scale):
    valid = error[mask]
    if not len(valid) or not torch.isfinite(valid).all():
        raise ValueError('Empty or nonfinite dynamics comparison')
    raw = [part.square().mean().sqrt().item() for part in valid.split(WIDTHS, dim=-1)]
    return {
        'valid_transitions': len(valid),
        'common_normalized_mse': (valid / scale).square().mean().item(),
        'component_rmse': dict(zip(COMPONENTS, raw)),
    }


@torch.no_grad()
def score(model, data, scale):
    state = unflatten_state(data['s'])
    successor = unflatten_state(data['s1'])
    context = normalize_xi(data['xi'])
    one_step = model.step(state, data['a'], context)
    prediction = tuple(value[:, 0] for value in state)
    errors = []
    for step in range(args.steps):
        prediction = model.step(prediction, data['a'][:, step], context[:, step])
        errors.append(model.state_error(prediction, tuple(value[:, step] for value in successor)))
    rollout_error = torch.stack(errors, dim=1)
    return {
        'one_step': metrics(model.state_error(one_step, successor), data['mask'], scale),
        'rollout_all_steps': metrics(rollout_error, data['mask'], scale),
        'rollout_final_step': metrics(rollout_error[:, -1], data['mask'][:, -1], scale),
    }


torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
kin = G1Kinematics(DEVICE)
motion = MotionLib.from_dir(str(args.motion_dir), kin, DEVICE)
assert len(motion.names) == 40
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=DEVICE))
clip_ids = torch.arange(40, device=DEVICE).repeat_interleave(args.frames_per_clip)
starts = motion.clip_start[clip_ids] + (torch.rand(len(clip_ids), device=DEVICE) * (motion.clip_len[clip_ids] - args.steps - 2)).long()
sim_cfg = make_sim(DEVICE, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, len(clip_ids), motion, mdp, eval_mode=True)
    probes = {name: collect(env, motion, starts, noise) for name, noise in [('reference_pd', 0.0), ('noisy_reference_pd', 0.1)]}

    deltas = torch.cat([WorldModel.targets(unflatten_state(data['s']), unflatten_state(data['s1']))[data['mask']] for data in probes.values()])
    floor = torch.tensor([value for value, width in zip((3e-3, 5e-3, 2e-2, 5e-2, 3e-3, 5e-2), WIDTHS) for _ in range(width)], device=DEVICE)
    scale = deltas.std(dim=0, correction=0).clamp_min(floor)
    # Check scorer units against identical states and a known one-metre offset.
    reference = unflatten_state(probes['reference_pd']['s1'])
    mask = probes['reference_pd']['mask']
    identity = metrics(WorldModel.state_error(reference, reference), mask, scale)
    assert identity['common_normalized_mse'] < 1e-6, identity
    shifted = (reference[0] + torch.tensor([1., 0., 0.], device=DEVICE), *reference[1:])
    offset = metrics(WorldModel.state_error(shifted, reference), mask, scale)
    assert abs(offset['component_rmse']['position_m'] - (1 / 3) ** 0.5) < 1e-5, offset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output.with_suffix('.pt')
    torch.save({'probes': {name: {key: value.cpu() for key, value in data.items()} for name, data in probes.items()},
                'starts': starts.cpu(), 'clip_ids': clip_ids.cpu(), 'motion_names': motion.names, 'scale': scale.cpu()}, dataset_path)
    result = {
        'scope': 'fixed reference-start native probes on the training motion library; not unseen-motion generalization or policy tracking',
        'seed': args.seed, 'frames_per_clip': args.frames_per_clip, 'steps': args.steps,
        'action_noise_std': {'reference_pd': 0.0, 'noisy_reference_pd': 0.1},
        'domain_randomization': False, 'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        'scorer_checks': {'identical_states': identity, 'one_metre_translation': offset},
        'models': {},
    }
    for path_string in args.checkpoints:
        path = Path(path_string)
        checkpoint = torch.load(path, map_location=DEVICE)
        model_args = {'latent_dim': checkpoint['args'].get('wm_latent_dim', 0)}
        if checkpoint['args'].get('wm_state_skip', False):
            model_args['state_skip'] = True
        model = WorldModel(**model_args).to(DEVICE).eval()
        model.load_state_dict(checkpoint['wm'], strict=True)
        networks.AMP_DTYPE = torch.bfloat16 if checkpoint['args'].get('amp') == 'bf16' else None
        result['models'][str(path)] = {
            'iteration': checkpoint['iteration'], 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sim_steps': checkpoint['sim_steps'], 'wall_time': checkpoint['wall_time'],
            'wm_parameters': sum(parameter.numel() for parameter in model.parameters()),
            'peak_torch_allocated_bytes': checkpoint.get('peak_gpu_allocated_bytes'),
            'peak_torch_reserved_bytes': checkpoint.get('peak_gpu_reserved_bytes'),
            'probes': {name: score(model, data, scale) for name, data in probes.items()},
        }
        del model, checkpoint
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Shared dynamics comparison saved:', args.output, flush=True)
