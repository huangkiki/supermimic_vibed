"""Inspect the original training actor update without importing its simulator entrypoint.

The actor and objective functions below are compiled from the repository's actual
train.py AST. No handwritten copy of its rollout, objective, clipping or optimizer
step is maintained here. The source file and the borrowed modules are not changed.
"""
from __future__ import annotations

import ast
import math
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

TRAIN_SOURCE = Path(__file__).resolve().parents[2] / "supermimic" / "train.py"
_HELPERS = ("_imag_step", "_reward_term", "_critic_input", "set_requires_grad")


def training_actor_source(path=TRAIN_SOURCE):
    """Extract unchanged actor statements, stopping before any critic update.

    Fail if the two explicit boundaries change. Execute only trusted repository
    source, never a checkpoint-supplied path. Module-level launch code is excluded.
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    update = functions["policy_update"]
    zero = [i for i, node in enumerate(update.body)
            if ast.unparse(node) == "opt_pi.zero_grad(set_to_none=True)"]
    thaw = [i for i, node in enumerate(update.body)
            if ast.unparse(node) == "set_requires_grad(wm, True)"]
    if len(zero) != 1 or len(thaw) != 1 or zero[0] >= thaw[0]:
        raise RuntimeError("Original policy_update actor boundaries changed; audit source before probing")
    extracted = [deepcopy(functions[name]) for name in _HELPERS]
    for name, end in (("actor_objective", zero[0]), ("actor_update", thaw[0])):
        function = deepcopy(update)
        function.name = name
        function.body = function.body[:end] + [ast.parse("return locals()").body[0]]
        extracted.append(function)
    module = ast.fix_missing_locations(ast.Module(body=extracted, type_ignores=[]))
    return compile(module, str(path), "exec"), {
        "train_source_sha256": sha256(source.encode()).hexdigest(),
        "actor_source_sha256": sha256(ast.dump(module).encode()).hexdigest(),
        "source": "supermimic/train.py::policy_update",
        "execution": "uncompiled original actor prefix; stop before critic normalization/update",
    }


def parameter_vector(policy, *, gradients=False):
    values = []
    for parameter in policy.parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            value = torch.zeros_like(parameter)
        values.append(value.detach().reshape(-1).clone())
    return torch.cat(values)


def parameter_layout(policy):
    return [{"name": name, "shape": list(p.shape), "numel": p.numel()}
            for name, p in policy.named_parameters()]


@contextmanager
def fixed_rng(seed, device):
    """Common random numbers, with caller CPU and selected-device RNG restored."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        yield


@contextmanager
def preserve_requires_grad(*modules):
    flags = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def validate_starts(starts, history, batch, hist):
    """Require raw replay fields, not already-noisy or fabricated observation history."""
    for key in ("s", "a_prev", "xi", "f"):
        if key not in starts or key not in history:
            raise ValueError(f"Raw replay start/history field missing: {key}")
        if starts[key].shape[0] != batch or history[key].shape[:2] != (batch, hist):
            raise ValueError(f"Invalid replay dimensions: {key}")
        if not torch.equal(starts[key], history[key][:, -1]):
            raise ValueError(f"Replay history must end at its start: {key}")
        if not torch.isfinite(starts[key]).all() or not torch.isfinite(history[key]).all():
            raise ValueError(f"Nonfinite replay input: {key}")


class PolicyGradientProbe:
    """A fixed replay minibatch and frozen model snapshot for original actor updates.

    Modules remain in their supplied train/eval modes. Normalizers are never
    updated. Only the policy and Adam are copied for the captured update; borrowed
    WM/critic requires_grad flags and the caller's RNG are restored on all paths.
    The saved starts describe a new diagnostic update, not a recovered historical
    minibatch unless the caller actually retained that minibatch and RNG.
    """

    def __init__(self, *, policy, wm, critic, mdp, motion, starts, history,
                 unflatten_state, normalize_xi, horizon=32, gamma=.99,
                 goal_frames=10, seed=9001, no_critic=False, detach_action_reward=False):
        if horizon not in (1, 8, 32):
            raise ValueError("Policy fidelity horizons must be 1, 8 or 32")
        self.policy, self.wm, self.critic = policy, wm, critic
        self.mdp, self.motion = mdp, motion
        self.starts = {k: v.detach().clone() for k, v in starts.items()}
        self.history = {k: v.detach().clone() for k, v in history.items()}
        self.batch, self.hist = len(starts["f"]), history["f"].shape[1]
        validate_starts(self.starts, self.history, self.batch, self.hist)
        self.device = next(policy.parameters()).device
        if any(v.device != self.device for d in (self.starts, self.history) for v in d.values()):
            raise ValueError("Policy and fixed replay must share a device")
        f = self.starts["f"]
        if torch.any(f + horizon > motion.clip_end[motion.clip_id[f]]):
            raise ValueError("Fixed replay would cross a motion clip boundary")
        self.horizon, self.gamma, self.goal_frames, self.seed = horizon, gamma, goal_frames, seed
        self.no_critic, self.detach_action_reward = no_critic, detach_action_reward
        self.unflatten_state, self.normalize_xi = unflatten_state, normalize_xi
        self.code, self.provenance = training_actor_source()

    def _namespace(self, policy, dynamics, optimizer=None, clipping=None):
        def sample(batch):
            if batch != self.batch:
                raise ValueError("Changed fixed minibatch size")
            return tuple({k: v.clone() for k, v in data.items()} for data in (self.starts, self.history))

        namespace = dict(torch=torch, nn=clipping or nn, pi=policy, wm=dynamics, critic=self.critic,
                         mdp=self.mdp, motion=self.motion, unflatten_state=self.unflatten_state,
                         normalize_xi=self.normalize_xi, sample_valid_starts=sample,
                         args=SimpleNamespace(pi_batch=self.batch, gamma=self.gamma, no_critic=self.no_critic),
                         K=self.goal_frames, Hh=self.hist, HZ=self.horizon, dev=self.device,
                         PROFILE=False, opt_pi=optimizer)
        exec(self.code, namespace)
        namespace["imag_step"] = namespace["_imag_step"]
        namespace["reward_term"] = namespace["_reward_term"]
        if self.detach_action_reward:
            def reward_term(state, frames, action, previous):
                # Same forward reward. Only its explicit action arguments are detached;
                # state -> action -> policy and soft-termination gradients remain.
                return (self.mdp.reward(state, frames, action.detach(), previous.detach()),
                        self.mdp.soft_termination(state, frames))
            namespace["reward_term"] = reward_term
        return namespace

    @torch.no_grad()
    def objective(self, *, policy=None, dynamics=None):
        policy = self.policy if policy is None else policy
        dynamics = self.wm if dynamics is None else dynamics
        namespace = self._namespace(policy, dynamics)
        with fixed_rng(self.seed, self.device), preserve_requires_grad(dynamics, self.critic):
            result = namespace["actor_objective"]()
        return {"proxy": result["J"].detach().clone(), "rewards": result["r"].detach().clone(),
                "soft_termination": result["d"].detach().clone(),
                "actions": torch.stack(result["As"]).detach().clone()}

    def capture(self, optimizer_state):
        """Return raw/clipped loss gradients and the actual historical-Adam displacement.

        Missing or partial historical Adam moments are an error, not silently
        replaced with a fresh optimizer. All vectors use named_parameters order.
        """
        policy = deepcopy(self.policy)
        parameters = list(policy.parameters())
        if not all(p.requires_grad for p in parameters):
            raise ValueError("Original actor update requires all policy parameters trainable")
        optimizer = torch.optim.Adam(parameters, lr=2e-4, betas=(.9, .95))
        optimizer.load_state_dict(deepcopy(optimizer_state))
        for parameter in parameters:
            state = optimizer.state[parameter]
            if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys() or float(state["step"]) <= 0:
                raise ValueError("Historical Adam state is missing; cannot recover actual update direction")
            if state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape:
                raise ValueError("Historical Adam moment shape differs from policy")
        before = parameter_vector(policy)
        captured = {}

        def clip_norm(params, *args, **kwargs):
            captured["raw_loss_gradient"] = parameter_vector(policy, gradients=True)
            norm = nn.utils.clip_grad_norm_(params, *args, **kwargs)
            captured["norm_clipped_loss_gradient"] = parameter_vector(policy, gradients=True)
            captured["raw_gradient_norm"] = norm.detach().clone()
            return norm

        def clip_value(params, *args, **kwargs):
            nn.utils.clip_grad_value_(params, *args, **kwargs)
            captured["clipped_loss_gradient"] = parameter_vector(policy, gradients=True)

        clipping = SimpleNamespace(Module=nn.Module, utils=SimpleNamespace(
            clip_grad_norm_=clip_norm, clip_grad_value_=clip_value))
        namespace = self._namespace(policy, self.wm, optimizer, clipping)
        with fixed_rng(self.seed, self.device), preserve_requires_grad(self.wm, self.critic):
            result = namespace["actor_update"]()
        if not torch.isfinite(captured["raw_gradient_norm"]) or not torch.isfinite(result["J"]).all():
            raise FloatingPointError("Original policy update skipped nonfinite gradient; no valid Adam direction")
        captured.update(
            objective=result["J"].detach().clone(), loss=result["loss"].detach().clone(),
            raw_ascent_gradient=-captured["raw_loss_gradient"],
            clipped_ascent_gradient=-captured["clipped_loss_gradient"],
            adam_displacement=parameter_vector(policy) - before,
            optimizer_before=deepcopy(optimizer_state), optimizer_after=optimizer.state_dict(),
            parameter_layout=parameter_layout(policy), horizon=self.horizon, gamma=self.gamma,
            seed=self.seed, no_critic=self.no_critic, detach_action_reward=self.detach_action_reward,
            provenance=self.provenance,
        )
        if not torch.isfinite(captured["adam_displacement"]).all():
            raise FloatingPointError("Historical Adam produced a nonfinite displacement")
        return captured


@contextmanager
def displaced_parameters(policy, displacement):
    """Apply a full vector without changing storage; restore even after failed physics."""
    parameters = list(policy.parameters())
    if displacement.numel() != sum(p.numel() for p in parameters) or not torch.isfinite(displacement).all():
        raise ValueError("Invalid policy displacement")
    originals = [p.detach().clone() for p in parameters]
    try:
        offset = 0
        with torch.no_grad():
            for p in parameters:
                p.add_(displacement[offset:offset + p.numel()].view_as(p))
                offset += p.numel()
        yield
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals):
                p.copy_(original)


class ClosedLoopPhysical:
    """Native physics transitions inside the SAME original closed-loop objective.

    The supplied TrackingSim must be in eval_mode (no resets after termination).
    Its trajectories continue for the soft+bootstrap diagnostic; a separate
    pre-transition hard-survival mask stops accumulation of the physical return.
    """

    def __init__(self, probe, env, bank, *, record_contacts=False):
        if not env.eval_mode:
            raise ValueError("Physical probe requires eval_mode without automatic resets")
        for key, start_key in (("state", "s"), ("frames", "f"), ("xi", "xi"), ("previous_action", "a_prev")):
            if not torch.equal(bank[key], probe.starts[start_key]):
                raise ValueError(f"Physical bank and raw replay start differ: {key}")
        self.probe, self.env, self.bank = probe, env, bank
        self.record_contacts = record_contacts

    @torch.no_grad()
    def __call__(self):
        from .fidelity import contact_signature, restore_probe_state, returns
        from supermimic.mdp import flatten_state

        restore_probe_state(self.env, self.bank)
        drift = (flatten_state(self.env.state()) - self.bank["state"]).abs().amax(-1)
        if not torch.isfinite(drift).all() or float(drift.max()) > 1e-5:
            raise RuntimeError(f"Native initial state restore drift exceeds 1e-5: {float(drift.max())}")
        env = self.env
        hard, contacts = [], []
        record_contacts = self.record_contacts

        class PhysicalTransitions(nn.Module):
            def step(self, state, action, xi):
                transition = env.step(action)
                hard.append(transition["done"].clone())
                if record_contacts:
                    contacts.append(contact_signature(env))
                # Snapshot tensors because the simulator reuses its state storage.
                return tuple(x.clone() for x in env.state())

        result = self.probe.objective(dynamics=PhysicalTransitions())
        result["hard"] = returns(result["rewards"], torch.stack(hard).float(), self.probe.gamma)
        result["hard_termination"] = torch.stack(hard)
        result["initial_state_drift"] = drift
        if record_contacts:
            result["geometry_contacts"] = contacts
        return result


def scan_parameter_directions(probe, captured, physical, *, epsilons, random_directions=4, repeats=3):
    """Paired FD of the fixed *batch mean*, plus actual Adam-step improvement.

    `physical` must restore all simulator state on every invocation and return
    per-start `proxy` and `hard` returns. Individual returns are retained but are
    not treated as independent predictions of the batch-mean policy gradient.
    The numerical resolution screen is an operational diagnostic, not truth.
    """
    import numpy as np
    from .fidelity_math import reference_quality

    if len(epsilons) < 2 or sorted(set(epsilons)) != list(epsilons) or any(not math.isfinite(e) or e <= 0 for e in epsilons):
        raise ValueError("At least two unique positive increasing epsilons required")
    if repeats < 2 or random_directions < 1:
        raise ValueError("Require at least two baseline repeats and one random direction")
    def evaluate():
        result = physical()
        for key in ("proxy", "hard"):
            if result[key].shape != (probe.batch,) or not torch.isfinite(result[key]).all():
                raise FloatingPointError(f"Invalid physical return: {key}")
        return result

    baseline = [evaluate() for _ in range(repeats)]
    noise = {key: max(float((result[key].mean() - baseline[0][key].mean()).abs()) for result in baseline[1:])
             for key in ("proxy", "hard")}
    directions = {"raw_gradient": captured["raw_ascent_gradient"],
                  "clipped_gradient": captured["clipped_ascent_gradient"],
                  "historical_adam": captured["adam_displacement"]}
    generator = torch.Generator(device=probe.device).manual_seed(probe.seed + 1)
    for i in range(random_directions):
        directions[f"random_{i}"] = torch.randn(captured["raw_ascent_gradient"].shape, device=probe.device,
                                                dtype=captured["raw_ascent_gradient"].dtype, generator=generator)
    rows, records, saved_directions, skipped = [], {}, {}, []
    for name, vector in directions.items():
        norm = vector.norm()
        if not torch.isfinite(norm) or norm <= 0:
            skipped.append({"direction": name, "reason": "zero or nonfinite norm"})
            continue
        direction = vector / norm
        saved_directions[name] = direction.detach().clone()
        predicted = float(captured["raw_ascent_gradient"] @ direction)
        measurements = []
        before = parameter_vector(probe.policy)
        for epsilon in epsilons:
            with displaced_parameters(probe.policy, epsilon * direction):
                plus_displacement = parameter_vector(probe.policy) - before
                plus = evaluate()
            with displaced_parameters(probe.policy, -epsilon * direction):
                minus_displacement = parameter_vector(probe.policy) - before
                minus = evaluate()
            measurements.append((plus, minus, float(plus_displacement.norm()), float(minus_displacement.norm())))
        records[name] = measurements
        for objective in ("proxy", "hard"):
            deltas = np.array([float(p[objective].mean() - m[objective].mean()) for p, m, _, _ in measurements])
            derivatives = deltas / (2 * np.array(epsilons))
            if not np.isfinite(derivatives).all():
                raise FloatingPointError("Nonfinite physical directional derivative")
            resolved = reference_quality(derivatives, deltas, noise[objective])
            for i, epsilon in enumerate(epsilons):
                plus, minus, plus_norm, minus_norm = measurements[i]
                rows.append(dict(direction=name, objective=objective, horizon=probe.horizon,
                    epsilon=epsilon, predicted_derivative=predicted, numerical_derivative=float(derivatives[i]),
                    sign_agrees=bool(np.sign(predicted) == np.sign(derivatives[i])), resolved=bool(resolved[i]),
                    baseline_return=float(baseline[0][objective].mean()),
                    positive_return=float(plus[objective].mean()), negative_return=float(minus[objective].mean()),
                    positive_improvement=float(plus[objective].mean() - baseline[0][objective].mean()),
                    baseline_repeat_noise=noise[objective], positive_parameter_displacement_l2=plus_norm,
                    negative_parameter_displacement_l2=minus_norm,
                    interpretation="matched_proxy_derivative" if objective == "proxy" and not probe.detach_action_reward
                    else "update_direction_effectiveness_only"))
    with displaced_parameters(probe.policy, captured["adam_displacement"]):
        adam_actual = evaluate()
    return dict(rows=rows, baselines=baseline, perturbations=records, unit_directions=saved_directions,
                skipped=skipped, actual_adam=adam_actual,
                actual_adam_improvement={key: float(adam_actual[key].mean() - baseline[0][key].mean())
                                         for key in ("proxy", "hard")},
                independent_unit="one fixed replay minibatch; directions/scales/repeats are paired diagnostics",
                hard_return_interpretation="physical effectiveness; not the derivative of the WM soft+bootstrap target")


def summarize_direction_scan(result):
    """Separate scales/objectives and random draws from selected optimizer directions."""
    import numpy as np
    from .fidelity_math import agreement
    groups = {}
    for row in result['rows']:
        family = 'random_directions' if row['direction'].startswith('random_') else 'selected_update_directions'
        key = row['objective'], row['epsilon'], family
        groups.setdefault(key, []).append(row)
    summaries = []
    for (objective, epsilon, family), rows in groups.items():
        stats = agreement(np.array([r['predicted_derivative'] for r in rows]),
                          np.array([r['numerical_derivative'] for r in rows]),
                          np.array([r['resolved'] for r in rows]))
        summaries.append(dict(objective=objective, epsilon=epsilon, direction_family=family,
                              **stats, scope=result['independent_unit']))
    return dict(summaries=summaries, actual_adam_improvement=result['actual_adam_improvement'],
                skipped=result['skipped'], hard_return_interpretation=result['hard_return_interpretation'])
