"""Numerical-reference diagnostics; no simulator Jacobian is assumed to be truth."""
from __future__ import annotations

import numpy as np


def reference_quality(derivatives, return_deltas, repeat_noise, *, absolute_floor=1e-7, relative_tolerance=0.25):
    """A direction is resolved only with signal and agreement at adjacent scales.

    Inputs are in INCREASING epsilon order. Repeated zero-perturbation return
    differences bound run-to-run noise, not floating-point truncation error.
    This flag is an operational reliability screen, never a truth certificate.
    """
    d = np.asarray(derivatives, dtype=float)
    signal = np.abs(np.asarray(return_deltas, dtype=float)) > max(5 * repeat_noise, absolute_floor)
    valid = np.isfinite(d) & signal
    stable = np.zeros(len(d), dtype=bool)
    for i in range(len(d) - 1):
        scale = max(abs(d[i]), abs(d[i + 1]), 1e-12)
        if valid[i] and valid[i + 1] and abs(d[i] - d[i + 1]) <= relative_tolerance * scale:
            stable[i:i + 2] = True
    return stable


def agreement(reference, predicted, resolved):
    ref, pred = np.asarray(reference), np.asarray(predicted)
    keep = np.asarray(resolved, dtype=bool) & np.isfinite(ref) & np.isfinite(pred)
    a, b = ref[keep], pred[keep]
    corr = None
    if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    return dict(total=int(len(ref)), resolved=int(keep.sum()), unresolved=int((~keep).sum()),
                sign_agreement=float(np.mean(np.sign(a) == np.sign(b))) if len(a) else None,
                pearson=corr)


def contact_groups(contact_signatures, radius=2):
    """Labels from observed solver contact-pair sets at control-step boundaries.

    Returns a label for each time/sample: within radius of a pair-set change is
    ``switch_nearby``; otherwise ``stable_contact`` or ``flight``. The signal does
    not claim substep event timing or force-grounded contact truth.
    """
    time_count, samples = len(contact_signatures), len(contact_signatures[0])
    result = []
    for t in range(time_count):
        row = []
        for n in range(samples):
            changed = any(contact_signatures[j][n] != contact_signatures[j - 1][n]
                          for j in range(max(1, t - radius), min(time_count, t + radius + 1)))
            row.append("switch_nearby" if changed else "stable_contact" if contact_signatures[t][n] else "flight")
        result.append(row)
    return result
