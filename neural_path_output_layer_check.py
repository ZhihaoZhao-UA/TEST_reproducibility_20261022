"""Numerical checks of Supplement S8; not Monte Carlo evidence of predictive superiority.
Run with Python 3 and NumPy. The analytical lemma, not these checks, is the proof.
"""
from pathlib import Path
import json, math
import numpy as np
P = Path(__file__).resolve().parent
rng = np.random.default_rng(1052026)
max_reconstruction = 0.0
max_dimension_violation = -math.inf
max_leverage_violation = -math.inf
cases = 30
for case in range(cases):
    h, m, N = 12, 40, 257
    distinct = 3 if case % 2 == 0 else 6
    inputs = rng.uniform(-1, 1, (N, 2))
    hidden = rng.normal(size=(distinct, 3))
    hidden = np.tile(hidden, (h // distinct, 1))
    psi = np.tanh(np.c_[inputs, np.ones(N)] @ hidden.T) / np.sqrt(h)
    ids = rng.integers(N, size=m)
    y = np.sin(inputs[ids, 0]) + .3 * rng.normal(size=m)
    H = psi[ids].T @ psi[ids] / m
    c = psi[ids].T @ y / m
    eig, V = np.linalg.eigh(H)
    ix = eig > 1e-11
    U = V[:, ix]
    assert U.shape[1] == distinct
    # Distinct active eigenvalues imply the Krylov space equals this range.
    assert np.min(np.diff(eig[ix])) > 1e-10
    assert np.min(np.abs(U.T @ c)) > 1e-12
    alpha = 1 / eig[-1]
    b = np.zeros(h)
    times = [0, 1, 2, 4, 8, 16, 64, 256, 1000]
    states = [b.copy()]
    for t in range(1, max(times) + 1):
        b = b - alpha * (H @ b - c)
        if t in times:
            states.append(b.copy())
    states = np.array(states)
    F = psi @ states.T
    z = F[:, :-1] - F[:, -1, None]
    phi = psi @ U
    B = (states[:-1] - states[-1]) @ U
    max_reconstruction = max(max_reconstruction, float(np.max(abs(z - phi @ B.T))))
    C = phi.T @ phi / N
    e, Q = np.linalg.eigh(C)
    leverage = np.sum((phi @ Q) ** 2 / e, axis=1).max()
    S = z.T @ z / N
    ev, W = np.linalg.eigh(S)
    for lam in (1 / 128, 1 / 512, 1 / 4096):
        rowlev = np.sum((z @ W) ** 2 / (np.maximum(ev, 0) + lam), axis=1)
        max_dimension_violation = max(max_dimension_violation, float(rowlev.mean() - distinct))
        max_leverage_violation = max(max_leverage_violation, float(rowlev.max() - leverage))
assert max_reconstruction < 1e-9
assert max_dimension_violation <= 1e-8 and max_leverage_violation <= 1e-8
H = math.tanh(1) ** 2
C = 1 - math.tanh(1)
L = H / C
oracle = C * (1 - H) ** 32
example = dict(input_second_moment=C, training_gram=H, directional_leverage=L,
    oracle_risk=oracle, M=6, n=512, sigma_squared=.09, delta_x=.05, delta_s=.05,
    sample_size_threshold=32 * L * math.log(2 * 5 / .05),
    excess_bound=2 / 512 + 24 * (.09 + L * oracle) / (11 * 512 * .05),
    hull_wide_excess=C - oracle)
assert example['sample_size_threshold'] < 512
assert example['excess_bound'] < example['hull_wide_excess']
report = dict(status='Analytic example arithmetic and random algebra checks, not a new predictive experiment',
    seed=1052026, random_cases=cases, max_path_representation_error=max_reconstruction,
    max_effective_dimension_violation=max_dimension_violation,
    max_leverage_violation=max_leverage_violation, closed_form_example=example)
(P / 'neural_path_output_layer_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
