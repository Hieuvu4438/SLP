# RUNNABLE_SMOKE
import numpy as np

EPS = 1e-12


def unit(x):
    return x / np.linalg.norm(x)


def project(q, v):
    return v - (q @ v) * q


def sphere_log(q, p):
    dot = float(np.clip(q @ p, -1.0, 1.0))
    h = p - dot * q
    length = np.linalg.norm(h)
    if length > EPS:
        return (np.arctan2(length, dot) / length) * h
    if dot >= 0.0:
        return np.zeros_like(q)
    basis = np.eye(len(q))[np.argmin(np.abs(q))]
    return np.pi * unit(project(q, basis))


def sphere_exp(q, v):
    a = np.linalg.norm(v)
    return np.cos(a) * q + np.sinc(a / np.pi) * v


def bridge(q, p, t, raw):
    tangent = t * sphere_log(q, p) + t * (1.0 - t) * project(q, raw)
    return sphere_exp(q, tangent)


def check_layout_and_grid():
    row = np.arange(133, dtype=float)
    pose = row[:123].reshape(41, 3)
    expr = row[123:]
    padded = np.r_[np.zeros(36), row]
    # Padded order: root+body = 66, two hands = 90, jaw = 3, face = 10.
    np.testing.assert_equal(padded[66:111], pose[10:25].ravel())
    np.testing.assert_equal(padded[111:156], pose[25:40].ravel())
    np.testing.assert_equal(padded[156:159], pose[40])
    np.testing.assert_equal(padded[159:169], expr)
    np.testing.assert_equal(np.r_[pose.ravel(), expr], row)

    def grid(frames, stride=4):
        count = 1 + (frames - 1 + stride - 1) // stride
        return np.linspace(0.0, 1.0, count)

    assert [len(grid(n)) for n in (1, 2, 4, 5, 6, 9, 10)] == [1, 2, 2, 2, 3, 3, 4]
    np.testing.assert_equal(grid(1), [0.0])
    np.testing.assert_equal(grid(6), [0.0, 0.5, 1.0])


def check_quaternion_geometry(q, p, raw):
    signed = sphere_log(q, p)
    physical = 2.0 * np.arccos(np.clip(abs(q @ p), 0.0, 1.0))
    assert np.linalg.norm(signed) > np.pi / 2.0 and physical < 0.5
    for endpoint in (p, q, -q):
        np.testing.assert_allclose(sphere_exp(q, sphere_log(q, endpoint)), endpoint, atol=1e-10)
    np.testing.assert_allclose(bridge(q, p, 0.0, raw), q, atol=1e-10)
    np.testing.assert_allclose(bridge(q, p, 1.0, raw), p, atol=1e-10)
    for t in np.linspace(0.0, 1.0, 11):
        np.testing.assert_allclose(np.linalg.norm(bridge(q, p, t, raw)), 1.0, atol=1e-10)


def check_energy(q, p, raw):
    t, step, rho, sigma = 0.37, 1e-5, 1e-6, 0.5
    x = bridge(q, p, t, raw)
    v = (bridge(q, p, t + step, raw) - bridge(q, p, t - step, raw)) / (2.0 * step)
    assert abs(x @ v) < 1e-8
    prototypes = np.array([q, p, unit(np.array([0.75, 0.1, 0.4, 0.5]))])
    delta = np.array([sphere_log(x, c) for c in prototypes])
    weights = np.array([0.3, 0.25, 0.45]) * np.exp(-np.sum(delta**2, axis=1) / (2 * sigma**2))
    mass = weights.sum()
    scatter = delta.T @ (weights[:, None] * delta)
    metric = mass * np.linalg.inv(scatter + rho * np.eye(4))  # dense reference only
    reference = v @ np.linalg.solve(metric, v)
    fast = (rho * (v @ v) + np.sum(weights * (delta @ v) ** 2)) / mass
    assert fast > 0.0
    np.testing.assert_allclose(fast, reference, rtol=1e-8, atol=1e-8)


def check_damped_ik():
    jac = np.array([[1.0, 0.3], [0.4, -0.9], [0.1, 0.5]])
    target = np.array([0.2, -0.1, 0.3])
    residual = -target
    delta = np.linalg.solve(jac.T @ jac + 0.1**2 * np.eye(2), -jac.T @ residual)
    assert np.linalg.norm(jac @ delta - target) < np.linalg.norm(residual)


def check_replanning(q, p, raw):
    for n in (1, 3, 5, 10):
        e0, end_e = np.array([0.4, -1.2]), np.array([1.3, 0.5])
        x, e, calls = q, e0.copy(), 0
        for k in range(n):
            t, s = k / n, (k + 1) / n
            calls += 1  # mock predictor always returns (p, end_e)
            x = bridge(q, p, s, raw) if k + 1 < n else p
            e = e + (s - t) / (1.0 - t) * (end_e - e)
            np.testing.assert_allclose(e, (1.0 - s) * e0 + s * end_e, atol=1e-10)
        assert calls == n
        np.testing.assert_allclose(x, p, atol=1e-10)
        np.testing.assert_allclose(e, end_e, atol=1e-10)


if __name__ == "__main__":
    q0 = np.array([0.1, np.sqrt(0.99), 0.0, 0.0])
    q1 = np.array([0.1, -np.sqrt(0.99), 0.0, 0.0])
    raw_delta = np.array([0.1, 0.2, -0.3, 0.4])
    check_layout_and_grid()  # codec plus variable-length grid
    check_quaternion_geometry(q0, q1, raw_delta)
    check_energy(q0, q1, raw_delta)
    check_damped_ik()
    check_replanning(q0, q1, raw_delta)
    print("PASS: codec, grid, sphere bridge, kinetic energy, DLS, replanning")
