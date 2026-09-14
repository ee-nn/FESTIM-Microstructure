"""Quaternion and cubic-symmetry utilities for EBSD orientation data.

Quaternions use ``(w, x, y, z)`` and Bunge's passive convention; crystal
symmetry acts on the right.
"""

import numpy as np


# Quaternion helpers.
def cubic_symmetry_quaternions():
    """The 24 rotations of the cubic group as unit quaternions: identity, nine
    90/180/270 deg about <100>, six 180 deg about <110>, eight 120/240 about
    <111>."""
    r = np.sqrt(0.5)
    q = [(1.0, 0.0, 0.0, 0.0)]
    for axis in range(3):
        for w, s in ((r, r), (0.0, 1.0), (r, -r)):
            v = [0.0, 0.0, 0.0]
            v[axis] = s
            q.append((w, v[0], v[1], v[2]))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for sign in (1.0, -1.0):
            v = [0.0, 0.0, 0.0]
            v[i], v[j] = r, sign * r
            q.append((0.0, v[0], v[1], v[2]))
    for sx in (0.5, -0.5):
        for sy in (0.5, -0.5):
            for sz in (0.5, -0.5):
                q.append((0.5, sx, sy, sz))
    out = np.array(q, dtype=float)
    assert out.shape == (24, 4), out.shape
    return out


def qmul(a, b):
    """Hamilton product, broadcasting over leading axes."""
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ),
        axis=-1,
    )


def qconj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def euler_bunge_to_quat(phi1, Phi, phi2, degrees=True):
    """Bunge Euler angles -> unit quaternion, passive convention."""
    if degrees:
        phi1, Phi, phi2 = np.radians(phi1), np.radians(Phi), np.radians(phi2)
    sigma = 0.5 * (phi1 + phi2)
    delta = 0.5 * (phi1 - phi2)
    c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
    q = np.stack(
        (c * np.cos(sigma), s * np.cos(delta), s * np.sin(delta), c * np.sin(sigma)),
        axis=-1,
    )
    # Canonicalize the sign for averaging and fundamental-zone reduction.
    return np.where(q[..., :1] < 0, -q, q)


def crystal_equivalents(q, sym):
    """Return right-multiplied cubic equivalents with shape ``(n, 24, 4)``."""
    return qmul(q[:, None, :], sym[None, :, :])


def to_fundamental_zone(q, sym, chunk=50_000):
    """Choose the cubic equivalent closest to identity for each orientation."""
    out = np.empty_like(q)
    for lo in range(0, len(q), chunk):
        blk = q[lo : lo + chunk]
        cand = crystal_equivalents(blk, sym)  # (n, 24, 4)
        best = np.argmax(np.abs(cand[..., 0]), axis=1)
        picked = cand[np.arange(len(blk)), best]
        out[lo : lo + chunk] = np.where(picked[..., :1] < 0, -picked, picked)
    return out


def rodrigues_to_quat(r):
    """Rodrigues vectors (n, 3) -> unit quaternions (n, 4), q0 > 0."""
    r = np.asarray(r, dtype=float).reshape(-1, 3)
    q = np.column_stack((np.ones(len(r)), r))
    return q / np.linalg.norm(q, axis=1)[:, None]


def quat_to_rodrigues(q):
    """Rodrigues vector = (q1, q2, q3) / q0. Requires q already in the FZ."""
    q0 = q[..., :1]
    if np.any(np.abs(q0) < 1e-8):
        raise ValueError("scalar part near zero; reduce to the fundamental zone first")
    return q[..., 1:] / q0


def cubic_disorientation_angle(m):
    """Disorientation angle (degrees) of a cubic misorientation quaternion.

    Closed form rather than a 24 x 24 search: with |components| sorted
    descending as a >= b >= c >= d, the largest attainable cos(omega/2) over
    the cubic group is max(a, (a + b)/sqrt(2), (a + b + c + d)/2) (Grimmer,
    Acta Cryst. A36 (1980) 382). The self-test checks it on two known cases.
    """
    s = np.sort(np.abs(m), axis=-1)[..., ::-1]
    a, b, c, d = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    best = np.maximum.reduce([a, (a + b) / np.sqrt(2.0), 0.5 * (a + b + c + d)])
    return np.degrees(2.0 * np.arccos(np.clip(best, -1.0, 1.0)))
