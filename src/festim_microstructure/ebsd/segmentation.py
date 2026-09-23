"""Flood-fill pixels into grains, prune, and average each grain's orientation."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from festim_microstructure.ebsd.orientation import (
    crystal_equivalents,
    cubic_disorientation_angle,
    qconj,
    qmul,
    to_fundamental_zone,
)

__all__ = ["grain_mean_orientations", "relabel_and_prune", "segment_grains"]


def segment_grains(qgrid, ok, threshold, sym):
    """Flood-fill across neighbours whose disorientation is below `threshold`.

    Four-connected: a pixel is joined to the one on its right and the one below
    when the two orientations are close enough. The resulting graph's connected
    components are the grains. Rejected pixels join nothing and end up as id 0.
    """
    ny, nx = ok.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols = [], []

    for shift_axis in (1, 0):  # right neighbour, then lower neighbour
        if shift_axis == 1:
            a, b = ok[:, :-1] & ok[:, 1:], None
            qa, qb = qgrid[:, :-1], qgrid[:, 1:]
            ia, ib = idx[:, :-1], idx[:, 1:]
        else:
            a, b = ok[:-1, :] & ok[1:, :], None
            qa, qb = qgrid[:-1, :], qgrid[1:, :]
            ia, ib = idx[:-1, :], idx[1:, :]
        del b
        pa, pb = qa[a], qb[a]
        if pa.size == 0:
            continue
        ang = cubic_disorientation_angle(qmul(qconj(pa), pb))
        same = ang < threshold
        rows.append(ia[a][same])
        cols.append(ib[a][same])

    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
    else:
        r = c = np.zeros(0, dtype=int)

    graph = coo_matrix(
        (np.ones(len(r)), (r, c)), shape=(ny * nx, ny * nx), dtype=np.int8
    )
    _n, labels = connected_components(graph, directed=False)
    labels = labels.reshape(ny, nx)
    labels[~ok] = -1
    return labels


def relabel_and_prune(labels, ok, min_pixels):
    """Drop tiny grains, then renumber what survives contiguously from 1.

    Neper requires the `**data` section to be numbered contiguously from 1, with
    0 for empty voxels. Pruned pixels become 0 and are treated exactly like
    unindexed ones -- the tessellation fit only uses cell boundaries, and the
    `grow` transform can fill the holes later if they matter.
    """
    flat = labels.ravel()
    valid = flat >= 0
    uniq, inv, counts = np.unique(flat[valid], return_inverse=True, return_counts=True)
    keep = counts >= min_pixels
    newid = np.zeros(len(uniq), dtype=np.int64)
    newid[keep] = np.arange(1, int(keep.sum()) + 1)

    out = np.zeros_like(flat, dtype=np.int64)
    out[valid] = newid[inv]
    out = out.reshape(labels.shape)
    dropped = int((~keep).sum())
    lost = int(counts[~keep].sum())
    return out, int(keep.sum()), dropped, lost


def grain_mean_orientations(
    qgrid, cellids, ncells, sym, chunk=50_000, sample_mask=None
):
    """One orientation per grain: the symmetry-aligned quaternion mean.

    Each pixel is first mapped to the symmetry equivalent closest to its grain's
    reference orientation, otherwise the average of two equivalent descriptions
    of the same orientation is not that orientation. ``sample_mask`` can exclude
    pixels that belong to the final grain geometry but should not influence its
    representative orientation, such as rejected or back-filled pixels.
    """
    flat_q = qgrid.reshape(-1, 4)
    flat_id = cellids.ravel()
    if sample_mask is not None:
        sample_mask = np.asarray(sample_mask, dtype=bool)
        if sample_mask.shape != cellids.shape:
            raise ValueError("sample_mask and cellids must have the same shape")
        flat_id = np.where(sample_mask.ravel(), flat_id, 0)
    order = np.argsort(flat_id, kind="stable")
    sorted_id = flat_id[order]
    starts = np.searchsorted(sorted_id, np.arange(1, ncells + 1))
    ends = np.searchsorted(sorted_id, np.arange(1, ncells + 1), side="right")

    means = np.zeros((ncells, 4))
    for k in range(ncells):
        members = order[starts[k] : ends[k]]
        if members.size == 0:
            raise ValueError(f"grain {k + 1} has no pixels usable for orientation")
        qs = flat_q[members]
        ref = qs[0]
        acc = np.zeros(4)
        for lo in range(0, len(qs), chunk):
            blk = qs[lo : lo + chunk]
            cand = crystal_equivalents(blk, sym)
            dots = cand @ ref
            best = np.argmax(np.abs(dots), axis=1)
            picked = cand[np.arange(len(blk)), best]
            sign = np.sign(dots[np.arange(len(blk)), best])
            sign[sign == 0] = 1.0
            acc += (picked * sign[:, None]).sum(axis=0)
        means[k] = acc / np.linalg.norm(acc)
    return to_fundamental_zone(means, sym)
