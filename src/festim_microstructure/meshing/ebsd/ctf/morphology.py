"""Raster clean-up: hole filling, and the two topologies ``neper -M`` rejects.

Separate from :mod:`.segmentation` because none of this changes which pixels
belong together -- it only changes the shape of what they form, so that Neper's
interface reconstruction can walk it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["STRUCT4", "fill_holes", "make_meshable"]


def fill_holes(cellids):
    """Assign every empty voxel to its nearest cell.

    Rejected points and pruned grains leave holes in `**data`, and a hole is an
    interior surface as far as the fit's `pts(region=surf)` control points are
    concerned -- so a map that is half holes has the objective function chasing
    the boundaries of the noise rather than the boundaries of the grains.

    Neper's own `grow` transform does this, but reaching it means putting the
    file back through `neper -T -transform`, which is the write path that
    produces an unreadable raster when the file carries `**oridata` (Neper
    5.0.0). Doing it here keeps the orientations and avoids that entirely.

    `**oridef` is deliberately left alone: it still records which points were
    actually indexed, so the provenance of a filled voxel is not lost.
    """
    from scipy.ndimage import distance_transform_edt

    empty = cellids == 0
    n = int(empty.sum())
    if n == 0 or n == empty.size:
        return cellids, n
    # The empty-mask distance transform identifies nearest filled voxels.
    _, idx = distance_transform_edt(empty, return_indices=True)
    return cellids[tuple(idx)], n


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _absorb_enclosed(cellids):
    """Absorb every group of cells fully enclosed by one grain into that grain.

    A grain containing an island has a boundary made of two loops. Neper's
    walk (see `make_meshable`) finishes the outer one and then has edges left
    over. The island itself is degenerate too: the ring around it carries no
    triple junction, so no vertex is created on it and its edge ends up with
    none.
    """
    from scipy.ndimage import label

    absorbed = []
    for cell in range(1, int(cellids.max()) + 1):
        # the complement of one grain, in the 4-connectivity of the voxel faces
        # the reconstruction works with; a component of it that does not reach
        # the map border is enclosed by that grain
        comp, n = label(cellids != cell, structure=STRUCT4)
        if n <= 1:
            continue
        border = set(comp[0]) | set(comp[-1]) | set(comp[:, 0]) | set(comp[:, -1])
        inside = [k for k in range(1, n + 1) if k not in border]
        if inside:
            mask = np.isin(comp, inside)
            absorbed += sorted(set(cellids[mask].tolist()))
            cellids[mask] = cell
    return absorbed


def _pinch_nodes(cellids):
    """Nodes where one grain occupies both pixels of a diagonal and neither of
    the other two, i.e. it touches itself at a point.

    Its boundary is then a figure-eight through that node rather than a loop,
    which breaks the same walk. The grain need not be in two pieces for this:
    an arm folding back on itself pinches while staying connected elsewhere.
    Pairs of pixels are returned, one pair per node.
    """
    a, b = cellids[:-1, :-1], cellids[:-1, 1:]
    c, d = cellids[1:, :-1], cellids[1:, 1:]
    out = []
    for mask, off in (
        ((a == d) & (a != b) & (a != c), ((0, 0), (1, 1))),
        ((b == c) & (b != a) & (b != d), ((0, 1), (1, 0))),
    ):
        ys, xs = np.nonzero(mask)
        out += [
            ((y + off[0][0], x + off[0][1]), (y + off[1][0], x + off[1][1]))
            for y, x in zip(ys.tolist(), xs.tolist())
        ]
    return out


def _unpinch(cellids):
    """Hand one pixel of each corner-only self-contact to a neighbouring grain.

    The pixel taken is whichever of the two is least attached to its own grain
    (a one-pixel spur left by the back-fill, usually), and it goes to whichever
    grain holds most of its four neighbours, so the change is one pixel per
    node and always in favour of an already-adjacent grain.
    """
    ny, nx = cellids.shape
    fixed = 0
    for p, q in _pinch_nodes(cellids):
        cell = cellids[p]
        if cellids[q] != cell:  # already resolved by an earlier fix
            continue

        def neighbours(pixel):
            y, x = pixel
            return [
                (y + dy, x + dx)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= y + dy < ny and 0 <= x + dx < nx
            ]

        victim = min(
            (p, q), key=lambda t: sum(cellids[n] == cell for n in neighbours(t))
        )
        others = [cellids[n] for n in neighbours(victim) if cellids[n] != cell]
        if not others:
            continue
        vals, counts = np.unique(others, return_counts=True)
        cellids[victim] = vals[np.argmax(counts)]
        fixed += 1
    return fixed


def make_meshable(cellids):
    """Remove the two raster configurations `neper -M` cannot reconstruct.

    Neper reconstructs the raster's interfaces into a vertex/edge/face topology
    and then orders each face's edges by walking them as a single closed loop
    (neut_tess_init_facetopo_fromver, in neut_tess_op1.c). Anything whose
    boundary is not one loop makes the walk run out of edges, and Neper aborts
    with `ut_print_neperbug()` right after "Reconstructing 0D mesh... 100%".
    Two things do that: a grain enclosing another (two loops) and a grain
    touching itself at a corner (a figure-eight).

    Both are fixed in one loop because each fix can create the other: absorbing
    an island can leave the host touching itself where the island had separated
    two arms, and handing away a pinched pixel can close a ring around a
    neighbour.

    What is lost is small and inert. An enclosed grain's boundary is a closed
    ring that meets nothing else, i.e. an isolated component of the grain
    boundary network (an enclosed pair gives a ring with one chord, likewise
    isolated), so removing it changes no connected path and leaves percolation
    untouched; a pinch is a one-pixel spur. The ids and counts are returned so
    the loss stays on the record.
    """
    cellids = cellids.copy()
    absorbed, pinches = [], 0
    for _ in range(20):
        n = _unpinch(cellids)
        new = _absorb_enclosed(cellids)
        pinches += n
        absorbed += new
        if not n and not new:
            break
    else:
        print(
            "  WARNING: enclosure/pinch cleanup did not converge; `neper -M` "
            "may still abort in neut_tess_init_facetopo_fromver"
        )

    ids = np.unique(cellids[cellids > 0])
    remap = np.zeros(int(cellids.max()) + 1, dtype=np.int64)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[cellids], sorted(absorbed), pinches
