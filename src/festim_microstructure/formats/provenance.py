"""The json sidecar that lets a .tesr be lined back up with its .ctf."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

__all__ = ["read_provenance", "write_provenance"]


def write_provenance(path, opt, seg, vox, log=print):
    """Everything needed to line the .tesr back up with the .ctf, plus the
    segmentation error, as json.

    The window, the mirror and the orientation convention are choices made in
    `Settings` and are not recoverable from the .tesr itself, so
    `measure_tesr_against_ctf` cannot re-measure the conversion without them.
    The statistics are copied in so that a later stage can quote stage 1's
    error without re-reading the .ctf. Dumping the whole dataclass means a
    field added to `Settings` reaches the file without a second edit here.
    """
    rec = asdict(opt)
    rec.update(
        {
            "ctf": str(opt.ctf),
            "unit": opt.unit,
            "voxsize": list(vox),
            "ncells": int(seg.ncells),
            "segmentation_error_deg": {
                name: asdict(getattr(seg, name))
                for name in ("all", "indexed", "backfilled")
            },
        }
    )
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    if log:
        log(f"  wrote {path}")
    return path


def read_provenance(path):
    """Read the conversion metadata stored alongside a TESR file."""
    return json.loads(Path(path).read_text())
