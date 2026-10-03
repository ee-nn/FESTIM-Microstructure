"""Source provenance for an imported EBSD archive."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

__all__ = ["read_provenance", "write_provenance"]


def write_provenance(path, opt, seg, vox, log=print):
    """Save source settings and quantitative orientation diagnostics."""
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
    """Read the conversion metadata stored alongside an EBSD archive."""
    return json.loads(Path(path).read_text())
