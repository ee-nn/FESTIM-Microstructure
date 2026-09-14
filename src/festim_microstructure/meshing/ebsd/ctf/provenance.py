"""The json sidecar that lets a .tesr be lined back up with its .ctf."""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import fields as dataclass_fields
from pathlib import Path

from ..segmentation_error import SegmentationError
from .settings import Settings

__all__ = ["settings_from_provenance", "write_provenance"]


def write_provenance(path, opt: Settings, seg: SegmentationError, vox, log=print):
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


def settings_from_provenance(path):
    """Rebuild the `Settings` of a conversion from its provenance json."""
    rec = json.loads(Path(path).read_text())
    known = {f.name for f in dataclass_fields(Settings)}
    return Settings(**{k: v for k, v in rec.items() if k in known})
