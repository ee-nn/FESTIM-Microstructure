"""Versioned EBSD arrays, independent of the importer and simulation stack."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["EbsdData", "read_ebsd", "write_ebsd"]


@dataclass
class EbsdData:
    """Grain geometry and orientations in the package Bunge convention.

    Positive labels are contiguous grain IDs. Zero represents unmeshed pixels.
    ``source_labels`` retains DefDAP's membership before filling; only these
    indexed pixels determine the grain orientations and indexed error.
    Spacings are in the source CTF units, multiplied by the configured scale.
    All arrays share the output map frame, including any y-axis frame change.
    """

    labels: np.ndarray
    grain_quats: np.ndarray
    vox: tuple
    pixel_quats: np.ndarray | None = None
    indexed: np.ndarray | None = None
    source_labels: np.ndarray | None = None
    crysym: str = "cubic"

    def __post_init__(self):
        self.labels = np.asarray(self.labels)
        self.grain_quats = np.asarray(self.grain_quats, dtype=float)
        self.vox = tuple(float(v) for v in self.vox)
        if self.labels.ndim != 2 or self.labels.dtype.kind not in "iu":
            raise ValueError("labels must be a 2D integer array")
        self.labels = self.labels.astype(np.int32)
        ids = np.unique(self.labels[self.labels > 0])
        if (
            np.any(self.labels < 0)
            or not np.array_equal(ids, np.arange(1, len(ids) + 1))
            or not len(ids)
        ):
            raise ValueError("labels must contain contiguous grain IDs starting at 1")
        if self.crysym != "cubic":
            raise ValueError(
                "EBSD disorientation diagnostics currently require cubic m-3m symmetry"
            )
        if len(self.vox) != 2 or not np.isfinite(self.vox).all() or min(self.vox) <= 0:
            raise ValueError("pixel spacings must be finite and positive")
        if (
            self.grain_quats.shape != (len(ids), 4)
            or not np.isfinite(self.grain_quats).all()
            or not np.allclose(np.linalg.norm(self.grain_quats, axis=-1), 1)
        ):
            raise ValueError("each grain must have a finite unit quaternion")
        if self.pixel_quats is None:
            self.pixel_quats = np.zeros((*self.labels.shape, 4))
            self.pixel_quats[..., 0] = 1
        self.pixel_quats = np.asarray(self.pixel_quats, dtype=float)
        if (
            self.pixel_quats.shape != (*self.labels.shape, 4)
            or not np.isfinite(self.pixel_quats).all()
            or not np.allclose(np.linalg.norm(self.pixel_quats, axis=-1), 1)
        ):
            raise ValueError("each pixel must have a finite unit quaternion")
        self.indexed = np.asarray(
            self.labels > 0 if self.indexed is None else self.indexed, dtype=bool
        )
        self.source_labels = np.asarray(
            self.labels if self.source_labels is None else self.source_labels,
            dtype=np.int32,
        )
        if (
            self.indexed.shape != self.labels.shape
            or self.source_labels.shape != self.labels.shape
        ):
            raise ValueError("pixel masks and labels must share a shape")
        sampled = self.source_labels > 0
        if np.any(sampled & (~self.indexed | (self.source_labels != self.labels))):
            raise ValueError("original grain membership must match indexed geometry")
        if not np.array_equal(np.unique(self.source_labels[sampled]), ids):
            raise ValueError("each grain needs original indexed pixels")

    @property
    def ncells(self):
        return len(self.grain_quats)

    @property
    def extent(self):
        ny, nx = self.labels.shape
        return nx * self.vox[0], ny * self.vox[1]


def write_ebsd(path, data):
    """Write numeric arrays without pickles or external sidecars."""
    data.__post_init__()
    with Path(path).open("wb") as stream:
        np.savez_compressed(
            stream,
            version=np.array(1),
            labels=data.labels,
            grain_quats=data.grain_quats,
            vox=data.vox,
            pixel_quats=data.pixel_quats,
            indexed=data.indexed,
            source_labels=data.source_labels,
            crysym=np.array(data.crysym),
        )
    return Path(path)


def read_ebsd(path):
    """Load and validate a native EBSD archive."""
    if isinstance(path, EbsdData):
        path.__post_init__()
        return path
    with np.load(path, allow_pickle=False) as arrays:
        if int(arrays["version"]) != 1:
            raise ValueError("unsupported EBSD archive version")
        return EbsdData(
            **{
                key: arrays[key]
                for key in (
                    "labels",
                    "grain_quats",
                    "vox",
                    "pixel_quats",
                    "indexed",
                    "source_labels",
                )
            },
            crysym=str(arrays["crysym"]),
        )
