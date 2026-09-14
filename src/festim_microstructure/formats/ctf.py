"""Parse Channel ``.ctf`` headers and pixel tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["LAUE_TO_CRYSYM", "CtfMap"]


class CtfMap:
    """A parsed Channel Text File: header fields plus the pixel table."""

    def __init__(self, path):
        self.path = Path(path)
        self.header = {}
        self.phases = []
        self._parse()

    def _parse(self):
        with open(self.path, errors="replace") as fh:
            lines = fh.read().splitlines()

        col_row = None
        for i, line in enumerate(lines):
            fields = line.split("\t")
            key = fields[0].strip()
            if key == "Phase" and len(fields) > 5 and "Euler1" in fields:
                col_row = i
                break
            if key in ("XCells", "YCells"):
                self.header[key] = int(float(fields[1]))
            elif key in ("XStep", "YStep"):
                self.header[key] = float(fields[1])
            elif key == "Phases":
                self.header["Phases"] = int(fields[1])
            elif ";" in key and len(fields) >= 5:
                # a phase line: "a;b;c <tab> al;be;ga <tab> name <tab> laue <tab> sg"
                try:
                    self.phases.append(
                        {"name": fields[2].strip(), "laue": int(fields[3])}
                    )
                except (ValueError, IndexError):
                    pass

        if col_row is None:
            raise ValueError(
                f"{self.path}: no column header row found. Expected a line "
                "starting with 'Phase' and containing 'Euler1'."
            )
        for key in ("XCells", "YCells", "XStep", "YStep"):
            if key not in self.header:
                raise ValueError(f"{self.path}: header is missing {key}")

        self.columns = [c.strip() for c in lines[col_row].split("\t") if c.strip()]
        for required in ("Phase", "X", "Y", "Euler1", "Euler2", "Euler3"):
            if required not in self.columns:
                raise ValueError(f"{self.path}: no '{required}' column")

        data = np.genfromtxt(
            self.path, skip_header=col_row + 1, usecols=range(len(self.columns))
        )
        if data.ndim == 1:
            data = data[None, :]
        self.table = {c: data[:, i] for i, c in enumerate(self.columns)}
        self.npoints = data.shape[0]

    def __getitem__(self, key):
        return self.table[key]

    def has(self, key):
        return key in self.table

    @property
    def shape(self):
        return self.header["YCells"], self.header["XCells"]

    def crysym(self, phase_index=1):
        if not self.phases:
            return None, None
        ph = self.phases[phase_index - 1]
        return LAUE_TO_CRYSYM.get(ph["laue"]), ph


LAUE_TO_CRYSYM = {
    1: "-1",
    2: "2/m",
    3: "mmm",
    4: "4/m",
    5: "4/mmm",
    6: "-3",
    7: "-3m",
    8: "6/m",
    9: "6/mmm",
    10: "m-3",
    11: "cubic",  # m-3m; Neper's `cubic` and `m-3m` both carry 24 operators
}
