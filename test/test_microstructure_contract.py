"""The microstructure contract: who satisfies it, and what the error says.

Needs neither dolfinx nor Neper: conformance is decided on attribute names, so
a stand-in object is enough, and the real classes are checked against the
protocol without being built.
"""

import numpy as np
import pytest

from festim_microstructure.microstructure import (
    BoundaryNetwork,
    MeshedMicrostructure,
    Microstructure,
    TaggedPolycrystal,
    missing_members,
    require,
)
from festim_microstructure.voronoi import VoronoiMicrostructure


def _meshed(**overrides):
    """A minimal object satisfying MeshedMicrostructure."""
    attrs = dict(
        mesh=object(),
        cell_tags=object(),
        facet_tags=None,
        gb_tag=None,
        orientations=np.zeros(3),
        grain_ids=np.arange(1, 4),
        n_grains=3,
        tolerance=1e-9,
        locator=lambda points: np.zeros(points.shape[1], dtype=bool),
        report=lambda: "",
    )
    attrs.update(overrides)
    return type("Stub", (), attrs)()


def test_a_complete_stub_satisfies_the_meshed_contract():
    assert missing_members(_meshed(), MeshedMicrostructure) == []
    assert require(_meshed(), MeshedMicrostructure) is not None


def test_missing_members_are_named_rather_than_just_rejected():
    stub = _meshed()
    del type(stub).facet_tags
    del type(stub).locator
    assert missing_members(stub, MeshedMicrostructure) == ["facet_tags", "locator"]


def test_require_raises_with_the_missing_names_and_the_call_site():
    stub = _meshed()
    del type(stub).orientations
    with pytest.raises(TypeError) as excinfo:
        require(stub, MeshedMicrostructure, context="in build()")
    message = str(excinfo.value)
    assert "orientations" in message
    assert "in build()" in message
    assert "MeshedMicrostructure" in message


def test_voronoi_class_declares_the_whole_meshed_contract():
    """Every protocol member is reachable before a dimension is selected."""
    assert missing_members(VoronoiMicrostructure, MeshedMicrostructure) == []


def test_dimension_agnostic_voronoi_fields():
    fields = {f.name for f in VoronoiMicrostructure.__dataclass_fields__.values()}
    assert "grain_ids" in fields
    assert "n_grains" not in fields  # derived from grain_ids
    assert {"facet_tags", "gb_tag"} <= fields


def test_boundary_network_implementations_are_found_dimension_agnostically(
    stubbed_heavy_deps,
):
    """Neper (faces) and EBSD (edges) answer to the same names."""
    from festim_microstructure.meshing.ebsd import EbsdMicrostructure
    from festim_microstructure.meshing.neper import NeperMesh

    for cls in (NeperMesh, EbsdMicrostructure):
        assert missing_members(cls, BoundaryNetwork) == []
        # the dimension-specific spellings stay available
        assert hasattr(cls, "network_entity_ids")

    assert NeperMesh.network_area is NeperMesh.network_measure
    assert EbsdMicrostructure.network_length is EbsdMicrostructure.network_measure


def test_tagged_polycrystal_satisfies_the_meshed_contract():
    assert missing_members(TaggedPolycrystal, MeshedMicrostructure) == []


def test_tagged_polycrystal_fills_in_the_ids_and_the_orientations():
    """Given ids, it needs no mesh to answer: the defaults are derived, not read."""
    poly = TaggedPolycrystal(
        mesh=object(), cell_tags=object(), grain_ids=np.array([1, 2, 5])
    )
    assert poly.n_grains == 3
    # indexed by id - 1, so the largest id sets the length, not the count
    assert poly.orientations.shape == (5,)
    assert not poly.orientations.any()
    assert missing_members(poly, MeshedMicrostructure) == []


def test_tagged_polycrystal_says_so_when_it_has_no_geometric_locator():
    poly = TaggedPolycrystal(
        mesh=object(), cell_tags=object(), grain_ids=np.array([1]), gb_tag=[3, 4]
    )
    with pytest.raises(TypeError, match="network_locator"):
        poly.locator(np.zeros((3, 2)))


def test_tagged_polycrystal_uses_the_locator_it_was_given():
    poly = TaggedPolycrystal(
        mesh=object(),
        cell_tags=object(),
        grain_ids=np.array([1]),
        network_locator=lambda points: np.ones(points.shape[1], dtype=bool),
    )
    assert poly.locator(np.zeros((3, 4))).all()


def test_protocols_are_not_a_class_hierarchy():
    """Conformance is structural, so nothing has to inherit from them.

    ``issubclass`` against a runtime_checkable protocol answers yes here, which
    is the point: it is a structural check, not a statement about the MRO.
    """
    assert Microstructure not in VoronoiMicrostructure.__mro__
    assert issubclass(VoronoiMicrostructure, Microstructure)
    assert missing_members(VoronoiMicrostructure, Microstructure) == []
