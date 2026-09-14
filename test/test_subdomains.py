"""How a grain-boundary network says where its facets are.

One class covers both routes, so what used to be the choice between two types
is now a choice of arguments, and the constructor is where a wrong combination
has to be caught. No mesh is needed for that: nothing here locates anything.
"""

import numpy as np
import pytest


@pytest.fixture
def network(stubbed_heavy_deps):
    from festim_microstructure.fem.subdomains import GrainBoundaryNetwork

    return GrainBoundaryNetwork


def tags(values):
    """A stand-in for dolfinx MeshTags: values, and the facets carrying them."""
    values = np.asarray(values, dtype=np.int32)
    indices = np.arange(values.size, dtype=np.int32)
    return type("Tags", (), {"values": values, "indices": indices})()


def test_a_locator_and_facet_tags_are_alternatives_not_a_pair(network):
    with pytest.raises(TypeError, match="both"):
        network(
            1,
            object(),
            1,
            facet_tags=tags([1]),
            entity_ids=[1],
            locator=lambda x: np.ones(x.shape[1], dtype=bool),
        )


def test_a_network_has_to_be_located_somehow(network):
    with pytest.raises(TypeError, match="neither"):
        network(1, object(), 1)


def test_facet_tags_without_ids_name_no_boundaries(network):
    with pytest.raises(TypeError, match="entity_ids"):
        network(1, object(), 1, facet_tags=tags([1, 2]))


def test_one_tag_and_a_sequence_of_them_are_both_accepted(network):
    """Neper and EBSD tag every face separately; a Voronoi mesh uses one marker."""
    assert network(1, object(), 2, facet_tags=tags([7]), entity_ids=7).entity_ids == [7]
    many = network(1, object(), 2, facet_tags=tags([7]), entity_ids=[3, 7])
    assert list(many.entity_ids) == [3, 7]


def test_the_exterior_filter_follows_the_route_unless_it_is_asked_not_to(network):
    """On for a predicate, which cannot tell the network from its trace on the
    wall of the box; off for tags, which say what a facet is."""
    located = network(1, object(), 1, locator=lambda x: x)
    tagged = network(1, object(), 1, facet_tags=tags([1]), entity_ids=[1])
    assert located.drop_exterior
    assert not tagged.drop_exterior
    loose = network(1, object(), 1, locator=lambda x: x, drop_exterior=False)
    assert not loose.drop_exterior
    assert network(
        1, object(), 1, facet_tags=tags([1]), entity_ids=[1], drop_exterior=True
    ).drop_exterior


def test_a_tagged_network_starts_with_nothing_mapped_back(network):
    """``entity_ids_of_facets`` is filled by the lookup, not by the constructor."""
    net = network(1, object(), 1, facet_tags=tags([1]), entity_ids=[1])
    assert net.entity_ids_of_facets is None
    assert net.n_dropped == 0
