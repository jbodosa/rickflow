# -*- coding: utf-8 -*-

from rflow import CharmmTrajectoryIterator, TransitionCounter, PermeationEventCounter, Distribution
from rflow.utility import abspath

import pytest


@pytest.fixture(scope="module")
def iterator():
    return CharmmTrajectoryIterator(
        filename_template=abspath("data/whex{}.dcd"),
        first_sequence=1, last_sequence=2,
        topology_file=abspath("data/whex.pdb")
    )


def test_transitions(iterator):
    tcount = TransitionCounter([1], 10, [51])
    for seq in iterator:
        tcount(seq)
    assert tcount.matrices[1].sum() == 199
    assert tcount.edges.shape == (11,)
    assert tcount.edges_around_zero.shape == (11,)
    assert tcount.edges_around_zero.mean() == pytest.approx(0.0)
    assert tcount.edges.mean() == pytest.approx(0.5*tcount.average_box_height)


def test_permeation(iterator):
    pcount = PermeationEventCounter([51], 0.25, 0.1, "not water")
    for seq in iterator:
        pcount(seq)
    assert len(pcount.events) == 0


def test_distribution(iterator):
    dist = Distribution(atom_selection=[51], coordinate=2, nbins=10)
    for seq in iterator:
        dist(seq)
    assert dist.counts.shape == (10,)
    assert dist.counts.sum() == 200*1
