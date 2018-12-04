# -*- coding: utf-8 -*-

from rflow import CharmmTrajectoryIterator, TransitionCounter, PermeationEventCounter
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


def test_permeation(iterator):
    pcount = PermeationEventCounter([51], 0.25, 0.1, "not water")
    for seq in iterator:
        pcount(seq)
    assert len(pcount.events) == 0