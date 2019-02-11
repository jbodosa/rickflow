# -*- coding: utf-8 -*-

from rflow import CharmmTrajectoryIterator, TransitionCounter, PermeationEventCounter, Distribution
from rflow.utility import abspath

import os
import pytest
import numpy as np


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
    print(tcount.matrices[1])
    assert (tcount.matrices[1] == np.array(
        [[61, 4, 0, 0, 0, 0, 0, 0, 0, 6],
         [4, 24, 0, 0, 0, 0, 0, 0, 0, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 0, 25, 5],
         [6, 0, 0, 0, 0, 0, 0, 0, 5, 59]]
       )).all()


def test_permeation(iterator):
    pcount = PermeationEventCounter([51], 0.1, 0.05, "not water")
    for seq in iterator:
        pcount(seq)
    assert len(pcount.events) == 0


def test_distribution(iterator):
    dist = Distribution(atom_selection=[51], coordinate=2, nbins=10)
    for seq in iterator:
        dist(seq)
    assert dist.counts.shape == (10,)
    assert dist.counts.sum() == 200*1


def test_distribution_save(iterator, tmpdir):
    datafile = os.path.join(str(tmpdir), "distribution.txt")
    dist = Distribution(atom_selection=[51], coordinate=2, nbins=10)
    for seq in iterator:
        dist(seq)
    dist.save(datafile)
    loaded = np.loadtxt(datafile)
    assert loaded.shape == (10, 5)
    Distribution.load_from_pic(datafile + ".pic")