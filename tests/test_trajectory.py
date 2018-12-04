# -*- coding: utf-8 -*-

from rflow import CharmmTrajectoryIterator, normalize
from rflow.utility import abspath

import pytest


@pytest.fixture(scope="module")
def iterator():
    return CharmmTrajectoryIterator(
        filename_template=abspath("data/whex{}.dcd"),
        first_sequence=1, last_sequence=2,
        topology_file=abspath("data/whex.pdb")
    )


def test_iterator(iterator):
    for seq in iterator:
        assert seq.n_frames == 100


def test_normalize(iterator):
    for seq in iterator:
        normalized = normalize(seq)
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0


def test_normalize_com(iterator):
    for seq in iterator:
        normalized = normalize(seq, [0, 1, 2], "not water", [52])
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0