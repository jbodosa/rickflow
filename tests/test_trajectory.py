# -*- coding: utf-8 -*-

from rflow import TrajectoryIterator, normalize, center_of_mass_of_selection
from rflow.utility import abspath
import numpy as np
import pytest
import mdtraj as md

@pytest.fixture(scope="module")
def iterator():
    return TrajectoryIterator(
        filename_template=abspath("data/whex{}.dcd"),
        first_sequence=1, last_sequence=2,
        topology_file=abspath("data/whex.pdb")
    )


def test_iterator(iterator):
    for seq in iterator:
        assert seq.n_frames == 100


def test_indexing(iterator):
    traj2 = iterator[2]
    for t in iterator:
        pass
    assert np.isclose(t.xyz, traj2.xyz).all()


def test_normalize(iterator):
    for seq in iterator:
        normalized = normalize(seq)
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0


def test_normalize_com(iterator):
    for seq in iterator:
        normalized = normalize(seq, [0, 1, 2], "not water", [25])
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0


def test_compute_center_of_mass_selection(iterator):
    selected = iterator.topology.select("not water")
    for seq in iterator:
        for coordinates in [2, [0,1,2], [0,1]]:
            slice = seq.atom_slice(selected)
            reference_com = md.compute_center_of_mass(
                slice
            )[:, coordinates]
            com = center_of_mass_of_selection(seq, selected, coordinates)
            assert com == pytest.approx(reference_com, 1e-4)


# INFERRING TIME FOR THE FRAMES IS TESTED IN test_rickflow.py::test_run_and_restart