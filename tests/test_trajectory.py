# -*- coding: utf-8 -*-

from rflow import TrajectoryIterator, normalize, center_of_mass_of_selection
from rflow.trajectory import rewrapped_coordinates_around_selection
from rflow.utility import abspath
import numpy as np
import pytest
import mdtraj as md

@pytest.fixture(scope="module")
def iterator():
    return TrajectoryIterator(
        filename_template=abspath("data/whex{:0d}.dcd"),
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
            com = center_of_mass_of_selection(seq, selected, coordinates, allow_rewrapping=False)
            assert com == pytest.approx(reference_com, 1e-4)


def test_file_format_by_name():
    traj = TrajectoryIterator(
        filename_template=abspath("data/whex{}.dcd"),
        first_sequence=1, last_sequence=2,
        topology_file=abspath("data/whex.pdb"),
        load_function="dcd")
    assert traj.first == 1
    assert traj.last == 2


class MockTrajectory:
    def __init__(self, xyz, unitcell_lengths):
        self.xyz = np.array(xyz)
        self.unitcell_lengths = np.array(unitcell_lengths)
        self.n_atoms = self.xyz.shape[1]
        self.n_frames = self.xyz.shape[0]
        assert self.xyz.shape == (self.n_frames, self.n_atoms, 3)
        assert self.unitcell_lengths.shape == (self.n_frames, 3)
        self.time = np.arange(self.n_frames)
        self.topology = md.Topology()
        chain = self.topology.add_chain()
        for i in range(self.n_atoms):
            residue = self.topology.add_residue("H", chain)
            self.topology.add_atom("H", md.element.hydrogen, residue)


@pytest.mark.parametrize("origin", [0, 1])
@pytest.mark.parametrize("wraps", [True, False])
def test_rewrapped_coordinates_around_selection(origin, wraps):
    unitcell_lengths = [[2,2,2]] * 2
    xyz = [[[0.5, 0.5, origin]]*100]*2
    traj = MockTrajectory(xyz, unitcell_lengths)
    if wraps:
        traj.xyz[:,0,2] -= 0.9
        traj.xyz[:,1,2] += 0.8
    rewrapped = rewrapped_coordinates_around_selection(traj, [0,1])
    if wraps and origin == 0:
        assert rewrapped[:,0,2] == pytest.approx(1.1)
        assert rewrapped[:,1,2] == pytest.approx(0.8)
    elif wraps and origin == 1:
        assert rewrapped[:,0,2] == pytest.approx(2.1)
        assert rewrapped[:,1,2] == pytest.approx(1.8)
    else:
        assert (rewrapped == traj.xyz).all()

# INFERRING TIME FOR THE FRAMES IS TESTED IN test_rickflow.py::test_run_and_restart
