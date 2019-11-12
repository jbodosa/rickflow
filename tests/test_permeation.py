# -*- coding: utf-8 -*-

from rflow import (
    TransitionCounter, PermeationEventCounter, Distribution, RickFlowException,
    PermeationEventCounterWithoutBuffer
)

import pytest
import warnings
import numpy as np

import mdtraj as md


def test_transitions(whex_iterator):
    iterator = whex_iterator
    #water = iterator.topology.select("water")
    tcount = TransitionCounter([1], 10, [51]) # water)
    for seq in iterator:
        tcount(seq)
    assert tcount.matrices[1].sum() == 199
    assert tcount.edges.shape == (11,)
    assert tcount.edges_around_zero.shape == (11,)
    assert tcount.edges_around_zero.mean() == pytest.approx(0.0)
    assert tcount.edges.mean() == pytest.approx(0.5*tcount.average_box_size)
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


def test_permeation(whex_iterator):
    iterator = whex_iterator
    water = iterator.topology.select("water")
    pcount = PermeationEventCounter(water, 0.1, 0.05, membrane=None, initialize_all_permeants=True)
    pcount2 = PermeationEventCounter(water, 0.4, 0.2, membrane=None, initialize_all_permeants=True)
    pcount2_noinit = PermeationEventCounter(water, 0.4, 0.2, membrane=None, initialize_all_permeants=True)
    pcount3 = PermeationEventCounter(water, 0.11, 0.1, membrane="water", initialize_all_permeants=True)
    for seq in iterator:
        pcount(seq)
        pcount2(seq)
        pcount2_noinit(seq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcount3(seq)
    assert len(pcount.events) == 0
    assert len(pcount2.events) == 33
    assert len(pcount2.events) >= len(pcount2_noinit.events)
    assert len(pcount3.events) > 100


def test_critical_transitions():
    print(PermeationEventCounter.make_critical_transitions())


def test_empty_list_error():
    with pytest.raises(RickFlowException):
        PermeationEventCounter([], 0.1, 0.05, membrane=None, initialize_all_permeants=True)


def test_invalid_ids_error():
    with pytest.raises(RickFlowException):
        PermeationEventCounter(None, 0.1, 0.05, membrane=None, initialize_all_permeants=True)


def test_permeability(whex_iterator):
    """Just testing the API at this point."""
    iterator = whex_iterator
    water_o = iterator.topology.select("water and mass > 15")
    dist = Distribution(atom_selection=water_o, coordinate=2, nbins=4)
    pcount = PermeationEventCounter(water_o, 0.11, 0.1, membrane="water", initialize_all_permeants=True)
    for seq in iterator:
        dist(seq)
        pcount(seq)
    assert pcount.permeability(dist, mode="semi-permeation") > 0.0


def test_permeability(whex_iterator):
    """Just testing the API at this point."""
    iterator = whex_iterator
    water_o = iterator.topology.select("water and mass > 15")
    dist = Distribution(atom_selection=water_o, coordinate=2, nbins=4)
    pcount = PermeationEventCounter(water_o, 0.11, 0.1, membrane="water", initialize_all_permeants=True)
    for seq in iterator:
        dist(seq)
        pcount(seq)
    min_p, max_p = pcount.permeability_error(dist, mode="semi-permeation", alpha=0.9)
    assert min_p > 0.0
    assert max_p > 0.0
    assert max_p > min_p

def test_permeation_warnings(whex_iterator):
    iterator = whex_iterator
    water = iterator.topology.select("water")
    pcount = PermeationEventCounter(water, 0.1, 0.05, membrane="water", initialize_all_permeants=True)
    for traj in iterator:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcount(traj)
    assert pcount.num_severe_warnings > 0
    #print(os.linesep, pcount.num_transitions_between_bins)
    assert pcount.num_severe_warnings == 10
    print(pcount.severe_warnings)


def test_permeability2():
    """Only test API"""
    p,mi,ma = PermeationEventCounter.calculate_permeability_and_errors(num_events=20, cw=0.1, area=30, tsim=50)
    assert mi > 0
    assert p > mi
    assert ma > p


class MockTrajectory:
    def __init__(self, z):
        # allow 1d and 2d input
        if len(z.shape) == 1: z = z[None,:] #.reshape((1, z.shape[0]))
        assert len(z.shape) == 2
        self.n_atoms = z.shape[0]
        self.n_frames = z.shape[1]
        z = np.transpose(z)[:,:,None]
        x = np.random.random((self.n_frames,self.n_atoms,1))
        y = np.random.random((self.n_frames,self.n_atoms,1))
        xyz = np.reshape(np.stack([x, y, z], axis=-1), (self.n_frames, self.n_atoms, 3))
        assert len(xyz.shape) == 3
        self.xyz = xyz
        assert xyz.shape == (self.n_frames, self.n_atoms, 3)
        self.unitcell_lengths = np.ones((self.n_frames,3))
        self.time = np.arange(self.n_frames)
        self.topology = md.Topology()
        chain = self.topology.add_chain()
        for i in range(self.n_atoms):
            residue = self.topology.add_residue("H", chain)
            self.topology.add_atom("H", md.element.hydrogen, residue)


def test_counter2_noevent():
    # bouncing back and forth, barely not crossing the center
    z = np.array([0.1, 0.49, 0.1, 0.8, 0.5001, 0.9, 0.2, 0.499])
    counter = PermeationEventCounterWithoutBuffer([0], 0.25)
    counter(MockTrajectory(z))
    assert not counter.events


def test_counter2_entry():
    z = np.array([
        np.interp(np.arange(21), [0, 20], [0.24999, 0.4999]), # particle 0 not entering
        np.interp(np.arange(21), [0, 20], [0.24999, 0.5001]), # particle 1 entering from one side
        np.interp(np.arange(21), [0, 20], [0.75001, 0.4999])  # particle 2 entering from other side
    ])
    counter = PermeationEventCounterWithoutBuffer([0, 1, 2], 0.25)
    counter(MockTrajectory(z))
    assert counter.events == [
        {'event_id': 0, 'atom': 1, 'type': 'entry', 'frame': 20,
            'from_water': 0, 'entry_time_nframes': 19},
        {'event_id': 1, 'atom': 2, 'type': 'entry', 'frame': 20,
            'from_water': 3, 'entry_time_nframes': 19},
    ]


def test_counter2_crossing():
    z = np.array([
        [0.22, 0.33, 0.44, 0.55, 0.44, 0.33, 0.44, 0.55, 0.66, 0.77],  # crossing from one side
        [0.77, 0.66, 0.55, 0.44, 0.55, 0.66, 0.55, 0.44, 0.33, 0.22],  # crossing from other side
    ])
    counter = PermeationEventCounterWithoutBuffer([0, 1], 0.25)
    counter(MockTrajectory(z))
    print(counter.events)
    assert counter.events == [
        {'event_id': 0, 'atom': 0, 'type': 'entry', 'frame': 3,
            'from_water': 0, 'entry_time_nframes': 2},
        {'event_id': 1, 'atom': 1, 'type': 'entry', 'frame': 3,
            'from_water': 3, 'entry_time_nframes': 2},
        {'event_id': 2, 'atom': 0, 'type': 'crossing', 'frame': 9,
            'from_water': 0, 'crossing_time_nframes': 8,
            'corresponding_entry_id': 0, 'exit_time_nframes': 2},
        {'event_id': 3, 'atom': 1, 'type': 'crossing', 'frame': 9,
            'from_water': 3, 'crossing_time_nframes': 8,
            'corresponding_entry_id': 1, 'exit_time_nframes': 2}
    ]


def test_counter2_rebound():
    z = np.array([
        [0.22,0.33,0.44,0.55,0.44,0.33,0.22],
        [0.77,0.66,0.55,0.44,0.55,0.66,0.77],
    ])
    counter = PermeationEventCounterWithoutBuffer([0, 1], 0.25)
    counter(MockTrajectory(z))
    assert counter.events == [
        {'event_id': 0, 'atom': 0, 'type': 'entry', 'frame': 3,
            'from_water': 0, 'entry_time_nframes': 2},
        {'event_id': 1, 'atom': 1, 'type': 'entry', 'frame': 3,
            'from_water': 3, 'entry_time_nframes': 2},
        {'event_id': 2, 'atom': 0, 'type': 'rebound', 'frame': 6,
            'from_water': 0, 'rebound_time_nframes': 5,
            'corresponding_entry_id': 0, 'exit_time_nframes': 2},
        {'event_id': 3, 'atom': 1, 'type': 'rebound', 'frame': 6,
            'from_water': 3, 'rebound_time_nframes': 5,
            'corresponding_entry_id': 1, 'exit_time_nframes': 2}
   ]
