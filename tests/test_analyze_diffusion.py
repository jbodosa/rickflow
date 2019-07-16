# -*- coding: utf-8 -*-

from rflow import TransitionCounter, PermeationEventCounter, Distribution, RickFlowException

import os
import pytest
import warnings
import numpy as np


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

