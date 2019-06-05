"""

"""

import os
import warnings
import numpy as np
from pytest import approx
from rflow.observables import *


def test_statistical_quantity():
    q = TimeSeries()
    q.append(1.0)
    q.append(2.0)
    assert q.data[0] == 1.0
    assert q.data[1] == 2.0
    assert q.mean == approx(1.5)


def test_statistical_quantity_save_to_file(tmpdir):
    datafile = os.path.join(str(tmpdir), "data.txt")
    q = TimeSeries(filename=datafile)
    q.append(1.0)
    q.append(2.0)
    assert q.mean == approx(1.5)
    assert os.path.isfile(datafile)
    retrieved = np.loadtxt(datafile)
    assert retrieved[0] == approx(1.0)
    assert retrieved[1] == approx(2.0)


def test_statistical_quantity_vector(tmpdir):
    datafile = os.path.join(str(tmpdir), "data.txt")
    q = TimeSeries(filename=datafile)
    q.append([1.0, 2.0])
    q.append([2.0, 3.0])
    assert np.isclose(q.data[0], [1.0, 2.0]).all()
    assert np.isclose(q.data[1], [2.0, 3.0]).all()
    assert np.isclose(q.mean, [1.5, 2.5]).all()
    assert os.path.isfile(datafile)
    retrieved = np.loadtxt(datafile)
    assert np.isclose(retrieved[0], [1.0, 2.0]).all()
    assert np.isclose(retrieved[1], [2.0, 3.0]).all()


def test_time_series_radd():
    q = TimeSeries()
    q += [1.0, 2.0]
    assert q.data[0] == 1.0
    assert q.data[1] == 2.0
    assert q.mean == approx(1.5)


def double(a):
    return np.array([2*a])

def something(a,b,c=0):
    return np.array([a*b+c])

def test_evaluator():
    q = TimeSeries(evaluator=double)
    q(1)
    q(2)
    q(3)
    assert q.data[0] == 2
    assert q.data[1] == 4
    assert q.data[2] == 6

def test_evaluator_with_args():
    q = TimeSeries(evaluator=something)
    q(1,3)
    q(2,3)
    q(3,3)
    assert q.data[0] == 3
    assert q.data[1] == 6
    assert q.data[2] == 9
    q(1,3,c=1)
    q(2,3,c=1)
    q(3,3,c=1)
    assert q.data[3] == 4
    assert q.data[4] == 7
    assert q.data[5] == 10


def test_evaluator_with_file(tmpdir):
    datafile = os.path.join(str(tmpdir), "data.txt")
    q = TimeSeries(evaluator=double, filename=datafile)
    q(1)
    q(2)
    q(3)
    assert os.path.isfile(datafile)
    retrieved = np.loadtxt(datafile)
    assert retrieved[0] == 2
    assert retrieved[1] == 4
    assert retrieved[2] == 6


def test_distribution(whex_iterator):
    dist = Distribution(atom_selection=[51], coordinate=2, nbins=10)
    for seq in whex_iterator:
        dist(seq)
    assert dist.counts.shape == (10,)
    assert dist.counts.sum() == 200*1


def test_distribution_save(whex_iterator, tmpdir):
    datafile = os.path.join(str(tmpdir), "distribution.txt")
    dist = Distribution(atom_selection=[51], coordinate=2, nbins=10)
    for seq in whex_iterator:
        dist(seq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist.save(datafile)
    loaded = np.loadtxt(datafile)
    assert loaded.shape == (10, 5)
    Distribution.load_from_pic(datafile + ".pic")


def test_distribution_add(whex_iterator):
    dist = Distribution(atom_selection=[51], coordinate=2, nbins=10)
    dist2 = Distribution(atom_selection=[51], coordinate=2, nbins=10)
    dist3 = Distribution(atom_selection=[51], coordinate=2, nbins=10)
    for seq in whex_iterator:
        dist(seq)
        dist2(seq)
        dist3(seq)
    sum_dist = sum([dist, dist2, dist3])
    assert sum_dist.average_box_size == (dist.average_box_size + dist2.average_box_size + dist3.average_box_size)/3
    assert (sum_dist.counts == dist.counts + dist2.counts + dist3.counts).all()
