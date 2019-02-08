"""

"""

from rflow.observables import *
import os
import numpy as np
from pytest import approx, fixture


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