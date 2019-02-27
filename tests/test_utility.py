
import numpy as np
from rflow.utility import increment_using_multiindices


def test_increment():
    a = np.zeros((2,2), dtype=int)
    res = increment_using_multiindices(a, np.array([[0,0], [1,1]], dtype=int))
    assert np.array_equal(res, np.array([[1,0], [0,1]], dtype=int))