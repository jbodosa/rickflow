# -*- coding: utf-8 -*-

import os
import inspect
import pkg_resources
import traceback
import numpy as np


def abspath(relative_path): # type (object) -> object
    """Get file from a path that is relative to caller's module.
    Returns:    absolute path as string"""
    caller = inspect.stack()[1]
    mod = inspect.getmodule(caller[0])
    return os.path.normpath(pkg_resources.resource_filename(mod.__name__, relative_path))


class CWD(object):
    """
    change dir in a with block
    """
    def __init__(self, path): self.old_path = os.getcwd(); self.new_path = str(path)

    def __enter__(self): os.chdir(self.new_path); return self

    def __exit__(self, exc_type, exc_value, tb):
        os.chdir(self.old_path)
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False # uncomment to pass exception through

        return True


def select_atoms(topology_from, sel):
    """
    A short helper function to enable selection via atom ids or selection strings.

    Args:
        topology_from (mdtraj.Trajectory or mdtraj.Topology): The object defining the topology.
        sel: Either a selection string or a list of atom ids.

    Returns:
        list of int: Selected atom ids.
    """
    if hasattr(topology_from, "topology"):
        topology = topology_from.topology
    else:
        topology = topology_from
    if sel is None:
        return []
    elif isinstance(sel, list) or isinstance(sel, np.ndarray):
        return sel
    else:
        return topology.select(sel)


def mydigitize(array, nbins, range):
    """
    An attempt to write a faster version of digitize.
    Args:
        array:
        nbins:
        range:

    Returns:

    """
    raise NotImplementedError


def increment_using_multiindices(array, index_array):
    """Increment an array by 1 at all multiindices defined in index_array.

    If a multiindex occurs multiple times in the index array, the element is incremented multiple times.
    
    Args:
        array (numpy.array): A (possibly highdimensional) array
        index_array (numpy.array): A two-dimensional array, whose rows specify multiindices.

    Returns:
        incremented_array (numpy.array): A copy of the input array, where 1 has been added at each index from
            the index_array.
    """
    unfolded_array = np.ravel(array)
    unfolded_indices = np.ravel_multi_index(index_array.T, array.shape)
    np.add.at(unfolded_array, unfolded_indices, 1)
    return np.reshape(unfolded_array, array.shape)
