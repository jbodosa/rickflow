# -*- coding: utf-8 -*-

import os
import inspect
import pkg_resources
import traceback
import numpy as np

from simtk.openmm import (
    MonteCarloBarostat, MonteCarloAnisotropicBarostat, MonteCarloMembraneBarostat,
    Platform, NonbondedForce, CustomNonbondedForce
)

from rflow.exceptions import RickFlowException, NoCuda


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
    elif isinstance(sel, str):
        return topology.select(sel)
    else:
        return sel


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


def get_barostat(system):
    return get_force(system, [MonteCarloMembraneBarostat, MonteCarloAnisotropicBarostat, MonteCarloBarostat])


def get_force(system, forcetypes):
    result = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if any(isinstance(force, forcetype) for forcetype in forcetypes):
            if result is None:
                result = (i, force)
            else:
                raise RickFlowException("Multiple forces found get_force.")
    return result


def require_cuda(gpu_id=None, precision="mixed"):
    """
    Require CUDA to be used for the simulation.

    Args:
        gpu_id (int): The id of the GPU to be used.
        precision (str): 'mixed', 'double', or 'single'

    Returns:
        A pair (platform, properties):

         - OpenMM Platform object: The platform to be passed to the simulation.
         - dict: A dictionary to be passed to the simulation.

    Raises:
        NoCuda: If CUDA is not present.
    """
    try:
        assert "LD_LIBRARY_PATH" in os.environ
        assert 'cuda' in os.environ["LD_LIBRARY_PATH"].lower()
        my_platform = Platform.getPlatformByName('CUDA')
    except Exception as e:
        raise NoCuda(e)
    if gpu_id is not None:
        my_properties = {'DeviceIndex': str(gpu_id),
                         'Precision': precision
                         }
    return my_platform, my_properties


def disable_long_range_correction(system):
    """
    Disable analytic long range correction.
    This is fixed in openmm > 7.3.1.
    """
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            force.setUseDispersionCorrection(False)
        if isinstance(force, CustomNonbondedForce):
            force.setUseLongRangeCorrection(False)


def read_input_coordinates(coordinate_specification):
    """
    Read input coordinates from diverse input.

    Args:
        coordinate_specification: Can be one of the following
            - a np.array - do nothing
            - a filename for a CHARMM crd file
            - a filename for a CHARMM pdb file
            - a 3-tuple (trajectory_filename, topology, frame_id)
            - a pair (trajectory_filename, frame_id)

    Returns: Input coordinates, box_size_guess
    """
    pass

