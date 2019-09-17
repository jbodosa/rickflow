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
from simtk.openmm.app import CharmmCrdFile
from simtk import unit as u
import mdtraj as md

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
                result = force
            else:
                raise RickFlowException("Multiple forces found get_force.")
    return result


def get_platform(gpu_id=None, precision="mixed"):
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
    if gpu_id is None:
        return None, None
    if isinstance(gpu_id, str):
        return Platform.getPlatformByName(gpu_id), {}
    elif isinstance(gpu_id, int):
        try:
            assert "LD_LIBRARY_PATH" in os.environ
            assert 'cuda' in os.environ["LD_LIBRARY_PATH"].lower()
            my_platform = Platform.getPlatformByName('CUDA')
            my_properties = {'DeviceIndex': str(gpu_id), 'Precision': precision}
            return my_platform, my_properties
        except Exception as e:
            raise NoCuda(e)
    else:
        raise RickFlowException(
            "Did not understand gpu_id. It has to be an integer, None, "
            "or a string that contains the platform name."
        )


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


def read_input_coordinates(input, topology=None, frame=-1):
    """
    Read input coordinates from diverse input.

    Args:
        input (str): A trajectory file or a np array.
        topology (str or openmm.Topology): A topology file or topology object
        frame (int): Trajectory frame to read the coordinates from.

    Returns:
        Input coordinates as a list of numpy arrays; in nanometer.

    Notes:
        - All files ending in .crd or .cor are considered CHARMM coordinate files
        - Returning as a list of arrays facilitates appending coordinates if needed.
    """
    if isinstance(input, np.ndarray):
        pos = input
    elif isinstance(input, u.Quantity):
        pos = input.value_in_unit(u.nanometer)
    elif input.endswith(".crd") or input.endswith(".cor"):
        crd = CharmmCrdFile(input)
        pos = crd.positions.value_in_unit(u.nanometer)
    elif input.endswith(".trj"):
        traj = md.load_dcd(input, top=topology)
        pos = traj.xyz[frame, :, :]
    else:
        traj = md.load(input, top=topology)
        pos = traj.xyz[frame, :, :]
    return list(np.array(pos))


def read_box_dimensions(input, topology=None, frame=-1):
    """Same as read_input_coordinates, except for box dimensions."""
    if isinstance(input, np.ndarray):
        return None
    elif isinstance(input, u.Quantity):
        return None
    elif input.endswith(".crd") or input.endswith(".cor"):
        return None
    elif input.endswith(".trj"):
        traj = md.load_dcd(input, top=topology)
        dimensions = traj.unitcell_lengths[frame, :]
    else:
        traj = md.load(input, top=topology)
        dimensions = traj.unitcell_lengths[frame, :]
    return list(dimensions)


def center_of_mass(positions, topology, selection):
    """
    Calculate the center of mass of a subset of atoms.

    Args:
        particle_ids (list of int): The particle ids that define the subset of the system

    Returns:
        float: center of mass in nanometer
    """
    masses = np.array([atom.element.mass.value_in_unit(u.dalton) for atom in topology.atoms()])
    positions = np.array(positions)
    return np.sum(
        positions[selection].transpose()
        * masses[selection],
        axis=1
    ) / np.sum(masses[selection])


def recenter_positions(positions, selection, topology, box_lengths):
    current_com = center_of_mass(positions, topology, selection)
    target_com = 0.5*np.array(box_lengths.value_in_unit(u.nanometer))
    move = target_com - current_com
    return positions + move

