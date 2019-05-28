# -*- coding: utf-8 -*-

"""
Iterate over dcd trajectories.
"""

import os
import glob

import numpy as np

from simtk.openmm.app import CharmmPsfFile

import mdtraj as md

from rflow.exceptions import RickFlowException, TrajectoryNotFound
from rflow.utility import selection


class CharmmTrajectoryIterator(object):
    """An iterator that runs over trajectory files.

    For trajectory files that were created using the rickflow workflow, the class
    is used as follows:

    >>> trajectories = CharmmTrajectoryIterator()
    >>> for traj in trajectories:
    >>>     ...

    Analysis classes in the rflow package are written so that they can be iteratively
    called on trajectories. For example, for assembling transition matrices, call:

    >>>
    """
    def __init__(self, first_sequence=None, last_sequence=None,
                 filename_template="trj/dyn{}.dcd", topology_file="system.pdb",
                 selection="all", load_function=md.load_dcd):

        # select sequences
        trajectory_files = glob.glob(filename_template.format("*"))
        lstr, rstr = filename_template.split("{}")
        sequence_ids = [int(trj[len(lstr):len(trj) - len(rstr)])
                        for trj in trajectory_files]
        if first_sequence is None:
            first_sequence = min(sequence_ids)
        if last_sequence is None:
            last_sequence = max(sequence_ids)
        for i in range(first_sequence, last_sequence + 1):
            if i not in sequence_ids:
                raise TrajectoryNotFound(str(format(i)))
        self.first = first_sequence
        self.last = last_sequence

        self.filename_template = filename_template

        # create topology
        top_suffix = os.path.basename(topology_file).split(".")[-1]
        if top_suffix == "pdb":
            self.topology = md.load(topology_file).topology
        elif top_suffix == "psf":
            self.topology = md.Topology.from_openmm(
                CharmmPsfFile(topology_file).topology
            )
        else:
            raise RickFlowException(
                "Error: topology_file has to be a pdb or psf file.")

        # create selection
        if isinstance(selection, str):
            self.selection = self.topology.select(selection)
        else:
            self.selection = selection

        self.load_function = load_function

    def __iter__(self):
        for i in range(self.first, self.last + 1):
            trajectory = self.load_function(self.filename_template.format(i),
                                            top=self.topology,
                                            atom_indices=self.selection)
            trajectory.i = i
            yield trajectory

    def __len__(self):
        return self.last - self.first + 1

    def __getitem__(self, index):
        trajectory = self.load_function(self.filename_template.format(index),
                                        top=self.topology,
                                        atom_indices=self.selection)
        trajectory.i = index
        return trajectory


def normalize(trajectory, coordinates=2, com_selection=None, subselect="all"):
    """
    Normalize the trajectory so that all coordinates are in [0,1] and the center of
    mass of the membrane is at 0.5.

    Args:
        trajectory:     An mdtraj trajectory object.
        coordinates:    0,1, or 2 (for x,y,z); or a list
        com_selection:  Selection of the membrane (to get the center of mass).
                        Can be a list of ints or a selection string.
        subselect:      Atom selection (usually the permeant). Can be a list of ints or a selection string.
                        The normalized array will only contain the atoms in this selection.

    Returns:
        np.array: The normalized coordinates.

    """
    membrane_center = center_of_mass_of_selection(trajectory, com_selection, coordinates)

    selected = selection(trajectory, subselect)
    # normalize z coordinates: scale to [0,1] and shift membrane center to 0.5
    z_normalized = trajectory.xyz[:, selected,
                   coordinates].transpose() - membrane_center.transpose()

    z_normalized /= trajectory.unitcell_lengths[:, coordinates].transpose()

    if com_selection is not None:
        z_normalized += 0.5  # shift so that com is at 0.5

    z_normalized = np.mod(z_normalized, 1.0).transpose()

    return z_normalized


def center_of_mass_of_selection(trajectory, com_selection=None, coordinates=[0,1,2]):
    """
    Compute the center of mass of a selection of atoms.
    Args:
        trajectory:    An mdtraj trajectory.
        com_selection: Either a DSL selection string or a list (np.array) of atom ids.
        coordinates (int in [0,1,2], or sublist of [0,1,2]): The coordinates for which to calculate the center of mass.

    Returns:

    """
    if com_selection is None:
        if isinstance(coordinates, int):
            return np.array(0.0)
        else:
            return np.array([0.0]*len(coordinates))
    selected_atom_ids = selection(trajectory, com_selection)

    for i, a in enumerate(trajectory.topology.atoms):
        assert i == a.index
    masses = np.array([atom.element.mass for atom in trajectory.topology.atoms])
    center_of_mass = np.einsum(
        "i,ni...->n...", masses[selected_atom_ids], trajectory.xyz[:, selected_atom_ids])
    center_of_mass /= np.sum(masses[selected_atom_ids])
    return center_of_mass[:,coordinates]

