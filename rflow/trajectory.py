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
    def __init__(self, first_sequence=None, last_sequence=None,
                 filename_template="trj/dyn{}.dcd", topology_file="system.pdb",
                 selection="all"):

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

    def __iter__(self):
        for i in range(self.first, self.last + 1):
            trajectory = md.load_dcd(self.filename_template.format(i),
                                     top=self.topology,
                                     atom_indices=self.selection)
            trajectory.i = i
            yield trajectory


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
    if com_selection is None:
        membrane_center = np.array([0.0])
    else:
        membrane = selection(trajectory, com_selection)
        membrane_trajectory = trajectory.atom_slice(membrane)
        membrane_center = md.compute_center_of_mass(
            membrane_trajectory
        )[:, coordinates]

    selected = selection(trajectory, subselect)
    # normalize z coordinates: scale to [0,1] and shift membrane center to 0.5
    z_normalized = trajectory.xyz[:, selected,
                   coordinates].transpose() - membrane_center.transpose()

    z_normalized /= trajectory.unitcell_lengths[:, coordinates].transpose()

    if com_selection is not None:
        z_normalized += 0.5  # shift so that com is at 0.5

    z_normalized = np.mod(z_normalized, 1.0).transpose()

    return z_normalized
