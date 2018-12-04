# -*- coding: utf-8 -*-

"""
Iterate over dcd trajectories.
"""

import os
import glob
from simtk.openmm.app import CharmmPsfFile

import mdtraj as md

from rflow.exceptions import RickFlowException


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
            assert i in sequence_ids, \
                "Error: Could not find trajectory file {}.".format(i)
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


