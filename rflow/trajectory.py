# -*- coding: utf-8 -*-

"""
Iterate over dcd trajectories.
"""

import os
import glob
import warnings

import numpy as np

from simtk.openmm.app import CharmmPsfFile

import mdtraj as md

from rflow.exceptions import RickFlowException, TrajectoryNotFound
from rflow.utility import select_atoms


def make_topology(topology_file):
    top_suffix = os.path.basename(topology_file).split(".")[-1]
    if top_suffix == "pdb":
        return md.load(topology_file).topology
    elif top_suffix == "psf":
        return md.Topology.from_openmm(
            CharmmPsfFile(topology_file).topology
        )
    else:
        raise RickFlowException(
            "Error: topology_file has to be a pdb or psf file.")


class TrajectoryIterator(object):
    """An iterator that runs over trajectory files.

    Args:
        first_sequence (int or None): ID of the first sequence; default=None.
                                      If None, take the first existing sequence ID matching the filename_template.
        last_sequence (int or None):  ID of the last sequence; default=None.
                                      If None, take the last existing sequence ID matching the filename_template.
        filename_template (str):      A template string, where {} serves as a placeholder for the filename;
                                      default="trj/dyn{}.dcd".
        topology_file (str):          Filename of the topology file (pdb or psf); default="system.pdb".
        select_atoms (str or list of int): A selection of atoms; default="all".
        time_between_frames (float): The time between trajectory frames in picoseconds; default=1.
        num_frames_per_trajectory (int or None): The number of frames in each sequence; default=None.
                                      If None, infer the the number of frames from the first sequence that is read in.
        infer_time (bool):            Whether the time field in the trajectory should be inferred heuristically from
                                      the time between frames and number of frames per trajectory file.
        load_function (callable or str): The function used to load the file (e.g. md.load),
                                      or the trajectory format as a string.
                                      By default the format is inferred from the file suffix.


    For trajectory files that were created using the rickflow workflow, the class
    is used as follows:

    >>> trajectories = TrajectoryIterator()
    >>> for traj in trajectories:
    >>>     pass  # do stuff here

    It also supports querying the length and getting individual sequences via

    >>> len(trajectories)
    and
    >>> trajectories[i]

    Analysis classes in the rflow package are written so that they can be iteratively
    called on trajectories. For example, for tabulating the box dimensions, call:

    >>> from rflow import TimeSeries, BoxSize
    >>> boxsize = TimeSeries(BoxSize(), filename="boxsize.txt")
    >>> for traj in trajectories:
    >>>     boxsize(traj)

    """
    def __init__(self, first_sequence=None, last_sequence=None,
                 filename_template="trj/dyn{}.dcd", topology_file="system.pdb",
                 atom_selection="all", time_between_frames=1.0, num_frames_per_trajectory=None,
                 infer_time=True, load_function=md.load):

        # select sequences
        if filename_template.count('{') == 1 and filename_template.count('}') == 1:
            lstr = filename_template.split("{")[0]
            rstr = filename_template.split("}")[1]
            trajectory_files = glob.glob(lstr+"*"+rstr)
            sequence_ids = [int(trj[len(lstr):len(trj) - len(rstr)])
                            for trj in trajectory_files]
            # check if the files really exist (formatting can corrupt this)
            sequence_ids = [i for i in sequence_ids if os.path.exists(filename_template.format(i))]
            if len(sequence_ids) == 0:
                raise RickFlowException("No trajectory files matching your filename template.")
            if first_sequence is None:
                first_sequence = min(sequence_ids)
            if last_sequence is None:
                last_sequence = max(sequence_ids)
            for i in range(first_sequence, last_sequence + 1):
                if i not in sequence_ids:
                    raise TrajectoryNotFound(str(format(i)))
            self.first = first_sequence
            self.last = last_sequence
        else:  # filename_template is one file, not a template
            self.first = 1
            self.last = 1

        self.filename_template = filename_template

        # create topology
        self.topology = make_topology(topology_file)

        # create selection
        self.selection = select_atoms(self.topology, atom_selection)

        # initialize time
        self.infer_time = infer_time
        self.time_between_frames = float(time_between_frames)
        self.num_frames_per_trajectory = num_frames_per_trajectory
        self.load_function = self.interpret_load_function(filename_template, load_function)

    def __iter__(self):
        for i in range(self.first, self.last + 1):
            trajectory = self.load_function(self.filename_template.format(i),
                                            top=self.topology,
                                            atom_indices=self.selection)
            trajectory.i = i
            self._infer_time(trajectory, i)
            yield trajectory

    def __len__(self):
        return self.last - self.first + 1

    def __getitem__(self, index):
        trajectory = self.load_function(self.filename_template.format(index),
                                        top=self.topology,
                                        atom_indices=self.selection)
        trajectory.i = index
        self._infer_time(trajectory, index)
        return trajectory

    def _infer_time(self, trajectory, i):
        """
        Initialize the time field in the trajectory.
        As of now, mdtraj does not read and write time information to and from dcd files correctly.
        Therefore, time is manually assigned here assuming that each trajectory sequence has the same
        number of frames as the one that is read first.
        """
        if self.infer_time:
            # initialize time information
            if self.num_frames_per_trajectory is None:
                self.num_frames_per_trajectory = trajectory.n_frames
            trajectory.time = np.arange(
                (i-1) * self.num_frames_per_trajectory * self.time_between_frames,
                ((i-1) * self.num_frames_per_trajectory + trajectory.n_frames) * self.time_between_frames,
                step=self.time_between_frames
            )

    def select(self, selection_string):
        try:
            return self.topology.select(selection_string)
        except ValueError:
            raise RickFlowException(f"Selection {selection_string} invalid.")

    @staticmethod
    def interpret_load_function(filename, load_function=md.load):
        # Interpret load_function argument
        suffix = filename.split(".")[-1]
        if load_function is None:
            load_function = md.load
        # Interpret .trj files as .dcd
        if load_function == md.load and suffix == "trj":
            return md.load_dcd
        elif isinstance(load_function, str):
            return getattr(md, "load_{}".format(load_function))
        else:
            return load_function



class CharmmTrajectoryIterator(TrajectoryIterator):
    """Old name for the trajectory iterator."""
    def __init__(self, *args, **kwargs):
        warnings.warn("The CharmmTrajectoryIterator has been renamed into TrajectoryIterator.", DeprecationWarning)
        super(CharmmTrajectoryIterator, self).__init__(*args, **kwargs)


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

    selected = select_atoms(trajectory, subselect)
    # normalize z coordinates: scale to [0,1] and shift membrane center to 0.5
    z_normalized = trajectory.xyz[:, selected,
                   coordinates].transpose() - membrane_center.transpose()

    z_normalized /= trajectory.unitcell_lengths[:, coordinates].transpose()

    if com_selection is not None and len(com_selection) > 0:
        z_normalized += 0.5  # shift so that com is at 0.5

    z_normalized = np.mod(z_normalized, 1.0).transpose()

    # calling mod again to avoid finite precision issues (e.g., np.mod(-1e-50,1.0) evaluates to 1.0 but we need to
    # make sure that normalized values are in [0,1) )
    return np.mod(z_normalized, 1.0)


def center_of_mass_of_selection(trajectory, com_selection=None, coordinates=[0,1,2], allow_rewrapping=False):
    """
    Compute the center of mass of a selection of atoms.
    Args:
        trajectory:    An mdtraj trajectory.
        com_selection: Either a DSL selection string or a list (np.array) of atom ids.
        coordinates (int in [0,1,2], or sublist of [0,1,2]): The coordinates for which to calculate the center of mass.

    Returns:

    """
    if com_selection is None or len(com_selection) == 0:
        if isinstance(coordinates, int):
            return np.array(0.0)
        else:
            return np.array([0.0]*len(coordinates))
    selected_atom_ids = select_atoms(trajectory, com_selection)

    if allow_rewrapping and com_selection is not None:
        xyz = rewrapped_coordinates_around_selection(trajectory, selected_atom_ids)
    else:
        xyz = trajectory.xyz
    for i, a in enumerate(trajectory.topology.atoms):
        assert i == a.index
    masses = np.array([atom.element.mass for atom in trajectory.topology.atoms])
    center_of_mass = np.einsum(
        "i,ni...->n...", masses[selected_atom_ids], xyz[:, selected_atom_ids])
    center_of_mass /= np.sum(masses[selected_atom_ids])
    return center_of_mass[:, coordinates]


def rewrapped_coordinates_around_selection(trajectory, selection=None):
    """
    Rewrap so that the selection does not penetrate a boundary.
    (Breaks molecules).

    Args:
        trajectory: mdtraj trajectory
        selection: a DSL selection string or a list of atom ids

    Returns:
        A numpy array with shape (trajectory.n_frames, 3). Last axis is for x, y, z.
        Elements are 0, if it does not wrap around a boundary, 1 for positive, -1 for negative
    """
    selected_atom_ids = select_atoms(trajectory, selection)
    box_center = np.mean(trajectory.xyz, axis=1)
    original = trajectory.xyz
    rewrapped = np.choose(
        original < box_center[:, None, :],
        [original, original + trajectory.unitcell_lengths[:,None,:]]
    )
    needs_wrapping = (
            np.std(original[:, selected_atom_ids, :], axis=1) > np.std(rewrapped[:, selected_atom_ids, :], axis=1)
    )
    wrapped_around_selection = np.choose(
        needs_wrapping[:, None, :],
        [original, rewrapped]
    )
    needs_shift_by_boxlength = np.mean(wrapped_around_selection[:, selected_atom_ids, :], axis=1) > trajectory.unitcell_lengths
    return wrapped_around_selection - (needs_shift_by_boxlength * trajectory.unitcell_lengths)[:, None, :]


