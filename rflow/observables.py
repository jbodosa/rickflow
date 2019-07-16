"""
Tools for analyzing MD observables
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd

from simtk.openmm import Context, LangevinIntegrator, NonbondedForce
from mdtraj.utils import lengths_and_angles_to_box_vectors
import simtk.unit as u

from rflow.trajectory import normalize
from rflow.utility import select_atoms
from rflow.exceptions import RickFlowException


class TimeSeries(object):
    """A time series."""
    def __init__(self, evaluator=None, name="", filename=None, append=False):
        """
        Args:
            evaluator (callable): The callable returns a numpy array.
            filename:
            append:
        """
        self.evaluator = evaluator
        if hasattr(evaluator, name) and name=="":
            self.name = evaluator.name
        else:
            self.name = name
        self._data = []
        self.filename = filename
        if append and os.path.isfile(filename):
            self.data = list(np.loadtxt(filename))

    @property
    def data(self):
        return self._data

    @property
    def mean(self):
        return np.mean(self._data, axis=0)

    @property
    def std(self):
        return np.std(self._data)

    def __len__(self):
        return len(self._data)

    def __iadd__(self, value):
        self._data += value
        self.update_file()
        return self

    def __call__(self, *args, **kwargs):
        self += list(self.evaluator(*args, **kwargs))

    def append(self, value):
        self._data.append(value)
        self.update_file()

    def update_file(self):
        if self.filename is not None:
            np.savetxt(self.filename, self._data, header=self.name)


class AreaPerLipid(object):
    def __init__(self, num_lipids_per_leaflet):
        self.num_lipids_per_leaflet = num_lipids_per_leaflet
        self.name = "Area per Lipid (nm^2)"

    def __call__(self, traj):
        return traj.unitcell_lengths[:,0]*traj.unitcell_lengths[:,1] / self.num_lipids_per_leaflet


class BoxSize(object):
    def __init__(self):
        self.name = "Box Vectors (nm)"

    def __call__(self, traj):
        return traj.unitcell_lengths


class Coordinates(object):
    def __init__(self, atom_ids, coordinates=2, normalize=False, com_selection=None):
        self.atom_ids = atom_ids
        self.coordinates = coordinates
        self.normalize = normalize
        self.com_selection = com_selection
        self.name = "Coordinates"

    def __call__(self, traj):
        if self.normalize:
            normalized = normalize(traj, coordinates=self.coordinates, com_selection=self.com_selection, subselect=self.atom_ids)
            return normalized
        else:
            return traj.xyz[:, self.atom_ids, self.coordinates]


class BinEdgeUpdater(object):
    """A class that keeps track of bins along one axis, with respect to the average box size.
    """
    def __init__(self, num_bins=100, coordinate=2):
        self.num_bins = num_bins
        self.coordinate = coordinate
        self.average_box_size = 0.0
        self.n_frames = 0

    def __call__(self, traj):
        box_size = traj.unitcell_lengths[:, self.coordinate]
        self.average_box_size = self.n_frames * self.average_box_size + traj.n_frames * box_size.mean()
        self.n_frames += traj.n_frames
        self.average_box_size /= self.n_frames

    @property
    def edges(self):
        return np.linspace(0.0, self.average_box_size,
                           self.num_bins+1, endpoint=True)

    @property
    def edges_around_zero(self):
        return np.linspace(-0.5*self.average_box_size, 0.5*self.average_box_size,
                           self.num_bins+1, endpoint=True)

    @property
    def bin_centers_around_zero(self):
        edges = self.edges_around_zero
        return 0.5*(edges[:-1] + edges[1:])

    @property
    def bin_centers(self):
        edges = self.edges
        return 0.5*(edges[:-1] + edges[1:])


class Distribution(BinEdgeUpdater):
    def __init__(self, atom_selection, coordinate, nbins=100, com_selection=None):
        """
        Args:
            atom_selection:
            coordinate:
            nbins:
            com_selection: List of atom ids to calculate the com of the membrane, to make the distribution relative to
                    the center of mass.
        """
        super(Distribution, self).__init__(num_bins=nbins, coordinate=coordinate)
        self.atom_selection = atom_selection
        self.counts = 0.0
        self.com_selection = com_selection

    @property
    def probability(self):
        return self.counts / self.counts.sum()

    @property
    def free_energy(self):
        """in kBT"""
        return - np.log(self.counts / np.max(self.counts))

    def __call__(self, trajectory):
        super(Distribution, self).__call__(trajectory)
        atom_ids = select_atoms(trajectory, self.atom_selection)
        com_ids = select_atoms(trajectory, self.com_selection)
        normalized = normalize(trajectory, self.coordinate, subselect=atom_ids, com_selection=com_ids)
        histogram = np.histogram(normalized, bins=self.num_bins, range=(0, 1))  # this is !much! faster than manual bins
        self.counts = self.counts + histogram[0]

    def __add__(self, other):
        assert self.atom_selection == other.atom_selection
        assert self.com_selection == other.com_selection
        assert self.num_bins == other.num_bins
        assert self.coordinate == other.coordinate
        sumdist = Distribution(self.atom_selection, self.coordinate, self.num_bins, self.com_selection)
        sumdist.counts = self.counts + other.counts
        sumdist.n_frames = self.n_frames + other.n_frames
        sumdist.average_box_size = ((self.average_box_size * self.n_frames + other.average_box_size * other.n_frames)
                                    / sumdist.n_frames)
        return sumdist

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(self, other)

    def save(self, filename):
        data = np.array([self.bin_centers, self.bin_centers_around_zero, self.counts,
                         self.probability, self.free_energy])
        np.savetxt(filename, data.transpose(),
                   header="bin_centers, bin_centers_around_0, counts, probability, free_energy_(kBT)\n")

        with open(filename + ".pic", 'wb') as pic:
            pickle.dump(self, pic)

    @staticmethod
    def load_from_pic(filename):
        with open(filename, 'rb') as pic:
            return pickle.load(pic)


class EnergyDecomposition(object):
    """Calculate different potential energy terms from a trajectory
    """
    def __init__(self, system):
        self.system = system
        dummy_integrator = LangevinIntegrator(1., 1., 1.)
        self.context = Context(system, dummy_integrator)
        self.forcegroups = []

    def assign_force_groups(self, forces):
        self.forcegroups.clear()
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            # make the force object to str and human readable
            forcename = str(force).split("::")[1].split("*")[0].strip()
            if forcename in forces or forces == "all":
                self.forcegroups.append(forcename)
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if isinstance(force, NonbondedForce):
                force.setReciprocalSpaceForceGroup(self.system.getNumForces())
                self.forcegroups.append("Reciprocal")

    def calculate_energies_from_context(self):
        energies = []
        for i in range(len(self.forcegroups)):
            energies.append(
                self.context.getState(
                    getEnergy=True, groups={i}
                ).getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
            )
        return energies

    def __call__(self, traj, **kwargs):
        """
        Args:
        - traj: the trajectory to get energies from
        - force_to_return (kwargs):
        HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, CustomTorsionForce,
        CMAPTorsionForce, NonbondedForce, CMMotionRemover, and MORE TO COME ...
        - n_frames (kwargs): int, only the first n frames of the trajectory
        are used to calculate energies if this is provided
        """
        n_frames = kwargs.get("n_frames", traj.n_frames)
        forces_to_return = kwargs.get("forces_to_return", "all")
        self.assign_force_groups(forces_to_return)
        self.context.reinitialize(preserveState=True)
        energy_terms = []
        for frame in range(n_frames):
            box_vectors = lengths_and_angles_to_box_vectors(
                *traj.unitcell_lengths[frame], *traj.unitcell_angles[frame]
            )
            self.context.setPeriodicBoxVectors(*box_vectors)
            self.context.setPositions(traj.xyz[frame])
            energy_terms.append(self.calculate_energies_from_context())
        return np.array(energy_terms)

    def as_data_frame(self, energies, **kwargs):
        forcegroups = kwargs.get("forcegroups", self.forcegroups)
        # delete zero columns
        nonzero_energies = []
        nonzero_forcegroups = []
        for i in range(energies.shape[1]):
            if np.any(np.abs(energies[:,i]) > 1e-5):
                nonzero_energies.append(energies[:,i])
                nonzero_forcegroups.append(forcegroups[i])
        return pd.DataFrame(np.transpose(nonzero_energies), columns=nonzero_forcegroups)


class ModuliInput(object):
    """
    Input generator for the bilayer moduli analysis.

    Args:
        head_atoms (array or list): Either a 1D or 2D array of integers.
                                    If 1D, each element refers to the particle ID that uniquely defines the lipid head.
                                    If 2D, the head is defined as a weighted average of different atoms in the lipid.
                                    The order of particle IDs is
                                    [[lipid1_atom1, lipid2_atom1, lipid3_atom1, ... ]
                                     [lipid1_atom2, lipid2_atom2, lipid3_atom2, ... ]]
        tail_atoms (array or list): see head_atoms.
        head_weights (None or array): weights for averaging the different atoms defining the head. The default refers
                                    to the geometric center (equal weighting).
        tail_weights (None or array): see head_weights.
        bilayer_normal (string): 'x', 'y', or 'z'.
        boxsizefile_prefix (string): The prefix for the file that stores the box dimensions.
        lipidfile_prefix (string): The prefix for the file that store the head and tail coordinates.
        append (bool): If True, append to existing files.

    Examples:
        Given sequences of a POPC trajectory dyn1.dcd, dyn2.dcd, ..., and a psf file popc.psf, you would do:

        >>> import mdtraj as md
        >>> from rflow import ModuliInput, TrajectoryIterator
        >>>
        >>> trajectories = TrajectoryIterator(filename_template="dyn{}.dcd", topology_file="popc.psf")
        >>> c2atoms = trajectories.topology.select("resname POPC and name C2")
        >>> c216atoms = trajectories.topology.select("resname POPC and name C216")
        >>> c316atoms = trajectories.topology.select("resname POPC and name C316")
        >>>
        >>> input_generator = ModuliInput(head_atoms=c2atoms, tail_atoms=[c216atoms, c316atoms])
        >>> for traj in trajectories:
        >>>     input_generator(traj)

        This command will generate input files for the bilayer moduli analysis.

    """
    def __init__(self, head_atoms, tail_atoms, head_weights=None, tail_weights=None, bilayer_normal='z',
                 boxsizefile_prefix="boxsize", lipidfile_prefix="Lipid", append=False):
        self.boxsizefile_prefix = boxsizefile_prefix
        self.lipidfile_prefix = lipidfile_prefix
        output_files = [self.boxfile(dim) for dim in range(3)] + [self.lipidfile(dim) for dim in range(3)]
        if not append and any(os.path.exists(f) for f in output_files):
            raise RickFlowException("At least one of the output files exists. Do one of the following:"
                                    " (a) remove the files, (b) specify append=True, or"
                                    " (c) set a different boxsize_filename/lipid_filename.")
        head_atoms = np.array(head_atoms, dtype=int)
        tail_atoms = np.array(tail_atoms, dtype=int)
        self.n_head_atoms_per_lipid = 1 if len(head_atoms.shape) == 1 else head_atoms.shape[0]
        self.n_tail_atoms_per_lipid = 1 if len(tail_atoms.shape) == 1 else tail_atoms.shape[0]
        self.n_lipids = head_atoms.shape[-1]
        assert tail_atoms.shape[-1] == self.n_lipids, "Last dimension of head and tail atoms must agree."
        self.head_atoms = np.reshape(head_atoms.transpose(), [self.n_lipids, self.n_head_atoms_per_lipid])
        self.tail_atoms = np.reshape(tail_atoms.transpose(), [self.n_lipids, self.n_tail_atoms_per_lipid])
        self.bilayer_normal = {'x':0,'y':1,'z':2}[bilayer_normal.lower()]
        self.head_weights = head_weights
        self.tail_weights = tail_weights

    def __call__(self, trajectory):
        head_positions = trajectory.xyz[:,self.head_atoms,:] * 10
        tail_positions = trajectory.xyz[:,self.tail_atoms,:] * 10
        # dimensions of head and tail positions are [frame, lipid, atom, coordinates]
        boxsize = trajectory.unitcell_lengths * 10
        # assert that the tails are in the center, otherwise recenter
        #if np.std(head_positions[:,:,self.bilayer_normal]) < np.std(tail_positions[:,:,:self.bilayer_normal]):
        #    warnings.warn("The trajectory appears to have the lipid tails "
        #                  "pointing outwards from the center of the box. "
        #                  "The ModuliInput analysis will operate on a reframed trajectory.")
        #    raise NotImplementedError()

        # geometric center of tail atoms in each lipid
        averaged_head_positions = np.average(head_positions, weights=self.head_weights, axis=2)
        averaged_tail_positions = np.average(tail_positions, weights=self.tail_weights, axis=2)
        # shift center to boxsize/2 in the x and y axis (only necessary for comparison and can
        # probably be removed at a later stage)
        for coordinate in (0,1):
            if abs(averaged_head_positions[:,:,coordinate].mean()) < 0.25 * boxsize[:, coordinate].mean():
                # Trajectory is centered around zero
                averaged_head_positions[:,:,coordinate] += 0.5*boxsize[:,None,coordinate]
                averaged_tail_positions[:,:,coordinate] += 0.5*boxsize[:,None,coordinate]
        self.write(boxsize, averaged_head_positions, averaged_tail_positions)

    def boxfile(self, coordinate):
        return f"{self.boxsizefile_prefix}{'XYZ'[coordinate]}.out"

    def lipidfile(self, coordinate):
        return f"{self.lipidfile_prefix}{'XYZ'[coordinate]}.out"

    @staticmethod
    def vector2string(vector):
        return "".join(map(lambda x: f"  {x: .14f}    {os.linesep}", list(vector)))

    def write(self, boxsize, head_positions, tail_positions):
        mode = "a+"
        for coordinate in range(3):
            with open(self.boxfile(coordinate), mode) as fp:
                fp.write(self.vector2string(boxsize[:,coordinate]))
            with open(self.lipidfile(coordinate), mode) as fp:
                vector = (np.stack([head_positions[:,:,coordinate], tail_positions[:,:,coordinate]], axis=-1)).flatten()
                fp.write(self.vector2string(vector))


