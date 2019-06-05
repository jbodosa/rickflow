"""
Tools for analyzing MD observables
"""

import os
import pickle
import numpy as np
from rflow.trajectory import normalize
from rflow.utility import select_atoms


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
