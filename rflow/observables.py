"""
Tools for analyzing MD observables
"""

import os
import numpy as np
from rflow.analyze_diffusion import normalize
import mdtraj as md

class TimeSeries(object):
    """A time series."""
    def __init__(self, evaluator=None, name="", filename=None, append=False):
        """
        Args:
            evaluator (callable): The callable takes an mdtraj trajectory as its only argument and returns a numpy array.
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

    def __call__(self, traj):
        self += list(self.evaluator(traj))

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

