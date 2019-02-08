"""

"""

import os
import numpy as np


class TimeSeries(object):
    """A time series."""
    def __init__(self, name="", filename=None, append=False):
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

    def __radd__(self, value):
        self._data += value
        self.update_file()

    def append(self, value):
        self._data.append(value)
        self.update_file()

    def update_file(self):
        if self.filename is not None:
            np.savetxt(self.filename, self._data, header=self.name)


class AreaPerLipid(object):
    def __init__(self, num_lipids_per_leaflet, filename=None, append=False):
        self.num_lipids_per_leaflet = num_lipids_per_leaflet
        self.area = TimeSeries(name="Area per Lipid", filename=filename, append=append)

    def __call__(self, traj):
        self.area += traj.unitcell_lengths[:,0]*traj.unitcell_lengths[:,1] / self.num_lipids_per_leaflet

