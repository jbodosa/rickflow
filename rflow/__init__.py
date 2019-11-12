# -*- coding: utf-8 -*-

"""Top-level package for rickflow."""

__author__ = """Andreas Krämer"""
__email__ = 'kraemer.research@gmail.com'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from rflow.exceptions import *
from rflow.utility import increment_using_multiindices, select_atoms, CWD, abspath
from rflow.trajectory import (TrajectoryIterator, CharmmTrajectoryIterator, make_topology, normalize,
                              center_of_mass_of_selection)
from rflow.observables import (TimeSeries, AreaPerLipid, Coordinates, BoxSize, BinEdgeUpdater, Distribution,
                               EnergyDecomposition, ModuliInput)

from rflow.biasing import *
from rflow.permeation import TransitionCounter, PermeationEventCounter, PermeationEventCounterWithoutBuffer
from rflow.biasing import *
from rflow.integrators import NonequilibriumLangevinIntegrator
from rflow.rickflow import *
from rflow.nearest import NearestNeighborAnalysis, NearestNeighborResult
from rflow.modcharge import scale_subsystem_charges