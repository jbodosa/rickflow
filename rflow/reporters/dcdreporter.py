"""
DCD reporter that does not report the zero-th step and inserts the correct "first step".
The reason for this is that CHARMM does not read frames with ID 0.
"""

from rflow.openmm.app import DCDFile
from rflow.openmm import unit as u
import numpy as np
import networkx as nx
import mdtraj as md
from mdtraj.utils import box_vectors_to_lengths_and_angles


class DCDReporter(object):
    """DCDReporter outputs a series of frames from a Simulation to a DCD file.
    To use it, create a DCDReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file, reportInterval, velocities=False, append=False, enforcePeriodicBox=None, centerAtOrigin=False):
        """Create a DCDReporter.
        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        velocities : bool
            If True, write velocities. If False (the default), write positions.
        append : bool=False
            If True, open an existing DCD file to append to.  If False, create a new file.
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        centerAtOrigin: bool (default: False)
            Specifies whether the molecules should be rewrapped so that the origin of the periodic box is at (0,0,0).
        """
        self._reportInterval = reportInterval
        self._append = append
        self._enforcePeriodicBox = enforcePeriodicBox
        self._report_velocities = velocities
        if append:
            mode = 'r+b'
        else:
            mode = 'wb'
        self._centerAtOrigin = centerAtOrigin
        self._out = open(file, mode)
        self._dcd = None
        self._molecules = None

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        if self._report_velocities:
            return (steps, False, True, False, False, self._enforcePeriodicBox)
        else:
            return (steps, True, False, False, False, self._enforcePeriodicBox)

    def report(self, simulation, state):
        """Generate a report.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if self._dcd is None:
            self._dcd = DCDFile(
                self._out,
                simulation.topology,
                simulation.integrator.getStepSize(),
                simulation.currentStep,
                self._reportInterval,
                self._append
            )
        if self._centerAtOrigin and self._molecules is None:
            self._compute_molecules(simulation.topology)
        if self._report_velocities:
            self._dcd.writeModel(
                state.getVelocities().value_in_unit(u.angstrom/u.picosecond)*u.angstrom,  # to trick the DCDFile instance,
                periodicBoxVectors=state.getPeriodicBoxVectors()
            )
        else:
            positions = state.getPositions()
            box_vectors = state.getPeriodicBoxVectors()
            if self._centerAtOrigin:
                positions = self._rewrap_around_zero(
                    state.getPositions(asNumpy=True),
                    state.getPeriodicBoxVectors(asNumpy=True)
                )
            self._dcd.writeModel(positions, periodicBoxVectors=box_vectors)

    def __del__(self):
        self._out.close()

    def _compute_molecules(self, topology):
        """fill self._molecules field with np.arrays of indices belonging to each molecule"""
        graph = md.Topology.from_openmm(topology).to_bondgraph()
        self._molecules = []
        for mol in nx.connected_components(graph):
            self._molecules.append(np.array([atom.index for atom in mol], dtype=np.int32))

    def _rewrap_around_zero(self, positions, box_vectors):
        # only for an orthorhombic box
        vectors = box_vectors.value_in_unit(u.nanometer)
        assert (all(abs(vectors[i][j]) < 1e-5 for i in range(3) for j in range(3) if i != j ),
                NotImplementedError("Centering DCDs around origin is only for orthorhombic boxes"))
        box_lengths = np.array([vectors[i][i] for i in range(3)])
        pos = positions.value_in_unit(u.nanometer)
        geometric_center = lambda molecule: pos[molecule].mean(axis=0)
        for molecule in self._molecules:
            center = geometric_center(molecule)
            needs_shift = center > 0.5 * box_lengths
            pos[molecule] -= needs_shift * box_lengths
        return u.Quantity(pos, unit=u.nanometer)