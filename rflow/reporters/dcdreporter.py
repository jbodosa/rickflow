"""
DCD reporter that does not report the zero-th step and inserts the correct "first step".
The reason for this is that CHARMM does not read frames with ID 0.
"""

from simtk.openmm.app import DCDFile
from simtk import unit as u


class DCDReporter(object):
    """DCDReporter outputs a series of frames from a Simulation to a DCD file.
    To use it, create a DCDReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file, reportInterval, velocities=False, append=False, enforcePeriodicBox=None):
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
        """
        self._reportInterval = reportInterval
        self._append = append
        self._enforcePeriodicBox = enforcePeriodicBox
        self._report_velocities = velocities
        if append:
            mode = 'r+b'
        else:
            mode = 'wb'
        self._out = open(file, mode)
        self._dcd = None

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
        if self._report_velocities:
            self._dcd.writeModel(
                state.getVelocities().value_in_unit(u.angstrom/u.picosecond)*u.angstrom,  # to trick the DCDFile instance,
                periodicBoxVectors=state.getPeriodicBoxVectors()
            )
        else:
            self._dcd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())

    def __del__(self):
        self._out.close()
