"""
Reports alchemical energies.
"""

import os
import textwrap
from simtk import unit as u


class AlchemyReporter(object):
    """
    Report alchemical energies to a table.
    """

    def __init__(self, file, reportInterval, alchemical_state, temperature, time_step,
                 lambdas_vdw, lambdas_elec, lambda_index, append=False,):
        """Create an AlchemyReporter.
        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        alchemical_state : openmmtools.AlchemicalState
            The alchemical state.
        append : bool=False
            If True, open an existing DCD file to append to.  If False, create a new file.
        """
        self._reportInterval = reportInterval
        self.alchemical_state = alchemical_state
        self.kT = u.AVOGADRO_CONSTANT_NA * u.BOLTZMANN_CONSTANT_kB * temperature
        self.lambdas_vdw = lambdas_vdw
        self.lambdas_elec = lambdas_elec
        self.lambda_index = lambda_index
        if not os.path.isfile(file):
            self._append = False
        else:
            self._append = append
        self.file = file
        self.mode = 'r+' if self._append else 'w'
        if not self._append:
            header = textwrap.dedent(
                """                # Energies of all lambda states in reduced units of kT. Time between samples: {} ps.
                #     Samples were created in lambda state {}/{} (lambda_vdw: {}, lambda_elec: {}).
                #     lambdas_vdw: {}
                #     lambdas_elec: {}
                #     State1                  State2                      .....
                """.format(
                    time_step,
                    lambda_index,
                    lambdas_elec,
                    lambdas_vdw[lambda_index],
                    lambdas_elec[lambda_index],
                    lambdas_vdw,
                    lambdas_elec
                )
            )
            with open(self.file, self.mode) as f:
                f.write(header)

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
        return (steps, True, False, False, False, False)

    def report(self, simulation, state):
        """Generate a report.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        # don't write zeroth step
        if simulation.currentStep == 0:
            return

        print("Saving step", simulation.currentStep)
        energies = []
        for lambda_vdw, lambda_elec in zip(self.lambdas_vdw, self.lambdas_elec):
            self.alchemical_state.lambda_sterics = lambda_vdw
            self.alchemical_state.lambda_electrostatics = lambda_elec
            self.alchemical_state.apply_to_context(simulation.context)
            energies.append(simulation.context.getState(getEnergy=True).getPotentialEnergy() / self.kT)

        # reset original state
        self.alchemical_state.lambda_sterics = self.lambdas_vdw[self.lambda_index]
        self.alchemical_state.lambda_electrostatics = self.lambdas_elec[self.lambda_index]
        self.alchemical_state.apply_to_context(simulation.context)

        with open(self.file, "a+") as f:
            f.write("".join(f"{e:25.15e}" for e in energies) + '\n')
