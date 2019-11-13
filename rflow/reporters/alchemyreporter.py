"""
Reports alchemical energies.
"""

import os
import textwrap
import warnings

import numpy as np

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
                """                # Energies of all lambda states in reduced units of kT. Time between samples: {}.
                #     Samples were created in lambda state {}/{} (lambda_vdw: {}, lambda_elec: {}).
                #     lambdas_vdw: {}
                #     lambdas_elec: {}
                #     State1                  State2                      .....
                """.format(
                    time_step*reportInterval,
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


class FreeEnergyDifference:
    """
    Calculate the free energy differences between all lambda states.
    """
    def __init__(self, file_template, n_lambdas, force_update=False, unit=None, temperature=None, n_blocks=5):
        self.file_template = file_template
        self.n_lambdas = n_lambdas
        self.n_blocks = n_blocks
        if unit is None:
            self.unit_factor = 1.0
        else:
            kT = (temperature * u.constants.BOLTZMANN_CONSTANT_kB * u.constants.AVOGADRO_CONSTANT_NA)
            self.unit_factor = kT.value_in_unit(unit)

        # energies and errors between all states; those are in units of kT
        self.result_matrix = np.zeros([n_lambdas, n_lambdas], dtype=np.float)
        self.error_matrix = np.zeros([n_lambdas, n_lambdas], dtype=np.float)
        self.mbar_error_matrix = np.zeros([n_lambdas, n_lambdas], dtype=np.float)

        # parsed from headers
        self.lambdas_vdw = None
        self.lambdas_elec = None

        if not os.path.isfile(self.summary_file) or force_update:
            self._parse_headers()
            self._calculate_free_energy_difference()
            self._save_summary_file()
        else:
            self._load_summary_file()

        self.index_vdwstart = list(self.lambdas_elec).index(0.0)
        assert self.lambdas_vdw[self.index_vdwstart] == 1.0
        assert self.lambdas_elec[self.index_vdwstart] == 0.0

    def _parse_headers(self):
        lambda_indices = set()
        for i in range(self.n_lambdas):
            file_i = self.file_template.format(i)
            with open(file_i, "r") as fp:
                lines = [fp.readline() for _ in range(5)]
            lambda_index = int(lines[1].split("/")[0].split(" ")[-1])
            lambdas_vdw = lines[2].split("[")[1].split("]")[0].split(",")
            lambdas_vdw = [float(lam) for lam in lambdas_vdw]
            lambdas_elec = lines[3].split("[")[1].split("]")[0].split(",")
            lambdas_elec = [float(lam) for lam in lambdas_elec]
            if lambda_index == 0:
                self.lambdas_vdw = lambdas_vdw
                self.lambdas_elec = lambdas_elec
            else:
                assert lambdas_vdw == self.lambdas_vdw
                assert lambdas_elec == self.lambdas_elec
            lambda_indices.add(lambda_index)
            assert lambda_index == i

    def _apply_mbar(self, time_slice=slice(None,None,None)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PendingDeprecationWarning)

            from pymbar import MBAR, timeseries
            energies = []
            for i in range(self.n_lambdas):
                file_i = self.file_template.format(i)
                assert os.path.isfile(file_i)
                energies.append(np.loadtxt(file_i, comments="#")[time_slice])
            u_kln = np.array([u.transpose() for u in energies])

            # Subsample data to extract uncorrelated equilibrium timeseries
            N_k = np.zeros(self.n_lambdas, np.int32)  # number of uncorrelated samples
            for k in range(self.n_lambdas):
                [nequil, g, Neff_max] = timeseries.detectEquilibration(u_kln[k, k, :])
                indices = timeseries.subsampleCorrelatedData(u_kln[k, k, :], g=g)
                N_k[k] = len(indices)
                u_kln[k, :, 0:N_k[k]] = u_kln[k, :, indices].T
            # Compute free energy differences and statistical uncertainties
            mbar = MBAR(u_kln, N_k)
            [DeltaF_ij, dDeltaF_ij, Theta_ij] = mbar.getFreeEnergyDifferences()
            return DeltaF_ij, dDeltaF_ij

    def _calculate_free_energy_difference(self):
        for i in range(self.n_lambdas):
            file_i = self.file_template.format(i)
            assert os.path.isfile(file_i)
            data = np.loadtxt(file_i, comments='#')
            if i == 0:
                n_samples = data.shape[0]
            else:
                assert n_samples == data.shape[0]
        df, ddf = self._apply_mbar()
        self.result_matrix = df
        self.mbar_error_matrix = ddf
        indices = np.split(np.arange(n_samples), self.n_blocks)
        all_df = []
        for i in range(self.n_blocks):
            df, ddf = self._apply_mbar(time_slice=indices[i])
            all_df.append(df[:])
        self.error_matrix = np.std(all_df, axis=0)/np.sqrt(self.n_blocks) * 2

    def _save_summary_file(self):
        with open(self.summary_file, "w") as fp:
            fp.write("# Free energy differences between lambda states in reduced units of kT.\n")
            fp.write("# lambdas_vdw\n")
            np.savetxt(fp, [self.lambdas_vdw], fmt='%-7.4f')
            fp.write("# lambdas_elec\n")
            np.savetxt(fp, [self.lambdas_elec], fmt='%-7.4f')
            fp.write("# result_matrix\n")
            np.savetxt(fp, self.result_matrix, fmt='%-7.4f')
            fp.write("# error_matrix\n")
            np.savetxt(fp, self.error_matrix, fmt='%-7.4f')
            fp.write("# mbar_error_matrix\n")
            np.savetxt(fp, self.mbar_error_matrix, fmt='%-7.4f')

    def _load_summary_file(self):
        with open(self.summary_file, "r") as fp:
            summary = np.loadtxt(fp, comments="#")
            assert summary.shape[0] == self.n_lambdas * 3 + 2
            assert summary.shape[1] == self.n_lambdas
            self.lambdas_vdw = summary[0,:]
            self.lambdas_elec = summary[1,:]
            self.result_matrix = summary[2:2+self.n_lambdas,:]
            self.error_matrix = summary[2+self.n_lambdas:2+2*self.n_lambdas,:]
            self.mbar_error_matrix = summary[2+2*self.n_lambdas:,:]

    @property
    def summary_file(self):
        return self.file_template.format("summary")

    @property
    def value(self):
        return self.unit_factor * self.result_matrix[0, -1]

    @property
    def error(self):
        return self.unit_factor * (self.error_matrix[0, -1])

    @property
    def mbar_error(self):
        return self.unit_factor * (self.mbar_error_matrix[0, -1])

    @property
    def value_elec(self):
        return self.unit_factor * self.result_matrix[0, self.index_vdwstart]

    @property
    def error_elec(self):
        return self.unit_factor * self.error_matrix[0, self.index_vdwstart]

    @property
    def mbar_error_elec(self):
        return self.unit_factor * self.mbar_error_matrix[0, self.index_vdwstart]

    @property
    def value_vdw(self):
        return self.unit_factor * self.result_matrix[self.index_vdwstart, -1]

    @property
    def error_vdw(self):
        return self.unit_factor * self.error_matrix[self.index_vdwstart, -1]

    @property
    def mbar_error_vdw(self):
        return self.unit_factor * self.mbar_error_matrix[self.index_vdwstart, -1]

    @property
    def value_steps(self):
        return self.unit_factor * self.result_matrix[0, :]

    @property
    def error_steps(self):
        return self.unit_factor * self.error_matrix[0, :]

    @property
    def mbar_error_steps(self):
        return self.unit_factor * self.mbar_error_matrix[0, :]

