
import os

from openmmtools import alchemy

from simtk import unit as u

from rflow.workflow import Workflow
from rflow.utility import read_input_coordinates
from rflow.exceptions import SoluteAtomsNotSet
from rflow.equilibration import equilibrate


class AlchemyFlow(Workflow):
    """
    Alchemical free energies of annihilation.
    """
    def __init__(
        self,
            *args,
            lambda_index,
            lambdas_vdw,
            lambdas_elec,
            dcd_file="dyn.{}.dcd",
            energy_file="ener.{}.txt",
            restart_file="restart.{}.txt",
            append=True,
            **kwargs
    ):
        super(AlchemyFlow, self).__init__(*args, **kwargs)
        self.lambda_index = lambda_index
        self.lambdas_vdw = lambdas_vdw
        self.lambdas_elec = lambdas_elec
        self._solute_atoms = None
        self.alchemical_state = None
        self.dcd_file = os.path.join(self.work_dir, dcd_file.format(lambda_index))
        self.energy_file = os.path.join(self.work_dir, energy_file.format(lambda_index))
        self.restart_file = os.path.join(self.work_dir, restart_file.format(lambda_index))
        self.append=append

    @property
    def solute_atoms(self):
        return self._solute_atoms

    @solute_atoms.setter
    def solute_atoms(self, solute_atoms):
        self.solute_atoms = solute_atoms

    def _finalize_system(self):
        """Alchemically modify the system."""
        if self.solute_atoms is None:
            raise SoluteAtomsNotSet()
        factory = alchemy.AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=True,
                                                    alchemical_pme_treatment='exact')
        region = alchemy.AlchemicalRegion(alchemical_atoms=self.solute_atoms,
                                          annihilate_electrostatics=True,
                                          annihilate_sterics=True)
        alchemical_system = factory.create_alchemical_system(self.system, region)
        alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
        alchemical_state.lambda_sterics = self.lambdas_vdw[self.lambda_index]
        alchemical_state.lambda_electrostatics = self.lambdas_elec[self.lambda_index]
        self.system = alchemical_system
        self.alchemical_state = alchemical_state

    def _initialize_state(self):
        if os.path.isfile(self.dcd_file) and self.append:
            # check if energy file is consistent with DCD file, if not set append to False and raise an exception
            # either read checkpoint (if exists) or last dcd frame
            self.append = False
            raise NotImplementedError("Appending alchemical simulations is not implemented, yet.")
        else:
            if self.psf.topology.getPeriodicBoxVectors():
                self.context.setPeriodicBoxVectors(
                    *self.psf.topology.getPeriodicBoxVectors())
            self.context.setPositions(self.positions)
            if self.initialize_velocities:
                temperature = self.simulation.integrator.getTemperature()
                print("Setting random initial velocities with temperature {}".format(temperature))
                self.context.setVelocitiesToTemperature(temperature)

    def run(self, **kwargs):
        """
        Args:
            **kwargs: Keyword arguments for equilibrate.
        """
        # ---- EQUILIBRATE ----
        if not self.append:
            equilibrate(self.simulation, **kwargs)

        # ---- WRITE FILES ----
        if not self.append or not os.path.isfile(self.dcd_file):
            # write headers of dcd and energy file

        # ---- RUN ----
        kT = u.AVOGADRO_CONSTANT_NA * u.BOLTZMANN_CONSTANT_kB * self.temperature

        energies = np.zeros([num_samples, num_states], np.float64)

        with open(dcd_filename, "wb") as dcd:

            dcd_file = DCDFile(dcd, topology, time_step, 0, steps_between_samples)

            for sample in range(num_samples):

                print('Production ... Sample {}/{} ... '.format(sample + 1, num_samples))

                # Set to current alchemical state
                self.alchemical_state.lambda_sterics = self.lambdas_vdw[self.lambda_index]
                self.alchemical_state.lambda_electrostatics = self.lambdas_elec[self.lambda_index]
                self.alchemical_state.apply_to_context(self.simulation.context)

                # Run some dynamics
                self.simulation.step(steps_between_samples)

                # Save snapshot
                state = context.getState(getPositions=True, enforcePeriodicBox=True)
                dcd_file.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())

                # Compute energies at all alchemical states
                for state in range(num_states):
                    alchemical_state.lambda_sterics = lambdas_vdw[state]
                    alchemical_state.lambda_electrostatics = lambdas_elec[state]
                    alchemical_state.apply_to_context(context)
                    energies[sample, state] = context.getState(getEnergy=True).getPotentialEnergy() / kT

        # # Save Energies
        header = """Energies of all lambda states for {} samples. Time between samples: {}.
         Samples were created in lambda state {}/{} (lambda_vdw: {}, lambda_elec: {}).
         State1                  State2                      .....""".format(
            num_samples, sample_interval, lambda_index, num_states - 1,
            lambdas_vdw[lambda_index], lambdas_elec[lambda_index]
        )
        np.savetxt(energy_filename, energies, header=header)

    def regenerate_energy_file(self):
        """
        Recalculate all energies, for example when lambda states have changed.
        """
        raise NotImplementedError()


