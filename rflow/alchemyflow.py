
import os
from copy import deepcopy

from rflow.openmm import unit as u

from openmmtools import alchemy

from rflow.workflow import PsfWorkflow
from rflow.exceptions import SoluteAtomsNotSet
from rflow.equilibration import equilibrate
from rflow.reporters.alchemyreporter import AlchemyReporter


class AlchemyFlow(PsfWorkflow):
    """
    Workflow to compute alchemical free energies of annihilation.
    """
    def __init__(
        self,
            lambda_index,
            lambdas_vdw,
            lambdas_elec,
            dcd_file="dyn.{}.dcd",
            energy_file="ener.{}.txt",
            restart_file="restart.{}.xml",
            output_frequency=1000,
            gpu_id=0,
            append=True,
            **kwargs
    ):
        """

        Args:
            lambda_index: can be the index of a list or the key of an ordered dict
            lambdas_vdw: a list
            lambdas_elec: a list
            dcd_file:
            energy_file:
            restart_file:
            append:
            **kwargs:
        """
        super(AlchemyFlow, self).__init__(**kwargs)
        self.lambda_index = lambda_index
        # convert lists to dicts
        self.lambdas_vdw = lambdas_vdw
        self.lambdas_elec = lambdas_elec
        assert len(lambdas_vdw) == len(lambdas_elec)
        self._solute_atoms = None
        self.alchemical_state = None
        self.dcd_file = dcd_file.format(lambda_index)
        self.energy_file = energy_file.format(lambda_index)
        self.restart_file = restart_file.format(lambda_index)
        self.output_frequency = output_frequency
        self.gpu_id = gpu_id
        if not os.path.isfile(self.dcd_file) or not os.path.isfile(self.energy_file):
            self.append=False
        else:
            self.append=append

    @property
    def solute_atoms(self):
        return self._solute_atoms

    @solute_atoms.setter
    def solute_atoms(self, atom_ids):
        self._solute_atoms = atom_ids

    @property
    def num_samples(self):
        return self.steps//self.output_frequency

    @property
    def num_lambda_states(self):
        assert len(self.lambdas_elec) == len(self.lambdas_vdw)
        return len(self.lambdas_elec)

    def create_system(
            self,
            disable_lrc=True,
            vdw_switching="vswitch",
            switch_distance=10 * u.angstrom,
            cutoff_distance=12 * u.angstrom,
            **kwargs
    ):
        super(AlchemyFlow, self).create_system(
            disable_lrc=disable_lrc,
            vdw_switching=vdw_switching,
            switch_distance=switch_distance,
            cutoff_distance=cutoff_distance,
            **kwargs
        )

        # Alchemically modify the system.
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

        self._original_system = deepcopy(self._system)
        self._system = deepcopy(alchemical_system)
        self.alchemical_state = alchemical_state

    def initialize_state(self, initialize_velocities=True, pdb_output_file=None):
        if self.append:
            # both out files exist
            # check if energy file is consistent with DCD file, if not set append to False and raise an exception
            # either read checkpoint (if exists) or last dcd frame
            # assert nframes > 0
            self.append = False
            raise NotImplementedError("Appending alchemical simulations is not implemented, yet.")
        if not self.append:
            super(AlchemyFlow, self).initialize_state(initialize_velocities, pdb_output_file)

    def create_simulation(
            self,
            integrator,
            barostat=None,
            table_output_interval=0,
            table_output_file="dyn.txt"
    ):
        super(AlchemyFlow, self).create_simulation(
            integrator,
            barostat,
            gpu_id=self.gpu_id,
            dcd_output_interval=self.output_frequency,
            dcd_output_file=self.dcd_file,
            table_output_interval=table_output_interval,
            table_output_file=table_output_file
        )
        self.alchemical_state.apply_to_context(self.context)
        self.simulation.reporters.append(
            AlchemyReporter(
                self.energy_file, self.output_frequency, self.alchemical_state, self.temperature, self.timestep,
                self.lambdas_vdw, self.lambdas_elec, self.lambda_index, self.append
            )
        )

    def run(self, num_steps=10000, **kwargs):
        """
        Args:
            **kwargs: Keyword arguments for equilibrate.
        """
        # equilibrate
        if not self.append:
            if "gpu_id" not in kwargs:
                kwargs["gpu_id"] = self.gpu_id
            equilibrate(self.simulation, **kwargs)

        # run
        self.simulation.step(num_steps)

        self.simulation.saveState(self.restart_file)
        print(f"Samples written to {self.dcd_file}")
        print(f"Energies written to {self.energy_file}")
        print(f"Restart written to {self.restart_file}")

