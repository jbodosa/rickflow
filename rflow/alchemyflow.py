
import os

from openmmtools import alchemy

from rflow.workflow import Workflow

from rflow.utility import read_input_coordinates
from rflow.exceptions import SoluteAtomsNotSet


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
        factory = alchemy.AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=True,
                                                    alchemical_pme_treatment='exact')
        region = alchemy.AlchemicalRegion(alchemical_atoms=self._solute_atoms,
                                          annihilate_electrostatics=True,
                                          annihilate_sterics=True)
        alchemical_system = factory.create_alchemical_system(self.system, region)
        alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
        self.system = alchemical_system
        self.alchemical_state = alchemical_state

    def _initialize_state(self):
        if os.path.isfile(self.dcd_file) and self.append:
            # check if energy file is consistent with DCD file
            pass

    def run(self):
        if self.solute_atoms is None:
            raise SoluteAtomsNotSet()
        # ---- EQUILIBRATE ----

        # ---- RUN ----

