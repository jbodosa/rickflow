

import os
import warnings
import numpy as np

from simtk import unit as u
from simtk.openmm.app import (
    PME, CharmmParameterSet, CharmmPsfFile, HBonds, Simulation, PDBReporter
)
import mdtraj as md

from rflow.utility import (
    CWD, read_input_coordinates, disable_long_range_correction, require_cuda, get_barostat
)
from rflow.exceptions import RickFlowException
from rflow import omm_vfswitch


class Workflow(object):
    """Basic simulation workflow class.

    Subclasses can override
    """

    def __init__(
            self,
            toppar,
            psf,
            crd,
            box_dimensions,
            gpu_id=0,
            nonbonded_method=PME,
            switch_distance=8*u.angstrom,
            cutoff_distance=12*u.angstrom,
            use_vdw_force_switch=True,
            tmp_output_dir=None,
            dcd_output_interval=1000,
            table_output_interval=1000,
            steps=1000000,
            use_only_xml_restarts=False,
            misc_psf_create_system_kwargs={},
            initialize_velocities=True,
            center_around="not water",
            analysis_mode=False,
            work_dir=".",
            **kwargs
    ):
        self.work_dir = work_dir
        self.gpu_id = gpu_id
        self.context = None
        self.simulation = None
        self._mdtraj_topology = None
        self.initialize_velocities = initialize_velocities
        self.analysis_mode = analysis_mode

        # prepare temporary output directory
        if tmp_output_dir is not None:
            assert os.path.exists(tmp_output_dir)
            self.tmp_output_dir = os.path.normpath(tmp_output_dir)
            with CWD(tmp_output_dir):
                if not os.path.isdir("trj"):  # directory for trajectories
                    os.mkdir("trj")
                if not os.path.isdir("out"):  # directory for state files
                    os.mkdir("out")
                if not os.path.isdir("res"):  # directory for restart files
                    os.mkdir("res")
        else:
            self.tmp_output_dir = None
        self.dcd_output_interval = dcd_output_interval
        self.table_output_interval = table_output_interval
        self.steps = steps
        self.use_only_xml_restarts = use_only_xml_restarts
        if not steps % dcd_output_interval== 0:
            raise RickFlowException("dcd_output_interval ({}) has to be a divisor of steps ({}).".format(
                dcd_output_interval, steps
            ))
        if not steps % table_output_interval== 0:
            raise RickFlowException("table_output_interval ({}) has to be a divisor of steps ({}).".format(
                table_output_interval, steps
            ))

        with CWD(self.work_dir):
            self.parameters = CharmmParameterSet(*toppar)
            self.psf = CharmmPsfFile(psf)
            box_dimensions = [dim * u.angstrom for dim in box_dimensions]
            self.psf.setBox(*box_dimensions)
            self.positions = read_input_coordinates(crd, self.psf.topology)
        # create system
        self._cutoff_distance = cutoff_distance
        self._switch_distance = switch_distance
        self.use_vdw_force_switch = use_vdw_force_switch
        psf_create_system_kwargs = {
            "nonbondedMethod": nonbonded_method,
            "nonbondedCutoff": cutoff_distance,
            "constraints": HBonds,
            "switchDistance": switch_distance
        }
        psf_create_system_kwargs.update(misc_psf_create_system_kwargs)
        self._system = self.psf.createSystem(
            self.parameters,
            **psf_create_system_kwargs
        )

        # translate system so that the center of mass of non-waters is in the middle
        if center_around is not None:
            center_selection = self.select(center_around)
            current_com = self.center_of_mass(center_selection)
            target_com = (0.5 * self.psf.boxLengths).value_in_unit(u.nanometer)
            move = target_com - current_com
            self.positions = [xyz + move for xyz in self.positions]

        # no LRC for charmm force fields
        disable_long_range_correction(self.system)

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, new_system):
        self._system = new_system

    @property
    def temperature(self):
        if self.simulation is None:
            raise RickFlowException("Simulation is not set up yet.")
        return self.simulation.context.getIntegrator().getTemperature()

    @temperature.setter
    def temperature(self, temp):
        if self.simulation is None:
            raise RickFlowException("Simulation is not set up yet.")
        self.simulation.context.getIntegrator().setTemperature(temp)
        barostat = get_barostat(self.system)
        if barostat is not None:
            barostat.setDefaultTemperature(temp)
            self.simulation.context.setParameter(barostat.Temperature(), temp)

    @property
    def timestep(self):
        if self.simulation is None:
            raise RickFlowException("Simulation is not set up yet.")
        return self.simulation.context.getIntegrator().getStepSize()

    def select(self, *args, **kwargs):
        if self._mdtraj_topology is None:
            self._mdtraj_topology = md.Topology.from_openmm(self.psf.topology)
        return self._mdtraj_topology.select(*args, **kwargs)

    def center_of_mass(self, particle_ids):
        """
        Calculate the center of mass of a subset of atoms.

        Args:
            particle_ids (list of int): The particle ids that define the subset of the system

        Returns:
            float: center of mass in nanometer
        """
        masses = np.array([atom.element.mass.value_in_unit(u.dalton) for atom in self.psf.topology.atoms()])
        positions = np.array(self.positions)
        return np.sum(
            positions[particle_ids].transpose()
            * masses[particle_ids],
            axis=1
        ) / np.sum(masses[particle_ids])

    def prepareSimulation(self, integrator, barostat=None):
        warnings.warn(
            "prepareSimulation has been renamed into prepare_simulation. "
            "Future versions of rickflow will only support the latter.",
            DeprecationWarning
        )
        return self.prepare_simulation(integrator, barostat)

    def prepare_simulation(self, integrator, barostat=None):
        """
        Initialize simulation object by passing an integrator and a barostat.

        Args:
            integrator (OpenMM integrator object): The integrator to be used.
            barostat (OpenMM barostat object): The barostat. Pass None for NVT.
        """
        if self.use_vdw_force_switch:
            self.apply_vdw_force_switch(switch_distance=self._switch_distance, cutoff_distance=self._cutoff_distance)
        self._finalize_system()
        if self.gpu_id is not None and not self.analysis_mode:
            platform, platform_properties = require_cuda(self.gpu_id)
        else:
            platform = None
            platform_properties = None
        if barostat:
            self.system.addForce(barostat)

        with CWD(self.work_dir):
            self.simulation = Simulation(
                self.psf.topology,
                self.system,
                integrator,
                platform,
                platform_properties
            )
            self.context = self.simulation.context
            self._initialize_state()
            # write the system as a pdb file (this is important for postprocessing,
            # if virtual sites were manually added to the system)
            if not self.analysis_mode:
                PDBReporter("system.pdb", 1).report(
                    self.simulation, self.context.getState(getPositions=True)
                )
                print("#Running on ", self.context.getPlatform().getName())

    def apply_vdw_force_switch(self, switch_distance, cutoff_distance):
        """Use van der Waals force switch.
        Should be called after every custom nonbonded force has been added to the system."""
        self._system = omm_vfswitch.vfswitch(self._system, self.psf, switch_distance, cutoff_distance)

    def _finalize_system(self):
        pass

    def _initialize_state(self):
        raise NotImplementedError("Workflow subclasses have to implement an _initialize_state method.")

    def run(self):
        raise NotImplementedError("Workflow is abstract at this point.")
