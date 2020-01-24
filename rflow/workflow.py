
from simtk import unit as u
from simtk.openmm.app import (
    CharmmParameterSet, CharmmPsfFile, Simulation, PDBReporter, StateDataReporter
)
import mdtraj as md

from rflow.utility import (
    read_input_coordinates, disable_long_range_correction, get_platform, get_barostat,
    recenter_positions, read_box_dimensions
)
from rflow.exceptions import RickFlowException
from rflow import omm_vfswitch
from rflow.reporters.dcdreporter import DCDReporter


class PsfWorkflow(object):
    """
    Basic simulation workflow class.
    """
    def __init__(
            self,
            toppar,
            psf,
            crd,
            box_dimensions=None,
            center_around="not water",
            center_relative_position=0.5,
            center_dcd_at_origin=False
    ):
        self._system = None
        self._simulation = None
        self._mdtraj_topology = None

        # Load input files
        self.parameters = CharmmParameterSet(*toppar)
        self.psf = CharmmPsfFile(psf)
        self.positions = read_input_coordinates(crd, topology=self.mdtraj_topology)
        parsed_dimensions = read_box_dimensions(crd, topology=self.mdtraj_topology)
        if parsed_dimensions is not None:
            self.psf.setBox(*parsed_dimensions)
        elif box_dimensions is not None:
            box_dimensions = [dim * u.angstrom for dim in box_dimensions]
            self.psf.setBox(*box_dimensions)

        # process coordinates
        if center_around is not None:
            self.positions = recenter_positions(
                positions=self.positions,
                selection=self.select(center_around),
                topology=self.psf.topology,
                box_lengths=self.psf.boxLengths,
                center_relative_position=center_relative_position
            )
        self._center_dcd_at_origin = center_dcd_at_origin

    @property
    def system(self):
        if self._system is None:
            raise RickFlowException("system not created, yet")
        return self._system

    @property
    def simulation(self):
        if self._simulation is None:
            raise RickFlowException("simulation not created, yet")
        return self._simulation

    @property
    def timestep(self):
        return self.simulation.context.getIntegrator().getStepSize()

    @property
    def topology(self):
        return self.psf.topology

    @property
    def mdtraj_topology(self):
        if self._mdtraj_topology is None:
            self._mdtraj_topology = md.Topology.from_openmm(self.topology)
        return self._mdtraj_topology

    @property
    def context(self):
        return self.simulation.context

    @property
    def temperature(self):
        return self.context.getIntegrator().getTemperature()

    @temperature.setter
    def temperature(self, temp):
        self.context.getIntegrator().setTemperature(temp)
        barostat = get_barostat(self.system)
        if barostat is not None:
            barostat.setDefaultTemperature(temp)
            self.context.setParameter(barostat.Temperature(), temp)

    def select(self, *args, **kwargs):
        return self.mdtraj_topology.select(*args, **kwargs)

    def create_system(
            self,
            disable_lrc=True,
            vdw_switching="openmm",
            switch_distance=8 * u.angstrom,
            cutoff_distance=12 * u.angstrom,
            **kwargs
    ):
        kwargs["switchDistance"] = switch_distance
        kwargs["nonbondedCutoff"] = cutoff_distance
        self._system = self.psf.createSystem(self.parameters, **kwargs)
        if disable_lrc:
            disable_long_range_correction(self._system)
        if vdw_switching == "charmm-gui":
            omm_vfswitch.vfswitch(self._system, self.psf, switch_distance, cutoff_distance)
        elif vdw_switching == "vfswitch":
            from openmmtools.forcefactories import use_vdw_with_charmm_force_switch
            use_vdw_with_charmm_force_switch(self._system, switch_distance, cutoff_distance)
        elif vdw_switching == "vswitch":
            from openmmtools.forcefactories import use_custom_vdw_switching_function
            use_custom_vdw_switching_function(self._system, switch_distance, cutoff_distance)
        elif vdw_switching == "openmm":
            pass
        else:
            raise RickFlowException(f"vdw_switching={vdw_switching} not understood.")

    def create_simulation(
            self,
            integrator,
            barostat=None,
            gpu_id=0,
            dcd_output_interval=0,
            dcd_output_file="dyn.dcd",
            table_output_interval=0,
            table_output_file="dyn.txt",
            precision="mixed"
    ):
        """
        Initialize simulation object by passing an integrator and a barostat.

        Args:
            integrator (OpenMM integrator object): The integrator to be used.
            barostat (OpenMM barostat object): The barostat. Pass None for NVT.
        """
        if barostat:
            self.system.addForce(barostat)
        platform, platform_properties = get_platform(gpu_id, precision)
        self._simulation = Simulation(
            self.topology,
            self.system,
            integrator,
            platform,
            platform_properties
        )
        if dcd_output_interval > 0:
            self.simulation.reporters.append(
                DCDReporter(
                    dcd_output_file,
                    dcd_output_interval,
                    enforcePeriodicBox=True,
                    centerAtOrigin=self._center_dcd_at_origin
                )
            )
        if table_output_interval > 0:
            self.simulation.reporters.append(StateDataReporter(
                table_output_file,
                table_output_interval,
                step=True, time=True,
                potentialEnergy=True, kineticEnergy=True,
                totalEnergy=True, temperature=True,
                volume=True, density=True, speed=True
            ))
        print("#Running on ", self.context.getPlatform().getName())

    def initialize_state(self, initialize_velocities=True, pdb_output_file="system.pdb"):
        self.context.setPositions(self.positions)
        self.context.setPeriodicBoxVectors(*self.topology.getPeriodicBoxVectors())
        if initialize_velocities:
            self.context.setVelocitiesToTemperature(self.temperature)
        self.context.applyConstraints(1e-7)
        # write the system as a pdb file (this is important for postprocessing,
        # if virtual sites were manually added to the system)
        if pdb_output_file is not None:
            PDBReporter(pdb_output_file, 1).report(
                self.simulation, self.context.getState(getPositions=True)
            )

    def run(self, steps):
        self.simulation.step(steps)
