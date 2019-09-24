#! /usr/bin/env python

from copy import deepcopy
import os
import warnings
import logging

import yaml
import numpy as np
import click

import mdtraj as md

from simtk import unit as u
from simtk.openmm import MonteCarloMembraneBarostat
from simtk.openmm.app import CharmmPsfFile, PME, CharmmParameterSet

from openmmtools.multistate import multistateanalyzer
from openmmtools.multistate import ReplicaExchangeSampler, ReplicaExchangeAnalyzer, MultiStateReporter
from openmmtools.mcmc import LangevinDynamicsMove
from openmmtools.states import ThermodynamicState, SamplerState

from rflow.utility import read_input_coordinates
from rflow.utility import read_box_dimensions
from rflow import scale_subsystem_charges, omm_vfswitch
from rflow.utility import disable_long_range_correction, recenter_positions


warnings.filterwarnings(
    "ignore",
    message="Warning: The openmmtools.multistate API is experimental and may change in future releases",
    category=UserWarning
)


class ReplicaExchangeWorkflow:
    """A workflow to run a replica exchange simulation."""
    def __init__(self, configuration_file="rex.yml"):
        with open(configuration_file, 'r') as f:
            self._configuration = yaml.load(f, yaml.SafeLoader)
        self.parameters = CharmmParameterSet(*self["toppar"])
        self.psf = CharmmPsfFile(self["psf"])

        # parse selections
        self.mdtraj_topology = md.Topology.from_openmm(self.psf.topology)
        self.selection = {
            key: self.mdtraj_topology.select(value) for key,value in self["selection"].items()
        }
        self._system = None
        self._thermodynamic_states = None
        self._sampler_states = None
        self._mcmc_move = None

    def __getitem__(self, item):
        """easy access to the configuration"""
        if item in self._configuration:
            return self._configuration[item]
        else:
            raise KeyError(f"ReplicaExchangeWorkflow has no key {item}")

    @property
    def system(self):
        if self._system is None:
            self.create_system()
        return self._system

    @property
    def thermodynamic_states(self):
        if self._thermodynamic_states is None:
            self.create_thermodynamic_states()
        return self._thermodynamic_states

    @property
    def sampler_states(self):
        if self._sampler_states is None:
            self.create_sampler_states()
        return self._sampler_states

    @property
    def mcmc_move(self):
        if self._mcmc_move is None:
            self.create_mcmc_move()
        return self._mcmc_move

    def create_system(self):
        self.psf.setBox(3,3,3)  # is going to be replaced by sampler state
        self._system = self.psf.createSystem(
            self.parameters,
            nonbondedMethod=PME,
            nonbondedCutoff=self["cutoff"]*u.nanometer
        )
        disable_long_range_correction(self.system)
        omm_vfswitch.vfswitch(
            self.system,
            self.psf,
            switch_distance=self["switch"]*u.nanometer,
            cutoff_distance=self["cutoff"]*u.nanometer
        )

    def create_thermodynamic_state(self, system, **kwargs):
        # electrostatic decoupling
        if "charge_scaling" in kwargs and kwargs["charge_scaling"] != 1.0:
            scale_subsystem_charges(
                system,
                self.psf.topology,
                self.selection[kwargs["charge_scaling_selection"]],
                kwargs["charge_scaling"],
                handle_internal_within=kwargs["charge_scaling_handle_internal_within"],
                handle_external_beyond=kwargs["charge_scaling_handle_external_beyond"]
            )
        # umbrella windows
        # ...
        # add barostat
        barostat = MonteCarloMembraneBarostat(
            kwargs["pressure"] * u.atmosphere, 0.0,
            kwargs["temperature"] * u.kelvin,
            MonteCarloMembraneBarostat.XYIsotropic,
            MonteCarloMembraneBarostat.ZFixed, 25
        )
        system.addForce(barostat)
        return ThermodynamicState(
            system,
            kwargs["temperature"] * u.kelvin,
            kwargs["pressure"] * u.atmosphere
        )

    def create_thermodynamic_states(self):
        """
        """
        thermodynamic_states = []
        for state in self["thermodynamic_states"]:
            system = deepcopy(self.system)
            settings = self["thermodynamic_state_settings"]
            settings.update(state)
            thermodynamic_states.append(self.create_thermodynamic_state(system, **settings))
        self._thermodynamic_states = thermodynamic_states

    def create_sampler_states(self):
        sampler_states = []
        for state in self["sampler_states"]:
            positions = read_input_coordinates(state["coordinates"], topology=self.psf.topology)
            box_dimensions = read_box_dimensions(state["coordinates"], self.psf.topology) * u.nanometer
            if not box_dimensions:
                box_dimensions = np.array(state["box_dimensions"]) * u.angstrom
            if "center_around" in state and state["center_around"] is not None:
                positions = recenter_positions(
                    positions,
                    list(self.selection[state["center_around"]]),
                    self.psf.topology, box_dimensions
                )
            box_vectors = np.diag(box_dimensions) * u.nanometer
            sampler_state = SamplerState(positions, box_vectors=box_vectors)
            sampler_states.append(sampler_state)
        self._sampler_states = sampler_states

    def create_mcmc_move(self):
        self._mcmc_move = LangevinDynamicsMove(
            collision_rate=5./u.picoseconds,
            reassign_velocities=False,
            **self["mcmc_move"]
        )

    def multistate_sampler(self):
        if os.path.isfile(self["storage_file"]):
            return ReplicaExchangeSampler.from_storage(self["storage_file"]), True
        else:
            sampler = ReplicaExchangeSampler(
                mcmc_moves=[self.mcmc_move] * len(self.thermodynamic_states),
                number_of_iterations=self["n_iterations"]
            )
            reporter = MultiStateReporter(self["storage_file"], checkpoint_interval=1)
            sampler.create(self.thermodynamic_states, self.sampler_states, storage=reporter)
            return sampler, False

    def to_mdtraj(self, iterations, replica=0, selection=None):
        reporter = MultiStateReporter(storage=self["storage_file"], open_mode='r', checkpoint_interval=1)
        selection = (
            np.arange(self.mdtraj_topology.n_atoms)
            if selection is None
            else self.selection[selection]
        )
        topology = self.mdtraj_topology
        topology = topology.subset(selection)
        xyz = np.zeros((len(iterations), len(selection), 3))
        unitcell_lengths = np.zeros((len(iterations), 3))
        unitcell_angles = 90.0 * np.ones((len(iterations), 3))
        states_indices = reporter.read_replica_thermodynamic_states(iteration=iterations)
        time = np.zeros(len(iterations))
        for i, iteration in enumerate(iterations):
            sampler_states = reporter.read_sampler_states(iteration)
            sampler_state = sampler_states[states_indices[i, replica]]
            xyz[i] = (sampler_state.positions).value_in_unit(u.nanometer)[selection]
            box_vectors = sampler_state.box_vectors.value_in_unit(u.nanometer)
            unitcell_lengths[i] = np.array([box_vectors[0,0], box_vectors[1,1], box_vectors[2,2]])
            time[i] = iteration * self["mcmc_move"]["timestep"]

        return md.Trajectory(xyz, topology, time, unitcell_lengths, unitcell_angles)


@click.group()
@click.option("-f", "--config-file", type=click.Path(exists=True), default="rex.yml",
              help="yaml file with configuration")
@click.pass_context
def rex(ctx, config_file):
    """Replica Exchange Simulation in OpenMM"""
    flow = ReplicaExchangeWorkflow(config_file)
    ctx.obj = flow


@rex.command()
@click.option("-n", type=int, default=1000, help="Number of iterations.")
@click.pass_context
def run(ctx, n):
    """Run replica exchange simulation."""
    flow = ctx.obj
    sampler, is_restart = flow.multistate_sampler()
    if is_restart:
        sampler.extend(n)
    else:
        sampler.run(n)


@rex.command()
@click.pass_context
def status(ctx):
    """Print current status of the replica exchange simulation."""
    flow = ctx.obj
    print(ReplicaExchangeSampler.read_status(flow["storage_file"]))


@rex.command()
@click.pass_context
def stats(ctx):
    """Print mixing statistics."""
    flow = ctx.obj

    # enable logging
    handler = logging.StreamHandler()
    logger = logging.getLogger(multistateanalyzer.__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # print mixing statistics
    reporter = MultiStateReporter(storage=flow["storage_file"], open_mode='r', checkpoint_interval=1)
    analyzer = ReplicaExchangeAnalyzer(reporter)
    analyzer.show_mixing_statistics(number_equilibrated=1)


@rex.command()
@click.option("-r", "--replica", default=0, type=int, help="index of the thermodynamic state")
@click.option("-b", "--begin", default=0, type=int, help="first frame")
@click.option("-e", "--end", default=1, type=int, help="last frame")
@click.option("-s", "--step", default=1, type=int, help="step between frames")
@click.option("-f", "--filename", default="dyn.dcd", type=str, help="filename of the trajectory")
@click.option("--selection", default=None, type=str, help="selection to write only a subset of atoms")
@click.pass_context
def traj(ctx, replica, begin, end, step, filename, selection):
    """
    Extract trajectory for one thermodynamic state from the results file.
    A pdb file is also written to enable reading into vmd.
    """
    flow = ctx.obj
    traj = flow.to_mdtraj(
        iterations=list(np.arange(begin, end, step)),
        replica=replica,
        selection=selection
    )
    traj.save(filename)
    traj.slice(0).save_pdb(filename+".pdb")


@rex.command()
@click.option("-i", "--iteration", default=-1, type=int, help="first frame")
@click.option("-f", "--filename", default=None, type=str, help="filename of the trajectory")
@click.option("--selection", default=None, type=str, help="selection to write only a subset of atoms")
@click.pass_context
def snapshot(ctx, iteration, filename, selection):
    """
    Extract snapshot of all replicas at one point in time.
    """
    flow = ctx.obj
    filename = f"snapshot.{iteration}.dcd" if filename is None else filename
    trajectories = []
    for replica, _ in enumerate(flow.thermodynamic_states):
        trajectories.append(flow.to_mdtraj(
            iterations=[iteration],
            replica=replica,
            selection=selection
        )
        )
    snapshot = md.join(trajectories)
    snapshot.save(filename)
    snapshot.slice(0).save_pdb(filename+".pdb")


if __name__ == "__main__":
    rex()
