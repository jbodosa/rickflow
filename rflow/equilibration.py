
import numpy as np

from simtk import unit as u
from simtk.openmm.app import Simulation, StateDataReporter
from simtk.openmm import (LangevinIntegrator, DrudeForce, DrudeLangevinIntegrator)

from rflow.exceptions import RickFlowException
from rflow.utility import get_barostat, get_force, CWD, require_cuda


def equilibrate(
        simulation,
        target_temperature,
        minimize=True,
        minimization_tolerance=10.000000000000004 * u.kilojoule /u.mole,
        max_minimization_iterations=0,
        number_of_equilibration_steps=100000,
        start_temperature=100 * u.kelvin,
        gpu_id=0,
        work_dir="."
    ):
    """
    Energy minimization and gradual heating to get a stable starting configuration.

    Args:
        simulation: A simulation object, whose context is already
            initialize with periodic box vectors and particle positions.
        target_temperature:
        minimize (bool): Whether to perform a minimization
        minimization_tolerance:
        max_minimization_iterations:
        number_of_equilibration_steps:
        start_temperature:
        gpu_id:

    Returns:

    """
    if simulation is None:
        raise RickFlowException("equilibrate can only be called after preparing the simulation "
                                "(RickFlow.prepareSimulation)")
    barostat = get_barostat(simulation.system)
    number_of_equilibration_steps //= 2  # two equilibration phases
    isdrude = get_force(simulation.system, [DrudeForce]) is not None

    with CWD(work_dir):
        # set up simulation
        if isdrude:
            print("Found Drude Force.")
            integrator = DrudeLangevinIntegrator(start_temperature, 5.0 / u.picosecond,
                                                 1.0 * u.kelvin, 20.0 / u.picosecond,
                                                 1.0 * u.femtosecond)
        else:
            integrator = LangevinIntegrator(start_temperature, 5.0 / u.picosecond, 1.0 * u.femtosecond)
        if gpu_id is not None:
            platform, platform_properties = require_cuda(gpu_id=gpu_id)
        else:
            platform = None
            platform_properties = None
        equilibration = Simulation(simulation.topology, simulation.system, integrator, platform, platform_properties)
        state = simulation.context.getState(getPositions=True)
        equilibration.context.setPositions(state.getPositions())
        equilibration.context.setPeriodicBoxVectors(*list(state.getPeriodicBoxVectors()))
        equilibration.context.setVelocities((state.getPositions(asNumpy=True)*0).tolist())
        equilibration.reporters.append(StateDataReporter(
            "equilibration.txt", 100, step=True, time=True,
            potentialEnergy=True, temperature=True,
            volume=True, density=True, speed=True)
        )

        if minimize:
            print("Starting Minimization...")
            equilibration.minimizeEnergy(minimization_tolerance, max_minimization_iterations)
            print("Minimization done.")

        print("Starting heating ({} steps)...".format(number_of_equilibration_steps))
        for i in range(number_of_equilibration_steps//100):
            switch = ( i *100 ) /number_of_equilibration_steps
            temperature = ( 1 -switch) * start_temperature + switch * target_temperature
            integrator.setTemperature(temperature)
            # apply temperature to barostat
            if barostat is not None:
                equilibration.context.setParameter(barostat.Temperature(), temperature)
            equilibration.step(100)

        print("...Running {} steps at target temperature...".format(number_of_equilibration_steps))
        integrator.setTemperature(target_temperature)
        if barostat is not None:
            equilibration.context.setParameter(barostat.Temperature(), target_temperature)
        equilibration.step(number_of_equilibration_steps)
        print("Equilibration done.")

        equilibration.saveState("equilibrated.xml")
        simulation.loadState("equilibrated.xml")
