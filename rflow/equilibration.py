
import os

from simtk import unit as u
from simtk.openmm.app import Simulation, StateDataReporter
from simtk.openmm import (LangevinIntegrator, DrudeForce, DrudeLangevinIntegrator)

from rflow.exceptions import RickFlowException
from rflow.utility import get_barostat, get_force, CWD, get_platform


def equilibrate(
        simulation,
        target_temperature=None,
        minimize=True,
        num_minimization_steps=0,
        num_high_pressure_steps=0,
        start_temperature=200.0 * u.kelvin,
        equilibration=500.*u.picosecond,
        time_step=1.0*u.femtosecond,
        gpu_id=0,
        out_file="equilibration.txt",
        restart_file="equilibrated.xml",
        work_dir="."
    ):
    """
    Energy minimization and gradual heating to get a stable starting configuration.

    Args:
        simulation: A simulation object, whose context is already
            initialize with periodic box vectors and particle positions.
        target_temperature: If None, get the temperature from the simulation's integrator.
        minimize (bool): Whether to perform a minimization
        num_minimization_steps:
        num_high_pressure_steps:
        equilibration:
        start_temperature:
        gpu_id:
        out_file: If None, don't write.
        restart_file: If None, don't write.
        work_dir:
        time_step:

    Returns:

    """
    if simulation is None:
        raise RickFlowException("equilibrate can only be called after preparing the simulation "
                                "(WorkFlow.prepareSimulation)")
    if target_temperature is None:
        target_temperature = simulation.context.getIntegrator().getTemperature()
    barostat = get_barostat(simulation.system)
    num_equilibration_steps = int(equilibration/time_step)  # two equilibration phases
    isdrude = get_force(simulation.system, [DrudeForce]) is not None
    with CWD(work_dir):
        # set up simulation
        if isdrude:
            print("Found Drude Force.")
            integrator = DrudeLangevinIntegrator(
                start_temperature,
                5.0 / u.picosecond,
                1.0 * u.kelvin,
                20.0 / u.picosecond,
                time_step
            )
        else:
            integrator = LangevinIntegrator(start_temperature, 5.0 / u.picosecond, time_step)
        if gpu_id is not None:
            platform, platform_properties = get_platform(gpu_id=gpu_id)
        else:
            platform = None
            platform_properties = None
        equilibration = Simulation(simulation.topology, simulation.system, integrator, platform, platform_properties)
        state = simulation.context.getState(getPositions=True)
        equilibration.context.setPositions(state.getPositions())
        equilibration.context.setPeriodicBoxVectors(*list(state.getPeriodicBoxVectors()))
        equilibration.context.setVelocities((state.getPositions(asNumpy=True)*0).tolist())

        if out_file is not None:
            equilibration.reporters.append(StateDataReporter(
                os.path.join(work_dir, out_file), 100, step=True, time=True,
                potentialEnergy=True, temperature=True,
                volume=True, density=True, speed=True)
            )

        print("Equilibration:")
        # Phase 0: Minimization
        if minimize:
            print("...Energy Minimization...")
            equilibration.minimizeEnergy(maxIterations=num_minimization_steps)

        # Phase 1: Equilibration at low temperature and high pressure to prevent blow-ups
        target_pressure = barostat.getDefaultPressure() if barostat is not None else None
        if num_high_pressure_steps:
            print("...Starting high-pressure equilibration ({} steps)...".format(num_high_pressure_steps))
            if barostat is not None:
                barostat.setDefaultTemperature(start_temperature)
                barostat.setDefaultPressure(1000.0 * u.atmosphere)
                equilibration.context.setParameter(barostat.Temperature(), start_temperature)
                equilibration.context.setParameter(barostat.Pressure(), 1000.0 * u.atmosphere)
            integrator.setTemperature(start_temperature)
            equilibration.step(num_high_pressure_steps)

        # Phase 2: Gradual heating to target temperature at target pressure
        print("...Starting heating ({} steps)...".format(num_equilibration_steps))
        if barostat is not None:
            barostat.setDefaultPressure(target_pressure)
            equilibration.context.setParameter(barostat.Pressure(), target_pressure)
        for i in range(num_equilibration_steps//100):
            switch = (i*100)/num_equilibration_steps
            temperature = (1 - switch) * start_temperature + switch * target_temperature
            integrator.setTemperature(temperature)
            # apply temperature to barostat
            if barostat is not None:
                barostat.setDefaultTemperature(temperature)
                equilibration.context.setParameter(barostat.Temperature(), temperature)
            equilibration.step(100)

        # Phase 3: Run Langevin Dynamics at target temperature and target pressure
        print("...Running {} steps at target temperature...".format(num_equilibration_steps))
        integrator.setTemperature(target_temperature)
        if barostat is not None:
            equilibration.context.setParameter(barostat.Temperature(), target_temperature)
            barostat.setDefaultTemperature(temperature)
        equilibration.step(num_equilibration_steps)
        print("Equilibration done.")
        if restart_file is not None:
            print(f"Writing state to {restart_file}.")
            equilibration.saveState(os.path.join(work_dir, restart_file))

        state = equilibration.context.getState(getPositions=True, getVelocities=True)
        simulation.context.setPositions(state.getPositions())
        simulation.context.setPeriodicBoxVectors(*list(state.getPeriodicBoxVectors()))
        simulation.context.setVelocities(state.getVelocities())

