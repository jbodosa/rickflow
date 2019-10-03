
import os
import shutil
import pytest
import numpy as np

from simtk import unit as u

from simtk.openmm import LangevinIntegrator
from simtk.openmm.app import NoCutoff, StateDataReporter

from rflow.alchemyflow import AlchemyFlow
from rflow.reporters.alchemyreporter import FreeEnergyDifference
from rflow.utility import abspath, CWD

import mdtraj as md


def test_alchemyflow(tmpdir):
    with CWD(tmpdir):
        flow = AlchemyFlow(
            psf=abspath("data/m2a.psf"),
            crd=abspath("data/m2a.mp2_opt.cor"),
            toppar=[abspath("data/m2a.prm"), abspath("data/m2a.rtf")],
            box_dimensions=[300.0]*3,
            output_frequency=10,
            lambda_index=1,
            lambdas_elec=[0.0, 0.5, 1.0],
            lambdas_vdw=[0.0, 1.0, 1.0],
            gpu_id="Reference"
        )
        flow.solute_atoms = flow.select("resname M2A")
        flow.create_system(nonbondedMethod=NoCutoff, vdw_switching="openmm")
        temperature = 1000*u.kelvin
        integrator = LangevinIntegrator(temperature, 5./u.picosecond, 1.0*u.femtosecond)
        flow.create_simulation(integrator)
        flow.simulation.reporters.append(StateDataReporter("statedata.xml", 10, potentialEnergy=True))
        flow.initialize_state(False)
        flow.run(
            num_steps=1000,
            num_minimization_steps=0,
            num_high_pressure_steps=0,
            equilibration=0*u.femtosecond,
        )
        assert os.path.isfile(flow.energy_file)
        assert os.path.isfile(flow.dcd_file)
        assert os.path.isfile(flow.restart_file)
        traj = md.load_dcd(flow.dcd_file, top=flow.mdtraj_topology)
        energies = np.loadtxt(flow.energy_file)
        kT = u.BOLTZMANN_CONSTANT_kB * u.AVOGADRO_CONSTANT_NA * temperature
        potential = np.loadtxt("statedata.xml") / kT.value_in_unit(u.kilojoule_per_mole)
        assert traj.n_frames == 100
        assert energies.shape == (100, 3)
        assert potential.shape == (100,)
        recalculated = []
        for i in range(traj.n_frames):
            flow.context.setPositions(traj.xyz[i])
            recalculated.append(flow.context.getState(getEnergy=True).getPotentialEnergy()/kT)
        assert energies[:,1] == pytest.approx(potential, abs=1e-4)
        assert np.array(recalculated) == pytest.approx(potential, rel=None, abs=1e-2)


def test_free_energy_difference(tmpdir):
    shutil.copytree(abspath("./data/alchemydata/"), str(tmpdir/"alchemydata"))
    file_template = str(tmpdir/"alchemydata/ener.{}.txt")
    n_lambdas = 13
    delta_g = FreeEnergyDifference(file_template, n_lambdas)
    delta_g._save_summary_file()
    # saving and loading
    with open(delta_g.summary_file, "r") as fp:
        read1 = fp.read()
    delta_g._load_summary_file()
    delta_g._save_summary_file()
    with open(delta_g.summary_file, "r") as fp:
        read2 = fp.read()
    assert read1 == read2
    # properties
    assert delta_g.value == pytest.approx(delta_g.value_elec + delta_g.value_vdw)
    assert delta_g.error == pytest.approx(delta_g.error_elec + delta_g.error_vdw, rel=0.1, abs=None)
    assert delta_g.mbar_error == pytest.approx(delta_g.mbar_error_elec + delta_g.mbar_error_vdw, rel=0.3, abs=None)

