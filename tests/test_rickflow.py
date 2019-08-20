#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `rickflow` module."""

from rflow import RickFlow, CWD, TrajectoryIterator, RickFlowException
from rflow.utility import abspath
import glob
import shutil
import os
import subprocess
import pytest
import numpy as np
from simtk import unit as u
from simtk.openmm import LangevinIntegrator, MonteCarloBarostat


@pytest.fixture(scope="module")
def rickflow_instance(tmpdir_factory):
    return RickFlow(
        toppar=glob.glob(os.path.join(abspath("data/toppar"), '*')),
        psf=abspath("data/2dlpc.psf"),
        crd=abspath("data/rb2dlpc.crd"),
        box_dimensions=[47.7695166, 47.7695166, 137.142387],
        gpu_id=None,
        work_dir=str(tmpdir_factory.mktemp('rflow')),
        misc_psf_create_system_kwargs={"constraints": None}
    )


@pytest.fixture(scope="module", params=[True, False])
def run_and_restart(tmpdir_factory, request):
    tmpdir = tmpdir_factory.mktemp('run_and_restart_barostat_'+str(request.param))
    rf = RickFlow(
        toppar=glob.glob(os.path.join(abspath("data/toppar"), '*')),
        psf=abspath("data/water.psf"),
        crd=abspath("data/water.crd"),
        box_dimensions=[25.1984]*3,
        gpu_id=None,
        steps_per_sequence=20,
        table_output_interval=10,
        dcd_output_interval=10,
        center_around=None,
        work_dir=str(tmpdir)
    )
    barostat = MonteCarloBarostat(1.0*u.atmosphere, 200.0*u.kelvin, 25) if request.param else None
    rf.prepare_simulation(
        integrator=LangevinIntegrator(200.*u.kelvin, 5.0/u.picosecond, 1.0*u.femtosecond),
        barostat=barostat
    )
    rf.run()

    rf = RickFlow(
        toppar=glob.glob(os.path.join(abspath("data/toppar"), '*')),
        psf=abspath("data/water.psf"),
        crd=abspath("data/water.crd"),
        box_dimensions=[25.1984]*3,
        gpu_id=None,
        steps_per_sequence=20,
        table_output_interval=10,
        dcd_output_interval=10,
        center_around=None,
        work_dir=str(tmpdir),
        use_only_xml_restarts=True
    )
    barostat = MonteCarloBarostat(1.0*u.atmosphere, 200.0*u.kelvin, 25) if request.param else None
    rf.prepare_simulation(
        integrator=LangevinIntegrator(200.*u.kelvin, 5.0/u.picosecond, 1.0*u.femtosecond ),
        barostat=barostat
    )
    rf.run()
    return rf


def test_run_and_restart(run_and_restart):
    assert os.path.isfile(os.path.join(run_and_restart.work_dir, "out", "out1.txt"))
    assert os.path.isfile(os.path.join(run_and_restart.work_dir, "trj", "dyn1.dcd"))
    assert os.path.isfile(os.path.join(run_and_restart.work_dir, "out", "out2.txt"))
    assert os.path.isfile(os.path.join(run_and_restart.work_dir, "trj", "dyn2.dcd"))
    # check dcd file sanity
    with CWD(run_and_restart.work_dir):
        trajectories = TrajectoryIterator(time_between_frames=0.01)
        trajectory = trajectories[2]
    assert np.all(trajectory.time == np.array([0.02, 0.03]))


@pytest.mark.skipif(True, reason="requires a working CHARMM installation 'c41n1'")
def test_charmm_postprocessing(run_and_restart):
    shutil.copytree(abspath("data/toppar"), os.path.join(run_and_restart.work_dir, "toppar"))
    shutil.copy(abspath("data/edges+area.inp"), run_and_restart.work_dir)
    shutil.copy(abspath("data/water.crd"), run_and_restart.work_dir)
    shutil.copy(abspath("data/water.psf"), run_and_restart.work_dir)
    print("workdir:", run_and_restart.work_dir)
    with CWD(run_and_restart.work_dir):
        with open("edges+area.inp", "r") as fin:
            with open("edges+area.out", 'w') as fout:
                rc = subprocess.call(["c41b1"], stdin=fin, stdout=fout, stderr=fout)
                assert rc == 0


def test_select(rickflow_instance):
    assert len(rickflow_instance.select("resname DLPC")) > 0


def test_directory_structure(rickflow_instance):
    assert os.path.exists(os.path.join(rickflow_instance.work_dir, "out"))
    assert os.path.exists(os.path.join(rickflow_instance.work_dir, "trj"))
    assert os.path.exists(os.path.join(rickflow_instance.work_dir, "res"))


def test_forces(rickflow_instance):
    for force in rickflow_instance.system.getForces():
        if hasattr(force, "getUseDispersionCorrection"):
            assert force.getUseDispersionCorrection() is False
        if hasattr(force, "getUseLongRangeCorrection"):
            assert force.getUseLongRangeCorrection() is False


def test_analysis_mode(tmpdir):
    work_dir = str(tmpdir)
    with CWD(work_dir):
        flow = RickFlow(
            toppar=glob.glob(os.path.join(abspath("data/toppar"), '*')),
            psf=abspath("data/2dlpc.psf"),
            crd=abspath("data/rb2dlpc.crd"),
            box_dimensions=[47.7695166, 47.7695166, 137.142387],
            work_dir=".",
            misc_psf_create_system_kwargs={"constraints": None},
            analysis_mode=True
        )
    flow.prepare_simulation(LangevinIntegrator(1, 1, 1))
    with pytest.raises(RickFlowException):
        flow.run()
    assert os.listdir(work_dir) == []

