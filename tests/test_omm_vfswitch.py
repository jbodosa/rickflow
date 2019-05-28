
import warnings
from rflow.rickflow import RickFlow
from rflow.utility import abspath
from simtk.openmm import LangevinIntegrator
from simtk.openmm.app.internal.charmm.exceptions import CharmmPSFWarning
from glob import glob
import os


def test_vfswitch(tmpdir):
    os.mkdir(str(tmpdir) + "sim1")
    os.mkdir(str(tmpdir) + "sim2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=CharmmPSFWarning)
        rflow1 = RickFlow(
            toppar=glob(abspath("data/toppar/") +"/*"),
            psf=abspath("data/2dlpc.psf"),
            crd=abspath("data/rb2dlpc.crd"),
            box_dimensions=[104.314,  104.314, 75.3316],
            gpu_id=None, use_vdw_force_switch=False, work_dir=str(tmpdir) + "sim1"
        )
        rflow1.prepareSimulation(LangevinIntegrator(1,1,1))
        rflow2 = RickFlow(
            toppar=glob(abspath("data/toppar/") +"/*"),
            psf=abspath("data/2dlpc.psf"),
            crd=abspath("data/rb2dlpc.crd"),
            box_dimensions=[104.314,  104.314, 75.3316],
            gpu_id=None, use_vdw_force_switch=True, work_dir=str(tmpdir) + "sim2"
        )
        rflow2.prepareSimulation(LangevinIntegrator(1,1,1))
    num_forces = rflow1.system.getNumForces()
    num_forces_vfswitch = rflow2.system.getNumForces()
    assert num_forces_vfswitch == num_forces + 2
