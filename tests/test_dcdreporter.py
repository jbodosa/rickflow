
from rflow.openmm import LangevinIntegrator
from rflow import RickFlow, abspath, CWD
import glob
import os
from rflow.openmm import unit as u
import mdtraj as md
import pytest

def test_rewrapping(tmpdir):
    with CWD(tmpdir):
        workflow = RickFlow(
            toppar=glob.glob(os.path.join(abspath("data/toppar"), '*')),
            psf=abspath("data/water.psf"),
            crd=abspath("data/water.crd"),
            box_dimensions=[25.1984,25.1984,25.1984],
            center_around=None,
            center_relative_position=0.0,
            center_dcd_at_origin=True,# CENTER OUTPUT AROUND ORIGIN
            dcd_output_interval=1,
            table_output_interval=1,
            steps_per_sequence=2,
            gpu_id="Reference"
        )
        workflow.prepare_simulation(LangevinIntegrator(300 * u.kelvin, 5. / u.picosecond, 1.0 * u.femtoseconds))
        workflow.run()
        traj = md.load_dcd("trj/dyn1.dcd", top=abspath("data/water.psf"))
        assert traj.xyz.mean() == pytest.approx(0.0, abs=0.05, rel=0.00)

