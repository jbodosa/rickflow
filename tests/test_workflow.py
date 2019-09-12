
import glob
import os
from rflow.workflow import PsfWorkflow
from rflow.utility import abspath, CWD
from simtk.openmm import LangevinIntegrator
from simtk import unit as u

def test_start_from_dcd(tmpdir):
    with CWD(tmpdir):
        workflow = PsfWorkflow(
            toppar=glob.glob(os.path.join(abspath("data/toppar"), '*')),
            psf=abspath("data/water.psf"),
            crd=abspath("data/water.dcd"),
            box_dimensions="some garbage that should not be read when starting from a dcd",
            center_around=None
        )
        assert workflow.positions is not None
        assert True
        print(workflow.topology.getPeriodicBoxVectors())
        workflow.create_system()
        workflow.create_simulation(LangevinIntegrator(300*u.kelvin, 5./u.picosecond, 1.0*u.femtoseconds), gpu_id=None)
        workflow.initialize_state()
        workflow.run(1)
