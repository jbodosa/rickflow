
import os
import glob

import numpy as np
import pytest

from simtk import unit as u
from simtk.openmm import LangevinIntegrator
from rflow.utility import CWD, abspath
from rflow.rickflow import RickFlow
from rflow.equilibration import equilibrate


def test_equilibrate(tmpdir):
    for do_equilibration in [True, False]:
        work_dir = os.path.join(str(tmpdir), str(do_equilibration))
        os.mkdir(work_dir)
        with CWD(work_dir):
            flow = RickFlow(
                toppar=glob.glob(os.path.join(abspath("data/toppar"), '*')),
                psf=abspath("data/water.psf"),
                crd=abspath("data/water.crd"),
                box_dimensions=[25.1984] * 3,
                gpu_id=None,
                steps_per_sequence=100,
                table_output_interval=10,
                dcd_output_interval=100,
                center_around=None
            )
            # compromise the particle positions to render the simulation unstable
            flow.positions = (np.array(flow.positions) * 0.5).tolist()
            flow.prepareSimulation(integrator=LangevinIntegrator(300*u.kelvin, 5.0/u.picosecond, 1.0*u.femtosecond))
            if do_equilibration:
                equilibrate(
                    flow.simulation,
                    300.0*u.kelvin,
                    gpu_id=None,
                    equilibration=300*u.femtosecond,
                    num_high_pressure_steps=50,
                    num_minimization_steps=100,
                    work_dir=str(tmpdir)
                )
                flow.run()
            else:
                ## check that the simulation would fail without equilibration
                with pytest.raises(Exception): # could be a ValueError or an OpenMMException
                    flow.run()
