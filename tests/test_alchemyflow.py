

from simtk import unit as u

from simtk.openmm import LangevinIntegrator
from simtk.openmm.app import NoCutoff

from rflow.alchemyflow import AlchemyFlow
from rflow.utility import abspath

import mdtraj as md


def test_alchemyflow(tmpdir):
    flow = AlchemyFlow(
        psf=abspath("data/m2a.psf"),
        crd=abspath("data/m2a.mp2_opt.crd"),
        toppar=[abspath("data/m2a.rtf"), abspath("data/m2a.prm")],
        box_dimensions=[30.0]*3,
        steps=100,
        dcd_output_interval=50,
        table_output_interval=50,
        lambda_index=0,
        lambdas_elec=[0.0, 0.5, 1.0],
        lambdas_vdw=[0.0, 0.5, 1.0],
        work_dir=str(tmpdir),
        gpu_id=None,
        use_vdw_force_switch=False,
        misc_psf_create_system_kwargs={"nonbondedMethod": NoCutoff}
    )
    flow.solute_atoms = flow.select("resname M2A")
    integrator = LangevinIntegrator(200.*u.kelvin, 5./u.picosecond, 1.0*u.femtosecond)
    flow.prepare_simulation(integrator)
    flow.run(
        num_minimization_steps=10,
        num_high_pressure_steps=10,
        equilibration=10*u.femtosecond
    )


