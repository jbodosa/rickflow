#! /usr/bin/env python

import os

import numpy as np

import simtk.unit as u
from simtk.openmm.app import PME
from simtk.openmm import LangevinIntegrator

from rflow import RickFlow

#
#  ========= Setup System ==========
#
workflow = RickFlow(
    toppar=["top_all36_lipid.rtf", "par_all36_lipid.prm"],
    psf="hxdwat.psf",
    crd="hxdwat.crd",
    box_dimensions=[50,50,53.1975],
    gpu_id=0,
    nonbonded_method=PME,
    #tmp_output_dir=os.path.join("/lscratch", os.environ['SLURM_JOB_ID']),
    dcd_output_interval=1000,
    table_output_interval=1000,
    steps_per_sequence=1000000
)


#
#  ========= Define Integrator and Barostat ==========
#
temperature = 310.0 * u.kelvin
# integrator (The Nose-Hoover integrator in openmm does is currently not using the right
# number of degrees of freedom. The implementation in nosehoover.py provides a short-term fix,
# which requires the system to be passed to the constructor)
# from rflow.integrators import NoseHooverChainVelocityVerletIntegrator
# integrator = NoseHooverChainVelocityVerletIntegrator(
#        workflow.system, temperature, 50.0 / u.picosecond, 1.0 * u.femtosecond,
#        chain_length=3, num_mts=3, num_yoshidasuzuki=3
#)
integrator = LangevinIntegrator(temperature, 1.0 / u.picosecond, 1.0 * u.femtosecond)
# from simtk.openmm import MonteCarloAnisotropicBarostat
# barostat = MonteCarloAnisotropicBarostat(
#     u.Quantity(value=np.array([0.0, 0.0, 1.0]), unit=u.atmosphere), temperature,
#     False, False, True
# )

workflow.prepare_simulation(integrator)#, barostat)



if __name__ == "__main__":   
    #
    #  ========= Run Simulation  ==========
    #
    workflow.run()
