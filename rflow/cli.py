# -*- coding: utf-8 -*-

"""Console script for rickflow."""

import os
import sys
import click

from rflow import CharmmTrajectoryIterator

@click.group()
def main(args=None):
    """Console script for rickflow."""
    click.echo("Rickflow: a python package to facilitate running jobs in OpenMM using CHARMM defaults.")
    return 0


@main.command()
@click.option("-t", "--template", type=str, help="Template for batch submission script.")
def create(template):
    """
    Setup a simulation. Create template dyn.py and sdyn.sh files.
    """
    dynpy = """
#! /usr/bin/env python

import simtk.unit as u
from simtk.openmm.app import PME, LJPME
from simtk.openmm import MonteCarloBarostat, MonteCarloAnisotropicBarostat, MonteCarloMembraneBarostat
from simtk.openmm import DrudeLangevinIntegrator, LangevinIntegrator
from rickflow import RickFlow, NoseHooverChainVelocityVerletIntegrator


    """

    dynpy += """
#
#  ========= Setup System ==========
#
workflow = RickFlow(
    toppar=["top_all36_lipid.rtf", "par_all36_lipid.prm"],
    psf="water+o2.psf",
    crd="water+o2.crd",
    box_dimensions=[54.3114178]*3,
    gpu_id=0,
    nonbonded_method=PME,
)


    """

    if not os.path.isfile("dyn.py"):
        with open("dyn.py", 'w') as fp:
            fp.write(dynpy)


@main.command()
@click.argument("batch")
def submit(batch):
    """
    Submit the workflow using a batch script.
    """
    assert os.path.isfile(batch)
    cwd = os.path.basename(os.getcwd())
    os.system("sbatch -o {}-%j.log -J {} {}".format(cwd, cwd, submit_script))


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
