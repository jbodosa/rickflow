# -*- coding: utf-8 -*-

"""Console script for rickflow."""

import os
import sys
import click

from rflow import CharmmTrajectoryIterator, Distribution, TransitionCounter
import mdtraj as md
import numpy as np

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


@main.command()
@click.option("-p", "--permeant", type=str, help="Permeant selection string")
@click.option("-f", "--first_seq", type=int, help="First sequence of trajectory", default=1)
@click.option("-m", "--membrane", type=str, help="Membrane selection string for com removal", default=None)
@click.option("-l", "--length", type=int, help="Number of sequences")
@click.option("-n", "--nbins", type=int, help="Number of bins")
@click.option("-o", "--outdir", type=str, help="Directory for the output files.")
def tmat(permeant, first_seq, membrane=None,
         length=100, lag_iterations=[10, 20, 30, 40, 50, 60],
         nbins=100, outdir="."):
    """
    Extract transition matrices and distributions from simulations generated with the rickflow workflow.
    """
    try:
        permeant = eval(permeant)
        assert isinstance(permeant, list)
    except:
        pass

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    ti = CharmmTrajectoryIterator(
        first_sequence=first_seq, last_sequence=first_seq + length - 1)

    distribution = Distribution(atom_selection=permeant, coordinate=2)
    distribution_with_com = Distribution(permeant, coordinate=2, com_selection=membrane)
    counter = TransitionCounter(num_bins=nbins, solute=permeant,
                                lag_iterations=lag_iterations,
                                membrane=membrane
                                )
    com = []
    for traj in ti:
        print("\r", "from ", first_seq, ":", traj.i, end="")
        distribution(traj)
        distribution_with_com(traj)
        counter(traj)
        if membrane is not None:
            com.append(md.compute_center_of_mass(traj.atom_slice(membrane)))

    # save transition matrices
    counter.save_matrices(os.path.join(
        outdir, "tmat.first{}.len{}.lag{{}}.nbins{}.txt".format(
            first_seq, length, nbins)))
    # save distributions
    np.savetxt(os.path.join(
        outdir, "distribution.first{}.len{}.nbins{}.txt".format(
            first_seq, length, nbins), distribution)
    )
    np.savetxt(os.path.join(
        outdir, "distribution_com.first{}.len{}.nbins{}.txt".format(
            first_seq, length, nbins), distribution_with_com)
    )
    np.savetxt(os.path.join(
        outdir, "com.first{}.len{}".format(first_seq, length)
    ))


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
