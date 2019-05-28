# -*- coding: utf-8 -*-

"""Console script for rickflow."""

import os
import sys
import click

import rflow
from rflow import CharmmTrajectoryIterator, Distribution, TransitionCounter, make_topology
import mdtraj as md
import numpy as np

@click.group()
@click.version_option(version=rflow.__version__)
def main(args=None):
    """Console script for rickflow."""
    click.echo("Rickflow: a python package to facilitate running and analyzing jobs in OpenMM using CHARMM defaults.")
    return 0


@main.command()
@click.option("-t", "--template", type=str, help="Template for batch submission script.")
def create(template):
    """
    Setup a simulation. Create template dyn.py and sdyn.sh files.
    """
    raise NotImplementedError
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
@click.argument("batch", type=click.Path(exists=True))
def submit(batch):
    """
    Submit the workflow using a batch script.
    """
    if not os.path.isdir("log"):  # directory for slurm output files
        os.mkdir("log")
    assert os.path.isfile(batch)
    cwd = os.path.basename(os.getcwd())
    os.system("sbatch -o log/{}-%j.log -J {} {}".format(cwd, cwd, batch))


@main.command()
@click.argument("selection", type=str)
@click.option("-t", "topology_file", default="system.pdb",
              type=click.Path(exists=True), help="Topology file (pdb or psf).")
def select(selection, topology_file):
    """
    Check the atom ids of a selection string SELECTION
    """
    topology = make_topology(topology_file)
    atom_ids = topology.select(selection)
    print("Selection contains {} atoms.\n ids={}".format(len(atom_ids),atom_ids))


@main.command()
def count_crossings():
    pass

@main.command()
@click.option("-p", "--permeant", type=str, help="Permeant selection string")
@click.option("-f", "--first_seq", type=int, help="First sequence of trajectory", default=1)
@click.option("-m", "--membrane", type=str, help="Membrane selection string for com removal", default=None)
@click.option("-l", "--length", type=int, help="Number of sequences", default=100)
@click.option("-n", "--nbins", type=int, help="Number of bins", default=100)
@click.option("-o", "--outdir", type=str, help="Directory for the output files.", default=".")
@click.option("-t", "--lag_iterations", type=int, multiple=True, default=[10, 20, 30, 40, 50, 60])
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
    to_save1 = np.array([distribution.bin_centers_around_zero, distribution.probability]).transpose()
    to_save2 = np.array([distribution_with_com.bin_centers_around_zero, distribution_with_com.probability]).transpose()
    np.savetxt(os.path.join(
        outdir, "distribution.first{}.len{}.nbins{}.txt".format(
            first_seq, length, nbins)), to_save1)
    np.savetxt(os.path.join(
        outdir, "distribution_com.first{}.len{}.nbins{}.txt".format(
            first_seq, length, nbins)), to_save2)
    if com:
        np.savetxt(os.path.join(
            outdir, "com.first{}.len{}".format(first_seq, length)), com)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
