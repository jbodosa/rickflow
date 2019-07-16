# -*- coding: utf-8 -*-

"""Console script for rickflow."""

import os
import sys
import click

import rflow
from rflow import (TrajectoryIterator, Distribution, TransitionCounter, make_topology, center_of_mass_of_selection,
                   ModuliInput)
import numpy as np
import mdtraj as md


@click.group()
@click.version_option(version=rflow.__version__)
@click.pass_context
def main(ctx, args=None):
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
@click.option("-t", "--top", default="system.pdb",
              type=click.Path(exists=True), help="Topology file (pdb or psf).")
def select(selection, top):
    """
    Check the atom ids of a selection string SELECTION
    """
    topology = make_topology(top)
    atom_ids = topology.select(selection)
    print("Selection contains {} atoms.\n ids={}".format(len(atom_ids), atom_ids))


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

    ti = TrajectoryIterator(
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
            com.append(center_of_mass_of_selection(traj,membrane,2))

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


@main.group()
@click.option("-f", "--first-seq", type=int, help="First sequence of trajectory. (default: infer)", default=None)
@click.option("-l", "--last-seq", type=int, help="Number of sequences. (default: infer)", default=None)
@click.option("-t", "--filename-template", help="Filenames, use {} as a wildcard for ids. (default: trj/dyn{}.dcd)",
              default="trj/dyn{}.dcd")
@click.option("--top", type=str, help="Topology file. (default: system.pdb)", default="system.pdb")
@click.option("-s", "--selection", type=str, help="Operate on a subset of atoms. (default: all)", default="all")
@click.option("--infer-time/--no-infer-time", help="Infer time from trajectory frames. (default: True)", default=True)
@click.option("--dt", type=float, help="Time between frames in picoseconds.", default=1.0)
@click.option("-n", "--nframes", type=int, help="Number of frames per trajectory (default: auto-detection)",
              default=None)
@click.option("--format", help="Trajectory format (default: infer from filename)", default=None)
@click.pass_context
def traj(ctx, first_seq, last_seq, filename_template, top, selection, infer_time, dt, nframes, format):
    "Trajectory analysis module."
    trajectories = TrajectoryIterator(
        first_sequence=first_seq, last_sequence=last_seq, filename_template=filename_template, atom_selection=selection,
        infer_time=infer_time, load_function=format, topology_file=top, time_between_frames=dt,
        num_frames_per_trajectory=nframes
    )
    click.echo(f"Found {len(trajectories)} trajectory sequence{'s' if  len(trajectories) != 1 else ''}.")
    ctx.obj["traj"] = trajectories


@traj.command()
@click.option("-h", "--head", type=str, multiple=True, help="Selection string for head atoms.")
@click.option("-t", "--tail", type=str, multiple=True, help="Selection string for tail atoms.")
@click.option("--whead", default=None, type=float, multiple=True, help="Weights for head atoms.")
@click.option("--wtail", default=None, type=float, multiple=True, help="Weights for tail atoms.")
@click.option("--bilayer_normal", type=str, default='z', help="x,y, or z")
@click.option("-b", "--box-prefix", type=str, default="boxsize", help="Filename prefix for boxsize (default: boxsize).")
@click.option("-l", "--lipid-prefix", type=str, default="Lipid", help="Filename prefix for coordinates (default: boxsize).")
@click.option("--append/--no-append", default=False, help="Whether output should be appended to existing files.")
@click.pass_context
def moduli_input(ctx, head, tail, whead, wtail, bilayer_normal, box_prefix="boxsize", lipid_prefix="Lipid", append=False):
    """
    Generate input files for bilayer moduli analysis.

    Example:
    >>> rflow traj -t tests/data/ord2.dcd --top=tests/data/ord+o2.psf
            moduli-input "resname DPPC and name C2"
            -t "resname DPPC and name C216"
            -t "resname DPPC and name C316"

    """
    trajectories = ctx.obj["traj"]
    head_atoms = []
    tail_atoms = []
    for h in head:
        selected = trajectories.select(h)
        if len(selected) == 0:
            click.echo(f"Error: Selection '{h}' empty.")
            sys.exit(1)
        head_atoms.append(selected)
    for t in tail:
        selected = trajectories.select(t)
        if len(selected) == 0:
            click.echo(f"Error: Selection '{t}' empty.")
            sys.exit(1)
        tail_atoms.append(selected)
    if len(whead) == 0: whead = None
    if len(wtail) == 0: wtail = None
    moduli_input = ModuliInput(
        head_atoms=head_atoms, tail_atoms=tail_atoms, head_weights=whead, tail_weights=wtail,
        bilayer_normal=bilayer_normal, boxsizefile_prefix=box_prefix, lipidfile_prefix=lipid_prefix,
        append=append
    )
    click.echo(f"Will save results to {box_prefix}[XYZ].out and {lipid_prefix}[XYZ].out")
    for traj in trajectories:
        print(f"Sequence {traj.i}/{trajectories.last}")
        moduli_input(traj)





def entrypoint():
    sys.exit(main(obj={}))

