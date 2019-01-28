# -*- coding: utf-8 -*-

"""
Translating Rick Venable's simulation workflow from CHARMM to OpenMM.
"""

import os
import glob
import numpy as np
import shutil

import simtk.unit as u
from simtk.openmm import Platform
from simtk.openmm import CustomNonbondedForce, NonbondedForce
from simtk.openmm.app import Simulation
from simtk.openmm.app import CharmmPsfFile, CharmmParameterSet, CharmmCrdFile
from simtk.openmm.app import LJPME, PME, HBonds
from simtk.openmm.app import DCDReporter, StateDataReporter, PDBReporter

import mdtraj as md

from rflow.exceptions import LastSequenceReached, NoCuda, RickFlowException
from rflow.utility import CWD


def get_next_seqno_and_checkpoints(work_dir="."):
    """
    Sets up the directory structure and reads the id of the next sequence
    to be simulated.

    Returns:
        A triple (next_seqno, checkpoint_file, state_file):

         - int: number of the next sequence
         - string: name of the checkpoint file to be read from
           (None, if next_seqno=1)
         - string: name of the state file to be read from
           (None, if next_seqno=1)

    Raises:
          LastSequenceReached: If the number in the file next.seqno is larger
          than the number in the file last.seqno
    """
    with CWD(work_dir):
        if not os.path.isdir("trj"):  # directory for trajectories
            os.mkdir("trj")
        if not os.path.isdir("out"):  # directory for state files
            os.mkdir("out")
        if not os.path.isdir("res"):  # directory for restart files
            os.mkdir("res")

        # a file containing the id of the next sequence to be simulated
        if not os.path.isfile("next.seqno"):
            with open("next.seqno", 'w') as fp:
                fp.write("1")
            next_seqno = 1
        else:
            with open("next.seqno", 'r') as fp:
                next_seqno = int(fp.read())

        # a file containing the id of the last sequence to be simulated
        if os.path.isfile("last.seqno"):
            with open("last.seqno", 'r') as fp:
                last_seqno = int(fp.read())
                if next_seqno > last_seqno:
                    raise LastSequenceReached(last_seqno)

        current_checkpoint_file = None
        current_state_file = None
        if next_seqno > 1:
            current_checkpoint_file = "res/checkpoint{}.chk".format(next_seqno - 1)
            current_state_file = "res/state{}.xml".format(next_seqno - 1)

    return next_seqno, current_checkpoint_file, current_state_file


def require_cuda(gpu_id=None, precision="mixed"):
    """
    Require CUDA to be used for the simulation.

    Args:
        gpu_id (int): The id of the GPU to be used.
        precision (str): 'mixed', 'double', or 'single'

    Returns:
        A pair (platform, properties):

         - OpenMM Platform object: The platform to be passed to the simulation.
         - dict: A dictionary to be passed to the simulation.

    Raises:
        NoCuda: If CUDA is not present.
    """
    try:
        assert "LD_LIBRARY_PATH" in os.environ
        assert 'cuda' in os.environ["LD_LIBRARY_PATH"].lower()
        my_platform = Platform.getPlatformByName('CUDA')
    except Exception as e:
        raise NoCuda(e)
    if gpu_id is not None:
        my_properties = {'DeviceIndex': str(gpu_id),
                         'Precision': precision
                         }
    return my_platform, my_properties


class RickFlow(object):
    """
    Runs a simulation in OpenMM in sequences.
    """

    def __init__(self, toppar, psf, crd,
                 box_dimensions, gpu_id=0,
                 nonbonded_method=PME,
                 recenter_coordinates=True,
                 switch_distance=1.0*u.nanometer,
                 work_dir=".",
                 tmp_output_dir=None
                 ):
        """
        The constructor sets up the system.

        Args:
            toppar (list): Filenames of parameter, rtf and stream files that define the parameters.
            psf (str): Charmm topology (.psf file).
            crd (str): Initial coordinates (.crd file).
            box_dimensions (list): Box dimensions in Angstrom.
            gpu_id (int or None): The device id of the CUDA-compatible GPU to be used. If None,
                the use of CUDA is not enforced and OpenMM may run on a slower platform.
                Note that slurm scheduler is smart enough to map the gpu_id 0
                to the first GPU that was allocated for the current job. Therefore multiple slurm jobs
                with `gpu_id=0` can be run on the same node and each will have its own GPU.
            nonbonded_method (OpenMM object): openmm.app.PME for cutoff-LJ, openmm.app.LJPME for LJ-PME
            recenter_coordinates (bool): If True, recenter initial coordinates around center of mass
                of non-water molecules.
            switch_distance (simtk.unit): Switching distance for LJ potential.
            work_dir (str): The working directory.
            tmp_output_dir (str): A temporary directory for output during simulations.
        """

        self.work_dir = work_dir
        self.next_seqno, self.current_checkpoint, self.current_state = (
            get_next_seqno_and_checkpoints(self.work_dir)
        )
        # prepare temporary output directory
        if tmp_output_dir is not None:
            assert os.path.exists(tmp_output_dir)
            self.tmp_output_dir = os.path.normpath(tmp_output_dir)
            with CWD(tmp_output_dir):
                if not os.path.isdir("trj"):  # directory for trajectories
                    os.mkdir("trj")
                if not os.path.isdir("out"):  # directory for state files
                    os.mkdir("out")
                if not os.path.isdir("res"):  # directory for restart files
                    os.mkdir("res")
        else:
            self.tmp_output_dir = None
        if gpu_id is not None:
            self.platform, self.platform_properties = require_cuda(gpu_id)
        else:
            self.platform = None
            self.platform_properties = None

        with CWD(work_dir):
            self.parameters = CharmmParameterSet(*toppar)
            self.psf = CharmmPsfFile(psf)
            box_dimensions = [dim * u.angstrom for dim in box_dimensions]
            self.psf.setBox(*box_dimensions)
            self.crd = CharmmCrdFile(crd)
        # create system
        self._system = self.psf.createSystem(
            self.parameters,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=1.2 * u.nanometer,
            constraints=HBonds,
            switchDistance=switch_distance
        )
        # translate system so that the center of mass of non-waters is in the middle
        if recenter_coordinates:
            non_waters = [i for i in range(self.crd.natom)
                          if "TIP" not in self.crd.resname[i]
                          ]
            current_com = self.centerOfMass(non_waters)
            target_com = 0.5 * self.psf.boxLengths
            move = target_com - current_com
            self.crd.positions += move
        # save positions as list, so we can append virtual sites, if needed
        self.positions = self.crd.positions.value_in_unit(u.angstrom).tolist()
        # manually remove long-range correction from the nonbonded force
        # (that will hopefully get fixed in CharmmPsfFile.createSystem
        # # at some point)
        for force in self.system.getForces():
            if isinstance(force, NonbondedForce):
                force.setUseDispersionCorrection(False)
            if isinstance(force, CustomNonbondedForce):
                force.setUseLongRangeCorrection(False)
        # set up misc fields
        self.context = None
        self.simulation = None

    @property
    def system(self):
        return self._system

    def centerOfMass(self, particle_ids):
        """
        Calculate the center of mass of a subset of atoms.

        Args:
            particle_ids (list of int): The particle ids that define the subset of the system

        Returns:
            float: center of mass in Angstrom
        """
        masses = np.array(
            [self.system.getParticleMass(i).value_in_unit(u.dalton) for i in
             range(self.crd.natom)])
        positions = np.array(self.crd.positions.value_in_unit(u.angstrom))
        return np.sum(
            positions[particle_ids].transpose()
            * masses[particle_ids],
            axis=1
        ) / np.sum(masses[particle_ids]) * u.angstrom

    def prepareSimulation(self, integrator, barostat=None):
        """
        Initialize simulation object by passing an integrator and a barostat.

        Args:
            integrator (OpenMM integrator object): The integrator to be used.
            barostat (OpenMM barostat object): The barostat. Pass None for NVT.
        """
        with CWD(self.work_dir):
            if barostat:
                self.system.addForce(barostat)
            self.simulation = Simulation(self.psf.topology, self.system,
                                         integrator, self.platform,
                                         self.platform_properties)
            self.context = self.simulation.context
            self._initializeSimulation()
            # write the system as a pdb file (this is important for postprocessing,
            # if virtual sites were manually added to the system)
            PDBReporter("system.pdb", 1).report(
                self.simulation, self.context.getState(getPositions=True)
            )
            print("#Running on ", self.context.getPlatform().getName())

    def _initializeSimulation(self):
        """
        Initialize state, use checkpoint for seqno > 1.
        """
        if self.next_seqno == 1:
            print("Setting up from initial coordinates.")
            if self.psf.topology.getPeriodicBoxVectors():
                self.context.setPeriodicBoxVectors(
                    *self.psf.topology.getPeriodicBoxVectors())
            self.context.setPositions(
                u.Quantity(
                    value=np.array(self.positions),
                    unit=u.angstrom
                )
            )
        else:
            try:
                print("Attempting restart...")
                self.simulation.loadCheckpoint(self.current_checkpoint)
                print("Restarting from checkpoint file {}.".format(
                    self.current_checkpoint))
            except:
                print("    ...could not read from checkpoint file...")
                self.simulation.loadState(self.current_state)
                print("Restarting from state file {}.".format(
                    self.current_state))
            # read time and timestep
            last_out = np.loadtxt("out/out{}.txt".format(self.next_seqno - 1),
                                  delimiter=",")
            last_step = int(last_out[-1][0])
            last_time = round(last_out[-1][1]) * u.picosecond
            print("Restart at step {}. Time = {}.".format(last_step, last_time))
            self.simulation.currentStep = last_step
            self.context.setTime(last_time)
        self.context.applyConstraints(1e-7)

    def run(self):
        """
        Run the simulation.
        """
        # define output for the current run
        output_dir = self.work_dir if self.tmp_output_dir is None else self.tmp_output_dir
        with CWD(output_dir):
            self.simulation.reporters.clear()
            self.simulation.reporters.append(
                DCDReporter("trj/dyn{}.dcd".format(self.next_seqno), 1000))
            self.simulation.reporters.append(
                StateDataReporter("out/out{}.txt".format(self.next_seqno), 1000,
                                  step=True, time=True,
                                  potentialEnergy=True, temperature=True,
                                  volume=True, density=True, speed=True))

            # run one sequence
            duration = 1. * u.nanosecond
            n_steps = round(duration / (1. * u.femtosecond))
            print("Starting {} steps".format(n_steps))

            self.simulation.step(n_steps)

            # save checkpoints
            self.simulation.saveState("res/state{}.xml".format(self.next_seqno))
            with open("res/checkpoint{}.chk".format(self.next_seqno), 'wb') as f:
                f.write(self.simulation.context.createCheckpoint())

            self.next_seqno += 1
            with open("next.seqno", 'w') as fp:
                fp.write(str(self.next_seqno))

        # if required, copy temporary files over
        if self.tmp_output_dir is not None:
            with CWD(self.work_dir):
                shutil.copy(os.path.join(self.tmp_output_dir, "trj/dyn{}.dcd".format(self.next_seqno)), "trj")
                shutil.copy(os.path.join(self.tmp_output_dir, "out/out{}.dcd".format(self.next_seqno)), "out")
                shutil.copy(os.path.join(self.tmp_output_dir, "res/state{}.xml".format(self.next_seqno)), "res")
                shutil.copy(os.path.join(self.tmp_output_dir, "res/checkpoint{}.chk".format(self.next_seqno)), "res")
                shutil.copy(os.path.join(self.tmp_output_dir, "next.seqno".format(self.next_seqno)), ".")


