# -*- coding: utf-8 -*-

"""
Translating Rick Venable's simulation workflow from CHARMM to OpenMM.
"""

import os
import numpy as np
import shutil

import simtk.unit as u
from simtk.openmm.app import DCDFile, StateDataReporter

from rflow.exceptions import LastSequenceReached, RickFlowException
from rflow.utility import CWD, read_input_coordinates, disable_long_range_correction
from rflow.workflow import Workflow


class RickFlow(Workflow):
    """
    Runs a simulation in OpenMM in sequences.
    """
    def __init__(
            self,
            *args,
            analysis_mode=False,
            steps_per_sequence=1000000,
            **kwargs
    ):
        """
        The constructor sets up the system.

        Args:
            toppar (list): Filenames of parameter, rtf and stream files that define the parameters.
            psf (str): Charmm topology (.psf file).
            crd (str): Initial coordinates (coordinate or trajectory file). If the trajectory contains multiple frames,
                the positions are initialized from the last frame.
            box_dimensions (list): Box dimensions in Angstrom.
            gpu_id (int or None): The device id of the CUDA-compatible GPU to be used. If None,
                the use of CUDA is not enforced and OpenMM may run on a slower platform.
                Note that slurm scheduler is smart enough to map the gpu_id 0
                to the first GPU that was allocated for the current job. Therefore multiple slurm jobs
                with `gpu_id=0` can be run on the same node and each will have its own GPU.
            nonbonded_method (OpenMM object): openmm.app.PME for cutoff-LJ, openmm.app.LJPME for LJ-PME
            switch_distance (simtk.unit): Switching distance for LJ potential.
            cutoff_distance (simtk.unit): Cutoff distance for LJ potential.
            use_vdw_force_switch (bool): If True, use the van-der-Waals force switch via a CustomNonbondedForce.
            work_dir (str): The working directory.
            tmp_output_dir (str): A temporary directory for output during simulations.
            dcd_output_interval (int): Number of time steps between trajectory frames.
            table_output_interval (int):  Number of time steps between lines in output tables.
            steps_per_sequence (int): Number of time steps per job.
            misc_psf_create_system_kwargs (dict): Provides an interface for the user to modify keyword
                arguments of the CharmmPsfFile.createSystem() call.
            use_only_xml_restarts (bool): If True, always use state files for restarts.
                If False, try checkpoint file first.
            center_around (selection string): Center initial system around the selection. If None, don't recenter.
            analysis_mode (bool): If True, create the workflow in its initial state without
                setting up the directory structure or requiring a GPU.

        Attributes:
            positions (list of np.ndarray): List of particle positions from input file; in nanometers
        """
        super(RickFlow, self).__init__(*args, analysis_mode=analysis_mode, steps=steps_per_sequence, **kwargs)
        if not analysis_mode:
            self.next_seqno, self.current_checkpoint, self.current_state = (
                get_next_seqno_and_checkpoints(self.work_dir)
            )

    def _initialize_state(self):
        """
        Initialize state, use checkpoint for seqno > 1.
        """
        if self.analysis_mode or self.next_seqno == 1:
            print("Setting up from initial coordinates.")
            if self.psf.topology.getPeriodicBoxVectors():
                self.context.setPeriodicBoxVectors(
                    *self.psf.topology.getPeriodicBoxVectors())
            self.context.setPositions(self.positions)
            if self.initialize_velocities:
                temperature = self.simulation.integrator.getTemperature()
                print("Setting random initial velocities with temperature {}".format(temperature))
                self.context.setVelocitiesToTemperature(temperature)
        else:
            print("Attempting restart...")
            if not self.use_only_xml_restarts:
                try:
                    self.simulation.loadCheckpoint(self.current_checkpoint)
                    print("Restarting from checkpoint file {}.".format(
                        self.current_checkpoint))
                except:
                    print("    ...could not read from checkpoint file...")
                    self.simulation.loadState(self.current_state)
                    print("Restarting from state file {}.".format(
                        self.current_state))
            else:
                self.simulation.loadState(self.current_state)
                print("Restarting from state file {}.".format(
                    self.current_state))

            # read time and timestep
            last_out = np.loadtxt("out/out{}.txt".format(self.next_seqno - 1),
                                  delimiter=",")
            last_step = int(last_out[-1][0])
            last_time = last_out[-1][1] * u.picosecond
            print("Restart at step {}. Time = {}.".format(last_step, last_time.format("%.4f")))
            self.simulation.currentStep = last_step
            self.context.setTime(last_time)
        self.context.applyConstraints(1e-7)

    def run(self):
        """
        Run the simulation.
        """
        if self.analysis_mode:
            raise RickFlowException("Cannot run an instance that has been created with analysis_mode=True.")
        # define output for the current run
        output_dir = self.work_dir if self.tmp_output_dir is None else self.tmp_output_dir
        with CWD(output_dir):
            with open("trj/dyn{}.dcd".format(self.next_seqno), "wb") as dcd:
                dcd_file = DCDFile(dcd, self.psf.topology, self.simulation.integrator.getStepSize(),
                                   self.simulation.currentStep + self.dcd_output_interval, self.dcd_output_interval)
                # first frame in the trajectory is after the 1000th integration step if the output interval is 1000.
                # I am using this to make CHARMM happy. Dcd files written by the DCDReporter cause warnings in charmm,
                # e.g. the first frame is 0 and cannot be read

                self.simulation.reporters.clear()
                self.simulation.reporters.append(
                    StateDataReporter("out/out{}.txt".format(self.next_seqno), self.table_output_interval,
                                      step=True, time=True,
                                      potentialEnergy=True, temperature=True,
                                      volume=True, density=True, speed=True))

                # run one sequence
                print("Starting {} steps".format(self.steps))

                for i in range(self.steps // self.dcd_output_interval):
                    self.simulation.step(self.dcd_output_interval)
                    # write to dcd
                    state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
                    dcd_file.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())

            # save checkpoints
            self.simulation.saveState("res/state{}.xml".format(self.next_seqno))
            with open("res/checkpoint{}.chk".format(self.next_seqno), 'wb') as f:
                f.write(self.simulation.context.createCheckpoint())

        with CWD(self.work_dir):
            # if required, copy temporary files over
            if self.tmp_output_dir is not None:
                shutil.copy(os.path.join(self.tmp_output_dir, "trj/dyn{}.dcd".format(self.next_seqno)), "trj")
                shutil.copy(os.path.join(self.tmp_output_dir, "out/out{}.txt".format(self.next_seqno)), "out")
                shutil.copy(os.path.join(self.tmp_output_dir, "res/state{}.xml".format(self.next_seqno)), "res")
                shutil.copy(os.path.join(self.tmp_output_dir, "res/checkpoint{}.chk".format(self.next_seqno)),
                            "res")
            # increment sequence number
            self.next_seqno += 1
            with open("next.seqno", 'w') as fp:
                fp.write(str(self.next_seqno))


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
        if not os.path.isdir("log"):  # directory for slurm output files
            os.mkdir("log")

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
