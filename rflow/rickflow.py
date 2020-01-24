# -*- coding: utf-8 -*-

"""
Translating Rick Venable's simulation workflow from CHARMM to OpenMM.
"""

import os
import warnings
import numpy as np
import shutil

from simtk import unit as u
from simtk.openmm.app import PME, HBonds

from rflow.exceptions import LastSequenceReached, RickFlowException
from rflow.reporters.dcdreporter import DCDReporter
from rflow.workflow import PsfWorkflow
from rflow.utility import CWD


class RickFlow(PsfWorkflow):
    """
    Runs a simulation in OpenMM in sequences.

    Attributes:
        positions (list of np.ndarray): List of particle positions from input file; in nanometers
    """
    # filenames
    _trjdir = "trj"
    _outdir = "out"
    _resdir = "res"
    _logdir = "log"
    _dcd = lambda i: os.path.join(RickFlow._trjdir, f"dyn{i}.dcd")
    _veldcd = lambda i: os.path.join(RickFlow._trjdir, f"vel{i}.dcd")
    _out = lambda i: os.path.join(RickFlow._outdir, f"out{i}.txt")
    _xml = lambda i: os.path.join(RickFlow._resdir, f"state{i}.xml")
    #_chk = lambda i: os.path.join(RickFlow._resdir, f"checkpoint{i}.chk")
    _nxt = "next.seqno"
    _lst = "last.seqno"

    def __init__(self, toppar, psf, crd,
                 box_dimensions=None, gpu_id=0,
                 nonbonded_method=PME,
                 switch_distance=8*u.angstrom,
                 cutoff_distance=12*u.angstrom,
                 vdw_switching="charmm-gui",
                 work_dir=".",
                 tmp_output_dir=None,
                 dcd_output_interval=1000,
                 table_output_interval=1000,
                 steps_per_sequence=1000000,
                 misc_psf_create_system_kwargs={},
                 initialize_velocities=True,
                 center_around=None,
                 center_relative_position=0.5,
                 center_dcd_at_origin=False,
                 analysis_mode=False,
                 precision="mixed",
                 report_velocities=False,
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
            gpu_id (int, str, or None): The device id of the CUDA-compatible GPU to be used. If None,
                the use of CUDA is not enforced and OpenMM may run on a slower platform.
                Note that slurm scheduler is smart enough to map the gpu_id 0
                to the first GPU that was allocated for the current job. Therefore multiple slurm jobs
                with `gpu_id=0` can be run on the same node and each will have its own GPU.
                To specify a different platform than CUDA, you can specify the platform name instead of an id
                ("Reference", "CPU", or "CUDA").
            nonbonded_method (OpenMM object): openmm.app.PME for cutoff-LJ, openmm.app.LJPME for LJ-PME
            switch_distance (simtk.unit): Switching distance for LJ potential.
            cutoff_distance (simtk.unit): Cutoff distance for LJ potential.
            vdw_switching (str): Can be "charmm-gui" (CHARMM-GUI implementation of vfswitch; the default),
                "vswitch" (from openmmtools), "vfswitch" (from openmmtools),
                or "openmm" (OpenMM's built-in switching function; the fastest option).
            work_dir (str): The working directory.
            tmp_output_dir (str): A temporary directory for output during simulations.
            dcd_output_interval (int): Number of time steps between trajectory frames.
            table_output_interval (int):  Number of time steps between lines in output tables.
            steps_per_sequence (int): Number of time steps per job.
            misc_psf_create_system_kwargs (dict): Provides an interface for the user to modify keyword
                arguments of the CharmmPsfFile.createSystem() call.
            center_around (selection or None): Center initial system around the selection.
                If None, center the system around all non-TIP3Ps.
            analysis_mode (bool): If True, create the workflow in its initial state without
                setting up the directory structure or requiring a GPU.
            precision (str): "mixed", "single", or "double". Only active on CUDA platform, default: mixed.
            center_relative_position (float): The relative position of the 'center_around' selection 
                with respect to the box dimensions.
            center_dcd_at_origin (bool): If True, the output trajectory fills the box as [-L/2,L/2] instead of [O,L]
        """
        self.work_dir = work_dir
        self.gpu_id = gpu_id
        self.precision = precision
        self.tmp_output_dir = tmp_output_dir
        self.dcd_output_interval = dcd_output_interval
        self.table_output_interval = table_output_interval
        self.steps_per_sequence = steps_per_sequence
        self.vdw_switching = vdw_switching
        self.analysis_mode = analysis_mode
        self.tmp_output_dir = tmp_output_dir
        self.initialize_velocities = initialize_velocities
        self.report_velocities = report_velocities
        self._last_seqno = None
        self._next_seqno = None
        if 'use_vdw_force_switch' in kwargs:
            warnings.warn(
                "use_vdw_force_switch is deprecated and will be removed in future versions. "
                "Use RickFlow(....,vdw_switching='openmm'/'charmm-gui', ...) instead."
            )
            self.vdw_switching = "charmm-gui" if kwargs['use_vdw_force_switch'] else "openmm"
            kwargs.pop('use_vdw_force_switch')
        if 'use_only_xml_restarts' in kwargs:
            warnings.warn(
                "use_only_xml_restarts is deprecated and will be removed in future versions. "
                "Only xml restart files are generated due to instability "
                "issues with the checkpoint files. "
            )
            kwargs.pop("use_only_xml_restarts")
        assert len(kwargs) == 0, "Keyword arguments not understood: {}".format(kwargs)

        if not self.analysis_mode:
            RickFlow.make_directory_structure(work_dir)
            if self.tmp_output_dir is not None:
                self.make_directory_structure(tmp_output_dir)

        with CWD(work_dir):
            super(RickFlow, self).__init__(
                toppar=toppar,
                psf=psf,
                crd=crd,
                box_dimensions=box_dimensions,
                center_around=center_around,
                center_relative_position=center_relative_position,
                center_dcd_at_origin=center_dcd_at_origin
            )

            if not steps_per_sequence % dcd_output_interval== 0:
                raise RickFlowException(
                    "dcd_output_interval ({}) has to be a divisor of steps_per_sequence ({}).".format(
                        dcd_output_interval, steps_per_sequence
                ))
            if not steps_per_sequence % table_output_interval== 0:
                raise RickFlowException(
                    "table_output_interval ({}) has to be a divisor of steps_per_sequence ({}).".format(
                        table_output_interval, steps_per_sequence
                ))

            # create system
            psf_create_system_kwargs = {
                "nonbondedMethod": nonbonded_method,
                "constraints": HBonds,
            }
            psf_create_system_kwargs.update(misc_psf_create_system_kwargs)
            self.create_system(
                disable_lrc=True,
                vdw_switching=self.vdw_switching,
                switch_distance=switch_distance,
                cutoff_distance=cutoff_distance,
                **psf_create_system_kwargs
            )

    @property
    def next_seqno(self):
        # increment sequence number
        if self._next_seqno is None:
            with CWD(self.work_dir):
                # a file containing the id of the next sequence to be simulated
                if os.path.isfile(self._nxt):
                    with open(self._nxt, 'r') as fp:
                        return int(fp.read())
                else:
                    with open(self._nxt, 'w') as fp:
                        fp.write("1")
                        return 1
        else:
            return self._next_seqno

    @next_seqno.setter
    def next_seqno(self, i):
        with CWD(self.work_dir):
            with open(self._nxt, 'w') as fp:
                fp.write(str(i))
        self._next_seqno = i

    @property
    def last_seqno(self):
        # a file containing the id of the last sequence to be simulated
        if self._last_seqno is None:
            with CWD(self.work_dir):
                if os.path.isfile(self._lst):
                    with open(self._lst, 'r') as fp:
                        return int(fp.read())
                else:
                    return 99999999999
        else:
            return self._last_seqno

    @property
    def previous_step(self):
        with CWD(self.work_dir):
            # read time and timestep
            last_out = np.loadtxt(RickFlow._out(self.next_seqno-1), delimiter=",")
            last_step = int(last_out[-1][0])
            last_time = last_out[-1][1] * u.picosecond
            return last_step, last_time

    @property
    def output_directory(self):
        return self.work_dir if self.tmp_output_dir is None else self.tmp_output_dir

    def prepareSimulation(self, integrator, barostat=None):
        warnings.warn(
            "The method `prepareSimulation` has been renamed "
            "into `prepare_simulation`. "
            "This will work for now, but will be removed in "
            "future versions.",
            DeprecationWarning
        )
        self.prepare_simulation(integrator, barostat)

    def prepare_simulation(self, integrator, barostat=None):
        """
        Initialize simulation object by passing an integrator and a barostat.

        Args:
            integrator (OpenMM integrator object): The integrator to be used.
            barostat (OpenMM barostat object): The barostat. Pass None for NVT.
        """
        self.create_simulation(integrator, barostat)
        self.initialize_state()

    def create_simulation(
            self,
            integrator,
            barostat=None
    ):
        with CWD(self.output_directory):
            super(RickFlow, self).create_simulation(
                integrator=integrator,
                barostat=barostat,
                gpu_id=None if self.analysis_mode else self.gpu_id,
                dcd_output_interval=0 if self.analysis_mode else self.dcd_output_interval,
                dcd_output_file=RickFlow._dcd(self.next_seqno),
                table_output_interval=0 if self.analysis_mode else self.table_output_interval,
                table_output_file=RickFlow._out(self.next_seqno),
                precision=self.precision
            )
            if self.dcd_output_interval > 0 and self.report_velocities:
                self.simulation.reporters.append(
                    DCDReporter(
                        RickFlow._veldcd(self.next_seqno),
                        self.dcd_output_interval,
                        velocities=True
                    )
                )

    def initialize_state(self):
        """
        Initialize state, use checkpoint for seqno > 1.
        """
        with CWD(self.work_dir):
            if self.analysis_mode or self.next_seqno == 1:
                print("Setting up from initial coordinates.")
                super(RickFlow, self).initialize_state(
                    self.initialize_velocities,
                    pdb_output_file = None if self.analysis_mode else "system.pdb"
                )
            else:
                print("Attempting restart...")
                #checkpoint_file = RickFlow._chk(self.next_seqno-1)
                state_file = RickFlow._xml(self.next_seqno-1)
                # if not self.use_only_xml_restarts:          # to be removed
                #     try:
                #         self.simulation.loadCheckpoint(checkpoint_file)
                #         print("Restarting from checkpoint file {}.".format(checkpoint_file))
                #     except:
                #         print("    ...could not read from checkpoint file...")
                #         self.simulation.loadState(state_file)
                #         print("Restarting from state file {}.".format(state_file))
                # else:
                self.simulation.loadState(state_file)
                print("Restarting from state file {}.".format(state_file))

                # read time and timestep
                last_step, last_time = self.previous_step
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

        print("Starting {} steps".format(self.steps_per_sequence))

        with CWD(self.output_directory):
            # run dynamics
            self.simulation.step(self.steps_per_sequence)

            # save checkpoints
            self.simulation.saveState(RickFlow._xml(self.next_seqno))
            #with open(RickFlow._chk(self.next_seqno), 'wb') as f:
            #    f.write(self.simulation.context.createCheckpoint())

            # if required, copy temporary files over
            if self.tmp_output_dir is not None:
                shutil.copy(RickFlow._dcd(self.next_seqno), os.path.join(self.work_dir, RickFlow._trjdir))
                shutil.copy(RickFlow._out(self.next_seqno), os.path.join(self.work_dir, RickFlow._outdir))
                shutil.copy(RickFlow._xml(self.next_seqno), os.path.join(self.work_dir, RickFlow._resdir))
                if self.report_velocities:
                    shutil.copy(RickFlow._veldcd(self.next_seqno), os.path.join(self.work_dir, RickFlow._trjdir))

                #shutil.copy(RickFlow._chk(self.next_seqno), os.path.join(self.work_dir, RickFlow._resdir))

        self.next_seqno = self.next_seqno + 1
        if self.next_seqno > self.last_seqno:
            raise LastSequenceReached(self.last_seqno)

    @classmethod
    def make_directory_structure(cls, directory):
        with CWD(directory):
            if not os.path.isdir(cls._trjdir):  # directory for trajectories
                os.mkdir(cls._trjdir)
            if not os.path.isdir(cls._outdir):  # directory for state files
                os.mkdir(cls._outdir)
            if not os.path.isdir(cls._resdir):  # directory for restart files
                os.mkdir(cls._resdir)
            if not os.path.isdir(cls._logdir):  # directory for slurm output files
                os.mkdir(cls._logdir)
