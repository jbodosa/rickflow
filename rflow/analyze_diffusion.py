# -*- coding: utf-8 -*-

"""
Analysis tools for diffusivity and membrane permeation.
"""


from warnings import warn
import pickle

import numpy as np

import mdtraj as md
from simtk import unit as u

from rflow.utility import select_atoms, increment_using_multiindices
from rflow.trajectory import normalize
from rflow.exceptions import RickFlowException


class TransitionCounter(object):
    """
    A class to extract transitions matrices.

    Examples:

        Usage on trajectories that have been created using the rflow standard protocol:

        >>> from rickflow import TrajectoryIterator as TI
        >>> trans_counter = TransitionCounter([10,20], 10, [0,1,2,3])
        >>> for frame in TI():
        >>>     trans_counter(frame)
        >>> print(frame.matrices)
    """
    def __init__(self, lag_iterations, num_bins, solute, membrane=None):
        self.lag_iterations = lag_iterations
        self.fifo_positions = [None for _ in range(max(lag_iterations) + 1)]
        self.matrices = {lag: np.zeros((num_bins, num_bins), dtype=np.int32)
                         for lag in lag_iterations}
        self.num_bins = num_bins
        self.solute = solute
        self.membrane = membrane
        self.n_timesteps = 0
        self.average_box_height = 0.0

    @property
    def edges(self):
        return np.arange(0.0, self.average_box_height + 1e-6,
                         self.average_box_height/self.num_bins)

    @property
    def edges_around_zero(self):
        return np.arange(-0.5*self.average_box_height,
                         0.5*self.average_box_height + 1e-6,
                         self.average_box_height/self.num_bins)

    def __call__(self, trajectory):
        z_normalized = normalize(trajectory, 2, self.membrane, self.solute)

        # update edges
        self.average_box_height = (self.average_box_height * self.n_timesteps
                                   + np.mean(trajectory.unitcell_lengths[:, 2]) * trajectory.n_frames)
        self.n_timesteps += trajectory.n_frames
        self.average_box_height /= self.n_timesteps

        # find bin indices
        h = 1.0 / self.num_bins
        bins = np.arange(h, 1.0 + h, h)
        z_digitized = np.digitize(z_normalized, bins)

        for i in range(trajectory.n_frames):
            # update FIFO queue of positions
            self.fifo_positions.pop()
            self.fifo_positions.insert(0, z_digitized[i])
            # add to transition matrices
            for lag in self.lag_iterations:
                if self.fifo_positions[lag] is not None:
                    increment_using_multiindices(
                        self.matrices[lag],
                        np.column_stack([self.fifo_positions[0],self.fifo_positions[lag]]))

    def save_matrices(self, filename_template):
        try:
            from dcma.matrices import Transitions
        except ImportError:
            raise RickFlowException("Saving transition matrices requires the dcma package to be installed.")
        for l in self.matrices:
            filename = filename_template.format(l)
            #                              dcma expects edges in angstrom: multiply by 10
            tmat = Transitions(lag_time=l, edges=self.edges_around_zero*10.0, matrix=self.matrices[l])
            tmat.save(filename)


class PermeationEventCounter(object):
    """Class to count permeation events (permeants entering/exiting/crossing a membrane).
    """
    def __init__(self, solute_ids, dividing_surface, center_threshold=0.02, membrane=None,
                 initialize_all_permeants=True):
        """
        Args:
            solute_ids (list of int): The permeant's atom ids.
            dividing_surface (float): Placement of the dividing surface that separates the water phase from the
                membrane. The dividing surface is specified relative to the height of the system,
                as a distance from the center (thus it must be a number between 0.0 and 0.5).
            center_threshold (float): Placement of the dividing surface that separates the central membrane region
                from the outer membrane. The dividing surface is specified relative to the height of the system,
                as a distance from the center (thus it must be a number between 0.0 and 0.5). In order to be counted
                as crossings, permeants have to enter the central region and exit to the water region on the opposite
                side.
            membrane (list or np.array of ints): The membrane's atom ids (required to properly center the system).
            initialize_all_permeants (bool): If true, the permeants in the membrane are initialized as having made
                half a crossing already. Setting this to False can lead to dramatic underestimates of the permeability,
                especially for lipid-soluble permeants.
        """
        try:
            assert len(solute_ids) > 0
        except AssertionError:
            raise RickFlowException("The list of solute_ids that you provided was empty.")
        except:
            raise RickFlowException("Invalid solute_ids. "
                                    "Please provide atom indices as a non-empty list "
                                    "(or numpy array) of integers.")
        self.center_threshold = center_threshold
        self.dividing_surface = dividing_surface

        self.bins = np.array([(0.5-dividing_surface)/2.0, 0.5 - dividing_surface,
                              0.5 - center_threshold, 0.5 + center_threshold,
                              0.5 + dividing_surface, 1.0 - (0.5-dividing_surface)/2.0,
                              1.0])
        # bin 0: control bin at the first edge of the periodic boundary
        # bin 1: first donor/acceptor bin
        # bin 2: first outer membrane bin
        # bin 3: membrane center
        # bin 4: second outer membrane bin
        # bin 5: second donor/acceptor bin
        # bin 6: control bin at the second edge of the periodic boundary

        #  dividing_surface:   |<------------>|
        #  center_threshold:             |<-->|
        #  |                                  |                                  |
        # 0.0                                0.5                                1.0
        #  |                                  |                                  |
        #  |---------|---------|---------|---------|---------|---------|---------|
        #       0         1         2         3         4         5         6
        #                      OX=~~~~~~~~~~~~ ~~~~~~~~~~~~=XO
        #                      OX=~~~~~~~~~~~~ ~~~~~~~~~~~~=XO
        #          WATER                   MEMBRANE                   WATER
        #                      OX=~~~~~~~~~~~~ ~~~~~~~~~~~~=XO
        #                      OX=~~~~~~~~~~~~ ~~~~~~~~~~~~=XO
        #  |---------|---------|---------|---------|---------|---------|---------|
        #
        #
        self.functional_bins = [1, 3, 5]
        self.startframe = 0
        self.solute_ids = solute_ids
        self.membrane = membrane
        # trackers
        self.previous_z_digitized = None
        self.last_water_bin = np.array([-999999 for _ in solute_ids],
                                       dtype=np.int32)
        self.framenr_of_last_seen_in_functional_bin = {
            b:
                np.array(
                    [-999999 for _ in
                     solute_ids],
                    dtype=np.int32)
            for b in
            self.functional_bins
        }
        self.last_functional_bin = np.array([-999999 for _ in solute_ids],
                                            dtype=np.int32)
        self.events = []
        self.severe_warnings = []
        self._num_mild_warnings = 0
        self.initialize_all_permeants = initialize_all_permeants
        self.severely_critical_transitions, self.mildly_critical_transitions = self.make_critical_transitions()

    @property
    def num_crossings(self):
        n = 0
        for e in self.events:
            if e["type"]=="crossing": n += 1
        return n

    @property
    def num_severe_warnings(self):
        return len(self.severe_warnings)

    @property
    def num_mild_warnings(self):
        return self._num_mild_warnings

    def _sanity_check(self, z_digitized_i, frame):
        # sanity check: particles should not hop over two or more bin
        # neither should they hop over the central bin
        if self.previous_z_digitized is not None:
            severely_critical_transitions = np.where(
                self.severely_critical_transitions[self.previous_z_digitized, z_digitized_i])[0]
            mildly_critical_transitions = np.where(
                self.mildly_critical_transitions[self.previous_z_digitized, z_digitized_i])[0]
            if severely_critical_transitions.any():
                for i in severely_critical_transitions:
                    warn("An infeasible transition was detected for particle {} in trajectory frame {} (bin {} to {})."
                         " This might or might not have been a transit through the bilayer. It is not counted as a"
                         " permeation event.".format(self.solute_ids[i], frame, self.previous_z_digitized[i],
                                                     z_digitized_i[i]))
                    self.severe_warnings += [{"frame": frame, "atom": self.solute_ids[i],
                                              "source_bin": self.previous_z_digitized[i],
                                              "target_bin": z_digitized_i[i]}]
            self._num_mild_warnings += mildly_critical_transitions.sum()
        self.previous_z_digitized = np.copy(z_digitized_i)

    def __call__(self, trajectory):
        z_normalized = normalize(trajectory, 2, self.membrane, self.solute_ids)

        # find bin indices
        z_digitized = np.digitize(z_normalized, self.bins)
        # initialize flags for all permeants
        if self.initialize_all_permeants:
            if (-999999 in self.last_functional_bin) or (-999999 in self.last_water_bin):
                self.last_functional_bin = np.copy(z_digitized[0])
                self.last_functional_bin[np.where(np.isin(self.last_functional_bin, [0,1,2]))] = 1
                self.last_functional_bin[np.where(np.isin(self.last_functional_bin, [4,5,6]))] = 5
                assert np.isin(self.last_functional_bin, self.functional_bins).all()
                self.last_water_bin[:] = (z_normalized[0] < 0.5)*1 + (z_normalized[0] >= 0.5)*5

        for i in range(trajectory.n_frames):
            frame = self.startframe + i
            self._sanity_check(z_digitized[i], frame)

            # check for transitions between functional bins
            important_transit = np.logical_and(
                np.isin(z_digitized[i], self.functional_bins),
                z_digitized[i] != self.last_functional_bin
            )
            not_water_to_water = np.logical_or(
                z_digitized[i] == 3,
                self.last_functional_bin == 3
            )
            particles_with_events = np.where(np.logical_and(
                important_transit, not_water_to_water))[0]

            for particle in particles_with_events:
                last_w_bin = self.last_water_bin[particle]
                if last_w_bin == -999999:
                    continue

                from_w_bin_time = frame - self.framenr_of_last_seen_in_functional_bin[last_w_bin][particle]
                event = {
                    "frame": frame,
                    "from_water": last_w_bin,
                    "atom": self.solute_ids[particle]
                }
                if self.last_functional_bin[particle] == 3:  # enter water from center
                    if self.framenr_of_last_seen_in_functional_bin[3][particle] != -999999:
                        exit_time = frame - self.framenr_of_last_seen_in_functional_bin[3][
                                        particle]
                        event["exit_time_nframes"] = exit_time
                    if self.last_water_bin[particle] == z_digitized[i][
                        particle]:
                        # rebound
                        event["type"] = "rebound"
                        if self.framenr_of_last_seen_in_functional_bin[last_w_bin][particle] != -999999:
                            event["rebound_time_nframes"] = from_w_bin_time
                    else:  # crossing
                        assert abs(
                            self.last_water_bin[particle] - z_digitized[i][
                                particle]) == 4

                        event["type"] = "crossing"
                        if self.framenr_of_last_seen_in_functional_bin[last_w_bin][particle] != -999999:
                            event["crossing_time_nframes"] = from_w_bin_time
                else:
                    assert z_digitized[i][particle] == 3
                    # entry
                    event["type"] = "entry"
                    if self.framenr_of_last_seen_in_functional_bin[last_w_bin][particle] != -999999:
                        event["entry_time_nframes"] = from_w_bin_time
                self.events += [event]

            # update
            self.last_water_bin[np.where(z_digitized[i] == 1)] = 1
            self.last_water_bin[np.where(z_digitized[i] == 5)] = 5
            for b in self.functional_bins:
                in_b = np.where(z_digitized[i] == b)[0]
                self.last_functional_bin[in_b] = b
                self.framenr_of_last_seen_in_functional_bin[b][in_b] = frame

        # finalize
        self.startframe += trajectory.n_frames

    @staticmethod
    def make_critical_transitions():
        severe = np.ones((7, 7), dtype=bool)
        mild = np.zeros((7, 7), dtype=bool)
        for i in range(7):
            # transitions from bin i to bin i are not critical
            severe[i, i] = False
            # transitions from bin i to bin i+1 are not critical
            severe[i, i-1] = False
            severe[i-1, i] = False
            # most hops over one bin are not severely critical
            severe[i, i-2] = False
            mild[i, i-2] = True
            severe[i-2, i] = False
            mild[i-2, i] = True
        # The only severely critical hop is the one over the central bin
        severe[2,4] = True
        mild[2,4] = False
        severe[4,2] = True
        mild[4,2] = False
        return severe, mild

    def permeability(self, permeant_distribution, start_frame=0, end_frame=None,
                     time_between_frames=1*u.picosecond, mode='crossings', num_bins_in_water=2):
        return self.calculate_permeability(
            self.events, permeant_distribution, num_permeants=len(self.solute_ids),
            start_frame=start_frame, end_frame=end_frame,
            time_between_frames=time_between_frames, mode=mode,
            num_bins_in_water=num_bins_in_water)

    @staticmethod
    def calculate_permeability(events, permeant_distribution, num_permeants=None, start_frame=0, end_frame=None,
                               time_between_frames=1*u.picosecond, mode='crossings', num_bins_in_water=2):
        # try to set default arguments
        if end_frame is None:
            end_frame = permeant_distribution.n_frames
        if num_permeants is None:
            num_permeants = len(permeant_distribution.atom_selection)
        # initialize chosen counting method
        if mode == 'crossings':
            factor = 2.0
            counted_events = ["crossing"]
        elif mode == 'rebounds':
            factor = 4.0
            counted_events = ["crossing", "rebound"]
        elif mode == 'semi-permeation':
            factor = 8.0
            counted_events = ["crossing", "rebound", "entry"]
        else:
            raise RickFlowException("mode has to be 'crossings', 'rebounds', or 'semi-permeation'")
        # count permeation events
        num_events = 0
        for event in events:
            if (event["type"] in counted_events) and (event["frame"] >= start_frame) and (event["frame"] <= end_frame):
                num_events += 1
        # get simulated time
        tsim = (end_frame - start_frame + 1) * time_between_frames
        # normalize free energy to be zero in the water
        free_energy = permeant_distribution.free_energy
        water_free_energy = np.mean(free_energy[:num_bins_in_water].tolist() + free_energy[-num_bins_in_water:].tolist())
        free_energy -= water_free_energy
        exponential = np.exp(-free_energy)
        # integrate free energy
        fe_integral = (permeant_distribution.average_box_size * u.nanometer * np.mean(
                       [0.5*exponential[0]] + exponential[1:-1].tolist() + [0.5*exponential[-1]]))
        # Calculate permeability
        return (fe_integral / (factor*num_permeants) * (num_events / tsim)).value_in_unit(u.centimeter/u.second)


class Distribution(object):
    def __init__(self, atom_selection, coordinate, nbins=100, com_selection=None):
        """
        Args:
            atom_selection:
            coordinate:
            nbins:
            com_selection: List of atom ids to calculate the com of the membrane, to make the distribution relative to
                    the center of mass.
        """
        self.atom_selection = atom_selection
        self.coordinate = coordinate
        self.average_box_size = 0.0
        self.n_frames = 0
        self.bins = np.arange(0, 1.0 + 1e-6, 1.0/nbins)
        self.nbins = nbins
        self.counts = 0.0
        self.com_selection = com_selection

    @property
    def bin_centers(self):
        return self.average_box_size * (self.bins[:-1] + 0.5*self.bins[1])

    @property
    def bin_centers_around_zero(self):
        return self.average_box_size * (self.bins[:-1] + 0.5 * self.bins[1] - 0.5)

    @property
    def probability(self):
        return self.counts / self.counts.sum()

    @property
    def free_energy(self):
        """in kBT"""
        return - np.log(self.counts / np.max(self.counts))

    def __call__(self, trajectory):
        atom_ids = select_atoms(trajectory, self.atom_selection)
        normalized = normalize(trajectory, self.coordinate, subselect=atom_ids, com_selection=self.com_selection)
        box_size = trajectory.unitcell_lengths[:, self.coordinate]
        self.average_box_size = self.n_frames * self.average_box_size + trajectory.n_frames * box_size.mean()
        self.n_frames += trajectory.n_frames
        self.average_box_size /= self.n_frames

        histogram = np.histogram(normalized, bins=self.nbins, range=(0,1))  # this is !much! faster than manual bins
        self.counts = self.counts + histogram[0]

    def save(self, filename):
        data = np.array([self.bin_centers, self.bin_centers_around_zero, self.counts,
                         self.probability, self.free_energy])
        np.savetxt(filename, data.transpose(),
                   header="bin_centers, bin_centers_around_0, counts, probability, free_energy_(kBT)\n")

        with open(filename + ".pic", 'wb') as pic:
            pickle.dump(self, pic)

    @staticmethod
    def load_from_pic(filename):
        with open(filename, 'rb') as pic:
            return pickle.load(pic)


