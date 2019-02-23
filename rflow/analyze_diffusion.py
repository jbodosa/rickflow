# -*- coding: utf-8 -*-

from warnings import warn
import pickle

import numpy as np

import mdtraj as md

from rflow.utility import selection
from rflow.trajectory import normalize
from rflow.exceptions import RickFlowException


class TransitionCounter(object):
    """
    A class to extract transitions matrices.
    Example usage:

    >>> from rickflow import CharmmTrajectoryIterator as TI
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
                    # TODO: Replace by np.add.at
                    for i, j in zip(self.fifo_positions[0],
                                    self.fifo_positions[lag]):
                        self.matrices[lag][i, j] += 1

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

    def __init__(self, solute_ids, dividing_surface, center_threshold=0.02, membrane=None,
                 initialize_all_permeants=True):
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
        self.initialize_all_permeants = initialize_all_permeants

    @property
    def num_crossings(self):
        n = 0
        for e in self.events:
            if e["type"]=="crossing": n += 1
        return n

    def _sanity_check(self, z_digitized_i, frame):
        # sanity check: transitions should occur only between adjacent bins
        if self.previous_z_digitized is not None:
            transition_step_size = np.mod(
                np.absolute(z_digitized_i - self.previous_z_digitized),
                len(self.bins) - 1
            )
            # print(transition_step_size)
            too_fast_particles = np.where(transition_step_size > 1)[0]
            # print(too_fast_particles, bool(too_fast_particles))
            if too_fast_particles.size > 0:
                message = (
                    "An infeasible transition was detected for particles {} "
                    "in trajectory frame {}.".format(
                        [self.solute_ids[i] for i in too_fast_particles],
                        frame
                    )
                )
                is_critical = np.any(
                    np.isin(z_digitized_i[too_fast_particles], [2, 3, 4]))
                if is_critical:
                    message += (
                    " This might have been a "
                    "transit through the bilayer."
                    "You should save your simulation "
                    "output more frequently or increase the "
                    "center_threshold.")
                    warn(message)

        self.previous_z_digitized = z_digitized_i

    def __call__(self, trajectory):
        z_normalized = normalize(trajectory, 2, self.membrane, self.solute_ids)

        # find bin indices
        z_digitized = np.digitize(z_normalized, self.bins)

        # initialize flags for all permeants
        if self.initialize_all_permeants:
            if (-999999 in self.last_functional_bin) or (-999999 in self.last_water_bin):
                self.last_functional_bin = z_digitized
                self.last_functional_bin[np.where[np.isin(self.last_functional_bin, [0,1,2])]] = 1
                self.last_functional_bin[np.where[np.isin(self.last_functional_bin, [4,5,6])]] = 5
                assert np.isin(self.last_functional_bin, self.functional_bins).all()
                self.last_water_bin = (z_normalized < 0.5)*1 + (z_normalized >= 0.5)*5

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

                from_w_bin_time = frame - self.framenr_of_last_seen_in_functional_bin[last_w_bin][
                    particle]
                event = {
                    "frame": frame,
                    "from_water": last_w_bin,
                    "atom": self.solute_ids[particle]
                }
                if self.last_functional_bin[
                    particle] == 3:  # enter water from center
                    exit_time = frame - self.framenr_of_last_seen_in_functional_bin[3][
                                    particle]
                    event["exit_time_nframes"] = exit_time
                    if self.last_water_bin[particle] == z_digitized[i][
                        particle]:
                        # rebound
                        event["type"] = "rebound"
                        event["rebound_time_nframes"] = from_w_bin_time
                    else:  # crossing
                        assert abs(
                            self.last_water_bin[particle] - z_digitized[i][
                                particle]) == 4
                        crossing_time = \
                        self.framenr_of_last_seen_in_functional_bin[
                            z_digitized[i][particle]][particle]
                        event["type"] = "crossing"
                        event["crossing_time_nframes"] = from_w_bin_time
                else:
                    assert z_digitized[i][particle] == 3
                    # entry
                    event["type"] = "entry"
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
        atom_ids = selection(trajectory, self.atom_selection)
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
