# -*- coding: utf-8 -*-

from warnings import warn

import numpy as np

import mdtraj as md


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
    def __init__(self, lag_iterations, num_bins, solute_ids, com_removal=False):
        self.lag_iterations = lag_iterations
        self.fifo_positions = [None for _ in range(max(lag_iterations) + 1)]
        self.matrices = {lag: np.zeros((num_bins, num_bins), dtype=np.int32)
                         for lag in lag_iterations}
        self.num_bins = num_bins
        self.solute_ids = solute_ids
        self.com_removal = com_removal
        self.n_timesteps = 0
        self.average_box_height = 0.0

    @property
    def edges(self):
        return np.arange(-0.5*self.average_box_height,
                         0.5*self.average_box_height + 1e-6,
                         self.average_box_height/self.num_bins)

    def __call__(self, trajectory):
        # get center of mass of the membrane (system minus water minus solute)
        if self.com_removal:
            membrane = trajectory.topology.select("not water")
            for sol in self.solute_ids:
                if sol in membrane:
                    membrane.remove(sol)
            membrane_trajectory = trajectory.atom_slice(membrane)
            membrane_center = md.compute_center_of_mass(
                membrane_trajectory
            )
        # normalize z coordinates: scale to [0,1] and shift membrane center to 0.5
            z_normalized = trajectory.xyz[:, self.solute_ids,
                           2].transpose() - membrane_center[:, 2]
        else:
            z_normalized = trajectory.xyz[:, self.solute_ids,
                           2].transpose()

        z_normalized = np.mod(
            z_normalized / trajectory.unitcell_lengths[:, 2] + 0.5,
            1.0).transpose()

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
                    for i, j in zip(self.fifo_positions[0],
                                    self.fifo_positions[lag]):
                        self.matrices[lag][i, j] += 1


class PermeationEventCounter(object):

    def __init__(self, solute_ids, dividing_surface, center_threshold=0.02):
        self.center_threshold = center_threshold
        self.dividing_surface = dividing_surface
        self.bins = np.array([center_threshold, 0.5 - dividing_surface,
                              0.5 - center_threshold, 0.5 + center_threshold,
                              0.5 + dividing_surface, 1.0 - center_threshold,
                              1.0])
        # bin 0: control bin at the first edge of the periodic boundary
        # bin 1: first water bin
        # bin 2: first outer membrane bin
        # bin 3: membrane center
        # bin 4: second outer membrane bin
        # bin 5: second water bin
        # bin 6: control bin at the second edge of the periodic boundary
        self.functional_bins = [1, 3, 5]
        self.startframe = 0
        self.solute_ids = solute_ids
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
                    "in trajectory frame {}. This indicates, that permeation "
                    "events may be missed.".format(
                        [self.solute_ids[i] for i in too_fast_particles],
                        frame
                    )
                )
                is_critical = np.any(
                    np.isin(z_digitized_i[too_fast_particles], [2, 3, 4]))
                if is_critical:
                    message += (
                    " This seems to have been a "
                    "transit through the bilayer."
                    "You should save your simulation "
                    "output more frequently or increase the "
                    "center_threshold.")
                    warn(message)

        self.previous_z_digitized = z_digitized_i

    def __call__(self, trajectory):
        # get center of mass of the membrane (system minus water minus solute)
        membrane = trajectory.topology.select("not water")
        for sol in self.solute_ids:
            if sol in membrane:
                membrane.remove(sol)
        membrane_trajectory = trajectory.atom_slice(membrane)
        membrane_center = md.compute_center_of_mass(
            membrane_trajectory
        )

        # normalize z coordinates: scale to [0,1] and shift membrane center to 0.5
        z_normalized = trajectory.xyz[:, self.solute_ids,
                       2].transpose() - membrane_center[:, 2]
        z_normalized = np.mod(
            z_normalized / trajectory.unitcell_lengths[:, 2] + 0.5,
            1.0).transpose()

        # find bin indices
        z_digitized = np.digitize(z_normalized, self.bins)

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
