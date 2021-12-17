# -*- coding: utf-8 -*-

"""
Analysis tools for diffusivity and membrane permeation.
"""


import warnings
from warnings import warn

import numpy as np
import pandas as pd

from rflow.openmm import unit as u

import rflow.observables
from rflow.utility import increment_using_multiindices
from rflow.observables import BinEdgeUpdater
from rflow.trajectory import normalize
from rflow.exceptions import RickFlowException


class TransitionCounter(BinEdgeUpdater):
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
        super(TransitionCounter, self).__init__(num_bins, coordinate=2)
        self.lag_iterations = lag_iterations
        self.fifo_positions = [None for _ in range(max(lag_iterations) + 1)]
        self.matrices = {lag: np.zeros((num_bins, num_bins), dtype=np.int32)
                         for lag in lag_iterations}
        self.solute = solute
        self.membrane = membrane

    def __call__(self, trajectory):
        super(TransitionCounter, self).__call__(trajectory)
        z_normalized = normalize(trajectory, 2, self.membrane, self.solute)

        # find bin indices
        h = 1.0 / self.num_bins
        bins = np.arange(h, 1.0 + h, h)
        z_digitized = np.digitize(z_normalized, bins, right=True)

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

    def save_matrices(self, filename_template, time_between_frames, dt=1.0):
        """
        Writes transitions matrices in a format that can be read by diffusioncma and mcdiff.
        Args: 
            filename_template (str): template for the files, where {} is a placeholder for the lag time.
            time_between_frames (float): time elapsed between two trajectory frames in picoseconds.
            dt (float): the time step dt written into the header of the files.
        """
        from rflow.utility import _DCMATransitions
        for lag in self.matrices:
            filename = filename_template.format(lag)
            #                              dcma expects edges in angstrom: multiply by 10
            tmat = _DCMATransitions(
                lag_time=lag*time_between_frames,
                edges=self.edges_around_zero*10.0,
                matrix=self.matrices[lag]
            )
            tmat.save(filename, dt=dt)

    def brownian_similarity(self, time_between_frames=1.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulation_time = time_between_frames*self.n_frames
            from scipy.optimize import curve_fit
            curve = lambda x,a,b: a*x**b
            fluxes = np.array([
                [0.5*(self.matrices[lag][i-1,i]+self.matrices[lag][i,i-1])/simulation_time for lag in self.lag_iterations]
                for i in range(self.num_bins)
            ])
            exponents = []
            factors = []
            for bin in range(self.num_bins):
                factor, exponent = curve_fit(curve, np.array(self.lag_iterations), fluxes[bin,:])[0]
                factors.append(factor)
                exponents.append(exponent)
            return np.array(exponents), np.array(factors), fluxes


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
        self.num_transitions_between_bins = np.zeros((7, 7), dtype=int)
        self.initialize_all_permeants = initialize_all_permeants
        self.severe_warnings = []

    @property
    def num_crossings(self):
        n = 0
        for e in self.events:
            if e["type"]=="crossing": n += 1
        return n

    @property
    def num_severe_warnings(self):
        return self.num_transitions_between_bins[2, 4] + self.num_transitions_between_bins[4, 2]

    def _sanity_check(self, z_digitized_i, frame):
        # sanity check: particles should not hop over the central bin
        if self.previous_z_digitized is not None:
            num_transitions_between_bins = np.zeros((7, 7), dtype=int)
            np.add.at(num_transitions_between_bins, (self.previous_z_digitized, z_digitized_i), 1)
            self.num_transitions_between_bins += num_transitions_between_bins
            if num_transitions_between_bins[2, 4] + num_transitions_between_bins[4, 2] > 0:
                severely_critical_transitions = np.where(
                    np.logical_or(
                        np.logical_and(
                            self.previous_z_digitized == 2,
                            z_digitized_i == 4
                        ),
                        np.logical_and(
                            self.previous_z_digitized == 4,
                            z_digitized_i == 2
                        ),
                    )
                )
                for i in severely_critical_transitions:
                    warn("An infeasible transition was detected for particle {} in trajectory frame {} (bin {} to {})."
                         " This might or might not have been a transit through the bilayer. It is not counted as a"
                         " permeation event.".format(self.solute_ids[i], frame, self.previous_z_digitized[i],
                                                     z_digitized_i[i]))
                    self.severe_warnings += [{"frame": frame, "atom": self.solute_ids[i],
                                              "source_bin": self.previous_z_digitized[i],
                                              "target_bin": z_digitized_i[i]}]

    def __call__(self, trajectory):
        z_normalized = normalize(trajectory, 2, self.membrane, self.solute_ids)

        # find bin indices
        z_digitized = np.digitize(z_normalized, self.bins, right=True)
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
                            event["in_membrane_time_nframes"] = from_w_bin_time
                else:
                    assert z_digitized[i][particle] == 3
                    # entry
                    event["type"] = "entry"
                    if self.framenr_of_last_seen_in_functional_bin[last_w_bin][particle] != -999999:
                        event["entry_time_nframes"] = from_w_bin_time
                self.events += [event]

            # update
            self.last_water_bin[np.where(np.isin(z_digitized[i], [0, 1]))] = 1
            self.last_water_bin[np.where(np.isin(z_digitized[i], [5, 6]))] = 5
            # consider all transitions into the outer membrane that have not entered from the central bin as having
            # entered through the water phase
            if self.previous_z_digitized is not None:
                self.last_functional_bin[np.where(
                    np.logical_and(z_digitized[i] == 2, np.logical_not(np.isin(self.previous_z_digitized, [2, 3]))))] = 1
                self.last_functional_bin[np.where(
                    np.logical_and(z_digitized[i] == 4, np.logical_not(np.isin(self.previous_z_digitized, [4, 3]))))] = 5
                self.last_water_bin[np.where(
                    np.logical_and(z_digitized[i] == 2, np.logical_not(np.isin(self.previous_z_digitized, [2, 3]))))] = 1
                self.last_water_bin[np.where(
                    np.logical_and(z_digitized[i] == 4, np.logical_not(np.isin(self.previous_z_digitized, [4, 3]))))] = 5
            for b in self.functional_bins:
                in_b = np.where(z_digitized[i] == b)[0]
                self.last_functional_bin[in_b] = b
                self.framenr_of_last_seen_in_functional_bin[b][in_b] = frame
            self.previous_z_digitized = np.copy(z_digitized[i])

        # finalize
        self.startframe += trajectory.n_frames

    @staticmethod
    def make_critical_transitions():
        severe = np.zeros((7, 7), dtype=bool)
        mild = np.zeros((7, 7), dtype=bool)
        # The only severely critical hop is the one over the central bin
        severe[2, 4] = True
        severe[4, 2] = True
        # Mild warnings: All hops 0 1 [2 3 4] 5 6
        mild[0, 3] = True
        mild[3, 0] = True
        mild[6, 3] = True
        mild[3, 6] = True

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

        return severe, mild

    def permeability(self, permeant_distribution, start_frame=0, end_frame=None,
                     time_between_frames=1*u.picosecond, mode='crossings', num_bins_in_water=2):
        """
        Calculate the permeability.

        Args:
            permeant_distribution (rflow.Distribution): The permeant distribution.
            start_frame (int): Trajectory frame to start counting events.
            end_frame (int or None): Trajectory frame to stop counting events. If None, take last frame.
            time_between_frames (float): Time between trajectory frames in ps.
            mode (str): Type of events on which the permeability calculation is based.
                        Either 'crossings', 'rebounds', or 'semi-permeation'.
            num_bins_in_water (int): Number of bins considered to be in the water phase.

        Returns:
            float: Permeability in cm/s
        """
        return self.calculate_permeability(
            self.events, permeant_distribution, num_permeants=len(self.solute_ids),
            start_frame=start_frame, end_frame=end_frame,
            time_between_frames=time_between_frames, mode=mode,
            num_bins_in_water=num_bins_in_water)

    def permeability_error(self, permeant_distribution, start_frame=0, end_frame=None,
                     time_between_frames=1*u.picosecond, mode='crossings', num_bins_in_water=2, alpha=0.95):
        """
        Calculate the error on the permeability based on the uncertainty in the number of events.
        This error estimate does not take into account the uncertainty in the permeant distribution.
        It may also underestimate the errors, when events are not independent (this is most likely to happen
        for mode='semi-permeation' but depends strongly on the simulation).

        Args:
            permeant_distribution (rflow.Distribution): The permeant distribution.
            start_frame (int): Trajectory frame to start counting events.
            end_frame (int or None): Trajectory frame to stop counting events. If None, take last frame.
            time_between_frames (float): Time between trajectory frames in ps.
            mode (str): Type of events on which the permeability calculation is based.
                        Either 'crossings', 'rebounds', or 'semi-permeation'.
            num_bins_in_water (int): Number of bins considered to be in the water phase.
            alpha(float): The confidence interval (default: 0.95 = 95%)

        Returns:
            A pair of floats:
                - Min. permeability in cm/s
                - Max. permeability in cm/s
        """
        from scipy.stats import poisson
        num_events = self.num_events(self.events,mode=mode, start_frame=start_frame, end_frame=end_frame)
        min_crossings, max_crossings = poisson.interval(alpha, num_events)
        min_p = self.calculate_permeability(
            min_crossings, permeant_distribution, num_permeants=len(self.solute_ids),
            start_frame=start_frame, end_frame=end_frame,
            time_between_frames=time_between_frames, mode=mode,
            num_bins_in_water=num_bins_in_water)
        max_p = self.calculate_permeability(
            max_crossings, permeant_distribution, num_permeants=len(self.solute_ids),
            start_frame=start_frame, end_frame=end_frame,
            time_between_frames=time_between_frames, mode=mode,
            num_bins_in_water=num_bins_in_water)
        return min_p, max_p

    @staticmethod
    def num_events(events, mode='crossings', start_frame=0, end_frame=None):
        """
        The number of events of a certain type.

        Args:
            events (pd.DataFrame or dictionary): The events.
            mode (str): The type of events ('crossings', 'rebounds', or 'semi-permeation')
            start_frame (int): Frame to start counting.
            end_frame (int): Frame to stop counting.

        Returns:
            int: The number of events.
        """
        try:
            [event['type'] for event in events]
        except:
            return int(events)
        if end_frame is None:
            end_frame = np.infty
        counted_events = {'crossings': ['crossing'], 'rebounds': ["crossing", "rebound"],
                          'semi-permeation':["crossing", "rebound", "entry"]}[mode]
        num_events = 0
        for event in events:
            if (event["type"] in counted_events) and (event["frame"] >= start_frame) and (event["frame"] <= end_frame):
                num_events += 1
        return num_events

    @staticmethod
    def calculate_permeability(events, permeant_distribution, num_permeants=None, start_frame=0, end_frame=None,
                               time_between_frames=1*u.picosecond, mode='crossings', num_bins_in_water=2):
        """
        Calculate the permeability.
        Args:
            events (a dict, pd.DataFrame, or integer): The events or the number of events.
            permeant_distribution (rflow.Distribution): The permeant distribution.
            num_permeants (int or None): If None, figure out the number of permeants from the distribution.
            start_frame (int): Trajectory frame to start counting events.
            end_frame (int or None): Trajectory frame to stop counting events. If None, take last frame.
            time_between_frames (float): Time between trajectory frames in ps.
            mode (str): Type of events on which the permeability calculation is based.
                        Either 'crossings', 'rebounds', or 'semi-permeation'.
            num_bins_in_water (int): Number of bins considered to be in the water phase.

        Returns:
            float: Permeability in cm/s
        """
        if end_frame is None:
            end_frame = permeant_distribution.n_frames
        # try to set default arguments
        if num_permeants is None:
            num_permeants = len(permeant_distribution.atom_selection)
        # initialize chosen counting method
        factor = {'crossings': 2.0, 'rebounds': 4.0, 'semi-permeation': 8.0}[mode]
        num_events = PermeationEventCounter.num_events(events, mode, start_frame, end_frame)
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

    @staticmethod
    def calculate_permeability_and_errors(num_events, cw, tsim, area, mode='crossings', alpha=0.95):
        """
        Calculate permeability and confidence interval.
        Args:
            num_events (int): number of events
            cw (float): Equilibrium concentration of permeant in water in #molecules/nm^3
            tsim (float): Simulation time in nanoseconds.
            area (float): Total bilayer cross-section area in nm^2
            mode (str): Type of events on which the permeability calculation is based.
                        Either 'crossings', 'rebounds', or 'semi-permeation'.
            alpha (float): The confidence interval (default: 0.95 = 95%)

        Returns:
            A tuple:
                - float: Permeability
                - float: Lower limit of confidence interval
                - float: Upper limit of confidence interval
        Notes:
            - At some point, I need to clean up this mess and put it into a single permeability function
        """
        from scipy.stats import poisson
        factor = {'crossings': 2.0, 'rebounds': 4.0, 'semi-permeation': 8.0}[mode]
        min_events, max_events = poisson.interval(alpha, num_events)
        perm = lambda n: ((n/(area*(u.nanometer**2) * tsim*u.nanoseconds)/(factor*cw/(u.nanometer**3))).
                          value_in_unit(u.centimeter/u.second))

        return perm(num_events), perm(min_events), perm(max_events)


class PermeationEventCounterWithoutBuffer:
    num_bins = 4
    membrane_bins = [1, 2]
    water_bins = [0, 3]

    def __init__(
            self,
            solute_ids,
            dividing_surface,
            membrane=None,
            initialize_all_permeants=True
    ):
        self.solute_ids = solute_ids
        self.membrane = membrane
        self.initialize_all_permeants = initialize_all_permeants
        self.bins = [0.5-dividing_surface, 0.5, 0.5+dividing_surface]
        self.num_bins = 4
        self.last_visited = -2*np.ones((len(self.solute_ids), self.num_bins), dtype=int)
        self.previous_digitized = None
        self.previous_last_visited = None
        self.startframe = 0
        self.events = []

    def __call__(self, trajectory):
        indices = np.arange(len(self.solute_ids))
        digitized = self._digitize(trajectory)

        # initialize flags for all permeants
        if self.startframe == 0:
            if self.initialize_all_permeants:
                self.last_visited[np.where(digitized[0] == 1), 0] = -1
                self.last_visited[np.where(digitized[0] == 2), 3] = -1

        # count events
        for i in range(trajectory.n_frames):

            # update last visited
            frame = self.startframe + i
            self.last_visited[(indices, digitized[i])] = frame

            if self.previous_digitized is not None:
                # -- jumps over a bin --
                particles_with_jumps = np.where(
                    np.mod(self.previous_digitized - digitized[i], self.num_bins) == 2
                )[0]
                for particle in particles_with_jumps:
                    warn(
                        f"Jump over bin in frame {frame}. Particle {self.solute_ids[particle]}: "
                        f"{self.previous_digitized[particle]}->{digitized[i][particle]}. Continuing "
                        f"with the assumption that it jumped over a water bin."
                    )
                    involved_bins = {self.previous_digitized[particle], digitized[i][particle]}
                    assert involved_bins in [{0,2}, {1,3}]
                    # assume the jump was over a water bin,
                    # since permeation through the membrane is usually much slower
                    bin_jumped_over = 0 if involved_bins == {1,3} else 3
                    # if the jump was an exit, restrain the jump to this bin to make sure that crossings get counted.
                    # This has no consequence in the following steps
                    # unless another jump over membrane bin occurs in the next step (in that case, the frame spacing
                    # is way too wide anyway).
                    if digitized[i][particle] in {0,3}:
                        self.last_visited[particle, bin_jumped_over] = frame
                        self.last_visited[particle, digitized[i, particle]] = -1
                        digitized[i, particle] = bin_jumped_over
                    else:
                        # if the jump was an entry, rewrite the history to make sure that the crossing/rebound
                        # is recognized when the particle exits
                        self.last_visited[particle, bin_jumped_over] = frame - 1
                        self.last_visited[particle, self.previous_digitized[particle]] = frame - 2
                        self.previous_digitized[particle] = bin_jumped_over


                # -- meaningful transitions --
                # first seave out most particles for efficiency
                particles_with_transition = np.where(
                    np.logical_and(
                        np.isin(np.mod(self.previous_digitized - digitized[i], self.num_bins), [1, 3]),
                        np.isin(self.previous_digitized, self.membrane_bins)
                    )
                )[0]
                particles_with_events = {
                    **{i: "entry" for i in particles_with_transition[
                        np.where(self._is_entry(particles_with_transition))[0]
                    ]},
                    **{i: "crossing" for i in particles_with_transition[
                        np.where(self._is_crossing(particles_with_transition))[0]
                    ]},
                    **{i: "rebound" for i in particles_with_transition[
                        np.where(self._is_rebound(particles_with_transition))[0]
                    ]},
                }

                # - handle events --
                for particle, event_type in particles_with_events.items():
                    # -- make event dictionary --
                    last_water = np.argmax(self.previous_last_visited[particle, [0,3]])*3
                    last_time_water = self.previous_last_visited[particle, last_water]
                    atom_id = self.solute_ids[particle]
                    event = {
                        "event_id": len(self.events),
                        "atom": atom_id,
                        "type": event_type,
                        "frame": frame,
                        "from_water": last_water,
                        f"{event_type}_time_nframes": (
                            None if last_time_water < 0 else frame - 1 - last_time_water
                            # the -1 accounts for the average time spend in the source and target bin
                            # before and after crossing the boundary
                        )
                    }
                    if event_type in ["rebound", "crossing"]:
                        # -- get corresponding entry event and exit time --
                        for other_event in reversed(self.events):
                            if other_event["type"] == "entry" and other_event["atom"] == atom_id:
                                event["corresponding_entry_id"] = other_event["event_id"]
                                break
                        otherside_membrane_bin = ( digitized[i][particle] + 2 ) % 4
                        last_otherside = self.previous_last_visited[particle, otherside_membrane_bin]
                        event["exit_time_nframes"] = (
                            None if last_otherside is None else frame - 1 - last_otherside
                            # the -1 accounts for the average time spend in the source and target bin
                            # before and after crossing the boundary
                        )

                    self.events += [event]

            self.previous_digitized = np.copy(digitized[i])
            self.previous_last_visited = np.copy(self.last_visited)
        self.startframe += trajectory.n_frames

    # ----- PRIVATE METHODS ------

    def _digitize(self, trajectory):
        z_normalized = normalize(trajectory, 2, self.membrane, self.solute_ids)
        return np.digitize(z_normalized, self.bins, right=True)

    @staticmethod
    def _is_in_order(array, order):
        in_order = np.ones(array.shape[:-1], dtype=bool)
        for i in range(len(order)-1):
            in_order = np.logical_and(in_order, np.less(array[...,order[i]], array[...,order[i+1]]))
        return in_order

    def _is_entry(self, particles_with_transition):
        return np.logical_or(
            np.logical_and(
                self._is_in_order(self.previous_last_visited[particles_with_transition], [2, 0, 1]),
                self._is_in_order(self.last_visited[particles_with_transition], [0, 1, 2])
            ),
            np.logical_and(
                self._is_in_order(self.previous_last_visited[particles_with_transition], [1, 3, 2]),
                self._is_in_order(self.last_visited[particles_with_transition], [3, 2, 1])
            )
        )

    def _is_crossing(self, particles_with_transition):
        return np.logical_or(
            np.logical_and(
                self._is_in_order(self.previous_last_visited[particles_with_transition], [3, 0, 1, 2]),
                self._is_in_order(self.last_visited[particles_with_transition], [0, 1, 2, 3]),
            ),
            np.logical_and(
                self._is_in_order(self.previous_last_visited[particles_with_transition], [0, 3, 2, 1]),
                self._is_in_order(self.last_visited[particles_with_transition], [3, 2, 1, 0]),
            )
        )

    def _is_rebound(self, particles_with_transition):
        return np.logical_or(
            np.logical_and(
                self._is_in_order(self.previous_last_visited[particles_with_transition], [3, 0, 2, 1]),
                self._is_in_order(self.last_visited[particles_with_transition], [2, 1, 0]),
            ),
            np.logical_and(
                self._is_in_order(self.previous_last_visited[particles_with_transition], [0, 3, 1, 2]),
                self._is_in_order(self.last_visited[particles_with_transition], [1, 2, 3]),
            )
        )


class RegionCrossingCounter:
    def __init__(self, solute_ids, lower_boundary, upper_boundary, membrane=None, initialize_permeants_from=None):
        """

        Args:
            solute_ids:
            lower_boundary:
            upper_boundary:
            membrane:
            initialize_permeants_from: can be None, "upper", and "lower". If None, don't impose a history on the
                particles inside the regions. If "lower", all are assumed to have entered from the lower boundary.
                If "upper", all are assumed to have entered from the upper boundary.
        """
        assert upper_boundary > lower_boundary
        assert len(solute_ids) > 0
        assert initialize_permeants_from in [None, "upper", "lower"]
        self.solute_ids = solute_ids
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.membrane = membrane
        self.initialize_permeants_from = initialize_permeants_from
        self.opposite_center = ((self.lower_boundary + self.upper_boundary)/2 + 0.5) % 1.0
        # make sure that the "opposite center" is not in the region
        assert self.opposite_center < lower_boundary or self.opposite_center > upper_boundary
        self.last_visited = -2*np.ones((len(self.solute_ids), 3), dtype=int)
        self.previous_digitized = None
        self.startframe = 0
        self.num_transitions = np.zeros((3,3), dtype=int)
        self.events = []

    def _digitize(self, trajectory):
        """
        bin 0: below region
        bin 1: in region
        bin 2: above region
        """
        normalized = normalize(trajectory, 2, self.membrane, self.solute_ids)
        if self.opposite_center > self.upper_boundary:
            digitized = np.digitize(normalized, [self.lower_boundary, self.upper_boundary, self.opposite_center], right=True)
            digitized[np.where(digitized == 3)] = 0
        elif self.opposite_center < self.lower_boundary:
            digitized = np.digitize(normalized, [self.opposite_center, self.lower_boundary, self.upper_boundary], right=True)
            digitized = np.array([2,0,1,2])[digitized]
        else:
            raise RickFlowException("Should never reach this part of the code.")
        return digitized

    def __call__(self, trajectory):
        digitized = self._digitize(trajectory)
        indices = np.arange(len(self.solute_ids))

        # initialize flags for all permeants in region
        if self.startframe == 0:
            if self.initialize_permeants_from == "lower":
                self.last_visited[np.where(digitized[0] == 1), 0] = -1
            elif self.initialize_permeants_from == "upper":
                self.last_visited[np.where(digitized[0] == 1), 2] = -1

        # count events
        for i in range(trajectory.n_frames):
            frame = self.startframe + i

            if self.previous_digitized is not None:
                increment_using_multiindices(
                    self.num_transitions,
                    np.column_stack([self.previous_digitized, digitized[i]])
                )
                is_exit = np.logical_and(self.previous_digitized == 1, np.isin(digitized[i], [0,2]))
                is_event = np.logical_and(
                    is_exit,
                    np.less(
                        self.last_visited[(indices, digitized[i])],
                        self.last_visited[(indices, 2-digitized[i])]
                    )
                )
                particle_with_events = np.where(is_event)[0]
                for particle in particle_with_events:
                    event = {
                        "atom": self.solute_ids[particle],
                        "frame": frame,
                        "crossing_time_nframes": (
                                frame - self.last_visited[particle, 2-digitized[i, particle]] - 1
                                if self.last_visited[particle, 2-digitized[i, particle]] >= 0 else None
                        ),
                        "from_lower_boundary": digitized[i, particle]//2
                    }
                    self.events.append(event)

            self.previous_digitized = np.copy(digitized[i])
            self.last_visited[indices, digitized[i]] = frame

        self.startframe += trajectory.n_frames


class Distribution(rflow.observables.Distribution):
    def __init__(self, *args, **kwargs):
        super(Distribution, self).__init__(*args, **kwargs)
        raise DeprecationWarning("The Distribution class has moved from rflow.analyze_diffusion to rflow.observables.")
