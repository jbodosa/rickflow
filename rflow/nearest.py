"""

"""

from copy import deepcopy
import pickle

import numpy as np
import mdtraj as md

from rflow import normalize, BinEdgeUpdater, RickFlowException


class NearestNeighorException(RickFlowException):
    pass


class NearestNeighborAnalysis(BinEdgeUpdater):
    """This class runs a very specific nearest neighbor analysis in a heterogeneous membrane.

    For some permeant molecules, it finds the nearest neighboring lipid chain atoms.
    For example, for a bilayer containing PSM, POPC, and cholesterol, it figures out the closest
    chain atoms.
    """

    def __init__(self, permeants, chains, num_residues_per_chain, num_z_bins=50,
                 num_neighbors=3, com_selection=None):
        """
        """
        # input arguments
        self.permeants = permeants
        self.chains = chains
        self.num_residues_per_chain = num_residues_per_chain
        self.num_z_bins = num_z_bins
        self.num_neighbors = num_neighbors
        self.com_selection = com_selection

        # initialize the parent object that keeps track of bin centers and bin edges
        super(NearestNeighborAnalysis, self).__init__(num_z_bins, coordinate=2)

        # derived quantities
        bin_width = 1.0 / num_z_bins
        self.bins = np.arange(bin_width, 1.0 + bin_width, bin_width)
        self.num_permeants = len(self.permeants)
        self.num_chains = len(self.chains)
        assert len(num_residues_per_chain) == len(chains)
        assert all([len(chains[i]) % num_residues_per_chain[i] == 0 for i in range(len(chains))])
        self.num_atoms_per_residue = [len(chains[i]) // num_residues_per_chain[i] for i in range(len(chains))]
        self.total_num_residues = sum(self.num_residues_per_chain)

        self.chain_atoms = np.concatenate(chains)
        self.num_chain_atoms = len(self.chain_atoms)
        self.num_atoms_per_chain = [self.num_residues_per_chain[i] * self.num_atoms_per_residue[i]
                                    for i in range(self.num_chains)]

        # build a lookup-table that gives the chain id for each residue_id, where
        # residue_id is a residue's index (return -1, if residue_id is not a valid residue)
        self.chain_id = np.zeros(self.total_num_residues, dtype=int)
        self.chain_id[:] = -1
        for i in range(self.num_chains):
            self.chain_id[sum(self.num_residues_per_chain[:i]):sum(self.num_residues_per_chain[:i+1])] = i

        # build all pairs of permeant and chain atoms [id of a permeant atom, id of a chain atom]
        # because we need to calculate the distance between these pairs for every trajectory frame
        self.permeant_chain_pairs = self.cartesian_product(np.array(self.permeants), self.chain_atoms)

        # build an array that stores the results. The results are stored in the form of a multidimensional
        # array, where counts[z_index, chain_index1, chain_index2, ...] refers to the number of permeant atoms
        # in the z_index-th bin, whose closest neighbors belong to the chains with chain_index1, chain_index2, etc.
        # e.g., counts[0,1,1,1] denots the number of occurences, where the nearest neighbors of a permeant atom
        # in the first bin all belonged to chain id 1
        self.counts = np.zeros([num_z_bins] + [self.num_chains] * num_neighbors, dtype=int)

    def __call__(self, traj):
        for i in range(traj.n_frames):
            frame = traj.slice([i])
            self.call_on_frame(frame)

    def call_on_frame(self, frame):
        # update bin centers and bin edges
        super(NearestNeighborAnalysis, self).__call__(frame)
        # normalize z axis and categorize z coordinates into bins
        z_normalized = normalize(frame, com_selection=self.com_selection,
                                 coordinates=2, subselect=self.permeants)
        z_digitized = np.digitize(z_normalized, self.bins)
        # compute interatomic distances between permeant and chain atoms
        distances = md.compute_distances(frame, self.permeant_chain_pairs, periodic=True, opt=True)
        # refold so that distances can be accessed using [permeant_id, chain_atom_id], where
        # permeant_id enumerates the permeant atoms starting at 0, 1, ...
        # and chain_atom_id enumerates all chain atoms starting at 0, 1, ...
        distances_reshaped = distances.reshape((self.num_permeants, self.num_chain_atoms))
        distances_to_residue_in_chain = []
        for i in range(self.num_chains):
            distances_per_residue = distances_reshaped[
                                    :, sum(self.num_atoms_per_chain[:i]):sum(self.num_atoms_per_chain[:i+1])].reshape(
                (self.num_permeants, self.num_residues_per_chain[i], self.num_atoms_per_residue[i]))
            distances_to_residue_in_chain.append(np.min(distances_per_residue, axis=2))
        distances_to_residues = np.column_stack(distances_to_residue_in_chain)

        # partition the residue indices in each row (i.e. for each permeant atom)
        # partitioning means that the k-th smallest number in the array will be at position k
        # in the partitioned array. All smaller numbers are left, all larger numbers right
        # of position k. The function np.argpartition returns the indices that the array
        # would have after partitioning, not the partitioned array itself (like argsort vs. sort).
        partitioned_residue_indices = np.argpartition(distances_to_residues, axis=1, kth=self.num_neighbors)
        # get the indices of the closest residues
        k_nearest_residue_indices = partitioned_residue_indices[:, :self.num_neighbors]
        # get the chain ids for the closest residues
        k_nearest_chains = self.chain_id[k_nearest_residue_indices]
        # sorting assures that, e.g., indices [0,0,1] and [1,0,0], are treated as equivalent
        k_nearest_chains.sort(axis=1)
        # building the indices, for which self.counts is going to be incremented.
        # each index is an array [z_index, chain_index1, chain_index2, ... ],
        # where chain_index1 <= chain_index2 <= ...
        count_indices = np.column_stack([z_digitized.T, k_nearest_chains])
        self.counts = self.increment_using_multiindices(self.counts, count_indices)

    def __add__(self, other):
        try:
            assert np.array_equal(other.permeants, self.permeants)
            assert np.array_equal(np.concatenate(other.chains), np.concatenate(self.chains))
            assert other.num_z_bins == self.num_z_bins
            assert np.array_equal(other.com_selection, self.com_selection)
        except AssertionError:
            raise NearestNeighorException("For adding nearest neighbor analysis instances, they "
                                          "have to share the same permeants, chains, num_z_bins, "
                                          "and com_selection.")
        assert self.counts.shape == other.counts.shape
        result = deepcopy(self)
        result.counts += other.counts
        result.n_frames = self.n_frames + other.n_frames
        result.average_box_size = (1.0/result.n_frames *
            (self.n_frames * self.average_box_size + other.n_frames * other.average_box_size)
        )
        return result

    def __radd__(self, other): # enables usage of sum(...)
        if other:
            return other + self
        else:
            return self

    def __eq__(self, other):
        if not abs(other.average_box_size - self.average_box_size) < 1e-6: return False
        if not np.array_equal(other.permeants, self.permeants): return False
        if not np.array_equal(np.concatenate(other.chains), np.concatenate(self.chains)): return False
        if other.num_z_bins != self.num_z_bins: return False
        if not np.array_equal(other.com_selection, self.com_selection): return False
        if not np.array_equal(other.counts, self.counts): return False
        return True

    def save(self, filename):
        probabilities = self.probabilities
        probabilities_iii = [self.bin_centers_around_zero]
        for i in range(self.num_chains):
            probabilities_iii.append(probabilities.T[tuple(i for _ in range(self.num_neighbors))])
        probabilities_iii.append(-np.sum(np.array(probabilities_iii)[1:], axis=0) + 1.0)
        header = ("{:<9}" * (self.num_chains+2)).format(
            *(['z'] + ['{}xChain{}'.format(self.num_neighbors, i) for i in range(self.num_chains)] + ['other']))
        np.savetxt(filename, np.column_stack(probabilities_iii), fmt="%.6f",
                   header='Probabilities for nearest neighbors\n' + header)
        with open(filename + '.pic', 'wb') as fp:
            pickle.dump(self, fp)

    @property
    def probabilities(self):
        probabilities = self.counts
        normalization = np.sum(probabilities, axis=tuple(np.arange(1, 1+self.num_neighbors)))
        probabilities = (probabilities.T / normalization).T  # transpose, as division is over last axis
        return probabilities

    @staticmethod
    def from_pickle_file(filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    @staticmethod
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    @staticmethod
    def increment_using_multiindices(array, index_array):
        """
        """
        unfolded_array = np.ravel(array)
        unfolded_indices = np.ravel_multi_index(index_array.T, array.shape)
        np.add.at(unfolded_array, unfolded_indices, 1)
        return np.reshape(unfolded_array, array.shape)