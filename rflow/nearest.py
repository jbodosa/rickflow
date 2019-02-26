"""
This module contains a neighbor analysis for permeants through a heterogeneous lipid bilayer.
"""

import pickle
from copy import copy

import numpy as np
import mdtraj as md

from rflow import normalize, BinEdgeUpdater, RickFlowException


class NearestNeighorException(RickFlowException):
    pass


class NearestNeighborResult(BinEdgeUpdater):
    """Result of a nearest neighbor analysis. This object is created by the NearestNeighborAnalysis class
    and can usually be accessed as NearestNeighborAnalysis.result.

    This class supports addition, testing for equality, saving and loading results, converting counts to probabilities,
    as well as reducing the number of bins:
    - Adding two instances will sum up the counts and n_frames, and average the box size.

    Properties:
        probabilities (numpy.array): The probability of finding a certain neighbor configuration in a bin,
            divided by the probability of finding a permeant in the bin (probabilities sum up to 1 in each bin).
    """
    def __init__(self, num_bins, average_box_size, n_frames, counts):
        """
        Args:
            num_bins (int): number of bins in the z direction (bilayer normal)
            average_box_size(float): average box height in the z direction
            n_frames (int): number of trajectory frames that were analyzed to get this result
            counts (numpy array): the number of tuples of neighbors for each class of permeants and bin in
                the z-direction. E.g., the element `counts[4][2][2][0]` stores the number of occurences,
                where the three nearest residues of a permeant in bin nr. 4 were of chaintype 2,2, and 0.
        """
        # initialize the parent object that keeps track of bin centers and bin edges
        super(NearestNeighborResult, self).__init__(num_bins, coordinate=2)
        self.average_box_size = average_box_size
        self.n_frames = n_frames
        self.counts = counts
        shape = counts.shape
        self.num_neighbors = len(shape) - 1
        assert all(shape[i] == shape[1] for i in range(2, self.num_neighbors))
        self.num_chains = shape[1]

    def __add__(self, other):
        assert self.num_bins == other.num_bins
        assert self.counts.shape == other.counts.shape
        n_frames = self.n_frames + other.n_frames
        average_box_size = (other.n_frames * other.average_box_size + self.n_frames * self.average_box_size) / n_frames
        return NearestNeighborResult(self.num_bins, average_box_size, n_frames, self.counts + other.counts)

    def __radd__(self, other):
        if other:
            return self + other
        else:
            return self

    def __iadd__(self, other): # enables usage of sum(...)
        if other:
            assert self.counts.shape == other.counts.shape
            self.counts += other.counts
            self.average_box_size = (
                    1.0 / (self.n_frames+other.n_frames) *
                    (self.n_frames * self.average_box_size
                     + other.n_frames * other.average_box_size)
                    )
            self.n_frames = self.n_frames + other.n_frames
            return self
        else:
            return self

    def __eq__(self, other):
        if not abs(other.average_box_size - self.average_box_size) < 1e-6: return False
        if other.num_bins != self.num_bins: return False
        if other.n_frames != self.n_frames: return False
        if not np.array_equal(other.counts, self.counts): return False
        return True

    @staticmethod
    def from_file(filename):
        """Load a result from file
        Args:
            filename (str): The file that contains the data.

        Returns:
            A NearestNeighborResult instance.
        """
        with open(filename, 'r') as fp:
            header_line_2 = fp.readlines()[1].replace('#', '').strip()
            header_dictionary = eval(header_line_2)
        num_bins = header_dictionary['num_bins']
        num_chains = header_dictionary['num_chains']
        num_neighbors = header_dictionary['num_neighbors']
        average_box_size  = header_dictionary['average_box_size']
        n_frames  = header_dictionary['n_frames']
        counts_inline = np.loadtxt(filename)[:, 1:]
        shape = tuple([num_bins] + [num_chains] * num_neighbors)
        counts_reshaped = np.reshape(counts_inline, shape)
        return NearestNeighborResult(num_bins,  average_box_size, n_frames, counts_reshaped)

    def save(self, filename):
        """Save this result to a (human-readable) file.

        Args:
            filename (str): The filename to write to.
        """
        header_dictionary = {
            'num_bins': self.num_bins,
            'average_box_size': self.average_box_size,
            'n_frames': self.n_frames,
            'num_neighbors': self.num_neighbors,
            'num_chains': self.num_chains
        }
        labels = 'z '
        for index in np.ndindex(*([self.num_chains] * self.num_neighbors)):
            labels += (''.join(str(i) for i in index) + ' ')
        header = "Result of a nearest-neighbor analysis with \n" + str(header_dictionary) + '\n' + labels
        counts_inline = self.counts.reshape(self.num_bins, self.num_chains**self.num_neighbors)

        np.savetxt(filename, np.column_stack([self.bin_centers_around_zero, counts_inline]),
                   fmt="%.6f " + "%i " * (self.num_chains ** self.num_neighbors), header=header)

    @property
    def probabilities(self):
        probabilities = self.counts
        normalization = np.sum(probabilities, axis=tuple(np.arange(1, 1+self.num_neighbors)))
        probabilities = (probabilities.T / normalization).T  # transpose, as division is over last axis
        return probabilities

    def coarsen(self, num_bins):
        """Coarsen the bin discretization.

        Args:
            num_bins (int): The number of bins of the new instance.
                Has to be a divisor of this instance's number of bins.

        Returns:
            A new NearestNeighborResult instance with the specified number of bins.
        """
        if not self.num_bins % num_bins == 0:
            raise NearestNeighorException("num_bins has to be a divisor of this result's number of bins.")
        #                            retain shape of all but bin axis  ----   last axis is going to be summed over
        transposed_new_shape = tuple(list(self.counts.shape[-1:0:-1]) + [num_bins, self.num_bins // num_bins])
        coarsened_counts = np.sum(np.reshape(self.counts.T, transposed_new_shape).T, axis=0)
        return NearestNeighborResult(
            num_bins, self.average_box_size, self.n_frames, coarsened_counts
        )


class NearestNeighborAnalysis(BinEdgeUpdater):
    """This class runs a very specific nearest neighbor analysis in a heterogeneous membrane.

    For some permeant molecules, it finds the nearest neighboring lipid chain atoms.
    For example, for a bilayer containing PSM, POPC, and cholesterol, it figures out the closest
    chain atoms.

    The analysis is run by calling this class as a function with an mdtraj trajectory as an argument.

    Properties:
        result (instance of NearestNeighborResult): Returns the result in a form that can be saved, added, etc.


    """
    def __init__(self, permeants, chains, num_residues_per_chain, num_bins=120,
                 num_neighbors=3, com_selection=None):
        """
        Args:
            permeants(list or np.array of int): List of the permeant's atom ids.
            chains(list (or np.array) of list (or np.array) of int): Each item of the list is a list of lipid atom ids.
                Each list specifies a different chain type.
            num_residues_per_chain (list of int): Number of residues per chain for each chain type. Has to have the
                same length as `chains`.
            num_bins (int): The number of bins in the z direction.
            num_neighbors (int): The number of nearest neighbors that should be stored for each permeant.
            com_selection (list or np.array of int): All membrane atom ids, to define the membrane's center of mass.
        """
        # input arguments
        self.permeants = permeants
        self.chains = chains
        self.num_residues_per_chain = num_residues_per_chain
        self.num_neighbors = num_neighbors
        self.com_selection = com_selection

        # initialize the parent object that keeps track of bin centers and bin edges
        super(NearestNeighborAnalysis, self).__init__(num_bins, coordinate=2)

        # derived quantities
        bin_width = 1.0 / num_bins
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
        self.counts = np.zeros([num_bins] + [self.num_chains] * num_neighbors, dtype=int)

    @property
    def result(self):
        return NearestNeighborResult(self.num_bins, self.average_box_size,
                                     self.n_frames, self.counts)

    def __call__(self, traj):
        for i in range(traj.n_frames):
            frame = traj.slice([i])
            self.call_on_frame(frame)

    def call_on_frame(self, frame):
        """Process the nearest neighbor analysis for one frame.

        Args:
            frame: An mdtraj trajectory containing one frame.
        """
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
        # find the minimum distance of each permeant to each chain residue
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

    @staticmethod
    def cartesian_product(*arrays):
        """Helper function that returns the cartesian product of multiple arrays.

        Args:
            *arrays: Any number of arrays.

        Returns:
            cartesian_product (numpy.array): The cartesian product of the input arrays.
        """
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    @staticmethod
    def increment_using_multiindices(array, index_array):
        """Increment the array by 1 at all multiindices defined in index_array.

        Args:
            array (numpy.array): A (possibly highdimensional) array
            index_array (numpy.array): A two-dimensional array, whose rows specify multiindices.

        Returns:
            incremented_array (numpy.array): A copy of the input array, where 1 has been added at each index from
                the index_array.
        """
        unfolded_array = np.ravel(array)
        unfolded_indices = np.ravel_multi_index(index_array.T, array.shape)
        np.add.at(unfolded_array, unfolded_indices, 1)
        return np.reshape(unfolded_array, array.shape)
