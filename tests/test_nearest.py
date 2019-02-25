
from rflow.nearest import NearestNeighborAnalysis, NearestNeighborResult, NearestNeighorException

import os
from copy import deepcopy
import numpy as np
import mdtraj as md

from rflow.utility import abspath

import pytest


@pytest.fixture(scope='module')
def traj():
    return md.load(abspath('data/psm_dopc_ld.hdf5'))


@pytest.fixture(scope='module')
def nna(traj):
    psm_chain_atoms = traj.topology.select("resname PSM and ((name =~ 'C.*S') or (name =~'C.*F'))")
    popc_chain_atoms = traj.topology.select("resname POPC and ((name =~ 'C2.*') or (name =~'C3.*'))")
    chol_chain_atoms = traj.topology.select("resname CHL1 and (name =~ 'C.*')")
    membrane = traj.topology.select("resname PSM or resname POPC or resname CHL1")
    water_oxygen = traj.topology.select("water and mass > 15")
    nna = NearestNeighborAnalysis(
        permeants=water_oxygen[:5], # for testing only few atoms
        chains=[water_oxygen, psm_chain_atoms, popc_chain_atoms, chol_chain_atoms],
        num_residues_per_chain=[20487, 128, 386, 44],
        com_selection=membrane,
        num_bins=10
    )
    nna(traj)
    return nna


def test_nna(nna):
    assert nna.counts.sum() == nna.num_permeants # one frame, two permeant molecules
    # most likely, water will be closest to water
    assert nna.counts[:,0,0,0].sum() == nna.num_permeants


def test_probabilities(nna):
    assert nna.result.probabilities.shape == nna.counts.shape


def test_nna_save_load_eq(nna, tmpdir):
    filename = os.path.join(str(tmpdir), 'nna.txt')
    nna.result.save(filename)
    assert os.path.isfile(filename)
    nnares2 = NearestNeighborResult.from_file(filename)
    assert nna.result == nnares2
    nnares2.counts[0,0,0,0] += 1
    assert nna != nnares2


def text_inherited_quantities(nna, traj):
    assert hasattr(nna, 'edges')
    zlength = traj.unitcell_lengths[0,2]
    assert len(nna.edges) == nna.num_bins + 1
    assert nna.average_box_size == zlength


def test_radd(nna):
    nnares1 = nna.result
    nnares2 = deepcopy(nna.result)
    nnares2.counts[0,0,0,0] += 1
    nnares2.n_frames += 1
    nna_sum = deepcopy(nnares1)
    nna_sum += nnares2
    assert np.array_equal(nna_sum.counts, nnares1.counts + nnares2.counts)
    # check if box size was updated
    assert nna_sum.n_frames == 3
    assert nna_sum.average_box_size == pytest.approx(
        1./3. * (nnares1.average_box_size + 2 * nnares2.average_box_size)
    )


def test_sum(nna):
    nnares1 = nna.result
    nnares2 = deepcopy(nna.result)
    nnares2.counts[0,0,0,0] += 1
    nnares2.n_frames += 1
    nna_sum = sum([nnares1, nnares2])
    assert np.array_equal(nna_sum.counts, nnares1.counts + nnares2.counts)
    # check if box size was updated
    assert nna_sum.n_frames == 3
    assert nna_sum.average_box_size == pytest.approx(
        1./3. * (nnares1.average_box_size + 2 * nnares2.average_box_size)
    )


def test_coarsen(nna):
    nnr_fine = nna.result
    with pytest.raises(NearestNeighorException):
        nnr_fine.coarsen(11)
    nnr_coarse = nna.result.coarsen(2)
    assert nnr_coarse.counts.shape == (2, 4, 4, 4)
    assert nnr_coarse.counts[0,0,0,0] == np.sum(nnr_fine.counts[:5,0,0,0])