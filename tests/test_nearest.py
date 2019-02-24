
from rflow.nearest import NearestNeighborAnalysis

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
        num_z_bins=10
    )
    nna(traj)
    return nna


def test_nna(nna):
    assert nna.counts.sum() == nna.num_permeants # one frame, two permeant molecules
    # most likely, water will be closest to water
    assert nna.counts[:,0,0,0].sum() == nna.num_permeants


def test_probabilities(nna):
    assert nna.probabilities.shape == nna.counts.shape


def test_nna_save_load_eq(nna, tmpdir):
    filename = os.path.join(str(tmpdir), 'nna.txt')
    nna.save(filename)
    assert os.path.isfile(filename)
    assert os.path.isfile(filename + ".pic")
    nna2 = NearestNeighborAnalysis.from_pickle_file(filename + ".pic")
    assert nna == nna2
    nna2.counts[0,0,0,0] += 1
    assert nna != nna2


def text_inherited_quantities(nna, traj):
    assert hasattr(nna, 'edges')
    zlength = traj.unitcell_lengths[0,2]
    assert len(nna.edges) == nna.num_z_bins + 1
    assert nna.average_box_size == zlength


def test_add(nna):
    nna2 = deepcopy(nna)
    nna2.counts[0,0,0,0] += 1
    nna2.n_frames += 1
    nna_sum = nna + nna2
    assert np.array_equal(nna_sum.counts, nna.counts + nna2.counts)
    # check if box size was updated
    assert nna_sum.n_frames == 3
    assert nna_sum.average_box_size == pytest.approx(
        1./3. * (nna.average_box_size + 2 * nna2.average_box_size)
    )


def test_sum(nna):
    nna2 = deepcopy(nna)
    sum([nna, nna2])