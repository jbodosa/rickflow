
from rflow.nearest import NearestNeighborAnalysis, NearestNeighborResult, NearestNeighorException

import os
from copy import deepcopy
import warnings
import numpy as np
import mdtraj as md
from collections import Counter

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
    assert nna.result.coarsen(2).probabilities.shape[1:] == nna.counts.shape[1:]


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


def test_example():
    """Test the neighbor counting against a slow but more-reliable version."""
    # setup system
    traj = md.load_dcd(abspath("data/ord2.dcd"), top=abspath("data/ord+o2.psf"))
    top = traj.topology
    # select atoms
    chainatoms = "((name =~ 'C2.*') or (name =~ 'C3.*'))"
    atom_ids = dict()
    atom_ids["O2"] = top.select("resname O2")
    atom_ids["DPPC"] = top.select("resname DPPC and " + chainatoms)
    atom_ids["DOPC"] = top.select("resname DOPC and " + chainatoms)
    atom_ids["CHL1"] = top.select("resname CHL1 and (name =~ 'C.*')")
    # count residues
    residues = Counter()
    for residue in top.residues:
        residues.update([residue.name])
    assert all(len(atom_ids[a]) > 0 for a in atom_ids)
    assert all(len(atom_ids[a]) % residues[a] == 0 for a in atom_ids)
    atoms_in_residue = {a: len(atom_ids[a]) // residues[a] for a in atom_ids}
    # for each o2: calculate distances and find closest residues
    closest = {"DPPC": 0, "DOPC": 0, "CHL1": 0, "MIX": 0}
    closest_atoms = {"DPPC": 0, "DOPC": 0, "CHL1": 0, "MIX": 0}
    closest_chains = {"DPPC": 0, "DOPC": 0, "CHL1": 0, "MIX": 0}
    for frame in traj:
        for o2 in atom_ids["O2"]:
            # find closest pairs
            pairlist = []
            lipid = []
            lipid_id = []
            for type in ["DPPC", "DOPC", "CHL1"]:
                for i, a in enumerate(atom_ids[type]):
                    pairlist.append([o2, a])
                    lipid.append(type)
                    lipid_id.append(i // atoms_in_residue[type])
            distances = md.compute_distances(frame, pairlist)[0]
            sorted_indices = np.argsort(distances)
            sorted_distances = distances[sorted_indices]
            assert (sorted_distances == sorted(distances)).all()
            sorted_lipid = np.array(lipid)[sorted_indices]
            sorted_lipid_id = np.array(lipid_id)[sorted_indices]
            i = 0
            # 1: find closest residues
            close = []
            while len(close) < 3:
                if not [sorted_lipid[i], sorted_lipid_id[i]] in close:
                    close.append([sorted_lipid[i], sorted_lipid_id[i]])
                i += 1
            if all(lip == "DPPC" for lip, _ in close): closest["DPPC"] += 1
            elif all(lip == "DOPC" for lip, _ in close): closest["DOPC"] += 1
            elif all(lip == "CHL1" for lip, _ in close): closest["CHL1"] += 1
            else: closest["MIX"] += 1
            # 2: find closest atoms (regardless of residue)
            close_atoms = sorted_lipid[:3]
            if all(lip == "DPPC" for lip in close_atoms): closest_atoms["DPPC"] += 1
            elif all(lip == "DOPC" for lip in close_atoms): closest_atoms["DOPC"] += 1
            elif all(lip == "CHL1" for lip in close_atoms): closest_atoms["CHL1"] += 1
            else: closest_atoms["MIX"] += 1

    # compare these reference solutions with the code
    nna = NearestNeighborAnalysis(
        permeants=atom_ids["O2"], # for testing only few atoms
        chains=[atom_ids["DPPC"], atom_ids["DOPC"], atom_ids["CHL1"]],
        num_residues_per_chain=[residues["DPPC"], residues["DOPC"], residues["CHL1"]],
        com_selection=atom_ids["DPPC"],
        num_bins=1
    )
    nna(traj)
    counts = nna.result.counts

    assert counts[0,0,0,0] == closest["DPPC"]
    assert counts[0,1,1,1] == closest["DOPC"]
    assert counts[0,2,2,2] == closest["CHL1"]
    assert np.sum(counts)-counts[0,0,0,0]-counts[0,1,1,1]-counts[0,2,2,2] == closest["MIX"]

    nna_atoms = NearestNeighborAnalysis(
        permeants=atom_ids["O2"], # for testing only few atoms
        chains=[atom_ids["DPPC"], atom_ids["DOPC"], atom_ids["CHL1"]],
        num_residues_per_chain=[len(atom_ids["DPPC"]), len(atom_ids["DOPC"]), len(atom_ids["CHL1"])],
        com_selection=atom_ids["DPPC"],
        num_bins=1
    )
    nna_atoms(traj)
    counts_atoms = nna_atoms.result.counts

    assert counts_atoms[0,0,0,0] == closest_atoms["DPPC"]
    assert counts_atoms[0,1,1,1] == closest_atoms["DOPC"]
    assert counts_atoms[0,2,2,2] == closest_atoms["CHL1"]
    assert np.sum(counts_atoms)-counts_atoms[0,0,0,0]-counts_atoms[0,1,1,1]-counts_atoms[0,2,2,2] == closest_atoms["MIX"]







