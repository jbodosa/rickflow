
from rflow.nearest import NearestNeighborAnalysis
from rflow.utility import abspath
import mdtraj as md

def test_nearest_neighbor_analysis():
    traj = md.load(abspath('data/psm_dopc_ld.hdf5'))
    psm_chain_atoms = traj.topology.select("resname PSM and ((name =~ 'C.*S') or (name =~'C.*F'))")
    popc_chain_atoms = traj.topology.select("resname POPC and ((name =~ 'C2.*') or (name =~'C3.*'))")
    chol_chain_atoms = traj.topology.select("resname CHL1 and (name =~ 'C.*')")
    membrane = traj.topology.select("resname PSM or resname POPC or resname CHL1")
    water_oxygen = traj.topology.select("water and mass > 15")
    nna = NearestNeighborAnalysis(
        permeants=water_oxygen,
        chains=[psm_chain_atoms, popc_chain_atoms, chol_chain_atoms],
        com_selection=membrane
    )
    nna(traj)