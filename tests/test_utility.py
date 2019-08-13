
import numpy as np
import pytest

import mdtraj as md
from simtk import unit as u
from simtk.openmm import VerletIntegrator, Context, System
from simtk.openmm.app import CharmmPsfFile

from rflow.utility import increment_using_multiindices, read_input_coordinates, abspath


def test_increment():
    a = np.zeros((2,2), dtype=int)
    res = increment_using_multiindices(a, np.array([[0,0], [1,1], [1,1]], dtype=int))
    assert np.array_equal(res, np.array([[1,0], [0,2]], dtype=int))


@pytest.mark.parametrize("frame", (0, -1))
@pytest.mark.parametrize(
    "trajectory_file,topology_file",
    [
        [np.zeros((31,3)), abspath("data/m2a.psf")],
        [np.zeros((31,3))*u.angstrom, abspath("data/m2a.psf")],
        [abspath("data/m2a.mp2_opt.crd"), abspath("data/m2a.psf")],
        [abspath("data/m2a.mp2_opt.cor"), abspath("data/m2a.psf")],
        [abspath("data/wat.m2a.pdb"), abspath("data/wat.m2a.psf")],
        [abspath("data/ord2.dcd"), abspath("data/ord+o2.psf")],
    ]
)
def test_read_input_coordinates(trajectory_file, topology_file, frame):
    """check if input coordinates can be read from various sources"""
    psf = CharmmPsfFile(topology_file)
    topology = md.Topology.from_openmm(psf.topology)
    for top in [topology, topology_file]:
        pos = read_input_coordinates(trajectory_file, top)
        pos = read_input_coordinates(trajectory_file, top, frame)
        # check if context can be created
        system = System()
        for _ in topology.atoms:
            system.addParticle(1.0)
        context = Context(system, VerletIntegrator(1.0))
        context.setPositions(pos)


