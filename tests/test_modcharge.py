
import pytest
from copy import deepcopy
import numpy as np
from simtk.openmm import Context, VerletIntegrator
from simtk import unit as u
from openmmtools import testsystems
from rflow.modcharge import scale_subsystem_charges


def get_energy(system, positions):
    ctx = Context(system, VerletIntegrator(1.0))
    ctx.setPositions(positions)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)


@pytest.mark.skipif(not hasattr(testsystems, "CharmmSolvated"),
                    reason="test requires a version of openmmtools that has the CharmmSolvated testsystem")
def test_scale_charges():
    testsystem = testsystems.CharmmSolvated()
    # first 27 particles belong to the solute
    subsystem = np.arange(27)
    modified_system = deepcopy(testsystem.system)
    num_added_exceptions, num_modified_exceptions = scale_subsystem_charges(
        modified_system, testsystem.topology, subsystem, 1.0,
        handle_internal_within=4,
        handle_external_beyond=4
    )

    # solute has 71 dihedrals that are not bonds or angles
    assert num_modified_exceptions == 57
    assert num_added_exceptions == 0

    # test energies (should be equal for scaling = 1.0)
    e1 = get_energy(testsystem.system, testsystem.positions)
    e2 = get_energy(modified_system, testsystem.positions)
    assert e1 == pytest.approx(e2, rel=None, abs=0.02)

    # assert that inverting the solute charges changes the energy unfavorably
    modified_system = deepcopy(testsystem.system)
    scale_subsystem_charges(modified_system, testsystem.topology, subsystem, -1)
    e3 = get_energy(modified_system, testsystem.positions)
    assert e3 > e1 + 1

    # assert that we don't hit any strange behavior at 0
    modified_system = deepcopy(testsystem.system)
    scale_subsystem_charges(modified_system, testsystem.topology, subsystem, 0)
    e4 = get_energy(modified_system, testsystem.positions)

    modified_system = deepcopy(testsystem.system)
    scale_subsystem_charges(modified_system, testsystem.topology, subsystem, 0.005)
    e5 = get_energy(modified_system, testsystem.positions)
    assert e4 == pytest.approx(e5, rel=None, abs=1)


@pytest.mark.parametrize("create_exceptions", [True, False])
def test_internal_electrostatic_correction(create_exceptions):
    from simtk.openmm import System, NonbondedForce
    from simtk.openmm.app import Topology, Chain, Residue, Atom
    system = System()
    topology = Topology()
    chain = topology.addChain(0)
    residue = topology.addResidue("R", chain)
    nonbonded_force = NonbondedForce()
    atoms = []
    bonds = []
    for i in range(6):
        system.addParticle(1.0)
        nonbonded_force.addParticle(1.0 if i % 2 == 0 else -1.0, 1.0, 0.0)
        atoms.append(topology.addAtom(f"p{i}", element=None, residue=residue))
    for i in range(5):
        topology.addBond(atoms[i],atoms[i+1])
        bonds.append([i, i+1])
    if create_exceptions:
        nonbonded_force.createExceptionsFromBonds(bonds, 1, 1)
    system.addForce(nonbonded_force)
    positions = np.array([[i*0.01, 0.0, 0.0] for i in range(6)])
    energy1 = get_energy(system, positions=positions)

    # check that scaling charges has no effect (because all interactions are internal)
    modified_system = deepcopy(system)
    scale_subsystem_charges(modified_system, topology, range(6), 0.1)
    energy2 = get_energy(modified_system, positions=positions)
    assert energy1 == pytest.approx(energy2, rel=None, abs=1e-2)

    # check that scaling charges has no effect (all interactions are bonded and handled internally within 10)
    modified_system = deepcopy(system)
    scale_subsystem_charges(modified_system, topology, range(2), 0.1)
    energy3 = get_energy(modified_system, positions)
    assert energy1 == pytest.approx(energy3, rel=None, abs=1e-2)

    # check that scaling charges has an effect (1-5 and 1-6 are handled hybrid; linearly scaled)
    modified_system = deepcopy(system)
    scale_subsystem_charges(modified_system, topology, range(2), 0.1, handle_internal_within=4)
    energy4 = get_energy(modified_system, positions)
    assert energy1 == pytest.approx(energy2, rel=None, abs=1e-2)
    assert abs(energy1 - energy4) > 1




