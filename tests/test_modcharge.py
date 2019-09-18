
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
    num_added_exceptions, num_modified_exceptions = scale_subsystem_charges(modified_system, [subsystem], 1.0)

    # solute has 71 dihedrals that are not bonds or angles
    assert num_added_exceptions + num_modified_exceptions == 71

    # test energies (should be equal for scaling = 1.0)
    e1 = get_energy(testsystem.system, testsystem.positions)
    e2 = get_energy(modified_system, testsystem.positions)
    assert e1 == pytest.approx(e2, rel=None, abs=0.02)

    # assert that inverting the solute charges changes the energy unfavorably
    modified_system = deepcopy(testsystem.system)
    scale_subsystem_charges(modified_system, [subsystem], -1)
    e3 = get_energy(modified_system, testsystem.positions)
    assert e3 > e1 + 1

    # assert that we don't hit any strange behavior at 0
    modified_system = deepcopy(testsystem.system)
    scale_subsystem_charges(modified_system, [subsystem], 0)
    e4 = get_energy(modified_system, testsystem.positions)

    modified_system = deepcopy(testsystem.system)
    scale_subsystem_charges(modified_system, [subsystem], 0.005)
    e5 = get_energy(modified_system, testsystem.positions)
    assert e4 == pytest.approx(e5, rel=None, abs=1)

    # test multiple subsystems one of which has no torsions (only test API)
    modified_system = deepcopy(testsystem.system)
    num_added_exceptions, num_modified_exceptions = scale_subsystem_charges(
        modified_system, [subsystem, np.arange(27,30)], 0.005)
    assert num_added_exceptions + num_modified_exceptions == 71
