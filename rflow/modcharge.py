
import warnings
import numpy as np
from simtk.openmm import NonbondedForce, PeriodicTorsionForce, HarmonicBondForce, HarmonicAngleForce
from rflow.utility import get_force


def scale_subsystem_charges(
        system,
        subsystems,
        lambda_electrostatics
):
    """
    Scale the charges for subparts of the system but retain the original 1-4 interactions in those subparts.
    The system is modified in-place.

    Args:
        system (openmm.System): the system, whose forces are modified in-place
        subsystems (list of iterables): each iterable holds the particle ids that define the subsystem.
            Subsystems must be pairwise disjoint
        lambda_electrostatics (float): the scaling parameter for the charges

    Returns:
        num_added_exceptions (int): The number of 1-4 exceptions that have been added
        num_modified_exceptions (int): The number of 1-4 exceptions that were touched.

    Examples:
        Scaling the charges of a solute by half.

        >>> testsystem = testsystems.CharmmSolvated()
        >>> subsystem = np.arange(27)  # first 27 particles belong to the solute
        >>> modified_system = deepcopy(testsystem.system)  # make a copy to avoid overwriting the original system
        >>> scale_subsystem_charges(modified_system, [subsystem], 0.5)


    Notes:
        This function is built and tested for the CHARMM force field.
        It expects Lorentz-Berthelot mixing rules and fully-coupled 1-4 interactions.
        The NonbondedForce object must not have any offsets in its per-particle and exception parameters.
    """
    # check that subsystems are pairwise disjunct
    for i in range(len(subsystems)):
        for j in range(i):
            assert np.intersect1d(subsystems[i], subsystems[j]).size == 0

    nonbonded_force = get_force(system, [NonbondedForce])
    torsion_force = get_force(system, [PeriodicTorsionForce])
    if torsion_force is None:
        torsion_force = PeriodicTorsionForce()
    bond_forces = [system.getForce(i) for i in range(system.getNumForces())
                   if isinstance(system.getForce(i), HarmonicBondForce)]
    angle_forces = [system.getForce(i) for i in range(system.getNumForces())
                   if isinstance(system.getForce(i), HarmonicAngleForce)]
    assert isinstance(nonbonded_force, NonbondedForce)
    assert isinstance(torsion_force, PeriodicTorsionForce)
    assert nonbonded_force.getNumParticleParameterOffsets() == 0
    assert nonbonded_force.getNumExceptionParameterOffsets() == 0

    # get nonbonded parameters
    charges = []
    sigmas = []
    epsilons = []
    for particle in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle)
        charges.append(charge)
        sigmas.append(sigma)
        epsilons.append(epsilon)
        if any(particle in subsystem for subsystem in subsystems):
            nonbonded_force.setParticleParameters(particle, lambda_electrostatics*charge, sigma, epsilon)

    # get exceptions to make sure that we don't overwrite anything
    exceptions = {}
    for exception in range(nonbonded_force.getNumExceptions()):
        p1, p2, _, _, _ = nonbonded_force.getExceptionParameters(exception)
        exceptions[(p1,p2)] = exception
        exceptions[(p2,p1)] = exception

    # get bonds to make sure that we don't add charges to bonded particles
    bonds = []
    for bond_force in bond_forces: #checking angles would be enough in principle; but who cares...
        for bond in range(bond_force.getNumBonds()):
            p1, p2, _, _ = bond_force.getBondParameters(bond)
            bonds.append((p1,p2))
            bonds.append((p2,p1))
    for angle_force in angle_forces:
        for bond in range(angle_force.getNumAngles()):
            p1, p2, p3, _, _ = angle_force.getAngleParameters(bond)
            bonds.append((p1,p3))
            bonds.append((p3,p1))
            bonds.append((p1,p2))
            bonds.append((p2,p1))
            bonds.append((p2,p3))
            bonds.append((p3,p2))

    # reset 1-4 interactions to original charges
    num_added_exceptions = 0
    num_modified_exceptions = 0
    for torsion in range(torsion_force.getNumTorsions()):
        p1, p2, p3, p4, _, _, _ = torsion_force.getTorsionParameters(torsion)
        for subsystem in subsystems:
            if p1 in subsystem or p4 in subsystem:
                if p1 not in subsystem or p4 not in subsystem:
                    warnings.warn(f"p1 ({p1}) and p4 ({p4}) of torsion {p1}-{p2}-{p3}-{p4} "
                                  f"are not in the same subsystem. Proceed with care.")
                if (p1, p4) in bonds:
                    continue
                if (p1, p4) not in exceptions:
                    num_added_exceptions += 1
                    nonbonded_force.addException(
                        p1, p4,
                        charges[p1]*charges[p4],
                        0.5*(sigmas[p1] + sigmas[p4]),
                        np.sqrt(epsilons[p1] * epsilons[p4])
                    )
                else:
                    exception = exceptions[(p1,p4)]
                    num_modified_exceptions += 1
                    pone, pfour, q, sigma, epsilon = nonbonded_force.getExceptionParameters(exception)
                    nonbonded_force.setExceptionParameters(
                        exception, pone, pfour, charges[p1]*charges[p4], sigma, epsilon
                    )
                    if np.abs(q-charges[p1]*charges[p4]) > 1e-5:
                        warnings.warn("Found different")
    return num_added_exceptions, num_modified_exceptions
