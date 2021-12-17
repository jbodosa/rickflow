
import numpy as np
import networkx as nx
import openmmtools
from rflow.openmm import unit as u
from rflow.openmm import NonbondedForce, CustomBondForce
from rflow.openmm.app import Topology
from rflow.utility import get_force


def scale_subsystem_charges(
        system,
        topology,
        particle_ids,
        lambda_electrostatics,
        handle_internal_within=10,
        handle_external_beyond=20
):
    """
    Scale the charges for some of the particles in the system but retain the
    original intramolecular Coulombic interactions within a certain distance of the
    molecular graph. Between handle_internal_within and handle_external_beyond,
    the scaling factor is gradually switched from 1 to lambda_electrostatics (linear interpolation).

    The system is modified in-place.

    Args:
        system (openmm.System): the system, whose forces are modified in-place
        topology (openmm.app.Topology): the topology
        particle_ids (iterable of int): particle ids whose charges should be scaled
        lambda_electrostatics (float): the scaling parameter for the charges
        handle_internal_within (int): retain the original interactions for all 1-... interactions
            with ... <= handle_internal_within
        handle_external_beyond (int): scale interactions according to lambda_electrostatics
            for all 1-... interactions with ... > handle_external_beyond


    Returns:
        num_added_interactions (int): The number of internal interactions that have been added
            to make up for the scaled charges.
        num_modified_exceptions (int): The number of 1-4 exceptions that were touched.

    Examples:
        Scaling the charges of a solute by half.

        >>> testsystem = testsystems.CharmmSolvated()
        >>> subsystem = np.arange(27)  # first 27 particles belong to the solute
        >>> modified_system = deepcopy(testsystem.system)  # make a copy to avoid overwriting the original system
        >>> scale_subsystem_charges(modified_system, [subsystem], 0.5)


    Notes:
        The NonbondedForce object must not have any offsets in its per-particle and exception parameters.
    """

    nonbonded_force = get_force(system, [NonbondedForce])
    assert isinstance(nonbonded_force, NonbondedForce)
    assert nonbonded_force.getNumParticleParameterOffsets() == 0
    assert nonbonded_force.getNumExceptionParameterOffsets() == 0
    assert handle_internal_within <= handle_external_beyond


    # === SCALE EXTERNAL ===

    # get nonbonded parameters; scale charges
    charges = []
    sigmas = []
    epsilons = []
    for particle in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle)
        charges.append(charge)
        sigmas.append(sigma)
        epsilons.append(epsilon)
        if particle in particle_ids:
            nonbonded_force.setParticleParameters(particle, lambda_electrostatics * charge, sigma, epsilon)

    # === SCALE INTERNAL ===

    # get exceptions to make sure that we don't overwrite anything
    exceptions = {}
    for exception in range(nonbonded_force.getNumExceptions()):
        p1, p2, _, _, _ = nonbonded_force.getExceptionParameters(exception)
        exceptions[(p1, p2)] = exception
        exceptions[(p2, p1)] = exception

    # Create the bond force to forge the internal charges
    handled_pairs = set()
    internal_electrostatic_force = CustomBondForce("one_over_4pi_eps0 * scaled_charge_prod / r")
    internal_electrostatic_force.addPerBondParameter("scaled_charge_prod")
    internal_electrostatic_force.addGlobalParameter("one_over_4pi_eps0", openmmtools.constants.ONE_4PI_EPS0)

    graph = TopologyGraph(topology)

    num_modified_exceptions = 0
    num_added_interactions = 0
    # scale 1-4 interactions and internal interactions
    for p1 in particle_ids:

        # loop over all connected atoms
        for p2, distance in graph.connected(p1):

            if distance == 0:
                continue

            if (p1, p2) in handled_pairs:
                # add each pair only once
                continue

            if distance < handle_internal_within:
                internal_lambda = 1.0
            elif distance >= handle_external_beyond:
                internal_lambda = lambda_electrostatics
            else:
                # linear interpolation
                alpha = ((distance - (handle_internal_within-1))
                         / (handle_external_beyond - (handle_internal_within-1))
                         )
                # alpha = 1 at distance=handle_external_beyond
                # alpha = 0 at distance=handle_internal_within-1
                internal_lambda = alpha * lambda_electrostatics + (1 - alpha) * 1.0

            if distance <= 3:
                # see if this should be added as an exception
                if (p1,p2) in exceptions:
                    exception_id = exceptions[(p1,p2)]
                    pone, ptwo, chargeprod, sigma, epsilon = (
                        nonbonded_force.getExceptionParameters(exception_id)
                    )
                    assert {pone,ptwo} == {p1, p2}
                    nonbonded_force.setExceptionParameters(
                        exception_id, pone, ptwo, internal_lambda ** 2 * chargeprod, sigma, epsilon
                    )
                    num_modified_exceptions += 1
                    handled_pairs.add((p1, p2))
                    handled_pairs.add((p2, p1))
                    continue

            if distance >= handle_external_beyond:
                continue

            # all forces have to agree on their exceptions, so from here on everything
            # has to be handled through an additional force instance
            # we calculate the scaling factor that gives
            #       scaling_factor  + lambda_ext(p1)*lambda_ext(p2) = lambda_internal^2
            # in order to remove the effect of external scaling from the internal interactions

            lambda_ext_p1 = lambda_electrostatics  # we know that this guy is in the particle ids
            lambda_ext_p2 = lambda_electrostatics if p2 in particle_ids else 1.0
            scaling_factor = internal_lambda**2 - lambda_ext_p1 * lambda_ext_p2
            scaled_charge_prod = scaling_factor * charges[p1] * charges[p2]
            internal_electrostatic_force.addBond(
                int(p1), int(p2), [scaled_charge_prod.value_in_unit(u.elementary_charge**2)]
            )
            num_added_interactions += 1
            handled_pairs.add((p1, p2))
            handled_pairs.add((p2, p1))
    system.addForce(internal_electrostatic_force)

    return num_added_interactions, num_modified_exceptions


class TopologyGraph(nx.Graph):
    """The topology as a graph."""
    def __init__(self, topology: Topology):
        super(TopologyGraph, self).__init__()
        for atom in topology.atoms():
            self.add_node(atom.index)
        for bond in topology.bonds():
            atom1, atom2 = bond.atom1.index, bond.atom2.index
            self.add_edge(atom1, atom2)
        self.distances = dict(nx.all_pairs_shortest_path_length(self))

    def connected(self, atom_id):
        """
        Iterate over atoms that are connected with the atom_id.

        Args:
            atom_id (int):

        Yields:
            atom2 (int): id of the connected atom
            distance (int): the number of bonds between the two atoms
        """
        for atom2 in self.distances[atom_id]:
            yield atom2, self.distances[atom_id][atom2]
