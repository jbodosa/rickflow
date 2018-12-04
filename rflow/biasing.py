# -*- coding: utf-8 -*-

"""
This module contains functionality to facilitate biased simulations in OpenMM.

Andreas Kraemer, 2018.
"""

import numpy as np
import simtk.unit as u
from simtk.openmm.openmm import CustomExternalForce, CustomCVForce
from simtk.openmm.openmm import System, TwoParticleAverageSite, NonbondedForce
from simtk.openmm.app import Element
import mdtraj as md


def make_center_of_mass_z_cv(particle_ids, masses, relative=True,
                             box_height=None):
    """
    Make a collective variable for the center of mass of some particles.

    Args:
        particle_ids (list of int): List of particle ids.
        masses (list of floats): The particle masses.
        relative (bool): If True, divide com by instantaneous box length.

    Returns:
        A dictionary with keys "x", "y", "z".
        Each item represents a component of the center of mass.
    """
    total_mass = np.sum(masses)
    L_str = "(0.75*LZ + periodicdistance(0,0,0,0,0,0.75*LZ))"
    if relative:
        collective_variable = CustomExternalForce("m/M * z/{}".format(L_str))
        collective_variable.addGlobalParameter("LZ", box_height)
    else:
        collective_variable = CustomExternalForce("m/M * z")
    collective_variable.addGlobalParameter("M", total_mass)
    collective_variable.addPerParticleParameter("m")
    for i in range(len(particle_ids)):
        collective_variable.addParticle(particle_ids[i], np.array([masses[i]]))
    return collective_variable


class FreeEnergyCosineSeries(object):
    """
    Representation of a free energy profile as a cosine series
    that can be used as a biasing potential.

    When using this class you have to work consistently with either unitless
    quantities (floats, np.arrays) or simtk.unit.Quantity.
    """

    def __init__(self, average_box_height, coefficients, constant_height=False):
        """
        Args:
            average_box_height (float or simtk.unit.Quantity):
                Average height of the simulation system.
            coefficients (np.array or simtk.unit.Quantity):
                Coefficients of the cosine series.
            constant_height (boolean): Whether or not the system should be
                simulated with a constant height.

        Note:
            The box height is expected to not deviate by more than 25% ever
            during the simulation run.
        """
        self.averageBoxHeight = average_box_height
        self.coefficients = coefficients
        self.constantHeight = constant_height

    def __str__(self):
        """
        Convert to a string that can be passed to the constructor of
        OpenMM's CustomExternalForce class.

        Returns:
            string: A string that is interpretable by OpenMM.

        """
        if self.constantHeight:
            L_str = "LZ"
        else:
            L_str = "(0.75*LZ + periodicdistance(0,0,0,0,0,0.75*LZ))"
        result = " + ".join(
            "A{} * cos( ( z / {} ) * 2 * PI * {} )".format(i, L_str, i)
            for i in range(len(self.coefficients))
        )
        return result

    def __call__(self, z, box_height=None):
        """
        Evaluate the series at a given data point.

        Args:
            z (float or np.array or simtk.unit.Quantity): The z coordinate.
            box_height (float): The instantaneous box height.

        Returns:
            float or np.array: The value of the cosine series.
        """
        if box_height is None:
            box_height = self.averageBoxHeight
        try:
            result = 0.0 * self.coefficients.unit
        except:
            result = 0.0
        for i in range(len(self.coefficients)):
            result += (self.coefficients[i] *
                       np.cos((z / box_height) * 2.0 * np.pi * i))
        return result

    def as_openmm_force(self, particle_ids=[]):
        """
        Args:
            particle_ids (list of int): A list of particle ids that the force is
                applied to.

        Returns:
            A CustomExternalForce object: The cosine series as an OpenMM force.
        """
        biasing_force = CustomExternalForce(str(self))
        biasing_force.addGlobalParameter("PI", np.pi)
        biasing_force.addGlobalParameter("LZ", self.averageBoxHeight)
        for i in range(len(self.coefficients)):
            biasing_force.addGlobalParameter("A{}".format(i),
                                             self.coefficients[i])
        for particle in particle_ids:
            biasing_force.addParticle(particle)
        return biasing_force

    def as_openmm_cv_forces(self, particle_id_list, system):
        """
        Args:
            particle_id_list (list of list of int):
            system (openmm System object): To retrieve the particle masses.

        Returns:
            A CustomCVForce object: The cosine series as an OpenMM force.
        """
        # collective variables z0, z1, z... are relative to box height
        forces = []
        for res_nr in range(len(particle_id_list)):
            energy_string = " + ".join(
                "A{} * cos( z{} * 2 * PI * {} )".format(i, res_nr, i)
                for i in range(len(self.coefficients))
            )
            biasing_force = CustomCVForce(energy_string)
            biasing_force.addGlobalParameter("PI", np.pi)
            biasing_force.addGlobalParameter("LZ", self.averageBoxHeight)
            for i in range(len(self.coefficients)):
                biasing_force.addGlobalParameter("A{}".format(i),
                                                 self.coefficients[i])

            #for res_nr in range(len(particle_id_list)):
            residue = particle_id_list[res_nr]
            masses = [system.getParticleMass(pid) for pid in residue]
            cv = make_center_of_mass_z_cv(residue, masses, relative=True,
                                          box_height=self.averageBoxHeight)
            biasing_force.addCollectiveVariable("z{}".format(res_nr), cv)

            forces.append(biasing_force)
        return forces

    def as_openmm_vsite_force(self, particle_id_pairs, system, topology,
                              positions, weights=[0.5, 0.5]):
        vsites = []
        for pair in particle_id_pairs:
            index = system.addParticle(0.0)
            system.setVirtualSite(
                index, TwoParticleAverageSite(*pair, *weights)
            )
            vsites.append(index)
            topology.addAtom(
                "V", Element.getBySymbol("D"),
                residue=topology.addResidue("V", chain=topology.addChain())
            )
            pos_virtual = [
                weights[0] * positions[pair[0]][i]
                + weights[1] * positions[pair[1]][i]
                for i in range(3)
            ]
            # add vsite to nonbonded forces (nonbonded forces must have
            # all particles in the sytem)
            for i in range(system.getNumForces()):
                if isinstance(system.getForce(i), NonbondedForce):
                    system.getForce(i).addParticle(0, 1.0, 0)
            positions.append(pos_virtual)
        return self.as_openmm_force(vsites)

    @staticmethod
    def from_histogram(box_height, counts, temperature, num_coefficients=6):
        """
        Static constructor to extract the cosine series from a histogram
        that was extracted from a previous simulation.

        Args:
            box_height (float): Height of the box in z-direction.
            counts (np.array): Counts from the histogram.
            temperature (float): in Kelvin.
            num_coefficients (int): Number of terms in the cosine series.

        Returns:
            An instance of this class.

        Note:
            This class sets up the coefficients as a simtk.unit.Quantity object.
        """
        summe = counts.sum()
        # free energy
        R = u.MOLAR_GAS_CONSTANT_R
        free_energy = - (R * temperature * np.log(counts / summe))
        return FreeEnergyCosineSeries.from_free_energy(box_height, free_energy,
                                                       num_coefficients)

    @staticmethod
    def from_free_energy(box_height, free_energy, num_coefficients=6):
        """
        Static constructor to extract the cosine series from a known free energy
        profile that was extracted from a previous simulation.

        Args:
            box_height (float): Height of the box in z-direction.
            free_energy (Quantity): Free energy in bins.
            num_coefficients (int): Number of terms in the cosine series.

        Returns:
            An instance of this class.

        Note:
            This class sets up the coefficients as a simtk.unit.Quantity object.
        """
        num_bins = len(free_energy)
        fe = (0.5 * (free_energy + free_energy[::-1]).
              value_in_unit(u.kilojoule_per_mole))  # symmetrize
        fe -= fe.min()  # normalize
        rfft = np.fft.rfft(fe).real
        cos_coeff = 2 * rfft / num_bins
        cos_coeff[0] = rfft[0] / num_bins
        cos_coeff = u.Quantity(value=(-1.0) * cos_coeff,
                               unit=u.kilojoule_per_mole)
        return FreeEnergyCosineSeries(box_height, cos_coeff[:num_coefficients])


def extract_z_histogram(trjfiles, topology, particle_ids, num_bins=100):
    """TODO: Deprecate and replace with something smarter"""
    try:
        top = md.Topology.from_openmm(topology)
    except:
        top = topology
    trajectory = md.load(trjfiles[0], top=top, atom_indices=particle_ids)
    for i in range(1, len(trjfiles)):
        next_sequence = md.load(trjfiles[i], top=top, atom_indices=particle_ids)
        trajectory = trajectory.join(next_sequence)
    z_coordinates = np.array([frame.xyz[0][i][2] / frame.unitcell_lengths[0][2]
                              for frame in trajectory
                              for i in particle_ids])
    box_sizes = [trajectory.openmm_boxes(i)[2][2].value_in_unit(u.angstrom)
                 for i in range(trajectory.n_frames)]
    box_size = np.mean(box_sizes)
    # print(z_coordinates)
    hist = np.histogram(np.mod(z_coordinates, 1.0),
                        np.arange(0.0, 1. + 1e-10, 1.0 / num_bins))
    hist = list(hist)
    hist[1] = box_size * hist[1]
    return hist
