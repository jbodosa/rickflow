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
from rflow.utility import selection


def make_center_of_mass_z_cv(particle_ids, masses, relative=True,
                             box_height=None):
    """
    Make a collective variable for the z-coordinate of the center of mass of some particles.

    Args:
        particle_ids (list of int): List of particle ids.
        masses (list of floats): The particle masses.
        relative (bool): If True, divide com by instantaneous box length.
        box_height (a simtk.unit for lengths): A guess for the instantaneous box_height.

    Returns:
        A collective variable (in OpenMM, a collective variable is expressed as a custom force object).

    Note:
        The instantaneous box height must not deviate more than 25% from the box_height.

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
    for i,particle_id in enumerate(particle_ids):
        collective_variable.addParticle(particle_id, np.array([masses[i]]))
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


class RelativePartialCenterOfMassRestraint(object):
    """
    Harmonic restraint for the center of mass of a part of the system, e.g. a membrane, relative to the box height.
    """
    def __init__(self, atom_ids, force_constant=None, position=0.5, box_height_guess=None):
        """
        Args:
            atom_ids (list of int): The atom ids on which the restraint acts.
            force_constant (simtk.unit.Quantity object): The force constant in units of energy.
            position (float): A float between 0 and 1, indicating the minimum of the restraint
                relative to the instantaneous box height.
            box_height_guess (simtk.unit.Quantity object): A length that provides a guess
                for the box height. Must not deviate from any instantaneous box height by more than 25%.
        """
        try:
            self.atom_ids = atom_ids.tolist()
        except AttributeError:
            self.atom_ids = atom_ids
        self.force_constant = force_constant
        self.position = position
        self.energy_string = "k0 * (zcom - zref)^2"
        self.box_height = box_height_guess

    def as_openmm_force(self, system):
        """
        Args:
            system: The OpenMM system.

        Returns::x
            An OpenMM force object.
        """
        biasing_force = CustomCVForce(self.energy_string)
        masses = [system.getParticleMass(atom) for atom in self.atom_ids]
        com_cv = make_center_of_mass_z_cv(self.atom_ids, masses, relative=True, box_height=self.box_height)
        biasing_force.addCollectiveVariable("zcom", com_cv)
        biasing_force.addGlobalParameter("zref", self.position)
        biasing_force.addGlobalParameter("k0", self.force_constant)
        return biasing_force

    def __str__(self):
        string = "{}; k0 = {}, zref={}".format(
            self.energy_string, self.force_constant, self.position
        )
        return string

    def __call__(self, positions, masses, box_height):
        z_com = 0.0 * u.nanometer * u.dalton
        total_mass = 0.0 * u.dalton
        for i in self.atom_ids:
            total_mass += masses[i]
            z_com += positions[i][2] * masses[i]
        z_com /= total_mass
        return self.force_constant * (z_com/box_height - self.position)**2