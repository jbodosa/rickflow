"""
Tests for biasing tools.
"""

from rflow import biasing
from rflow.biasing import FreeEnergyCosineSeries, RelativePartialCenterOfMassRestraint, ConstantPullingForce
from rflow.utility import abspath

import pytest
from pytest import approx

import os
import random
import numpy as np

from simtk.openmm import System, Context, LangevinIntegrator, VerletIntegrator, Platform
from simtk.openmm.app import Simulation, Topology, DCDReporter, Element, PDBReporter
from simtk import unit as u

def add_centroid_force(system, custom_centroid_bond_force, platform):
    """A wrapper function to acknowledge that the h22 magic variable is not standard OpenMM right now"""
    system.addForce(custom_centroid_bond_force)
    try:
        Context(system, LangevinIntegrator(
            500.0, 1. / u.picosecond, 1.0 * u.femtosecond), platform)
    except Exception as e:
        if "Unknown variable 'h22'" in str(e):
            pytest.skip("This version of OpenMM does not support h22 in CustomCentroidBondForce.")
        else:
            raise e

class OpenMMEnergyEvaluator(object):
    """
    calculating the energies of a biasing potential directly in openmm
    """
    def __init__(self):
        # minimal system with one particle
        self.system = System()
        self.system.addParticle(1.0 * u.dalton)
        self.context = None

    def addForce(self, force, box_length):
        self.system.addForce(force)
        self.context = Context(self.system, LangevinIntegrator(
            500.0, 1. / u.picosecond, 1.0 * u.femtosecond))
        self.context.setPeriodicBoxVectors(*np.eye(3) * box_length)

    def __call__(self, z):
        self.context.setPositions(u.Quantity(value=np.array([[0.0, 0.0, z]]),
                                        unit=u.angstrom))
        state = self.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(
            u.kilojoule_per_mole)
        return energy


def get_custom_force_classes():
    """
    Yields all classes in biasing_tools that have a method as_openmm_force()`.
    """
    for name in biasing.__dir__():
        obj = getattr(biasing, name)
        if hasattr(obj, "as_openmm_force"):
            yield obj


def test_cos_series_with_floats():
    """
    Test usage of cosine series with unitless quantities.
    """
    densities = np.loadtxt(abspath("data/density_whex.txt"))
    densities[0, 1] = densities[1, 1]
    densities[-1, 1] = densities[-2, 1]
    box_height = abs(2 * densities[0, 0])

    fe_series = FreeEnergyCosineSeries(average_box_height=box_height,
                                       coefficients=np.array([1.])
                                       )
    assert fe_series(2.0) == approx(1.0)


def test_cos_series_with_quantities():
    """
    Test usage of cosine series with unit.Quantity objects.
    """
    box_height = 10.0 * u.angstrom
    fe_series = FreeEnergyCosineSeries(
        average_box_height=box_height,
        coefficients=u.Quantity(value=np.array([1.]),
                                unit=u.kilocalorie_per_mole)
    )
    assert (fe_series(2.0*u.angstrom).value_in_unit(u.kilojoule_per_mole)
            == approx(4.184))


@pytest.mark.parametrize("force_type", ["standard", "cv", "centroid"])
@pytest.mark.parametrize("constant_height", [True, False])
def test_cos_openmm_force(force_type, constant_height):
    """
    Tests if the forces applied by openmm are consistent with the ones
    that are output by the call function.
    """
    platform = Platform.getPlatformByName("Reference")
    series = FreeEnergyCosineSeries(
        average_box_height=20.0 * u.angstrom,
        coefficients=u.Quantity(value=np.array([1.0, 1.0]), unit=u.kilojoule_per_mole),
        constant_height=constant_height
    )

    # minimal system with one particle
    system = System()
    system.addParticle(1.0)
    system.getNumParticles()
    if force_type == "cv":
        system.addForce(series.as_openmm_cv_forces(particle_id_list=[[0]], system=system)[0])
    elif force_type == "centroid":
        add_centroid_force(system, series.as_openmm_centroid_force(particle_id_list=[[0]]), platform)
    else:
        system.addForce(series.as_openmm_force(particle_ids=[0]))
    context = Context(system, LangevinIntegrator(
        500.0, 1./u.picosecond, 1.0* u.femtosecond), platform)
    context.setPeriodicBoxVectors(*np.eye(3)*20*u.angstrom)

    for z in np.arange(0,10.0, 0.1):
        context.setPositions(u.Quantity(value=np.array([[0.0, 0.0, z]]),
                                       unit=u.angstrom)
                            )
        state = context.getState(getEnergy=True)
        target = series(z * u.angstrom).value_in_unit(
            u.kilojoule_per_mole)
        energy = state.getPotentialEnergy().value_in_unit(
            u.kilojoule_per_mole)
        assert target == approx(energy, abs=1e-5)


@pytest.mark.parametrize("force_type", ["cv", "centroid"])
def test_partial_com_restraint(force_type):
    # minimal system with one particle
    platform = Platform.getPlatformByName("Reference")
    system = System()
    system.addParticle(1.0)
    system.getNumParticles()
    system.addParticle(2.0)
    k = 100.0 * u.kilojoule_per_mole
    box_length = 10.0*u.angstrom
    bias = RelativePartialCenterOfMassRestraint([0,1],
                                        force_constant=k,
                                        position=0.7,
                                        box_height_guess=box_length*1.2 # an inaccurate guess is OK
                                        )
    if force_type == "cv":
        system.addForce(bias.as_openmm_cv_force(system))
    elif force_type == "centroid":
        add_centroid_force(system, bias.as_openmm_centroid_force(), platform)
    context = Context(system, LangevinIntegrator(
        500.0, 1. / u.picosecond, 1.0 * u.femtosecond))
    context.setPeriodicBoxVectors(*np.eye(3)*box_length)
    for z in np.arange(0, 10.0, 0.1):
        positions = u.Quantity(
            value=np.array([[0.0,0.0,0.0],[0.0, 0.0, z]]), unit=u.angstrom)
        context.setPositions(positions)
        ener = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            u.kilojoule_per_mole)
        expected = (k * ((2.*z*u.angstrom)/3./box_length - 0.7)**2).value_in_unit(u.kilojoule_per_mole)
        assert ener == approx(expected, abs=1e-4)
        assert ener == approx(
            bias(positions, [1.0*u.dalton,2.0*u.dalton], box_length).value_in_unit(u.kilojoule_per_mole),
            abs=1e-4
        )


def test_constant_pull_periodic():
    force = 10.0 * u.kilojoule_per_mole / u.nanometer
    constant_pulling_force = ConstantPullingForce(force)
    mass = 1.0 * u.amu
    system = System()
    system.addParticle(mass)
    system.addForce(constant_pulling_force.as_openmm_force([0]))
    integrator = VerletIntegrator(1.0 * u.femtosecond)
    context = Context(system, integrator)
    context.setPeriodicBoxVectors(*np.eye(3) * 1.0 * u.nanometer)
    context.setPositions([[0.0, 0.0, 0.0]])
    context.setVelocities([[0.0, 0.0, 0.0]])
    for i in range(1000):
        integrator.step(1)
        state = context.getState(getPositions=True)
        t = state.getTime()
        z = (state.getPositions()[0][2]).value_in_unit(u.nanometer)
        analytic_solution = (0.5*force/mass*t**2).value_in_unit(u.nanometer)
        assert z == pytest.approx(analytic_solution, abs=0.01)


def test_centroid_force():
    platform = Platform.getPlatformByName("Reference")
    system = System()
    for i in range(4):
        system.addParticle(1.0)
    from simtk.openmm import CustomCentroidBondForce
    force = CustomCentroidBondForce(1, "z1 + h22")# + k * distance(g1,g2)")
    force.setUsesPeriodicBoundaryConditions(True)
    force.addGroup([0])
    #force.addPerBondParameter("k")
    force.addBond([0])
    system.addForce(force)

    integrator = VerletIntegrator(1.0 * u.femtosecond)
    try:
        context = Context(system, integrator, platform)
        context.setPeriodicBoxVectors(*np.eye(3) * 4.0 * u.nanometer)
        context.setPositions([[0.0, 0.0, float(i)] for i in range(4)])

        ener = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            u.kilojoule_per_mole)
        print(ener)
    except Exception as e:
        if "Unknown variable 'h22'" in str(e):
            pytest.skip("This version of OpenMM does not support h22 in CustomCentroidBondForce.")
        else:
            raise e
