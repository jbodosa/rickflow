"""
Tests for biasing tools.
"""

from rflow import biasing
from rflow.biasing import FreeEnergyCosineSeries, RelativePartialCenterOfMassRestraint
from rflow.utility import abspath

import pytest
from pytest import approx

import os
import random
import numpy as np

from simtk.openmm import System, Context, LangevinIntegrator
from simtk.openmm.app import Simulation, Topology, DCDReporter, Element, PDBReporter
from simtk import unit as u


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


@pytest.mark.parametrize("use_com_cv", [True, False])
@pytest.mark.parametrize("constant_height", [True, False])
def test_cos_openmm_force(use_com_cv, constant_height):
    """
    Tests if the forces applied by openmm are consistent with the ones
    that are output by the call function.
    """
    series = FreeEnergyCosineSeries(
        average_box_height=10.0 * u.angstrom,
        coefficients=u.Quantity(value=np.array([1.0, 1.0]), unit=u.kilojoule_per_mole),
        constant_height=constant_height
    )

    # minimal system with one particle
    system = System()
    system.addParticle(1.0)
    system.getNumParticles()
    if use_com_cv:
        system.addForce(series.as_openmm_cv_forces(particle_id_list=[[0]], system=system)[0])
    else:
        system.addForce(series.as_openmm_force(particle_ids=[0]))
    context = Context(system, LangevinIntegrator(
        500.0, 1./u.picosecond, 1.0* u.femtosecond))
    context.setPeriodicBoxVectors(*np.eye(3)*10.0*u.angstrom)

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


def test_partial_com_restraint():
    # minimal system with one particle
    system = System()
    system.addParticle(1.0)
    system.getNumParticles()
    system.addParticle(2.0)
    k = 100.0 * u.kilojoule_per_mole
    box_length = 10.0*u.angstrom
    bias = RelativePartialCenterOfMassRestraint([0,1],
                                        force_constant=k,
                                        position=0.5,
                                        box_height_guess=box_length*1.2 # an inaccurate guess is OK
                                        )
    system.addForce(bias.as_openmm_force(system))
    context = Context(system, LangevinIntegrator(
        500.0, 1. / u.picosecond, 1.0 * u.femtosecond))
    context.setPeriodicBoxVectors(*np.eye(3)*box_length)
    for z in np.arange(0, 10.0, 0.1):
        positions = u.Quantity(
            value=np.array([[0.0,0.0,0.0],[0.0, 0.0, z]]), unit=u.angstrom)
        context.setPositions(positions)
        ener = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            u.kilojoule_per_mole)
        expected = (k * ((2.*z*u.angstrom)/3./box_length - 0.5)**2).value_in_unit(u.kilojoule_per_mole)
        assert ener == approx(expected, abs=1e-4)
        assert ener == approx(
            bias(positions, [1.0*u.dalton,2.0*u.dalton], box_length).value_in_unit(u.kilojoule_per_mole),
            abs=1e-4
        )


@pytest.mark.skipif(True, reason="Deprecated")
def test_extract_z_histogram(tmpdir):

    #for use_cv_force in [False, True]:
    #for force_mode in ["per_atom", "collective_variable", "virtual_site"]:
    for force_mode in ["per_atom", "virtual_site"]:

        # I tried the test for the cv force once and it worked
        # However, it is too expensive to be repeated all the time.
        n_particles = 500

        topology = Topology()
        for i in range(n_particles):
            topology.addAtom(
                "H", Element.getBySymbol("H"),
                residue=topology.addResidue("H", chain=topology.addChain())
            )
        topology.setUnitCellDimensions(
            u.Quantity(np.array([5.0] * 3), unit=u.angstrom)
        )

        # simulate
        if True:
            series = FreeEnergyCosineSeries(average_box_height=5.0 * u.angstrom,
                                             coefficients=u.Quantity(
                                                 value=np.array([0.0, 0.1]),
                                                 unit=u.kilojoule_per_mole
                                             ),
                                             constant_height=True
                                             )
            print(str(series))
            print(series.coefficients)
            # minimal system
            system = System()
            for i in range(n_particles):
                system.addParticle(1.0 * u.dalton)
            positions = [[0.0, 0.0, 5.0*random.random()] for _ in range(n_particles)]
            system.setDefaultPeriodicBoxVectors(*np.eye(3) * 5.0 * u.angstrom)
            pairs_of_particles = [[i,i+1] for i in range(0, n_particles, 2)]
            if force_mode == "collective_variable":
                forces = series.as_openmm_cv_forces(particle_id_list=pairs_of_particles, system=system)
                for force in forces:
                    system.addForce(force)
            elif force_mode == "per_atom":
                system.addForce(series.as_openmm_force(particle_ids=list(range(n_particles))))
            else:
                assert force_mode == "virtual_site"
                system.addForce(series.as_openmm_vsite_force(
                    particle_id_pairs=pairs_of_particles, system=system,
                    topology=topology, positions=positions
                ))

            #integrator = MetropolisMonteCarloIntegrator(temperature=310.*u.kelvin)
            integrator = LangevinIntegrator(310. * u.kelvin, 1.0 / u.picosecond, 1.0 * u.femtosecond)
            simulation = Simulation(topology, system, integrator)
            context = simulation.context
            context.setPeriodicBoxVectors(*np.eye(3) * 5.0 * u.angstrom)
            context.setPositions(u.Quantity(value=np.array(positions),
                                            unit=u.angstrom))
            context.applyConstraints(0.001)
            pdb = PDBReporter(os.path.join(str(tmpdir),"out.pdb"), 1)
            pdb.report(simulation, context.getState(getPositions=True))
            print("run 1000")
            simulation.step(1000)
            simulation.reporters.append(DCDReporter(os.path.join(str(tmpdir),"out.dcd"), 100, enforcePeriodicBox=True))
            print("run 10000")
            simulation.step(10000)
            #assert False

        histogram = extract_z_histogram([os.path.join(str(tmpdir),"out.dcd")], topology,
                                        particle_ids=list(range(n_particles)),
                                        num_bins=10)
        free_energy = - u.MOLAR_GAS_CONSTANT_R * 310.0 * u.kelvin * np.log(histogram[0])
        free_energy = (free_energy - np.min(free_energy)).value_in_unit(u.kilojoule_per_mole)
        assert free_energy[0] == approx(0.2, abs=0.5)
        #plt.plot(0.5*(histogram[1][:-1] + histogram[1][1:]), free_energy)
        #plt.show()