#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `rickflow` module."""

import pytest

from rflow import RickFlow
from rflow.utility import abspath
import glob
import os
import pytest
from simtk.openmm.app.internal.charmm.exceptions import CharmmPSFWarning

pytestmark = pytest.mark.filterwarnings("ignore:Detected PSF molecule section that is WRONG")


@pytest.fixture(scope="module")
def rickflow_instance(tmpdir_factory):
    return RickFlow(
        toppar=glob.glob(os.path.join(abspath("data/toppar"), '*')),
        psf=abspath("data/2dlpc.psf"),
        crd=abspath("data/rb2dlpc.crd"),
        box_dimensions=[47.7695166, 47.7695166, 137.142387],
        gpu_id=None,
        work_dir=str(tmpdir_factory.mktemp('rflow'))
    )


def test_directory_structure(rickflow_instance):
    assert os.path.exists(os.path.join(rickflow_instance.work_dir, "out"))
    assert os.path.exists(os.path.join(rickflow_instance.work_dir, "trj"))
    assert os.path.exists(os.path.join(rickflow_instance.work_dir, "res"))


def test_forces(rickflow_instance):
    for force in rickflow_instance.system.getForces():
        if hasattr(force, "getUseDispersionCorrection"):
            assert force.getUseDispersionCorrection() is False
        if hasattr(force, "getUseLongRangeCorrection"):
            assert force.getUseLongRangeCorrection() is False

