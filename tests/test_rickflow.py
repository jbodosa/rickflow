#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `rickflow` module."""

import pytest

from rickflow import RickFlow
import glob
import os
import pytest


class CWD(object):
    def __init__(self, path): self.old_path = os.getcwd(); self.new_path = str(path)
    def __enter__(self): os.chdir(self.new_path); return self
    def __exit__(self): os.chdir(self.old_path)


@pytest.fixture(scope="module")
def rickflow_instance():
    return RickFlow(
        toppar=glob.glob("./toppar/*"),
        psf="2dlpc.psf",
        crd="rb2dlpc.crd",
        box_dimensions=[47.7695166, 47.7695166, 137.142387],
        gpu_id=None
    )


def test_directory_structure(rickflow_instance):
    assert os.path.exists("out")
    assert os.path.exists("trj")
    assert os.path.exists("res")


def test_forces(rickflow_instance):
    for force in rickflow_instance.system.getForces():
        if hasattr(force, "getUseDispersionCorrection"):
            assert force.getUseDispersionCorrection() is False
        if hasattr(force, "getUseLongRangeCorrection"):
            assert force.getUseLongRangeCorrection() is False

