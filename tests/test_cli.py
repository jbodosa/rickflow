
import warnings
from click.testing import CliRunner
from simtk.openmm.app.internal.charmm.exceptions import CharmmPSFWarning
from rflow import abspath
from rflow.cli import main, select


def test_selection_psf():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=CharmmPSFWarning)
        runner = CliRunner()
        topology_file = abspath("data/2dlpc.psf")
        result = runner.invoke(select, ["--topology", topology_file, "water"])
        assert "17280 atoms" in result.output


def test_selection_pdb():
    pass