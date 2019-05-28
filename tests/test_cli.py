from click.testing import CliRunner
from rflow import abspath
from rflow.cli import main


def test_selection_psf():
    runner = CliRunner()
    topology_file = abspath("data/2dlpc.psf")
    runner.invoke(main, "select", "water", "-t", topology_file)


def test_selection_pdb():
    pass