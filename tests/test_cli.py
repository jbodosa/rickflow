
import os
from click.testing import CliRunner
from rflow import abspath, CWD
from rflow.cli import main, select


def test_selection_psf():
    runner = CliRunner()
    topology_file = abspath("data/2dlpc.psf")
    result = runner.invoke(select, ["--top", topology_file, "water"])
    assert "17280 atoms" in result.output


def test_bilayer_moduli(tmpdir):
    dcd = abspath("data/ord2.dcd")
    psf = abspath("data/ord+o2.psf")
    with CWD(str(tmpdir)):
        #runner = CliRunner()
        args = ["traj", "-t",  dcd, "--top", psf,
                "moduli-input", "-h", "'resname DPPC and name C2'",
                "-t", "'resname DPPC and name C216'", "-t", "'resname DPPC and name C316'"]
        os.system("rflow " + " ".join(args))
        #result = runner.invoke(main, args)
        #print(result.output)
        #assert result.exit_code == 0
        assert all(os.path.isfile(f"boxsize{dim}.out") for dim in "XYZ")
        assert all(os.path.isfile(f"Lipid{dim}.out") for dim in "XYZ")



def xxxtest_selection_pdb():
    pass