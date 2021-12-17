
import warnings
import pytest
import mdtraj as md
from rflow.openmm import app
from rflow.trajectory import TrajectoryIterator
from rflow.utility import abspath


warnings.filterwarnings("ignore", message="numpy.dtype size changed", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Detected PSF molecule section that is WRONG", category=app.internal.charmm.exceptions.CharmmPSFWarning)


@pytest.fixture(scope="session")
def whex_iterator():
    return TrajectoryIterator(
        filename_template=abspath("data/whex{}.dcd"),
        first_sequence=1, last_sequence=2,
        topology_file=abspath("data/whex.pdb")
    )


@pytest.fixture(scope="session")
def ord2_traj():
    return md.load_dcd(abspath("data/ord2.dcd"), top=abspath("data/ord+o2.psf"))
