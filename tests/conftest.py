
import warnings
import pytest
from simtk.openmm.app.internal.charmm.exceptions import CharmmPSFWarning
from rflow.trajectory import TrajectoryIterator
from rflow.utility import abspath

warnings.filterwarnings("ignore", message="numpy.dtype size changed", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Detected PSF molecule section that is WRONG", category=CharmmPSFWarning)

@pytest.fixture(scope="session")
def whex_iterator():
    return TrajectoryIterator(
        filename_template=abspath("data/whex{}.dcd"),
        first_sequence=1, last_sequence=2,
        topology_file=abspath("data/whex.pdb")
    )
