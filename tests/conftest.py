
import warnings
import pytest
from rflow.trajectory import TrajectoryIterator
from rflow.utility import abspath

warnings.filterwarnings("ignore", message="numpy.dtype size changed", category=RuntimeWarning)


@pytest.fixture(scope="session")
def whex_iterator():
    return TrajectoryIterator(
        filename_template=abspath("data/whex{}.dcd"),
        first_sequence=1, last_sequence=2,
        topology_file=abspath("data/whex.pdb")
    )
