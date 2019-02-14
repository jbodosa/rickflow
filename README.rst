========
rickflow
========


.. image:: https://img.shields.io/pypi/v/rickflow.svg
        :target: https://pypi.python.org/pypi/rickflow

.. image:: https://img.shields.io/travis/Olllom/rickflow.svg
        :target: https://travis-ci.org/Olllom/rickflow

.. image:: https://readthedocs.org/projects/rickflow/badge/?version=latest
        :target: https://rickflow.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Running and Analyzing OpenMM Jobs


* Free software: MIT license
* Documentation: https://rickflow.readthedocs.io.


Features
--------

* A handy front-end to run simulations in OpenMM
* Setting up biased simulations in OpenMM to enhance permeation
* Analysis of permeation from OpenMM and CHARMM trajectories


Getting Started
---------------

1) Download and install anaconda_. And add the required lines to your .bashrc file.

.. _anaconda:https://www.anaconda.com/download/#macos

2) Create a conda environment with all the requirements::

    $ conda create -n rflow -c omnia python=3.6 openmm mdtraj numpy pandas

3) Activate the environment (has to be done every time you open a new terminal)::

    $ conda activate rflow

4) Clone the repository from gitlab::

    $ git clone https://gitlab.com/Olllom/rickflow.git

5) Install::

    $ cd rickflow
    $ python setup.py install

6) Take a look at the examples_::

.. _examples: examples/

* `Running a Simulation from Charmm input files`_   
* `Counting Crossings`_
* `Extracting Diffusion Profiles`_

.. _Running a Simulation from Charmm input files: examples/start_simulation/start_simulation.ipynb
.. _Counting Crossings: examples/counting_crossings.ipynb
.. _Extracting Diffusion Profiles: examples/diffusion_profiles/bayesian_diffusion_profiles.ipynb


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
