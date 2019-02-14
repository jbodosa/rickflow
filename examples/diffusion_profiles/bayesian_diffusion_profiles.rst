
Calculating Diffusion Profiles through Bayesian Analysis
========================================================

Diffusion profiles can be extracted from unrestrained molecular dynamics
via Bayesian Analysis (BA), as first described by Hummer (2005). The BA
can be performed using the python packages
`diffusioncma <https://gitlab.com/Olllom/diffusioncma>`__ or
`mcdiff <https://github.com/annekegh/mcdiff>`__.

The most important input argument for these packages is a transition
matrix, which can be extracted using the present rickflow package.

This tutorial shows the calculation of diffusion profiles D(z) in two
steps: 1. Extraction of a transition matrix from an MD trajectory using
rickflow. 2. Bayesian Analysis using diffusioncma.

Get the Code (Recommended Way)
==============================

Both *rickflow* and *diffusioncma* plus all their requirements should be
installed in a conda environment.

-  Download and install anaconda
-  Create a conda environment named *dcma_py36* that has all the
   requirements installed

::

   conda -c omnia -c conda-forge -c olllom create -n dcma_py36 "python=3.6" diffusioncma mdtraj "openmm>=7.3.0"

-  Activate the environment (has to be done every time you open a new
   terminal)

::

   conda activate dcma_py36

(on older anaconda versions, it is ``source activate dcma_py36``
instead).

-  Clone the rickflow repository from gitlab:

::

   git clone git@gitlab.com:Olllom/rickflow.git

-  cd into the rickflow directory and install the code

::

   cd rickflow
   python setup.py install

Step 1: Extract Transition Matrix
---------------------------------

The transition matrix stores the number N_ij of all transitions from bin
i to bin j. In a Python script, we do the following:

First, import the two classes that we need from the ``rflow`` module.
The ``CharmmTrajectoryIterator`` is a helper class that loops over
*dyn\ ``i``.dcd* files. The ``TransitionCounter`` is responsible for
assembling the transition matrices.

.. code:: ipython3

    from rflow import CharmmTrajectoryIterator, TransitionCounter
    
    filename_template="/u/ewang/Projects/bacterial/popc-etoh-s170/1/namd/dyn{}.dcd"
    topology_file="/u/ewang/Projects/bacterial/popc-etoh-s170/1/step5_assembly.psf"

Note that ``{}`` in the ``filename_template`` acts as a wildcard that
will assume the sequence numbers, i.e. *dyn1.dcd* etc.

I am using Eric’s files here, assuming that you have more or less the
same file structure. If not, that is no problem – you would just have to
manually load the trajectories using ``mdtraj.load()`` instead of using
the ``CharmmTrajectoryIterator``.

The trajectory iterator is set up with the above-defined dcd and psf
files. It also takes two arguments ``first_sequence`` and
``last_sequence`` that specify the range of the trajectory files that
you want to analyze. For this example, I will use all sequences from
``dyn10.dcd`` to ``dyn20.dcd``, avoiding the equilibration phase, but
still including a reasonable number of frames. Note that you can also
omit the ``last_sequence`` argument. The code will automatically figure
out the last sequence number in the directory.

.. code:: ipython3

    trajectories = CharmmTrajectoryIterator(
        filename_template=filename_template,
        topology_file=topology_file,
        first_sequence=10, last_sequence=20, 
    )
    
    counter = TransitionCounter(
        lag_iterations=[10, 20, 30, 40, 50, 60],
        num_bins=100,
        solute="resname ETOH",
        membrane="resname POPC"
    )


.. parsed-literal::

    /u/akraemer/.conda/envs/openmm-7.4.0_cuda-9.2/lib/python3.6/site-packages/simtk/openmm/app/charmmpsffile.py:363: CharmmPSFWarning: Detected PSF molecule section that is WRONG. Resetting molecularity.
      'Resetting molecularity.', CharmmPSFWarning)


The two parameters ``solute`` and ``membrane`` take atom selection
strings using mdtraj’s DSL language, as documented
`here <http://mdtraj.org/latest/atom_selection.html>`__. If you have a
membrane with various lipid types you have to do, e.g.,

``membrane="resname POPC and resname DOPC and resname CHL1"``, or
``membrane="(not water) and (not resname ETOH)"`` instead.

The parameter ``num_bins`` specifies the number of bins along the
z-direction. 100 is usually a good choice.

The parameter ``lag_iterations`` is a list of lag times for the Bayesian
Analysis (specified in number of frames). If you save your output once
every picosecond, the numbers will be in picoseconds.

The next step runs the assembly of the matrices. This is the
computationally expensive part.

.. code:: ipython3

    for i,traj in enumerate(trajectories):
        print("\rTrajectory {}/{} ...".format(i+trajectories.first, trajectories.last), end="")
        counter(traj)
    print(" done.")


.. parsed-literal::

    Trajectory 20/20 ... done.


Let’s save the matrices to files.

.. code:: ipython3

    counter.save_matrices("transition_matrix_lag{}.txt")

And you should see files ``transition_matrix_lag10.txt``, … in your
directory.

Step 2: Optimize Diffusion profiles
-----------------------------------

The next step is the Bayesian Analysis. Please see `the diffusioncma
README file <https://gitlab.com/Olllom/diffusioncma>`__ for a long
description of how to do this. What follows here is just a very basic
analysis. Diffusioncma can be used as a command line tool or through
Python. I will describe the command line usage. To see the help message
type

::

   dcma opt --help

I will only be doing the optimization for the shortest lag time:

.. code:: bash

    %%bash
    dcma opt transition_matrix_lag10.txt --outfile profiles_lag10.txt


.. parsed-literal::

    Optimizing Diffusion and Free Energy Profiles...
    STARTING CMA-ES
    Optimization finished.
    Running DCMA


The optimal profiles are stored in ``profiles_lag10.txt``. The header of
this file also lists the permeability.

.. code:: bash

    %%bash
    dcma plot profiles_lag10.txt -o profiles_lag10


.. parsed-literal::

    No labels given or wrong number of labels. Using file names as labels.
    The diffusion and free energy profiles have been saved to
      profiles_lag10_D.png
      profiles_lag10_F.png
    --------------------------------------------------------


This is how it looks like (the labels got messed up, but anyway :-), you
can plot directly from the .txt files).


