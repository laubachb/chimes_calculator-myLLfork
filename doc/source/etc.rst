
.. _etc:

ChIMES in LAMMPS
=============================================

We are currently working toward ChIMES calculator implementation in `LAMMPS <https://lammps.sandia.gov>`_ as a USER package. In the interim, we have provided a capability for compiling the `stable_29Aug2024_update1 <https://github.com/lammps/lammps.git build/>`_ version of LAMMPS with ChIMES as a pairstyle in this package.



Quick start
^^^^^^^^^^^^^^^^

.. Note ::

    To compile LAMMPS with ChIMES, you must have C++11 and MPI compilers avillable. As with installation of the ChIMES Calculator itself, if you are on a HPC using module files, you may need to load them first. Module files are already configured for a handful of HPC - inspect the contents of modfiles to see if yours is listed. If it is (e.g., LLNL-LC.mod), execute ``export hosttype=LLNL-LC`` Otherwise, load the appropriate modules by hand before running the install script.

    Note that Intel oneapi compilers (which are now free) can be used to properly configure your enviroment for all Intel capabilities (e.g., icc, mpiicpc, mkl, etc.) - simply locate and execute the setvars.sh script within your Intel installation.



Navigate to ``etc/lmp`` and execute  ``./install.sh`` to install. Once complete, the installation can be tested by navigating to ``etc/lmp/tests`` and either running the entire test suite via ``./run_tests.sh``, which takes roughly 15 minutes, or running an individual test by entering an example folder and executing ``../../exe/lmp_mpi_chimes -i in.lammps``. 

The install script compiles with LAMMPS packages ``manybody`` and ``extra-pair``. Additional packages can be included by adding appropriate commands to the ``etc/lmp/install.sh`` script. For example, to add the MOLECULE package, one would add the line highlighted in yellow below:


.. code-block :: 
    :lineno-start: 114
    :emphasize-lines: 6
    
    # Compile
    
    cd build/${lammps}/src
    make yes-manybody
    make yes-extra-pair
    make yes-molecule
    make -j 4 mpi_chimes
    cd -

    


.. Tip ::

    Additional flags can be specified during installation to enable features such as model tabulation and configuration fingerprint generation. For more details, see the :ref:`utils` page.
    

Running
^^^^^^^^^^^^^^^^

To run a simulation using ChIMES parameters, a block like the following is needed in the main LAMMPS input file (i.e. ``in.lammps``):

.. code-block:: text

    pair_style	chimesFF
    pair_coeff	* *   some_standard_chimes_parameter_file.txt 

Note that the following must also be set in the main LAMMPS input file, to use ChIMES:

.. code-block:: text

    units       real		
    newton      on 		
    atom_style  atomic		
    atom_modify sort 0 0.0	

**Important Considerations:**

Atom Type Mass Matching
"""""""""""""""""""""

The element masses defined in the ChIMES parameter file must match the masses defined in the LAMMPS data file. The matching is done by comparing masses to within 0.001 atomic mass units. The order of atom types in the files does not matter - only the mass values need to match.

For example, if your ChIMES parameter file defines elements with masses 12.011 and 15.999, then your LAMMPS data file must have atom types with matching masses, regardless of the order they appear in the files.

**ChIMES Parameter File Example:**
The parameter file defines atom types with specific masses:

.. code-block:: text

    ATOM TYPES: 2

    # TYPEIDX #     # ATM_TYP #     # ATMCHRG #     # ATMMASS #
    0               C               0               12.011
    1               O               0               15.999

**LAMMPS Data File Example:**
The data file must have atom types with matching masses (order doesn't matter):

.. code-block:: text
  
    2 atom types

    Masses

    1 12.0107  # C (matches ChIMES mass 12.011 within tolerance)
    2 15.9994  # O (matches ChIMES mass 15.999 within tolerance)

**Important:** If no mass matches are found between the ChIMES parameter file and LAMMPS data file, the simulation will terminate with an error, as ChIMES cannot be used for any interactions.

Fast Many-Body Evaluation (fastmb)
""""""""""""""""""""""""""""""""""

Adding the keyword ``fastmb`` to the pair style evaluates the 3- and 4-body ChIMES terms with the same
arithmetic as the default code, reordered. Energies, forces and the virial agree with the default
evaluation to floating-point roundoff (relative differences of 1e-13 or less, the same size as the
differences the default code shows between MPI rank counts), and parameter files are unchanged.

.. code-block:: text

    neighbor        0.3 bin
    neigh_modify    delay 0 every 1 check yes
    pair_style      chimesFF fastmb
    pair_coeff      * * some_standard_chimes_parameter_file.txt

``fastmb`` combines:

* a per-pair cache of the cutoff-weighted Chebyshev polynomials and their derivatives, computed once per pair per step instead of once per cluster;
* sum factorization: each coefficient tensor is stored as a compressed sparse fiber tree, and one sweep returns the cluster energy and every pair derivative;
* SIMD batches: eight clusters of the same type pass through the tree together (double precision);
* cluster lists built as the triangles and tetrahedra of the Vietoris-Rips complex, with adjacency bitsets instead of nested distance loops (the default loops are used automatically when the box is smaller than twice the padded many-body cutoff);
* cluster ownership by a hash of the atom tag rather than the tag itself, which balances work across MPI ranks.

Optional keywords, for A/B testing: ``nosimd`` (one cluster at a time), ``norips`` (default cluster-list loops)
and ``orient tag`` (default ownership rule). Setting the environment variable ``CHIMES_TIMING=1`` prints per-phase
timings at exit. ``fastmb`` is ignored, with a message, for tabulated 3-body parameters and in FINGERPRINT builds.

For full speed compile with OpenMP SIMD and the target vector instruction set (the UT-TACC makefile uses
``-qopenmp-simd -xCORE-AVX512``). A neighbor skin of about 0.3 Angstrom is recommended: cluster lists are rebuilt
cheaply, and a smaller skin keeps fewer inactive clusters in them.

On one Stampede3 SPR node (112 MPI ranks), ``fastmb`` with a 0.3 Angstrom skin reduced the time per MD step
relative to the default code by 186x for Carbon-2.0 Large (12,800 atoms), 87x for the reshock nitrogen model
(13,824 atoms) and 14x for Carbon-2.0 Small (102,400 atoms). To check a build, run
``etc/lmp/tests/example-fastmb/check_fastmb.sh``.

Hybrid Overlay Usage
"""""""""""""""""""

ChIMES can be used simultaneously with other potentials using LAMMPS ``hybrid/overlay`` pair style so that forces from each sub-style add together. The subsections below each use their own heading for a distinct overlay pattern.

ChIMES with MOMB and Lennard-Jones
++++++++++++++++++++++++++++++++++

Combine one ChIMES parameter file with MOMB (many-body van der Waals) and Lennard-Jones for different atom-type pairs.

.. code-block:: text

    pair_style      hybrid/overlay chimesFF momb 9.0 0.75 20.0 lj/cut 10.0
    pair_coeff      * * chimesFF ${param_file}
    pair_coeff      1 1 momb 0.0 1.0 1.0 418.26 2.904
    pair_coeff      1 2 lj/cut 0.25   3.5
    pair_coeff      2 2 lj/cut 0.25   3.5

Here ChIMES supplies the bonded / short-range ML-IAP description, while MOMB and ``lj/cut`` supply additional van der Waals channels as specified by the ``pair_coeff`` lines.

TurboChIMES multilayer (two ``chimesFF`` layers)
++++++++++++++++++++++++++++++++++++++++++++++++

A **multilayer (TurboChIMES)** fit from the Active Learning Driver and ChIMES LSQ produces **two** reduced parameter files—one per hyperparameter layer—typically named ``0params.txt.reduced`` and ``1params.txt.reduced`` by convention. The **0** file is the **short-ranged** layer (dense basis, smaller effective outer cutoffs in the fit); the **1** file is the **long-ranged** layer (sparser basis, smoother mid- to long-range behavior). Use **two** ``chimesFF`` entries in ``hybrid/overlay`` and assign each file to a different sub-style index.

.. code-block:: text

    pair_style      hybrid/overlay chimesFF for_fitting chimesFF for_fitting
    pair_coeff      * * chimesFF  1  0params.txt.reduced
    pair_coeff      * * chimesFF  2  1params.txt.reduced

The integer after ``chimesFF`` on each ``pair_coeff`` line selects which ``chimesFF`` instance in the ``pair_style`` list receives that file: **1** → short-range layer (**0params**), **2** → long-range layer (**1params**). For production runs (outside the LSQ fitting workflow), replace ``for_fitting`` with the appropriate keyword for your build, as for a single ``pair_style chimesFF`` run.

**Important:** Both parameter files must remain **consistent** with the same LAMMPS data file (atom typing, masses, and ``atom_style``) and with how the model was trained; only the split of resolution across range differs between the two files.

D2 dispersion correction (MOMB with ChIMES)
++++++++++++++++++++++++++++++++++++++++++++++

This pattern uses **hybrid ChIMES + MOMB** specifically to add **D2 dispersion correction** in LAMMPS. Whether to include MOMB and which parameters to use follow your ChIMES parameterization; do not add MOMB arbitrarily.

**Important:** When using MOMB with ChIMES, you must include the ``make yes-extra-pair`` command in the ``install.sh`` script when compiling LAMMPS to enable the MOMB potential support.

.. code-block:: text

    # Compile

    cd build/${lammps}/src
    make yes-manybody
    make yes-extra-pair

.. Warning::

    1. Implementation assumes outer cutoffs for (n+1)-body interactions are always :math:`\le` those for n-body interactions
    2. This capability is still under testing - please `let us know <https://groups.google.com/g/chimes_software>`_ if you observe strange behavior
    3. Assumes user wants single-atom energies to be added to the system energy. If you don't want to, zero the energy offsets in the parameter file

