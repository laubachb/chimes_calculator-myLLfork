
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


GPU Acceleration (CUDA)
^^^^^^^^^^^^^^^^^^^^^^^^

ChIMES supports optional CUDA GPU acceleration for force evaluation.
All GPU code is guarded by ``#ifdef USE_CUDA``; CPU-only builds are unaffected.

Requirements
""""""""""""

* NVIDIA GPU with Compute Capability ≥ 6.0 (Pascal or newer)
* CUDA Toolkit ≥ 11.0
* GCC ≥ 9 (Intel compilers are **not** supported for CUDA compilation)

Identify your GPU's SM architecture before building::

    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

Common SM architecture values:

.. list-table::
   :header-rows: 1

   * - GPU
     - SM arch
     - ``CUDA_ARCH``
   * - RTX PRO 6000 Blackwell / B200
     - sm_120
     - ``120``
   * - H100
     - sm_90
     - ``90``
   * - A100
     - sm_80
     - ``80``
   * - A30 / A40 / RTX 3090
     - sm_86
     - ``86``
   * - L40S / RTX 4090
     - sm_89
     - ``89``
   * - V100
     - sm_70
     - ``70``

Building GPU-enabled LAMMPS
""""""""""""""""""""""""""""

Navigate to ``etc/lmp`` and run ``install_gpu.sh`` with the SM architecture as its argument:

.. code-block:: bash

    # Example: Stampede3 with Blackwell GPUs (SM 120)
    module load gcc/13.2.0 cuda/12.8
    ./install_gpu.sh 120
    # Binary: etc/lmp/exe/lmp_mpi_chimes_gpu

    # Example: H100 (SM 90)
    ./install_gpu.sh 90

    # Example: A100 (SM 80)
    ./install_gpu.sh 80

No changes to the LAMMPS input script are required.  The GPU path
activates automatically at runtime when the binary was compiled with
``USE_CUDA`` and no per-atom energy/virial output is requested.  If
per-atom thermo quantities, ``TABULATION``, or ``FINGERPRINT`` mode are
requested, the pair style falls back silently to the CPU path.

.. Note ::

    On Stampede3, the ``cuda`` module requires GCC (not Intel) as its
    host compiler.  Load modules in this order::

        module load intel/24.0 impi/21.11 gcc/13.2.0 cuda/12.8 python/3.12.11


Running the GPU test
"""""""""""""""""""""

A full LAMMPS GPU NVE test is provided in ``etc/lmp/tests/test_suite-GPU_NVE/``.
It runs 100 NVE steps on 512 carbon atoms and compares GPU thermo output
against the CPU reference within a tolerance of 1×10\ :sup:`-3`.

On Stampede3 (batch submission)::

    cd etc/lmp/tests/test_suite-GPU_NVE
    sbatch submit_stampede3.slurm
    # Results appear in current_output/

For interactive runs::

    idev -p rtx-small -N 1 -n 1 -t 00:30:00 -A <YOUR_ALLOCATION>
    module load intel/24.0 impi/21.11 gcc/13.2.0 cuda/12.8 python/3.12.11
    cd etc/lmp/tests/test_suite-GPU_NVE
    ./run_gpu_test.sh

See ``etc/lmp/tests/test_suite-GPU_NVE/README`` for complete details.

Standalone GPU vs. CPU validation
"""""""""""""""""""""""""""""""""""

A standalone unit test executable validates that GPU and CPU forces agree
to within 1×10\ :sup:`-9` eV/Å for random cluster geometries:

.. code-block:: bash

    # Build (requires WITH_CUDA=ON):
    module load gcc/13.2.0 cuda/12.8
    cmake -B build_gpu -DWITH_CUDA=ON -DCUDA_ARCH=<SM_ARCH> .
    cmake --build build_gpu --target chimescalc-gpu-validate -j4

    # Run against the included liquid-carbon force field:
    ./build_gpu/chimescalc-gpu-validate \
        serial_interface/tests/force_fields/published_params.liqC.2+3b.cubic.txt

.. Note ::

    GPU floating-point reductions (``atomicAdd``) are non-associative, so
    GPU results may differ from CPU by ≈10\ :sup:`-12` – 10\ :sup:`-10` eV
    (machine-epsilon level).  This is the standard trade-off for GPU
    parallelism and does not affect physical observables.

