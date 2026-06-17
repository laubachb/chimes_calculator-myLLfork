/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Rebecca K. Lindsey (LLNL)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(chimesFF,PairCHIMES); // PairStyle(key, class)

#else

#ifndef LMP_PAIR_CHIMES_H
#define LMP_PAIR_CHIMES_H


#include "pair.h"

#include "chimesFF.h"
#ifdef USE_CUDA
#  include "chimesFF_gpu.cuh"
#endif
#include <vector>	


/*	Functions required by LAMMPS:


settings 	(done)		reads the input script line with arguments defined here
coeff		(done)		set coefficients for one i,j pair type
compute		(done)		workhorse routine that computes pairwise interactions
init_one	(done)		perform initalization for one i,j type pair
init_style 	(done)		initialization specific to this pair style

write_restart			write i,j pair coeffs to restart file
read_restart			read i,j pair coeffs from restart file
write_restart_settings	write global settings to restart file
read_restart_settings	read global settings from restart file
single				    force and energy fo a single pairwise interaction between two atoms
*/

	
namespace LAMMPS_NS
{
	class PairCHIMES : public Pair 
	{
	 	public:
			
			// Variable definitions
			
			chimesFF chimes_calculator;   // chimesFF instance
			
			char * chimesFF_paramfile;	  // ChIMES parameter file
			
			std::vector<int> chimes_type; // For i = LMP atom type indx, chimes_type[i-1] gives the ChIMES parameter file type idx
			
			double maxcut_3b;
			double maxcut_4b;
				
			int n_3mers;				   // number of neighborlist_Xmers entries
			int n_4mers;
			
			std::vector<std::vector<int> > neighborlist_3mers;	// custom neighbor list; neighborlist_Xmers[cluster idx][atom in cluster idx]
			std::vector<std::vector<int> > neighborlist_4mers;
            
            // Prepare files necessary for ChIMES fitting 
            
            bool     for_fitting;
            bool     fingerprint;
			int IO_freq;
            ofstream badness_stream;			

			// 2-body vars for chimesFF access

			std::vector        <double>   dr;
		    std::vector        <double>   dr_3b;
		    std::vector        <double >  dr_4b;

			double                        dist;
			std::vector        <double>   dist_3b;				
			std::vector        <double>   dist_4b;	

			std::vector<double> force_2b;
		    std::vector<double> force_3b;
		    std::vector<double> force_4b;

			std::vector<int> typ_idxs_2b;
			std::vector<int> typ_idxs_3b;
			std::vector<int> typ_idxs_4b;	

			// Vars for neighlist construction

			std::vector <int> tmp_3mer;
			std::vector <int> tmp_4mer;	
			
			// Constructor/Deconstructor
			
			PairCHIMES(class LAMMPS *);
			
			virtual ~PairCHIMES();
			
			// Functions that have been written

			void   settings(int narg, char **arg);
			void   init_style();	
			void   coeff(int narg, char **arg);
			void   allocate();
			double init_one(int i, int j);	
			void   compute(int eflag, int vflag);
			void   build_mb_neighlists();
		    inline double get_dist(int i, int j, double* dr);
		    inline double get_dist(int i, int j);
			void   set_chimes_type();

		// Functions I haven't worked on 
					
		void write_restart();		
		void read_restart();				
		void write_restart_settings();	
		void read_restart_settings();
		void single();

		// --------------------------------------------------------
		// GPU batch-evaluation path (compiled only with USE_CUDA)
		// --------------------------------------------------------
#ifdef USE_CUDA
		// Called from coeff() after parameter upload; safe to call again.
		void init_gpu_buffers();
		// Called from destructor.
		void free_gpu_buffers();

		// Ensure host + device batch arrays can hold at least n entries.
		void ensure_batch_2b(int n);
		void ensure_batch_3b(int n);
		void ensure_batch_4b(int n);
		// Ensure output force/energy arrays are sized for natoms atoms.
		void ensure_output_arrays(int natoms);

		// Executes GPU-accelerated 1B+2B+3B+4B compute and returns.
		// Called from compute() when GPU is active.
		void compute_gpu(int eflag, int vflag);

		// GPU batch input arrays (host-side, raw new[]/delete[])
		double *h_dx_2b, *h_dr_2b;
		int    *h_typ_2b, *h_ai_2b, *h_aj_2b;
		int     cap_2b;

		double *h_dx_3b, *h_dr_3b;
		int    *h_typ_3b, *h_ai_3b, *h_aj_3b, *h_ak_3b;
		int     cap_3b;

		double *h_dx_4b, *h_dr_4b;
		int    *h_typ_4b, *h_ai_4b, *h_aj_4b, *h_ak_4b, *h_al_4b;
		int     cap_4b;

		// GPU batch input arrays (device-side)
		double *d_dx_2b, *d_dr_2b;
		int    *d_typ_2b, *d_ai_2b, *d_aj_2b;

		double *d_dx_3b, *d_dr_3b;
		int    *d_typ_3b, *d_ai_3b, *d_aj_3b, *d_ak_3b;

		double *d_dx_4b, *d_dr_4b;
		int    *d_typ_4b, *d_ai_4b, *d_aj_4b, *d_ak_4b, *d_al_4b;

		// GPU output arrays (device-side)
		double *d_forces_out;    // [natoms*3]  zeroed each step
		double *d_energy_out;    // [1]         zeroed each step

		// GPU output arrays (host-side mirrors)
		double *h_forces_out;
		double  h_energy_gpu;

		int     out_natoms;      // current d_forces_out allocation size
		bool    gpu_ready;       // true after init_gpu_buffers() succeeds
#endif
	};
}	


	
#endif
#endif
