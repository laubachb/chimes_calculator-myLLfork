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

#ifdef FASTMB
			// Fast many-body evaluation (pair_style chimesFF fastmb)

			bool                use_fastmb;             // Evaluate the 3- and 4-body terms through the chimesFF pair cache
			bool                fastmb_simd;            // Sweep clusters of one type in SIMD batches (keyword nosimd: one at a time)
			bool                fastmb_rips;            // Build cluster lists by bitset intersection (keyword norips: the nested loops of build_mb_neighlists)
			bool                fastmb_hash_ownership;  // A cluster is owned by its atom with the smallest hashed tag (keyword orient tag: smallest tag)
			bool                fastmb_warned_self_image; // The fall-back to build_mb_neighlists has been reported once

			std::vector<int>    fastmb_pair_i;          // [pair idx] local index of the first atom of a registered pair (i < j)
			std::vector<int>    fastmb_pair_j;          // [pair idx] local index of the second atom
			std::vector<double> fastmb_pair_r;          // [pair idx] pair distance
			std::vector<double> fastmb_pair_dr;         // [pair idx*CHDIM + dim] distance components, atom j minus atom i
			std::vector<int>    fastmb_3mer_pair_idx;   // [3mer idx*3 + slot] 2*pair idx, plus 1 if the slot runs from j to i
			std::vector<int>    fastmb_4mer_pair_idx;   // [4mer idx*6 + slot]
			std::vector<long long> fastmb_registry_keys; // Open-addressing hash table of the registered pairs: pair key, -1 if empty
			std::vector<int>       fastmb_registry_vals; // Pair index of the pair stored in the same table slot
			std::vector<std::vector<int> > fastmb_batches; // [cluster key] clusters waiting for a full SIMD batch

			// Benchmark instrumentation, printed at exit: phase timers (env CHIMES_TIMING=1) and the
			// memory held by the many-body data structures (env CHIMES_MEMORY=1)

			bool   fastmb_timing;
			double fastmb_time_neighlist;               // Seconds in the many-body neighbor list builds, this rank
			double fastmb_time_registry;                // ... in the pair registry builds
			double fastmb_time_cache;                   // ... filling the pair cache
			double fastmb_time_2b;                      // ... in the 1- and 2-body loop
			double fastmb_time_3b;                      // ... in the 3-body term
			double fastmb_time_4b;                      // ... in the 4-body term
			void   report_fastmb_instrumentation();     // The two reports, rank 0

			bool   build_fastmb_neighlists();           // Cluster lists by bitset intersection; false if the box is too small (caller then uses build_mb_neighlists on every rank)
			void   build_fastmb_pair_registry();        // Assigns pair indices to the pairs of the many-body neighbor lists
			void   compute_fastmb(int nbody);           // 3- or 4-body forces, energy and virial from the pair cache
#endif
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

		};
}	


	
#endif
#endif
