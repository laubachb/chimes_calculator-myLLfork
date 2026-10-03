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
#include "chimesFF_mbfast.h"
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

			// Sum-factorized many-body path (pair_style chimesFF ... fastmb)
			bool                use_fastmb;
			chimesMBFast        mbfast;
			std::vector<int>    mb_pair_a, mb_pair_b;   // registered pairs, local indices a < b
			std::vector<int>    mb_tri_pid;             // 3 per triplet: 2*pair id + flip
			std::vector<int>    mb_quad_pid;            // 6 per quadruplet
			std::vector<double> mb_pair_r, mb_pair_dr;
			void   build_mb_pair_registry();
			bool   build_mb_neighlists_rips();      // false -> caller falls back to the stock loops (decided on all ranks)
			bool   warned_self_image = false;

			// fastmb options (pair_style keywords nosimd, norips, orient tag|hash)
			bool   use_simd, use_rips, orient_hash;   // orient_hash: order atoms by a bijective hash of the tag
			std::vector<std::vector<int> > mb_bucket;  // per cluster type key: pending cluster ids
			std::vector<long long> mb_hkeys;           // flat open-addressing pair registry
			std::vector<int>       mb_hvals;

			// Optional phase timers (env CHIMES_TIMING=1): seconds, accumulated on each rank
			bool   timing;
			double t_mbnl, t_reg, t_cache, t_2b, t_3b, t_4b;
			void   compute_mb_fast(int nbody);
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
