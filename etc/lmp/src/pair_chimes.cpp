/* ----------------------------------------------------------------------
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


#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "mpi.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "my_page.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "error.h"
#include "pair_chimes.h"
#include "group.h"
#include "update.h" // Needed for mb neighlist updates and info printing for fitting
#include "output.h" // Needed for infor printing for fitting -- dump 1 must be the "main" dump file used for fitting
#include  "utils.h" // Needed for infor printing for fitting
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>

using namespace LAMMPS_NS;

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


PairCHIMES::PairCHIMES(LAMMPS *lmp) : Pair(lmp)
{
	restartinfo = 0;

	int me = comm->me;
	MPI_Comm_rank(world,&me);
	
	chimes_calculator.init(me);  
    for_fitting = false;
    use_fastmb  = false;
    use_simd    = true;
    use_rips    = true;
    orient_hash = true;
    timing      = getenv("CHIMES_TIMING") && atoi(getenv("CHIMES_TIMING")) > 0;
    t_mbnl = t_reg = t_cache = t_2b = t_3b = t_4b = 0.0;
#ifdef FINGERPRINT
    fingerprint = false;
#endif
	
	// 2, 3, and 4-body vars for chimesFF access

	dr     .resize(CHDIM);
	dr_3b  .resize(3*CHDIM); 
	dr_4b  .resize(6*CHDIM);
	
	dist_3b.resize(3);			       
	dist_4b.resize(6);			       

	// CHDIM is the number of spatial dimensions (usually 3).
	force_2b.resize(2*CHDIM);
	force_3b.resize(3*CHDIM) ;
	force_4b.resize(4*CHDIM) ;
	
	typ_idxs_2b.resize(2);
	typ_idxs_3b.resize(3);
	typ_idxs_4b.resize(4);
	
	// Vars for neighlist construction
	
	tmp_3mer.resize(3);
	tmp_4mer.resize(4);   
	
	if (chimes_calculator.rank == 0)
	{ 
		std::cout << std::endl;
		std::cout << "************************* WARNING (pair_style chimesFF) ************************" << std::endl;
		std::cout << "Assuming n-body interactions have longer cutoffs than all (n+1)-body interactions" << std::endl;
		std::cout << "************************* WARNING (pair_style chimesFF) ************************" << std::endl;
		std::cout << std::endl;
	}
		
}

PairCHIMES::~PairCHIMES()
{
	if (timing)
	{
		double loc[6] = {t_mbnl, t_reg, t_cache, t_2b, t_3b, t_4b}, mx[6], sum[6];
		MPI_Allreduce(loc, mx,  6, MPI_DOUBLE, MPI_MAX, world);
		MPI_Allreduce(loc, sum, 6, MPI_DOUBLE, MPI_SUM, world);
		if (comm->me == 0)
		{
			const char *nm[6] = {"mb_neighlist", "pair_registry", "pair_cache", "2body", "3body", "4body"};
			int np = comm->nprocs;
			for (int k = 0; k < 6; k++)
				printf("CHIMES_TIMING %-14s max %12.6f s  mean %12.6f s\n", nm[k], mx[k], sum[k]/np);
		}
	}
	if (allocated) 
	{   	    
	    memory->destroy(setflag);
	    memory->destroy(cutsq);
	}
    
    if (badness_stream.is_open())
        badness_stream.close();
}	

void PairCHIMES::settings(int narg_in, char **arg_in)
{
	// Fast many-body evaluation (exact; see chimesFF_mbfast.h). Keywords may appear anywhere:
	//   fastmb        enable it (SIMD batching, Vietoris-Rips cluster lists, hash ownership)
	//   nosimd        evaluate clusters one at a time instead of in SIMD batches
	//   norips        build cluster lists with the stock nested loops
	//   orient tag    assign each cluster to its lowest-tag atom (stock rule) instead of by hash
	// All other arguments are handled as before (fitting, fingerprint <iofreq>).
	std::vector<char *> kept;
	for (int a = 0; a < narg_in; a++)
	{
		if      (utils::strmatch(arg_in[a], "^fastmb$")) use_fastmb = true;
		else if (utils::strmatch(arg_in[a], "^nosimd$")) use_simd   = false;
		else if (utils::strmatch(arg_in[a], "^norips$")) use_rips   = false;
		else if (utils::strmatch(arg_in[a], "^orient$"))
		{
			if (a + 1 >= narg_in) error->all(FLERR, "Illegal pair_style chimesFF command: orient needs tag or hash");
			const std::string v = arg_in[++a];
			if      (v == "tag")  orient_hash = false;
			else if (v == "hash") orient_hash = true;
			else error->all(FLERR, "Illegal pair_style chimesFF command: orient must be tag or hash");
		}
		else kept.push_back(arg_in[a]);
	}
	int    narg = kept.size();
	char **arg  = kept.data();

	if (narg > 3) 
		error -> all(FLERR,"Illegal pair_style command. Expects no more than three arguments (string: fitting or fingerprint and iofreq)");
        
    if (narg == 1) 
    {  
        if (utils::strmatch(arg[0],"fitting"))
        {
            for_fitting   = true;
            stringstream ss;
            ss << chimes_calculator.rank;
            badness_stream.open("rank-" + ss.str() + ".badness.log");    
        }
    }
	#ifdef FINGERPRINT
    if (narg == 2) 
    {  
        if (utils::strmatch(arg[0],"fingerprint"))
        {
			
            fingerprint   = true;
			IO_freq = std::stoi(arg[1]);;  
        }
    }
	
    if (narg == 3) 
    {  
        if (utils::strmatch(arg[0],"fingerprint"))
        {
			
            fingerprint   = true;
			IO_freq = std::stoi(arg[1]);;  
        }
        if (utils::strmatch(arg[1],"fingerprint"))
        {
			
            fingerprint   = true;
			IO_freq = std::stoi(arg[2]);;  
        }
        if (utils::strmatch(arg[0],"fitting") || utils::strmatch(arg[1],"fitting") || utils::strmatch(arg[2],"fitting"))
        {
            for_fitting   = true;
            stringstream ss;
            ss << chimes_calculator.rank;
            badness_stream.open("rank-" + ss.str() + ".badness.log");  
        }
    }
	#endif

	return;	
}

void PairCHIMES::coeff(int narg, char **arg)
{
	// Expect: pair_coeff * * <parameter file name>

	if (narg != 3) 
		error -> all(FLERR,"Illegal pair_style command. Expects \"pair_coeff * * <parameter file name>\" ");
	
	chimesFF_paramfile = arg[2]; 
	
	chimes_calculator.read_parameters(chimesFF_paramfile);

	set_chimes_type();
    
    //chimes_calculator.set_atomtypes(chimes_type);
    chimes_calculator.build_pair_int_trip_map() ; 
    chimes_calculator.build_pair_int_quad_map() ;

	// Set special LAMMPS flags/cutoffs
	
	if (!allocated)
		allocate();
		
	vector<vector<double> > cutoff_2b;
	chimes_calculator.get_cutoff_2B(cutoff_2b);

	for(int i=1; i<=atom->ntypes; i++)
	{
		for(int j=i; j<=atom->ntypes; j++)
		{
			setflag[i][j] = 1;
			setflag[j][i] = 1;

			if (chimes_type[i-1] < 0 || chimes_type[j-1] < 0)
			{
				cutsq[i][j] = 0.0;
				if (i!=j)
					cutsq[j][i] = 0.0;
				continue;
			}
			
			cutsq[i][j]  = cutoff_2b[ chimes_calculator.get_atom_pair_index( chimes_type[i-1]*chimes_calculator.natmtyps + chimes_type[j-1] ) ][1];
			cutsq[i][j] *= cutsq[i][j];
			
			if (i!=j)
			{
				cutsq[j][i]  = cutoff_2b[ chimes_calculator.get_atom_pair_index( chimes_type[j-1]*chimes_calculator.natmtyps + chimes_type[i-1]) ][1];
				cutsq[j][i] *= cutsq[j][i];
			}			
		}
	}

	maxcut_3b = chimes_calculator.max_cutoff_3B();
	// if (maxcut_3b==0.0 && fingerprint){
	// 	double max_val = 20;
	// 	for (const auto& row : cutoff_2b) {
	// 		for (double val : row) {
	// 			if (val > max_val) {
	// 				max_val = val;
	// 			}
	// 		}
	// 	}
	// 	maxcut_3b=max_val;
	// }
	maxcut_4b = chimes_calculator.max_cutoff_4B();
	// if (maxcut_4b==0.0 && fingerprint){maxcut_4b=maxcut_3b;}

#ifdef TABULATION
	if (use_fastmb && chimes_calculator.tabulate_3B)
	{
		if (comm->me == 0) std::cout << "chimesFF: fastmb ignored for tabulated 3-body parameters" << std::endl;
		use_fastmb = false;
	}
#endif
#ifdef FINGERPRINT
	if (use_fastmb && fingerprint)
	{
		if (comm->me == 0) std::cout << "chimesFF: fastmb ignored when fingerprinting" << std::endl;
		use_fastmb = false;
	}
#endif
	if (use_fastmb)
	{
		const std::string why = mbfast.init(chimes_calculator);
		if (!why.empty()) error->all(FLERR, "pair_style chimesFF fastmb: " + why);
		if (comm->me == 0)
			std::cout << "chimesFF: fastmb many-body evaluation; SIMD batches " << (use_simd ? "on" : "off")
			          << ", Rips cluster lists " << (use_rips ? "on" : "off")
			          << ", cluster ownership by " << (orient_hash ? "hash" : "tag") << std::endl;
	}
}
void writeClusterDataComp(const string& filename, const vector<vector<double>>& data) 
{
    ofstream ofs(filename);
    if (!ofs) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }
    
    ostringstream buffer;
    
    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); j++) {
            buffer << row[j];
            if (j < row.size() - 1) buffer << " ";  // Add space between elements
        }
        buffer << "\n";
    }

    ofs << buffer.str(); // Single large write operation
    ofs.close();
}

void PairCHIMES::allocate()
{
	allocated = 1;
	
	memory->create(setflag,atom->ntypes+1,atom->ntypes+1,"pair:setflag");
	
	for(int i=1; i<=atom->ntypes; i++)
		for(int j=i; j<=atom->ntypes; j++)
			setflag[i][j] = 0;

	memory->create(cutsq,atom->ntypes+1,atom->ntypes+1,"pair:cutsq");
}	

void PairCHIMES::init_style()
{
	if (atom -> tag_enable == 0)
		error -> all(FLERR,"Pair style ChIMES requires atom IDs");
	
	if (force->newton_pair == 0)
		error->all(FLERR,"Pair style ChIMES requires newton pair on");	
	
	// Set up neighbor lists... borrowing this from pair_airebo:
	// need a full neighbor list, including neighbors of ghosts

	// int irequest = neighbor->request(this,instance_me);
	// neighbor->requests[irequest]->half = 0;
	// neighbor->requests[irequest]->full = 1;
	// neighbor->requests[irequest]->ghost = 1;

    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
}

double PairCHIMES::init_one(int i, int j)
{
	// Sets the cutoff for each pair interaction.
	// The maximum of the returned values are used to set outer cutoff for neighbor lists
	// WARNING: This means linking won't work properly if 2-b interactions do not have larger cutoffs than all other
	// higher bodied interactions!!
	
	if (setflag[i][j] == 0) 
		error->all(FLERR,"All pair coeffs are not set");
	
	return sqrt(cutsq[i][j]);
}

inline double PairCHIMES::get_dist(int i, int j, double *dr)
{
	double 	**x    = atom -> x;	// Access to system coordinates

	dr[0] = x[j][0] - x[i][0];  
	dr[1] = x[j][1] - x[i][1];
	dr[2] = x[j][2] - x[i][2];

	return sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
}

inline double PairCHIMES::get_dist(int i, int j)
{
	double dummy_dr[3] ;

	return get_dist(i,j, dummy_dr);
}

void PairCHIMES::build_mb_neighlists()
{


	if ( (chimes_calculator.poly_orders[1] == 0) &&  (chimes_calculator.poly_orders[2] == 0) 
		#ifdef FINGERPRINT
			&& (!fingerprint)
		#endif
		)
		return;

	// List gets built based on atoms owned by calling proc. 
	
	neighborlist_3mers.clear();
	neighborlist_4mers.clear();
	
	int i,j,k,l,inum,jnum,knum,lnum, ii, jj, kk, ll;		 // Local iterator vars
	int *ilist,*jlist,*klist,*llist, *numneigh,**firstneigh; // Local neighborlist vars
	tagint 	*tag   = atom -> tag;					         // Access to global atom indices
	int     itag, jtag, ktag, ltag;					         // holds tags	
	int     *type  = atom -> type;					         // Access to system atom types
	double 	**x    = atom -> x;					             // Access to system coordinates
	
	double maxcut_3b_padded = maxcut_3b + neighbor-> skin;
	double maxcut_4b_padded = maxcut_4b + neighbor-> skin;
	
	double dist_ij, dist_ik, dist_il, dist_jk, dist_jl, dist_kl;
	
	////////////////////////////////////////
	// Access to neighbor list vars
	////////////////////////////////////////

	inum       = list -> inum; 		 // length of the list
	ilist      = list -> ilist; 	 // list of i atoms for which neighbor list exists
	numneigh   = list -> numneigh;	 // length of each of the ilist neighbor lists
	firstneigh = list -> firstneigh; // point to the list of neighbors of i	
	
	for (ii = 0; ii < inum; ii++) // Loop over real atoms (ai)	
	{
		i     = ilist[ii];		
		itag  = tag[i];			
		jlist = firstneigh[i];		
		jnum  = numneigh[i];	

		if (chimes_type[type[i]-1] < 0)
			continue;

		for (jj = 0; jj < jnum; jj++)	
		{
			j     = jlist[jj];	
			jtag  = tag[j];		
			j    &= NEIGHMASK;

			if (chimes_type[type[j]-1] < 0)
				continue;

			if (j == i)
				continue;
			if (jtag < itag) 
				continue;				
				
			// Check ij distance

			dist_ij = get_dist(i,j);
			
			if ( (dist_ij >= maxcut_3b_padded) && (dist_ij >= maxcut_4b_padded) )
				continue;

			klist = firstneigh[i];	// ChIMES assumes all atoms must be within cutoff of eachother for a valid interaction	
			knum  = numneigh[i];	
			
			for (kk = 0; kk < knum; kk++)	
			{
				k     = klist[kk];	
				ktag  = tag[k];		
				k    &= NEIGHMASK;

				if (chimes_type[type[k]-1] < 0)
					continue;

				if ( (k==i) || (k==j) )
					continue;
				if ( (ktag < itag) || (ktag < jtag) )
					continue;						
							
 	 			// Check ik distance			

				dist_ik = get_dist(i,k);

				if ( (dist_ik >= maxcut_3b_padded) && (dist_ik >= maxcut_4b_padded) )
					continue;
					
				// Check jk distance			

				dist_jk = get_dist(j,k);
				
				if( (dist_ij < maxcut_3b_padded) &&  (dist_ik < maxcut_3b_padded) && (dist_jk < maxcut_3b_padded) )
				{
					// If we're here and valid_3mer == true, then add the triplet to the chimes neigh list        

					tmp_3mer[0] = i;
					tmp_3mer[1] = j;
					tmp_3mer[2] = k;
				
					neighborlist_3mers.push_back(tmp_3mer);
				}
									
				if ((dist_ij >= maxcut_4b_padded) || (dist_ik >= maxcut_4b_padded) || (dist_jk >= maxcut_4b_padded) )	
					continue;					
				
				// Now decide if we should continue on to 4-body neighbor list construction

				if (chimes_calculator.poly_orders[2] == 0 
					#ifdef FINGERPRINT
						&& !fingerprint
					#endif
					)
					continue;

				llist = firstneigh[i];	
				lnum  = numneigh[i];	
					
				for (ll = 0; ll < lnum; ll++)	
				{
					l     = llist[ll];	
					ltag  = tag[l];		
					l    &= NEIGHMASK;

					if (chimes_type[type[l]-1] < 0)
						continue;
					
					if ( (l==i) || (l==j) || (l==k))
						continue;
					if ((ltag < itag) ||(ltag < jtag)||(ltag < ktag)) 
						continue;
											
					// Check il distance			

					dist_il = get_dist(i,l); 

					if (dist_il >= maxcut_4b_padded)
						continue;	

					// Check jl distance			
	
					dist_jl = get_dist(j,l);
	
					if (dist_jl >= maxcut_4b_padded)
						continue;
								
					// Check kl distance			

					dist_kl = get_dist(k,l);
	
					if (dist_kl >= maxcut_4b_padded)
						continue;
		
					// If we're here and valid_4mer == true, then add the quadruplet to the chimes neigh list
					
					tmp_4mer[0] = i;
					tmp_4mer[1] = j;
					tmp_4mer[2] = k;
					tmp_4mer[3] = l;
					
					neighborlist_4mers.push_back(tmp_4mer);
					
				}				
			}
		}
	}

}

void PairCHIMES::compute(int eflag, int vflag)
{
	// Vars for access to chimesFF compute_XB functions
	
	std::vector  <double>  stensor(6);	// pointers to system stress tensor
    
	// Temp vars to hold chimes output for passing to ev_tally function
	
    std::vector<double>  fscalar(6);
    std::vector<double>  tmp_dist(1);
    std::vector<double>  tmp_dr(6);
    int                  atmidxlst[6][2];
	
	// General LAMMPS compute vars
	
	int 	i,j,k,l,inum,jnum, ii, jj;	// Local iterator vars
	int 	*ilist,*jlist, *numneigh,**firstneigh;	// Local neighborlist vars
	int     idx;

	double 	**x    = atom -> x;		    // Access to system coordinates
	double 	**f    = atom -> f;		    // Access to system forces
	
	int 	*type  = atom -> type;		// Acces to system atom types (countng starts from 1, chimesFF class expects counting from 0!)
	tagint 	*tag   = atom -> tag;		// Access to global atom indices (sort of like "parent" indices)
	int     itag, jtag, ktag, ltag;		// holds tags
	int 	nlocal = atom -> nlocal;	// Number of real atoms owned by current process .. used used to assure force assignments aren't duplicated
	int 	newton_pair = force -> newton_pair;	// Should f_j be automatically set to -f_i (true) or manually calculated (false)
	double  energy;				        // pair energy 

	int me = comm->me;
	MPI_Comm_rank(world,&me);	
	
	// Set up vars controlling if energy/pressure (virial) contributions are computed

	if (eflag || vflag) 
	{
  		ev_setup(eflag,vflag);
	}
	else 
	{
		evflag      = 0;
		vflag_fdotr = 0;
  		vflag_atom  = 0;
	}

	// Compile if fingerprinting desired
#ifdef FINGERPRINT
	std::vector<std::vector<double>> tmp_dist_2b;
    std::vector<std::vector<double>> tmp_dist_3b;
    std::vector<std::vector<double>> tmp_dist_4b;
	bool 				 tmp_FP;
	bool 				 valid_order;
	if (fingerprint){
		if(update->ntimestep % IO_freq == 0){tmp_FP = true;}
	}else{tmp_FP = false;}
#endif

	////////////////////////////////////////
	// Access to (2-body) neighbor list vars
	////////////////////////////////////////

	inum       = list -> inum; 		 // length of the list
	ilist      = list -> ilist; 	 // list of i atoms for which neighbor list exists
	numneigh   = list -> numneigh;	 // length of each of the ilist neighbor lists
	firstneigh = list -> firstneigh; // point to the list of neighbors of i


    chimes2BTmp chimes_2btmp(chimes_calculator.poly_orders[0]) ;
    chimes3BTmp chimes_3btmp(chimes_calculator.poly_orders[1]) ;
    chimes4BTmp chimes_4btmp(chimes_calculator.poly_orders[2]) ;      
	
	// Build the ChIMES many-body neighbor lists.. only do so when LAMMPS neighborlist has been updated
	
	if ( neighbor->ago == 0)
	{
		if (chimes_calculator.rank == 0)
			std::cout << "Updating chimesFF neighbor lists..." << std::endl;
			
		double tt0 = MPI_Wtime();
		if (!(use_fastmb && use_rips && build_mb_neighlists_rips()))
			build_mb_neighlists();
		double tt1 = MPI_Wtime();
		if (use_fastmb)
			build_mb_pair_registry();
		t_mbnl += tt1 - tt0;
		t_reg  += MPI_Wtime() - tt1;
		if (chimes_calculator.rank == 0)
		{
			std::cout << "	Rank " << me << " 3-body list size: " << neighborlist_3mers.size() << std::endl;
			std::cout << "	Rank " << me << " 4-body list size: " << neighborlist_4mers.size() << std::endl;
			std::cout << "	...update complete" << std::endl;
		}
	}
    
    // Prepare the badness variable
    
    chimes_calculator.reset_badness();

	////////////////////////////////////////
	// Compute 1- and 2-body interactions
	////////////////////////////////////////
	double t2b0 = MPI_Wtime();
	for (ii = 0; ii < inum; ii++)		// Loop over the atoms owned by the current process
	{
		i     = ilist[ii];				// Index of the current atom
		itag  = tag[i];					// Get i's global atom index (sort of like its "parent")

		if (chimes_type[type[i]-1] < 0)
			continue;

		jlist = firstneigh[i];			// Neighborlist for atom i
		jnum  = numneigh[i];			// Number of neighbors of atom i
		
		// First, get the single-atom energy contribution
		
		energy = 0.0;
		
		chimes_calculator.compute_1B(chimes_type[type[i]-1], energy);
        
        atmidxlst[0][0] = i;
		
		if(evflag)
			ev_tally_mb(1, 0, atmidxlst, energy, stensor);

		// Now move on to two-body force, stress, and energy
		
		for (jj = 0; jj < jnum; jj++) // Loop over neighbors of i
		{
			j     = jlist[jj];			// Index of the jj atom
			jtag  = tag[j];				// Get j's global atom index (sort of like its "parent")
			j    &= NEIGHMASK;			// Strip possible extra bits of j

			if (chimes_type[type[j]-1] < 0)
				continue;
				
			
			if (jtag <= itag) // only allow calculation for j<i, since we've requested a full neighbor list
				continue;
				
			// Get distance using ghost atoms... don't need MIC since we're using ghost atoms

			dist = get_dist(i,j,&dr[0]);
			
			typ_idxs_2b[0] = chimes_type[type[i]-1]; // Type (index) of the current atom... subtract 1 to account for chimesFF vs LAMMPS numbering convention
			typ_idxs_2b[1] = chimes_type[type[j]-1];

			// Using std::fill for maximum efficiency.
			std::fill(force_2b.begin(), force_2b.end(), 0.0) ;

			// Do the same for stress tensors
			std::fill(stensor.begin(), stensor.end(), 0.0) ;

			energy = 0.0;	
#ifdef TABULATION
			if (chimes_calculator.tabulate_2B)
                chimes_calculator.compute_2B_tab( dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp);
            else
#endif
			#ifdef FINGERPRINT
			valid_order = (i < j);
			if (tmp_FP && valid_order){
				double tmp_force_scalar;
				chimes_calculator.compute_2B( dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp, tmp_force_scalar, tmp_dist_2b, tmp_FP && valid_order);	// Auto-updates badness
			} else {
			#endif
				chimes_calculator.compute_2B( dist, dr, typ_idxs_2b, force_2b, stensor, energy, chimes_2btmp);	// Auto-updates badness		
			#ifdef FINGERPRINT
			}
			#endif
			for (idx=0; idx<3; idx++)
			{
				f[i][idx] += force_2b[0*CHDIM+idx] ;
				f[j][idx] += force_2b[1*CHDIM+idx] ;
			}

			// "Save"/tally up the energy and stresses to the global virial/energy data objects (see pair.cpp ~ line 1000)
			// Compute pressure, (in contrast to chimes_md) AFTER penalty has been added		
			
			if(vflag_atom)
            {
			    atmidxlst[0][0] = i;
			    atmidxlst[0][1] = j;
            }
			tmp_dist    [0] = dist;
			
			if (evflag)
				ev_tally_mb(2, 1, atmidxlst, energy, stensor);         
		}
	}
	#ifdef FINGERPRINT
	std::string ts = std::to_string(update->ntimestep);
	if (tmp_FP)
	{
		std::stringstream filename;
		filename << ts << "." << std::to_string(chimes_calculator.rank) <<".2b_clusters.txt";
		writeClusterDataComp(filename.str(), tmp_dist_2b);
	}
	#endif

	t_2b += MPI_Wtime() - t2b0;

    // Document badness for configuration: current timestep, current rank, worst badness seen by rank
    if (for_fitting)
        if(update->ntimestep % output->every_dump[0] == 0)
            badness_stream << update->ntimestep << " " <<  chimes_calculator.get_badness() << endl;

	// if (chimes_calculator.poly_orders[1] > 0 || tmp_FP)
	if (use_fastmb && chimes_calculator.poly_orders[1] > 0)
	{
		// Pair cache: distances and fcut-weighted Chebyshev vectors, once per pair per step
		double **xx = atom->x;
		int    *tt  = atom->type;
		double tc0 = MPI_Wtime();
		const int np = mb_pair_a.size();
		for (int p = 0; p < np; p++)
		{
			const int a = mb_pair_a[p], b = mb_pair_b[p];
			double *d = &mb_pair_dr[3*p];
			d[0] = xx[b][0] - xx[a][0];
			d[1] = xx[b][1] - xx[a][1];
			d[2] = xx[b][2] - xx[a][2];
			mb_pair_r[p] = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
			const int pt = mbfast.pair_type(chimes_type[tt[a]-1], chimes_type[tt[b]-1]);
			mbfast.fill_pair(p, pt, mb_pair_r[p]);
		}
		double tc1 = MPI_Wtime();
		compute_mb_fast(3);
		double tc2 = MPI_Wtime();
		if (chimes_calculator.poly_orders[2] > 0)
			compute_mb_fast(4);
		t_cache += tc1 - tc0;
		t_3b    += tc2 - tc1;
		t_4b    += MPI_Wtime() - tc2;
	}
	else if (chimes_calculator.poly_orders[1] > 0)
	{
		double ts3 = MPI_Wtime();
		////////////////////////////////////////
		// Compute 3-body interactions
		////////////////////////////////////////

		for (ii = 0; ii < neighborlist_3mers.size(); ii++)		
		{
			i     = neighborlist_3mers[ii][0];
			j     = neighborlist_3mers[ii][1];
			k     = neighborlist_3mers[ii][2];

			dist_3b[0] = get_dist(i,j,&dr_3b[0*CHDIM]);
			dist_3b[1] = get_dist(i,k,&dr_3b[1*CHDIM]);
			dist_3b[2] = get_dist(j,k,&dr_3b[2*CHDIM]);

			typ_idxs_3b[0] = chimes_type[type[i]-1];
			typ_idxs_3b[1] = chimes_type[type[j]-1];
			typ_idxs_3b[2] = chimes_type[type[k]-1];

			std::fill(force_3b.begin(), force_3b.end(), 0.0) ;
			std::fill(stensor.begin(), stensor.end(), 0.0) ;
				
			energy = 0.0 ;
      
#ifdef TABULATION
			if (chimes_calculator.tabulate_3B){
                chimes_calculator.compute_3B_tab( dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp);}
            else
#endif

			#ifdef FINGERPRINT
			valid_order = (tag[i] < tag[j] && tag[i] < tag[k] && tag[j] < tag[k]);
			if (tmp_FP && valid_order){
				vector<double> tmp_force_scalar_3b(3);
				chimes_calculator.compute_3B( dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp, tmp_force_scalar_3b, tmp_dist_3b, tmp_FP && valid_order);
			} else {
			#endif
				chimes_calculator.compute_3B( dist_3b, dr_3b, typ_idxs_3b, force_3b, stensor, energy, chimes_3btmp);
			#ifdef FINGERPRINT
			}
			#endif

			for (idx=0; idx<3; idx++)
			{
				f[i][idx] += force_3b[0*CHDIM+idx] ;
				f[j][idx] += force_3b[1*CHDIM+idx] ;
				f[k][idx] += force_3b[2*CHDIM+idx] ;
			}

            if (vflag_atom)
            {
			    atmidxlst[0][0] = i;
			    atmidxlst[0][1] = j;
			    atmidxlst[1][0] = i;
			    atmidxlst[1][1] = k;
			    atmidxlst[2][0] = j;
			    atmidxlst[2][1] = k;
            }
			
			if (evflag)
				ev_tally_mb(3, 3, atmidxlst, energy, stensor);		            
		}		
		t_3b += MPI_Wtime() - ts3;
	}
	#ifdef FINGERPRINT
	if (tmp_FP)
	{
		std::stringstream filename_3b;
		filename_3b << ts << "." << std::to_string(chimes_calculator.rank) <<".3b_clusters.txt";
		writeClusterDataComp(filename_3b.str(), tmp_dist_3b);
	}
	#endif

    // if (chimes_calculator.poly_orders[2] > 0 || tmp_FP)
	double ts4 = MPI_Wtime();
	if (!use_fastmb && chimes_calculator.poly_orders[2] > 0)

	{
		////////////////////////////////////////
		// Compute 4-body interactions
		////////////////////////////////////////

		for (ii = 0; ii < neighborlist_4mers.size(); ii++)		
		{
			i     = neighborlist_4mers[ii][0];
			j     = neighborlist_4mers[ii][1];
			k     = neighborlist_4mers[ii][2];
			l     = neighborlist_4mers[ii][3];			
			
			dist_4b[0] = get_dist(i,j,&dr_4b[0*CHDIM]);				      
			dist_4b[1] = get_dist(i,k,&dr_4b[1*CHDIM]);
			dist_4b[2] = get_dist(i,l,&dr_4b[2*CHDIM]);
			dist_4b[3] = get_dist(j,k,&dr_4b[3*CHDIM]);
			dist_4b[4] = get_dist(j,l,&dr_4b[4*CHDIM]);
			dist_4b[5] = get_dist(k,l,&dr_4b[5*CHDIM]);

			typ_idxs_4b[0] = chimes_type[type[i]-1];
			typ_idxs_4b[1] = chimes_type[type[j]-1];
			typ_idxs_4b[2] = chimes_type[type[k]-1];
			typ_idxs_4b[3] = chimes_type[type[l]-1];

			std::fill(force_4b.begin(), force_4b.end(), 0.0) ;
			std::fill(stensor.begin(), stensor.end(), 0.0) ;

			energy = 0.0 ;	
			
			#ifdef FINGERPRINT
			valid_order = (tag[i] < tag[j] && tag[j] < tag[k] && tag[k] < tag[l]);
			if (tmp_FP && valid_order){
				vector<double> tmp_force_scalar_4b(6);
				chimes_calculator.compute_4B( dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp, tmp_force_scalar_4b, tmp_dist_4b, tmp_FP && valid_order);
			} else {
			#endif
				chimes_calculator.compute_4B( dist_4b, dr_4b, typ_idxs_4b, force_4b, stensor, energy, chimes_4btmp);
			#ifdef FINGERPRINT
			}
			#endif

			for (idx=0; idx<3; idx++)
			{
				f[i][idx] += force_4b[0*CHDIM+idx] ;
				f[j][idx] += force_4b[1*CHDIM+idx] ;
				f[k][idx] += force_4b[2*CHDIM+idx] ;
				f[l][idx] += force_4b[3*CHDIM+idx] ;
			}
			
            if (vflag_atom) 
            {
			    atmidxlst[0][0] = i;
			    atmidxlst[0][1] = j;
			    atmidxlst[1][0] = i;
			    atmidxlst[1][1] = k;
			    atmidxlst[2][0] = i;
			    atmidxlst[2][1] = l;
			    atmidxlst[3][0] = j;
			    atmidxlst[3][1] = k;
			    atmidxlst[4][0] = j;
			    atmidxlst[4][1] = l;
			    atmidxlst[5][0] = k;
			    atmidxlst[5][1] = l;
            }
			
			if (evflag)
				ev_tally_mb(4, 6, atmidxlst, energy, stensor);	
            
		}
	}
	#ifdef FINGERPRINT
	if (tmp_FP)
	{
		std::stringstream filename_4b;
		filename_4b << ts << "." << std::to_string(chimes_calculator.rank) <<".4b_clusters.txt";
		writeClusterDataComp(filename_4b.str(), tmp_dist_4b);
	}
	#endif

	if (!use_fastmb && chimes_calculator.poly_orders[2] > 0)
		t_4b += MPI_Wtime() - ts4;

if (vflag_fdotr) 
        virial_fdotr_compute();

	return;
}

// Assign an id to every distinct atom pair used by a 3/4-body cluster, and record for each
// cluster slot (u,v) the pair id and whether the slot runs against the stored a<b order.
void PairCHIMES::build_mb_pair_registry()
{
	mb_pair_a.clear(); mb_pair_b.clear();
	mb_tri_pid.resize(3*neighborlist_3mers.size());
	mb_quad_pid.resize(6*neighborlist_4mers.size());

	// open addressing, linear probing, power-of-two table grown at load 1/2
	const long long base = atom->nlocal + atom->nghost + 1;
	size_t cap = 1024;
	while (cap < 8 * (size_t)(atom->nlocal + atom->nghost)) cap <<= 1;
	mb_hkeys.assign(cap, -1); mb_hvals.resize(cap);
	auto rehash = [&]() {
		std::vector<long long> ok(mb_hkeys); std::vector<int> ov(mb_hvals);
		cap <<= 1; mb_hkeys.assign(cap, -1); mb_hvals.resize(cap);
		for (size_t t = 0; t < ok.size(); t++) if (ok[t] >= 0)
		{
			size_t h = (size_t)(ok[t] * 0x9E3779B97F4A7C15ULL) & (cap - 1);
			while (mb_hkeys[h] >= 0) h = (h + 1) & (cap - 1);
			mb_hkeys[h] = ok[t]; mb_hvals[h] = ov[t];
		}
	};
	auto pid = [&](int u, int v) {
		const int a = u < v ? u : v, b = u < v ? v : u;
		const long long key = (long long)a*base + b;
		size_t h = (size_t)(key * 0x9E3779B97F4A7C15ULL) & (cap - 1);
		while (mb_hkeys[h] >= 0 && mb_hkeys[h] != key) h = (h + 1) & (cap - 1);
		int id;
		if (mb_hkeys[h] == key) id = mb_hvals[h];
		else
		{
			id = mb_pair_a.size(); mb_hkeys[h] = key; mb_hvals[h] = id;
			mb_pair_a.push_back(a); mb_pair_b.push_back(b);
			if (2 * mb_pair_a.size() > cap) rehash();
		}
		return 2*id + (u < v ? 0 : 1);
	};
	for (size_t t = 0; t < neighborlist_3mers.size(); t++)
	{
		const std::vector<int> &c = neighborlist_3mers[t];
		mb_tri_pid[3*t+0] = pid(c[0], c[1]);
		mb_tri_pid[3*t+1] = pid(c[0], c[2]);
		mb_tri_pid[3*t+2] = pid(c[1], c[2]);
	}
	for (size_t q = 0; q < neighborlist_4mers.size(); q++)
	{
		const std::vector<int> &c = neighborlist_4mers[q];
		mb_quad_pid[6*q+0] = pid(c[0], c[1]);
		mb_quad_pid[6*q+1] = pid(c[0], c[2]);
		mb_quad_pid[6*q+2] = pid(c[0], c[3]);
		mb_quad_pid[6*q+3] = pid(c[1], c[2]);
		mb_quad_pid[6*q+4] = pid(c[1], c[3]);
		mb_quad_pid[6*q+5] = pid(c[2], c[3]);
	}
	mb_pair_r.resize(mb_pair_a.size());
	mb_pair_dr.resize(3*mb_pair_a.size());
	mbfast.resize(mb_pair_a.size());
}

// ChIMES clusters are the 2- and 3-simplices of the Vietoris-Rips complex of the atoms at the padded
// many-body cutoffs. For each local atom i, take its forward neighbors (atoms later than i in the
// ownership order: tag, or a bijective hash of the tag), compute their pairwise distances once into
// adjacency bitsets, and list triangles and tetrahedra by bitset intersection. Every cluster is listed
// exactly once, by the rank that owns its first atom in that order. Returns false on every rank (the
// caller then uses the stock loops) if any rank finds two candidates with the same tag, i.e. periodic
// self-images in a box smaller than twice the padded cutoff.
bool PairCHIMES::build_mb_neighlists_rips()
{
	if ((chimes_calculator.poly_orders[1] == 0) && (chimes_calculator.poly_orders[2] == 0)) return true;
	neighborlist_3mers.clear();
	neighborlist_4mers.clear();

	tagint *tag  = atom->tag;
	int    *type = atom->type;
	double **x   = atom->x;
	const double c3 = maxcut_3b + neighbor->skin, c4 = maxcut_4b + neighbor->skin;
	const double c3sq = c3*c3, c4sq = c4*c4, cmsq = std::max(c3sq, c4sq);
	const bool do4 = chimes_calculator.poly_orders[2] > 0;

	// Cluster ownership = the member that comes first in this order. Tag order (stock) is spatially
	// correlated after replicate/read_data, which unbalances ranks; a bijective hash of the tag
	// gives every atom the same expected number of owned clusters. The cluster set is unchanged.
	auto ord = [&](tagint t) -> unsigned long long {
		return orient_hash ? (unsigned long long)t * 0x9E3779B97F4A7C15ULL : (unsigned long long)t; };
	std::vector<std::pair<unsigned long long,int> > cand;
	std::vector<unsigned long long> adj3, adj4;
	std::vector<char> in3, in4;
	std::vector<int> tmp3(3), tmp4(4);

	int self_image = 0;
	for (int ii = 0; ii < list->inum; ii++)
	{
		const int i = list->ilist[ii];
		if (chimes_type[type[i]-1] < 0) continue;
		const unsigned long long iord = ord(tag[i]);
		const int *jl = list->firstneigh[i];
		cand.clear();
		for (int jj = 0; jj < list->numneigh[i]; jj++)
		{
			const int j = jl[jj] & NEIGHMASK;
			if (j == i || chimes_type[type[j]-1] < 0) continue;
			const unsigned long long jord = ord(tag[j]);
			if (jord < iord) continue;
			const double dx = x[j][0]-x[i][0], dy = x[j][1]-x[i][1], dz = x[j][2]-x[i][2];
			if (dx*dx + dy*dy + dz*dz < cmsq) cand.push_back({jord, j});
		}
		std::sort(cand.begin(), cand.end());
		const int nf = cand.size();
		for (int a = 1; a < nf; a++) if (cand[a].first == cand[a-1].first) self_image = 1;
		if (self_image) break;
		const int nw = (nf + 63) / 64;
		adj3.assign((size_t)nf*nw, 0ULL); adj4.assign((size_t)nf*nw, 0ULL);
		in3.assign(nf, 0); in4.assign(nf, 0);
		for (int a = 0; a < nf; a++)
		{
			const int ja = cand[a].second;
			const double ax = x[ja][0]-x[i][0], ay = x[ja][1]-x[i][1], az = x[ja][2]-x[i][2];
			const double ra = ax*ax + ay*ay + az*az;
			in3[a] = ra < c3sq; in4[a] = ra < c4sq;
			for (int b = a+1; b < nf; b++)
			{
				const int jb = cand[b].second;
				const double bx = x[jb][0]-x[ja][0], by = x[jb][1]-x[ja][1], bz = x[jb][2]-x[ja][2];
				const double r2 = bx*bx + by*by + bz*bz;
				if (r2 < c3sq) { adj3[(size_t)a*nw + b/64] |= 1ULL << (b%64); adj3[(size_t)b*nw + a/64] |= 1ULL << (a%64); }
				if (r2 < c4sq) { adj4[(size_t)a*nw + b/64] |= 1ULL << (b%64); adj4[(size_t)b*nw + a/64] |= 1ULL << (a%64); }
			}
		}
		for (int a = 0; a < nf; a++)
		{
			for (int b = a+1; b < nf; b++)
			{
				const bool e3 = (adj3[(size_t)a*nw + b/64] >> (b%64)) & 1ULL;
				const bool e4 = (adj4[(size_t)a*nw + b/64] >> (b%64)) & 1ULL;
				if (in3[a] && in3[b] && e3)
				{
					tmp3[0] = i; tmp3[1] = cand[a].second; tmp3[2] = cand[b].second;
					neighborlist_3mers.push_back(tmp3);
				}
				if (!do4 || !(in4[a] && in4[b] && e4)) continue;
				// tetrahedra (i, a, b, c), c > b: common 4B neighbors of a and b that are 4B neighbors of i
				for (int w = b/64; w < nw; w++)
				{
					unsigned long long bits = adj4[(size_t)a*nw + w] & adj4[(size_t)b*nw + w];
					if (w == b/64) bits &= (b%64 == 63) ? 0ULL : (~0ULL << (b%64 + 1));
					while (bits)
					{
						const int cidx = w*64 + __builtin_ctzll(bits);
						bits &= bits - 1;
						if (!in4[cidx]) continue;
						tmp4[0] = i; tmp4[1] = cand[a].second; tmp4[2] = cand[b].second; tmp4[3] = cand[cidx].second;
						neighborlist_4mers.push_back(tmp4);
					}
				}
			}
		}
	}
	// The choice between this enumerator and the stock loops must be the same on every rank: the
	// two assign cluster ownership differently, and mixing them double-counts or drops clusters.
	int any_self_image = 0;
	MPI_Allreduce(&self_image, &any_self_image, 1, MPI_INT, MPI_MAX, world);
	if (any_self_image)
	{
		neighborlist_3mers.clear();
		neighborlist_4mers.clear();
		if (comm->me == 0 && !warned_self_image)
			std::cout << "chimesFF: fastmb cluster lists fall back to the stock loops (box smaller than twice the padded many-body cutoff)" << std::endl;
		warned_self_image = true;
		return false;
	}
	return true;
}

// Sum-factorized 3- or 4-body forces, energy and virial over the ChIMES cluster lists.
void PairCHIMES::compute_mb_fast(int nbody)
{
	static const int slots3[3][2] = {{0,1},{0,2},{1,2}};
	static const int slots4[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
	const int npair = nbody == 3 ? 3 : 6;
	const int (*slots)[2] = nbody == 3 ? slots3 : slots4;
	const std::vector<std::vector<int> > &clusters = nbody == 3 ? neighborlist_3mers : neighborlist_4mers;
	const int *cpid = nbody == 3 ? mb_tri_pid.data() : mb_quad_pid.data();

	double **f   = atom->f;
	int     *typ = atom->type;
	const bool per_cluster_tally = evflag && (eflag_atom || vflag_atom);

	std::vector<double> stensor(6);
	double E_sum = 0.0, S_sum[6] = {0,0,0,0,0,0};
	int    atmidxlst[6][2];
	int    pids[6], ctyp[4];
	double dEdr[6];

	// scatter one evaluated cluster
	auto apply = [&](size_t c, double E, const double *dE) {
		const int *at = clusters[c].data();
		double S[6] = {0,0,0,0,0,0};
		for (int s = 0; s < npair; s++)
		{
			const int    p    = cpid[npair*c + s] >> 1;
			const double sgn  = (cpid[npair*c + s] & 1) ? -1.0 : 1.0;
			const double *d   = &mb_pair_dr[3*p];
			const double fs   = dE[s] / mb_pair_r[p];
			const double fx = fs*sgn*d[0], fy = fs*sgn*d[1], fz = fs*sgn*d[2];
			const int u = at[slots[s][0]], v = at[slots[s][1]];
			f[u][0] += fx; f[u][1] += fy; f[u][2] += fz;
			f[v][0] -= fx; f[v][1] -= fy; f[v][2] -= fz;
			S[0] -= fs*d[0]*d[0]; S[1] -= fs*d[0]*d[1]; S[2] -= fs*d[0]*d[2];
			S[3] -= fs*d[1]*d[1]; S[4] -= fs*d[1]*d[2]; S[5] -= fs*d[2]*d[2];
		}
		if (per_cluster_tally)
		{
			for (int s = 0; s < npair; s++) { atmidxlst[s][0] = at[slots[s][0]]; atmidxlst[s][1] = at[slots[s][1]]; }
			for (int k = 0; k < 6; k++) stensor[k] = S[k];
			ev_tally_mb(nbody, npair, atmidxlst, E, stensor);
		}
		else
		{
			E_sum += E;
			for (int k = 0; k < 6; k++) S_sum[k] += S[k];
		}
	};

	if (use_simd)
	{
		const int W = chimesMBFast::LANES;
		if ((int)mb_bucket.size() < mbfast.nkeys(nbody)) mb_bucket.resize(mbfast.nkeys(nbody));
		for (auto &b : mb_bucket) b.clear();
		int bp[chimesMBFast::LANES * 6];
		double bE[chimesMBFast::LANES], bG[chimesMBFast::LANES * 6];
		auto flush = [&](int key) {
			std::vector<int> &b = mb_bucket[key];
			const int nb = b.size();
			for (int w = 0; w < nb; w++) for (int s = 0; s < npair; s++) bp[w*npair + s] = cpid[npair*b[w] + s] >> 1;
			mbfast.eval_batch(nbody, key, bp, nb, bE, bG);
			for (int w = 0; w < nb; w++) apply(b[w], bE[w], &bG[w*npair]);
			b.clear();
		};
		for (size_t c = 0; c < clusters.size(); c++)
		{
			const int *at = clusters[c].data();
			for (int a = 0; a < nbody; a++) ctyp[a] = chimes_type[typ[at[a]]-1];
			for (int s = 0; s < npair; s++) pids[s] = cpid[npair*c + s] >> 1;
			const int key = nbody == 3 ? mbfast.key3(ctyp) : mbfast.key4(ctyp);
			if (!mbfast.active(nbody, key, pids)) continue;
			mb_bucket[key].push_back(c);
			if ((int)mb_bucket[key].size() == W) flush(key);
		}
		for (int k = 0; k < (int)mb_bucket.size(); k++) if (!mb_bucket[k].empty()) flush(k);
	}
	else
	for (size_t c = 0; c < clusters.size(); c++)
	{
		const int *at = clusters[c].data();
		for (int a = 0; a < nbody; a++) ctyp[a] = chimes_type[typ[at[a]]-1];
		for (int s = 0; s < npair; s++) pids[s] = cpid[npair*c + s] >> 1;
		double E;
		const bool hit = nbody == 3 ? mbfast.eval3(pids, ctyp, E, dEdr) : mbfast.eval4(pids, ctyp, E, dEdr);
		if (hit) apply(c, E, dEdr);
	}
	// Global-only tallies: one call carries the summed energy and virial
	if (evflag && !per_cluster_tally && !clusters.empty())
	{
		const int *at = clusters[0].data();
		for (int s = 0; s < npair; s++) { atmidxlst[s][0] = at[slots[s][0]]; atmidxlst[s][1] = at[slots[s][1]]; }
		for (int k = 0; k < 6; k++) stensor[k] = S_sum[k];
		ev_tally_mb(nbody, npair, atmidxlst, E_sum, stensor);
	}
}

void PairCHIMES::set_chimes_type()
{
	if(comm->me == 0)
		std::cout << "Attempting to match LAMMPS and ChIMES atom types by comparing masses. Looking for matches within a tolerance of 1e-3." << std::endl;
	int nmatches = 0;

	chimes_type.assign(atom->ntypes, -1);

	for (int i=1; i<= atom->ntypes; i++) // Lammps indexing starts at 1
	{
		bool matched = false;
		for (int j=0; j<chimes_calculator.natmtyps; j++) // ChIMES indexing starts at 0
		{
			if (comm->me == 0)
				std::cout << "LAMMPS atom type idx: " << i << " LAMMPS mass: " << atom->mass[i] << " ; " << "ChIMES atom type idx: " << j << " ChIMES mass: " << chimes_calculator.masses[j] << " ... status: " << std::endl;
			if (abs(atom->mass[i] - chimes_calculator.masses[j]) < 1e-3) // Masses should match to at least 3 decimal places
			{
				chimes_type[i-1] = j;
				nmatches++;
				matched = true;
				if(comm->me == 0)
					std::cout << "Match" << std::endl;
				break;
			}
		}
		if (!matched && comm->me == 0)
			std::cout << "No match" << std::endl;
	}

	if (nmatches == 0 && comm->me == 0)
	{
		std::cout << "ERROR: No element matches found between LAMMPS data file and ChIMES parameter file!" << std::endl;
		std::cout << "       Either check your atom masses or do not request ChIMES interactions" << std::endl;
		exit(0);
	}
	else if (nmatches < atom->ntypes && comm->me == 0)
	{
		// Commented out to allow for hybrid/overlay pair style. For example, combining ChIMES description for carbon 
		// with LJ description of Ar for simulations of carbon nanoparticles in an argon bath
		//std::cout << "ERROR: LAMMPS coordinate file has " << atom->ntypes << " atom type masses" << std::endl; 
		//std::cout << "       but only found " << nmatches << " matches with the ChIMES parameter file." << std::endl;
		//exit(0);
 
		std::cout << "WARNING: LAMMPS coordinate file has " << atom->ntypes << " atom type masses" << std::endl;
		std::cout << "       but only found " << nmatches << " matches with the ChIMES parameter file." << std::endl;
		std::cout << "       Will not use ChIMES to evaluate interactions related to unmatched atom types!" << std::endl;
	}
}
							
void PairCHIMES::write_restart(){}			
void PairCHIMES::read_restart(){}				
void PairCHIMES::write_restart_settings(){}		
void PairCHIMES::read_restart_settings(){}	
void PairCHIMES::single(){}					
