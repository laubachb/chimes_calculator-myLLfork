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
#ifdef FASTMB
    use_fastmb  = false;
    fastmb_simd = true;
#endif
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
	if (allocated) 
	{   	    
	    memory->destroy(setflag);
	    memory->destroy(cutsq);
	}
    
    if (badness_stream.is_open())
        badness_stream.close();
}	

void PairCHIMES::settings(int narg, char **arg)
{
#ifdef FASTMB
	// The keyword fastmb, anywhere in the argument list, selects the fast many-body evaluation;
	// nosimd evaluates clusters one at a time instead of in SIMD batches (for testing).
	// The remaining arguments are handled as before.

	std::vector<char *> other_args;

	for (int a = 0; a < narg; a++)
	{
		if (utils::strmatch(arg[a], "^fastmb$"))
			use_fastmb = true;
		else if (utils::strmatch(arg[a], "^nosimd$"))
			fastmb_simd = false;
		else
			other_args.push_back(arg[a]);
	}

	narg = other_args.size();
	arg  = other_args.data();
#else
	for (int a = 0; a < narg; a++)
		if (utils::strmatch(arg[a], "^fastmb$"))
			error -> all(FLERR,"pair_style chimesFF fastmb requires LAMMPS to be installed with ./install.sh FASTMB");
#endif

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

#ifdef FASTMB
	// fastmb evaluates the analytic 3- and 4-body polynomials; it does not apply to tabulated
	// 3-body parameters or to fingerprint runs, which need the per-cluster distances
#ifdef TABULATION
	if (use_fastmb && chimes_calculator.tabulate_3B)
	{
		if (comm->me == 0)
			std::cout << "chimesFF: fastmb ignored for tabulated 3-body parameters" << std::endl;
		use_fastmb = false;
	}
#endif
#ifdef FINGERPRINT
	if (use_fastmb && fingerprint)
	{
		if (comm->me == 0)
			std::cout << "chimesFF: fastmb ignored when fingerprinting" << std::endl;
		use_fastmb = false;
	}
#endif
	if (use_fastmb)
	{
		std::string why = chimes_calculator.fastmb_init();

		if (!why.empty())
			error -> all(FLERR, "pair_style chimesFF fastmb: " + why);

		if (comm->me == 0)
			std::cout << "chimesFF: fastmb many-body evaluation enabled; SIMD batches " << (fastmb_simd ? "on" : "off") << std::endl;
	}
#endif
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
			
		build_mb_neighlists();		
#ifdef FASTMB
		if (use_fastmb)
			build_fastmb_pair_registry();
#endif
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

    // Document badness for configuration: current timestep, current rank, worst badness seen by rank
    if (for_fitting)
        if(update->ntimestep % output->every_dump[0] == 0)
            badness_stream << update->ntimestep << " " <<  chimes_calculator.get_badness() << endl;

	// if (chimes_calculator.poly_orders[1] > 0 || tmp_FP)
#ifdef FASTMB
	if (use_fastmb && chimes_calculator.poly_orders[1] > 0)
	{
		////////////////////////////////////////
		// Compute 3- and 4-body interactions through the pair cache
		////////////////////////////////////////

		// Distance and cutoff-weighted polynomials of every registered pair, once per step

		for (int p = 0; p < fastmb_pair_i.size(); p++)
		{
			i = fastmb_pair_i[p];
			j = fastmb_pair_j[p];

			fastmb_pair_r[p] = get_dist(i, j, &fastmb_pair_dr[CHDIM*p]);

			chimes_calculator.fastmb_fill_cache(p, chimes_calculator.fastmb_pair_type(chimes_type[type[i]-1], chimes_type[type[j]-1]), fastmb_pair_r[p]);
		}

		compute_fastmb(3);

		if (chimes_calculator.poly_orders[2] > 0)
			compute_fastmb(4);
	}
	else
#endif
	if (chimes_calculator.poly_orders[1] > 0)
	{
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
#ifdef FASTMB
	if (!use_fastmb)    // Otherwise already done with the 3-body term above
#endif
	if (chimes_calculator.poly_orders[2] > 0)

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

if (vflag_fdotr) 
        virial_fdotr_compute();

	return;
}

#ifdef FASTMB

void PairCHIMES::build_fastmb_pair_registry()
{
	// Give every distinct atom pair of the 3- and 4-body neighbor lists an index, and record for each
	// cluster slot its pair index and whether the slot runs against the stored (i < j) direction.
	// Called whenever the many-body neighbor lists are rebuilt.

	fastmb_pair_i.clear();
	fastmb_pair_j.clear();
	fastmb_3mer_pair_idx.resize(3*neighborlist_3mers.size());
	fastmb_4mer_pair_idx.resize(6*neighborlist_4mers.size());

	std::unordered_map<long long, int> pair_index;	// (i, j) with i < j -> pair idx
	long long nall = atom->nlocal + atom->nghost;

	auto register_pair = [&](int u, int v) -> int
	{
		int       i   = (u < v) ? u : v;
		int       j   = (u < v) ? v : u;
		long long key = (long long)i*nall + j;
		int       idx;

		std::unordered_map<long long, int>::iterator it = pair_index.find(key);

		if (it != pair_index.end())
			idx = it->second;
		else
		{
			idx = fastmb_pair_i.size();
			pair_index[key] = idx;
			fastmb_pair_i.push_back(i);
			fastmb_pair_j.push_back(j);
		}

		return 2*idx + ((u < v) ? 0 : 1);
	};

	for (int c = 0; c < neighborlist_3mers.size(); c++)
	{
		const std::vector<int> & atoms = neighborlist_3mers[c];

		fastmb_3mer_pair_idx[3*c+0] = register_pair(atoms[0], atoms[1]);	// ij
		fastmb_3mer_pair_idx[3*c+1] = register_pair(atoms[0], atoms[2]);	// ik
		fastmb_3mer_pair_idx[3*c+2] = register_pair(atoms[1], atoms[2]);	// jk
	}

	for (int c = 0; c < neighborlist_4mers.size(); c++)
	{
		const std::vector<int> & atoms = neighborlist_4mers[c];

		fastmb_4mer_pair_idx[6*c+0] = register_pair(atoms[0], atoms[1]);	// ij
		fastmb_4mer_pair_idx[6*c+1] = register_pair(atoms[0], atoms[2]);	// ik
		fastmb_4mer_pair_idx[6*c+2] = register_pair(atoms[0], atoms[3]);	// il
		fastmb_4mer_pair_idx[6*c+3] = register_pair(atoms[1], atoms[2]);	// jk
		fastmb_4mer_pair_idx[6*c+4] = register_pair(atoms[1], atoms[3]);	// jl
		fastmb_4mer_pair_idx[6*c+5] = register_pair(atoms[2], atoms[3]);	// kl
	}

	fastmb_pair_r .resize(fastmb_pair_i.size());
	fastmb_pair_dr.resize(CHDIM*fastmb_pair_i.size());

	chimes_calculator.fastmb_resize_cache(fastmb_pair_i.size());
}

void PairCHIMES::compute_fastmb(int nbody)
{
	// 3- or 4-body forces, energy and virial from the pair cache. Each cluster returns its energy and
	// dE/dr for every slot; the forces and stress of a slot follow from dE/dr and the pair vector,
	// as in chimesFF::compute_3B/compute_4B.

	static const int slot_atoms_3b[3][2] = {{0,1},{0,2},{1,2}};
	static const int slot_atoms_4b[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

	const int   npairs         = (nbody == 3) ? 3 : 6;
	const int (*slot_atoms)[2] = (nbody == 3) ? slot_atoms_3b : slot_atoms_4b;

	const std::vector<std::vector<int> > & clusters         = (nbody == 3) ? neighborlist_3mers : neighborlist_4mers;
	const int                            * cluster_pair_idx = (nbody == 3) ? fastmb_3mer_pair_idx.data() : fastmb_4mer_pair_idx.data();

	double **f    = atom->f;
	int     *type = atom->type;

	// Per-atom energy or virial tallies need one ev_tally_mb call per cluster; otherwise the summed
	// energy and virial are tallied once at the end

	bool per_cluster_tally = evflag && (eflag_atom || vflag_atom);

	std::vector<double> stensor(6);
	int    atmidxlst[6][2];
	int    typ_idxs[4];
	int    pair_idxs[6];
	double deriv[6];
	double energy;
	double energy_sum = 0.0;
	double stress_sum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	// Forces and stress of each slot of one evaluated cluster from its dE/dr, then the tallies

	auto apply_cluster = [&](int c, double energy, const double *deriv)
	{
		const int *atoms = clusters[c].data();

		double stress[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

		for (int s = 0; s < npairs; s++)
		{
			int     p       = cluster_pair_idx[npairs*c + s] >> 1;
			double  sign    = (cluster_pair_idx[npairs*c + s] & 1) ? -1.0 : 1.0;	// slot runs against the stored direction
			double *dr      = &fastmb_pair_dr[CHDIM*p];
			double  fscalar = deriv[s] / fastmb_pair_r[p];
			int     u       = atoms[slot_atoms[s][0]];
			int     v       = atoms[slot_atoms[s][1]];

			for (int idx = 0; idx < CHDIM; idx++)
			{
				f[u][idx] += sign * fscalar * dr[idx];
				f[v][idx] -= sign * fscalar * dr[idx];
			}

			stress[0] -= fscalar * dr[0] * dr[0];	// xx tensor component
			stress[1] -= fscalar * dr[0] * dr[1];	// xy tensor component
			stress[2] -= fscalar * dr[0] * dr[2];	// xz tensor component
			stress[3] -= fscalar * dr[1] * dr[1];	// yy tensor component
			stress[4] -= fscalar * dr[1] * dr[2];	// yz tensor component
			stress[5] -= fscalar * dr[2] * dr[2];	// zz tensor component
		}

		if (per_cluster_tally)
		{
			for (int s = 0; s < npairs; s++)
			{
				atmidxlst[s][0] = atoms[slot_atoms[s][0]];
				atmidxlst[s][1] = atoms[slot_atoms[s][1]];
			}

			for (int k = 0; k < 6; k++)
				stensor[k] = stress[k];

			ev_tally_mb(nbody, npairs, atmidxlst, energy, stensor);
		}
		else
		{
			energy_sum += energy;

			for (int k = 0; k < 6; k++)
				stress_sum[k] += stress[k];
		}
	};

	if (fastmb_simd)
	{
		// Active clusters are collected per cluster key and swept fastmb_lanes at a time

		const int lanes = chimesFF::fastmb_lanes;
		const int nkeys = chimes_calculator.fastmb_ncluster_keys(nbody);

		if (fastmb_batches.size() < nkeys)
			fastmb_batches.resize(nkeys);

		for (int k = 0; k < nkeys; k++)
			fastmb_batches[k].clear();

		int    batch_pair_idxs[chimesFF::fastmb_lanes * 6];
		double batch_energy[chimesFF::fastmb_lanes];
		double batch_deriv[chimesFF::fastmb_lanes * 6];

		auto sweep_batch = [&](int key)
		{
			std::vector<int> & batch = fastmb_batches[key];
			int nclusters = batch.size();

			for (int w = 0; w < nclusters; w++)
				for (int s = 0; s < npairs; s++)
					batch_pair_idxs[w*npairs + s] = cluster_pair_idx[npairs*batch[w] + s] >> 1;

			chimes_calculator.fastmb_compute_batch(nbody, key, batch_pair_idxs, nclusters, batch_energy, batch_deriv);

			for (int w = 0; w < nclusters; w++)
				apply_cluster(batch[w], batch_energy[w], &batch_deriv[w*npairs]);

			batch.clear();
		};

		for (int c = 0; c < clusters.size(); c++)
		{
			const int *atoms = clusters[c].data();

			for (int a = 0; a < nbody; a++)
				typ_idxs[a] = chimes_type[type[atoms[a]]-1];

			for (int s = 0; s < npairs; s++)
				pair_idxs[s] = cluster_pair_idx[npairs*c + s] >> 1;

			int key = chimes_calculator.fastmb_cluster_key(nbody, typ_idxs);

			if (!chimes_calculator.fastmb_cluster_active(nbody, key, pair_idxs))
				continue;

			fastmb_batches[key].push_back(c);

			if (fastmb_batches[key].size() == lanes)
				sweep_batch(key);
		}

		for (int k = 0; k < nkeys; k++)
			if (!fastmb_batches[k].empty())
				sweep_batch(k);
	}
	else
	{
		for (int c = 0; c < clusters.size(); c++)
		{
			const int *atoms = clusters[c].data();

			for (int a = 0; a < nbody; a++)
				typ_idxs[a] = chimes_type[type[atoms[a]]-1];

			for (int s = 0; s < npairs; s++)
				pair_idxs[s] = cluster_pair_idx[npairs*c + s] >> 1;

			int key = chimes_calculator.fastmb_cluster_key(nbody, typ_idxs);

			if (chimes_calculator.fastmb_compute_cluster(nbody, key, pair_idxs, energy, deriv))
				apply_cluster(c, energy, deriv);
		}
	}

	// Global-only tallies: one call carries the summed energy and virial (atmidxlst is not used)

	if (evflag && !per_cluster_tally && !clusters.empty())
	{
		const int *atoms = clusters[0].data();

		for (int s = 0; s < npairs; s++)
		{
			atmidxlst[s][0] = atoms[slot_atoms[s][0]];
			atmidxlst[s][1] = atoms[slot_atoms[s][1]];
		}

		for (int k = 0; k < 6; k++)
			stensor[k] = stress_sum[k];

		ev_tally_mb(nbody, npairs, atmidxlst, energy_sum, stensor);
	}
}

#endif

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
