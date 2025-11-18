#include <mpi.h>

#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<cmath>
#include<algorithm> // For sort
#include <iomanip>
#include <dirent.h>
#include <cstring>

#include "chimesFF.h"

using namespace std;
using namespace GlobalParams;

int nprocs;
int my_rank;
double alpha = 1.0;
double max_composition_difference_2b = 0.0;
double max_composition_difference_3b = 0.0;
double max_composition_difference_4b = 0.0;

// Debug flag - set to true to enable detailed debug output
bool DEBUG_MODE = false;
int DEBUG_SAMPLE_SIZE = 10; // Number of samples to print debug info for

#include <iomanip>
#include "chimesFF.h"  // Make sure this includes the GlobalParams namespace declaration

void print_global_params()
{

    // Print 2-body cutoffs
    if (my_rank == 0)
    {
    cout << "\n2-Body Cutoffs:\n";
    for (size_t i = 0; i < rcut_2b_list.size(); ++i)
    {
        cout << "Pair " << i << ": ";
        cout << "Inner = " << fixed << setprecision(4) << rcut_2b_list[i][0];
        cout << ", Outer = " << rcut_2b_list[i][1] << "\n";
    }


    // Print 3-body cutoffs
    cout << "\n3-Body Cutoffs:\n";
    for (size_t i = 0; i < rcut_3b_list.size(); ++i)
    {
        cout << "Triplet " << i << ":\n";
        for (size_t j = 0; j < rcut_3b_list[i].size(); ++j)
    {
            cout << "  Pair " << j << ": ";
            cout << "Inner = " << rcut_3b_list[i][j][0];
            cout << ", Outer = " << rcut_3b_list[i][j][1] << "\n";
        }
    }

    // Print 4-body cutoffs
    cout << "\n4-Body Cutoffs:\n";
    for (size_t i = 0; i < rcut_4b_list.size(); ++i)
    {
        cout << "Quadruplet " << i << ":\n";
        for (size_t j = 0; j < rcut_4b_list[i].size(); ++j)
    {
            cout << "  Pair " << j << ": ";
            cout << "Inner = " << rcut_4b_list[i][j][0];
            cout << ", Outer = " << rcut_4b_list[i][j][1] << "\n";
        }
    }

    // Print Morse lambda values
    cout << "\nMorse Lambda Values:\n";
    for (size_t i = 0; i < morse_lambda_list.size(); ++i)
    {
        cout << "Pair " << i << ": λ = "
                  << fixed << setprecision(4)
                  << morse_lambda_list[i] << "\n";
    }
    }
}

int split_lines(string line, vector<string> & items)
{
    // Break a line up into tokens based on space separators.
    // Returns the number of tokens parsed.

    string       contents;
    stringstream sstream;

    // Strip comments beginining with ! or ## and terminal new line

    int pos = line.find('!');

    if ( pos != string::npos )
        line.erase(pos, line.length() - pos);

    pos = line.find("##");
    if ( pos != string::npos )
        line.erase(pos, line.length()-pos);

    pos = line.find('\n');
    if ( pos != string::npos )
        line.erase(pos, 1);

    sstream.str(line);

    items.clear();

    while ( sstream >> contents )
        items.push_back(contents);

    return items.size();
}

bool get_next_line(istream& str, string & line)
{
    // Read a line and return it, with error checking.

        getline(str, line);

        if(!str)
            return false;

    return true;
}

double transform(double rcin, double rcout, double lambda, double rij)
{
    double x_min = exp(-1*rcin/lambda);
    double x_max = exp(-1*rcout/lambda);

    double x_avg   = 0.5 * (x_max + x_min);
    double x_diff  = 0.5 * (x_max - x_min);

    x_diff *= -1.0; // Special for Morse style

    return (exp(-1*rij/lambda) - x_avg)/x_diff;
}

// Function to read the clusters and construct adjacency matrices
void read_flat_clusters(string clufile, int npairs_per_cluster, vector<double > & clusters, vector<double> & atom_types, const int body_cnt) {
    ifstream clustream(clufile);
    if (!clustream.is_open()) {
        cerr << "ERROR: Could not open file " << clufile << endl;
        exit(0);
    }

    string line;
    vector<string> line_contents;
    int n_contents;

    while (get_next_line(clustream, line)) {
        n_contents = split_lines(line, line_contents);

        if (n_contents != npairs_per_cluster + body_cnt) { // body_cnt additional columns for atom types
            cout << "ERROR: Read the wrong number of clusters!" << endl;
            cout << "n_contents: " << n_contents << endl;
            cout << "Expected: " << npairs_per_cluster + body_cnt << endl;
            exit(0);
        }

        // Extract edge lengths and atom types
        vector<double> edge_lengths(npairs_per_cluster);
        vector<int> typ_idxs(body_cnt);
        vector<double> tmp_desc(body_cnt);

        for (int i = 0; i < body_cnt; i++) {
            typ_idxs[i] = stoi(line_contents[npairs_per_cluster + i]);
            tmp_desc[i] = atomic_descriptors[stoi(line_contents[npairs_per_cluster + i])];
        }
        atom_types.insert(atom_types.end(), tmp_desc.begin(), tmp_desc.end());

        double cutoff_0, cutoff_00;
        double cutoff_1, cutoff_01;
        double cutoff_2, cutoff_02;
        double cutoff_3, cutoff_03;
        double cutoff_4, cutoff_04;
        double cutoff_5, cutoff_05;
        double cutoff_6, cutoff_06;

        double morse_pair_1, morse_pair_2, morse_pair_3, morse_pair_4, morse_pair_5, morse_pair_6;

        if (body_cnt == 2){
            int pair_idx = atom_int_pair_mapping[ typ_idxs[0]*atom_typ_cnt + typ_idxs[1] ];
            cutoff_0 = rcut_2b_list[pair_idx][1];
            cutoff_00 = rcut_2b_list[pair_idx][0];
            morse_pair_1 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[0]*atom_typ_cnt + typ_idxs[1]]];
            edge_lengths[0] = transform(cutoff_00, cutoff_0, morse_pair_1, stod(line_contents[0]));
        } else if (body_cnt == 3){
            int type_idx =  typ_idxs[0]*atom_typ_cnt*atom_typ_cnt + typ_idxs[1]*atom_typ_cnt + typ_idxs[2];
            int tripidx = atom_int_trip_mapping[type_idx];
            vector<int> & mapped_pair_idx_3b = pair_int_trip_mapping[type_idx];
            // Get cutoffs
            cutoff_0  = rcut_3b_list[ tripidx ][1][mapped_pair_idx_3b[0]]; // outer cutoff
            cutoff_00 = rcut_3b_list[ tripidx ][0][mapped_pair_idx_3b[0]]; // inner cutoff
            cutoff_1  = rcut_3b_list[ tripidx ][1][mapped_pair_idx_3b[1]];
            cutoff_01 = rcut_3b_list[ tripidx ][0][mapped_pair_idx_3b[1]];
            cutoff_2  = rcut_3b_list[ tripidx ][1][mapped_pair_idx_3b[2]];
            cutoff_02 = rcut_3b_list[ tripidx ][0][mapped_pair_idx_3b[2]];
            // Get morse variables
            morse_pair_1 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[0]*atom_typ_cnt + typ_idxs[1]]];
            morse_pair_2 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[0]*atom_typ_cnt + typ_idxs[2]]];
            morse_pair_3 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[1]*atom_typ_cnt + typ_idxs[2]]];
            // Assign edge lengths
            edge_lengths[0] = transform(cutoff_00, cutoff_0, morse_pair_1, stod(line_contents[0]));
            edge_lengths[1] = transform(cutoff_01, cutoff_1, morse_pair_2, stod(line_contents[1]));
            edge_lengths[2] = transform(cutoff_02, cutoff_2, morse_pair_3, stod(line_contents[2]));
        } else {
            int idx = typ_idxs[0]*atom_typ_cnt*atom_typ_cnt*atom_typ_cnt + typ_idxs[1]*atom_typ_cnt*atom_typ_cnt + typ_idxs[2]*atom_typ_cnt + typ_idxs[3] ;
            int quadidx = atom_int_quad_mapping[idx] ;
            vector<int> & mapped_pair_idx_4b = pair_int_quad_mapping[idx] ;
            // Get cutoffs
            cutoff_0  = rcut_4b_list[ quadidx ][1][mapped_pair_idx_4b[0]];
            cutoff_00 = rcut_4b_list[ quadidx ][0][mapped_pair_idx_4b[0]];
            cutoff_1  = rcut_4b_list[ quadidx ][1][mapped_pair_idx_4b[1]];
            cutoff_01 = rcut_4b_list[ quadidx ][0][mapped_pair_idx_4b[1]];
            cutoff_2  = rcut_4b_list[ quadidx ][1][mapped_pair_idx_4b[2]];
            cutoff_02 = rcut_4b_list[ quadidx ][0][mapped_pair_idx_4b[2]];
            cutoff_3  = rcut_4b_list[ quadidx ][1][mapped_pair_idx_4b[3]];
            cutoff_03 = rcut_4b_list[ quadidx ][0][mapped_pair_idx_4b[3]];
            cutoff_4  = rcut_4b_list[ quadidx ][1][mapped_pair_idx_4b[4]];
            cutoff_04 = rcut_4b_list[ quadidx ][0][mapped_pair_idx_4b[4]];
            cutoff_5  = rcut_4b_list[ quadidx ][1][mapped_pair_idx_4b[5]];
            cutoff_05 = rcut_4b_list[ quadidx ][0][mapped_pair_idx_4b[5]];
            // Get morse variables
            morse_pair_1 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[0]*atom_typ_cnt + typ_idxs[1]]];
            morse_pair_2 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[0]*atom_typ_cnt + typ_idxs[2]]];
            morse_pair_3 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[0]*atom_typ_cnt + typ_idxs[3]]];
            morse_pair_4 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[1]*atom_typ_cnt + typ_idxs[2]]];
            morse_pair_5 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[1]*atom_typ_cnt + typ_idxs[3]]];
            morse_pair_6 = morse_lambda_list[atom_int_pair_mapping[ typ_idxs[2]*atom_typ_cnt + typ_idxs[3]]];
            // Assign edge lengths
            edge_lengths[0] = transform(cutoff_00, cutoff_0, morse_pair_1, stod(line_contents[0]));
            edge_lengths[1] = transform(cutoff_01, cutoff_1, morse_pair_2, stod(line_contents[1]));
            edge_lengths[2] = transform(cutoff_02, cutoff_2, morse_pair_3, stod(line_contents[2]));
            edge_lengths[3] = transform(cutoff_03, cutoff_3, morse_pair_4, stod(line_contents[3]));
            edge_lengths[4] = transform(cutoff_04, cutoff_4, morse_pair_5, stod(line_contents[4]));
            edge_lengths[5] = transform(cutoff_05, cutoff_5, morse_pair_6, stod(line_contents[5]));
        }
        sort(edge_lengths.begin(), edge_lengths.end());
        clusters.insert(clusters.end(), edge_lengths.begin(), edge_lengths.end());
    }

    clustream.close();
}

int get_bin(double binw, double maxval, double dist)
{
    int bin = floor(dist/binw);

    if (dist == maxval)
        return bin-1;
    else
        return bin;
}

void divide_task(int & my_rank_start, int & my_rank_end, int tasks)
{
    int procs_used;

    // Deal with no tasks to perform.
    if ( tasks <= 0 )
    {
      my_rank_start = 1 ;
      my_rank_end = 0 ;
      return ;
    }

    // Deal gracefully with more tasks than processors.
    if ( nprocs <= tasks )
        procs_used = nprocs;
    else
        procs_used = tasks;

    // Use ceil so the last process always has fewer tasks than the other
    // This improves load balancing.
    my_rank_start = ceil( (double) my_rank * tasks / procs_used);

    if ( my_rank > tasks )
    {
        my_rank_start = tasks + 1;
        my_rank_end = tasks - 1;
    }
    else if ( my_rank == procs_used - 1 )
    {
        // End of the list.
        my_rank_end = tasks - 1;
    }
    else
    {
        // Next starting value - 1 .
        my_rank_end   = ceil( (double) (my_rank+1) * tasks / procs_used ) - 1;
        if ( my_rank_end > tasks - 1 )
            my_rank_end = tasks - 1;
    }
}

// ============================================================================
// HYBRID COMPOSITION DISTANCE APPROACH
// ============================================================================
// This approach combines:
// 1. Position-aware weighted composition (via edge ranking) for 3b and 4b
// 2. Direct Wasserstein distance for 2b (since position doesn't matter)
//
// The key insight: Your original edge-ranking approach is correct for
// capturing geometric position, but the issue was in the normalization
// and the fact that it computed a "weight" rather than a "distance".
//
// Now we compute composition descriptors for each cluster, then take the
// Euclidean distance between them.
// ============================================================================

double compute_composition_descriptor_2b(vector<double> edges, vector<double> atom_types) {
    // For 2-body, position doesn't matter, just return sorted average
    vector<double> sorted_types = atom_types;
    sort(sorted_types.begin(), sorted_types.end());
    return (sorted_types[0] + sorted_types[1]) / 2.0;
}

vector<double> compute_composition_descriptor_3b(vector<double> edges, vector<double> atom_types) {
    // For 3-body, use edge-ranking to preserve positional information
    // Edge ordering (after sorting): edge[0] < edge[1] < edge[2]
    // Edge[0] connects atoms closest together
    // Edge[2] connects atoms furthest apart

    // Mapping: edge[0] = 0-1, edge[1] = 0-2, edge[2] = 1-2
    // Sum of edge lengths for each atom:
    // Atom 0: edge[0] + edge[1]
    // Atom 1: edge[0] + edge[2]
    // Atom 2: edge[1] + edge[2]

    vector<pair<int, double>> sum_edge_lengths(3);
    sum_edge_lengths[0] = {0, edges[0] + edges[1]};
    sum_edge_lengths[1] = {1, edges[0] + edges[2]};
    sum_edge_lengths[2] = {2, edges[1] + edges[2]};

    // Sort by edge sum to get position ranking
    sort(sum_edge_lengths.begin(), sum_edge_lengths.end(),
        [](const pair<int, double>& a, const pair<int, double>& b) {
            return a.second < b.second;
        });

    // Create position-weighted descriptor vector
    // Atoms with smaller edge sums (more central) get higher weight
    vector<double> descriptor(3);
    descriptor[0] = atom_types[sum_edge_lengths[0].first]; // Most central
    descriptor[1] = atom_types[sum_edge_lengths[1].first]; // Middle
    descriptor[2] = atom_types[sum_edge_lengths[2].first]; // Most peripheral

    return descriptor;
}

vector<double> compute_composition_descriptor_4b(vector<double> edges, vector<double> atom_types) {
    // For 4-body, use edge-ranking to preserve positional information
    // Edge ordering (after sorting): edges in increasing order

    // Mapping for 4 atoms (0,1,2,3):
    // edge[0] = 0-1, edge[1] = 0-2, edge[2] = 0-3
    // edge[3] = 1-2, edge[4] = 1-3, edge[5] = 2-3

    // Sum of edge lengths for each atom:
    vector<pair<int, double>> sum_edge_lengths(4);
    sum_edge_lengths[0] = {0, edges[0] + edges[1] + edges[2]};
    sum_edge_lengths[1] = {1, edges[0] + edges[3] + edges[4]};
    sum_edge_lengths[2] = {2, edges[1] + edges[3] + edges[5]};
    sum_edge_lengths[3] = {3, edges[2] + edges[4] + edges[5]};

    // Sort by edge sum to get position ranking
    sort(sum_edge_lengths.begin(), sum_edge_lengths.end(),
        [](const pair<int, double>& a, const pair<int, double>& b) {
            return a.second < b.second;
        });

    // Create position-weighted descriptor vector
    vector<double> descriptor(4);
    descriptor[0] = atom_types[sum_edge_lengths[0].first]; // Most central
    descriptor[1] = atom_types[sum_edge_lengths[1].first];
    descriptor[2] = atom_types[sum_edge_lengths[2].first];
    descriptor[3] = atom_types[sum_edge_lengths[3].first]; // Most peripheral

    return descriptor;
}

double compute_composition_distance_2b(vector<double> edges1, vector<double> atom_types1,
                                        vector<double> edges2, vector<double> atom_types2) {
    // For 2-body: Use Wasserstein distance (position doesn't matter)
    vector<double> sorted1 = atom_types1;
    vector<double> sorted2 = atom_types2;
    sort(sorted1.begin(), sorted1.end());
    sort(sorted2.begin(), sorted2.end());

    double dist = 0.0;
    for (size_t i = 0; i < sorted1.size(); i++) {
        dist += abs(sorted1[i] - sorted2[i]);
    }
    return dist / sorted1.size();
}

double compute_composition_distance_3b(vector<double> edges1, vector<double> atom_types1,
                                        vector<double> edges2, vector<double> atom_types2) {
    // Get position-aware descriptors for both clusters
    vector<double> desc1 = compute_composition_descriptor_3b(edges1, atom_types1);
    vector<double> desc2 = compute_composition_descriptor_3b(edges2, atom_types2);

    // Compute Euclidean distance between descriptors
    double dist = 0.0;
    for (size_t i = 0; i < desc1.size(); i++) {
        dist += pow(desc1[i] - desc2[i], 2.0);
    }
    return sqrt(dist) / sqrt(desc1.size()); // Normalize by sqrt(n) for consistency
}

double compute_composition_distance_4b(vector<double> edges1, vector<double> atom_types1,
                                        vector<double> edges2, vector<double> atom_types2) {
    // Get position-aware descriptors for both clusters
    vector<double> desc1 = compute_composition_descriptor_4b(edges1, atom_types1);
    vector<double> desc2 = compute_composition_descriptor_4b(edges2, atom_types2);

    // Compute Euclidean distance between descriptors
    double dist = 0.0;
    for (size_t i = 0; i < desc1.size(); i++) {
        dist += pow(desc1[i] - desc2[i], 2.0);
    }
    return sqrt(dist) / sqrt(desc1.size()); // Normalize by sqrt(n) for consistency
}

void gen_flat_hists(vector<double > & clu1, vector<double > & clu2, vector<double> & clu1_atm_types, vector<double> & clu2_atm_types, int n_cluster_pairs, int nbin, double binw, double maxd, string histfile, bool same = false, int body_cnt = 2)
{
    int                     bin;
    double                  normalized_structure_distance;
    double                  normalized_composition_distance;
    double                  total_dist;
    double                  dist_structure;
    vector<long long int>   my_hist(nbin,0);
    vector<long long int>   hist(nbin,0);
    long long int           my_nsamples = 0;
    long long int           nsamples = 0;
    int                     my_rank_start;
    int                     my_rank_end;
    int                     looptwo_start;
    int                     total_tasks;
    int                     status;
    int                     maxIntValue = numeric_limits<int>::max();
    double                  dist_struct;
    double                  comp_dist;

    // Error checking variables
    int                     debug_sample_count = 0;
    long long int           out_of_bounds_count = 0;
    long long int           negative_dist_count = 0;
    double                  min_total_dist = 1e10;
    double                  max_total_dist = -1e10;
    double                  min_comp_dist = 1e10;
    double                  max_comp_dist = -1e10;

    // Distribute outer loop over processors

    divide_task(my_rank_start, my_rank_end, clu1.size()/n_cluster_pairs);    // Divide atoms on a per-processor basis.
    total_tasks = my_rank_end-my_rank_start;

    if(my_rank ==0)
    {
        cout << "Dividing " << clu1.size()/n_cluster_pairs << " tasks across " << nprocs << " processors" << endl;
        cout << "Alpha = " << alpha << endl;
        cout << "Atom type count = " << atom_typ_cnt << endl;
        cout << "Max descriptor = " << max_descr << endl;
        cout << "Min descriptor = " << min_descr << endl;

        if (body_cnt == 2)
            cout << "Max composition difference (2b) = " << max_composition_difference_2b << endl;
        else if (body_cnt == 3)
            cout << "Max composition difference (3b) = " << max_composition_difference_3b << endl;
        else if (body_cnt == 4)
            cout << "Max composition difference (4b) = " << max_composition_difference_4b << endl;

        cout << "Number of bins = " << nbin << endl;
        cout << "Bin width = " << binw << endl;
        cout << "Max distance = " << maxd << endl;
        cout << "Using hybrid composition distance (position-aware for 3b/4b)" << endl;
    }

    if (total_tasks>0)
    {

    for (int i=my_rank_start; i<=my_rank_end; i++)
    {

        // Print progress

        status = double(i-my_rank_start)/(total_tasks)*100.0;


    // This logic needed to avoid div by zero when total_tasks/10 is zero (since they are integer types)
        if (my_rank == 0)
        if ((total_tasks/10) == 0)
                    cout << histfile << " Completion percent: " << status << " " << i << " of " << total_tasks << " assigned" << endl;
        else if(i%(total_tasks/10) == 0)
            cout << histfile << " Completion percent: " << status << " " << i << " of " << total_tasks << " assigned" << endl;

        // Modify bounds in case this is a self-calculation
        if (same)
            looptwo_start = i+1;
        else
            looptwo_start = 0;

        // Compute the distances
        for (int j=looptwo_start; j<clu2.size()/n_cluster_pairs; j++)
        {
            dist_struct = 0;
            vector<double> edge_length_1(n_cluster_pairs);
            vector<double> edge_length_2(n_cluster_pairs);
            vector<double>    atom_list_1(body_cnt);
            vector<double>    atom_list_2(body_cnt);

            for (int k=0; k<n_cluster_pairs; k++){
                // Distance calculation between two clusters
                edge_length_1[k] = clu1[i*n_cluster_pairs+k];
                edge_length_2[k] = clu2[j*n_cluster_pairs+k];
                dist_struct += pow(clu1[i*n_cluster_pairs+k] - clu2[j*n_cluster_pairs+k],2.0);
            }

            for (int l=0; l<body_cnt; l++){
                atom_list_1[l] = clu1_atm_types[i*body_cnt+l];
                atom_list_2[l] = clu2_atm_types[j*body_cnt+l];
            }

            // Compute composition distance (method depends on body count)
            if (body_cnt == 2) {
                comp_dist = compute_composition_distance_2b(edge_length_1, atom_list_1,
                                                            edge_length_2, atom_list_2);
            } else if (body_cnt == 3) {
                comp_dist = compute_composition_distance_3b(edge_length_1, atom_list_1,
                                                            edge_length_2, atom_list_2);
            } else if (body_cnt == 4) {
                comp_dist = compute_composition_distance_4b(edge_length_1, atom_list_1,
                                                            edge_length_2, atom_list_2);
            } else {
                cout << "Improper body count: " << body_cnt << endl;
                exit(1);
            }

            // Track min/max composition distances
            if (comp_dist < min_comp_dist) min_comp_dist = comp_dist;
            if (comp_dist > max_comp_dist) max_comp_dist = comp_dist;

            // Normalize composition distance
            double max_comp_diff;
            if (body_cnt == 2)
                max_comp_diff = max_composition_difference_2b;
            else if (body_cnt == 3)
                max_comp_diff = max_composition_difference_3b;
            else if (body_cnt == 4)
                max_comp_diff = max_composition_difference_4b;
            else {
                cout << "Improper body count: " << body_cnt << endl;
                exit(1);
            }

            // Error checking: ensure normalization denominator is not zero
            if (max_comp_diff < 1e-10) {
                if (my_rank == 0 && debug_sample_count == 0) {
                    cout << "WARNING: max_composition_difference is nearly zero ("
                         << max_comp_diff << ")" << endl;
                    cout << "This will cause all composition distances to be zero or infinite!" << endl;
                }
                normalized_composition_distance = 0.0;
            } else {
                normalized_composition_distance = comp_dist / max_comp_diff;
            }

            // Compute normalized structure distance
            // Maximum possible Euclidean distance in n-dimensional space with coordinates in [-1,1]
            // is sqrt(4*n) = 2*sqrt(n), where n is the number of pairs
            normalized_structure_distance = sqrt(dist_struct) / (2.0*sqrt(n_cluster_pairs));

            // Combine structural and compositional distances
            total_dist = alpha * normalized_structure_distance + (1.0 - alpha) * normalized_composition_distance;

            // Track min/max distances for diagnostics
            if (total_dist < min_total_dist) min_total_dist = total_dist;
            if (total_dist > max_total_dist) max_total_dist = total_dist;

            // Debug output for first few samples
            if (DEBUG_MODE && my_rank == 0 && debug_sample_count < DEBUG_SAMPLE_SIZE) {
                cout << "\n=== DEBUG Sample " << debug_sample_count << " (i=" << i << ", j=" << j << ") ===" << endl;
                cout << "  Edge lengths 1: [";
                for (int k=0; k<n_cluster_pairs; k++)
                    cout << edge_length_1[k] << (k<n_cluster_pairs-1 ? ", " : "]\n");
                cout << "  Edge lengths 2: [";
                for (int k=0; k<n_cluster_pairs; k++)
                    cout << edge_length_2[k] << (k<n_cluster_pairs-1 ? ", " : "]\n");
                cout << "  Atom types 1: [";
                for (int l=0; l<body_cnt; l++)
                    cout << atom_list_1[l] << (l<body_cnt-1 ? ", " : "]\n");
                cout << "  Atom types 2: [";
                for (int l=0; l<body_cnt; l++)
                    cout << atom_list_2[l] << (l<body_cnt-1 ? ", " : "]\n");

                // Show composition descriptors for 3b/4b
                if (body_cnt == 3) {
                    vector<double> desc1 = compute_composition_descriptor_3b(edge_length_1, atom_list_1);
                    vector<double> desc2 = compute_composition_descriptor_3b(edge_length_2, atom_list_2);
                    cout << "  Composition descriptor 1: [";
                    for (int l=0; l<body_cnt; l++)
                        cout << desc1[l] << (l<body_cnt-1 ? ", " : "]\n");
                    cout << "  Composition descriptor 2: [";
                    for (int l=0; l<body_cnt; l++)
                        cout << desc2[l] << (l<body_cnt-1 ? ", " : "]\n");
                } else if (body_cnt == 4) {
                    vector<double> desc1 = compute_composition_descriptor_4b(edge_length_1, atom_list_1);
                    vector<double> desc2 = compute_composition_descriptor_4b(edge_length_2, atom_list_2);
                    cout << "  Composition descriptor 1: [";
                    for (int l=0; l<body_cnt; l++)
                        cout << desc1[l] << (l<body_cnt-1 ? ", " : "]\n");
                    cout << "  Composition descriptor 2: [";
                    for (int l=0; l<body_cnt; l++)
                        cout << desc2[l] << (l<body_cnt-1 ? ", " : "]\n");
                }

                cout << "  Composition distance (raw): " << comp_dist << endl;
                cout << "  Max composition difference: " << max_comp_diff << endl;
                cout << "  Normalized composition distance: " << normalized_composition_distance << endl;
                cout << "  Structure distance (raw): " << sqrt(dist_struct) << endl;
                cout << "  Normalized structure distance: " << normalized_structure_distance << endl;
                cout << "  Alpha: " << alpha << endl;
                cout << "  Total distance: " << total_dist << endl;
                debug_sample_count++;
            }

            // Error checking: negative distances
            if (total_dist < 0) {
                negative_dist_count++;
                if (my_rank == 0 && negative_dist_count <= 5) {
                    cout << "ERROR: Negative total distance detected: " << total_dist << endl;
                    cout << "  normalized_structure_distance: " << normalized_structure_distance << endl;
                    cout << "  normalized_composition_distance: " << normalized_composition_distance << endl;
                }
                total_dist = 0.0; // Clamp to zero
            }

            // Compute bin with improved bounds checking
            bin = get_bin(binw, maxd, total_dist);

            // Robust bounds checking
            if (bin < 0) {
                out_of_bounds_count++;
                if (my_rank == 0 && out_of_bounds_count <= 5) {
                    cout << "WARNING: Computed bin < 0 (bin=" << bin << ", dist=" << total_dist
                         << "). Clamping to bin 0." << endl;
                }
                bin = 0;
            }

            if (bin >= nbin) {
                out_of_bounds_count++;
                if (my_rank == 0 && out_of_bounds_count <= 5) {
                    cout << "WARNING: Computed bin >= nbin (bin=" << bin << ", nbin=" << nbin
                         << ", dist=" << total_dist << "). Clamping to bin " << nbin-1 << endl;
                    cout << "  Details: binw=" << binw << ", maxd=" << maxd << endl;
                }
                bin = nbin - 1;
            }

            my_hist[bin] += 1;
            my_nsamples += 1;
        }
    }
    }

    if (my_rank == 0)
    {
        cout << "\nLoop done, printing statistics: " << endl;
        cout << "  My samples counted: " << my_nsamples << endl;
        cout << "  Min total distance: " << min_total_dist << endl;
        cout << "  Max total distance: " << max_total_dist << endl;
        cout << "  Min composition distance: " << min_comp_dist << endl;
        cout << "  Max composition distance: " << max_comp_dist << endl;
        if (out_of_bounds_count > 0)
            cout << "  Out-of-bounds bins: " << out_of_bounds_count << endl;
        if (negative_dist_count > 0)
            cout << "  Negative distances: " << negative_dist_count << endl;
    }

    MPI_Reduce(my_hist.data(), hist.data(), nbin, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_nsamples, &nsamples, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);


    // Print results

    if(my_rank == 0)
    {
        cout << "Total samples across all ranks: " << nsamples << endl;

        // Additional histogram statistics
        long long int total_count = 0;
        long long int nonempty_bins = 0;
        for (int i=0; i<hist.size(); i++) {
            total_count += hist[i];
            if (hist[i] > 0) nonempty_bins++;
        }

        cout << "Histogram statistics:" << endl;
        cout << "  Total count in histogram: " << total_count << endl;
        cout << "  Non-empty bins: " << nonempty_bins << " / " << nbin << endl;
        cout << "  First bin count: " << hist[0] << " ("
             << 100.0*hist[0]/total_count << "%)" << endl;
        cout << "  Last bin count: " << hist[nbin-1] << " ("
             << 100.0*hist[nbin-1]/total_count << "%)" << endl;

        ofstream cluhist;
        cluhist.open(histfile);

        for (int i=0; i<hist.size(); i++)
             cluhist << i*binw+0.5*binw <<  " " << double(hist[i]) /nsamples << endl;

        cluhist.close();

        cout << "Histogram written to: " << histfile << endl;
    }

}

// To compile - mpiicc -O3 -o histogram multi_calc_histogram_hybrid.cpp chimesFF.cpp
int main(int argc, char *argv[])
{
    my_rank = 0;
    nprocs = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Start total timing
    double time_total_start = MPI_Wtime();

    if (my_rank == 0) {
        cout << "=== Cluster Histogram Generator (Hybrid Composition Distance) ===" << endl;
        cout << "Nprocs = " << nprocs << endl;
    }

    // Parse command line arguments
    if (argc < 3) {
        if (my_rank == 0) {
            cout << "ERROR: Insufficient arguments!" << endl;
            cout << "Usage: " << argv[0] << " <parameter_file> <alpha> [debug]" << endl;
            cout << "  parameter_file: ChIMES parameter file" << endl;
            cout << "  alpha: mixing parameter (0=composition only, 1=structure only)" << endl;
            cout << "  debug: optional flag to enable debug output" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string param_file = argv[1];
    alpha = atof(argv[2]);

    // Check for debug flag
    if (argc > 3 && string(argv[3]) == "debug") {
        DEBUG_MODE = true;
        if (my_rank == 0)
            cout << "Debug mode enabled" << endl;
    }

    // Validate alpha
    if (alpha < 0.0 || alpha > 1.0) {
        if (my_rank == 0) {
            cout << "ERROR: Alpha must be between 0.0 and 1.0" << endl;
            cout << "  Provided alpha = " << alpha << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (my_rank == 0) {
        cout << "Alpha parameter: " << alpha << endl;
        if (alpha == 0.0)
            cout << "  -> Using composition distance only (position-aware)" << endl;
        else if (alpha == 1.0)
            cout << "  -> Using structure distance only" << endl;
        else
            cout << "  -> Using mixed distance metric" << endl;
    }

    vector<string> all_files;
    int num_files = 0;

    // File discovery and sorting (rank 0 only)
    if (my_rank == 0) {
        DIR *dir = opendir(".");
        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr) {
            string filename = entry->d_name;
            size_t num_len = 0;
            while (num_len < filename.size() && isdigit(filename[num_len]))
                num_len++;

            if (num_len > 0 && filename.substr(num_len) == ".all-2b-clusters.txt")
                all_files.push_back(filename.substr(0, num_len));
        }
        closedir(dir);

        sort(all_files.begin(), all_files.end(), [](const string &a, const string &b) {
            return stoi(a) < stoi(b);
        });

        num_files = all_files.size();
        cout << "Found " << num_files << " cluster files" << endl;
    }

    // Broadcast file list
    MPI_Bcast(&num_files, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (num_files == 0) {
        if (my_rank == 0)
            cout << "ERROR: No cluster files found in current directory!" << endl;
        MPI_Finalize();
        return 1;
    }

    all_files.resize(num_files);

    for (int i = 0; i < num_files; ++i) {
        int str_len = 0;
        char buffer[256] = {0};

        if (my_rank == 0) {
            str_len = all_files[i].size();
            strncpy(buffer, all_files[i].c_str(), 255);
        }

        MPI_Bcast(&str_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(buffer, 256, MPI_CHAR, 0, MPI_COMM_WORLD);

        all_files[i] = string(buffer, str_len);
    }

    // Initialize ChIMES calculator
    if (my_rank == 0)
        cout << "\nInitializing ChIMES force field..." << endl;

    double time_init_start = MPI_Wtime();
    chimesFF ff;
    ff.init(my_rank);
    ff.read_parameters(param_file);
    ff.build_pair_int_trip_map();
    ff.build_pair_int_quad_map();
    double time_init_end = MPI_Wtime();

    if (my_rank == 0) {
        cout << "ChIMES initialization complete" << endl;
        cout << "  Initialization time: " << time_init_end - time_init_start << " seconds" << endl;
        print_global_params();
    }

    // Compute max composition differences
    // For 2b: Wasserstein distance, max = (max_descr - min_descr)
    // For 3b/4b: Euclidean distance of position-aware descriptors
    //   Max occurs when one cluster has all min_descr and other has all max_descr
    //   in the most dissimilar positional arrangement

    max_composition_difference_2b = (max_descr - min_descr);

    // For 3b: max Euclidean distance when descriptors are [min, min, min] vs [max, max, max]
    // Distance = sqrt(3 * (max - min)^2) / sqrt(3) = (max - min)
    max_composition_difference_3b = (max_descr - min_descr);

    // For 4b: similarly
    max_composition_difference_4b = (max_descr - min_descr);

    if (my_rank == 0) {
        cout << "\nComposition distance normalization factors (Hybrid):" << endl;
        cout << "  2-body max composition diff: " << max_composition_difference_2b
             << " (Wasserstein)" << endl;
        cout << "  3-body max composition diff: " << max_composition_difference_3b
             << " (Position-aware Euclidean)" << endl;
        cout << "  4-body max composition diff: " << max_composition_difference_4b
             << " (Position-aware Euclidean)" << endl;

        // Warnings for potential issues
        if (max_composition_difference_2b < 1e-10) {
            cout << "\nWARNING: max_composition_difference is nearly zero!" << endl;
            cout << "This suggests min_descr ~= max_descr (single element type?)" << endl;
            cout << "Composition distances will be zero or ill-defined." << endl;

            if (alpha == 0.0) {
                cout << "CRITICAL: Alpha=0 with no composition variation!" << endl;
                cout << "All distances will be zero - all clusters in first bin!" << endl;
            }
        }
    }

    // Process individual files
    for (const auto &file_idx : all_files) {
        string f1_idx = file_idx;
        string f2_idx = file_idx;  // Same file for both indices

        double time_file_start = MPI_Wtime();

        if (my_rank == 0) {
            cout << "\n========================================" << endl;
            cout << "Processing file: " << file_idx << endl;
            cout << "========================================" << endl;
        }

        // File paths - using same index for both
        string f1_2b = f1_idx + ".all-2b-clusters.txt";
        string f2_2b = f2_idx + ".all-2b-clusters.txt";

        // Histogram parameters
        const int nbin_2b = 100, nbin_3b = 100, nbin_4b = 100;
        const double maxd_2b = 1.0,
                     maxd_3b = 1.0,
                     maxd_4b = 1.0;

        double binw_2b = maxd_2b/nbin_2b;
        double binw_3b = maxd_3b/nbin_3b;
        double binw_4b = maxd_4b/nbin_4b;

        // Process 2B clusters
        if (my_rank == 0)
            cout << "\n--- Processing 2-body clusters ---" << endl;

        double time_2b_read_start = MPI_Wtime();
        vector<double> f1_2b_flat_clusters, f2_2b_flat_clusters;
        vector<double> f1_2b_atom_types, f2_2b_atom_types;
        int npairs_2b = 1;
        read_flat_clusters(f1_2b, npairs_2b, f1_2b_flat_clusters, f1_2b_atom_types, 2);
        if (my_rank == 0)
            cout << "Read " << f1_2b_flat_clusters.size()/npairs_2b << " 2B clusters from file 1" << endl;
        read_flat_clusters(f2_2b, npairs_2b, f2_2b_flat_clusters, f2_2b_atom_types, 2);
        if (my_rank == 0)
            cout << "Read " << f2_2b_flat_clusters.size()/npairs_2b << " 2B clusters from file 2" << endl;
        double time_2b_read_end = MPI_Wtime();
        if (my_rank == 0)
            cout << "  2B read time: " << time_2b_read_end - time_2b_read_start << " seconds" << endl;

        // Process 3B clusters
        if (my_rank == 0)
            cout << "\n--- Processing 3-body clusters ---" << endl;

        double time_3b_read_start = MPI_Wtime();
        string f1_3b = f1_idx + ".all-3b-clusters.txt";
        string f2_3b = f2_idx + ".all-3b-clusters.txt";
        vector<double> f1_3b_flat_clusters, f2_3b_flat_clusters;
        vector<double> f1_3b_atom_types, f2_3b_atom_types;
        int npairs_3b = 3;
        read_flat_clusters(f1_3b, npairs_3b, f1_3b_flat_clusters, f1_3b_atom_types, 3);
        if (my_rank == 0)
            cout << "Read " << f1_3b_flat_clusters.size()/npairs_3b << " 3B clusters from file 1" << endl;
        read_flat_clusters(f2_3b, npairs_3b, f2_3b_flat_clusters, f2_3b_atom_types, 3);
        if (my_rank == 0)
            cout << "Read " << f2_3b_flat_clusters.size()/npairs_3b << " 3B clusters from file 2" << endl;
        double time_3b_read_end = MPI_Wtime();
        if (my_rank == 0)
            cout << "  3B read time: " << time_3b_read_end - time_3b_read_start << " seconds" << endl;

        // Process 4B clusters
        if (my_rank == 0)
            cout << "\n--- Processing 4-body clusters ---" << endl;

        double time_4b_read_start = MPI_Wtime();
        string f1_4b = f1_idx + ".all-4b-clusters.txt";
        string f2_4b = f2_idx + ".all-4b-clusters.txt";
        vector<double> f1_4b_flat_clusters, f2_4b_flat_clusters;
        vector<double> f1_4b_atom_types, f2_4b_atom_types;
        int npairs_4b = 6;
        read_flat_clusters(f1_4b, npairs_4b, f1_4b_flat_clusters, f1_4b_atom_types, 4);
        if (my_rank == 0)
            cout << "Read " << f1_4b_flat_clusters.size()/npairs_4b << " 4B clusters from file 1" << endl;
        read_flat_clusters(f2_4b, npairs_4b, f2_4b_flat_clusters, f2_4b_atom_types, 4);
        if (my_rank == 0)
            cout << "Read " << f2_4b_flat_clusters.size()/npairs_4b << " 4B clusters from file 2" << endl;
        double time_4b_read_end = MPI_Wtime();
        if (my_rank == 0)
            cout << "  4B read time: " << time_4b_read_end - time_4b_read_start << " seconds" << endl;

        if (my_rank == 0) {
            cout << "\nHistogram parameters:" << endl;
            cout << "  Max distances: 2b=" << maxd_2b << " 3b=" << maxd_3b << " 4b=" << maxd_4b << endl;
            cout << "  Bin widths: 2b=" << binw_2b << " 3b=" << binw_3b << " 4b=" << binw_4b << endl;
        }

        bool same = (f1_idx == f2_idx);

        if (my_rank == 0)
            cout << "\n--- Generating 2-body histogram ---" << endl;
        double time_2b_hist_start = MPI_Wtime();
        gen_flat_hists(f1_2b_flat_clusters, f2_2b_flat_clusters, f1_2b_atom_types,
                      f2_2b_atom_types, npairs_2b, nbin_2b, binw_2b, maxd_2b,
                      f1_idx + "-" + f2_idx + ".2b_clu-s.hist", same, 2);
        double time_2b_hist_end = MPI_Wtime();
        if (my_rank == 0)
            cout << "  2B histogram time: " << time_2b_hist_end - time_2b_hist_start << " seconds" << endl;

        if (my_rank == 0)
            cout << "\n--- Generating 3-body histogram ---" << endl;
        double time_3b_hist_start = MPI_Wtime();
        gen_flat_hists(f1_3b_flat_clusters, f2_3b_flat_clusters, f1_3b_atom_types,
                      f2_3b_atom_types, npairs_3b, nbin_3b, binw_3b, maxd_3b,
                      f1_idx + "-" + f2_idx + ".3b_clu-s.hist", same, 3);
        double time_3b_hist_end = MPI_Wtime();
        if (my_rank == 0)
            cout << "  3B histogram time: " << time_3b_hist_end - time_3b_hist_start << " seconds" << endl;

        if (my_rank == 0)
            cout << "\n--- Generating 4-body histogram ---" << endl;
        double time_4b_hist_start = MPI_Wtime();
        gen_flat_hists(f1_4b_flat_clusters, f2_4b_flat_clusters, f1_4b_atom_types,
                      f2_4b_atom_types, npairs_4b, nbin_4b, binw_4b, maxd_4b,
                      f1_idx + "-" + f2_idx + ".4b_clu-s.hist", same, 4);
        double time_4b_hist_end = MPI_Wtime();
        if (my_rank == 0)
            cout << "  4B histogram time: " << time_4b_hist_end - time_4b_hist_start << " seconds" << endl;

        double time_file_end = MPI_Wtime();
        if (my_rank == 0) {
            cout << "\n========================================" << endl;
            cout << "File " << file_idx << " total time: " << time_file_end - time_file_start << " seconds" << endl;
            cout << "========================================" << endl;
        }
    }

    double time_total_end = MPI_Wtime();

    if (my_rank == 0) {
        cout << "\n=== All histograms generated successfully ===" << endl;
        cout << "\n========================================" << endl;
        cout << "TOTAL EXECUTION TIME: " << time_total_end - time_total_start << " seconds" << endl;
        cout << "========================================" << endl;
    }

    MPI_Finalize();
    return 0;
}
