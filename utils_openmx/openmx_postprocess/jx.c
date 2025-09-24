/**********************************************************************

jx.c:

This program code calculates spin-spin interaction
coupling constant J between the selected two atoms

Log of jx.c:
   30/Aug/2003  Released by Myung Joon Han (supervised by Prof. J. Yu)
    7/Dec/2003  Modified by Taisuke Ozaki
   03/Mar/2011  Modified by Fumiyuki Ishii for MPI
   02/May/2018  Modified by Asako Terasawa for blas
   10/Jul/2019  Modified by Asako Terasawa for jx.config etc.
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
//#include <malloc/malloc.h>
//#include <assert.h>

#include "lapack_prototypes.h"
#include "f77func.h"
#include "mpi.h"

#include "read_scfout.h"
#include "jx_LNO.h"
#include "jx.h"
#include "jx_tools.h"
#include "jx_config.h"
#include "jx_total_mem.h"
//#include "jx_param_species_atoms.h"
//#include "jx_minimal_basis.h"

#define Host_ID       0         /* ID of the host CPU in MPI */

int main(int argc, char *argv[])
{

  MPI_Comm comm1;
  int numprocs,myid,ID,ID1, *arpo;
  int flag_cell;
  double TStime,TEtime;
  char *s_vec[20];

  int i,j,k,l,n,p,spin;

  MPI_Status stat;
  MPI_Request request;
  MPI_Init(&argc, &argv);
  comm1 = MPI_COMM_WORLD;
  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

  dtime(&TStime);
  total_mem=0.0;

  if (myid==Host_ID){
    printf("\n********************************************************************\n");
    printf("********************************************************************\n");
    printf(" jx: code for calculating an effective exchange coupling constant J \n");
    printf(" Copyright (C), 2003, Myung Joon Han, Jaejun Yu, and Taisuke Ozaki \n");
    printf("                2019, Asako Terasawa and Taisuke Ozaki \n");
    printf(" This is free software, and you are welcome to         \n");
    printf(" redistribute it under the constitution of the GNU-GPL.\n");
    printf("********************************************************************\n");
    printf("********************************************************************\n");
  }
  
  if (argc<=1){
    if (myid==0) printf("\n  scfout file is not found.\n\n");
    MPI_Finalize();
    exit(0);
  }
  else if (argc<=2){
    if (myid==0) printf("\n  config file is not found.\n\n");
    MPI_Finalize();
    exit(0);
  }

  read_scfout(argv);
  MPI_Barrier(comm1);
  filename_jxconfig=argv[2];

  read_jx_config(filename_jxconfig,Solver);
  MPI_Barrier(comm1);

/*  if (flag_minimal == 1 && Solver==3 ){
    read_species_from_omxinput("./temporal_12345.input");
    define_minimal_basis();
    basis_to_minimal_matrix();
    read_atoms_from_omxinput("./temporal_12345.input");
    SpeMinMat_to_IfMinMat();
    if ( myid == Host_ID ){
      printf("\n flag_minimal = 1 : minimal basis projection\n");
      printf("   Number of species %i\n\n",SpeciesNum);
      for (p=0; p<SpeciesNum; p++){
        printf("   Species %i\n",p);
        printf("     Name of species: %s\n",SpeName[p]);
        printf("     Name of basis: %s\n",SpeBasisName[p]);
        printf("     Number of basis taken into account:\n");
        for (l=0; l<=Supported_MaxL; l++){
          printf("         l = %i : %i \n", l, Spe_Num_Basis[p][l]);
        }
        printf("     Number of minimal basis:\n");
        for (l=0; l<=Supported_MaxL; l++){
          printf("         l = %i : %i \n", l, Spe_Min_Basis[p][l]);
        }
        printf("     Projection function to minimal basis matrix \n");
        printf("       ");
        for (n=0; n<Spe_NumOrb[p]; n++){
          printf("%i ", Spe_Min_Mat[p][n]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
  else if (flag_minimal == 1){
    printf(" WARNING: flag_minimal is currently not supported in cluster calculations\n\n");
  } */

  if (flag_LNO==1 && Solver==3){
    if (myid==Host_ID){
      printf("\n flag_LNO = 1: Localized natural orbital representation\n");
      printf("   lambda = %f\n",LNO_Occ_Cutoff);
    }
    LNO_alloc(comm1);
    LNO_Col_Diag(comm1);
    LNO_occ_trns_mat(comm1);
  }
  else if ( flag_LNO==1 ){
    printf("\n WARNING: LNO representation is not supported in cluster calculations.\n");
  }

  s_vec[0]="Recursion"; s_vec[1]="Cluster"; s_vec[2]="Band";
  s_vec[3]="NEGF";      s_vec[4]="DC";      s_vec[5]="GDC";
  s_vec[6]="Cluster2";

  if (myid==Host_ID){
    printf("\n Previous eigenvalue solver = %s\n",s_vec[Solver-1]);
    printf(" atomnum                    = %i\n",atomnum);
    printf(" ChemP                      = %15.12f (Hartree)\n",ChemP);
    printf(" E_Temp                     = %15.12f (K)\n",E_Temp);
  }

  MPI_Barrier(comm1);

  if (Solver==2 || Solver==7){
    Jij_cluster(argc, argv, comm1);
  }
  else if (Solver==3) {
    if (myid==Host_ID){
      printf("\n Jij calculation for a periodic structure\n");
      printf("   Number of k-grids: %i %i %i \n", num_Kgrid[0], num_Kgrid[1], num_Kgrid[2]);
      if ( num_Kgrid[0]<=0 || num_Kgrid[1]<=0 || num_Kgrid[2]<=0 ){
        printf("\n   ERROR: invalid number of k-grids\n");
        MPI_Abort(comm1,0);
        MPI_Finalize();
        exit(0);
      }
    }
    if (flag_periodic_sum == 1 ){
      if (myid==Host_ID){
        printf("   flag_periodic_sum = 1: summation of couplings between periodic images of sites i and j \n");
      }
      Jij_band_psum(comm1);
    }
    else{
      if (myid==Host_ID){
        printf("   flag_periodic_sum = 0: coupling between site i at cell 0 and site j at cell R \n");
        printf("     Number of poles of Fermi-Dirac continued fraction (PRB.75.035123): %i\n",num_poles);
      }
      Jij_band_indiv(comm1);
    }
  }

  MPI_Barrier(comm1);

  dtime(&TEtime);
  if (myid == Host_ID ){
    printf("\n Elapsed time = %lf (s)",TEtime-TStime);
    printf("\n");
  }
  MPI_Barrier(comm1);

  if (flag_LNO==1 && Solver==3){
    LNO_free(comm1);
  }

  MPI_Finalize();
  exit(0);

}
