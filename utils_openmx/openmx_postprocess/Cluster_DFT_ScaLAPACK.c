/**********************************************************************
  Cluster_DFT_ScaLAPACK.c:

     Cluster_DFT_ScaLAPACK.c is a subroutine to perform cluster calculations

  Log of Cluster_DFT.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time   0




static double Cluster_collinear(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double **ko,
                   double *****nh, 
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2],
		   int myworld1,
		   int *NPROCS_ID1,
		   int *Comm_World1,
		   int *NPROCS_WD1,
		   int *Comm_World_StartID1,
		   MPI_Comm *MPI_CommWD1,
                   int *MP,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *Hs,
		   double *CDM1,
		   int size_H1,
                   int *SP_NZeros,
                   int *SP_Atoms,
                   double **EVec1,
                   double *Work1);


static double Cluster_non_collinear(char *mode,
				    int SCF_iter,
				    int SpinP_switch,
				    double *****nh,
				    double *****ImNL,
				    double ****CntOLP,
				    double *****CDM,
				    double *****EDM,
                                    int *is2, int *ie2,
                                    int *MP,
                                    double *DM1,
                                    double *Work1,
                                    dcomplex *EVec1,
                                    double *ko,
                                    int size_H1, 
				    double Eele0[2], double Eele1[2]);

static void Save_DOS_Col(int n, int MaxN, int *MP, double ****OLP0, double **EVec1, 
                         double **ko, int *NPROCS_WD1, int *Comm_World_StartID1);

static void Save_DOS_NonCol(int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko);


double Cluster_DFT_ScaLAPACK(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double **Cluster_ko,
                   double *ko_NonCol,
                   double *****nh,
                   double *****ImNL,
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2],
		   int myworld1,
		   int *NPROCS_ID1,
		   int *Comm_World1,
		   int *NPROCS_WD1,
		   int *Comm_World_StartID1,
		   MPI_Comm *MPI_CommWD1,
                   int *MP,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *Hs, 
		   double *CDM1,
		   int size_H1, 
                   int *SP_NZeros,
                   int *SP_Atoms, 
                   double **EVec1_Re,
                   dcomplex *EVec1_NonCol,
                   double *Work1)
{
  static double time0;

  /****************************************************
         collinear without spin-orbit coupling
  ****************************************************/

  if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==0 ){

    time0 = Cluster_collinear(mode,SCF_iter,SpinP_switch,Cluster_ko,
                              nh,CntOLP,CDM,EDM,Eele0,Eele1,
			      myworld1,NPROCS_ID1,Comm_World1,NPROCS_WD1,
			      Comm_World_StartID1,MPI_CommWD1,MP,is2,ie2,
    			      Ss,Cs,Hs,CDM1,size_H1,SP_NZeros,SP_Atoms,EVec1_Re,Work1);
  }

  /****************************************************
           collinear with spin-orbit coupling
  ****************************************************/

  else if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==1 ){
    printf("Spin-orbit coupling is not supported for collinear DFT calculations.\n");
    MPI_Finalize();
    exit(1);
  }

  /****************************************************
   non-collinear with and without spin-orbit coupling
  ****************************************************/

  else if (SpinP_switch==3){  

    time0 = Cluster_non_collinear( mode,SCF_iter,SpinP_switch,nh,ImNL,CntOLP,CDM,EDM,is2,ie2,MP,
                                   CDM1,Work1,EVec1_NonCol,ko_NonCol,size_H1,Eele0,Eele1 );
  }

  return time0;
}




static double Cluster_collinear(
				char *mode,
				int SCF_iter,
				int SpinP_switch,
				double **ko,
				double *****nh, double ****CntOLP,
				double *****CDM,
				double *****EDM,
				double Eele0[2], double Eele1[2],
				int myworld1,
				int *NPROCS_ID1,
				int *Comm_World1,
				int *NPROCS_WD1,
				int *Comm_World_StartID1,
				MPI_Comm *MPI_CommWD1,
                                int *MP,
                                int *is2,
                                int *ie2,
				double *Ss,
				double *Cs,
				double *Hs, 
				double *CDM1,
                                int size_H1, 
                                int *SP_NZeros,
                                int *SP_Atoms, 
                                double **EVec1,
                                double *Work1)
{
  static int firsttime=1;
  int i,j,l,n,n2,n1,i1,i1s,j1,k1,l1;
  int wan,HOMO0,HOMO1;
  int spin,po,num0,num1,ires;
  int ct_AN,k,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,loopN,Gc_AN;
  int MA_AN,LB_AN,GB_AN,Bnum,MaxN;
  int wan1,mul,m,bcast_flag;
  int *is1,*ie1;
  double time0,lumos,av_num;
  double *OneD_Mat1;
  double ***H;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double sum0,sum1,sum2,sum3;
  double My_Eele1[2],tmp1,tmp2;
  double Num_State,x,FermiF,Dnum,Dnum2;
  double FermiF2,x2,diffF;
  double dum,ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  double EV_cut0;
  double res;
  int numprocs0,myid0;
  int numprocs1,myid1;
  int ID,p,world_Snd,world_Rcv; 
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  double OLP_eigen_cut=Threshold_OLP_Eigen;
  char file_EV[YOUSO10] = ".EV";
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp_EV;
  double stime, etime;
  double time1,time2,time3,time4,time5,time6,time7;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int info,ig,jg,il,jl,prow,pcol,brow,bcol;
  int ZERO=0, ONE=1;
  double alpha = 1.0; double beta = 0.0;
  int LOCr, LOCc, node, irow, icol;
  double C_spin_i1,mC_spin_i1;
  int sp;

  int ID0,IDS,IDR,Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;
  MPI_Status stat;
  MPI_Request request;

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n = n + Spe_Total_CNO[wanA];
  }
  n2 = n + 2;

  /****************************************************
   Allocation

   double  H[List_YOUSO[23]][n2][n2]  
  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);

  is1 = (int*)malloc(sizeof(int)*numprocs1);
  ie1 = (int*)malloc(sizeof(int)*numprocs1);

  Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs1);
  Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs1);

  /* for PrintMemory */
  firsttime=0;

  if (measure_time){
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
  }

  if      (SpinP_switch==0) spin_degeneracy = 2.0;
  else if (SpinP_switch==1) spin_degeneracy = 1.0;

  /****************************************************
                   total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ = TZ + Spe_Core_Charge[wan];
  }

  /****************************************************
         find the numbers of partions for MPI
  ****************************************************/

  if ( numprocs1<=n ){

    av_num = (double)n/(double)numprocs1;

    for (ID=0; ID<numprocs1; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs1-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs1; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  /****************************************************
       1. diagonalize the overlap matrix     
       2. search negative eigenvalues
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  if (SCF_iter==1){
    Overlap_Cluster_Ss(CntOLP,Cs,MP,myworld1);
  }

  if (SpinP_switch==1 && numprocs0==1){
    Hamiltonian_Cluster_Hs(nh[0],Hs,MP,0,0);
  }
  else{
    for (spin=0; spin<=SpinP_switch; spin++){
      Hamiltonian_Cluster_Hs(nh[spin],Hs,MP,spin,myworld1);
    } 
  }

  if (SCF_iter==1){

    if (measure_time) dtime(&stime);

    MPI_Comm_split(MPI_CommWD1[myworld1],my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(MPI_CommWD1[myworld1],my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    if (scf_eigen_lib_flag==1){
      F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&n, &n, Cs, &na_rows, &ko[0][1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
    }

    else if (scf_eigen_lib_flag==2){

#ifndef kcomp

      int mpiworld;
      mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

      F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(&n, &n, Cs, &na_rows, &ko[0][1], 
                                            Ss, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);

#endif
    }

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    /* print to the standard output */

    if (2<=level_stdout && myid0==Host_ID){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[0][l]);fflush(stdout);
      }
    }

    /* minus eigenvalues to 1.0e-10 */

    for (l=1; l<=n; l++){
      if (ko[0][l]<0.0) ko[0][l] = 1.0e-10;
    }

    /* calculate S*1/sqrt(ko) */

    for (l=1; l<=n; l++){
      ko[0][l] = 1.0/sqrt(ko[0][l]);
    }

    for(i=0; i<na_rows; i++){
      for(j=0; j<na_cols; j++){
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	Ss[j*na_rows+i] = Ss[j*na_rows+i]*ko[0][jg];
      }
    }

    if (measure_time){
      dtime(&etime);
      time1 += etime - stime; 
    }
  }

  /****************************************************
    calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  /* find the maximum states in solved eigenvalues */

  if (SCF_iter==1){
    MaxN = n;
  }
  else{

    if ( strcasecmp(mode,"scf")==0 ) 
      lumos = (double)n*0.100;
    else if ( strcasecmp(mode,"dos")==0 )
      lumos = (double)n*0.200;      

    if (lumos<400.0) lumos = 400.0;
    MaxN = (TZ-system_charge)/2 + (int)lumos;
    if (n<MaxN) MaxN = n;
    
    if (cal_partial_charge) MaxN = n; 
  }

  if ( numprocs1<=MaxN ){
    
    av_num = (double)MaxN/(double)numprocs1;
    for (ID=0; ID<numprocs1; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }
    
    is2[0] = 1;
    ie2[numprocs1-1] = MaxN; 
  }

  else{

    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }

    for (ID=MaxN; ID<numprocs1; ID++){
      is2[ID] =  1;
      ie2[ID] =  0;
    }
  }

  /* making data structure of MPI communicaition for eigenvectors */

  for (ID=0; ID<numprocs1; ID++){
    Num_Snd_EV[ID] = 0;
    Num_Rcv_EV[ID] = 0;
  }

  for (i=0; i<na_rows; i++){

    ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;

    po = 0;
    for (ID=0; ID<numprocs1; ID++){
      if (is2[ID]<=ig && ig <=ie2[ID]){
        po = 1;
        ID0 = ID;
        break;
      }
    }

    if (po==1) Num_Snd_EV[ID0] += na_cols;
  }

  for (ID=0; ID<numprocs1; ID++){
    IDS = (myid1 + ID) % numprocs1;
    IDR = (myid1 - ID + numprocs1) % numprocs1;
    if (ID!=0){
      MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
    }
  }

  Max_Num_Snd_EV = 0;
  Max_Num_Rcv_EV = 0;
  for (ID=0; ID<numprocs1; ID++){
    if (Max_Num_Snd_EV<Num_Snd_EV[ID]) Max_Num_Snd_EV = Num_Snd_EV[ID];
    if (Max_Num_Rcv_EV<Num_Rcv_EV[ID]) Max_Num_Rcv_EV = Num_Rcv_EV[ID];
  }  

  Max_Num_Snd_EV++;
  Max_Num_Rcv_EV++;

  index_Snd_i = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  index_Snd_j = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  EVec_Snd = (double*)malloc(sizeof(double)*Max_Num_Snd_EV);
  index_Rcv_i = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  index_Rcv_j = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  EVec_Rcv = (double*)malloc(sizeof(double)*Max_Num_Rcv_EV);

  /* initialize ko */
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i1=1; i1<=n; i1++){
      ko[spin][i1] = 10000.0;
    }
  }

  /* spin=myworld1 */

  spin = myworld1;

 diagonalize:

  if (measure_time) dtime(&stime);

  /* pdgemm */

  /* H * U * 1.0/sqrt(ko[l]) */

  for(i=0; i<na_rows_max*na_cols_max; i++){
    Cs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

  for(i=0; i<na_rows*na_cols; i++){
    Hs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /* The output C matrix is distributed by column. */

  if (measure_time) dtime(&stime);

  MPI_Comm_split(MPI_CommWD1[myworld1],my_pcol,my_prow,&mpi_comm_rows);
  MPI_Comm_split(MPI_CommWD1[myworld1],my_prow,my_pcol,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  if (scf_eigen_lib_flag==1){
    F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&n, &MaxN, Hs, &na_rows, &ko[spin][1], Cs, 
                                            &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
  }
  else if (scf_eigen_lib_flag==2){

#ifndef kcomp

    int mpiworld;
    mpiworld = MPI_Comm_c2f(MPI_CommWD1[myworld1]);

    F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(&n, &MaxN, Hs, &na_rows, &ko[spin][1], 
									 		    Cs, &na_rows, &nblk, &na_cols, 
                                                                           &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);
#endif
  }

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  if (measure_time){
    dtime(&etime);
    time3 += etime - stime;
  }

  /****************************************************
      transformation to the original eigenvectors.
                       NOTE 244P
  ****************************************************/

  if (measure_time) dtime(&stime);

  for(i=0;i<na_rows*na_cols;i++){
    Hs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("T","T",&n,&n,&n,&alpha,Cs,&ONE,&ONE,descC,Ss,&ONE,&ONE,descS,&beta,Hs,&ONE,&ONE,descH);

  /* MPI communications of Hs */

  for (ID=0; ID<numprocs1; ID++){
    
    IDS = (myid1 + ID) % numprocs1;
    IDR = (myid1 - ID + numprocs1) % numprocs1;

    k = 0;
    for(i=0; i<na_rows; i++){
      ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
      if (is2[IDS]<=ig && ig <=ie2[IDS]){

        for (j=0; j<na_cols; j++){
          jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
 
          index_Snd_i[k] = ig;
          index_Snd_j[k] = jg;
          EVec_Snd[k] = Hs[j*na_rows+i];
          k++; 
	}
      }
    }

    if (ID!=0){

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(EVec_Snd, Num_Snd_EV[IDS], MPI_DOUBLE, IDS, 999, MPI_CommWD1[myworld1], &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR], MPI_DOUBLE, IDR, 999, MPI_CommWD1[myworld1], &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }
    }
    else{
      for(k=0; k<Num_Snd_EV[IDS]; k++){
        index_Rcv_i[k] = index_Snd_i[k];
        index_Rcv_j[k] = index_Snd_j[k];
        EVec_Rcv[k] = EVec_Snd[k];
      } 
    }

    for(k=0; k<Num_Rcv_EV[IDR]; k++){
      ig = index_Rcv_i[k];
      jg = index_Rcv_j[k];
      m = (ig-is2[myid1])*n + jg - 1;
      EVec1[spin][m] = EVec_Rcv[k];
    }
  }

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;
    Hamiltonian_Cluster_Hs(nh[spin],Hs,MP,spin,spin);
    goto diagonalize; 
  }

  /*********************************************** 
    MPI: ko
  ***********************************************/

  if (measure_time) dtime(&stime);

  for (sp=0; sp<=SpinP_switch; sp++){
    MPI_Bcast(&ko[sp][1],MaxN,MPI_DOUBLE,Comm_World_StartID1[sp],mpi_comm_level1);
  }

  if ( strcasecmp(mode,"scf")==0 ){

    if (2<=level_stdout){
      for (i1=1; i1<=MaxN; i1++){
	if (SpinP_switch==0)
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,ko[0][i1],ko[0][i1]);
	else 
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,ko[0][i1],ko[1][i1]);
      }
    }

    /****************************************************
              searching of chemical potential
    ****************************************************/

    /* first, find ChemP at five times large temperatue */

    po = 0;
    loopN = 0;

    ChemP_MAX = 30.0;  
    ChemP_MIN =-30.0;
  
    do {

      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      Num_State = 0.0;

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i1=1; i1<=MaxN; i1++){
	  x = (ko[spin][i1] - ChemP)*Beta*0.2;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = 1.0/(1.0 + exp(x));
	  Num_State = Num_State + spin_degeneracy*FermiF;
	  if (0.5<FermiF) Cluster_HOMO[spin] = i1;
	}
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-14) po = 1;

      if (myid1==Host_ID && 2<=level_stdout){
	printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",ChemP,TZ,Num_State); 
      }

      loopN++;

    } while (po==0 && loopN<1000); 

    /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

    po = 0;
    loopN = 0;

    ChemP_MAX = 30.0;  
    ChemP_MIN =-30.0;
  
    do {

      if (loopN!=0){
	ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      }

      Num_State = 0.0;

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i1=1; i1<=MaxN; i1++){
	  x = (ko[spin][i1] - ChemP)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = 1.0/(1.0 + exp(x));
	  Num_State = Num_State + spin_degeneracy*FermiF;
	  if (0.5<FermiF) Cluster_HOMO[spin] = i1;
	}
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-14) po = 1;

      if (myid1==Host_ID && 2<=level_stdout){
	printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",ChemP,TZ,Num_State); 
      }

      loopN++;

    } 
    while (po==0 && loopN<1000); 

    if (2<=level_stdout){
      printf("  ChemP=%15.12f\n",ChemP);
    }

    if (measure_time){
      dtime(&etime);
      time5 += etime - stime;
    }

    /****************************************************
            Energies by summing up eigenvalues
    ****************************************************/

    Eele0[0] = 0.0;
    Eele0[1] = 0.0;
    for (spin=0; spin<=SpinP_switch; spin++){
      for (i1=1; i1<=MaxN; i1++){

	x = (ko[spin][i1] - ChemP)*Beta;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = FermiFunc(x,spin,i1,&po,&x);
	Eele0[spin] += ko[spin][i1]*FermiF;

      }
    }
    if (SpinP_switch==0){
      Eele0[1] = Eele0[0];
    }

    /****************************************************
      LCAO coefficients are stored for calculating
               values of MOs on grids
    ****************************************************/

    if (SpinP_switch==0){
      if ( (Cluster_HOMO[0]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[0];
      if ( (Cluster_HOMO[0]+num_LUMOs)>MaxN ) num_LUMOs = MaxN - Cluster_HOMO[0];
    }
    else if (SpinP_switch==1){
      if ( (Cluster_HOMO[0]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[0];
      if ( (Cluster_HOMO[1]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[1];
      if ( (Cluster_HOMO[0]+num_LUMOs)>MaxN ) num_LUMOs = MaxN - Cluster_HOMO[0];
      if ( (Cluster_HOMO[1]+num_LUMOs)>MaxN ) num_LUMOs = MaxN - Cluster_HOMO[1];
    }

    if (myid0==Host_ID){
      if (SpinP_switch==0 && 2<=level_stdout){
	printf("  HOMO = %2d\n",Cluster_HOMO[0]);
      }
      else if (SpinP_switch==1 && 2<=level_stdout){
	printf("  HOMO for up-spin   = %2d\n",Cluster_HOMO[0]);
	printf("  HOMO for down-spin = %2d\n",Cluster_HOMO[1]);
      }
    }

    if (MO_fileout==1){  

      /* allocation of arrays */
      double *array0;    
      int *is3,*ie3;
      int numprocs3,ID1;

      array0 = (double*)malloc(sizeof(double)*(n+2));
      is3 = (int*)malloc(sizeof(int)*numprocs0);
      ie3 = (int*)malloc(sizeof(int)*numprocs0);

      /* HOMOs */

      for (spin=0; spin<=SpinP_switch; spin++){

	/* set is3 and ie3 */

	numprocs3 = NPROCS_WD1[spin];

	if ( numprocs3<=MaxN ){

	  av_num = (double)MaxN/(double)numprocs3;
	  for (ID=0; ID<numprocs3; ID++){
	    is3[ID] = (int)(av_num*(double)ID) + 1; 
	    ie3[ID] = (int)(av_num*(double)(ID+1)); 
	  }

	  is3[0] = 1;
	  ie3[numprocs3-1] = MaxN; 
	}

	else{
	  for (ID=0; ID<MaxN; ID++){
	    is3[ID] = ID + 1; 
	    ie3[ID] = ID + 1;
	  }
	  for (ID=MaxN; ID<numprocs3; ID++){
	    is3[ID] =  1;
	    ie3[ID] =  0;
	  }
	}

        /* loop for j */

	for (j=0; j<num_HOMOs; j++){

	  j1 = Cluster_HOMO[spin] - j;

	  /* store eigenvalue */
	  HOMOs_Coef[0][spin][j][0][0].r = ko[spin][j1] ;

	  /* store EVec1 */
          if (numprocs0==1){
            for (k=0; k<n; k++){
              m = (j1-1)*n + k;
	      array0[k] = EVec1[spin][m];
	    }
	  }
          else{

	    po = 0;
	    for (ID=0; ID<numprocs3; ID++){
	      if (is3[ID]<=j1 && j1 <=ie3[ID]){
		po = 1;
		ID0 = ID;
		break;
	      }
	    }

            ID1 = Comm_World_StartID1[spin] + ID0;

            if (myid0==ID1){
	      for (k=0; k<n; k++){
		m = (j1-is3[ID0])*n + k;
		array0[k] = EVec1[spin][m];
	      }
	    }

            /* MPI communications */
            MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);  
	  }

	  /* store eigenvector */
	  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	    wanA = WhatSpecies[GA_AN];
	    tnoA = Spe_Total_CNO[wanA];
	    Anum = MP[GA_AN];
	    for (i=0; i<tnoA; i++){
	      HOMOs_Coef[0][spin][j][GA_AN][i].r = array0[Anum+i-1];
	      HOMOs_Coef[0][spin][j][GA_AN][i].i = 0.0;
	    }
	  }
	}
      }

      /* LUMOs */
   
      for (spin=0; spin<=SpinP_switch; spin++){

	/* set is3 and ie3 */

	numprocs3 = NPROCS_WD1[spin];

	if ( numprocs3<=MaxN ){

	  av_num = (double)MaxN/(double)numprocs3;
	  for (ID=0; ID<numprocs3; ID++){
	    is3[ID] = (int)(av_num*(double)ID) + 1; 
	    ie3[ID] = (int)(av_num*(double)(ID+1)); 
	  }

	  is3[0] = 1;
	  ie3[numprocs3-1] = MaxN; 
	}

	else{
	  for (ID=0; ID<MaxN; ID++){
	    is3[ID] = ID + 1; 
	    ie3[ID] = ID + 1;
	  }
	  for (ID=MaxN; ID<numprocs3; ID++){
	    is3[ID] =  1;
	    ie3[ID] = -2;
	  }
	}

        /* loop for j */

	for (j=0; j<num_LUMOs; j++){

	  j1 = Cluster_HOMO[spin] + 1 + j;

	  /* store eigenvalue */
	  LUMOs_Coef[0][spin][j][0][0].r = ko[spin][j1];

	  /* store EVec1 */
          if (numprocs0==1){
            for (k=0; k<n; k++){
              m = (j1-1)*n + k;
	      array0[k] = EVec1[spin][m];
	    }
	  }
          else{

	    po = 0;
	    for (ID=0; ID<numprocs3; ID++){
	      if (is3[ID]<=j1 && j1 <=ie3[ID]){
		po = 1;
		ID0 = ID;
		break;
	      }
	    }

            ID1 = Comm_World_StartID1[spin] + ID0;

            if (myid0==ID1){
	      for (k=0; k<n; k++){
		m = (j1-is3[ID0])*n + k;
		array0[k] = EVec1[spin][m];
	      }
	    }

            /* MPI communications */
            MPI_Bcast(array0, n, MPI_DOUBLE, ID1, mpi_comm_level1);  
	  }

	  /* store eigenvector */
	  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	    wanA = WhatSpecies[GA_AN];
	    tnoA = Spe_Total_CNO[wanA];
	    Anum = MP[GA_AN];
	    for (i=0; i<tnoA; i++){
	      LUMOs_Coef[0][spin][j][GA_AN][i].r = array0[Anum+i-1];
	      LUMOs_Coef[0][spin][j][GA_AN][i].i = 0.0;
	    }
	  }
	}
      }

      /* freeing of array0 */
      free(array0);
      free(is3);
      free(ie3);
    }

    /****************************************************
          density matrix and energy density matrix
                  for up and down spins
    ****************************************************/

    time6 += Calc_DM_Cluster_collinear_ScaLAPACK( 1,myid0,numprocs0,myid1,numprocs1,myworld1,
                                                  size_H1,is2,ie2,MP,n,MPI_CommWD1,Comm_World_StartID1,
                                                  CDM,EDM,ko,CDM1,Work1,EVec1,SP_NZeros,SP_Atoms);

    if (Cnt_switch==1){
    time6 += Calc_DM_Cluster_collinear_ScaLAPACK( 2,myid0,numprocs0,myid1,numprocs1,myworld1,
                                                  size_H1,is2,ie2,MP,n,MPI_CommWD1,Comm_World_StartID1,
                                                  CDM,EDM,ko,CDM1,Work1,EVec1,SP_NZeros,SP_Atoms);
    }

    /****************************************************
                        Bond Energies
    ****************************************************/
  
    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;

    for (spin=0; spin<=SpinP_switch; spin++){
      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
	GA_AN = M2G[MA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (j=0; j<=FNAN[GA_AN]; j++){
	  wanB = WhatSpecies[natn[GA_AN][j]];
	  tnoB = Spe_Total_CNO[wanB];
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){
	      My_Eele1[spin] += CDM[spin][MA_AN][j][k][l]*nh[spin][MA_AN][j][k][l];
	    }
	  }
	}
      }
    }
  
    /* MPI, My_Eele1 */
    for (spin=0; spin<=SpinP_switch; spin++){
      MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch==0) Eele1[1] = Eele1[0];

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

    /****************************************************
                           Output
    ****************************************************/

    if (measure_time) dtime(&stime);

    if (myid0==Host_ID){

      sprintf(file_EV,"%s%s.EV",filepath,filename);

      if ((fp_EV = fopen(file_EV,"w")) != NULL){

	setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"            Eigenvalues (Hartree) for SCF KS-eq.           \n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n\n");

	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
	fprintf(fp_EV,"   Number of States             = %18.14f\n",Num_State);
	if (SpinP_switch==0){
	  fprintf(fp_EV,"   HOMO = %2d\n",Cluster_HOMO[0]);
	}
	else if (SpinP_switch==1){
	  fprintf(fp_EV,"   HOMO for up-spin   = %2d\n",Cluster_HOMO[0]);
	  fprintf(fp_EV,"   HOMO for down-spin = %2d\n",Cluster_HOMO[1]);
	}

	fprintf(fp_EV,"   Eigenvalues\n");
	fprintf(fp_EV,"                Up-spin            Down-spin\n");
	for (i1=1; i1<=MaxN; i1++){
	  if (SpinP_switch==0)
	    fprintf(fp_EV,"      %5d %18.14f %18.14f\n",i1,ko[0][i1],ko[0][i1]);
	  else if (SpinP_switch==1)
	    fprintf(fp_EV,"      %5d %18.14f %18.14f\n",i1,ko[0][i1],ko[1][i1]);
	}
      }

      /* fclose of fp_EV */
      fclose(fp_EV);
    }

    if (2<=level_fileout){

      if (myid0==Host_ID){

        sprintf(file_EV,"%s%s.EV",filepath,filename);

        if ((fp_EV = fopen(file_EV,"a")) != NULL){
  	  setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */
        }
	else{
	  printf("Failure of saving the EV file.\n");
	}
      }

      if (myid0==Host_ID){

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"  Eigenvalues (Hartree) and Eigenvectors for SCF KS-eq.  \n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");

	fprintf(fp_EV,"\n\n");
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
	if (SpinP_switch==0){
	  fprintf(fp_EV,"   HOMO = %2d\n",Cluster_HOMO[0]);
	}
	else if (SpinP_switch==1){
	  fprintf(fp_EV,"   HOMO for up-spin   = %2d\n",Cluster_HOMO[0]);
	  fprintf(fp_EV,"   HOMO for down-spin = %2d\n",Cluster_HOMO[1]);
	}

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"   LCAO coefficients for up (U) and down (D) spins\n\n");
      }

      /* allocation of arrays */
      double *array0;    
      int *is3,*ie3;
      int numprocs3,ID1;

      array0 = (double*)malloc(sizeof(double)*(n+2)*7);
      is3 = (int*)malloc(sizeof(int)*numprocs0);
      ie3 = (int*)malloc(sizeof(int)*numprocs0);

      /* start of loop spin and i */

      num0 = 6;
      num1 = MaxN/num0 + 1*(MaxN%num0!=0);

      for (spin=0; spin<=SpinP_switch; spin++){

	/* set is3 and ie3 */

	numprocs3 = NPROCS_WD1[spin];

	if ( numprocs3<=MaxN ){

	  av_num = (double)MaxN/(double)numprocs3;
	  for (ID=0; ID<numprocs3; ID++){
	    is3[ID] = (int)(av_num*(double)ID) + 1; 
	    ie3[ID] = (int)(av_num*(double)(ID+1)); 
	  }

	  is3[0] = 1;
	  ie3[numprocs3-1] = MaxN; 
	}

	else{
	  for (ID=0; ID<MaxN; ID++){
	    is3[ID] = ID + 1; 
	    ie3[ID] = ID + 1;
	  }
	  for (ID=MaxN; ID<numprocs3; ID++){
	    is3[ID] =  1;
	    ie3[ID] = -2;
	  }
	}

	for (i=1; i<=num1; i++){

	  if (myid0==Host_ID){ 

	    /* header */ 
	    fprintf(fp_EV,"\n");
	    for (i1=-1; i1<=0; i1++){

	      if      (i1==-1) fprintf(fp_EV,"                     ");
	      else if (i1==0)  fprintf(fp_EV,"                      ");

	      for (j=1; j<=num0; j++){
		j1 = num0*(i-1) + j;
		if (j1<=MaxN){ 
		  if (i1==-1){

  		    if (spin==0)      fprintf(fp_EV," %5d (U)",j1);
		    else if (spin==1) fprintf(fp_EV," %5d (D)",j1);
		  }
		  else if (i1==0){
  		    fprintf(fp_EV,"  %8.5f",ko[spin][j1]);
		  }
		}
	      }
	      fprintf(fp_EV,"\n");
	      if (i1==0)  fprintf(fp_EV,"\n");
	    }
	  }

	  /* MPI communication of EVec1 */

          if (numprocs0==1){

	    for (j=1; j<=num0; j++){

	      j1 = num0*(i-1) + j;

  	      if (j1<=MaxN){
		for (k=0; k<n; k++){
		  m = (j1-1)*n + k;
		  array0[(j-1)*n+k] = EVec1[spin][m];
		}
	      }
	    }
	  }
          else{

	    for (j=1; j<=num0; j++){

	      j1 = num0*(i-1) + j;

	      po = 0;
	      for (ID=0; ID<numprocs3; ID++){
		if (is3[ID]<=j1 && j1 <=ie3[ID]){
		  po = 1;
		  ID0 = ID;
		  break;
		}
	      }

	      ID1 = Comm_World_StartID1[spin] + ID0;

  	      if (j1<=MaxN){

		if (myid0==ID1){
		  for (k=0; k<n; k++){
		    m = (j1-is3[ID0])*n + k;
		    array0[(j-1)*n+k] = EVec1[spin][m];
		  }
		}

		/* MPI communications */
		MPI_Bcast(&array0[(j-1)*n], n, MPI_DOUBLE, ID1, mpi_comm_level1);
	      }

	    } /* j */
	  } /* else */

	  /* LCAO coefficients */ 

	  Name_Angular[0][0] = "s          ";
	  Name_Angular[1][0] = "px         ";
	  Name_Angular[1][1] = "py         ";
	  Name_Angular[1][2] = "pz         ";
	  Name_Angular[2][0] = "d3z^2-r^2  ";
	  Name_Angular[2][1] = "dx^2-y^2   ";
	  Name_Angular[2][2] = "dxy        ";
	  Name_Angular[2][3] = "dxz        ";
	  Name_Angular[2][4] = "dyz        ";
	  Name_Angular[3][0] = "f5z^2-3r^2 ";
	  Name_Angular[3][1] = "f5xz^2-xr^2";
	  Name_Angular[3][2] = "f5yz^2-yr^2";
	  Name_Angular[3][3] = "fzx^2-zy^2 ";
	  Name_Angular[3][4] = "fxyz       ";
	  Name_Angular[3][5] = "fx^3-3*xy^2";
	  Name_Angular[3][6] = "f3yx^2-y^3 ";
	  Name_Angular[4][0] = "g1         ";
	  Name_Angular[4][1] = "g2         ";
	  Name_Angular[4][2] = "g3         ";
	  Name_Angular[4][3] = "g4         ";
	  Name_Angular[4][4] = "g5         ";
	  Name_Angular[4][5] = "g6         ";
	  Name_Angular[4][6] = "g7         ";
	  Name_Angular[4][7] = "g8         ";
	  Name_Angular[4][8] = "g9         ";

	  Name_Multiple[0] = "0";
	  Name_Multiple[1] = "1";
	  Name_Multiple[2] = "2";
	  Name_Multiple[3] = "3";
	  Name_Multiple[4] = "4";
	  Name_Multiple[5] = "5";

	  if (myid0==Host_ID){ 

	    i1 = 1; 
	    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	      wan1 = WhatSpecies[Gc_AN];
            
	      for (l=0; l<=Supported_MaxL; l++){
		for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
		  for (m=0; m<(2*l+1); m++){

		    if (l==0 && mul==0 && m==0)
		      fprintf(fp_EV,"%4d %3s %s %s", 
			      Gc_AN,SpeName[wan1],Name_Multiple[mul],Name_Angular[l][m]);
		    else
		      fprintf(fp_EV,"         %s %s", 
			      Name_Multiple[mul],Name_Angular[l][m]);
		    for (j=1; j<=num0; j++){
		      j1 = num0*(i-1) + j;
		      if (0<i1 && j1<=MaxN){
			fprintf(fp_EV,"  %8.5f",array0[(j-1)*n+(i1-1)]);
		      }
		    }
		  
		    fprintf(fp_EV,"\n");

		    i1++;
		  }
		}
	      }
	    }
	  }

	} /* i */
      } /* spin */

      /* freeing of arrays */

      free(array0);
      free(is3);
      free(ie3);

      /* fclose of fp_EV */

      if (myid0==Host_ID){
        fclose(fp_EV);
      }

    } /* end of if (2<=level_fileout) */

    if (measure_time){
      dtime(&etime);
      time7 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  else if ( strcasecmp(mode,"dos")==0 ){
    Save_DOS_Col(n,MaxN,MP,CntOLP,EVec1,ko,NPROCS_WD1,Comm_World_StartID1);
  }

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid0,time1,time2,time3,time4,time5,time6,time7);fflush(stdout); 
  }

  /****************************************************
                          Free
  ****************************************************/

  free(is1);
  free(ie1);
  free(Num_Snd_EV);
  free(Num_Rcv_EV);
  
  free(index_Snd_i);
  free(index_Snd_j);
  free(EVec_Snd);
  free(index_Rcv_i);
  free(index_Rcv_j);
  free(EVec_Rcv);

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}






double Calc_DM_Cluster_collinear_ScaLAPACK(
    int calc_flag,
    int myid0,
    int numprocs0,
    int myid1,
    int numprocs1,
    int myworld1,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    MPI_Comm *MPI_CommWD1,
    int *Comm_World_StartID1,
    double *****CDM,
    double *****EDM,
    double **ko,
    double *DM1,
    double *Work1,
    double **EVec1, 
    int *SP_NZeros,
    int *SP_Atoms )
{
  int spin,i,j,k,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID;
  double max_x=60.0,dum;
  double FermiF,FermiF2,x,x2,diffF;
  double FermiEps = 1.0e-13;
  double stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;

  dtime(&stime);

  /* spin=myworld1 */

  spin = myworld1;

 calc_dm_collinear:

  /* initialize DM1 */

  for (i=0; i<size_H1; i++){
    DM1[i] = 0.0;
  }

  /* calculation of DM1 */ 

  for (k=is2[myid1]; k<=ie2[myid1]; k++){

    x = (ko[spin][k] - ChemP)*Beta;
    if (x<=-max_x) x = -max_x;
    if (max_x<=x)  x = max_x;
    FermiF = FermiFunc(x,spin,k,&po,&x);

    if (cal_partial_charge){
      x2 = (ko[spin][k] - (ChemP+ene_win_partial_charge))*Beta;
      if (x2<=-max_x) x2 = -max_x;
      if (max_x<=x2)  x2 = max_x;
      FermiF2 = FermiFunc(x2,spin,k,&po,&x);
      diffF = fabs(FermiF-FermiF2);
    }

    if (FermiF>FermiEps || (cal_partial_charge && (FermiEps<FermiF || FermiEps<FermiF2) )) {

      p = 0;
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      i1 = (k-is2[myid1])*n + Anum + i - 1; 
	      j1 = (k-is2[myid1])*n + Bnum + j - 1;

              if (calc_flag==1){
                DM1[p] += FermiF*EVec1[spin][i1]*EVec1[spin][j1];;
	      }
              else if (calc_flag==2){
                DM1[p] += FermiF*EVec1[spin][i1]*EVec1[spin][j1]*ko[spin][k];
	      }
              else if (calc_flag==3){
                DM1[p] += diffF*EVec1[spin][i1]*EVec1[spin][j1];
	      }

	      /* increment of p */
	      p++;  

	    }
	  }
	}
      } /* GA_AN */
    }
  }

  /* MPI_Allreduce */

  MPI_Allreduce(DM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
  for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

  /* store DM1 to a proper place */

  p = 0;
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

    MA_AN = F_G2M[GA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];
    ID = G2ID[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];

      if (myid0==ID){
         
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            if      (calc_flag==1) CDM[spin][MA_AN][LB_AN][i][j] = DM1[p];
            else if (calc_flag==2) EDM[spin][MA_AN][LB_AN][i][j] = DM1[p];
            else if (calc_flag==3) PDM[spin][MA_AN][LB_AN][i][j] = DM1[p];

	    /* increment of p */
	    p++;  
	  }
	}
      }
      else{
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    /* increment of p */
	    p++;  
	  }
	}
      }

    } /* LB_AN */
  } /* GA_AN */

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;  
    goto calc_dm_collinear;
  }
 
  else if (SpinP_switch==1 && numprocs0!=1){

    /* MPI communication of DM1 */

    if (Comm_World_StartID1[0]==myid0){
      MPI_Isend(DM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],10,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Isend(DM1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],20,mpi_comm_level1,&request);
    }
    if (Comm_World_StartID1[1]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[0],10,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }
    if (Comm_World_StartID1[0]==myid0){
      MPI_Recv(Work1,size_H1,MPI_DOUBLE,Comm_World_StartID1[1],20,mpi_comm_level1,&stat);
      MPI_Wait(&request,&stat);
    }

    MPI_Bcast(Work1, size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);  
    for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

    /* store DM1 to a proper place */

    if      (myworld1==0) spin = 1;
    else if (myworld1==1) spin = 0;

    p = 0;
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

      MA_AN = F_G2M[GA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];
      ID = G2ID[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	if (myid0==ID){
         
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

              if      (calc_flag==1) CDM[spin][MA_AN][LB_AN][i][j] = DM1[p];
              else if (calc_flag==2) EDM[spin][MA_AN][LB_AN][i][j] = DM1[p];
              else if (calc_flag==3) PDM[spin][MA_AN][LB_AN][i][j] = DM1[p];

	      /* increment of p */
	      p++;  
	    }
	  }
	}
	else{
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
	      /* increment of p */
	      p++;  
	    }
	  }
	}

      } /* LB_AN */
    } /* GA_AN */
  } /* end of else if (SpinP_switch==1) */

  dtime(&etime);
  return (etime-stime);

}


static double Cluster_non_collinear(char *mode,
				    int SCF_iter,
				    int SpinP_switch,
				    double *****nh,
				    double *****ImNL,
				    double ****CntOLP,
				    double *****CDM,
				    double *****EDM,
                                    int *is2, int *ie2,
                                    int *MP,
                                    double *DM1,
                                    double *Work1,
                                    dcomplex *EVec1,
                                    double *ko,
                                    int size_H1, 
				    double Eele0[2], double Eele1[2])
{
  static int firsttime=1;
  int i,j,l,n,n2,n1,i1,i1s,j1,k1,l1;
  int ii1,jj1,jj2,ki,kj;
  int wan,HOMO0,HOMO1;
  int po,num0,num1;
  int mul,m,wan1,Gc_AN,bcast_flag;
  double time0,lumos;
  int ct_AN,k,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,loopN;
  int MA_AN,LB_AN,GB_AN,Bnum,MaxN;
  int *is1,*ie1;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double My_Eele1[2];
  double Num_State,x,FermiF,Dnum,Dnum2;
  double FermiF2,x2,diffF;
  double dum,ChemP_MAX,ChemP_MIN;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10] = ".EV";
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */
  double time1,time2,time3,time4,time5,time6,time7;
  double stime,etime;
  double av_num,tmp;
  int ig,jg;
  int numprocs,myid,ID;
  int ke,ks,nblk_m,nblk_m2;
  int ID0,IDS,IDR,Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;
  int ZERO=0,ONE=1,info;
  double alpha = 1.0; double beta = 0.0;
  double *rHs11,*rHs12,*rHs22,*iHs11,*iHs12,*iHs22,*Cs;
  dcomplex *Hs2,*Cs2;
  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  MPI_Status stat;
  MPI_Request request;

  double *Ss;
  dcomplex *Ss2;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* ***************************************************
             calculation of the array size
  *************************************************** */

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }
  n2 = 2*n;

  /* ***************************************************
                  allocation of arrays
  *************************************************** */

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs);
  Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs);

  /* ***************************************************
        setting for BLACS in the matrix size of n
  *************************************************** */

  np_cols = (int)(sqrt((float)numprocs));
  do{
    if((numprocs%np_cols)==0) break;
    np_cols--;
  } while (np_cols>=2);
  np_rows = numprocs/np_cols;

  nblk_m = NBLK;
  while((nblk_m*np_rows>n || nblk_m*np_cols>n) && (nblk_m > 1)){
    nblk_m /= 2;
  }
  if(nblk_m<1) nblk_m = 1;

  MPI_Allreduce(&nblk_m,&nblk,1,MPI_INT,MPI_MIN,mpi_comm_level1);

  my_prow = myid/np_cols;
  my_pcol = myid%np_cols;

  na_rows = numroc_(&n, &nblk, &my_prow, &ZERO, &np_rows);
  na_cols = numroc_(&n, &nblk, &my_pcol, &ZERO, &np_cols );

  bhandle1 = Csys2blacs_handle(mpi_comm_level1);
  ictxt1 = bhandle1;

  Cblacs_gridinit(&ictxt1, "Row", np_rows, np_cols);
  Ss = (double*)malloc(sizeof(double)*na_rows*na_cols);
  rHs11 = (double*)malloc(sizeof(double)*na_rows*na_cols);
  rHs12 = (double*)malloc(sizeof(double)*na_rows*na_cols);
  rHs22 = (double*)malloc(sizeof(double)*na_rows*na_cols);
  iHs11 = (double*)malloc(sizeof(double)*na_rows*na_cols);
  iHs12 = (double*)malloc(sizeof(double)*na_rows*na_cols);
  iHs22 = (double*)malloc(sizeof(double)*na_rows*na_cols);

  MPI_Allreduce(&na_rows,&na_rows_max,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  MPI_Allreduce(&na_cols,&na_cols_max,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  Cs = (double*)malloc(sizeof(double)*na_rows_max*na_cols_max);

  descinit_(descS, &n,  &n,  &nblk,  &nblk,  &ZERO, &ZERO, &ictxt1, &na_rows,  &info);
  descinit_(descH, &n,  &n,  &nblk,  &nblk,  &ZERO, &ZERO, &ictxt1, &na_rows,  &info);
  descinit_(descC, &n,  &n,  &nblk,  &nblk,  &ZERO, &ZERO, &ictxt1, &na_rows,  &info);

  /* ***************************************************
       setting for BLACS in the matrix size of n2
  *************************************************** */

  np_cols2 = (int)(sqrt((float)numprocs));
  do{
    if((numprocs%np_cols2)==0) break;
    np_cols2--;
  } while (np_cols2>=2);
  np_rows2 = numprocs/np_cols2;

  nblk_m2 = NBLK;
  while((nblk_m2*np_rows2>n2 || nblk_m2*np_cols2>n2) && (nblk_m2 > 1)){
    nblk_m2 /= 2;
  }
  if(nblk_m2<1) nblk_m2 = 1;

  MPI_Allreduce(&nblk_m2,&nblk2,1,MPI_INT,MPI_MIN,mpi_comm_level1);

  my_prow2 = myid/np_cols2;
  my_pcol2 = myid%np_cols2;

  na_rows2 = numroc_(&n2, &nblk2, &my_prow2, &ZERO, &np_rows2 );
  na_cols2 = numroc_(&n2, &nblk2, &my_pcol2, &ZERO, &np_cols2 );

  bhandle1_2 = Csys2blacs_handle(mpi_comm_level1);
  ictxt1_2 = bhandle1_2;

  Cblacs_gridinit(&ictxt1_2, "Row", np_rows2, np_cols2);
  Hs2 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows2*na_cols2);
  Ss2 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows2*na_cols2);

  MPI_Allreduce(&na_rows2,&na_rows_max2,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  MPI_Allreduce(&na_cols2,&na_cols_max2,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  Cs2 = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max2*na_cols_max2);

  descinit_(descH2, &n2,  &n2,  &nblk2,  &nblk2,  &ZERO, &ZERO, &ictxt1_2, &na_rows2,  &info);
  descinit_(descC2, &n2,  &n2,  &nblk2,  &nblk2,  &ZERO, &ZERO, &ictxt1_2, &na_rows2,  &info);
  descinit_(descS2, &n2,  &n2,  &nblk2,  &nblk2,  &ZERO, &ZERO, &ictxt1_2, &na_rows2,  &info);

  /* print memory size */

  if (firsttime){
    PrintMemory("Cluster_DFT_ScaLAPACK: Ss",sizeof(double)*na_rows*na_cols,NULL);
    PrintMemory("Cluster_DFT_ScaLAPACK: rHs11",sizeof(double)*na_rows*na_cols,NULL);
    PrintMemory("Cluster_DFT_ScaLAPACK: rHs12",sizeof(double)*na_rows*na_cols,NULL);
    PrintMemory("Cluster_DFT_ScaLAPACK: rHs22",sizeof(double)*na_rows*na_cols,NULL);
    PrintMemory("Cluster_DFT_ScaLAPACK: iHs11",sizeof(double)*na_rows*na_cols,NULL);
    PrintMemory("Cluster_DFT_ScaLAPACK: iHs12",sizeof(double)*na_rows*na_cols,NULL);
    PrintMemory("Cluster_DFT_ScaLAPACK: iHs22",sizeof(double)*na_rows*na_cols,NULL);
    PrintMemory("Cluster_DFT_ScaLAPACK: Cs",sizeof(double)*na_rows_max*na_cols_max,NULL);
    PrintMemory("Cluster_DFT_ScaLAPACK: Hs2",sizeof(dcomplex)*na_rows2*na_cols2,NULL);
    PrintMemory("Cluster_DFT_ScaLAPACK: Cs2",sizeof(dcomplex)*na_rows_max2*na_cols_max2,NULL);
  }
  firsttime=0;

  /* initialize variables of measuring elapsed time */

  if (measure_time){
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
  }

  /* ***************************************************
                  total core charge
  *************************************************** */

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ += Spe_Core_Charge[wan];
  }

  /* ***************************************************
         find the numbers of partions for MPI
  *************************************************** */

  if ( numprocs<=n ){

    av_num = (double)n/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = n; 
  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }

    for (ID=n; ID<numprocs; ID++){
      is1[ID] = 1;
      ie1[ID] = 0;
    }
  }

  /* find the maximum states in solved eigenvalues */
  
  if (SCF_iter==1){
    MaxN = n2; 
  }   
  else {

    if ( strcasecmp(mode,"scf")==0 ) 
      lumos = (double)n2*0.20;      
    else if ( strcasecmp(mode,"dos")==0 )
      lumos = (double)n2*0.40;

    if (lumos<400.0) lumos = 400.0;
    MaxN = Cluster_HOMO[0] + (int)lumos;
    if (n2<MaxN) MaxN = n2;

    if (cal_partial_charge) MaxN = n2; 
  }

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is2[0] = 1;
    ie2[numprocs-1] = MaxN; 
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is2[ID] = 1;
      ie2[ID] = 0;
    }
  }

  /* making data structure of MPI communicaition for eigenvectors */

  for (ID=0; ID<numprocs; ID++){
    Num_Snd_EV[ID] = 0;
    Num_Rcv_EV[ID] = 0;
  }

  for (i=0; i<na_rows2; i++){

    ig = np_rows2*nblk2*((i)/nblk2) + (i)%nblk2 + ((np_rows2+my_prow2)%np_rows2)*nblk2 + 1;

    po = 0;
    for (ID=0; ID<numprocs; ID++){
      if (is2[ID]<=ig && ig <=ie2[ID]){
        po = 1;
        ID0 = ID;
        break;
      }
    }

    if (po==1) Num_Snd_EV[ID0] += na_cols2;
  }

  for (ID=0; ID<numprocs; ID++){
    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;
    if (ID!=0){
      MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, mpi_comm_level1, &request);
      MPI_Recv(&Num_Rcv_EV[IDR], 1, MPI_INT, IDR, 999, mpi_comm_level1, &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
    }
  }

  Max_Num_Snd_EV = 0;
  Max_Num_Rcv_EV = 0;
  for (ID=0; ID<numprocs; ID++){
    if (Max_Num_Snd_EV<Num_Snd_EV[ID]) Max_Num_Snd_EV = Num_Snd_EV[ID];
    if (Max_Num_Rcv_EV<Num_Rcv_EV[ID]) Max_Num_Rcv_EV = Num_Rcv_EV[ID];
  }  

  Max_Num_Snd_EV++;
  Max_Num_Rcv_EV++;

  index_Snd_i = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  index_Snd_j = (int*)malloc(sizeof(int)*Max_Num_Snd_EV);
  EVec_Snd = (double*)malloc(sizeof(double)*Max_Num_Snd_EV*2);
  index_Rcv_i = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  index_Rcv_j = (int*)malloc(sizeof(int)*Max_Num_Rcv_EV);
  EVec_Rcv = (double*)malloc(sizeof(double)*Max_Num_Rcv_EV*2);

  /* ***************************************************
            diagonalize the overlap matrix     
  *************************************************** */

  /*
  if (SCF_iter==1){
  */

  if (1){

    if (measure_time) dtime(&stime);

    Overlap_Cluster_Ss(CntOLP,Cs,MP,0);

    MPI_Comm_split(mpi_comm_level1,my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(mpi_comm_level1,my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

    /* diagonalize Cs */

    if (scf_eigen_lib_flag==1){
      F77_NAME(solve_evp_real,SOLVE_EVP_REAL)( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, 
                                               &mpi_comm_rows_int, &mpi_comm_cols_int );
    }
    else if (scf_eigen_lib_flag==2){

#ifndef kcomp

      int mpiworld;
      mpiworld = MPI_Comm_c2f(mpi_comm_level1);

      F77_NAME(elpa_solve_evp_real_2stage_double_impl,ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)
	( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, 
          &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld ); 
#endif
    }

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    /* print to the standard output */

    if (2<=level_stdout){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[l]);
      }
    }

    /* minus eigenvalues to 1.0e-10 */

    for (l=1; l<=n; l++){
      if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
      ko[l] = 1.0/sqrt(ko[l]);
    }

    /* calculate S*1/sqrt(ko) */

    for(i=0; i<na_rows; i++){
      for(j=0; j<na_cols; j++){
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	Ss[j*na_rows+i] = Ss[j*na_rows+i]*ko[jg];
      }
    }

    /* make Ss2 */

    Overlap_Cluster_NC_Ss2( Ss, Ss2 );

    if (measure_time){
      dtime(&etime);
      time1 += etime - stime; 
    }
  }

  /* ***************************************************
             transformation of H with Ss

    in case of SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
               && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0

    H[i    ][j    ].r = RH[0];
    H[i    ][j    ].i = 0.0;
    H[i+NUM][j+NUM].r = RH[1];
    H[i+NUM][j+NUM].i = 0.0;
    H[i    ][j+NUM].r = RH[2];
    H[i    ][j+NUM].i = RH[3];

    in case of SO_switch==1 or Hub_U_switch==1 or 1<=Constraint_NCS_switch 
               or Zeeman_NCS_switch==1 or Zeeman_NCO_switch==1 

    H[i    ][j    ].r = RH[0];  
    H[i    ][j    ].i = IH[0];
    H[i+NUM][j+NUM].r = RH[1];
    H[i+NUM][j+NUM].i = IH[1];
    H[i    ][j+NUM].r = RH[2];
    H[i    ][j+NUM].i = RH[3] + IH[2];
  *************************************************** */

  if (measure_time) dtime(&stime);

  /* set rHs and iHs */

  Hamiltonian_Cluster_Hs(nh[0],rHs11,MP,0,0);
  Hamiltonian_Cluster_Hs(nh[1],rHs22,MP,0,0);
  Hamiltonian_Cluster_Hs(nh[2],rHs12,MP,0,0);
  Hamiltonian_Cluster_Hs(nh[3],iHs12,MP,0,0);

  Hamiltonian_Cluster_Hs(ImNL[0],iHs11,MP,0,0);
  Hamiltonian_Cluster_Hs(ImNL[1],iHs22,MP,0,0);
  Hamiltonian_Cluster_Hs(ImNL[2],Cs,MP,0,0);

  for (i=0; i<na_rows*na_cols; i++) iHs12[i] += Cs[i];

  /* S^t x rHs11 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,rHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs11[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,rHs11,&ONE,&ONE,descH);

  /* S^t x rHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,rHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs12[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,rHs12,&ONE,&ONE,descH);

  /* S^t x rHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,rHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) rHs22[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,rHs22,&ONE,&ONE,descH);

  /* S^t x iHs11 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,iHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs11[i] = 0.0;

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,iHs11,&ONE,&ONE,descH);

  /* S^t x iHs12 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,iHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs12[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,iHs12,&ONE,&ONE,descH);

  /* S^t x iHs22 x S */

  for (i=0; i<na_rows*na_cols; i++) Cs[i] = 0.0;

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,iHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  for (i=0; i<na_rows*na_cols; i++) iHs22[i] = 0.0;
  
  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,iHs22,&ONE,&ONE,descH);

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /* ***************************************************
             diagonalize the transformed H
  *************************************************** */

  if (measure_time) dtime(&stime);

  Hamiltonian_Cluster_NC_Hs2( rHs11, rHs22, rHs12, iHs11, iHs22, iHs12, Hs2 );

  MPI_Comm_split(mpi_comm_level1,my_pcol2,my_prow2,&mpi_comm_rows);
  MPI_Comm_split(mpi_comm_level1,my_prow2,my_pcol2,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
  
  if (scf_eigen_lib_flag==1){
    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, 
                                                   &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
  }
  else if (scf_eigen_lib_flag==2){

#ifndef kcomp

    int mpiworld;
    mpiworld = MPI_Comm_c2f(mpi_comm_level1);

    F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
      ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &na_cols2, 
        &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#endif
  }
  

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);
  
  if (2<=level_stdout){
    for (i1=1; i1<=MaxN; i1++){
      printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1,ko[i1]);
    }
  }
  
  if (measure_time){
    dtime(&etime);
    time3 += etime - stime;
  }

  /* ***************************************************
      Transformation to the original eigenvectors.
      JRCAT NOTE 244P  C = U * lambda^{-1/2} * D
  *************************************************** */

  if (measure_time) dtime(&stime);

  for(k=0; k<na_rows2*na_cols2; k++){
    Hs2[k].r = 0.0;
    Hs2[k].i = 0.0;
  }

  Cblacs_barrier(ictxt1_2,"A");
  F77_NAME(pzgemm,PZGEMM)("T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,&ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);

  /* MPI communications of Hs2 */

  for (ID=0; ID<numprocs; ID++){
    
    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    k = 0;
    for(i=0; i<na_rows2; i++){
      ig = np_rows2*nblk2*((i)/nblk2) + (i)%nblk2 + ((np_rows2+my_prow2)%np_rows2)*nblk2 + 1;

      if (is2[IDS]<=ig && ig <=ie2[IDS]){

        for (j=0; j<na_cols2; j++){
          jg = np_cols2*nblk2*((j)/nblk2) + (j)%nblk2 + ((np_cols2+my_pcol2)%np_cols2)*nblk2 + 1;
 
          index_Snd_i[k] = ig;
          index_Snd_j[k] = jg;
          EVec_Snd[2*k  ] = Hs2[j*na_rows2+i].r;
          EVec_Snd[2*k+1] = Hs2[j*na_rows2+i].i;
          k++; 
	}
      }
    }

    if (ID!=0){

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, mpi_comm_level1, &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, mpi_comm_level1, &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, mpi_comm_level1, &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, mpi_comm_level1, &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }

      if (Num_Snd_EV[IDS]!=0){
        MPI_Isend(EVec_Snd, Num_Snd_EV[IDS]*2, MPI_DOUBLE, IDS, 999, mpi_comm_level1, &request);
      }
      if (Num_Rcv_EV[IDR]!=0){
        MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR]*2, MPI_DOUBLE, IDR, 999, mpi_comm_level1, &stat);
      }
      if (Num_Snd_EV[IDS]!=0){
        MPI_Wait(&request,&stat);
      }
    }
    else{
      for(k=0; k<Num_Snd_EV[IDS]; k++){
        index_Rcv_i[k] = index_Snd_i[k];
        index_Rcv_j[k] = index_Snd_j[k];
        EVec_Rcv[2*k  ] = EVec_Snd[2*k  ];
        EVec_Rcv[2*k+1] = EVec_Snd[2*k+1];
      } 
    }

    for(k=0; k<Num_Rcv_EV[IDR]; k++){
      ig = index_Rcv_i[k];
      jg = index_Rcv_j[k];
      m = (ig-is2[myid])*n2 + jg - 1;
      EVec1[m].r = EVec_Rcv[2*k  ];
      EVec1[m].i = EVec_Rcv[2*k+1];
    }
  }

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }

  if ( strcasecmp(mode,"scf")==0 ){

    /****************************************************
                  find chemical potential
    ****************************************************/

    if (measure_time) dtime(&stime);

    /* first, find ChemP at five times large temperatue */

    po = 0;
    loopN = 0;

    ChemP_MAX = 15.0;  
    ChemP_MIN =-15.0;

    do {
      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      Num_State = 0.0;

      for (i1=1; i1<=MaxN; i1++){
	x = (ko[i1] - ChemP)*Beta*0.2;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = FermiFunc_NC(x,i1);
	Num_State = Num_State + FermiF;
	if (0.5<FermiF) Cluster_HOMO[0] = i1;
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-14) po = 1;
      loopN++;
    } 
    while (po==0 && loopN<1000); 

    /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

    po = 0;
    loopN = 0;

    ChemP_MAX = 15.0;  
    ChemP_MIN =-15.0;

    do {

      if (loopN!=0){
	ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      }

      Num_State = 0.0;

      for (i1=1; i1<=MaxN; i1++){
	x = (ko[i1] - ChemP)*Beta;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = FermiFunc_NC(x,i1);
	Num_State = Num_State + FermiF;
	if (0.5<FermiF) Cluster_HOMO[0] = i1;
      }

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-14) po = 1;
      loopN++;
    } 
    while (po==0 && loopN<1000); 

    if (2<=level_stdout){
      printf("  ChemP=%15.12f\n",ChemP);
    }

    if (measure_time){
      dtime(&etime);
      time5 += etime - stime;
    }

    /****************************************************
            Energies by summing up eigenvalues
    ****************************************************/

    Eele0[0] = 0.0;
    Eele0[1] = 0.0;

    for (i1=1; i1<=MaxN; i1++){

      x = (ko[i1] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = FermiFunc_NC(x,i1);
      Eele0[0] = Eele0[0] + ko[i1]*FermiF;
    }

    /****************************************************
      LCAO coefficients are stored for calculating
               values of MOs on grids
    ****************************************************/

    if ( (Cluster_HOMO[0]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[0];
    if ( (Cluster_HOMO[0]+num_LUMOs)>MaxN ) num_LUMOs = MaxN - Cluster_HOMO[0];

    if (myid==Host_ID && 2<=level_stdout){
      printf("  HOMO = %2d\n",Cluster_HOMO[0]);
    }

    if (MO_fileout==1){  

      /* allocation of arrays */
      double *array0;    
      int *is3,*ie3;

      array0 = (double*)malloc(sizeof(double)*(n2+2)*2);
      is3 = (int*)malloc(sizeof(int)*numprocs);
      ie3 = (int*)malloc(sizeof(int)*numprocs);

      /* set is3 and ie3 */

      if ( numprocs<=MaxN ){

	av_num = (double)MaxN/(double)numprocs;
	for (ID=0; ID<numprocs; ID++){
	  is3[ID] = (int)(av_num*(double)ID) + 1; 
	  ie3[ID] = (int)(av_num*(double)(ID+1)); 
	}

	is3[0] = 1;
	ie3[numprocs-1] = MaxN; 
      }
      else{
	for (ID=0; ID<MaxN; ID++){
	  is3[ID] = ID + 1; 
	  ie3[ID] = ID + 1;
	}
	for (ID=MaxN; ID<numprocs; ID++){
	  is3[ID] =  1;
	  ie3[ID] =  0;
	}
      }

      /* HOMOs */

      for (j=0; j<num_HOMOs; j++){

	j1 = Cluster_HOMO[0] - j;

        /* store eigenvalue */
        HOMOs_Coef[0][0][j][0][0].r = ko[j1];
        HOMOs_Coef[0][1][j][0][0].r = ko[j1];

	/* store EVec1 */
	if (numprocs==1){
	  for (k=0; k<n2; k++){
	    m = (j1-1)*n2 + k;
	    array0[2*k  ] = EVec1[m].r;
	    array0[2*k+1] = EVec1[m].i;
	  }
	}
	else{

	  po = 0;
	  for (ID=0; ID<numprocs; ID++){
	    if (is3[ID]<=j1 && j1 <=ie3[ID]){
	      po = 1;
	      ID0 = ID;
	      break;
	    }
	  }

	  if (myid==ID0){
	    for (k=0; k<n2; k++){
	      m = (j1-is3[ID0])*n2 + k;
	      array0[2*k  ] = EVec1[m].r;
	      array0[2*k+1] = EVec1[m].i;
	    }
	  }

	  /* MPI communications */
	  MPI_Bcast(array0, n2*2, MPI_DOUBLE, ID0, mpi_comm_level1);  
	}

	/* store eigenvector */
	for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){

	    HOMOs_Coef[0][0][j][GA_AN][i].r = array0[ (Anum+i-1)*2      ];
	    HOMOs_Coef[0][0][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1    ];
	    HOMOs_Coef[0][1][j][GA_AN][i].r = array0[ (Anum+i-1)*2  +n2 ];
	    HOMOs_Coef[0][1][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1+n2 ];
	  }
	}
      }
      
      /* LUMOs */

      for (j=0; j<num_LUMOs; j++){

	j1 = Cluster_HOMO[0] + 1 + j;

        /* store eigenvalue */
        LUMOs_Coef[0][0][j][0][0].r = ko[j1];
        LUMOs_Coef[0][1][j][0][0].r = ko[j1];

	/* store EVec1 */
	if (numprocs==1){
          for (k=0; k<n2; k++){
	    m = (j1-1)*n2 + k;
	    array0[2*k  ] = EVec1[m].r;
	    array0[2*k+1] = EVec1[m].i;
	  }
        }
	else{

	  po = 0;
	  for (ID=0; ID<numprocs; ID++){
	    if (is3[ID]<=j1 && j1 <=ie3[ID]){
	      po = 1;
	      ID0 = ID;
	      break;
	    }
	  }

	  if (myid==ID0){
	    for (k=0; k<n2; k++){
	      m = (j1-is3[ID0])*n2 + k;
	      array0[2*k  ] = EVec1[m].r;
	      array0[2*k+1] = EVec1[m].i;
	    }
	  }

	  /* MPI communications */
	  MPI_Bcast(array0, n2*2, MPI_DOUBLE, ID0, mpi_comm_level1);  
	}

	/* store eigenvector */
	for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    LUMOs_Coef[0][0][j][GA_AN][i].r = array0[ (Anum+i-1)*2      ];
	    LUMOs_Coef[0][0][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1    ];
	    LUMOs_Coef[0][1][j][GA_AN][i].r = array0[ (Anum+i-1)*2  +n2 ];
	    LUMOs_Coef[0][1][j][GA_AN][i].i = array0[ (Anum+i-1)*2+1+n2 ];
	  }
	}
      }

      free(array0);
      free(is3);
      free(ie3);

    } /* end of if (MO_fileout==1) */

    /****************************************************
        density matrix and energy density matrix

        CDM[0]  Re alpha alpha density matrix
        CDM[1]  Re beta  beta  density matrix
        CDM[2]  Re alpha beta  density matrix
        CDM[3]  Im alpha beta  density matrix
        iDM[0][0]  Im alpha alpha density matrix
        iDM[0][1]  Im beta  beta  density matrix

        EDM[0]  Re alpha alpha energy density matrix
        EDM[1]  Re beta  beta  energy density matrix
        EDM[2]  Re alpha beta  energy density matrix
        EDM[3]  Im alpha beta  energy density matrix
    ****************************************************/

    if (measure_time) dtime(&stime);

    /* DM */

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 1, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 2, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 3, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 4, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 5, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 6, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

    /* EDM */

    if (Cnt_switch==1){

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 7, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 8, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 9, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);

    time6 += Calc_DM_Cluster_non_collinear_ScaLAPACK( 10, myid, numprocs, size_H1, is2, ie2, MP, n, n2,
                                                      CDM, iDM[0], EDM, ko, DM1, Work1, EVec1);
    }

    /****************************************************
                      Bond Energies
    ****************************************************/

    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;
  
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (j=0; j<=FNAN[GA_AN]; j++){
	wanB = WhatSpecies[natn[GA_AN][j]];
	tnoB = Spe_Total_CNO[wanB];

	/* non-spin-orbit coupling and non-LDA+U */  
	if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
	    && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){
	      My_Eele1[0] += 
		+ CDM[0][MA_AN][j][k][l]*nh[0][MA_AN][j][k][l]
		+ CDM[1][MA_AN][j][k][l]*nh[1][MA_AN][j][k][l]
		+ 2.0*CDM[2][MA_AN][j][k][l]*nh[2][MA_AN][j][k][l]
		- 2.0*CDM[3][MA_AN][j][k][l]*nh[3][MA_AN][j][k][l];
	    }
	  }
	}

	/* spin-orbit coupling or LDA+U */
	else {
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){
	      My_Eele1[0] += 
		+ CDM[0][MA_AN][j][k][l]*nh[0][MA_AN][j][k][l]
		- iDM[0][0][MA_AN][j][k][l]*ImNL[0][MA_AN][j][k][l]
		+ CDM[1][MA_AN][j][k][l]*nh[1][MA_AN][j][k][l]
		- iDM[0][1][MA_AN][j][k][l]*ImNL[1][MA_AN][j][k][l]
		+ 2.0*CDM[2][MA_AN][j][k][l]*nh[2][MA_AN][j][k][l]
		- 2.0*CDM[3][MA_AN][j][k][l]*(nh[3][MA_AN][j][k][l]
					      +ImNL[2][MA_AN][j][k][l]);
	    }
	  }
	}

      }
    }

    /* MPI, My_Eele1 */
    MPI_Allreduce(&My_Eele1[0], &Eele1[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_Eele1[1], &Eele1[1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (2<=level_stdout && myid==Host_ID){
      printf("Eele0=%15.12f Eele1=%15.12f\n",Eele0[0],Eele1[0]);
    }

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

    /****************************************************
                           Output
    ****************************************************/

    if (measure_time) dtime(&stime);

    if (myid==Host_ID){

      sprintf(file_EV,"%s%s.EV",filepath,filename);

      if ((fp_EV = fopen(file_EV,"w")) != NULL){

	setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"      Eigenvalues (Hartree) for non-collinear KS-eq.       \n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n\n");

	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
	fprintf(fp_EV,"   Number of States             = %18.14f\n",Num_State);
	fprintf(fp_EV,"   HOMO = %2d\n",Cluster_HOMO[0]);

	fprintf(fp_EV,"   Eigenvalues\n");
	fprintf(fp_EV,"\n");
	for (i1=1; i1<=MaxN; i1++){
	  fprintf(fp_EV,"      %5d %18.14f\n",i1,ko[i1]);
	}

      }
      /* fclose of fp_EV */
      fclose(fp_EV);
    }

    if (2<=level_fileout){

      if (myid==Host_ID){

	sprintf(file_EV,"%s%s.EV",filepath,filename);

	if ((fp_EV = fopen(file_EV,"a")) != NULL){
	  setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */
	}
	else{
	  printf("Failure of saving the EV file.\n");
	}
      }

      if (myid==Host_ID){

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"  Eigenvalues (Hartree) and Eigenvectors for SCF KS-eq.  \n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");

	fprintf(fp_EV,"\n\n");
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
	fprintf(fp_EV,"   HOMO = %2d\n",Cluster_HOMO[0]);

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"   Real (Re) and imaginary (Im) parts of LCAO coefficients\n\n");

      }

      /* allocation of arrays */
      double *array0;    
      int *is3,*ie3;

      array0 = (double*)malloc(sizeof(double)*(n2+2)*4);
      is3 = (int*)malloc(sizeof(int)*numprocs);
      ie3 = (int*)malloc(sizeof(int)*numprocs);

      /* set is3 and ie3 */

      if ( numprocs<=MaxN ){

	av_num = (double)MaxN/(double)numprocs;
	for (ID=0; ID<numprocs; ID++){
	  is3[ID] = (int)(av_num*(double)ID) + 1; 
	  ie3[ID] = (int)(av_num*(double)(ID+1)); 
	}

	is3[0] = 1;
	ie3[numprocs-1] = MaxN; 
      }

      else{
	for (ID=0; ID<MaxN; ID++){
	  is3[ID] = ID + 1; 
	  ie3[ID] = ID + 1;
	}
	for (ID=MaxN; ID<numprocs; ID++){
	  is3[ID] =  1;
	  ie3[ID] = -2;
	}
      }

      num0 = 2;
      num1 = MaxN/num0 + 1*(MaxN%num0!=0);

      for (i=1; i<=num1; i++){

	if (myid==Host_ID){ 

	  /* header */ 

	  fprintf(fp_EV,"\n");

	  for (i1=-2; i1<=0; i1++){

	    fprintf(fp_EV,"                     ");

	    for (j=1; j<=num0; j++){
	      j1 = num0*(i-1) + j;

	      if (j1<=MaxN){ 
		if (i1==-2){
		  fprintf(fp_EV," %4d",j1);
		  fprintf(fp_EV,"                                   ");
		}
		else if (i1==-1){
		  fprintf(fp_EV,"   %8.5f",ko[j1]);
		  fprintf(fp_EV,"                             ");
		}

		else if (i1==0){
		  fprintf(fp_EV,"     Re(U)");
		  fprintf(fp_EV,"     Im(U)");
		  fprintf(fp_EV,"     Re(D)");
		  fprintf(fp_EV,"     Im(D)");
		}

	      }
	    }
	    fprintf(fp_EV,"\n");
	    if (i1==-1)  fprintf(fp_EV,"\n");
	    if (i1==0)   fprintf(fp_EV,"\n");
	  }
	}

	/* MPI communication of EVec1 */

	if (numprocs==1){

	  for (j=1; j<=num0; j++){

	    j1 = num0*(i-1) + j;

	    if (j1<=MaxN){
	      for (k=0; k<n2; k++){
		m = (j1-1)*n2 + k;
		array0[(j-1)*n2*2+2*k  ] = EVec1[m].r;
		array0[(j-1)*n2*2+2*k+1] = EVec1[m].i;
	      }
	    }
	  }
	}
	else{

	  for (j=1; j<=num0; j++){

	    j1 = num0*(i-1) + j;

	    po = 0;
	    for (ID=0; ID<numprocs; ID++){
	      if (is3[ID]<=j1 && j1 <=ie3[ID]){
		po = 1;
		ID0 = ID;
		break;
	      }
	    }

	    if (j1<=MaxN){

	      if (myid==ID0){
		for (k=0; k<n2; k++){
		  m = (j1-is3[ID0])*n2 + k;
		  array0[(j-1)*n2*2+2*k  ] = EVec1[m].r;
		  array0[(j-1)*n2*2+2*k+1] = EVec1[m].i;
		}
	      }

	      /* MPI communications */
	      MPI_Bcast(&array0[(j-1)*n2*2], n2*2, MPI_DOUBLE, ID0, mpi_comm_level1);
	    }

	  } /* j */
	} /* else */

        /* LCAO coefficients */ 

	Name_Angular[0][0] = "s          ";
	Name_Angular[1][0] = "px         ";
	Name_Angular[1][1] = "py         ";
	Name_Angular[1][2] = "pz         ";
	Name_Angular[2][0] = "d3z^2-r^2  ";
	Name_Angular[2][1] = "dx^2-y^2   ";
	Name_Angular[2][2] = "dxy        ";
	Name_Angular[2][3] = "dxz        ";
	Name_Angular[2][4] = "dyz        ";
	Name_Angular[3][0] = "f5z^2-3r^2 ";
	Name_Angular[3][1] = "f5xz^2-xr^2";
	Name_Angular[3][2] = "f5yz^2-yr^2";
	Name_Angular[3][3] = "fzx^2-zy^2 ";
	Name_Angular[3][4] = "fxyz       ";
	Name_Angular[3][5] = "fx^3-3*xy^2";
	Name_Angular[3][6] = "f3yx^2-y^3 ";
	Name_Angular[4][0] = "g1         ";
	Name_Angular[4][1] = "g2         ";
	Name_Angular[4][2] = "g3         ";
	Name_Angular[4][3] = "g4         ";
	Name_Angular[4][4] = "g5         ";
	Name_Angular[4][5] = "g6         ";
	Name_Angular[4][6] = "g7         ";
	Name_Angular[4][7] = "g8         ";
	Name_Angular[4][8] = "g9         ";

	Name_Multiple[0] = "0";
	Name_Multiple[1] = "1";
	Name_Multiple[2] = "2";
	Name_Multiple[3] = "3";
	Name_Multiple[4] = "4";
	Name_Multiple[5] = "5";

	if (myid==Host_ID){ 

	  i1 = 1; 

	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	    wan1 = WhatSpecies[Gc_AN];

	    for (l=0; l<=Supported_MaxL; l++){
	      for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
		for (m=0; m<(2*l+1); m++){

		  if (l==0 && mul==0 && m==0)
		    fprintf(fp_EV,"%4d %3s %s %s", 
			    Gc_AN,SpeName[wan1],Name_Multiple[mul],Name_Angular[l][m]);
		  else
		    fprintf(fp_EV,"         %s %s", 
			    Name_Multiple[mul],Name_Angular[l][m]);

		  for (j=1; j<=num0; j++){

		    j1 = num0*(i-1) + j;

		    if (0<i1 && j1<=MaxN){
		      fprintf(fp_EV,"  %8.5f",array0[ (j-1)*n2*2+2*(i1-1)      ]);
		      fprintf(fp_EV,"  %8.5f",array0[ (j-1)*n2*2+2*(i1-1)+1    ]);
		      fprintf(fp_EV,"  %8.5f",array0[ (j-1)*n2*2+2*(i1-1)  +n2 ]);
		      fprintf(fp_EV,"  %8.5f",array0[ (j-1)*n2*2+2*(i1-1)+1+n2 ]);
		    }
		  }

		  fprintf(fp_EV,"\n");
		  if (i1==-1)  fprintf(fp_EV,"\n");
		  if (i1==0)   fprintf(fp_EV,"\n");

		  i1++;

		}
	      }
	    }
	  }
	}
      }

      /* freeing of arrays */

      free(array0);
      free(is3);
      free(ie3);
	
      /* fclose of fp_EV */

      if (myid==Host_ID){
	fclose(fp_EV);
      }

    } /* end of if (2<=level_fileout) */

    if (measure_time){
      dtime(&etime);
      time7 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  else if ( strcasecmp(mode,"dos")==0 ){
    Save_DOS_NonCol(n,n2,MaxN,MP,CntOLP,EVec1,ko);
  }

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid,time1,time2,time3,time4,time5,time6,time7);fflush(stdout); 
  }

  /****************************************************
                          Free
  ****************************************************/

  free(is1);
  free(ie1);

  free(Num_Snd_EV);
  free(Num_Rcv_EV);

  free(Ss);
  free(rHs11);
  free(rHs12);
  free(rHs22);
  free(iHs11);
  free(iHs12);
  free(iHs22);
  free(Cs);
  free(Hs2);
  free(Ss2);
  free(Cs2);

  free(index_Snd_i);
  free(index_Snd_j);
  free(EVec_Snd);
  free(index_Rcv_i);
  free(index_Rcv_j);
  free(EVec_Rcv);

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}





double Calc_DM_Cluster_non_collinear_ScaLAPACK(
    int calc_flag,
    int myid,
    int numprocs,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    double *****CDM,
    double *****iDM0,
    double *****EDM,
    double *ko,
    double *DM1,
    double *Work1,
    dcomplex *EVec1 )
{
  int i,j,k,po,p,GA_AN,MA_AN,wanA,tnoA,Anum;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j1,ID;
  double max_x=60.0,dum;
  double FermiF,FermiF2,x,x2,diffF;
  double FermiEps = 1.0e-13;
  double stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;

  dtime(&stime);

  /* initialize DM1 */

  for (i=0; i<size_H1; i++){
    DM1[i] = 0.0;
  }

  /* calculation of DM1 */ 

  for (k=is2[myid]; k<=ie2[myid]; k++){

    x = (ko[k] - ChemP)*Beta;
    if (x<=-max_x) x = -max_x;
    if (max_x<=x)  x = max_x;
    FermiF = FermiFunc_NC(x,k);

    if (11<=calc_flag && cal_partial_charge){
      x2 = (ko[k] - (ChemP+ene_win_partial_charge))*Beta;
      if (x2<=-max_x) x2 = -max_x;
      if (max_x<=x2)  x2 = max_x;
      FermiF2 = FermiFunc_NC(x2,k);
      diffF = fabs(FermiF-FermiF2);
    }

    if ( FermiEps<FermiF ) {

      p = 0;
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

              i1 = (k-is2[myid])*n2 + Anum + i - 1; 
              j1 = (k-is2[myid])*n2 + Bnum + j - 1;

              switch (calc_flag){

		/* Re11 */
              case 1:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1].r + EVec1[i1].i*EVec1[j1].i);
		break;

		/* Re22 */
              case 2:
		DM1[p] += FermiF*(EVec1[i1+n].r*EVec1[j1+n].r + EVec1[i1+n].i*EVec1[j1+n].i);
		break;

		/* Re12 */
              case 3:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].r + EVec1[i1].i*EVec1[j1+n].i);
		break;

		/* Im12 */
              case 4:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].i - EVec1[i1].i*EVec1[j1+n].r);
		break;

		/* Im11 */
              case 5:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1].i - EVec1[i1].i*EVec1[j1].r);
		break;

		/* Im22 */
              case 6:
		DM1[p] += FermiF*(EVec1[i1+n].r*EVec1[j1+n].i - EVec1[i1+n].i*EVec1[j1+n].r);
		break;

		/* ReEDM11 */
              case 7:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1].r + EVec1[i1].i*EVec1[j1].i)*ko[k];
		break;

		/* ReEDM22 */
              case 8:
		DM1[p] += FermiF*(EVec1[i1+n].r*EVec1[j1+n].r + EVec1[i1+n].i*EVec1[j1+n].i)*ko[k];
		break;

		/* ReEDM12 */
              case 9:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].r + EVec1[i1].i*EVec1[j1+n].i)*ko[k];
		break;

		/* ImEDM12 */
              case 10:
		DM1[p] += FermiF*(EVec1[i1].r*EVec1[j1+n].i - EVec1[i1].i*EVec1[j1+n].r)*ko[k];
		break;

		/* Partial_DM11 */
              case 11:
		DM1[p] += diffF*(EVec1[i1].r*EVec1[j1].r + EVec1[i1].i*EVec1[j1].i);
		break;

		/* Partial_DM22 */
              case 12:
		DM1[p] += diffF*(EVec1[i1+n].r*EVec1[j1+n].r + EVec1[i1+n].i*EVec1[j1+n].i);
		break;

	      }

	      /* increment of p */
	      p++;  

	    }
	  }
	}
      } /* GA_AN */
    }
  }

  /* MPI_Allreduce */

  MPI_Allreduce(DM1, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  for (i=0; i<size_H1; i++) DM1[i] = Work1[i];

  /* store DM1 to a proper place */

  p = 0;
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

    MA_AN = F_G2M[GA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];
    ID = G2ID[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];

      if (myid==ID){
         
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            switch (calc_flag){

              /* Re11 */ 
              case 1:
		CDM[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Re22 */ 
              case 2:
		CDM[1][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Re12 */ 
              case 3:
		CDM[2][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Im12 */ 
              case 4:
		CDM[3][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Im11 */ 
              case 5:
		iDM0[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Im22 */ 
              case 6:
		iDM0[1][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* ReEDM11 */ 
              case 7:
		EDM[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* ReEDM22 */ 
              case 8:
		EDM[1][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* ReEDM12 */ 
              case 9:
		EDM[2][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* ImEDM12 */ 
              case 10:
		EDM[3][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Partial_DM11 */
              case 11:
		Partial_DM[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;

              /* Partial_DM22 */
              case 12:
		Partial_DM[0][MA_AN][LB_AN][i][j] = DM1[p];
    	        break;
	    }

	    /* increment of p */
	    p++;  
	  }
	}
      }
      else{
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    /* increment of p */
	    p++;  
	  }
	}
      }

    } /* LB_AN */
  } /* GA_AN */

  dtime(&etime);
  return (etime-stime);
}





void Save_DOS_Col( int n, int MaxN, int *MP, double ****OLP0, double **EVec1, 
                   double **ko, int *NPROCS_WD1, int *Comm_World_StartID1 )
{
  int spin,i,j,iemin,iemax,GA_AN,k,l;
  int Anum,Bnum,tnoA,tnoB,wanA,wanB;
  int MA_AN,LB_AN,GB_AN,MaxL,num,m,p,po;
  int numprocs,myid,ID,tag,ID0;
  int i_vec[10];  
  double dum,tmp,av_num;
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  MPI_Status stat;
  MPI_Request request;
  float *array0;    
  float *SD;
  int *is3,*ie3;
  int numprocs3,ID1;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */

  array0 = (float*)malloc(sizeof(float)*(n+2));
  is3 = (int*)malloc(sizeof(int)*numprocs);
  ie3 = (int*)malloc(sizeof(int)*numprocs);
  SD = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);

  /* open file pointers */

  if (myid==Host_ID){

    strcpy(file_eig,".Dos.val");
    fnjoint(filepath,filename,file_eig);
    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_eig);
    }
  
    strcpy(file_ev,".Dos.vec");
    fnjoint(filepath,filename,file_ev);
    if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_ev);
    }
  }

  /* find iemin */

  iemin = n;
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<n; i++) {
      if ( ((ChemP+Dos_Erange[0])<ko[spin][i]) && (i-1)<iemin ) {
        iemin = i - 1;
        break;
      }
    }
  }
  if (iemin<1) iemin = 1;

  /* find iemax */

  iemax = 1;
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<=n; i++) {
      if ( ((ChemP+Dos_Erange[1])<ko[spin][i])) {
        if (iemax<i) iemax = i;
        break;
      }
    }
  }
  if (iemax==1)   iemax = MaxN;
  if (MaxN<iemax) iemax = MaxN;

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){

    /* set is3 and ie3 */

    numprocs3 = NPROCS_WD1[spin];

    if ( numprocs3<=MaxN ){

      av_num = (double)MaxN/(double)numprocs3;
      for (ID=0; ID<numprocs3; ID++){
	is3[ID] = (int)(av_num*(double)ID) + 1; 
	ie3[ID] = (int)(av_num*(double)(ID+1)); 
      }

      is3[0] = 1;
      ie3[numprocs3-1] = MaxN; 
    }

    else{
      for (ID=0; ID<MaxN; ID++){
	is3[ID] = ID + 1; 
	ie3[ID] = ID + 1;
      }
      for (ID=MaxN; ID<numprocs3; ID++){
	is3[ID] =  1;
	ie3[ID] = -2;
      }
    }

    /* loop for p */

    for (p=iemin; p<=iemax; p++){

      /* store EVec1 */
      if (numprocs==1){
	for (k=0; k<n; k++){
	  m = (p-1)*n + k;
	  array0[k] = (float)EVec1[spin][m];
	}
      }
      else{

	po = 0;
	for (ID=0; ID<numprocs3; ID++){
	  if (is3[ID]<=p && p <=ie3[ID]){
	    po = 1;
	    ID0 = ID;
	    break;
	  }
	}

	ID1 = Comm_World_StartID1[spin] + ID0;

	if (myid==ID1){
	  for (k=0; k<n; k++){
	    m = (p-is3[ID0])*n + k;
	    array0[k] = (float)EVec1[spin][m];
	  }
	}

	/* MPI communications */
	MPI_Bcast(array0, n+1, MPI_FLOAT, ID1, mpi_comm_level1);
      }

      /* initialize SD */

      for (i=0; i<(atomnum+1)*List_YOUSO[7]; i++) SD[i] = 0.0;

      i_vec[0]=i_vec[1]=i_vec[2]=0;
      if (myid==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);

      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

	GA_AN = M2G[MA_AN]; 
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (i=0; i<tnoA; i++){
	  for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	    GB_AN = natn[GA_AN][LB_AN];
	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_CNO[wanB];
	    Bnum = MP[GB_AN];

	    for (j=0; j<tnoB; j++){
	      SD[Anum+i] += array0[Anum-1+i]*array0[Bnum-1+j]*(float)OLP0[MA_AN][LB_AN][i][j];
	    }

	  } /* LB_AN */
	} /* i */
      } /* MA_AN */

      /* MPI communication */

      for (i=1; i<=n; i++) array0[i] = SD[i];
      MPI_Reduce(array0, SD, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);

      /* write *.Dos.vec */

      if (myid==Host_ID){
        fwrite(&SD[1],sizeof(float),n,fp_ev);
      }

    } /* p */
  } /* spin */

  /****************************************************
                   save *.Dos.val
  ****************************************************/

  if (myid==Host_ID){

    fprintf(fp_eig,"mode        1\n");
    fprintf(fp_eig,"NonCol      0\n");
    fprintf(fp_eig,"N           %d\n",n);
    fprintf(fp_eig,"Nspin       %d\n",SpinP_switch);
    fprintf(fp_eig,"Erange      %lf %lf\n",Dos_Erange[0],Dos_Erange[1]);
    /*  fprintf(fp_eig,"irange      %d %d\n",iemin,iemax); */
    fprintf(fp_eig,"Kgrid       %d %d %d\n",1,1,1);
    fprintf(fp_eig,"atomnum     %d\n",atomnum);
    fprintf(fp_eig,"<WhatSpecies\n");
    for (i=1;i<=atomnum;i++) {
      fprintf(fp_eig,"%d ",WhatSpecies[i]);
    }
    fprintf(fp_eig,"\nWhatSpecies>\n");
    fprintf(fp_eig,"SpeciesNum     %d\n",SpeciesNum);
    fprintf(fp_eig,"<Spe_Total_CNO\n");
    for (i=0;i<SpeciesNum;i++) {
      fprintf(fp_eig,"%d ",Spe_Total_CNO[i]);
    }
    fprintf(fp_eig,"\nSpe_Total_CNO>\n");
    MaxL=Supported_MaxL; 
    fprintf(fp_eig,"MaxL           %d\n",Supported_MaxL);
    fprintf(fp_eig,"<Spe_Num_CBasis\n");
    for (i=0;i<SpeciesNum;i++) {
      for (l=0;l<=MaxL;l++) {
	fprintf(fp_eig,"%d ",Spe_Num_CBasis[i][l]);
      }
      fprintf(fp_eig,"\n");
    }
    fprintf(fp_eig,"Spe_Num_CBasis>\n");
    fprintf(fp_eig,"ChemP       %lf\n",ChemP);

    fprintf(fp_eig,"irange      %d %d\n",iemin,iemax);
    fprintf(fp_eig,"<Eigenvalues\n");

    for (spin=0; spin<=SpinP_switch; spin++) {
      fprintf(fp_eig,"%d %d %d ",0,0,0);
      for (i=iemin;i<=iemax;i++) {
	fprintf(fp_eig,"%lf ",ko[spin][i]);
	/* printf("%lf ",ko[spin][ie]); */
      }
      fprintf(fp_eig,"\n");
      /* printf("\n"); */
    }
    fprintf(fp_eig,"Eigenvalues>\n");

    printf("write eigenvalues\n");
    printf("write eigenvectors\n");

  } /* if (myid==Host_ID) */

  /* close file pointers */

  if (myid==Host_ID){
    if (fp_eig) fclose(fp_eig);
    if (fp_ev)  fclose(fp_ev);
  }

  /* freeing of array */

  free(SD);
  free(array0);
  free(is3);
  free(ie3);
}




void Save_DOS_NonCol(int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko)
{
  int spin,ie,i,j,iemin,iemax,MA_AN,GA_AN,k,p,l;
  int Anum,Bnum,tnoA,tnoB,wanA,wanB;
  int LB_AN,GB_AN,MaxL,num,num0,num1,m,po,ID0;
  int numprocs,myid,ID,tag;
  int i_vec[10];  
  double d0,d1,d2,d3,dum,av_num;
  double tmp,tmp1,tmp2,tmp3;
  double theta,phi,sit,cot,sip,cop;
  float *array0;    
  float *SD,*Re11,*Re22,*Re12,*Im12;
  int *is3,*ie3;
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */

  array0 = (float*)malloc(sizeof(float)*(n2+2)*2);
  is3 = (int*)malloc(sizeof(int)*numprocs);
  ie3 = (int*)malloc(sizeof(int)*numprocs);
  SD = (float*)malloc(sizeof(float)*2*(atomnum+1)*List_YOUSO[7]);
  Re11 = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);
  Re22 = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);
  Re12 = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);
  Im12 = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);

  /* open file pointers */

  if (myid==Host_ID){

    strcpy(file_eig,".Dos.val");
    fnjoint(filepath,filename,file_eig);
    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_eig);
    }
  
    strcpy(file_ev,".Dos.vec");
    fnjoint(filepath,filename,file_ev);
    if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_ev);
    }
  }

  /* find iemin */

  iemin = n2;

  for (i=1; i<n2; i++) {
    if ( ((ChemP+Dos_Erange[0])<ko[i]) && (i-1)<iemin ) {
      iemin = i - 1;
      break;
    }
  }
  if (iemin<1) iemin = 1;

  /* find iemax */

  iemax = 1;
  for (i=1; i<=2*n; i++) {
    if ( ((ChemP+Dos_Erange[1])<ko[i])) {
      if (iemax<i) iemax = i;
      break;
    }
  }

  if (iemax==1)   iemax = MaxN;
  if (MaxN<iemax) iemax = MaxN;

  /* set is3 and ie3 */

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;
    for (ID=0; ID<numprocs; ID++){
      is3[ID] = (int)(av_num*(double)ID) + 1; 
      ie3[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is3[0] = 1;
    ie3[numprocs-1] = MaxN; 
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is3[ID] = ID + 1; 
      ie3[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is3[ID] =  1;
      ie3[ID] = -2;
    }
  }

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

  for (p=iemin; p<=iemax; p++){

    /* store EVec1 */

    if (numprocs==1){
      for (k=0; k<n2; k++){
	m = (p-1)*n2 + k;
	array0[2*k  ] = (float)EVec1[m].r;
	array0[2*k+1] = (float)EVec1[m].i;
      }
    }

    else{

      po = 0;
      for (ID=0; ID<numprocs; ID++){
	if (is3[ID]<=p && p <=ie3[ID]){
	  po = 1;
	  ID0 = ID;
	  break;
	}
      }

      if (myid==ID0){
	for (k=0; k<n2; k++){
	  m = (p-is3[ID0])*n2 + k;
	  array0[2*k  ] = (float)EVec1[m].r;
	  array0[2*k+1] = (float)EVec1[m].i;
	}
      }

      /* MPI communications */
      MPI_Bcast(array0, n2*2, MPI_FLOAT, ID0, mpi_comm_level1);
    }

    /* initialize Re11, Re22, Re12, and Im12 */

    for (i=0; i<(atomnum+1)*List_YOUSO[7]; i++){ 
      Re11[i] = 0.0;
      Re22[i] = 0.0;
      Re12[i] = 0.0;
      Im12[i] = 0.0;
    }

    /* write a header */

    i_vec[0]=i_vec[1]=i_vec[2]=0;
    if (myid==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);

    /* loop of MA_AN */

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

      GA_AN = M2G[MA_AN]; 
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      for (i=0; i<tnoA; i++){
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  for (j=0; j<tnoB; j++){

	    /* Re11 */

            Re11[Anum+i] += ( array0[(Anum+i-1)*2  ]*array0[(Bnum+j-1)*2  ]
                             +array0[(Anum+i-1)*2+1]*array0[(Bnum+j-1)*2+1] )*(float)OLP0[MA_AN][LB_AN][i][j];

	    /* Re22 */

            Re22[Anum+i] += ( array0[(Anum+i-1)*2  +n2]*array0[(Bnum+j-1)*2+ +n2]
                             +array0[(Anum+i-1)*2+1+n2]*array0[(Bnum+j-1)*2+1+n2] )*(float)OLP0[MA_AN][LB_AN][i][j];

	    /* Re12 */

            Re12[Anum+i] += ( array0[(Anum+i-1)*2  ]*array0[(Bnum+j-1)*2+ +n2]
                             +array0[(Anum+i-1)*2+1]*array0[(Bnum+j-1)*2+1+n2] )*(float)OLP0[MA_AN][LB_AN][i][j];

	    /* Im12
	       conjugate complex of Im12 due to difference in the definition
	       between density matrix and charge density
	    */

            Im12[Anum+i] +=-( array0[(Anum+i-1)*2  ]*array0[(Bnum+j-1)*2+1+n2]
                             -array0[(Anum+i-1)*2+1]*array0[(Bnum+j-1)*2  +n2] )*(float)OLP0[MA_AN][LB_AN][i][j];


	  } /* j */
	} /* LB_AN */
      } /* i */
    } /* MA_AN */

    /* MPI communication */

    MPI_Reduce(Re11, array0, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);
    for (i=0; i<=n; i++) Re11[i] = array0[i];

    MPI_Reduce(Re22, array0, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);
    for (i=0; i<=n; i++) Re22[i] = array0[i];

    MPI_Reduce(Re12, array0, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);
    for (i=0; i<=n; i++) Re12[i] = array0[i];

    MPI_Reduce(Im12, array0, n+1, MPI_FLOAT, MPI_SUM, Host_ID, mpi_comm_level1);
    for (i=0; i<=n; i++) Im12[i] = array0[i];

    /*  transform to up and down states */

    if (myid==Host_ID){

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (i=0; i<tnoA; i++){

	  d0 = Re11[Anum+i];
	  d1 = Re22[Anum+i];
	  d2 = Re12[Anum+i];
	  d3 = Im12[Anum+i];

	  theta = Angle0_Spin[GA_AN];
	  phi   = Angle1_Spin[GA_AN];

	  sit = sin(theta);
	  cot = cos(theta);
	  sip = sin(phi);
	  cop = cos(phi);     

	  tmp1 = 0.5*(d0 + d1);
	  tmp2 = 0.5*cot*(d0 - d1);
	  tmp3 = (d2*cop - d3*sip)*sit;

	  SD[2*(Anum-1)+i     ] = (float)(tmp1 + tmp2 + tmp3); /* up   */
	  SD[2*(Anum-1)+tnoA+i] = (float)(tmp1 - tmp2 - tmp3); /* down */

	}
      }
    
      /* write *.Dos.vec */

      fwrite(&SD[0],sizeof(float),n2,fp_ev);
    }

  } /* p */

  /****************************************************
                   save *.Dos.val
  ****************************************************/

  if (myid==Host_ID){

    fprintf(fp_eig,"mode        1\n");
    fprintf(fp_eig,"NonCol      1\n");
    fprintf(fp_eig,"N           %d\n",n);
    fprintf(fp_eig,"Nspin       %d\n",1); /* switch to 1 */ 
    fprintf(fp_eig,"Erange      %lf %lf\n",Dos_Erange[0],Dos_Erange[1]);
    /*  fprintf(fp_eig,"irange      %d %d\n",iemin,iemax); */
    fprintf(fp_eig,"Kgrid       %d %d %d\n",1,1,1);
    fprintf(fp_eig,"atomnum     %d\n",atomnum);
    fprintf(fp_eig,"<WhatSpecies\n");
    for (i=1;i<=atomnum;i++) {
      fprintf(fp_eig,"%d ",WhatSpecies[i]);
    }
    fprintf(fp_eig,"\nWhatSpecies>\n");
    fprintf(fp_eig,"SpeciesNum     %d\n",SpeciesNum);
    fprintf(fp_eig,"<Spe_Total_CNO\n");
    for (i=0;i<SpeciesNum;i++) {
      fprintf(fp_eig,"%d ",Spe_Total_CNO[i]);
    }
    fprintf(fp_eig,"\nSpe_Total_CNO>\n");
    MaxL=Supported_MaxL; 
    fprintf(fp_eig,"MaxL           %d\n",Supported_MaxL);
    fprintf(fp_eig,"<Spe_Num_CBasis\n");
    for (i=0;i<SpeciesNum;i++) {
      for (l=0;l<=MaxL;l++) {
	fprintf(fp_eig,"%d ",Spe_Num_CBasis[i][l]);
      }
      fprintf(fp_eig,"\n");
    }
    fprintf(fp_eig,"Spe_Num_CBasis>\n");
    fprintf(fp_eig,"ChemP       %lf\n",ChemP);

    fprintf(fp_eig,"<SpinAngle\n");
    for (i=1; i<=atomnum; i++) {
      fprintf(fp_eig,"%lf %lf\n",Angle0_Spin[i],Angle1_Spin[i]);
    }
    fprintf(fp_eig,"SpinAngle>\n");

    fprintf(fp_eig,"irange      %d %d\n",iemin,iemax);
    fprintf(fp_eig,"<Eigenvalues\n");

    for (spin=0; spin<=1; spin++) {
      fprintf(fp_eig,"%d %d %d ",0,0,0);
      for (ie=iemin; ie<=iemax; ie++) {
	fprintf(fp_eig,"%lf ",ko[ie]);
      }
      fprintf(fp_eig,"\n");
      /* printf("\n"); */
    }
    fprintf(fp_eig,"Eigenvalues>\n");

    printf("write eigenvalues\n");
    printf("write eigenvectors\n");
  }

  /* close file pointers */

  if (myid==Host_ID){
    if (fp_eig) fclose(fp_eig);
    if (fp_ev)  fclose(fp_ev);
  }

  /* freeing of array */

  free(array0);
  free(is3);
  free(ie3);
  free(SD);
  free(Re11);
  free(Re22);
  free(Re12);
  free(Im12);
}
