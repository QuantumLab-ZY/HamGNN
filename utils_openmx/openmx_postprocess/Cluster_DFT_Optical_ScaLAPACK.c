/**********************************************************************
  Cluster_DFT_Optical_ScaLAPACK.c:

  Cluster_DFT_Optical_ScaLAPACK.c is a subroutine to calculate optical
  conductivities and dielectric functions based on a collinear DFT

  Log of Cluster_DFT_Optical_ScaLAPACK.c:

     14/Sep./2018  Released by Yung-Ting Lee

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



static double Cluster_collinear_Optical(
				char *mode,
				int SCF_iter,
				int SpinP_switch,
				double **ko,
				double *****nh, double ****CntOLP,
				double *****CDM,
				double *****EDM,
				EXX_t *exx,
				dcomplex ****exx_CDM,
				double *Uexx,
				double Eele0[2], double Eele1[2],
				int myworld1,
				int *NPROCS_ID1,
				int *Comm_World1,
				int *NPROCS_WD1,
				int *Comm_World_StartID1,
				MPI_Comm *MPI_CommWD1,
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



static double Cluster_non_collinear_Optical(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double *****nh,
                   double *****ImNL,
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2]);



double Cluster_DFT_Optical_ScaLAPACK(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double **Cluster_ko,
                   double *****nh,
                   double *****ImNL,
                   double ****OLP0,
                   double *****CDM,
                   double *****EDM,
                   EXX_t *exx, 
                   dcomplex ****exx_CDM,
                   double *Uexx,
                   double Eele0[2], double Eele1[2],
		   int myworld1,
		   int *NPROCS_ID1,
		   int *Comm_World1,
		   int *NPROCS_WD1,
		   int *Comm_World_StartID1,
		   MPI_Comm *MPI_CommWD1,
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
  static double time0;

  /****************************************************
         collinear without spin-orbit coupling
  ****************************************************/

  if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==0 ){

    time0 = Cluster_collinear_Optical( mode,SCF_iter,SpinP_switch,Cluster_ko,
				       nh,  OLP0,CDM,EDM,exx,exx_CDM,Uexx,Eele0,Eele1,
				       myworld1,NPROCS_ID1,Comm_World1,NPROCS_WD1,
				       Comm_World_StartID1,MPI_CommWD1,is2,ie2,
				       Ss,Cs,Hs,CDM1,size_H1,SP_NZeros,SP_Atoms,EVec1,Work1 );
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
    time0 = Cluster_non_collinear_Optical(mode,SCF_iter,SpinP_switch,nh,ImNL,OLP0,CDM,EDM,Eele0,Eele1);
  }

  return time0;
}




static double Cluster_collinear_Optical(
				char *mode,
				int SCF_iter,
				int SpinP_switch,
				double **ko,
				double *****nh, double ****CntOLP,
				double *****CDM,
				double *****EDM,
				EXX_t *exx,
				dcomplex ****exx_CDM,
				double *Uexx,
				double Eele0[2], double Eele1[2],
				int myworld1,
				int *NPROCS_ID1,
				int *Comm_World1,
				int *NPROCS_WD1,
				int *Comm_World_StartID1,
				MPI_Comm *MPI_CommWD1,
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
  int *MP;
  int step,wstep,nstep,istart,iend;
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
  double tmp1,tmp2,eig;
  double Num_State,x,FermiF,Dnum,Dnum2;
  double dum,ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  double EV_cut0;
  double res;
  int numprocs0,myid0;
  int numprocs1,myid1;
  int ID;
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

/* YTL-start */
  double* fd_dist;
  dcomplex** LCAO;
/* YTL-end */

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

   int     MP[List_YOUSO[1]]
   double  H[List_YOUSO[23]][n2][n2]  
  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);

  is1 = (int*)malloc(sizeof(int)*numprocs1);
  ie1 = (int*)malloc(sizeof(int)*numprocs1);

  Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs1);
  Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs1);

  /* YTL-start */

  /* Step (1) : calculate < phi( atom a, orbital alpha ) | nabla operator | phi( atom b, orbital beta ) > */
  Calc_NabraMatrixElements();

  Set_MPIworld_for_optical(myid0,numprocs0); /* MPI for bands */
  Initialize_optical();

  fd_dist = (double*)malloc(sizeof(double)*n);

  LCAO = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i1=0;i1<n2;i1++) LCAO[i1] = (dcomplex*)malloc(sizeof(dcomplex)*n2);

  /* YTL-end */

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
         set the matrix size n
  ****************************************************/

  n = Size_Total_Matrix;

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

    F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&n, &n, Cs, &na_rows, &ko[0][1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    /* minus eigenvalues to 1.0e-10 */

    for (l=1; l<=n; l++){
      if (ko[0][l]<0.0) ko[0][l] = 1.0e-10;
      EV_S[l] = ko[0][l];
    }

    /* print to the standard output */

    if (2<=level_stdout && myid0==Host_ID){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[0][l]);fflush(stdout);
      }
    }

    /* calculate S*1/sqrt(ko) */

    for (l=1; l<=n; l++){
      IEV_S[l] = 1.0/sqrt(ko[0][l]);
    }

    for(i=0;i<na_rows;i++){
      for(j=0;j<na_cols;j++){
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	Ss[j*na_rows+i] = Ss[j*na_rows+i]*IEV_S[jg];
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
  else {

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

  for(i=0; i<na_rows; i++){

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

  for(i=0;i<na_rows_max*na_cols_max;i++){
    Cs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"A");
  F77_NAME(pdgemm,PDGEMM)("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

  /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

  for(i=0;i<na_rows*na_cols;i++){
    Hs[i] = 0.0;
  }

  Cblacs_barrier(ictxt1,"C");
  F77_NAME(pdgemm,PDGEMM)("T","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /*  The output C matrix is distributed by column. */

  if (measure_time) dtime(&stime);

  MPI_Comm_split(MPI_CommWD1[myworld1],my_pcol,my_prow,&mpi_comm_rows);
  MPI_Comm_split(MPI_CommWD1[myworld1],my_prow,my_pcol,&mpi_comm_cols);

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  F77_NAME(solve_evp_real,SOLVE_EVP_REAL)(&n, &MaxN, Hs, &na_rows, &ko[spin][1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);

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

  F77_NAME(pdgemm,PDGEMM)("T","T",&n,&n,&n,&alpha,Cs,&ONE,&ONE,descC,Ss,&ONE,&ONE,descS,&beta,Hs,&ONE,&ONE,descH);
  Cblacs_barrier(ictxt1,"A");

  /* MPI communications of Hs, and store them into EVec1 */

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

      MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      MPI_Wait(&request,&stat);
      MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD1[myworld1], &request);
      MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD1[myworld1], &stat);
      MPI_Wait(&request,&stat);
      MPI_Isend(EVec_Snd, Num_Snd_EV[IDS], MPI_DOUBLE, IDS, 999, MPI_CommWD1[myworld1], &request);
      MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR], MPI_DOUBLE, IDR, 999, MPI_CommWD1[myworld1], &stat);
      MPI_Wait(&request,&stat);
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
  
  if (measure_time) dtime(&stime);
  
  /*********************************************** 
    MPI: ko
  ***********************************************/
  
  for (sp=0; sp<=SpinP_switch; sp++){
    MPI_Bcast(&ko[sp][1],MaxN,MPI_DOUBLE,Comm_World_StartID1[sp],mpi_comm_level1);
  }
 
  if ( strcasecmp(mode,"scf")==0 ){

    if (2<=level_stdout){
      for (i1=1; i1<=MaxN; i1++){
	if (SpinP_switch==0){
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,ko[0][i1],ko[0][i1]);
	}
	else{
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,ko[0][i1],ko[1][i1]);
	}
      }
    }

    /****************************************************
              searching of chemical potential
    ****************************************************/

    po = 0;
    loopN = 0;

    ChemP_MAX = 30.0;  
    ChemP_MIN =-30.0;
  
    do {

      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
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

    /* YTL-start */

    /*************************************************
        determination of CDDF_max_unoccupied_state
    *************************************************/

    /* determine the maximum unoccupied state */

    double range_Ha=(CDDF_max_eV-CDDF_min_eV)/eV2Hartree; /* in Hartree */
    double p1 = CDDF_AddMaxE/eV2Hartree,k2,k3;
    double FDFi,FDFl;
    double x_cut=30.0;

    j = 0;
    for (spin=0; spin<=SpinP_switch; spin++){
      for (i=0;i<MaxN;i++){ /* occupied state */

        eig = ko[spin][i+1];
        x = (eig - ChemP)*Beta;
	if      (x<-x_cut)  FDFi = 1.0;
	else if (x>x_cut)   FDFi = 0.0;
	else                FDFi = 1.0/(1.0 + exp(x));

	for (l=i+1;l<MaxN;l++){ /* unoccupied state */

	  eig = ko[spin][l+1];
	  x = (eig - ChemP)*Beta;
	  if      (x<-x_cut)  FDFl = 1.0;
	  else if (x>x_cut)   FDFl = 0.0;
	  else                FDFl = 1.0/(1.0 + exp(x));

          k2 = FDFi - FDFl; /* dFE */
          k3 = fabs( ko[spin][i+1] - ko[spin][l+1] );       /* dE */
          if ( k2!=0 && k3 <= range_Ha + p1 && l > j) j=l;  /* update the highest state */

	}
      }
    }

    if (j > CDDF_max_unoccupied_state){
      CDDF_max_unoccupied_state = j; /* set the highest state */
    }else{ j = CDDF_max_unoccupied_state; }

    MPI_Allreduce(&j, &CDDF_max_unoccupied_state, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

    /* YTL-end */

    /*
      if (spin==1){
	printf("ABC3 myid0=%2d\n",myid0);
	MPI_Finalize(); 
	exit(0);
      }
    */

    /****************************************************
            calculation of Fermi function
    ****************************************************/

    for (spin=0; spin<=SpinP_switch; spin++){
      for (i1=1; i1<=MaxN; i1++){

	x = (ko[spin][i1] - ChemP)*Beta;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = 1.0/(1.0 + exp(x));

        /* YTL-start */
        /* get Fermi-Dirac distribution */
        if (SpinP_switch==0){
          fd_dist[i1-1] = 2.0*FermiF;
        }else if (SpinP_switch==1){
          fd_dist[i1-1] = FermiF;
        }
        /* YTL-end */
      }

      /* YTL-start */

      double *array0;    
      int *is3,*ie3;
      int numprocs3,ID1,max_width;

      numprocs3 = NPROCS_WD1[spin];

      is3 = (int*)malloc(sizeof(int)*numprocs3);
      ie3 = (int*)malloc(sizeof(int)*numprocs3);

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

      max_width = -100;
      for (ID=0; ID<numprocs3; ID++){
        if (max_width<(ie3[ID]-is3[ID]+1)) max_width = ie3[ID] - is3[ID] + 1;
      }

      array0 = (double*)malloc(sizeof(double)*n*max_width);

      for (ID=0; ID<numprocs3; ID++){

	ID0 = Comm_World_StartID1[spin] + ID; 

        if (myid0==ID0){
          for (i1=0; i1<(ie3[ID]-is3[ID]+1)*n; i1++){
            array0[i1] = EVec1[spin][i1]; 
	  }
	}

        if (0<(ie3[ID]-is3[ID]+1)){

	  /*
          printf("VVV1 n=%2d ID0=%2d  %2d\n",n,ID0,(ie3[ID]-is3[ID]+1));
	  */

	  MPI_Bcast(array0,(ie3[ID]-is3[ID]+1)*n,MPI_DOUBLE,ID0,mpi_comm_level1);

	  k = 0;
	  for (i1=is3[ID]; i1<=ie3[ID]; i1++){
	    for (j1=1; j1<=n; j1++){
	      LCAO[j1][i1] = Complex(array0[k],0.0); 
	      k++;
	    }
	  }
	}
      }

      /*
      printf("ABC6 spin=%2d myid0=%2d numprocs3=%2d max_width=%2d MaxN=%2d Comm_World_StartID1=%2d\n",
      spin,myid0,numprocs3,max_width,MaxN,Comm_World_StartID1[spin]);
      if (spin==1){
	MPI_Finalize(); 
	exit(0);
      }
      */


      Calc_band_optical_col_1(0.0,0.0,0.0,spin,n,ko[spin],LCAO,fd_dist,ChemP);

      free(is3);
      free(ie3);
      free(array0);

      /* YTL-end */

    } /* spin */

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid0,time1,time2,time3,time4,time5,time6,time7);fflush(stdout); 
  }

  /* YTL-start */

  /****************************************************
    collect all the contributions conductivities and 
    dielectric functions
  ****************************************************/

  Calc_optical_col_2(n,1.0);

  /* YTL-end */

  /****************************************************
                          Free
  ****************************************************/

  free(MP);

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

  free(fd_dist);

  for (i1=0;i1<n2;i1++) free(LCAO[i1]);
  free(LCAO);

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}








static double Cluster_non_collinear_Optical(
				    char *mode,
				    int SCF_iter,
				    int SpinP_switch,
				    double *****nh,
				    double *****ImNL,
				    double ****CntOLP,
				    double *****CDM,
				    double *****EDM,
				    double Eele0[2], double Eele1[2])
{
  static int firsttime=1;
  int i,j,l,n,n2,n1,i1,i1s,j1,k1,l1;
  int ii1,jj1,jj2,ki,kj;
  int wan,HOMO0,HOMO1;
  int *MP;
  int spin,po,num0,num1;
  int mul,m,wan1,Gc_AN,bcast_flag;
  double time0,lumos,av_num;
  int ct_AN,k,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,loopN;
  int MA_AN,LB_AN,GB_AN,Bnum,MaxN;
  int *is1,*ie1,*is12,*ie12,*is2,*ie2;
  double *ko,eig;
  dcomplex **H,**C;
  dcomplex Ctmp1,Ctmp2;
  double EV_cut0;
  double sum_i,sum_r,tmp1,tmp2;
  double sum_r0,sum_i0,sum_r1,sum_i1;
  double sum_r00,sum_i00;
  double sum_r01,sum_i01;
  double sum_r10,sum_i10;
  double sum_r11,sum_i11;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double Num_State,x,FermiF,Dnum,Dnum2;
  double dum,ChemP_MAX,ChemP_MIN;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  int numprocs,myid,ID;
  double OLP_eigen_cut=Threshold_OLP_Eigen;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10] = ".EV";
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */
  double time1,time2,time3,time4,time5,time6,time7;
  double stime,etime;

  /* YTL-start */
  double* fd_dist;
  dcomplex** LCAO;
  /* YTL-end */

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* allocation of arrays */

  stat_send = malloc(sizeof(MPI_Status)*numprocs);
  request_send = malloc(sizeof(MPI_Request)*numprocs);
  request_recv = malloc(sizeof(MPI_Request)*numprocs);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }
  n2 = 2*n + 2;

  /****************************************************
   Allocation

   int     MP[List_YOUSO[1]]
   double  ko[n2]
   dcomplex H[n2][n2]  
   dcomplex C[n2][n2]
  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);
  ko = (double*)malloc(sizeof(double)*n2);

  H = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i=0; i<n2; i++){
    H[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  C = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i=0; i<n2; i++){
    C[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  is12 = (int*)malloc(sizeof(int)*numprocs);
  ie12 = (int*)malloc(sizeof(int)*numprocs);

  is2 = (int*)malloc(sizeof(int)*numprocs);
  ie2 = (int*)malloc(sizeof(int)*numprocs);
/*
  if (firsttime){
    PrintMemory("Cluster_DFT: H",sizeof(dcomplex)*n2*n2,NULL);
    PrintMemory("Cluster_DFT: C",sizeof(dcomplex)*n2*n2,NULL);
  }
*/
  /* YTL-start */
  /* Step (1) : calculate < phi( atom a, orbital alpha ) | nabla operator | phi( atom b, orbital beta ) > */
  Calc_NabraMatrixElements();

  Set_MPIworld_for_optical(myid,numprocs); /* MPI for bands */
  Initialize_optical();

  fd_dist = (double*)malloc(sizeof(double)*2*n);

  LCAO = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i1=0;i1<n2;i1++) LCAO[i1] = (dcomplex*)malloc(sizeof(dcomplex)*n2);

  /* YTL-end */

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

  /****************************************************
                  total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ = TZ + Spe_Core_Charge[wan];
  }

  /****************************************************
       1. diagonalize the overlap matrix     
       2. search negative eigenvalues
  ****************************************************/

  n = Size_Total_Matrix;

  /****************************************************
         find the numbers of partions for MPI
  ****************************************************/

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
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  for (ID=0; ID<numprocs; ID++){
    is12[ID] = 2*is1[ID] - 1;
    ie12[ID] = 2*ie1[ID];
  }

  /****************************************************
       1. diagonalize the overlap matrix     
       2. search negative eigenvalues
  ****************************************************/

  if (SCF_iter<=1){

    if (measure_time) dtime(&stime);

    Overlap_Cluster(CntOLP,S,MP);

    /* all the processors have the full output S matrix */

    bcast_flag = 1;

    if (numprocs<n && 1<numprocs)
      Eigen_PReHH(mpi_comm_level1,S,ko,n,n,bcast_flag);
    else 
      Eigen_lapack(S,ko,n,n);

    /* minus eigenvalues to 1.0e-14 */
    for (l=1; l<=n; l++){
      if (ko[l]<0.0) ko[l] = 1.0e-14;
      EV_S[l] = ko[l];
    }

    /* print to the standard output */
/*
    if (2<=level_stdout){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[l]);
      }
    }
*/
    /* calculate S*1/sqrt(ko) */

    for (l=1; l<=n; l++){
      IEV_S[l] = 1.0/sqrt(ko[l]);
    }

    SP_PEV = 1;

    for (i1=1; i1<=(n-3); i1+=4){

#pragma omp parallel shared(n,S,IEV_S) private(OMPID,Nthrds,Nprocs,j1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for ( j1=1+OMPID; j1<=n; j1+=Nthrds ){

	  S[i1+0][j1] = S[i1+0][j1]*IEV_S[j1];
	  S[i1+1][j1] = S[i1+1][j1]*IEV_S[j1];
	  S[i1+2][j1] = S[i1+2][j1]*IEV_S[j1];
	  S[i1+3][j1] = S[i1+3][j1]*IEV_S[j1];
	}

      } /* #pragma omp parallel */
    }

    i1s = n - n%4 + 1;
    for (i1=i1s; i1<=n; i1++){
      for (j1=1; j1<=n; j1++){
	S[i1][j1] = S[i1][j1]*IEV_S[j1];
      }
    }

    if (measure_time){
      dtime(&etime);
      time1 += etime - stime; 
    }
  }

  /****************************************************
    Calculations of eigenvalues

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/
   
  if (measure_time) dtime(&stime);
 
  Hamiltonian_Cluster_NC(nh, ImNL, H, MP);

  /* initialize ko */
  for (i1=1; i1<=2*n; i1++){
    ko[i1] = 10000.0;
  }

  /* transpose S */
  for (i1=1; i1<=n; i1++){
    for (j1=i1+1; j1<=n; j1++){
      tmp1 = S[i1][j1];
      tmp2 = S[j1][i1];
      S[i1][j1] = tmp2;
      S[j1][i1] = tmp1;
    }
  }

  /* C is distributed by row in each processor */
  /*            H * U * lambda^{-1/2}          */

  for (j1=is1[myid]; j1<=ie1[myid]; j1++){

#pragma omp parallel shared(j1,C,S,H,n) private(OMPID,Nthrds,Nprocs,i1,sum_r0,sum_i0,sum_r1,sum_i1,l)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (i1=1+OMPID; i1<=2*n; i1+=Nthrds){

	sum_r0 = 0.0;
	sum_i0 = 0.0;

	sum_r1 = 0.0;
	sum_i1 = 0.0;

	for (l=1; l<=n; l++){
	  sum_r0 += H[i1][l  ].r*S[j1][l];
	  sum_i0 += H[i1][l  ].i*S[j1][l];
	  sum_r1 += H[i1][n+l].r*S[j1][l];
	  sum_i1 += H[i1][n+l].i*S[j1][l];
	}

	C[2*j1-1][i1].r = sum_r0;
	C[2*j1-1][i1].i = sum_i0;

	C[2*j1  ][i1].r = sum_r1;
	C[2*j1  ][i1].i = sum_i1;
      }

    } /* #pragma omp parallel */
  }

  /* H is distributed by row in each processor */
  /* lambda^{-1/2} * U^+ H * U * lambda^{-1/2} */

  for (j1=is1[myid]; j1<=ie1[myid]; j1++){

#pragma omp parallel shared(j1,C,S,H,n) private(OMPID,Nthrds,Nprocs,i1,sum_r00,sum_i00,sum_r01,sum_i01,sum_r10,sum_i10,sum_r11,sum_i11,l,jj1,jj2)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (i1=1+OMPID; i1<=n; i1+=Nthrds){

	sum_r00 = 0.0;
	sum_i00 = 0.0;

	sum_r01 = 0.0;
	sum_i01 = 0.0;

	sum_r10 = 0.0;
	sum_i10 = 0.0;

	sum_r11 = 0.0;
	sum_i11 = 0.0;

	jj1 = 2*j1 - 1;
	jj2 = 2*j1;

	for (l=1; l<=n; l++){

	  sum_r00 += S[i1][l]*C[jj1][l  ].r;
	  sum_i00 += S[i1][l]*C[jj1][l  ].i;

	  sum_r01 += S[i1][l]*C[jj1][l+n].r;
	  sum_i01 += S[i1][l]*C[jj1][l+n].i;

	  sum_r10 += S[i1][l]*C[jj2][l  ].r;
	  sum_i10 += S[i1][l]*C[jj2][l  ].i;

	  sum_r11 += S[i1][l]*C[jj2][l+n].r;
	  sum_i11 += S[i1][l]*C[jj2][l+n].i;
	}

	H[jj1][2*i1-1].r = sum_r00;
	H[jj1][2*i1-1].i = sum_i00;

	H[jj1][2*i1  ].r = sum_r01;
	H[jj1][2*i1  ].i = sum_i01;

	H[jj2][2*i1-1].r = sum_r10;
	H[jj2][2*i1-1].i = sum_i10;

	H[jj2][2*i1  ].r = sum_r11;
	H[jj2][2*i1  ].i = sum_i11;
      }

    } /* #pragma omp parallel */
  }

  /* broadcast H */

  BroadCast_ComplexMatrix(mpi_comm_level1,H,2*n,is12,ie12,myid,numprocs,
                          stat_send,request_send,request_recv);

  /* H to C (transposition) */

  for (i1=1; i1<=2*n; i1++){
    for (j1=1; j1<=2*n; j1++){
      C[j1][i1].r = H[i1][j1].r;
      C[j1][i1].i = H[i1][j1].i;
    }
  }

  /* penalty for ill-conditioning states */

  EV_cut0 = Threshold_OLP_Eigen;

  for (i1=1; i1<=n; i1++){

    if (EV_S[i1]<EV_cut0){
      C[2*i1-1][2*i1-1].r += pow((EV_S[i1]/EV_cut0),-2.0) - 1.0;
      C[2*i1  ][2*i1  ].r += pow((EV_S[i1]/EV_cut0),-2.0) - 1.0;
    }

    /* cutoff the interaction between the ill-conditioned state */

    if (1.0e+3<C[2*i1-1][2*i1-1].r){
      for (j1=1; j1<=2*n; j1++){
	C[2*i1-1][j1    ] = Complex(0.0,0.0);
	C[j1    ][2*i1-1] = Complex(0.0,0.0);
	C[2*i1  ][j1    ] = Complex(0.0,0.0);
	C[j1    ][2*i1  ] = Complex(0.0,0.0);
      }
      C[2*i1-1][2*i1-1] = Complex(1.0e+4,0.0);
      C[2*i1  ][2*i1  ] = Complex(1.0e+4,0.0);
    }
  }

  /* find the maximum states in solved eigenvalues */
  
  n1 = 2*n;

  if (SCF_iter==1){
    MaxN = n1; 
  }   
  else {

    if ( strcasecmp(mode,"scf")==0 ) 
      lumos = (double)n1*0.20;      
    else if ( strcasecmp(mode,"dos")==0 )
      lumos = (double)n1*0.40;

    if (lumos<400.0) lumos = 400.0;
    MaxN = Cluster_HOMO[0] + (int)lumos;
    if (n1<MaxN) MaxN = n1;

    if (cal_partial_charge) MaxN = n1; 
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
      is2[ID] =  1;
      ie2[ID] = -2;
    }
  }

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /*  The output C matrix is distributed by column. */
  /*  solve eigenvalue problem                      */

  if (measure_time) dtime(&stime);

  bcast_flag = 0;

  if (numprocs<n1 && 1<numprocs)
    Eigen_PHH(mpi_comm_level1, C, ko, n1, MaxN, bcast_flag);
  else 
    EigenBand_lapack(C, ko, n1, MaxN, 1);
/*
  if (2<=level_stdout){
    for (i1=1; i1<=MaxN; i1++){
      printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1,ko[i1]);
    }
  }
*/
  /*  The H matrix is distributed by row */

  for (i1=1; i1<=2*n; i1++){
    for (j1=is2[myid]; j1<=ie2[myid]; j1++){
      H[j1][i1] = C[i1][j1];
    }
  }

  if (measure_time){
    dtime(&etime);
    time3 += etime - stime;
  }

  /****************************************************
      Transformation to the original eigenvectors.
      JRCAT NOTE 244P  C = U * lambda^{-1/2} * D
  ****************************************************/

  if (measure_time) dtime(&stime);

  /* transpose S */

  for (i1=1; i1<=n; i1++){
    for (j1=i1+1; j1<=n; j1++){
      tmp1 = S[i1][j1];
      tmp2 = S[j1][i1];
      S[i1][j1] = tmp2;
      S[j1][i1] = tmp1;
    }
  }

  for (i1=1; i1<=2*n; i1++){
    for (j1=1; j1<=2*n; j1++){
      C[i1][j1].r = 0.0;
      C[i1][j1].i = 0.0;
    }
  }

  /* C is distributed by row in each processor */

  for (j1=is2[myid]; j1<=ie2[myid]; j1++){

#pragma omp parallel shared(j1,C,S,H,n) private(OMPID,Nthrds,Nprocs,i1,sum_r0,sum_i0,sum_r1,sum_i1,l,l1)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (i1=1+OMPID; i1<=n; i1+=Nthrds){

	sum_r0 = 0.0;
	sum_i0 = 0.0;

	sum_r1 = 0.0;
	sum_i1 = 0.0;

	l1 = 1;
	for (l=1; l<=n; l++){
	  sum_r0 += S[i1][l]*H[j1][l1].r;
	  sum_i0 += S[i1][l]*H[j1][l1].i;  l1++;

	  sum_r1 += S[i1][l]*H[j1][l1].r;
	  sum_i1 += S[i1][l]*H[j1][l1].i;  l1++;
	}

	C[j1][i1  ].r = sum_r0;
	C[j1][i1  ].i = sum_i0;

	C[j1][i1+n].r = sum_r1;
	C[j1][i1+n].i = sum_i1;

      }

    } /* #pragma omp parallel */
  }

  /****************************************************
     broadcast C
     C is distributed by row in each processor
  ****************************************************/

  BroadCast_ComplexMatrix(mpi_comm_level1,C,2*n,is2,ie2,myid,numprocs, 
                          stat_send,request_send,request_recv);

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }

  if ( strcasecmp(mode,"scf")==0 ){

    /****************************************************
                  find chemical potential
    ****************************************************/

    if (measure_time) dtime(&stime);

    po = 0;
    loopN = 0;

    ChemP_MAX = 5.0;  
    ChemP_MIN =-5.0;

    do {
      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
      Num_State = 0.0;

      for (i1=1; i1<=MaxN; i1++){
	x = (ko[i1] - ChemP)*Beta;
	if (x<=-max_x) x = -max_x;
	if (max_x<=x)  x = max_x;
	FermiF = 1.0/(1.0 + exp(x));
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

    if (measure_time){
      dtime(&etime);
      time5 += etime - stime;
    }

    /* YTL-start */

    /*************************************************
       determination of CDDF_max_unoccupied_state
    *************************************************/

    /* determine the maximum unoccupied state */

    double range_Ha=(CDDF_max_eV-CDDF_min_eV)/eV2Hartree; /* in Hartree */
    double p1 = CDDF_AddMaxE/eV2Hartree,k2,k3;
    double FDFi, FDFl;
    double x_cut=30.0;

    j=0;
      for (i=0;i<MaxN;i++){ /* occupied state */

        eig = ko[i+1];
        x = (eig - ChemP)*Beta;
  if      (x<-x_cut)  FDFi = 1.0;
  else if (x>x_cut)   FDFi = 0.0;
  else                FDFi = 1.0/(1.0 + exp(x));

  for (l=i+1;l<MaxN;l++){ /* unoccupied state */

    eig = ko[l+1];
    x = (eig - ChemP)*Beta;
    if      (x<-x_cut)  FDFl = 1.0;
    else if (x>x_cut)   FDFl = 0.0;
    else                FDFl = 1.0/(1.0 + exp(x));

          k2 = FDFi - FDFl; /* dFE */
          k3 = fabs( ko[i+1] - ko[l+1] ) ; /* dE */
          if ( k2!=0 && k3 <= range_Ha + p1 && l > j) j=l; /* update the highest state */

  }
      }

    if (j > CDDF_max_unoccupied_state){
      CDDF_max_unoccupied_state = j; /* set the highest state */
    }else{ j = CDDF_max_unoccupied_state; }

    MPI_Allreduce(&j, &CDDF_max_unoccupied_state, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

    /* YTL-end */

    /****************************************************
            Calculation of Fermi function
    ****************************************************/

    for (i1=1; i1<=MaxN; i1++){

      x = (ko[i1] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = 1.0/(1.0 + exp(x));

      /* YTL-start */
      fd_dist[i1-1] = FermiF;
      /* YTL-end */
    }

    /* YTL-start */
    /* spin-polarization (on) -> spin = 0, (off) -> spin = 0,1 */
    for (i1=0;i1<2*n;i1++)
      for (j1=0;j1<2*n;j1++)
        LCAO[i1][j1] = Complex(C[i1+1][j1+1].r,C[i1+1][j1+1].i);

    Calc_band_optical_noncol_1(0.0,0.0,0.0,2*n,ko,LCAO,fd_dist,ChemP);
    /* YTL-end */

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid,time1,time2,time3,time4,time5,time6,time7);fflush(stdout); 
  }

  /* YTL-start */

  /****************************************************
    collect all the contributions conductivities and 
    dielectric functions
  ****************************************************/

  Calc_optical_noncol_2(2*n,1.0);

  /* YTL-end */

  /****************************************************
                          Free
  ****************************************************/

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(MP);
  free(ko);

  for (i=0; i<n2; i++){
    free(H[i]);
  }
  free(H);

  for (i=0; i<n2; i++){
    free(C[i]);
  }
  free(C);

  free(is1);
  free(ie1);

  free(is12);
  free(ie12);

  free(is2);
  free(ie2);

  free(fd_dist);

  for (i1=0;i1<n2;i1++) free(LCAO[i1]);
  free(LCAO);

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}

