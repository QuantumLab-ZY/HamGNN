/**********************************************************************
  Band_DFT_Col_Optical_ScaLAPACK.c:

  Band_DFT_Col_Optical_ScaLAPACK.c is a subroutine to calculate optical
  conductivities and dielectric functions based on a collinear DFT

  Log of Band_DFT_Col_Optical_ScaLAPACK.c:

     14/Sep./2018  Released by Yung-Ting Lee

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"
#include <omp.h>


#define  measure_time  0



double Band_DFT_Col_Optical_ScaLAPACK(
                    int SCF_iter,
                    int knum_i, int knum_j, int knum_k,
		    int SpinP_switch,
		    double *****nh,
		    double *****ImNL,
		    double ****CntOLP,
		    double *****CDM,
		    double *****EDM,
		    double Eele0[2], double Eele1[2], 
		    int *MP,
		    int *order_GA,
		    double *ko,
		    double *koS,
		    double *H1,
		    double *S1,
		    double *CDM1,
		    double *EDM1,
		    dcomplex **H)
{
  static int firsttime=1;
  int i,j,k,l,m,n,wan,MaxN,i0,ks;
  int i1,i1s,j1,ia,jb,lmax,po,po1,spin,s1,e1;
  int num2,RnB,l1,l2,l3,loop_num,ns,ne;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int MA_AN,GA_AN,Anum,num_kloop0;
  int T_knum,S_knum,E_knum,kloop,kloop0;
  double av_num,lumos;
  double time0;
  int LB_AN,GB_AN,Bnum;
  double k1,k2,k3,Fkw;
  double sum,sumi,sum_weights;
  double Num_State;
  double My_Num_State;
  double FermiF;
  double tmp,eig,kw,EV_cut0;
  double x,Dnum,Dnum2,AcP,ChemP_MAX,ChemP_MIN;
  int *is1,*ie1;
  int *is2,*ie2;
  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;
  MPI_Comm *MPI_CommWD_CDM1; 
  int *MPI_CDM1_flag;  

  int all_knum; 
  dcomplex Ctmp1,Ctmp2;
  int ii,ij,ik;
  int BM,BN,BK;
  double u2,v2,uv,vu;
  double d1,d2;
  double My_Eele1[2]; 
  double TZ,dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2;
  double Imsum,ImsumE,Imdum,Imdum2;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double Stime,Etime,Stime0,Etime0;
  double FermiEps=1.0e-14;
  double x_cut=30.0;

  char file_EV[YOUSO10];
  FILE *fp_EV;
#ifdef xt3
  char buf[fp_bsize];          /* setvbuf */
#endif
  int AN,Rn,size_H1;
  int parallel_mode;
  int numprocs0,myid0;
  int ID,ID0,ID1;
  int numprocs1,myid1;
  int numprocs2,myid2;
  int Num_Comm_World1;
  int Num_Comm_World2;

  int tag=999,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;

  double time1,time2,time3;
  double time4,time5,time6;
  double time8,time9;
  double time10,time11;
  double time81;
  double time51;

  int ***k_op;
  int *T_k_op;
  int **T_k_ID;
  double *T_KGrids1;
  double *T_KGrids2;
  double *T_KGrids3;
  double ***EIGEN;

  dcomplex *Ss;
  dcomplex *Cs;
  dcomplex *Hs;

  int nblk_m;

  int myworld1;
  int myworld2;

  int *NPROCS_ID1;
  int *Comm_World1;
  int *NPROCS_WD1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;

  int *NPROCS_ID2;
  int *NPROCS_WD2;
  int *Comm_World2;
  int *Comm_World_StartID2;
  MPI_Comm *MPI_CommWD2;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  FILE* file;
  char* BUF[1000];

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int info,ig,jg,il,jl,prow,pcol,brow,bcol;
  int ZERO=0, ONE=1;
  double alpha = 1.0; double beta = 0.0;
  int LOCr, LOCc, node, irow, icol;
  MPI_Status *stat_send2;
  double mC_spin_i1,C_spin_i1;

  /* YTL-start */
  double* fd_dist;
  /* YTL-end */

  /* for time */
  dtime(&TStime);

  time1 = 0.0;
  time2 = 0.0;
  time3 = 0.0;
  time4 = 0.0;
  time5 = 0.0;
  time6 = 0.0;
  time8 = 0.0;
  time9 = 0.0;
  time10 = 0.0;
  time11 = 0.0;
  time81 = 0.0;
  time51 = 0.0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);
  MPI_Barrier(mpi_comm_level1);

  /****************************************************
   find the number of basis functions, n
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n += Spe_Total_CNO[wanA];
  }

  /****************************************************
   find TZ
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ += Spe_Core_Charge[wan];
  }

  /***********************************************
     find the number of states to be solved 
  ***********************************************/

  lumos = (double)n*0.300;
  if (lumos<60.0) lumos = 400.0;
  MaxN = (TZ-system_charge)/2 + (int)lumos;
  if (n<MaxN) MaxN = n;

  /***********************************************
     allocation of arrays
  ***********************************************/

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  MPI_CommWD_CDM1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*numprocs0);
  MPI_CDM1_flag = (int*)malloc(sizeof(int)*numprocs0);
  for (i=0; i<numprocs0; i++) MPI_CDM1_flag[i] = 0;

  k_op = (int***)malloc(sizeof(int**)*knum_i);
  for (i=0;i<knum_i; i++) {
    k_op[i] = (int**)malloc(sizeof(int*)*knum_j);
    for (j=0;j<knum_j; j++) {
      k_op[i][j] = (int*)malloc(sizeof(int)*knum_k);
    }
  }

/*
  if (firsttime) {
  PrintMemory("Band_DFT: ko", sizeof(double)*(n+1),NULL);
  PrintMemory("Band_DFT: koS",sizeof(double)*(n+1),NULL);
  PrintMemory("Band_DFT: H",  sizeof(dcomplex)*(n+1)*(n+1),NULL);
  PrintMemory("Band_DFT: S",  sizeof(dcomplex)*(n+1)*(n+1),NULL);
  PrintMemory("Band_DFT: C",  sizeof(dcomplex)*(n+1)*(n+1),NULL);
  }
*/
  /***********************************************
              k-points by regular mesh 
  ***********************************************/

  /**************************************************************
     k_op[i][j][k]: weight of DOS 
                 =0   no calc.
                 =1   G-point
                 =2   which has k<->-k point
        Now , only the relation, E(k)=E(-k), is used. 

    Future release: k_op will be used for symmetry operation 
  *************************************************************/

  for (i=0;i<=knum_i-1;i++) {
    for (j=0;j<=knum_j-1;j++) {
      for (k=0;k<=knum_k-1;k++) {
	k_op[i][j][k]=-999;
      }
    }
  }

  for (i=0;i<=knum_i-1;i++) {
    for (j=0;j<=knum_j-1;j++) {
      for (k=0;k<=knum_k-1;k++) {
	if ( k_op[i][j][k]==-999 ) {
	  k_inversion(i,j,k,knum_i,knum_j,knum_k,&ii,&ij,&ik);
	  if ( i==ii && j==ij && k==ik ) {
	    k_op[i][j][k]    = 1;
	  }

	  else {
	    k_op[i][j][k]    = 2;
	    k_op[ii][ij][ik] = 0;
	  }
	}
      } /* k */
    } /* j */
  } /* i */

  /* find T_knum */

  T_knum = 0;
  for (i=0; i<knum_i; i++) {
    for (j=0; j<knum_j; j++) {
      for (k=0; k<knum_k; k++) {
	if (0<k_op[i][j][k]){
	  T_knum++;
	}
      }
    }
  }

  T_KGrids1 = (double*)malloc(sizeof(double)*T_knum);
  T_KGrids2 = (double*)malloc(sizeof(double)*T_knum);
  T_KGrids3 = (double*)malloc(sizeof(double)*T_knum);
  T_k_op    = (int*)malloc(sizeof(int)*T_knum);

  T_k_ID    = (int**)malloc(sizeof(int*)*2);
  for (i=0; i<2; i++){
    T_k_ID[i] = (int*)malloc(sizeof(int)*T_knum);
  }

  EIGEN  = (double***)malloc(sizeof(double**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    EIGEN[i] = (double**)malloc(sizeof(double*)*T_knum);
    for (j=0; j<T_knum; j++){
      EIGEN[i][j] = (double*)malloc(sizeof(double)*(n+1));
      for (k=0; k<(n+1); k++) EIGEN[i][j][k] = 1.0e+5;
    }
  }

  Num_Comm_World1 = SpinP_switch + 1; 

  NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs0); 
  Comm_World1 = (int*)malloc(sizeof(int)*numprocs0); 
  NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
  Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);

  Make_Comm_Worlds(mpi_comm_level1, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1, 
		   NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

  stat_send2 = malloc(sizeof(MPI_Status)*numprocs1);

  if (T_knum<=numprocs1){

    Num_Comm_World2 = T_knum;

    NPROCS_ID2 = (int*)malloc(sizeof(int)*numprocs1);
    Comm_World2 = (int*)malloc(sizeof(int)*numprocs1);
    NPROCS_WD2 = (int*)malloc(sizeof(int)*Num_Comm_World2);
    Comm_World_StartID2 = (int*)malloc(sizeof(int)*Num_Comm_World2);
    MPI_CommWD2 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2);

    Make_Comm_Worlds(MPI_CommWD1[myworld1], myid1, numprocs1, Num_Comm_World2, 
		     &myworld2, MPI_CommWD2, NPROCS_ID2, Comm_World2, 
		     NPROCS_WD2, Comm_World_StartID2);


    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);

    np_cols = (int)(sqrt((float)numprocs2));
    do{
      if((numprocs2%np_cols)==0) break;
      np_cols--;
    } while (np_cols>=2);
    np_rows = numprocs2/np_cols;

    nblk_m = NBLK;
    while((nblk_m*np_rows>n || nblk_m*np_cols>n) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    my_prow = myid2/np_cols;
    my_pcol = myid2%np_cols;

    na_rows = numroc_(&n, &nblk, &my_prow, &ZERO, &np_rows);
    na_cols = numroc_(&n, &nblk, &my_pcol, &ZERO, &np_cols );

    bhandle2 = Csys2blacs_handle(MPI_CommWD2[myworld2]);
    ictxt2 = bhandle2;
    Cblacs_gridinit(&ictxt2, "Row", np_rows, np_cols);
	
    Ss = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
    Hs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);

    MPI_Allreduce(&na_rows,&na_rows_max,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols,&na_cols_max,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    Cs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max*na_cols_max);

    descinit_(descS, &n,   &n,   &nblk,  &nblk,  &ZERO, &ZERO, &ictxt2, &na_rows,  &info);
    descinit_(descH, &n,   &n,   &nblk,  &nblk,  &ZERO, &ZERO, &ictxt2, &na_rows,  &info);
    descinit_(descC, &n,   &n,   &nblk,  &nblk,  &ZERO, &ZERO, &ictxt2, &na_rows,  &info);
  }

  else {

    Num_Comm_World2 = numprocs1;

    NPROCS_ID2 = (int*)malloc(sizeof(int)*numprocs1);
    Comm_World2 = (int*)malloc(sizeof(int)*numprocs1);
    NPROCS_WD2 = (int*)malloc(sizeof(int)*Num_Comm_World2);
    Comm_World_StartID2 = (int*)malloc(sizeof(int)*Num_Comm_World2);
    MPI_CommWD2 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2);

    Make_Comm_Worlds(MPI_CommWD1[myworld1], myid1, numprocs1, Num_Comm_World2,
		     &myworld2, MPI_CommWD2, 
		     NPROCS_ID2, Comm_World2, NPROCS_WD2, Comm_World_StartID2);


    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);

    np_cols = (int)(sqrt((float)numprocs2));
    do{
      if((numprocs2%np_cols)==0) break;
      np_cols--;
    } while (np_cols>=2);
    np_rows = numprocs2/np_cols;

    nblk_m = NBLK;
    while((nblk_m*np_rows>n || nblk_m*np_cols>n) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    my_prow = myid2/np_cols;
    my_pcol = myid2%np_cols;

    na_rows = numroc_(&n, &nblk, &my_prow, &ZERO, &np_rows);
    na_cols = numroc_(&n, &nblk, &my_pcol, &ZERO, &np_cols );

    bhandle2 = Csys2blacs_handle(MPI_CommWD2[myworld2]);
    ictxt2 = bhandle2;
    Cblacs_gridinit(&ictxt2, "Row", np_rows, np_cols);

    Ss = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
    Hs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);

    MPI_Allreduce(&na_rows,&na_rows_max,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols,&na_cols_max,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    Cs = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max*na_cols_max);

    descinit_(descS, &n,   &n,   &nblk,  &nblk,  &ZERO, &ZERO, &ictxt2, &na_rows,  &info);
    descinit_(descH, &n,   &n,   &nblk,  &nblk,  &ZERO, &ZERO, &ictxt2, &na_rows,  &info);
    descinit_(descC, &n,   &n,   &nblk,  &nblk,  &ZERO, &ZERO, &ictxt2, &na_rows,  &info);

    /*
      printf("myid0=%d myid2=%d np2=%d my_pr=%d my_pc=%d na_r=%d na_c=%d np_r=%d np_c=%d nblk=%d T_k=%d ictxt2=%d n=%d\n",
      myid0,myid2,numprocs2,my_prow,my_pcol,na_rows,na_cols,np_rows,np_cols,nblk,T_knum,ictxt2,n);
    */
  }

  /***********************************
     one-dimentionalize for MPI
  ************************************/

  /* YTL-start */

  /* Step (1) : calculate < phi( atom a, orbital alpha) | nabla | phi( atom b, orbital beta) > */
  Calc_NabraMatrixElements();

  Initialize_optical();

  /* YTL-end */


  /* set T_KGrids1,2,3 and T_k_op */

  T_knum = 0;
  for (i=0; i<knum_i; i++){

    if (knum_i==1)  k1 = 0.0;
    else            k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_i) + Shift_K_Point;

    for (j=0; j<knum_j; j++){

      if (knum_j==1)  k2 = 0.0;
      else            k2 = -0.5 + (2.0*(double)j+1.0)/(2.0*(double)knum_j) - Shift_K_Point;

      for (k=0; k<knum_k; k++){

	if (knum_k==1)  k3 = 0.0;
	else            k3 = -0.5 + (2.0*(double)k+1.0)/(2.0*(double)knum_k) + 2.0*Shift_K_Point;

	if (0<k_op[i][j][k]){

	  T_KGrids1[T_knum] = k1;
	  T_KGrids2[T_knum] = k2;
	  T_KGrids3[T_knum] = k3;
	  T_k_op[T_knum]    = k_op[i][j][k];

	  T_knum++;
	}
      }
    }
  }

  if (myid0==Host_ID && 0<level_stdout){
    printf(" CDDF.KGrids1: ");fflush(stdout);
    for (i=0;i<=knum_i-1;i++){
      if (knum_i==1)  k1 = 0.0;
      else            k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_i) + Shift_K_Point;
      printf("%9.5f ",k1);fflush(stdout);
    }
    printf("\n");fflush(stdout);

    printf(" CDDF.KGrids2: ");fflush(stdout);

    for (i=0;i<=knum_j-1;i++){
      if (knum_j==1)  k2 = 0.0;
      else            k2 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_j) - Shift_K_Point;
      printf("%9.5f ",k2);fflush(stdout);
    }
    printf("\n");fflush(stdout);

    printf(" CDDF.KGrids3: ");fflush(stdout);
    for (i=0;i<=knum_k-1;i++){
      if (knum_k==1)  k3 = 0.0;
      else            k3 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_k) + 2.0*Shift_K_Point;
      printf("%9.5f ",k3);fflush(stdout);
    }
    printf("\n");fflush(stdout);
  }

  /***********************************************
            calculate the sum of weights
  ***********************************************/

  sum_weights = 0.0;
  for (k=0; k<T_knum; k++){
    sum_weights += (double)T_k_op[k];
  }

  /***********************************************
           allocate k-points into processors 
  ***********************************************/

  if (numprocs1<T_knum){

    /* set parallel_mode */
    parallel_mode = 0;

    /* allocation of kloop to ID */     

    for (ID=0; ID<numprocs1; ID++){
      tmp = (double)T_knum/(double)numprocs1;
      S_knum = (int)((double)ID*(tmp+1.0e-12)); 
      E_knum = (int)((double)(ID+1)*(tmp+1.0e-12)) - 1;
      if (ID==(numprocs1-1)) E_knum = T_knum - 1;
      if (E_knum<0)          E_knum = 0;

      for (k=S_knum; k<=E_knum; k++){
        /* ID in the first level world */
        T_k_ID[myworld1][k] = ID;
      }
    }

    /* find own informations */

    tmp = (double)T_knum/(double)numprocs1; 
    S_knum = (int)((double)myid1*(tmp+1.0e-12)); 
    E_knum = (int)((double)(myid1+1)*(tmp+1.0e-12)) - 1;
    if (myid1==(numprocs1-1)) E_knum = T_knum - 1;
    if (E_knum<0)             E_knum = 0;

    num_kloop0 = E_knum - S_knum + 1;

    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);

    Set_MPIworld_for_optical(myid2,numprocs2);
  }

  else {

    /* set parallel_mode */
    parallel_mode = 1;
    num_kloop0 = 1;

    Num_Comm_World2 = T_knum;
    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);

    S_knum = myworld2;

    /* allocate k-points into processors */
    
    for (k=0; k<T_knum; k++){
      /* ID in the first level world */
      T_k_ID[myworld1][k] = Comm_World_StartID2[k];
    }

    Set_MPIworld_for_optical(myid2,numprocs2);
  }

  /****************************************************
   find all_knum
   if (all_knum==1), all the calculation will be made 
   by the first diagonalization loop, and the second 
   diagonalization will be skipped. 
  ****************************************************/

  MPI_Allreduce(&num_kloop0, &all_knum, 1, MPI_INT, MPI_PROD, mpi_comm_level1);

  if (SpinP_switch==1 && numprocs0==1 && all_knum==1){
    all_knum = 0;
  }

  /****************************************************
    if (parallel_mode==1 && all_knum==1)
     make is1, ie1, is2, ie2
  ****************************************************/

  if (all_knum==1){

    /* allocation */ 

    stat_send = malloc(sizeof(MPI_Status)*numprocs2);
    request_send = malloc(sizeof(MPI_Request)*numprocs2);
    request_recv = malloc(sizeof(MPI_Request)*numprocs2);

    is1 = (int*)malloc(sizeof(int)*numprocs2);
    ie1 = (int*)malloc(sizeof(int)*numprocs2);

    is2 = (int*)malloc(sizeof(int)*numprocs2);
    ie2 = (int*)malloc(sizeof(int)*numprocs2);

    /* make is1 and ie1 */ 

    if ( numprocs2<=n ){

      av_num = (double)n/(double)numprocs2;

      for (ID=0; ID<numprocs2; ID++){
      	is1[ID] = (int)(av_num*(double)ID) + 1; 
      	ie1[ID] = (int)(av_num*(double)(ID+1)); 
      }

      is1[0] = 1;
      ie1[numprocs2-1] = n; 

    }

    else{

      for (ID=0; ID<n; ID++){
	is1[ID] = ID + 1; 
	ie1[ID] = ID + 1;
      }
      for (ID=n; ID<numprocs2; ID++){
	is1[ID] =  1;
	ie1[ID] = -2;
      }
    }

    /* make is2 and ie2 */ 

    if ( numprocs2<=MaxN ){

      av_num = (double)MaxN/(double)numprocs2;

      for (ID=0; ID<numprocs2; ID++){
	is2[ID] = (int)(av_num*(double)ID) + 1; 
	ie2[ID] = (int)(av_num*(double)(ID+1)); 
      }

      is2[0] = 1;
      ie2[numprocs2-1] = MaxN; 
    }

    else{
      for (ID=0; ID<MaxN; ID++){
	is2[ID] = ID + 1; 
	ie2[ID] = ID + 1;
      }
      for (ID=MaxN; ID<numprocs2; ID++){
	is2[ID] =  1;
	ie2[ID] = -2;
      }
    }

  } /* if (all_knum==1) */

  /****************************************************
     communicate T_k_ID
  ****************************************************/

  if (numprocs0==1 && SpinP_switch==1){
    for (k=0; k<T_knum; k++){
      T_k_ID[1][k] = T_k_ID[0][k];
    }
  }
  else{
    for (spin=0; spin<=SpinP_switch; spin++){
      ID = Comm_World_StartID1[spin];
      MPI_Bcast(&T_k_ID[spin][0], T_knum, MPI_INT, ID, mpi_comm_level1);
    }
  }

  /****************************************************
     store in each processor all the matrix elements
        for overlap and Hamiltonian matrices
  ****************************************************/

  dtime(&Stime);

  /* spin=myworld1 */

  spin = myworld1;

  /* set S1 */

  if (SCF_iter==1 || all_knum!=1){
    size_H1 = Get_OneD_HS_Col(1, CntOLP,   S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

diagonalize1:

  /* set H1 */

  if (SpinP_switch==0){ 
    size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }
  else if (1<numprocs0){
    size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, nh[1], CDM1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    if (myworld1){
      for (i=0; i<size_H1; i++){
        H1[i] = CDM1[i];
      }
    }
  }
  else{
    size_H1 = Get_OneD_HS_Col(1, nh[spin], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  dtime(&Etime);
  time1 += Etime - Stime;

  /****************************************************
                      start kloop
  ****************************************************/

  dtime(&SiloopTime);

  for (kloop0=0; kloop0<num_kloop0; kloop0++){

    kloop = S_knum + kloop0;

    k1 = T_KGrids1[kloop];
    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];

    /* make S and H */

    for (i1=1; i1<=n; i1++){
      for (j1=1; j1<=n; j1++){
	H[i1][j1] = Complex(0.0,0.0);
      } 
    } 

    if (SCF_iter==1 || all_knum!=1){
      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){

	    for (j=0; j<tnoB; j++){

	      H[Anum+i][Bnum+j].r += S1[k]*co;
	      H[Anum+i][Bnum+j].i += S1[k]*si;

	      k++;

	    }
	  }
	}
      }

      for(i=0;i<na_rows;i++){
	for(j=0;j<na_cols;j++){
	  ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
	  jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	  Cs[j*na_rows+i].r = H[ig][jg].r;
	  Cs[j*na_rows+i].i = H[ig][jg].i;
	}
      }
      
    }


    for (i1=1; i1<=n; i1++){
      for (j1=1; j1<=n; j1++){
	H[i1][j1] = Complex(0.0,0.0);
      } 
    } 

    k = 0;
    for (AN=1; AN<=atomnum; AN++){
      GA_AN = order_GA[AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);

	for (i=0; i<tnoA; i++){

	  for (j=0; j<tnoB; j++){

	    H[Anum+i][Bnum+j].r += H1[k]*co;
	    H[Anum+i][Bnum+j].i += H1[k]*si;

	    k++;

	  }

	}
      }
    }
    

    for(i=0;i<na_rows;i++){
      for(j=0;j<na_cols;j++){
	ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	Hs[j*na_rows+i].r = H[ig][jg].r;
	Hs[j*na_rows+i].i = H[ig][jg].i;
      }
    }


    /* diagonalize S */

    dtime(&Stime);

    if (parallel_mode==0 || (SCF_iter==1 || all_knum!=1)){
      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)(&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);
    }

    dtime(&Etime);
    time2 += Etime - Stime;

    if (SCF_iter==1 || all_knum!=1){
/*
      if (3<=level_stdout){
	printf(" myid0=%2d spin=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,spin,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,ko[i1]);
	}
      }
*/

      /* minus eigenvalues to 1.0e-14 */

      for (l=1; l<=n; l++){
	if (ko[l]<0.0) ko[l] = 1.0e-14;
	koS[l] = ko[l];
      }

      /* calculate S*1/sqrt(ko) */

      for (l=1; l<=n; l++) ko[l] = 1.0/sqrt(ko[l]);

      /* S * 1.0/sqrt(ko[l]) */

      for(i=0;i<na_rows;i++){
	for(j=0;j<na_cols;j++){
	  jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	  Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	  Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
	}
      }

    }

    /****************************************************
     1.0/sqrt(ko[l]) * U^t * H * U * 1.0/sqrt(ko[l])
    ****************************************************/

    dtime(&Stime);

    /* pzgemm */

    /* H * U * 1.0/sqrt(ko[l]) */

    for(i=0;i<na_rows_max*na_cols_max;i++){
      Cs[i].r = 0.0;
      Cs[i].i = 0.0;
    }
    Cblacs_barrier(ictxt2,"A");

    F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

    /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

    for(i=0;i<na_rows*na_cols;i++){
      Hs[i].r = 0.0;
      Hs[i].i = 0.0;
    }
    Cblacs_barrier(ictxt2,"C");

    F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);


    /* penalty for ill-conditioning states */

    EV_cut0 = Threshold_OLP_Eigen;

    for (i1=1; i1<=n; i1++){

      ig = i1;
      jg = i1;
              
      brow = (ig-1)/nblk;
      bcol = (jg-1)/nblk;
    
      prow = brow%np_rows;
      pcol = bcol%np_cols;

      if(my_prow==prow && my_pcol==pcol){
	il = (brow/np_rows+1)*nblk+1;
	jl = (bcol/np_cols+1)*nblk+1;
	if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	  if(my_prow==prow){
	    il = il+(ig-1)%nblk;
	  }
	  il = il-nblk;
	}
	if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	  if(my_pcol==pcol){
	    jl = jl+(jg-1)%nblk;
	  }
	  jl = jl-nblk;
	}
	if(koS[i1]<EV_cut0){
	  Hs[(jl-1)*na_rows+il-1].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
	}
	mC_spin_i1 = Hs[(jl-1)*na_rows+il-1].r;
      }
      else{
	mC_spin_i1 = 0.0;
      }

      MPI_Allreduce(&mC_spin_i1, &C_spin_i1, 1, MPI_DOUBLE, MPI_SUM, MPI_CommWD2[myworld2]);
 
      /* cutoff the interaction between the ill-conditioned state */

      if (1.0e+3<C_spin_i1){
	for (j1=1; j1<=n; j1++){

	  ig = i1;
	  jg = j1;
              
	  brow = (ig-1)/nblk;
	  bcol = (jg-1)/nblk;

	  prow = brow%np_rows;
	  pcol = bcol%np_cols;

	  if(my_prow==prow && my_pcol==pcol){
	    il = (brow/np_rows+1)*nblk+1;
	    jl = (bcol/np_cols+1)*nblk+1;
	    if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	      if(my_prow==prow){
		il = il+(ig-1)%nblk;
	      }
	      il = il-nblk;
	    }
	    if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	      if(my_pcol==pcol){
		jl = jl+(jg-1)%nblk;
	      }
	      jl = jl-nblk;
	    }
	    Hs[(jl-1)*na_rows+il-1] = Complex(0.0,0.0);
	  }

	  ig = j1;
	  jg = i1;
              
	  brow = (ig-1)/nblk;
	  bcol = (jg-1)/nblk;

	  prow = brow%np_rows;
	  pcol = bcol%np_cols;

	  if(my_prow==prow && my_pcol==pcol){
	    il = (brow/np_rows+1)*nblk+1;
	    jl = (bcol/np_cols+1)*nblk+1;
	    if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	      if(my_prow==prow){
		il = il+(ig-1)%nblk;
	      }
	      il = il-nblk;
	    }
	    if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	      if(my_pcol==pcol){
		jl = jl+(jg-1)%nblk;
	      }
	      jl = jl-nblk;
	    }
	    Hs[(jl-1)*na_rows+il-1] = Complex(0.0,0.0);
	  }
	}

	ig = i1;
	jg = i1;
              
	brow = (ig-1)/nblk;
	bcol = (jg-1)/nblk;

	prow = brow%np_rows;
	pcol = bcol%np_cols;

	if(my_prow==prow && my_pcol==pcol){
	  il = (brow/np_rows+1)*nblk+1;
	  jl = (bcol/np_cols+1)*nblk+1;
	  if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	    if(my_prow==prow){
	      il = il+(ig-1)%nblk;
	    }
	    il = il-nblk;
	  }
	  if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	    if(my_pcol==pcol){
	      jl = jl+(jg-1)%nblk;
	    }
	    jl = jl-nblk;
	  }
	  Hs[(jl-1)*na_rows+il-1].r = 1.0e+4;
	}
      }
      
    }



    dtime(&Etime);
    time3 += Etime - Stime;

    /* diagonalize H' */

    dtime(&Stime);

    MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
	
    F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)(&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);


    dtime(&Etime);
    time4 += Etime - Stime;

    for (l=1; l<=MaxN; l++){
      EIGEN[spin][kloop][l] = ko[l];
    }
/*
    if (3<=level_stdout && 0<=kloop){
      printf(" myid0=%2d spin=%2d kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	     myid0,spin,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
      for (i1=1; i1<=n; i1++){
	if (SpinP_switch==0)
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,EIGEN[0][kloop][i1],EIGEN[0][kloop][i1]);
	else 
	  printf("  Eigenvalues of Kohn-Sham %2d %15.12f %15.12f\n",
		 i1,EIGEN[0][kloop][i1],EIGEN[1][kloop][i1]);
      }
    }
*/
    /* calculation of wave functions */

    dtime(&Stime);

    if (all_knum==1){

      for(i=0;i<na_rows*na_cols;i++){
	Hs[i].r = 0.0;
	Hs[i].i = 0.0;
      }

      F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);
      Cblacs_barrier(ictxt2,"A");

      if (myid2==0){

	for(node=0;node<numprocs2;node++){
     
	  if(node==0){

	    for(i=0;i<na_rows;i++){
	      for(j=0;j<na_cols;j++){
		ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
		jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
		H[ig][jg].r = Hs[j*na_rows+i].r;
		H[ig][jg].i = Hs[j*na_rows+i].i;
	      }
	    }
	  }

	  else{
         
	    MPI_Recv(&irow, 1, MPI_INT, node, 10, MPI_CommWD2[myworld2], stat_send2);
	    MPI_Recv(&icol, 1, MPI_INT, node, 20, MPI_CommWD2[myworld2], stat_send2);
	    MPI_Recv(&LOCr, 1, MPI_INT, node, 40, MPI_CommWD2[myworld2], stat_send2);
	    MPI_Recv(&LOCc, 1, MPI_INT, node, 50, MPI_CommWD2[myworld2], stat_send2);
	    
	    MPI_Recv(Cs, LOCr*LOCc*2, MPI_DOUBLE, node, 30, MPI_CommWD2[myworld2], stat_send2);

	    for(i=0;i<LOCr;i++){
	      for(j=0;j<LOCc;j++){
		ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+irow)%np_rows)*nblk + 1;
		jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+icol)%np_cols)*nblk + 1;
		H[ig][jg].r = Cs[j*LOCr+i].r;
		H[ig][jg].i = Cs[j*LOCr+i].i;
	      }
	    }
	  }
	}
      }
      else{
	MPI_Send(&my_prow, 1, MPI_INT, 0, 10, MPI_CommWD2[myworld2]);
	MPI_Send(&my_pcol, 1, MPI_INT, 0, 20, MPI_CommWD2[myworld2]);
	MPI_Send(&na_rows, 1, MPI_INT, 0, 40, MPI_CommWD2[myworld2]);
	MPI_Send(&na_cols, 1, MPI_INT, 0, 50, MPI_CommWD2[myworld2]);
	MPI_Send(Hs, na_rows*na_cols*2, MPI_DOUBLE, 0, 30, MPI_CommWD2[myworld2]);
      } 

      /*
	! Broadcast the eigenvectors to all proceses 
      */

      for(i=1;i<=n;i++){
	MPI_Bcast(H[i],(n+1)*2,MPI_DOUBLE,0,MPI_CommWD2[myworld2]);
      }

      dtime(&Etime0);
      time51 += Etime0 - Stime0;

    } /* if (all_knum==1) */

    dtime(&Etime);
    time5 += Etime - Stime;

  } /* kloop0 */

  dtime(&EiloopTime);

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;  
    goto diagonalize1; 
  }

  /****************************************************
     MPI:

     EIGEN
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&Stime);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (kloop=0; kloop<T_knum; kloop++){

      /* get ID in the zeroth world */
      ID = Comm_World_StartID1[spin] + T_k_ID[spin][kloop];
      MPI_Bcast(&EIGEN[spin][kloop][0], MaxN+1, MPI_DOUBLE, ID, mpi_comm_level1);
    } 
  }

  dtime(&Etime);
  time6 += Etime - Stime;

  /**************************************
         find chemical potential
  **************************************/

  dtime(&Stime);
  
  po = 0;
  loop_num = 0;
  ChemP_MAX = 10.0;  
  ChemP_MIN =-10.0;

  do {

    loop_num++;

    ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
    Num_State = 0.0;

    for (kloop=0; kloop<T_knum; kloop++){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (l=1; l<=MaxN; l++){

	  x = (EIGEN[spin][kloop][l] - ChemP)*Beta;

	  if (x<=-x_cut)      FermiF = 1.0;
	  else if (x>=x_cut)  FermiF = 0.0;
	  else                FermiF = 1.0/(1.0 + exp(x));

	  Num_State += FermiF*(double)T_k_op[kloop];

	} 
      }  
    } 

    if (SpinP_switch==0) 
      Num_State = 2.0*Num_State/sum_weights;
    else 
      Num_State = Num_State/sum_weights;
 
    Dnum = TZ - Num_State - system_charge;

    if (0.0<=Dnum) ChemP_MIN = ChemP;
    else           ChemP_MAX = ChemP;
    if (fabs(Dnum)<10e-14) po = 1;
  }
  while (po==0 && loop_num<2000);

  /*************************************************
     determination of CDDF_max_unoccupied_state
  *************************************************/

  /* YTL-start */
  fd_dist = (double*)malloc(sizeof(double)*n);
  for (i=0; i<n; i++) fd_dist[i] = 0.0; /* initialize fermi-dirac distribution */
  /* YTL-end */

  /* determine the maximum unoccupied state */

  double range_Ha=(CDDF_max_eV-CDDF_min_eV)/eV2Hartree; /* in Hartree */
  double p1 = CDDF_AddMaxE/eV2Hartree;
  double FDFi, FDFl;

  j = 0;
  for (spin=0; spin<=SpinP_switch; spin++){
    for (kloop=0; kloop<T_knum; kloop++){

      if (0<T_k_op[kloop]){

  for (i=0; i<MaxN; i++){ /* occupied state */

          eig = EIGEN[spin][kloop][i+1];
          x = (eig - ChemP)*Beta;
    if      (x<-x_cut)  FDFi = 1.0;
    else if (x>x_cut)   FDFi = 0.0;
    else                FDFi = 1.0/(1.0 + exp(x));

    for (l=i+1; l<MaxN; l++){ /* unoccupied state */

      eig = EIGEN[spin][kloop][l+1];
      x = (eig - ChemP)*Beta;
      if      (x<-x_cut)  FDFl = 1.0;
      else if (x>x_cut)   FDFl = 0.0;
      else                FDFl = 1.0/(1.0 + exp(x));

      k2 = FDFi - FDFl; /* dFE */
            k3 = fabs( EIGEN[spin][kloop][i+1] - EIGEN[spin][kloop][l+1] ) ; /* dE */
        if ( k2!=0 && k3 <= range_Ha + p1 && l > j) j = l; /* update the highest state */
    }
  }
      }
    }
  }

  if (j > CDDF_max_unoccupied_state){
    CDDF_max_unoccupied_state = j; /* set the highest state. */
  }
  else{ 
    j = CDDF_max_unoccupied_state;
  }
      
  MPI_Allreduce(&j, &CDDF_max_unoccupied_state, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

  /* YTL-end */

  /****************************************************
         if all_knum==1, calculate CDM and EDM
  ****************************************************/

  dtime(&Stime);

  if (all_knum==1){

    dtime(&Stime0);

    /* calculate CDM and EDM */

    spin = myworld1;
    kloop = S_knum;

    k1 = T_KGrids1[kloop];
    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];

    /* weight of k-point */ 

    kw = (double)T_k_op[kloop];

    /* store Fermi function */

    po = 0;
    l = 1;
    do{

      eig = EIGEN[spin][kloop][l];
      x = (eig - ChemP)*Beta;

      if      (x<-x_cut)  FermiF = 1.0;
      else if (x>x_cut)   FermiF = 0.0;
      else                FermiF = 1.0/(1.0 + exp(x));

      /* YTL-start */
      if (SpinP_switch==0){
        fd_dist[l-1] = 2.0*FermiF*kw;
      }else if (SpinP_switch==1){
        fd_dist[l-1] = FermiF*kw;
      }
      /* YTL-end */

      if ( FermiF<=FermiEps && po==0 ) {
      	lmax = l;  
      	po = 1;
      }

      l++;

    } while(po==0 && l<=MaxN);

    if (po==0) lmax = MaxN;

    /* YTL-start */
    Calc_band_optical_col_1(T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop],spin,n,EIGEN[spin][kloop],H,fd_dist,ChemP);
    /* YTL-end */

    dtime(&Etime0);
    time81 += Etime0 - Stime0;

  } /* if (all_knum==1) */

  dtime(&Etime);
  time8 += Etime - Stime;

  dtime(&SiloopTime);

  /****************************************************
   ****************************************************
     diagonalization for calculating density matrix
   ****************************************************
  ****************************************************/

  if (all_knum!=1){

    /* spin=myworld1 */

    spin = myworld1;

  diagonalize2:

    /* set S1 */

    size_H1 = Get_OneD_HS_Col(1, CntOLP,   S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    /* set H1 */

    if (SpinP_switch==0){ 
      size_H1 = Get_OneD_HS_Col(1, nh[0], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }
    else if (1<numprocs0){
      size_H1 = Get_OneD_HS_Col(1, nh[0], H1,   MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
      size_H1 = Get_OneD_HS_Col(1, nh[1], CDM1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

      if (myworld1){
	for (i=0; i<size_H1; i++){
	  H1[i] = CDM1[i];
	}
      }
    }
    else{
      size_H1 = Get_OneD_HS_Col(1, nh[spin], H1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    }

    /* for kloop */

    for (kloop0=0; kloop0<num_kloop0; kloop0++){

      kloop = kloop0 + S_knum;

      k1 = T_KGrids1[kloop];
      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];

      /* make S and H */

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  H[i1][j1] = Complex(0.0,0.0);
	} 
      } 

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      H[Anum+i][Bnum+j].r += S1[k]*co;
	      H[Anum+i][Bnum+j].i += S1[k]*si;

	      k++;

	    }
	  }
	}
      }

      for(i=0;i<na_rows;i++){
	for(j=0;j<na_cols;j++){
	  ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
	  jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	  Cs[j*na_rows+i].r = H[ig][jg].r;
	  Cs[j*na_rows+i].i = H[ig][jg].i;
	}
      }


      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  H[i1][j1] = Complex(0.0,0.0);
	} 
      } 

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      H[Anum+i][Bnum+j].r += H1[k]*co;
	      H[Anum+i][Bnum+j].i += H1[k]*si;

	      k++;

	    }
	  }
	}
      }

      for(i=0;i<na_rows;i++){
	for(j=0;j<na_cols;j++){
	  ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
	  jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	  Hs[j*na_rows+i].r = H[ig][jg].r;
	  Hs[j*na_rows+i].i = H[ig][jg].i;
	}
      }

      /* diagonalize S */

      dtime(&Stime);


      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)(&n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);

      dtime(&Etime);
      time9 += Etime - Stime;
/*
      if (3<=level_stdout){
	printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,ko[i1]);
	}
      }
*/
      /* minus eigenvalues to 1.0e-14 */

      for (l=1; l<=n; l++){
	if (ko[l]<0.0) ko[l] = 1.0e-14;
	koS[l] = ko[l];
      }

      /* calculate S*1/sqrt(ko) */

      for (l=1; l<=n; l++) ko[l] = 1.0/sqrt(ko[l]);

      /* S * 1.0/sqrt(ko[l])  */


    for(i=0;i<na_rows;i++){
      for(j=0;j<na_cols;j++){
	jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
      }
    }

    /****************************************************
          1/sqrt(ko) * U^t * H * U * 1/sqrt(ko)
    ****************************************************/

    /* pzgemm */

    /* H * U * 1/sqrt(ko) */
    
    for(i=0;i<na_rows_max*na_cols_max;i++){
      Cs[i].r = 0.0;
      Cs[i].i = 0.0;
    }

    Cblacs_barrier(ictxt2,"A");

    F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Hs,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

    /* 1/sqrt(ko) * U^+ H * U * 1/sqrt(ko) */

    for(i=0;i<na_rows*na_cols;i++){
      Hs[i].r = 0.0;
      Hs[i].i = 0.0;
    }

    Cblacs_barrier(ictxt2,"C");

    F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);

    /* penalty for ill-conditioning states */

    EV_cut0 = Threshold_OLP_Eigen;

    for (i1=1; i1<=n; i1++){

      ig = i1;
      jg = i1;
              
      brow = (ig-1)/nblk;
      bcol = (jg-1)/nblk;
    
      prow = brow%np_rows;
      pcol = bcol%np_cols;

      if(my_prow==prow && my_pcol==pcol){
	il = (brow/np_rows+1)*nblk+1;
	jl = (bcol/np_cols+1)*nblk+1;
	if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	  if(my_prow==prow){
	    il = il+(ig-1)%nblk;
	  }
	  il = il-nblk;
	}
	if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	  if(my_pcol==pcol){
	    jl = jl+(jg-1)%nblk;
	  }
	  jl = jl-nblk;
	}
	if(koS[i1]<EV_cut0){
	  Hs[(jl-1)*na_rows+il-1].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
	}
	mC_spin_i1 = Hs[(jl-1)*na_rows+il-1].r;
      }
      else{
	mC_spin_i1 = 0.0;
      }

      MPI_Allreduce(&mC_spin_i1, &C_spin_i1, 1, MPI_DOUBLE, MPI_SUM, MPI_CommWD2[myworld2]);
 
      /* cutoff the interaction between the ill-conditioned state */

      if (1.0e+3<C_spin_i1){
	for (j1=1; j1<=n; j1++){

	  ig = i1;
	  jg = j1;
              
	  brow = (ig-1)/nblk;
	  bcol = (jg-1)/nblk;

	  prow = brow%np_rows;
	  pcol = bcol%np_cols;

	  if(my_prow==prow && my_pcol==pcol){
	    il = (brow/np_rows+1)*nblk+1;
	    jl = (bcol/np_cols+1)*nblk+1;
	    if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	      if(my_prow==prow){
		il = il+(ig-1)%nblk;
	      }
	      il = il-nblk;
	    }
	    if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	      if(my_pcol==pcol){
		jl = jl+(jg-1)%nblk;
	      }
	      jl = jl-nblk;
	    }
	    Hs[(jl-1)*na_rows+il-1] = Complex(0.0,0.0);
	  }

	  ig = j1;
	  jg = i1;
              
	  brow = (ig-1)/nblk;
	  bcol = (jg-1)/nblk;

	  prow = brow%np_rows;
	  pcol = bcol%np_cols;

	  if(my_prow==prow && my_pcol==pcol){
	    il = (brow/np_rows+1)*nblk+1;
	    jl = (bcol/np_cols+1)*nblk+1;
	    if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	      if(my_prow==prow){
		il = il+(ig-1)%nblk;
	      }
	      il = il-nblk;
	    }
	    if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	      if(my_pcol==pcol){
		jl = jl+(jg-1)%nblk;
	      }
	      jl = jl-nblk;
	    }
	    Hs[(jl-1)*na_rows+il-1] = Complex(0.0,0.0);
	  }
	}

	ig = i1;
	jg = i1;
              
	brow = (ig-1)/nblk;
	bcol = (jg-1)/nblk;

	prow = brow%np_rows;
	pcol = bcol%np_cols;

	if(my_prow==prow && my_pcol==pcol){
	  il = (brow/np_rows+1)*nblk+1;
	  jl = (bcol/np_cols+1)*nblk+1;
	  if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
	    if(my_prow==prow){
	      il = il+(ig-1)%nblk;
	    }
	    il = il-nblk;
	  }
	  if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
	    if(my_pcol==pcol){
	      jl = jl+(jg-1)%nblk;
	    }
	    jl = jl-nblk;
	  }
	  Hs[(jl-1)*na_rows+il-1].r = 1.0e+4;
	}
      }
      
    }

      /* diagonalize H' */

      dtime(&Stime);

      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)(&n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);


      dtime(&Etime);
      time10 += Etime - Stime;
/*
      if (3<=level_stdout && 0<=kloop){
	printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of Kohn-Sham(DM) spin=%2d i1=%2d %15.12f\n",
		 spin,i1,ko[i1]);
	}
      }
*/
      /****************************************************
        transformation to the original eigenvectors.
	     NOTE JRCAT-244p and JAIST-2122p 
      ****************************************************/

      for(i=0;i<na_rows*na_cols;i++){
	Hs[i].r = 0.0;
	Hs[i].i = 0.0;
      }

      F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);
      Cblacs_barrier(ictxt2,"A");


      if(myid2==0){

	for(node=0;node<numprocs2;node++){

	  if(node==0){

	    for(i=0;i<na_rows;i++){
	      for(j=0;j<na_cols;j++){
		ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
		jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
		H[ig][jg].r = Hs[j*na_rows+i].r;
		H[ig][jg].i = Hs[j*na_rows+i].i;
	      }
	    }
	  }

	  else{
         
	    MPI_Recv(&irow, 1, MPI_INT, node, 10, MPI_CommWD2[myworld2], stat_send2);
	    MPI_Recv(&icol, 1, MPI_INT, node, 20, MPI_CommWD2[myworld2], stat_send2);
	    MPI_Recv(&LOCr, 1, MPI_INT, node, 40, MPI_CommWD2[myworld2], stat_send2);
	    MPI_Recv(&LOCc, 1, MPI_INT, node, 50, MPI_CommWD2[myworld2], stat_send2);
         
	    MPI_Recv(Cs, LOCr*LOCc*2, MPI_DOUBLE, node, 30, MPI_CommWD2[myworld2], stat_send2);

	    for(i=0;i<LOCr;i++){
	      for(j=0;j<LOCc;j++){
		ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+irow)%np_rows)*nblk + 1;
		jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+icol)%np_cols)*nblk + 1;
		H[ig][jg].r = Cs[j*LOCr+i].r;
		H[ig][jg].i = Cs[j*LOCr+i].i;
	      }
	    }
	  }
	}
      }
      else{
	MPI_Send(&my_prow, 1, MPI_INT, 0, 10, MPI_CommWD2[myworld2]);
	MPI_Send(&my_pcol, 1, MPI_INT, 0, 20, MPI_CommWD2[myworld2]);
	MPI_Send(&na_rows, 1, MPI_INT, 0, 40, MPI_CommWD2[myworld2]);
	MPI_Send(&na_cols, 1, MPI_INT, 0, 50, MPI_CommWD2[myworld2]);
	MPI_Send(Hs, na_rows*na_cols*2, MPI_DOUBLE, 0, 30, MPI_CommWD2[myworld2]);
      } 

      /*
	! Broadcast the eigenvectors to all proceses 
      */

      for(i=1;i<=n;i++){
	MPI_Bcast(H[i],(n+1)*2,MPI_DOUBLE,0,MPI_CommWD2[myworld2]);
      }

      /* YTL-start */
      fd_dist = (double*)malloc(sizeof(double)*n); /* num_kloop0 = number of kloop at each CPU */
      for (i=0; i<n; i++) fd_dist[i] = 0.0; /* initialize fermi-dirac distribution */
      /* YTL-end */

      /****************************************************
                   calculation of Fermi function
      ****************************************************/

      dtime(&Stime);

      /* weight of k-point */ 

      kw = (double)T_k_op[kloop];

      /* store Fermi function */

      po = 0;
      l = 1;
      do{

      	eig = EIGEN[spin][kloop][l];
      	x = (eig - ChemP)*Beta;

      	if      (x<-x_cut)  FermiF = 1.0;
      	else if (x>x_cut)   FermiF = 0.0;
      	else                FermiF = 1.0/(1.0 + exp(x));

	/* YTL-start */
	if (SpinP_switch==0){
	  fd_dist[l-1] = 2.0*FermiF*kw;
	}else if (SpinP_switch==1){
	  fd_dist[l-1] = FermiF*kw;
	}
	/* YTL-end */

      	if ( FermiF<=FermiEps && po==0 ) {
      	  lmax = l;  
      	  po = 1;
      	}

      	l++;

      } while(po==0 && l<=MaxN);

      if (po==0) lmax = MaxN;

/* YTL-start */
      Calc_band_optical_col_1(T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop],spin,n,EIGEN[spin][kloop],H,fd_dist,ChemP);
/* YTL-end */

      dtime(&Etime);
      time11 += Etime - Stime;

    } /* kloop0 */

    if (SpinP_switch==1 && numprocs0==1 && spin==0){
      spin++;  
      goto diagonalize2; 
    }

  } /* if (all_knum!=1) */

  /* YTL-start */

  /****************************************************
    collect all the contributions conductivities and 
    dielectric functions
  ****************************************************/

  Calc_optical_col_2(n,sum_weights);

  /* YTL-end */
  
  /****************************************************
                       free arrays
  ****************************************************/

  if (all_knum==1){

    free(stat_send);
    free(request_send);
    free(request_recv);

    free(is1);
    free(ie1);

    free(is2);
    free(ie2);

    for (ID=0; ID<numprocs0; ID++){
      if (MPI_CDM1_flag[ID]){
        MPI_Comm_free(&MPI_CommWD_CDM1[ID]);
      }
    }
  }

  free(stat_send2);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  free(MPI_CommWD_CDM1);
  free(MPI_CDM1_flag);

  free(fd_dist);

  for (i=0;i<knum_i; i++) {
    for (j=0;j<knum_j; j++) {
      free(k_op[i][j]);
    }
    free(k_op[i]);
  }
  free(k_op);

  free(T_KGrids1);
  free(T_KGrids2);
  free(T_KGrids3);
  free(T_k_op);

  for (i=0; i<2; i++){
    free(T_k_ID[i]);
  }
  free(T_k_ID);

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<T_knum; j++){
      free(EIGEN[i][j]);
    }
    free(EIGEN[i]);
  }
  free(EIGEN);

  if (Num_Comm_World1<=numprocs0){
    MPI_Comm_free(&MPI_CommWD1[myworld1]);
  }

  free(MPI_CommWD1);
  free(Comm_World_StartID1);
  free(NPROCS_WD1);
  free(Comm_World1);
  free(NPROCS_ID1);

  if (T_knum<=numprocs1){

    if (Num_Comm_World2<=numprocs1){
      MPI_Comm_free(&MPI_CommWD2[myworld2]);
    }

    free(MPI_CommWD2);
    free(Comm_World_StartID2);
    free(NPROCS_WD2);
    free(Comm_World2);
    free(NPROCS_ID2);
  }

  free(Ss);
  free(Hs);
  free(Cs);

  /* for PrintMemory and allocation */
  firsttime=0;

  /* for elapsed time */

  if (measure_time){
    printf("myid0=%2d time1 =%9.4f\n",myid0,time1);fflush(stdout);
    printf("myid0=%2d time2 =%9.4f\n",myid0,time2);fflush(stdout);
    printf("myid0=%2d time3 =%9.4f\n",myid0,time3);fflush(stdout);
    printf("myid0=%2d time4 =%9.4f\n",myid0,time4);fflush(stdout);
    printf("myid0=%2d time5 =%9.4f\n",myid0,time5);fflush(stdout);
    printf("myid0=%2d time51=%9.4f\n",myid0,time51);fflush(stdout);
    printf("myid0=%2d time6 =%9.4f\n",myid0,time6);fflush(stdout);
    printf("myid0=%2d time8 =%9.4f\n",myid0,time8);fflush(stdout);
    printf("myid0=%2d time81=%9.4f\n",myid0,time81);fflush(stdout);
    printf("myid0=%2d time9 =%9.4f\n",myid0,time9);fflush(stdout);
    printf("myid0=%2d time10=%9.4f\n",myid0,time10);fflush(stdout);
    printf("myid0=%2d time11=%9.4f\n",myid0,time11);fflush(stdout);
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}

