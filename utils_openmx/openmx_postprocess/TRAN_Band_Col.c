/**********************************************************************
  Band_DFT_Col.c:

     Band_DFT_Col.c is a subroutine to perform band calculations
     based on a collinear DFT

  Log of Band_DFT_Col.c:

     22/Nov/2001  Released by T.Ozaki

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



double TRAN_Band_Col(int SCF_iter,
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
		     double ***EIGEN,
		     double *H1,
		     double *S1,
		     double *CDM1,
		     double *EDM1,
		     dcomplex **H,
		     dcomplex **S,
		     dcomplex **C,
		     dcomplex *BLAS_S,
		     int ***k_op,
		     int *T_k_op,
		     int **T_k_ID,
		     double *T_KGrids1,
		     double *T_KGrids2,
		     double *T_KGrids3,
		     int myworld1,
		     int *NPROCS_ID1,
		     int *Comm_World1,
		     int *NPROCS_WD1,
		     int *Comm_World_StartID1,
		     MPI_Comm *MPI_CommWD1,
		     int myworld2,
		     int *NPROCS_ID2,
		     int *NPROCS_WD2,
		     int *Comm_World2,
		     int *Comm_World_StartID2,
		     MPI_Comm *MPI_CommWD2)
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
  double *VecFkw;
  double *VecFkwE;
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
  double My_Eele0[2];

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
  double time7,time8,time9;
  double time10,time11,time12;
  double time81,time82,time83;
  double time84,time85;
  double time51;

  dcomplex *BLAS_H;
  dcomplex *BLAS_C;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* for time */
  dtime(&TStime);

  time1 = 0.0;
  time2 = 0.0;
  time3 = 0.0;
  time4 = 0.0;
  time5 = 0.0;
  time6 = 0.0;
  time7 = 0.0;
  time8 = 0.0;
  time9 = 0.0;
  time10 = 0.0;
  time11 = 0.0;
  time12 = 0.0;
  time81 = 0.0;
  time82 = 0.0;
  time83 = 0.0;
  time84 = 0.0;
  time85 = 0.0;
  time51 = 0.0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);
  MPI_Barrier(mpi_comm_level1);

  Num_Comm_World1 = SpinP_switch + 1; 

  /*********************************************** 
       for pallalel calculations in myworld1
  ***********************************************/

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

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

  VecFkw  = (double*)malloc(sizeof(double)*(n+2));
  VecFkwE = (double*)malloc(sizeof(double)*(n+2));

  BLAS_H = (dcomplex*)malloc(sizeof(dcomplex)*n*n);
  BLAS_C = (dcomplex*)malloc(sizeof(dcomplex)*n*n);

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  MPI_CommWD_CDM1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*numprocs0);
  MPI_CDM1_flag = (int*)malloc(sizeof(int)*numprocs0);
  for (i=0; i<numprocs0; i++) MPI_CDM1_flag[i] = 0;

  if (firsttime) {
  PrintMemory("Band_DFT: ko", sizeof(double)*(n+1),NULL);
  PrintMemory("Band_DFT: koS",sizeof(double)*(n+1),NULL);
  PrintMemory("Band_DFT: H",  sizeof(dcomplex)*(n+1)*(n+1),NULL);
  PrintMemory("Band_DFT: S",  sizeof(dcomplex)*(n+1)*(n+1),NULL);
  PrintMemory("Band_DFT: C",  sizeof(dcomplex)*(n+1)*(n+1),NULL);
  }

  /***********************************************
              k-points by regular mesh 
  ***********************************************/

  if (way_of_kpoint==1){

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

    /***********************************
       one-dimentionalize for MPI
    ************************************/

    T_knum = 0;
    for (i=0; i<=(knum_i-1); i++){
      for (j=0; j<=(knum_j-1); j++){
	for (k=0; k<=(knum_k-1); k++){
	  if (0<k_op[i][j][k]){
	    T_knum++;
	  }
	}
      }
    }

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

      printf(" KGrids1: ");fflush(stdout);
      for (i=0;i<=knum_i-1;i++){
	if (knum_i==1)  k1 = 0.0;
	else            k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_i) + Shift_K_Point;
	printf("%9.5f ",k1);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids2: ");fflush(stdout);

      for (i=0;i<=knum_j-1;i++){
	if (knum_j==1)  k2 = 0.0;
	else            k2 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_j) - Shift_K_Point;
	printf("%9.5f ",k2);fflush(stdout);
      }
      printf("\n");fflush(stdout);

      printf(" KGrids3: ");fflush(stdout);
      for (i=0;i<=knum_k-1;i++){
	if (knum_k==1)  k3 = 0.0;
	else            k3 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)knum_k) + 2.0*Shift_K_Point;
	printf("%9.5f ",k3);fflush(stdout);
      }
      printf("\n");fflush(stdout);
    }

  }

  /***********************************************
                Monkhorst-Pack k-points 
  ***********************************************/

  else if (way_of_kpoint==2){

    T_knum = num_non_eq_kpt; 
   
    for (k=0; k<num_non_eq_kpt; k++){
      T_KGrids1[k] = NE_KGrids1[k];
      T_KGrids2[k] = NE_KGrids2[k];
      T_KGrids3[k] = NE_KGrids3[k];
      T_k_op[k]    = NE_T_k_op[k];
    }
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

    /**********************************************
       for MPI communication of CDM1 and EDM1        
    **********************************************/

    {
    MPI_Group new_group,old_group; 
    int *new_ranks; 

    new_ranks = (int*)malloc(sizeof(int)*T_knum);

    /* ID: zeroth world */

    for (ID=Comm_World_StartID1[myworld1]; ID<(numprocs0+Comm_World_StartID1[myworld1]); ID++){

      /* ID0: zeroth world */

      ID0 = ID % numprocs0;

      /* ID1: first world */

      ID1 = (ID-Comm_World_StartID1[myworld1]) % numprocs1;
         
      for (i=0; i<T_knum; i++){
        if (Comm_World_StartID2[i]<=ID1){
          ks = i;
        }
      }

      new_ranks[0] = ID1;
      if (myid1==ID1) MPI_CDM1_flag[ID0] = 1;

      for (i=(ks+1); i<(T_knum+ks); i++){
        i0 = i % T_knum;
        /* id in the first world */
        ID1 = Comm_World_StartID2[i0] + (ID0 % NPROCS_WD2[i0]);
        new_ranks[i-ks] = ID1;
        if (myid1==ID1) MPI_CDM1_flag[ID0] = 1;
      }

      MPI_Comm_group(MPI_CommWD1[myworld1], &old_group);

      /* define a new group */
      MPI_Group_incl(old_group,T_knum,new_ranks,&new_group);
      MPI_Comm_create(MPI_CommWD1[myworld1], new_group, &MPI_CommWD_CDM1[ID0]);
      MPI_Group_free(&new_group);
    }

    free(new_ranks); /* never forget cleaning! */
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

  if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){
    size_H1 = Get_OneD_HS_Col(1, CntOLP,   S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

diagonalize1:

  /* set H1 */

  if (SpinP_switch==0){ 
    size_H1 = Get_OneD_HS_Col(1, nh[0], H1,   MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
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

    if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){

      for (i1=0; i1<n*n; i1++) BLAS_S[i1] = Complex(0.0,0.0);

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  S[i1][j1] = Complex(0.0,0.0);
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

          if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){

            k -= tnoB; 

  	    for (j=0; j<tnoB; j++){

	      S[Anum+i][Bnum+j].r += S1[k]*co;
	      S[Anum+i][Bnum+j].i += S1[k]*si;

  	      k++;
	    }
	  }

	}
      }
    }

    /* for blas */

    for (i1=1; i1<=n; i1++){
      for (j1=1; j1<=n; j1++){
	BLAS_H[(j1-1)*n+i1-1] = H[i1][j1];
      } 
    } 

    /* diagonalize S */

    dtime(&Stime);

    if (parallel_mode==0){
      EigenBand_lapack(S,ko,n,n,1);
    }
    else if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){
      Eigen_PHH(MPI_CommWD2[myworld2],S,ko,n,n,1);
    }

    dtime(&Etime);
    time2 += Etime - Stime;

    if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){

      if (3<=level_stdout){
	printf(" myid0=%2d spin=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,spin,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,ko[i1]);
	}
      }
 
      /* minus eigenvalues to 1.0e-14 */

      for (l=1; l<=n; l++){
	if (ko[l]<0.0) ko[l] = 1.0e-14;
	koS[l] = ko[l];
      }

      /* calculate S*1/sqrt(ko) */

      for (l=1; l<=n; l++) ko[l] = 1.0/sqrt(ko[l]);

      /* S * 1.0/sqrt(ko[l]) */

#pragma omp parallel shared(BLAS_S,ko,S,n) private(OMPID,Nthrds,Nprocs,i1,j1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (i1=1+OMPID; i1<=n; i1+=Nthrds){
	  for (j1=1; j1<=n; j1++){

	    S[i1][j1].r = S[i1][j1].r*ko[j1];
	    S[i1][j1].i = S[i1][j1].i*ko[j1];

	    BLAS_S[(j1-1)*n+i1-1] = S[i1][j1];
	  } 
	} 
      } /* #pragma omp parallel */
    }

    /****************************************************
     1.0/sqrt(ko[l]) * U^t * H * U * 1.0/sqrt(ko[l])
    ****************************************************/

    dtime(&Stime);

    /* first transposition of S */

    /*
    for (i1=1; i1<=n; i1++){
      for (j1=i1+1; j1<=n; j1++){
	Ctmp1 = S[i1][j1];
	Ctmp2 = S[j1][i1];
	S[i1][j1] = Ctmp2;
	S[j1][i1] = Ctmp1;
      }
    }
    */

    /**********************************
      for parallel in the second world
    **********************************/

    if (all_knum==1){

      /* H * U * 1.0/sqrt(ko[l]) */
      /* C is distributed by row in each processor */

      /*
      for (j1=is1[myid2]; j1<=ie1[myid2]; j1++){
	for (i1=1; i1<=n; i1++){

	  sum  = 0.0;
	  sumi = 0.0;

	  for (l=1; l<=n; l++){
	    sum  += H[i1][l].r*S[j1][l].r - H[i1][l].i*S[j1][l].i;
	    sumi += H[i1][l].r*S[j1][l].i + H[i1][l].i*S[j1][l].r;
	  }

	  C[j1][i1].r = sum;
	  C[j1][i1].i = sumi;

	}
      } 
      */    

      /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */


    /*
    printf("ABC3 myid0=%2d myid1=%2d myid2=%2d numprocs2=%2d\n",myid0,myid1,myid2,numprocs2);
    MPI_Finalize();
    exit(0);
    */

#pragma omp parallel shared(myid2,ie1,is1,BLAS_S,BLAS_H,BLAS_C,n) private(OMPID,Nthrds,Nprocs,Ctmp1,Ctmp2,BM,BN,BK,ns,ne)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

        ns = is1[myid2] + OMPID*(ie1[myid2]-is1[myid2]+1)/Nthrds;
        ne = is1[myid2] + (OMPID+1)*(ie1[myid2]-is1[myid2]+1)/Nthrds - 1;

	BM = n;
	BN = ne - ns + 1;
	BK = n;

	Ctmp1.r = 1.0;
	Ctmp1.i = 0.0;
	Ctmp2.r = 0.0;
	Ctmp2.i = 0.0;

        if (0<BN){

	  F77_NAME(zgemm,ZGEMM)("N","N", &BM,&BN,&BK, 
		  	        &Ctmp1, 
			        BLAS_H, &BM,
			        &BLAS_S[(ns-1)*n], &BK,
			        &Ctmp2, 
			        &BLAS_C[(ns-1)*n], &BM);
	}

      } /* #pragma omp parallel */

      /*
      BM = n;
      BN = ie1[myid2] - is1[myid2] + 1;  
      BK = n;

      Ctmp1.r = 1.0;
      Ctmp1.i = 0.0;
      Ctmp2.r = 0.0;
      Ctmp2.i = 0.0;

      F77_NAME(zgemm,ZGEMM)("N","N", &BM,&BN,&BK, 
                            &Ctmp1, 
                            BLAS_H, &BM,
			    &BLAS_S[(is1[myid2]-1)*n], &BK,
                            &Ctmp2, 
			    &BLAS_C[(is1[myid2]-1)*n], &BM);
      */


      /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */
      /* H is distributed by row in each processor */

      /*
      for (j1=is1[myid2]; j1<=ie1[myid2]; j1++){
        for (i1=1; i1<=n; i1++){

	  sum  = 0.0;
	  sumi = 0.0;

	  for (l=1; l<=n; l++){
	    sum  +=  S[i1][l].r*C[j1][l].r + S[i1][l].i*C[j1][l].i;
	    sumi +=  S[i1][l].r*C[j1][l].i - S[i1][l].i*C[j1][l].r;
	  }

	  H[j1][i1].r = sum;
	  H[j1][i1].i = sumi;

	}
      } 
      */

      /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

#pragma omp parallel shared(H,myid2,ie1,is1,BLAS_S,BLAS_H,BLAS_C,n) private(OMPID,Nthrds,Nprocs,Ctmp1,Ctmp2,BM,BN,BK,ns,ne,i1,j1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

        ns = is1[myid2] + OMPID*(ie1[myid2]-is1[myid2]+1)/Nthrds;
        ne = is1[myid2] + (OMPID+1)*(ie1[myid2]-is1[myid2]+1)/Nthrds - 1;

	BM = n;
	BN = ne - ns + 1;
	BK = n;

	Ctmp1.r = 1.0;
	Ctmp1.i = 0.0;
	Ctmp2.r = 0.0;
	Ctmp2.i = 0.0;

        if (0<BN){
 	  F77_NAME(zgemm,ZGEMM)("C","N", &BM,&BN,&BK, 
		  	        &Ctmp1,
			        BLAS_S, &BM,
			        &BLAS_C[(ns-1)*n], &BK, 
			        &Ctmp2, 
			        &BLAS_H[(ns-1)*n], &BM);
	}

	for (j1=ns; j1<=ne; j1++){
	  for (i1=1; i1<=n; i1++){
	    H[j1][i1] = BLAS_H[(j1-1)*n+i1-1];            
	  }
	}

      } /* #pragma omp parallel */

      /* broadcast H */

      BroadCast_ComplexMatrix(MPI_CommWD2[myworld2],H,n,is1,ie1,myid2,numprocs2,
                              stat_send,request_send,request_recv);
    }

    else{

      /* H * U * 1.0/sqrt(ko[l]) */

      /*
      for (j1=1; j1<=n; j1++){
	for (i1=1; i1<=n; i1++){

	  sum  = 0.0;
	  sumi = 0.0;

	  for (l=1; l<=n; l++){
	    sum  += H[i1][l].r*S[j1][l].r - H[i1][l].i*S[j1][l].i;
	    sumi += H[i1][l].r*S[j1][l].i + H[i1][l].i*S[j1][l].r;
	  }

	  C[j1][i1].r = sum;
	  C[j1][i1].i = sumi;

	}
      }
      */

      /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

#pragma omp parallel shared(BLAS_S,BLAS_H,BLAS_C,n) private(OMPID,Nthrds,Nprocs,Ctmp1,Ctmp2,BM,BN,BK)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	BM = n;
	BN = (OMPID+1)*n/Nthrds - (OMPID*n/Nthrds+1) + 1;
	BK = n;

	Ctmp1.r = 1.0;
	Ctmp1.i = 0.0;
	Ctmp2.r = 0.0;
	Ctmp2.i = 0.0;

        if (0<BN){
	  F77_NAME(zgemm,ZGEMM)("N","N", &BM,&BN,&BK, 
		 	        &Ctmp1, 
			        BLAS_H, &BM,
			        &BLAS_S[(OMPID*n/Nthrds)*n], &BK,
			        &Ctmp2, 
			        &BLAS_C[(OMPID*n/Nthrds)*n], &BM);
	}

      } /* #pragma omp parallel */

      /* 1.0/sqrt(ko[l]) * U^+ H * U * 1.0/sqrt(ko[l]) */

      /*
      for (j1=1; j1<=n; j1++){
	for (i1=1; i1<=n; i1++){

	  sum  = 0.0;
	  sumi = 0.0;

	  for (l=1; l<=n; l++){
	    sum  += S[i1][l].r*C[j1][l].r + S[i1][l].i*C[j1][l].i;
	    sumi += S[i1][l].r*C[j1][l].i - S[i1][l].i*C[j1][l].r;
	  }

	  H[j1][i1].r = sum;
	  H[j1][i1].i = sumi;
	}
      } 
      */

      /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

#pragma omp parallel shared(H,BLAS_S,BLAS_H,BLAS_C,n) private(OMPID,Nthrds,Nprocs,Ctmp1,Ctmp2,BM,BN,BK,i1,j1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	BM = n;
	BN = (OMPID+1)*n/Nthrds - (OMPID*n/Nthrds+1) + 1;
	BK = n;

	Ctmp1.r = 1.0;
	Ctmp1.i = 0.0;
	Ctmp2.r = 0.0;
	Ctmp2.i = 0.0;

        if (0<BN){
	  F77_NAME(zgemm,ZGEMM)("C","N", &BM,&BN,&BK,
			        &Ctmp1, 
			        BLAS_S, &BM,
			        &BLAS_C[(OMPID*n/Nthrds)*n], &BK, 
			        &Ctmp2, 
			        &BLAS_H[(OMPID*n/Nthrds)*n], &BM);
	}

	for (j1=(OMPID*n/Nthrds+1); j1<=(OMPID+1)*n/Nthrds; j1++){
	  for (i1=1; i1<=n; i1++){
	    H[j1][i1] = BLAS_H[(j1-1)*n+i1-1];            
	  }
	}

      } /* #pragma omp parallel */

    } /* else */

    /* H to C (transposition) */

    for (i1=1; i1<=n; i1++){
      for (j1=1; j1<=n; j1++){
	C[j1][i1] = H[i1][j1];
      }
    }

    /* penalty for ill-conditioning states */

    EV_cut0 = Threshold_OLP_Eigen;

    for (i1=1; i1<=n; i1++){

      if (koS[i1]<EV_cut0){
	C[i1][i1].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
      }
 
      /* cutoff the interaction between the ill-conditioned state */
 
      if (1.0e+3<C[i1][i1].r){
	for (j1=1; j1<=n; j1++){
	  C[i1][j1] = Complex(0.0,0.0);
	  C[j1][i1] = Complex(0.0,0.0);
	}
	C[i1][i1].r = 1.0e+4;
      }
    }

    dtime(&Etime);
    time3 += Etime - Stime;

    /* diagonalize H' */

    dtime(&Stime);

    if (parallel_mode==0){
      EigenBand_lapack(C,ko,n,MaxN,all_knum);
    }
    else{
      /*  The output C matrix is distributed by column. */
      Eigen_PHH(MPI_CommWD2[myworld2],C,ko,n,MaxN,0);
    }

    dtime(&Etime);
    time4 += Etime - Stime;

    for (l=1; l<=MaxN; l++){
      EIGEN[spin][kloop][l] = ko[l];
    }

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

    /* calculation of wave functions */

    dtime(&Stime);

    if (all_knum==1){

      /*  The H matrix is distributed by row */

      /*
      for (i1=1; i1<=n; i1++){
	for (j1=is2[myid2]; j1<=ie2[myid2]; j1++){
	  H[j1][i1] = C[i1][j1];
	}
      }
      */

      for (i1=1; i1<=n; i1++){
	for (j1=is2[myid2]; j1<=ie2[myid2]; j1++){
          BLAS_H[(j1-1)*n+i1-1] = C[i1][j1];
	}
      }

      /* the second transposition of S */

      /*
      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }
      */

      /* C is distributed by row in each processor */

      /*
      for (j1=is2[myid2]; j1<=ie2[myid2]; j1++){
        for (i1=1; i1<=n; i1++){

	  sum  = 0.0;
	  sumi = 0.0;
	  for (l=1; l<=n; l++){
	    sum  += S[i1][l].r*H[j1][l].r - S[i1][l].i*H[j1][l].i;
	    sumi += S[i1][l].r*H[j1][l].i + S[i1][l].i*H[j1][l].r;
	  }

	  C[j1][i1].r = sum;
	  C[j1][i1].i = sumi;
	}
      }
      */

      /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

#pragma omp parallel shared(C,myid2,ie2,is2,BLAS_S,BLAS_H,BLAS_C,n) private(OMPID,Nthrds,Nprocs,Ctmp1,Ctmp2,BM,BN,BK,ns,ne,i1,j1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

        ns = is2[myid2] + OMPID*(ie2[myid2]-is2[myid2]+1)/Nthrds;
        ne = is2[myid2] + (OMPID+1)*(ie2[myid2]-is2[myid2]+1)/Nthrds - 1;

	BM = n;
	BN = ne - ns + 1;
	BK = n;

	Ctmp1.r = 1.0;
	Ctmp1.i = 0.0;
	Ctmp2.r = 0.0;
	Ctmp2.i = 0.0;

        if (0<BN){
	  F77_NAME(zgemm,ZGEMM)("N","N", &BM,&BN,&BK, 
			        &Ctmp1, 
			        BLAS_S, &BM,
			        &BLAS_H[(ns-1)*n], &BK,
			        &Ctmp2, 
			        &BLAS_C[(ns-1)*n], &BM);
	}

	for (j1=ns; j1<=ne; j1++){
	  for (i1=1; i1<=n; i1++){
	    C[j1][i1] = BLAS_C[(j1-1)*n+i1-1];            
	  }
	}

      } /* #pragma omp parallel */

      /* broadcast C:
       C is distributed by row in each processor
      */

      dtime(&Stime0);
  
      BroadCast_ComplexMatrix(MPI_CommWD2[myworld2],C,n,is2,ie2,myid2,numprocs2,
			      stat_send,request_send,request_recv);

      /* C to H (transposition)
         H consists of column vectors
      */ 

      for (i1=1; i1<=MaxN; i1++){
        for (j1=1; j1<=n; j1++){
	  H[j1][i1] = C[i1][j1];
	}
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

  /****************************************************
           band energy in a finite temperature
  ****************************************************/

  Eele0[0] = 0.0;
  Eele0[1] = 0.0;

  for (kloop=0; kloop<T_knum; kloop++){
    for (spin=0; spin<=SpinP_switch; spin++){
      for (l=1; l<=MaxN; l++){

	x = (EIGEN[spin][kloop][l] - ChemP)*Beta;

	if (x<=-x_cut)      FermiF = 1.0;
	else if (x_cut<=x)  FermiF = 0.0;
	else                FermiF = 1.0/(1.0 + exp(x));

	Eele0[spin] += FermiF*EIGEN[spin][kloop][l]*(double)T_k_op[kloop];

      }
    }
  } 

  if (SpinP_switch==0){
    Eele0[0] = Eele0[0]/sum_weights;
    Eele0[1] = Eele0[0];
  }
  else {
    Eele0[0] = Eele0[0]/sum_weights;
    Eele0[1] = Eele0[1]/sum_weights;
  }

  Uele = Eele0[0] + Eele0[1];

  if (2<=level_stdout){
    printf("myid0=%2d ChemP=%lf, Eele0[0]=%lf, Eele0[1]=%lf\n",myid0,ChemP,Eele0[0],Eele0[1]);
  }

  dtime(&Etime);
  time7 += Etime - Stime;

  /****************************************************
         if all_knum==1, calculate CDM and EDM
  ****************************************************/

  dtime(&Stime);

  if (all_knum==1){

    dtime(&Stime0);

    /* initialize CDM1 and EDM1 */

    for (i=0; i<size_H1; i++){
      CDM1[i] = 0.0;
      EDM1[i] = 0.0;
    }

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

      VecFkw[l]  = FermiF*kw; 
      VecFkwE[l] = VecFkw[l]*eig;

      if ( FermiF<=FermiEps && po==0 ) {
	lmax = l;  
	po = 1;
      }

      l++;

    } while(po==0 && l<=MaxN);

    if (po==0) lmax = MaxN;

    dtime(&Etime0);
    time81 += Etime0 - Stime0;

    dtime(&Stime0);

    /* predetermination of k 
       H is used as temporal array 
    */ 

    k = 0;
    for (AN=1; AN<=atomnum; AN++){

      GA_AN = order_GA[AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      H[0][AN].r = (double)k;

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	k += tnoA*tnoB;
      }
    }

    /* DM and EDM */

#pragma omp parallel shared(numprocs0,My_NZeros,SP_NZeros,MPI_CDM1_flag,EDM1,CDM1,VecFkwE,VecFkw,H,lmax,k1,k2,k3,atv_ijk,ncn,natn,FNAN,MP,Spe_Total_CNO,WhatSpecies,order_GA,atomnum) private(OMPID,Nthrds,Nprocs,AN,GA_AN,wanA,tnoA,Anum,k,LB_AN,GB_AN,Rn,wanB,tnoB,Bnum,l1,l2,l3,kRn,si,co,i,ia,j,jb,d1,d2,l,tmp,po,ID)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (AN=1+OMPID; AN<=atomnum; AN+=Nthrds){

	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	k = (int)H[0][AN].r;

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

	    ia = Anum + i; 

	    for (j=0; j<tnoB; j++){

	      jb = Bnum + j;

	      /***************************************************************
               Note that the imagiary part is zero, 
               since

               at k 
               A = (co + i si)(Re + i Im) = (co*Re - si*Im) + i (co*Im + si*Re) 
               at -k
               B = (co - i si)(Re - i Im) = (co*Re - si*Im) - i (co*Im + si*Re) 
               Thus, Re(A+B) = 2*(co*Re - si*Im)
                     Im(A+B) = 0
	      ***************************************************************/

	      po = 0;
	      ID = 0;
	      do { 
		if (MPI_CDM1_flag[ID] && SP_NZeros[ID]<=k && k<(SP_NZeros[ID]+My_NZeros[ID])) po = 1;
		ID++;
	      } while (po==0 && ID<numprocs0);

	      if (po==1){

		d1 = 0.0;
		d2 = 0.0;
	
		for (l=1; l<=lmax; l++){

		  tmp = co*(H[ia][l].r*H[jb][l].r + H[ia][l].i*H[jb][l].i)
		       -si*(H[ia][l].r*H[jb][l].i - H[ia][l].i*H[jb][l].r);

		  d1 += VecFkw[l]*tmp;
		  d2 += VecFkwE[l]*tmp;;
		}

		CDM1[k] += d1;
		EDM1[k] += d2; 
	      }

	      /* increment of k */

	      k++;

	    }
	  }

	} /* LB_AN */
      } /* AN */

    } /* #pragma omp parallel */

    /* sum of CDM1 and EDM1 by Allreduce in MPI */

    dtime(&Etime0);
    time82 += Etime0 - Stime0;

    dtime(&Stime0);

    for (ID=0; ID<numprocs0; ID++){
      if (MPI_CDM1_flag[ID]){
        k = SP_NZeros[ID];
        MPI_Reduce(&CDM1[k], &H1[k], My_NZeros[ID], MPI_DOUBLE, MPI_SUM, 0, MPI_CommWD_CDM1[ID]);
        MPI_Reduce(&EDM1[k], &S1[k], My_NZeros[ID], MPI_DOUBLE, MPI_SUM, 0, MPI_CommWD_CDM1[ID]);
      }
    }

    dtime(&Etime0);
    time83 += Etime0 - Stime0;

    dtime(&Stime0);

    /* CDM and EDM */

    k = 0;
    for (AN=1; AN<=atomnum; AN++){
      GA_AN = order_GA[AN];
      MA_AN = F_G2M[GA_AN]; 

      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

	    if (1<=MA_AN && MA_AN<=Matomnum){   

	      CDM[spin][MA_AN][LB_AN][i][j] = H1[k];
              EDM[spin][MA_AN][LB_AN][i][j] = S1[k];
	    }

	    k++;

	  }
	}
      }
    }

    dtime(&Etime0);
    time84 += Etime0 - Stime0;

    dtime(&Stime0);

    /* if necessary, MPI communication of CDM and EDM */

    if (1<numprocs0 && SpinP_switch==1){

      /* set spin */

      if (myworld1==0){
	spin = 1;
      }
      else{
	spin = 0;
      } 

      /* communicate CDM1 and EDM1 */

      for (i=0; i<=1; i++){

        for (ID=Comm_World_StartID1[i]; ID<(numprocs0+Comm_World_StartID1[i]); ID++){

          /* ID0: zeroth world */

          ID0 = ID % numprocs0;

          /* ID's for sending and receiving in the zeroth world */

          IDS = Comm_World_StartID1[i] + (ID-Comm_World_StartID1[i]) % NPROCS_WD1[i];
          IDR = ID0;

          k = SP_NZeros[IDR];

          if (IDS!=IDR){

	    /* CDM1 */

	    if (myid0==IDS){
	      MPI_Isend(&H1[k], My_NZeros[IDR], MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
	    } 

	    if (myid0==IDR){
	      MPI_Recv(&CDM1[k], My_NZeros[IDR], MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
	    }

	    if (myid0==IDS){
	      MPI_Wait(&request,&stat);
	    }

	    /* EDM1 */

	    if (myid0==IDS){
	      MPI_Isend(&S1[k], My_NZeros[IDR], MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
	    } 

	    if (myid0==IDR){
	      MPI_Recv(&EDM1[k], My_NZeros[IDR], MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
	    }

	    if (myid0==IDS){
	      MPI_Wait(&request,&stat);
	    }

	  }
        }
      }

      /* put CDM1 and EDM1 into CDM and EDM */

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	MA_AN = F_G2M[GA_AN]; 

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      if (1<=MA_AN && MA_AN<=Matomnum){   
		CDM[spin][MA_AN][LB_AN][i][j] = CDM1[k];
                EDM[spin][MA_AN][LB_AN][i][j] = EDM1[k];
	      }

	      k++;

	    }
	  }
	}
      }
    }

    dtime(&Etime0);
    time85 += Etime0 - Stime0;

  } /* if (all_knum==1) */

  dtime(&Etime);
  time8 += Etime - Stime;

  if (myid0==Host_ID && 0<level_stdout){
    printf("<Band_DFT>  Eigen, time=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
  }

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
      size_H1 = Get_OneD_HS_Col(1, nh[0], H1,   MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
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

    /* initialize CDM1 and EDM1 */

    for (i=0; i<size_H1; i++){
      CDM1[i] = 0.0;
      EDM1[i] = 0.0;
    }

    /* initialize CDM, EDM, and iDM */

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
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
	    CDM[spin][MA_AN][LB_AN][i][j] = 0.0;
            EDM[spin][MA_AN][LB_AN][i][j] = 0.0;
	    iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
	    iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
	  }
	}
      }
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
	  S[i1][j1] = Complex(0.0,0.0);
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

	      S[Anum+i][Bnum+j].r += S1[k]*co;
	      S[Anum+i][Bnum+j].i += S1[k]*si;
	      H[Anum+i][Bnum+j].r += H1[k]*co;
	      H[Anum+i][Bnum+j].i += H1[k]*si;

	      k++;

	    }
	  }
	}
      }

      /* for blas */

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  BLAS_H[(j1-1)*n+i1-1] = H[i1][j1];
	} 
      } 

      /* diagonalize S */

      dtime(&Stime);

      if (parallel_mode==0){
	EigenBand_lapack(S,ko,n,n,1);
      }
      else{
	Eigen_PHH(MPI_CommWD2[myworld2],S,ko,n,n,1);
      }

      dtime(&Etime);
      time9 += Etime - Stime;

      if (3<=level_stdout){
	printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,ko[i1]);
	}
      }

      /* minus eigenvalues to 1.0e-14 */

      for (l=1; l<=n; l++){
	if (ko[l]<0.0) ko[l] = 1.0e-14;
	koS[l] = ko[l];
      }

      /* calculate S*1/sqrt(ko) */

      for (l=1; l<=n; l++) ko[l] = 1.0/sqrt(ko[l]);

      /* S * 1.0/sqrt(ko[l])  */

#pragma omp parallel shared(BLAS_S,ko,S,n) private(OMPID,Nthrds,Nprocs,i1,j1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (i1=1+OMPID; i1<=n; i1+=Nthrds){
	  for (j1=1; j1<=n; j1++){

	    S[i1][j1].r = S[i1][j1].r*ko[j1];
	    S[i1][j1].i = S[i1][j1].i*ko[j1];
	    BLAS_S[(j1-1)*n+i1-1] = S[i1][j1];
	  } 
	} 

      } /* #pragma omp parallel */

      /****************************************************
            1/sqrt(ko) * U^t * H * U * 1/sqrt(ko)
      ****************************************************/

      /* transpose S */

      /*
      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }
      */

      /* H * U * 1/sqrt(ko) */

      /*
      for (j1=1; j1<=n; j1++){
	for (i1=1; i1<=n; i1++){

	  sum  = 0.0;
	  sumi = 0.0;

	  for (l=1; l<=n; l++){
	    sum  += H[i1][l].r*S[j1][l].r - H[i1][l].i*S[j1][l].i;
	    sumi += H[i1][l].r*S[j1][l].i + H[i1][l].i*S[j1][l].r;
	  }

	  C[j1][i1].r = sum;
	  C[j1][i1].i = sumi;
	}
      } 
      */

      /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

#pragma omp parallel shared(BLAS_S,BLAS_H,BLAS_C,n) private(OMPID,Nthrds,Nprocs,Ctmp1,Ctmp2,BM,BN,BK)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	BM = n;
	BN = (OMPID+1)*n/Nthrds - (OMPID*n/Nthrds+1) + 1;
	BK = n;

	Ctmp1.r = 1.0;
	Ctmp1.i = 0.0;
	Ctmp2.r = 0.0;
	Ctmp2.i = 0.0;

        if (0<BN){
	  F77_NAME(zgemm,ZGEMM)("N","N", &BM,&BN,&BK, 
		  	        &Ctmp1, 
			        BLAS_H, &BM,
			        &BLAS_S[(OMPID*n/Nthrds)*n], &BK,
			        &Ctmp2, 
			        &BLAS_C[(OMPID*n/Nthrds)*n], &BM);
	}

      } /* #pragma omp parallel */

      /* 1/sqrt(ko) * U^+ H * U * 1/sqrt(ko) */

      /*
      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  sum  = 0.0;
	  sumi = 0.0;
	  for (l=1; l<=n; l++){
	    sum  +=  S[i1][l].r*C[j1][l].r + S[i1][l].i*C[j1][l].i;
	    sumi +=  S[i1][l].r*C[j1][l].i - S[i1][l].i*C[j1][l].r;
	  }
	  H[i1][j1].r = sum;
	  H[i1][j1].i = sumi;
	}
      } 
      */

      /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

#pragma omp parallel shared(C,BLAS_S,BLAS_H,BLAS_C,n) private(OMPID,Nthrds,Nprocs,Ctmp1,Ctmp2,BM,BN,BK,i1,j1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	BM = n;
	BN = (OMPID+1)*n/Nthrds - (OMPID*n/Nthrds+1) + 1;
	BK = n;

	Ctmp1.r = 1.0;
	Ctmp1.i = 0.0;
	Ctmp2.r = 0.0;
	Ctmp2.i = 0.0;

        if (0<BN){
	  F77_NAME(zgemm,ZGEMM)("C","N", &BM,&BN,&BK, 
			        &Ctmp1,
			        BLAS_S, &BM,
			        &BLAS_C[(OMPID*n/Nthrds)*n], &BK, 
			        &Ctmp2, 
			        &BLAS_H[(OMPID*n/Nthrds)*n], &BM);
	}

	for (j1=(OMPID*n/Nthrds+1); j1<=(OMPID+1)*n/Nthrds; j1++){
	  for (i1=1; i1<=n; i1++){
	    C[i1][j1] = BLAS_H[(j1-1)*n+i1-1];            
	  }
	}

      } /* #pragma omp parallel */

      /* H to C */

      /*
      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  C[i1][j1] = H[i1][j1];
	}
      }
      */

      /* penalty for ill-conditioning states */

      EV_cut0 = Threshold_OLP_Eigen;

      for (i1=1; i1<=n; i1++){

	if (koS[i1]<EV_cut0){
	  C[i1][i1].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
	}
 
	/* cutoff the interaction between the ill-conditioned state */
 
	if (1.0e+3<C[i1][i1].r){
	  for (j1=1; j1<=n; j1++){
	    C[i1][j1] = Complex(0.0,0.0);
	    C[j1][i1] = Complex(0.0,0.0);
	  }
	  C[i1][i1].r = 1.0e+4;
	}
      }

      /* diagonalize H' */

      dtime(&Stime);

      if (parallel_mode==0){
	EigenBand_lapack(C,ko,n,MaxN,1);
      }
      else{
	/*  The C matrix is distributed by row */
	Eigen_PHH(MPI_CommWD2[myworld2],C,ko,n,MaxN,1);
      }

      dtime(&Etime);
      time10 += Etime - Stime;

      if (3<=level_stdout && 0<=kloop){
	printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of Kohn-Sham(DM) spin=%2d i1=%2d %15.12f\n",
		 spin,i1,ko[i1]);
	}
      }

      /****************************************************
        transformation to the original eigenvectors.
	     NOTE JRCAT-244p and JAIST-2122p 
      ****************************************************/

      /* transpose */

      /*
      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }
      */

      /* transpose */

      /*
      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = C[i1][j1];
	  Ctmp2 = C[j1][i1];
	  C[i1][j1] = Ctmp2;
	  C[j1][i1] = Ctmp1;
	}
      }
      */

      /*
      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=MaxN; j1++){

	  sum  = 0.0;
	  sumi = 0.0;
	  for (l=1; l<=n; l++){
	    sum  += S[i1][l].r*C[j1][l].r - S[i1][l].i*C[j1][l].i;
	    sumi += S[i1][l].r*C[j1][l].i + S[i1][l].i*C[j1][l].r;
	  }
	  H[i1][j1].r = sum;
	  H[i1][j1].i = sumi;
	}
      }
      */

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
          BLAS_H[(j1-1)*n+i1-1] = C[i1][j1];
	}
      }

      /* note for BLAS, A[M*K] * B[K*N] = C[M*N] */

#pragma omp parallel shared(H,BLAS_S,BLAS_H,BLAS_C,n,MaxN) private(OMPID,Nthrds,Nprocs,j1,i1,Ctmp1,Ctmp2,BM,BN,BK)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	BM = n;
	BN = (OMPID+1)*MaxN/Nthrds - (OMPID*MaxN/Nthrds+1) + 1;
	BK = n;

	Ctmp1.r = 1.0;
	Ctmp1.i = 0.0;
	Ctmp2.r = 0.0;
	Ctmp2.i = 0.0;

        if (0<BN){
	  F77_NAME(zgemm,ZGEMM)("N","N", &BM,&BN,&BK, 
			        &Ctmp1, 
			        BLAS_S, &BM,
			        &BLAS_H[(OMPID*MaxN/Nthrds)*n], &BK,
			        &Ctmp2, 
			        &BLAS_C[(OMPID*MaxN/Nthrds)*n], &BM);
	}

	for (j1=(OMPID*MaxN/Nthrds+1); j1<=(OMPID+1)*MaxN/Nthrds; j1++){
	  for (i1=1; i1<=n; i1++){
	    H[i1][j1] = BLAS_C[(j1-1)*n+i1-1];            
	  }
	}

      } /* #pragma omp parallel */

      /****************************************************
                   calculate DM and EDM
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

	VecFkw[l]  = FermiF*kw; 
	VecFkwE[l] = VecFkw[l]*eig;

	if ( FermiF<=FermiEps && po==0 ) {
	  lmax = l;  
	  po = 1;
	}

	l++;

      } while(po==0 && l<=MaxN);

      if (po==0) lmax = MaxN;

      /* predetermination of k 
         H is used as temporal array 
      */ 

      k = 0;
      for (AN=1; AN<=atomnum; AN++){

	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];

        H[0][AN].r = (double)k;

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
          k += tnoA*tnoB;
	}
      }

      /* DM and EDM */

#pragma omp parallel shared(EDM1,CDM1,VecFkwE,VecFkw,H,lmax,k1,k2,k3,atv_ijk,ncn,natn,FNAN,MP,Spe_Total_CNO,WhatSpecies,order_GA,atomnum) private(OMPID,Nthrds,Nprocs,AN,GA_AN,wanA,tnoA,Anum,k,LB_AN,GB_AN,Rn,wanB,tnoB,Bnum,l1,l2,l3,kRn,si,co,i,ia,j,jb,d1,d2,l,tmp)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (AN=1+OMPID; AN<=atomnum; AN+=Nthrds){

	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];

	  k = (int)H[0][AN].r;

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

	      ia = Anum + i; 

	      for (j=0; j<tnoB; j++){

		jb = Bnum + j;

		/***************************************************************
                Note that the imagiary part is zero, 
                 since

                 at k 
                 A = (co + i si)(Re + i Im) = (co*Re - si*Im) + i (co*Im + si*Re) 
                 at -k
                 B = (co - i si)(Re - i Im) = (co*Re - si*Im) - i (co*Im + si*Re) 
                 Thus, Re(A+B) = 2*(co*Re - si*Im)
                       Im(A+B) = 0
		***************************************************************/

		d1 = 0.0;
		d2 = 0.0;

		for (l=1; l<=lmax; l++){

		  tmp = co*(H[ia][l].r*H[jb][l].r + H[ia][l].i*H[jb][l].i)
		       -si*(H[ia][l].r*H[jb][l].i - H[ia][l].i*H[jb][l].r);

		  d1 += VecFkw[l]*tmp;
		  d2 += VecFkwE[l]*tmp;;

		}

		CDM1[k] += d1; 
		EDM1[k] += d2; 

		/* increment of k */

		k++;

	      }
	    }
	  }
	}

      } /* #pragma omp parallel */

      dtime(&Etime);
      time11 += Etime - Stime;

    } /* kloop0 */

    /*******************************************************
         sum of CDM1 and EDM1 by Allreduce in MPI
    *******************************************************/

    dtime(&Stime);

    if (parallel_mode){
      tmp = 1.0/(double)numprocs2;
      for (i=0; i<size_H1; i++){
	CDM1[i] *= tmp;
	EDM1[i] *= tmp;
      }
    }

    MPI_Allreduce(&CDM1[0], &H1[0], size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
    MPI_Allreduce(&EDM1[0], &S1[0], size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);

    /* CDM and EDM */

    k = 0;
    for (AN=1; AN<=atomnum; AN++){
      GA_AN = order_GA[AN];
      MA_AN = F_G2M[GA_AN]; 

      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

	    if (1<=MA_AN && MA_AN<=Matomnum){   

	      CDM[spin][MA_AN][LB_AN][i][j] = H1[k];
              EDM[spin][MA_AN][LB_AN][i][j] = S1[k];
	    }

	    k++;

	  }
	}
      }
    }

    dtime(&Etime);
    time12 += Etime - Stime;

    if (SpinP_switch==1 && numprocs0==1 && spin==0){
      spin++;  
      goto diagonalize2; 
    }

    /* if necessary, MPI communication of CDM and EDM */

    if (1<numprocs0 && SpinP_switch==1){

      /* set spin */

      if (myworld1==0){
	spin = 1;
      }
      else{
	spin = 0;
      } 

      /* communicate CDM1 and EDM1 */

      for (i=0; i<=1; i++){
    
	IDS = Comm_World_StartID1[i%2];
	IDR = Comm_World_StartID1[(i+1)%2];

	if (myid0==IDS){
	  MPI_Isend(&H1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
	}

	if (myid0==IDR){
	  MPI_Recv(&CDM1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
	}

	if (myid0==IDS){
	  MPI_Wait(&request,&stat);
	}

	if (myid0==IDS){
	  MPI_Isend(&S1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
	}

	if (myid0==IDR){
	  MPI_Recv(&EDM1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
	}

	if (myid0==IDS){
	  MPI_Wait(&request,&stat);
	}
      }

      MPI_Bcast(&CDM1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
      MPI_Bcast(&EDM1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);

      /* put CDM1 and EDM1 into CDM and EDM */

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	MA_AN = F_G2M[GA_AN]; 

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      if (1<=MA_AN && MA_AN<=Matomnum){   
		CDM[spin][MA_AN][LB_AN][i][j] = CDM1[k];
   	        EDM[spin][MA_AN][LB_AN][i][j] = EDM1[k];
	      }

	      k++;

	    }
	  }
	}
      }
    }

  } /* if (all_knum!=1) */

  /****************************************************
           normalization of CDM, EDM, and iDM 
  ****************************************************/

  dtime(&EiloopTime);

  if (myid0==Host_ID && 0<level_stdout){
    printf("<Band_DFT>  DM, time=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
  }

  dum = 1.0/sum_weights;

  for (spin=0; spin<=SpinP_switch; spin++){
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
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
	    CDM[spin][MA_AN][LB_AN][i][j]    = CDM[spin][MA_AN][LB_AN][i][j]*dum;
            EDM[spin][MA_AN][LB_AN][i][j]    = EDM[spin][MA_AN][LB_AN][i][j]*dum;
	    iDM[0][spin][MA_AN][LB_AN][i][j] = iDM[0][spin][MA_AN][LB_AN][i][j]*dum;
	  }
	}
      }
    }
  }

  /****************************************************
                       bond-energies
  ****************************************************/

  My_Eele1[0] = 0.0;
  My_Eele1[1] = 0.0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    for (j=0; j<=FNAN[GA_AN]; j++){
      GB_AN = natn[GA_AN][j];  
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];

      for (k=0; k<tnoA; k++){
	for (l=0; l<tnoB; l++){
	  for (spin=0; spin<=SpinP_switch; spin++){
	    My_Eele1[spin] += CDM[spin][MA_AN][j][k][l]*nh[spin][MA_AN][j][k][l];
	  }
	}
      }

      /*
      printf("GA_AN=%2d LB_AN=%2d GB_AN=%2d RnB=%2d\n",GA_AN,LB_AN,GB_AN,ncn[GA_AN][j]);
      for (k=0; k<tnoA; k++){
	for (l=0; l<tnoB; l++){
	  for (spin=0; spin<=SpinP_switch; spin++){
	    printf("%15.12f ",CDM[spin][MA_AN][j][k][l]);
	  }
	}
        printf("\n");
      }
      */
  
    }
  }

  /* MPI, My_Eele1 */
  MPI_Barrier(mpi_comm_level1);
  for (spin=0; spin<=SpinP_switch; spin++){
    MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE,
                   MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0){
    Eele1[1] = Eele1[0];
  }

  if (3<=level_stdout && myid0==Host_ID){
    printf("Eele00=%15.12f Eele01=%15.12f\n",Eele0[0],Eele0[1]);
    printf("Eele10=%15.12f Eele11=%15.12f\n",Eele1[0],Eele1[1]);
  }

  /****************************************************
                        output
  ****************************************************/

  if (myid0==Host_ID){
  
    strcpy(file_EV,".EV");
    fnjoint(filepath,filename,file_EV);

    if ((fp_EV = fopen(file_EV,"w")) != NULL){

#ifdef xt3
      setvbuf(fp_EV,buf,_IOFBF,fp_bsize); 
#endif

      fprintf(fp_EV,"\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"           Eigenvalues (Hartree) of SCF KS-eq.           \n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n\n");
      fprintf(fp_EV,"   Chemical Potential (Hatree) = %18.14f\n",ChemP);
      fprintf(fp_EV,"   Number of States            = %18.14f\n",Num_State);
      fprintf(fp_EV,"   Eigenvalues\n");
      fprintf(fp_EV,"              Up-spin           Down-spin\n");

      for (kloop=0; kloop<T_knum; kloop++){

	k1 = T_KGrids1[kloop];
	k2 = T_KGrids2[kloop];
	k3 = T_KGrids3[kloop];

	if (0<T_k_op[kloop]){

	  fprintf(fp_EV,"\n");
	  fprintf(fp_EV,"   kloop=%i\n",kloop);
	  fprintf(fp_EV,"   k1=%10.5f k2=%10.5f k3=%10.5f\n\n",k1,k2,k3);
	  for (l=1; l<=MaxN; l++){
	    if (SpinP_switch==0){
	      fprintf(fp_EV,"%5d  %18.14f %18.14f\n",
		      l,EIGEN[0][kloop][l],EIGEN[0][kloop][l]);
	    }
	    else if (SpinP_switch==1){
	      fprintf(fp_EV,"%5d  %18.14f %18.14f\n",
		      l,EIGEN[0][kloop][l],EIGEN[1][kloop][l]);
	    }
	  }
	}
      }
      fclose(fp_EV);
    }
    else{
      printf("Failure of saving the EV file.\n");
      fclose(fp_EV);
    }  
  }

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

  free(VecFkw);
  free(VecFkwE);

  free(BLAS_H);
  free(BLAS_C);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  free(MPI_CommWD_CDM1);
  free(MPI_CDM1_flag);

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
    printf("myid0=%2d time7 =%9.4f\n",myid0,time7);fflush(stdout);
    printf("myid0=%2d time8 =%9.4f\n",myid0,time8);fflush(stdout);
    printf("myid0=%2d time81=%9.4f\n",myid0,time81);fflush(stdout);
    printf("myid0=%2d time82=%9.4f\n",myid0,time82);fflush(stdout);
    printf("myid0=%2d time83=%9.4f\n",myid0,time83);fflush(stdout);
    printf("myid0=%2d time84=%9.4f\n",myid0,time84);fflush(stdout);
    printf("myid0=%2d time85=%9.4f\n",myid0,time85);fflush(stdout);
    printf("myid0=%2d time9 =%9.4f\n",myid0,time9);fflush(stdout);
    printf("myid0=%2d time10=%9.4f\n",myid0,time10);fflush(stdout);
    printf("myid0=%2d time11=%9.4f\n",myid0,time11);fflush(stdout);
    printf("myid0=%2d time12=%9.4f\n",myid0,time12);fflush(stdout);

  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}
