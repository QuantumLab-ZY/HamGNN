/**********************************************************************
  Band_DFT_Col.c:

     Band_DFT_Col.c is a subroutine to perform band calculations
     based on a collinear DFT

  Log of Band_DFT_Col.c:

     22/Nov/2001  Released by T. Ozaki

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
 

double Band_DFT_Col(
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
		    double ***EIGEN,
		    double *H1,   
		    double *S1,   
		    double *CDM1,  
		    double *EDM1,
		    dcomplex **EVec1,
		    dcomplex *Ss,
		    dcomplex *Cs,
                    dcomplex *Hs,
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
  int i,j,k,l,m,n,p,wan,MaxN,i0,ks;
  int i1,i1s,j1,ia,jb,lmax,kmin,kmax,po,po1,spin,s1,e1;
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
  double FermiF,tmp1;
  double tmp,eig,kw,EV_cut0;
  double x,Dnum,Dnum2,AcP,ChemP_MAX,ChemP_MIN;
  int *is1,*ie1;
  int *is2,*ie2;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;

  int all_knum; 
  dcomplex Ctmp1,Ctmp2;
  int ii,ij,ik;
  int BM,BN,BK;
  double u2,v2,uv,vu;
  double d1,d2,d3,d4,ReA,ImA;
  double My_Eele1[2]; 
  double TZ,dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2;
  double Imsum,ImsumE,Imdum,Imdum2;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double Stime,Etime,Stime0,Etime0;
  double FermiEps=1.0e-13;
  double x_cut=60.0;
  double My_Eele0[2];

  char file_EV[YOUSO10];
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */
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

  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int info,ig,jg,il,jl,prow,pcol,brow,bcol;
  int ZERO=0, ONE=1;
  double alpha = 1.0; double beta = 0.0;
  int LOCr, LOCc, node, irow, icol;
  double mC_spin_i1,C_spin_i1;

  int Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;

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

  lumos = (double)n*0.200;
  if (lumos<60.0) lumos = 400.0;
  MaxN = (TZ-system_charge)/2 + (int)lumos;
  if (n<MaxN) MaxN = n;

  /***********************************************
     allocation of arrays
  ***********************************************/

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  /***********************************************
              k-points by regular mesh 
  ***********************************************/

  if (way_of_kpoint==1){

    /**************************************************************
     k_op[i][j][k]: weight of DOS 
                 =0   no calc.
                 =1   G-point
                 =2   which has k<->-k point
        Now, only the relation, E(k)=E(-k), is used. 

    Future release: k_op will be used for symmetry operation 
    *************************************************************/

    for (i=0;i<knum_i;i++) {
      for (j=0;j<knum_j;j++) {
	for (k=0;k<knum_k;k++) {
	  k_op[i][j][k]=-999;
	}
      }
    }

    for (i=0;i<knum_i;i++) {
      for (j=0;j<knum_j;j++) {
	for (k=0;k<knum_k;k++) {
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
    for (i=0; i<knum_i; i++){
      for (j=0; j<knum_j; j++){
	for (k=0; k<knum_k; k++){
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

    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);
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

    is1 = (int*)malloc(sizeof(int)*numprocs2);
    ie1 = (int*)malloc(sizeof(int)*numprocs2);

    is2 = (int*)malloc(sizeof(int)*numprocs2);
    ie2 = (int*)malloc(sizeof(int)*numprocs2);

    Num_Snd_EV = (int*)malloc(sizeof(int)*numprocs2);
    Num_Rcv_EV = (int*)malloc(sizeof(int)*numprocs2);

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
	ie1[ID] =  0;
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
	is2[ID] = 1;
	ie2[ID] = 0;
      }
    }

    /****************************************************************
       making data structure of MPI communicaition for eigenvectors 
    ****************************************************************/

    for (ID=0; ID<numprocs2; ID++){
      Num_Snd_EV[ID] = 0;
      Num_Rcv_EV[ID] = 0;
    }

    for(i=0; i<na_rows; i++){

      ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;

      po = 0;
      for (ID=0; ID<numprocs2; ID++){
	if (is2[ID]<=ig && ig <=ie2[ID]){
	  po = 1;
	  ID0 = ID;
	  break;
	}
      }

      if (po==1) Num_Snd_EV[ID0] += na_cols;
    }

    for (ID=0; ID<numprocs2; ID++){
      IDS = (myid2 + ID) % numprocs2;
      IDR = (myid2 - ID + numprocs2) % numprocs2;
      if (ID!=0){
	MPI_Isend(&Num_Snd_EV[IDS], 1, MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
	MPI_Recv(&Num_Rcv_EV[IDR],  1, MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
	MPI_Wait(&request,&stat);
      }
      else{
	Num_Rcv_EV[IDR] = Num_Snd_EV[IDS];
      }
    }

    Max_Num_Snd_EV = 0;
    Max_Num_Rcv_EV = 0;
    for (ID=0; ID<numprocs2; ID++){
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

  } /* if (all_knum==1) */

  /****************************************************
     PrintMemory
  ****************************************************/

  if (firsttime && memoryusage_fileout) {
    PrintMemory("Band_DFT_Col: My_NZeros", sizeof(int)*numprocs0,NULL);
    PrintMemory("Band_DFT_Col: SP_NZeros", sizeof(int)*numprocs0,NULL);
    PrintMemory("Band_DFT_Col: SP_Atoms", sizeof(int)*numprocs0,NULL);
    if (all_knum==1){
      PrintMemory("Band_DFT_Col: is1", sizeof(int)*numprocs2,NULL);
      PrintMemory("Band_DFT_Col: ie1", sizeof(int)*numprocs2,NULL);
      PrintMemory("Band_DFT_Col: is2", sizeof(int)*numprocs2,NULL);
      PrintMemory("Band_DFT_Col: ie2", sizeof(int)*numprocs2,NULL);
      PrintMemory("Band_DFT_Col: Num_Snd_EV", sizeof(int)*numprocs2,NULL);
      PrintMemory("Band_DFT_Col: Num_Rcv_EV", sizeof(int)*numprocs2,NULL);
      PrintMemory("Band_DFT_Col: index_Snd_i", sizeof(int)*Max_Num_Snd_EV,NULL);
      PrintMemory("Band_DFT_Col: index_Snd_j", sizeof(int)*Max_Num_Snd_EV,NULL);
      PrintMemory("Band_DFT_Col: EVec_Snd", sizeof(double)*Max_Num_Snd_EV*2,NULL);
      PrintMemory("Band_DFT_Col: index_Rcv_i", sizeof(int)*Max_Num_Rcv_EV,NULL);
      PrintMemory("Band_DFT_Col: index_Rcv_j", sizeof(int)*Max_Num_Rcv_EV,NULL);
      PrintMemory("Band_DFT_Col: EVec_Rcv", sizeof(double)*Max_Num_Rcv_EV*2,NULL);
    }
  }

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

  if (measure_time) dtime(&Stime);

  /* spin=myworld1 */

  spin = myworld1;

  /* set S1 */

  if (SCF_iter==1 || all_knum!=1){
    size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

diagonalize1:

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

  if (measure_time){ 
    dtime(&Etime);
    time1 += Etime - Stime;
  }

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

    if (SCF_iter==1 || all_knum!=1){
      for(i=0;i<na_rows;i++){
	for(j=0;j<na_cols;j++){
	  Cs[j*na_rows+i].r = 0.0;
	  Cs[j*na_rows+i].i = 0.0;
	}
      }
    }

    for(i=0;i<na_rows;i++){
      for(j=0;j<na_cols;j++){
	Hs[j*na_rows+i].r = 0.0;
	Hs[j*na_rows+i].i = 0.0;
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

	  ig = Anum + i;
	  brow = (ig-1)/nblk;
	  prow = brow%np_rows;

	  for (j=0; j<tnoB; j++){

	    jg = Bnum + j;
	    bcol = (jg-1)/nblk;
	    pcol = bcol%np_cols;

	    if (my_prow==prow && my_pcol==pcol){

	      il = (brow/np_rows+1)*nblk+1;
	      jl = (bcol/np_cols+1)*nblk+1;

	      if (((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
		if(my_prow==prow){
		  il = il+(ig-1)%nblk;
		}
		il = il-nblk;
	      }

	      if (((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
		if(my_pcol==pcol){
		  jl = jl+(jg-1)%nblk;
		}
		jl = jl-nblk;
	      }

	      if (SCF_iter==1 || all_knum!=1){
		Cs[(jl-1)*na_rows+il-1].r += S1[k]*co;
		Cs[(jl-1)*na_rows+il-1].i += S1[k]*si;
	      }

	      Hs[(jl-1)*na_rows+il-1].r += H1[k]*co;
	      Hs[(jl-1)*na_rows+il-1].i += H1[k]*si;
	    }

	    k++;

	  }
	}
      }
    }

    /* diagonalize S */

    if (measure_time) dtime(&Stime);

    if (parallel_mode==0 || (SCF_iter==1 || all_knum!=1)){
      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      if (scf_eigen_lib_flag==1){

        F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
        ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
      }
      else if (scf_eigen_lib_flag==2){

#ifndef kcomp
	int mpiworld;
	mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
	F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
        ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols, 
          &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#endif
      }

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);
    }

    if (measure_time){
      dtime(&Etime);
      time2 += Etime - Stime;
    }

    if (SCF_iter==1 || all_knum!=1){

      if (3<=level_stdout){
	printf(" myid0=%2d spin=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,spin,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,ko[i1]);
	}
      }

      /* minus eigenvalues to 1.0e-10 */

      for (l=1; l<=n; l++){
	if (ko[l]<0.0) ko[l] = 1.0e-10;
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

    if (measure_time) dtime(&Stime);

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

    if (measure_time){
      dtime(&Etime);
      time3 += Etime - Stime;
    }

    /* diagonalize H' */

    if (measure_time) dtime(&Stime);

    MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
    MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

    mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
    mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
	
    if (scf_eigen_lib_flag==1){

      F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
      ( &n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
    }
    else if (scf_eigen_lib_flag==2){

#ifndef kcomp
      int mpiworld;
      mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
      F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
      ( &n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &na_cols,
        &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#endif
    }

    MPI_Comm_free(&mpi_comm_rows);
    MPI_Comm_free(&mpi_comm_cols);

    if (measure_time){
      dtime(&Etime);
      time4 += Etime - Stime;
    }

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

    /**************************************************
      if (all_knum==1), wave functions are calculated. 
    **************************************************/

    if (measure_time) dtime(&Stime);

    if (all_knum==1){

      for(i=0; i<na_rows*na_cols; i++){
        Hs[i].r = 0.0;
        Hs[i].i = 0.0;
      }

      F77_NAME(pzgemm,PZGEMM)("T","T",&n,&n,&n,&alpha,Cs,&ONE,&ONE,descS,Ss,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);
      Cblacs_barrier(ictxt2,"A");

      /* MPI communications of Hs and store them to EVec1 */

      for (ID=0; ID<numprocs2; ID++){

	IDS = (myid2 + ID) % numprocs2;
	IDR = (myid2 - ID + numprocs2) % numprocs2;

	k = 0;
	for(i=0; i<na_rows; i++){

	  ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
	  if (is2[IDS]<=ig && ig <=ie2[IDS]){

	    for (j=0; j<na_cols; j++){
	      jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;

	      index_Snd_i[k] = ig;
	      index_Snd_j[k] = jg;
	      EVec_Snd[2*k  ] = Hs[j*na_rows+i].r;
	      EVec_Snd[2*k+1] = Hs[j*na_rows+i].i;

	      k++;
	    }
	  }
	}

	if (ID!=0){

	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Isend(index_Snd_i, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
	  }
	  if (Num_Rcv_EV[IDR]!=0){
	    MPI_Recv(index_Rcv_i, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
	  }
	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Wait(&request,&stat);
	  }

	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Isend(index_Snd_j, Num_Snd_EV[IDS], MPI_INT, IDS, 999, MPI_CommWD2[myworld2], &request);
	  }
	  if (Num_Rcv_EV[IDR]!=0){
	    MPI_Recv(index_Rcv_j, Num_Rcv_EV[IDR], MPI_INT, IDR, 999, MPI_CommWD2[myworld2], &stat);
	  }
	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Wait(&request,&stat);
	  }

	  if (Num_Snd_EV[IDS]!=0){
	    MPI_Isend(EVec_Snd, Num_Snd_EV[IDS]*2, MPI_DOUBLE, IDS, 999, MPI_CommWD2[myworld2], &request);
	  }
	  if (Num_Rcv_EV[IDR]!=0){
	    MPI_Recv(EVec_Rcv, Num_Rcv_EV[IDR]*2, MPI_DOUBLE, IDR, 999, MPI_CommWD2[myworld2], &stat);
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
	  m = (jg-1)*(ie2[myid2]-is2[myid2]+1)+ig-is2[myid2];

	  EVec1[spin][m].r = EVec_Rcv[2*k  ];
	  EVec1[spin][m].i = EVec_Rcv[2*k+1];

	}

      } /* ID */

    } /* if (all_knum==1) */

    if (measure_time){
      dtime(&Etime);
      time5 += Etime - Stime;
    }

  } /* kloop0 */

  if (measure_time) dtime(&EiloopTime);

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;  
    goto diagonalize1; 
  }

  /****************************************************
     MPI:

     EIGEN
  ****************************************************/

  if (measure_time){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Stime);
  }

  for (spin=0; spin<=SpinP_switch; spin++){
    for (kloop=0; kloop<T_knum; kloop++){

      /* get ID in the zeroth world */
      ID = Comm_World_StartID1[spin] + T_k_ID[spin][kloop];
      MPI_Bcast(&EIGEN[spin][kloop][0], MaxN+1, MPI_DOUBLE, ID, mpi_comm_level1);
    } 
  }

  if (measure_time){ 
    dtime(&Etime);
    time6 += Etime - Stime;
  }

  /**************************************
         find chemical potential
  **************************************/

  if (measure_time) dtime(&Stime);
  
  /* first, find ChemP at five times large temperatue */

  po = 0;
  loop_num = 0;
  ChemP_MAX = 20.0;  
  ChemP_MIN =-20.0;

  do {

    loop_num++;

    ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
    Num_State = 0.0;

    for (kloop=0; kloop<T_knum; kloop++){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (l=1; l<=MaxN; l++){

	  x = (EIGEN[spin][kloop][l] - ChemP)*Beta*0.2;

	  if (x<=-x_cut) x = -x_cut;
	  if (x_cut<=x)  x =  x_cut;
	  FermiF = FermiFunc(x,spin,l,&l,&x);

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

  /* second, find ChemP at the temperatue, starting from the previously found ChemP. */

  po = 0;
  loop_num = 0;
  ChemP_MAX = 20.0;  
  ChemP_MIN =-20.0;

  do {

    loop_num++;

    if (loop_num!=1){
      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);
    }

    Num_State = 0.0;

    for (kloop=0; kloop<T_knum; kloop++){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (l=1; l<=MaxN; l++){

	  x = (EIGEN[spin][kloop][l] - ChemP)*Beta;

	  if (x<=-x_cut) x = -x_cut;
	  if (x_cut<=x)  x =  x_cut;
	  FermiF = FermiFunc(x,spin,l,&l,&x);

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

        if (x<=-x_cut) x = -x_cut;
	if (x_cut<=x)  x = x_cut;
	FermiF = FermiFunc(x,spin,l,&l,&x);

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

  if (measure_time){
    dtime(&Etime);
    time7 += Etime - Stime;
  }

  /****************************************************
         if all_knum==1, calculate CDM and EDM
  ****************************************************/

  if (measure_time) dtime(&Stime);

  if (all_knum==1){

    if (measure_time) dtime(&Stime0);

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

    /* pre-calculation of the Fermi function */

    po = 0;
    kmin = is2[myid2];
    kmax = ie2[myid2];

    for (k=is2[myid2]; k<=ie2[myid2]; k++){

      eig = EIGEN[spin][kloop][k];
      x = (eig - ChemP)*Beta;
      if (x<=-x_cut) x = -x_cut;
      if (x_cut<=x)  x = x_cut;
      FermiF = FermiFunc(x,spin,k,&k,&x);
      tmp1 = sqrt(kw*FermiF);

      for (i1=1; i1<=n; i1++){
	i = (i1-1)*(ie2[myid2]-is2[myid2]+1) + k - is2[myid2];
	EVec1[spin][i].r *= tmp1;
	EVec1[spin][i].i *= tmp1;
      }

      /* find kmax */

      if ( FermiF<FermiEps && po==0 ) {
        kmax = k;
        po = 1;         
      }
    }    

    if (measure_time){
      dtime(&Etime0);
      time81 += Etime0 - Stime0;
      dtime(&Stime0);
    }

    /* calculation of CDM1 and EDM1 */ 

    p = 0;
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

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

	    i1 = (Anum + i - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];
	    j1 = (Bnum + j - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];

	    d1 = 0.0;
	    d2 = 0.0;
	    d3 = 0.0;
	    d4 = 0.0;

	    for (k=kmin; k<=kmax; k++){

	      ReA = EVec1[spin][i1+k].r*EVec1[spin][j1+k].r + EVec1[spin][i1+k].i*EVec1[spin][j1+k].i; 
	      ImA = EVec1[spin][i1+k].r*EVec1[spin][j1+k].i - EVec1[spin][i1+k].i*EVec1[spin][j1+k].r;

	      d1 += ReA;
	      d2 += ImA;
	      d3 += ReA*EIGEN[spin][kloop][k];
	      d4 += ImA*EIGEN[spin][kloop][k];
	    }

	    CDM1[p] += co*d1 - si*d2;
	    EDM1[p] += co*d3 - si*d4;

	    /* increment of p */
	    p++;  

	  }
	}
      }
    } /* GA_AN */

    if (measure_time){
      dtime(&Etime0);
      time82 += Etime0 - Stime0;
      dtime(&Stime0);
     }

    /* sum of CDM1 and EDM1 by Allreduce in MPI */

    MPI_Allreduce(CDM1, H1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
    for (i=0; i<size_H1; i++) CDM1[i] = H1[i];

    MPI_Allreduce(EDM1, H1, size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
    for (i=0; i<size_H1; i++) EDM1[i] = H1[i];

    if (measure_time){
      dtime(&Etime0);
      time83 += Etime0 - Stime0;
      dtime(&Stime0);
    }

    /* store DM1 to a proper place in CDM and EDM */

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

              CDM[spin][MA_AN][LB_AN][i][j] = CDM1[p];
              EDM[spin][MA_AN][LB_AN][i][j] = EDM1[p];

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

    if (measure_time){
      dtime(&Etime0);
      time84 += Etime0 - Stime0;
      dtime(&Stime0);
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
	  MPI_Isend(&CDM1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
	}

	if (myid0==IDR){
	  MPI_Recv(&H1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
	}

	if (myid0==IDS){
	  MPI_Wait(&request,&stat);
	}

	if (myid0==IDS){
	  MPI_Isend(&EDM1[0], size_H1, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request);
	}

	if (myid0==IDR){
	  MPI_Recv(&S1[0], size_H1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &stat);
	}

	if (myid0==IDS){
	  MPI_Wait(&request,&stat);
	}
      }

      MPI_Bcast(&H1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);
      MPI_Bcast(&S1[0], size_H1, MPI_DOUBLE, 0, MPI_CommWD1[myworld1]);

      /* put CDM1 and EDM1 into CDM and EDM */

      k = 0;
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

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
    }

    if (measure_time){
      dtime(&Etime0);
      time85 += Etime0 - Stime0;
    }

  } /* if (all_knum==1) */

  if (measure_time){
    dtime(&Etime);
    time8 += Etime - Stime;
  }

  dtime(&EiloopTime);

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

    size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

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

      for(i=0;i<na_rows;i++){
	for(j=0;j<na_cols;j++){
	  Cs[j*na_rows+i] = Complex(0.0,0.0);
	  Hs[j*na_rows+i] = Complex(0.0,0.0);
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

	    ig = Anum + i;
	    brow = (ig-1)/nblk;
	    prow = brow%np_rows;

	    for (j=0; j<tnoB; j++){

	      jg = Bnum + j;
	      bcol = (jg-1)/nblk;
	      pcol = bcol%np_cols;

	      if (my_prow==prow && my_pcol==pcol){

		il = (brow/np_rows+1)*nblk+1;
		jl = (bcol/np_cols+1)*nblk+1;

		if (((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
		  if(my_prow==prow){
		    il = il+(ig-1)%nblk;
		  }
		  il = il-nblk;
		}

		if (((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
		  if(my_pcol==pcol){
		    jl = jl+(jg-1)%nblk;
		  }
		  jl = jl-nblk;
		}

		if (SCF_iter==1 || all_knum!=1){
		  Cs[(jl-1)*na_rows+il-1].r += S1[k]*co;
		  Cs[(jl-1)*na_rows+il-1].i += S1[k]*si;
		}

		Hs[(jl-1)*na_rows+il-1].r += H1[k]*co;
		Hs[(jl-1)*na_rows+il-1].i += H1[k]*si;
	      }

	      k++;

	    }
	  }
	}
      }

      /* diagonalize S */

      if (measure_time) dtime(&Stime);

      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      if (scf_eigen_lib_flag==1){

        F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
        ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);
      }
      else if (scf_eigen_lib_flag==2){

#ifndef kcomp
        int mpiworld;
        mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
        F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
        ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &na_cols,
          &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);
#endif
      }

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);

      if (measure_time){
        dtime(&Etime);
        time9 += Etime - Stime;
      }

      if (3<=level_stdout){
	printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,ko[i1]);
	}
      }

      /* minus eigenvalues to 1.0e-14 */

      for (l=1; l<=n; l++){
	if (ko[l]<0.0) ko[l] = 1.0e-10;
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

      /* diagonalize H' */

      if (measure_time) dtime(&Stime);

      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      if (scf_eigen_lib_flag==1){
        F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
        ( &n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
      }
      else if (scf_eigen_lib_flag==2){

#ifndef kcomp
	int mpiworld;
	mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
	F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)
	  ( &n, &MaxN, Hs, &na_rows, &ko[1], Cs, &na_rows, &nblk, &na_cols, 
            &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld );
#endif
      }

      MPI_Comm_free(&mpi_comm_rows);
      MPI_Comm_free(&mpi_comm_cols);

      if (measure_time){
        dtime(&Etime);
        time10 += Etime - Stime;
      }

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

      for(i=0;i<na_rows*na_cols;i++){
	Hs[i].r = 0.0;
	Hs[i].i = 0.0;
      }

      F77_NAME(pzgemm,PZGEMM)("T","T",&n,&n,&n,&alpha,Cs,&ONE,&ONE,descS,Ss,&ONE,&ONE,descC,&beta,Hs,&ONE,&ONE,descH);
      Cblacs_barrier(ictxt2,"A");

      /* Hs are stored to EVec1 */

      k = 0;
      for (j=0; j<na_cols; j++){
        for(i=0; i<na_rows; i++){

	  EVec1[spin][k].r = Hs[j*na_rows+i].r;
	  EVec1[spin][k].i = Hs[j*na_rows+i].i;

	  k++;
	}
      }

      /****************************************************
                     calculate DM and EDM
      ****************************************************/

      if (measure_time) dtime(&Stime);

      /* weight of k-point */ 

      kw = (double)T_k_op[kloop];

      po = 0;
      kmin = 1;
      kmax = MaxN;

      for (k=1; k<=MaxN; k++){

	eig = EIGEN[spin][kloop][k];
	x = (eig - ChemP)*Beta;
	if (x<=-x_cut) x = -x_cut;
	if (x_cut<=x)  x = x_cut;
	FermiF = FermiFunc(x,spin,k,&k,&x);
	tmp1 = sqrt(kw*FermiF);

	for (i1=1; i1<=n; i1++){
	  i = (i1-1)*n + k - 1;
	  EVec1[spin][i].r *= tmp1;
	  EVec1[spin][i].i *= tmp1;
	}

	/* find kmax */

	if ( FermiF<FermiEps && po==0 ) {
	  kmax = k;
	  po = 1;         
	}
      }    

      /* calculation of CDM1 and EDM1 */ 

      p = 0;
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

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

	      i1 = (Anum + i - 1)*n - 1;
	      j1 = (Bnum + j - 1)*n - 1;

	      d1 = 0.0;
	      d2 = 0.0;
	      d3 = 0.0;
	      d4 = 0.0;

	      for (k=1; k<=MaxN; k++){

		ReA = EVec1[spin][i1+k].r*EVec1[spin][j1+k].r + EVec1[spin][i1+k].i*EVec1[spin][j1+k].i; 
		ImA = EVec1[spin][i1+k].r*EVec1[spin][j1+k].i - EVec1[spin][i1+k].i*EVec1[spin][j1+k].r;

		d1 += ReA;
		d2 += ImA;
		d3 += ReA*EIGEN[spin][kloop][k];
		d4 += ImA*EIGEN[spin][kloop][k];
	      }

	      CDM1[p] += co*d1 - si*d2;
	      EDM1[p] += co*d3 - si*d4;

	      /* increment of p */
	      p++;  

	    }
	  }
	}
      } /* GA_AN */

      if (measure_time){
        dtime(&Etime);
        time11 += Etime - Stime;
      }

    } /* kloop0 */

    /*******************************************************
         sum of CDM1 and EDM1 by Allreduce in MPI
    *******************************************************/

    if (measure_time) dtime(&Stime);

    MPI_Allreduce(&CDM1[0], &H1[0], size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);
    MPI_Allreduce(&EDM1[0], &S1[0], size_H1, MPI_DOUBLE, MPI_SUM, MPI_CommWD1[myworld1]);

    /* CDM and EDM */

    k = 0;
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

      MA_AN = F_G2M[GA_AN]; 
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      ID = G2ID[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	GB_AN = natn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];

	if (myid0==ID){

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
	      CDM[spin][MA_AN][LB_AN][i][j] = H1[k];
              EDM[spin][MA_AN][LB_AN][i][j] = S1[k];
	      k++;
	    }
	  }
	}

	else{
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
	      k++;
	    }
	  }
	}
      }
    }

    if (measure_time){
      dtime(&Etime);
      time12 += Etime - Stime;
    }

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
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

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

      setvbuf(fp_EV,buf,_IOFBF,fp_bsize); 

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

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  if (all_knum==1){

    free(is1);
    free(ie1);

    free(is2);
    free(ie2);

    free(Num_Snd_EV);
    free(Num_Rcv_EV);
    free(index_Snd_i);
    free(index_Snd_j);
    free(EVec_Snd);
    free(index_Rcv_i);
    free(index_Rcv_j);
    free(EVec_Rcv);
  }

  /* for PrintMemory and allocation */
  firsttime=0;

  /* for elapsed time */

  if (measure_time){
    printf("myid0=%2d time1 =%9.4f\n",myid0,time1);fflush(stdout);
    printf("myid0=%2d time2 =%9.4f\n",myid0,time2);fflush(stdout);
    printf("myid0=%2d time3 =%9.4f\n",myid0,time3);fflush(stdout);
    printf("myid0=%2d time4 =%9.4f\n",myid0,time4);fflush(stdout);
    printf("myid0=%2d time5 =%9.4f\n",myid0,time5);fflush(stdout);
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

