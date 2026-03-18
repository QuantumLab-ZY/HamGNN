/**********************************************************************
  Band_DFT_NonCol.c:

     Band_DFT_NonCol.c is a subroutine to perform band calculations
     based on a non-collinear DFT. 

  Log of Band_DFT_NonCol.c:

     16/Feb./2019  Released by T.Ozaki

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


static void Construct_Band_Ms( int cpx_flag, double ****Mat, double *M1, double *M2, dcomplex *Ms, 
                               int *MP, double k1, double k2, double k3);


static double Calc_DM_Band_non_collinear(
    int calc_flag,
    int store_flag,
    int myid0,
    int myid2,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    int MaxN,
    double k1,
    double k2,
    double k3,
    double *****CDM,
    double *****iDM0,
    double *****EDM,
    double *ko,
    double *DM1,
    double *Work1,
    dcomplex *EVec1,
    double *rDM11,
    double *rDM22,
    double *rDM12,
    double *iDM12,
    double *iDM11,
    double *iDM22, 
    double *rEDM11,
    double *rEDM22);



double Band_DFT_NonCol(
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
		    dcomplex *rHs11,   
		    dcomplex *rHs22,   
		    dcomplex *rHs12,   
		    dcomplex *iHs11,   
		    dcomplex *iHs22,   
		    dcomplex *iHs12, 
		    dcomplex **EVec1,
		    dcomplex *Ss,
		    dcomplex *Cs,
                    dcomplex *Hs,
		    dcomplex *Ss2,
		    dcomplex *Cs2,
                    dcomplex *Hs2,
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
  int i,j,k,l,m,n,n2,p,wan,MaxN,i0,ks;
  int i1,i1s,j1,ia,jb,lmax,po,po1,spin,s1,e1;
  int num2,RnB,l1,l2,l3,loop_num,ns,ne;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int MA_AN,GA_AN,Anum,num_kloop0,max_num_kloop0;
  int T_knum,S_knum,E_knum,kloop,kloop0;
  double av_num,lumos;
  double time0;
  int LB_AN,GB_AN,Bnum;
  double k1,k2,k3,Fkw;
  double sum,sumi,sum_weights;
  double Num_State;
  double My_Num_State;
  double FermiF;
  double tmp,tmp1,eig,kw,EV_cut0;
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
  double d1,d2;
  double My_Eele1[2]; 
  double TZ,dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2;
  double Imsum,ImsumE,Imdum,Imdum2;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double Stime,Etime,Stime0,Etime0;
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
  double time10,time11,time12,time13;

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
  double mC_spin_i1,C_spin_i1;

  int Max_Num_Snd_EV,Max_Num_Rcv_EV;
  int *Num_Snd_EV,*Num_Rcv_EV;
  int *index_Snd_i,*index_Snd_j,*index_Rcv_i,*index_Rcv_j;
  double *EVec_Snd,*EVec_Rcv;
  double *rDM11,*rDM22,*rDM12,*iDM12,*iDM11,*iDM22,*rEDM11,*rEDM22;

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
  time13 = 0.0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);
  MPI_Barrier(mpi_comm_level1);

  Num_Comm_World1 = 1;

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
  n2 = n*2; 

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

  lumos = (double)n2*0.200;
  if (lumos<60.0) lumos = 400.0;
  MaxN = (TZ-system_charge)/2 + (int)lumos;
  if (n2<MaxN) MaxN = n2;

  /***********************************************
     allocation of arrays
  ***********************************************/

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  /****************************************************
   find size_H
  ****************************************************/

  size_H1 = Get_OneD_HS_Col(0, nh[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /***********************************************
     allocation of arrays 
  ***********************************************/

  rDM11 = (double*)malloc(sizeof(double)*size_H1);
  rDM22= (double*)malloc(sizeof(double)*size_H1);
  rDM12 = (double*)malloc(sizeof(double)*size_H1);
  iDM11 = (double*)malloc(sizeof(double)*size_H1);
  iDM22 = (double*)malloc(sizeof(double)*size_H1);
  iDM12 = (double*)malloc(sizeof(double)*size_H1);
  rEDM11 = (double*)malloc(sizeof(double)*size_H1);
  rEDM22 = (double*)malloc(sizeof(double)*size_H1);

  for (i=0; i<size_H1; i++){
    rDM11[i] = 0.0;
    rDM22[i] = 0.0;
    rDM12[i] = 0.0;
    iDM11[i] = 0.0;
    iDM22[i] = 0.0;
    iDM12[i] = 0.0;
    rEDM11[i] = 0.0;
    rEDM22[i] = 0.0;
  }

  /***********************************************
          initialize CDM, EDM, and iDM
  ***********************************************/

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
	  CDM[0][MA_AN][LB_AN][i][j] = 0.0;
	  CDM[1][MA_AN][LB_AN][i][j] = 0.0;
	  CDM[2][MA_AN][LB_AN][i][j] = 0.0;
	  CDM[3][MA_AN][LB_AN][i][j] = 0.0;
	  EDM[0][MA_AN][LB_AN][i][j] = 0.0;
	  EDM[1][MA_AN][LB_AN][i][j] = 0.0;
	  EDM[2][MA_AN][LB_AN][i][j] = 0.0;
	  EDM[3][MA_AN][LB_AN][i][j] = 0.0;
	  iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
	  iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
	}
      }
    }
  }

  /***********************************************
              k-points by regular mesh 
  ***********************************************/

  for (i=0;i<knum_i;i++) {
    for (j=0;j<knum_j;j++) {
      for (k=0;k<knum_k;k++) {
	k_op[i][j][k] = 1;
      }
    }
  }

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
  MPI_Allreduce(&num_kloop0, &max_num_kloop0, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

  /****************************************************
                make is1, ie1, is2, ie2
  ****************************************************/

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

  for(i=0; i<na_rows2; i++){

    ig = np_rows2*nblk2*((i)/nblk2) + (i)%nblk2 + ((np_rows2+my_prow2)%np_rows2)*nblk2 + 1;

    po = 0;
    for (ID=0; ID<numprocs2; ID++){
      if (is2[ID]<=ig && ig <=ie2[ID]){
	po = 1;
	ID0 = ID;
	break;
      }
    }

    if (po==1) Num_Snd_EV[ID0] += na_cols2;
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

  /****************************************************
                      PrintMemory
  ****************************************************/

  if (firsttime && memoryusage_fileout) {
  PrintMemory("Band_DFT_NonCol: My_NZeros", sizeof(int)*numprocs0,NULL);
  PrintMemory("Band_DFT_NonCol: SP_NZeros", sizeof(int)*numprocs0,NULL);
  PrintMemory("Band_DFT_NonCol: SP_Atoms", sizeof(int)*numprocs0,NULL);
  PrintMemory("Band_DFT_NonCol: is1", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: ie1", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: is2", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: ie2", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: Num_Snd_EV", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: Num_Rcv_EV", sizeof(int)*numprocs2,NULL);
  PrintMemory("Band_DFT_NonCol: index_Snd_i", sizeof(int)*Max_Num_Snd_EV,NULL);
  PrintMemory("Band_DFT_NonCol: index_Snd_j", sizeof(int)*Max_Num_Snd_EV,NULL);
  PrintMemory("Band_DFT_NonCol: EVec_Snd", sizeof(double)*Max_Num_Snd_EV*2,NULL);
  PrintMemory("Band_DFT_NonCol: index_Rcv_i", sizeof(int)*Max_Num_Rcv_EV,NULL);
  PrintMemory("Band_DFT_NonCol: index_Rcv_j", sizeof(int)*Max_Num_Rcv_EV,NULL);
  PrintMemory("Band_DFT_NonCol: EVec_Rcv", sizeof(double)*Max_Num_Rcv_EV*2,NULL);

  PrintMemory("Band_DFT_NonCol: rDM11", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: rDM22", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: rDM12", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: iDM11", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: iDM22", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: iDM12", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: rEDM11", sizeof(double)*size_H1,NULL);
  PrintMemory("Band_DFT_NonCol: rEDM22", sizeof(double)*size_H1,NULL);
  }

  /****************************************************
                      start kloop
  ****************************************************/

  dtime(&SiloopTime);

  for (kloop0=0; kloop0<max_num_kloop0; kloop0++){

    /* get k1, k2, and k3 */

    if (kloop0<num_kloop0){

      kloop = S_knum + kloop0;
      k1 = T_KGrids1[kloop];
      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];
    }

    if (measure_time) dtime(&Stime);

    if (SCF_iter==1 || all_knum!=1){

      /* make Cs */

      Construct_Band_Ms(0,CntOLP,H1,S1,Cs,MP,k1,k2,k3);

      /* diagonalize Cs */

      if (kloop0<num_kloop0){

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

	/* print to the standard output */

	if (3<=level_stdout){
	  printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i=1; i<=n; i++){
	    printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	  }
	}

	/*
	  printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	  myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i=1; i<=n; i++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	  }
	*/

	/* minus eigenvalues to 1.0e-10 */

	for (l=1; l<=n; l++){
	  if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
	  ko[l] = 1.0/sqrt(ko[l]);
	}

	/* calculate S*1/sqrt(ko) */

	for(i=0; i<na_rows; i++){
	  for(j=0; j<na_cols; j++){
	    jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	    Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	    Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
	  }
	}

	/* make Ss2 */

	Overlap_Band_NC_Ss2( Ss, Ss2, MPI_CommWD2[myworld2] );
      }
    }

    if (measure_time){
      dtime(&Etime);
      time1 += Etime - Stime;
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

    if (measure_time) dtime(&Stime);

    /* set rHs and iHs */

    Construct_Band_Ms(0,nh[0],H1,S1,rHs11,MP,k1,k2,k3);
    Construct_Band_Ms(0,nh[1],H1,S1,rHs22,MP,k1,k2,k3);
    Construct_Band_Ms(0,nh[2],H1,S1,rHs12,MP,k1,k2,k3);
    Construct_Band_Ms(1,nh[3],H1,S1,iHs12,MP,k1,k2,k3);

    Construct_Band_Ms(1,ImNL[0],H1,S1,iHs11,MP,k1,k2,k3);
    Construct_Band_Ms(1,ImNL[1],H1,S1,iHs22,MP,k1,k2,k3);
    Construct_Band_Ms(1,ImNL[2],H1,S1,Cs,MP,k1,k2,k3);

    if (measure_time){
      dtime(&Etime);
      time2 += Etime - Stime;
      dtime(&Stime);
    }

    for (i=0; i<na_rows*na_cols; i++){
      iHs12[i].r += Cs[i].r;
      iHs12[i].i += Cs[i].i;
    }

    for (i=0; i<na_rows*na_cols; i++){
      rHs11[i].r += iHs11[i].r;  
      rHs11[i].i += iHs11[i].i;  
      rHs22[i].r += iHs22[i].r;  
      rHs22[i].i += iHs22[i].i;  
      rHs12[i].r += iHs12[i].r;  
      rHs12[i].i += iHs12[i].i;
    }

    Hamiltonian_Band_NC_Hs2( rHs11, rHs22, rHs12, Hs2, MPI_CommWD2[myworld2] );
    
    if (measure_time){
      dtime(&Etime);
      time13 += Etime - Stime;
    }

    if (kloop0<num_kloop0){

      if (measure_time) dtime(&Stime);

      /* S^t x rHs11 x S */

      for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

      Cblacs_barrier(ictxt1,"A");
      F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,rHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

      for (i=0; i<na_rows*na_cols; i++) rHs11[i] = Complex(0.0,0.0);

      Cblacs_barrier(ictxt1,"C");
      F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,rHs11,&ONE,&ONE,descH);

      /* S^t x rHs12 x S */

      for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

      Cblacs_barrier(ictxt1,"A");
      F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,rHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

      for (i=0; i<na_rows*na_cols; i++) rHs12[i] = Complex(0.0,0.0);

      Cblacs_barrier(ictxt1,"C");
      F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,rHs12,&ONE,&ONE,descH);

      /* S^t x rHs22 x S */

      for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

      Cblacs_barrier(ictxt1,"A");
      F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,rHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

      for (i=0; i<na_rows*na_cols; i++) rHs22[i] = Complex(0.0,0.0);

      Cblacs_barrier(ictxt1,"C");
      F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,rHs22,&ONE,&ONE,descH);

      if (measure_time){
        dtime(&Etime);
        time3 += Etime - Stime;
      }

      /* ***************************************************
	 diagonalize the transformed H
      *************************************************** */

      if (measure_time){
        dtime(&Stime);
      }

      Hamiltonian_Band_NC_Hs2( rHs11, rHs22, rHs12, Hs2, MPI_CommWD2[myworld2] );

      /*
      for(i=0; i<3; i++){
        for(j=0; j<3; j++){
  	  printf("QQQ2 kloop=%2d i=%2d j=%2d %15.12f %15.12f\n",
                  kloop,i,j,Hs2[j*na_rows2+i].r,Hs2[j*na_rows2+i].i);
        }
      }
      */

      MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol2,my_prow2,&mpi_comm_rows);
      MPI_Comm_split(MPI_CommWD2[myworld2],my_prow2,my_pcol2,&mpi_comm_cols);

      mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
      mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

      if (scf_eigen_lib_flag==1){
        F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
        ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
      }
      else if (scf_eigen_lib_flag==2){

#ifndef kcomp
        int mpiworld;
        mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
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

      for (l=1; l<=MaxN; l++){
	EIGEN[0][kloop][l] = ko[l];
      }

      if (3<=level_stdout && 0<=kloop){
	printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n2; i1++){
	  printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
	}
      }

      /*
      printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	     myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
      for (i1=1; i1<=n2; i1++){
	printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
      }
      */

    } /* end of if (kloop0<num_kloop0) */

    if (measure_time){
      dtime(&Etime);
      time4 += Etime - Stime;
    }

    /**************************************************
      if (all_knum==1), wave functions are calculated. 
    **************************************************/

    if (measure_time) dtime(&Stime);

    if (all_knum==1){

      for(k=0; k<na_rows2*na_cols2; k++){
	Hs2[k].r = 0.0;
	Hs2[k].i = 0.0;
      }

      Cblacs_barrier(ictxt1_2,"A");
      F77_NAME(pzgemm,PZGEMM)( "T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,
                               &ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);

      /* MPI communications of Hs2 */

      for (ID=0; ID<numprocs2; ID++){
    
	IDS = (myid2 + ID) % numprocs2;
	IDR = (myid2 - ID + numprocs2) % numprocs2;

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

	  EVec1[0][m].r = EVec_Rcv[2*k  ];
	  EVec1[0][m].i = EVec_Rcv[2*k+1];
	}
      }

    } /* if (all_knum==1) */

    if (measure_time){
      dtime(&Etime);
      time5 += Etime - Stime;
    }

  } /* kloop0 */

  /****************************************************
     MPI:

     EIGEN
  ****************************************************/

  if (measure_time){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Stime);
  }

  for (kloop=0; kloop<T_knum; kloop++){
    /* get ID in the zeroth world */
    ID = Comm_World_StartID1[0] + T_k_ID[myworld1][kloop];
    MPI_Bcast(&EIGEN[0][kloop][0], MaxN+1, MPI_DOUBLE, ID, mpi_comm_level1);
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
      for (l=1; l<=MaxN; l++){

	x = (EIGEN[0][kloop][l] - ChemP)*Beta*0.2;

	if (x<=-x_cut) x = -x_cut;
	if (x_cut<=x)  x =  x_cut;
	FermiF = FermiFunc_NC(x,l);
	Num_State += FermiF*(double)T_k_op[kloop];
      } 
    } 

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
      for (l=1; l<=MaxN; l++){

	x = (EIGEN[0][kloop][l] - ChemP)*Beta;

	if (x<=-x_cut) x = -x_cut;
	if (x_cut<=x)  x =  x_cut;
	FermiF = FermiFunc_NC(x,l);

	Num_State += FermiF*(double)T_k_op[kloop];

      } 
    } 

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
    for (l=1; l<=MaxN; l++){

      x = (EIGEN[0][kloop][l] - ChemP)*Beta;

      if (x<=-x_cut) x = -x_cut;
      if (x_cut<=x)  x = x_cut;
      FermiF = FermiFunc_NC(x,l);

      Eele0[0] += FermiF*EIGEN[0][kloop][l]*(double)T_k_op[kloop];
    }
  } 

  Eele0[0] = Eele0[0]/sum_weights;
  Uele = Eele0[0];

  if (2<=level_stdout){
    printf("myid0=%2d ChemP=%lf, Eele0[0]=%lf, Eele0[1]=%lf\n",myid0,ChemP,Eele0[0],Eele0[1]);
  }

  if (measure_time){
    dtime(&Etime);
    time7 += Etime - Stime;
  }

  /****************************************************
       if all_knum==1, calculate CDM and EDM

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

  if (measure_time) dtime(&Stime);

  if (all_knum==1){

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
	    CDM[0][MA_AN][LB_AN][i][j] = 0.0;
	    CDM[1][MA_AN][LB_AN][i][j] = 0.0;
	    CDM[2][MA_AN][LB_AN][i][j] = 0.0;
	    CDM[3][MA_AN][LB_AN][i][j] = 0.0;
            EDM[0][MA_AN][LB_AN][i][j] = 0.0;
            EDM[1][MA_AN][LB_AN][i][j] = 0.0;
            EDM[2][MA_AN][LB_AN][i][j] = 0.0;
            EDM[3][MA_AN][LB_AN][i][j] = 0.0;
	    iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
	    iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
	  }
	}
      }
    }

    /* get k1, k2, and k3 */

    kloop = S_knum;

    k1 = T_KGrids1[kloop];
    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];

    /* calculate DM, iDM, and EDM */
    Calc_DM_Band_non_collinear( 1,1,
				myid0,myid2,size_H1,
				is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				CDM,iDM[0],EDM,EIGEN[0][kloop],
				H1,S1,EVec1[0],
				rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				rEDM11,rEDM22 );

  } /* if (all_knum==1) */

  if (measure_time){
    dtime(&Etime);
    time8 += Etime - Stime;
  }

  dtime(&EiloopTime);

  if (myid0==Host_ID && 0<level_stdout){
    printf("<Band_DFT>  Eigen, time=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
  }

  /****************************************************
   ****************************************************
     diagonalization for calculating density matrix
   ****************************************************
  ****************************************************/

  dtime(&SiloopTime);

  if (all_knum!=1){

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
	    CDM[0][MA_AN][LB_AN][i][j] = 0.0;
	    CDM[1][MA_AN][LB_AN][i][j] = 0.0;
	    CDM[2][MA_AN][LB_AN][i][j] = 0.0;
	    CDM[3][MA_AN][LB_AN][i][j] = 0.0;
            EDM[0][MA_AN][LB_AN][i][j] = 0.0;
            EDM[1][MA_AN][LB_AN][i][j] = 0.0;
            EDM[2][MA_AN][LB_AN][i][j] = 0.0;
            EDM[3][MA_AN][LB_AN][i][j] = 0.0;
	    iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
	    iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
	  }
	}
      }
    }

    /* for kloop */

    for (kloop0=0; kloop0<max_num_kloop0; kloop0++){

      /* get k1, k2, and k3 */

      if (kloop0<num_kloop0){

	kloop = S_knum + kloop0;
	k1 = T_KGrids1[kloop];
	k2 = T_KGrids2[kloop];
	k3 = T_KGrids3[kloop];
      }

      if (measure_time) dtime(&Stime);

      /* make Cs */

      Construct_Band_Ms(0,CntOLP,H1,S1,Cs,MP,k1,k2,k3);

      /* diagonalize Cs */

      if (kloop0<num_kloop0){

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

	/* print to the standard output */

	if (3<=level_stdout){
	  printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i=1; i<=n; i++){
	    printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	  }
	}

	/*
	printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i=1; i<=n; i++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
	}
	*/

	/* minus eigenvalues to 1.0e-10 */

	for (l=1; l<=n; l++){
	  if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
	  ko[l] = 1.0/sqrt(ko[l]);
	}

	/* calculate S*1/sqrt(ko) */

	for(i=0; i<na_rows; i++){
	  for(j=0; j<na_cols; j++){
	    jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	    Ss[j*na_rows+i].r = Ss[j*na_rows+i].r*ko[jg];
	    Ss[j*na_rows+i].i = Ss[j*na_rows+i].i*ko[jg];
	  }
	}

	/* make Ss2 */

	Overlap_Band_NC_Ss2( Ss, Ss2, MPI_CommWD2[myworld2] );

      } /* end of if (kloop0<num_kloop0) */

      if (measure_time){
        dtime(&Etime);
        time9 += Etime - Stime;
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
      
      if (measure_time) dtime(&Stime);
      
      /* set rHs and iHs */
      
      Construct_Band_Ms(0,nh[0],H1,S1,rHs11,MP,k1,k2,k3);
      Construct_Band_Ms(0,nh[1],H1,S1,rHs22,MP,k1,k2,k3);
      Construct_Band_Ms(0,nh[2],H1,S1,rHs12,MP,k1,k2,k3);
      Construct_Band_Ms(1,nh[3],H1,S1,iHs12,MP,k1,k2,k3);
      
      Construct_Band_Ms(1,ImNL[0],H1,S1,iHs11,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[1],H1,S1,iHs22,MP,k1,k2,k3);
      Construct_Band_Ms(1,ImNL[2],H1,S1,Cs,MP,k1,k2,k3);
      
      if (kloop0<num_kloop0){

	/* set Hs2 */

	for (i=0; i<na_rows*na_cols; i++){
	  iHs12[i].r += Cs[i].r;
	  iHs12[i].i += Cs[i].i;
	}

	for (i=0; i<na_rows*na_cols; i++){
	  rHs11[i].r += iHs11[i].r;  
	  rHs11[i].i += iHs11[i].i;  
	  rHs22[i].r += iHs22[i].r;  
	  rHs22[i].i += iHs22[i].i;  
	  rHs12[i].r += iHs12[i].r;  
	  rHs12[i].i += iHs12[i].i;
	}

	Hamiltonian_Band_NC_Hs2( rHs11, rHs22, rHs12, Hs2, MPI_CommWD2[myworld2] );

	/* S^t x rHs11 x S */

	for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"A");
	F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,rHs11,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	for (i=0; i<na_rows*na_cols; i++) rHs11[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"C");
	F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,rHs11,&ONE,&ONE,descH);

	/* S^t x rHs12 x S */

	for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"A");
	F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,rHs12,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	for (i=0; i<na_rows*na_cols; i++) rHs12[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"C");
	F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,rHs12,&ONE,&ONE,descH);

	/* S^t x rHs22 x S */

	for (i=0; i<na_rows*na_cols; i++) Cs[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"A");
	F77_NAME(pzgemm,PZGEMM)("N","N",&n,&n,&n,&alpha,rHs22,&ONE,&ONE,descH,Ss,&ONE,&ONE,descS,&beta,Cs,&ONE,&ONE,descC);

	for (i=0; i<na_rows*na_cols; i++) rHs22[i] = Complex(0.0,0.0);

	Cblacs_barrier(ictxt1,"C");
	F77_NAME(pzgemm,PZGEMM)("C","N",&n,&n,&n,&alpha,Ss,&ONE,&ONE,descS,Cs,&ONE,&ONE,descC,&beta,rHs22,&ONE,&ONE,descH);

        if (measure_time){
  	  dtime(&Etime);
	  time10 += Etime - Stime;
	}

	/* ***************************************************
	   diagonalize the transformed H
	   *************************************************** */

	if (measure_time) dtime(&Stime);

	Hamiltonian_Band_NC_Hs2( rHs11, rHs22, rHs12, Hs2, MPI_CommWD2[myworld2] );

	/*
	  for(i=0; i<3; i++){
	  for(j=0; j<3; j++){
	  printf("QQQ2 kloop=%2d i=%2d j=%2d %15.12f %15.12f\n",kloop,i,j,Hs2[j*na_rows2+i].r,Hs2[j*na_rows2+i].i);
	  }
	  }
	*/

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol2,my_prow2,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow2,my_pcol2,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);
  
        if (scf_eigen_lib_flag==1){
  	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}
        else if (scf_eigen_lib_flag==2){

#ifndef kcomp
          int mpiworld;
          mpiworld = MPI_Comm_c2f(MPI_CommWD2[myworld2]);
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

	for (l=1; l<=MaxN; l++){
	  EIGEN[0][kloop][l] = ko[l];
	}

	if (3<=level_stdout && 0<=kloop && kloop0<num_kloop0){
	  printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
		 myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	  for (i1=1; i1<=n2; i1++){
	    printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
	  }
	}

	/*
	if (0<=kloop && kloop0<num_kloop0){

	printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=n2; i1++){
	  printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
	}
	}
	*/

        if (measure_time){
          dtime(&Etime);
          time11 += Etime - Stime;
	}

        /**************************************************
                  calculation of wave functions
        **************************************************/

	for(k=0; k<na_rows2*na_cols2; k++){
	  Hs2[k].r = 0.0;
	  Hs2[k].i = 0.0;
	}

	Cblacs_barrier(ictxt1_2,"A");
	F77_NAME(pzgemm,PZGEMM)( "T","T",&n2,&n2,&n2,&alpha,Cs2,&ONE,&ONE,descC2,Ss2,
				 &ONE,&ONE,descS2,&beta,Hs2,&ONE,&ONE,descH2);

	/* MPI communications of Hs2 */

        ID = 0;
	IDS = (myid2 + ID) % numprocs2;
	IDR = (myid2 - ID + numprocs2) % numprocs2;

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

	for(k=0; k<Num_Rcv_EV[IDR]; k++){
	  ig = index_Snd_i[k];
	  jg = index_Snd_j[k];
	  m = (jg-1)*(ie2[myid2]-is2[myid2]+1)+ig-is2[myid2];

	  EVec1[0][m].r = EVec_Snd[2*k  ];
	  EVec1[0][m].i = EVec_Snd[2*k+1];
	}

      } /* end of if (kloop0<num_kloop0) */

      /****************************************************
                     calculate DM and EDM
      ****************************************************/

      if (measure_time) dtime(&Stime);

      /* calculate DM and iDM */

      Calc_DM_Band_non_collinear( (kloop0<num_kloop0),0,
				  myid0,myid2,size_H1,
				  is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				  CDM,iDM[0],EDM,EIGEN[0][kloop],
				  H1,S1,EVec1[0],
				  rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				  rEDM11,rEDM22 );

      if (measure_time){
        dtime(&Etime);
        time12 += Etime - Stime;
      }

    } /* kloop0 */

    /* store DM and iDM */

    Calc_DM_Band_non_collinear( 0,1,
				myid0,myid2,size_H1,
				is2,ie2,MP,n,n2,MaxN,k1,k2,k3, 
				CDM,iDM[0],EDM,EIGEN[0][kloop],
				H1,S1,EVec1[0],
				rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				rEDM11,rEDM22 );

  } /* if (all_knum!=1) */

  /****************************************************
           normalization of CDM, EDM, and iDM 
  ****************************************************/

  dtime(&EiloopTime);

  if (myid0==Host_ID && 0<level_stdout){
    printf("<Band_DFT>  DM, time=%lf\n", EiloopTime-SiloopTime);fflush(stdout);
  }

  dum = 1.0/sum_weights;

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
	  CDM[0][MA_AN][LB_AN][i][j] = CDM[0][MA_AN][LB_AN][i][j]*dum;
	  CDM[1][MA_AN][LB_AN][i][j] = CDM[1][MA_AN][LB_AN][i][j]*dum;
	  CDM[2][MA_AN][LB_AN][i][j] = CDM[2][MA_AN][LB_AN][i][j]*dum;
	  CDM[3][MA_AN][LB_AN][i][j] = CDM[3][MA_AN][LB_AN][i][j]*dum;
	  EDM[0][MA_AN][LB_AN][i][j] = EDM[0][MA_AN][LB_AN][i][j]*dum;
	  EDM[1][MA_AN][LB_AN][i][j] = EDM[1][MA_AN][LB_AN][i][j]*dum;
	  EDM[2][MA_AN][LB_AN][i][j] = EDM[2][MA_AN][LB_AN][i][j]*dum;
	  EDM[3][MA_AN][LB_AN][i][j] = EDM[3][MA_AN][LB_AN][i][j]*dum;
	  iDM[0][0][MA_AN][LB_AN][i][j] = iDM[0][0][MA_AN][LB_AN][i][j]*dum;
	  iDM[0][1][MA_AN][LB_AN][i][j] = iDM[0][1][MA_AN][LB_AN][i][j]*dum;
	}
      }
    }
  }

  /****************************************************
                       bond-energies
  ****************************************************/

  My_Eele1[0] = 0.0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    for (j=0; j<=FNAN[GA_AN]; j++){
      GB_AN = natn[GA_AN][j];  
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];

      /* non-spin-orbit coupling and non-LDA+U */  
      if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
          && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){
        for (k=0; k<tnoA; k++){
	  for (l=0; l<tnoB; l++){
            My_Eele1[0] = My_Eele1[0]
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
            My_Eele1[0] = My_Eele1[0]
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

  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&My_Eele1[0], &Eele1[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  Eele1[1] = 0.0;

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

      setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp_EV,"\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"           Eigenvalues (Hartree) of SCF KS-eq.           \n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n\n");
      fprintf(fp_EV,"   Chemical Potential (Hatree) = %18.14f\n",ChemP);
      fprintf(fp_EV,"   Number of States            = %18.14f\n",Num_State);
      fprintf(fp_EV,"   Eigenvalues\n\n");

      for (kloop=0; kloop<T_knum; kloop++){

	if (0<T_k_op[kloop]){

	  fprintf(fp_EV,"\n");
	  fprintf(fp_EV,"   kloop=%i\n",kloop);
	  fprintf(fp_EV,"   k1=%10.5f k2=%10.5f k3=%10.5f\n\n",
		  T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);

	  for (l=1; l<=MaxN; l++){
	    fprintf(fp_EV,"%5d  %18.14f\n",l,EIGEN[0][kloop][l]);
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

  free(rDM11);
  free(rDM22);
  free(rDM12);
  free(iDM11);
  free(iDM22);
  free(iDM12);
  free(rEDM11);
  free(rEDM22);

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
    printf("myid0=%2d time9 =%9.4f\n",myid0,time9);fflush(stdout);
    printf("myid0=%2d time10=%9.4f\n",myid0,time10);fflush(stdout);
    printf("myid0=%2d time11=%9.4f\n",myid0,time11);fflush(stdout);
    printf("myid0=%2d time12=%9.4f\n",myid0,time12);fflush(stdout);
    printf("myid0=%2d time13=%9.4f\n",myid0,time12);fflush(stdout);
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}




void Construct_Band_Ms( int cpx_flag, double ****Mat, double *M1, double *M2, dcomplex *Ms, 
                        int *MP, double k1, double k2, double k3)
{
  static int firsttime=1;
  int i,j,k;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN,Rn,l1,l2,l3;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int num,tnum,num_orbitals;
  int ID,myid,numprocs,tag=999;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double sum,kRn,si,co;
  double Stime,Etime;
  double AStime,AEtime;
  MPI_Status stat;
  MPI_Request request;
  int ig,jg,il,jl,prow,pcol,brow,bcol;

  if (measure_time){
    dtime(&AStime);
    dtime(&Stime);
  }

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* allocation of arrays */

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);
  order_GA = (int*)malloc(sizeof(int)*(atomnum+2));

  if (firsttime && memoryusage_fileout) {
  PrintMemory("Construct_Band_Ms: My_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: SP_NZeros", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: SP_Atoms", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: is1", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: ie1", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: is2", sizeof(int)*numprocs,NULL);
  PrintMemory("Band_DFT_NonCol: order_GA", sizeof(int)*(atomnum+2),NULL);
  }
  firsttime = 1;

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /* set is2 and order_GA */

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }
  
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;

  /* set M1 */

  for (i=0; i<tnum; i++) M1[i] = 0.0;

  k = is1[myid];
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){
          M1[k] = Mat[MA_AN][LB_AN][i][j]; 
          k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB1 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* MPI M1 */

  MPI_Allreduce(&M1[0], &M2[0], tnum, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  if (measure_time){
    dtime(&Etime);
    printf("timeB2 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  if (measure_time){
    dtime(&Stime);
  }

  /* M2 -> Ms */
  
  for(i=0; i<na_rows*na_cols; i++){
    Ms[i].r = 0.0;
    Ms[i].i = 0.0;
  }

  k = 0;
  for (AN=1; AN<=atomnum; AN++){
    GA_AN = order_GA[AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];

    for (i=0; i<tnoA; i++){

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

	for (j=0; j<tnoB; j++){
	  ig = Anum+i;
	  jg = Bnum+j;
	    
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

            if (cpx_flag==0){
	      Ms[(jl-1)*na_rows+il-1].r += M2[k]*co;
	      Ms[(jl-1)*na_rows+il-1].i += M2[k]*si;
	    }
            else if (cpx_flag==1){
	      Ms[(jl-1)*na_rows+il-1].r -= M2[k]*si;
	      Ms[(jl-1)*na_rows+il-1].i += M2[k]*co;
	    }
	  }
	    
	  k++;
	}
      }
    }
  }

  if (measure_time){
    dtime(&Etime);
    printf("timeB3 myid=%2d %15.12f\n",myid,Etime-Stime);
  }

  /* freeing of arrays */

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);

  if (measure_time){
    dtime(&AEtime);
    printf("timeB_all myid=%2d %15.12f\n",myid,AEtime-AStime);
  }
}




double Calc_DM_Band_non_collinear(
    int calc_flag,
    int store_flag,
    int myid0,
    int myid2,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    int MaxN,
    double k1,
    double k2,
    double k3,
    double *****CDM,
    double *****iDM0,
    double *****EDM,
    double *ko,
    double *DM1,
    double *Work1,
    dcomplex *EVec1,
    double *rDM11,
    double *rDM22,
    double *rDM12,
    double *iDM12,
    double *iDM11,
    double *iDM22, 
    double *rEDM11,
    double *rEDM22)
{
  int i,j,k,po,p,GA_AN,MA_AN,wanA,tnoA,Anum,Rn,kmin,kmax;
  int LB_AN,GB_AN,wanB,tnoB,Bnum,i1,j0,j1,i2,j2,ID,l1,l2,l3;
  double max_x=60.0,dum,co,si,kRn,tmp1,tmp2;
  double FermiF,x,x2,ReA,ReB,ReC,ImA,ImB,ImC;
  double d1,d2,d3,d4,d5,d6,d7,d8,d9,d10;
  double FermiEps = 1.0e-13;
  double Stime,Etime,stime,etime,time,lumos;
  MPI_Status stat;
  MPI_Request request;

  dtime(&stime);

  if (measure_time){
    dtime(&Stime);
  }

  /******************************
      calculation of DM, EDM 
  *******************************/ 

  if (calc_flag==1){

    /* pre-calculation of the Fermi function */
    
    po = 0;
    kmin = is2[myid2];
    kmax = ie2[myid2];
 
    for (k=is2[myid2]; k<=ie2[myid2]; k++){

      x = (ko[k] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = FermiFunc_NC(x,k); 
      tmp1 = sqrt(FermiF);

      for (i1=1; i1<=n2; i1++){
	i = (i1-1)*(ie2[myid2]-is2[myid2]+1) + k - is2[myid2];
	EVec1[i].r *= tmp1;
	EVec1[i].i *= tmp1;
      }

      /* find kmax */

      if ( FermiF<FermiEps && po==0 ) {
        kmax = k;
        po = 1;         
      }
    }    

    /* loop for GA_AN */
    
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

	  i1 = (Anum + i - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];
	  i2 = (Anum + i + n - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];

	  for (j=0; j<tnoB; j++){

	    j1 = (Bnum + j - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];
	    j2 = (Bnum + j + n - 1)*(ie2[myid2]-is2[myid2]+1) - is2[myid2];

	    d1 = 0.0;
	    d2 = 0.0;
	    d3 = 0.0;
	    d4 = 0.0;
	    d5 = 0.0;
	    d6 = 0.0;
	    d7 = 0.0;
	    d8 = 0.0;
	    d9 = 0.0;
	    d10= 0.0;

	    for (k=kmin; k<=kmax; k++){
	      ReA = EVec1[i1+k].r*EVec1[j1+k].r + EVec1[i1+k].i*EVec1[j1+k].i; 
	      d1 += ReA;
	      d7 += ReA*ko[k];
	    }

	    for (k=kmin; k<=kmax; k++){
	      ImA = EVec1[i1+k].r*EVec1[j1+k].i - EVec1[i1+k].i*EVec1[j1+k].r;
	      d2 += ImA;
	      d8 += ImA*ko[k];
	    }

	    for (k=kmin; k<=kmax; k++){
	      ReB = EVec1[i2+k].r*EVec1[j2+k].r + EVec1[i2+k].i*EVec1[j2+k].i;
	      d3 += ReB;
	      d9 += ReB*ko[k];
	    }

	    for (k=kmin; k<=kmax; k++){
	      ImB = EVec1[i2+k].r*EVec1[j2+k].i - EVec1[i2+k].i*EVec1[j2+k].r;
	      d4  += ImB;
	      d10 += ImB*ko[k];
	    }

	    for (k=kmin; k<=kmax; k++){
	      ReC = EVec1[i1+k].r*EVec1[j2+k].r + EVec1[i1+k].i*EVec1[j2+k].i;
	      d5 += ReC;
	    }

	    for (k=kmin; k<=kmax; k++){
	      ImC = EVec1[i1+k].r*EVec1[j2+k].i - EVec1[i1+k].i*EVec1[j2+k].r; 
	      d6 += ImC;
	    }

	    /* Re DM11 */
	    rDM11[p] += co*d1 - si*d2; 

	    /* Re DM22 */
	    rDM22[p] += co*d3 - si*d4;

	    /* Re DM12 */
	    rDM12[p] += co*d5 - si*d6; 

	    /* Im DM12 */
	    iDM12[p] += co*d6 + si*d5;

	    /* Im DM11 */
	    iDM11[p] += co*d2 + si*d1;

	    /* Im DM22 */
	    iDM22[p] += co*d4 + si*d3;

	    /* ReEDM11 */
	    rEDM11[p] += co*d7 - si*d8;

	    /* rEDM22 */
	    rEDM22[p] += co*d9 - si*d10;

	    /* increment of p */
	    p++;  

	  }
	}
      }
    } /* GA_AN */

  } /* if (calc_flag==1) */

  if (measure_time){
    dtime(&Etime);
    printf("timeA1 myid0=%2d myid2=%2d ie2-is2+1=%2d  %15.12f\n",
            myid0,myid2,ie2[myid2]-is2[myid2]+1,Etime-Stime);
  }

  /***********************************
     store the data to proper arrays
  ************************************/ 

  if (store_flag==1){

    /* MPI_Allreduce */

    if (measure_time){
      dtime(&Stime);
    }

    MPI_Allreduce(rDM11, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    for (i=0; i<size_H1; i++) rDM11[i] = Work1[i];

    MPI_Allreduce(rDM22, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    for (i=0; i<size_H1; i++) rDM22[i] = Work1[i];

    MPI_Allreduce(rDM12, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    for (i=0; i<size_H1; i++) rDM12[i] = Work1[i];

    MPI_Allreduce(iDM11, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    for (i=0; i<size_H1; i++) iDM11[i] = Work1[i];

    MPI_Allreduce(iDM22, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    for (i=0; i<size_H1; i++) iDM22[i] = Work1[i];

    MPI_Allreduce(iDM12, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    for (i=0; i<size_H1; i++) iDM12[i] = Work1[i];
   
    MPI_Allreduce(rEDM11, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    for (i=0; i<size_H1; i++) rEDM11[i] = Work1[i];

    MPI_Allreduce(rEDM22, Work1, size_H1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    for (i=0; i<size_H1; i++) rEDM22[i] = Work1[i];

    if (measure_time){
      dtime(&Etime);
      printf("timeA2 %15.12f\n",Etime-Stime);
    }

    /* store DM1 to a proper place */

    if (measure_time){
      dtime(&Stime);
    }

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

	      CDM[0][MA_AN][LB_AN][i][j] += rDM11[p];  /* Re11 */ 
	      CDM[1][MA_AN][LB_AN][i][j] += rDM22[p];  /* Re22 */ 
	      CDM[2][MA_AN][LB_AN][i][j] += rDM12[p];  /* Re12 */ 
	      CDM[3][MA_AN][LB_AN][i][j] += iDM12[p];  /* Im12 */ 
	      iDM0[0][MA_AN][LB_AN][i][j] += iDM11[p]; /* Im11 */ 
	      iDM0[1][MA_AN][LB_AN][i][j] += iDM22[p]; /* Im22 */ 

	      EDM[0][MA_AN][LB_AN][i][j] += rEDM11[p]; /* ReEDM11 */ 
	      EDM[1][MA_AN][LB_AN][i][j] += rEDM22[p]; /* ReEDM22 */ 

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
      dtime(&Etime);
      printf("timeA3 %15.12f\n",Etime-Stime);
    }

  } /* if (store_flag==1) */

  dtime(&etime);
  return (etime-stime);
}
