/**********************************************************************
  Band_DFT_NonCol_Optical.c:

  Band_DFT_NonCol_Optical.c is a subroutine to calculate optical
  conductivities and dielectric functions based on a non-collinear DFT

  Log of Band_DFT_NonCol_Optical.c:

     13/Sep./2018  Released by Yung-Ting Lee

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

void get_LCAO_Coef(dcomplex** C2, dcomplex** LCAO,dcomplex** S,int n);



double Band_DFT_NonCol_Optical(int SCF_iter,
			       double *koS,
			       dcomplex **S,
			       int knum_i, int knum_j, int knum_k,
			       int SpinP_switch,
			       double *****nh,
			       double *****ImNL,
			       double ****CntOLP,
			       double *****CDM,
			       double *****EDM,
			       double Eele0[2], double Eele1[2])
{
  static int firsttime=1;
  int i,j,k,l,n,n2,wan;
  int ii1,m,mn,lmn,lmax;
  int i1,j1,jj1,jj2,po,spin,n1;
  int ia,ib,ja,jb;
  int num2,RnB,l1,l2,l3,loop_num;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int MA_AN,GA_AN,Anum;
  double time0,sum_weights;
  double Imdum2,Imsum,ImsumE;
  int LB_AN,GB_AN,Bnum;
  double EV_cut0;
  double snum_i,snum_j,snum_k,k1,k2,k3,sum,sumi,Num_State,FermiF;
  double sum_r0,sum_i0;
  double sum_r1,sum_i1;
  double sum_r00,sum_i00;
  double sum_r01,sum_i01;
  double sum_r10,sum_i10;
  double sum_r11,sum_i11;
  double av_num;
  double x,Dnum,Dnum2,AcP,ChemP_MAX,ChemP_MIN;
  double **EIGEN;
  double d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12;
  double *VecFkw;
  double *VecFkwE;
  dcomplex **H,**C;
  dcomplex Ctmp1,Ctmp2;
  int ***k_op,*T_k_op,*T_k_ID;
  int ii,ij,ik;
  int S_knum,E_knum,T_knum,e1,s1,kloop,kloop0,num_kloop0;
  double *KGrids1,*KGrids2,*KGrids3;
  double *T_KGrids1,*T_KGrids2,*T_KGrids3;
  double u2,v2,uv,vu,tmp,tmp1;
  double My_Eele1[2]; 
  double kw,eig,Fkw;
  double TZ,dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2,Imdum;

  double ResumA,ResumEA,RedumA,Redum2A,ImdumA;
  double ResumB,RedumB,Redum2B,ImdumB;
  double ResumC,RedumC,Redum2C,ImdumC;
  double Imdum2D,ImsumED;
  double u2A,v2A,uvA,vuA;
  double u2B,v2B,uvB,vuB;
  double u2C,v2C,uvC,vuC;

  double TStime,TEtime,SiloopTime,EiloopTime;
  double Stime,Etime;
  double FermiEps=1.0e-14;
  double x_cut=30.0;
  char file_EV[YOUSO10];
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */

  int *MP;
  int *order_GA;
  double *ko;
  double *S1;
  double *RH0;
  double *RH1;
  double *RH2;
  double *RH3;
  double *IH0;
  double *IH1;
  double *IH2;

  int *is1,*ie1;
  int *is12,*ie12;
  int *is2,*ie2;
  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;
  double lumos;
  int Rn,AN;
  int size_H1;
  int all_knum;
  int MaxN,ks,i0;
  int ID,ID0,ID1;
  int numprocs0,myid0;
  int numprocs1,myid1;
  int myworld1;
  int parallel_mode;
  int Num_Comm_World1;
  int *NPROCS_ID1;
  int *NPROCS_WD1;
  int *Comm_World1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;

  double time1,time2,time3;
  double time4,time5,time6;
  double time7,time8,time9;
  double time10,time11,time12;
  double time13,time14,time15;
  double time16,time17,time18;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  /* YTL-start */
  double* fd_dist;
  dcomplex** C2;
  dcomplex** LCAO;
  /* YTL-end */

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
  time14 = 0.0;
  time15 = 0.0;
  time16 = 0.0;
  time17 = 0.0;
  time18 = 0.0;

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n += Spe_Total_CNO[wanA];
  }
  n2 = 2*n + 2;

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

  lumos = (double)2*n*0.100;
  if (lumos<60.0) lumos = 400.0;
  MaxN = (TZ-system_charge) + (int)lumos;
  if ( (2*n)<MaxN ) MaxN = 2*n;

  /****************************************************
   Allocation

   int       MP[List_YOUSO[1]]
   double    ko[n2]
   double    koS[n+1];
   dcomplex  H[n2][n2]
   dcomplex  S[n2][n2]
   double    M1[n2]
   dcomplex  C[n2][n2]
   double    EIGEN[T_knum]
                  [n2] 
   double    KGrids1[knum_i]
   double    KGrids2[knum_j]
   double    KGrids3[knum_k]
   int       k_op[knum_i][knum_j][knum_k]
  ****************************************************/

  VecFkw = (double*)malloc(sizeof(double)*n2);
  VecFkwE = (double*)malloc(sizeof(double)*n2);

  H = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (j=0; j<n2; j++){
    H[j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  C = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (j=0; j<n2; j++){
    C[j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  KGrids1 = (double*)malloc(sizeof(double)*knum_i);
  KGrids2 = (double*)malloc(sizeof(double)*knum_j);
  KGrids3 = (double*)malloc(sizeof(double)*knum_k);

  k_op=(int***)malloc(sizeof(int**)*knum_i);
  for (i=0;i<knum_i; i++) {
    k_op[i]=(int**)malloc(sizeof(int*)*knum_j);
    for (j=0;j<knum_j; j++) {
      k_op[i][j]=(int*)malloc(sizeof(int)*knum_k);
    }
  }
  /*  k_op[0:knum_i-1][0:knum_j-1][0:knum_k-1] */

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);
  order_GA = (int*)malloc(sizeof(int)*(List_YOUSO[1]+1));

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  ko = (double*)malloc(sizeof(double)*n2);

  /****************************************************
    PrintMemory
  ****************************************************/
/*
  if (firsttime){

    PrintMemory("Band_DFT: ko",sizeof(double)*n2,NULL);
    PrintMemory("Band_DFT: H",sizeof(dcomplex)*n2*n2,NULL);
    PrintMemory("Band_DFT: S",sizeof(dcomplex)*n2*n2,NULL);
    PrintMemory("Band_DFT: C",sizeof(dcomplex)*n2*n2,NULL);
    PrintMemory("Band_DFT: M1",sizeof(double)*n2,NULL);

    PrintMemory("Band_DFT: EIGEN",
          sizeof(double)*List_YOUSO[27]*
                         List_YOUSO[28]*
                         List_YOUSO[29]*n2,NULL);
    PrintMemory("Band_DFT: ko_op",
          sizeof(int)*knum_i*knum_j*knum_k,NULL);
  }
*/
  /* for PrintMemory */
  firsttime=0;

  /***********************************************
              k-points by regular mesh 
  ***********************************************/

  if (way_of_kpoint==1){

    snum_i = (double)knum_i;
    snum_j = (double)knum_j;
    snum_k = (double)knum_k;

    for (i=0; i<knum_i; i++){
      if (knum_i==1){
	k1 = 0.0;
      }
      else {
	k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*snum_i) + Shift_K_Point;
      }
      KGrids1[i]=k1;
    }

    for (i=0; i<knum_j; i++){
      if (knum_j==1){
	k1 = 0.0;
      }
      else {
	k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*snum_j) - Shift_K_Point;
      }
      KGrids2[i] = k1;
    }

    for (i=0; i<knum_k; i++){
      if (knum_k==1){
	k1 = 0.0;
      }
      else {
	k1 = -0.5 + (2.0*(double)i+1.0)/(2.0*snum_k) + 2.0*Shift_K_Point;
      }
      KGrids3[i]=k1;
    }

    if (myid0==Host_ID && 0<level_stdout){
      printf(" CDDF.KGrids1: ");
      for (i=0;i<=knum_i-1;i++) printf("%9.5f ",KGrids1[i]);
      printf("\n");
      printf(" CDDF.KGrids2: ");
      for (i=0;i<=knum_j-1;i++) printf("%9.5f ",KGrids2[i]);
      printf("\n");
      printf(" CDDF.KGrids3: ");
      for (i=0;i<=knum_k-1;i++) printf("%9.5f ",KGrids3[i]);
      printf("\n");
    }

    /**************************************************************
     k_op[i][j][k]: weight of DOS 
                 =0   no calc.
                 =1   G-point
             note that there is no simple inversion symmetry for
               the wave functions in noncollinear case 
                 =1  other  
    *************************************************************/

    for (i=0; i<knum_i; i++) {
      for (j=0; j<knum_j; j++) {
	for (k=0; k<knum_k; k++) {
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
	      /*
		printf("* : %d %d %d (%f %f %f)\n",i,j,k,
		KGrids1[i], KGrids2[j], KGrids3[k]);
	      */
	      k_op[i][j][k]    = 1;
	    }

	    else {

	      /* note that there is no simple inversion symmetry for
		 the wave functions in noncollinear case */

	      k_op[i][j][k]    = 1;
	      k_op[ii][ij][ik] = 1;
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

    /* YTL-start */

    /* Step (1) : calculate < phi( atom a, orbital alpha) | nabla | phi( atom b, orbital beta) > */
    Calc_NabraMatrixElements();

    Initialize_optical();

    /* YTL-end */

    /* allocation of arrays */

    T_KGrids1 = (double*)malloc(sizeof(double)*T_knum);
    T_KGrids2 = (double*)malloc(sizeof(double)*T_knum);
    T_KGrids3 = (double*)malloc(sizeof(double)*T_knum);
    T_k_op    = (int*)malloc(sizeof(int)*T_knum);
    T_k_ID    = (int*)malloc(sizeof(int)*T_knum);

    EIGEN = (double**)malloc(sizeof(double*)*T_knum);
    for (j=0; j<T_knum; j++){
      EIGEN[j] = (double*)malloc(sizeof(double)*n2);
      for (i=0; i<n2; i++)  EIGEN[j][i] = 1.0e+5;
    }

    /* set T_KGrid1,2,3 and T_k_op */

    T_knum = 0;
    for (i=0; i<=(knum_i-1); i++){
      for (j=0; j<=(knum_j-1); j++){
	for (k=0; k<=(knum_k-1); k++){
	  if (0<k_op[i][j][k]){

	    T_KGrids1[T_knum] = KGrids1[i];
	    T_KGrids2[T_knum] = KGrids2[j];
	    T_KGrids3[T_knum] = KGrids3[k];

	    T_k_op[T_knum]    = k_op[i][j][k];

	    T_knum++;
	  }
	}
      }
    }
  }

  /***********************************************
                 Monkhorst-Pack k-points 
  ***********************************************/

  else if (way_of_kpoint==2){

    T_knum = num_non_eq_kpt; 

    T_KGrids1 = (double*)malloc(sizeof(double)*T_knum);
    T_KGrids2 = (double*)malloc(sizeof(double)*T_knum);
    T_KGrids3 = (double*)malloc(sizeof(double)*T_knum);
    T_k_op    = (int*)malloc(sizeof(int)*T_knum);
    T_k_ID    = (int*)malloc(sizeof(int)*T_knum);

    EIGEN = (double**)malloc(sizeof(double*)*T_knum);
    for (j=0; j<T_knum; j++){
      EIGEN[j] = (double*)malloc(sizeof(double)*n2);
      for (i=0; i<n2; i++)  EIGEN[j][i] = 1.0e+5;
    }
   
    for (k=0; k<T_knum; k++){
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
         allocate k-points into proccessors 
  ***********************************************/

  if (numprocs0<T_knum){

    /* set parallel_mode */
    parallel_mode = 0;

    /* allocation of kloop to ID */     

    for (ID=0; ID<numprocs0; ID++){
      tmp = (double)T_knum/(double)numprocs0;
      S_knum = (int)((double)ID*(tmp+1.0e-12)); 
      E_knum = (int)((double)(ID+1)*(tmp+1.0e-12)) - 1;
      if (ID==(numprocs0-1)) E_knum = T_knum - 1;
      if (E_knum<0)          E_knum = 0;

      for (k=S_knum; k<=E_knum; k++){
        /* ID in the zeroth level world */
        T_k_ID[k] = ID;
      }
    }

    /* find own informations */

    tmp = (double)T_knum/(double)numprocs0; 
    S_knum = (int)((double)myid0*(tmp+1.0e-12)); 
    E_knum = (int)((double)(myid0+1)*(tmp+1.0e-12)) - 1;
    if (myid0==(numprocs0-1)) E_knum = T_knum - 1;
    if (E_knum<0)             E_knum = 0;

    num_kloop0 = E_knum - S_knum + 1;

  }

  else {

    /* set parallel_mode */
    parallel_mode = 1;
    num_kloop0 = 1;

    Num_Comm_World1 = T_knum;

    NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs0);
    Comm_World1 = (int*)malloc(sizeof(int)*numprocs0);
    NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
    Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
    MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);

    Make_Comm_Worlds(mpi_comm_level1, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1, 
		     NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

    MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

    S_knum = myworld1;

    /* allocate k-points into processors */
    
    for (k=0; k<T_knum; k++){
      /* ID in the first level world */
      T_k_ID[k] = Comm_World_StartID1[k];
    }

    /* YTL-start */
    Set_MPIworld_for_optical(myid1,numprocs1);
    /* YTL-end */
  }

  /****************************************************
   find all_knum
   if (all_knum==1), all the calculation will be made 
   by the first diagonalization loop, and the second 
   diagonalization will be skipped. 
  ****************************************************/

  MPI_Allreduce(&num_kloop0, &all_knum, 1, MPI_INT, MPI_PROD, mpi_comm_level1);

  if (all_knum==1){

    /* allocation */ 

    stat_send = malloc(sizeof(MPI_Status)*numprocs1);
    request_send = malloc(sizeof(MPI_Request)*numprocs1);
    request_recv = malloc(sizeof(MPI_Request)*numprocs1);

    is1 = (int*)malloc(sizeof(int)*numprocs1);
    ie1 = (int*)malloc(sizeof(int)*numprocs1);

    is12 = (int*)malloc(sizeof(int)*numprocs1);
    ie12 = (int*)malloc(sizeof(int)*numprocs1);

    is2 = (int*)malloc(sizeof(int)*numprocs1);
    ie2 = (int*)malloc(sizeof(int)*numprocs1);

    /* make is1 and ie1, is12 and ie12 */ 

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

    for (ID=0; ID<numprocs1; ID++){
      is12[ID] = 2*is1[ID] - 1;
      ie12[ID] = 2*ie1[ID];
    }

    /* make is2 and ie2 */ 

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
	ie2[ID] = -2;
      }
    }

  } /* if (all_knum==1) */

  /****************************************************
     store in each processor all the matrix elements
        for overlap and Hamiltonian matrices
  ****************************************************/

  dtime(&Stime);

  /* get size_H1 */

  size_H1 = Get_OneD_HS_Col(0, CntOLP, &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  S1 = (double*)malloc(sizeof(double)*size_H1);
  RH0 = (double*)malloc(sizeof(double)*size_H1);
  RH1 = (double*)malloc(sizeof(double)*size_H1);
  RH2 = (double*)malloc(sizeof(double)*size_H1);
  RH3 = (double*)malloc(sizeof(double)*size_H1);
  IH0 = (double*)malloc(sizeof(double)*size_H1);
  IH1 = (double*)malloc(sizeof(double)*size_H1);
  IH2 = (double*)malloc(sizeof(double)*size_H1);

  /* set S1, RH0, RH1, RH2, RH3, IH0, IH1, IH2 */

  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1,  MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[0],  RH0, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[1],  RH1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[2],  RH2, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[3],  RH3, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
       && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){  
    
    /* nothing is done. */
  }
  else {
    size_H1 = Get_OneD_HS_Col(1, ImNL[0], IH0, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, ImNL[1], IH1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, ImNL[2], IH2, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  dtime(&Etime);
  time1 += Etime - Stime; 

  /****************************************************
                      start kloop
  ****************************************************/

  dtime(&SiloopTime);

  for (kloop0=0; kloop0<num_kloop0; kloop0++){

    dtime(&Stime);

    kloop = S_knum + kloop0;

    k1 = T_KGrids1[kloop];
    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];

    /* make S and H */

    if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){

      for (i=1; i<=n; i++){
        for (j=1; j<=n; j++){
	  S[i][j] = Complex(0.0,0.0);
        } 
      } 
    }

    for (i=1; i<=2*n; i++){
      for (j=1; j<=2*n; j++){
	H[i][j] = Complex(0.0,0.0);
      } 
    } 

    /* non-spin-orbit coupling and non-LDA+U */
    if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
	&& Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){  

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

	      H[Anum+i  ][Bnum+j  ].r += co*RH0[k];
	      H[Anum+i  ][Bnum+j  ].i += si*RH0[k];

	      H[Anum+i+n][Bnum+j+n].r += co*RH1[k];
	      H[Anum+i+n][Bnum+j+n].i += si*RH1[k];
            
	      H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*RH3[k];
	      H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*RH3[k];

	      k++;

	    }

            if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){
              
              k -= tnoB; 

	      for (j=0; j<tnoB; j++){

	        S[Anum+i  ][Bnum+j  ].r += co*S1[k];
	        S[Anum+i  ][Bnum+j  ].i += si*S1[k];

		k++;
	      }
	    }

	  }
	}
      }
    }

    /* spin-orbit coupling or LDA+U */
    else {  

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

	      H[Anum+i  ][Bnum+j  ].r += co*RH0[k] - si*IH0[k];
	      H[Anum+i  ][Bnum+j  ].i += si*RH0[k] + co*IH0[k];

	      H[Anum+i+n][Bnum+j+n].r += co*RH1[k] - si*IH1[k];
	      H[Anum+i+n][Bnum+j+n].i += si*RH1[k] + co*IH1[k];
            
	      H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*(RH3[k]+IH2[k]);
	      H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*(RH3[k]+IH2[k]);

	      k++;

	    }

            if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){

	      k -= tnoB; 

	      for (j=0; j<tnoB; j++){

		S[Anum+i  ][Bnum+j  ].r += co*S1[k];
		S[Anum+i  ][Bnum+j  ].i += si*S1[k];

		k++;
	      }
	    }

	  }
	}
      }
    }

    /* set off-diagonal part */

    for (i=1; i<=n; i++){
      for (j=1; j<=n; j++){
	H[j+n][i].r = H[i][j+n].r;
	H[j+n][i].i =-H[i][j+n].i;
      } 
    } 

    dtime(&Etime);
    time2 += Etime - Stime; 

    /* diagonalize S */

    dtime(&Stime);

    if (parallel_mode==0){
      EigenBand_lapack(S,ko,n,n,1);
    }
    else if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){
      Eigen_PHH(MPI_CommWD1[myworld1],S,ko,n,n,1);
    }

    dtime(&Etime);
    time3 += Etime - Stime; 

    if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1 || all_knum!=1){
/*
      if (3<=level_stdout){
	printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i=1; i<=n; i++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i,ko[i]);
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

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  S[i1][j1].r = S[i1][j1].r*ko[j1];
	  S[i1][j1].i = S[i1][j1].i*ko[j1];
	} 
      } 
    }

    /****************************************************
                  set H' and diagonalize it
    ****************************************************/

    /* U'^+ * H * U * M1 */

    /* transpose S */

    for (i1=1; i1<=n; i1++){
      for (j1=i1+1; j1<=n; j1++){
	Ctmp1 = S[i1][j1];
	Ctmp2 = S[j1][i1];
	S[i1][j1] = Ctmp2;
	S[j1][i1] = Ctmp1;
      }
    }

    /* for parallelization in the first world */

    if (all_knum==1){

      /* H * U' */
      /* C is distributed by row in each processor */

      dtime(&Stime);

#pragma omp parallel shared(C,S,H,n,is1,ie1,myid1) private(OMPID,Nthrds,Nprocs,i1,j1,sum_r0,sum_i0,sum_r1,sum_i1,l)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

        for (i1=1+OMPID; i1<=2*n; i1+=Nthrds){
          for (j1=is1[myid1]; j1<=ie1[myid1]; j1++){

	    sum_r0 = 0.0;
	    sum_i0 = 0.0;

	    sum_r1 = 0.0;
	    sum_i1 = 0.0;

	    for (l=1; l<=n; l++){
	      sum_r0 += H[i1][l  ].r*S[j1][l].r - H[i1][l  ].i*S[j1][l].i;
	      sum_i0 += H[i1][l  ].r*S[j1][l].i + H[i1][l  ].i*S[j1][l].r;

	      sum_r1 += H[i1][n+l].r*S[j1][l].r - H[i1][n+l].i*S[j1][l].i;
	      sum_i1 += H[i1][n+l].r*S[j1][l].i + H[i1][n+l].i*S[j1][l].r;
	    }

	    C[2*j1-1][i1].r = sum_r0;
	    C[2*j1-1][i1].i = sum_i0;

	    C[2*j1  ][i1].r = sum_r1;
	    C[2*j1  ][i1].i = sum_i1;
	  }
	} 
    

      } /* #pragma omp parallel */

      dtime(&Etime);
      time4 += Etime - Stime; 

      /* U'^+ H * U' */
      /* H is distributed by row in each processor */

      dtime(&Stime);

#pragma omp parallel shared(C,S,H,n,is1,ie1,myid1) private(OMPID,Nthrds,Nprocs,i1,j1,sum_r00,sum_i00,sum_r01,sum_i01,sum_r10,sum_i10,sum_r11,sum_i11,l,jj1,jj2)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

        for (j1=is1[myid1]+OMPID; j1<=ie1[myid1]; j1+=Nthrds){
          for (i1=1; i1<=n; i1++){

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

	      sum_r00 +=  S[i1][l].r*C[jj1][l  ].r + S[i1][l].i*C[jj1][l  ].i;
	      sum_i00 +=  S[i1][l].r*C[jj1][l  ].i - S[i1][l].i*C[jj1][l  ].r;

	      sum_r01 +=  S[i1][l].r*C[jj1][l+n].r + S[i1][l].i*C[jj1][l+n].i;
	      sum_i01 +=  S[i1][l].r*C[jj1][l+n].i - S[i1][l].i*C[jj1][l+n].r;

	      sum_r10 +=  S[i1][l].r*C[jj2][l  ].r + S[i1][l].i*C[jj2][l  ].i;
	      sum_i10 +=  S[i1][l].r*C[jj2][l  ].i - S[i1][l].i*C[jj2][l  ].r;

	      sum_r11 +=  S[i1][l].r*C[jj2][l+n].r + S[i1][l].i*C[jj2][l+n].i;
	      sum_i11 +=  S[i1][l].r*C[jj2][l+n].i - S[i1][l].i*C[jj2][l+n].r;
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
	}

      } /* #pragma omp parallel */

      /* broadcast H */

      BroadCast_ComplexMatrix(MPI_CommWD1[myworld1],H,2*n,is12,ie12,myid1,numprocs1,
			      stat_send,request_send,request_recv);

      dtime(&Etime);
      time5 += Etime - Stime; 

    }

    else{

      /* H * U' */

      dtime(&Stime);

#pragma omp parallel shared(C,S,H,n) private(OMPID,Nthrds,Nprocs,i1,j1,sum_r0,sum_i0,sum_r1,sum_i1,l)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

        for (i1=1+OMPID; i1<=2*n; i1+=Nthrds){
	  for (j1=1; j1<=n; j1++){

	    sum_r0 = 0.0;
	    sum_i0 = 0.0;

	    sum_r1 = 0.0;
	    sum_i1 = 0.0;

	    for (l=1; l<=n; l++){
	      sum_r0 += H[i1][l  ].r*S[j1][l].r - H[i1][l  ].i*S[j1][l].i;
	      sum_i0 += H[i1][l  ].r*S[j1][l].i + H[i1][l  ].i*S[j1][l].r;

	      sum_r1 += H[i1][n+l].r*S[j1][l].r - H[i1][n+l].i*S[j1][l].i;
	      sum_i1 += H[i1][n+l].r*S[j1][l].i + H[i1][n+l].i*S[j1][l].r;
	    }

	    C[2*j1-1][i1].r = sum_r0;
	    C[2*j1-1][i1].i = sum_i0;

	    C[2*j1  ][i1].r = sum_r1;
	    C[2*j1  ][i1].i = sum_i1;
	  }
	} 

      } /* #pragma omp parallel */

      dtime(&Etime);
      time6 += Etime - Stime; 

      /* U'^+ H * U' */

      dtime(&Stime);

#pragma omp parallel shared(C,S,H,n) private(OMPID,Nthrds,Nprocs,i1,j1,sum_r00,sum_i00,sum_r01,sum_i01,sum_r10,sum_i10,sum_r11,sum_i11,l,jj1,jj2)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (j1=1+OMPID; j1<=n; j1+=Nthrds){
	  for (i1=1; i1<=n; i1++){

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

	      sum_r00 +=  S[i1][l].r*C[jj1][l  ].r + S[i1][l].i*C[jj1][l  ].i;
	      sum_i00 +=  S[i1][l].r*C[jj1][l  ].i - S[i1][l].i*C[jj1][l  ].r;

	      sum_r01 +=  S[i1][l].r*C[jj1][l+n].r + S[i1][l].i*C[jj1][l+n].i;
	      sum_i01 +=  S[i1][l].r*C[jj1][l+n].i - S[i1][l].i*C[jj1][l+n].r;

	      sum_r10 +=  S[i1][l].r*C[jj2][l  ].r + S[i1][l].i*C[jj2][l  ].i;
	      sum_i10 +=  S[i1][l].r*C[jj2][l  ].i - S[i1][l].i*C[jj2][l  ].r;

	      sum_r11 +=  S[i1][l].r*C[jj2][l+n].r + S[i1][l].i*C[jj2][l+n].i;
	      sum_i11 +=  S[i1][l].r*C[jj2][l+n].i - S[i1][l].i*C[jj2][l+n].r;
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
	}

      } /* #pragma omp parallel */

      dtime(&Etime);
      time7 += Etime - Stime; 

    }

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

      if (koS[i1]<EV_cut0){
	C[2*i1-1][2*i1-1].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
	C[2*i1  ][2*i1  ].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
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

    /* solve the standard eigenvalue problem */

    dtime(&Stime);

    n1 = 2*n;

    /* YTL-start */
    LCAO = (dcomplex**)malloc(sizeof(dcomplex*)*n1);
    for (j=0;j<n1;j++){
      LCAO[j] = (dcomplex*)malloc(sizeof(dcomplex)*n1);
      for (k=0;k<n1;k++) LCAO[j][k] = Complex(0.0,0.0);
    }
    /* YTL-end */

    if (parallel_mode==0){
      EigenBand_lapack(C, ko, 2*n, MaxN, all_knum);
    }
    else{
      /*  The output C matrix is distributed by column. */
      Eigen_PHH(MPI_CommWD1[myworld1],C,ko,2*n,MaxN,0);
    }

    dtime(&Etime);
    time8 += Etime - Stime; 

    for (l=1; l<=MaxN; l++)  EIGEN[kloop][l] = ko[l];
/*
    if (3<=level_stdout && 0<=kloop){
      printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
              myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
      for (i1=1; i1<=2*n; i1++){
	printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
      }
    }
*/
    /* calculation of wave functions */

    if (all_knum==1){

      dtime(&Stime);

      /*  The H matrix is distributed by row */

      for (i1=1; i1<=2*n; i1++){
	for (j1=is2[myid1]; j1<=ie2[myid1]; j1++){
	  H[j1][i1] = C[i1][j1];
	}
      }

      /* transpose */

      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }

      /* C is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is2,ie2,myid1) private(OMPID,Nthrds,Nprocs,i1,j1,sum_r0,sum_i0,sum_r1,sum_i1,l,l1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (j1=is2[myid1]+OMPID; j1<=ie2[myid1]; j1+=Nthrds){
	  for (i1=1; i1<=n; i1++){

	    sum_r0 = 0.0; 
	    sum_i0 = 0.0;

	    sum_r1 = 0.0; 
	    sum_i1 = 0.0;

	    l1 = 0; 

	    for (l=1; l<=n; l++){

	      l1++; 

	      sum_r0 +=  S[i1][l].r*H[j1][l1].r - S[i1][l].i*H[j1][l1].i;
	      sum_i0 +=  S[i1][l].r*H[j1][l1].i + S[i1][l].i*H[j1][l1].r;

	      l1++; 

	      sum_r1 +=  S[i1][l].r*H[j1][l1].r - S[i1][l].i*H[j1][l1].i;
	      sum_i1 +=  S[i1][l].r*H[j1][l1].i + S[i1][l].i*H[j1][l1].r;
	    } 

	    C[j1][i1  ].r = sum_r0;
	    C[j1][i1  ].i = sum_i0;

	    C[j1][i1+n].r = sum_r1;
	    C[j1][i1+n].i = sum_i1;

	  }
	}
      } /* #pragma omp parallel */

      /* broadcast C:
       C is distributed by row in each processor
      */

      BroadCast_ComplexMatrix(MPI_CommWD1[myworld1],C,2*n,is2,ie2,myid1,numprocs1,
            stat_send,request_send,request_recv);

      /* C to H (transposition)
         H consists of column vectors
      */ 

      for (i1=1; i1<=MaxN; i1++){
        for (j1=1; j1<=2*n; j1++){
	  H[j1][i1] = C[i1][j1];
	}
      }

      /* YTL-start */
      for (j=0;j<n1;j++){
	for (k=0;k<n1;k++){
	  LCAO[j][k].r = C[j+1][k+1].r;
	  LCAO[j][k].i = C[j+1][k+1].i;
	}
      }
      /* YTL-end */

      dtime(&Etime);
      time9 += Etime - Stime; 
    } /* if (all_knum==1) */

  } /* kloop0 */

  /****************************************************
     MPI:

     EIGEN
  ****************************************************/

  dtime(&Stime);

  for (kloop=0; kloop<T_knum; kloop++){
    ID = T_k_ID[kloop];
    MPI_Bcast(&EIGEN[kloop][0], MaxN+1, MPI_DOUBLE, ID, mpi_comm_level1);
  } 

  dtime(&Etime);
  time10 += Etime - Stime; 

  /****************************************************
                  find chemical potential
  ****************************************************/

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

      for (l=1; l<=2*n; l++){

	x = (EIGEN[kloop][l] - ChemP)*Beta;

	if      (x<=-x_cut)  FermiF = 1.0;
	else if (x>=x_cut)   FermiF = 0.0;
	else                 FermiF = 1.0/(1.0 + exp(x));

	Num_State += FermiF*(double)T_k_op[kloop];

      } /* l */
    } /* kloop */

    Num_State = Num_State/sum_weights;

    Dnum = TZ - Num_State - system_charge;
    if (0.0<=Dnum) ChemP_MIN = ChemP;
    else           ChemP_MAX = ChemP;
    if (fabs(Dnum)<10e-14) po = 1;
  }
  while (po==0 && loop_num<2000);

  dtime(&Etime);
  time11 += Etime - Stime; 

  /*************************************************
     determination of CDDF_max_unoccupied_state
  *************************************************/

  /* YTL-start */
  fd_dist = (double*)malloc(sizeof(double)*n1); /* num_kloop0 = number of kloop at each CPU */
  for (i=0; i<n1; i++) fd_dist[i] = 0.0; /* initialize fermi-dirac distribution */
  /* YTL-end */

  /* determine the maximum unoccupied state */

  double range_Ha=(CDDF_max_eV-CDDF_min_eV)/eV2Hartree; /* in Hartree */
  double p1 = CDDF_AddMaxE/eV2Hartree;
  double FDFi, FDFl;

  j = 0;

  for (kloop=0; kloop<T_knum; kloop++){

    if (0<T_k_op[kloop]){

      for (i=0; i<MaxN; i++){ /* occupied state */

	eig = EIGEN[kloop][i+1];
	x = (eig - ChemP)*Beta;
	if      (x<-x_cut)  FDFi = 1.0;
	else if (x>x_cut)   FDFi = 0.0;
	else                FDFi = 1.0/(1.0 + exp(x));

	for (l=i+1; l<MaxN; l++){ /* unoccupied state */

	  eig = EIGEN[kloop][l+1];
	  x = (eig - ChemP)*Beta;
	  if      (x<-x_cut)  FDFl = 1.0;
	  else if (x>x_cut)   FDFl = 0.0;
	  else                FDFl = 1.0/(1.0 + exp(x));

	  k2 = FDFi - FDFl; /* dFE */
	  k3 = fabs( EIGEN[kloop][i+1] - EIGEN[kloop][l+1] ) ; /* dE */
	  if ( k2!=0 && k3 <= range_Ha + p1 && l > j) j = l; /* update the highest state */
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

  /********************************************************
  if all_knum==1, calculate Fermi function and 
  momentum matrix elements and optical conductivity and 
  dielectric functions at a given k-point 
  ********************************************************/

  if (all_knum==1){

    dtime(&Stime);

    kloop = S_knum;

    /* weight of k-point */

    kw = (double)T_k_op[kloop];

    /* store Fermi function */

    l = 1;
    do{

      eig = EIGEN[kloop][l];
      x = (eig - ChemP)*Beta;

      if      (x<-x_cut)  FermiF = 1.0;
      else if (x>x_cut)   FermiF = 0.0;
      else                FermiF = 1.0/(1.0 + exp(x));
      fd_dist[l-1] = FermiF*kw;
      l++;
    } while(l<=MaxN);

    /* YTL-start */
    Calc_band_optical_noncol_1(T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop],2*n,EIGEN[kloop],LCAO,fd_dist,ChemP);
    /* YTL-end */

    dtime(&Etime);
    time12 += Etime - Stime; 

  } /*if (all_knum==1) */

  dtime(&EiloopTime);
  if (myid0==Host_ID && 0<level_stdout){
  }
  SiloopTime=EiloopTime;
  fflush(stdout);

  /****************************************************
    if all_knum!=1, calculate optical properties by 
    diagonalizing again. 
  ****************************************************/

  dtime(&SiloopTime);

  if (all_knum!=1){

    /* for kloop */

    for (kloop0=0; kloop0<num_kloop0; kloop0++){

      kloop = S_knum + kloop0;

      k1 = T_KGrids1[kloop];
      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];

      /* make S and H */

      for (i=1; i<=n; i++){
	for (j=1; j<=n; j++){
	  S[i][j] = Complex(0.0,0.0);
	} 
      } 

      for (i=1; i<=2*n; i++){
	for (j=1; j<=2*n; j++){
	  H[i][j] = Complex(0.0,0.0);
	} 
      } 

      /* non-spin-orbit coupling and non-LDA+U */

      if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
	  && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){  

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

		S[Anum+i  ][Bnum+j  ].r += co*S1[k];
		S[Anum+i  ][Bnum+j  ].i += si*S1[k];

		H[Anum+i  ][Bnum+j  ].r += co*RH0[k];
		H[Anum+i  ][Bnum+j  ].i += si*RH0[k];

		H[Anum+i+n][Bnum+j+n].r += co*RH1[k];
		H[Anum+i+n][Bnum+j+n].i += si*RH1[k];
            
		H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*RH3[k];
		H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*RH3[k];

		k++;

	      }
	    }
	  }
	}
      }

      /* spin-orbit coupling or LDA+U */

      else {  

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

		S[Anum+i  ][Bnum+j  ].r += co*S1[k];
		S[Anum+i  ][Bnum+j  ].i += si*S1[k];

		H[Anum+i  ][Bnum+j  ].r += co*RH0[k] - si*IH0[k];
		H[Anum+i  ][Bnum+j  ].i += si*RH0[k] + co*IH0[k];

		H[Anum+i+n][Bnum+j+n].r += co*RH1[k] - si*IH1[k];
		H[Anum+i+n][Bnum+j+n].i += si*RH1[k] + co*IH1[k];
            
		H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*(RH3[k]+IH2[k]);
		H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*(RH3[k]+IH2[k]);

		k++;

	      }
	    }
	  }
	}
      }

      /* set off-diagonal part */

      for (i=1; i<=n; i++){
	for (j=1; j<=n; j++){
	  H[j+n][i].r = H[i][j+n].r;
	  H[j+n][i].i =-H[i][j+n].i;
	} 
      } 

      /* diagonalize S */

      if (parallel_mode==0){
	EigenBand_lapack(S,ko,n,n,1);
      }
      else{
	Eigen_PHH(MPI_CommWD1[myworld1],S,ko,n,n,1);
      }
/*
      if (3<=level_stdout){
	printf(" myid0=%2d kloop %2d  k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i=1; i<=n; i++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,ko[i]);
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

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  S[i1][j1].r = S[i1][j1].r*ko[j1];
	  S[i1][j1].i = S[i1][j1].i*ko[j1];
	} 
      } 

      /****************************************************
                  set H' and diagonalize it
      ****************************************************/

      /* U'^+ * H * U * M1 */

      /* transpose S */

      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }

      /* H * U' */

#pragma omp parallel shared(C,S,H,n) private(OMPID,Nthrds,Nprocs,i1,j1,sum_r0,sum_i0,sum_r1,sum_i1,l)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

        for (i1=1+OMPID; i1<=2*n; i1+=Nthrds){
	  for (j1=1; j1<=n; j1++){

	    sum_r0 = 0.0;
	    sum_i0 = 0.0;

	    sum_r1 = 0.0;
	    sum_i1 = 0.0;

	    for (l=1; l<=n; l++){
	      sum_r0 += H[i1][l  ].r*S[j1][l].r - H[i1][l  ].i*S[j1][l].i;
	      sum_i0 += H[i1][l  ].r*S[j1][l].i + H[i1][l  ].i*S[j1][l].r;

	      sum_r1 += H[i1][n+l].r*S[j1][l].r - H[i1][n+l].i*S[j1][l].i;
	      sum_i1 += H[i1][n+l].r*S[j1][l].i + H[i1][n+l].i*S[j1][l].r;
	    }

	    C[2*j1-1][i1].r = sum_r0;
	    C[2*j1-1][i1].i = sum_i0;

	    C[2*j1  ][i1].r = sum_r1;
	    C[2*j1  ][i1].i = sum_i1;
	  }
	}     

      } /* #pragma omp parallel */

      /* U'^+ H * U' */

#pragma omp parallel shared(C,S,H,n) private(OMPID,Nthrds,Nprocs,i1,j1,sum_r0,sum_i0,sum_r1,sum_i1,l)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (i1=1+OMPID; i1<=n; i1+=Nthrds){
	  for (j1=1; j1<=2*n; j1++){

	    sum_r0 = 0.0;
	    sum_i0 = 0.0;

	    sum_r1 = 0.0;
	    sum_i1 = 0.0;

	    for (l=1; l<=n; l++){
	      sum_r0 +=  S[i1][l].r*C[j1][l  ].r + S[i1][l].i*C[j1][l  ].i;
	      sum_i0 +=  S[i1][l].r*C[j1][l  ].i - S[i1][l].i*C[j1][l  ].r;

	      sum_r1 +=  S[i1][l].r*C[j1][l+n].r + S[i1][l].i*C[j1][l+n].i;
	      sum_i1 +=  S[i1][l].r*C[j1][l+n].i - S[i1][l].i*C[j1][l+n].r;
	    }

	    H[2*i1-1][j1].r = sum_r0;
	    H[2*i1-1][j1].i = sum_i0;

	    H[2*i1  ][j1].r = sum_r1;
	    H[2*i1  ][j1].i = sum_i1;

	  }
	}

      } /* #pragma omp parallel */

      /* H to C */

      for (i1=1; i1<=2*n; i1++){
	for (j1=1; j1<=2*n; j1++){
	  C[i1][j1].r = H[i1][j1].r;
	  C[i1][j1].i = H[i1][j1].i;
	}
      }

      /* penalty for ill-conditioning states */

      EV_cut0 = Threshold_OLP_Eigen;

      for (i1=1; i1<=n; i1++){

	if (koS[i1]<EV_cut0){
	  C[2*i1-1][2*i1-1].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
	  C[2*i1  ][2*i1  ].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
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

      /* solve eigenvalue problem */

      dtime(&Stime);

      n1 = 2*n;

      if (parallel_mode==0){
	EigenBand_lapack(C, ko, 2*n, MaxN, 1);
      }
      else{
        Eigen_PHH(MPI_CommWD1[myworld1],C,ko,2*n,MaxN,1);
      }

      C2 = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
      for (j=0; j<n2; j++) C2[j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);

      LCAO = (dcomplex**)malloc(sizeof(dcomplex*)*n1);
      for (j=0;j<n1;j++){
	LCAO[j] = (dcomplex*)malloc(sizeof(dcomplex)*n1);
	for (k=0;k<n1;k++) LCAO[j][k] = Complex(0.0,0.0);
      }

      for (j=1;j<=n1;j++)
	for (k=1;k<=n1;k++){
	  C2[j][k].r = C[j][k].r;
	  C2[j][k].i = C[j][k].i;
	}

      /****************************************************
          transformation to the original eigenvectors
                 NOTE JRCAT-244p and JAIST-2122p 
                    C = U * lambda^{-1/2} * D, 
          and store them in LCAO
      ****************************************************/

      get_LCAO_Coef(C2,LCAO,S,n);

      for (j=0; j<n2; j++) free(C2[j]);
      free(C2);

      dtime(&Etime);
      time5 += Etime - Stime; 

      for (l=1; l<=MaxN; l++)  EIGEN[kloop][l] = ko[l];
/*
      if (3<=level_stdout && 0<=kloop){
	printf(" myid0=%2d  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	       myid0,kloop,T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop]);
	for (i1=1; i1<=2*n; i1++){
	  printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1, ko[i1]);
	}
      }
*/
      /****************************************************************
        calculate Fermi function, and call Calc_band_optical_noncol_1
      ****************************************************************/

      /* weight of k-point */

      kw = (double)T_k_op[kloop];

      /* store Fermi function */

      po = 0;
      l = 1;
      do{

	eig = EIGEN[kloop][l];
	x = (eig - ChemP)*Beta;

	if      (x<-x_cut)  FermiF = 1.0;
	else if (x>x_cut)   FermiF = 0.0;
	else                FermiF = 1.0/(1.0 + exp(x));
  /* YTL-start */
	fd_dist[l-1] = FermiF*kw;
  /* YTL-end */
        l++;

      } while(po==0 && l<=MaxN);

      /* YTL-start */
      Calc_band_optical_noncol_1(T_KGrids1[kloop],T_KGrids2[kloop],T_KGrids3[kloop],n1,EIGEN[kloop],LCAO,fd_dist,ChemP);
      /* YTL-end */

    } /* kloop0 */

  } /* if (all_knum!=1) */

  dtime(&EiloopTime);
  SiloopTime=EiloopTime;
  fflush(stdout);

  /* YTL-start */

  /****************************************************
    collect all the contributions conductivities and 
    dielectric functions
  ****************************************************/

  Calc_optical_noncol_2(n,sum_weights);

  /* YTL-end */

  /****************************************************
                       free arrays
  ****************************************************/

  free(VecFkw);
  free(VecFkwE);

  for (j=0; j<n2; j++){
    free(H[j]);
  }
  free(H);

  for (j=0; j<n2; j++){
    free(C[j]);
  }
  free(C);

  free(KGrids1); free(KGrids2); free(KGrids3);

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
  free(T_k_ID);

  for (j=0; j<T_knum; j++){
    free(EIGEN[j]);
  }
  free(EIGEN);

  free(MP);
  free(order_GA);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  free(ko);

  free(S1);
  free(RH0);
  free(RH1);
  free(RH2);
  free(RH3);

  free(IH0);
  free(IH1);
  free(IH2);

  if (T_knum<=numprocs0){

    if (Num_Comm_World1<=numprocs0){
      MPI_Comm_free(&MPI_CommWD1[myworld1]);
    }

    free(NPROCS_ID1);
    free(Comm_World1);
    free(NPROCS_WD1);
    free(Comm_World_StartID1);
    free(MPI_CommWD1);
  }

  if (all_knum==1){

    free(stat_send);
    free(request_send);
    free(request_recv);

    free(is1);
    free(ie1);
    free(is12);
    free(ie12);
    free(is2);
    free(ie2);
  }

  for (i=0; i<n1; i++) free(LCAO[i]);
  free(LCAO);
  free(fd_dist);

  /* for timing */

  if (measure_time){
    printf("myid0=%2d  time1 =%10.5f\n",myid0,time1);fflush(stdout);
    printf("myid0=%2d  time2 =%10.5f\n",myid0,time2);fflush(stdout);
    printf("myid0=%2d  time3 =%10.5f\n",myid0,time3);fflush(stdout);
    printf("myid0=%2d  time4 =%10.5f\n",myid0,time4);fflush(stdout);
    printf("myid0=%2d  time5 =%10.5f\n",myid0,time5);fflush(stdout);
    printf("myid0=%2d  time6 =%10.5f\n",myid0,time6);fflush(stdout);
    printf("myid0=%2d  time7 =%10.5f\n",myid0,time7);fflush(stdout);
    printf("myid0=%2d  time8 =%10.5f\n",myid0,time8);fflush(stdout);
    printf("myid0=%2d  time9 =%10.5f\n",myid0,time9);fflush(stdout);
    printf("myid0=%2d  time10=%10.5f\n",myid0,time10);fflush(stdout);
    printf("myid0=%2d  time11=%10.5f\n",myid0,time11);fflush(stdout);
    printf("myid0=%2d  time12=%10.5f\n",myid0,time12);fflush(stdout);
    printf("myid0=%2d  time13=%10.5f\n",myid0,time13);fflush(stdout);
    printf("myid0=%2d  time14=%10.5f\n",myid0,time14);fflush(stdout);
    printf("myid0=%2d  time15=%10.5f\n",myid0,time15);fflush(stdout);
    printf("myid0=%2d  time16=%10.5f\n",myid0,time16);fflush(stdout);
    printf("myid0=%2d  time17=%10.5f\n",myid0,time17);fflush(stdout);
  }

  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}


/* YTL-start */
/* calculation of wave functions */
void get_LCAO_Coef(dcomplex** C2, dcomplex** LCAO,dcomplex** S,int n){
  int i1,j1,l1,l,i2=2*n,j2=i2+2;
  double sum_r0,sum_i0,sum_r1,sum_i1;
  dcomplex Ctmp1,Ctmp2,C3[j2][j2];

  for (i1=1; i1<=i2; i1++)
    for (j1=1; j1<=i2; j1++)
      C3[j1][i1] = C2[i1][j1];

  for (i1=1; i1<=n; i1++)
    for (j1=i1+1; j1<=n; j1++){
      Ctmp1 = S[i1][j1];
      Ctmp2 = S[j1][i1];
      S[i1][j1] = Ctmp2;
      S[j1][i1] = Ctmp1;
    }

  for (j1=0; j1<2*n; j1++){
    j2 = j1 + 1;
    for (i1=0; i1<n; i1++){
      sum_r0 = 0.0; 
      sum_i0 = 0.0;
      sum_r1 = 0.0; 
      sum_i1 = 0.0;
      i2 = i1 + 1;
      l1 = 0; 
      for (l=1; l<=n; l++){
        l1++; 
        sum_r0 +=  S[i2][l].r*C3[j2][l1].r - S[i2][l].i*C3[j2][l1].i;
        sum_i0 +=  S[i2][l].r*C3[j2][l1].i + S[i2][l].i*C3[j2][l1].r;
        l1++; 
        sum_r1 +=  S[i2][l].r*C3[j2][l1].r - S[i2][l].i*C3[j2][l1].i;
        sum_i1 +=  S[i2][l].r*C3[j2][l1].i + S[i2][l].i*C3[j2][l1].r;
      } 
      LCAO[j1][i1  ].r = sum_r0;
      LCAO[j1][i1  ].i = sum_i0;
      LCAO[j1][i1+n].r = sum_r1;
      LCAO[j1][i1+n].i = sum_i1;
    }
  }

  for (i1=1; i1<=n; i1++)
    for (j1=i1+1; j1<=n; j1++){
      Ctmp1 = S[i1][j1];
      Ctmp2 = S[j1][i1];
      S[i1][j1] = Ctmp2;
      S[j1][i1] = Ctmp1;
    }
}
/* YTL-end */
