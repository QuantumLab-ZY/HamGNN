/**********************************************************************
  Cluster_DFT_Optical.c:

  Cluster_DFT_Optical.c is a subroutine to calculate optical
  conductivities and dielectric functions based on a collinear DFT

  Log of Cluster_DFT_Optical.c:

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
                   double ***C,
                   double **ko,
                   double *****nh, 
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   EXX_t *exx,
                   dcomplex ****exx_CDM,
                   double *Uexx,
                   double Eele0[2], double Eele1[2]);

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



double Cluster_DFT_Optical(char *mode,
			   int SCF_iter,
			   int SpinP_switch,
			   double ***Cluster_ReCoes,
			   double **Cluster_ko,
			   double *****nh,
			   double *****ImNL,
			   double ****CntOLP,
			   double *****CDM,
			   double *****EDM,
			   EXX_t *exx, 
			   dcomplex ****exx_CDM,
			   double *Uexx,
			   double Eele0[2], double Eele1[2])
{
  static double time0;

  /****************************************************
         collinear without spin-orbit coupling
  ****************************************************/

  if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==0 ){

    time0 = Cluster_collinear_Optical(mode,SCF_iter,SpinP_switch,Cluster_ReCoes,Cluster_ko,
                              nh,CntOLP,CDM,EDM,exx,exx_CDM,Uexx,Eele0,Eele1);
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
    time0 = Cluster_non_collinear_Optical(mode,SCF_iter,SpinP_switch,nh,ImNL,CntOLP,CDM,EDM,Eele0,Eele1);
  }

  return time0;
}




static double Cluster_collinear_Optical(
				char *mode,
				int SCF_iter,
				int SpinP_switch,
				double ***C,
				double **ko,
				double *****nh, double ****CntOLP,
				double *****CDM,
				double *****EDM,
				EXX_t *exx,
				dcomplex ****exx_CDM,
				double *Uexx,
				double Eele0[2], double Eele1[2])
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
  int *is1,*ie1,*is2,*ie2;
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
  int Num_Comm_World1;
  int ID,myworld1;
  int *NPROCS_ID1,*NPROCS_WD1;
  int *Comm_World1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10] = ".EV";
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp_EV;
  double stime, etime;
  double time1,time2,time3,time4,time5,time6,time7;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

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

  Num_Comm_World1 = SpinP_switch + 1; 

  stat_send = malloc(sizeof(MPI_Status)*numprocs0);
  request_send = malloc(sizeof(MPI_Request)*numprocs0);
  request_recv = malloc(sizeof(MPI_Request)*numprocs0);

  /***********************************************
      allocation of arrays for the first world 
  ***********************************************/

  NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs0); 
  Comm_World1 = (int*)malloc(sizeof(int)*numprocs0); 
  NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);

  /*********************************************** 
            make the first level worlds 
  ***********************************************/

  Make_Comm_Worlds(mpi_comm_level1, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1, 
                   NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

  /*********************************************** 
       for pallalel calculations in myworld1
  ***********************************************/

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

  H = (double***)malloc(sizeof(double**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    H[i] = (double**)malloc(sizeof(double*)*n2);
    for (j=0; j<n2; j++){
      H[i][j] = (double*)malloc(sizeof(double)*n2);
    }
  }

  is1 = (int*)malloc(sizeof(int)*numprocs1);
  ie1 = (int*)malloc(sizeof(int)*numprocs1);

  is2 = (int*)malloc(sizeof(int)*numprocs1);
  ie2 = (int*)malloc(sizeof(int)*numprocs1);
/*
  if (firsttime){
    PrintMemory("Cluster_DFT: H",sizeof(double)*List_YOUSO[23]*n2*n2,NULL);
  }
*/
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

  if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1){
    Overlap_Cluster(CntOLP,S,MP);
  }

  for (spin=0; spin<=SpinP_switch; spin++){
    Hamiltonian_Cluster(nh[spin],H[spin],MP);
  } 

  if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1){

    if (measure_time) dtime(&stime);

    /* all the processors have the full output S matrix */

    bcast_flag = 1;

    if (numprocs0<n && 1<numprocs0)
      Eigen_PReHH(mpi_comm_level1,S,ko[0],n,n,bcast_flag);
    else 
      Eigen_lapack(S,ko[0],n,n);

    /* minus eigenvalues to 1.0e-14 */

    for (l=1; l<=n; l++){
      if (ko[0][l]<0.0) ko[0][l] = 1.0e-14;
      EV_S[l] = ko[0][l];
    }

    /* print to the standard output */
/*
    if (2<=level_stdout && myid0==Host_ID){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[0][l]);fflush(stdout);
      }
    }
*/
    /* calculate S*1/sqrt(ko) */

    for (l=1; l<=n; l++){
      IEV_S[l] = 1.0/sqrt(ko[0][l]);
    }

    SP_PEV = 1;

    for (i1=1; i1<=(n-3); i1+=4){

#pragma omp parallel shared(i1,n,S,IEV_S) private(OMPID,Nthrds,Nprocs,j1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (j1=1+OMPID; j1<=n; j1+=Nthrds ){
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
    calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

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

  for (i1=1; i1<=(n-3); i1+=4){

#pragma omp parallel shared(i1,n,spin,C,S,H,is1,ie1,myid1) private(OMPID,Nthrds,Nprocs,j1,sum0,sum1,sum2,sum3,l)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (j1=is1[myid1]+OMPID; j1<=ie1[myid1]; j1+=Nthrds){

	sum0 = 0.0;
	sum1 = 0.0;
	sum2 = 0.0;
	sum3 = 0.0;

	for (l=1; l<=n; l++){
	  sum0 += H[spin][i1+0][l]*S[j1][l];
	  sum1 += H[spin][i1+1][l]*S[j1][l];
	  sum2 += H[spin][i1+2][l]*S[j1][l];
	  sum3 += H[spin][i1+3][l]*S[j1][l];
	}

	C[spin][j1][i1+0] = sum0;
	C[spin][j1][i1+1] = sum1;
	C[spin][j1][i1+2] = sum2;
	C[spin][j1][i1+3] = sum3;

      } /* j1 */
    } /* #pragma omp parallel */
  } /* i1 */

  i1s = n - n%4 + 1;

  for (i1s=i1s; i1<=n; i1++){
    for (j1=is1[myid1]; j1<=ie1[myid1]; j1++){
      sum = 0.0;
      for (l=1; l<=n; l++){
	sum += H[spin][i1][l]*S[j1][l];
      }
      C[spin][j1][i1] = sum;
    }
  }

  /* H is distributed by row in each processor */

  for (i1=1; i1<=(n-3); i1+=4){

#pragma omp parallel shared(i1,n,spin,C,S,H,is1,ie1,myid1) private(OMPID,Nthrds,Nprocs,j1,sum0,sum1,sum2,sum3,l)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (j1=is1[myid1]+OMPID; j1<=ie1[myid1]; j1+=Nthrds){

	sum0 = 0.0;
	sum1 = 0.0;
	sum2 = 0.0;
	sum3 = 0.0;

	for (l=1; l<=n; l++){
	  sum0 += S[i1+0][l]*C[spin][j1][l];
	  sum1 += S[i1+1][l]*C[spin][j1][l];
	  sum2 += S[i1+2][l]*C[spin][j1][l];
	  sum3 += S[i1+3][l]*C[spin][j1][l];
	}

	H[spin][j1][i1+0] = sum0;
	H[spin][j1][i1+1] = sum1;
	H[spin][j1][i1+2] = sum2;
	H[spin][j1][i1+3] = sum3;

      } /* j1 */
    } /* #pragma omp parallel */
  } /* i1 */

  i1s = n - n%4 + 1;

  for (i1=i1s; i1<=n; i1++){
    for (j1=is1[myid1]; j1<=ie1[myid1]; j1++){
      sum = 0.0;
      for (l=1; l<=n; l++){
	sum += S[i1][l]*C[spin][j1][l];
      }
      H[spin][j1][i1] = sum;
    }
  }

  /* broadcast H[spin] */

  BroadCast_ReMatrix(MPI_CommWD1[myworld1],H[spin],n,is1,ie1,myid1,numprocs1,
                       stat_send,request_send,request_recv);

  /* H to C (transposition) */

  for (i1=1; i1<=n; i1++){
    for (j1=1; j1<=n; j1++){
      C[spin][j1][i1] = H[spin][i1][j1];
    }
  }

  /* penalty for ill-conditioning states */

  EV_cut0 = Threshold_OLP_Eigen;

  for (i1=1; i1<=n; i1++){

    if (EV_S[i1]<EV_cut0){
      C[spin][i1][i1] += pow((EV_S[i1]/EV_cut0),-2.0) - 1.0;
    }
 
    /* cutoff the interaction between the ill-conditioned state */
 
    if (1.0e+3<C[spin][i1][i1]){
      for (j1=1; j1<=n; j1++){
	C[spin][i1][j1] = 0.0;
	C[spin][j1][i1] = 0.0;
      }
      C[spin][i1][i1] = 1.0e+4;
    }
  }

  /* find the maximum states in solved eigenvalues */

  if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1){
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
      ie2[ID] = -2;
    }
  }

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /*  The output C matrix is distributed by column. */

  if (measure_time) dtime(&stime);

  bcast_flag = 0;

  if (numprocs1<n && 1<numprocs1)
    Eigen_PReHH(MPI_CommWD1[myworld1],C[spin],ko[spin],n,MaxN,bcast_flag);
  else 
    Eigen_lapack(C[spin],ko[spin],n,n);

  /*  The H matrix is distributed by row */
  for (i1=1; i1<=n; i1++){
    for (j1=is2[myid1]; j1<=ie2[myid1]; j1++){
      H[spin][j1][i1] = C[spin][i1][j1];
    }
  }

  if (measure_time){
    dtime(&etime);
    time3 += etime - stime;
  }

  /****************************************************
      transformation to the original eigenvectors.
                       NOTE 244P
  ****************************************************/

  if (measure_time) dtime(&stime);

  /* transpose */

  for (i1=1; i1<=n; i1++){
    for (j1=i1+1; j1<=n; j1++){
      tmp1 = S[i1][j1];
      tmp2 = S[j1][i1];
      S[i1][j1] = tmp2;
      S[j1][i1] = tmp1;
    }
  }

  /* C is distributed by row in each processor */

  for (i1=1; i1<=(n-3); i1+=4){

#pragma omp parallel shared(i1,n,spin,C,S,H,is2,ie2,myid1) private(OMPID,Nthrds,Nprocs,j1,sum0,sum1,sum2,sum3,l)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (j1=is2[myid1]; j1<=ie2[myid1]; j1++){

	sum0 = 0.0;
	sum1 = 0.0;
	sum2 = 0.0;
	sum3 = 0.0;

	for (l=1; l<=n; l++){
	  sum0 += S[i1+0][l]*H[spin][j1][l];
	  sum1 += S[i1+1][l]*H[spin][j1][l];
	  sum2 += S[i1+2][l]*H[spin][j1][l];
	  sum3 += S[i1+3][l]*H[spin][j1][l];
	}

	C[spin][j1][i1+0] = sum0;
	C[spin][j1][i1+1] = sum1;
	C[spin][j1][i1+2] = sum2;
	C[spin][j1][i1+3] = sum3;

      } /* j1 */
    } /* #pragma omp parallel */
  } /* i1 */

  i1s = n - n%4 + 1;

  for (i1=i1s; i1<=n; i1++){
    for (j1=is2[myid1]; j1<=ie2[myid1]; j1++){
      sum = 0.0;
      for (l=1; l<=n; l++){
	sum += S[i1][l]*H[spin][j1][l];
      }
      C[spin][j1][i1] = sum;
    }
  }

  /****************************************************
   MPI: C

   Since is2 and ie2 depend on the spin index, we must
   call BroadCast_ReMatrix in the spin loop.
  ****************************************************/

  /* broadcast C:
     C is distributed by row in each processor
  */

  BroadCast_ReMatrix(MPI_CommWD1[myworld1],C[spin],n,is2,ie2,myid1,numprocs1,
                     stat_send,request_send,request_recv);

  if (measure_time){
    dtime(&etime);
    time4 += etime - stime;
  }

  if (SpinP_switch==1 && numprocs0==1 && spin==0){
    spin++;  
    goto diagonalize; 
  }

  if (measure_time) dtime(&stime);

  /*********************************************** 
    MPI: ko and C 
  ***********************************************/
 
  /* communicate every 100 MByte */

  nstep = 8*MaxN*n/(1000000*100) + 1;
  wstep = MaxN/nstep;
  res = ((double)MaxN/(double)nstep - (double)wstep)*(double)nstep + 3.0;
  ires = (int)res;

  /* allocate */
  OneD_Mat1 = (double*)malloc(sizeof(double)*(wstep+ires)*(n+1));

  for (spin=0; spin<=SpinP_switch; spin++){

    for (step=0; step<nstep; step++){

      istart = step*wstep + 1;
      iend   = (step+1)*wstep;
      if (MaxN<iend)       iend = MaxN;
      if (step==(nstep-1)) iend = MaxN; 

      if (istart<=iend){

	for (i=istart; i<=iend; i++){
	  i1 = (i - istart)*n;
	  for (j=1; j<=n; j++){
	    k = i1 + j - 1;
	    OneD_Mat1[k] = C[spin][i][j];
	  }
	}

        MPI_Barrier(mpi_comm_level1);
	MPI_Bcast(&OneD_Mat1[0],(iend-istart+1)*n,MPI_DOUBLE,
		  Comm_World_StartID1[spin],mpi_comm_level1);

	for (i=istart; i<=iend; i++){
	  i1 = (i - istart)*n;
	  for (j=1; j<=n; j++){
	    k = i1 + j - 1;
	    C[spin][i][j] = OneD_Mat1[k];
	  }
	}
      }
    }
  }

  for (spin=0; spin<=SpinP_switch; spin++){
    MPI_Bcast(&ko[spin][1],MaxN,MPI_DOUBLE,Comm_World_StartID1[spin],mpi_comm_level1);
  }

  /* free */
  free(OneD_Mat1);

  if ( strcasecmp(mode,"scf")==0 ){
/*
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
*/
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
/*
      if (myid1==Host_ID && 2<=level_stdout){
	printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",ChemP,TZ,Num_State); 
      }
*/
      loopN++;

    } 
    while (po==0 && loopN<1000); 
/*
    if (2<=level_stdout){
      printf("  ChemP=%15.12f\n",ChemP);
    }
*/
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
          k3 = fabs( ko[spin][i+1] - ko[spin][l+1] ) ; /* dE */
          if ( k2!=0 && k3 <= range_Ha + p1 && l > j) j=l; /* update the highest state */

	}
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

      /* spin-polarization (on) -> spin = 0, (off) -> spin = 0,1 */
      for (i1=1;i1<n2-1;i1++)
      	for (j1=1;j1<n2-1;j1++)
      	  LCAO[j1][i1] = Complex(C[spin][i1][j1],0.0);

      Calc_band_optical_col_1(0.0,0.0,0.0,spin,n,ko[spin],LCAO,fd_dist,ChemP);
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

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);

  free(is2);
  free(ie2);

  free(MP);

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<n2; j++){
      free(H[i][j]);
    }
    free(H[i]);
  }
  free(H);  

  for (i1=0;i1<n2;i1++) free(LCAO[i1]);
  free(LCAO);

  free(fd_dist);

  /* freeing of arrays for the first world */

  if (Num_Comm_World1<=numprocs0){
    MPI_Comm_free(&MPI_CommWD1[myworld1]);
  }

  free(NPROCS_ID1);
  free(Comm_World1);
  free(NPROCS_WD1);
  free(Comm_World_StartID1);
  free(MPI_CommWD1);

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

  if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1){

    if (measure_time) dtime(&stime);

    Overlap_Cluster(CntOLP,S,MP);

    /* all the processors have the full output S matrix */

    bcast_flag = 1;

    if (numprocs<n)
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

  if (SCF_iter==1 || rediagonalize_flag_overlap_matrix==1){
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

  if (numprocs<n1)
    Eigen_PHH(mpi_comm_level1, C, ko, n1, MaxN, bcast_flag);
  else 
    EigenBand_lapack(C, ko, n1, MaxN, 1);

  if (2<=level_stdout){
    for (i1=1; i1<=MaxN; i1++){
      printf("  Eigenvalues of Kohn-Sham %2d  %15.12f\n", i1,ko[i1]);
    }
  }

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
            calculation of Fermi function
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
