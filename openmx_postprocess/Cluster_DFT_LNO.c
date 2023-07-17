/**********************************************************************
  Cluster_DFT_LNO.c:

   Cluster_DFT_LNO.c is a subroutine to perform cluster calculations
   with localized natural orbitals (LNO).

  Log of Cluster_DFT_LNO.c:

     07/March/2018  Released by T.Ozaki

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
                   double ***C,
                   double **ko,
                   double *****nh, 
                   double ****OLP0,
                   double *****CDM,
                   double *****EDM,
                   EXX_t *exx,
                   dcomplex ****exx_CDM,
                   double *Uexx,
                   double Eele0[2], double Eele1[2]);

static double Cluster_non_collinear(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double *****nh,
                   double *****ImNL,
                   double ****OLP0,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2]);

static void Save_DOS_Col(int n, int MaxN, int *MP, double ****OLP0, double ***C, double **ko);
static void Save_DOS_NonCol(int n, int MaxN, int *MP, double ****OLP0, dcomplex **C, double *ko);



double Cluster_DFT_LNO(char *mode,
		       int SCF_iter,
		       int SpinP_switch,
		       double ***Cluster_ReCoes,
		       double **Cluster_ko,
		       double *****nh,
		       double *****ImNL,
		       double ****OLP0,
		       double *****CDM,
		       double *****EDM,
		       EXX_t *exx, 
		       dcomplex ****exx_CDM,
		       double *Uexx,
		       double Eele0[2], double Eele1[2])
{
  static double time0;

  /****************************************************
                     collinear DFT 
  ****************************************************/

  if ( SpinP_switch==0 || SpinP_switch==1 ){
      time0 = Cluster_collinear(mode,SCF_iter,SpinP_switch,Cluster_ReCoes,Cluster_ko,
                                nh,OLP0,CDM,EDM,exx,exx_CDM,Uexx,Eele0,Eele1);
  }

  /****************************************************
                  non-collinear DFT
  ****************************************************/

  else if ( SpinP_switch==3 ){
    time0 = Cluster_non_collinear(mode,SCF_iter,SpinP_switch,nh,ImNL,OLP0,CDM,EDM,Eele0,Eele1);
  }

  

  return time0;
}




static double Cluster_collinear(
				char *mode,
				int SCF_iter,
				int SpinP_switch,
				double ***C,
				double **ko,
				double *****nh, double ****OLP0,
				double *****CDM,
				double *****EDM,
				EXX_t *exx,
				dcomplex ****exx_CDM,
				double *Uexx,
				double Eele0[2], double Eele1[2])
{
  static int firsttime=1;
  int i,j,l,n1,i1,i1s,j1,k1,l1;
  int wan,HOMO0,HOMO1;
  int *MP;
  int step,wstep,nstep,istart,iend;
  int spin,po,num0,num1,ires;
  int N0,N02,N1,N12;
  int ct_AN,k,wanA,tnoA,wanB,tnoB,tnoA0,tnoB0;
  int GA_AN,Anum,loopN,Gc_AN;
  int MA_AN,MB_AN,LB_AN,GB_AN,Bnum;
  int wan1,mul,m,bcast_flag;
  int *is1,*ie1;
  double lumos,av_num;
  double *OneD_Mat1;
  double ***H,***S;
  double TZ,my_sum,sum,sumE,max_x=60.0;
  double sum0,sum1,sum2,sum3;
  double My_Eele1[2],tmp1,tmp2;
  double Num_State,x,FermiF,Dnum,Dnum2;
  double FermiF2,x2,diffF;
  double dum,ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  double EV_cut0,res;
  double *Mat1,*Mat2,*Mat3;
  int numprocs0,myid0;
  int numprocs1,myid1;
  int Num_Comm_World1;
  int ID,myworld1;
  int *NPROCS_ID1,*NPROCS_WD1;
  int *Comm_World1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char file_EV[YOUSO10] = ".EV";
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp_EV;
  double stime, etime;
  double time0,time1,time2,time3,time4,time5,time6,time7;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

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
                  calculation of LNO
  ****************************************************/

  time0 = LNO("o-n3",SCF_iter,OLP0,nh,CDM);

  /****************************************************
   set the size of original matrix and contracted one
  ****************************************************/

  /* original */

  N0 = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    N0 += Spe_Total_CNO[wanA];
  }
  N02 = N0 + 2;

  /* contracted */

  N1 = 0;
  for (i=1; i<=atomnum; i++){
    N1 += LNO_Num[i];
  }
  N12 = N1 + 2;

  /****************************************************
   Allocation

   int     MP[List_YOUSO[1]]
   double  H[List_YOUSO[23]][n2][n2]  
  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);

  H = (double***)malloc(sizeof(double**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    H[i] = (double**)malloc(sizeof(double*)*N12);
    for (j=0; j<N12; j++){
      H[i][j] = (double*)malloc(sizeof(double)*N12);
    }
  }

  S = (double***)malloc(sizeof(double**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    S[i] = (double**)malloc(sizeof(double*)*N12);
    for (j=0; j<N12; j++){
      S[i][j] = (double*)malloc(sizeof(double)*N12);
    }
  }

  Mat1 = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Mat2 = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Mat3 = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);

  is1 = (int*)malloc(sizeof(int)*numprocs1);
  ie1 = (int*)malloc(sizeof(int)*numprocs1);

  if (firsttime){
    PrintMemory("Cluster_DFT_LNO: H",sizeof(double)*List_YOUSO[23]*N12*N12,NULL);
  }

  /* for PrintMemory */
  firsttime=0;

  if (measure_time){
    time0 = 0.0; 
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

  if ( numprocs1<=N1 ){

    av_num = (double)N1/(double)numprocs1;

    for (ID=0; ID<numprocs1; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs1-1] = N1; 

  }

  else{

    for (ID=0; ID<N1; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=N1; ID<numprocs1; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  /****************************************************
       1. diagonalize the overlap matrix     
       2. search negative eigenvalues
  ****************************************************/

  Set_ContMat_Cluster_LNO(OLP0,nh,S,H,MP);

  /*
  if(myid0==0){
    printf("S before\n");
    for(i=1;i<=N1;i++){
      for(j=1;j<=N1;j++){
	printf("%6.3f ",S[0][i][j]);
      }
      printf("\n");
    }

    printf("H before\n");
    for(i=1;i<=N1;i++){
      for(j=1;j<=N1;j++){
	printf("%6.3f ",H[0][i][j]);
      }
      printf("\n");
    }
  }
  */

  if (measure_time) dtime(&stime);

  /* spin=myworld1 */

  spin = myworld1;

 diagonalize:

  /* all the processors have the full output S matrix */

  bcast_flag = 1;

  if (numprocs1<N1 && 1<numprocs1){
    Eigen_PReHH(MPI_CommWD1[myworld1],S[spin],ko[spin],N1,N1,bcast_flag);
  }
  else{ 
    Eigen_lapack(S[spin],ko[spin],N1,N1);
  }

  /* print to the standard output */

  if (2<=level_stdout && myid0==Host_ID){
    for (l=1; l<=N1; l++){
      printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[spin][l]);fflush(stdout);
    }
  }

  /* set minus eigenvalues to 1.0e-14 and calculate 1/sqrt(ko) */

  for (l=1; l<=N1; l++){
    if (ko[spin][l]<0.0) ko[spin][l] = 1.0e-13;
    ko[spin][l] = 1.0/sqrt(ko[spin][l]);
  }

  for (i1=1; i1<=(N1-3); i1+=4){

#pragma omp parallel shared(i1,N1,S,IEV_S) private(OMPID,Nthrds,Nprocs,j1)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (j1=1+OMPID; j1<=N1; j1+=Nthrds ){
	S[spin][i1+0][j1] = S[spin][i1+0][j1]*ko[spin][j1];
	S[spin][i1+1][j1] = S[spin][i1+1][j1]*ko[spin][j1];
	S[spin][i1+2][j1] = S[spin][i1+2][j1]*ko[spin][j1];
	S[spin][i1+3][j1] = S[spin][i1+3][j1]*ko[spin][j1];
      }

    } /* #pragma omp parallel */
  }

  i1s = N1 - N1%4 + 1;

  for (i1=i1s; i1<=N1; i1++){
    for (j1=1; j1<=N1; j1++){
      S[spin][i1][j1] = S[spin][i1][j1]*ko[spin][j1];
    }
  }

  if (measure_time){
    dtime(&etime);
    time1 += etime - stime; 
  }

  /****************************************************
    calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  /*
  if(myid0==0){
    printf("H\n");
    for(i=1;i<=n;i++){
      for(j=1;j<=n;j++){
	printf("%5.2f ",H[0][i][j]);
      }
      printf("\n");
    }
  }
  */

  /* initialize ko */

  for (i1=1; i1<=N1; i1++){
    ko[spin][i1] = 10000.0;
  }

  if (measure_time) dtime(&stime);

  /* transpose S */

  for (i1=1; i1<=N1; i1++){
    for (j1=i1+1; j1<=N1; j1++){
      tmp1 = S[spin][i1][j1];
      tmp2 = S[spin][j1][i1];
      S[spin][i1][j1] = tmp2;
      S[spin][j1][i1] = tmp1;
    }
  }

  /* C is distributed by row in each processor */

  for (i1=1; i1<=(N1-3); i1+=4){

#pragma omp parallel shared(i1,N1,spin,C,S,H,is1,ie1,myid1) private(OMPID,Nthrds,Nprocs,j1,sum0,sum1,sum2,sum3,l)
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

	for (l=1; l<=N1; l++){
	  sum0 += H[spin][i1+0][l]*S[spin][j1][l];
	  sum1 += H[spin][i1+1][l]*S[spin][j1][l];
	  sum2 += H[spin][i1+2][l]*S[spin][j1][l];
	  sum3 += H[spin][i1+3][l]*S[spin][j1][l];
	}

	C[spin][j1][i1+0] = sum0;
	C[spin][j1][i1+1] = sum1;
	C[spin][j1][i1+2] = sum2;
	C[spin][j1][i1+3] = sum3;

      } /* j1 */
    } /* #pragma omp parallel */
  } /* i1 */

  i1s = N1 - N1%4 + 1;

  for (i1s=i1s; i1<=N1; i1++){
    for (j1=is1[myid1]; j1<=ie1[myid1]; j1++){

      sum = 0.0;

      for (l=1; l<=N1; l++){
	sum += H[spin][i1][l]*S[spin][j1][l];
      }
      C[spin][j1][i1] = sum;
    }
  }


  i1s = N1 - N1%4 + 1;
  for (i1=i1s; i1<=N1; i1++){
    for (j1=is1[myid1]; j1<=ie1[myid1]; j1++){
      sum = 0.0;
      for (l=1; l<=N1; l++){
	sum += S[spin][i1][l]*C[spin][j1][l];
      }
      H[spin][j1][i1] = sum;
    }
  }

  /* H is distributed by row in each processor */

  for (i1=1; i1<=(N1-3); i1+=4){

#pragma omp parallel shared(i1,N1,spin,C,S,H,is1,ie1,myid1) private(OMPID,Nthrds,Nprocs,j1,sum0,sum1,sum2,sum3,l)
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

	for (l=1; l<=N1; l++){
	  sum0 += S[spin][i1+0][l]*C[spin][j1][l];
	  sum1 += S[spin][i1+1][l]*C[spin][j1][l];
	  sum2 += S[spin][i1+2][l]*C[spin][j1][l];
	  sum3 += S[spin][i1+3][l]*C[spin][j1][l];
	}

	H[spin][j1][i1+0] = sum0;
	H[spin][j1][i1+1] = sum1;
	H[spin][j1][i1+2] = sum2;
	H[spin][j1][i1+3] = sum3;

      } /* j1 */
    } /* #pragma omp parallel */
  } /* i1 */


  i1s = N1 - N1%4 + 1;
  for (i1=i1s; i1<=N1; i1++){
    for (j1=is1[myid1]; j1<=ie1[myid1]; j1++){
      sum = 0.0;
      for (l=1; l<=N1; l++){
	sum += S[spin][i1][l]*C[spin][j1][l];
      }
      H[spin][j1][i1] = sum;
    }
  }

  /* broadcast H[spin] */

  BroadCast_ReMatrix(MPI_CommWD1[myworld1],H[spin],N1,is1,ie1,myid1,numprocs1,
                       stat_send,request_send,request_recv);

  /* H to C (transposition) */

  for (i1=1; i1<=N1; i1++){
    for (j1=1; j1<=N1; j1++){
      C[spin][j1][i1] = H[spin][i1][j1];
    }
  }

  if (measure_time){
    dtime(&etime);
    time2 += etime - stime;
  }

  /*  The output C matrix is distributed by column. */

  if (measure_time) dtime(&stime);

  /*
  printf("n=%d, MaxN=%d\n",n,MaxN);
  if(myid0==0){
    printf("C before\n");
    for(i=1;i<=Size_Total_Matrix;i++){
      for(j=1;j<=Size_Total_Matrix;j++){
	printf("%8.4f ",C[spin][i][j]);
      }
      printf("\n");
    }
  }
  */

  bcast_flag = 0;

  if (numprocs1<N1 && 1<numprocs1)
    Eigen_PReHH(MPI_CommWD1[myworld1],C[spin],ko[spin],N1,N1,bcast_flag);
  else 
    Eigen_lapack(C[spin],ko[spin],N1,N1);

  /*
    if(myid0==0){
      printf("C after\n");
      for(i=0;i<Size_Total_Matrix+2;i++){
	for(j=0;j<Size_Total_Matrix+2;j++){
	  printf("%5.3f ",C[spin][i][j]);
	}
        printf("\n");
      }
    }


  if(myid0==0){
    printf("ko after eigen\n");
    for(i=1;i<=MaxN;i++){
      printf("%d %2.1f\n",i,ko[spin][i]);
      }
    }
  */

  /*
  if(myid0==0){
    printf("ko after eigen\n");
    for(i=1;i<=N1;i++){
      printf("%d %15.12f\n",i,ko[spin][i]);
    }
  }
  */


  /*  The H matrix is distributed by row */

  for (i1=1; i1<=N1; i1++){
    for (j1=is1[myid1]; j1<=ie1[myid1]; j1++){
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

  for (i1=1; i1<=N1; i1++){
    for (j1=i1+1; j1<=N1; j1++){
      tmp1 = S[spin][i1][j1];
      tmp2 = S[spin][j1][i1];
      S[spin][i1][j1] = tmp2;
      S[spin][j1][i1] = tmp1;
    }
  }

  /* C is distributed by row in each processor */

  for (i1=1; i1<=(N1-3); i1+=4){

#pragma omp parallel shared(i1,N1,spin,C,S,H,is1,ie1,myid1) private(OMPID,Nthrds,Nprocs,j1,sum0,sum1,sum2,sum3,l)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (j1=is1[myid1]; j1<=ie1[myid1]; j1++){

	sum0 = 0.0;
	sum1 = 0.0;
	sum2 = 0.0;
	sum3 = 0.0;

	for (l=1; l<=N1; l++){

	  sum0 += S[spin][i1+0][l]*H[spin][j1][l];
	  sum1 += S[spin][i1+1][l]*H[spin][j1][l];
	  sum2 += S[spin][i1+2][l]*H[spin][j1][l];
	  sum3 += S[spin][i1+3][l]*H[spin][j1][l];
	}

	C[spin][j1][i1+0] = sum0;
	C[spin][j1][i1+1] = sum1;
	C[spin][j1][i1+2] = sum2;
	C[spin][j1][i1+3] = sum3;

      } /* j1 */
    } /* #pragma omp parallel */
  } /* i1 */

  i1s = N1 - N1%4 + 1;

  for (i1=i1s; i1<=N1; i1++){
    for (j1=is1[myid1]; j1<=ie1[myid1]; j1++){
      sum = 0.0;
      for (l=1; l<=N1; l++){
	sum += S[spin][i1][l]*H[spin][j1][l];
      }
      C[spin][j1][i1] = sum;
    }
  }

  /****************************************************
   MPI: C

   Since is1 and ie1 depend on the spin index, we must
   call BroadCast_ReMatrix in the spin loop.
  ****************************************************/

  /* broadcast C:
     C is distributed by row in each processor
  */

  BroadCast_ReMatrix(MPI_CommWD1[myworld1],C[spin],N1,is1,ie1,myid1,numprocs1,
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

  nstep = 8*N1*N1/(1000000*100) + 1;
  wstep = N1/nstep;
  res = ((double)N1/(double)nstep - (double)wstep)*(double)nstep + 3.0;
  ires = (int)res;

  /* allocate */
  OneD_Mat1 = (double*)malloc(sizeof(double)*(wstep+ires)*(N1+1));

  for (spin=0; spin<=SpinP_switch; spin++){

    for (step=0; step<nstep; step++){

      istart = step*wstep + 1;
      iend   = (step+1)*wstep;
      if (N1<iend)         iend = N1;
      if (step==(nstep-1)) iend = N1; 

      if (istart<=iend){

	for (i=istart; i<=iend; i++){
	  i1 = (i - istart)*N1;
	  for (j=1; j<=N1; j++){
	    k = i1 + j - 1;
	    OneD_Mat1[k] = C[spin][i][j];
	  }
	}

        MPI_Barrier(mpi_comm_level1);
	MPI_Bcast(&OneD_Mat1[0],(iend-istart+1)*N1,MPI_DOUBLE,
		  Comm_World_StartID1[spin],mpi_comm_level1);

	for (i=istart; i<=iend; i++){
	  i1 = (i - istart)*N1;
	  for (j=1; j<=N1; j++){
	    k = i1 + j - 1;
	    C[spin][i][j] = OneD_Mat1[k];
	  }
	}
      }
    }
  }

  for (spin=0; spin<=SpinP_switch; spin++){
    MPI_Bcast(&ko[spin][1],N1,MPI_DOUBLE,Comm_World_StartID1[spin],mpi_comm_level1);
  }

  /* free */
  free(OneD_Mat1);

  /* if ( strcasecmp(mode,"scf")==0 ) */

  if ( strcasecmp(mode,"scf")==0 ){

    if (2<=level_stdout){
      for (i1=1; i1<=N1; i1++){
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
	for (i1=1; i1<=N1; i1++){

	  x = (ko[spin][i1] - ChemP)*Beta*0.2;

	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x =  max_x;
	  FermiF = FermiFunc(x,spin,i1,&po,&x);

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
	for (i1=1; i1<=N1; i1++){
	  x = (ko[spin][i1] - ChemP)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = FermiFunc(x,spin,i1,&po,&x);

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
      for (i1=1; i1<=N1; i1++){

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

    if (0){

    if (SpinP_switch==0){
      if ( (Cluster_HOMO[0]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[0];
      if ( (Cluster_HOMO[0]+num_LUMOs)>N1 )   num_LUMOs = N1 - Cluster_HOMO[0];
    }
    else if (SpinP_switch==1){
      if ( (Cluster_HOMO[0]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[0];
      if ( (Cluster_HOMO[1]-num_HOMOs+1)<1 )  num_HOMOs = Cluster_HOMO[1];
      if ( (Cluster_HOMO[0]+num_LUMOs)>N1 )   num_LUMOs = N1 - Cluster_HOMO[0];
      if ( (Cluster_HOMO[1]+num_LUMOs)>N1 )   num_LUMOs = N1 - Cluster_HOMO[1];
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

      /* HOMOs */

      for (spin=0; spin<=SpinP_switch; spin++){
	for (j=0; j<num_HOMOs; j++){

	  j1 = Cluster_HOMO[spin] - j;

          /* store eigenvalue */
          HOMOs_Coef[0][spin][j][0][0].r = ko[spin][j1] ;

          /* store eigenvector */
	  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	    wanA = WhatSpecies[GA_AN];
	    tnoA = Spe_Total_CNO[wanA];
	    Anum = MP[GA_AN];
	    for (i=0; i<tnoA; i++){
	      HOMOs_Coef[0][spin][j][GA_AN][i].r = C[spin][j1][Anum+i];
	    }
	  }
	}
      }
      
      /* LUMOs */
   
      for (spin=0; spin<=SpinP_switch; spin++){
	for (j=0; j<num_LUMOs; j++){

	  j1 = Cluster_HOMO[spin] + 1 + j;

          /* store eigenvalue */
	  LUMOs_Coef[0][spin][j][0][0].r = ko[spin][j1];

          /* store eigenvector */
	  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	    wanA = WhatSpecies[GA_AN];
	    tnoA = Spe_Total_CNO[wanA];
	    Anum = MP[GA_AN];
	    for (i=0; i<tnoA; i++){
	      LUMOs_Coef[0][spin][j][GA_AN][i].r = C[spin][j1][Anum+i];
	    }
	  }
	}
      }
    }

    }


    /*************************************************************
       density matrix and energy density matrix
       for up and down spins. 

       1. density matrix is calculated in the LNO representation
       2. transformation of the representation from LNO to PAO
    *************************************************************/

    if (measure_time) dtime(&stime);

    /* initialization of CDM, EDM, and Partial_DM */ 

    for (spin=0; spin<=SpinP_switch; spin++){
      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
	GA_AN = M2G[MA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
	      CDM[spin][MA_AN][LB_AN][i][j] = 0.0;
	      EDM[spin][MA_AN][LB_AN][i][j] = 0.0;

              if (cal_partial_charge){
 	        Partial_DM[spin][MA_AN][LB_AN][i][j] = 0.0;
              }
	    }
	  }
	}
      }
    }

    /* 1. density matrix is calculated in the LNO representation */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (k=1; k<=N1; k++){

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

#pragma omp parallel shared(cal_partial_charge,ko,EDM,CDM,Partial_DM,k,spin,C,FermiF,diffF,natn,FNAN,MP,Spe_Total_CNO,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,MA_AN,GA_AN,wanA,tnoA,Anum,LB_AN,GB_AN,wanB,tnoB,Bnum,i,j,dum)
	  { 

	    /* get info. on OpenMP */ 

	    OMPID = omp_get_thread_num();
	    Nthrds = omp_get_num_threads();
	    Nprocs = omp_get_num_procs();

	    for (MA_AN=1+OMPID; MA_AN<=Matomnum; MA_AN+=Nthrds){

	      GA_AN = M2G[MA_AN];
	      tnoA = LNO_Num[GA_AN];
	      Anum = MP[GA_AN];

	      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

		GB_AN = natn[GA_AN][LB_AN];
		tnoB = LNO_Num[GB_AN];
		Bnum = MP[GB_AN];

		for (i=0; i<tnoA; i++){
		  for (j=0; j<tnoB; j++){

		    dum = FermiF*C[spin][k][Anum+i]*C[spin][k][Bnum+j];
		    CDM[spin][MA_AN][LB_AN][i][j] += dum;
		    EDM[spin][MA_AN][LB_AN][i][j] += dum*ko[spin][k];

                    if (cal_partial_charge){
                       dum = diffF*C[spin][k][Anum+i]*C[spin][k][Bnum+j];
		       Partial_DM[spin][MA_AN][LB_AN][i][j] += dum;
                    }
		  }
		}
	      }
	    }

	  } /* #pragma omp parallel */

	}
      }
    }

    /* 2. transformation of the representation from LNO to PAO */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

	GA_AN = M2G[MA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA0 = Spe_Total_CNO[wanA];
        tnoA = LNO_Num[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	  GB_AN = natn[GA_AN][LB_AN];
          MB_AN = S_G2M[GB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB0 = Spe_Total_CNO[wanB];
          tnoB = LNO_Num[GB_AN];

          /* transformation of CDM */

	  for (i=0; i<tnoA0; i++){
	    for (j=0; j<tnoB; j++){

              sum1 = 0.0;
              sum2 = 0.0;

  	      for (k=0; k<tnoA; k++){
                sum1 += LNO_coes[spin][MA_AN][k*tnoA0+i]*CDM[spin][MA_AN][LB_AN][k][j]; /* cache miss */
                sum2 += LNO_coes[spin][MA_AN][k*tnoA0+i]*EDM[spin][MA_AN][LB_AN][k][j]; /* cache miss */
	      }
              
              Mat1[j*tnoA0+i] = sum1; 
              Mat2[j*tnoA0+i] = sum2; 

	      if (cal_partial_charge){

                sum3 = 0.0;
    	        for (k=0; k<tnoA; k++){
                  sum3 += LNO_coes[spin][MA_AN][k*tnoA0+i]*Partial_DM[spin][MA_AN][LB_AN][k][j]; /* cache miss */
	        }
                Mat3[j*tnoA0+i] = sum3; 
	      }
	    }
	  }

	  for (i=0; i<tnoA0; i++){
	    for (j=0; j<tnoB0; j++){

              sum1 = 0.0;
              sum2 = 0.0;

  	      for (k=0; k<tnoB; k++){
                sum1 += Mat1[k*tnoA0+i]*LNO_coes[spin][MB_AN][k*tnoB0+j]; /* cache miss */
                sum2 += Mat2[k*tnoA0+i]*LNO_coes[spin][MB_AN][k*tnoB0+j]; /* cache miss */
	      }
             
              CDM[spin][MA_AN][LB_AN][i][j] = sum1;
              EDM[spin][MA_AN][LB_AN][i][j] = sum2;

	      if (cal_partial_charge){

                sum3 = 0.0;
    	        for (k=0; k<tnoA; k++){
                  sum3 += Mat3[k*tnoA0+i]*LNO_coes[spin][MB_AN][k*tnoB0+j]; /* cache miss */
	        }
                Partial_DM[spin][MA_AN][LB_AN][k][j] = sum3; 
	      }
	    }
	  }

	} /* LB_AN */
      } /* MA_AN */
    } /* spin */

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

    /*
    printf("Eele00=%15.12f Eele01=%15.12f\n",Eele0[0],Eele0[1]);
    printf("Eele10=%15.12f Eele11=%15.12f\n",Eele1[0],Eele1[1]);
    */

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

    /****************************************************
                        Output
    ****************************************************/

    if (measure_time) dtime(&stime);

    if (myid0==Host_ID){

      fnjoint(filepath,filename,file_EV);

      if ((fp_EV = fopen(file_EV,"w")) != NULL){

#ifdef xt3
	setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

	fprintf(fp_EV,"\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"***********************************************************\n");
	fprintf(fp_EV,"           Eigenvalues (Hartree) for SCF KS-eq.           \n");
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
	for (i1=1; i1<=N1; i1++){
	  if (SpinP_switch==0)
	    fprintf(fp_EV,"      %5d %18.14f %18.14f\n",i1,ko[0][i1],ko[0][i1]);
	  else if (SpinP_switch==1)
	    fprintf(fp_EV,"      %5d %18.14f %18.14f\n",i1,ko[0][i1],ko[1][i1]);
	}

	if (2<=level_fileout){

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

	  num0 = 6;
	  num1 = N1/num0 + 1*(N1%num0!=0);

	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (i=1; i<=num1; i++){

	      /* header */ 
	      fprintf(fp_EV,"\n");
	      for (i1=-1; i1<=0; i1++){

		if      (i1==-1) fprintf(fp_EV,"                     ");
		else if (i1==0)  fprintf(fp_EV,"                      ");

		for (j=1; j<=num0; j++){
		  j1 = num0*(i-1) + j;
		  if (j1<=N1){ 
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

	      i1 = 1; 

	      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

		wan1 = WhatSpecies[Gc_AN];
            
		for (l=0; l<=Supported_MaxL; l++){
		  for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
		    for (m=0; m<(2*l+1); m++){

		      if (l==0 && mul==0 && m==0)
			fprintf(fp_EV,"%4d %3s %2d %s",Gc_AN,SpeName[wan1],mul,Name_Angular[l][m]);
		      else
			fprintf(fp_EV,"         %2d %s",mul,Name_Angular[l][m]);

		      for (j=1; j<=num0; j++){
			j1 = num0*(i-1) + j;
			if (0<i1 && j1<=N1){
			  fprintf(fp_EV,"  %8.5f",C[spin][j1][i1]);
			}
		      }
		  
		      fprintf(fp_EV,"\n");

		      i1++;
		    }
		  }
		}
	      }

	    }
	  }

	}

	fclose(fp_EV);
      }
      else
	printf("Failure of saving the EV file.\n");
    }

    if (measure_time){
      dtime(&etime);
      time7 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  else if ( strcasecmp(mode,"dos")==0 ){
    Save_DOS_Col(N1,N1,MP,OLP0,C,ko);
  }

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid0,time1,time2,time3,time4,time5,time6,time7);fflush(stdout); 
  }

  /****************************************************
                          Free
  ****************************************************/

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);

  free(MP);

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<N12; j++){
      free(H[i][j]);
    }
    free(H[i]);
  }
  free(H);  

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<N12; j++){
      free(S[i][j]);
    }
    free(S[i]);
  }
  free(S);  

  free(Mat1);
  free(Mat2);
  free(Mat3);

  /* freeing of arrays for the first world */

  if (Num_Comm_World1<=numprocs0){
    MPI_Comm_free(&MPI_CommWD1[myworld1]);
  }

  free(NPROCS_ID1);
  free(Comm_World1);
  free(NPROCS_WD1);
  free(Comm_World_StartID1);
  free(MPI_CommWD1);

  /*
  printf("VVV3 myid0=%2d\n",myid0);
  MPI_Finalize();
  exit(0);
  */

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}








static double Cluster_non_collinear(
				    char *mode,
				    int SCF_iter,
				    int SpinP_switch,
				    double *****nh,
				    double *****ImNL,
				    double ****OLP0,
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
  double *ko;
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
  double My_Eele1[2];
  double Num_State,x,FermiF,Dnum,Dnum2;
  double FermiF2,x2,diffF;
  double dum,ChemP_MAX,ChemP_MIN;
  double TStime,TEtime;
  double FermiEps = 1.0e-13;
  int numprocs,myid,ID;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char file_EV[YOUSO10] = ".EV";
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */
  double time1,time2,time3,time4,time5,time6,time7;
  double stime,etime;

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

  if (firsttime){
    PrintMemory("Cluster_DFT: H",sizeof(dcomplex)*n2*n2,NULL);
    PrintMemory("Cluster_DFT: C",sizeof(dcomplex)*n2*n2,NULL);
  }

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

    Overlap_Cluster(OLP0,S,MP);

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

    if (2<=level_stdout){
      for (l=1; l<=n; l++){
	printf("  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[l]);
      }
    }

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

    /* first, find ChemP at five times large temperatue */

    po = 0;
    loopN = 0;

    ChemP_MAX = 30.0;  
    ChemP_MIN =-30.0;

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

    ChemP_MAX = 30.0;  
    ChemP_MIN =-30.0;

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

      /* HOMOs */

      for (j=0; j<num_HOMOs; j++){

	j1 = Cluster_HOMO[0] - j;

        /* store eigenvalue */
        HOMOs_Coef[0][0][j][0][0].r = ko[j1];
        HOMOs_Coef[0][1][j][0][0].r = ko[j1];

        /* store eigenvector */
	for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    HOMOs_Coef[0][0][j][GA_AN][i].r = C[j1][Anum+i].r;
	    HOMOs_Coef[0][0][j][GA_AN][i].i = C[j1][Anum+i].i;
	    HOMOs_Coef[0][1][j][GA_AN][i].r = C[j1][Anum+i+n].r;
	    HOMOs_Coef[0][1][j][GA_AN][i].i = C[j1][Anum+i+n].i;
	  }
	}
      }
      
      /* LUMOs */

      for (j=0; j<num_LUMOs; j++){

	j1 = Cluster_HOMO[0] + 1 + j;

        /* store eigenvalue */
        LUMOs_Coef[0][0][j][0][0].r = ko[j1];
        LUMOs_Coef[0][1][j][0][0].r = ko[j1];

        /* store eigenvector */
	for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    LUMOs_Coef[0][0][j][GA_AN][i].r = C[j1][Anum+i].r;
	    LUMOs_Coef[0][0][j][GA_AN][i].i = C[j1][Anum+i].i;
	    LUMOs_Coef[0][1][j][GA_AN][i].r = C[j1][Anum+i+n].r;
	    LUMOs_Coef[0][1][j][GA_AN][i].i = C[j1][Anum+i+n].i;
	  }
	}
      }
    }

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
	      CDM[spin][MA_AN][LB_AN][i][j] = 0.0;
	      EDM[spin][MA_AN][LB_AN][i][j] = 0.0;

	    }
	  }

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
	      iDM[0][0][MA_AN][LB_AN][i][j] = 0.0;
	      iDM[0][1][MA_AN][LB_AN][i][j] = 0.0;
	    }
	  }

          if (cal_partial_charge && spin<=1){
	    for (i=0; i<tnoA; i++){
	      for (j=0; j<tnoB; j++){
		Partial_DM[spin][MA_AN][LB_AN][i][j] = 0.0;
	      }
	    }
	  }
	}
      }
    }

    for (k=1; k<=MaxN; k++){

      x = (ko[k] - ChemP)*Beta;
      if (x<=-max_x) x = -max_x;
      if (max_x<=x)  x = max_x;
      FermiF = FermiFunc_NC(x,k);

      if (cal_partial_charge){
	x2 = (ko[k] - (ChemP+ene_win_partial_charge))*Beta;
	if (x2<=-max_x) x2 = -max_x;
	if (max_x<=x2)  x2 = max_x;
	FermiF2 = FermiFunc_NC(x2,k);
	diffF = fabs(FermiF-FermiF2);
      }

      if ( FermiF>FermiEps || (cal_partial_charge && (FermiEps<FermiF || FermiEps<FermiF2) ) ) {

#pragma omp parallel shared(ko,iDM,EDM,CDM,Partial_DM,k,spin,C,FermiF,diffF,natn,FNAN,MP,Spe_Total_CNO,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,MA_AN,GA_AN,wanA,tnoA,Anum,LB_AN,GB_AN,wanB,tnoB,Bnum,i,j,dum)
	{ 

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (MA_AN=1+OMPID; MA_AN<=Matomnum; MA_AN+=Nthrds){
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
              
		  /* Re11 */
		  dum = FermiF*(C[k][Anum+i].r*C[k][Bnum+j].r
			      + C[k][Anum+i].i*C[k][Bnum+j].i);
		  CDM[0][MA_AN][LB_AN][i][j] += dum;
		  EDM[0][MA_AN][LB_AN][i][j] += dum*ko[k];

		  /* Re22 */
		  dum = FermiF*(C[k][Anum+i+n].r*C[k][Bnum+j+n].r
			      + C[k][Anum+i+n].i*C[k][Bnum+j+n].i);
		  CDM[1][MA_AN][LB_AN][i][j] += dum;
		  EDM[1][MA_AN][LB_AN][i][j] += dum*ko[k];
               
		  /* Re12 */
		  dum = FermiF*(C[k][Anum+i].r*C[k][Bnum+j+n].r
			      + C[k][Anum+i].i*C[k][Bnum+j+n].i);
		  CDM[2][MA_AN][LB_AN][i][j] += dum;
		  EDM[2][MA_AN][LB_AN][i][j] += dum*ko[k];
              
		  /* Im12 */
		  dum = FermiF*(C[k][Anum+i].r*C[k][Bnum+j+n].i
			       -C[k][Anum+i].i*C[k][Bnum+j+n].r);
		  CDM[3][MA_AN][LB_AN][i][j] += dum;
		  EDM[3][MA_AN][LB_AN][i][j] += dum*ko[k];

		  /* Im11 */
		  dum = FermiF*(C[k][Anum+i].r*C[k][Bnum+j].i
			       -C[k][Anum+i].i*C[k][Bnum+j].r);
		  iDM[0][0][MA_AN][LB_AN][i][j] += dum;

		  /* Im22 */
		  dum = FermiF*(C[k][Anum+i+n].r*C[k][Bnum+j+n].i
			       -C[k][Anum+i+n].i*C[k][Bnum+j+n].r);
		  iDM[0][1][MA_AN][LB_AN][i][j] += dum;

                  /* partial density matrix for STM simulation */
                  if (cal_partial_charge){

		    dum = diffF*(C[k][Anum+i].r*C[k][Bnum+j].r
			       + C[k][Anum+i].i*C[k][Bnum+j].i);
		    Partial_DM[0][MA_AN][LB_AN][i][j] += dum;

		    dum = diffF*(C[k][Anum+i+n].r*C[k][Bnum+j+n].r
			       + C[k][Anum+i+n].i*C[k][Bnum+j+n].i);
		    Partial_DM[1][MA_AN][LB_AN][i][j] += dum;
		  }
		}
	      }
	    }
	  }

	} /* #pragma omp parallel */

      }
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

      fnjoint(filepath,filename,file_EV);

      if ((fp_EV = fopen(file_EV,"w")) != NULL){

#ifdef xt3
	setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

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

	if (2<=level_fileout){

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

	  num0 = 2;
	  num1 = MaxN/num0 + 1*(MaxN%num0!=0);

	  for (i=1; i<=num1; i++){

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

	    i1 = 1; 

	    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	      wan1 = WhatSpecies[Gc_AN];

	      for (l=0; l<=Supported_MaxL; l++){
		for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
		  for (m=0; m<(2*l+1); m++){

		    if (l==0 && mul==0 && m==0)
		      fprintf(fp_EV,"%4d %3s %2d %s",Gc_AN,SpeName[wan1],mul,Name_Angular[l][m]);
		    else
		      fprintf(fp_EV,"         %2d %s",mul,Name_Angular[l][m]);

		    for (j=1; j<=num0; j++){

		      j1 = num0*(i-1) + j;

		      if (0<i1 && j1<=MaxN){
			fprintf(fp_EV,"  %8.5f",C[j1][i1].r);
			fprintf(fp_EV,"  %8.5f",C[j1][i1].i);
			fprintf(fp_EV,"  %8.5f",C[j1][i1+n].r);
			fprintf(fp_EV,"  %8.5f",C[j1][i1+n].i);
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

	fclose(fp_EV);
      }
      else
	printf("Failure of saving the EV file.\n");
    }

    if (measure_time){
      dtime(&etime);
      time7 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  else if ( strcasecmp(mode,"dos")==0 ){
    Save_DOS_NonCol(n,MaxN,MP,OLP0,C,ko);
  }

  if (measure_time){
    printf("Cluster_DFT myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f time7=%7.3f\n",
            myid,time1,time2,time3,time4,time5,time6,time7);fflush(stdout); 
  }

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

  /* for elapsed time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}







void Save_DOS_Col(int n, int MaxN, int *MP, double ****OLP0, double ***C, double **ko)
{
  int spin,i,j,iemin,iemax,GA_AN,k,l;
  int Anum,Bnum,tnoA,tnoB,wanA,wanB;
  int LB_AN,GB_AN,MaxL,num,num0,num1;
  int sAN,eAN,sAN0,eAN0;
  int numprocs,myid,ID,tag;
  int i_vec[10];  
  double dum,tmp;
  float *SD;
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  MPI_Status stat;
  MPI_Request request;
  MPI_Request *request_recv;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  request_recv = malloc(sizeof(MPI_Request)*numprocs);

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

  /* allocation of array */

  SD = (float*)malloc(sizeof(float)*(atomnum+1)*List_YOUSO[7]);

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

  /* make S */

  Overlap_Cluster(OLP0,S,MP);

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (k=iemin; k<=iemax; k++){

      for (i=0; i<(atomnum+1)*List_YOUSO[7]; i++) SD[i] = 0.0;

      i_vec[0]=i_vec[1]=i_vec[2]=0;
      if (myid==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);

      tmp = (double)atomnum/(double)numprocs;

      sAN = (int)(tmp*(double)myid) + 1;
      eAN = (int)(tmp*(double)(myid+1));

      if      (myid==0)            sAN = 1;
      else if (myid==(numprocs-1)) eAN = atomnum;

#pragma omp parallel shared(S,SD,k,spin,C,natn,FNAN,MP,Spe_Total_CNO,WhatSpecies,sAN,eAN) private(OMPID,Nthrds,Nprocs,GA_AN,wanA,tnoA,Anum,i,LB_AN,GB_AN,wanB,tnoB,Bnum,j,dum)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (GA_AN=sAN+OMPID; GA_AN<=eAN; GA_AN+=Nthrds){

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
		dum = C[spin][k][Anum+i]*C[spin][k][Bnum+j];
		SD[Anum+i] += (float)(dum*S[Anum+i][Bnum+j]);
	      }
	    }
	  }
	}

      } /* #pragma omp parallel */

      /* MPI communication */

      if (myid!=Host_ID){
      
        wanA = WhatSpecies[eAN];
        tnoA = Spe_Total_CNO[wanA];
        num0 = MP[eAN] + tnoA - MP[sAN]; 

        tag = 999;
        MPI_Send(&SD[MP[sAN]],num0,MPI_FLOAT,Host_ID,tag,mpi_comm_level1);
      }
   
      else {

        for (ID=1; ID<numprocs; ID++){ 

          tmp = (double)atomnum/(double)numprocs;

          sAN0 = (int)(tmp*(double)ID) + 1;
          eAN0 = (int)(tmp*(double)(ID+1));

	  wanA = WhatSpecies[eAN0];
	  tnoA = Spe_Total_CNO[wanA];
	  num1 = MP[eAN0] + tnoA - MP[sAN0]; 

	  tag = 999;
	  MPI_Irecv(&SD[MP[sAN0]], num1, MPI_FLOAT, ID, tag, mpi_comm_level1, &request_recv[ID]);
	}

        for (ID=1; ID<numprocs; ID++){ 
          MPI_Wait(&request_recv[ID],&stat);
	}
      }      

      /* write *.Dos.vec */

      if (myid==Host_ID){
	wanA = WhatSpecies[atomnum];
	tnoA = Spe_Total_CNO[wanA];
        num = MP[atomnum] + tnoA - 1;
        fwrite(&SD[1],sizeof(float),num,fp_ev);
      }

    }
  }

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
  free(request_recv);
}





void Save_DOS_NonCol(int n, int MaxN, int *MP, double ****OLP0, dcomplex **C, double *ko)
{
  int spin,ie,i,j,iemin,iemax,GA_AN,k,l;
  int Anum,Bnum,tnoA,tnoB,wanA,wanB;
  int LB_AN,GB_AN,MaxL,num,num0,num1;
  int sAN,eAN,sAN0,eAN0;
  int numprocs,myid,ID,tag;
  int i_vec[10];  
  double d0,d1,d2,d3,dum;
  double tmp,tmp1,tmp2,tmp3;
  double theta,phi,sit,cot,sip,cop;
  float *SD;
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  MPI_Status stat;
  MPI_Request request;
  MPI_Request *request_recv;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  request_recv = malloc(sizeof(MPI_Request)*numprocs);

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

  /* allocation of array */

  SD = (float*)malloc(sizeof(float)*2*(atomnum+1)*List_YOUSO[7]);

  /* find iemin */

  iemin = 2*n;

  for (i=1; i<2*n; i++) {
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

  /* make S */

  Overlap_Cluster(OLP0,S,MP);

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

  for (k=iemin; k<=iemax; k++){

    i_vec[0]=i_vec[1]=i_vec[2]=0;
    if (myid==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);

    tmp = (double)atomnum/(double)numprocs;

    sAN = (int)(tmp*(double)myid) + 1;
    eAN = (int)(tmp*(double)(myid+1));

    if      (myid==0)            sAN = 1;
    else if (myid==(numprocs-1)) eAN = atomnum;

#pragma omp parallel shared(SD,Angle1_Spin,Angle0_Spin,S,k,C,natn,FNAN,MP,Spe_Total_CNO,WhatSpecies,sAN,eAN) private(OMPID,Nthrds,Nprocs,GA_AN,wanA,tnoA,Anum,i,d0,d1,d2,d3,LB_AN,GB_AN,wanB,tnoB,Bnum,dum,j,theta,phi,sit,cot,sip,cop,tmp1,tmp2,tmp3)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (GA_AN=sAN+OMPID; GA_AN<=eAN; GA_AN+=Nthrds){

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (i=0; i<tnoA; i++){

	  d0 = 0.0;
	  d1 = 0.0;
	  d2 = 0.0;
	  d3 = 0.0;

	  for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	    GB_AN = natn[GA_AN][LB_AN];
	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_CNO[wanB];
	    Bnum = MP[GB_AN];

	    for (j=0; j<tnoB; j++){

	      /* Re11 */
	      dum = C[k][Anum+i].r*C[k][Bnum+j].r + C[k][Anum+i].i*C[k][Bnum+j].i;
	      d0 += dum*S[Anum+i][Bnum+j];

	      /* Re22 */
	      dum = C[k][Anum+i+n].r*C[k][Bnum+j+n].r + C[k][Anum+i+n].i*C[k][Bnum+j+n].i;
	      d1 += dum*S[Anum+i][Bnum+j];

	      /* Re12 */
	      dum = C[k][Anum+i].r*C[k][Bnum+j+n].r + C[k][Anum+i].i*C[k][Bnum+j+n].i;
	      d2 += dum*S[Anum+i][Bnum+j];

	      /* Im12
		 conjugate complex of Im12 due to difference in the definition
		 between density matrix and charge density
	      */

	      dum = -(C[k][Anum+i].r*C[k][Bnum+j+n].i - C[k][Anum+i].i*C[k][Bnum+j+n].r);
	      d3 += dum*S[Anum+i][Bnum+j];

	    } /* j */
	  } /* LB_AN */

	  /*  transform to up and down states */

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

	} /* i */
      } /* GA_AN */

    } /* #pragma omp parallel */

    /* MPI communication */

    if (myid!=Host_ID){
      
      wanA = WhatSpecies[eAN];
      tnoA = Spe_Total_CNO[wanA];
      num0 = 2*(MP[eAN] + tnoA - MP[sAN]); 
      tag = 999;
      MPI_Send(&SD[2*(MP[sAN]-1)],num0,MPI_FLOAT,Host_ID,tag,mpi_comm_level1);
    }
   
    else {

      for (ID=1; ID<numprocs; ID++){ 

        tmp = (double)atomnum/(double)numprocs;

        sAN0 = (int)(tmp*(double)ID) + 1;
        eAN0 = (int)(tmp*(double)(ID+1));

	wanA = WhatSpecies[eAN0];
	tnoA = Spe_Total_CNO[wanA];
	num1 = 2*(MP[eAN0] + tnoA - MP[sAN0]); 

	tag = 999;
	MPI_Irecv(&SD[2*(MP[sAN0]-1)], num1, MPI_FLOAT, ID, tag, mpi_comm_level1, &request_recv[ID]);
      }

      for (ID=1; ID<numprocs; ID++){ 
	MPI_Wait(&request_recv[ID],&stat);
      }
    }      
    
    /* write *.Dos.vec */

    if (myid==Host_ID){
      wanA = WhatSpecies[atomnum];
      tnoA = Spe_Total_CNO[wanA];
      num = 2*MP[atomnum] + 2*tnoA - 2;
      fwrite(&SD[0],sizeof(float),num,fp_ev);
    }

  } /* k */

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

  free(SD);
  free(request_recv);
}


