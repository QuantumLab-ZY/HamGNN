/**********************************************************************
  Cluster_DFT_NonCol.c:

     Cluster_DFT_NonCol.c is a subroutine to perform non-collinear 
     cluster calculations.

  Log of Cluster_DFT_NonCol.c:

     21/Feb./2019  Released by T.Ozaki

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


static void Save_DOS_NonCol(int n, int n2, int MaxN, int *MP, double ****OLP0, dcomplex *EVec1, double *ko);


double Cluster_DFT_NonCol(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double *ko,
                   double *****nh,
                   double *****ImNL,
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2],
                   int *MP,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *rHs11,
		   double *rHs12,
		   double *rHs22,
		   double *iHs11,
		   double *iHs12,
		   double *iHs22,
                   dcomplex *Ss2,
                   dcomplex *Hs2,
                   dcomplex *Cs2,
		   double *DM1,
		   int size_H1, 
                   dcomplex *EVec1,
                   double *Work1)
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
  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  MPI_Status stat;
  MPI_Request request;

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

  /* print memory size */

  if (firsttime && memoryusage_fileout){
    PrintMemory("Cluster_DFT_NonCol: is1",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: ie1",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: Num_Snd_EV",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: Num_Ecv_EV",sizeof(int)*numprocs,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Snd_i",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Snd_j",sizeof(int)*Max_Num_Snd_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Rcv_i",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: index_Rcv_j",sizeof(int)*Max_Num_Rcv_EV,NULL);
    PrintMemory("Cluster_DFT_NonCol: EVec_Snd",sizeof(double)*Max_Num_Snd_EV*2,NULL);
    PrintMemory("Cluster_DFT_NonCol: EVec_Rcv",sizeof(double)*Max_Num_Rcv_EV*2,NULL);
  }
  firsttime=0;

  /* ***************************************************
            diagonalize the overlap matrix     
  *************************************************** */

  if (SCF_iter==1){

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
