/**********************************************************************
  Opt_Contraction.c:
  
    Opt_Contraction.c is a subroutine to update the contraction
    coefficients using the gradient of the total energy with respect
    to the contraction coefficients for the orbital optimization method.
  
  Log of Opt_Contraction.c:
  
     22/Nov/2001  Released by T.Ozaki
  
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"


static void D_RCntCoes(double ***CntCoes0, double ***D_CntCoes,
                       double *****H,      double *****OLP,
                       double *****CDM,    double *****EDM);
static void D_ACntCoes(double ***CntCoes0,
                       double ***D_CntCoes,
                       double ***DCntCoes_Spe,
                       double *****H, double *****OLP,
                       double *****CDM,
                       double *****EDM);
static double NormD_RCnt(double ***D_CntCoes);
static double NormD_ACnt(double ***DCntCoes_Spe);
static void Inverse(int n, double **a, double **ia);
static void Inverse_Mat(int n, double **A0, double **IA0);

static double Species_Opt_Contraction_DIIS(
         int orbitalOpt_iter,
         double TotalE,
         double *****H, 
         double *****OLP,
         double *****CDM,
         double *****EDM,
         double ****His_CntCoes,
         double ****His_D_CntCoes,
         double ****His_CntCoes_Species,
         double ****His_D_CntCoes_Species,
         double *His_OrbOpt_Etot,
         double **OrbOpt_Hessian);

static double Species_Opt_Contraction_EDIIS(
         int orbitalOpt_iter,
         double TotalE,
         double *****H, 
         double *****OLP,
         double *****CDM,
         double *****EDM,
         double ****His_CntCoes,
         double ****His_D_CntCoes,
         double ****His_CntCoes_Species,
         double ****His_D_CntCoes_Species,
         double *His_OrbOpt_Etot,
         double **OrbOpt_Hessian);

static double Atoms_Opt_Contraction_DIIS(
         int orbitalOpt_iter,
         double *****H, 
         double *****OLP,
         double *****CDM,
         double *****EDM,
         double ****His_CntCoes,
         double ****His_D_CntCoes);



double Opt_Contraction(
         int orbitalOpt_iter,
         double TotalE,
         double *****H, 
         double *****OLP,
         double *****CDM,
         double *****EDM,
         double ****His_CntCoes,
         double ****His_D_CntCoes,
         double ****His_CntCoes_Species,
         double ****His_D_CntCoes_Species,
         double *His_OrbOpt_Etot,
         double **OrbOpt_Hessian)
{

  double norm_deri;

  if (ACnt_switch==0){

    norm_deri = Atoms_Opt_Contraction_DIIS( 
                     orbitalOpt_iter,
                     H, 
                     OLP,
                     CDM,
                     EDM,
                     His_CntCoes,
		     His_D_CntCoes);
  }

  else if (ACnt_switch==1){

    norm_deri = Species_Opt_Contraction_DIIS( 
                     orbitalOpt_iter,
                     TotalE,
                     H, 
                     OLP,
                     CDM,
                     EDM,
                     His_CntCoes,
		     His_D_CntCoes,
                     His_CntCoes_Species,
		     His_D_CntCoes_Species,
                     His_OrbOpt_Etot,
                     OrbOpt_Hessian);


    /*
    norm_deri = Species_Opt_Contraction_EDIIS( 
                     orbitalOpt_iter,
                     TotalE,
                     H, 
                     OLP,
                     CDM,
                     EDM,
                     His_CntCoes,
		     His_D_CntCoes,
                     His_CntCoes_Species,
		     His_D_CntCoes_Species,
                     His_OrbOpt_Etot,
                     OrbOpt_Hessian);
    */

  }

  /* return */
  return norm_deri;
}


double Atoms_Opt_Contraction_DIIS(
         int orbitalOpt_iter,
         double *****H, 
         double *****OLP,
         double *****CDM,
         double *****EDM,
         double ****His_CntCoes,
         double ****His_D_CntCoes)
{
  static int firsttime=1;
  int i,j,Mc_AN,Gc_AN,Cwan,be,his;
  int his1,his2,diis_dim,al,p,q,p0,q0;
  int n,num,wan,size1,size2;
  double nd[4],x01,x12,x20,y12,y01;
  double a,b,c,alpha;
  double norm_deri,sum,sum1,sum2;
  double coef_OptG,tmp0,mixing_coef;
  double **My_A,**A,**IA;
  double *DIIS_coes;
  double ***CntCoes0;
  double ***D_CntCoes;
  double ***D_CntCoes0;
  double *tmp_array;
  double *tmp_array2;
  int *Snd_CntCoes_Size;
  int *Rcv_CntCoes_Size;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************************
    allocation of arrays:

    int Snd_CntCoes_Size[numprocs];
    int Rcv_CntCoes_Size[numprocs];

    double CntCoes0[Matomnum+MatonumF+1]
                          [List_YOUSO[7]]
                          [List_YOUSO[24]];

    double D_CntCoes[Matomnum+1]
                           [List_YOUSO[7]]
                           [List_YOUSO[24]];

    double D_CntCoes0[Matomnum+1]
                            [List_YOUSO[7]]
                            [List_YOUSO[24]];

  ********************************************************/

  DIIS_coes = (double*)malloc(sizeof(double)*(orbitalOpt_iter+2));

  IA = (double**)malloc(sizeof(double*)*(orbitalOpt_iter+2));
  for (i=0; i<(orbitalOpt_iter+2); i++){
    IA[i] = (double*)malloc(sizeof(double)*(orbitalOpt_iter+2));
  }

  My_A = (double**)malloc(sizeof(double*)*(orbitalOpt_iter+2));
  for (i=0; i<(orbitalOpt_iter+2); i++){
    My_A[i] = (double*)malloc(sizeof(double)*(orbitalOpt_iter+2));
  }

  A = (double**)malloc(sizeof(double*)*(orbitalOpt_iter+2));
  for (i=0; i<(orbitalOpt_iter+2); i++){
    A[i] = (double*)malloc(sizeof(double)*(orbitalOpt_iter+2));
  }
  
  Snd_CntCoes_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_CntCoes_Size = (int*)malloc(sizeof(int)*numprocs);

  CntCoes0 = (double***)malloc(sizeof(double**)*(Matomnum+MatomnumF+1));
  for (i=0; i<=(Matomnum+MatomnumF); i++){
    CntCoes0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      CntCoes0[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  D_CntCoes = (double***)malloc(sizeof(double**)*(Matomnum+1));
  for (i=0; i<=Matomnum; i++){
    D_CntCoes[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      D_CntCoes[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  D_CntCoes0 = (double***)malloc(sizeof(double**)*(Matomnum+1));
  for (i=0; i<=Matomnum; i++){
    D_CntCoes0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      D_CntCoes0[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  /********************************************************
    PrintMemory
  ********************************************************/

  if (firsttime) {
    PrintMemory("Opt_Contraction: CntCoes0",sizeof(double)*
           (Matomnum+1)*List_YOUSO[7]*List_YOUSO[24], NULL);
    PrintMemory("Opt_Contraction: D_CntCoes",sizeof(double)*
           (Matomnum+1)*List_YOUSO[7]*List_YOUSO[24], NULL);
    PrintMemory("Opt_Contraction: D_CntCoes0",sizeof(double)*
           (Matomnum+1)*List_YOUSO[7]*List_YOUSO[24], NULL);
    firsttime=0;
  }

  /********************************************************
    start calc.
  ********************************************************/

  /* calculate gradients of total energy with respect to contraction coefficients */
  /* atom dependent contraction */

  D_RCntCoes(CntCoes,D_CntCoes,H,OLP,CDM,EDM);
  norm_deri = NormD_RCnt(D_CntCoes);

  /* find diis_dim */

  if (orbitalOpt_History<(orbitalOpt_iter-orbitalOpt_StartPulay+1)) 
    diis_dim = orbitalOpt_History;
  else 
    diis_dim = orbitalOpt_iter-orbitalOpt_StartPulay+1;

  if (diis_dim<1) diis_dim = 1;

  /* if diis_dim==1, find mixing_coef */

  mixing_coef = 4.0*orbitalOpt_SD_step/norm_deri;
  if (0.05<mixing_coef) mixing_coef = 0.05;

  /* shift the His arrays */

  for (his=(diis_dim-1); 0<his; his--){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (be=0; be<Spe_Total_CNO[Cwan]; be++){
	for (p=0; p<Spe_Specified_Num[Cwan][be]; p++){
	  His_CntCoes[his][Mc_AN][be][p] = His_CntCoes[his-1][Mc_AN][be][p];
	  His_D_CntCoes[his][Mc_AN][be][p] = His_D_CntCoes[his-1][Mc_AN][be][p];
	}
      }
    }
  }

  /* store CntCoes and D_CntCoes into the His arrays */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (be=0; be<Spe_Total_CNO[Cwan]; be++){
      for (p=0; p<Spe_Specified_Num[Cwan][be]; p++){
	His_CntCoes[0][Mc_AN][be][p] = CntCoes[Mc_AN][be][p];
	His_D_CntCoes[0][Mc_AN][be][p] = D_CntCoes[Mc_AN][be][p];
      }
    }
  }

  /* update CntCoes by a simple mixing if (diis_dim==1) */

  if (diis_dim==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (be=0; be<Spe_Total_CNO[Cwan]; be++){
	for (p=0; p<Spe_Specified_Num[Cwan][be]; p++){
	  CntCoes[Mc_AN][be][p] -= mixing_coef*D_CntCoes[Mc_AN][be][p];
	}
      }
    }
  }

  /* otherwise, update CntCoes by RMM-DIIS */

  else {

    /* construct the norm matrix of His_D_CntCoes */

    for (his1=0; his1<diis_dim; his1++){
      for (his2=0; his2<diis_dim; his2++){

	sum = 0.0;  
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  for (be=0; be<Spe_Total_CNO[Cwan]; be++){
	    for (p=0; p<Spe_Specified_Num[Cwan][be]; p++){

	      sum += His_D_CntCoes[his1][Mc_AN][be][p]*His_D_CntCoes[his2][Mc_AN][be][p];
	    }
	  }
	}

	My_A[his1][his2] = sum;
      }
    }

    /* MPI My_A */

    for (his1=0; his1<diis_dim; his1++){
      for (his2=his1; his2<diis_dim; his2++){

	MPI_Allreduce(&My_A[his1][his2], &A[his1][his2],
		      1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

	A[his2][his1] = A[his1][his2];
      }
    }

    /*
    printf("A\n");
    for (his1=0; his1<diis_dim; his1++){
      for (his2=0; his2<diis_dim; his2++){
        printf("%15.10f ",A[his1][his2]);
      }
      printf("\n");
    }
    */

    /* solve the linear equation */

    for (his1=1; his1<=diis_dim; his1++){
      A[his1-1][diis_dim] = 1.0;
      A[diis_dim][his1-1] = 1.0;
    }
    A[diis_dim][diis_dim] = 0.0;

    Inverse(diis_dim,A,IA);

    for (his1=0; his1<diis_dim; his1++){
      DIIS_coes[his1] = IA[his1][diis_dim]; 
    }

    /*
    for (his1=0; his1<diis_dim; his1++){
      printf("his1=%2d DIIS_coes[his1]=%15.10f\n",his1,DIIS_coes[his1]);
    }
    */

    /* construct optimum contraction coefficients */

    if      (1.0e-2<norm_deri)    coef_OptG = 0.1;
    else if (1.0e-3<norm_deri)    coef_OptG = 0.2;
    else if (1.0e-4<norm_deri)    coef_OptG = 0.4;
    else if (1.0e-5<norm_deri)    coef_OptG = 0.5;
    else                          coef_OptG = 0.7;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (be=0; be<Spe_Total_CNO[Cwan]; be++){
	for (p=0; p<Spe_Specified_Num[Cwan][be]; p++){

	  sum1 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum1 += DIIS_coes[his1]*His_CntCoes[his1][Mc_AN][be][p];
	  }      

	  sum2 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum2 += DIIS_coes[his1]*His_D_CntCoes[his1][Mc_AN][be][p];
	  }      

	  CntCoes[Mc_AN][be][p] = sum1 - coef_OptG*sum2; 
	}
      }
    }
  } /* else */

  /********************************************************
    output norm_deri
  ********************************************************/

  Oopt_NormD[1] = norm_deri;  
  if (myid==Host_ID) printf("<Opt_Contraction> Norm of derivatives = %15.12f \n",norm_deri);

  /********************************************************
        normalization of contracted basis functions
  ********************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (al=0; al<Spe_Total_CNO[Cwan]; al++){

      sum = 0.0;
      for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	p0 = Spe_Trans_Orbital[Cwan][al][p];

	for (q=0; q<Spe_Specified_Num[Cwan][al]; q++){
          q0 = Spe_Trans_Orbital[Cwan][al][q];

          tmp0 = CntCoes[Mc_AN][al][p]*CntCoes[Mc_AN][al][q];
          sum += tmp0*OLP[0][Mc_AN][0][p0][q0]; 
        }
      }

      tmp0 = 1.0/sqrt(sum);
      for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
        CntCoes[Mc_AN][al][p] = CntCoes[Mc_AN][al][p]*tmp0;
      }        

    }
  }

  if (2<=level_stdout){
    printf("<Opt_Contraction> Contraction coefficients\n");
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (be=0; be<Spe_Total_CNO[Cwan]; be++){
        for (p=0; p<Spe_Specified_Num[Cwan][be]; p++){
          printf("  Mc_AN=%2d Gc_AN=%2d be=%2d p=%2d  %15.12f\n",
                    Mc_AN,Gc_AN,be,p,CntCoes[Mc_AN][be][p]);
        }
      }
    }
  }

  /****************************************************
    MPI:

    CntCoes
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;

      /* find data size to send block data */
      if (F_Snd_Num[IDS]!=0){

        size1 = 0;
        for (n=0; n<F_Snd_Num[IDS]; n++){
          Mc_AN = Snd_MAN[IDS][n];
          Gc_AN = Snd_GAN[IDS][n];
          wan = WhatSpecies[Gc_AN]; 
          for (be=0; be<Spe_Total_CNO[wan]; be++){
            size1 += Spe_Specified_Num[wan][be];
	  }
	}

        Snd_CntCoes_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_CntCoes_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR]!=0){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_CntCoes_Size[IDR] = size2;
      }
      else{
        Rcv_CntCoes_Size[IDR] = 0;
      }
    
      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

    }
    else {
      Snd_CntCoes_Size[myid] = 0;
      Rcv_CntCoes_Size[myid] = 0;
    }
  }

  /***********************************
             data transfer
  ************************************/

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
              sending of data 
      *****************************/

      if (F_Snd_Num[IDS]!=0){

        size1 = Snd_CntCoes_Size[IDS];

        /* allocation of array */
        tmp_array = (double*)malloc(sizeof(double)*size1);

        /* multidimentional array to vector array */

        num = 0;
        for (n=0; n<F_Snd_Num[IDS]; n++){
          Mc_AN = Snd_MAN[IDS][n];
          Gc_AN = Snd_GAN[IDS][n];
          wan = WhatSpecies[Gc_AN]; 
          for (be=0; be<Spe_Total_CNO[wan]; be++){
            for (p=0; p<Spe_Specified_Num[wan][be]; p++){
              tmp_array[num] = CntCoes[Mc_AN][be][p];
              num++;
  	    }
	  }
        }

        MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR]!=0){

        size2 = Rcv_CntCoes_Size[IDR];

        /* allocation of array */
        tmp_array2 = (double*)malloc(sizeof(double)*size2);

        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        Mc_AN = F_TopMAN[IDR] - 1;
        for (n=0; n<F_Rcv_Num[IDR]; n++){
          Mc_AN++;
          Gc_AN = Rcv_GAN[IDR][n];
          wan = WhatSpecies[Gc_AN];
          for (be=0; be<Spe_Total_CNO[wan]; be++){
            for (p=0; p<Spe_Specified_Num[wan][be]; p++){
              CntCoes[Mc_AN][be][p] = tmp_array2[num];
              num++;
	    }
  	  }
        }

        /* freeing of array */
        free(tmp_array2);
      }

      if (F_Snd_Num[IDS]!=0){
        MPI_Wait(&request,&stat);
        free(tmp_array); /* freeing of array */
      }
    }
  }

  /********************************************************
    freeing of arrays:

    int Snd_CntCoes_Size[numprocs];
    int Rcv_CntCoes_Size[numprocs];

    double CntCoes0[Matomnum+MatomnumF+1]
                          [List_YOUSO[7]]
                          [List_YOUSO[24]];

    double D_CntCoes[Matomnum+1]
                           [List_YOUSO[7]]
                           [List_YOUSO[24]];

    double D_CntCoes0[Matomnum+1]
                            [List_YOUSO[7]]
                            [List_YOUSO[24]];
  ********************************************************/

  free(Snd_CntCoes_Size);
  free(Rcv_CntCoes_Size);

  for (i=0; i<=(Matomnum+MatomnumF); i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(CntCoes0[i][j]);
    }
    free(CntCoes0[i]);
  }
  free(CntCoes0);

  for (i=0; i<=Matomnum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(D_CntCoes[i][j]);
    }
    free(D_CntCoes[i]);
  }
  free(D_CntCoes);

  for (i=0; i<=Matomnum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(D_CntCoes0[i][j]);
    }
    free(D_CntCoes0[i]);
  }
  free(D_CntCoes0);

  for (i=0; i<(orbitalOpt_iter+2); i++){
    free(My_A[i]);
  }
  free(My_A);

  for (i=0; i<(orbitalOpt_iter+2); i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<(orbitalOpt_iter+2); i++){
    free(IA[i]);
  }
  free(IA);

  free(DIIS_coes);

  if (firsttime) {
    firsttime=0;
  }

  /* return */
  return Oopt_NormD[1];
}


double Species_Opt_Contraction_DIIS(
         int orbitalOpt_iter,
         double TotalE,
         double *****H, 
         double *****OLP,
         double *****CDM,
         double *****EDM,
         double ****His_CntCoes,
         double ****His_D_CntCoes,
         double ****His_CntCoes_Species,
         double ****His_D_CntCoes_Species,
         double *His_OrbOpt_Etot,
         double **OrbOpt_Hessian)
{
  static int firsttime=1;
  int i,j,Mc_AN,Mc_AN1,Cwan,Gc_AN,be,his,po,my_po;
  int his1,his2,diis_dim,al,p,q,p0,q0;
  int n,num,wan,size1,size2,dim_H;
  int wan1,al1,p1,wan2,al2,p2;
  double nd[4],x01,x12,x20,y12,y01;
  double a,b,c,alpha,c0,c1,MinKo,MaxStep;
  double tmp1,tmp2,d1,d2,d;
  double norm_deri,sum,sum1,sum2;
  double coef_OptG,tmp0,mixing_coef;
  double **My_A,**A,**IA,*B;
  double *DIIS_coes;
  double ***D_CntCoes;
  double ***DCntCoes_Spe;
  double ***OLPtmp;
  double *TmpVec1;
  double *TmpVec2;
  double **U,*ko;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************************
    allocation of arrays:

    double D_CntCoes[Matomnum+1]
                           [List_YOUSO[7]]
                           [List_YOUSO[24]];

    double DCntCoes_Spe[SpeciesNum+1]
                              [List_YOUSO[7]]
                              [List_YOUSO[24]];
  ********************************************************/

  DIIS_coes = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));

  IA = (double**)malloc(sizeof(double*)*(orbitalOpt_History+2));
  for (i=0; i<(orbitalOpt_History+2); i++){
    IA[i] = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));
  }

  My_A = (double**)malloc(sizeof(double*)*(orbitalOpt_History+2));
  for (i=0; i<(orbitalOpt_History+2); i++){
    My_A[i] = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));
  }

  A = (double**)malloc(sizeof(double*)*(orbitalOpt_History+2));
  for (i=0; i<(orbitalOpt_History+2); i++){
    A[i] = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));
  }

  B = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));

  D_CntCoes = (double***)malloc(sizeof(double**)*(Matomnum+1));
  for (i=0; i<=Matomnum; i++){
    D_CntCoes[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      D_CntCoes[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  DCntCoes_Spe = (double***)malloc(sizeof(double**)*(SpeciesNum+1));
  for (i=0; i<=SpeciesNum; i++){
    DCntCoes_Spe[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      DCntCoes_Spe[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  OLPtmp = (double***)malloc(sizeof(double**)*(SpeciesNum+1));
  for (i=0; i<=SpeciesNum; i++){
    OLPtmp[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      OLPtmp[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }
  }

  /********************************************************
    PrintMemory
  ********************************************************/

  if (firsttime) {
    PrintMemory("Opt_Contraction: D_CntCoes",sizeof(double)*
           (Matomnum+1)*List_YOUSO[7]*List_YOUSO[24], NULL);
    firsttime=0;
  }

  /********************************************************
    start calc.
  ********************************************************/

  /* find the dimension of Hessian */ 

  dim_H = 0;
  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        dim_H++;
      }
    }
  }

  /* initialize OrbOpt_Hessian */

  if (orbitalOpt_iter==1){

    for (i=0; i<dim_H; i++){
      for (j=0; j<dim_H; j++){
        OrbOpt_Hessian[i][j] = 0.0;
      }    
      OrbOpt_Hessian[i][i] = 1.0;
    }    
  } 

  TmpVec1 = (double*)malloc(sizeof(double)*(dim_H+1));
  TmpVec2 = (double*)malloc(sizeof(double)*(dim_H+1));

  U = (double**)malloc(sizeof(double*)*(dim_H+5));
  for (i=0; i<(dim_H+5); i++){
    U[i] = (double*)malloc(sizeof(double)*(dim_H+5));
  }

  ko = (double*)malloc(sizeof(double)*(dim_H+5));

  /* calculate gradients of total energy with respect to contraction coefficients */
  /* species dependent contraction */

  {
    int p,Mc_AN,Gc_AN,wan,al;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];

    if (3<=level_stdout){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          printf("E Init_CntCoes Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
                  Mc_AN,Gc_AN,al,p,CntCoes[Mc_AN][al][p]);
        }
      }
    }
  }
  }

  D_ACntCoes(CntCoes, D_CntCoes, DCntCoes_Spe, H, OLP, CDM, EDM);
  norm_deri = NormD_ACnt(DCntCoes_Spe);

  /* find diis_dim */

  if (orbitalOpt_History<(orbitalOpt_iter-orbitalOpt_StartPulay+1)) 
    diis_dim = orbitalOpt_History;
  else 
    diis_dim = orbitalOpt_iter-orbitalOpt_StartPulay+1;

  if (diis_dim<1) diis_dim = 1;

  /* shift the His arrays */

  for (his=(diis_dim-1); 0<his; his--){
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
	for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	  His_CntCoes_Species[his][wan][al][p] = His_CntCoes_Species[his-1][wan][al][p];
	  His_D_CntCoes_Species[his][wan][al][p] = His_D_CntCoes_Species[his-1][wan][al][p];
	}
      }
    }    

    His_OrbOpt_Etot[his] = His_OrbOpt_Etot[his-1];
  }

  /* store CntCoes_Species and D_CntCoes_Spe into the His arrays */

  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	His_CntCoes_Species[0][wan][al][p]   = CntCoes_Species[wan][al][p];
	His_D_CntCoes_Species[0][wan][al][p] = DCntCoes_Spe[wan][al][p];
      }
    }
  }    

  /* store the total energy into the His arrays */

  His_OrbOpt_Etot[0] = TotalE;

  /* update CntCoes by a simple mixing if (diis_dim==1) */
  /* if diis_dim==1, find mixing_coef */

  if (diis_dim==1){

    MaxStep = 0.0;
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	  if (MaxStep<fabs(DCntCoes_Spe[wan][al][p])) MaxStep = fabs(DCntCoes_Spe[wan][al][p]);
	}
      }
    }

    mixing_coef = orbitalOpt_SD_step/MaxStep;

    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	  CntCoes_Species[wan][al][p] -= mixing_coef*DCntCoes_Spe[wan][al][p];
	}
      }
    }
  }

  /* otherwise, update CntCoes by RMM-DIIS */

  else {

    /* construct the norm matrix of His_D_CntCoes */

    for (his1=0; his1<diis_dim; his1++){
      for (his2=0; his2<diis_dim; his2++){

	sum = 0.0;  

        for (wan=0; wan<SpeciesNum; wan++){
          for (al=0; al<Spe_Total_CNO[wan]; al++){
            for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	      sum += His_D_CntCoes_Species[his1][wan][al][p]*His_D_CntCoes_Species[his2][wan][al][p];
	    }
	  }
	}

	A[his1][his2] = sum;

	/*
        if (his1==his2) B[his1] =-0.001*His_OrbOpt_Etot[his1];
	*/

	if (his1==his2) B[his1] = 0.0;

      }
    }
    B[diis_dim] = 1.0;

    /*
    printf("A\n");
    for (his1=0; his1<diis_dim; his1++){
      for (his2=0; his2<diis_dim; his2++){
        printf("%15.10f ",A[his1][his2]);
      }
      printf("\n");
    }
    */

    /* solve the linear equation */

    for (his1=1; his1<=diis_dim; his1++){
      A[his1-1][diis_dim] = 1.0;
      A[diis_dim][his1-1] = 1.0;
    }
    A[diis_dim][diis_dim] = 0.0;

    /*
    Inverse(diis_dim,A,IA);
    for (his1=0; his1<diis_dim; his1++){
      DIIS_coes[his1] = IA[his1][diis_dim]; 
    }
    for (his1=0; his1<diis_dim; his1++){
      printf("his1=%2d DIIS_coes[his1]=%15.10f\n",his1,DIIS_coes[his1]);
    }
    */

    Inverse_Mat(diis_dim,A,IA);

    for (his1=0; his1<diis_dim; his1++){

      sum = 0.0; 
      for (his2=0; his2<=diis_dim; his2++){
        sum += IA[his1][his2]*B[his2];
      }    
      DIIS_coes[his1] = sum;
    }    

    /*
    for (his1=0; his1<diis_dim; his1++){
      DIIS_coes[his1] = IA[his1][diis_dim]; 
    }
    */

    /*
    for (his1=0; his1<diis_dim; his1++){
      printf("his1=%2d DIIS_coes[his1]=%15.10f\n",his1,DIIS_coes[his1]);
    }
    */

    /* construct optimum contraction coefficients in the DIIS sence */

    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
	for (p=0; p<Spe_Specified_Num[wan][al]; p++){

	  sum1 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum1 += DIIS_coes[his1]*His_CntCoes_Species[his1][wan][al][p];
	  }      

	  sum2 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum2 += DIIS_coes[his1]*His_D_CntCoes_Species[his1][wan][al][p];
	  }      

	  CntCoes_Species[wan][al][p] = sum1;
          DCntCoes_Spe[wan][al][p]    = sum2;
	}
      }
    }

    /* update an approximate Hessian matrix by the BFGS method */
    /*
     H_k = H_{k-1} + y*y^t/(y^t*y) - H_{k-1}*s * s^t * H_{k-1}^t /(s^t * H_{k-1} * s)
     y = His_D_CntCoes_Species[0][wan][al][p] - His_D_CntCoes_Species[1][wan][al][p];
     s = His_CntCoes_Species[0][wan][al][p] - His_CntCoes_Species[1][wan][al][p];
    */

    /* H*s */

    for (i=0; i<dim_H; i++){

      j = 0;
      sum = 0.0;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
  	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){

            d = His_CntCoes_Species[0][wan][al][p] - His_CntCoes_Species[1][wan][al][p];    
	    sum += OrbOpt_Hessian[i][j]*d;
            j++;
	  }
	}
      }

      /* store H*s */

      TmpVec1[i] = sum;
    }

    /* tmp1 = y^t*s, tmp2 = s^t*H*s */ 

    tmp1 = 0.0;
    tmp2 = 0.0;

    i = 0;
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
	for (p=0; p<Spe_Specified_Num[wan][al]; p++){

          d1 = His_D_CntCoes_Species[0][wan][al][p]-His_D_CntCoes_Species[1][wan][al][p];
          d2 = His_CntCoes_Species[0][wan][al][p] - His_CntCoes_Species[1][wan][al][p];

          tmp1 += d1*d2;
          tmp2 += d2*TmpVec1[i]; 

          i++;
	}
      }
    }

    /* c0=1.0/tmp1, c1=1.0/tmp2 */

    c0 = 1.0/tmp1;
    c1 = 1.0/tmp2;

    /* update the approximate Hessian by the BFGS method if 0.0<c0 */

    if (0.0<c0){

      i = 0;
      for (wan1=0; wan1<SpeciesNum; wan1++){
	for (al1=0; al1<Spe_Total_CNO[wan1]; al1++){
	  for (p1=0; p1<Spe_Specified_Num[wan1][al1]; p1++){

	    j = 0;
	    for (wan2=0; wan2<SpeciesNum; wan2++){
	      for (al2=0; al2<Spe_Total_CNO[wan2]; al2++){
		for (p2=0; p2<Spe_Specified_Num[wan2][al2]; p2++){

		  d1 = His_D_CntCoes_Species[0][wan1][al1][p1]-His_D_CntCoes_Species[1][wan1][al1][p1];
		  d2 = His_D_CntCoes_Species[0][wan2][al2][p2]-His_D_CntCoes_Species[1][wan2][al2][p2];
		  OrbOpt_Hessian[i][j] += c0*d1*d2 -c1*TmpVec1[i]*TmpVec1[j];                

		  j++;
		}
	      }
	    }

	    i++; 
	  }
	}
      }
    }

    /* diagonalize the approximate Hessian */

    for (i=0; i<dim_H; i++){
      for (j=0; j<dim_H; j++){
        U[i+1][j+1] = OrbOpt_Hessian[i][j];
      }    
    }    

    Eigen_lapack(U,ko,dim_H,dim_H);

    /* correct small eigenvalues by MinKo */

    MinKo = 0.1;
    for (i=0; i<dim_H; i++){
      if (ko[i+1]<MinKo) ko[i+1] = MinKo;
    }    

    /* U*lambda^{-1} U^{dag} * g */   

    for (i=0; i<dim_H; i++){

      j = 1;
      sum = 0.0;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){

	    sum += U[j][i+1]*DCntCoes_Spe[wan][al][p]; 
	    j++;
	  }
	}
      }

      U[i+1][0] = sum;
    }

    for (i=0; i<dim_H; i++){

      j = 1;
      sum = 0.0;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){

	    sum += U[i+1][j]*U[j][0]/ko[j];
	    j++;
	  }
	}
      }

      U[0][i+1] = sum;
    }

    /* avoid a too large movement */

    MaxStep = 0.0;
    for (i=1; i<=dim_H; i++){
      if (MaxStep<fabs(U[0][i])) MaxStep = fabs(U[0][i]);
    }

    if (0.001<MaxStep){
      for (i=1; i<=dim_H; i++){
	U[0][i] = 0.001*U[0][i]/MaxStep;
      }
    }

    /* perform a quasi Newton method */

    if (OrbOpt_OptMethod==2){

      i = 1;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){

	    CntCoes_Species[wan][al][p] -= U[0][i];
	    i++;
	  }
	}
      }

    }
 
    else if (OrbOpt_OptMethod==1) {

      if      (1.0e-2<norm_deri)  coef_OptG = 0.05;
      else if (1.0e-3<norm_deri)  coef_OptG = 0.10;
      else if (1.0e-4<norm_deri)  coef_OptG = 0.15;
      else if (1.0e-5<norm_deri)  coef_OptG = 0.20;
      else if (1.0e-6<norm_deri)  coef_OptG = 0.40;

      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	    CntCoes_Species[wan][al][p] -= coef_OptG*DCntCoes_Spe[wan][al][p];
	  }
	}
      }

    }

  } /* else */

  /********************************************************
    output norm_deri
  ********************************************************/

  Oopt_NormD[1] = norm_deri;  
  if (myid==Host_ID) printf("<Opt_Contraction> Norm of derivatives = %15.12f \n",norm_deri);

  /********************************************************
     MPI: diagonal block elements of OLP
  ********************************************************/

  for (wan=0; wan<SpeciesNum; wan++){

    Mc_AN = 1;
    po = 0;

    while (po==0 && Mc_AN<=Matomnum){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      if (wan==Cwan){
        po = 1;
        Mc_AN1 = Mc_AN;
      }

      Mc_AN++;
    } 

    my_po = po*(myid+1);
    MPI_Allreduce(&my_po, &po, 1, MPI_INT, MPI_MAX, mpi_comm_level1);    
    ID = po - 1;

    if (myid==ID){
      for (al=0; al<Spe_Total_NO[wan]; al++){
	for (be=0; be<Spe_Total_NO[wan]; be++){
          OLPtmp[wan][al][be] = OLP[0][Mc_AN1][0][al][be];
	}
      }
    }

    for (al=0; al<Spe_Total_NO[wan]; al++){
      MPI_Bcast(&OLPtmp[wan][al][0], Spe_Total_NO[wan], MPI_DOUBLE, ID, mpi_comm_level1);  
    }

  } /* wan */

  /********************************************************
        normalization of contracted basis functions
  ********************************************************/

  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){

      sum = 0.0;
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	p0 = Spe_Trans_Orbital[wan][al][p];

	for (q=0; q<Spe_Specified_Num[wan][al]; q++){
	  q0 = Spe_Trans_Orbital[wan][al][q];

	  tmp0 = CntCoes_Species[wan][al][p]*CntCoes_Species[wan][al][q];
	  sum += tmp0*OLPtmp[wan][p0][q0]; 
	}
      }

      tmp0 = 1.0/sqrt(sum);
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	CntCoes_Species[wan][al][p] = CntCoes_Species[wan][al][p]*tmp0;
      }        
    }
  }

  if (2<=level_stdout){
    printf("<Opt_Contraction> Contraction coefficients\n");fflush(stdout);  
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          printf("  wan=%2d al=%2d p=%2d  %15.12f\n",
                    wan,al,p,CntCoes_Species[wan][al][p]);fflush(stdout);  
        }
      }
    }
  }

  /* CntCoes_Species to CntCoes */

  for (Mc_AN=1; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        CntCoes[Mc_AN][al][p] = CntCoes_Species[wan][al][p];
      }        
    }
  }

  /********************************************************
    freeing of arrays:

    double D_CntCoes[Matomnum+1]
                           [List_YOUSO[7]]
                           [List_YOUSO[24]];
  ********************************************************/

  free(ko);

  for (i=0; i<(dim_H+5); i++){
    free(U[i]);
  }
  free(U);

  free(TmpVec1);
  free(TmpVec2);

  for (i=0; i<=SpeciesNum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(OLPtmp[i][j]);
    }
    free(OLPtmp[i]);
  }
  free(OLPtmp);

  for (i=0; i<=Matomnum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(D_CntCoes[i][j]);
    }
    free(D_CntCoes[i]);
  }
  free(D_CntCoes);

  for (i=0; i<=SpeciesNum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(DCntCoes_Spe[i][j]);
    }
    free(DCntCoes_Spe[i]);
  }
  free(DCntCoes_Spe);

  free(B);

  for (i=0; i<(orbitalOpt_History+2); i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<(orbitalOpt_History+2); i++){
    free(My_A[i]);
  }
  free(My_A);

  for (i=0; i<(orbitalOpt_History+2); i++){
    free(IA[i]);
  }
  free(IA);

  free(DIIS_coes);

  if (firsttime) {
    firsttime=0;
  }

  /* return */
  return Oopt_NormD[1];
}



double Species_Opt_Contraction_EDIIS(
         int orbitalOpt_iter,
         double TotalE,
         double *****H, 
         double *****OLP,
         double *****CDM,
         double *****EDM,
         double ****His_CntCoes,
         double ****His_D_CntCoes,
         double ****His_CntCoes_Species,
         double ****His_D_CntCoes_Species,
         double *His_OrbOpt_Etot,
         double **OrbOpt_Hessian)
{
  static int firsttime=1;
  int i,j,Mc_AN,Mc_AN1,Cwan,Gc_AN,be,his,po,my_po;
  int his1,his2,diis_dim,al,p,q,p0,q0;
  int n,num,wan,size1,size2,dim_H;
  int wan1,al1,p1,wan2,al2,p2;
  double nd[4],x01,x12,x20,y12,y01;
  double a,b,c,alpha,c0,c1,MinKo,MaxStep;
  double tmp1,tmp2,d1,d2,d;
  double norm_deri,sum,sum1,sum2;
  double coef_OptG,tmp0,mixing_coef;
  double **My_A,**A,**IA,*B;
  double *DIIS_coes;
  double ***D_CntCoes;
  double ***DCntCoes_Spe;
  double ***OLPtmp;
  double *TmpVec1;
  double *TmpVec2;
  double **U,*ko;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************************
    allocation of arrays:

    double D_CntCoes[Matomnum+1]
                           [List_YOUSO[7]]
                           [List_YOUSO[24]];

    double DCntCoes_Spe[SpeciesNum+1]
                              [List_YOUSO[7]]
                              [List_YOUSO[24]];
  ********************************************************/

  DIIS_coes = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));

  IA = (double**)malloc(sizeof(double*)*(orbitalOpt_History+2));
  for (i=0; i<(orbitalOpt_History+2); i++){
    IA[i] = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));
  }

  My_A = (double**)malloc(sizeof(double*)*(orbitalOpt_History+2));
  for (i=0; i<(orbitalOpt_History+2); i++){
    My_A[i] = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));
  }

  A = (double**)malloc(sizeof(double*)*(orbitalOpt_History+2));
  for (i=0; i<(orbitalOpt_History+2); i++){
    A[i] = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));
  }

  B = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));

  D_CntCoes = (double***)malloc(sizeof(double**)*(Matomnum+1));
  for (i=0; i<=Matomnum; i++){
    D_CntCoes[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      D_CntCoes[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  DCntCoes_Spe = (double***)malloc(sizeof(double**)*(SpeciesNum+1));
  for (i=0; i<=SpeciesNum; i++){
    DCntCoes_Spe[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      DCntCoes_Spe[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  OLPtmp = (double***)malloc(sizeof(double**)*(SpeciesNum+1));
  for (i=0; i<=SpeciesNum; i++){
    OLPtmp[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      OLPtmp[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }
  }

  /********************************************************
    PrintMemory
  ********************************************************/

  if (firsttime) {
    PrintMemory("Opt_Contraction: D_CntCoes",sizeof(double)*
           (Matomnum+1)*List_YOUSO[7]*List_YOUSO[24], NULL);
    firsttime=0;
  }

  /********************************************************
    start calc.
  ********************************************************/

  /* find the dimension of Hessian */ 

  dim_H = 0;
  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        dim_H++;
      }
    }
  }

  /* initialize OrbOpt_Hessian */

  if (orbitalOpt_iter==1){

    for (i=0; i<dim_H; i++){
      for (j=0; j<dim_H; j++){
        OrbOpt_Hessian[i][j] = 0.0;
      }    
      OrbOpt_Hessian[i][i] = 1.0;
    }    
  } 

  TmpVec1 = (double*)malloc(sizeof(double)*(dim_H+1));
  TmpVec2 = (double*)malloc(sizeof(double)*(dim_H+1));

  U = (double**)malloc(sizeof(double*)*(dim_H+5));
  for (i=0; i<(dim_H+5); i++){
    U[i] = (double*)malloc(sizeof(double)*(dim_H+5));
  }

  ko = (double*)malloc(sizeof(double)*(dim_H+5));

  /* calculate gradients of total energy with respect to contraction coefficients */
  /* species dependent contraction */

  D_ACntCoes(CntCoes, D_CntCoes, DCntCoes_Spe, H, OLP, CDM, EDM);
  norm_deri = NormD_ACnt(DCntCoes_Spe);

  /* find diis_dim */

  if (orbitalOpt_History<(orbitalOpt_iter-orbitalOpt_StartPulay+1)) 
    diis_dim = orbitalOpt_History;
  else 
    diis_dim = orbitalOpt_iter-orbitalOpt_StartPulay+1;

  if (diis_dim<1) diis_dim = 1;

  /* shift the His arrays */

  for (his=(diis_dim-1); 0<his; his--){
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
	for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	  His_CntCoes_Species[his][wan][al][p] = His_CntCoes_Species[his-1][wan][al][p];
	  His_D_CntCoes_Species[his][wan][al][p] = His_D_CntCoes_Species[his-1][wan][al][p];
	}
      }
    }    

    His_OrbOpt_Etot[his] = His_OrbOpt_Etot[his-1];
  }

  /* store CntCoes_Species and D_CntCoes_Spe into the His arrays */

  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	His_CntCoes_Species[0][wan][al][p]   = CntCoes_Species[wan][al][p];
	His_D_CntCoes_Species[0][wan][al][p] = DCntCoes_Spe[wan][al][p];
      }
    }
  }    

  /* store the total energy into the His arrays */

  His_OrbOpt_Etot[0] = TotalE;

  /* update CntCoes by a simple mixing if (diis_dim==1) */
  /* if diis_dim==1, find mixing_coef */

  if (diis_dim==1){

    MaxStep = 0.0;
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	  if (MaxStep<fabs(DCntCoes_Spe[wan][al][p])) MaxStep = fabs(DCntCoes_Spe[wan][al][p]);
	}
      }
    }

    mixing_coef = orbitalOpt_SD_step/MaxStep;

    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	  CntCoes_Species[wan][al][p] -= mixing_coef*DCntCoes_Spe[wan][al][p];
	}
      }
    }
  }

  /* otherwise, update CntCoes by Energy DIIS */

  else {

    /* construct the norm matrix of His_CntCoes and His_D_CntCoes */

    for (his1=0; his1<diis_dim; his1++){
      for (his2=0; his2<diis_dim; his2++){

	sum1 = 0.0;  
	sum2 = 0.0;  

        for (wan=0; wan<SpeciesNum; wan++){
          for (al=0; al<Spe_Total_CNO[wan]; al++){
            for (p=0; p<Spe_Specified_Num[wan][al]; p++){

	      sum1 += (His_CntCoes_Species[his1][wan][al][p]-His_CntCoes_Species[his2][wan][al][p])
                     *(His_D_CntCoes_Species[his1][wan][al][p]-His_D_CntCoes_Species[his2][wan][al][p]);

	      sum2 += (His_CntCoes_Species[his2][wan][al][p]-His_CntCoes_Species[his1][wan][al][p])
                     *(His_D_CntCoes_Species[his2][wan][al][p]-His_D_CntCoes_Species[his1][wan][al][p]);


	      /*
	      sum1 += His_CntCoes_Species[his1][wan][al][p]*His_D_CntCoes_Species[his2][wan][al][p];
	      sum2 += His_D_CntCoes_Species[his1][wan][al][p]*His_CntCoes_Species[his2][wan][al][p];

              printf("his1=%2d his2=%2d C=%18.15f D=%18.15f\n",his1,his2,
                    His_CntCoes_Species[his1][wan][al][p],His_D_CntCoes_Species[his2][wan][al][p]);
              printf("his1=%2d his2=%2d sum1=%18.15f sum2=%18.15f\n",his1,his2,sum1,sum2);
	      */

	    }
	  }
	}

        printf("\n");

	A[his1][his2] = 0.5*(sum1 + sum2);

	/*
        if (his1==his2) B[his1] = sum1 - His_OrbOpt_Etot[his1];
	*/

        if (his1==his2) B[his1] = His_OrbOpt_Etot[his1];

      }
    }
    B[diis_dim] = 1.0;

    /* solve the linear equation */

    for (his1=1; his1<=diis_dim; his1++){
      A[his1-1][diis_dim] = 1.0;
      A[diis_dim][his1-1] = 1.0;
    }
    A[diis_dim][diis_dim] = 0.0;


    printf("ZZZ A\n");
    for (his1=0; his1<=diis_dim; his1++){
      for (his2=0; his2<=diis_dim; his2++){
        printf("%15.10f ",A[his1][his2]);
      }
      printf("\n");
    }

    printf("ZZZ B\n");
    for (his1=0; his1<=diis_dim; his1++){
      printf("%15.10f\n",B[his1]);
    }

    /*
    Inverse(diis_dim,A,IA);
    for (his1=0; his1<diis_dim; his1++){
      DIIS_coes[his1] = IA[his1][diis_dim]; 
    }
    for (his1=0; his1<diis_dim; his1++){
      printf("his1=%2d DIIS_coes[his1]=%15.10f\n",his1,DIIS_coes[his1]);
    }
    */

    Inverse_Mat(diis_dim,A,IA);

    for (his1=0; his1<diis_dim; his1++){

      sum = 0.0; 
      for (his2=0; his2<=diis_dim; his2++){
        sum += IA[his1][his2]*B[his2];
      }    
      DIIS_coes[his1] = sum;
    }    

    for (his1=0; his1<diis_dim; his1++){
      printf("his1=%2d DIIS_coes[his1]=%15.10f\n",his1,DIIS_coes[his1]);
    }


    /* construct optimum contraction coefficients in the DIIS sence */

    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
	for (p=0; p<Spe_Specified_Num[wan][al]; p++){

	  sum1 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum1 += DIIS_coes[his1]*His_CntCoes_Species[his1][wan][al][p];
	  }      

	  sum2 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum2 += DIIS_coes[his1]*His_D_CntCoes_Species[his1][wan][al][p];
	  }      

	  CntCoes_Species[wan][al][p] = sum1;
          DCntCoes_Spe[wan][al][p]    = sum2;
	}
      }
    }

    /* update an approximate Hessian matrix by the BFGS method */
    /*
     H_k = H_{k-1} + y*y^t/(y^t*y) - H_{k-1}*s * s^t * H_{k-1}^t /(s^t * H_{k-1} * s)
     y = His_D_CntCoes_Species[0][wan][al][p] - His_D_CntCoes_Species[1][wan][al][p];
     s = His_CntCoes_Species[0][wan][al][p] - His_CntCoes_Species[1][wan][al][p];
    */

    /* H*s */

    for (i=0; i<dim_H; i++){

      j = 0;
      sum = 0.0;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
  	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){

            d = His_CntCoes_Species[0][wan][al][p] - His_CntCoes_Species[1][wan][al][p];    
	    sum += OrbOpt_Hessian[i][j]*d;
            j++;
	  }
	}
      }

      /* store H*s */

      TmpVec1[i] = sum;
    }

    /* tmp1 = y^t*s, tmp2 = s^t*H*s */ 

    tmp1 = 0.0;
    tmp2 = 0.0;

    i = 0;
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
	for (p=0; p<Spe_Specified_Num[wan][al]; p++){

          d1 = His_D_CntCoes_Species[0][wan][al][p]-His_D_CntCoes_Species[1][wan][al][p];
          d2 = His_CntCoes_Species[0][wan][al][p] - His_CntCoes_Species[1][wan][al][p];

          tmp1 += d1*d2;
          tmp2 += d2*TmpVec1[i]; 

          i++;
	}
      }
    }

    /* c0=1.0/tmp1, c1=1.0/tmp2 */

    c0 = 1.0/tmp1;
    c1 = 1.0/tmp2;

    /* update the approximate Hessian by the BFGS method if 0.0<c0 */

    if (0.0<c0){

      i = 0;
      for (wan1=0; wan1<SpeciesNum; wan1++){
	for (al1=0; al1<Spe_Total_CNO[wan1]; al1++){
	  for (p1=0; p1<Spe_Specified_Num[wan1][al1]; p1++){

	    j = 0;
	    for (wan2=0; wan2<SpeciesNum; wan2++){
	      for (al2=0; al2<Spe_Total_CNO[wan2]; al2++){
		for (p2=0; p2<Spe_Specified_Num[wan2][al2]; p2++){

		  d1 = His_D_CntCoes_Species[0][wan1][al1][p1]-His_D_CntCoes_Species[1][wan1][al1][p1];
		  d2 = His_D_CntCoes_Species[0][wan2][al2][p2]-His_D_CntCoes_Species[1][wan2][al2][p2];
		  OrbOpt_Hessian[i][j] += c0*d1*d2 -c1*TmpVec1[i]*TmpVec1[j];                

		  j++;
		}
	      }
	    }

	    i++; 
	  }
	}
      }
    }

    /* diagonalize the approximate Hessian */

    for (i=0; i<dim_H; i++){
      for (j=0; j<dim_H; j++){
        U[i+1][j+1] = OrbOpt_Hessian[i][j];
      }    
    }    

    Eigen_lapack(U,ko,dim_H,dim_H);

    /* correct small eigenvalues by MinKo */

    MinKo = 0.1;
    for (i=0; i<dim_H; i++){
      if (ko[i+1]<MinKo) ko[i+1] = MinKo;
    }    

    /* U*lambda^{-1} U^{dag} * g */   

    for (i=0; i<dim_H; i++){

      j = 1;
      sum = 0.0;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){

	    sum += U[j][i+1]*DCntCoes_Spe[wan][al][p]; 
	    j++;
	  }
	}
      }

      U[i+1][0] = sum;
    }

    for (i=0; i<dim_H; i++){

      j = 1;
      sum = 0.0;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){

	    sum += U[i+1][j]*U[j][0]/ko[j];
	    j++;
	  }
	}
      }

      U[0][i+1] = sum;
    }

    /* avoid a too large movement */

    MaxStep = 0.0;
    for (i=1; i<=dim_H; i++){
      if (MaxStep<fabs(U[0][i])) MaxStep = fabs(U[0][i]);
    }

    if (0.001<MaxStep){
      for (i=1; i<=dim_H; i++){
	U[0][i] = 0.001*U[0][i]/MaxStep;
      }
    }

    /* perform a quasi Newton method */

    if (OrbOpt_OptMethod==2){

      i = 1;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){

	    CntCoes_Species[wan][al][p] -= U[0][i];
	    i++;
	  }
	}
      }

    }
 
    else if (OrbOpt_OptMethod==1) {

      if      (1.0e-2<norm_deri)  coef_OptG = 0.05;
      else if (1.0e-3<norm_deri)  coef_OptG = 0.10;
      else if (1.0e-4<norm_deri)  coef_OptG = 0.15;
      else if (1.0e-5<norm_deri)  coef_OptG = 0.20;
      else if (1.0e-6<norm_deri)  coef_OptG = 0.40;

      coef_OptG = 0.00;


      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	    CntCoes_Species[wan][al][p] -= coef_OptG*DCntCoes_Spe[wan][al][p];
	  }
	}
      }

    }

  } /* else */

  /********************************************************
    output norm_deri
  ********************************************************/

  Oopt_NormD[1] = norm_deri;  
  if (myid==Host_ID) printf("<Opt_Contraction> Norm of derivatives = %15.12f \n",norm_deri);

  /********************************************************
     MPI: diagonal block elements of OLP
  ********************************************************/

  for (wan=0; wan<SpeciesNum; wan++){

    Mc_AN = 1;
    po = 0;
    do {

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      if (wan==Cwan){
        po = 1;
        Mc_AN1 = Mc_AN;
      }

      Mc_AN++;

    } while (po==0 && Mc_AN<=Matomnum);

    my_po = po*(myid+1);
    MPI_Allreduce(&my_po, &po, 1, MPI_INT, MPI_MAX, mpi_comm_level1);    
    ID = po - 1;

    if (myid==ID){
      for (al=0; al<Spe_Total_NO[wan]; al++){
	for (be=0; be<Spe_Total_NO[wan]; be++){
          OLPtmp[wan][al][be] = OLP[0][Mc_AN1][0][al][be];
	}
      }
    }

    for (al=0; al<Spe_Total_NO[wan]; al++){
      MPI_Bcast(&OLPtmp[wan][al][0], Spe_Total_NO[wan], MPI_DOUBLE, ID, mpi_comm_level1);  
    }

  } /* wan */

  /********************************************************
        normalization of contracted basis functions
  ********************************************************/

  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){

      sum = 0.0;
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	p0 = Spe_Trans_Orbital[wan][al][p];

	for (q=0; q<Spe_Specified_Num[wan][al]; q++){
          q0 = Spe_Trans_Orbital[wan][al][q];

          tmp0 = CntCoes_Species[wan][al][p]*CntCoes_Species[wan][al][q];
          sum += tmp0*OLPtmp[wan][p0][q0]; 
        }
      }

      tmp0 = 1.0/sqrt(sum);
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        CntCoes_Species[wan][al][p] = CntCoes_Species[wan][al][p]*tmp0;
      }        
    }
  }

  if (2<=level_stdout){
    printf("<Opt_Contraction> Contraction coefficients and gradients\n");fflush(stdout);  
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          printf("  wan=%2d al=%2d p=%2d  %15.12f %15.12f\n",
                    wan,al,p,CntCoes_Species[wan][al][p],
                    His_D_CntCoes_Species[0][wan][al][p]);fflush(stdout);  
        }
      }
    }
  }

  /* CntCoes_Species to CntCoes */
  for (Mc_AN=1; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        CntCoes[Mc_AN][al][p] = CntCoes_Species[wan][al][p];
      }        
    }
  }

  /********************************************************
    freeing of arrays:

    double D_CntCoes[Matomnum+1]
                           [List_YOUSO[7]]
                           [List_YOUSO[24]];
  ********************************************************/

  free(ko);

  for (i=0; i<(dim_H+5); i++){
    free(U[i]);
  }
  free(U);

  free(TmpVec1);
  free(TmpVec2);

  for (i=0; i<=SpeciesNum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(OLPtmp[i][j]);
    }
    free(OLPtmp[i]);
  }
  free(OLPtmp);

  for (i=0; i<=Matomnum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(D_CntCoes[i][j]);
    }
    free(D_CntCoes[i]);
  }
  free(D_CntCoes);

  for (i=0; i<=SpeciesNum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(DCntCoes_Spe[i][j]);
    }
    free(DCntCoes_Spe[i]);
  }
  free(DCntCoes_Spe);

  free(B);

  for (i=0; i<(orbitalOpt_History+2); i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<(orbitalOpt_History+2); i++){
    free(My_A[i]);
  }
  free(My_A);

  for (i=0; i<(orbitalOpt_History+2); i++){
    free(IA[i]);
  }
  free(IA);

  free(DIIS_coes);

  if (firsttime) {
    firsttime=0;
  }

  /* return */
  return Oopt_NormD[1];
}




void D_RCntCoes(double ***CntCoes0,
                double ***D_CntCoes,
                double *****H, double *****OLP,
                double *****CDM,
                double *****EDM)
{
  static int firsttime=1;
  int i,j,Cwan,Mc_AN,Gc_AN,Mh_AN;
  int p,q,al,al0,be,p0,q0,wan;
  int h_AN,Gh_AN,Hwan,spin;
  int L0,Mul0,M0;
  double sum,sum0;
  double *TmpD;
  double ***D0_CntCoes;

  /********************************************************
    allocation of arrays:

    double TmpD[List_YOUSO[24]];

    double D0_CntCoes[Matomnum+1]
                            [List_YOUSO[7]]
                            [List_YOUSO[24]];

  ********************************************************/

  TmpD = (double*)malloc(sizeof(double)*List_YOUSO[24]);

  D0_CntCoes = (double***)malloc(sizeof(double**)*(Matomnum+1));
  for (i=0; i<=Matomnum; i++){
    D0_CntCoes[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      D0_CntCoes[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  /* PrintMemory */

  if (firsttime) {
    PrintMemory("D_RCntCoes: D0_CntCoes",sizeof(D0_CntCoes),NULL);
    firsttime=0;
  }

  /****************************************************
            Calculate D_CntCoes[Mc_AN][al][p]
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[Cwan]; al++){
      for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	p0 = Spe_Trans_Orbital[Cwan][al][p];

	sum = 0.0;
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
          Mh_AN = F_G2M[Gh_AN];
	  for (be=0; be<Spe_Total_CNO[Hwan]; be++){
	    for (q=0; q<Spe_Specified_Num[Hwan][be]; q++){
	      q0 = Spe_Trans_Orbital[Hwan][be][q];

	      sum0 = 0.0;
	      for (spin=0; spin<=SpinP_switch; spin++){
		sum0 = sum0 +
		   CDM[spin][Mc_AN][h_AN][al][be]*H[spin][Mc_AN][h_AN][p0][q0]
		  -EDM[spin][Mc_AN][h_AN][al][be]*OLP[0][Mc_AN][h_AN][p0][q0];
	      } 
	      if (SpinP_switch==0) sum0 = 2.0*sum0; 

	      sum = sum + 2.0*sum0*CntCoes0[Mh_AN][be][q];
        
	    }
	  }
	}

	D0_CntCoes[Mc_AN][al][p] = sum;

	/*
          printf("step=%2d Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
	  step,Mc_AN,Gc_AN,al,p,sum);
	*/

      }
    }

    /****************************************************
          taking into account of the restriction
    ****************************************************/
          
    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_CBasis[Cwan][L0]; Mul0++){

	for (p=0; p<Spe_Specified_Num[Cwan][al+1]; p++){
	  TmpD[p] = 0.0;
	}

	al0 = al;
	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	    TmpD[p] = TmpD[p] + D0_CntCoes[Mc_AN][al][p];
	  }
	}
	al = al0;

	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	    D_CntCoes[Mc_AN][al][p] = TmpD[p];
	  }
	}
      }
    }      
  }

  /*************************************************************
    in case of optimization of only the last orbital in each L
  *************************************************************/

  if (SCnt_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];    
      wan = WhatSpecies[Gc_AN];

      al = -1;
      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	  for (M0=0; M0<=2*L0; M0++){

	    al++;

            if ( Mul0!=(Spe_Num_CBasis[wan][L0]-1) ){

	      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
		D_CntCoes[Mc_AN][al][p] = 0.0;
	      }
            }

	  }
	}
      }
    }

  }

  /********************************************************
    freeing of arrays:

    double TmpD[List_YOUSO[24]];

    double D0_CntCoes[Matomnum+1]
                            [List_YOUSO[7]]
                            [List_YOUSO[24]];
  ********************************************************/

  free(TmpD);

  for (i=0; i<=Matomnum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(D0_CntCoes[i][j]);
    }
    free(D0_CntCoes[i]);
  }
  free(D0_CntCoes);

}




void D_ACntCoes(double ***CntCoes0,
                double ***D_CntCoes,
                double ***DCntCoes_Spe,
                double *****H, double *****OLP,
                double *****CDM,
                double *****EDM)
{
  static int firsttime=1;
  int i,j,Cwan,Mc_AN,Gc_AN,Mh_AN;
  int p,q,al,al0,be,p0,q0;
  int h_AN,Gh_AN,Hwan,spin;
  int L0,Mul0,M0,wan;
  double sum,sum0;
  double *TmpD;
  double ***D0_CntCoes;
  double ***My_DCntCoes_Spe;

  /********************************************************
    allocation of arrays:

    double TmpD[List_YOUSO[24]];

    double D0_CntCoes[Matomnum+1]
                            [List_YOUSO[7]]
                            [List_YOUSO[24]];


    double My_DCntCoes_Spe[SpeciesNum+1]
                                 [List_YOUSO[7]]
                                 [List_YOUSO[24]];
  ********************************************************/

  TmpD = (double*)malloc(sizeof(double)*List_YOUSO[24]);

  D0_CntCoes = (double***)malloc(sizeof(double**)*(Matomnum+1));
  for (i=0; i<=Matomnum; i++){
    D0_CntCoes[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      D0_CntCoes[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  My_DCntCoes_Spe = (double***)malloc(sizeof(double**)*(SpeciesNum+1));
  for (i=0; i<=SpeciesNum; i++){
    My_DCntCoes_Spe[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      My_DCntCoes_Spe[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
    }
  }

  /* PrintMemory */

  if (firsttime) {
    PrintMemory("D_RCntCoes: D0_CntCoes",sizeof(D0_CntCoes),NULL);
    firsttime=0;
  }

  /****************************************************
            Calculate D_CntCoes[Mc_AN][al][p]
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[Cwan]; al++){
      for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	p0 = Spe_Trans_Orbital[Cwan][al][p];

	sum = 0.0;
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
          Mh_AN = F_G2M[Gh_AN];
	  for (be=0; be<Spe_Total_CNO[Hwan]; be++){
	    for (q=0; q<Spe_Specified_Num[Hwan][be]; q++){
	      q0 = Spe_Trans_Orbital[Hwan][be][q];

	      sum0 = 0.0;
	      for (spin=0; spin<=SpinP_switch; spin++){
		sum0 = sum0 +
		   CDM[spin][Mc_AN][h_AN][al][be]*H[spin][Mc_AN][h_AN][p0][q0]
		  -EDM[spin][Mc_AN][h_AN][al][be]*OLP[0][Mc_AN][h_AN][p0][q0];
	      } 

	      if (SpinP_switch==0) sum0 = 2.0*sum0; 

	      sum = sum + 2.0*sum0*CntCoes0[Mh_AN][be][q];
        
	    }
	  }
	}

	D0_CntCoes[Mc_AN][al][p] = sum;

	/*
          printf("step=%2d Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
	  step,Mc_AN,Gc_AN,al,p,sum);
	*/

      }
    }

    /****************************************************
          taking into account of the restriction
    ****************************************************/
          
    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_CBasis[Cwan][L0]; Mul0++){

	for (p=0; p<Spe_Specified_Num[Cwan][al+1]; p++){
	  TmpD[p] = 0.0;
	}

	al0 = al;
	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	    TmpD[p] = TmpD[p] + D0_CntCoes[Mc_AN][al][p];
	  }
	}
	al = al0;

	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	    D_CntCoes[Mc_AN][al][p] = TmpD[p];
	  }
	}
      }
    }      
  }

  /*************************************************************
    in case of optimization of only the last orbital in each L
  *************************************************************/

  if (SCnt_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];    
      wan = WhatSpecies[Gc_AN];

      al = -1;
      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	  for (M0=0; M0<=2*L0; M0++){

	    al++;

            if ( Mul0!=(Spe_Num_CBasis[wan][L0]-1) ){

	      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
		D_CntCoes[Mc_AN][al][p] = 0.0;
	      }
            }

	  }
	}
      }
    }

  }

  /********************************************************
    calculate derivatives with respect to CntCoes_Spe
  ********************************************************/

  for (wan=0; wan<SpeciesNum; wan++){
    for (i=0; i<List_YOUSO[7]; i++){
      for (j=0; j<List_YOUSO[24]; j++){
        My_DCntCoes_Spe[wan][i][j] = 0.0;
      }
    }
  }

  /* local sum in a proccessor */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        My_DCntCoes_Spe[wan][al][p] += D_CntCoes[Mc_AN][al][p];
      }        
    }
  }

  /* global sum by MPI */
  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        MPI_Allreduce(&My_DCntCoes_Spe[wan][al][p], &DCntCoes_Spe[wan][al][p],
                       1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        
      }
    }
  }    

  /* DCntCoes_Spe to DCntCoes */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        D_CntCoes[Mc_AN][al][p] = DCntCoes_Spe[wan][al][p];
      }        
    }
  }

  /********************************************************
    freeing of arrays:

    double TmpD[List_YOUSO[24]];

    double D0_CntCoes[Matomnum+1]
                            [List_YOUSO[7]]
                            [List_YOUSO[24]];
  ********************************************************/

  free(TmpD);

  for (i=0; i<=Matomnum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(D0_CntCoes[i][j]);
    }
    free(D0_CntCoes[i]);
  }
  free(D0_CntCoes);

  for (i=0; i<=SpeciesNum; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(My_DCntCoes_Spe[i][j]);
    }
    free(My_DCntCoes_Spe[i]);
  }
  free(My_DCntCoes_Spe);
}











double NormD_RCnt(double ***D_CntCoes)
{
  int Cwan,Mc_AN,Gc_AN,p,q,al;
  int L0,Mul0,M0;
  double My_NormD,NormD;

  /****************************************************
                 Calculate NormD_RCnt
  ****************************************************/

  My_NormD = 0.0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_CBasis[Cwan][L0]; Mul0++){
	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	    My_NormD += D_CntCoes[Mc_AN][al][p]*D_CntCoes[Mc_AN][al][p];  
	  }
	}
      }
    }      
  }

  /* MPI My_NormD */
  MPI_Allreduce(&My_NormD, &NormD, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  return NormD;
}


double NormD_ACnt(double ***DCntCoes_Spe)
{
  int Cwan,Mc_AN,Gc_AN,p,q,al;
  int L0,Mul0,M0;
  double My_NormD,NormD;

  /****************************************************
                 Calculate NormD_ACnt
  ****************************************************/

  My_NormD = 0.0;

  for (Cwan=0; Cwan<SpeciesNum; Cwan++){
    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_CBasis[Cwan][L0]; Mul0++){
	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	    My_NormD += DCntCoes_Spe[Cwan][al][p]*DCntCoes_Spe[Cwan][al][p];  
	  }
	}
      }
    }      
  }

  NormD = My_NormD;

  return NormD;
}



static void Inverse(int n, double **a, double **ia)
{
  /****************************************************
                  LU decomposition
                      0 to n
   NOTE:
   This routine does not consider the reduction of rank
  ****************************************************/

  int i,j,k;
  double w;
  double *x,*y;
  double **da;

  /***************************************************
    allocation of arrays: 

    x[(orbitalOpt_History+2)]
    y[(orbitalOpt_History+2)]
    da[(orbitalOpt_History+2)][(orbitalOpt_History+2)]
  ***************************************************/

  x = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));
  y = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));
  for (i=0; i<(orbitalOpt_History+2); i++){
    x[i] = 0.0;
    y[i] = 0.0;
  }

  da = (double**)malloc(sizeof(double*)*(orbitalOpt_History+2));
  for (i=0; i<(orbitalOpt_History+2); i++){
    da[i] = (double*)malloc(sizeof(double)*(orbitalOpt_History+2));
    for (j=0; j<(orbitalOpt_History+2); j++){
      da[i][j] = 0.0;
    }
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<(orbitalOpt_History+2); i++){
      for (j=0; j<(orbitalOpt_History+2); j++){
	a[i][j] = 0.0;
      }
    }
  }
  else{
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	da[i][j] = a[i][j];
      }
    }

    /****************************************************
                     LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){
      w = 1.0/a[k][k];
      for (i=k+1; i<=n; i++){
	a[i][k] = w*a[i][k];
	for (j=k+1; j<=n; j++){
	  a[i][j] = a[i][j] - a[i][k]*a[k][j];
	}
      }
    }

    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k)
	  y[i] = 1.0;
	else
	  y[i] = 0.0;
	for (j=0; j<=i-1; j++){
	  y[i] = y[i] - a[i][j]*y[j];
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
	  x[i] = x[i] - a[i][j]*x[j];
	}
	x[i] = x[i]/a[i][i];
	ia[i][k] = x[i];
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	a[i][j] = da[i][j];
      }
    }
  }

  /***************************************************
    freeing of arrays: 

    x[(orbitalOpt_History+2)]
    y[(orbitalOpt_History+2)]
    da[(orbitalOpt_History+2)][(orbitalOpt_History+2)]
  ***************************************************/

  free(x);
  free(y);

  for (i=0; i<(orbitalOpt_History+2); i++){
    free(da[i]);
  }
  free(da);
}



void Inverse_Mat(int n, double **A0, double **IA0)
{
  int i,j,N;
  double *A,*work;
  INTEGER lda,info,lwork,*ipiv;

  A = (double*)malloc(sizeof(double)*(n+2)*(n+2));
  work = (double*)malloc(sizeof(double)*(n+2));
  ipiv = (INTEGER*)malloc(sizeof(INTEGER)*(n+2));

  N = n + 1;

  lda = N;
  lwork = N;

  /****************************************************
      A0 -> A
  ****************************************************/

  for (i=0;i<=n;i++) {
    for (j=0;j<=n;j++) {
       A[i*(n+1)+j]= A0[i][j];
    }
  }

  /****************************************************
                call zgetrf_() in clapack
  ****************************************************/

  F77_NAME(dgetrf,DGETRF)(&N, &N, A, &lda, ipiv, &info);

  if (info!=0){
    printf("error in dgetrf_() which is called from IS_Lanczos info=%2d\n",info);
  }

  /****************************************************
                Call dgetri_() in clapack
  ****************************************************/

  F77_NAME(dgetri,DGETRI)(&N, A, &lda, ipiv, work, &lwork, &info);
  if (info!=0){
    printf("error in dgetri_() which is called from IS_Lanczos info=%2d\n",info);
  }

  /****************************************************
               A -> IA0
  ****************************************************/

  for (i=0; i<=n; i++) {
    for (j=0; j<=n; j++) {
      IA0[i][j] = A[i*(n+1)+j];
    }
  }

  /* free arrays */
  free(ipiv);
  free(work);
  free(A);
}

