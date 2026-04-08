/**********************************************************************
  LNO.c:

   LNO.c is a subroutine to calculate strictly localized non-orthogonal 
   natural orbitals.

  Log of LNO.c:

     06/March/2018  Released by T. Ozaki

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

#define  measure_time   0

static double LNO_Col_Projection(char *mode, int SCF_iter,
			         double ****OLP0, double *****Hks, double *****CDM);

static double LNO_Col_Schur(char *mode, int SCF_iter,
			    double ****OLP0, double *****Hks, double *****CDM);

static double LNO_Col_Diag(char *mode, int SCF_iter,
			   double ****OLP0, double *****CDM);

static double LNO_NonCol_Diag(char *mode, int SCF_iter,
		   	      double ****OLP0, double *****CDM);

static void Calc_Inverse(int n, double *A);
static logical myselect(double wr, double wi);

double LNO(char *mode,
           int SCF_iter,
           double ****OLP0,
           double *****Hks,
           double *****CDM)
{
  double time0;

  /****************************************************
                    for collinear DFT
  ****************************************************/

  if ( SpinP_switch==0 || SpinP_switch==1){

    time0 = LNO_Col_Diag(mode, SCF_iter, OLP0, CDM);


    /*
    time0 = LNO_Col_Schur(mode, SCF_iter, OLP0, Hks, CDM);
    */

    /*
    time0 = LNO_Col_Projection(mode, SCF_iter, OLP0, Hks, CDM);
    */

    /*
    if (SCF_iter<=2){ 
      time0 = LNO_Col_Schur(mode, SCF_iter, OLP0, Hks, CDM);
    }
    else{ 
      time0 = LNO_Col_Opt(mode, SCF_iter, OLP0, Hks, CDM);
    }
    */

  }

  else if ( SpinP_switch==3 ){

    time0 = LNO_NonCol_Diag(mode, SCF_iter, OLP0, CDM);
  }

  /****************************************************
                   for non-collinear DFT
  ****************************************************/

  else if (SpinP_switch==3){
    printf("LNO is not supported for non-collinear DFT calculations.\n");
    MPI_Finalize();
    exit(1);
  }

  return time0;
}


static double LNO_Col_Projection( char *mode, int SCF_iter,
                                  double ****OLP0, double *****Hks, double *****CDM)
{
  int i,j,k,l,n,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int Cwan,Hwan,num,wan1,wan2,tno0,tno1,tno2,spin;
  int Nloop,po,p,q,tnoA,tnoB,GA,GB,na,nb;
  int hA,hB,kl,ih,wanA,wanB,matD,N;
  double sum;
  double *tmp_array,*tmp_array2;
  double **mat1,**matS,**mat0,**mat2,*ko;
  double TStime,TEtime;
  int ID,IDS,IDR,myid,numprocs,tag=999;
  int *Snd_Size,*Rcv_Size;
  int size1,size2;
  MPI_Status stat;
  MPI_Request request;

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************
             allocation of arrays
  ********************************************/

  Snd_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_Size = (int*)malloc(sizeof(int)*numprocs);

  matD = 0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      
    Gc_AN = M2G[Mc_AN];

    na = 0;
    for (hA=0; hA<=FNAN[Gc_AN]; hA++){
      GA = natn[Gc_AN][hA];
      wanA = WhatSpecies[GA];
      tnoA = Spe_Total_CNO[wanA];
      na += tnoA;
    }

    if (matD<na) matD = na;
  }

  mat1 = (double**)malloc(sizeof(double*)*matD);
  for (i=0; i<matD; i++){
    mat1[i] = (double*)malloc(sizeof(double)*matD);
  }

  matS = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    matS[i] = (double*)malloc(sizeof(double)*matD);
  }

  mat2 = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    mat2[i] = (double*)malloc(sizeof(double)*matD);
  }

  mat0 = (double**)malloc(sizeof(double*)*(List_YOUSO[7]+1));
  for (i=0; i<(List_YOUSO[7]+1); i++){
    mat0[i] = (double*)malloc(sizeof(double)*(List_YOUSO[7]+1));
  }

  ko = (double*)malloc(sizeof(double)*(List_YOUSO[7]+1));

  /********************************************
    MPI communication of DM0
  ********************************************/

  /* copy CDM to DM0 */

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_NO[Hwan];

        for (i=0; i<tno1; i++){
          for (j=0; j<tno2; j++){
            DM0[spin][Mc_AN][h_AN][i][j] = CDM[spin][Mc_AN][h_AN][i][j];
  	  }
	}
      }
    }
  }

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
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              size1 += tno1*tno2;
	    }
          }
	}

        Snd_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR]!=0){

        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_Size[IDR] = size2;
      }
      else{
        Rcv_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
    }
    else{
      Snd_Size[IDS] = 0;
      Rcv_Size[IDR] = 0;
    }
  }

  /***********************************
        data transfer of DM0
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

        size1 = Snd_Size[IDS];

        /* allocation of array */

        tmp_array = (double*)malloc(sizeof(double)*size1);

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  tmp_array[num] = DM0[spin][Mc_AN][h_AN][i][j];
                  num++;
                } 
              } 
	    }
          }
	}

        MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);

      }

      /*****************************
         receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR]!=0){

        size2 = Rcv_Size[IDR];
        
        /* allocation of array */
        tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = F_TopMAN[IDR] - 1;  
          for (n=0; n<F_Rcv_Num[IDR]; n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  DM0[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
		}
	      }
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

  /********************************************
     diagonalization of a projection matrix 
  ********************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wan1];

      /* construct a matrix of DM for the FNAN region */

      na = 0;
      for (hA=0; hA<=FNAN[Gc_AN]; hA++){

	GA = natn[Gc_AN][hA];
	wanA = WhatSpecies[GA];
        tnoA = Spe_Total_CNO[wanA];
	ih = F_G2M[GA];

        for (i=0; i<tnoA; i++){

	  nb = 0;
	  for (hB=0; hB<=FNAN[Gc_AN]; hB++){

  	    kl = RMI1[Mc_AN][hA][hB];

	    GB = natn[Gc_AN][hB];
	    wanB = WhatSpecies[GB];
	    tnoB = Spe_Total_CNO[wanB];

	    if (0<=kl){

	      for (j=0; j<tnoB; j++){
		mat1[na+i][nb+j] = DM0[spin][ih][kl][i][j]; 
	      }
	    }

            else{
	      for (j=0; j<tnoB; j++){
		mat1[na+i][nb+j] = 0.0;
	      }
	    } 

            nb += tnoB;

	  } /* hB */
	} /* i */

        na += tnoA;

      } /* hA */

      /* construct matS of OLP0 for the FNAN region */

      na = 0;
      for (hA=0; hA<=FNAN[Gc_AN]; hA++){

	GA = natn[Gc_AN][hA];
	wanA = WhatSpecies[GA];
        tnoA = Spe_Total_CNO[wanA];

        for (i=0; i<tno1; i++){
          for (j=0; j<tnoA; j++){
            matS[i][na+j] = OLP0[Mc_AN][hA][i][j];
  	  } 
	} 

        na += tnoA;
      }

      /* dimension of matrices */

      N = na;

      /* symmetrizaton of mat1 */

      for (i=0; i<N; i++){
        for (j=(i+1); j<N; j++){
          sum = 0.5*(mat1[i][j] + mat1[j][i]); 
          mat1[i][j] = sum;
          mat1[j][i] = sum;
	}
      }

      /*
      printf("QQQ0 Mc_AN=%2d\n",Mc_AN);
      for (i=0; i<N; i++){
        for (j=0; j<N; j++){
          printf("%7.4f",mat1[i][j]);
	}
        printf("\n");
      }
      */

      /********************************************
          matrix product of matS*mat1*matS
      ********************************************/

      /* mat1*matS -> mat2 */

      for (j=0; j<tno1; j++){
        for (i=0; i<N; i++){

          sum = 0.0;
          for (k=0; k<N; k++){
            sum += mat1[i][k]*matS[j][k];
	  }
          
          mat2[j][i] = sum;
        }
      }

      /* matS*mat2 -> mat0 */

      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){

          sum = 0.0;
          for (k=0; k<N; k++){
            sum += matS[i][k]*mat2[j][k];
	  }
          
          mat0[i+1][j+1] = sum;
        }
      }

      /*
      printf("ABC0 Mc_AN=%2d\n",Mc_AN);
      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){
          printf("%7.4f ",mat0[i+1][j+1]);
	}
        printf("\n");
      }
      */


      /*
      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){
          printf("%7.4f ",mat0[i+1][j+1]);
	}
        printf("\n");
      }

      MPI_Finalize();
      exit(0);
      */

      /********************************************
                diagonalization of mat0
      ********************************************/

      Eigen_lapack(mat0,ko,tno1,tno1);

      /* copy mat0 to LNO_coes, where vectors in LNO_coes are stored in column */
      
      for (j=0; j<tno1; j++){
	for (i=0; i<tno1; i++){
	  LNO_coes[spin][Mc_AN][tno1*j+i] = mat0[i+1][(tno1-j-1)+1];
	}
      }

      /* store the eigenvalues */

      for (i=0; i<tno1; i++){
        LNO_pops[spin][Mc_AN][i] = ko[(tno1-i-1)+1];
      }

      /*
      printf("ABC1 Mc_AN=%2d ko\n",Mc_AN);
      for (i=0; i<tno1; i++){
        printf("i=%2d ko=%15.12f\n",i+1,ko[(tno1-i-1)+1]);
      }

      printf("ABC1 Mc_AN=%2d\n",Mc_AN);
      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){
          printf("%7.4f ",mat0[i+1][(tno1-j-1)+1]);
	}
        printf("\n");
      }
      */

      /********************************************
       determine the number of LNOs which will be 
       included in proceeding calculations.
      ********************************************/

      if ( (SCF_iter==1 && mode=="o-n3") || mode=="full" ){
        LNO_Num[Gc_AN] = tno1;
      }

      else{

	if (spin==0) LNO_Num[Gc_AN] = 0;

	for (i=0; i<tno1; i++){

	  if (LNO_pops[spin][Mc_AN][i]<LNO_Occ_Cutoff && LNO_Num[Gc_AN]<=i){

	    LNO_Num[Gc_AN] = i;
	    break;
	  }
	}

	if (i==tno1){ 
	  LNO_Num[Gc_AN] = tno1;
	}
      }

    } /* Mc_AN */
  } /* spin */

  /********************************************
    MPI communication of LNO_Num and LNO_coes
  ********************************************/

  /* LNO_Num */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&LNO_Num[Gc_AN], 1, MPI_INT, ID, mpi_comm_level1);
  }

  /* LNO_coes */

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      tag = 999;

      /* find data size to send block data */

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	size1 = 0;
	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  
  	  size1 += tno1*tno2;
	}
        size1 = (SpinP_switch+1)*size1;

	Snd_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_Size[IDR] = size2;
      }
      else{
	Rcv_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
    }

    else{
      Snd_Size[IDS] = 0;
      Rcv_Size[IDR] = 0;
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

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	size1 = Snd_Size[IDS];

	/* allocation of array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){

	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  

          for (spin=0; spin<=SpinP_switch; spin++){
	    for (j=0; j<tno2; j++){
	      for (i=0; i<tno1; i++){
		tmp_array[num] = LNO_coes[spin][Mc_AN][tno1*j+i];
		num++;
	      }
	    }
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
          
	size2 = Rcv_Size[IDR];
        
	/* allocation of array */
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
         
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	num = 0;
	Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */

	for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){

	  Mc_AN++;
	  Gc_AN = Rcv_GAN[IDR][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  

          for (spin=0; spin<=SpinP_switch; spin++){
	    for (j=0; j<tno2; j++){
	      for (i=0; i<tno1; i++){
		LNO_coes[spin][Mc_AN][tno1*j+i] = tmp_array2[num];
		num++;
	      }
	    }
	  }
	}        

	/* freeing of array */
	free(tmp_array2);
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
	MPI_Wait(&request,&stat);
	free(tmp_array); /* freeing of array */
      }
    }
  }

  /********************************************
             freeing of arrays
  ********************************************/

  free(Snd_Size);
  free(Rcv_Size);

  for (i=0; i<matD; i++){
    free(mat1[i]);
  }
  free(mat1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(matS[i]);
  }
  free(matS);

  for (i=0; i<List_YOUSO[7]; i++){
    free(mat2[i]);
  }
  free(mat2);

  for (i=0; i<(List_YOUSO[7]+1); i++){
    free(mat0[i]);
  }
  free(mat0);

  free(ko);

  /* for time */
  dtime(&TEtime);

  return (TEtime-TStime);
}



static double LNO_Col_Schur(char *mode, int SCF_iter,
			    double ****OLP0, double *****Hks, double *****CDM)
{
  int i,j,k,l,n,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int Cwan,num,wan1,wan2,tno0,tno1,tno2,spin;
  int Nloop,po,p,q,i1,j1,k1,no;
  char *JOBVS,*SORT,*SENSE;
  int N,A,LDA,LDVS,SDIM,LWORK,*IWORK,LIWORK,INFO;
  double ***DMS,*WR,*WI,*VS,*WORK,RCONDE,RCONDV;
  double *B,*C,*IC,sum,sum0,F;
  logical *BWORK;
  double *tmp_array;
  double *tmp_array2;
  double TStime,TEtime;
  int ID,IDS,IDR,myid,numprocs,tag=999;
  int *Snd_Size,*Rcv_Size;
  int size1,size2;
  MPI_Status stat;
  MPI_Request request;

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************
             allocation of arrays
  ********************************************/

  Snd_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_Size = (int*)malloc(sizeof(int)*numprocs);

  DMS = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    DMS[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){
      DMS[spin][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]*List_YOUSO[7]; i++) DMS[spin][Mc_AN][i] = 0.0;
    }
  }

  WR = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  WI = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  VS = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*10);
  WORK = (double*)malloc(sizeof(double)*List_YOUSO[7]*10);
  IWORK = (int*)malloc(sizeof(int)*List_YOUSO[7]*10);
  BWORK = (logical*)malloc(sizeof(logical)*List_YOUSO[7]);
  B = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  C = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  IC = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  
  /********************************************
        calculation of DMS defined by DM*S
  ********************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
 
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){

	    sum = 0.0;
            for (k=0; k<tno2; k++){
              sum += CDM[spin][Mc_AN][h_AN][i][k]*OLP0[Mc_AN][h_AN][j][k];
            }

	    DMS[spin][Mc_AN][tno1*j+i] += sum; 
	  }
	}
      }


      /*
      printf("QQQ1 DMS spin=%2d Mc_AN=%2d Gc_AN=%2d\n",spin,Mc_AN,Gc_AN);
      for (i=0; i<tno1; i++){
	for (j=0; j<tno1; j++){
	  printf("%10.5f ",DMS[spin][Mc_AN][tno1*j+i]);
	}
	printf("\n");
      }
      */


    } /* Mc_AN */
  } /* spin */

  /************************************************************
    Schur decomposition of DMS and the proceeding on-site 
    diagonalization within a subspace orthogonal to 
    the occupied space calculated by the Schur decomposition
  ************************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wan1];

      /* Schur decomposition */
      /* call the dgeesx routine in lapack */

      JOBVS = "V";
      SORT = "N";
      SENSE = "N";
      N = tno1;
      LDA = tno1;
      SDIM = tno1;
      LDVS = tno1;    
      LWORK = tno1*10;
      LIWORK = 1;

      for (i=0; i<tno1*tno1; i++) B[i] = DMS[spin][Mc_AN][i];

      F77_NAME(dgeesx,DGEESX)( JOBVS, SORT, myselect, SENSE, &N, B, &LDA, &SDIM, WR, WI, VS, &LDVS, 
                               &RCONDE, &RCONDV, WORK, &LWORK, IWORK, &LIWORK, BWORK, &INFO );

      if (INFO!=0){
        printf("warning: INFO=%2d in calling dgeesx in a function 'LNO'\n",INFO);
      }

      /* ordering the eigenvalues and the orthogonal matrix */

      for (i=0; i<tno1; i++) IWORK[i] = i;

      qsort_double_int2(tno1,WR,IWORK);

      /* copy VS to LNO_coes, where vectors in LNO_coes are stored in column */

      for (j=0; j<tno1; j++){
        k = IWORK[j];
	for (i=0; i<tno1; i++){
	  LNO_coes[spin][Mc_AN][tno1*j+i] = VS[tno1*k+i];
	}
      }

      /* store the eigenvalues */

      for (i=0; i<tno1; i++){
        LNO_pops[spin][Mc_AN][i] = WR[i];
      }

      /********************************************
       determine the number of LNOs which will be 
       included in proceeding calculations.
      ********************************************/

      if ( (SCF_iter==1 && mode=="o-n3") || mode=="full" ){
        LNO_Num[Gc_AN] = tno1;
      }

      else{

        if (LNOs_Num_predefined_flag==0){

	  if (spin==0) LNO_Num[Gc_AN] = 0;

	  for (i=0; i<tno1; i++){

	    if (LNO_pops[spin][Mc_AN][i]<LNO_Occ_Cutoff && LNO_Num[Gc_AN]<=i){

	      LNO_Num[Gc_AN] = i;
	      break;
	    }
	  }

	  if (i==tno1 && 1){ 
	    LNO_Num[Gc_AN] = tno1;
	  }
	}

        /* LNOs_Num_predefined_flag==1 */
        else{
          wan1 = WhatSpecies[Gc_AN];
          LNO_Num[Gc_AN] = LNOs_Num_predefined[wan1];
	}
      }

      /* show the result */

      if (0 && myid==0){

        printf("ABC1 Gc_AN=%2d LNO_Num=%2d\n",Gc_AN,LNO_Num[Gc_AN]);

	for (i=0; i<tno1; i++){
	  printf("ABC myid=%2d spin=%2d Mc_AN=%2d i=%2d IWORK=%2d WR=%15.11f WI=%15.11f\n",
                      myid,spin,Mc_AN,i,IWORK[i],WR[i],WI[i]);fflush(stdout);
	}
	
        printf("QQQ myid=%2d spin=%2d Mc_AN=%2d LNO_Num=%2d\n",myid,spin,Mc_AN,LNO_Num[Gc_AN]);fflush(stdout);

	printf("WWW1 myid=%2d LNO_coes spin=%2d Mc_AN=%2d\n",myid,spin,Mc_AN);fflush(stdout);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
	    printf("%10.5f ",LNO_coes[spin][Mc_AN][tno1*j+i]);fflush(stdout);
	  }
	  printf("\n");fflush(stdout);
	}

        printf("Check orthogonalization\n"); 
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
 
            sum = 0.0;
	    for (k=0; k<tno1; k++){
              sum += LNO_coes[spin][Mc_AN][tno1*i+k]*LNO_coes[spin][Mc_AN][tno1*j+k];
  	    }
            printf("%10.5f ",sum);
	  }
	  printf("\n");fflush(stdout);
	}

      }

    } /* Mc_AN */
  } /* spin */

  /********************************************
              MPI communication
  ********************************************/

  /* LNO_Num */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&LNO_Num[Gc_AN], 1, MPI_INT, ID, mpi_comm_level1);
  }

  /* LNO_coes */

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      tag = 999;

      /* find data size to send block data */

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	size1 = 0;
	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  
  	  size1 += tno1*tno2;
	}
        size1 = (SpinP_switch+1)*size1;

	Snd_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_Size[IDR] = size2;
      }
      else{
	Rcv_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
    }

    else{
      Snd_Size[IDS] = 0;
      Rcv_Size[IDR] = 0;
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

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	size1 = Snd_Size[IDS];

	/* allocation of array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){

	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  

          for (spin=0; spin<=SpinP_switch; spin++){
	    for (j=0; j<tno2; j++){
	      for (i=0; i<tno1; i++){
		tmp_array[num] = LNO_coes[spin][Mc_AN][tno1*j+i];
		num++;
	      }
	    }
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
          
	size2 = Rcv_Size[IDR];
        
	/* allocation of array */
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
         
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	num = 0;
	Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */

	for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){

	  Mc_AN++;
	  Gc_AN = Rcv_GAN[IDR][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  

          for (spin=0; spin<=SpinP_switch; spin++){
	    for (j=0; j<tno2; j++){
	      for (i=0; i<tno1; i++){
		LNO_coes[spin][Mc_AN][tno1*j+i] = tmp_array2[num];
		num++;
	      }
	    }
	  }
	}        

	/* freeing of array */
	free(tmp_array2);
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
	MPI_Wait(&request,&stat);
	free(tmp_array); /* freeing of array */
      }
    }
  }


  /*
  if (myid==0){

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	Gc_AN = S_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_CNO[Cwan];
	tno2 = LNO_Num[Gc_AN];  
       
	printf("VVV1 spin=%2d Mc_AN=%2d Gc_AN=%2d LNO_coes\n",spin,Mc_AN,Gc_AN);fflush(stdout);

        for (i=0; i<tno1; i++){
  	  for (j=0; j<tno2; j++){
            printf("%10.5f ",LNO_coes[spin][Mc_AN][tno1*j+i]); fflush(stdout);
	  }
          printf("\n"); fflush(stdout);
	}
      }
    }
  }
  */

  /********************************************
             freeing of arrays
  ********************************************/

  free(Snd_Size);
  free(Rcv_Size);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){
      free(DMS[spin][Mc_AN]);
    }
    free(DMS[spin]);
  }
  free(DMS);

  free(WR);
  free(WI);
  free(VS);
  free(WORK);
  free(IWORK);
  free(BWORK);

  free(B);
  free(C);
  free(IC);

  /* for time */
  dtime(&TEtime);

  return (TEtime-TStime);
}









static double LNO_Col_Diag(char *mode, int SCF_iter,
			   double ****OLP0, double *****CDM)
{
  int i,j,k,l,n,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int Cwan,num,wan1,wan2,tno0,tno1,tno2,spin;
  int Nloop,po,p,q;
  char *JOBVL,*JOBVR;
  int N,A,LDA,LDVL,LDVR,SDIM,LWORK,INFO,*IWORK;
  double ***DMS,*WR,*WI,*VL,*VR,*WORK,RCONDE,RCONDV;
  double *B,*C,*IC,sum,sum0,F;
  double *tmp_array;
  double *tmp_array2;
  double TStime,TEtime;
  int ID,IDS,IDR,myid,numprocs,tag=999;
  int *Snd_Size,*Rcv_Size;
  int size1,size2;
  MPI_Status stat;
  MPI_Request request;

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************
             allocation of arrays
  ********************************************/

  Snd_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_Size = (int*)malloc(sizeof(int)*numprocs);

  DMS = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    DMS[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){
      DMS[spin][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]*List_YOUSO[7]; i++) DMS[spin][Mc_AN][i] = 0.0;
    }
  }

  WR = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  WI = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  VL = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*2);
  VR = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*2);

  WORK = (double*)malloc(sizeof(double)*List_YOUSO[7]*10);
  IWORK = (int*)malloc(sizeof(int)*List_YOUSO[7]);

  B = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  C = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  IC = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  
  /********************************************
        calculation of DMS defined by DM*S
  ********************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
 
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wan1];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wan2];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){

	    sum = 0.0;
            for (k=0; k<tno2; k++){
              sum += CDM[spin][Mc_AN][h_AN][i][k]*OLP0[Mc_AN][h_AN][j][k];
            }

	    DMS[spin][Mc_AN][tno1*j+i] += sum; 
	  }
	}
      }

    } /* Mc_AN */
  } /* spin */

  /********************************************
            diagonalization of DMS
  ********************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wan1];

      /* call the dgeev routine in lapack */

      JOBVL = "V";
      JOBVR = "V";
      N = tno1;
      LDA = tno1;
      LDVL = tno1*2;
      LDVR = tno1*2;
      LWORK = tno1*10;

      for (i=0; i<tno1*tno1; i++) B[i] = DMS[spin][Mc_AN][i];

      F77_NAME(dgeev,DGEEV)( JOBVL, JOBVR, &N, B, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, 
                             WORK, &LWORK, &INFO );

      if (INFO!=0){
        printf("warning: INFO=%2d in calling dgeev in a function 'LNO'\n",INFO);
      }

      /* ordering the eigenvalues and the orthogonal matrix */

      for (i=0; i<tno1; i++) IWORK[i] = i;
      qsort_double_int2(tno1,WR,IWORK);

      /* calculations of Frobenius norm */

      if (0 && myid==0){

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
	    B[j*tno1+i] = 0.0; 
	  }
	}     

        for (k=0; k<tno1; k++){
          l = IWORK[k];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno1; j++){
              B[j*tno1+i] += VR[LDVR*l+i]*WR[k]*VL[LDVL*l+j];
	    }
	  }     
	}

        printf("DMS\n");
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
            printf("%10.6f ",DMS[spin][Mc_AN][tno1*j+i]);
	  }
          printf("\n");
	}

        printf("B\n");
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
            printf("%10.6f ",B[tno1*j+i]);
	  }
          printf("\n");
	}
      }

      /* copy VR to LNO_coes, where vectors in LNO_coes are stored in column */

      for (j=0; j<tno1; j++){
        k = IWORK[j];
	for (i=0; i<tno1; i++){
	  LNO_coes[spin][Mc_AN][tno1*j+i] = VR[LDVR*k+i];
	}

        sum = 0.0;
	for (i=0; i<tno1; i++){
	  sum += LNO_coes[spin][Mc_AN][tno1*j+i]*LNO_coes[spin][Mc_AN][tno1*j+i];
	}
        sum = 1.0/sqrt(sum);

	for (i=0; i<tno1; i++){
	  LNO_coes[spin][Mc_AN][tno1*j+i] *= sum;
	}
      }

      /* store the eigenvalues */

      for (i=0; i<tno1; i++){
        LNO_pops[spin][Mc_AN][i] = WR[i];
      }

      /********************************************
       determine the number of LNOs which will be 
       included in proceeding calculations.
      ********************************************/

      if ( (SCF_iter==1 && mode=="o-n3") || mode=="full" ){
        LNO_Num[Gc_AN] = tno1;
      }

      else{

        if (LNOs_Num_predefined_flag==0){

	  if (spin==0) LNO_Num[Gc_AN] = 0;

	  for (i=0; i<tno1; i++){

	    if (LNO_pops[spin][Mc_AN][i]<LNO_Occ_Cutoff && LNO_Num[Gc_AN]<=i){

	      LNO_Num[Gc_AN] = i;
	      break;
	    }
	  }

	  if (i==tno1){ 
	    LNO_Num[Gc_AN] = tno1;
	  }
	}

        /* LNOs_Num_predefined_flag==1 */
        else{
          wan1 = WhatSpecies[Gc_AN];
          LNO_Num[Gc_AN] = LNOs_Num_predefined[wan1];
	}

      }

      if (0 && myid==0){

        printf("ABC1 Gc_AN=%2d LNO_Num=%2d\n",Gc_AN,LNO_Num[Gc_AN]);

	for (i=0; i<tno1; i++){
	  printf("ABC myid=%2d spin=%2d Mc_AN=%2d i=%2d IWORK=%2d WR=%15.11f WI=%15.11f\n",
                      myid,spin,Mc_AN,i,IWORK[i],WR[i],WI[i]);fflush(stdout);
	}

        printf("QQQ myid=%2d spin=%2d Mc_AN=%2d LNO_Num=%2d\n",myid,spin,Mc_AN,LNO_Num[Gc_AN]);fflush(stdout);

	printf("WWW1 myid=%2d LNO_coes spin=%2d Mc_AN=%2d\n",myid,spin,Mc_AN);fflush(stdout);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
	    printf("%10.5f ",LNO_coes[spin][Mc_AN][tno1*j+i]);fflush(stdout);
	  }
	  printf("\n");fflush(stdout);
	}

        printf("Check orthogonalization\n"); 
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
 
            sum = 0.0;
	    for (k=0; k<tno1; k++){
              sum += VL[LDVL*i+k]*VR[LDVR*j+k];
  	    }
            printf("%10.5f ",sum);
	  }
	  printf("\n");fflush(stdout);
	}

	/*
	printf("WWW1 myid=%2d VL spin=%2d Mc_AN=%2d\n",myid,spin,Mc_AN);fflush(stdout);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1*2; j++){
	    printf("%10.5f ",VL[tno1*j+i]);fflush(stdout);
	  }
	  printf("\n");fflush(stdout);
	}

	printf("WWW1 myid=%2d VR spin=%2d Mc_AN=%2d\n",myid,spin,Mc_AN);fflush(stdout);
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1*2; j++){
	    printf("%10.5f ",VR[tno1*j+i]);fflush(stdout);
	  }
	  printf("\n");fflush(stdout);
	}
	*/

	/*
        MPI_Finalize();
        exit(0);
	*/

        



      }

    } /* Mc_AN */
  } /* spin */

  /********************************************
              MPI communication
  ********************************************/

  /* LNO_Num */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&LNO_Num[Gc_AN], 1, MPI_INT, ID, mpi_comm_level1);
  }

  /* LNO_coes */

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      tag = 999;

      /* find data size to send block data */

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	size1 = 0;
	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  
  	  size1 += tno1*tno2;
	}
        size1 = (SpinP_switch+1)*size1;

	Snd_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_Size[IDR] = size2;
      }
      else{
	Rcv_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
    }

    else{
      Snd_Size[IDS] = 0;
      Rcv_Size[IDR] = 0;
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

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	size1 = Snd_Size[IDS];

	/* allocation of array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){

	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  

          for (spin=0; spin<=SpinP_switch; spin++){
	    for (j=0; j<tno2; j++){
	      for (i=0; i<tno1; i++){
		tmp_array[num] = LNO_coes[spin][Mc_AN][tno1*j+i];
		num++;
	      }
	    }
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
          
	size2 = Rcv_Size[IDR];
        
	/* allocation of array */
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
         
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	num = 0;
	Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */

	for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){

	  Mc_AN++;
	  Gc_AN = Rcv_GAN[IDR][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  

          for (spin=0; spin<=SpinP_switch; spin++){
	    for (j=0; j<tno2; j++){
	      for (i=0; i<tno1; i++){
		LNO_coes[spin][Mc_AN][tno1*j+i] = tmp_array2[num];
		num++;
	      }
	    }
	  }
	}        

	/* freeing of array */
	free(tmp_array2);
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
	MPI_Wait(&request,&stat);
	free(tmp_array); /* freeing of array */
      }
    }
  }

  /*
  if (myid==0){

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	Gc_AN = S_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_CNO[Cwan];
	tno2 = LNO_Num[Gc_AN];  
       
	printf("VVV1 spin=%2d Mc_AN=%2d Gc_AN=%2d LNO_coes\n",spin,Mc_AN,Gc_AN);fflush(stdout);

        for (i=0; i<tno1; i++){
  	  for (j=0; j<tno2; j++){
            printf("%10.5f ",LNO_coes[spin][Mc_AN][tno1*j+i]); fflush(stdout);
	  }
          printf("\n"); fflush(stdout);
	}
      }
    }
  }
  */

  /********************************************
             freeing of arrays
  ********************************************/

  free(Snd_Size);
  free(Rcv_Size);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){
      free(DMS[spin][Mc_AN]);
    }
    free(DMS[spin]);
  }
  free(DMS);

  free(WR);
  free(WI);
  free(VL);
  free(VR);
  free(WORK);
  free(IWORK);
  free(B);
  free(C);
  free(IC);

  /* for time */
  dtime(&TEtime);

  return (TEtime-TStime);
}


static double LNO_NonCol_Diag(char *mode, int SCF_iter,
		   	      double ****OLP0, double *****CDM)
{
  int i,j,k,l,n,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int Cwan,num,wan1,wan2,tno0,tno1,tno2;
  int Nloop,po,p,q;
  char *JOBVL,*JOBVR;
  int N,A,LDA,LDVL,LDVR,SDIM,LWORK,INFO,*IWORK;
  double **DMS,*WR,*WI,*VL,*VR,*WORK,RCONDE,RCONDV;
  double *B,*C,*IC,sum,sum0,F;
  double *tmp_array;
  double *tmp_array2;
  double TStime,TEtime;
  int ID,IDS,IDR,myid,numprocs,tag=999;
  int *Snd_Size,*Rcv_Size;
  int size1,size2;
  MPI_Status stat;
  MPI_Request request;

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /********************************************
             allocation of arrays
  ********************************************/

  Snd_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_Size = (int*)malloc(sizeof(int)*numprocs);

  DMS = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){
    DMS[Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
    for (i=0; i<List_YOUSO[7]*List_YOUSO[7]; i++) DMS[Mc_AN][i] = 0.0;
  }

  WR = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  WI = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  VL = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*2);
  VR = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]*2);

  WORK = (double*)malloc(sizeof(double)*List_YOUSO[7]*10);
  IWORK = (int*)malloc(sizeof(int)*List_YOUSO[7]);

  B = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  C = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  IC = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  
  /********************************************
        calculation of DMS defined by DM*S
  ********************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[wan1];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      wan2 = WhatSpecies[Gh_AN];
      tno2 = Spe_Total_CNO[wan2];

      for (i=0; i<tno1; i++){
	for (j=0; j<tno1; j++){

	  sum = 0.0;
	  for (k=0; k<tno2; k++){
	    sum += (CDM[0][Mc_AN][h_AN][i][k]+CDM[1][Mc_AN][h_AN][i][k])*OLP0[Mc_AN][h_AN][j][k];
	  }

	  DMS[Mc_AN][tno1*j+i] += sum; 
	}
      }

    } /* h_AN */
  } /* Mc_AN */

  /********************************************
            diagonalization of DMS
  ********************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[wan1];

    /* call the dgeev routine in lapack */

    JOBVL = "V";
    JOBVR = "V";
    N = tno1;
    LDA = tno1;
    LDVL = tno1*2;
    LDVR = tno1*2;
    LWORK = tno1*10;

    for (i=0; i<tno1*tno1; i++) B[i] = DMS[Mc_AN][i];

    F77_NAME(dgeev,DGEEV)( JOBVL, JOBVR, &N, B, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, 
			   WORK, &LWORK, &INFO );

    if (INFO!=0){
      printf("warning: INFO=%2d in calling dgeev in a function 'LNO'\n",INFO);
    }

    /* ordering the eigenvalues and the orthogonal matrix */

    for (i=0; i<tno1; i++) IWORK[i] = i;
    qsort_double_int2(tno1,WR,IWORK);

    /* calculations of Frobenius norm */

    if (0 && myid==0){

      for (i=0; i<tno1; i++){
	for (j=0; j<tno1; j++){
	  B[j*tno1+i] = 0.0; 
	}
      }     

      for (k=0; k<tno1; k++){
	l = IWORK[k];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno1; j++){
	    B[j*tno1+i] += VR[LDVR*l+i]*WR[k]*VL[LDVL*l+j];
	  }
	}     
      }

      printf("DMS\n");
      for (i=0; i<tno1; i++){
	for (j=0; j<tno1; j++){
	  printf("%10.6f ",DMS[Mc_AN][tno1*j+i]);
	}
	printf("\n");
      }

      printf("B\n");
      for (i=0; i<tno1; i++){
	for (j=0; j<tno1; j++){
	  printf("%10.6f ",B[tno1*j+i]);
	}
	printf("\n");
      }
    }

    /* copy VR to LNO_coes, where vectors in LNO_coes are stored in column */

    for (j=0; j<tno1; j++){
      k = IWORK[j];
      for (i=0; i<tno1; i++){
	LNO_coes[0][Mc_AN][tno1*j+i] = VR[LDVR*k+i];
      }

      sum = 0.0;
      for (i=0; i<tno1; i++){
	sum += LNO_coes[0][Mc_AN][tno1*j+i]*LNO_coes[0][Mc_AN][tno1*j+i];
      }
      sum = 1.0/sqrt(sum);

      for (i=0; i<tno1; i++){
	LNO_coes[0][Mc_AN][tno1*j+i] *= sum;
      }
    }

    /* store the eigenvalues */

    for (i=0; i<tno1; i++){
      LNO_pops[0][Mc_AN][i] = WR[i];
    }

    /********************************************
       determine the number of LNOs which will be 
       included in proceeding calculations.
    ********************************************/

    if ( (SCF_iter==1 && mode=="o-n3") || mode=="full" ){
      LNO_Num[Gc_AN] = tno1;
    }

    else{

      if (LNOs_Num_predefined_flag==0){

        LNO_Num[Gc_AN] = 0;
	for (i=0; i<tno1; i++){

	  if (LNO_pops[0][Mc_AN][i]<(2.0*LNO_Occ_Cutoff) && LNO_Num[Gc_AN]<=i){

	    LNO_Num[Gc_AN] = i;
	    break;
	  }
	}

	if (i==tno1){ 
	  LNO_Num[Gc_AN] = tno1;
	}
      }

      /* LNOs_Num_predefined_flag==1 */
      else{
	wan1 = WhatSpecies[Gc_AN];
	LNO_Num[Gc_AN] = LNOs_Num_predefined[wan1];
      }

    }

    if (0 && myid==0){

      printf("ABC1 Gc_AN=%2d LNO_Num=%2d\n",Gc_AN,LNO_Num[Gc_AN]);

      for (i=0; i<tno1; i++){
	printf("ABC myid=%2d Mc_AN=%2d i=%2d IWORK=%2d WR=%15.11f WI=%15.11f\n",
	       myid,Mc_AN,i,IWORK[i],WR[i],WI[i]);fflush(stdout);
      }

      printf("QQQ myid=%2d Mc_AN=%2d LNO_Num=%2d\n",myid,Mc_AN,LNO_Num[Gc_AN]);fflush(stdout);

      printf("WWW1 myid=%2d LNO_coes Mc_AN=%2d\n",myid,Mc_AN);fflush(stdout);
      for (i=0; i<tno1; i++){
	for (j=0; j<tno1; j++){
	  printf("%10.5f ",LNO_coes[0][Mc_AN][tno1*j+i]);fflush(stdout);
	}
	printf("\n");fflush(stdout);
      }

      printf("Check orthogonalization\n"); 
      for (i=0; i<tno1; i++){
	for (j=0; j<tno1; j++){
 
	  sum = 0.0;
	  for (k=0; k<tno1; k++){
	    sum += VL[LDVL*i+k]*VR[LDVR*j+k];
	  }
	  printf("%10.5f ",sum);
	}
	printf("\n");fflush(stdout);
      }

      /*
	printf("WWW1 myid=%2d VL Mc_AN=%2d\n",myid,Mc_AN);fflush(stdout);
	for (i=0; i<tno1; i++){
	for (j=0; j<tno1*2; j++){
	printf("%10.5f ",VL[tno1*j+i]);fflush(stdout);
	}
	printf("\n");fflush(stdout);
	}

	printf("WWW1 myid=%2d VR Mc_AN=%2d\n",myid,Mc_AN);fflush(stdout);
	for (i=0; i<tno1; i++){
	for (j=0; j<tno1*2; j++){
	printf("%10.5f ",VR[tno1*j+i]);fflush(stdout);
	}
	printf("\n");fflush(stdout);
	}
      */

      /*
        MPI_Finalize();
        exit(0);
      */

    }

  } /* Mc_AN */

  /********************************************
              MPI communication
  ********************************************/

  /* LNO_Num */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&LNO_Num[Gc_AN], 1, MPI_INT, ID, mpi_comm_level1);
  }

  /* LNO_coes */

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      tag = 999;

      /* find data size to send block data */

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	size1 = 0;
	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  
  	  size1 += tno1*tno2;
	}

	Snd_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_Size[IDR] = size2;
      }
      else{
	Rcv_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
    }

    else{
      Snd_Size[IDS] = 0;
      Rcv_Size[IDR] = 0;
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

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	size1 = Snd_Size[IDS];

	/* allocation of array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){

	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  

	  for (j=0; j<tno2; j++){
	    for (i=0; i<tno1; i++){
	      tmp_array[num] = LNO_coes[0][Mc_AN][tno1*j+i];
	      num++;
	    }
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
          
	size2 = Rcv_Size[IDR];
        
	/* allocation of array */
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
         
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	num = 0;
	Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */

	for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){

	  Mc_AN++;
	  Gc_AN = Rcv_GAN[IDR][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];
          tno2 = LNO_Num[Gc_AN];  

	  for (j=0; j<tno2; j++){
	    for (i=0; i<tno1; i++){
	      LNO_coes[0][Mc_AN][tno1*j+i] = tmp_array2[num];
	      num++;
	    }
	  }
	}        

	/* freeing of array */
	free(tmp_array2);
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
	MPI_Wait(&request,&stat);
	free(tmp_array); /* freeing of array */
      }
    }
  }

  /*
  if (myid==0){

    for (Mc_AN=1; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

      Gc_AN = S_M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN]; 
      tno1 = Spe_Total_CNO[Cwan];
      tno2 = LNO_Num[Gc_AN];  
       
      printf("VVV1 Mc_AN=%2d Gc_AN=%2d LNO_coes\n",Mc_AN,Gc_AN);fflush(stdout);

      for (i=0; i<tno1; i++){
	for (j=0; j<tno2; j++){
	  printf("%10.5f ",LNO_coes[0][Mc_AN][tno1*j+i]); fflush(stdout);
	}
	printf("\n"); fflush(stdout);
      }
    }
  }
  */

  /********************************************
             freeing of arrays
  ********************************************/

  free(Snd_Size);
  free(Rcv_Size);

  for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){
    free(DMS[Mc_AN]);
  }
  free(DMS);

  free(WR);
  free(WI);
  free(VL);
  free(VR);
  free(WORK);
  free(IWORK);
  free(B);
  free(C);
  free(IC);

  /* for time */
  dtime(&TEtime);

  return (TEtime-TStime);
}




void Calc_Inverse(int n, double *A)
{
  static char *thisprogram="Lapack_LU_inverse";
  int *ipiv;
  double *work;
  int lwork;
  int info;

  /* L*U factorization */

  ipiv = (int*) malloc(sizeof(int)*n);

  F77_NAME(dgetrf,DGETRF)(&n,&n,A,&n,ipiv,&info);

  if ( info !=0 ) {
    printf("zgetrf failed, info=%i, %s\n",info,thisprogram);
  }

  /* inverse L*U factorization */

  lwork = 4*n;
  work = (double*)malloc(sizeof(double)*lwork);

  F77_NAME(dgetri,DGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

  if ( info !=0 ) {
    printf("zgetrf failed, info=%i, %s\n",info,thisprogram);
  }

  free(work); free(ipiv);
}



logical myselect(double wr, double wi)
{
  /* This is a dummy function. Nothing is defined. */
  return 1;
}
