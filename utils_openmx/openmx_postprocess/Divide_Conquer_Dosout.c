/**********************************************************************
  Divide_Conquer_DFT_Dosout.c:

     Divide_Conquer_Dosout.c is a subroutine to set up density of states
     based on a divide and conquer method.

  Log of Divide_Conquer_DFT_Dosout.c:

     10/Oct/2003  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"


static double DC_Dosout_Col(double *****Hks, double ****OLP0);
static double DC_Dosout_NonCol(double *****Hks,
                               double *****ImNL,
                               double ****OLP0);



double Divide_Conquer_Dosout(double *****Hks,
                             double *****ImNL,
                             double ****OLP0)
{
  double time0;

  /****************************************************
         collinear without spin-orbit coupling
  ****************************************************/

  if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==0 ){
    time0 = DC_Dosout_Col(Hks,OLP0);
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
    time0 = DC_Dosout_NonCol(Hks,ImNL,OLP0);
  }

  return time0;
}




static double DC_Dosout_Col(double *****Hks, double ****OLP0)
{
  static int firsttime=1;
  int Mc_AN,Gc_AN,i,Gi,wan,wanA,wanB,Anum;
  int size1,size2,num,NUM,NUM1,n2,Cwan,Hwan;
  int ih,ig,ian,j,kl,jg,jan,Bnum,m,n,spin;
  int l,i1,j1,P_min,m_size;
  int po,loopN,tno1,tno2,h_AN,Gh_AN,MaxL;
  double sum,FermiF,time0;
  double My_Num_State,Num_State,x,Dnum;
  double TStime,TEtime;
  double My_Eele0[2],My_Eele1[2];
  double max_x=50.0;
  double ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double **S_DC,***H_DC,*ko,*M1;
  double **B,**C,**D;
  double ***EVal;
  double ******Residues;
  int *MP,*Msize,*Msize1;
  double *tmp_array;
  double *tmp_array2;
  int *Snd_H_Size,*Rcv_H_Size;
  int *Snd_S_Size,*Rcv_S_Size;
  int numprocs,myid,ID,IDS,IDR,tag=999;
  double Stime_atom, Etime_atom;
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  char buf1[fp_bsize];          /* setvbuf */
  char buf2[fp_bsize];          /* setvbuf */

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  /****************************************************
    allocation of arrays:

    int MP[List_YOUSO[2]];
    int Msize[Matomnum+1];
    double EVal[SpinP_switch+1][Matomnum+1][n2];
  ****************************************************/

  Snd_H_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_H_Size = (int*)malloc(sizeof(int)*numprocs);
  Snd_S_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_S_Size = (int*)malloc(sizeof(int)*numprocs);

  m_size = 0;
  MP = (int*)malloc(sizeof(int)*List_YOUSO[2]);
  Msize = (int*)malloc(sizeof(int)*(Matomnum+1));
  Msize1 = (int*)malloc(sizeof(int)*(Matomnum+1));

  EVal = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    EVal[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      if (Mc_AN==0){
        Gc_AN = 0;
        FNAN[0] = 1;
        SNAN[0] = 0;
        n2 = 1;
        Msize[Mc_AN] = 1;
      }
      else{
        Anum = 1;
        for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
          Gi = natn[Gc_AN][i];
          wanA = WhatSpecies[Gi];
          Anum = Anum + Spe_Total_CNO[wanA];
        }
        NUM = Anum - 1;
        Msize[Mc_AN] = NUM;
        n2 = NUM + 3;
      }

      m_size += n2;

      EVal[spin][Mc_AN] = (double*)malloc(sizeof(double)*n2);
    }
  }

  if (firsttime)
  PrintMemory("Divide_Conquer_Dosout: EVal",sizeof(double)*m_size,NULL);

  if (2<=level_stdout){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        printf("<DC> myid=%i Mc_AN=%2d Gc_AN=%2d Msize=%3d\n",
        myid,Mc_AN,M2G[Mc_AN],Msize[Mc_AN]);
    }
  }

  /****************************************************
    allocation of arrays:

    double Residues[SpinP_switch+1]
                   [Matomnum+1]
                   [FNAN[Gc_AN]+1]
                   [Spe_Total_CNO[Gc_AN]] 
                   [Spe_Total_CNO[Gh_AN]] 
                   [NUM2]
     To reduce the memory size, the size of NUM2 is
     needed to be found in the loop.  
  ****************************************************/

  m_size = 0;
  Residues = (double******)malloc(sizeof(double*****)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Residues[spin] = (double*****)malloc(sizeof(double****)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      if (Mc_AN==0){
        Gc_AN = 0;
        FNAN[0] = 1;
        tno1 = 1;
        n2 = 1;
      }
      else{
        wanA = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_CNO[wanA];
        n2 = Msize[Mc_AN] + 2;
      }

      Residues[spin][Mc_AN] =
           (double****)malloc(sizeof(double***)*(FNAN[Gc_AN]+1));

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno2 = 1;
        }
        else {
          Gh_AN = natn[Gc_AN][h_AN];
          wanB = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_CNO[wanB];
        }

        Residues[spin][Mc_AN][h_AN] = (double***)malloc(sizeof(double**)*tno1);
        for (i=0; i<tno1; i++){
          Residues[spin][Mc_AN][h_AN][i] = (double**)malloc(sizeof(double*)*tno2);
          for (j=0; j<tno2; j++){
            Residues[spin][Mc_AN][h_AN][i][j] = (double*)malloc(sizeof(double)*n2);
	  }
        }

        m_size += tno1*tno2*n2;
      }
    }
  }


  if (firsttime)
  PrintMemory("Divide_Conquer_Dosout: Residues",sizeof(double)*m_size,NULL);

  /****************************************************
   MPI

   Hks
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
      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

        size1 = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
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
                  size1++; 
                } 
              } 
	    }
          }
	}
 
        Snd_H_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_H_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_H_Size[IDR] = size2;
      }
      else{
        Rcv_H_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);

    }
    else{
      Snd_H_Size[IDS] = 0;
      Rcv_H_Size[IDR] = 0;
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

        size1 = Snd_H_Size[IDS];

        /* allocation of array */

        tmp_array = (double*)malloc(sizeof(double)*size1);

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
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
                  tmp_array[num] = Hks[spin][Mc_AN][h_AN][i][j];
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

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

        size2 = Rcv_H_Size[IDR];
        
        /* allocation of array */
        tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = S_TopMAN[IDR] - 1;  /* S_TopMAN should be used. */
          for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
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
                  Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
		}
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

  /****************************************************
   MPI

   OLP0
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
      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
	size1 = 0;
	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
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
		size1++; 
	      } 
	    } 
	  }
	}

	Snd_S_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_S_Size[IDS] = 0;
      }

      /* receiving of size of data */
 
      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_S_Size[IDR] = size2;
      }
      else{
	Rcv_S_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
    }
    else{
      Snd_S_Size[IDS] = 0;
      Rcv_S_Size[IDR] = 0;
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

	size1 = Snd_S_Size[IDS];

	/* allocation of array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
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
		tmp_array[num] = OLP0[Mc_AN][h_AN][i][j];
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
          
	size2 = Rcv_S_Size[IDR];
        
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

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
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

  /****************************************************
      Setting of Hamiltonian and overlap matrices

         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];

    /***********************************************
      find the size of matrix for the atom Mc_AN
                and set the MP vector

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
    ***********************************************/
    
    Anum = 1;
    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      MP[i] = Anum;
      Gi = natn[Gc_AN][i];
      wanA = WhatSpecies[Gi];
      Anum = Anum + Spe_Total_CNO[wanA];
    }
    NUM = Anum - 1;

    n2 = NUM + 3;

    /***********************************************
     allocation of arrays:
     
     double S_DC[n2][n2];     
     double H_DC[n2][n2];     
     double ko[n2];
    ***********************************************/

    S_DC = (double**)malloc(sizeof(double*)*n2);
    for (i=0; i<n2; i++){
      S_DC[i] = (double*)malloc(sizeof(double)*n2);
    }

    H_DC = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      H_DC[spin] = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
        H_DC[spin][i] = (double*)malloc(sizeof(double)*n2);
      }
    }

    ko = (double*)malloc(sizeof(double)*n2);
    M1 = (double*)malloc(sizeof(double)*n2);

    B = (double**)malloc(sizeof(double*)*n2);
    for (i=0; i<n2; i++){
      B[i] = (double*)malloc(sizeof(double)*n2);
    }

    C = (double**)malloc(sizeof(double*)*n2);
    for (i=0; i<n2; i++){
      C[i] = (double*)malloc(sizeof(double)*n2);
    }

    D = (double**)malloc(sizeof(double*)*n2);
    for (i=0; i<n2; i++){
      D[i] = (double*)malloc(sizeof(double)*n2);
    }

    /***********************************************
     construct cluster full matrices of Hamiltonian
              and overlap for the atom Mc_AN             
    ***********************************************/

    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      ig = natn[Gc_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i];
      ih = S_G2M[ig];

      for (j=0; j<=(FNAN[Gc_AN]+SNAN[Gc_AN]); j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[Gc_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j];

	if (0<=kl){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      S_DC[Anum+m][Bnum+n] = OLP0[ih][kl][m][n];
	    }
	  }

	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		H_DC[spin][Anum+m][Bnum+n] = Hks[spin][ih][kl][m][n];
	      }
	    }
	  }
	}

	else{
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      S_DC[Anum+m][Bnum+n] = 0.0;
	    }
	  }

	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		H_DC[spin][Anum+m][Bnum+n] = 0.0;
	      }
	    }
	  }
	}
      }
    }

    /****************************************************
     Solve the generalized eigenvalue problem
     HC = SCE

     1) diagonalize S
     2) search negative eigenvalues of S  
    ****************************************************/

    Eigen_lapack(S_DC,ko,NUM,NUM);

    /***********************************************
           Searching of negative eigenvalues
    ************************************************/

    P_min = 1;
    for (l=1; l<=NUM; l++){
      if (ko[l]<Threshold_OLP_Eigen){
        P_min = l + 1;
        if (3<=level_stdout){
          printf("<DC>  Negative EV of OLP %2d %15.12f\n",l,ko[l]);
	}
      }
    }
    for (l=P_min; l<=NUM; l++){
      M1[l] = 1.0/sqrt(ko[l]);
    }

    /***********************************************
      transform Hamiltonian matrix
    ************************************************/

    for (spin=0; spin<=SpinP_switch; spin++){

      for (i1=1; i1<=NUM; i1++){
        for (j1=P_min; j1<=NUM; j1++){
          sum = 0.0;
          for (l=1; l<=NUM; l++){
	    sum = sum + H_DC[spin][i1][l]*S_DC[l][j1]*M1[j1]; 
          }
          C[i1][j1] = sum;
        }
      }

      for (i1=P_min; i1<=NUM; i1++){
        for (j1=1; j1<=NUM; j1++){
          sum = 0.0;
          for (l=1; l<=NUM; l++){
	    sum = sum + M1[i1]*S_DC[l][i1]*C[l][j1];
          }
          B[i1][j1] = sum;
        }
      }

      for (i1=P_min; i1<=NUM; i1++){
        for (j1=P_min; j1<=NUM; j1++){
          D[i1-(P_min-1)][j1-(P_min-1)] = B[i1][j1];       
        }
      }

      /***********************************************
       diagonalize the trasformed Hamiltonian matrix
      ************************************************/

      NUM1 = NUM - (P_min - 1);
      Msize1[Mc_AN] = NUM1;

      Eigen_lapack(D,ko,NUM1,NUM1);

      /***********************************************
        transformation to the original eigenvectors.
                        NOTE 244P
      ***********************************************/

      for (i1=1; i1<=NUM; i1++){
        for (j1=1; j1<=NUM; j1++){
          C[i1][j1] = 0.0;
        }
      }

      for (i1=1; i1<=NUM; i1++){
        for (j1=1; j1<=NUM1; j1++){
          sum = 0.0;
          for (l=P_min; l<=NUM; l++){
            sum = sum + S_DC[i1][l]*M1[l]*D[l-(P_min-1)][j1];
          }
          C[i1][j1] = sum;
        }
      }

      /***********************************************
           store eigenvalues and residues of poles
      ***********************************************/

      for (i1=1; i1<=NUM; i1++){
        EVal[spin][Mc_AN][i1-1] = 1000.0;
      }
      for (i1=1; i1<=NUM1; i1++){
        EVal[spin][Mc_AN][i1-1] = ko[i1];
      }

      wanA = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wanA];

      for (i=0; i<tno1; i++){
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          wanB = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_CNO[wanB];
          Bnum = MP[h_AN];
          for (j=0; j<tno2; j++){
            for (i1=1; i1<=NUM1; i1++){
              Residues[spin][Mc_AN][h_AN][i][j][i1-1] = C[1+i][i1]*C[Bnum+j][i1];
	    }
	  }
	}
      }      

    } /* end of spin */

    /****************************************************
                        free arrays
    ****************************************************/

    for (i=0; i<n2; i++){
      free(S_DC[i]);
    }
    free(S_DC);

    for (spin=0; spin<=SpinP_switch; spin++){
      for (i=0; i<n2; i++){
        free(H_DC[spin][i]);
      }
      free(H_DC[spin]);
    }
    free(H_DC);

    free(ko);
    free(M1);

    for (i=0; i<n2; i++){
      free(B[i]);
    }
    free(B);

    for (i=0; i<n2; i++){
      free(C[i]);
    }
    free(C);

    for (i=0; i<n2; i++){
      free(D[i]);
    }
    free(D);

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  } /* end of Mc_AN */

  /****************************************************
                   fprintf *.Dos.val
  ****************************************************/

  sprintf(file_eig,"%s%s.Dos.val",filepath,filename);

  if (myid==Host_ID){
    if ( (fp_eig=fopen(file_eig,"w")) != NULL ) {

#ifdef xt3
      setvbuf(fp_eig,buf1,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      fprintf(fp_eig,"mode        5\n");
      fprintf(fp_eig,"NonCol      0\n");
      /*      fprintf(fp_eig,"N           %d\n",n); */
      fprintf(fp_eig,"Nspin       %d\n",SpinP_switch);
      fprintf(fp_eig,"Erange      %lf %lf\n",Dos_Erange[0],Dos_Erange[1]);
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

      fclose(fp_eig);
    }
    else{
      printf("failure of saving %s\n",file_eig);
    }
  }

  /****************************************************
              calculate projected DOS
                      and 
               fprintf *.Dos.vec
  ****************************************************/

  sprintf(file_ev, "%s%s.Dos.vec%i",filepath,filename,myid);

  if ( (fp_ev=fopen(file_ev,"w")) != NULL ) {

#ifdef xt3
    setvbuf(fp_ev,buf2,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

        fprintf(fp_ev,"<AN%dAN%d\n",Gc_AN,spin);
        fprintf(fp_ev,"%d\n",Msize1[Mc_AN]);

        for (i1=0; i1<Msize1[Mc_AN]; i1++){

          fprintf(fp_ev,"%4d  %10.6f  ",i1,EVal[spin][Mc_AN][i1]);

          for (i=0; i<tno1; i++){

	    sum = 0.0;
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      wanB = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_CNO[wanB];
	      for (j=0; j<tno2; j++){
		sum += Residues[spin][Mc_AN][h_AN][i][j][i1]*
                                 OLP0[Mc_AN][h_AN][i][j];
	      }
	    }

            fprintf(fp_ev,"%8.5f",sum);
	  }
          fprintf(fp_ev,"\n");
	}

        fprintf(fp_ev,"AN%dAN%d>\n",Gc_AN,spin);

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }
    }

    fclose(fp_ev);
  }
  else {
    printf("failure of saving %s\n",file_ev);
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/

  free(Snd_H_Size);
  free(Rcv_H_Size);

  free(Snd_S_Size);
  free(Rcv_S_Size);

  free(MP);
  free(Msize);
  free(Msize1);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(EVal[spin][Mc_AN]);
    }
    free(EVal[spin]);
  }
  free(EVal);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      if (Mc_AN==0){
        Gc_AN = 0;
        FNAN[0] = 1;
        tno1 = 1;
      }
      else{
        wanA = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_CNO[wanA];
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        if (Mc_AN==0){
          tno2 = 1;
        }
        else {
          Gh_AN = natn[Gc_AN][h_AN];
          wanB = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_CNO[wanB];
        }

        for (i=0; i<tno1; i++){
          for (j=0; j<tno2; j++){
            free(Residues[spin][Mc_AN][h_AN][i][j]);
	  }
          free(Residues[spin][Mc_AN][h_AN][i]);
        }
        free(Residues[spin][Mc_AN][h_AN]);
      }
      free(Residues[spin][Mc_AN]);
    }
    free(Residues[spin]);
  }
  free(Residues);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  /* for PrintMemory */
  firsttime=0;

  return time0;
}









static double DC_Dosout_NonCol(double *****Hks,
                               double *****ImNL,
                               double ****OLP0)
{
  static int firsttime=1;
  int Mc_AN,Gc_AN,i,Gi,wan,wanA,wanB,Anum;
  int size1,size2,num,NUM,NUM1,n2,Cwan,Hwan;
  int ih,ig,ian,j,kl,jg,jan,Bnum,m,n,spin;
  int l,i1,j1,P_min,m_size;
  int ii1,jj1,k;
  int po,loopN,tno1,tno2,h_AN,Gh_AN,MaxL;
  double sum,FermiF,time0,sum_r,sum_i;
  double My_Num_State,Num_State,x,Dnum;
  double TStime,TEtime;
  double My_Eele0[2],My_Eele1[2];
  double max_x=50.0;
  double ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double tmp1,tmp2,tmp3,SDup,SDdn;
  double Re11,Re22,Re12,Im12;
  double cot,sit,cop,sip,theta,phi;
  double **S_DC,*ko,*M1;
  dcomplex **C,**H_DC;
  double **EVal;
  double ******Residues;
  int *MP,*Msize,*Msize1;
  double *tmp_array;
  double *tmp_array2;
  int *Snd_H_Size,*Rcv_H_Size;
  int *Snd_S_Size,*Rcv_S_Size;
  int numprocs,myid,ID,IDS,IDR,tag=999;
  double Stime_atom, Etime_atom;
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  char buf1[fp_bsize];          /* setvbuf */
  char buf2[fp_bsize];          /* setvbuf */

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  /****************************************************
    allocation of arrays:

    int MP[List_YOUSO[2]];
    int Msize[Matomnum+1];
    int Msize1[Matomnum+1];
    double EVal[Matomnum+1][n2];
  ****************************************************/

  Snd_H_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_H_Size = (int*)malloc(sizeof(int)*numprocs);
  Snd_S_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_S_Size = (int*)malloc(sizeof(int)*numprocs);

  m_size = 0;
  MP = (int*)malloc(sizeof(int)*List_YOUSO[2]);
  Msize = (int*)malloc(sizeof(int)*(Matomnum+1));
  Msize1 = (int*)malloc(sizeof(int)*(Matomnum+1));

  EVal = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (Mc_AN==0){
      Gc_AN = 0;
      FNAN[0] = 1;
      SNAN[0] = 0;
      n2 = 1;
      Msize[Mc_AN] = 1;
    }
    else{
      Anum = 1;
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	Anum = Anum + Spe_Total_CNO[wanA];
      }
      NUM = Anum - 1;
      Msize[Mc_AN] = NUM;
      n2 = 2*NUM + 3;
    }

    m_size += n2;

    EVal[Mc_AN] = (double*)malloc(sizeof(double)*n2);
  }

  if (firsttime)
  PrintMemory("Divide_Conquer_Dosout: EVal",sizeof(double)*m_size,NULL);

  if (2<=level_stdout){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        printf("<DC> myid=%i Mc_AN=%2d Gc_AN=%2d Msize=%3d\n",
        myid,Mc_AN,M2G[Mc_AN],Msize[Mc_AN]);
    }
  }

  /****************************************************
    allocation of arrays:

    double Residues[4]
                   [Matomnum+1]
                   [FNAN[Gc_AN]+1]
                   [Spe_Total_CNO[Gc_AN]] 
                   [Spe_Total_CNO[Gh_AN]] 
                   [NUM2]
     To reduce the memory size, the size of NUM2 is
     needed to be found in the loop.  
  ****************************************************/

  m_size = 0;
  Residues = (double******)malloc(sizeof(double*****)*4);
  for (spin=0; spin<4; spin++){
    Residues[spin] = (double*****)malloc(sizeof(double****)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      if (Mc_AN==0){
        Gc_AN = 0;
        FNAN[0] = 1;
        tno1 = 1;
        n2 = 1;
      }
      else{
        wanA = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_CNO[wanA];
        n2 = 2*Msize[Mc_AN] + 2;
      }

      Residues[spin][Mc_AN] = (double****)malloc(sizeof(double***)*(FNAN[Gc_AN]+1));

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno2 = 1;
        }
        else {
          Gh_AN = natn[Gc_AN][h_AN];
          wanB = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_CNO[wanB];
        }

        Residues[spin][Mc_AN][h_AN] = (double***)malloc(sizeof(double**)*tno1);
        for (i=0; i<tno1; i++){
          Residues[spin][Mc_AN][h_AN][i] = (double**)malloc(sizeof(double*)*tno2);
          for (j=0; j<tno2; j++){
            Residues[spin][Mc_AN][h_AN][i][j] = (double*)malloc(sizeof(double)*n2);
	  }
        }

        m_size += tno1*tno2*n2;
      }
    }
  }


  if (firsttime)
  PrintMemory("Divide_Conquer_Dosout: Residues",sizeof(double)*m_size,NULL);

  /****************************************************
   MPI

   Hks
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
      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

        size1 = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
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
                  size1++; 
                } 
              } 
	    }
          }
	}
 
        Snd_H_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_H_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_H_Size[IDR] = size2;
      }
      else{
        Rcv_H_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);

    }
    else{
      Snd_H_Size[IDS] = 0;
      Rcv_H_Size[IDR] = 0;
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

        size1 = Snd_H_Size[IDS];

        /* allocation of array */

        tmp_array = (double*)malloc(sizeof(double)*size1);

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
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
                  tmp_array[num] = Hks[spin][Mc_AN][h_AN][i][j];
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

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

        size2 = Rcv_H_Size[IDR];
        
        /* allocation of array */
        tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = S_TopMAN[IDR] - 1;  /* S_TopMAN should be used. */
          for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
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
                  Hks[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
		}
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

  /****************************************************
   MPI

   OLP0
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
      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
	size1 = 0;
	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
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
		size1++; 
	      } 
	    } 
	  }
	}

	Snd_S_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_S_Size[IDS] = 0;
      }

      /* receiving of size of data */
 
      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_S_Size[IDR] = size2;
      }
      else{
	Rcv_S_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
    }
    else{
      Snd_S_Size[IDS] = 0;
      Rcv_S_Size[IDR] = 0;
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

	size1 = Snd_S_Size[IDS];

	/* allocation of array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
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
		tmp_array[num] = OLP0[Mc_AN][h_AN][i][j];
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
          
	size2 = Rcv_S_Size[IDR];
        
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

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
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

  /****************************************************
      Setting of Hamiltonian and overlap matrices

         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];

    /***********************************************
      find the size of matrix for the atom Mc_AN
                and set the MP vector

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
    ***********************************************/
    
    Anum = 1;
    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      MP[i] = Anum;
      Gi = natn[Gc_AN][i];
      wanA = WhatSpecies[Gi];
      Anum = Anum + Spe_Total_CNO[wanA];
    }
    NUM = Anum - 1;

    n2 = 2*NUM + 3;

    /***********************************************
     allocation of arrays:
     
     double   S_DC[NUM+2][NUM+2];
     dcomplex H_DC[n2][n2];     
     double   ko[n2];
     double   M1[n2];
     dcomplex C[n2][n2];     
    ***********************************************/

    S_DC = (double**)malloc(sizeof(double*)*(NUM+2));
    for (i=0; i<(NUM+2); i++){
      S_DC[i] = (double*)malloc(sizeof(double)*(NUM+2));
    }

    H_DC = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
    for (i=0; i<n2; i++){
      H_DC[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
    }

    ko = (double*)malloc(sizeof(double)*n2);
    M1 = (double*)malloc(sizeof(double)*n2);

    C = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
    for (i=0; i<n2; i++){
      C[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
    }

    /***********************************************
     construct cluster full matrices of Hamiltonian
              and overlap for the atom Mc_AN             
    ***********************************************/

    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      ig = natn[Gc_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i];
      ih = S_G2M[ig];

      for (j=0; j<=(FNAN[Gc_AN]+SNAN[Gc_AN]); j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[Gc_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j];

	if (0<=kl){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      S_DC[Anum+m][Bnum+n] = OLP0[ih][kl][m][n];
	    }
	  }

          /* non-spin-orbit coupling and non-LDA+U */  
          if ( SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
              && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0 ){
  	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
	        H_DC[Anum+m    ][Bnum+n    ].r =  Hks[0][ih][kl][m][n];
	        H_DC[Anum+m    ][Bnum+n    ].i =  0.0;
	        H_DC[Anum+m+NUM][Bnum+n+NUM].r =  Hks[1][ih][kl][m][n];
	        H_DC[Anum+m+NUM][Bnum+n+NUM].i =  0.0;
	        H_DC[Anum+m    ][Bnum+n+NUM].r =  Hks[2][ih][kl][m][n];
	        H_DC[Anum+m    ][Bnum+n+NUM].i =  Hks[3][ih][kl][m][n];
                H_DC[Bnum+n+NUM][Anum+m    ].r =  H_DC[Anum+m    ][Bnum+n+NUM].r; 
	        H_DC[Bnum+n+NUM][Anum+m    ].i = -H_DC[Anum+m    ][Bnum+n+NUM].i;
	      }
	    }
	  }

          /* spin-orbit coupling or LDA+U */  
          else {
  	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
	        H_DC[Anum+m    ][Bnum+n    ].r =  Hks[0][ih][kl][m][n];
	        H_DC[Anum+m    ][Bnum+n    ].i =  ImNL[0][ih][kl][m][n];
	        H_DC[Anum+m+NUM][Bnum+n+NUM].r =  Hks[1][ih][kl][m][n];
	        H_DC[Anum+m+NUM][Bnum+n+NUM].i =  ImNL[1][ih][kl][m][n];
	        H_DC[Anum+m    ][Bnum+n+NUM].r =  Hks[2][ih][kl][m][n];
	        H_DC[Anum+m    ][Bnum+n+NUM].i =  Hks[3][ih][kl][m][n] + ImNL[2][ih][kl][m][n];
                H_DC[Bnum+n+NUM][Anum+m    ].r =  H_DC[Anum+m    ][Bnum+n+NUM].r; 
	        H_DC[Bnum+n+NUM][Anum+m    ].i = -H_DC[Anum+m    ][Bnum+n+NUM].i;
	      }
	    }
	  }

	}

	else{

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      S_DC[Anum+m][Bnum+n] = 0.0;
	    }
	  }

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      H_DC[Anum+m    ][Bnum+n    ].r = 0.0;
	      H_DC[Anum+m    ][Bnum+n    ].i = 0.0;
	      H_DC[Anum+m+NUM][Bnum+n+NUM].r = 0.0;
	      H_DC[Anum+m+NUM][Bnum+n+NUM].i = 0.0;
	      H_DC[Anum+m    ][Bnum+n+NUM].r = 0.0;
	      H_DC[Anum+m    ][Bnum+n+NUM].i = 0.0;
	      H_DC[Anum+m+NUM][Bnum+n    ].r = 0.0;
	      H_DC[Anum+m+NUM][Bnum+n    ].i = 0.0;
	    }
	  }

	}
      }
    }

    /****************************************************
     Solve the generalized eigenvalue problem
     HC = SCE

     1) diagonalize S
     2) search negative eigenvalues of S  
    ****************************************************/

    Eigen_lapack(S_DC,ko,NUM,NUM);

    /***********************************************
           Searching of negative eigenvalues
    ************************************************/

    P_min = 1;
    for (l=1; l<=NUM; l++){
      if (ko[l]<Threshold_OLP_Eigen){
        P_min = l + 1;
        if (3<=level_stdout){
          printf("<DC>  Negative EV of OLP %2d %15.12f\n",l,ko[l]);
	}
      }
    }
    for (l=P_min; l<=NUM; l++){
      M1[l] = 1.0/sqrt(ko[l]);
    }

    /***********************************************
      transform Hamiltonian matrix
    ************************************************/

    /* H * U * M1 */

    for (i1=1; i1<=2*NUM; i1++){
      for (j1=P_min; j1<=NUM; j1++){

        for (k=0; k<=1; k++){

	  sum_r = 0.0;
	  sum_i = 0.0;

	  for (l=1; l<=NUM; l++){
	    sum_r += H_DC[i1][l+k*NUM].r*S_DC[l][j1]*M1[j1]; 
	    sum_i += H_DC[i1][l+k*NUM].i*S_DC[l][j1]*M1[j1]; 
	  }

          jj1 = 2*j1 - P_min + k;

	  C[i1][jj1].r = sum_r;
	  C[i1][jj1].i = sum_i;

	}
      }
    }

    /* M1 * U^+ H * U * M1 */

    for (i1=P_min; i1<=NUM; i1++){
      for (k=0; k<=1; k++){

        ii1 = 2*i1 - P_min + k;

        for (j1=1; j1<=2*NUM; j1++){
	  sum_r = 0.0;
	  sum_i = 0.0;
	  for (l=1; l<=NUM; l++){
	    sum_r += M1[i1]*S_DC[l][i1]*C[l+k*NUM][j1].r;
	    sum_i += M1[i1]*S_DC[l][i1]*C[l+k*NUM][j1].i;
	  }
	  H_DC[ii1][j1].r = sum_r;
	  H_DC[ii1][j1].i = sum_i;
        }
      }
    }

    /* H to C */

    for (i1=P_min; i1<=2*NUM; i1++){
      for (j1=P_min; j1<=2*NUM; j1++){
	C[i1-(P_min-1)][j1-(P_min-1)].r = H_DC[i1][j1].r;
	C[i1-(P_min-1)][j1-(P_min-1)].i = H_DC[i1][j1].i;
      }
    }

    /***********************************************
     diagonalize the trasformed Hamiltonian matrix
    ************************************************/

    NUM1 = 2*NUM - (P_min - 1);
    Msize1[Mc_AN] = NUM1;
    EigenBand_lapack(C, ko, NUM1, NUM1, 1);

    for (i1=1; i1<=NUM1; i1++){
      for (j1=1; j1<=NUM1; j1++){
        H_DC[i1][j1].r = C[i1][j1].r;
        H_DC[i1][j1].i = C[i1][j1].i;
      }
    }

    /***********************************************
      transformation to the original eigen vectors.
      NOTE 244P    C = U * lambda^{-1/2} * D
    ***********************************************/

    for (i1=1; i1<=2*NUM; i1++){
      for (j1=1; j1<=2*NUM; j1++){
	C[i1][j1].r = 0.0;
	C[i1][j1].i = 0.0;
      }
    }

    for (k=0; k<=1; k++){
      for (i1=1; i1<=NUM; i1++){
        for (j1=1; j1<=NUM1; j1++){
	  sum_r = 0.0;
	  sum_i = 0.0;
	  for (l=P_min; l<=NUM; l++){
	    sum_r += S_DC[i1][l]*M1[l]*H_DC[2*(l-P_min)+1+k][j1].r;
	    sum_i += S_DC[i1][l]*M1[l]*H_DC[2*(l-P_min)+1+k][j1].i;
	  }
	  C[i1+k*NUM][j1].r = sum_r;
	  C[i1+k*NUM][j1].i = sum_i;
        }
      }
    }

    /***********************************************
         store eigenvalues and residues of poles
    ***********************************************/

    for (i1=1; i1<=2*NUM; i1++){
      EVal[Mc_AN][i1-1] = 1000.0;
    }
    for (i1=1; i1<=NUM1; i1++){
      EVal[Mc_AN][i1-1] = ko[i1];
    }

    wanA = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[wanA];

    for (i=0; i<tno1; i++){
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	wanB = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[wanB];
	Bnum = MP[h_AN];
	for (j=0; j<tno2; j++){
	  for (i1=1; i1<=NUM1; i1++){

            /* Re11 */
	    Residues[0][Mc_AN][h_AN][i][j][i1-1] = C[1+i    ][i1].r*C[Bnum+j    ][i1].r
                                                  +C[1+i    ][i1].i*C[Bnum+j    ][i1].i;

            /* Re22 */
	    Residues[1][Mc_AN][h_AN][i][j][i1-1] = C[1+i+NUM][i1].r*C[Bnum+j+NUM][i1].r
                                                  +C[1+i+NUM][i1].i*C[Bnum+j+NUM][i1].i;

            /* Re12 */
	    Residues[2][Mc_AN][h_AN][i][j][i1-1] = C[1+i    ][i1].r*C[Bnum+j+NUM][i1].r
                                                  +C[1+i    ][i1].i*C[Bnum+j+NUM][i1].i;

	    /* Im12
	       conjugate complex of Im12 due to difference in the definition
	       between density matrix and charge density
	    */
	    Residues[3][Mc_AN][h_AN][i][j][i1-1] =-(C[1+i    ][i1].r*C[Bnum+j+NUM][i1].i
                                                   -C[1+i    ][i1].i*C[Bnum+j+NUM][i1].r);

	  }
	}
      }
    }      

    /****************************************************
                        free arrays
    ****************************************************/

    for (i=0; i<(NUM+2); i++){
      free(S_DC[i]);
    }
    free(S_DC);

    for (i=0; i<n2; i++){
      free(H_DC[i]);
    }
    free(H_DC);

    free(ko);
    free(M1);

    for (i=0; i<n2; i++){
      free(C[i]);
    }
    free(C);

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  } /* end of Mc_AN */

  /****************************************************
                   fprintf *.Dos.val
  ****************************************************/

  sprintf(file_eig,"%s%s.Dos.val",filepath,filename);

  if (myid==Host_ID){
    if ( (fp_eig=fopen(file_eig,"w")) != NULL ) {

#ifdef xt3
      setvbuf(fp_eig,buf1,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      fprintf(fp_eig,"mode        5\n");
      fprintf(fp_eig,"NonCol      1\n");
      /*      fprintf(fp_eig,"N           %d\n",n); */
      fprintf(fp_eig,"Nspin       %d\n",1);  /* switch to 1 */
      fprintf(fp_eig,"Erange      %lf %lf\n",Dos_Erange[0],Dos_Erange[1]);
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

      fclose(fp_eig);
    }
    else{
      printf("failure of saving %s\n",file_eig);
    }
  }

  /****************************************************
              calculate projected DOS
                      and 
               fprintf *.Dos.vec
  ****************************************************/

  sprintf(file_ev, "%s%s.Dos.vec%i",filepath,filename,myid);

  if ( (fp_ev=fopen(file_ev,"w")) != NULL ) {

#ifdef xt3
    setvbuf(fp_ev,buf2,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wanA = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wanA];

      theta = Angle0_Spin[Gc_AN];
      phi   = Angle1_Spin[Gc_AN];

      sit = sin(theta);
      cot = cos(theta);
      sip = sin(phi);
      cop = cos(phi);     

      fprintf(fp_ev,"<AN%d\n",Gc_AN);
      fprintf(fp_ev,"%d %d\n",Msize1[Mc_AN],Msize1[Mc_AN]);

      for (i1=0; i1<Msize1[Mc_AN]; i1++){

	fprintf(fp_ev,"%4d  %10.6f %10.6f ",i1,EVal[Mc_AN][i1],EVal[Mc_AN][i1]);

	for (i=0; i<tno1; i++){

	  Re11 = 0.0;
	  Re22 = 0.0;
	  Re12 = 0.0;
	  Im12 = 0.0;

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    wanB = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wanB];

	    for (j=0; j<tno2; j++){

	      Re11 += Residues[0][Mc_AN][h_AN][i][j][i1]*
 		                 OLP0[Mc_AN][h_AN][i][j];

	      Re22 += Residues[1][Mc_AN][h_AN][i][j][i1]*
 		                 OLP0[Mc_AN][h_AN][i][j];

	      Re12 += Residues[2][Mc_AN][h_AN][i][j][i1]*
 		                 OLP0[Mc_AN][h_AN][i][j];

	      Im12 += Residues[3][Mc_AN][h_AN][i][j][i1]*
 		                 OLP0[Mc_AN][h_AN][i][j];

	    }
	  }

          tmp1 = 0.5*(Re11 + Re22);
          tmp2 = 0.5*cot*(Re11 - Re22);
          tmp3 = (Re12*cop - Im12*sip)*sit;

          SDup = tmp1 + tmp2 + tmp3;
          SDdn = tmp1 - tmp2 - tmp3;

	  fprintf(fp_ev,"%8.5f %8.5f ",SDup,SDdn);
	}
	fprintf(fp_ev,"\n");
      }

      fprintf(fp_ev,"AN%d>\n",Gc_AN);

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    fclose(fp_ev);
  }
  else {
    printf("failure of saving %s\n",file_ev);
  }

  /****************************************************
    freeing of arrays:

  ****************************************************/

  free(Snd_H_Size);
  free(Rcv_H_Size);

  free(Snd_S_Size);
  free(Rcv_S_Size);

  free(MP);
  free(Msize);
  free(Msize1);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(EVal[Mc_AN]);
  }
  free(EVal);

  for (spin=0; spin<4; spin++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      if (Mc_AN==0){
        Gc_AN = 0;
        FNAN[0] = 1;
        tno1 = 1;
      }
      else{
        wanA = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_CNO[wanA];
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        if (Mc_AN==0){
          tno2 = 1;
        }
        else {
          Gh_AN = natn[Gc_AN][h_AN];
          wanB = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_CNO[wanB];
        }

        for (i=0; i<tno1; i++){
          for (j=0; j<tno2; j++){
            free(Residues[spin][Mc_AN][h_AN][i][j]);
	  }
          free(Residues[spin][Mc_AN][h_AN][i]);
        }
        free(Residues[spin][Mc_AN][h_AN]);
      }
      free(Residues[spin][Mc_AN]);
    }
    free(Residues[spin]);
  }
  free(Residues);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  /* for PrintMemory */
  firsttime=0;

  return time0;
}

