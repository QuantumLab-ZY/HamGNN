/**********************************************************************
  Initial_CntCoes.c:

     Initial_CntCoes.c is a subroutine to prepare initial contraction
     coefficients for the orbital optimization method.

  Log of Initial_CntCoes.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"

#pragma optimization_level 1
void Initial_CntCoes2(double *****nh, double *****OLP)
{
  static int firsttime=1;
  int i,j,l,n,n2,i1,j1;
  int wan;
  int po;
  int Second_switch;
  double time0;
  int Mc_AN,Gc_AN,wanA;
  int q,q0,al0,al1,pmax;
  int Mul0,Mul1,deg_on,deg_num;
  int al,p,L0,M0,p0,Np;
  int ig,im,jg,ian,jan,kl,m,Gi;
  int mu,nu,Anum,Bnum,NUM,maxp;
  int h_AN,Gh_AN,Hwan,tno1,tno2,Cwan,spin;
  double Beta0,scaleF,maxc;
  double *ko,*C0,*koSys;
  double **S,**Hks,**D,*abs_sum,*M1,**C,**B;
  int *jun,*ponu;
  double tmp0,tmp1,Max0,rc1,fugou,MaxV;
  double sum,TZ;
  double Num_State,x,FermiF,Dnum;
  double LChemP_MAX,LChemP_MIN,LChemP;
  double TStime,TEtime;

  double *tmp_array;
  double *tmp_array2;
  int *MP,*dege;
  int **tmp_index;
  int ***tmp_index1;
  int ***tmp_index2;
  double *Tmp_CntCoes;
  double **Check_ko;
  double *Weight_ko;
  double ***CntCoes_Spe;
  double ***My_CntCoes_Spe;
  double **InProd;
  int *Snd_CntCoes_Size;
  int *Rcv_CntCoes_Size;
  int *Snd_H_Size,*Rcv_H_Size;
  int *Snd_S_Size,*Rcv_S_Size;
  int size1,size2,num;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  /****************************************************
    allocation of arrays:

     int MP[List_YOUSO[8]];

     int tmp_index[List_YOUSO[25]+1]
                         [2*(List_YOUSO[25]+1)+1];
     int tmp_index1[List_YOUSO[25]+1]
                          [List_YOUSO[24]]
                          [2*(List_YOUSO[25]+1)+1];

     int tmp_index2[List_YOUSO[25]+1]
                          [List_YOUSO[24]]
                          [2*(List_YOUSO[25]+1)+1];
 
     double Tmp_CntCoes[List_YOUSO[24]] 

     double Check_ko[List_YOUSO[25]+1]
                           [2*(List_YOUSO[25]+1)+1];

     double Weight_ko[List_YOUSO[7]];

  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[8]);
  
  tmp_index = (int**)malloc(sizeof(int*)*(List_YOUSO[25]+1)); 
  for (i=0; i<(List_YOUSO[25]+1); i++){
    tmp_index[i] = (int*)malloc(sizeof(int)*(2*(List_YOUSO[25]+1)+1)); 
  }

  tmp_index1 = (int***)malloc(sizeof(int**)*(List_YOUSO[25]+1)); 
  for (i=0; i<(List_YOUSO[25]+1); i++){
    tmp_index1[i] = (int**)malloc(sizeof(int*)*List_YOUSO[24]); 
    for (j=0; j<List_YOUSO[24]; j++){
      tmp_index1[i][j] = (int*)malloc(sizeof(int)*(2*(List_YOUSO[25]+1)+1)); 
    }
  }

  tmp_index2 = (int***)malloc(sizeof(int**)*(List_YOUSO[25]+1)); 
  for (i=0; i<(List_YOUSO[25]+1); i++){
    tmp_index2[i] = (int**)malloc(sizeof(int*)*List_YOUSO[24]); 
    for (j=0; j<List_YOUSO[24]; j++){
      tmp_index2[i][j] = (int*)malloc(sizeof(int)*(2*(List_YOUSO[25]+1)+1)); 
    }
  }

  Tmp_CntCoes = (double*)malloc(sizeof(double)*List_YOUSO[24]); 

  Check_ko = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1)); 
  for (i=0; i<(List_YOUSO[25]+1); i++){
    Check_ko[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1)); 
  }

  Weight_ko = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  Snd_CntCoes_Size = (int*)malloc(sizeof(int)*Num_Procs);
  Rcv_CntCoes_Size = (int*)malloc(sizeof(int)*Num_Procs);
  Snd_H_Size = (int*)malloc(sizeof(int)*Num_Procs);
  Rcv_H_Size = (int*)malloc(sizeof(int)*Num_Procs);
  Snd_S_Size = (int*)malloc(sizeof(int)*Num_Procs);
  Rcv_S_Size = (int*)malloc(sizeof(int)*Num_Procs);

  /* PrintMemory */

  if (firsttime) {
    PrintMemory("Initial_CntCoes: tmp_index",sizeof(int)*(List_YOUSO[25]+1)*
                                            (2*(List_YOUSO[25]+1)+1),NULL);
    PrintMemory("Initial_CntCoes: tmp_index1",sizeof(int)*(List_YOUSO[25]+1)*
                             List_YOUSO[24]*(2*(List_YOUSO[25]+1)+1) ,NULL);
    PrintMemory("Initial_CntCoes: tmp_index2",sizeof(int)*(List_YOUSO[25]+1)*
                             List_YOUSO[24]*(2*(List_YOUSO[25]+1)+1) ,NULL);
    PrintMemory("Initial_CntCoes: Check_ko",sizeof(double)*(List_YOUSO[25]+1)*
                                            (2*(List_YOUSO[25]+1)+1),NULL);
    firsttime=0;
  }

  /****************************************************
    MPI:

    nh(=H)
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
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_NO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_NO[Hwan];
              size1 += tno1*tno2; 
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

      if (F_Rcv_Num[IDR]!=0){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_H_Size[IDR] = size2;
      }
      else{
        Rcv_H_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

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

      if (F_Snd_Num[IDS]!=0){

        size1 = Snd_H_Size[IDS];

        /* allocation of array */

        tmp_array = (double*)malloc(sizeof(double)*size1);

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_NO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_NO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  tmp_array[num] = nh[spin][Mc_AN][h_AN][i][j];
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
        
        size2 = Rcv_H_Size[IDR];
        
        /* allocation of array */
        tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
        
        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = S_TopMAN[IDR] - 1;  /* S_TopMAN should be used. */
          for (n=0; n<F_Rcv_Num[IDR]; n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_NO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_NO[Hwan];

              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  nh[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
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
        free(tmp_array);  /* freeing of array */
      } 
    }
  }

  /****************************************************
    MPI:

    OLP[0]
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
          Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_NO[Cwan];
          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
            Gh_AN = natn[Gc_AN][h_AN];        
            Hwan = WhatSpecies[Gh_AN];
            tno2 = Spe_Total_NO[Hwan];
            size1 += tno1*tno2; 
	  }
        }
 
        Snd_S_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_S_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR]!=0){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_S_Size[IDR] = size2;
      }
      else{
        Rcv_S_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

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

      if (F_Snd_Num[IDS]!=0){

        size1 = Snd_S_Size[IDS];

        /* allocation of array */

        tmp_array = (double*)malloc(sizeof(double)*size1);

        /* multidimentional array to vector array */

        num = 0;

        for (n=0; n<F_Snd_Num[IDS]; n++){
          Mc_AN = Snd_MAN[IDS][n];
          Gc_AN = Snd_GAN[IDS][n];
          Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_NO[Cwan];
          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
            Gh_AN = natn[Gc_AN][h_AN];        
            Hwan = WhatSpecies[Gh_AN];
            tno2 = Spe_Total_NO[Hwan];
            for (i=0; i<tno1; i++){
              for (j=0; j<tno2; j++){
                tmp_array[num] = OLP[0][Mc_AN][h_AN][i][j];
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

      if (F_Rcv_Num[IDR]!=0){
        
        size2 = Rcv_S_Size[IDR];
        
        /* allocation of array */
        tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
        
        num = 0;
        Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
        for (n=0; n<F_Rcv_Num[IDR]; n++){
          Mc_AN++;
          Gc_AN = Rcv_GAN[IDR][n];
          Cwan = WhatSpecies[Gc_AN]; 
          tno1 = Spe_Total_NO[Cwan];

          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
            Gh_AN = natn[Gc_AN][h_AN];        
            Hwan = WhatSpecies[Gh_AN];
            tno2 = Spe_Total_NO[Hwan];
            for (i=0; i<tno1; i++){
              for (j=0; j<tno2; j++){
                OLP[0][Mc_AN][h_AN][i][j] = tmp_array2[num];
                num++;
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

  /****************************************************
     set of "initial" coefficients of the contraction
  ****************************************************/

  Second_switch = 1;

  Simple_InitCnt[0] = 0;
  Simple_InitCnt[1] = 0;
  Simple_InitCnt[2] = 0;
  Simple_InitCnt[3] = 0;
  Simple_InitCnt[4] = 0;
  Simple_InitCnt[5] = 0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];

    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        CntCoes[Mc_AN][al][p] = 0.0;
      }
    }

    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
        for (M0=0; M0<=2*L0; M0++){
          al++;
          CntCoes[Mc_AN][al][Mul0] = 1.0;
	}
      }
    }
  }

  if (SICnt_switch==2) goto Simple_Init;

  /****************************************************
      Setting of Hamiltonian and overlap matrices

         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];

    if      ( FNAN[Gc_AN]<30 )  scaleF = 1.6;
    else if ( FNAN[Gc_AN]<40 )  scaleF = 1.4; 
    else if ( FNAN[Gc_AN]<50 )  scaleF = 1.2; 
    else                        scaleF = 1.0; 

    rc1 = scaleF*Spe_Atom_Cut1[wan];

    /***********************************************
         MP indicates the starting position of
              atom i in arraies H and S
    ***********************************************/
    
    Anum = 1;
    TZ = 0.0;
    for (i=0; i<=FNAN[Gc_AN]; i++){
      if (Dis[Gc_AN][i]<=rc1){
        MP[i] = Anum;
        Gi = natn[Gc_AN][i];
        wanA = WhatSpecies[Gi];
        Anum = Anum + Spe_Total_NO[wanA];
        TZ = TZ + Spe_Core_Charge[wanA];
      }
    }
    NUM = Anum - 1;

    /****************************************************
                       allocation
    ****************************************************/

    n2 = NUM + 3;

    koSys   = (double*)malloc(sizeof(double)*n2);
    ko      = (double*)malloc(sizeof(double)*n2);
    abs_sum = (double*)malloc(sizeof(double)*n2);
    M1      = (double*)malloc(sizeof(double)*n2);
    dege    = (int*)malloc(sizeof(int)*n2);
    C0      = (double*)malloc(sizeof(double)*n2);

    S   = (double**)malloc(sizeof(double*)*n2);
    Hks = (double**)malloc(sizeof(double*)*n2);
    D   = (double**)malloc(sizeof(double*)*n2);
    C   = (double**)malloc(sizeof(double*)*n2);
    B   = (double**)malloc(sizeof(double*)*n2);

    for (i=0; i<n2; i++){
      S[i]   = (double*)malloc(sizeof(double)*n2);
      Hks[i] = (double*)malloc(sizeof(double)*n2);
      D[i]   = (double*)malloc(sizeof(double)*n2);
      C[i]   = (double*)malloc(sizeof(double)*n2);
      B[i]   = (double*)malloc(sizeof(double)*n2);
    }

    jun  = (int*)malloc(sizeof(int)*n2);
    ponu = (int*)malloc(sizeof(int)*n2);

    InProd = (double**)malloc(sizeof(double*)*n2);
    for (i=0; i<n2; i++){
      InProd[i] = (double*)malloc(sizeof(double)*n2);
    }

    /****************************************************
                           calc
    ****************************************************/

    if (3<=level_stdout){
      printf("<Initial_CntCoes> Mc_AN=%2d Gc_AN=%2d  NUM=%2d\n",Mc_AN,Gc_AN,NUM);
    }
    
    for (i=0; i<=FNAN[Gc_AN]; i++){

      if (Dis[Gc_AN][i]<=rc1){

	ig = natn[Gc_AN][i];
        im = S_G2M[ig];  /* S_G2M must be used. */

	ian = Spe_Total_NO[WhatSpecies[ig]];
	Anum = MP[i];

	for (j=0; j<=FNAN[Gc_AN]; j++){

	  if (Dis[Gc_AN][j]<=rc1){

	    kl = RMI1[Mc_AN][i][j];
	    jg = natn[Gc_AN][j];
	    jan = Spe_Total_NO[WhatSpecies[jg]];
	    Bnum = MP[j];

	    if (0<=kl){
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){

		  S[Anum+m][Bnum+n] = OLP[0][im][kl][m][n];

		  if (SpinP_switch==0)
		    Hks[Anum+m][Bnum+n] = nh[0][im][kl][m][n];
		  else 
		    Hks[Anum+m][Bnum+n] = 0.5*(nh[0][im][kl][m][n]
				             + nh[1][im][kl][m][n]);
		}
	      }
	    }

	    else{
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  S[Anum+m][Bnum+n] = 0.0;
		  Hks[Anum+m][Bnum+n] = 0.0;
		}
	      }
	    }
	  }
	}
      }
    }

    if (3<=level_stdout){
      printf("\n");
      printf("overlap matrix Gc_AN=%2d\n",Gc_AN);
      for (i1=1; i1<=NUM; i1++){
        for (j1=1; j1<=NUM; j1++){
          printf("%6.3f ",S[i1][j1]); 
        }
        printf("\n");
      }

      printf("\n");
      printf("hamiltonian matrix Gc_AN=%2d\n",Gc_AN);
      for (i1=1; i1<=NUM; i1++){
        for (j1=1; j1<=NUM; j1++){
          printf("%6.3f ",Hks[i1][j1]); 
        }
        printf("\n");
      }
      printf("\n");

    }

    /***********************************************
       Solve the generalized eigenvalue problem
    ***********************************************/

    Eigen_lapack(S,koSys,NUM,NUM);
  
    /***********************************************
           Searching of negative eigenvalues
    ************************************************/

    for (l=1; l<=NUM; l++){
      if (koSys[l]<0.0){

        koSys[l] = 1.0e-7;

        if (3<=level_stdout){
          printf("<Init_CntCoes>  Negative EV of OLP %2d %15.12f\n",l,koSys[l]);
	}
      }
    }
    for (l=1; l<=NUM; l++){
      M1[l] = 1.0/sqrt(koSys[l]);
    }

    for (i1=1; i1<=NUM; i1++){
      for (j1=i1+1; j1<=NUM; j1++){

        tmp0 = S[i1][j1];
        tmp1 = S[j1][i1];

        S[j1][i1] = tmp0;
        S[i1][j1] = tmp1; 
      }
    }
    
    for (i1=1; i1<=NUM; i1++){
      for (j1=1; j1<=NUM; j1++){

        sum = 0.0;
        tmp0 = M1[j1]; 
 
        for (l=1; l<=NUM; l++){
	  sum = sum + Hks[i1][l]*S[j1][l]*tmp0;
        }
        C[j1][i1] = sum;
      }
    }

    for (i1=1; i1<=NUM; i1++){
      for (j1=1; j1<=NUM; j1++){

        sum = 0.0;
        tmp0 = M1[i1]; 

        for (l=1; l<=NUM; l++){
	  sum = sum + tmp0*S[i1][l]*C[j1][l];
        }
        B[i1][j1] = sum;
      }
    }

    for (i1=1; i1<=NUM; i1++){
      for (j1=1; j1<=NUM; j1++){
        D[i1][j1] = B[i1][j1];       
      }
    }

    Eigen_lapack(D,koSys,NUM,NUM);

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
      for (j1=i1+1; j1<=NUM; j1++){

        tmp0 = S[i1][j1];
        tmp1 = S[j1][i1];

        S[j1][i1] = tmp0;
        S[i1][j1] = tmp1; 

        tmp0 = D[i1][j1];
        tmp1 = D[j1][i1];

        D[j1][i1] = tmp0;
        D[i1][j1] = tmp1; 

      }
    }

    for (i1=1; i1<=NUM; i1++){
      for (j1=1; j1<=NUM; j1++){

        sum = 0.0;

        for (l=1; l<=NUM; l++){
          sum = sum + S[i1][l]*M1[l]*D[j1][l];
        }

        C[i1][j1] = sum;
      }
    }

    /****************************************************
           searching of a local chemical potential
    ****************************************************/

    po = 0;
    LChemP_MAX = 10.0;  
    LChemP_MIN =-10.0;

    Beta0 = 1.0/(kB*1500.0/eV2Hartree);

    do{
      LChemP = 0.50*(LChemP_MAX + LChemP_MIN);
      Num_State = 0.0;
      for (i1=1; i1<=NUM; i1++){
        x = (koSys[i1] - LChemP)*Beta0;
        if (x<=-30.0) x = -30.0;
        if (30.0<=x)  x = 30.0;
        FermiF = 2.0/(1.0 + exp(x)); 
        Num_State = Num_State + FermiF;
      }
      Dnum = TZ - Num_State;
      if (0.0<=Dnum) LChemP_MIN = LChemP;
      else           LChemP_MAX = LChemP;
      if (fabs(Dnum)<0.000000000001) po = 1;
    }
    while (po==0); 

    if (3<=level_stdout){
      for (i1=1; i1<=NUM; i1++){
        x = (koSys[i1] - LChemP)*Beta0;
        if (x<=-30.0) x = -30.0;
        if (30.0<=x)  x = 30.0;
        FermiF = 1.0/(1.0 + exp(x)); 
        printf("<Init_CntCoes>  %2d  eigenvalue=%15.12f  FermiF=%15.12f\n",
                i1,koSys[i1],FermiF);
      }
    }

    if (3<=level_stdout){
      printf("\n");
      printf("first C Gc_AN=%2d\n",Gc_AN);
      for (i1=1; i1<=NUM; i1++){
        for (j1=1; j1<=NUM; j1++){
          printf("%10.6f ",C[i1][j1]); 
        }
        printf("\n");
      }
    }
    
    if (3<=level_stdout){
      printf("<Init_CntCoes>  LChemP=%15.12f\n",LChemP);
    }
    
    /************************************************
       maximize the "overlap" between wave functions 
       and contracted basis functions              
    ************************************************/

    /* make a table function converting [L0][Mul0][M0] to "al" for primitive orbitals */

    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
        for (M0=0; M0<=2*L0; M0++){
          al++;
	  tmp_index1[L0][Mul0][M0] = al;
        }
      }
    }

    /* make a table function converting [L0][Mul0][M0] to "al" for contracted orbitals */

    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  tmp_index2[L0][Mul0][M0] = al;
	}
      }
    }

    /* loop for L0 */
     
    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
        for (Mul1=0; Mul1<Spe_Num_Basis[wan][L0]; Mul1++){
          Hks[Mul0+1][Mul1+1] = 0.0;
        }
      }

      for (M0=0; M0<=2*L0; M0++){

	for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

          i = tmp_index1[L0][Mul0][M0]; 

	  for (mu=1; mu<=NUM; mu++){
            InProd[mu][Mul0] = C[MP[0]+i][mu];
	  } /* mu */
	} /* Mul0 */

	for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	  for (Mul1=0; Mul1<Spe_Num_Basis[wan][L0]; Mul1++){

            sum = 0.0;
	    for (mu=1; mu<=NUM; mu++){

	      x = (koSys[mu] - LChemP)*Beta0;
	      if (x<=-30.0) x = -30.0;
	      if (30.0<=x)  x = 30.0;
	      FermiF = 1.0/(1.0 + exp(x)); 

              sum += FermiF*InProd[mu][Mul0]*InProd[mu][Mul1];   
	    }

            Hks[Mul0+1][Mul1+1] -= sum; 
  	  }

          /* for calculation of a single atom */

          tmp0 = (double)(Spe_Num_Basis[wan][L0]-Mul0); 
          Hks[Mul0+1][Mul0+1] += -1.0e-9*tmp0*tmp0;

	}

      } /* M0 */

      /*
      M0 = 0; 
      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
        i = tmp_index1[L0][Mul0][M0]; 
	for (Mul1=0; Mul1<Spe_Num_Basis[wan][L0]; Mul1++){
          j = tmp_index1[L0][Mul1][M0]; 
          S[Mul0+1][Mul1+1] = OLP[0][Mc_AN][0][i][j];
	}
      }
      */

      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	for (Mul1=0; Mul1<Spe_Num_Basis[wan][L0]; Mul1++){
	  S[Mul0+1][Mul1+1] = 0.0;
	}
        S[Mul0+1][Mul0+1] = 1.0;
      }


      if (3<=level_stdout){
	printf("<Hks Gc_AN=%2d L0=%2d>\n",Gc_AN,L0);
	for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	  for (Mul1=0; Mul1<Spe_Num_Basis[wan][L0]; Mul1++){
	    printf("%15.10f ",Hks[Mul0+1][Mul1+1]);
	  }
	  printf("\n");
	}
      }

      if (3<=level_stdout){
	printf("<S Gc_AN=%2d L0=%2d>\n",Gc_AN,L0);
	for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	  for (Mul1=0; Mul1<Spe_Num_Basis[wan][L0]; Mul1++){
	    printf("%15.10f ",S[Mul0+1][Mul1+1]);
	  }
	  printf("\n");
	}
      }

      /* diagonalization */

      Np = Spe_Num_Basis[wan][L0];

      Eigen_lapack(S,ko,Np,Np);

      for (l=1; l<=Np; l++){
        M1[l] = 1.0/sqrt(ko[l]);
      }

      for (i1=1; i1<=Np; i1++){
	for (j1=1; j1<=Np; j1++){
	  sum = 0.0;
	  for (l=1; l<=Np; l++){
	    sum = sum + Hks[i1][l]*S[l][j1]*M1[j1]; 
	  }
	  C[i1][j1] = sum;
	}
      }

      for (i1=1; i1<=Np; i1++){
	for (j1=1; j1<=Np; j1++){
	  sum = 0.0;
	  for (l=1; l<=Np; l++){
	    sum = sum + M1[i1]*S[l][i1]*C[l][j1];
	  }
	  B[i1][j1] = sum;
	}
      }

      for (i1=1; i1<=Np; i1++){
	for (j1=1; j1<=Np; j1++){
	  D[i1][j1] = B[i1][j1];
	}
      }

      Eigen_lapack(D,ko,Np,Np);

      /* transformation to the original eigenvectors */
 
      for (i1=1; i1<=Np; i1++){
	for (j1=1; j1<=Np; j1++){
	  C[i1][j1] = 0.0;
	}
      }

      for (i1=1; i1<=Np; i1++){
	for (j1=1; j1<=Np; j1++){
	  sum = 0.0;
	  for (l=1; l<=Np; l++){
	    sum = sum + S[i1][l]*M1[l]*D[l][j1];
	  }
	  C[i1][j1] = sum;
	}
      }

      if (3<=level_stdout){
	printf("<Eigenvalues Gc_AN=%2d L0=%2d>\n",Gc_AN,L0);
	for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	  printf("Mul=%2d ko=%15.12f\n",Mul0,ko[Mul0+1]);
	}

	printf("<C Gc_AN=%2d L0=%2d>\n",Gc_AN,L0);
	for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	  for (Mul1=0; Mul1<Spe_Num_Basis[wan][L0]; Mul1++){
	    printf("%15.10f ",C[Mul0+1][Mul1+1]);
	  }
	  printf("\n");
	}
      }

      /* set up contraction coefficients */
     
      for (M0=0; M0<=2*L0; M0++){
	for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){

	  al = tmp_index2[L0][Mul0][M0];

          /* if (SCnt_switch==1) */
          if ( SCnt_switch==1 && Mul0==(Spe_Num_CBasis[wan][L0]-1) ){
	    for (p=0; p<Spe_Num_Basis[wan][L0]; p++){
	      CntCoes[Mc_AN][al][p] = C[p+1][1];
  	    }
	  }
 
          else {
	    for (p=0; p<Spe_Num_Basis[wan][L0]; p++){
	      CntCoes[Mc_AN][al][p] = C[p+1][Mul0+1];;
	    }
	  }

          maxc = -1.0e+10;  
	  for (p=0; p<Spe_Num_Basis[wan][L0]; p++){
	    if (maxc<fabs(CntCoes[Mc_AN][al][p])){
              maxc = fabs(CntCoes[Mc_AN][al][p]); 
              maxp = p;
	    }
	  }

          tmp0 = sgn(CntCoes[Mc_AN][al][maxp]);
	  for (p=0; p<Spe_Num_Basis[wan][L0]; p++){
	    CntCoes[Mc_AN][al][p] *= tmp0;
	  }

	}
      }

    } /* L0 */    

    if (3<=level_stdout){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          printf("A Init_CntCoes Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
                  Mc_AN,Gc_AN,al,p,CntCoes[Mc_AN][al][p]);
        }
      }
    }

    /****************************************************
                        free arrays
    ****************************************************/

    for (i=0; i<n2; i++){
      free(InProd[i]);
    }
    free(InProd);

    free(koSys);
    free(ko);
    free(abs_sum);
    free(M1);
    free(dege);
    free(jun);
    free(ponu);
    free(C0);

    for (i=0; i<n2; i++){
      free(S[i]);
      free(Hks[i]);
      free(D[i]);
      free(C[i]);
      free(B[i]);
    }
    free(S);
    free(Hks);
    free(D);
    free(C);
    free(B);

  } /* Mc_AN */ 

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
		CntCoes[Mc_AN][al][p] = 0.0;
	      }

  	      CntCoes[Mc_AN][al][Mul0] = 1.0;
            }

	  }
	}
      }
    }
  }

  /****************************************************
            average contraction coefficients
  ****************************************************/

  if (ACnt_switch==1){

    /* allocation */
    My_CntCoes_Spe = (double***)malloc(sizeof(double**)*(SpeciesNum+1));
    for (i=0; i<=SpeciesNum; i++){
      My_CntCoes_Spe[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
        My_CntCoes_Spe[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
      }
    }

    CntCoes_Spe = (double***)malloc(sizeof(double**)*(SpeciesNum+1));
    for (i=0; i<=SpeciesNum; i++){
      CntCoes_Spe[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
        CntCoes_Spe[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
      }
    }

    /* initialize */
    for (wan=0; wan<SpeciesNum; wan++){
      for (i=0; i<List_YOUSO[7]; i++){
        for (j=0; j<List_YOUSO[24]; j++){
          My_CntCoes_Spe[wan][i][j] = 0.0;
	}
      }
    }

    /* local sum in a proccessor */
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      wan = WhatSpecies[Gc_AN];
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          My_CntCoes_Spe[wan][al][p] += CntCoes[Mc_AN][al][p];
        }        
      }
    }

    /* global sum by MPI */
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          MPI_Allreduce(&My_CntCoes_Spe[wan][al][p], &CntCoes_Spe[wan][al][p],
                         1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
	}
      }
    }    

    /* copy CntCoes_Spe to CntCoes_Species */
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          CntCoes_Species[wan][al][p] = CntCoes_Spe[wan][al][p];
	}
      }
    }    

    /* CntCoes_Spe to CntCoes */
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      wan = WhatSpecies[Gc_AN];
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          CntCoes[Mc_AN][al][p] = CntCoes_Spe[wan][al][p];
        }        
      }
    }

    /* free */
    for (i=0; i<=SpeciesNum; i++){
      for (j=0; j<List_YOUSO[7]; j++){
        free(My_CntCoes_Spe[i][j]);
      }
      free(My_CntCoes_Spe[i]);
    }
    free(My_CntCoes_Spe);

    for (i=0; i<=SpeciesNum; i++){
      for (j=0; j<List_YOUSO[7]; j++){
        free(CntCoes_Spe[i][j]);
      }
      free(CntCoes_Spe[i]);
    }
    free(CntCoes_Spe);

  }

  /**********************************************
    transformation of optimized orbitals by 
    an extended Gauss elimination and 
    the Gram-Schmidt orthogonalization
  ***********************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];

    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  tmp_index2[L0][Mul0][M0] = al;
	}
      }
    }

    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (M0=0; M0<=2*L0; M0++){

	/**********************************************
                  extended Gauss elimination
	***********************************************/

	for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	  al0 = tmp_index2[L0][Mul0][M0]; 
	  for (Mul1=0; Mul1<Spe_Num_CBasis[wan][L0]; Mul1++){
	    al1 = tmp_index2[L0][Mul1][M0];

	    if (Mul1!=Mul0){

	      tmp0 = CntCoes[Mc_AN][al0][Mul0]; 
	      tmp1 = CntCoes[Mc_AN][al1][Mul0]; 

	      for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
		CntCoes[Mc_AN][al1][p] -= CntCoes[Mc_AN][al0][p]/tmp0*tmp1;
	      }
	    }

	  }
	}

	/**********************************************
           orthonormalization of initial contraction 
           coefficients
        ***********************************************/

	for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	  al0 = tmp_index2[L0][Mul0][M0]; 

	  /* x - sum_i <x|e_i>e_i */

	  for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	    Tmp_CntCoes[p] = 0.0;
	  }
         
	  for (Mul1=0; Mul1<Mul0; Mul1++){
	    al1 = tmp_index2[L0][Mul1][M0];

	    sum = 0.0;
	    for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	      sum = sum + CntCoes[Mc_AN][al0][p]*CntCoes[Mc_AN][al1][p];
	    }

	    for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	      Tmp_CntCoes[p] = Tmp_CntCoes[p] + sum*CntCoes[Mc_AN][al1][p];
	    }
	  }

	  for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	    CntCoes[Mc_AN][al0][p] = CntCoes[Mc_AN][al0][p] - Tmp_CntCoes[p];
	  }

	  /* Normalize */

	  sum = 0.0;
	  Max0 = -100.0;
	  pmax = 0;
	  for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	    sum = sum + CntCoes[Mc_AN][al0][p]*CntCoes[Mc_AN][al0][p];
	    if (Max0<fabs(CntCoes[Mc_AN][al0][p])){
	      Max0 = fabs(CntCoes[Mc_AN][al0][p]);
	      pmax = p;
	    }
	  }

	  if (fabs(sum)<1.0e-11)
	    tmp0 = 0.0;
	  else 
	    tmp0 = 1.0/sqrt(sum); 

	  tmp1 = sgn(CntCoes[Mc_AN][al0][pmax]);
            
	  for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	    CntCoes[Mc_AN][al0][p] = tmp0*tmp1*CntCoes[Mc_AN][al0][p];
	  }

	}
      }
    }

  } /* Mc_AN */

  /****************************************************
                     Normalization
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[wan]; al++){

      sum = 0.0;
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	p0 = Spe_Trans_Orbital[wan][al][p];

	for (q=0; q<Spe_Specified_Num[wan][al]; q++){
          q0 = Spe_Trans_Orbital[wan][al][q];

          tmp0 = CntCoes[Mc_AN][al][p]*CntCoes[Mc_AN][al][q];
          sum = sum + tmp0*OLP[0][Mc_AN][0][p0][q0]; 
        }
      }

      tmp0 = 1.0/sqrt(sum);
      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
        CntCoes[Mc_AN][al][p] = CntCoes[Mc_AN][al][p]*tmp0;
      } 
    }

    if (3<=level_stdout){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          printf("B Init_CntCoes Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
                  Mc_AN,Gc_AN,al,p,CntCoes[Mc_AN][al][p]);
        }
      }
    }

  } /* Mc_AN */

 Simple_Init:

  /****************************************************
    MPI:

    CntCoes_Species
  ****************************************************/

  for (wan=0; wan<SpeciesNum; wan++){

    Gc_AN = 1;
    po = 0;

    do {

      wanA = WhatSpecies[Gc_AN];

      if (wan==wanA){

        ID = G2ID[Gc_AN];
        Mc_AN = F_G2M[Gc_AN]; 

        for (al=0; al<Spe_Total_CNO[wan]; al++){
          for (p=0; p<Spe_Specified_Num[wan][al]; p++){

            if (ID==myid) tmp0 = CntCoes[Mc_AN][al][p];

            MPI_Bcast(&tmp0, 1, MPI_DOUBLE, ID, mpi_comm_level1);
            CntCoes_Species[wan][al][p] = tmp0; 
	  }
	}

        po = 1;
      }

      Gc_AN++;

    } while (po==0 && Gc_AN<=atomnum);
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
          for (al=0; al<Spe_Total_CNO[wan]; al++){
            size1 += Spe_Specified_Num[wan][al];
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
          for (al=0; al<Spe_Total_CNO[wan]; al++){
            for (p=0; p<Spe_Specified_Num[wan][al]; p++){
              tmp_array[num] = CntCoes[Mc_AN][al][p];
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
          for (al=0; al<Spe_Total_CNO[wan]; al++){
            for (p=0; p<Spe_Specified_Num[wan][al]; p++){
              CntCoes[Mc_AN][al][p] = tmp_array2[num];
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

  /****************************************************
    freeing of arrays:

     int MP[List_YOUSO[8]];

     int tmp_index[List_YOUSO[25]+1]
                         [2*(List_YOUSO[25]+1)+1];
     int tmp_index1[List_YOUSO[25]+1]
                          [List_YOUSO[24]]
                          [2*(List_YOUSO[25]+1)+1];
     int tmp_index2[List_YOUSO[25]+1]
                          [List_YOUSO[24]]
                          [2*(List_YOUSO[25]+1)+1];
 
     double Tmp_CntCoes[List_YOUSO[24]] 

     double Check_ko[List_YOUSO[25]+1]
                           [2*(List_YOUSO[25]+1)+1];

     double Weight_ko[List_YOUSO[7]];
  ****************************************************/

  free(MP);
  
  for (i=0; i<(List_YOUSO[25]+1); i++){
    free(tmp_index[i]);
  }
  free(tmp_index);

  for (i=0; i<(List_YOUSO[25]+1); i++){
    for (j=0; j<List_YOUSO[24]; j++){
      free(tmp_index1[i][j]);
    }
    free(tmp_index1[i]);
  }
  free(tmp_index1);

  for (i=0; i<(List_YOUSO[25]+1); i++){
    for (j=0; j<List_YOUSO[24]; j++){
      free(tmp_index2[i][j]);
    }
    free(tmp_index2[i]);
  }
  free(tmp_index2);

  free(Tmp_CntCoes);

  for (i=0; i<(List_YOUSO[25]+1); i++){
    free(Check_ko[i]);
  }
  free(Check_ko);

  free(Weight_ko);

  free(Snd_CntCoes_Size);
  free(Rcv_CntCoes_Size);

  free(Snd_H_Size);
  free(Rcv_H_Size);

  free(Snd_S_Size);
  free(Rcv_S_Size);

  /* for elapsed time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
}

