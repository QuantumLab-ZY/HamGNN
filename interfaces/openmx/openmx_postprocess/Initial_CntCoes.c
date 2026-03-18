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


void Initial_CntCoes(double *****nh, double *****OLP)
{
  static int firsttime=1;
  int i,j,l,n,n2,i1,j1;
  int wan;
  int P_min,po;
  int Second_switch;
  double time0;
  int Mc_AN,Gc_AN,wanA;
  int q,q0,al0,al1,pmax;
  int Mul0,Mul1,deg_on,deg_num;
  int al,p,L0,M0,p0;
  int ig,im,jg,ian,jan,kl,m,Gi;
  int mu,nu,Anum,Bnum,NUM,NUM1;
  int h_AN,Gh_AN,Hwan,tno1,tno2,Cwan,spin;
  double Beta0,scaleF;
  double *ko,*C0;
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
  int ***tmp_index2;
  double *Tmp_CntCoes;
  double **Check_ko;
  double *Weight_ko;
  double ***CntCoes_Spe;
  double ***My_CntCoes_Spe;
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

  /****************************************************
      Setting of Hamiltonian and overlap matrices

         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    wan = WhatSpecies[Gc_AN];

    if      ( FNAN[Gc_AN]<40 )  scaleF = 2.0;
    else if ( FNAN[Gc_AN]<55 )  scaleF = 1.6; 
    else if ( FNAN[Gc_AN]<82 )  scaleF = 1.2; 
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

    /***********************************************
       Solve the generalized eigenvalue problem
    ***********************************************/

    Eigen_lapack(S,ko,NUM,NUM);
  
    /***********************************************
           Searching of negative eigenvalues
    ************************************************/

    P_min = 1;
    for (l=1; l<=NUM; l++){
      if (ko[l]<0.0){
        P_min = l + 1;
        if (3<=level_stdout){
          printf("<Init_CntCoes>  Negative EV of OLP %2d %15.12f\n",l,ko[l]);
	}
      }
    }
    for (l=P_min; l<=NUM; l++){
      M1[l] = 1.0/sqrt(ko[l]);
    }
    
    for (i1=1; i1<=NUM; i1++){
      for (j1=P_min; j1<=NUM; j1++){
        sum = 0.0;
        for (l=1; l<=NUM; l++){
	  sum = sum + Hks[i1][l]*S[l][j1]*M1[j1]; 
        }
        C[i1][j1] = sum;
      }
    }

    for (i1=P_min; i1<=NUM; i1++){
      for (j1=1; j1<=NUM; j1++){
        sum = 0.0;
        for (l=1; l<=NUM; l++){
	  sum = sum + M1[i1]*S[l][i1]*C[l][j1];
        }
        B[i1][j1] = sum;
      }
    }

    for (i1=P_min; i1<=NUM; i1++){
      for (j1=P_min; j1<=NUM; j1++){
        D[i1-(P_min-1)][j1-(P_min-1)] = B[i1][j1];       
      }
    }

    NUM1 = NUM - (P_min - 1);
    Eigen_lapack(D,ko,NUM1,NUM1);

    /***********************************************
      transformation to the original eigen vectors.
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
          sum = sum + S[i1][l]*M1[l]*D[l-(P_min-1)][j1];
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
        x = (ko[i1] - LChemP)*Beta0;
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
        x = (ko[i1] - LChemP)*Beta0;
        if (x<=-30.0) x = -30.0;
        if (30.0<=x)  x = 30.0;
        FermiF = 1.0/(1.0 + exp(x)); 
        printf("<Init_CntCoes>  %2d  eigenvalue=%15.12f  FermiF=%15.12f\n",
                i1,ko[i1],FermiF);
      }
    }

    if (3<=level_stdout){
      printf("\n");
      printf("first C Gc_AN=%2d\n",Gc_AN);
      for (i1=1; i1<=NUM; i1++){
        for (j1=1; j1<=10; j1++){
          printf("%6.3f ",C[i1][j1]); 
        }
        printf("\n");
      }
    }
    
    if (3<=level_stdout){
      printf("<Init_CntCoes>  LChemP=%15.12f\n",LChemP);
    }

    /************************************************
             find degenerate eigenvalues
    ************************************************/
    
    for (i1=0; i1<=NUM1; i1++) dege[i1] = 0;

    i1 = 0;
    l = 0;
    deg_on = 0;

    do {
      i1++;
      
      if (fabs(ko[i1]-ko[i1+1])<1.0e-7){
        if (deg_on==0) deg_on = i1;
      }

      else {

        if (deg_on!=0){
          l++;
          for (j1=deg_on; j1<=i1; j1++){
            dege[j1] = l;            
          }
        }

        deg_on = 0;
      }      

    } while(i1<NUM1);

    if (3<=level_stdout){
      for (i1=1; i1<=NUM1; i1++){
        printf("i1=%i  dege=%i\n",i1,dege[i1]);
      }
    }

    /************************************************
          avaraging of degenerate eigenvectors
    ************************************************/

    deg_on  = 0;
    deg_num = 0;

    for (i1=1; i1<=NUM1; i1++){

      if (dege[i1]!=0) {
                  
        for (j1=1; j1<=NUM; j1++){
          if (deg_on==0){ 
            C0[j1]  = C[j1][i1]*C[j1][i1];
	  }
          else if (dege[i1-1]!=dege[i1]){
            C0[j1]  = C[j1][i1]*C[j1][i1];
          }
          else{  
            C0[j1] += C[j1][i1]*C[j1][i1];
	  }
        }         

        if (dege[i1-1]!=dege[i1]) deg_on  = 0;

        if (deg_on==0){
          deg_on = i1;
          deg_num = 0;
	}
      
        deg_num++;
      }   

      if ( deg_on!=0 && (dege[i1]!=dege[i1+1]) ){

        tmp0 = 1.0/sqrt((double)deg_num);
        for (j1=1; j1<=NUM; j1++){
          C0[j1] = sqrt(fabs(C0[j1]))*tmp0;
        }
        
        for (l=deg_on; l<(deg_on+deg_num); l++){
          for (j1=1; j1<=NUM; j1++){
            C[j1][l] = sgn(C[j1][l])*C0[j1];
          }        
        }

        deg_on = 0;
      }

    }

    if (3<=level_stdout){
      printf("\n");
      printf("averaged C\n");
      for (i1=1; i1<=NUM; i1++){
        for (j1=1; j1<=10; j1++){
          printf("%6.3f ",C[i1][j1]); 
        }
        printf("\n");
      }
    }

    /************************************************
                       Contraction     
    ************************************************/

    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
        B[i][j] = 0.0; 
      }
    }

    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
        for (M0=0; M0<=2*L0; M0++){
          al++;
	  tmp_index2[L0][Mul0][M0] = al;
        }
      }
    }

    Beta0 = 1.0/(kB*1500.0/eV2Hartree);

    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

      if (0<Spe_Num_Basis[wan][L0]){

	for (M0=0; M0<=2*L0; M0++){

	  for (mu=1; mu<=NUM1; mu++){

	    p = tmp_index2[L0][0][M0];
	    fugou = sgn(C[p+1][mu]);
	    x = (ko[mu] - LChemP)*Beta0;
	    if (x<=-30.0) x = -30.0;
	    if (30.0<=x)  x = 30.0;
	    FermiF = 1.0/(1.0 + exp(x)); 

	    sum = 0.0;
	    for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	      al = tmp_index2[L0][Mul0][M0]; 
	      B[al+1][1] = B[al+1][1] + fugou*FermiF*C[al+1][mu];
	      sum = sum + FermiF*fabs(C[al+1][mu]);
	    }

	    abs_sum[mu] = sum;
	  }

	  if (Second_switch==0){

	    for (mu=1; mu<=NUM1; mu++) ponu[mu] = 0;
	    for (mu=1; mu<=NUM1; mu++){
	      MaxV = -1000.0;
	      for (nu=1; nu<=NUM1; nu++){
		if (MaxV<abs_sum[nu] && ponu[nu]!=1){
		  MaxV = abs_sum[nu];
		  p = nu;   
		} 
	      }
	      jun[mu] = p;
	      ponu[p] = 1;
	    }


	    if (3<=level_stdout){
	      for (mu=1; mu<=NUM1; mu++){
		printf("L0=%2d  M0= %2d mu=%2d  jun=%2d\n",L0,M0,mu,jun[mu]);
	      }
	    }

	    for (mu=1; mu<=NUM1; mu++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		al = tmp_index2[L0][Mul0][M0]; 
		B[al+1][mu+1] = C[al+1][jun[mu]];
	      }          
	    }
	  }

	  else if (Second_switch==1){

	    for (mu=1; mu<=NUM1; mu++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		al = tmp_index2[L0][Mul0][M0];
		if (mu==Mul0) B[al+1][mu+1] = 1.0;
	      }          
	    }
	  }
	}
      }
    }

    if (3<=level_stdout){
      printf("B\n");
      for (i1=1; i1<=NUM; i1++){
        for (j1=1; j1<=10; j1++){
          printf("%6.3f ",B[i1][j1]); 
        }
        printf("\n");
      }
    }


    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (M0=0; M0<=2*L0; M0++){
        tmp_index[L0][M0] = 1;
        Check_ko[L0][M0] = 10e+10;
      }
    }
    
    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
        for (M0=0; M0<=2*L0; M0++){
    
          al++;
    
          po = 0;
	  i = tmp_index[L0][M0] - 1;
	  do{
    	    i++;
     
            for (p=0; p<Spe_Specified_Num[wan][al]; p++){
              p0 = Spe_Trans_Orbital[wan][al][p];

	      if (0.001<fabs(B[p0+1][i])){
	        tmp_index[L0][M0] = i;
                Check_ko[L0][M0] = ko[i];
                Weight_ko[al] = ko[i];
	        po = 1;
	      }
              if (po==1) break;
	    }

	  }while(po==0);

          for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	    p0 = Spe_Trans_Orbital[wan][al][p];
	    i = tmp_index[L0][M0];
            if (Simple_InitCnt[L0]==0)
              CntCoes[Mc_AN][al][p] = B[p0+1][i];
	  }

          tmp_index[L0][M0] = tmp_index[L0][M0] + 1;

	}
      }
    }

    if (3<=level_stdout){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          printf("A Init_CntCoes Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
                  Mc_AN,Gc_AN,al,p,CntCoes[Mc_AN][al][p]);
        }
      }
    }

    /****************************************************
        Orthonormalization by the Gram-Schmidt method
    ****************************************************/

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

    if (3<=level_stdout){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          printf("B Init_CntCoes Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
                  Mc_AN,Gc_AN,al,p,CntCoes[Mc_AN][al][p]);
        }
      }
    }

    /****************************************************
                   Restricted contraction
    ****************************************************/

    if (RCnt_switch==1 || SICnt_switch==1){

      al = -1;
      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
        for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){

          for (p=0; p<Spe_Specified_Num[wan][al+1]; p++){
	    Tmp_CntCoes[p] = 0.0;
          }

          al0 = al; 
          for (M0=0; M0<=2*L0; M0++){
            al++;

            x = (Weight_ko[al] - LChemP)*Beta;
            if (x<=-30.0) x = -30.0;
            if (30.0<=x)  x = 30.0;
            FermiF = 1.0/(1.0 + exp(x));

            for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	      Tmp_CntCoes[p] = Tmp_CntCoes[p] + CntCoes[Mc_AN][al][p]*FermiF;
            }
          }
          al = al0;

          for (M0=0; M0<=2*L0; M0++){
            al++;
            for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	      CntCoes[Mc_AN][al][p] = Tmp_CntCoes[p];
            }
          }

	}
      }

      /****************************************************
          Orthonormalization by the Gram-Schmidt method
      ****************************************************/

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
	    tmp0 = 1.0/sqrt(sum);
	    tmp1 = sgn(CntCoes[Mc_AN][al0][pmax]);
            
	    for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	      CntCoes[Mc_AN][al0][p] = tmp0*tmp1*CntCoes[Mc_AN][al0][p];
	    }

	  }
	}
      }
    }

    if (3<=level_stdout){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          printf("C Init_CntCoes Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
                  Mc_AN,Gc_AN,al,p,CntCoes[Mc_AN][al][p]);
        }
      }
    }

    /****************************************************
                        free arrays
    ****************************************************/

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
	  printf("D Init_CntCoes Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
		 Mc_AN,Gc_AN,al,p,CntCoes[Mc_AN][al][p]);
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

