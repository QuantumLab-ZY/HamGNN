/**********************************************************************
  Embedded_GFM.c:

     Embedded_GFM.c is a subroutine to perform an embedded Green function 
     method to calculate density matrix. 

  Log of Embedded_GFM.c:

     15/Aug./2017  Released by T. Ozaki

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



static double EGFM_Col(char *mode,
		       int SCF_iter,
		       double *****Hks, double ****OLP0,
		       double *****CDM,
		       double *****EDM,
		       double Eele0[2], double Eele1[2]);



double Embedded_GFM(char *mode,
                      int SCF_iter,
                      double *****Hks,
                      double *****ImNL,
                      double ****OLP0,
                      double *****CDM,
                      double *****EDM,
                      double Eele0[2], double Eele1[2])
{
  double time0;

  /****************************************************
         collinear without spin-orbit coupling
  ****************************************************/

  if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==0 ){
    time0 = EGFM_Col(mode,SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1);
  }

  /****************************************************
   non-collinear with and without spin-orbit coupling
  ****************************************************/

  else if (SpinP_switch==3){
    /*
    time0 = DC_NonCol(mode,SCF_iter, Hks, ImNL, OLP0, CDM, EDM, Eele0, Eele1);
    */
  }

  return time0;
}







static double EGFM_Col(char *mode,
		       int SCF_iter,
		       double *****Hks, double ****OLP0,
		       double *****CDM,
		       double *****EDM,
		       double Eele0[2], double Eele1[2])
{
  static int firsttime=1;
  int Mc_AN,Gc_AN,i,Gi,wan,wanA,wanB,Anum;
  int size1,size2,num,NUM,NUM1,n2,Cwan,Hwan;
  int ih,ig,ian,j,kl,jg,jan,Bnum,m,n,spin;
  int l,i1,j1,P_min,m_size;
  int po,loopN,tno1,tno2,h_AN,Gh_AN;
  int MA_AN,GA_AN,GB_AN,tnoA,tnoB,k;
  double My_TZ,TZ,sum,FermiF,time0;
  double tmp1,tmp2;
  double My_Num_State,Num_State,x,Dnum;
  double TStime,TEtime;
  double My_Eele0[2],My_Eele1[2];
  double max_x=30.0;
  double ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double **S_DC,***H_DC,*ko,*M1;
  double **C;
  double ***EVal;
  double ******Residues;
  double ***PDOS_DC;
  int *MP,*Msize;
  double *tmp_array;
  double *tmp_array2;
  int *Snd_H_Size,*Rcv_H_Size;
  int *Snd_S_Size,*Rcv_S_Size;
  int numprocs,myid,ID,IDS,IDR,tag=999;
  double Stime_atom, Etime_atom;
  double OLP_eigen_cut=Threshold_OLP_Eigen;
  double stime, etime;
  double time1,time2,time3,time4,time5,time6;

  double sum1,sum2,sum3,sum4;
  int i1s, j1s;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  if (measure_time){
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
  }

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
  Msize = (int*)malloc(sizeof(int)*(Matomnum+1));

  EVal = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    EVal[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));


    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        FNAN[0] = 1;
        SNAN[0] = 0;
        n2 = 1;
        Msize[Mc_AN] = 1;
      }
      else{

        Gc_AN = M2G[Mc_AN];
        Anum = 1;
        for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
          Gi = natn[Gc_AN][i];
          wanA = WhatSpecies[Gi];
          Anum += Spe_Total_CNO[wanA];
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
  PrintMemory("Divide_Conquer: EVal",sizeof(double)*m_size,NULL);

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

      if (Mc_AN==0){
        Gc_AN = 0;
        FNAN[0] = 1;
        tno1 = 1;
        n2 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
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
  PrintMemory("Divide_Conquer: Residues",sizeof(double)*m_size,NULL);

  /****************************************************
    allocation of arrays:

    double PDOS[SpinP_switch+1]
               [Matomnum+1]
               [NUM]
  ****************************************************/

  m_size = 0;
  PDOS_DC = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    PDOS_DC[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0)  n2 = 1;
      else           n2 = Msize[Mc_AN] + 2;

      m_size += n2;
      PDOS_DC[spin][Mc_AN] = (double*)malloc(sizeof(double)*n2);
    }
  }

  if (firsttime)
  PrintMemory("Divide_Conquer: PDOS_DC",sizeof(double)*m_size,NULL);

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

  if (SCF_iter<=2){

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
  }

  /****************************************************
            find the total number of electrons 
  ****************************************************/

  My_TZ = 0.0;
  for (i=1; i<=Matomnum; i++){
    Gc_AN = M2G[i];
    wan = WhatSpecies[Gc_AN];
    My_TZ = My_TZ + Spe_Core_Charge[wan];
  }

  /* MPI, My_TZ */

  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&My_TZ, &TZ, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
      Setting of Hamiltonian and overlap matrices

         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

#pragma omp parallel shared(OLP_eigen_cut,List_YOUSO,Etime_atom,time_per_atom,time3,Residues,EVal,time2,time1,S12,level_stdout,SpinP_switch,Hks,OLP0,SCF_iter,RMI1,S_G2M,Spe_Total_CNO,natn,FNAN,SNAN,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,wan,Anum,i,j,MP,Gi,wanA,NUM,NUM1,n2,spin,S_DC,H_DC,ko,M1,C,ig,ian,ih,kl,jg,jan,Bnum,m,n,stime,P_min,l,i1,j1,etime,tmp1,tmp2,sum1,sum2,sum3,sum4,j1s,sum,tno1,h_AN,Gh_AN,wanB,tno2)
  { 

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /* allocation of arrays */

    MP = (int*)malloc(sizeof(int)*List_YOUSO[2]);

    /* start of the Mc_AN loop which is parallelized by OpenMP */

    for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){

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
	Anum += Spe_Total_CNO[wanA];
      }
      NUM = Anum - 1;
      n2 = NUM + 3;

      /***********************************************
       allocation of arrays:
     
       double S_DC[n2][n2];     
       double H_DC[SpinP_switch+1][n2][n2];     
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

      C = (double**)malloc(sizeof(double*)*n2);
      for (i=0; i<n2; i++){
	C[i] = (double*)malloc(sizeof(double)*n2);
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

	    if (SCF_iter<=2){
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  S_DC[Anum+m][Bnum+n] = OLP0[ih][kl][m][n];
		}
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

	    if (SCF_iter<=2){
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  S_DC[Anum+m][Bnum+n] = 0.0;
		}
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

      if (SCF_iter<=2){

	if (measure_time) dtime(&stime);

	Eigen_lapack(S_DC,ko,NUM,NUM);

	/***********************************************
              Searching of negative eigenvalues
	************************************************/

	P_min = 1;
	for (l=1; l<=NUM; l++){

	  if (ko[l]<OLP_eigen_cut){
	    P_min = l + 1;
	    if (3<=level_stdout){
	      printf("<DC>  Negative EV of OLP %2d %15.12f\n",l,ko[l]);
	    }
	  }
	}

	S12[Mc_AN][0][0] = P_min;

	for (l=1; l<P_min; l++)     M1[l] = 0.0;
	for (l=P_min; l<=NUM; l++)  M1[l] = 1.0/sqrt(ko[l]);

	for (i1=1; i1<=NUM; i1++){
	  for (j1=1; j1<=NUM; j1++){
	    S_DC[i1][j1]       = S_DC[i1][j1]*M1[j1];
	    S12[Mc_AN][i1][j1] = S_DC[i1][j1];
	  }
	}

	if (measure_time){
	  dtime(&etime);
	  time1 += etime - stime; 
	}

      }

      else{

	P_min = (int)S12[Mc_AN][0][0];

	for (i1=1; i1<=NUM; i1++){
	  for (j1=1; j1<=NUM; j1++){
	    S_DC[i1][j1] = S12[Mc_AN][i1][j1];
	  }
	}
      }

      /***********************************************
        transform Hamiltonian matrix
      ************************************************/

      for (spin=0; spin<=SpinP_switch; spin++){

	if (measure_time) dtime(&stime);

	/* transpose S */
	for (i1=1; i1<=NUM; i1++){
	  for (j1=i1+1; j1<=NUM; j1++){
	    tmp1 = S_DC[i1][j1];
	    tmp2 = S_DC[j1][i1];
	    S_DC[i1][j1] = tmp2;
	    S_DC[j1][i1] = tmp1;
	  }
	}

	/* H * U * M1 */

	for (j1=1; j1<=NUM-3; j1=j1+4){
	  for (i1=1; i1<=NUM; i1++){
	    sum1 = 0.0;
	    sum2 = 0.0;
	    sum3 = 0.0;
	    sum4 = 0.0;
	    for (l=1; l<=NUM; l++){
	      sum1 += H_DC[spin][i1][l]*S_DC[j1][l];
	      sum2 += H_DC[spin][i1][l]*S_DC[j1+1][l];
	      sum3 += H_DC[spin][i1][l]*S_DC[j1+2][l];
	      sum4 += H_DC[spin][i1][l]*S_DC[j1+3][l];
	    }
	    C[j1][i1] = sum1;
	    C[j1+1][i1] = sum2;
	    C[j1+2][i1] = sum3;
	    C[j1+3][i1] = sum4;
	  }
	}

	j1s =  NUM - NUM%4 + 1;
	for (j1=j1s; j1<=NUM; j1++){
	  for (i1=1; i1<=NUM; i1++){
	    sum = 0.0;
	    for (l=1; l<=NUM; l++){
	      sum += H_DC[spin][i1][l]*S_DC[j1][l];
	    }
	    C[j1][i1] = sum;
	  }
	}

	/* M1 * U^+ H * U * M1 */

	for (j1=1; j1<=NUM-3; j1=j1+4){
	  for (i1=1; i1<=NUM; i1++){
	    sum1 = 0.0;
	    sum2 = 0.0;
	    sum3 = 0.0;
	    sum4 = 0.0;
	    for (l=1; l<=NUM; l++){
	      sum1 += S_DC[i1][l]*C[j1  ][l];
	      sum2 += S_DC[i1][l]*C[j1+1][l];
	      sum3 += S_DC[i1][l]*C[j1+2][l];
	      sum4 += S_DC[i1][l]*C[j1+3][l];
	    }
	    H_DC[spin][j1  ][i1] = sum1;
	    H_DC[spin][j1+1][i1] = sum2;
	    H_DC[spin][j1+2][i1] = sum3;
	    H_DC[spin][j1+3][i1] = sum4;
	  }
	}
	j1s =  NUM - NUM%4 + 1;
	for (j1=j1s; j1<=NUM; j1++){
	  for (i1=1; i1<=NUM; i1++){
	    sum1 = 0.0;
	    for (l=1; l<=NUM; l++){
	      sum1 += S_DC[i1][l]*C[j1][l];
	    }
	    H_DC[spin][j1][i1] = sum1;
	  }
	}

	/* H_DC to C (transposition) */

	for (i1=P_min; i1<=NUM; i1++){
	  for (j1=P_min; j1<=NUM; j1++){
	    C[j1-(P_min-1)][i1-(P_min-1)] = H_DC[spin][i1][j1];
	  }
	}

	/***********************************************
         diagonalize the trasformed Hamiltonian matrix
	************************************************/

	NUM1 = NUM - (P_min - 1);
	Eigen_lapack(C,ko,NUM1,NUM1);

	/*
        for (i=1; i<=NUM1; i++){
          printf("DCQ1 i=%2d ko=%16.12f\n",i,ko[i]);
        }
        MPI_Finalize(); 
        exit(0);
	*/

	/* C to H (transposition) */

	for (i1=1; i1<=NUM; i1++){
	  for (j1=1; j1<=NUM1; j1++){
	    H_DC[spin][j1][i1] = C[i1][j1];
	  }
	}

	/***********************************************
         transformation to the original eigen vectors.
                        NOTE 244P
	***********************************************/

	/* transpose */

	for (i1=1; i1<=NUM; i1++){
	  for (j1=i1+1; j1<=NUM; j1++){
	    tmp1 = S_DC[i1][j1];
	    tmp2 = S_DC[j1][i1];
	    S_DC[i1][j1] = tmp2;
	    S_DC[j1][i1] = tmp1;
	  }
	}

	for (j1=1; j1<=NUM1; j1++){
	  for (l=NUM; P_min<=l; l--){
	    H_DC[spin][j1][l] = H_DC[spin][j1][l-(P_min-1)];
	  }
	}

	for (j1=1; j1<=NUM-3; j1=j1+4){
	  for (i1=1; i1<=NUM; i1++){
	    sum1 = 0.0;
	    sum2 = 0.0;
	    sum3 = 0.0;
	    sum4 = 0.0;
	    for (l=P_min; l<=NUM; l++){
	      sum1 += S_DC[i1][l]*H_DC[spin][j1  ][l];
	      sum2 += S_DC[i1][l]*H_DC[spin][j1+1][l];
	      sum3 += S_DC[i1][l]*H_DC[spin][j1+2][l];
	      sum4 += S_DC[i1][l]*H_DC[spin][j1+3][l];
	    }
	    C[i1][j1  ] = sum1;
	    C[i1][j1+1] = sum2;
	    C[i1][j1+2] = sum3;
	    C[i1][j1+3] = sum4;
	  }
	}
	j1s =  NUM - NUM%4 + 1;
	for (j1=j1s; j1<=NUM; j1++){
	  for (i1=1; i1<=NUM; i1++){
	    sum = 0.0;
	    for (l=P_min; l<=NUM; l++){
	      sum += S_DC[i1][l]*H_DC[spin][j1][l];
	    }
	    C[i1][j1] = sum;
	  }
	}

	if (measure_time){
	  dtime(&etime);
	  time2 += etime - stime;
	}

	/*
	  for (i=1; i<=NUM1; i++){
	  printf("ko %15.12f 1.0\n",ko[i]);
	  }

	  for (i=1; i<=NUM; i++){
	  for (j=1; j<=NUM; j++){
	  printf("%15.12f ",C[i][j]); 
	  }
	  printf("\n");
	  }
	*/

	/*
	  MPI_Finalize();
	  exit(0);
	*/

	/***********************************************
           store eigenvalues and residues of poles
	***********************************************/

	if (measure_time) dtime(&stime);

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

	if (measure_time){
	  dtime(&etime);
	  time3 += etime - stime;
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
	free(C[i]);
      }
      free(C);

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* end of Mc_AN */

    /* freeing of arrays */

    free(MP);

  } /* #pragma omp parallel */

  if ( strcasecmp(mode,"scf")==0 ){

    /****************************************************
              calculate projected DOS
    ****************************************************/

    if (measure_time) dtime(&stime);

#pragma omp parallel shared(FNAN,time_per_atom,Residues,OLP0,natn,PDOS_DC,Msize,Spe_Total_CNO,WhatSpecies,M2G,Matomnum,SpinP_switch) private(OMPID,Nthrds,Nprocs,Mc_AN,spin,Stime_atom,Etime_atom,Gc_AN,wanA,tno1,i1,i,h_AN,Gh_AN,wanB,tno2,j,tmp1)
    {

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){
	for (spin=0; spin<=SpinP_switch; spin++){

	  dtime(&Stime_atom);

	  Gc_AN = M2G[Mc_AN];
	  wanA = WhatSpecies[Gc_AN];
	  tno1 = Spe_Total_CNO[wanA];

	  for (i1=0; i1<Msize[Mc_AN]; i1++){
	    PDOS_DC[spin][Mc_AN][i1] = 0.0;
	  }

	  for (i=0; i<tno1; i++){
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      wanB = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_CNO[wanB];
	      for (j=0; j<tno2; j++){

		tmp1 = OLP0[Mc_AN][h_AN][i][j];
		for (i1=0; i1<Msize[Mc_AN]; i1++){
		  PDOS_DC[spin][Mc_AN][i1] += Residues[spin][Mc_AN][h_AN][i][j][i1]*tmp1;
		}
	      }            
	    }        
	  }

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	  /*
	    for (i1=0; i1<Msize[Mc_AN]; i1++){
	    printf("%4d  %18.15f\n",i1,PDOS_DC[spin][Mc_AN][i1]);
	    }
	  */

	}
      }

    } /* #pragma omp parallel */

    if (measure_time){
      dtime(&etime);
      time4 += etime - stime;
    }

    /****************************************************
                   find chemical potential
    ****************************************************/

    if (measure_time) dtime(&stime);

    po = 0;
    loopN = 0;

    ChemP_MAX = 30.0;  
    ChemP_MIN =-30.0;
    if      (SpinP_switch==0) spin_degeneracy = 2.0;
    else if (SpinP_switch==1) spin_degeneracy = 1.0;

    do {
      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);

      My_Num_State = 0.0;
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  dtime(&Stime_atom);

	  Gc_AN = M2G[Mc_AN];

	  for (i=0; i<Msize[Mc_AN]; i++){
	    x = (EVal[spin][Mc_AN][i] - ChemP)*Beta;
	    if (x<=-max_x) x = -max_x;
	    if (max_x<=x)  x = max_x;
	    FermiF = 1.0/(1.0 + exp(x));
	    My_Num_State += spin_degeneracy*FermiF*PDOS_DC[spin][Mc_AN][i];
	  }

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
	}
      }

      /* MPI, My_Num_State */

      MPI_Barrier(mpi_comm_level1);
      MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-13) po = 1;


      if (myid==Host_ID && 2<=level_stdout){
	printf("ChemP=%15.12f TZ=%15.12f Num_state=%15.12f\n",ChemP,TZ,Num_State); 
      }

      loopN++;
    }
    while (po==0 && loopN<1000); 

    /****************************************************
        eigenenergy by summing up eigenvalues
    ****************************************************/

    My_Eele0[0] = 0.0;
    My_Eele0[1] = 0.0;
    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	for (i=0; i<Msize[Mc_AN]; i++){
	  x = (EVal[spin][Mc_AN][i] - ChemP)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = 1.0/(1.0 + exp(x));
	  My_Eele0[spin] += FermiF*EVal[spin][Mc_AN][i]*PDOS_DC[spin][Mc_AN][i];
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }
    }

    /* MPI, My_Eele0 */
    for (spin=0; spin<=SpinP_switch; spin++){
      MPI_Barrier(mpi_comm_level1);
      MPI_Allreduce(&My_Eele0[spin], &Eele0[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch==0){
      Eele0[1] = Eele0[0];
    }

    if (measure_time){
      dtime(&etime);
      time5 += etime - stime;
    }

    /****************************************************
       calculate density and energy density matrices
    ****************************************************/

    if (measure_time) dtime(&stime);

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];
	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      CDM[spin][Mc_AN][h_AN][i][j] = 0.0;
	      EDM[spin][Mc_AN][h_AN][i][j] = 0.0;
	    }
	  }
	}
      }
    }

#pragma omp parallel shared(FNAN,time_per_atom,EDM,CDM,Residues,natn,max_x,Beta,ChemP,EVal,Msize,Spe_Total_CNO,WhatSpecies,M2G,SpinP_switch,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,spin,Stime_atom,Gc_AN,wanA,tno1,i1,x,FermiF,h_AN,Gh_AN,wanB,tno2,i,j,tmp1,Etime_atom)
    {

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){
	for (spin=0; spin<=SpinP_switch; spin++){

	  dtime(&Stime_atom);

	  Gc_AN = M2G[Mc_AN];
	  wanA = WhatSpecies[Gc_AN];
	  tno1 = Spe_Total_CNO[wanA];

	  for (i1=0; i1<Msize[Mc_AN]; i1++){
	    x = (EVal[spin][Mc_AN][i1] - ChemP)*Beta;
	    if (x<=-max_x) x = -max_x;
	    if (max_x<=x)  x = max_x;
	    FermiF = 1.0/(1.0 + exp(x));

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      wanB = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_CNO[wanB];
	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){
		  tmp1 = FermiF*Residues[spin][Mc_AN][h_AN][i][j][i1];
		  CDM[spin][Mc_AN][h_AN][i][j] += tmp1;
		  EDM[spin][Mc_AN][h_AN][i][j] += tmp1*EVal[spin][Mc_AN][i1];
		}
	      }
	    }
	  }

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
	}
      }

    } /* #pragma omp parallel */

    /****************************************************
                     bond energies
    ****************************************************/

    My_Eele1[0] = 0.0;
    My_Eele1[1] = 0.0;
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (j=0; j<=FNAN[GA_AN]; j++){
	GB_AN = natn[GA_AN][j];  
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];

	for (k=0; k<tnoA; k++){
	  for (l=0; l<tnoB; l++){
	    for (spin=0; spin<=SpinP_switch; spin++){
	      My_Eele1[spin] += CDM[spin][MA_AN][j][k][l]*Hks[spin][MA_AN][j][k][l];
	    }
	  }
	}
  
      }
    }

    /* MPI, My_Eele1 */
    MPI_Barrier(mpi_comm_level1);
    for (spin=0; spin<=SpinP_switch; spin++){
      MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE,
		    MPI_SUM, mpi_comm_level1);
    }

    if (SpinP_switch==0){
      Eele1[1] = Eele1[0];
    }

    if (3<=level_stdout && myid==Host_ID){
      printf("Eele00=%15.12f Eele01=%15.12f\n",Eele0[0],Eele0[1]);
      printf("Eele10=%15.12f Eele11=%15.12f\n",Eele1[0],Eele1[1]);
    }

    if (measure_time){
      dtime(&etime);
      time6 += etime - stime;
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  else if ( strcasecmp(mode,"dos")==0 ){
    Save_DOS_Col(Residues,OLP0,EVal,Msize);
  }

  if (measure_time){
    printf("Divide_Conquer myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f time6=%7.3f\n",
            myid,time1,time2,time3,time4,time5,time6);fflush(stdout); 
  }

  /****************************************************
    freeing of arrays:

  ****************************************************/

  free(Snd_H_Size);
  free(Rcv_H_Size);

  free(Snd_S_Size);
  free(Rcv_S_Size);

  free(Msize);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(EVal[spin][Mc_AN]);
    }
    free(EVal[spin]);
  }
  free(EVal);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        FNAN[0] = 1;
        tno1 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
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

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(PDOS_DC[spin][Mc_AN]);
    }
    free(PDOS_DC[spin]);
  }
  free(PDOS_DC);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  /* for PrintMemory */
  firsttime=0;

  return time0;
}
