/**********************************************************************
  Divide_Conquer.c:

     Divide_Conquer.c is a subroutine to perform a divide and conquer
     method for eigenvalue problem

  Log of Divide_Conquer.c:

     10/Dec/2003  Released by T.Ozaki

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



static double DC_Col(char *mode,
		     int SCF_iter,
		     double *****Hks, double ****OLP0,
		     double *****CDM,
		     double *****EDM,
		     double Eele0[2], double Eele1[2]);

static double DC_NonCol(char *mode,
			int SCF_iter,
			double *****Hks,
			double *****ImNL,
			double ****OLP0,
			double *****CDM,
			double *****EDM,
			double Eele0[2],
			double Eele1[2]);

static void Save_DOS_Col(double ******Residues, double ****OLP0, double ***EVal, int *Msize);
static void Save_DOS_NonCol(dcomplex ******Residues, double ****OLP0, double **EVal, int *Msize);


double Divide_Conquer(char *mode,
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
    time0 = DC_Col(mode,SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1);
  }

  /****************************************************
   non-collinear with and without spin-orbit coupling
  ****************************************************/

  else if (SpinP_switch==3){
    time0 = DC_NonCol(mode,SCF_iter, Hks, ImNL, OLP0, CDM, EDM, Eele0, Eele1);
  }

  return time0;
}






#pragma optimization_level 2
static double DC_Col(char *mode,
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







#pragma optimization_level 2
static double DC_NonCol(char *mode,
			int SCF_iter,
			double *****Hks,
			double *****ImNL,
			double ****OLP0,
			double *****CDM,
			double *****EDM,
			double Eele0[2],
			double Eele1[2])
{
  static int firsttime=1;
  int Mc_AN,Gc_AN,i,Gi,wan,wanA,wanB,Anum;
  int size1,size2,num,NUM,NUM1,n2,Cwan,Hwan;
  int ih,ig,ian,j,kl,jg,jan,Bnum,m,n,spin,so;
  int l,i1,j1,P_min,m_size;
  int po,loopN,tno1,tno2,h_AN,Gh_AN;
  int k,ii1,jj1,k1,l1,ompi;
  int MA_AN,GA_AN,GB_AN,tnoA,tnoB;
  double My_TZ,TZ,sum,FermiF,time0;
  double My_Num_State,Num_State,x,Dnum;
  double sum_r,sum_i,tmp1,tmp2;
  double sum1_r,sum1_i,sum2_r,sum2_i;
  double TStime,TEtime;
  double My_Eele0[2],My_Eele1[2];
  double max_x=50.0;
  double ChemP_MAX,ChemP_MIN,spin_degeneracy;
  double **S_DC,*ko,*M1;
  dcomplex **C,**H_DC;
  double **EVal;
  dcomplex ******Residues;
  double **PDOS_DC;
  int *MP,*Msize;
  double *tmp_array;
  double *tmp_array2;
  double *omp_tmp;
  int *Snd_H_Size,*Rcv_H_Size;
  int *Snd_iHNL_Size,*Rcv_iHNL_Size;
  int *Snd_S_Size,*Rcv_S_Size;
  int numprocs,myid,ID,IDS,IDR,tag=999;
  double Stime_atom, Etime_atom;
  double OLP_eigen_cut=Threshold_OLP_Eigen;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs,Nthrds0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  /****************************************************
    allocation of arrays:

    int MP[List_YOUSO[2]];
    int Msize[Matomnum+1];
    double EVal[Matomnum+1][n2];
  ****************************************************/

  Snd_H_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_H_Size = (int*)malloc(sizeof(int)*numprocs);
  Snd_iHNL_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_iHNL_Size = (int*)malloc(sizeof(int)*numprocs);
  Snd_S_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_S_Size = (int*)malloc(sizeof(int)*numprocs);

  m_size = 0;
  Msize = (int*)malloc(sizeof(int)*(Matomnum+1));

  EVal = (double**)malloc(sizeof(double*)*(Matomnum+1));

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
  PrintMemory("Divide_Conquer: EVal",sizeof(double)*m_size,NULL);

  if (2<=level_stdout){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        printf("<DC> myid=%i Mc_AN=%2d Gc_AN=%2d Msize=%3d\n",
        myid,Mc_AN,M2G[Mc_AN],Msize[Mc_AN]);
    }
  }

  /****************************************************
    allocation of arrays:

    dcomplex Residues[3]
                     [Matomnum+1]
                     [FNAN[Gc_AN]+1]
                     [Spe_Total_CNO[Gc_AN]] 
                     [Spe_Total_CNO[Gh_AN]] 
                     [n2]
     To reduce the memory size, the size of NUM2
     should be found in the loop.  
  ****************************************************/

  m_size = 0;

  Residues = (dcomplex******)malloc(sizeof(dcomplex*****)*3);
  for (spin=0; spin<3; spin++){
    Residues[spin] = (dcomplex*****)malloc(sizeof(dcomplex****)*(Matomnum+1));
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
	n2 = 2*Msize[Mc_AN] + 2;
      }

      Residues[spin][Mc_AN] = (dcomplex****)malloc(sizeof(dcomplex***)*(FNAN[Gc_AN]+1));

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno2 = 1;
	}
	else {
	  Gh_AN = natn[Gc_AN][h_AN];
	  wanB = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[wanB];
	}

	Residues[spin][Mc_AN][h_AN] = (dcomplex***)malloc(sizeof(dcomplex**)*tno1);
	for (i=0; i<tno1; i++){
	  Residues[spin][Mc_AN][h_AN][i] = (dcomplex**)malloc(sizeof(dcomplex*)*tno2);
	  for (j=0; j<tno2; j++){
	    Residues[spin][Mc_AN][h_AN][i][j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
	  }
	}

	m_size += tno1*tno2*n2;
      }
    }
  }

  if (firsttime)
  PrintMemory("Divide_Conquer: Residues",sizeof(dcomplex)*m_size,NULL);

  /****************************************************
    allocation of arrays:

    double PDOS_DC[Matomnum+1]
                  [n2]
  ****************************************************/

  m_size = 0;

  PDOS_DC = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0) n2 = 1;
    else          n2 = 2*Msize[Mc_AN] + 2;

    m_size += n2;
    PDOS_DC[Mc_AN] = (double*)malloc(sizeof(double)*n2);
  }

  if (firsttime)
  PrintMemory("Divide_Conquer: PDOS_DC",sizeof(double)*m_size,NULL);

  /* allocation of a temporal array for OpenMP */

#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  omp_tmp = (double*)malloc(sizeof(double)*Nthrds0);

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

   ImNL
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  /* spin-orbit coupling or LDA+U */  

  if (SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
      || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){
	tag = 999;

	/* find data size to send block data */
	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	  size1 = 0;
	  for (so=0; so<List_YOUSO[5]; so++){
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
 
	  Snd_iHNL_Size[IDS] = size1;
	  MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
	}
	else{
	  Snd_iHNL_Size[IDS] = 0;
	}

	/* receiving of size of data */

	if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
	  MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	  Rcv_iHNL_Size[IDR] = size2;
	}
	else{
	  Rcv_iHNL_Size[IDR] = 0;
	}

	if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);

      }
      else{
	Snd_iHNL_Size[IDS] = 0;
	Rcv_iHNL_Size[IDR] = 0;
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

	  size1 = Snd_iHNL_Size[IDS];

	  /* allocation of array */

	  tmp_array = (double*)malloc(sizeof(double)*size1);

	  /* multidimentional array to vector array */

	  num = 0;
	  for (so=0; so<List_YOUSO[5]; so++){
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
		    tmp_array[num] = ImNL[so][Mc_AN][h_AN][i][j];
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

	  size2 = Rcv_iHNL_Size[IDR];
        
	  /* allocation of array */
	  tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
	  MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	  num = 0;
	  for (so=0; so<List_YOUSO[5]; so++){
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
		    ImNL[so][Mc_AN][h_AN][i][j] = tmp_array2[num];
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

#pragma omp parallel shared(List_YOUSO,time_per_atom,Residues,EVal,S12,OLP_eigen_cut,ImNL,Hks,Zeeman_NCO_switch,Zeeman_NCS_switch,Constraint_NCS_switch,Hub_U_switch,SO_switch,OLP0,SCF_iter,RMI1,S_G2M,Spe_Total_CNO,natn,SNAN,FNAN,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Etime_atom,Stime_atom,Gc_AN,wan,Anum,i,MP,Gi,wanA,NUM,n2,S_DC,H_DC,ko,M1,C,ig,ian,ih,j,kl,jg,jan,Bnum,m,n,P_min,l,i1,j1,tmp1,tmp2,k,jj1,k1,sum_r,sum_i,l1,sum1_r,sum1_i,sum2_r,sum2_i,ii1,NUM1,tno1,h_AN,Gh_AN,wanB,tno2)
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

	    if (SCF_iter<=2){
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  S_DC[Anum+m][Bnum+n] = OLP0[ih][kl][m][n];
		}
	      }
	    }

	    /* non-spin-orbit coupling and non-LDA+U */  
	    if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
		&& Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){
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

	    if (SCF_iter<=2){
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  S_DC[Anum+m][Bnum+n] = 0.0;
		}
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
		H_DC[Bnum+n+NUM][Anum+m    ].r = 0.0;
		H_DC[Bnum+n+NUM][Anum+m    ].i = 0.0;
	      }
	    }

	  }
	}
      }

      /*
    if (Gc_AN==1){

    printf("H_DC.r\n");
    for (i=1; i<=NUM*2; i++){
      for (j=1; j<=NUM*2; j++){
        printf("H_DC.r i=%2d j=%2d %7.4f\n",i,j,H_DC[i][j].r);
      }
    }

    printf("H_DC.i\n");
    for (i=1; i<=NUM*2; i++){
      for (j=1; j<=NUM*2; j++){
        printf("H_DC.i i=%2d j=%2d %7.4f\n",i,j,H_DC[i][j].i);
      }
    }
    }
      */

    /*
    EigenBand_lapack(H_DC, ko, NUM*2, NUM*2, 1);

    if (Gc_AN==1){
      for (l=1; l<=NUM*2; l++){
	printf("ABC0 myid=%2d l=%2d ko=%15.12f\n",myid,l,ko[l]);fflush(stdout); 
      }
    }
    */

      /****************************************************
       Solve the generalized eigenvalue problem
       HC = SCE

       1) diagonalize S
       2) search negative eigenvalues of S  
      ****************************************************/

      if (SCF_iter<=2){

	Eigen_lapack(S_DC, ko, NUM, NUM);

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
      }

      else{

	P_min = (int)S12[Mc_AN][0][0];

	for (i1=1; i1<=NUM; i1++){
	  for (j1=1; j1<=NUM; j1++){
	    S_DC[i1][j1] = S12[Mc_AN][i1][j1];
	  }
	}
      }


      /*
      printf("S_DC Gc_AN=%2d\n",Gc_AN);

    if (Gc_AN==1){

    printf("S_DC\n");


    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
        printf("S_DC i=%2d j=%2d %7.4f\n",i,j,S_DC[i][j]);
      }
    }

    }
      */

      /***********************************************
             transform Hamiltonian matrix
      ************************************************/

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

      for (j1=P_min; j1<=NUM; j1++){
	for (k=0; k<=1; k++){
	  jj1 = 2*j1 - P_min + k;
	  k1 = k*NUM;
        
	  for (i1=1; i1<=2*NUM; i1++){

	    sum_r = 0.0;
	    sum_i = 0.0;

	    for (l=1; l<=NUM; l++){
	      l1 = k1 + l;
	      sum_r += H_DC[i1][l1].r*S_DC[j1][l];
	      sum_i += H_DC[i1][l1].i*S_DC[j1][l];
	    }

	    C[jj1][i1].r = sum_r;
	    C[jj1][i1].i = sum_i;
	  }
	}
      }

      /*
    if (Gc_AN==1){
    printf("C.r\n");
    for (i=1; i<=NUM*2; i++){
      for (j=1; j<=NUM*2; j++){
        printf("ABC1 C.r i=%2d j=%2d %7.4f\n",i,j,C[i][j].r);
      }
    }

    printf("C.i\n");
    for (i=1; i<=NUM*2; i++){
      for (j=1; j<=NUM*2; j++){
        printf("ABC1 C.i i=%2d j=%2d %7.4f\n",i,j,C[i][j].i);
      }
    }
    }
      */

      /* M1 * U^+ H * U * M1 */

      for (j1=1; j1<=2*NUM; j1++){
	for (i1=P_min; i1<=NUM; i1++){

	  sum1_r = 0.0;
	  sum1_i = 0.0;
	  sum2_r = 0.0;
	  sum2_i = 0.0;

	  for (l=1; l<=NUM; l++){
	    sum1_r += S_DC[i1][l]*C[j1][l].r;
	    sum1_i += S_DC[i1][l]*C[j1][l].i;
	  }

	  for (l=NUM+1; l<=2*NUM; l++){
	    l1 = l - NUM; 
	    sum2_r += S_DC[i1][l1]*C[j1][l].r;
	    sum2_i += S_DC[i1][l1]*C[j1][l].i;
	  }

	  ii1 = 2*i1 - P_min;
	  H_DC[j1][ii1].r = sum1_r;
	  H_DC[j1][ii1].i = sum1_i;

	  ii1 = 2*i1 - P_min + 1;
	  H_DC[j1][ii1].r = sum2_r;
	  H_DC[j1][ii1].i = sum2_i;
	}
      }

      /* H to C (transposition) */

      for (i1=P_min; i1<=2*NUM; i1++){
	for (j1=P_min; j1<=2*NUM; j1++){
	  C[j1-(P_min-1)][i1-(P_min-1)].r = H_DC[i1][j1].r;
	  C[j1-(P_min-1)][i1-(P_min-1)].i = H_DC[i1][j1].i;
	}
      }

    if (Gc_AN==1){

      /*
    printf("C.r\n");
    for (i=1; i<=NUM*2; i++){
      for (j=1; j<=NUM*2; j++){
        printf("ABC2 C.r i=%2d j=%2d %7.4f\n",i,j,C[i][j].r);
      }
    }

    printf("C.i\n");
    for (i=1; i<=NUM*2; i++){
      for (j=1; j<=NUM*2; j++){
        printf("ABC2 C.i i=%2d j=%2d %7.4f\n",i,j,C[i][j].i);
      }
    }
      */

      /*
    printf("C.r\n");
    for (i=1; i<=NUM*2; i++){
      for (j=1; j<=NUM*2; j++){
        printf("%7.4f",C[i][j].r);
      }
      printf("\n");
    }

    printf("C.i\n");
    for (i=1; i<=NUM*2; i++){
      for (j=1; j<=NUM*2; j++){
        printf("%7.4f",C[i][j].i);
      }
      printf("\n");
    }
      */


    }

      /***********************************************
       diagonalize the trasformed Hamiltonian matrix
      ************************************************/

      NUM1 = 2*NUM - (P_min - 1);
      EigenBand_lapack(C, ko, NUM1, NUM1, 1);

      /*
    if (Gc_AN==1){
      for (l=1; l<=NUM1; l++){
	printf("ABC10 myid=%2d l=%2d ko=%15.12f\n",myid,l,ko[l]);fflush(stdout); 
      }
    }

    MPI_Finalize();
    exit(0);
      */

      /* C to H (transposition) */

      for (i1=1; i1<=NUM1; i1++){
	for (j1=1; j1<=NUM1; j1++){
	  H_DC[j1][i1].r = C[i1][j1].r;
	  H_DC[j1][i1].i = C[i1][j1].i;
	}
      }

      /***********************************************
      transformation to the original eigen vectors.
      NOTE 244P    C = U * lambda^{-1/2} * D
      ***********************************************/

      /* transpose S */

      for (i1=1; i1<=NUM; i1++){
	for (j1=i1+1; j1<=NUM; j1++){
	  tmp1 = S_DC[i1][j1];
	  tmp2 = S_DC[j1][i1];
	  S_DC[i1][j1] = tmp2;
	  S_DC[j1][i1] = tmp1;
	}
      }

      for (j1=1; j1<=NUM1; j1++){
	for (k=0; k<=1; k++){
	  for (i1=1; i1<=NUM; i1++){
	    sum_r = 0.0;
	    sum_i = 0.0;
	    for (l=P_min; l<=NUM; l++){
	      sum_r += S_DC[i1][l]*H_DC[j1][2*(l-P_min)+1+k].r;
	      sum_i += S_DC[i1][l]*H_DC[j1][2*(l-P_min)+1+k].i;
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
	      Residues[0][Mc_AN][h_AN][i][j][i1-1].r = C[1+i    ][i1].r*C[Bnum+j    ][i1].r
		                                      +C[1+i    ][i1].i*C[Bnum+j    ][i1].i;

	      /* Re22 */
	      Residues[1][Mc_AN][h_AN][i][j][i1-1].r = C[1+i+NUM][i1].r*C[Bnum+j+NUM][i1].r
		                                      +C[1+i+NUM][i1].i*C[Bnum+j+NUM][i1].i;

	      /* Re12 */
	      Residues[2][Mc_AN][h_AN][i][j][i1-1].r = C[1+i    ][i1].r*C[Bnum+j+NUM][i1].r
		                                      +C[1+i    ][i1].i*C[Bnum+j+NUM][i1].i;

	      /* Im12 */
	      Residues[2][Mc_AN][h_AN][i][j][i1-1].i = C[1+i    ][i1].r*C[Bnum+j+NUM][i1].i 
		                                      -C[1+i    ][i1].i*C[Bnum+j+NUM][i1].r; 

	      /* spin-orbit coupling or LDA+U */
	      if ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
		   || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

		/* Im11 */
		Residues[0][Mc_AN][h_AN][i][j][i1-1].i = C[1+i    ][i1].r*C[Bnum+j    ][i1].i
	 	                                        -C[1+i    ][i1].i*C[Bnum+j    ][i1].r;

		/*
                printf("VVV1 Mc_AN=%2d h_AN=%2d i=%2d j=%2d i1=%2d %15.10f\n",
                        Mc_AN,h_AN,i,j,i1,Residues[0][Mc_AN][h_AN][i][j][i1-1].i);
		*/

		/* Im22 */
		Residues[1][Mc_AN][h_AN][i][j][i1-1].i = C[1+i+NUM][i1].r*C[Bnum+j+NUM][i1].i
		                                        -C[1+i+NUM][i1].i*C[Bnum+j+NUM][i1].r;
	      } 

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

      /*
	for (i1=1; i1<=NUM1; i1++){
	printf("Mc_AN=%2d i1=%2d EVal=%15.12f\n",Mc_AN,i1,EVal[Mc_AN][i1-1]);
	}
      */

    } /* end of Mc_AN */

    /* freeing of arrays */

    free(MP);

  } /* #pragma omp parallel */

  if ( strcasecmp(mode,"scf")==0 ){

    /****************************************************
              calculate projected DOS
    ****************************************************/

#pragma omp parallel shared(time_per_atom,Residues,OLP0,natn,PDOS_DC,Msize,Spe_Total_CNO,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,wanA,tno1,i1,i,h_AN,Gh_AN,wanB,tno2,j,tmp1)
    {

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

	for (i1=0; i1<2*Msize[Mc_AN]; i1++){
	  PDOS_DC[Mc_AN][i1] = 0.0;
	}

	for (i=0; i<tno1; i++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wanB = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wanB];
	    for (j=0; j<tno2; j++){

	      tmp1 = OLP0[Mc_AN][h_AN][i][j];
	      for (i1=0; i1<2*Msize[Mc_AN]; i1++){
		PDOS_DC[Mc_AN][i1] += ( Residues[0][Mc_AN][h_AN][i][j][i1].r
			 	      + Residues[1][Mc_AN][h_AN][i][j][i1].r )*tmp1;
	      }

	    }            
	  }        
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }

    } /* #pragma omp parallel */

    /****************************************************
    find chemical potential
    ****************************************************/

    po = 0;
    loopN = 0;

    ChemP_MAX = 15.0;  
    ChemP_MIN =-15.0;

    do {
      ChemP = 0.50*(ChemP_MAX + ChemP_MIN);

#pragma omp parallel shared(omp_tmp,max_x,ChemP,Beta,EVal,Msize,Matomnum,M2G) private(OMPID,Nthrds,Nprocs,Mc_AN,Gc_AN,i,x,FermiF)
      {

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	omp_tmp[OMPID] = 0.0;

	for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){

	  Gc_AN = M2G[Mc_AN];

	  for (i=0; i<2*Msize[Mc_AN]; i++){
	    x = (EVal[Mc_AN][i] - ChemP)*Beta;
	    if (x<=-max_x) x = -max_x;
	    if (max_x<=x)  x = max_x;
	    FermiF = 1.0/(1.0 + exp(x));
	    omp_tmp[OMPID] += FermiF*PDOS_DC[Mc_AN][i];
	  }

	}

      } /* #pragma omp parallel */

      My_Num_State = 0.0;
      for (ompi=0; ompi<Nthrds0; ompi++){
	My_Num_State += omp_tmp[ompi];
      }

      /* MPI, My_Num_State */

      MPI_Barrier(mpi_comm_level1);
      MPI_Allreduce(&My_Num_State, &Num_State, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      Dnum = (TZ - Num_State) - system_charge;
      if (0.0<=Dnum) ChemP_MIN = ChemP;
      else           ChemP_MAX = ChemP;
      if (fabs(Dnum)<1.0e-11) po = 1;

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

#pragma omp parallel shared(PDOS_DC,Beta,ChemP,max_x,EVal,Msize,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Gc_AN,i,x,FermiF)
    {

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      omp_tmp[OMPID] = 0.0;

      for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){

	Gc_AN = M2G[Mc_AN];
	for (i=0; i<2*Msize[Mc_AN]; i++){
	  x = (EVal[Mc_AN][i] - ChemP)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = 1.0/(1.0 + exp(x));
	  omp_tmp[OMPID] += FermiF*EVal[Mc_AN][i]*PDOS_DC[Mc_AN][i];
	}
      }

    } /* #pragma omp parallel */

    for (ompi=0; ompi<Nthrds0; ompi++){
      My_Eele0[0] += omp_tmp[ompi];
    }

    /* MPI, My_Eele0 */

    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_Eele0[0], &Eele0[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_Eele0[1], &Eele0[1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /****************************************************
      calculate density and energy density matrices

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

	  /* spin-orbit coupling or LDA+U */
	  if ( (SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
		|| Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) && spin==0 ){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		iDM[0][0][Mc_AN][h_AN][i][j] = 0.0;
		iDM[0][1][Mc_AN][h_AN][i][j] = 0.0;
	      }
	    }
	  }

	}
      }
    }

#pragma omp parallel shared(iDM,Zeeman_NCO_switch,Zeeman_NCS_switch,Constraint_NCS_switch,Hub_U_switch,SO_switch,time_per_atom,EDM,CDM,Residues,natn,max_x,Beta,ChemP,EVal,Msize,Spe_Total_CNO,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,wanA,tno1,i1,x,FermiF,h_AN,Gh_AN,wanB,tno2,i,j,tmp1,Etime_atom)
    {

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[wanA];

	for (i1=0; i1<2*Msize[Mc_AN]; i1++){
	  x = (EVal[Mc_AN][i1] - ChemP)*Beta;
	  if (x<=-max_x) x = -max_x;
	  if (max_x<=x)  x = max_x;
	  FermiF = 1.0/(1.0 + exp(x));

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wanB = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[wanB];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){

		tmp1 = FermiF*Residues[0][Mc_AN][h_AN][i][j][i1].r;

		CDM[0][Mc_AN][h_AN][i][j] += tmp1;
		EDM[0][Mc_AN][h_AN][i][j] += tmp1*EVal[Mc_AN][i1];

		tmp1 = FermiF*Residues[1][Mc_AN][h_AN][i][j][i1].r;

		CDM[1][Mc_AN][h_AN][i][j] += tmp1;
		EDM[1][Mc_AN][h_AN][i][j] += tmp1*EVal[Mc_AN][i1];

		tmp1 = FermiF*Residues[2][Mc_AN][h_AN][i][j][i1].r;

		CDM[2][Mc_AN][h_AN][i][j] += tmp1;
		EDM[2][Mc_AN][h_AN][i][j] += tmp1*EVal[Mc_AN][i1];

		tmp1 = FermiF*Residues[2][Mc_AN][h_AN][i][j][i1].i; 

		CDM[3][Mc_AN][h_AN][i][j] += tmp1;
		EDM[3][Mc_AN][h_AN][i][j] += tmp1*EVal[Mc_AN][i1];

		/* spin-orbit coupling or LDA+U */
		if (SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
		    || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){
		  iDM[0][0][Mc_AN][h_AN][i][j] += FermiF*Residues[0][Mc_AN][h_AN][i][j][i1].i;
		  iDM[0][1][Mc_AN][h_AN][i][j] += FermiF*Residues[1][Mc_AN][h_AN][i][j][i1].i;
		}

	      }
	    }
	  }
	}

      } /* #pragma omp parallel */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    /****************************************************
                        bond energies
    ****************************************************/
 
    My_Eele1[0] = 0.0;

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (j=0; j<=FNAN[GA_AN]; j++){

        GB_AN = natn[GA_AN][j];  
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];

	/* non-spin-orbit coupling and non-LDA+U */  
	if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
	    && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){
	      My_Eele1[0] += 
		+ CDM[0][MA_AN][j][k][l]*Hks[0][MA_AN][j][k][l]
		+ CDM[1][MA_AN][j][k][l]*Hks[1][MA_AN][j][k][l]
		+ 2.0*CDM[2][MA_AN][j][k][l]*Hks[2][MA_AN][j][k][l]
		- 2.0*CDM[3][MA_AN][j][k][l]*Hks[3][MA_AN][j][k][l];
	    }
	  }
	}

	/* spin-orbit coupling or LDA+U */
	else {
	  for (k=0; k<tnoA; k++){
	    for (l=0; l<tnoB; l++){

	      My_Eele1[0] += 
		+ CDM[0][MA_AN][j][k][l]*Hks[0][MA_AN][j][k][l]
		- iDM[0][0][MA_AN][j][k][l]*ImNL[0][MA_AN][j][k][l]
		+ CDM[1][MA_AN][j][k][l]*Hks[1][MA_AN][j][k][l]
		- iDM[0][1][MA_AN][j][k][l]*ImNL[1][MA_AN][j][k][l]
		+ 2.0*CDM[2][MA_AN][j][k][l]*Hks[2][MA_AN][j][k][l]
		- 2.0*CDM[3][MA_AN][j][k][l]*(Hks[3][MA_AN][j][k][l]
				            +ImNL[2][MA_AN][j][k][l]);

	    }
	  }
	}

      }
    }

    /* MPI, My_Eele1 */
    MPI_Barrier(mpi_comm_level1);
    MPI_Allreduce(&My_Eele1[0], &Eele1[0], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (3<=level_stdout && myid==Host_ID){
      printf("Eele0=%15.12f\n",Eele0[0]);
      printf("Eele1=%15.12f\n",Eele1[0]);
    }

  } /* if ( strcasecmp(mode,"scf")==0 ) */

  else if ( strcasecmp(mode,"dos")==0 ){
    Save_DOS_NonCol(Residues,OLP0,EVal,Msize);
  }

  /****************************************************
    freeing of arrays:

  ****************************************************/

  free(Snd_H_Size);
  free(Rcv_H_Size);

  free(Snd_iHNL_Size);
  free(Rcv_iHNL_Size);

  free(Snd_S_Size);
  free(Rcv_S_Size);

  free(Msize);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(EVal[Mc_AN]);
  }
  free(EVal);

  for (spin=0; spin<3; spin++){
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

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(PDOS_DC[Mc_AN]);
  }
  free(PDOS_DC);

  free(omp_tmp);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  /* for PrintMemory */
  firsttime=0;

  return time0;
}



void Save_DOS_Col(double ******Residues, double ****OLP0, double ***EVal, int *Msize)
{
  int spin,Mc_AN,wanA,Gc_AN,tno1;
  int i1,i,j,MaxL,l,h_AN,Gh_AN,wanB,tno2;
  double Stime_atom,Etime_atom; 
  double sum;
  int i_vec[10];  
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  int numprocs,myid,ID,tag;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* open file pointers */

  if (myid==Host_ID){

    sprintf(file_eig,"%s%s.Dos.val",filepath,filename);
    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_eig);
    }
  }
  
  sprintf(file_ev, "%s%s.Dos.vec%i",filepath,filename,myid);
  if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {
    printf("cannot open a file %s\n",file_ev);
  }

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wanA = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[wanA];

      fprintf(fp_ev,"<AN%dAN%d\n",Gc_AN,spin);
      fprintf(fp_ev,"%d\n",Msize[Mc_AN]);

      for (i1=0; i1<Msize[Mc_AN]; i1++){

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

  /****************************************************
                   save *.Dos.val
  ****************************************************/

  if (myid==Host_ID){

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
  }

  /* close file pointers */

  if (myid==Host_ID){
    if (fp_eig) fclose(fp_eig);
  }

  if (fp_ev)  fclose(fp_ev);
}



void Save_DOS_NonCol(dcomplex ******Residues, double ****OLP0, double **EVal, int *Msize)
{
  int spin,Mc_AN,wanA,Gc_AN,tno1;
  int i1,i,j,MaxL,l,h_AN,Gh_AN,wanB,tno2;
  double Stime_atom,Etime_atom; 
  double tmp1,tmp2,tmp3,sum,SDup,SDdn;
  double Re11,Re22,Re12,Im12;
  double theta,phi,sit,cot,sip,cop; 
  int i_vec[10];  
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  int numprocs,myid,ID,tag;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* open file pointers */

  if (myid==Host_ID){

    sprintf(file_eig,"%s%s.Dos.val",filepath,filename);
    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {
      printf("cannot open a file %s\n",file_eig);
    }
  }
  
  sprintf(file_ev, "%s%s.Dos.vec%i",filepath,filename,myid);
  if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {
    printf("cannot open a file %s\n",file_ev);
  }

  /****************************************************
                   save *.Dos.vec
  ****************************************************/

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
    fprintf(fp_ev,"%d %d\n",2*Msize[Mc_AN],2*Msize[Mc_AN]);

    for (i1=0; i1<2*Msize[Mc_AN]; i1++){

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

	    Re11 += Residues[0][Mc_AN][h_AN][i][j][i1].r*
	                   OLP0[Mc_AN][h_AN][i][j];

	    Re22 += Residues[1][Mc_AN][h_AN][i][j][i1].r*
	                   OLP0[Mc_AN][h_AN][i][j];

	    Re12 += Residues[2][Mc_AN][h_AN][i][j][i1].r*
 	                   OLP0[Mc_AN][h_AN][i][j];

	    Im12 += Residues[2][Mc_AN][h_AN][i][j][i1].i*
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

  /****************************************************
                   save *.Dos.val
  ****************************************************/

  if (myid==Host_ID){

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
  }

  /* close file pointers */

  if (myid==Host_ID){
    if (fp_eig) fclose(fp_eig);
  }

  if (fp_ev)  fclose(fp_ev);

}

