/**********************************************************************
  NBO_Krylov.c

     NBO_Krylov.c is a subroutine to calculate NBO 
     in case that solver is "Krylov".

  Log of NBO_Krylov.c

     --/---/2012  Released by T.Ohwaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"

#define  error_check    0
#define  cutoff_value  Threshold_OLP_Eigen

#include "tran_prototypes.h"

#define  measure_time   0

static double Krylov_Col_NAO(double *****Hks, double ****OLP0, double *****CDM);

static void Calc_NAO(int FS_atm, int Nc_AN, double **D_full, double **S_full, double **H_full,
                      int FS_atomnum, int SizeMat, double *NP_total, double *NP_partial);

static void Calc_NBO(int spin0, int Num_FSCenter, int Num_FCenter, int LVsize, 
                     int *FSCenter, int *LMindx2LMposi,
                     double **P_full, double **H_full, double **T_full);

void Calc_NAO_Krylov(double *****Hks, double ****OLP0, double *****CDM)
{

  /****************************************************
         collinear without spin-orbit coupling
  ****************************************************/

  if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==0 ){

    Krylov_Col_NAO( Hks, OLP0, CDM );

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
    printf("The O(N) Krylov subspace method is not supported for non-collinear DFT calculations.\n");
    MPI_Finalize();
    exit(1);
  }

}



static double Krylov_Col_NAO(double *****Hks, double ****OLP0, double *****CDM)
{

  int Mc_AN,Gc_AN,i,is,js,Gi,wan,wanA,wanB,Anum;
  int Mc_AN0,Gc_AN0,Gc_AN1,Mc_AN1,Mh_AN;
  int size1,size2,num,NUM0,NUM,NUM1,n2,Cwan,Hwan,Rn2;
  int ih,ig,ian,j,kl,jg,jan,Bnum,m,n,spin,i2,ip;
  int k,l,i1,j1,L1,L2,M1,M2,k1,k2,nn,mm;
  int P_min,m_size,q1,q2,csize,csize2;
  int h_AN1,Mh_AN1,h_AN2,Gh_AN1,Gh_AN2,Ga_AN0,Ga_AN1,wan1,wan2;
  int po,po1,loopN,tno0,tno1,tno2,h_AN,Gh_AN,Gh_AN0,rl1,rl2,rl;
  int MA_AN,GA_AN,tnoA,GB_AN,tnoB;
  int Msize2_max,rlmax_EC_max,EKC_core_size_max;
  double sum,FermiF;
  double sum00,sum10,sum20,sum30;
  double sum01,sum11,sum21,sum31;
  double sum02,sum12,sum22,sum32;
  double sum03,sum13,sum23,sum33;
  double tmp0,tmp1,tmp2,tmp3,b2,co,x0,xmin,xmax;
  double max_x=30.0;
  double *tmp_array;
  double *tmp_array2;
  double *****CDMn0,*temp_vec1,*temp_vec2,*temp_vec3,*temp_vec4;
  double tmp_ele0,tmp_ele1,tmp_ele2,tmp_ele3;

  int *MP;

  /*****************************************************
  Msize:   \sum_FNAN Spe_Total_CNO  
  Msize2:  \sum_FNAN+SNAN Spe_Total_CNO  
  Msize3:  rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]
  Msize4:  rlmax_EC2[Mc_AN]*EKC_core_size[Mc_AN]
  Msize5:  dimension for the last column of Residues
  *****************************************************/

  int *Msize;
  int *Msize2;
  int *Msize3;
  int *Msize4;
  int *Msize5;
  int Matomnum_NAO2;
  int numprocs,myid,ID,IDS,IDR,tag=999;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* BLAS */
   int M,N,K,lda,ldb,ldc;
   double alpha,beta;

   if (myid == Host_ID){

   printf("\n");fflush(stdout);
   printf(" *********************************************** \n");fflush(stdout);
   printf("                                                 \n");fflush(stdout);
   printf("      NATURAL ATOMIC ORBITAL (NAO) Analysis      \n");fflush(stdout);
   printf("                                                 \n");fflush(stdout);
   printf(" *********************************************** \n");fflush(stdout);
   printf("\n");fflush(stdout);

   printf("## Threshold for occupied or Rydberg orbital in NAO calc.: %lf elec.\n",
                                                 NAO_Occ_or_Ryd);fflush(stdout);
#if 0
   printf("\n");fflush(stdout);
   printf(" *********************************************** \n");fflush(stdout);
   printf("                                                 \n");fflush(stdout);
   printf("       NATURAL BOND ORBITAL (NBO) Analysis       \n");fflush(stdout);
   printf("                                                 \n");fflush(stdout);
   printf(" *********************************************** \n");fflush(stdout);
   printf("\n");fflush(stdout);
   printf(" =============================================== \n");fflush(stdout);
   printf("                                                 \n");fflush(stdout);
   printf("    STAGE-1: Natural Atomic Orbital (NAO) &      \n");fflush(stdout);
   printf("             Natural Population Analysis (NPA)   \n");fflush(stdout);
   printf("    STAGE-2: Natural Hybrid Orbital (NHO)        \n");fflush(stdout);
   printf("    STAGE-3: Natural Bond Orbital (NBO)          \n");fflush(stdout);
   printf("                                                 \n");fflush(stdout);
   printf(" =============================================== \n");fflush(stdout);
   printf("\n");fflush(stdout);
#endif
   }



   Matomnum_NAO2 = Matomnum + MatomnumF + MatomnumS;

  /****************************************************
                   allocation of arrays:
  ****************************************************/

  Msize  = (int*)malloc(sizeof(int)*(Matomnum+1));
  Msize2 = (int*)malloc(sizeof(int)*(Matomnum+1));
  Msize3 = (int*)malloc(sizeof(int)*(Matomnum+1));
  Msize4 = (int*)malloc(sizeof(int)*(Matomnum+1));

  Msize2_max = 0;
  rlmax_EC_max = 0;
  EKC_core_size_max = 0;

/* find Msize */

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
  /* 
   Gc_AN = M2G[Mc_AN]; 
  */
    if (Mc_AN==0){
      Msize[Mc_AN] = 1;
    }
    else{
      Gc_AN = M2G[Mc_AN];

      NUM = 0;
      for (i=0; i<=FNAN[Gc_AN]; i++){
        Gi = natn[Gc_AN][i];
        wanA = WhatSpecies[Gi];
        NUM += Spe_Total_CNO[wanA];
      }
      Msize[Mc_AN] = NUM;
    }
  }

  /* find Msize2 and Msize2_max */

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
  /* 
   Gc_AN = M2G[Mc_AN]; 
  */
    if (Mc_AN==0){
      Msize2[Mc_AN] = 1;
    }
    else{
      Gc_AN = M2G[Mc_AN];

      NUM = 0;
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
        Gi = natn[Gc_AN][i];
        wanA = WhatSpecies[Gi];
        NUM += Spe_Total_CNO[wanA];
      }
      Msize2[Mc_AN] = NUM;
    }

    if (Msize2_max<Msize2[Mc_AN]) Msize2_max = Msize2[Mc_AN];
    if (rlmax_EC_max<rlmax_EC[Mc_AN]) rlmax_EC_max = rlmax_EC[Mc_AN];
    if (EKC_core_size_max<EKC_core_size[Mc_AN]) EKC_core_size_max = EKC_core_size[Mc_AN];

  }


#if 0
  /* find Msize */

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (Mc_AN==0){
      Msize[Mc_AN] = 1;
    }
    else{
      NUM = 0;
      for (i=0; i<=FNAN[Gc_AN]; i++){
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	NUM += Spe_Total_CNO[wanA];
      }
      Msize[Mc_AN] = NUM;
    }
  }

  /* find Msize2 and Msize2_max */

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (Mc_AN==0){
      Msize2[Mc_AN] = 1;
    }
    else{
      NUM = 0;
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	NUM += Spe_Total_CNO[wanA];
      }
      Msize2[Mc_AN] = NUM;
    }
 
    if (Msize2_max<Msize2[Mc_AN]) Msize2_max = Msize2[Mc_AN];
    if (rlmax_EC_max<rlmax_EC[Mc_AN]) rlmax_EC_max = rlmax_EC[Mc_AN];
    if (EKC_core_size_max<EKC_core_size[Mc_AN]) EKC_core_size_max = EKC_core_size[Mc_AN];

  }
#endif
  Msize2_max = Msize2_max + 4;
  rlmax_EC_max = rlmax_EC_max + 4;
  EKC_core_size_max = EKC_core_size_max + 4;


  /* find Msize3 and Msize4 */

  Msize3[0] = 1;
  Msize4[0] = 1;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];
    Msize3[Mc_AN] = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN];
    Msize4[Mc_AN] = rlmax_EC2[Mc_AN]*EKC_core_size[Mc_AN];
  }

  if (2<=level_stdout){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        printf("<Krylov> myid=%4d Mc_AN=%4d Gc_AN=%4d Msize=%4d\n",
        myid,Mc_AN,M2G[Mc_AN],Msize[Mc_AN]);
    }
  }

  /****************************************************
                    density matrix
  ****************************************************/

    CDMn0 = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
    for (k=0; k<=SpinP_switch; k++){
      CDMn0[k] = (double****)malloc(sizeof(double***)*(Matomnum_NAO2+1));
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum_NAO2; Mc_AN++){

        if (Mc_AN==0){
          Gc_AN = 0;
          tno0 = 1;
        }
        else{
          Gc_AN = S_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        CDMn0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+SNAN[Gc_AN]+1));
        for (h_AN=0; h_AN<=FNAN[Gc_AN]+SNAN[Gc_AN]; h_AN++){

          if (Mc_AN==0){
            tno1 = 1;
          }
          else{
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            tno1 = Spe_Total_NO[Hwan];
          }

          CDMn0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
          for (i=0; i<tno0; i++){
            CDMn0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
            for (j=0; j<tno1; j++) CDMn0[k][Mc_AN][h_AN][i][j] = 0.0;
          }
        }
      }
    }

  /****************************************************
   MPI

   Hks
  ****************************************************/

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

        size1 = Snd_HFS_Size[IDS];

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

        size2 = Rcv_HFS_Size[IDR];
        
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
  } /* ID */

  /***********************************************
    for regeneration of the buffer matrix
  ***********************************************/

#pragma omp parallel shared(List_YOUSO,ChemP,RMI1,S_G2M,EC_matrix,Krylov_U,SpinP_switch,EKC_core_size,rlmax_EC,rlmax_EC2,Hks,OLP0,Msize3,Msize2,Msize,natn,FNAN,SNAN,Spe_Total_CNO,WhatSpecies,M2G,Matomnum,myid)
  {

    int OMPID,Nthrds,Nprocs;
    int Mc_AN,Gc_AN,wan,ct_on,spin;
    int ig,ian,ih,kl,jg,jan,Bnum,m,n,rl;
    int Anum,i,j,k,Gi,wanA,NUM,n2,csize,is,i2;
    int i1,rl1,js,ip,po1,tno1,h_AN,Gh_AN,wanB,tno2;
    int *MP;
    int KU_d1,KU_d2;

    double **C,***Krylov_U_NBO;
    double **H_DC,*ko;
    double sum00,sum10,sum20,sum30,sum;
    double tmp0,tmp1,tmp2,tmp3,x;
/*
    double **tmpvec0;
    double **tmpvec1;
    double **tmpvec2;
*/
    double **H_DCn,**OLP0n,**Cn,**CDMn;

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /* allocation of arrays */

    MP = (int*)malloc(sizeof(int)*List_YOUSO[2]);
/*
    tmpvec0 = (double**)malloc(sizeof(double*)*EKC_core_size_max);
    for (i=0; i<EKC_core_size_max; i++){
      tmpvec0[i] = (double*)malloc(sizeof(double)*Msize2_max);
    }

    tmpvec1 = (double**)malloc(sizeof(double*)*EKC_core_size_max);
    for (i=0; i<EKC_core_size_max; i++){
      tmpvec1[i] = (double*)malloc(sizeof(double)*Msize2_max);
    }

    tmpvec2 = (double**)malloc(sizeof(double*)*EKC_core_size_max);
    for (i=0; i<EKC_core_size_max; i++){
      tmpvec2[i] = (double*)malloc(sizeof(double)*Msize2_max);
    }
*/
    /* allocation of arrays (for NAO) */
/*
    int *Table_M2G0;
    double *NP_cluster0, *NP_atom0;

    Table_M2G0 = (int*)malloc(sizeof(int)*(Matomnum+1));
    for (i=0; i<=Matomnum; i++){
      Table_M2G0[i] = 0;
    }

    NP_cluster0 = (double*)malloc(sizeof(double)*(Matomnum+1));
    for (i=0; i<=Matomnum; i++){
      NP_cluster0[i] = 0.0;
    }

    NP_atom0 = (double*)malloc(sizeof(double)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      NP_atom0[i] = 0.0;
    }
*/
    Krylov_U_NBO = (double***)malloc(sizeof(double**)*rlmax_EC_max);
    for (rl=0; rl<rlmax_EC_max; rl++){
      Krylov_U_NBO[rl] = (double**)malloc(sizeof(double*)*EKC_core_size_max);
    for (n=0; n<EKC_core_size_max; n++){
      Krylov_U_NBO[rl][n] = (double*)malloc(sizeof(double)*(Msize2_max+1));
    for (i=0; i<=Msize2_max; i++){
      Krylov_U_NBO[rl][n][i] = 0.0;  
    }
    }
    }
    

    double Beta_NAO;
/*
    Beta_NAO = 1.0/kB/(5.0/eV2Hartree);
*/
    Beta_NAO = Beta;

    /***********************************************
              Main Loop of Calculation (1)
    ***********************************************/

    for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){

      Gc_AN = M2G[Mc_AN];
      wan = WhatSpecies[Gc_AN];

      /* MP array */

      Anum = 1;
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
	MP[i] = Anum;
	Gi = natn[Gc_AN][i];
	wanA = WhatSpecies[Gi];
	Anum += Spe_Total_CNO[wanA];
      }
      NUM = Anum - 1;
      n2 = NUM + 40;

      /***********************************************
                 allocation of arrays:
      ***********************************************/

      if (Msize[Mc_AN]<Msize3[Mc_AN])
	csize = Msize3[Mc_AN] + 40;
      else
	csize = Msize[Mc_AN] + 40;

      H_DC = (double**)malloc(sizeof(double*)*csize);
      for (i=0; i<csize; i++){
	H_DC[i] = (double*)malloc(sizeof(double)*csize);
      }

      ko = (double*)malloc(sizeof(double)*csize);

      C = (double**)malloc(sizeof(double*)*csize);
      for (i=0; i<csize; i++){
	C[i] = (double*)malloc(sizeof(double)*csize);
      }

      /***********************************************
              allocation of arrays (for NAO): 
      ***********************************************/

      if (Msize2[Mc_AN]<Msize3[Mc_AN])
        csize2 = Msize3[Mc_AN] + 40;
      else
        csize2 = Msize2[Mc_AN] + 40;

      H_DCn = (double**)malloc(sizeof(double*)*csize2);
      for (i=0; i<csize2; i++){
        H_DCn[i] = (double*)malloc(sizeof(double)*csize2);
      }

      OLP0n = (double**)malloc(sizeof(double*)*csize2);
      for (i=0; i<csize2; i++){
        OLP0n[i] = (double*)malloc(sizeof(double)*csize2);
      }

      Cn = (double**)malloc(sizeof(double*)*csize2);
      for (i=0; i<csize2; i++){
        Cn[i] = (double*)malloc(sizeof(double)*csize2);
      }

      CDMn = (double**)malloc(sizeof(double*)*csize2);
      for (i=0; i<csize2; i++){
        CDMn[i] = (double*)malloc(sizeof(double)*csize2);
      }


      for (spin=0; spin<=SpinP_switch; spin++){

	/****************************************************
                construct the Hamiltonian matrix
	****************************************************/

	for (i=0; i<=FNAN[Gc_AN]; i++){
	  ig = natn[Gc_AN][i];
	  ian = Spe_Total_CNO[WhatSpecies[ig]];
	  Anum = MP[i];
	  ih = S_G2M[ig]; /* S_G2M should be used */

	  for (j=0; j<=FNAN[Gc_AN]; j++){

	    kl = RMI1[Mc_AN][i][j];
	    jg = natn[Gc_AN][j];
	    jan = Spe_Total_CNO[WhatSpecies[jg]];
	    Bnum = MP[j];

	    if (0<=kl){
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  H_DC[Anum+m][Bnum+n] = Hks[spin][ih][kl][m][n];
		}
	      }
	    }

	    else{
	      for (m=0; m<ian; m++){
		for (n=0; n<jan; n++){
		  H_DC[Anum+m][Bnum+n] = 0.0;
		}
	      }
	    }
	  }
	}

        /****************************************************
               prepalation of matrices H and S for NAO 
        ****************************************************/

        for (i=0; i<=FNAN[Gc_AN]+SNAN[Gc_AN]; i++){
          ig = natn[Gc_AN][i];
          ian = Spe_Total_CNO[WhatSpecies[ig]];
          Anum = MP[i]-1;
          ih = S_G2M[ig]; /* S_G2M should be used */

          for (j=0; j<=FNAN[Gc_AN]+SNAN[Gc_AN]; j++){

            kl = RMI1[Mc_AN][i][j];
            jg = natn[Gc_AN][j];
            jan = Spe_Total_CNO[WhatSpecies[jg]];
            Bnum = MP[j]-1;

            if (0<=kl){
              for (m=0; m<ian; m++){
                for (n=0; n<jan; n++){
                  H_DCn[Anum+m][Bnum+n] = Hks[spin][ih][kl][m][n];
                  OLP0n[Anum+m][Bnum+n] =      OLP0[ih][kl][m][n];
                }
              }
            }

            else{
              for (m=0; m<ian; m++){
                for (n=0; n<jan; n++){
                  H_DCn[Anum+m][Bnum+n] = 0.0;
                  OLP0n[Anum+m][Bnum+n] = 0.0;
                }
              }
            }
          }
        }

        /****************************************************
               conversion of Krylov_U to the old array    
        ****************************************************/

        KU_d1 = EKC_core_size[Mc_AN]*Msize2[Mc_AN];
        KU_d2 = Msize2[Mc_AN];

        for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
          for (n=0; n<EKC_core_size[Mc_AN]; n++){
            for (i=0; i<Msize2[Mc_AN]; i++){
              Krylov_U_NBO[rl][n][i+1] = Krylov_U[spin][Mc_AN][rl*KU_d1+n*KU_d2+i+1];
            }
          }
        }

	/****************************************************
                   transform u1^+ * H_DC * u1
	****************************************************/

	/* H_DC * u1 */

	/* original version */
	/*
	  for (i=1; i<=Msize[Mc_AN]; i++){
	  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
          for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  sum = 0.0;
	  for (j=1; j<=Msize[Mc_AN]; j++){
	  sum += H_DC[i][j]*Krylov_U[spin][Mc_AN][rl][n][j];  
	  }

	  C[rl*EKC_core_size[Mc_AN]+n+1][i] = sum;
	  }      
	  }      
	  } 
	*/
     
	/* unrolling version */

	for (i=1; i<=(Msize[Mc_AN]-3); i+=4){
	  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum00 = 0.0;
	      sum10 = 0.0;
	      sum20 = 0.0;
	      sum30 = 0.0;

	      for (j=1; j<=Msize[Mc_AN]; j++){
		sum00 += H_DC[i+0][j]*Krylov_U_NBO[rl][n][j];  
		sum10 += H_DC[i+1][j]*Krylov_U_NBO[rl][n][j];  
		sum20 += H_DC[i+2][j]*Krylov_U_NBO[rl][n][j];  
		sum30 += H_DC[i+3][j]*Krylov_U_NBO[rl][n][j];  
	      }

	      C[rl*EKC_core_size[Mc_AN]+n+1][i+0] = sum00;
	      C[rl*EKC_core_size[Mc_AN]+n+1][i+1] = sum10;
	      C[rl*EKC_core_size[Mc_AN]+n+1][i+2] = sum20;
	      C[rl*EKC_core_size[Mc_AN]+n+1][i+3] = sum30;

	    }      
	  }      
	} 

	is = Msize[Mc_AN] - Msize[Mc_AN]%4 + 1;

	for (i=is; i<=Msize[Mc_AN]; i++){
	  for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
	    for (n=0; n<EKC_core_size[Mc_AN]; n++){

	      sum = 0.0;
	      for (j=1; j<=Msize[Mc_AN]; j++){
		sum += H_DC[i][j]*Krylov_U_NBO[rl][n][j];  
	      }

	      C[rl*EKC_core_size[Mc_AN]+n+1][i] = sum;
	    }      
	  }      
	} 

	/* u1^+ * H_DC * u1 */

	/* original version */

	/*
	  for (rl1=0; rl1<rlmax_EC[Mc_AN]; rl1++){
	  for (m=0; m<EKC_core_size[Mc_AN]; m++){
	  for (rl2=rl1; rl2<rlmax_EC[Mc_AN]; rl2++){
	  for (n=0; n<EKC_core_size[Mc_AN]; n++){

	  sum = 0.0;

	  i2 = rl2*EKC_core_size[Mc_AN] + n + 1;

	  for (i=1; i<=Msize[Mc_AN]; i++){
	  sum += Krylov_U[spin][Mc_AN][rl1][m][i]*C[i2][i];
	  }

	  H_DC[rl1*EKC_core_size[Mc_AN]+m+1][rl2*EKC_core_size[Mc_AN]+n+1] = sum;
	  H_DC[rl2*EKC_core_size[Mc_AN]+n+1][rl1*EKC_core_size[Mc_AN]+m+1] = sum;
	  }
	  }
	  }
	  }
	*/

	/* unrolling version */

	for (i2=1; i2<=(rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]-3); i2+=4){
	  for (i1=i2; i1<=rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]; i1++){

	    rl1 = (i1-1)/EKC_core_size[Mc_AN];
	    m = (i1-1) % EKC_core_size[Mc_AN];

	    sum00 = 0.0;
	    sum10 = 0.0;
	    sum20 = 0.0;
	    sum30 = 0.0;

	    for (i=1; i<=Msize[Mc_AN]; i++){
	      /* transpose */
	      sum00 += Krylov_U_NBO[rl1][m][i]*C[i2+0][i];
	      sum10 += Krylov_U_NBO[rl1][m][i]*C[i2+1][i];
	      sum20 += Krylov_U_NBO[rl1][m][i]*C[i2+2][i];
	      sum30 += Krylov_U_NBO[rl1][m][i]*C[i2+3][i];
	    }

	    H_DC[i1][i2+0] = sum00;
	    H_DC[i2+0][i1] = sum00;

	    H_DC[i1][i2+1] = sum10;
	    H_DC[i2+1][i1] = sum10;

	    H_DC[i1][i2+2] = sum20;
	    H_DC[i2+2][i1] = sum20;

	    H_DC[i1][i2+3] = sum30;
	    H_DC[i2+3][i1] = sum30;
	  }
	}

	is = rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN] - (rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN])%4 + 1;

	for (i2=is; i2<=rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]; i2++){
	  for (i1=i2; i1<=rlmax_EC[Mc_AN]*EKC_core_size[Mc_AN]; i1++){

	    rl1 = (i1-1)/EKC_core_size[Mc_AN];
	    m = (i1-1) % EKC_core_size[Mc_AN];

	    sum = 0.0;

	    for (i=1; i<=Msize[Mc_AN]; i++){
	      /* transpose */
	      sum += Krylov_U_NBO[rl1][m][i]*C[i2][i];
	    }

	    H_DC[i1][i2] = sum;
	    H_DC[i2][i1] = sum;
	  }
	}

	/* correction for ZeroNum */

	m = (int)Krylov_U[spin][Mc_AN][0];
	for (i=1; i<=m; i++) H_DC[i][i] = 1.0e+3;

	/****************************************************
            H0 = u1^+ * H_DC * u1 + D 
	****************************************************/

	for (i=1; i<=Msize3[Mc_AN]; i++){
	  for (j=1; j<=Msize3[Mc_AN]; j++){
	    H_DC[i][j] += EC_matrix[spin][Mc_AN][i][j];
	  }
	}

	/****************************************************
           diagonalize
	****************************************************/

	Eigen_lapack(H_DC,ko,Msize3[Mc_AN],Msize3[Mc_AN]);

        /********************************************
             back transformation of eigenvectors
                    c = u1 * b (for NAO)
        *********************************************/

        for (i=1; i<=Msize2[Mc_AN]; i++){
        for (j=1; j<=Msize3[Mc_AN]; j++){
          Cn[i][j] = 0.0;
        }
        }

        /* for NAO */
        for (i=1; i<=Msize2[Mc_AN]; i++){
        for (rl=0; rl<rlmax_EC[Mc_AN]; rl++){
        for (n=0; n<EKC_core_size[Mc_AN]; n++){

          tmp1 = Krylov_U_NBO[rl][n][i];
          i1 = rl*EKC_core_size[Mc_AN] + n + 1;

          for (j=1; j<=Msize3[Mc_AN]; j++){
            Cn[i][j] += tmp1*H_DC[i1][j];
          }

        }
        }
        }

        /**************************************************
           Construction of partial density matrix (CDMn0)
        **************************************************/

        wanA = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_CNO[wanA];

        for (i=0; i<tno1; i++){
          for (h_AN=0; h_AN<=FNAN[Gc_AN]+SNAN[Gc_AN]; h_AN++){
            Gh_AN = natn[Gc_AN][h_AN];
            wanB = WhatSpecies[Gh_AN];
            tno2 = Spe_Total_CNO[wanB];
            Bnum = MP[h_AN];
            for (j=0; j<tno2; j++){
            for (i1=1; i1<=Msize3[Mc_AN]; i1++){
              x = (ko[i1] - ChemP)*Beta_NAO;
              if (x <= -max_x) x = -max_x;
              if (x >=  max_x) x =  max_x;
              if (SpinP_switch==0) FermiF = 2.0/(1.0 + exp(x));
              else                 FermiF = 1.0/(1.0 + exp(x));

              CDMn0[spin][Mc_AN][h_AN][i][j] += FermiF * Cn[1+i][i1] * Cn[Bnum+j][i1];

            }
            }
          }
        }

      } /* spin */

      /***********************************************
                    freeing of arrays:
      ***********************************************/

      for (i=0; i<csize; i++){
	free(H_DC[i]);
      }
      free(H_DC);

      free(ko);

      for (i=0; i<csize; i++){
	free(C[i]);
      }
      free(C);


      for (i=0; i<csize2; i++){
        free(H_DCn[i]);
      }
      free(H_DCn);

      for (i=0; i<csize2; i++){
        free(Cn[i]);
      }
      free(Cn);

      for (i=0; i<csize2; i++){
        free(CDMn[i]);
      }
      free(CDMn);

      for (i=0; i<csize2; i++){
        free(OLP0n[i]);
      }
      free(OLP0n);

    } /* Mc_AN */

    for (rl=0; rl<rlmax_EC_max; rl++){
    for (n=0; n<EKC_core_size_max; n++){
      free(Krylov_U_NBO[rl][n]);
    }
      free(Krylov_U_NBO[rl]);
    }
      free(Krylov_U_NBO);

    free(MP);

  }   /* #pragma omp parallel */


  /***************************************************
      Natural Atomic Orbital & Natural Bond Orbital 
  ***************************************************/

   /****************************************************
     MPI : CDMn0
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
              for (h_AN=0; h_AN<=FNAN[Gc_AN]+SNAN[Gc_AN]; h_AN++){
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_CNO[Hwan];
                size1 += tno1*tno2;
              }
            }
          }

          Snd_HFS_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else{
          Snd_HFS_Size[IDS] = 0;
        }

        /* receiving of size of data */

        if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_HFS_Size[IDR] = size2;
        }
        else{
          Rcv_HFS_Size[IDR] = 0;
        }
        if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
      }
      else{
        Snd_HFS_Size[IDS] = 0;
        Rcv_HFS_Size[IDR] = 0;
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

        size1 = Snd_HFS_Size[IDS];

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
            for (h_AN=0; h_AN<=FNAN[Gc_AN]+SNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
              for (j=0; j<tno2; j++){
                tmp_array[num] = CDMn0[spin][Mc_AN][h_AN][i][j];
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

        size2 = Rcv_HFS_Size[IDR];

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

            for (h_AN=0; h_AN<=FNAN[Gc_AN]+SNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
              for (j=0; j<tno2; j++){
                CDMn0[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
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

  /*************************************************
          Preparation for NBO Calc. (1):
    Search for neighbour atoms of NBO-center atoms
  **************************************************/

  int counter, i2ID, tmp_num1, tmp_num2, tmp_max_num;
  int posi1, posi2, posi3, posi4, leng1, leng2;
  int jc, jp, kc, kg, kp;
  int Num_NBO_SCenter, Num_NBO_SCenter2, Num_NBO_FSCenter;
  int *NBO_SCenter, *NBO_SCenter2, *NBO_FSCenter; 
  int **NBO_FCenter_GAN, **NBO_SCenter_GAN, **NBO_FSCenter_GAN;

  NBO_SCenter  = (int*)malloc(sizeof(int)*(Num_NBO_FCenter*Max_FSNAN));
  NBO_SCenter2 = (int*)malloc(sizeof(int)*(Num_NBO_FCenter*Max_FSNAN));

  NBO_FCenter_GAN = (int**)malloc(sizeof(int*)*Num_NBO_FCenter);
  for (i=0; i<Num_NBO_FCenter; i++){
    NBO_FCenter_GAN[i] = (int*)malloc(sizeof(int)*(Max_FSNAN+1));
  }


  if (myid == Host_ID){

   counter = 0;
   for (i=0; i<Num_NBO_FCenter; i++){
     Gc_AN0 = NBO_FCenter[i];

     for (j=1; j<=FNAN[Gc_AN0]; j++){
/*
       printf("natn[%d] = %d Dis[%d] = %10.6f (Bohr)\n",j,natn[Gc_AN0][j],j,Dis[Gc_AN0][j]);
*/
       if (Dis[Gc_AN0][j] <= 3.6){ /* Bohr */
         counter++;
         NBO_SCenter[counter-1] = natn[Gc_AN0][j];
       }
     }

   }
   Num_NBO_SCenter = counter;

/*
   printf("<< NBO Prep (1) >> Num_NBO_SCenter = %d \n",Num_NBO_SCenter);
   for (i=0; i<Num_NBO_SCenter; i++){
     printf("NBO_SCenter[%d] = %d \n",i,NBO_SCenter[i]);
   }
*/

  /* True NBO_SCenter & Num_NBO_SCenter */

   counter = 0;
   for (i=0; i<Num_NBO_SCenter; i++){
     Gc_AN0 = NBO_SCenter[i];
     po = 0;

     for (j=0; j<Num_NBO_FCenter; j++){
       Gc_AN1 = NBO_FCenter[j];
       if (Gc_AN0 == Gc_AN1){
         po = 1;
         break;
       }
     }

     if (po == 0){
       counter++;
       NBO_SCenter2[counter-1] = Gc_AN0;
     }

   }
   Num_NBO_SCenter2 = counter; 

/*
   printf("<< NBO Prep (1) >> Num_NBO_SCenter2 = %d \n",Num_NBO_SCenter2);
   for (i=0; i<Num_NBO_SCenter2; i++){
     printf("NBO_SCenter2[%d] = %d \n",i,NBO_SCenter2[i]);
   }
*/

   counter = 0;
   for (i=0; i<Num_NBO_SCenter2; i++){
     Gc_AN0 = NBO_SCenter2[i];
     po = 0;

     for (j=i+1; j<Num_NBO_SCenter2; j++){
       Gc_AN1 = NBO_SCenter2[j];
       if (Gc_AN0 == Gc_AN1){
         po = 1;
         break;
       }
     }

     if (po == 0){
       counter++;
       NBO_SCenter[counter-1] = Gc_AN0;
     }

   }
   Num_NBO_SCenter = counter; 

/*
   printf("<< NBO Prep (1) >> Num_NBO_SCenter = %d \n",Num_NBO_SCenter);
   for (i=0; i<Num_NBO_SCenter; i++){
     printf("NBO_SCenter[%d] = %d \n",i,NBO_SCenter[i]);
   }
*/
  } /* if (myid == Host_ID) */

   MPI_Bcast(&Num_NBO_SCenter, 1, MPI_INT, Host_ID, mpi_comm_level1);

   NBO_SCenter_GAN = (int**)malloc(sizeof(int*)*Num_NBO_SCenter);
   for (i=0; i<Num_NBO_SCenter; i++){
     NBO_SCenter_GAN[i] = (int*)malloc(sizeof(int)*(Max_FSNAN+1));
   }


  if (myid == Host_ID){

   for (i=0; i<Num_NBO_FCenter; i++){
     Gc_AN0 = NBO_FCenter[i];
     for (j=0; j<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); j++){
       jg = natn[Gc_AN0][j];
       NBO_FCenter_GAN[i][j] = jg + atomnum*ncn[Gc_AN0][j];
/*
       printf("### NBO_FCenter_GAN[%3d][%3d] = %4d (Gc_AN0=%4d, jg=%4d, ncn=%4d)\n",
              i,j,NBO_FCenter_GAN[i][j],Gc_AN0,jg,ncn[Gc_AN0][j]);
*/
     }
   }
   printf("\n");

   for (i=0; i<Num_NBO_SCenter; i++){
     Gc_AN0 = NBO_SCenter[i];
     for (j=0; j<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); j++){
       jg = natn[Gc_AN0][j];
       NBO_SCenter_GAN[i][j] = jg + atomnum*ncn[Gc_AN0][j];
/*
       printf("### NBO_SCenter_GAN[%3d][%3d] = %4d (Gc_AN0=%4d, jg=%4d, ncn=%4d)\n",
              i,j,NBO_SCenter_GAN[i][j],Gc_AN0,jg,ncn[Gc_AN0][j]);
*/
     }
   }

    /* NBO_FCenter + NBO_SCenter => NBO_FSCenter */

    Num_NBO_FSCenter = Num_NBO_FCenter + Num_NBO_SCenter;

/*
    printf("<< NBO Prep (1) >> Num_NBO_FSCenter = %d \n",Num_NBO_FSCenter);
*/

  } /* if (myid == Host_ID) */


   MPI_Bcast(&Num_NBO_FSCenter, 1, MPI_INT, Host_ID, mpi_comm_level1);

   NBO_FSCenter = (int*)malloc(sizeof(int)*(Num_NBO_FSCenter*Max_FSNAN));

   NBO_FSCenter_GAN = (int**)malloc(sizeof(int*)*(Num_NBO_FSCenter+1));
   for (i=0; i<=Num_NBO_FSCenter; i++){
     NBO_FSCenter_GAN[i] = (int*)malloc(sizeof(int)*(Max_FSNAN+1));
   }


  if (myid == Host_ID){

    for (i=0; i<Num_NBO_FCenter; i++){
      NBO_FSCenter[i] = NBO_FCenter[i];
      Gc_AN0 = NBO_FCenter[i];
    for (j=0; j<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); j++){
      NBO_FSCenter_GAN[i][j] = NBO_FCenter_GAN[i][j];
    }
    }

    for (i=0; i<Num_NBO_SCenter; i++){
      NBO_FSCenter[Num_NBO_FCenter + i] = NBO_SCenter[i];
      Gc_AN0 = NBO_SCenter[i];
    for (j=0; j<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); j++){
      NBO_FSCenter_GAN[Num_NBO_FCenter + i][j] = NBO_SCenter_GAN[i][j];
    }
    }
/*
    for (i=0; i<Num_NBO_FSCenter; i++){
      Gc_AN0 = NBO_FSCenter[i];
    for (j=0; j<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); j++){
      printf("$$$ NBO_FSCenter_GAN[%3d][%3d] = %4d \n",i,j,NBO_FSCenter_GAN[i][j]);
    }
    }
*/
/*
    for (i=0; i<Num_NBO_FSCenter; i++){
      Gc_AN0 = NBO_FSCenter[i];
      printf("$$$ NBO_FSCenter[%3d] = %4d : ID = %d\n",i,Gc_AN0,G2ID[Gc_AN0]);
    }
*/
  } /* if (myid == Host_ID) */

    MPI_Bcast(&NBO_FSCenter[0], Num_NBO_FSCenter, MPI_INT, Host_ID, mpi_comm_level1);

  int NBO_LMat_size;
  int *NBO_LMindx2LMatom, *NBO_LMindx2LMAtom, *NBO_LMindx2LMposi;
  int *NBO_LMindx2SMindx, *NBO_LMindx2SMposi;

    /** Constrution of NAO-based Density Matrix **/
    /* Set of Density Matrix for Contructed Cluster with NBO-center atoms */

    NBO_LMindx2LMatom = (int*)malloc(sizeof(int)*(Num_NBO_FSCenter*Max_FSNAN));
    NBO_LMindx2LMAtom = (int*)malloc(sizeof(int)*(Num_NBO_FSCenter*Max_FSNAN));
    NBO_LMindx2LMposi = (int*)malloc(sizeof(int)*(Num_NBO_FSCenter*Max_FSNAN));
    NBO_LMindx2SMindx = (int*)malloc(sizeof(int)*(Num_NBO_FSCenter*Max_FSNAN));
    NBO_LMindx2SMposi = (int*)malloc(sizeof(int)*(Num_NBO_FSCenter*Max_FSNAN));

  if (myid==Host_ID){

    posi1= 0;

    for (j=0; j<Num_NBO_FSCenter; j++){
      Gh_AN0 = NBO_FSCenter[j];
      Gh_AN1 = NBO_FSCenter_GAN[j][0];
      Cwan = WhatSpecies[Gh_AN0];
      posi2 = posi1 + Spe_Total_CNO[Cwan];

      NBO_LMindx2LMAtom[j] = Gh_AN1;
      NBO_LMindx2LMatom[j] = Gh_AN0;
      NBO_LMindx2LMposi[j] = posi1;
      NBO_LMindx2SMindx[j] = j;
      NBO_LMindx2SMposi[j] = 0;

      posi1 = posi2;
    }

    tmp_max_num = Num_NBO_FSCenter;

    for (i=0; i<Num_NBO_FSCenter; i++){
      Gc_AN0 = NBO_FSCenter[i];
      tmp_num1 = FNAN[Gc_AN0] + SNAN[Gc_AN0];
      counter = 0;
      Cwan = WhatSpecies[Gc_AN0];
      posi3 = Spe_Total_CNO[Cwan];

    for (j=1; j<=tmp_num1; j++){
      Gh_AN0 = natn[Gc_AN0][j];
      Gh_AN1 = NBO_FSCenter_GAN[i][j];
      Cwan = WhatSpecies[Gh_AN0];
      posi4 = posi3 + Spe_Total_CNO[Cwan];
      po = 0;

      for (k=0; k<tmp_max_num; k++){
        if (Gh_AN1 == NBO_LMindx2LMAtom[k]){
          po = 1;
          break;
        }
      }

      if (po == 0){
        posi2 = posi1 + Spe_Total_CNO[Cwan];

        NBO_LMindx2LMAtom[tmp_max_num + counter] = Gh_AN1;
        NBO_LMindx2LMatom[tmp_max_num + counter] = Gh_AN0;
        NBO_LMindx2LMposi[tmp_max_num + counter] = posi1;
        NBO_LMindx2SMindx[tmp_max_num + counter] = i;
        NBO_LMindx2SMposi[tmp_max_num + counter] = posi3;

        counter++;
      }

      posi1 = posi2;
      posi3 = posi4;

    }

      tmp_max_num += counter;

    } 

    NBO_LMat_size = tmp_max_num;

/*
printf("<< NBO Prep (1) >> NBO_LMat_size = %d List_YOUSO[7] = %d\n",NBO_LMat_size,List_YOUSO[7]);

for (i=0; i<NBO_LMat_size; i++){
 printf(" ### i= %4d : LMAtom= %4d LMatom= %4d LMposi= %4d SMindx= %4d SMposi= %4d \n",
   i,NBO_LMindx2LMAtom[i],NBO_LMindx2LMatom[i],NBO_LMindx2LMposi[i],
     NBO_LMindx2SMindx[i],NBO_LMindx2SMposi[i]);
}
*/

  } /* if (myid == Host_ID) */


    MPI_Bcast(&NBO_LMat_size, 1, MPI_INT, Host_ID, mpi_comm_level1);
    MPI_Bcast(&NBO_LMindx2LMatom[0], NBO_LMat_size, MPI_INT, Host_ID, mpi_comm_level1);

    int ii;
    int *NBO_SMsize, *NBO_SMsize_tmp, NBO_LMsize, NBO_LVsize;
    int *NBO_SVsize, *NBO_SVsize_tmp;

    NBO_SMsize     = (int*)malloc(sizeof(int)*Num_NBO_FSCenter);
    NBO_SMsize_tmp = (int*)malloc(sizeof(int)*Num_NBO_FSCenter);

    NBO_SVsize     = (int*)malloc(sizeof(int)*Num_NBO_FSCenter);
    NBO_SVsize_tmp = (int*)malloc(sizeof(int)*Num_NBO_FSCenter);


   /** NBO_SMsize **/
   for (Mc_AN0=1; Mc_AN0<=Matomnum; Mc_AN0++){
     Gc_AN0 = M2G[Mc_AN0];
     for (i=0; i<Num_NBO_FSCenter; i++){
       if (Gc_AN0 == NBO_FSCenter[i]){
         NBO_SMsize_tmp[i] = Msize2[Mc_AN0];
       }
     }
   }

   for (i=0; i<Num_NBO_FSCenter; i++){
     i2ID = G2ID[NBO_FSCenter[i]];
     if (myid == i2ID){
       MPI_Isend(&NBO_SMsize_tmp[i], 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
     }
     if (myid == Host_ID){
       MPI_Recv(&NBO_SMsize[i], 1, MPI_INT, i2ID, tag, mpi_comm_level1, &stat);
     }
     if (myid == i2ID) MPI_Wait(&request,&stat);
   }

   MPI_Bcast(&NBO_SMsize[0], Num_NBO_FSCenter, MPI_INT, Host_ID, mpi_comm_level1);


   /** NBO_SVsize **/
   for (Mc_AN0=1; Mc_AN0<=Matomnum; Mc_AN0++){
     Gc_AN0 = M2G[Mc_AN0];
     for (i=0; i<Num_NBO_FSCenter; i++){
       if (Gc_AN0 == NBO_FSCenter[i]){
         Cwan  = WhatSpecies[Gc_AN0];
         leng1 = Spe_Total_CNO[Cwan];
         NBO_SVsize_tmp[i] = leng1;
       }
     }
   }

   for (i=0; i<Num_NBO_FSCenter; i++){
     i2ID = G2ID[NBO_FSCenter[i]];
     if (myid == i2ID){
       MPI_Isend(&NBO_SVsize_tmp[i], 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
     }
     if (myid == Host_ID){
       MPI_Recv(&NBO_SVsize[i], 1, MPI_INT, i2ID, tag, mpi_comm_level1, &stat);
     }
     if (myid == i2ID) MPI_Wait(&request,&stat);
   }

    MPI_Bcast(&NBO_SVsize[0], Num_NBO_FSCenter, MPI_INT, Host_ID, mpi_comm_level1);


   int NBO_SMsize_max=0;
   for (i=0; i<Num_NBO_FSCenter; i++){
     i2ID = G2ID[NBO_FSCenter[i]];
     if(NBO_SMsize_max < NBO_SMsize[i]) NBO_SMsize_max = NBO_SMsize[i];
   }

   int NBO_SVsize_max=0;
   for (i=0; i<Num_NBO_FSCenter; i++){
     i2ID = G2ID[NBO_FSCenter[i]];
     if(NBO_SVsize_max < NBO_SVsize[i]) NBO_SVsize_max = NBO_SVsize[i];
   }

    MP = (int*)malloc(sizeof(int)*List_YOUSO[2]);

  /***********************************************
            Main Loop of Calculation (2)
  ***********************************************/

  for (spin=0; spin<=SpinP_switch; spin++){ 
/*  for (spin=0; spin<=SpinP_switch-1; spin++){ */ 
/*  for (spin=1; spin<=SpinP_switch; spin++){ */ 

      size1 = NBO_SMsize_max;
      leng1 = NBO_SVsize_max;

    int *Table_M2G0;
    double *NP_cluster0, *NP_atom0;

    Table_M2G0 = (int*)malloc(sizeof(int)*(Matomnum+1));
    for (i=0; i<=Matomnum; i++){
      Table_M2G0[i] = 0;
    }

    NP_cluster0 = (double*)malloc(sizeof(double)*(Matomnum+1));
    for (i=0; i<=Matomnum; i++){
      NP_cluster0[i] = 0.0;
    }

    NP_atom0 = (double*)malloc(sizeof(double)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      NP_atom0[i] = 0.0;
    }

    NBO_CDMn_tmp = (double**)malloc(sizeof(double*)*size1);
    for (j=0; j<size1; j++){
      NBO_CDMn_tmp[j] = (double*)malloc(sizeof(double)*size1);
      for (k=0; k<size1; k++){
        NBO_CDMn_tmp[j][k] = 0.0;
      }
    }

    NBO_OLPn_tmp = (double**)malloc(sizeof(double*)*size1);
    for (j=0; j<size1; j++){
      NBO_OLPn_tmp[j] = (double*)malloc(sizeof(double)*size1);
      for (k=0; k<size1; k++){
        NBO_OLPn_tmp[j][k] = 0.0;
      }
    }

    NBO_Fock_tmp = (double**)malloc(sizeof(double*)*size1);
    for (j=0; j<size1; j++){
      NBO_Fock_tmp[j] = (double*)malloc(sizeof(double)*size1);
      for (k=0; k<size1; k++){
        NBO_Fock_tmp[j][k] = 0.0;
      }
    }

    NAO_vec_tmp = (double***)malloc(sizeof(double**)*Num_NBO_FSCenter);
    for (i=0; i<Num_NBO_FSCenter; i++){
      NAO_vec_tmp[i] = (double**)malloc(sizeof(double*)*size1);
      for (j=0; j<size1; j++){
        NAO_vec_tmp[i][j] = (double*)malloc(sizeof(double)*leng1);
        for (k=0; k<leng1; k++){
          NAO_vec_tmp[i][j][k] = 0.0;
        }
      }
    }
  
    NAO_partial_pop = (double**)malloc(sizeof(double*)*Num_NBO_FSCenter);
    for (i=0; i<Num_NBO_FSCenter; i++){
      NAO_partial_pop[i] = (double*)malloc(sizeof(double)*(NBO_SVsize_max+1));
      for (j=0; j<=NBO_SVsize_max; j++){
        NAO_partial_pop[i][j] = 0.0;
      }
    }

    NAO_ene_level = (double**)malloc(sizeof(double*)*Num_NBO_FSCenter);
    for (i=0; i<Num_NBO_FSCenter; i++){
      NAO_ene_level[i] = (double*)malloc(sizeof(double)*(NBO_SVsize_max+1));
      for (j=0; j<=NBO_SVsize_max; j++){
        NAO_ene_level[i][j] = 0.0;
      }
    }


  /***********************************************
            Main Loop of Calculation (2)
  ***********************************************/

   if (myid == Host_ID){

     printf("<< 1 >> NAO calculation\n");

   if (SpinP_switch==1 && spin==0){

     printf("\n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("                   for up-spin                   \n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("\n");fflush(stdout);

   }
   else if (SpinP_switch==1 && spin==1){

     printf("\n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("                   for down-spin                 \n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("\n");fflush(stdout);

   }
#if 0
   if (SpinP_switch==0){
     printf("\n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("                STAGE-1: NAO & NPA               \n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("\n");fflush(stdout);
   }
   else if (SpinP_switch==1 && spin==0){
     printf("\n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("        STAGE-1: NAO & NPA (for up-spin)         \n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("\n");fflush(stdout);
   }
   else if (SpinP_switch==1 && spin==1){
     printf("\n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("        STAGE-1: NAO & NPA (for down-spin)       \n");fflush(stdout);
     printf(" *********************************************** \n");fflush(stdout);
     printf("\n");fflush(stdout);
   }
#endif
   }

  for (Mc_AN0=1; Mc_AN0<=Matomnum; Mc_AN0++){
    Gc_AN0 = M2G[Mc_AN0];

    po = 0;
    for (j=0; j<Num_NBO_FSCenter; j++){
      if (Gc_AN0 == NBO_FSCenter[j]){
        po = 1;
        ii = j;
      }
    }

   if (po == 1){

 /* MP array */

    Anum = 1;
    for (i=0; i<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); i++){
      MP[i] = Anum;
      Gi = natn[Gc_AN0][i];
      wanA = WhatSpecies[Gi];
      Anum += Spe_Total_CNO[wanA];
    }

        /****************************************************
               prepalation of matrices H and S for NAO
        ****************************************************/

        for (i=0; i<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; i++){
          ig = natn[Gc_AN0][i];
          ian = Spe_Total_CNO[WhatSpecies[ig]];
          Anum = MP[i]-1;
          ih = S_G2M[ig]; /* S_G2M should be used */

          for (j=0; j<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; j++){

            kl = RMI1[Mc_AN0][i][j];
            jg = natn[Gc_AN0][j];
            jan = Spe_Total_CNO[WhatSpecies[jg]];
            Bnum = MP[j]-1;

            if (0<=kl){
              for (m=0; m<ian; m++){
              for (n=0; n<jan; n++){
                NBO_Fock_tmp[Anum+m][Bnum+n] = Hks[spin][ih][kl][m][n];
                NBO_OLPn_tmp[Anum+m][Bnum+n] =      OLP0[ih][kl][m][n];
              }
              }
            }
            else{
              for (m=0; m<ian; m++){
              for (n=0; n<jan; n++){
                NBO_Fock_tmp[Anum+m][Bnum+n] = 0.0;
                NBO_OLPn_tmp[Anum+m][Bnum+n] = 0.0;
              }
              }
            }
          }
        }

        /****************************************************
               construction of density matrix for NAO
        ****************************************************/

       for (i=0; i<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; i++){
          ig = natn[Gc_AN0][i];
          ian = Spe_Total_CNO[WhatSpecies[ig]];
          Anum = MP[i]-1;
          ih = S_G2M[ig]; /* S_G2M should be used */

          for (j=0; j<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; j++){
            kl = RMI1[Mc_AN0][i][j];
            jg = natn[Gc_AN0][j];
            jan = Spe_Total_CNO[WhatSpecies[jg]];
            Bnum = MP[j]-1;

            if (0<=kl){
              for (m=0; m<ian; m++){
              for (n=0; n<jan; n++){
                NBO_CDMn_tmp[Anum+m][Bnum+n] = CDMn0[spin][ih][kl][m][n];
              }
              }
            }
            else{
              for (m=0; m<ian; m++){
              for (n=0; n<jan; n++){
                NBO_CDMn_tmp[Anum+m][Bnum+n] = 0.0;
              }
              }
            }
          }
        }


  /*************************************************
          Preparation for NBO Calc. (2):
    Search for neighbour atoms of NBO-center atoms
  **************************************************/

        /***********************************************
                        NAO calculation
        ***********************************************/

    int FS_atomnum;

    FS_atomnum = FNAN[Gc_AN0] + SNAN[Gc_AN0];
#if 0
   if (myid == Host_ID){
     printf("\n### Mc_AN0 = %d / Gc_AN0 = %d / Msize = %d / Msize2 = %d / Msize3 = %d \n",
           Mc_AN0,Gc_AN0,Msize[Mc_AN0],Msize2[Mc_AN0],Msize3[Mc_AN0]);
     printf(" # Mc_AN0 / Matomnum = : %d / %d \n",Mc_AN0,Matomnum);
     printf(" # FNAN[%d], SNAN[%d] = : %d. %d \n",Gc_AN0,Gc_AN0,FNAN[Gc_AN0],SNAN[Gc_AN0]);
     printf(" # FS_atomnum, SizeMat = : %d, %d \n",FS_atomnum,Msize2[Mc_AN0]);
     printf(" # Global # of neigh. atoms:");
     for (i=1; i<=FS_atomnum; i++){
       printf(" %d",natn[Gc_AN0][i]);
     }
     printf("\n\n");
   }
#endif

    Calc_NAO( ii, Gc_AN0, NBO_CDMn_tmp, NBO_OLPn_tmp, NBO_Fock_tmp, 
              FS_atomnum, Msize2[Mc_AN0], &NP_cluster0[Mc_AN0], &NP_atom0[Mc_AN0] ); 

    Table_M2G0[Mc_AN0] = Gc_AN0;

    } /* if(po==1) (if Mc_AN is NBO_FSCenter) */

  } /* Main loop (2) (Mc_AN) */

/*
  for (i=0; i<NBO_SMsize_max; i++){
    free(NBO_CDMn_tmp[i]);
  }
  free(NBO_CDMn_tmp);

  for (i=0; i<NBO_SMsize_max; i++){
    free(NBO_OLPn_tmp[i]);
  }
  free(NBO_OLPn_tmp);

  for (i=0; i<NBO_SMsize_max; i++){
    free(NBO_Fock_tmp[i]);
  }
  free(NBO_Fock_tmp);
*/
/*
  MPI_Barrier(mpi_comm_level1);
*/
/*
printf("$$$$$$ End of Main loop (2) : ID = %d $$$$$$\n",myid);
*/

   /***********************************************
         Summalize NAO results at each process
   ***********************************************/

    int *ID2atomnump, *ID2atomnum, *ID2atomposi, *Table_M2G1;
    double *Tempvec, *NP_cluster1,*NP_atom1,Total_NP;
    double *NAO_vec_tmp1, *NAO_vec_tmp2;

    ID2atomnump = (int*)malloc(sizeof(int)*(numprocs));
    for (i=0; i<numprocs; i++){
      ID2atomnump[i] = 0;
    }

    ID2atomnum  = (int*)malloc(sizeof(int)*(numprocs));
    for (i=0; i<numprocs; i++){
      ID2atomnum[i] = 0;
    }

    ID2atomposi = (int*)malloc(sizeof(int)*(numprocs));
    for (i=0; i<numprocs; i++){
      ID2atomposi[i] = 0;
    }

    Table_M2G1 = (int*)malloc(sizeof(int)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      Table_M2G1[i] = 0;
    }

    Tempvec = (double*)malloc(sizeof(double)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      Tempvec[i] = 0.0;
    }

    NP_cluster1 = (double*)malloc(sizeof(double)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      NP_cluster1[i] = 0.0;
    }

    NP_atom1 = (double*)malloc(sizeof(double)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      NP_atom1[i] = 0.0;
    }

    NAO_vec_tmp1 = (double*)malloc(sizeof(double)*(Num_NBO_FSCenter*(NBO_SVsize_max+1)));
    for (j=0; j<(Num_NBO_FSCenter*(NBO_SVsize_max+1)); j++){
      NAO_vec_tmp1[j] = 0.0;
    }

    NAO_vec_tmp2 = (double*)malloc(sizeof(double)*(Num_NBO_FSCenter*(NBO_SVsize_max+1)));
    for (j=0; j<(Num_NBO_FSCenter*(NBO_SVsize_max+1)); j++){
      NAO_vec_tmp2[j] = 0.0;
    }


    if(myid==Host_ID) printf("<< 2 >> Sending & Receiving atom data\n");

    ID2atomnump[myid] = Matomnum;

    MPI_Gather(&ID2atomnump[myid],1,MPI_INT, &ID2atomnum[0],1,MPI_INT, Host_ID, mpi_comm_level1);

    if (myid==Host_ID){
      for (i=1; i<numprocs; i++){
        ID2atomposi[i] = ID2atomposi[i-1] + ID2atomnum[i-1];
      }
    }

    MPI_Gatherv(&Table_M2G0[1], Matomnum, MPI_INT,
                &Table_M2G1[1], ID2atomnum, ID2atomposi, MPI_INT, Host_ID, mpi_comm_level1);

    if(myid==Host_ID) printf("<< 3 >> Collection of NAO data\n\n");
/*
    if(1<0){
    MPI_Gatherv(&NP_cluster0[1], Matomnum, MPI_DOUBLE,
                &NP_cluster1[1], ID2atomnum, ID2atomposi, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    }
*/
    MPI_Gatherv(&NP_atom0[1], Matomnum, MPI_DOUBLE,
                &NP_atom1[1], ID2atomnum, ID2atomposi, MPI_DOUBLE, Host_ID, mpi_comm_level1);


  /* Transportation of energy levels of each NAO */
   for (i=0; i<Num_NBO_FCenter; i++){
     i2ID = G2ID[NBO_FCenter[i]];
     Gh_AN0 = NBO_FCenter[i];
     Cwan  = WhatSpecies[Gh_AN0];
     size1 = Spe_Total_NO[Cwan];

     if (myid == i2ID){
       counter = -1;
       for (j=0; j<size1; j++){
         counter++;
         NAO_vec_tmp1[counter] = NAO_ene_level[i][j];
       }

       MPI_Isend(&NAO_vec_tmp1[0], size1, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
     }

     if (myid == Host_ID){
       MPI_Recv(&NAO_vec_tmp2[0], size1, MPI_DOUBLE, i2ID, tag, mpi_comm_level1, &stat);
     }

     if (myid == i2ID) MPI_Wait(&request,&stat);

     if (myid == Host_ID){
       counter = -1;
       for (j=0; j<size1; j++){
         counter++;
         NAO_ene_level[i][j] = NAO_vec_tmp2[counter];
       }
     }
   }


  /* Transportation of populations in each NAO */
   for (i=0; i<Num_NBO_FCenter; i++){
     i2ID = G2ID[NBO_FCenter[i]];
     Gh_AN0 = NBO_FCenter[i];
     Cwan  = WhatSpecies[Gh_AN0];
     size1 = Spe_Total_NO[Cwan];

     if (myid == i2ID){
       counter = -1;
       for (j=0; j<size1; j++){
         counter++;
         NAO_vec_tmp1[counter] = NAO_partial_pop[i][j];
       }

       MPI_Isend(&NAO_vec_tmp1[0], size1, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
     }

     if (myid == Host_ID){
       MPI_Recv(&NAO_vec_tmp2[0], size1, MPI_DOUBLE, i2ID, tag, mpi_comm_level1, &stat);
     }

     if (myid == i2ID) MPI_Wait(&request,&stat);

     if (myid == Host_ID){
       counter = -1;
       for (j=0; j<size1; j++){
         counter++;
         NAO_partial_pop[i][j] = NAO_vec_tmp2[counter];
       }
     }
   }


    char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
    char *Name_Multiple[20];

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
/*
      for (i=1; i<=atomnum; i++){
        printf("$$$$$ %5d : %d \n",i,Table_M2G1[i]);
      }
      printf("\n");
      }

      if(1<0){
      for (i=1; i<=atomnum; i++){
        Tempvec[Table_M2G1[i]] = NP_cluster1[i];
      }
      for (i=1; i<=atomnum; i++){
        NP_cluster1[i] = Tempvec[i];
      }
*/
      for (i=1; i<=atomnum; i++){
        Tempvec[Table_M2G1[i]] = NP_atom1[i];
      }
      for (i=1; i<=atomnum; i++){
        NP_atom1[i] = Tempvec[i];
      }
/*
      printf("### Total natural pop. for each truncated cluster ###\n");
      for (i=1; i<=atomnum; i++){
        printf(" %5d : %9.4f \n",i,NP_cluster1[i]);
      }
      printf("\n");
*/
      printf("### Natural populations of target atoms ###\n");

      for (i=0; i<Num_NBO_FCenter; i++){
        i2ID = G2ID[NBO_FCenter[i]];
        Gh_AN0 = NBO_FCenter[i];
        Cwan  = WhatSpecies[Gh_AN0];

        printf(" (%d) %5d %3s : %12.8f \n",i+1,Gh_AN0,SpeName[Cwan],NP_atom1[Gh_AN0]);

      }
      printf("\n");

#if 0
      printf("### Natural population of target & second neghbour atoms ###\n");

      for (i=0; i<Num_NBO_FSCenter; i++){
        i2ID = G2ID[NBO_FSCenter[i]];
        Gh_AN0 = NBO_FSCenter[i];
        Cwan  = WhatSpecies[Gh_AN0];

        printf(" (%d) %5d %3s : %12.8f \n",i+1,Gh_AN0,SpeName[Cwan],NP_atom1[Gh_AN0]);

      }
      printf("\n");
#endif

      Total_NP = 0.0;
      for (i=1; i<=atomnum; i++){
        Total_NP += NP_atom1[i];
      }

      printf("### Total natural pop. of target atoms = %12.5f \n\n",Total_NP);

#if 0
      printf("### NAO for each target atom (details) ###\n");

      for (i=0; i<Num_NBO_FCenter; i++){
        Gc_AN = NBO_FCenter[i];
        wan1 = WhatSpecies[Gc_AN];
        printf(" ## Global atom num.: %d ( %s )\n",Gc_AN,SpeName[wan1]);
        printf(" ---------------------------------------\n");
        printf("  NAO      Energy (Hartree)  Population \n");
        printf(" ---------------------------------------\n");
        k = 0;
      for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Spe_Num_Basis[wan1][L1];
      for (M1=1; M1<=2*L1+1; M1++){
      for (j=0; j<leng1; j++){
        printf(" %s %3s   ",Name_Multiple[j+1],Name_Angular[L1][M1-1]);
        printf("%9.5f %9.5f\n",NAO_ene_level[i][k],NAO_partial_pop[i][k]);
        k++;
      } /* j */
      } /* M1 */
      } /* L1 */
        printf(" -------------------------------------\n\n");
      } /* i */
      printf("\n");
#endif
   }

   free(NAO_vec_tmp1);
   free(NAO_vec_tmp2);


   double *NBO_vec_tmp1, *NBO_vec_tmp2;

    NBO_LMsize = 0;
    for (i1=0; i1<NBO_LMat_size; i1++){
      jg = NBO_LMindx2LMatom[i1];
      Cwan  = WhatSpecies[jg];
      NBO_LMsize += Spe_Total_CNO[Cwan];
    }

    MPI_Bcast(&NBO_LMsize, 1, MPI_INT, Host_ID, mpi_comm_level1);


   NBO_vec_tmp1 = (double*)malloc(sizeof(double)*(NBO_LMsize*NBO_LMsize));
   for (j=0; j<(NBO_LMsize*NBO_LMsize); j++){
     NBO_vec_tmp1[j] = 0.0;
   }

   NBO_vec_tmp2 = (double*)malloc(sizeof(double)*(NBO_LMsize*NBO_LMsize));
   for (j=0; j<(NBO_LMsize*NBO_LMsize); j++){
     NBO_vec_tmp2[j] = 0.0;
   }

   for (i=0; i<Num_NBO_FSCenter; i++){
     i2ID = G2ID[NBO_FSCenter[i]];
     Gh_AN0 = NBO_FSCenter[i];
     Cwan  = WhatSpecies[Gh_AN0];
     leng1 = Spe_Total_CNO[Cwan];
     size1 = NBO_SMsize[i];
     size2 = size1 * leng1;

     if (myid == i2ID){
       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<leng1; k++){
         counter++;
         NBO_vec_tmp1[counter] = NAO_vec_tmp[i][j][k];
       }
       }

       MPI_Isend(&NBO_vec_tmp1[0], size2, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
     }

     if (myid == Host_ID){
       MPI_Recv(&NBO_vec_tmp2[0], size2, MPI_DOUBLE, i2ID, tag, mpi_comm_level1, &stat);
     }

     if (myid == i2ID) MPI_Wait(&request,&stat);

     if (myid == Host_ID){
       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<leng1; k++){
         counter++;
         NAO_vec_tmp[i][j][k] = NBO_vec_tmp2[counter];
       }
       }
     }
   }

   free(NBO_vec_tmp1);
   free(NBO_vec_tmp2);


  if (myid == Host_ID){

   printf("### NAO for each target atom (details) ###\n");

   for (i=0; i<Num_NBO_FCenter; i++){
     k = 0;
     i2ID = G2ID[NBO_FCenter[i]];
     Gh_AN0 = NBO_FSCenter[i];
     Cwan  = WhatSpecies[Gh_AN0];

     leng1 = Spe_Total_CNO[Cwan];
     size1 = NBO_SMsize[i];

     printf(" #### Global atom num.: %d ( %s ) / NP = %8.4f\n",
            Gh_AN0,SpeName[Cwan],NP_atom1[Gh_AN0]);fflush(stdout);

     printf("-----------------");
      for (j=0; j<leng1; j++){
        printf("--------");
      } /* j */
      printf("\n");

     printf(" NP in NAO       ");fflush(stdout);
      for (j=0; j<leng1; j++){
        printf("%8.4f",NAO_partial_pop[i][j]);
      } /* j */
      printf("\n");

     printf(" Energy (Hartree)");fflush(stdout);
      for (j=0; j<leng1; j++){
        printf("%8.4f",NAO_ene_level[i][j]);
      } /* j */
      printf("\n");

     printf("-----------------");
      for (j=0; j<leng1; j++){
        printf("--------");
      } /* j */
      printf("\n");

      for (L1=0; L1<=Spe_MaxL_Basis[Cwan]; L1++){
      for (M1=1; M1<=2*L1+1; M1++){
        leng2 = Spe_Num_Basis[Cwan][L1];
      for (j=0; j<leng2; j++){
        printf(" %s %3s   ",Name_Multiple[j+1],Name_Angular[L1][M1-1]);fflush(stdout);
      for (n=0; n<leng1; n++){
        printf("%8.4f",NAO_vec_tmp[i][k][n]);fflush(stdout);
      } /* n */
        k++;
        printf("\n");fflush(stdout);
      } /* j */
      } /* M1 */
      } /* L1 */
      printf("\n");fflush(stdout);

   }
   }


   for (i=0; i<Num_NBO_FSCenter; i++){
     free(NAO_partial_pop[i]);
   }
   free(NAO_partial_pop);

   for (i=0; i<Num_NBO_FSCenter; i++){
     free(NAO_ene_level[i]);
   }
   free(NAO_ene_level);



    free(ID2atomnump);
    free(ID2atomnum);
    free(ID2atomposi);

    free(Table_M2G0);
    free(Table_M2G1);

    free(Tempvec);
    free(NP_cluster0);
    free(NP_cluster1);
    free(NP_atom0);
    free(NP_atom1);


  /*************************************************
          Preparation for NBO Calc. (3):
          Transportation of matrix data 
  **************************************************/

  if(NAO_only==0){

    double **NBO_CDMn_full, **NBO_OLPn_full, **NBO_Fock_full, **NAO_vec_full;
    double *NBO_vec_tmp3;
/*
    double *NBO_vec_tmp1, *NBO_vec_tmp2, *NBO_vec_tmp3;
*/
    NBO_LMsize = 0;
    for (i1=0; i1<NBO_LMat_size; i1++){
      jg = NBO_LMindx2LMatom[i1];
      Cwan  = WhatSpecies[jg];
      NBO_LMsize += Spe_Total_CNO[Cwan];
    }

    MPI_Bcast(&NBO_LMsize, 1, MPI_INT, Host_ID, mpi_comm_level1);

    NBO_LVsize = 0;
    for (i1=0; i1<Num_NBO_FSCenter; i1++){
      jg = NBO_LMindx2LMatom[i1];
      Cwan  = WhatSpecies[jg];
      NBO_LVsize += Spe_Total_CNO[Cwan];
    }

    MPI_Bcast(&NBO_LVsize, 1, MPI_INT, Host_ID, mpi_comm_level1);


   NBO_vec_tmp1 = (double*)malloc(sizeof(double)*(NBO_LMsize*NBO_LMsize));
   for (j=0; j<(NBO_LMsize*NBO_LMsize); j++){
     NBO_vec_tmp1[j] = 0.0;
   }

   NBO_vec_tmp2 = (double*)malloc(sizeof(double)*(NBO_LMsize*NBO_LMsize));
   for (j=0; j<(NBO_LMsize*NBO_LMsize); j++){
     NBO_vec_tmp2[j] = 0.0;
   }

   NBO_vec_tmp3 = (double*)malloc(sizeof(double)*(NBO_LMsize*NBO_LMsize));
   for (j=0; j<(NBO_LMsize*NBO_LMsize); j++){
     NBO_vec_tmp3[j] = 0.0;
   }

/*
   for (i=0; i<Num_NBO_FSCenter; i++){
     i2ID = G2ID[NBO_FSCenter[i]];
     Gh_AN0 = NBO_FSCenter[i];
     Cwan  = WhatSpecies[Gh_AN0];
     leng1 = Spe_Total_CNO[Cwan];
     size1 = NBO_SMsize[i];
     size2 = size1 * leng1;

     if (myid == i2ID){
       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<leng1; k++){
         counter++;
         NBO_vec_tmp1[counter] = NAO_vec_tmp[i][j][k];
       }
       }

       MPI_Isend(&NBO_vec_tmp1[0], size2, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
     }

     if (myid == Host_ID){
       MPI_Recv(&NBO_vec_tmp2[0], size2, MPI_DOUBLE, i2ID, tag, mpi_comm_level1, &stat);
     }

     if (myid == i2ID) MPI_Wait(&request,&stat);

     if (myid == Host_ID){
       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<leng1; k++){
         counter++;
         NAO_vec_tmp[i][j][k] = NBO_vec_tmp2[counter];
       }
       }
     }
   }

*/


    NBO_CDMn_full = (double**)malloc(sizeof(double*)*NBO_LMsize);
    for (i=0; i<NBO_LMsize; i++){
      NBO_CDMn_full[i] = (double*)malloc(sizeof(double)*NBO_LMsize);
      for (j=0; j<NBO_LMsize; j++){
        NBO_CDMn_full[i][j] = 0.0;
      }
    }

    NBO_OLPn_full = (double**)malloc(sizeof(double*)*NBO_LMsize);
    for (i=0; i<NBO_LMsize; i++){
      NBO_OLPn_full[i] = (double*)malloc(sizeof(double)*NBO_LMsize);
      for (j=0; j<NBO_LMsize; j++){
        NBO_OLPn_full[i][j] = 0.0;
      }
    }

    NAO_vec_full = (double**)malloc(sizeof(double*)*NBO_LMsize);
    for (i=0; i<NBO_LMsize; i++){
      NAO_vec_full[i] = (double*)malloc(sizeof(double)*NBO_LVsize);
      for (j=0; j<NBO_LVsize; j++){
        NAO_vec_full[i][j] = 0.0;
      }
    }

   /* Full Density & Overlap Matrices for NBO Calc. */

    for (ii=0; ii<Num_NBO_FSCenter; ii++){
      Gc_AN0 = NBO_FSCenter[ii];
      i2ID = G2ID[Gc_AN0];
      size1 = NBO_SMsize[ii];
      size2 = size1*size1;

      Anum = 1;
      for (i=0; i<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); i++){
        MP[i] = Anum;
        Gi = natn[Gc_AN0][i];
        wanA = WhatSpecies[Gi];
        Anum += Spe_Total_CNO[wanA];
      }
/*
      tmp_num2 = 0;
      for (i=0; i<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); i++){
        Gi = natn[Gc_AN0][i];
        wanA = WhatSpecies[Gi];
      for (L1=0; L1<Spe_MaxL_Basis[wanA]; L1++){
        Posi_AL[i][L1] = tmp_num2;
        tmp_num2 += Spe_Num_Basis[wanA][L1] * (2*L1+1);
      }
      }
*/
     if (myid == i2ID){

       Mc_AN0 = S_G2M[Gc_AN0];

       for (i=0; i<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; i++){
         ig = natn[Gc_AN0][i];
         ian = Spe_Total_CNO[WhatSpecies[ig]];
         Anum = MP[i]-1;
         ih = S_G2M[ig]; /* S_G2M should be used */

         for (j=0; j<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; j++){
           kl = RMI1[Mc_AN0][i][j];
           jg = natn[Gc_AN0][j];
           jan = Spe_Total_CNO[WhatSpecies[jg]];
           Bnum = MP[j]-1;

           if (0<=kl){
             for (m=0; m<ian; m++){
             for (n=0; n<jan; n++){
               NBO_CDMn_tmp[Anum+m][Bnum+n] = CDMn0[spin][ih][kl][m][n];
             }
             }
           }
           else{
             for (m=0; m<ian; m++){
             for (n=0; n<jan; n++){
               NBO_CDMn_tmp[Anum+m][Bnum+n] = 0.0;
             }
             }
           }
         }
       }

       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<size1; k++){
         counter++;
         NBO_vec_tmp1[counter] = NBO_CDMn_tmp[j][k];
       }
       }

       MPI_Isend(&NBO_vec_tmp1[0], size2, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);

     } /* myid == i2ID */

     if (myid == Host_ID){
       MPI_Recv(&NBO_vec_tmp2[0], size2, MPI_DOUBLE, i2ID, tag, mpi_comm_level1, &stat);
     }

     if (myid == i2ID) MPI_Wait(&request,&stat);

     if (myid==Host_ID){
       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<size1; k++){
         counter++;
         NBO_CDMn_tmp[j][k] = NBO_vec_tmp2[counter];
       }
       }

     } /* if myid == Host_ID */

     if (myid == i2ID){

       Mc_AN0 = S_G2M[Gc_AN0];

       for (i=0; i<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; i++){
         ig = natn[Gc_AN0][i];
         ian = Spe_Total_CNO[WhatSpecies[ig]];
         Anum = MP[i]-1;
         ih = S_G2M[ig]; /* S_G2M should be used */

         for (j=0; j<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; j++){
           kl = RMI1[Mc_AN0][i][j];
           jg = natn[Gc_AN0][j];
           jan = Spe_Total_CNO[WhatSpecies[jg]];
           Bnum = MP[j]-1;

           if (0<=kl){
             for (m=0; m<ian; m++){
             for (n=0; n<jan; n++){
               NBO_OLPn_tmp[Anum+m][Bnum+n] = OLP0[ih][kl][m][n];
             }
             }
           }
           else{
             for (m=0; m<ian; m++){
             for (n=0; n<jan; n++){
               NBO_OLPn_tmp[Anum+m][Bnum+n] = 0.0;
             }
             }
           }
         }
       }

       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<size1; k++){
         counter++;
         NBO_vec_tmp1[counter] = NBO_OLPn_tmp[j][k];
       }
       }

       MPI_Isend(&NBO_vec_tmp1[0], size2, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
     }

     if (myid == Host_ID){
       MPI_Recv(&NBO_vec_tmp2[0], size2, MPI_DOUBLE, i2ID, tag, mpi_comm_level1, &stat);
     }

     if (myid == i2ID) MPI_Wait(&request,&stat);

     if (myid==Host_ID){
       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<size1; k++){
         counter++;
         NBO_OLPn_tmp[j][k] = NBO_vec_tmp2[counter];
       }
       }
     } /* if myid == Host_ID */

     if (myid==Host_ID){

      for (j=0; j<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); j++){
        Gh_AN0 = natn[Gc_AN0][j];
        Gh_AN1 = Gh_AN0 + atomnum*ncn[Gc_AN0][j];
        wan1 = WhatSpecies[Gh_AN0];
        posi1 = MP[j]-1;

        for (i1=0; i1<NBO_LMat_size; i1++){
          if (Gh_AN1 == NBO_LMindx2LMAtom[i1]){
            jp = NBO_LMindx2LMposi[i1];
            break;
          }
        }

      for (k=0; k<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); k++){
        Ga_AN0 = natn[Gc_AN0][k];
        Ga_AN1 = Ga_AN0 + atomnum*ncn[Gc_AN0][k];
        wan2 = WhatSpecies[Ga_AN0];
        posi2 = MP[k]-1;

        for (i1=0; i1<NBO_LMat_size; i1++){
          if (Ga_AN1 == NBO_LMindx2LMAtom[i1]){
            kp = NBO_LMindx2LMposi[i1];
            break;
          }
        }

       m = -1; mm = 0;
       for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
         tmp_num1 = Spe_Num_Basis[wan1][L1];
       for (M1=1; M1<=2*L1+1; M1++){
       for (k1=1; k1<=tmp_num1; k1++){
         m += 1;
         n = -1; nn = 0;
         for (L2=0; L2<=Spe_MaxL_Basis[wan2]; L2++){
           tmp_num2 = Spe_Num_Basis[wan2][L2];
         for (M2=1; M2<=2*L2+1; M2++){
         for (k2=1; k2<=tmp_num2; k2++){
           n += 1;

           NBO_CDMn_full[jp + m][kp + n] = NBO_CDMn_tmp[posi1 + mm + (2*L1+1)*(k1-1)+(M1-1)]
                                                       [posi2 + nn + (2*L2+1)*(k2-1)+(M2-1)];
           NBO_OLPn_full[jp + m][kp + n] = NBO_OLPn_tmp[posi1 + mm + (2*L1+1)*(k1-1)+(M1-1)]
                                                       [posi2 + nn + (2*L2+1)*(k2-1)+(M2-1)];

         } /* k2 */
         } /* M2 */
           nn += tmp_num2 * (2*L2+1);
         } /* L2 */

       } /* k1 */
       } /* M1 */
         mm += tmp_num1 * (2*L1+1);
       } /* L1 */

      } /* k */
      } /* j */

     } /* if myid == Host_ID */

    } /* ii */


  for (i=0; i<NBO_SMsize_max; i++){
    free(NBO_CDMn_tmp[i]);
  }
  free(NBO_CDMn_tmp);

  for (i=0; i<NBO_SMsize_max; i++){
    free(NBO_OLPn_tmp[i]);
  }
  free(NBO_OLPn_tmp);

/*
if(myid==Host_ID){
     printf("$$$$ TEST NBO_CDMn_full $$$$\n");
     for (i=0; i<11; i++){
     for (j=0; j<11; j++){
       printf("%9.5f",NBO_CDMn_full[i][j]);
     }
       printf("\n");
     }
       printf("\n");

     printf("$$$$ TEST NBO_OLPn_full $$$$\n");
     for (i=0; i<11; i++){
     for (j=0; j<11; j++){
       printf("%9.5f",NBO_OLPn_full[i][j]);
     }
       printf("\n");
     }
       printf("\n");
}
*/

    /* NAO-Vectors for NBO calc. */

 if (myid==Host_ID){

      for (i=0; i<Num_NBO_FSCenter; i++){
        Gc_AN0 = NBO_FSCenter[i];
        Gc_AN1 = NBO_FSCenter_GAN[i][0];
        Cwan = WhatSpecies[Gc_AN0];
        leng1 = Spe_Total_CNO[Cwan];

        for (i1=0; i1<NBO_LMat_size; i1++){
          if (Gc_AN1 == NBO_LMindx2LMAtom[i1]){
            jp = NBO_LMindx2LMposi[i1];
            break;
          }
        }

      posi3 = 0;

      for (k=0; k<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); k++){
        Gh_AN0 = natn[Gc_AN0][k];
        Gh_AN1 = Gh_AN0 + atomnum*ncn[Gc_AN0][k];
        Cwan = WhatSpecies[Gh_AN0];
        leng2 = Spe_Total_CNO[Cwan];
        posi4 = posi3 + Spe_Total_CNO[Cwan];

        for (i1=0; i1<NBO_LMat_size; i1++){
          if (Gh_AN1 == NBO_LMindx2LMAtom[i1]){
            kp = NBO_LMindx2LMposi[i1];
            break;
          }
        }

        for (m=0; m<leng1; m++){
        for (n=0; n<leng2; n++){
          NAO_vec_full[kp + n][jp + m] = NAO_vec_tmp[i][posi3 + n][m];
        }
        }
        posi3 = posi4;
      }
      }

 } /* if myid==Host_ID */


   for (i=0; i<Num_NBO_FSCenter; i++){
   for (j=0; j<NBO_SMsize_max; j++){
     free(NAO_vec_tmp[i][j]);
   }
     free(NAO_vec_tmp[i]);
   }
   free(NAO_vec_tmp);

  if(myid==Host_ID){

    /* Constructon of P_NAO (= S * D * S) */
/*
      for (i=0; i<NBO_LMsize; i++){
      for (j=0; j<NBO_LMsize; j++){
        tmp_ele1 = 0.0;
        for (k=0; k<NBO_LMsize; k++){
          tmp_ele1 += NBO_CDMn_full[i][k] * NBO_OLPn_full[k][j];
        }
        tmp_M1[i][j] = tmp_ele1;
      }
      }

      for (i=0; i<NBO_LMsize; i++){
      for (j=0; j<NBO_LMsize; j++){
        tmp_ele1 = 0.0;
        for (k=0; k<NBO_LMsize; k++){
          tmp_ele1 += NBO_OLPn_full[i][k] * tmp_M1[k][j];
        }
        NBO_CDMn_full[i][j] = tmp_ele1;
      }
      }
*/

   /* BLAS */
     num = 0;
     for (i=0; i<NBO_LMsize; i++){
     for (j=0; j<NBO_LMsize; j++){
       NBO_vec_tmp1[num] = NBO_OLPn_full[j][i];
       NBO_vec_tmp2[num] = NBO_CDMn_full[i][j];
       num++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = NBO_LMsize; N = NBO_LMsize; K = NBO_LMsize;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           NBO_vec_tmp2, &lda, NBO_vec_tmp1, &ldb, &beta, NBO_vec_tmp3, &ldc);
     alpha = 1.0; beta = 0.0;
     M = NBO_LMsize; N = NBO_LMsize; K = NBO_LMsize;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           NBO_vec_tmp1, &lda, NBO_vec_tmp3, &ldb, &beta, NBO_vec_tmp2, &ldc);

     num = 0;
     for (i=0; i<NBO_LMsize; i++){
     for (j=0; j<NBO_LMsize; j++){
       NBO_CDMn_full[i][j]  = NBO_vec_tmp2[num];
       num++;
     }
     }


  } /* myid==Host_ID */


 for (i=0; i<NBO_LMsize; i++){
    free(NBO_OLPn_full[i]);
  }
  free(NBO_OLPn_full);


    NBO_Fock_full = (double**)malloc(sizeof(double*)*NBO_LMsize);
    for (i=0; i<NBO_LMsize; i++){
      NBO_Fock_full[i] = (double*)malloc(sizeof(double)*NBO_LMsize);
      for (j=0; j<NBO_LMsize; j++){
        NBO_Fock_full[i][j] = 0.0;
      }
    }


   /* Full Fock Matrix for NBO Calc. */

    for (ii=0; ii<Num_NBO_FSCenter; ii++){
      Gc_AN0 = NBO_FSCenter[ii];
      i2ID = G2ID[Gc_AN0];
      size1 = NBO_SMsize[ii];
      size2 = size1*size1;

      Anum = 1;
      for (i=0; i<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); i++){
        MP[i] = Anum;
        Gi = natn[Gc_AN0][i];
        wanA = WhatSpecies[Gi];
        Anum += Spe_Total_CNO[wanA];
      }

     if (myid == i2ID){

       Mc_AN0 = S_G2M[Gc_AN0];

       for (i=0; i<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; i++){
         ig = natn[Gc_AN0][i];
         ian = Spe_Total_CNO[WhatSpecies[ig]];
         Anum = MP[i]-1;
         ih = S_G2M[ig]; /* S_G2M should be used */

         for (j=0; j<=FNAN[Gc_AN0]+SNAN[Gc_AN0]; j++){
           kl = RMI1[Mc_AN0][i][j];
           jg = natn[Gc_AN0][j];
           jan = Spe_Total_CNO[WhatSpecies[jg]];
           Bnum = MP[j]-1;

           if (0<=kl){
             for (m=0; m<ian; m++){
             for (n=0; n<jan; n++){
               NBO_Fock_tmp[Anum+m][Bnum+n] = Hks[spin][ih][kl][m][n];
             }
             }
           }
           else{
             for (m=0; m<ian; m++){
             for (n=0; n<jan; n++){
               NBO_Fock_tmp[Anum+m][Bnum+n] = 0.0;
             }
             }
           }
         }
       }

       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<size1; k++){
         counter++;
         NBO_vec_tmp1[counter] = NBO_Fock_tmp[j][k];
       }
       }

       MPI_Isend(&NBO_vec_tmp1[0], size2, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
     }

     if (myid == Host_ID){
       MPI_Recv(&NBO_vec_tmp2[0], size2, MPI_DOUBLE, i2ID, tag, mpi_comm_level1, &stat);
     }

     if (myid == i2ID) MPI_Wait(&request,&stat);

     if (myid==Host_ID){

       counter = -1;
       for (j=0; j<size1; j++){
       for (k=0; k<size1; k++){
         counter++;
         NBO_Fock_tmp[j][k] = NBO_vec_tmp2[counter];
       }
       }

      for (j=0; j<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); j++){
        Gh_AN0 = natn[Gc_AN0][j];
        Gh_AN1 = Gh_AN0 + atomnum*ncn[Gc_AN0][j];
        wan1 = WhatSpecies[Gh_AN0];
        posi1 = MP[j]-1;

        for (i1=0; i1<NBO_LMat_size; i1++){
          if (Gh_AN1 == NBO_LMindx2LMAtom[i1]){
            jp = NBO_LMindx2LMposi[i1];
            break;
          }
        }

      for (k=0; k<=(FNAN[Gc_AN0]+SNAN[Gc_AN0]); k++){
        Ga_AN0 = natn[Gc_AN0][k];
        Ga_AN1 = Ga_AN0 + atomnum*ncn[Gc_AN0][k];
        wan2 = WhatSpecies[Ga_AN0];
        posi2 = MP[k]-1;

        for (i1=0; i1<NBO_LMat_size; i1++){
          if (Ga_AN1 == NBO_LMindx2LMAtom[i1]){
            kp = NBO_LMindx2LMposi[i1];
            break;
          }
        }

       m = -1; mm = 0;
       for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
         tmp_num1 = Spe_Num_Basis[wan1][L1];
       for (M1=1; M1<=2*L1+1; M1++){
       for (k1=1; k1<=tmp_num1; k1++){
         m += 1;

         n = -1; nn = 0;
         for (L2=0; L2<=Spe_MaxL_Basis[wan2]; L2++){
           tmp_num2 = Spe_Num_Basis[wan2][L2];
         for (M2=1; M2<=2*L2+1; M2++){
         for (k2=1; k2<=tmp_num2; k2++){
           n += 1;

           NBO_Fock_full[jp + m][kp + n] = NBO_Fock_tmp[posi1 + mm + (2*L1+1)*(k1-1)+(M1-1)]
                                                       [posi2 + nn + (2*L2+1)*(k2-1)+(M2-1)];

         } /* k2 */
         } /* M2 */
           nn += tmp_num2 * (2*L2+1);
         } /* L2 */

       } /* k1 */
       } /* M1 */
         mm += tmp_num1 * (2*L1+1);
       } /* L1 */

      } /* k */
      } /* j */

     } /* if myid == Host_ID  */

    } /* ii */


  for (i=0; i<NBO_SMsize_max; i++){
    free(NBO_Fock_tmp[i]);
  }
  free(NBO_Fock_tmp);


 if (myid == Host_ID){
/*
     printf("$$$$ TEST NBO_Fock_full $$$$\n");
     for (i=0; i<11; i++){
     for (j=0; j<11; j++){
       printf("%9.5f",NBO_Fock_full[i][j]);
     }
       printf("\n");
     }
       printf("\n");
*/

    /* Constructon of P_NAO & F_NAO */
/*
      for (i=0; i<NBO_LMsize; i++){
      for (j=0; j<NBO_LVsize; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
        for (k=0; k<NBO_LMsize; k++){
          tmp_ele1 += NBO_CDMn_full[i][k] * NAO_vec_full[k][j];
          tmp_ele2 += NBO_Fock_full[i][k] * NAO_vec_full[k][j];
        }
        tmp_M1[i][j] = tmp_ele1;
        tmp_M2[i][j] = tmp_ele2;
      }
      }

      for (i=0; i<NBO_LVsize; i++){
      for (j=0; j<NBO_LVsize; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
        for (k=0; k<NBO_LMsize; k++){
          tmp_ele1 += NAO_vec_full[k][i] * tmp_M1[k][j];
          tmp_ele2 += NAO_vec_full[k][i] * tmp_M2[k][j];
        }
        NBO_CDMn_full[i][j] = tmp_ele1;
        NBO_Fock_full[i][j] = tmp_ele2;
      }
      }
*/
   /** BLAS **/
     /* P_NAO */
     num = 0;
     for (i=0; i<NBO_LVsize; i++){
     for (j=0; j<NBO_LMsize; j++){
       NBO_vec_tmp1[num] = NAO_vec_full[j][i];
       num++;
     }
     }

     num = 0;
     for (i=0; i<NBO_LMsize; i++){
     for (j=0; j<NBO_LMsize; j++){
       NBO_vec_tmp2[num] = NBO_CDMn_full[i][j];
       num++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = NBO_LMsize; N = NBO_LVsize; K = NBO_LMsize;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           NBO_vec_tmp2, &lda, NBO_vec_tmp1, &ldb, &beta, NBO_vec_tmp3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = NBO_LVsize; N = NBO_LVsize; K = NBO_LMsize;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("T", "N", &M, &N, &K, &alpha,
                           NBO_vec_tmp1, &lda, NBO_vec_tmp3, &ldb, &beta, NBO_vec_tmp2, &ldc);

     for (i=0; i<NBO_LVsize; i++){
     for (j=0; j<NBO_LVsize; j++){
       NBO_CDMn_full[i][j] = NBO_vec_tmp2[NBO_LMsize*i+j];
     }
     }

     /* F_NAO */
     num = 0;
     for (i=0; i<NBO_LMsize; i++){
     for (j=0; j<NBO_LMsize; j++){
       NBO_vec_tmp2[num] = NBO_Fock_full[i][j];
       num++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = NBO_LMsize; N = NBO_LVsize; K = NBO_LMsize;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           NBO_vec_tmp2, &lda, NBO_vec_tmp1, &ldb, &beta, NBO_vec_tmp3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = NBO_LVsize; N = NBO_LVsize; K = NBO_LMsize;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("T", "N", &M, &N, &K, &alpha,
                           NBO_vec_tmp1, &lda, NBO_vec_tmp3, &ldb, &beta, NBO_vec_tmp2, &ldc);

     for (i=0; i<NBO_LVsize; i++){
     for (j=0; j<NBO_LVsize; j++){
       NBO_Fock_full[i][j] = NBO_vec_tmp2[NBO_LMsize*i+j];
     }
     }


    /** NBO Calculation **/

      Calc_NBO( spin, Num_NBO_FSCenter, Num_NBO_FCenter, NBO_LVsize,
                NBO_FSCenter, NBO_LMindx2LMposi,
                NBO_CDMn_full, NBO_Fock_full, NAO_vec_full );

 } /* myid==Host_ID */

   free(NBO_vec_tmp1);
   free(NBO_vec_tmp2);
   free(NBO_vec_tmp3);

   for (i=0; i<NBO_LMsize; i++){
     free(NBO_CDMn_full[i]);
   }
   free(NBO_CDMn_full);

   for (i=0; i<NBO_LMsize; i++){
     free(NBO_Fock_full[i]);
   }
   free(NBO_Fock_full);

   for (i=0; i<NBO_LMsize; i++){
     free(NAO_vec_full[i]);
   }
   free(NAO_vec_full);

 } /* if NAO_only=off */
 else if(NAO_only==1){

   for (i=0; i<NBO_SMsize_max; i++){
     free(NBO_Fock_tmp[i]);
   }
   free(NBO_Fock_tmp);

#if 0
   for (i=0; i<Num_NBO_FSCenter; i++){
     size1 = NBO_SMsize[i];
   for (j=0; j<size1; j++){
     free(NAO_vec_tmp[i][j]);
   }
     free(NAO_vec_tmp[i]);
   }
   free(NAO_vec_tmp);
#endif

   for (i=0; i<NBO_SMsize_max; i++){
     free(NBO_CDMn_tmp[i]);
   }
   free(NBO_CDMn_tmp);

   for (i=0; i<NBO_SMsize_max; i++){
     free(NBO_OLPn_tmp[i]);
   }
   free(NBO_OLPn_tmp);

 } /* if NAO_only=on */

  MPI_Barrier(mpi_comm_level1);

 } /* spin */

   /* freeing of array */

    free(MP);
/*
    for (i=0; i<EKC_core_size_max; i++){
      free(tmpvec0[i]);
    }
    free(tmpvec0);

    for (i=0; i<EKC_core_size_max; i++){
      free(tmpvec1[i]);
    }
    free(tmpvec1);

    for (i=0; i<EKC_core_size_max; i++){
      free(tmpvec2[i]);
    }
    free(tmpvec2);
*/
  free(NBO_SCenter);
  free(NBO_SCenter2);
  free(NBO_FSCenter);

  for (i=0; i<Num_NBO_FCenter; i++){
    free(NBO_FCenter_GAN[i]);
  }
  free(NBO_FCenter_GAN);

  for (i=0; i<Num_NBO_SCenter; i++){
    free(NBO_SCenter_GAN[i]);
  }
  free(NBO_SCenter_GAN);

  for (i=0; i<Num_NBO_FSCenter; i++){
    free(NBO_FSCenter_GAN[i]);
  }
  free(NBO_FSCenter_GAN);

  free(NBO_LMindx2LMatom);
  free(NBO_LMindx2LMAtom);
  free(NBO_LMindx2LMposi);
  free(NBO_LMindx2SMindx);
  free(NBO_LMindx2SMposi);

  free(NBO_SMsize);
  free(NBO_SMsize_tmp);
  free(NBO_SVsize);
  free(NBO_SVsize_tmp);

/*}*/ /* #pragma omp parallel */

  /****************************************************
    freeing of arrays:
  ****************************************************/

      for (k=0; k<=SpinP_switch; k++){
        FNAN[0] = 0;
        for (Mc_AN=0; Mc_AN<=Matomnum_NAO2; Mc_AN++){

          if (Mc_AN==0){
            Gc_AN = 0;
            tno0 = 1;
            FNAN[0] = 0;
          }
          else{
            Gc_AN = S_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
          }

          for (h_AN=0; h_AN<=FNAN[Gc_AN]+SNAN[Gc_AN]; h_AN++){

            if (Mc_AN==0){
              tno1 = 1;
            }
            else{
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno1 = Spe_Total_NO[Hwan];
            }

            for (i=0; i<tno0; i++){
              free(CDMn0[k][Mc_AN][h_AN][i]);
            }
            free(CDMn0[k][Mc_AN][h_AN]);
          }
          free(CDMn0[k][Mc_AN]);
        }
        free(CDMn0[k]);
      }
      free(CDMn0);

  free(Msize);
  free(Msize2);
  free(Msize3);
  free(Msize4);

  MPI_Barrier(mpi_comm_level1);

  /* Cube-data of Orbitals */

  if(NAO_only==0){

  if(NHO_fileout==1){

    if (myid == Host_ID) printf("<NBO> MPI transfer of NHO-cube data \n");

    MPI_Bcast(&Total_Num_NHO, 1, MPI_INT, Host_ID, mpi_comm_level1);

    for (spin=0; spin<=SpinP_switch; spin++){
    for (i=0; i<Total_Num_NHO; i++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      wan1 = WhatSpecies[Gc_AN];
      MPI_Bcast(&NHOs_Coef[spin][i][Gc_AN][0], Spe_Total_CNO[wan1], MPI_DOUBLE, Host_ID, mpi_comm_level1);
    }
    }
    }

    if (myid == Host_ID) printf("<NBO> MPI transfer of NHO-cube data finished \n\n");

  }

  if(NBO_fileout==1){

    if (myid == Host_ID) printf("<NBO> MPI transfer of NBO-cube data \n");

    MPI_Bcast(&Total_Num_NBO, 1, MPI_INT, Host_ID, mpi_comm_level1);

    for (spin=0; spin<=SpinP_switch; spin++){
    for (i=0; i<Total_Num_NBO; i++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      wan1 = WhatSpecies[Gc_AN];
      MPI_Bcast(&NBOs_Coef_b[spin][i][Gc_AN][0], Spe_Total_CNO[wan1], MPI_DOUBLE, Host_ID, mpi_comm_level1);
    }
    }
    }

    for (spin=0; spin<=SpinP_switch; spin++){
    for (i=0; i<Total_Num_NBO; i++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      wan1 = WhatSpecies[Gc_AN];
      MPI_Bcast(&NBOs_Coef_a[spin][i][Gc_AN][0], Spe_Total_CNO[wan1], MPI_DOUBLE, Host_ID, mpi_comm_level1);
    }
    }
    }

    if (myid == Host_ID) printf("<NBO> MPI transfer of NBO-cube data finished \n\n");

  }

  } /* if(NAO_only==1) */

  return (0);
}



static void Calc_NAO(int FS_atm, int Nc_AN, double **D_full, double **S_full, double **H_full,
                     int FS_atomnum, int SizeMat, double *NP_total, double *NP_partial)
{

  /* loop */
   int Gc_AN,FS_AN,Mc_AN,L1,M1,i,j,i2,j2,k,l;

  /* temporary */
   int tnum,pnum,num,is;
   int wan1,wan2,tmp_num1,tmp_num2,tmp_num3,mtps1,mtps2;
   int posi1,posi2,leng1,leng2,Mmax1,Mmax2;
   int outputlev = 0;
   double tmp_ele0,tmp_ele1,tmp_ele2,tmp_ele3,tmp_ele4,sum;
   double tmp_ele00,tmp_ele01,tmp_ele02,tmp_ele03;
   double tmp_ele10,tmp_ele11,tmp_ele12,tmp_ele13;
   double tmp_ele20,tmp_ele21,tmp_ele22,tmp_ele23;
   double **temp_M1,**temp_M2,**temp_M3,**temp_M4;
   double *temp_V1,*temp_V2,*temp_V3;

  /* BLAS */
   int M,N,K,lda,ldb,ldc;
   double alpha,beta;

  /* data */
   int *Leng_a,**Leng_al,**Leng_alm,*Posi_a,**Posi_al,***Posi_alm;
   int Lmax,Mulmax,SizeMat2,***M_or_R1,*M_or_R2,*S_or_L;
   int Num_NMB,**Num_NMB1,*NMB_posi1,*NMB_posi2;
   int Num_NRB,**Num_NRB1,Num_NRB2,Num_NRB3;
   int *NRB_posi1,*NRB_posi2,*NRB_posi3,*NRB_posi4;
   int *NRB_posi5,*NRB_posi6,*NRB_posi7,*NRB_posi8,*NRB_posi9;
   int MaxLeng_a,MaxLeng_al,MaxLeng_alm;
   double thredens = NAO_Occ_or_Ryd/(1.0+SpinP_switch);
   double *Tmp_Vec0,*Tmp_Vec1,*Tmp_Vec2,*Tmp_Vec3;
/*
   double **H_full0,**D_full0;
   double **temp_M5,**temp_M6,**temp_M8,**temp_M9;
   double **N_diag,**N_ryd,**N_red,**Ow_NMB,**Ow_NRB,**O_schm,**O_schm2,**O_sym;
*/
   double **S_full0,**P_full0;
   double **P_full,**T_full,*W_full;
   double *****P_alm,*****S_alm;
   double **P_alm2,**S_alm2,**N_alm2,*W_alm2;
   double **N_tran1,**N_tran2;
   double *W_NMB,**S_NMB;
   double **Sw1,**Uw1,*Rw1,**Ow_NMB2;
   double **Sw2,**Uw2,*Rw2;
   double *W_NRB,*W_NRB2,**S_NRB,**S_NRB2,**Ow_NRB2;
   double *NP_part_full;

   double Stime1,Etime1,StimeF,EtimeF;
   double time1,time2,time3,time4,time5;
   double time6,time7,time8,time9,time10;
   double time11,time12,time13,time14,time15;

  /* MPI */
   int numprocs,myid,ID,IDS,IDR,tag=999;

   MPI_Status stat;
   MPI_Request request;

   MPI_Comm_size(mpi_comm_level1,&numprocs);
   MPI_Comm_rank(mpi_comm_level1,&myid);

  if (measure_time==1){ dtime(&StimeF);
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    time4 = 0.0;
    time5 = 0.0;
    time6 = 0.0;
    time7 = 0.0;
    time8 = 0.0;
    time9 = 0.0;
    time10 = 0.0;
    time11 = 0.0;
    time12 = 0.0;
    time13 = 0.0;
    time14 = 0.0;
    time15 = 0.0;
  }

  /***  Maximum number of L (angular momentum)  ***/

     Lmax = 0;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
       if (Lmax < Spe_MaxL_Basis[wan1]) Lmax = Spe_MaxL_Basis[wan1];
     }

     if (outputlev==1) printf(" Maximum number of L = %d \n",Lmax);fflush(stdout);

  /***  Maximum number of multiplicity for magnetic momentum  ***/

     Mulmax = 0;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        if (Mulmax < Spe_Num_Basis[wan1][L1]) Mulmax = Spe_Num_Basis[wan1][L1];
     } /* L1 */
     } /* FS_AN */

     if (outputlev==1) printf(" Maximum number of M = %d \n",Mulmax);fflush(stdout);

  /***  Total number of basis sets (= size of full matrix)  ***/

     SizeMat2 = 0;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
       SizeMat2 += Spe_Total_NO[wan1];
     }

     if (outputlev==1) printf(" Size of full matrix = %d \n\n",SizeMat2);fflush(stdout);

  /***  Allocation of arrays (1)  ***/

   Leng_a = (int*)malloc(sizeof(int)*(FS_atomnum*2+1));

   Leng_al = (int**)malloc(sizeof(int*)*(FS_atomnum+1));
    for (i=0; i<=FS_atomnum; i++){
    Leng_al[i] = (int*)malloc(sizeof(int)*(Lmax+1));
     for (j=0; j<=Lmax; j++){
     Leng_al[i][j] = 0;
     }
    }

   Leng_alm = (int**)malloc(sizeof(int*)*(FS_atomnum+1));
    for (i=0; i<=FS_atomnum; i++){
    Leng_alm[i] = (int*)malloc(sizeof(int)*(Lmax+1));
     for (j=0; j<=Lmax; j++){
     Leng_alm[i][j] = 0;
     }
    }

   Posi_a = (int*)malloc(sizeof(int)*(FS_atomnum+1));
    for (i=0; i<=FS_atomnum; i++){
     Posi_a[i] = 0;
    }

   Posi_al = (int**)malloc(sizeof(int*)*(FS_atomnum+1));
    for (i=0; i<=FS_atomnum; i++){
    Posi_al[i] = (int*)malloc(sizeof(int)*(Lmax+1));
     for (j=0; j<=Lmax; j++){
     Posi_al[i][j] = 0;
     }
    }

   Posi_alm = (int***)malloc(sizeof(int**)*(FS_atomnum+1));
    for (i=0; i<=FS_atomnum; i++){
    Posi_alm[i] = (int**)malloc(sizeof(int*)*(Lmax+1));
     for (j=0; j<=Lmax; j++){
     Posi_alm[i][j] = (int*)malloc(sizeof(int)*(Lmax*2+2));
      for (k=0; k<=Lmax*2+1; k++){
      Posi_alm[i][j][k] = 0;
      }
     }
    }

   Num_NMB1 = (int**)malloc(sizeof(int*)*(FS_atomnum+1));
    for (i=0; i<=FS_atomnum; i++){
    Num_NMB1[i] = (int*)malloc(sizeof(int)*(Lmax+1));
     for (j=0; j<=Lmax; j++){
     Num_NMB1[i][j] = 0;
     }
    }

   Num_NRB1 = (int**)malloc(sizeof(int*)*(FS_atomnum+1));
    for (i=0; i<=FS_atomnum; i++){
    Num_NRB1[i] = (int*)malloc(sizeof(int)*(Lmax+1));
     for (j=0; j<=Lmax; j++){
     Num_NRB1[i][j] = 0;
     }
    }

    S_full0 = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     S_full0[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      S_full0[i][j] = 0.0;
      }
     }

/*
    D_full0 = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     D_full0[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      D_full0[i][j] = 0.0;
      }
     }
*/
    P_full = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     P_full[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      P_full[i][j] = 0.0;
      }
     }

    P_full0= (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     P_full0[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      P_full0[i][j] = 0.0;
      }
     }
/*
    H_full0 = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     H_full0[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      H_full0[i][j] = 0.0;
      }
     }
*/
    T_full = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     T_full[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      T_full[i][j] = 0.0;
      }
     }

    W_full = (double*)malloc(sizeof(double)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     W_full[i] = 0.0;
     }

    N_tran1 = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     N_tran1[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      N_tran1[i][j] = 0.0;
      }
     }
/*
    N_ryd = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     N_ryd[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      N_ryd[i][j] = 0.0;
      }
     }
*/
/*
    N_red = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     N_red[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      N_red[i][j] = 0.0;
      }
     }
*/
/*
    O_schm = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     O_schm[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      O_schm[i][j] = 0.0;
      }
     }
*/
/*
    O_schm2 = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     O_schm2[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      O_schm2[i][j] = 0.0;
      }
     }
*/
     N_tran2= (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     N_tran2[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      N_tran2[i][j] = 0.0;
      }
     }
/*
    Ow_NRB = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     Ow_NRB[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      Ow_NRB[i][j] = 0.0;
      }
     }
*/
/*
    O_sym = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     O_sym[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      O_sym[i][j] = 0.0;
      }
     }
*/
   for (i=0; i<=SizeMat; i++){
     N_tran1[i][i]  = 1.0;
     N_tran2[i][i] = 1.0;
/*   N_diag[i][i]  = 1.0; */
/*   N_ryd[i][i]   = 1.0; */
/*   N_red[i][i]   = 1.0; */
/*   O_schm[i][i]  = 1.0; */
/*   O_schm2[i][i] = 1.0; */
/*   Ow_NMB[i][i]  = 1.0; */
/*   Ow_NRB[i][i]  = 1.0; */
/*   O_sym[i][i]   = 1.0; */
     T_full[i][i]  = 1.0;
   }

    temp_M1 = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     temp_M1[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      temp_M1[i][j] = 0.0;
      }
     }

    temp_M2 = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     temp_M2[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      temp_M2[i][j] = 0.0;
      }
     }

    temp_M3 = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     temp_M3[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      temp_M3[i][j] = 0.0;
      }
     }

    temp_M4 = (double**)malloc(sizeof(double*)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     temp_M4[i] = (double*)malloc(sizeof(double)*(SizeMat+1));
      for (j=0; j<=SizeMat; j++){
      temp_M4[i][j] = 0.0;
      }
     }

   temp_V1 = (double*)malloc(sizeof(double)*(SizeMat*SizeMat+1));
    for (j=0; j<=SizeMat*SizeMat; j++){
    temp_V1[j] = 0.0;
    }

   temp_V2 = (double*)malloc(sizeof(double)*(SizeMat*SizeMat+1));
    for (j=0; j<=SizeMat*SizeMat; j++){
    temp_V2[j] = 0.0;
    }

   temp_V3 = (double*)malloc(sizeof(double)*(SizeMat*SizeMat+1));
    for (j=0; j<=SizeMat*SizeMat; j++){
    temp_V3[j] = 0.0;
    }

   NP_part_full = (double*)malloc(sizeof(double)*(FS_atomnum+1));
    for (i=0; i<=FS_atomnum; i++){
     NP_part_full[i] = 0.0;
    }

/**************************************************************************
   1. Setting density & overlap matrices P & S
**************************************************************************/

if (myid == Host_ID && outputlev == 1) 
   printf("<< 1 >> Setting density & overlap matrices P & S \n");
if (measure_time==1) dtime(&Stime1);

  /***  Table functions of atomic positions on full matrix  ***/

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
       Leng_a[FS_AN] = Spe_Total_NO[wan1];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       Leng_alm[FS_AN][L1] = Spe_Num_Basis[wan1][L1];
     for (M1=1; M1<=2*L1+1; M1++){
       Leng_al[FS_AN][L1] += Leng_alm[FS_AN][L1];
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

if (myid == Host_ID){
if (outputlev==1){
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       printf(" Leng_a[%d] = %d \n",FS_AN,Leng_a[FS_AN]);fflush(stdout);
     } /* FS_AN */

     printf("\n");fflush(stdout);

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       printf(" Leng_al[%d][%d] = %d \n",FS_AN,L1,Leng_al[FS_AN][L1]);fflush(stdout);
     } /* L1 */
     } /* FS_AN */

     printf("\n");fflush(stdout);

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       printf(" Leng_alm[%d][%d] = %d \n",FS_AN,L1,Leng_alm[FS_AN][L1]);fflush(stdout);
     } /* L1 */
     } /* FS_AN */

     printf("\n");fflush(stdout);
}
}

      MaxLeng_a   = 0;
      MaxLeng_al  = 0;
      MaxLeng_alm = 0;

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
       if (MaxLeng_a < Leng_a[FS_AN]) MaxLeng_a = Leng_a[FS_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       if (MaxLeng_al  < Leng_al[FS_AN][L1])  MaxLeng_al  = Leng_al[FS_AN][L1];
       if (MaxLeng_alm < Leng_alm[FS_AN][L1]) MaxLeng_alm = Leng_alm[FS_AN][L1];
     } /* L1 */
     } /* FS_AN */

if (myid == Host_ID){
if (outputlev==1){
      printf(" MaxLeng_a   = %d \n",MaxLeng_a  );fflush(stdout);
      printf(" MaxLeng_al  = %d \n",MaxLeng_al );fflush(stdout);
      printf(" MaxLeng_alm = %d \n",MaxLeng_alm);fflush(stdout);
}
}

      tmp_num1 =0;
      tmp_num2 =0;
      tmp_num3 =0;

     for (FS_AN=1; FS_AN<=FS_atomnum; FS_AN++){
       tmp_num1 += Leng_a[FS_AN-1];
       Posi_a[FS_AN] = tmp_num1;
     } /* FS_AN */

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       tmp_num2 += Leng_al[FS_AN][L1];
       if (L1 != Spe_MaxL_Basis[wan1])                        Posi_al[FS_AN][L1+1] = tmp_num2;
       if (L1 == Spe_MaxL_Basis[wan1] && FS_AN != FS_atomnum) Posi_al[FS_AN+1][0]  = tmp_num2;
     } /* L1 */
     } /* FS_AN */

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
     for (M1=1; M1<=2*L1+1; M1++){
       tmp_num3 += Leng_alm[FS_AN][L1];
       if (M1 != 2*L1+1)                               Posi_alm[FS_AN][L1][M1+1] = tmp_num3;
       if (M1 == 2*L1+1 && L1 != Spe_MaxL_Basis[wan1]) Posi_alm[FS_AN][L1+1][1]  = tmp_num3;
       if (M1 == 2*L1+1 && L1 == Spe_MaxL_Basis[wan1]
                        && FS_AN != FS_atomnum)        Posi_alm[FS_AN+1][0][1]   = tmp_num3;
     }
     }
     }

if (myid == Host_ID){
if (outputlev==1){
     printf("\n ### Check Positions on Matrices ###\n");fflush(stdout);

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       printf(" Posi_a[%d] = %d \n",FS_AN,Posi_a[FS_AN]);fflush(stdout);
     } /* FS_AN */

     printf("\n");fflush(stdout);

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       printf(" Posi_al[%d][%d] = %d \n",FS_AN,L1,Posi_al[FS_AN][L1]);fflush(stdout);
     } /* L1 */
     } /* FS_AN */

     printf("\n");fflush(stdout);

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
     for (M1=1; M1<=2*L1+1; M1++){
       printf(" Posi_alm[%d][%d][%d] = %d \n",FS_AN,L1,M1,Posi_alm[FS_AN][L1][M1]);fflush(stdout);
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

     printf("\n");fflush(stdout);
}
}

  /***  Bond-order matrix D, overlap matrix S, Hamiltonian matrix H  ***/
/*
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       D_full[i][j] = D_full[i][j]*(2.0-(double)SpinP_switch);
     } 
     } 
*/

   int SizeMat0;
   SizeMat0 = 18;

   if (outputlev==1){
     printf("### Overlap Matrix (full size) ###\n");fflush(stdout);
     for (i=0; i<SizeMat0; i++){
     for (j=0; j<SizeMat0; j++){
       printf("%9.5f",S_full[i][j]);fflush(stdout);
     } /* j */
       printf("\n");fflush(stdout);
     } /* i */

     printf("\n");fflush(stdout);

     printf("### Bond-Order Matrix (full size) ###\n");fflush(stdout);
     for (i=0; i<SizeMat0; i++){
     for (j=0; j<SizeMat0; j++){
       printf("%9.5f",D_full[i][j]);fflush(stdout);
     } /* j */
       printf("\n");fflush(stdout);
     } /* i */

     printf("\n");fflush(stdout);

     printf("### Hamiltonian Matrix (full size) ###\n");fflush(stdout);
     for (i=0; i<SizeMat0; i++){
     for (j=0; j<SizeMat0; j++){
       printf("%9.5f",H_full[i][j]);fflush(stdout);
     } /* j */
       printf("\n");fflush(stdout);
     } /* i */

     printf("\n");fflush(stdout);
   }

  /***  Density matrix P = S * D * S  ***/
/*
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele1 = 0.0;
     for (k=0; k<SizeMat; k++){
        tmp_ele1 += D_full[i][k] * S_full[k][j];
     } 
       temp_M1[i][j] = tmp_ele1;
     } 
     } 


if (myid == Host_ID){
if (outputlev==1){
     printf("\n### Mulliken check for central atom ###\n");
     for (k=0; k<4; k++){
     for (i=0; i<Leng_a[k]; i++){
     for (j=0; j<Leng_a[k]; j++){
       printf("%8.4f",temp_M1[Posi_a[k]+i][Posi_a[k]+j]);
     }
     printf("\n");
     }
     printf("\n");
     }

     tmp_ele1 = 0.0;
     for (i=0; i<Leng_a[0]; i++){
       tmp_ele1 += temp_M1[i][i];
     }
     printf("### Mulliken pop. of central atom  = %9.4f \n",tmp_ele1);fflush(stdout);

     for (i=0; i<FS_atomnum; i++){
     tmp_ele1 = 0.0;
     for (j=0; j<Leng_a[i]; j++){
       tmp_ele1 += temp_M1[Posi_a[i]+j][Posi_a[i]+j];
     }
     printf("# %3d %8.4f\n",i,tmp_ele1);
     }
     printf("\n");

     tmp_ele1 = 0.0;
     for (i=0; i<SizeMat; i++){
       tmp_ele1 += temp_M1[i][i];
     }
     printf("### Mulliken total pop. = %8.4f \n",tmp_ele1);fflush(stdout);
}
}

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele1 = 0.0;
     for (k=0; k<SizeMat; k++){
       tmp_ele1 += S_full[i][k] * temp_M1[k][j];
     } 
       P_full[i][j] = tmp_ele1;
     } 
     } 
*/
   /* BLAS */
if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = S_full[i][j];
       temp_V2[tnum] = D_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

}
else{

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M2[j][i] = S_full[i][j];
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele0 = 0.0;
       tmp_ele1 = 0.0;
       tmp_ele2 = 0.0;
       tmp_ele3 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
        tmp_ele0 += D_full[i][k]   * temp_M2[j][k];
        tmp_ele1 += D_full[i][k+1] * temp_M2[j][k+1];
        tmp_ele2 += D_full[i][k+2] * temp_M2[j][k+2];
        tmp_ele3 += D_full[i][k+3] * temp_M2[j][k+3];
     }
       temp_M1[j][i] = tmp_ele0 + tmp_ele1 + tmp_ele2 + tmp_ele3;
     }
     }

     is = SizeMat - SizeMat%4;

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele0 = 0.0;
     for (k=is; k<SizeMat; k++){
        tmp_ele0 += D_full[i][k] * temp_M2[j][k];
     }
       temp_M1[j][i] += tmp_ele0;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele0 = 0.0;
       tmp_ele1 = 0.0;
       tmp_ele2 = 0.0;
       tmp_ele3 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 += S_full[i][k]   * temp_M1[j][k];
       tmp_ele1 += S_full[i][k+1] * temp_M1[j][k+1];
       tmp_ele2 += S_full[i][k+2] * temp_M1[j][k+2];
       tmp_ele3 += S_full[i][k+3] * temp_M1[j][k+3];
     }
       P_full[i][j] = tmp_ele0 + tmp_ele1 + tmp_ele2 + tmp_ele3;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele0 = 0.0;
     for (k=is; k<SizeMat; k++){
       tmp_ele0 += S_full[i][k] * temp_M1[j][k];
     }
       P_full[i][j] += tmp_ele0;
     }
     }
}

   if (outputlev==1){
     printf("### Density Matrix (full size) ###\n");fflush(stdout);
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       printf("%9.5f",P_full[i][j]);fflush(stdout);
     } /* j */
       printf("\n");fflush(stdout);
     } /* i */

     printf("\n");fflush(stdout);
   }

  /***  Rearrangement of overlap, density & Hamiltonian matrices  ***/
  /***  Overlap Matrix  ***/

     tmp_num3 = -1;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       posi1 = Posi_al[FS_AN][L1];
       tmp_num1 = Spe_Num_Basis[wan1][L1];
     for (M1=1; M1<=2*L1+1; M1++){
     for (k=1; k<=tmp_num1; k++){
       tmp_num3 += 1;
     for (j=0; j<SizeMat; j++){
       temp_M1[j][tmp_num3] = S_full[j][posi1+(2*L1+1)*(k-1)+(M1-1)];
     } /* j */
     } /* k */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

     tmp_num3 = -1;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       posi1 = Posi_al[FS_AN][L1];
       tmp_num1 = Spe_Num_Basis[wan1][L1];
     for (M1=1; M1<=2*L1+1; M1++){
     for (k=1; k<=tmp_num1; k++){
       tmp_num3 += 1;
     for (j=0; j<SizeMat; j++){
       temp_M2[tmp_num3][j] = temp_M1[posi1+(2*L1+1)*(k-1)+(M1-1)][j];
     } /* j */
     } /* k */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       S_full[i][j] = temp_M2[i][j];
       S_full0[i][j] = S_full[i][j];
     } /* j */
     } /* i */


  /***  Hamiltonian Matrix  ***/

     tmp_num3 = -1;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       posi1 = Posi_al[FS_AN][L1];
       tmp_num1 = Spe_Num_Basis[wan1][L1];
     for (M1=1; M1<=2*L1+1; M1++){
     for (k=1; k<=tmp_num1; k++){
       tmp_num3 += 1;
     for (j=0; j<SizeMat; j++){
       temp_M1[j][tmp_num3] = H_full[j][posi1+(2*L1+1)*(k-1)+(M1-1)];
     } /* j */
     } /* k */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

     tmp_num3 = -1;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       posi1 = Posi_al[FS_AN][L1];
       tmp_num1 = Spe_Num_Basis[wan1][L1];
     for (M1=1; M1<=2*L1+1; M1++){
     for (k=1; k<=tmp_num1; k++){
       tmp_num3 += 1;
     for (j=0; j<SizeMat; j++){
       temp_M2[tmp_num3][j] = temp_M1[posi1+(2*L1+1)*(k-1)+(M1-1)][j];
     } /* j */
     } /* k */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       H_full[i][j] = temp_M2[i][j];
/*
       H_full0[i][j] = H_full[i][j];
*/
     } /* j */
     } /* i */


  /***  Density Matrix  ***/

     tmp_num3 = -1;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       posi1 = Posi_al[FS_AN][L1];
       tmp_num1 = Spe_Num_Basis[wan1][L1];
     for (M1=1; M1<=2*L1+1; M1++){
     for (k=1; k<=tmp_num1; k++){
       tmp_num3 += 1;
     for (j=0; j<SizeMat; j++){
       temp_M1[j][tmp_num3] = D_full[j][posi1+(2*L1+1)*(k-1)+(M1-1)];
     } /* j */
     } /* k */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

     tmp_num3 = -1;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       posi1 = Posi_al[FS_AN][L1];
       tmp_num1 = Spe_Num_Basis[wan1][L1];
     for (M1=1; M1<=2*L1+1; M1++){
     for (k=1; k<=tmp_num1; k++){
       tmp_num3 += 1;
     for (j=0; j<SizeMat; j++){
       temp_M2[tmp_num3][j] = temp_M1[posi1+(2*L1+1)*(k-1)+(M1-1)][j];
     } /* j */
     } /* k */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       D_full[i][j] = temp_M2[i][j];
/*
       D_full0[i][j] = D_full[i][j];
*/
     } /* j */
     } /* i */

  /***  Density Matrix  ***/

     tmp_num3 = -1;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       posi1 = Posi_al[FS_AN][L1];
       tmp_num1 = Spe_Num_Basis[wan1][L1];
     for (M1=1; M1<=2*L1+1; M1++){
     for (k=1; k<=tmp_num1; k++){
       tmp_num3 += 1;
     for (j=0; j<SizeMat; j++){
       temp_M1[j][tmp_num3] = P_full[j][posi1+(2*L1+1)*(k-1)+(M1-1)];
     } /* j */
     } /* k */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

     tmp_num3 = -1;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
       posi1 = Posi_al[FS_AN][L1];
       tmp_num1 = Spe_Num_Basis[wan1][L1];
     for (M1=1; M1<=2*L1+1; M1++){
     for (k=1; k<=tmp_num1; k++){
       tmp_num3 += 1;
     for (j=0; j<SizeMat; j++){
       temp_M2[tmp_num3][j] = temp_M1[posi1+(2*L1+1)*(k-1)+(M1-1)][j];
     } /* j */
     } /* k */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_M2[i][j];
       P_full0[i][j] = P_full[i][j];
     } /* j */
     } /* i */

   SizeMat0 = 18;

   if (outputlev==1){
     printf("### Overlap Matrix (full size) 2 ###\n");fflush(stdout);
     for (i=0; i<SizeMat0; i++){
     for (j=0; j<SizeMat0; j++){
       printf("%9.5f",S_full[i][j]);fflush(stdout);
     } /* j */
       printf("\n");fflush(stdout);
     } /* i */

     printf("\n");

     printf("### Density Matrix (full size) 2 ###\n");fflush(stdout);
     for (i=0; i<SizeMat0; i++){
     for (j=0; j<SizeMat0; j++){
       printf("%9.5f",P_full[i][j]);fflush(stdout);
     } /* j */
       printf("\n");fflush(stdout);
     } /* i */

     printf("\n");

     printf("### Hamiltonian Matrix (full size) 2 ###\n");fflush(stdout);
     for (i=0; i<SizeMat0; i++){
     for (j=0; j<SizeMat0; j++){
       printf("%9.5f",H_full[i][j]);fflush(stdout);
     } /* j */
       printf("\n");fflush(stdout);
     } /* i */

     printf("\n");
   }


  /***  Allocation of arrays (2)  ***/

    P_alm = (double*****)malloc(sizeof(double****)*(FS_atomnum+1));
    for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
     P_alm[FS_AN] = (double****)malloc(sizeof(double***)*(Lmax+1));
     for (L1=0; L1<=Lmax; L1++){
      P_alm[FS_AN][L1] = (double***)malloc(sizeof(double**)*(2*Lmax+2));
      for (M1=0; M1<=2*Lmax+1; M1++){
       P_alm[FS_AN][L1][M1] = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
       for (i=0; i<=MaxLeng_alm; i++){
        P_alm[FS_AN][L1][M1][i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
        for (j=0; j<=MaxLeng_alm; j++){
         P_alm[FS_AN][L1][M1][i][j] = 0.0;
        }
       }
      }
     }
    }

    S_alm = (double*****)malloc(sizeof(double****)*(FS_atomnum+1));
    for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
     S_alm[FS_AN] = (double****)malloc(sizeof(double***)*(Lmax+1));
     for (L1=0; L1<=Lmax; L1++){
      S_alm[FS_AN][L1] = (double***)malloc(sizeof(double**)*(2*Lmax+2));
      for (M1=0; M1<=2*Lmax+1; M1++){
       S_alm[FS_AN][L1][M1] = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
       for (i=0; i<=MaxLeng_alm; i++){
        S_alm[FS_AN][L1][M1][i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
        for (j=0; j<=MaxLeng_alm; j++){
         S_alm[FS_AN][L1][M1][i][j] = 0.0;
        }
       }
      }
     }
    }

  /***  Atomic & anglar- & magetic-momentum block matrices  ***/

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Leng_alm[FS_AN][L1];
     for (M1=1; M1<=2*L1+1; M1++){
        posi1 = Posi_alm[FS_AN][L1][M1];
       for (i=0; i<leng1; i++){
       for (j=0; j<leng1; j++){
          P_alm[FS_AN][L1][M1][i][j] = P_full[i+posi1][j+posi1];
          S_alm[FS_AN][L1][M1][i][j] = S_full[i+posi1][j+posi1];
       } /* j */
       } /* i */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

if (measure_time==1){
  dtime(&Etime1);
  time1 = Etime1 - Stime1;
}

/**************************************************************************
   2. Intraatomic orthogonalization
**************************************************************************/

if (myid == Host_ID && outputlev == 1) printf("<< 2 >> Intraatomic orthogonalization \n");

  /***  Allocation of arrays (3)  ***/

    P_alm2 = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
    for (i=0; i<=MaxLeng_alm; i++){
     P_alm2[i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
     for (j=0; j<=MaxLeng_alm; j++){
      P_alm2[i][j] = 0.0;
     }
    }

    S_alm2 = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
    for (i=0; i<=MaxLeng_alm; i++){
     S_alm2[i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
     for (j=0; j<=MaxLeng_alm; j++){
      S_alm2[i][j] = 0.0;
     }
    }

    N_alm2 = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
    for (i=0; i<=MaxLeng_alm; i++){
     N_alm2[i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
     for (j=0; j<=MaxLeng_alm; j++){
      N_alm2[i][j] = 0.0;
     }
    }

    W_alm2 = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
    for (i=0; i<=MaxLeng_alm; i++){
     W_alm2[i] = 0.0;
    }

   NRB_posi3= (int*)malloc(sizeof(int)*(MaxLeng_alm+1));
   for (i=0; i<=MaxLeng_alm; i++){
    NRB_posi3[i] = 0;
   }

   M_or_R1= (int***)malloc(sizeof(int**)*(FS_atomnum+1));
   for (i=0; i<=FS_atomnum; i++){
    M_or_R1[i] = (int**)malloc(sizeof(int*)*(Lmax+1));
    for (j=0; j<=Lmax; j++){
     M_or_R1[i][j] = (int*)malloc(sizeof(int)*(MaxLeng_alm+1));
     for (k=0; k<=MaxLeng_alm; k++){
      M_or_R1[i][j][k] = 0;
     }
    }
   }

   M_or_R2 = (int*)malloc(sizeof(int)*(SizeMat+1));
   for (i=0; i<=SizeMat; i++){
    M_or_R2[i] = 0;
   }
 
  /*********************************************************
     2-1. Transformation from Cartesian to pure d,f,g AOs
  *********************************************************/

  /**************************************
     2-2. Symmetry averaging of P & S
  **************************************/

if (myid == Host_ID && outputlev == 1) printf("<< 2-2 >> Symmetry averaging of P & S \n");
if (measure_time==1) dtime(&Stime1);

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Leng_alm[FS_AN][L1];
       for (i=0; i<leng1; i++){
       for (j=0; j<leng1; j++){
          tmp_ele1 = 0.0;
          tmp_ele2 = 0.0;
         for (M1=1; M1<=L1*2+1; M1++){
              tmp_ele1 += P_alm[FS_AN][L1][M1][i][j];
              tmp_ele2 += S_alm[FS_AN][L1][M1][i][j];
         } /* M1 */
          tmp_ele1 /= L1*2.0+1.0;
          tmp_ele2 /= L1*2.0+1.0;
         for (M1=1; M1<=L1*2+1; M1++){
              P_alm[FS_AN][L1][M1][i][j] = tmp_ele1;
              S_alm[FS_AN][L1][M1][i][j] = tmp_ele2;
         } /* M1 */
       } /* j */
       } /* i */
     } /* L1 */
     } /* FS_AN */

if (measure_time==1){
  dtime(&Etime1);
  time2 = Etime1 - Stime1;
}

  /*******************************
     2-3. Formation of pre-NAOs
  *******************************/

if (myid == Host_ID && outputlev == 1) printf("<< 2-3 >> Formation of pre-NAOs \n");
if (measure_time==1) dtime(&Stime1);

  /***  Atom & anglar-momentum loop  ***/

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        Mmax1 = Leng_alm[FS_AN][L1];

  /***  Transferring of matrices (1)  ***/

       for (i=0; i<Mmax1; i++){
       for (j=0; j<Mmax1; j++){
          P_alm2[i+1][j+1] = P_alm[FS_AN][L1][1][i][j];
          S_alm2[i+1][j+1] = S_alm[FS_AN][L1][1][i][j];
       } /* j */
       } /* i */

    if (outputlev==1){
       printf("### S_alm2[%d][%d] ### \n",FS_AN,L1);fflush(stdout);
       for (i=0; i<Mmax1; i++){
       for (j=0; j<Mmax1; j++){
          printf("%9.5f",S_alm2[i+1][j+1]);fflush(stdout);
       } /* j */
          printf("\n");fflush(stdout);
       } /* j */

       printf("\n");fflush(stdout);
       printf("### P_alm2[%d][%d] ### \n",FS_AN,L1);fflush(stdout);
       for (i=0; i<Mmax1; i++){
       for (j=0; j<Mmax1; j++){
          printf("%9.5f",P_alm2[i+1][j+1]);fflush(stdout);
       } /* j */
          printf("\n");fflush(stdout);
       } /* i */
    }

  /***  Generalized eigenvalue problem, P * N = S * N * W  ***/

  /*******************************************
     Diagonalizing the overlap matrix

      First:
        S -> OLP matrix
      After calling Eigen_lapack:
        S -> eigenvectors of OLP matrix
   *******************************************/

  if (outputlev==1) printf("## Diagonalize the overlap matrix (Gc_AN=%d L1=%d)\n",Gc_AN,L1);

    Eigen_lapack(S_alm2,W_alm2,Mmax1,Mmax1);

  if (outputlev==1){
    for (k=1; k<=Mmax1; k++){
      printf("k W %2d %15.12f\n",k,W_alm2[k]);
    }
  }

    /* check ill-conditioned eigenvalues */

    for (k=1; k<=Mmax1; k++){
      if (W_alm2[k]<1.0e-14) {
        if (myid==Host_ID){
          printf("Found ill-conditioned eigenvalues (1)\n");
          printf("Stopped calculation\n");
        }
       exit(1);
      }
    }

    for (k=1; k<=Mmax1; k++){
      temp_V3[k] = 1.0/sqrt(W_alm2[k]);
    }

   /***  Calculations of eigenvalues  ***/

    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
         sum = 0.0;
        for (k=1; k<=Mmax1; k++){
          sum = sum + P_alm2[i][k]*S_alm2[k][j]*temp_V3[j];
        }
        N_alm2[i][j] = sum;
      }
    }

    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
         sum = 0.0;
        for (k=1; k<=Mmax1; k++){
          sum = sum + temp_V3[i]*S_alm2[k][i]*N_alm2[k][j];   
        }
        temp_M1[i][j] = sum;
      }
    }

  if (outputlev==1){
    printf("\n ##### TEST 1 ##### \n");
    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
        printf("%9.5f",temp_M1[i][j]);
      }
        printf("\n");
    }
  } 

    Eigen_lapack(temp_M1,W_alm2,Mmax1,Mmax1);

  if (outputlev==1){
    printf("\n ##### TEST 2 ##### \n");
    for (k=1; k<=Mmax1; k++){
      printf("k W %2d %9.5f\n",k,W_alm2[k]);
    }

    printf("\n ##### TEST 3 ##### \n");
    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
        printf("%9.5f",temp_M1[i][j]);
      }
        printf("\n");
    }
    printf("\n");
  }

  /***  Transformation to the original eigen vectors  ***/

    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
        N_alm2[i][j] = 0.0;
      }
    }

    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
         sum = 0.0;
        for (k=1; k<=Mmax1; k++){
          sum = sum + S_alm2[i][k]*temp_V3[k]*temp_M1[k][j];
        }
        N_alm2[i][j] = sum;
      }
    }

  /* printing out eigenvalues and the eigenvectors */

    if (outputlev==1){
      for (i=1; i<=Mmax1; i++){
        printf("%ith eigenvalue of HC=eSC: %15.12f\n",i,W_alm2[i]);
      }

        for (i=1; i<=Mmax1; i++){
           printf("%ith eigenvector: ",i);
           printf("{");
          for (j=1; j<=Mmax1; j++){
            printf("%15.12f,",N_alm2[i][j]);
          }
          printf("}\n");
        }

          printf("\n");
    }

  /***  Selection of NMB & NRB orbitals  ***/

       for (i=1; i<=Mmax1; i++){
         if (W_alm2[i] >= thredens) {M_or_R1[FS_AN][L1][i-1] = 1;}
         else                          {M_or_R1[FS_AN][L1][i-1] = 0;}
       } /* i */

       leng1 = Leng_alm[FS_AN][L1];

       for (M1=1; M1<=L1*2+1; M1++){
          posi1 = Posi_alm[FS_AN][L1][M1];
       for (i=0; i<leng1; i++){
          W_full[posi1+i] = W_alm2[i+1];
          if (W_alm2[i+1] >= thredens) {M_or_R2[posi1+i] = 1;}
          else                            {M_or_R2[posi1+i] = 0;}
       for (j=0; j<leng1; j++){
          N_tran1[posi1+i][posi1+j] = N_alm2[i+1][j+1];
       } /* j */
       } /* i */
       } /* M1 */

     } /* L1 */
     } /* FS_AN */

    if (outputlev==1){
       printf("### T_full (1) (= N_diag) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",N_tran1[i][j]);fflush(stdout);
       }
         printf("\n");
       }
     }

  /***  Counting number of NMB & NRB orbitals  ***/

       tmp_num1 = 0;

       for (i=0; i<SizeMat; i++){
        tmp_num1 += M_or_R2[i];
       } /* i */

       Num_NMB = tmp_num1;
       Num_NRB = SizeMat - Num_NMB;

    if (outputlev==1){
       printf("## Number of NMBs = %d \n",Num_NMB);fflush(stdout);
       printf("## Number of NRBs = %d \n",Num_NRB);fflush(stdout);
    }

if (measure_time==1){
  dtime(&Etime1);
  time3 = Etime1 - Stime1;
}

  /***  Overlap & density matrices for pre-NAOs  ***/

if (myid == Host_ID && outputlev == 1) printf("<< 2-4 >> Construction of S & P for pre-NAOs \n");
if (measure_time==1) dtime(&Stime1);
/*
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full[i][k] * N_diag[k][j];
        tmp_ele2 += P_full[i][k] * N_diag[k][j];
      } 
        temp_M1[i][j] = tmp_ele1;
        temp_M2[i][j] = tmp_ele2;
      } 
      } 

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += N_diag[k][i] * temp_M1[k][j];
        tmp_ele2 += N_diag[k][i] * temp_M2[k][j];
      }
        S_full[i][j] = tmp_ele1;
        P_full[i][j] = tmp_ele2;
      } 
      } 
*/
   /* BLAS */
     /*  N_diag * S * N_diag  */
if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = N_tran1[i][j];
       temp_V2[tnum] = S_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("T", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       S_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  N_diag * P * N_diag  */
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = P_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("T", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }
}
else{
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M3[j][i] = N_tran1[i][j];
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M3[j][k];
       tmp_ele1 = temp_M3[j][k+1];
       tmp_ele2 = temp_M3[j][k+2];
       tmp_ele3 = temp_M3[j][k+3];

       tmp_ele00 += S_full[i][k]   * tmp_ele0;
       tmp_ele01 += S_full[i][k+1] * tmp_ele1;
       tmp_ele02 += S_full[i][k+2] * tmp_ele2;
       tmp_ele03 += S_full[i][k+3] * tmp_ele3;

       tmp_ele10 += P_full[i][k]   * tmp_ele0;
       tmp_ele11 += P_full[i][k+1] * tmp_ele1;
       tmp_ele12 += P_full[i][k+2] * tmp_ele2;
       tmp_ele13 += P_full[i][k+3] * tmp_ele3;
     }
       temp_M1[j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       temp_M2[j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     is = SizeMat - SizeMat%4;

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
        tmp_ele0 = temp_M3[j][k];
        tmp_ele00 += S_full[i][k] * tmp_ele0;
        tmp_ele10 += P_full[i][k] * tmp_ele0;
     }
       temp_M1[j][i] += tmp_ele00;
       temp_M2[j][i] += tmp_ele10;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M3[i][k];
       tmp_ele1 = temp_M3[i][k+1];
       tmp_ele2 = temp_M3[i][k+2];
       tmp_ele3 = temp_M3[i][k+3];

       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele01 += tmp_ele1 * temp_M1[j][k+1];
       tmp_ele02 += tmp_ele2 * temp_M1[j][k+2];
       tmp_ele03 += tmp_ele3 * temp_M1[j][k+3];

       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
       tmp_ele11 += tmp_ele1 * temp_M2[j][k+1];
       tmp_ele12 += tmp_ele2 * temp_M2[j][k+2];
       tmp_ele13 += tmp_ele3 * temp_M2[j][k+3];
     }
       S_full[i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       P_full[i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }


     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
       tmp_ele0 = temp_M3[i][k];
       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
     }
       S_full[i][j] += tmp_ele00;
       P_full[i][j] += tmp_ele10;
     }
     }
}

if (measure_time==1){
  dtime(&Etime1);
  time4 = Etime1 - Stime1;
}


     if (outputlev==1){
       printf("\n ### S_full (1) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",S_full[i][j]);fflush(stdout);
       }
         printf("\n");
       }

       printf("\n ### P_full (1) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",P_full[i][j]);fflush(stdout);
       }
         printf("\n");
       }

       printf("\n ### W_full (1) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
         printf("%9.5f \n",W_full[i]);fflush(stdout);
       }
     }

  /***  Allocation of arrays (4)  ***/

    S_NMB = (double**)malloc(sizeof(double*)*(Num_NMB+1));
     for (i=0; i<=Num_NMB; i++){
     S_NMB[i] = (double*)malloc(sizeof(double)*(Num_NMB+1));
      for (j=0; j<=Num_NMB; j++){
      S_NMB[i][j] = 0.0;
      }
     }

    W_NMB = (double*)malloc(sizeof(double)*(Num_NMB+1));
     for (i=0; i<=Num_NMB; i++){
     W_NMB[i] = 0.0;
     }

    Ow_NMB2 = (double**)malloc(sizeof(double*)*(Num_NMB+1));
     for (i=0; i<=Num_NMB; i++){
     Ow_NMB2[i] = (double*)malloc(sizeof(double)*(Num_NMB+1));
      for (j=0; j<=Num_NMB; j++){
      Ow_NMB2[i][j] = 0.0;
      }
     }

    Sw1 = (double**)malloc(sizeof(double*)*(Num_NMB+1));
     for (i=0; i<=Num_NMB; i++){
     Sw1[i] = (double*)malloc(sizeof(double)*(Num_NMB+1));
      for (j=0; j<=Num_NMB; j++){
      Sw1[i][j] = 0.0;
      }
     }

    Uw1 = (double**)malloc(sizeof(double*)*(Num_NMB+1));
     for (i=0; i<=Num_NMB; i++){
     Uw1[i] = (double*)malloc(sizeof(double)*(Num_NMB+1));
      for (j=0; j<=Num_NMB; j++){
      Uw1[i][j] = 0.0;
      }
     }

    Rw1 = (double*)malloc(sizeof(double)*(Num_NMB+1));
     for (i=0; i<=Num_NMB; i++){
     Rw1[i] = 0.0;
     }

   NMB_posi1 = (int*)malloc(sizeof(int)*(Num_NMB+1));
     for (i=0; i<=Num_NMB; i++){
     NMB_posi1[i] = 0;
     }

   NMB_posi2 = (int*)malloc(sizeof(int)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     NMB_posi2[i] = 0;
     }

   NRB_posi2 = (int*)malloc(sizeof(int)*(SizeMat+1));
     for (i=0; i<=SizeMat; i++){
     NRB_posi2[i] = 0;
     }

   NRB_posi1 = (int*)malloc(sizeof(int)*(Num_NRB+1));
     for (i=0; i<=Num_NRB; i++){
     NRB_posi1[i] = 0;
     }

  /***  Matrices for NMB  ***/

      tmp_num1 = -1;

      for (i=0; i<SizeMat; i++){
        if (M_or_R2[i]==1){
          tmp_num1 += 1;
          W_NMB[tmp_num1] = W_full[i];
          NMB_posi1[tmp_num1] = i;
      if(1<0) printf(" ** NMB_posi1[%d] = %d \n",tmp_num1,i);fflush(stdout);
          NMB_posi2[i] = tmp_num1;
          tmp_num2 = -1;
         for (j=0; j<SizeMat; j++){
           if (M_or_R2[j]==1){
             tmp_num2 += 1;
             S_NMB[tmp_num1][tmp_num2] = S_full[i][j];
           }
         } /* j */
        }
      } /* i */

    if (outputlev==1){
      printf("\n ### S_NMB ### \n");fflush(stdout);
      for (i=0; i<Num_NMB; i++){
      for (j=0; j<Num_NMB; j++){        
        printf("%9.5f",S_NMB[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### W_NMB ### \n");fflush(stdout);
      for (i=0; i<Num_NMB; i++){
        printf("%9.5f \n",W_NMB[i]);fflush(stdout);
      }
    }

  /***  Matrices for NRB  ***/

      tmp_num1 = -1;

      for (i=0; i<SizeMat; i++){
        if (M_or_R2[i]==0){
          tmp_num1 += 1;
          NRB_posi1[tmp_num1] = i;
      if(1<0) printf(" ** NRB_posi1[%d] = %d \n",tmp_num1,i);fflush(stdout);
          NRB_posi2[i] = tmp_num1;
          tmp_num2 = -1;
        }
      } /* i */


/*************************************************************************
   3. Initial division between orthogonal valence and Rydberg AO spaces
*************************************************************************/

  /************************************
     3-1. OWSO of NMB orbitals (Ow)
  ************************************/

if (myid == Host_ID && outputlev == 1) printf("<< 3-1 >> OWSO of NMB orbitals (Ow) \n");
if (measure_time==1) dtime(&Stime1);

  /***  Sw (= W * S * W) Matrix  ***/
/*
      for (i=0; i<Num_NMB; i++){
      for (j=0; j<Num_NMB; j++){
        Sw1[i+1][j+1] = W_NMB[i] * S_NMB[i][j] * W_NMB[j];
        Uw1[i+1][j+1] = Sw1[i+1][j+1];
      } 
      } 
*/
      for (i=0; i<Num_NMB; i++){
        tmp_ele0 = W_NMB[i];
      for (j=0; j<Num_NMB; j++){
        tmp_ele2 = tmp_ele0 * S_NMB[i][j] * W_NMB[j];
        Sw1[i+1][j+1] = tmp_ele2;
        Uw1[i+1][j+1] = tmp_ele2;
      }
      }

    if (outputlev==1){
      printf("\n ### Sw1 ### \n");fflush(stdout);
      for (i=1; i<=Num_NMB; i++){
      for (j=1; j<=Num_NMB; j++){
        printf("%9.5f",Sw1[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

  /***  Eigenvalue problem, Sw * Uw = Rw * Uw  ***/

       Eigen_lapack(Uw1, Rw1, Num_NMB, Num_NMB);

    if (outputlev==1){
       printf("\n ### Uw1 ### \n");fflush(stdout);
       for (i=1; i<=Num_NMB; i++){
       for (j=1; j<=Num_NMB; j++){
         printf("%9.5f",Uw1[i][j]);fflush(stdout);
       }
         printf("\n");
       }

       printf("\n ### Rw1 (1) ### \n");fflush(stdout);
       for (i=1; i<=Num_NMB; i++){
         printf("%9.5f \n",Rw1[i]);fflush(stdout);
       }
    }

       for (l=1; l<=Num_NMB; l++){
         if (Rw1[l]<1.0e-14) {
          printf("Found ill-conditioned eigenvalues (2)\n");
          printf("Stopped calculation\n");
          exit(1);
         }
       }

  /***  Rw^{-1/2}  ***/

       for (l=1; l<=Num_NMB; l++){
         Rw1[l] = 1.0/sqrt(Rw1[l]);
       }

    if (outputlev==1){
       printf("\n ### Rw1 (2) ### \n");fflush(stdout);
       for (i=1; i<=Num_NMB; i++){
         printf("%9.5f \n",Rw1[i]);fflush(stdout);
       }
    }

  /***  Sw^{-1/2} = Uw * Rw^{-1/2} * Uw^{t}  ***/
/*
      for (i=1; i<=Num_NMB; i++){
      for (j=1; j<=Num_NMB; j++){
        temp_M1[i][j] = Rw1[i] * Uw1[j][i];
      } 
      } 

      for (i=1; i<=Num_NMB; i++){
      for (j=1; j<=Num_NMB; j++){
        tmp_ele1 = 0.0;
      for (k=1; k<=Num_NMB; k++){
        tmp_ele1 += Uw1[i][k] * temp_M1[k][j];
      } 
        Sw1[i][j] = tmp_ele1;
      } 
      } 
*/
      for (i=1; i<=Num_NMB; i++){
        tmp_ele0 = Rw1[i];
      for (j=1; j<=Num_NMB; j++){
        temp_M1[j][i] = tmp_ele0 * Uw1[j][i];
      }
      }

      for (i=1; i<=Num_NMB; i++){
      for (j=1; j<=Num_NMB; j++){
        tmp_ele1 = 0.0;
      for (k=1; k<=Num_NMB; k++){
        tmp_ele1 += Uw1[i][k] * temp_M1[j][k];
      }
        Sw1[i][j] = tmp_ele1;
      }
      }

    if (outputlev==1){
      printf("\n ### Sw1^{-1/2} ### \n");fflush(stdout);
      for (i=1; i<=Num_NMB; i++){
      for (j=1; j<=Num_NMB; j++){
        printf("%9.5f",Sw1[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

  /***  Ow = W * Sw^{-1/2}  ***/
/*
      for (i=0; i<Num_NMB; i++){
      for (j=0; j<Num_NMB; j++){
        Ow_NMB2[i+1][j+1] = W_NMB[i] * Sw1[i+1][j+1];
      } 
      } 
*/
      for (i=0; i<Num_NMB; i++){
        tmp_ele0 = W_NMB[i];
      for (j=0; j<Num_NMB; j++){
        Ow_NMB2[i+1][j+1] = tmp_ele0 * Sw1[i+1][j+1];
      } /* j */
      } /* i */

    if (outputlev==1){
      printf("\n ### Ow_NMB2 ### \n");fflush(stdout);
      for (i=1; i<=Num_NMB; i++){
      for (j=1; j<=Num_NMB; j++){
        printf("%9.5f",Ow_NMB2[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

if (measure_time==1){
  dtime(&Etime1);
  time5 = Etime1 - Stime1;
}

  /*******************************************************************************
     3-2. Schmidt interatomic orthogonalization of NRB to NMB orbitals (O_schm)
  *******************************************************************************/

if (myid == Host_ID && outputlev == 1) 
   printf("<< 3-2 >> Schmidt interatomic orthogonalization of NRB to NMB (O_schm) \n");
if (measure_time==1) dtime(&Stime1);

  /***  Overlap matrix for NMB and NRB  ***/

      for (i=0; i<Num_NMB; i++){
      for (j=0; j<Num_NMB; j++){
        N_tran2[NMB_posi1[i]][NMB_posi1[j]] = Ow_NMB2[i+1][j+1];
      } /* j */
      } /* i */

    if (outputlev==1){
      printf("\n ### Ow_NMB ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",N_tran2[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }
/*
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
        tmp_ele3 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full[i][k] * Ow_NMB[k][j];
        tmp_ele2 += P_full[i][k] * Ow_NMB[k][j];
        tmp_ele3 += N_diag[i][k] * Ow_NMB[k][j];
      } 
        temp_M1[i][j] = tmp_ele1;
        temp_M2[i][j] = tmp_ele2;
        T_full[i][j]  = tmp_ele3;
      } 
      } 

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += Ow_NMB[k][i] * temp_M1[k][j];
        tmp_ele2 += Ow_NMB[k][i] * temp_M2[k][j];
      } 
        S_full[i][j] = tmp_ele1;
        P_full[i][j] = tmp_ele2;
        O_schm[i][j] = T_full[i][j];
      } 
      } 
*/
   /* BLAS */
     /*  Ow_NMB * S * Ow_NMB  */
if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = N_tran2[i][j];
       temp_V2[tnum] = S_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       S_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  Ow_NMB * P * Ow_NMB  */
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = P_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  T * Ow_NMB  */

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = N_tran1[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[j][i] = temp_V3[tnum];
       tnum++;
     }
     }
}
else{

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M3[j][i] = N_tran2[i][j];
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M3[j][k];
       tmp_ele1 = temp_M3[j][k+1];
       tmp_ele2 = temp_M3[j][k+2];
       tmp_ele3 = temp_M3[j][k+3];

       tmp_ele00 += S_full[i][k]   * tmp_ele0;
       tmp_ele01 += S_full[i][k+1] * tmp_ele1;
       tmp_ele02 += S_full[i][k+2] * tmp_ele2;
       tmp_ele03 += S_full[i][k+3] * tmp_ele3;

       tmp_ele10 += P_full[i][k]   * tmp_ele0;
       tmp_ele11 += P_full[i][k+1] * tmp_ele1;
       tmp_ele12 += P_full[i][k+2] * tmp_ele2;
       tmp_ele13 += P_full[i][k+3] * tmp_ele3;

       tmp_ele20 += N_tran1[i][k]   * tmp_ele0;
       tmp_ele21 += N_tran1[i][k+1] * tmp_ele1;
       tmp_ele22 += N_tran1[i][k+2] * tmp_ele2;
       tmp_ele23 += N_tran1[i][k+3] * tmp_ele3;
     }
       temp_M1[j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       temp_M2[j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
       T_full[i][j]  = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
     }
     }

     is = SizeMat - SizeMat%4;

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
     for (k=is; k<SizeMat; k++){
/*
        tmp_ele00 += S_full[i][k] * Ow_NMB[k][j];
        tmp_ele10 += P_full[i][k] * Ow_NMB[k][j];
        tmp_ele20 += N_diag[i][k] * Ow_NMB[k][j];
*/
        tmp_ele0 = temp_M3[j][k];
        tmp_ele00 += S_full[i][k] * tmp_ele0;
        tmp_ele10 += P_full[i][k] * tmp_ele0;
        tmp_ele20 += N_tran1[i][k] * tmp_ele0;
     }
       temp_M1[j][i] += tmp_ele00;
       temp_M2[j][i] += tmp_ele10;
       T_full[i][j]  += tmp_ele20;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M3[i][k];
       tmp_ele1 = temp_M3[i][k+1];
       tmp_ele2 = temp_M3[i][k+2];
       tmp_ele3 = temp_M3[i][k+3];

       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele01 += tmp_ele1 * temp_M1[j][k+1];
       tmp_ele02 += tmp_ele2 * temp_M1[j][k+2];
       tmp_ele03 += tmp_ele3 * temp_M1[j][k+3];

       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
       tmp_ele11 += tmp_ele1 * temp_M2[j][k+1];
       tmp_ele12 += tmp_ele2 * temp_M2[j][k+2];
       tmp_ele13 += tmp_ele3 * temp_M2[j][k+3];

     }
       S_full[i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       P_full[i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
/*
       tmp_ele00 += Ow_NMB[k][i] * temp_M1[j][k];
       tmp_ele10 += Ow_NMB[k][i] * temp_M2[j][k];
*/
       tmp_ele0 = temp_M3[i][k];
       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
     }
       S_full[i][j] += tmp_ele00;
       P_full[i][j] += tmp_ele10;
     }
     }
}

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       N_tran1[i][j] = T_full[i][j];
     }
     }


    if (outputlev==1){
       printf("\n ### S_full (2) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",S_full[i][j]);fflush(stdout);
       }
         printf("\n");
       }

       printf("\n ### P_full (2) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",P_full[i][j]);fflush(stdout);
       }
         printf("\n");
       }

       printf("\n ### T_full (2) (= N_diag * Ow_NMB) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",T_full[i][j]);fflush(stdout);
       }
         printf("\n");
       }
    }

/* ### TEST ### */
    if (outputlev==1){
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full0[i][k] * T_full[k][j];
/*
        tmp_ele1 += NBO_OLPn_tmp[FS_atm][i][k] * T_full[k][j];
*/
      } /* k */
        temp_M1[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += T_full[k][i] * temp_M1[k][j];
      } /* k */
        temp_M3[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      printf("\n ### S_full (TEST 1) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",temp_M3[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }
/* ### TEST ### */


  /***  Schmidt orthogonalization  ***/

      for (i=0; i<Num_NRB; i++){
      for (k=0; k<SizeMat; k++){
        tmp_ele1 = 0.0;
      for (j=0; j<Num_NMB; j++){
        tmp_ele1 += T_full[k][NMB_posi1[j]] * S_full[NMB_posi1[j]][NRB_posi1[i]];
      } /* j */
        N_tran1[k][NRB_posi1[i]] -= tmp_ele1;
      } /* k */
      } /* i */

    if (outputlev==1){
      printf("\n ### O_schm ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",N_tran1[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         N_tran2[i][j]=0.0;
       }
       }
       for (i=0; i<SizeMat; i++){
         N_tran2[i][i]=1.0;
       }

if (measure_time==1){
  dtime(&Etime1);
  time6 = Etime1 - Stime1;
}


  /****************************************************
     3-3. Symmetry averaging of NRB orbitals (N_ryd)
  ****************************************************/

if (myid == Host_ID && outputlev == 1)
   printf("<< 3-3 >> Symmetry averaging of NRB orbitals (N_ryd) \n");
if (measure_time==1) dtime(&Stime1);

  /***  Overlap & density matrices (full size)  ***/
/*
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full0[i][k] * O_schm[k][j];
        tmp_ele2 += P_full0[i][k] * O_schm[k][j];
      } 
        temp_M1[i][j] = tmp_ele1;
        temp_M2[i][j] = tmp_ele2;
      } 
      } 

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += O_schm[k][i] * temp_M1[k][j];
        tmp_ele2 += O_schm[k][i] * temp_M2[k][j];
      } 
        S_full[i][j] = tmp_ele1;
        P_full[i][j] = tmp_ele2;
        T_full[i][j] = O_schm[i][j];
      } 
      } 
*/

   /* BLAS */
     /*  O_schm * S * O_schm  */
if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = N_tran1[i][j];
       temp_V2[tnum] = S_full0[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       S_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  O_schm * P * O_schm  */
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = P_full0[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }
}
else{

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M3[j][i] = N_tran1[i][j];
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){

       tmp_ele0 = temp_M3[j][k];
       tmp_ele1 = temp_M3[j][k+1];
       tmp_ele2 = temp_M3[j][k+2];
       tmp_ele3 = temp_M3[j][k+3];

       tmp_ele00 += S_full0[i][k]   * tmp_ele0;
       tmp_ele01 += S_full0[i][k+1] * tmp_ele1;
       tmp_ele02 += S_full0[i][k+2] * tmp_ele2;
       tmp_ele03 += S_full0[i][k+3] * tmp_ele3;

       tmp_ele10 += P_full0[i][k]   * tmp_ele0;
       tmp_ele11 += P_full0[i][k+1] * tmp_ele1;
       tmp_ele12 += P_full0[i][k+2] * tmp_ele2;
       tmp_ele13 += P_full0[i][k+3] * tmp_ele3;

     }
       temp_M1[j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       temp_M2[j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     is = SizeMat - SizeMat%4;

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
        tmp_ele0 = temp_M3[j][k];
        tmp_ele00 += S_full0[i][k] * tmp_ele0;
        tmp_ele10 += P_full0[i][k] * tmp_ele0;
     }
       temp_M1[j][i] += tmp_ele00;
       temp_M2[j][i] += tmp_ele10;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M3[i][k];
       tmp_ele1 = temp_M3[i][k+1];
       tmp_ele2 = temp_M3[i][k+2];
       tmp_ele3 = temp_M3[i][k+3];

       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele01 += tmp_ele1 * temp_M1[j][k+1];
       tmp_ele02 += tmp_ele2 * temp_M1[j][k+2];
       tmp_ele03 += tmp_ele3 * temp_M1[j][k+3];

       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
       tmp_ele11 += tmp_ele1 * temp_M2[j][k+1];
       tmp_ele12 += tmp_ele2 * temp_M2[j][k+2];
       tmp_ele13 += tmp_ele3 * temp_M2[j][k+3];
     }
       S_full[i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       P_full[i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
       tmp_ele0 = temp_M3[i][k];
       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
     }
       S_full[i][j] += tmp_ele00;
       P_full[i][j] += tmp_ele10;
     }
     }
}

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[i][j] = N_tran1[i][j];
     }
     }

    if (outputlev==1){
      printf("\n ### S_full (3) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",S_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### P_full (3) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",P_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### T_full (3) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",T_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

  /***  Atomic & anglar- & magetic-momentum block matrices  ***/

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Leng_alm[FS_AN][L1];
     for (M1=1; M1<=2*L1+1; M1++){
        posi1 = Posi_alm[FS_AN][L1][M1];
       for (i=0; i<leng1; i++){
       for (j=0; j<leng1; j++){
          P_alm[FS_AN][L1][M1][i][j] = P_full[i+posi1][j+posi1];
          S_alm[FS_AN][L1][M1][i][j] = S_full[i+posi1][j+posi1];
       } /* j */
       } /* i */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

  /***  Partitioning and symmetry averaging of P & S  ***/

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Leng_alm[FS_AN][L1];
       for (i=0; i<leng1; i++){
       for (j=0; j<leng1; j++){
         tmp_ele1 = 0.0;
         tmp_ele2 = 0.0;
         if (M_or_R1[FS_AN][L1][i]==0 && M_or_R1[FS_AN][L1][j]==0){
           for (M1=1; M1<=L1*2+1; M1++){
             tmp_ele1 += P_alm[FS_AN][L1][M1][i][j];
             tmp_ele2 += S_alm[FS_AN][L1][M1][i][j];
           } /* M1 */
             tmp_ele1 /= L1*2.0+1.0;
             tmp_ele2 /= L1*2.0+1.0;
           for (M1=1; M1<=L1*2+1; M1++){
             P_alm[FS_AN][L1][M1][i][j] = tmp_ele1;
             S_alm[FS_AN][L1][M1][i][j] = tmp_ele2;
           } /* M1 */
         }     
       } /* j */
       } /* i */
     } /* L1 */
     } /* FS_AN */

  if (1<0){
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Leng_alm[FS_AN][L1];
     for (M1=1; M1<=L1*2+1; M1++){
       for (i=0; i<leng1; i++){
       for (j=0; j<leng1; j++){
         printf("%9.5f",S_alm[FS_AN][L1][M1][i][j]);fflush(stdout);
       }
         printf("\n");fflush(stdout);
       }
     } /* M1 */
     } /* L1 */
     } /* FS_AN */
  }

  /***  Atom & anglar-momentum loop  ***/

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        Mmax1 = Leng_alm[FS_AN][L1];

  /***  Transferring of matrices (1)  ***/

       tmp_num1 = -1;

       for (i=0; i<Mmax1; i++){
        if (M_or_R1[FS_AN][L1][i]==0){
          tmp_num1 += 1;
          NRB_posi3[tmp_num1+1] = i;
          tmp_num2 = -1;
       for (j=0; j<Mmax1; j++){
        if (M_or_R1[FS_AN][L1][j]==0){
          tmp_num2 += 1;
          P_alm2[tmp_num1+1][tmp_num2+1] = P_alm[FS_AN][L1][1][i][j];
          S_alm2[tmp_num1+1][tmp_num2+1] = S_alm[FS_AN][L1][1][i][j];
        }
       } /* j */
        }
       } /* i */

       Mmax2 = tmp_num1 + 1;
    if (outputlev==1) printf("\n  Mmax2 = %d \n",Mmax2);fflush(stdout);

       if (Mmax2 != 0){

    if (outputlev==1){
       printf("### S_alm2[%d][%d] ### \n",FS_AN,L1);fflush(stdout);
       for (i=0; i<Mmax2; i++){
       for (j=0; j<Mmax2; j++){
          printf("%16.12f",S_alm2[i+1][j+1]);fflush(stdout);
       } /* j */
          printf("\n");fflush(stdout);
       } /* j */

       printf("\n");fflush(stdout);

       printf("### P_alm2[%d][%d] ### \n",FS_AN,L1);fflush(stdout);
       for (i=0; i<Mmax2; i++){
       for (j=0; j<Mmax2; j++){
          printf("%16.12f",P_alm2[i+1][j+1]);fflush(stdout);
       } /* j */
          printf("\n");fflush(stdout);
       } /* i */
    }

  /***  Generalized eigenvalue problem, P * N = S * N * W  ***/

  /*******************************************
     Diagonalizing the overlap matrix

      First:
        S -> OLP matrix
      After calling Eigen_lapack:
        S -> eigenvectors of OLP matrix
   *******************************************/

  /***  Generalized eigenvalue problem, P * N = S * N * W  ***/

  if (outputlev==1) printf("\n Diagonalize the overlap matrix \n");

    Eigen_lapack(S_alm2,W_alm2,Mmax2,Mmax2);

  if (outputlev==1){
    for (k=1; k<=Mmax2; k++){
      printf("k W %2d %15.12f\n",k,W_alm2[k]);
    }
  }

    /* check ill-conditioned eigenvalues */

    for (k=1; k<=Mmax2; k++){
      if (W_alm2[k]<1.0e-14) {
       printf("Found ill-conditioned eigenvalues (3)\n");
       printf("Stopped calculation\n");
       exit(1);
      }
    }

    for (k=1; k<=Mmax2; k++){
      temp_V3[k] = 1.0/sqrt(W_alm2[k]);
    }

   /***  Calculations of eigenvalues  ***/

    for (i=1; i<=Mmax2; i++){
      for (j=1; j<=Mmax2; j++){
         sum = 0.0;
        for (k=1; k<=Mmax2; k++){
          sum = sum + P_alm2[i][k]*S_alm2[k][j]*temp_V3[j];
        }
        N_alm2[i][j] = sum;
      }
    }

    for (i=1; i<=Mmax2; i++){
      for (j=1; j<=Mmax2; j++){
         sum = 0.0;
        for (k=1; k<=Mmax2; k++){
          sum = sum + temp_V3[i]*S_alm2[k][i]*N_alm2[k][j];
        }
        temp_M1[i][j] = sum;
      }
    }

  if (outputlev==1){
    printf("\n ##### TEST 1 ##### \n");
    for (i=1; i<=Mmax2; i++){
      for (j=1; j<=Mmax2; j++){
        printf("%9.5f",temp_M1[i][j]);
      }
        printf("\n");
    }
  }
/*
    printf("\n Diagonalize the D matrix \n");
*/
    Eigen_lapack(temp_M1,W_alm2,Mmax2,Mmax2);

  if (outputlev==1){
    printf("\n ##### TEST 2 ##### \n");
    for (k=1; k<=Mmax2; k++){
      printf("k W %2d %9.5f\n",k,W_alm2[k]);
    }

    printf("\n ##### TEST 3 ##### \n");
    for (i=1; i<=Mmax2; i++){
      for (j=1; j<=Mmax2; j++){
        printf("%9.5f",temp_M1[i][j]);
      }
        printf("\n");
    }
    printf("\n");
  }

    /***  Transformation to the original eigen vectors  ***/

    for (i=1; i<=Mmax2; i++){
      for (j=1; j<=Mmax2; j++){
        N_alm2[i][j] = 0.0;
      }
    }

    for (i=1; i<=Mmax2; i++){
      for (j=1; j<=Mmax2; j++){
         sum = 0.0;
        for (k=1; k<=Mmax2; k++){
          sum = sum + S_alm2[i][k]*temp_V3[k]*temp_M1[k][j];
        }
    /*  N_alm2[j][Mmax1-i+1] = sum;  */
        N_alm2[i][j] = sum;
      }
    }

  /* printing out eigenvalues and the eigenvectors */

    if (outputlev==1){
      for (i=1; i<=Mmax2; i++){
        printf("%ith eigenvalue of HC=eSC: %15.12f\n",i,W_alm2[i]);
      }

        for (i=1; i<=Mmax2; i++){
           printf("%ith eigenvector: ",i);
           printf("{");
        for (j=1; j<=Mmax2; j++){
           printf("%7.4f,",N_alm2[i][j]);
        }
           printf("}\n");
        }
    }

  /***  Transferring of matrices (2)  ***/

       for (M1=1; M1<=L1*2+1; M1++){
         posi1 = Posi_alm[FS_AN][L1][M1];
       for (i=1; i<=Mmax2; i++){
         W_full[posi1+NRB_posi3[i]] = W_alm2[i];
       for (j=1; j<=Mmax2; j++){
         N_tran2[posi1+NRB_posi3[i]][posi1+NRB_posi3[j]] = N_alm2[i][j];
       } /* j */
       } /* i */
       } /* M1 */

     } /* if Mmax2 != 0 */

     } /* L1 */
     } /* FS_AN */

     if (outputlev==1){
       printf("\n ### N_ryd ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",N_tran2[i][j]);fflush(stdout);
       }
         printf("\n");
       }
     }

  /***  Overlap matrix (full size)  ***/
/*
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
        tmp_ele3 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full[i][k] * N_ryd[k][j];
        tmp_ele2 += P_full[i][k] * N_ryd[k][j];
        tmp_ele3 += T_full[i][k] * N_ryd[k][j];
      } 
        temp_M1[i][j] = tmp_ele1;
        temp_M2[i][j] = tmp_ele2;
        temp_M3[i][j] = tmp_ele3;
      } 
      } 

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += N_ryd[k][i] * temp_M1[k][j];
        tmp_ele2 += N_ryd[k][i] * temp_M2[k][j];
      } 
        S_full[i][j] = tmp_ele1;
        P_full[i][j] = tmp_ele2;
        T_full[i][j] = temp_M3[i][j];
      } 
      } 
*/
   /* BLAS */
     /*  N_ryd * S * N_ryd  */
if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = N_tran2[i][j];
       temp_V2[tnum] = S_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       S_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  N_ryd * P * N_ryd  */
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = P_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  T * N_ryd  */

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = T_full[j][i];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[j][i] = temp_V3[tnum];
       tnum++;
     }
     }
}
else{
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M4[j][i] = N_tran2[i][j];
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M4[j][k];
       tmp_ele1 = temp_M4[j][k+1];
       tmp_ele2 = temp_M4[j][k+2];
       tmp_ele3 = temp_M4[j][k+3];

       tmp_ele00 += S_full[i][k]   * tmp_ele0;
       tmp_ele01 += S_full[i][k+1] * tmp_ele1;
       tmp_ele02 += S_full[i][k+2] * tmp_ele2;
       tmp_ele03 += S_full[i][k+3] * tmp_ele3;

       tmp_ele10 += P_full[i][k]   * tmp_ele0;
       tmp_ele11 += P_full[i][k+1] * tmp_ele1;
       tmp_ele12 += P_full[i][k+2] * tmp_ele2;
       tmp_ele13 += P_full[i][k+3] * tmp_ele3;

       tmp_ele20 += T_full[i][k]   * tmp_ele0;
       tmp_ele21 += T_full[i][k+1] * tmp_ele1;
       tmp_ele22 += T_full[i][k+2] * tmp_ele2;
       tmp_ele23 += T_full[i][k+3] * tmp_ele3;
     }
       temp_M1[j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       temp_M2[j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
       temp_M3[i][j] = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
     }
     }

     is = SizeMat - SizeMat%4;

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
     for (k=is; k<SizeMat; k++){
        tmp_ele0 = temp_M4[j][k];
        tmp_ele00 += S_full[i][k] * tmp_ele0;
        tmp_ele10 += P_full[i][k] * tmp_ele0;
        tmp_ele20 += T_full[i][k] * tmp_ele0;
     }
       temp_M1[j][i] += tmp_ele00;
       temp_M2[j][i] += tmp_ele10;
       temp_M3[i][j] += tmp_ele20;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M4[i][k];
       tmp_ele1 = temp_M4[i][k+1];
       tmp_ele2 = temp_M4[i][k+2];
       tmp_ele3 = temp_M4[i][k+3];

       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele01 += tmp_ele1 * temp_M1[j][k+1];
       tmp_ele02 += tmp_ele2 * temp_M1[j][k+2];
       tmp_ele03 += tmp_ele3 * temp_M1[j][k+3];

       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
       tmp_ele11 += tmp_ele1 * temp_M2[j][k+1];
       tmp_ele12 += tmp_ele2 * temp_M2[j][k+2];
       tmp_ele13 += tmp_ele3 * temp_M2[j][k+3];
     }
       S_full[i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       P_full[i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
       tmp_ele0 = temp_M4[i][k];
       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
     }
       S_full[i][j] += tmp_ele00;
       P_full[i][j] += tmp_ele10;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[i][j] = temp_M3[i][j];
    }
    }
}

    if (outputlev==1){
       printf("\n ### S_full (4) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",S_full[i][j]);fflush(stdout);
       }
         printf("\n");
       }

       printf("\n ### P_full (4) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",P_full[i][j]);fflush(stdout);
       }
         printf("\n");
       }

       printf("\n ### W_full (4) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
         printf("%9.5f \n",W_full[i]);fflush(stdout);
       }

       printf("\n ### T_full (4) (= N_diag * Ow_NMB * O_schm * N_ryd) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",T_full[i][j]);fflush(stdout);
       }
         printf("\n");
       }
    }


/* ### TEST ### */
    if (outputlev==1){
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full0[i][k] * T_full[k][j];
/*
        tmp_ele1 += NBO_OLPn_tmp[FS_atm][i][k] * T_full[k][j];
*/
      } /* k */
        temp_M1[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += T_full[k][i] * temp_M1[k][j];
      } /* k */
        temp_M3[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      printf("\n ### S_full (TEST 2) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",temp_M3[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }
/* ### TEST ### */

  /***  Allocation of arrays (5)  ***/

    S_NRB = (double**)malloc(sizeof(double*)*(Num_NRB+1));
     for (i=0; i<=Num_NRB; i++){
     S_NRB[i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
      for (j=0; j<=Num_NRB; j++){
      S_NRB[i][j] = 0.0;
      }
     }

    S_NRB2 = (double**)malloc(sizeof(double*)*(Num_NRB+1));
     for (i=0; i<=Num_NRB; i++){
     S_NRB2[i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
      for (j=0; j<=Num_NRB; j++){
      S_NRB2[i][j] = 0.0;
      }
     }

    W_NRB = (double*)malloc(sizeof(double)*(Num_NRB+1));
     for (i=0; i<=Num_NRB; i++){
     W_NRB[i] = 0.0;
     }

    W_NRB2 = (double*)malloc(sizeof(double)*(Num_NRB+1));
     for (i=0; i<=Num_NRB; i++){
     W_NRB2[i] = 0.0;
     }

    Ow_NRB2 = (double**)malloc(sizeof(double*)*(Num_NRB+1));
     for (i=0; i<=Num_NRB; i++){
     Ow_NRB2[i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
      for (j=0; j<=Num_NRB; j++){
      Ow_NRB2[i][j] = 0.0;
      }
     }

    Sw2 = (double**)malloc(sizeof(double*)*(Num_NRB+1));
     for (i=0; i<=Num_NRB; i++){
     Sw2[i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
      for (j=0; j<=Num_NRB; j++){
      Sw2[i][j] = 0.0;
      }
     }

    Uw2 = (double**)malloc(sizeof(double*)*(Num_NRB+1));
     for (i=0; i<=Num_NRB; i++){
     Uw2[i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
      for (j=0; j<=Num_NRB; j++){
      Uw2[i][j] = 0.0;
      }
     }

    Rw2 = (double*)malloc(sizeof(double)*(Num_NRB+1));
     for (i=0; i<=Num_NRB; i++){
     Rw2[i] = 0.0;
     }

   S_or_L = (int*)malloc(sizeof(int)*(Num_NRB+1));
    for (i=0; i<=Num_NRB; i++){
    S_or_L[i] = 0;
    }

   NRB_posi4= (int*)malloc(sizeof(int)*(Num_NRB+1));
    for (i=0; i<=Num_NRB; i++){
    NRB_posi4[i] = 0;
    }

   NRB_posi5= (int*)malloc(sizeof(int)*(Num_NRB+1));
    for (i=0; i<=Num_NRB; i++){
    NRB_posi5[i] = 0;
    }

   NRB_posi6= (int*)malloc(sizeof(int)*(Num_NRB+1));
    for (i=0; i<=Num_NRB; i++){
    NRB_posi6[i] = 0;
    }

   NRB_posi7= (int*)malloc(sizeof(int)*(Num_NRB+1));
    for (i=0; i<=Num_NRB; i++){
    NRB_posi7[i] = 0;
    }

   NRB_posi8= (int*)malloc(sizeof(int)*(Num_NRB+1));
    for (i=0; i<=Num_NRB; i++){
    NRB_posi8[i] = 0;
    }

   NRB_posi9= (int*)malloc(sizeof(int)*(SizeMat+1));
    for (i=0; i<=SizeMat; i++){
    NRB_posi9[i] = 0;
    }

  /***  Matrices for NRB  ***/

      tmp_num1 = -1;

      for (i=0; i<SizeMat; i++){
        if (M_or_R2[i]==0){
          tmp_num1 += 1;
          W_NRB[tmp_num1] = W_full[i];
          NRB_posi8[tmp_num1] = i;
          NRB_posi9[i] = tmp_num1;
          tmp_num2 = -1;
         for (j=0; j<SizeMat; j++){
           if (M_or_R2[j]==0){
             tmp_num2 += 1;
             S_NRB[tmp_num1][tmp_num2] = S_full[i][j];
           }
         } /* j */
        }
      } /* i */

    if (outputlev==1){
      printf("\n ### S_NRB (5) ### \n");fflush(stdout);
      for (i=0; i<Num_NRB; i++){
      for (j=0; j<Num_NRB; j++){
        printf("%9.5f",S_NRB[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

    if (outputlev==1) printf("\n ### W_NRB ### \n");fflush(stdout);

      tmp_num1 = -1;
      tmp_num2 = -1;
      for (i=0; i<Num_NRB; i++){
       if(1<0) printf("%9.5f \n",W_NRB[i]);fflush(stdout);
        if(W_NRB[i] >= 1.0e-4){
         S_or_L[i] = 1;
         tmp_num1 += 1;
         NRB_posi4[i] = tmp_num1;
         NRB_posi5[tmp_num1+1] = i;
        }
        else{
         S_or_L[i] = 0;
         tmp_num2 += 1;
         NRB_posi6[i] = tmp_num2;
         NRB_posi7[tmp_num2+1] = i;
        }
      }

      Num_NRB2 = tmp_num1+1;
      Num_NRB3 = Num_NRB-Num_NRB2;

    if (outputlev==1){
    if (myid == Host_ID){
      printf("\n Num_NRB2 = %d   ",Num_NRB2);fflush(stdout);
      printf("\n Num_NRB3 = %d \n",Num_NRB3);fflush(stdout);
    }
    }

    if (Num_NRB2 == 0 && Num_NRB3 == 0){
      printf("\n ### STOP (Num_NRB2=0 & Num_NRB3=0) (ID=%d) \n",myid);
      exit(0);
    }

       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         N_tran1[i][j]=0.0;
       }
       }
       for (i=0; i<SizeMat; i++){
         N_tran1[i][i]=1.0;
       }

if (measure_time==1){
  dtime(&Etime1);
  time7 = Etime1 - Stime1;
}


  /************************************
     3-4. OWSO of NRB orbitals (Ow)
  ************************************/

if (myid == Host_ID && outputlev == 1) printf("<< 3-4 >> OWSO of NRB orbitals (Ow) \n");
if (measure_time==1) dtime(&Stime1);

  /***  Sw (= W * S * W) Matrix  ***/
/*
      for (i=0; i<Num_NRB; i++){
      for (j=0; j<Num_NRB; j++){
       if(S_or_L[i] == 1 && S_or_L[j] == 1){
        k = NRB_posi4[i];
        l = NRB_posi4[j];
        Sw2[k+1][l+1] = W_NRB[i] * S_NRB[i][j] * W_NRB[j];
        Uw2[k+1][l+1] = Sw2[k+1][l+1];
        W_NRB2[k+1]   = W_NRB[i];
       }
      } 
      } 
*/
      for (i=0; i<Num_NRB; i++){
        tmp_ele1 = W_NRB[i];
      for (j=0; j<Num_NRB; j++){
       if(S_or_L[i] == 1 && S_or_L[j] == 1){
        k = NRB_posi4[i];
        l = NRB_posi4[j];
        tmp_ele2 = tmp_ele1 * S_NRB[i][j] * W_NRB[j];
        Sw2[k+1][l+1] = tmp_ele2;
        Uw2[k+1][l+1] = tmp_ele2;
        W_NRB2[k+1]   = tmp_ele1;
       }
      } /* j */
      } /* i */

    if (outputlev==1){
      printf("\n ### Sw2 ### \n");fflush(stdout);
      for (i=1; i<=Num_NRB2; i++){
      for (j=1; j<=Num_NRB2; j++){
        printf("%9.5f",Sw2[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }
  
  /***  Eigenvalue problem, Sw * Uw = Rw * Uw  ***/

     Eigen_lapack(Uw2, Rw2, Num_NRB2, Num_NRB2);

     if (outputlev==1){
       printf("\n ### Uw2 ### \n");fflush(stdout);
       for (i=1; i<=Num_NRB2; i++){
       for (j=1; j<=Num_NRB2; j++){
         printf("%9.5f",Uw2[i][j]);fflush(stdout);
       }
         printf("\n");
       }

       printf("\n ### Rw2 (1) ### \n");fflush(stdout);
       for (i=1; i<=Num_NRB2; i++){
         printf("%9.5f \n",Rw2[i]);fflush(stdout);
       }
     }

       for (l=1; l<=Num_NRB2; l++){
         if (Rw2[l]<1.0e-14) {
          printf("Found ill-conditioned eigenvalues (4)\n");
          printf("Stopped calculation\n");
          exit(1);
         }
       }

  /***  Rw^{-1/2}  ***/

       for (l=1; l<=Num_NRB2; l++){
         Rw2[l] = 1.0/sqrt(Rw2[l]);
       }

    if (outputlev==1){
       printf("\n ### Rw2 (2) ### \n");fflush(stdout);
       for (i=1; i<=Num_NRB2; i++){
         printf("%9.5f \n",Rw2[i]);fflush(stdout);
       }
    }

  /***  Sw^{-1/2} = Uw * Rw^{-1/2} * Uw^{t}  ***/
/*
      for (i=1; i<=Num_NRB2; i++){
      for (j=1; j<=Num_NRB2; j++){
        temp_M1[i][j] = Rw2[i] * Uw2[j][i];
      } 
      } 

      for (i=1; i<=Num_NRB2; i++){
      for (j=1; j<=Num_NRB2; j++){
        tmp_ele1 = 0.0;
      for (k=1; k<=Num_NRB2; k++){
        tmp_ele1 += Uw2[i][k] * temp_M1[k][j];
      } 
        Sw2[i][j] = tmp_ele1;
      } 
      }
*/
      for (i=1; i<=Num_NRB2; i++){
      for (j=1; j<=Num_NRB2; j++){
        temp_M2[j][i] = Uw2[i][j];
      }
      }

      for (i=1; i<=Num_NRB2; i++){
        tmp_ele0 = Rw2[i];
      for (j=1; j<=Num_NRB2; j++){
        temp_M1[j][i] = tmp_ele0 * temp_M2[i][j];
      } /* j */
      } /* i */

/* BLAS */
if(1>0){
     tnum = 0;
     for (i=1; i<=Num_NRB2; i++){
     for (j=1; j<=Num_NRB2; j++){
       temp_V1[tnum] =     Uw2[j][i];
       temp_V2[tnum] = temp_M1[j][i];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = Num_NRB2; N = Num_NRB2; K = Num_NRB2;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V2, &ldb, &beta, temp_V3, &ldc);

     tnum = 0;
     for (i=1; i<=Num_NRB2; i++){
     for (j=1; j<=Num_NRB2; j++){
       Sw2[i][j] = temp_V3[tnum];
       tnum++;
     }
     }

}
else{
      for (i=1; i<=Num_NRB2; i++){
      for (j=1; j<=Num_NRB2; j++){
        tmp_ele1 = 0.0;
      for (k=1; k<=Num_NRB2; k++){
        tmp_ele1 += Uw2[i][k] * temp_M1[j][k];
      } /* k */
        Sw2[i][j] = tmp_ele1;
      } /* j */
      } /* i */
}


    if (outputlev==1){
      printf("\n ### Sw2^{-1/2} ### \n");fflush(stdout);
      for (i=1; i<=Num_NRB2; i++){
      for (j=1; j<=Num_NRB2; j++){
        printf("%19.10f",Sw2[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### W_NRB2 ### \n");fflush(stdout);
      for (i=1; i<=Num_NRB2; i++){
        printf("%9.5f \n",W_NRB2[i]);fflush(stdout);
      }
    }

  /***  Ow = W * Sw^{-1/2}  ***/
/*
      for (i=1; i<=Num_NRB2; i++){
      for (j=1; j<=Num_NRB2; j++){
        Ow_NRB2[i][j] = W_NRB2[i] * Sw2[i][j];
      } 
      } 
*/
      for (i=1; i<=Num_NRB2; i++){
        tmp_ele0 = W_NRB2[i];
      for (j=1; j<=Num_NRB2; j++){
        Ow_NRB2[i][j] = tmp_ele0 * Sw2[i][j];
      } /* j */
      } /* i */

     if (outputlev==1){
      printf("\n ### Ow_NRB2 ### \n");fflush(stdout);
      for (i=1; i<=Num_NRB2; i++){
      for (j=1; j<=Num_NRB2; j++){
        printf("%9.5f",Ow_NRB2[i][j]);fflush(stdout);
      }
        printf("\n");
      }
     }

      for (i=1; i<=Num_NRB2; i++){
      for (j=1; j<=Num_NRB2; j++){
        k = NRB_posi5[i];
        l = NRB_posi5[j];
        N_tran1[NRB_posi1[k]][NRB_posi1[l]] = Ow_NRB2[i][j];
      } /* j */
      } /* i */

    if (outputlev==1){
      printf("\n ### Ow_NRB ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",N_tran1[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

  /***  Overlap matrix for NMB and NRB  ***/
/*
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
        tmp_ele3 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full[i][k] * Ow_NRB[k][j];
        tmp_ele2 += P_full[i][k] * Ow_NRB[k][j];
        tmp_ele3 += T_full[i][k] * Ow_NRB[k][j];
      } 
        temp_M1[i][j] = tmp_ele1;
        temp_M2[i][j] = tmp_ele2;
        temp_M3[i][j] = tmp_ele3;
      } 
      } 

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += Ow_NRB[k][i] * temp_M1[k][j];
        tmp_ele2 += Ow_NRB[k][i] * temp_M2[k][j];
      } 
        S_full[i][j] = tmp_ele1;
        P_full[i][j] = tmp_ele2;
        T_full[i][j] = temp_M3[i][j];
        O_schm2[i][j] = T_full[i][j];
      } 
      } 
*/
   /* BLAS */
     /*  Ow_NRB * S * Ow_NRB  */
if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = N_tran1[i][j];
       temp_V2[tnum] = S_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       S_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  Ow_NRB * P * Ow_NRB  */
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = P_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  T * Ow_NRB  */

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = T_full[j][i];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[j][i] = temp_V3[tnum];
       N_tran2[j][i] = temp_V3[tnum];
       tnum++;
     }
     }
}
else{
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M4[j][i] = N_tran1[i][j];
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M4[j][k];
       tmp_ele1 = temp_M4[j][k+1];
       tmp_ele2 = temp_M4[j][k+2];
       tmp_ele3 = temp_M4[j][k+3];

       tmp_ele00 += S_full[i][k]   * tmp_ele0;
       tmp_ele01 += S_full[i][k+1] * tmp_ele1;
       tmp_ele02 += S_full[i][k+2] * tmp_ele2;
       tmp_ele03 += S_full[i][k+3] * tmp_ele3;

       tmp_ele10 += P_full[i][k]   * tmp_ele0;
       tmp_ele11 += P_full[i][k+1] * tmp_ele1;
       tmp_ele12 += P_full[i][k+2] * tmp_ele2;
       tmp_ele13 += P_full[i][k+3] * tmp_ele3;

       tmp_ele20 += T_full[i][k]   * tmp_ele0;
       tmp_ele21 += T_full[i][k+1] * tmp_ele1;
       tmp_ele22 += T_full[i][k+2] * tmp_ele2;
       tmp_ele23 += T_full[i][k+3] * tmp_ele3;
     }
       temp_M1[j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       temp_M2[j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
       temp_M3[i][j] = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
     }
     }

     is = SizeMat - SizeMat%4;

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
     for (k=is; k<SizeMat; k++){
        tmp_ele00 += S_full[i][k] * tmp_ele0;
        tmp_ele10 += P_full[i][k] * tmp_ele0;
        tmp_ele20 += T_full[i][k] * tmp_ele0;
     }
       temp_M1[j][i] += tmp_ele00;
       temp_M2[j][i] += tmp_ele10;
       temp_M3[i][j] += tmp_ele20;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M4[i][k];
       tmp_ele1 = temp_M4[i][k+1];
       tmp_ele2 = temp_M4[i][k+2];
       tmp_ele3 = temp_M4[i][k+3];

       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele01 += tmp_ele1 * temp_M1[j][k+1];
       tmp_ele02 += tmp_ele2 * temp_M1[j][k+2];
       tmp_ele03 += tmp_ele3 * temp_M1[j][k+3];

       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
       tmp_ele11 += tmp_ele1 * temp_M2[j][k+1];
       tmp_ele12 += tmp_ele2 * temp_M2[j][k+2];
       tmp_ele13 += tmp_ele3 * temp_M2[j][k+3];
     }
       S_full[i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       P_full[i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
       tmp_ele0 = temp_M4[i][k];
       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
     }
       S_full[i][j] += tmp_ele00;
       P_full[i][j] += tmp_ele10;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele0 = temp_M3[i][j];
       T_full[i][j]  = tmp_ele0;
       N_tran2[i][j] = tmp_ele0;
     }
     }
}

/* ### TEST ### */
    if (outputlev==1){
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full0[i][k] * T_full[k][j];
/*
        tmp_ele1 += NBO_OLPn_tmp[FS_atm][i][k] * T_full[k][j];
*/
      } /* k */
        temp_M1[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += T_full[k][i] * temp_M1[k][j];
      } /* k */
        temp_M3[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      printf("\n ### S_full (TEST 3) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",temp_M3[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }
/* ### TEST ### */

    if (outputlev==1){
      printf("\n ### S_full (5) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",S_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### P_full (5) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",P_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### T_full (5) (= N_diag * Ow_NMB * O_schm * N_ryd * Ow_NRB) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",T_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

  /***  Treatment of low-occupied NRBs  ***/

      if(Num_NRB3>0){
/*
      printf("\n ** Schmidt orthogonalization of low-occupied NRBs to high ones **\n");fflush(stdout);
*/
      for (i=0; i<Num_NRB3; i++){
        i2 = NRB_posi8[NRB_posi7[i+1]];
      for (k=0; k<SizeMat;  k++){
        tmp_ele1 = 0.0;
      for (j=0; j<Num_NRB2; j++){
        j2 = NRB_posi8[NRB_posi5[j+1]];
        tmp_ele1 += T_full[k][j2] * S_full[j2][i2];
      } /* j */
        N_tran2[k][i2] -= tmp_ele1;
      } /* k */
      } /* i */

    if(outputlev == 1){
      printf("\n ### O_schm2 ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",N_tran2[i][j]);fflush(stdout);
      }
        printf("\n");
      }
      printf("\n");
    }
/*
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full0[i][k] * O_schm2[k][j];
        tmp_ele2 += P_full0[i][k] * O_schm2[k][j];
      }
        temp_M1[i][j] = tmp_ele1;
        temp_M2[i][j] = tmp_ele2;
      }
      }

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += O_schm2[k][i] * temp_M1[k][j];
        tmp_ele2 += O_schm2[k][i] * temp_M2[k][j];
      }
        S_full[i][j] = tmp_ele1;
        P_full[i][j] = tmp_ele2;
        T_full[i][j] = O_schm2[i][j];
      }
      }
*/

   /* BLAS */
     /*  O_schm * S * O_schm  */
if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = N_tran2[i][j];
       temp_V2[tnum] = S_full0[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       S_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  O_schm * P * O_schm  */
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = P_full0[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }
}
else{
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M3[j][i] = N_tran2[i][j];
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){

       tmp_ele0 = temp_M3[j][k];
       tmp_ele1 = temp_M3[j][k+1];
       tmp_ele2 = temp_M3[j][k+2];
       tmp_ele3 = temp_M3[j][k+3];

       tmp_ele00 += S_full0[i][k]   * tmp_ele0;
       tmp_ele01 += S_full0[i][k+1] * tmp_ele1;
       tmp_ele02 += S_full0[i][k+2] * tmp_ele2;
       tmp_ele03 += S_full0[i][k+3] * tmp_ele3;

       tmp_ele10 += P_full0[i][k]   * tmp_ele0;
       tmp_ele11 += P_full0[i][k+1] * tmp_ele1;
       tmp_ele12 += P_full0[i][k+2] * tmp_ele2;
       tmp_ele13 += P_full0[i][k+3] * tmp_ele3;

     }
       temp_M1[j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       temp_M2[j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     is = SizeMat - SizeMat%4;

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
        tmp_ele0 = temp_M3[j][k];
        tmp_ele00 += S_full0[i][k] * tmp_ele0;
     }
       temp_M1[j][i] += tmp_ele00;
       temp_M2[j][i] += tmp_ele10;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M3[i][k];
       tmp_ele1 = temp_M3[i][k+1];
       tmp_ele2 = temp_M3[i][k+2];
       tmp_ele3 = temp_M3[i][k+3];

       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele01 += tmp_ele1 * temp_M1[j][k+1];
       tmp_ele02 += tmp_ele2 * temp_M1[j][k+2];
       tmp_ele03 += tmp_ele3 * temp_M1[j][k+3];

       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
       tmp_ele11 += tmp_ele1 * temp_M2[j][k+1];
       tmp_ele12 += tmp_ele2 * temp_M2[j][k+2];
       tmp_ele13 += tmp_ele3 * temp_M2[j][k+3];
     }
       S_full[i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       P_full[i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
     }
       S_full[i][j] += tmp_ele00;
       P_full[i][j] += tmp_ele10;
     }
     }
}

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[i][j] = N_tran2[i][j];
     }
     }


/* ### TEST ### */
    if (outputlev==1){
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full0[i][k] * T_full[k][j];
/*
        tmp_ele1 += NBO_OLPn_tmp[FS_atm][i][k] * T_full[k][j];
*/
      } /* k */
        temp_M1[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += T_full[k][i] * temp_M1[k][j];
      } /* k */
        temp_M3[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      printf("\n ### S_full (TEST 2) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",temp_M3[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }
/* ### TEST ### */


    if (outputlev==1){
      printf("\n ### S_full (6) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",S_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### P_full (6) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",P_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### T_full (6) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",T_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }
/*
      printf("\n ** Loewdin symmetric orthogonalization **\n");fflush(stdout);
*/

  /***  Set of overlap matrix for low-occpied NRBs (S_NRB2)  ***/

       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         N_tran1[i][j]=0.0;
       }
       }
       for (i=0; i<SizeMat; i++){
         N_tran1[i][i]=1.0;
       }

      for (i=1; i<=Num_NRB3; i++){
        i2 = NRB_posi8[NRB_posi7[i]];
      for (j=1; j<=Num_NRB3; j++){
        j2 = NRB_posi8[NRB_posi7[j]];
        S_NRB2[i][j] = S_full[i2][j2];
        Uw2[i][j]    = S_NRB2[i][j];
      } /* j */
      } /* i */

    if (outputlev==1){
      printf("\n ### S_NRB2 ### \n");fflush(stdout);
      for (i=1; i<=Num_NRB3; i++){
      for (j=1; j<=Num_NRB3; j++){
        printf("%9.5f",S_NRB2[i][j]);fflush(stdout);
      } /* j */
        printf("\n");
      } /* i */
    }

  /***  Eigenvalue problem, S_NRB2 * Uw = Rw * Uw  ***/

       Eigen_lapack(Uw2, Rw2, Num_NRB3,Num_NRB3);

     if (outputlev==1){
       printf("\n ### Uw1 ### \n");fflush(stdout);
       for (i=1; i<=Num_NRB3; i++){
       for (j=1; j<=Num_NRB3; j++){
         printf("%9.5f",Uw2[i][j]);fflush(stdout);
       }
         printf("\n");
       }

       printf("\n ### Rw1 (1) ### \n");fflush(stdout);
       for (i=1; i<=Num_NRB3; i++){
         printf("%9.5f \n",Rw2[i]);fflush(stdout);
       }
     }

       for (l=1; l<=Num_NRB3; l++){
         if (Rw2[l]<1.0e-14) {
          printf("Found ill-conditioned eigenvalues (5)\n");
          printf("Stopped calculation\n");
          exit(1);
         }
       }

  /***  Rw^{-1/2}  ***/

       for (l=1; l<=Num_NRB3; l++){
         Rw2[l] = 1.0/sqrt(Rw2[l]);
       }

    if (outputlev==1){
       printf("\n ### Rw1 (2) ### \n");fflush(stdout);
       for (i=1; i<=Num_NRB3; i++){
         printf("%9.5f \n",Rw2[i]);fflush(stdout);
       }
    }

  /***  Sw^{-1/2} = Uw * Rw^{-1/2} * Uw^{t}  ***/

      for (i=1; i<=Num_NRB3; i++){
      for (j=1; j<=Num_NRB3; j++){
        temp_M1[i][j] = Rw2[i] * Uw2[j][i];
      } /* j */
      } /* i */

      for (i=1; i<=Num_NRB3; i++){
      for (j=1; j<=Num_NRB3; j++){
        tmp_ele1 = 0.0;
      for (k=1; k<=Num_NRB3; k++){
        tmp_ele1 += Uw2[i][k] * temp_M1[k][j];
      } /* k */
        Sw2[i][j] = tmp_ele1;
      } /* j */
      } /* i */

    if (outputlev==1){
      printf("\n ### Sw2^{-1/2} ### \n");fflush(stdout);
      for (i=1; i<=Num_NRB3; i++){
      for (j=1; j<=Num_NRB3; j++){
        printf("%9.5f",Sw2[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

      for (i=1; i<=Num_NRB3; i++){
        i2 = NRB_posi8[NRB_posi7[i]];
      for (j=1; j<=Num_NRB3; j++){
        j2 = NRB_posi8[NRB_posi7[j]];
        N_tran1[i2][j2] = Sw2[i][j];
      } /* j */
      } /* i */

     if (outputlev==1){

      printf("\n ### O_sym ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",N_tran1[i][j]);fflush(stdout);
      }
        printf("\n");
      }

     }
/*
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
        tmp_ele3 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full[i][k] * O_sym[k][j];
        tmp_ele2 += P_full[i][k] * O_sym[k][j];
        tmp_ele3 += T_full[i][k] * O_sym[k][j];
      }
        temp_M1[i][j] = tmp_ele1;
        temp_M2[i][j] = tmp_ele2;
        temp_M3[i][j] = tmp_ele3;
      }
      }

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += O_sym[k][i] * temp_M1[k][j];
        tmp_ele2 += O_sym[k][i] * temp_M2[k][j];
      }
        S_full[i][j] = tmp_ele1;
        P_full[i][j] = tmp_ele2;
        T_full[i][j] = temp_M3[i][j];
      }
      }
*/

   /* BLAS */
     /*  Ow_NRB * S * Ow_NRB  */
if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = N_tran1[i][j];
       temp_V2[tnum] = S_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       S_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  Ow_NRB * P * Ow_NRB  */
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = P_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  T * Ow_NRB  */

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = T_full[j][i];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[j][i] = temp_V3[tnum];
       tnum++;
     }
     }
}
else{
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M4[j][i] = N_tran1[i][j];
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M4[j][k];
       tmp_ele1 = temp_M4[j][k+1];
       tmp_ele2 = temp_M4[j][k+2];
       tmp_ele3 = temp_M4[j][k+3];

       tmp_ele00 += S_full[i][k]   * tmp_ele0;
       tmp_ele01 += S_full[i][k+1] * tmp_ele1;
       tmp_ele02 += S_full[i][k+2] * tmp_ele2;
       tmp_ele03 += S_full[i][k+3] * tmp_ele3;

       tmp_ele10 += P_full[i][k]   * tmp_ele0;
       tmp_ele11 += P_full[i][k+1] * tmp_ele1;
       tmp_ele12 += P_full[i][k+2] * tmp_ele2;
       tmp_ele13 += P_full[i][k+3] * tmp_ele3;

       tmp_ele20 += T_full[i][k]   * tmp_ele0;
       tmp_ele21 += T_full[i][k+1] * tmp_ele1;
       tmp_ele22 += T_full[i][k+2] * tmp_ele2;
       tmp_ele23 += T_full[i][k+3] * tmp_ele3;
     }
       temp_M1[j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       temp_M2[j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
       temp_M3[i][j] = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
     }
     }

     is = SizeMat - SizeMat%4;

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
     for (k=is; k<SizeMat; k++){
        tmp_ele0 = temp_M4[j][k];
        tmp_ele00 += S_full[i][k] * tmp_ele0;
        tmp_ele10 += P_full[i][k] * tmp_ele0;
        tmp_ele20 += T_full[i][k] * tmp_ele0;
     }
       temp_M1[j][i] += tmp_ele00;
       temp_M2[j][i] += tmp_ele10;
       temp_M3[i][j] += tmp_ele20;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M4[i][k];
       tmp_ele1 = temp_M4[i][k+1];
       tmp_ele2 = temp_M4[i][k+2];
       tmp_ele3 = temp_M4[i][k+3];

       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele01 += tmp_ele1 * temp_M1[j][k+1];
       tmp_ele02 += tmp_ele2 * temp_M1[j][k+2];
       tmp_ele03 += tmp_ele3 * temp_M1[j][k+3];

       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
       tmp_ele11 += tmp_ele1 * temp_M2[j][k+1];
       tmp_ele12 += tmp_ele2 * temp_M2[j][k+2];
       tmp_ele13 += tmp_ele3 * temp_M2[j][k+3];
     }
       S_full[i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       P_full[i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
       tmp_ele0 = temp_M4[i][k];
       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
     }
       S_full[i][j] += tmp_ele00;
       P_full[i][j] += tmp_ele10;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[i][j] = temp_M3[i][j];
    }
    }
}

      } /* if Num_NRB3 > 0 */

    if (outputlev==1){
      printf("\n ### S_full (7) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",S_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### P_full (7) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",P_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }

      printf("\n ### T_full (7) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",T_full[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    }

if (measure_time==1){
  dtime(&Etime1);
  time8 = Etime1 - Stime1;
}


  /************************************************************************
     3-5. Symmetry averaging of whole orbitals to construct NAOs (N_red)
  ************************************************************************/

if (myid == Host_ID && outputlev == 1)
    printf("<< 3-5 >> Symmetry averaging of whole orbitals to construct NAOs (N_red) \n");
if (measure_time==1) dtime(&Stime1);

       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         N_tran2[i][j]=0.0;
       }
       }
       for (i=0; i<SizeMat; i++){
         N_tran2[i][i]=1.0;
       }

  /***  Atomic & anglar- & magetic-momentum block matrices  ***/

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Leng_alm[FS_AN][L1];
     for (M1=1; M1<=2*L1+1; M1++){
        posi1 = Posi_alm[FS_AN][L1][M1];
       for (i=0; i<leng1; i++){
       for (j=0; j<leng1; j++){
          P_alm[FS_AN][L1][M1][i][j] = P_full[i+posi1][j+posi1];
          S_alm[FS_AN][L1][M1][i][j] = S_full[i+posi1][j+posi1];
       } /* j */
       } /* i */
     } /* M1 */
     } /* L1 */
     } /* FS_AN */

  /***  Partitioning and symmetry averaging of P & S  ***/

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Leng_alm[FS_AN][L1];
       for (i=0; i<leng1; i++){
       for (j=0; j<leng1; j++){
          tmp_ele1 = 0.0;
          tmp_ele2 = 0.0;
         for (M1=1; M1<=L1*2+1; M1++){
              tmp_ele1 += P_alm[FS_AN][L1][M1][i][j];
              tmp_ele2 += S_alm[FS_AN][L1][M1][i][j];
         } /* M1 */
          tmp_ele1 /= L1*2.0+1.0;
          tmp_ele2 /= L1*2.0+1.0;
         for (M1=1; M1<=L1*2+1; M1++){
              P_alm[FS_AN][L1][M1][i][j] = tmp_ele1;
              S_alm[FS_AN][L1][M1][i][j] = tmp_ele2;
         } /* M1 */
       } /* j */
       } /* i */
     } /* L1 */
     } /* FS_AN */

  /***  Atom & anglar-momentum loop  ***/

     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
        Gc_AN = natn[Nc_AN][FS_AN];
        wan1 = WhatSpecies[Gc_AN];
     for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        Mmax1 = Leng_alm[FS_AN][L1];

  /***  Transferring of matrices (1)  ***/

       for (i=0; i<Mmax1; i++){
       for (j=0; j<Mmax1; j++){
          P_alm2[i+1][j+1] = P_alm[FS_AN][L1][1][i][j];
          S_alm2[i+1][j+1] = S_alm[FS_AN][L1][1][i][j];
       } /* j */
       } /* i */

    if (outputlev==1){
       printf("\n### S_alm2[%d][%d] ### \n",FS_AN,L1);fflush(stdout);
       for (i=0; i<Mmax1; i++){
       for (j=0; j<Mmax1; j++){
          printf("%9.5f",S_alm2[i+1][j+1]);fflush(stdout);
       } /* j */
          printf("\n");fflush(stdout);
       } /* j */

       printf("\n");fflush(stdout);

       printf("### P_alm2[%d][%d] ### \n",FS_AN,L1);fflush(stdout);
       for (i=0; i<Mmax1; i++){
       for (j=0; j<Mmax1; j++){
          printf("%9.5f",P_alm2[i+1][j+1]);fflush(stdout);
       } /* j */
          printf("\n");fflush(stdout);
       } /* i */
     }

  /***  Generalized eigenvalue problem, P * N = S * N * W  ***/

  /*******************************************
     Diagonalizing the overlap matrix

      First:
        S -> OLP matrix
      After calling Eigen_lapack:
        S -> eigenvectors of OLP matrix
   *******************************************/

  /***  Generalized eigenvalue problem, P * N = S * N * W  ***/

   if (outputlev==1) printf("\n Diagonalize the overlap matrix \n");

    Eigen_lapack(S_alm2,W_alm2,Mmax1,Mmax1);

  if (outputlev==1){
    for (k=1; k<=Mmax1; k++){
      printf("k W %2d %15.12f\n",k,W_alm2[k]);
    }
  }

    /* check ill-conditioned eigenvalues */

    for (k=1; k<=Mmax1; k++){
      if (W_alm2[k]<1.0e-14) {
       printf("Found ill-conditioned eigenvalues (6)\n");
       printf("Stopped calculation\n");
       exit(1);
      }
    }

    for (k=1; k<=Mmax1; k++){
      temp_V3[k] = 1.0/sqrt(W_alm2[k]);
    }

   /***  Calculations of eigenvalues  ***/

    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
         sum = 0.0;
        for (k=1; k<=Mmax1; k++){
          sum = sum + P_alm2[i][k]*S_alm2[k][j]*temp_V3[j];
        }
        N_alm2[i][j] = sum;
      }
    }

    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
         sum = 0.0;
        for (k=1; k<=Mmax1; k++){
          sum = sum + temp_V3[i]*S_alm2[k][i]*N_alm2[k][j];
        }
        temp_M1[i][j] = sum;
      }
    }

  if (outputlev==1){
    printf("\n ##### TEST 1 ##### \n");
    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
        printf("%9.5f",temp_M1[i][j]);
      }
        printf("\n");
    }
  }

    Eigen_lapack(temp_M1,W_alm2,Mmax1,Mmax1);

  if (outputlev==1){
    printf("\n ##### TEST 2 ##### \n");
    for (k=1; k<=Mmax1; k++){
      printf("k W %2d %9.5f\n",k,W_alm2[k]);
    }

    printf("\n ##### TEST 3 ##### \n");
    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
        printf("%9.5f",temp_M1[i][j]);
      }
        printf("\n");
    }
    printf("\n");
  }

    /***  Transformation to the original eigen vectors  ***/

    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
        N_alm2[i][j] = 0.0;
      }
    }

    for (i=1; i<=Mmax1; i++){
      for (j=1; j<=Mmax1; j++){
         sum = 0.0;
        for (k=1; k<=Mmax1; k++){
          sum = sum + S_alm2[i][k]*temp_V3[k]*temp_M1[k][j];
        }
        N_alm2[i][j] = sum;
      }
    }

  /* printing out eigenvalues and the eigenvectors */

  if (outputlev==1){
      for (i=1; i<=Mmax1; i++){
        printf("%ith eigenvalue of HC=eSC: %15.12f\n",i,W_alm2[i]);
      }

        for (i=1; i<=Mmax1; i++){
           printf("%ith eigenvector: ",i);
           printf("{");
          for (j=1; j<=Mmax1; j++){
            printf("%7.4f,",N_alm2[i][j]);
          }
          printf("}\n");
        }

          printf("\n");
   }

  /***  Selection of NMB & NRB orbitals (2)  ***/

       for (M1=1; M1<=L1*2+1; M1++){
          posi1 = Posi_alm[FS_AN][L1][M1];
       for (i=0; i<Mmax1; i++){
          W_full[posi1+i] = W_alm2[i+1];
       for (j=0; j<Mmax1; j++){
          N_tran2[posi1+i][posi1+j] = N_alm2[i+1][j+1];
       } /* j */
       } /* i */
       } /* M1 */

     } /* L1 */
     } /* FS_AN */

    if (outputlev==1){
       printf("### N_red ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",N_tran2[i][j]);fflush(stdout);
       }
         printf("\n");
       }
    } 


  /***  Overlap & density matrices for NAOs  ***/
/*
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
        tmp_ele3 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full[i][k] * N_red[k][j];
        tmp_ele2 += P_full[i][k] * N_red[k][j];
        tmp_ele3 += T_full[i][k] * N_red[k][j];
      } 
        temp_M1[i][j] = tmp_ele1;
        temp_M2[i][j] = tmp_ele2;
        temp_M3[i][j] = tmp_ele3;
      } 
      } 

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += N_red[k][i] * temp_M1[k][j];
        tmp_ele2 += N_red[k][i] * temp_M2[k][j];
      } 
        S_full[i][j] = tmp_ele1;
        P_full[i][j] = tmp_ele2;
        T_full[i][j] = temp_M3[i][j];
      } 
      } 
*/

   /* BLAS */
     /*  N_red * S * N_red  */
if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = N_tran2[i][j];
       temp_V2[tnum] = S_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       S_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  N_red * P * N_red  */
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = P_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       P_full[i][j] = temp_V2[tnum];
       tnum++;
     }
     }

     /*  T * N_red  */

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V2[tnum] = T_full[j][i];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "T", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[j][i] = temp_V3[tnum];
       tnum++;
     }
     }
}
else{
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M4[j][i] = N_tran2[i][j];
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M4[j][k];
       tmp_ele1 = temp_M4[j][k+1];
       tmp_ele2 = temp_M4[j][k+2];
       tmp_ele3 = temp_M4[j][k+3];

       tmp_ele00 += S_full[i][k]   * tmp_ele0;
       tmp_ele01 += S_full[i][k+1] * tmp_ele1;
       tmp_ele02 += S_full[i][k+2] * tmp_ele2;
       tmp_ele03 += S_full[i][k+3] * tmp_ele3;

       tmp_ele10 += P_full[i][k]   * tmp_ele0;
       tmp_ele11 += P_full[i][k+1] * tmp_ele1;
       tmp_ele12 += P_full[i][k+2] * tmp_ele2;
       tmp_ele13 += P_full[i][k+3] * tmp_ele3;

       tmp_ele20 += T_full[i][k]   * tmp_ele0;
       tmp_ele21 += T_full[i][k+1] * tmp_ele1;
       tmp_ele22 += T_full[i][k+2] * tmp_ele2;
       tmp_ele23 += T_full[i][k+3] * tmp_ele3;
     }
       temp_M1[j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       temp_M2[j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
       temp_M3[i][j] = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
     }
     }

     is = SizeMat - SizeMat%4;

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
     for (k=is; k<SizeMat; k++){
        tmp_ele0 = temp_M4[j][k];
        tmp_ele00 += S_full[i][k] * tmp_ele0;
        tmp_ele10 += P_full[i][k] * tmp_ele0;
        tmp_ele20 += T_full[i][k] * tmp_ele0;
     }
       temp_M1[j][i] += tmp_ele00;
       temp_M2[j][i] += tmp_ele10;
       temp_M3[i][j] += tmp_ele20;
     }
     }


     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
       tmp_ele01 = 0.0; tmp_ele11 = 0.0;
       tmp_ele02 = 0.0; tmp_ele12 = 0.0;
       tmp_ele03 = 0.0; tmp_ele13 = 0.0;
     for (k=0; k<(SizeMat-3); k+=4){
       tmp_ele0 = temp_M4[i][k];
       tmp_ele1 = temp_M4[i][k+1];
       tmp_ele2 = temp_M4[i][k+2];
       tmp_ele3 = temp_M4[i][k+3];

       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele01 += tmp_ele1 * temp_M1[j][k+1];
       tmp_ele02 += tmp_ele2 * temp_M1[j][k+2];
       tmp_ele03 += tmp_ele3 * temp_M1[j][k+3];

       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
       tmp_ele11 += tmp_ele1 * temp_M2[j][k+1];
       tmp_ele12 += tmp_ele2 * temp_M2[j][k+2];
       tmp_ele13 += tmp_ele3 * temp_M2[j][k+3];
     }
       S_full[i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
       P_full[i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       tmp_ele00 = 0.0; tmp_ele10 = 0.0;
     for (k=is; k<SizeMat; k++){
       tmp_ele0 = temp_M4[i][k];
       tmp_ele00 += tmp_ele0 * temp_M1[j][k];
       tmp_ele10 += tmp_ele0 * temp_M2[j][k];
     }
       S_full[i][j] += tmp_ele00;
       P_full[i][j] += tmp_ele10;
     }
     }

     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       T_full[i][j] = temp_M3[i][j];
    }
    }
}

/* ### TEST ### */
    if (outputlev==1){
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full0[i][k] * T_full[k][j];
/*
        tmp_ele1 += NBO_OLPn_tmp[FS_atm][i][k] * T_full[k][j];
*/
      } /* k */
        temp_M1[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += T_full[k][i] * temp_M1[k][j];
      } /* k */
        temp_M3[i][j] = tmp_ele1;
      } /* j */
      } /* i */

      printf("\n ### S_full (TEST 4) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",temp_M3[i][j]);fflush(stdout);
      }
        printf("\n");
      }
    } 
/* ### TEST ### */

if (measure_time==1){
  dtime(&Etime1);
  time9 = Etime1 - Stime1;
}


  /*****************************
     3-6. Energy of each NAO
  *****************************/

if (myid == Host_ID && outputlev == 1) printf("<< 3-6 >> Energy of each NAO \n");
if (measure_time==1) dtime(&Stime1);
/*
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += H_full[i][k] * T_full[k][j];
      } 
        temp_M1[i][j] = tmp_ele1;
      } 
      } 

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += T_full[k][i] * temp_M1[k][j];
      } 
        temp_M3[i][j] = tmp_ele1;
      }
      } 
*/

   /* BLAS */

if(1>0){
     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_V1[tnum] = T_full[j][i];
       temp_V2[tnum] = H_full[i][j];
       tnum++;
     }
     }

     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("N", "N", &M, &N, &K, &alpha,
                           temp_V2, &lda, temp_V1, &ldb, &beta, temp_V3, &ldc);
     alpha = 1.0; beta = 0.0;
     M = SizeMat; N = SizeMat; K = SizeMat;
     lda = K; ldb = K; ldc = K;

     F77_NAME(dgemm,DGEMM)("T", "N", &M, &N, &K, &alpha,
                           temp_V1, &lda, temp_V3, &ldb, &beta, temp_V2, &ldc);

     tnum = 0;
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       temp_M3[i][j]  = temp_V2[tnum];
       tnum++;
     }
     }
}
else{
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        temp_M2[j][i] = T_full[i][j];
      }
      }

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele00 = 0.0;
        tmp_ele01 = 0.0;
        tmp_ele02 = 0.0;
        tmp_ele03 = 0.0;
      for (k=0; k<(SizeMat-3); k+=4){
        tmp_ele00 += H_full[i][k]   * temp_M2[j][k];
        tmp_ele01 += H_full[i][k+1] * temp_M2[j][k+1];
        tmp_ele02 += H_full[i][k+2] * temp_M2[j][k+2];
        tmp_ele03 += H_full[i][k+3] * temp_M2[j][k+3];
      } 
        temp_M1[j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
      } 
      } 

      is = SizeMat - SizeMat%4;

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele00 = 0.0;
      for (k=is; k<SizeMat; k++){
        tmp_ele00 += H_full[i][k] * temp_M2[j][k];
      }
        temp_M1[j][i] += tmp_ele00;
      }
      }

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele00 = 0.0;
        tmp_ele01 = 0.0;
        tmp_ele02 = 0.0;
        tmp_ele03 = 0.0;
      for (k=0; k<(SizeMat-3); k+=4){
        tmp_ele00 += temp_M2[i][k]   * temp_M1[j][k];
        tmp_ele01 += temp_M2[i][k+1] * temp_M1[j][k+1];
        tmp_ele02 += temp_M2[i][k+2] * temp_M1[j][k+2];
        tmp_ele03 += temp_M2[i][k+3] * temp_M1[j][k+3];
      } 
        temp_M3[i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
      } 
      } 

      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        tmp_ele00 = 0.0;
      for (k=is; k<SizeMat; k++){
        tmp_ele00 += temp_M2[i][k] * temp_M1[j][k];
      }
        temp_M3[i][j] += tmp_ele00;
      }
      }
}

/*
if(Nc_AN == 228){
      printf("\n ### T^t * H_full * T (TEST 4) ### \n");fflush(stdout);
      for (i=0; i<10; i++){
      for (j=0; j<10; j++){
        printf("%9.5f",temp_M3[i][j]);fflush(stdout);
      }
      printf("\n");
      }
}
*/
    if (outputlev==1){
/*
      printf("\n ### T^t * H_full * T (TEST 4) ### \n");fflush(stdout);
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",temp_M3[i][j]);fflush(stdout);
      }
      printf("\n");
      }
*/
       printf("\n ### S_full (10) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",S_full[i][j]);fflush(stdout);
       }
       printf("\n");
       }

       printf("\n ### P_full (10) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",P_full[i][j]);fflush(stdout);
       }
       printf("\n");
       }

       printf("\n ### T_full (10) ### \n");fflush(stdout);
       for (i=0; i<SizeMat; i++){
       for (j=0; j<SizeMat; j++){
         printf("%9.5f",T_full[i][j]);fflush(stdout);
       }
       printf("\n");
       }
     }


if (measure_time==1){
  dtime(&Etime1);
  time10 = Etime1 - Stime1;
}
/*
  MPI_Barrier(mpi_comm_level1);
*/
if(1<0){
if(Nc_AN == NBO_FCenter[2]){
/*
      for (j=0; j<Leng_a[0]; j++){
        printf("%9.5f",P_full[j][j]);fflush(stdout);
      }
      printf("\n");
      for (j=0; j<Leng_a[0]; j++){
        printf("%9.5f",temp_M3[j][j]);fflush(stdout);
      }
      printf("\n");
*/

      printf("--- P_full (Gc_AN=%d) ---------------\n",Nc_AN);
      for (i=0; i<Leng_a[0]; i++){
      for (j=0; j<Leng_a[0]; j++){
        printf("%9.5f",P_full[i][j]);fflush(stdout);
      }
      printf("\n");
      }
      printf("-------------------------------------\n");
      printf("\n");

/*
      printf("--- S_full0 -------------------------\n");
      for (i=0; i<SizeMat; i++){
      for (j=0; j<SizeMat; j++){
        printf("%9.5f",S_full0[i][j]);fflush(stdout);
      }
      printf("\n");
      }
      printf("-------------------------------------\n");
      printf("\n");
*/
      printf("--- D_full (Gc_AN=%d) ---------------\n",Nc_AN);
      for (i=0; i<Leng_a[0]; i++){
      for (j=0; j<Leng_a[0]; j++){
        printf("%9.5f",D_full[i][j]);fflush(stdout);
      }
      printf("\n");
      }
      printf("-------------------------------------\n");
      printf("\n");

      printf("--- P_full0 (Gc_AN=%d) --------------\n",Nc_AN);
      for (i=0; i<Leng_a[0]; i++){
      for (j=0; j<Leng_a[0]; j++){
        printf("%9.5f",P_full0[i][j]);fflush(stdout);
      }
      printf("\n");
      }
      printf("-------------------------------------\n");
      printf("\n");


      printf("--- T_full (Gc_AN=%d) ---------------\n",Nc_AN);
      for (i=0; i<Leng_a[0];   i++){
      for (j=0; j<Leng_a[0]; j++){
        printf("%9.5f",T_full[i][j]);fflush(stdout);
      }
      printf("\n");
      }
      printf("-------------------------------------\n");
      printf("\n");
/*

      printf("  &&& SizeMat = %d \n",SizeMat);

      for (i=0; i<SizeMat; i++){
      for (j=0; j<Leng_a[0]; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += P_full0[i][k] * T_full[k][j];
      }
        temp_M1[i][j] = tmp_ele1;
      }
      }

      for (i=0; i<Leng_a[0]; i++){
      for (j=0; j<Leng_a[0]; j++){
        tmp_ele1 = 0.0;
      for (k=0; k<SizeMat; k++){
        tmp_ele1 += T_full[k][i] * temp_M1[k][j];
      }
        temp_M3[i][j] = tmp_ele1;
      }
      }

      printf("--- P_full(2) -----------------------\n");
      for (i=0; i<Leng_a[0]; i++){
      for (j=0; j<Leng_a[0]; j++){
        printf("%9.5f",temp_M3[i][j]);fflush(stdout);
      }
      printf("\n");
      }
      printf("-------------------------------------\n");
      printf("\n");
*/

}
}

     k=0;
     tmp_ele2 = 0.0;
     for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
       Gc_AN = natn[Nc_AN][FS_AN];
       wan1 = WhatSpecies[Gc_AN];
       leng1 = Leng_a[FS_AN];
       tmp_ele1 = 0.0;
     for (i=0; i<leng1; i++){
       tmp_ele1 += P_full[k+i][k+i];
     } /* i */
       k += leng1;
       tmp_ele2 += tmp_ele1;
       NP_part_full[FS_AN] = tmp_ele1;
     } /* FS_AN */

     *NP_total   = tmp_ele2;
     *NP_partial = NP_part_full[0];

    /* copy of NAO vector for central atom */

     for (i=0; i<SizeMat; i++){
     for (j=0; j<Leng_a[0]; j++){
       NAO_vec_tmp[FS_atm][i][j] = T_full[i][j];
     }
     }

    /* copy of NAP of central atom */

     for (i=0; i<Leng_a[0]; i++){
       NAO_partial_pop[FS_atm][i] = P_full[i][i];
     }

    /* copy of energy levels of NAO of central atom */

     for (i=0; i<Leng_a[0]; i++){
       NAO_ene_level[FS_atm][i] = temp_M3[i][i];
     }

/*
if(Nc_AN == NBO_FCenter[0]){
for (i=0; i<Leng_a[0]; i++){
printf("TEST : %9.5f\n",NAO_partial_pop[FS_atm][i]);
}
}
*/

    /* copy of P_full0 and H_full0 */
/*
     for (i=0; i<SizeMat; i++){
     for (j=0; j<SizeMat; j++){
       NBO_OLPn_tmp[i][j] = S_full0[i][j];
     }
     }
*/
/*
     if(myid == Host_ID){
       printf("\n#### Natural population of each atom in cluster ####\n");
       for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
         printf(" %4d : %4d  %9.4f \n",FS_AN,natn[Nc_AN][FS_AN],NP_part_full[FS_AN]);  
       }
     }
*/

  /***  Freeing arrays (1)  ***/

    free(Leng_a);

   for (i=0; i<=FS_atomnum; i++){
    free(Leng_al[i]);
   }
    free(Leng_al);

   for (i=0; i<=FS_atomnum; i++){
    free(Leng_alm[i]);
   }
    free(Leng_alm);

    free(Posi_a);

   for (i=0; i<=FS_atomnum; i++){
    free(Posi_al[i]);
   }
    free(Posi_al);

   for (i=0; i<=FS_atomnum; i++){
   for (j=0; j<=Lmax; j++){
    free(Posi_alm[i][j]);
   }
    free(Posi_alm[i]);
   }
    free(Posi_alm);

   for (i=0; i<=FS_atomnum; i++){
    free(Num_NMB1[i]);
   }
    free(Num_NMB1);

   for (i=0; i<=FS_atomnum; i++){
    free(Num_NRB1[i]);
   }
    free(Num_NRB1);

   for (i=0; i<=SizeMat; i++){
    free(S_full0[i]);
   }
    free(S_full0);

/*
   for (i=0; i<=SizeMat; i++){
    free(D_full0[i]);
   }
    free(D_full0);
*/
   for (i=0; i<=SizeMat; i++){
    free(P_full[i]);
   }
    free(P_full);

   for (i=0; i<=SizeMat; i++){
    free(P_full0[i]);
   }
    free(P_full0);
/*
   for (i=0; i<=SizeMat; i++){
    free(H_full0[i]);
   }
    free(H_full0);
*/
   for (i=0; i<=SizeMat; i++){
    free(T_full[i]);
   }
    free(T_full);

    free(W_full);

   for (i=0; i<=SizeMat; i++){
    free(N_tran1[i]);
   }
    free(N_tran1);
/*
   for (i=0; i<=SizeMat; i++){
    free(N_ryd[i]);
   }
    free(N_ryd);
*/
/*
   for (i=0; i<=SizeMat; i++){
    free(N_red[i]);
   }
    free(N_red);
*/
/*
   for (i=0; i<=SizeMat; i++){
    free(O_schm[i]);
   }
    free(O_schm);
*/
/*
   for (i=0; i<=SizeMat; i++){
    free(O_schm2[i]);
   }
    free(O_schm2);
*/
   for (i=0; i<=SizeMat; i++){
    free(N_tran2[i]);
   }
    free(N_tran2);
/*
   for (i=0; i<=SizeMat; i++){
    free(Ow_NRB[i]);
   }
    free(Ow_NRB);
*/
   for (i=0; i<=SizeMat; i++){
    free(temp_M1[i]);
   }
    free(temp_M1);

   for (i=0; i<=SizeMat; i++){
    free(temp_M2[i]);
   }
    free(temp_M2);

   for (i=0; i<=SizeMat; i++){
    free(temp_M3[i]);
   }
    free(temp_M3);

   for (i=0; i<=SizeMat; i++){
    free(temp_M4[i]);
   }
    free(temp_M4);

    free(temp_V1);
    free(temp_V2);
    free(temp_V3);

  /***  Freeing arrays (2)  ***/

   for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
   for (L1=0; L1<=Lmax; L1++){
   for (M1=0; M1<=2*Lmax+1; M1++){
   for (i=0; i<=MaxLeng_alm; i++){
    free(P_alm[FS_AN][L1][M1][i]);
   }
    free(P_alm[FS_AN][L1][M1]);
   }
    free(P_alm[FS_AN][L1]);
   }
    free(P_alm[FS_AN]);
   }
    free(P_alm);

   for (FS_AN=0; FS_AN<=FS_atomnum; FS_AN++){
   for (L1=0; L1<=Lmax; L1++){
   for (M1=0; M1<=2*Lmax+1; M1++){
   for (i=0; i<=MaxLeng_alm; i++){
    free(S_alm[FS_AN][L1][M1][i]);
   }
    free(S_alm[FS_AN][L1][M1]);
   }
    free(S_alm[FS_AN][L1]);
   }
    free(S_alm[FS_AN]);
   }
    free(S_alm);

  /***  Freeing arrays (3)  ***/

   for (i=0; i<=MaxLeng_alm; i++){
    free(P_alm2[i]);
   }
    free(P_alm2);

   for (i=0; i<=MaxLeng_alm; i++){
    free(S_alm2[i]);
   }
    free(S_alm2);

   for (i=0; i<=MaxLeng_alm; i++){
    free(N_alm2[i]);
   }
    free(N_alm2);

    free(W_alm2);

   for (i=0; i<=FS_atomnum; i++){
   for (j=0; j<=Lmax; j++){
    free(M_or_R1[i][j]);
   }
    free(M_or_R1[i]);
   }
    free(M_or_R1);

    free(M_or_R2);

  /***  Freeing arrays (4)  ***/

    for (i=0; i<=Num_NMB; i++){
     free(S_NMB[i]);
    }
     free(S_NMB);

     free(W_NMB);
/*
    for (i=0; i<=Num_NMB; i++){
     free(Ow_NMB2[i]);
    }
     free(Ow_NMB2);
*/
    for (i=0; i<=Num_NMB; i++){
     free(Sw1[i]);
    }
     free(Sw1);

    for (i=0; i<=Num_NMB; i++){
     free(Uw1[i]);
    }
     free(Uw1);

     free(Rw1);

     free(NMB_posi1);

     free(NMB_posi2);

     free(NRB_posi1);

     free(NRB_posi2);

     free(NRB_posi3);

     free(NRB_posi4);

     free(NRB_posi5);

     free(NRB_posi6);

     free(NRB_posi7);

     free(NRB_posi8);

     free(NRB_posi9);

  /***  Freeing arrays (5)  ***/

     free(S_or_L);

    for (i=0; i<=Num_NRB; i++){
     free(S_NRB[i]);
    }
     free(S_NRB);

    for (i=0; i<=Num_NRB; i++){
     free(S_NRB2[i]);
    }
     free(S_NRB2);

     free(W_NRB);

     free(W_NRB2);
/*
    for (i=0; i<=Num_NRB; i++){
     free(Ow_NRB2[i]);
    }
     free(Ow_NRB2);
*/
/*
    for (i=0; i<=SizeMat; i++){
     free(O_sym[i]);
    }
     free(O_sym);
*/
    for (i=0; i<=Num_NRB; i++){
     free(Sw2[i]);
    }
     free(Sw2);

    for (i=0; i<=Num_NRB; i++){
     free(Uw2[i]);
    }
     free(Uw2);

     free(Rw2);

     free(NP_part_full);


if (measure_time==1) dtime(&EtimeF);


if (measure_time==1 && myid == Host_ID){

 printf(" \n######## Elapsed time in NAO calc. (sec.) ########\n\n");
 printf("  1. Setting density & overlap matrices P & S :             %8.4f \n",time1);
 printf("  2. Intraatomic orthogonalization \n");
 printf("   2-2 Symmetry averaging of P & S :                        %8.4f \n",time2);
 printf("   2-3 Formation of pre-NAOs :                              %8.4f \n",time3);
 printf("   2-4 Construction of S & P for pre-NAOs :                 %8.4f \n",time4);
 printf("  3. Initial division between valence and Rydberg AO spaces \n");
 printf("   3-1 OWSO of NMB orbitals (Ow) :                          %8.4f \n",time5);
 printf("   3-2 Schmidt interatomic orth. of NRB to NMB (O_schm) :   %8.4f \n",time6);
 printf("   3-3 Symmetry averaging of NRB orbitals (N_ryd) :         %8.4f \n",time7);
 printf("   3-4 OWSO of NRB orbitals (Ow) :                          %8.4f \n",time8);
 printf("   3-5 Symmetry averaging of whole orbitals (N_red) :       %8.4f \n",time9);
 printf("   3-6 Energy of each NAO :                                 %8.4f \n",time10);
 printf(" ------------------------------------------------------------------------\n");
 printf("                                                Total :     %8.4f \n\n",EtimeF-StimeF);

}


if(myid == Host_ID && outputlev == 1){
  printf("\n##### End of Calc_NAO #####\n\n");
}

} /** end of "Calc_NAO" **/



static void Calc_NBO(int spin0, int Num_FSCenter, int Num_FCenter, int LVsize,
                     int *FSCenter, int *LMindx2LMposi,
                     double **P_full, double **H_full, double **T_full)
{

  /* loop */
   int Gc_AN,Gh_AN,Gb_AN,FS_AN,Mc_AN,L1,M1,i,j,i1,i2,j1,j2,k,l;

  /* temporary */
   int tnum,pnum,snum,qnum,num,is,Mulmax,Lmax;
   int wan1,wan2,wan3,wan4,tmp_num1,tmp_num2,tmp_num3,mtps1,mtps2;
   int posi1,posi2,leng1,leng2,leng3,Mmax1,Mmax2;
   int outputlev = 0;
   double tmp_threshold,sum;
   double tmp_ele0,tmp_ele1,tmp_ele2,tmp_ele3,tmp_ele4;
   double tmp_ele00,tmp_ele01,tmp_ele02,tmp_ele03;
   double tmp_ele10,tmp_ele11,tmp_ele12,tmp_ele13;
   double tmp_ele20,tmp_ele21,tmp_ele22,tmp_ele23;
   double **temp_M1,**temp_M2,**temp_M10;
   double *temp_M7,*temp_M8;

   int *Posi2Gatm,*Num_LP,*Num_NHO,Num_LonePair,Num_NBO,MaxLeng_a;
   int **Table_NBO, **Table_LP, **NHO_indexL2G;
   int Gb1_AN,Gb2_AN,Ga1_AN,Ga2_AN;
   int posi[5],t_posi[5],leng[5],t_leng,m,n;
   double ***pLP_vec,***pNHO_vec,**Bond_Pair_vec;
   double ****pLP_mat,**pNHO_mat,**X_mat,***NHO_mat,****P_pair;
   double **NBO_bond_vec,**NBO_anti_vec,**LonePair_vec,Fij[2][2];
   double *Pop_bond_NBO,*Pop_anti_NBO,*Pop_LonePair;


   Posi2Gatm = (int*)malloc(sizeof(int)*LVsize);

   Num_LP = (int*)malloc(sizeof(int)*(Num_FSCenter+1));
    for (i=0; i<=Num_FSCenter; i++){
     Num_LP[i] = 0;
    }

   Num_NHO = (int*)malloc(sizeof(int)*(Num_FSCenter+1));
    for (i=0; i<=Num_FSCenter; i++){
     Num_NHO[i] = 0;
    }


 /*******************************************************************

   Table_NBO[i] : Table for i th bonding and anti-bonding NBO's

   Table_NBO[i][0] : atom-1 (global number)
   Table_NBO[i][1] : atom-2 (global number)
   Table_NBO[i][2] : index of NHO on atom-1
   Table_NBO[i][3] : index of NHO on atom-2

 *********************************************************************/

   Table_NBO = (int**)malloc(sizeof(int*)*(Num_FSCenter*10+1));
    for (i=0; i<=Num_FSCenter*10; i++){
     Table_NBO[i] = (int*)malloc(sizeof(int)*(6));
    for (j=0; j<=5; j++){
     Table_NBO[i][j] = 0;
    }
    }

 /*******************************************************************

   Table_LP[i] : Table for i th LP's

   Table_LP[i][0] : atom (global number)

 *********************************************************************/

   Table_LP = (int**)malloc(sizeof(int*)*(Num_FSCenter*10+1));
    for (i=0; i<=Num_FSCenter*10; i++){
     Table_LP[i] = (int*)malloc(sizeof(int)*(3));
    for (j=0; j<=2; j++){
     Table_LP[i][j] = 0;
    }
    }

    NHO_indexL2G = (int**)malloc(sizeof(int*)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
     NHO_indexL2G[i] = (int*)malloc(sizeof(int)*(8));
    for (j=0; j<=7; j++){
     NHO_indexL2G[i][j] = 0;
    }
    }

   MaxLeng_a = 0;
   for (i1=0; i1<Num_FSCenter; i1++){
     Gc_AN = FSCenter[i1];
     wan1 = WhatSpecies[Gc_AN];
     leng1 = Spe_Total_CNO[wan1];

     if (MaxLeng_a < leng1) MaxLeng_a = leng1;

   }

   Pop_bond_NBO = (double*)malloc(sizeof(double)*(Num_FSCenter*10+1));
    for (i=0; i<=Num_FSCenter*10; i++){
     Pop_bond_NBO[i] = 0.0;
    }

   Pop_anti_NBO = (double*)malloc(sizeof(double)*(Num_FSCenter*10+1));
    for (i=0; i<=Num_FSCenter*10; i++){
     Pop_anti_NBO[i] = 0.0;
    }

   Pop_LonePair = (double*)malloc(sizeof(double)*(Num_FSCenter*10+1));
    for (i=0; i<=Num_FSCenter*10; i++){
     Pop_LonePair[i] = 0.0;
    }

   pLP_vec = (double***)malloc(sizeof(double**)*(Num_FSCenter+1));
    for (i=0; i<=Num_FSCenter; i++){
     pLP_vec[i] = (double**)malloc(sizeof(double*)*(MaxLeng_a+1));
    for (j=0; j<=MaxLeng_a; j++){
     pLP_vec[i][j] = (double*)malloc(sizeof(double)*(MaxLeng_a+1));
    for (k=0; k<=MaxLeng_a; k++){
     pLP_vec[i][j][k] = 0.0;
    }
    }
    }

   pNHO_vec = (double***)malloc(sizeof(double**)*(Num_FSCenter+1));
    for (i=0; i<=Num_FSCenter; i++){
     pNHO_vec[i] = (double**)malloc(sizeof(double*)*(MaxLeng_a*2+1));
    for (j=0; j<=MaxLeng_a*2; j++){
     pNHO_vec[i][j] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
    for (k=0; k<=MaxLeng_a*2; k++){
     pNHO_vec[i][j][k] = 0.0;
    }
    }
    }

   pLP_mat = (double****)malloc(sizeof(double***)*(Num_FSCenter+1));
    for (i=0; i<=Num_FSCenter; i++){
     pLP_mat[i] = (double***)malloc(sizeof(double**)*(MaxLeng_a+1));
    for (j=0; j<=MaxLeng_a; j++){
     pLP_mat[i][j] = (double**)malloc(sizeof(double*)*(MaxLeng_a+1));
    for (k=0; k<=MaxLeng_a; k++){
     pLP_mat[i][j][k] = (double*)malloc(sizeof(double)*(MaxLeng_a+1));
    for (l=0; l<=MaxLeng_a; l++){
     pLP_mat[i][j][k][l] = 0.0;
    }
    }
    }
    }

   pNHO_mat = (double**)malloc(sizeof(double*)*(MaxLeng_a*2+1));
    for (i=0; i<=MaxLeng_a*2; i++){
     pNHO_mat[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
    for (j=0; j<=MaxLeng_a*2; j++){
     pNHO_mat[i][j] = 0.0;
    }
    }

   X_mat = (double**)malloc(sizeof(double*)*(MaxLeng_a*2+1));
    for (i=0; i<=MaxLeng_a*2; i++){
     X_mat[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
    for (j=0; j<=MaxLeng_a*2; j++){
     X_mat[i][j] = 0.0;
    }
    }

   NHO_mat = (double***)malloc(sizeof(double**)*(Num_FSCenter+1));
    for (i=0; i<=Num_FSCenter; i++){
     NHO_mat[i] = (double**)malloc(sizeof(double*)*(MaxLeng_a*2+1));
    for (j=0; j<=MaxLeng_a*2; j++){
     NHO_mat[i][j] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
    for (k=0; k<=MaxLeng_a*2; k++){
     NHO_mat[i][j][k] = 0.0;
    }
    }
    }

   Bond_Pair_vec = (double**)malloc(sizeof(double*)*(MaxLeng_a*2+1));
    for (i=0; i<=MaxLeng_a*2; i++){
     Bond_Pair_vec[i] = (double*)malloc(sizeof(double)*(3));
    for (j=0; j<=2; j++){
     Bond_Pair_vec[i][j] = 0.0;
    }
    }

   NBO_bond_vec = (double**)malloc(sizeof(double*)*(Num_FSCenter*10+1));
    for (i=0; i<=Num_FSCenter*10; i++){
     NBO_bond_vec[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
    for (j=0; j<=MaxLeng_a*2; j++){
     NBO_bond_vec[i][j] = 0.0;
    }
    }

   NBO_anti_vec = (double**)malloc(sizeof(double*)*(Num_FSCenter*10+1));
    for (i=0; i<=Num_FSCenter*10; i++){
     NBO_anti_vec[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
    for (j=0; j<=MaxLeng_a*2; j++){
     NBO_anti_vec[i][j] = 0.0;
    }
    }

   LonePair_vec = (double**)malloc(sizeof(double*)*(Num_FSCenter*10+1));
    for (i=0; i<=Num_FSCenter*10; i++){
     LonePair_vec[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
    for (j=0; j<=MaxLeng_a*2; j++){
     LonePair_vec[i][j] = 0.0;
    }
    }

   P_pair = (double****)malloc(sizeof(double***)*(Num_FSCenter+1));
    for (i=0; i<=Num_FSCenter; i++){
     P_pair[i] = (double***)malloc(sizeof(double**)*(Num_FSCenter+1));
    for (j=0; j<=Num_FSCenter; j++){
     P_pair[i][j] = (double**)malloc(sizeof(double*)*(MaxLeng_a*2+1));
    for (k=0; k<=MaxLeng_a*2; k++){
     P_pair[i][j][k] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
    for (l=0; l<=MaxLeng_a*2; l++){
     P_pair[i][j][k][l] = 0.0;
    }
    }
    }
    }

    temp_M1 = (double**)malloc(sizeof(double*)*(MaxLeng_a*5+1));
     for (i=0; i<=MaxLeng_a*5; i++){
      temp_M1[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*5+1));
     for (j=0; j<=MaxLeng_a*5; j++){
      temp_M1[i][j] = 0.0;
     }
     }

    temp_M2 = (double**)malloc(sizeof(double*)*(MaxLeng_a*5+1));
     for (i=0; i<=MaxLeng_a*5; i++){
      temp_M2[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*5+1));
     for (j=0; j<=MaxLeng_a*5; j++){
      temp_M2[i][j] = 0.0;
     }
     }

    temp_M10 = (double**)malloc(sizeof(double*)*(MaxLeng_a*5+1));
     for (i=0; i<=MaxLeng_a*5; i++){
      temp_M10[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*5+1));
     for (j=0; j<=MaxLeng_a*5; j++){
      temp_M10[i][j] = 0.0;
     }
     }

    temp_M7 = (double*)malloc(sizeof(double)*(MaxLeng_a*5+1));

    temp_M8 = (double*)malloc(sizeof(double)*(MaxLeng_a*5+1));


   printf("******************************************\n");
   printf("           NATURAL HYBRID ORBITAL         \n");
   printf("******************************************\n");


   for (i1=0; i1<Num_FSCenter; i1++){
     Gc_AN = FSCenter[i1];
     wan1 = WhatSpecies[Gc_AN];
     leng1 = Spe_Total_CNO[wan1];
     posi1 = LMindx2LMposi[i1];
     tnum = 0;

  /** (1) Diagonalization of atom-block density matrices **/

     for (i=0; i<leng1; i++){
     for (j=0; j<leng1; j++){
       temp_M1[i+1][j+1] = P_full[i+posi1][j+posi1];
     } /* j */
     } /* i */

/*
     printf("### Density Matrix of Atom %d ###\n",Gc_AN);fflush(stdout);
     for (i=0; i<leng1; i++){
     for (j=0; j<leng1; j++){
       printf("%9.5f",temp_M1[i+1][j+1]);fflush(stdout);
     } 
     printf("\n");fflush(stdout);
     } 
     printf("\n");fflush(stdout);
*/

     Eigen_lapack(temp_M1,temp_M7,leng1,leng1);

     if (outputlev==0){
       for (k=0; k<leng1; k++){
         printf("## atom-DM ev.(Gc_AN=%d): %9.5f \n",Gc_AN,temp_M7[k+1]);
       }
       printf("\n");fflush(stdout);
     }

  /** (2) Finding lone pairs **/

     if (spin0 == 0) tmp_threshold = 1.780/(1.0+(double)SpinP_switch);
     if (spin0 == 1) tmp_threshold = 1.892/(1.0+(double)SpinP_switch);

     for (k=0; k<leng1; k++){
       if (temp_M7[k+1] >= tmp_threshold){
         tnum++;

         for (i=0; i<leng1; i++){
           pLP_vec[i1][tnum][i] = temp_M1[i+1][k+1];
         }

         for (i=0; i<leng1; i++){
         for (j=0; j<leng1; j++){
           pLP_mat[i1][tnum][i][j] = temp_M7[k+1]
                                      * pLP_vec[i1][tnum][i]
                                      * pLP_vec[i1][tnum][j];
         }
         }

       }
     }
     Num_LP[i1] = tnum;

/*
   if (Num_LP[i1] != 0){
     printf("### Lone-pair vector (atom: %d (%d)) ###\n",Gc_AN,i1);fflush(stdout);
     for (k=1; k<=Num_LP[i1]; k++){
     for (i=0; i<leng1; i++){
       printf("%9.5f\n",pLP_vec[i1][k][i]);fflush(stdout);
     }
     printf("\n");fflush(stdout);
     }
     printf("\n");fflush(stdout);

     printf("### Density Matrix for Lone-Pair (atom: %d (%d)) ###\n",Gc_AN,i1);fflush(stdout);
     for (k=1; k<=Num_LP[i1]; k++){
     for (i=0; i<leng1; i++){
     for (j=0; j<leng1; j++){
       printf("%9.5f",pLP_mat[i1][k][i][j]);fflush(stdout);
     } 
     printf("\n");fflush(stdout);
     } 
     printf("\n");fflush(stdout);
     } 
   }
   else {
     printf("### Lone-pair not found on atom %d  (%d) ###\n\n",Gc_AN,i1);fflush(stdout);
   }
*/

   } /* i1 */

   Num_LonePair = 0;
   for (i1=0; i1<Num_FCenter; i1++){
     Num_LonePair += Num_LP[i1];
   }

  /** (3) Construction of atom-pair density matrix without LP's **/

   Num_NBO = 0;

   for (i1=0; i1<Num_FCenter; i1++){
     Gc_AN = FSCenter[i1];
     wan1 = WhatSpecies[Gc_AN];
     leng1 = Spe_Total_CNO[wan1];
     posi1 = LMindx2LMposi[i1];
     tnum = 0;

     for (i=0; i<leng1; i++){
     for (j=0; j<leng1; j++){
       temp_M1[i+1][j+1] = P_full[i+posi1][j+posi1];
       if (Num_LP[i1] != 0){
         for (k=1; k<=Num_LP[i1]; k++){
           temp_M1[i+1][j+1] -= pLP_mat[i1][k][i][j];
         }
       }
     }
     }

   for (i2=i1+1; i2<Num_FSCenter; i2++){
     Gh_AN = FSCenter[i2];
     wan2 = WhatSpecies[Gh_AN];
     leng2 = Spe_Total_CNO[wan2];
     posi2 = LMindx2LMposi[i2];

     for (i=0; i<leng2; i++){
     for (j=0; j<leng2; j++){
       temp_M1[leng1+i+1][leng1+j+1] = P_full[i+posi2][j+posi2];
       if (Num_LP[i2] != 0){
         for (k=1; k<=Num_LP[i2]; k++){
           temp_M1[leng1+i+1][leng1+j+1] -= pLP_mat[i2][k][i][j];
         }
       }
     }
     }

     for (i=0; i<leng2; i++){
     for (j=0; j<leng1; j++){
       temp_M1[leng1+i+1][j+1] = P_full[i+posi2][j+posi1];
       temp_M1[j+1][leng1+i+1] = P_full[j+posi1][i+posi2];
     }
     }

     for (i=0; i<leng1+leng2; i++){
     for (j=0; j<leng1+leng2; j++){
       temp_M2[i+1][j+1]    = temp_M1[i+1][j+1];
       P_pair[i1][i2][i][j] = temp_M1[i+1][j+1];
     }
     }

/*
     printf("### Density Matrix of (%d (%d),%d (%d)) Atom Pair w/o LP's ###\n",
            Gc_AN,i1,Gh_AN,i2);fflush(stdout);

     for (i=0; i<leng1+leng2; i++){
     for (j=0; j<leng1+leng2; j++){
       printf("%9.5f",temp_M1[i+1][j+1]);fflush(stdout);
     } 
     printf("\n");fflush(stdout);
     } 
     printf("\n");fflush(stdout);
*/

  /** (5) Orthogonalization of atom-pair matrices **/

     Eigen_lapack(temp_M2,temp_M7,leng1+leng2,leng1+leng2);

     if (outputlev==0){
       printf("### pNHO without LP's (%d (%d),%d (%d)) ###\n",Gc_AN,i1,Gh_AN,i2);fflush(stdout);
       for (k=1; k<=leng1+leng2; k++){
         printf("k W %2d %15.12f\n",k,temp_M7[k]);
       }
       printf("\n");fflush(stdout);
     }

   if (1<0){
     printf("### pNHO without LP's (%d (%d),%d (%d)) ###\n",Gc_AN,i1,Gh_AN,i2);fflush(stdout);

     for (i=0; i<leng1+leng2; i++){
     for (j=0; j<leng1+leng2; j++){
       printf("%9.5f",temp_M2[i+1][j+1]);fflush(stdout);
     } 
     printf("\n");fflush(stdout);
     } 
     printf("\n");fflush(stdout);
   }

  /** (6) Finding doublly occupied states **/

     if (spin0 == 0) tmp_threshold = 1.840/(1.0+(double)SpinP_switch);
     if (spin0 == 1) tmp_threshold = 1.388/(1.0+(double)SpinP_switch);

     for (k=0; k<leng1+leng2; k++){
       if (temp_M7[k+1] >= tmp_threshold){

         Num_NHO[i1]++;  tnum = Num_NHO[i1];

         for (i=0; i<leng1; i++){
           pNHO_vec[i1][tnum][i]= temp_M2[i+1][k+1];
         }

         if (i2 < Num_FCenter){
           Num_NBO++;
           Num_NHO[i2]++;  pnum = Num_NHO[i2];
           Table_NBO[Num_NBO][0] = i1;
           Table_NBO[Num_NBO][1] = i2;
           Table_NBO[Num_NBO][2] = tnum;
           Table_NBO[Num_NBO][3] = pnum;
           for (i=0; i<leng2; i++){
             pNHO_vec[i2][pnum][i]= temp_M2[i+leng1+1][k+1];
           }
         }

       }
     }

     } /* i2 */
   } /* i1 */

   Total_Num_NBO = Num_NBO;

     printf("### Table of NBO ###\n");fflush(stdout);
     for (i=1; i<=Num_NBO; i++){
       printf(" NBO: %d | atom1: %d (%d) | atom2: %d (%d)\n",
              i,Table_NBO[i][0],Table_NBO[i][2],Table_NBO[i][1],Table_NBO[i][3]);
     }
     printf("\n");fflush(stdout);

     printf("### Number of NHO's & LP's ###\n");fflush(stdout);
     for (i=0; i<Num_FCenter; i++){
       printf("  %d :  %d  %d\n",i,Num_NHO[i],Num_LP[i]);fflush(stdout);
     }
     printf("\n");fflush(stdout);

  /** (7) Symmetrically orthogonalization of pre-NHOs in each atom **/

   tnum = 0; /* for counting the num. of LP's in system */

   for (i1=0; i1<Num_FCenter; i1++){
     Gc_AN = FSCenter[i1];
     wan1 = WhatSpecies[Gc_AN];
     leng1 = Spe_Total_CNO[wan1];
     leng2 = Num_NHO[i1];
     leng3 = Num_LP[i1];

   /* construction of pNHO matrix C */

     for (i=0; i<leng1; i++){
     for (j=1; j<=leng2; j++){
       pNHO_mat[i][j-1] = pNHO_vec[i1][j][i];
     }
     }

     if (leng3 != 0){
     for (i=0; i<leng1; i++){
     for (j=1; j<=leng3; j++){
       pNHO_mat[i][leng2+j-1] = pLP_vec[i1][j][i];
     }
     }
     }

/*
     printf("### pNHO Matrix (Gatom = %d (%d)) ###\n",Gc_AN,i1);fflush(stdout);
     for (i=0; i<leng1; i++){
     for (j=0; j<leng2+leng3; j++){
       printf("%9.5f",pNHO_mat[i][j]);fflush(stdout);
     } 
     printf("\n");fflush(stdout);
     } 
     printf("\n");fflush(stdout);
*/

   /* overlap matrix of pNHO */

     for (i=0; i<leng2+leng3; i++){
     for (j=0; j<leng2+leng3; j++){
       tmp_ele1 = 0.0;
       for (k=0; k<leng1; k++){
         tmp_ele1 += pNHO_mat[k][i] * pNHO_mat[k][j];
       }
       temp_M2[i+1][j+1] = tmp_ele1;
     }
     }

/*
     printf("### Overlap Matrix of pNHO (Gatom = %d (%d)) ###\n",Gc_AN,i1);fflush(stdout);
     for (i=0; i<leng2+leng3; i++){
     for (j=0; j<leng2+leng3; j++){
       printf("%9.5f",temp_M2[i+1][j+1]);fflush(stdout);
     } 
     printf("\n");fflush(stdout);
     } 
     printf("\n");fflush(stdout);
*/

   /* diagonalization of overlap matrix */

     Eigen_lapack(temp_M2,temp_M7,leng2+leng3,leng2+leng3);

     if (outputlev==1){
       for (k=1; k<=leng2+leng3; k++){
         printf("k W %2d %15.12f\n",k,temp_M7[k]);
       }
       printf("\n");fflush(stdout);
     }

   /* check ill-conditioned eigenvalues */

     for (k=1; k<=leng2+leng3; k++){
       if (temp_M7[k]<1.0e-14) {
         printf("Found ill-conditioned eigenvalues (7)\n");
         printf("Stopped calculation\n");
         exit(1);
        }
     }

   /* construction of transfer matrix X */

     for (k=1; k<=leng2+leng3; k++){
       temp_M7[k] = 1.0/sqrt(temp_M7[k]);
     }

     for (i=1; i<=leng2+leng3; i++){
     for (j=1; j<=leng2+leng3; j++){
       temp_M1[i][j] = temp_M7[i] * temp_M2[j][i];
     }
     }

     for (i=1; i<=leng2+leng3; i++){
     for (j=1; j<=leng2+leng3; j++){
       tmp_ele1 = 0.0;
     for (k=1; k<=leng2+leng3; k++){
       tmp_ele1 += temp_M2[i][k] * temp_M1[k][j];
     }
       X_mat[i-1][j-1] = tmp_ele1;
     }
     }

     for (i=0; i<leng1; i++){
     for (j=0; j<leng2+leng3; j++){
       tmp_ele1 = 0.0;
       for (k=0; k<leng2+leng3; k++){
         tmp_ele1 += pNHO_mat[i][k] * X_mat[k][j];
       }
       NHO_mat[i1][i][j] = tmp_ele1;
     }
     }

   /* extraction of LP vector */

     if (leng3 != 0){
     for (j=1; j<=leng3; j++){
       tnum++;
     for (i=0; i<leng1; i++){
       LonePair_vec[tnum][i] = NHO_mat[i1][i][j+leng2-1];
       Table_LP[tnum][0] = i1;
     }
     }
     }

   /* overlap matrix of NHO */

     for (i=0; i<leng2+leng3; i++){
     for (j=0; j<leng2+leng3; j++){
       tmp_ele1 = 0.0;
       for (k=0; k<leng1; k++){
         tmp_ele1 += NHO_mat[i1][k][i] * NHO_mat[i1][k][j];
       }
       temp_M2[i+1][j+1] = tmp_ele1;
     }
     }

/*
     printf("### Overlap Matrix of NHO (Gatom = %d (%d)) ###\n",Gc_AN,i1);fflush(stdout);
     for (i=0; i<leng2+leng3; i++){
     for (j=0; j<leng2+leng3; j++){
       printf("%9.5f",temp_M2[i+1][j+1]);fflush(stdout);
     } 
     printf("\n");fflush(stdout);
     } 
     printf("\n");fflush(stdout);
*/
   /* NHO */
/*
     printf("### NHO Matrix (Gatom = %d (%d)) ###\n",Gc_AN,i1);fflush(stdout);
     for (i=0; i<leng1; i++){
     for (j=0; j<leng2+leng3; j++){
       printf("%9.5f",NHO_mat[i1][i][j]);fflush(stdout);
     } 
     printf("\n");fflush(stdout);
     } 
     printf("\n");fflush(stdout);
*/

   } /* i1 */





   /* Cuber-data output of NHO */

   printf("### Cube-data of NHOs ###\n\n");fflush(stdout);

   num = 0;
   for (i1=0; i1<Num_FCenter; i1++){
     Gc_AN = FSCenter[i1];
     wan1 = WhatSpecies[Gc_AN];

   for (i=0; i<Spe_Total_NO[wan1]; i++){
     Posi2Gatm[num] = Gc_AN;
     num++;
   }
   }

   LVsize = num;

   tnum = -1;
   for (i1=0; i1<Num_FCenter; i1++){
     Gc_AN = FSCenter[i1];
     wan1 = WhatSpecies[Gc_AN];
     posi1 = LMindx2LMposi[i1];
     leng1 = Spe_Total_CNO[wan1];
     leng2 = Num_NHO[i1];
     leng3 = Num_LP[i1];

     for (j=0; j<leng2+leng3; j++){
       tnum++;
       pnum = -1;

     for (i=0; i<LVsize; i++){
       tmp_ele1 = 0.0;
       Gh_AN = Posi2Gatm[i];

       if (i==0) Gb_AN = Gh_AN;

       if (Gh_AN == Gb_AN) pnum++;
       if (Gh_AN != Gb_AN) pnum = 0;

       for (k=0; k<leng1; k++){
         tmp_ele1 += T_full[i][posi1+k] * NHO_mat[i1][k][j];
       }

       NHOs_Coef[spin0][tnum][Gh_AN][pnum] = tmp_ele1;
       Gb_AN = Gh_AN;

     }
     }

     Num_NHOs[i1] = leng2+leng3;

   }

   Total_Num_NHO = tnum;



    char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
    char *Name_Multiple[20];

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


   printf("### Component of NHOs on the basis of PAO (TEST) ###\n\n");fflush(stdout);

      tnum = 0;

      for (i=0; i<Num_NBO_FCenter; i++){
        Gc_AN = NBO_FCenter[i];
        wan1 = WhatSpecies[Gc_AN];
        pnum = Num_NHOs[i];

      for (i1=0; i1<pnum; i1++){
        printf(" # Global atom num.: %d ( %s ) / NHO: %d \n",Gc_AN,SpeName[wan1],tnum);
        printf(" ---------------------------------------\n");
        printf("  PAO            Coefficient            \n");
        printf(" ---------------------------------------\n");
        k = 0;

      for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Spe_Num_Basis[wan1][L1];

      for (M1=1; M1<=2*L1+1; M1++){

      for (j=0; j<leng1; j++){
        printf(" %s %3s   ",Name_Multiple[j+1],Name_Angular[L1][M1-1]);
        printf("%9.5f \n",NHOs_Coef[spin0][tnum][Gc_AN][k]);
        k++;
      } /* j */
      } /* M1 */
      } /* L1 */
        printf(" -------------------------------------\n\n");
        tnum++;
      } /* i1 */
      } /* i */
      printf("\n");






   /*** Rearranging order of basis ***/

   printf("# Rearranging order of basis in Cube-data #\n\n");fflush(stdout);

   for (i=0; i<=Total_Num_NHO; i++){
   for (i1=0; i1<Num_FCenter; i1++){
     Gc_AN = FSCenter[i1];
     wan1 = WhatSpecies[Gc_AN];
     Lmax = Spe_MaxL_Basis[wan1];
     tnum = 0;
     pnum = 0;

     for (L1=0; L1<=Lmax; L1++){
       Mulmax = Spe_Num_Basis[wan1][L1];
       pnum = tnum;

       for (j=0; j<Mulmax*(2*L1+1); j++){
         temp_M7[j] = NHOs_Coef[spin0][i][Gc_AN][j+pnum];
#if 0
printf("### L1=%d  j=%d  pnum=%d  temp_M7=%9.5f \n",L1,j,pnum,temp_M7[j]);
#endif
         tnum++;
       }

       if (Mulmax > 1){
         qnum = 0;
         for (k=0; k<Mulmax; k++){
         for (j=0; j<(2*L1+1); j++){
           temp_M8[qnum] = temp_M7[k + j*Mulmax];
#if 0
printf("### L1=%d  qnum=%d  temp_M8=%9.5f  k=%d  j=%d\n",L1,qnum,temp_M8[qnum],k,j);
#endif
           qnum++;
         }
         }

         for (j=0; j<Mulmax*(2*L1+1); j++){
           NHOs_Coef[spin0][i][Gc_AN][j+pnum] = temp_M8[j];
         }
       }

     }
   }
   }






   /*** Component of NHOs ***/

   printf("### Component of NHOs on the basis of PAO ###\n\n");fflush(stdout);
/*
    char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
    char *Name_Multiple[20];

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
*/

      tnum = 0;

      for (i=0; i<Num_NBO_FCenter; i++){
        Gc_AN = NBO_FCenter[i];
        wan1 = WhatSpecies[Gc_AN];
        pnum = Num_NHOs[i];       
 
      for (i1=0; i1<pnum; i1++){
        printf(" # Global atom num.: %d ( %s ) / NHO: %d \n",Gc_AN,SpeName[wan1],tnum);
        printf(" ---------------------------------------\n");
        printf("  PAO            Coefficient            \n");
        printf(" ---------------------------------------\n");
        k = 0;
     
      for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Spe_Num_Basis[wan1][L1];

      for (M1=1; M1<=2*L1+1; M1++){

      for (j=0; j<leng1; j++){
        printf(" %s %3s   ",Name_Multiple[j+1],Name_Angular[L1][M1-1]);
        printf("%9.5f \n",NHOs_Coef[spin0][tnum][Gc_AN][k]);
        k++;
      } /* j */
      } /* M1 */
      } /* L1 */
        printf(" -------------------------------------\n\n");
        tnum++;
      } /* i1 */ 
      } /* i */
      printf("\n");


   printf("### Component of NHOs on the basis of NAO ###\n\n");fflush(stdout);

      tnum = 0;

      for (i=0; i<Num_NBO_FCenter; i++){
        Gc_AN = NBO_FCenter[i];
        wan1 = WhatSpecies[Gc_AN];
        leng1 = Spe_Total_CNO[wan1];
        pnum = Num_NHO[i]+Num_LP[i];

      for (i1=0; i1<pnum; i1++){
        printf(" # Global atom num.: %d ( %s ) / NHO: %d \n",Gc_AN,SpeName[wan1],tnum);
        printf(" ---------------------------------------\n");
        printf("  NAO            Coefficient            \n");
        printf(" ---------------------------------------\n");
        k = 0;

      for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
        leng1 = Spe_Num_Basis[wan1][L1];

      for (M1=1; M1<=2*L1+1; M1++){

      for (j=0; j<leng1; j++){
        printf(" %s %3s   ",Name_Multiple[j+1],Name_Angular[L1][M1-1]);
        printf("%9.5f \n",NHO_mat[i][k][i1]);
        k++;
      } /* j */
      } /* M1 */
      } /* L1 */
        printf(" -------------------------------------\n\n");
        tnum++;
      } /* i1 */
      } /* i */
      printf("\n");


   pnum = -1;
   for (i1=0; i1<Num_FCenter; i1++){
     tnum = Num_NHO[i1];
   for (i=1; i<=tnum; i++){
     pnum++;
     NHO_indexL2G[i1][i] = pnum;
     printf("### NHO_indexL2G[%d][%d] = %d ###\n",i1,i,pnum);fflush(stdout);
   }
     pnum += Num_LP[i1];
   }
   printf("\n");


   printf("###  Total Num. of NHOs = %d \n\n",Total_Num_NHO+1);fflush(stdout);


   printf("******************************************\n");
   printf("           NATURAL BOND ORBITAL           \n");
   printf("******************************************\n");
   printf("\n");

   for (i=1; i<=Num_NBO; i++){
     i1   = Table_NBO[i][0];
     i2   = Table_NBO[i][1];
     tnum = Table_NBO[i][2];
     pnum = Table_NBO[i][3];

     Gc_AN = FSCenter[i1];
     Gh_AN = FSCenter[i2];
     wan1  = WhatSpecies[Gc_AN];
     wan2  = WhatSpecies[Gh_AN];
     leng1 = Spe_Total_CNO[wan1];
     leng2 = Spe_Total_CNO[wan2]; 

     snum = NHO_indexL2G[i1][tnum];
     qnum = NHO_indexL2G[i2][pnum];
/*
     printf("### NHO_indexL2G = %d , %d\n\n",snum,qnum);fflush(stdout);
*/
   /* construction of bond-pair NHO vector V_nho */

     for (j=0; j<=MaxLeng_a*2; j++){
       Bond_Pair_vec[j][1] = 0.0;
       Bond_Pair_vec[j][2] = 0.0;
     }

     for (j=0; j<leng1; j++){
       Bond_Pair_vec[j][1] = NHO_mat[i1][j][tnum-1];
     }

     for (j=0; j<leng2; j++){
       Bond_Pair_vec[leng1+j][2] = NHO_mat[i2][j][pnum-1];
     }

/*
     printf("### Bond_Pair_vector (%d (%d),%d (%d)) (NBO=%d) ###\n",Gc_AN,i1,Gh_AN,i2,i);
     for (k=0; k<leng1+leng2; k++){
     for (j=1; j<=2; j++){
       printf("%9.5f",Bond_Pair_vec[k][j]);
     } 
     printf("\n");
     } 
     printf("\n");
*/

   /* V_nho^t * P_pair * V_nho  =>  2x2 P-matrix */

     for (j=0; j<leng1+leng2; j++){
     for (k=1; k<=2; k++){
       tmp_ele1 = 0.0;
       for (l=0; l<leng1+leng2; l++){
         tmp_ele1 += P_pair[i1][i2][j][l] * Bond_Pair_vec[l][k];
       }
       temp_M1[j][k] = tmp_ele1;
     }
     }

     for (j=1; j<=2; j++){
     for (k=1; k<=2; k++){
       tmp_ele1 = 0.0;
       for (l=0; l<leng1+leng2; l++){
         tmp_ele1 += Bond_Pair_vec[l][j] * temp_M1[l][k];
       }
       temp_M2[j][k] = tmp_ele1;
     }
     }

/*
     printf("### 2x2 P-matrix (%d,%d)(%d) ###\n",Gc_AN,Gh_AN,i);
     printf("  %9.5f  %9.5f\n",temp_M2[1][1],temp_M2[1][2]);
     printf("  %9.5f  %9.5f\n",temp_M2[2][1],temp_M2[2][2]);
     printf("\n");
*/

   /* diagonalization of 2x2 P-matrix */

     Eigen_lapack(temp_M2,temp_M7,2,2);

     printf("### NBO:%d (%s%d-%s%d) ###\n",i,SpeName[wan1],Gc_AN,SpeName[wan2],Gh_AN);
     printf("  elec. in bond orb.:      %9.5f\n",temp_M7[2]);
     printf("  elec. in anti-bond orb.: %9.5f\n",temp_M7[1]);
     printf("\n");
     printf("  %9.5f  %9.5f\n",temp_M2[1][1],temp_M2[1][2]);
     printf("  %9.5f  %9.5f\n",temp_M2[2][1],temp_M2[2][2]);
     printf("\n");

     Pop_bond_NBO[i] = temp_M7[2];
     Pop_anti_NBO[i] = temp_M7[1];

   /* construction of NBO vectors */

     for (j=0; j<leng1; j++){
       NBO_bond_vec[i][j]       = NHO_mat[i1][j][tnum-1] * temp_M2[1][2];
       NBO_anti_vec[i][j]       = NHO_mat[i1][j][tnum-1] * temp_M2[1][1];
     }
     for (j=0; j<leng2; j++){
       NBO_bond_vec[i][leng1+j] = NHO_mat[i2][j][pnum-1] * temp_M2[2][2];
       NBO_anti_vec[i][leng1+j] = NHO_mat[i2][j][pnum-1] * temp_M2[2][1];
     }

   /* Cube-data of NBOs */

     for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
       wan1  = WhatSpecies[Gc_AN];
       leng3 = Spe_Total_CNO[wan1];
     for (j=0; j<leng3; j++){
       NBOs_Coef_b[spin0][i-1][Gc_AN][j] = NHOs_Coef[spin0][snum][Gc_AN][j] * temp_M2[1][2]
                                         + NHOs_Coef[spin0][qnum][Gc_AN][j] * temp_M2[2][2];
       NBOs_Coef_a[spin0][i-1][Gc_AN][j] = NHOs_Coef[spin0][snum][Gc_AN][j] * temp_M2[1][1]
                                         + NHOs_Coef[spin0][qnum][Gc_AN][j] * temp_M2[2][1];
     }
     }

/*
     printf("### NBO vector (%d) ###\n",i);
     for (j=0; j<leng1+leng2; j++){
       printf("  %9.5f  %9.5f\n",NBO_bond_vec[i][j],NBO_anti_vec[i][j]);
     }
     printf("\n");
*/
   } /* i (NBO-loop) */


   /* population on LP */

   for (i=1; i<=Num_LonePair; i++){
     i1    = Table_LP[i][0];

   if (i1 < Num_FCenter){
     Gc_AN = FSCenter[i1];
     wan1  = WhatSpecies[Gc_AN];
     leng1 = Spe_Total_CNO[wan1];
     posi1 = LMindx2LMposi[i1];

     for (j=0; j<leng1; j++){
     for (k=0; k<leng1; k++){
       temp_M1[j][k] = P_full[j+posi1][k+posi1];
     } /* j */
     } /* i */

     for (j=0; j<leng1; j++){
       tmp_ele1 = 0.0;
     for (k=0; k<leng1; k++){
       tmp_ele1 += temp_M1[j][k] * LonePair_vec[i][k];
     }
       temp_M2[j][0] = tmp_ele1;
     }

     tmp_ele1 = 0.0;
     for (j=0; j<leng1; j++){
       tmp_ele1 += LonePair_vec[i][j] * temp_M2[j][0];
     }

     Pop_LonePair[i] = tmp_ele1;

     printf("### Lone Pair (%d) (atom : %s%d) ###\n",i,SpeName[wan1],Gc_AN);
     printf("  elec. in lone pair :      %9.5f\n",Pop_LonePair[i]);
     printf("\n");
/*
     for (j=0; j<leng1; j++){
       printf("  %9.5f\n",LonePair_vec[i][j]);
     }
     printf("\n");
*/
   }
   } /* i (LP-loop) */


   printf("******************************************\n");
   printf("     Orbital Interaction Analysis         \n");
   printf("     based on 2nd Perturbation Theory     \n");
   printf("******************************************\n");
   printf("\n");

   /*** construction of 2x2 Fock matrix for NBO basis ***/

   /** bonding NBO and antibonding NBO **/

   for (i=1; i<=Num_NBO; i++){
     i1 = Table_NBO[i][0];
     i2 = Table_NBO[i][1];

     Gb1_AN = FSCenter[i1];
     Gb2_AN = FSCenter[i2];

     wan1  = WhatSpecies[Gb1_AN];
     wan2  = WhatSpecies[Gb2_AN];

     leng[1] = Spe_Total_CNO[wan1];
     leng[2] = Spe_Total_CNO[wan2];
     posi[1] = LMindx2LMposi[i1];
     posi[2] = LMindx2LMposi[i2];

     t_posi[1] = 0;
     t_posi[2] = leng[1];
     t_posi[3] = leng[1] + leng[2];

   for (j=1; j<=Num_NBO; j++){
     j1 = Table_NBO[j][0];
     j2 = Table_NBO[j][1];

     Ga1_AN = FSCenter[j1];
     Ga2_AN = FSCenter[j2];

     wan3  = WhatSpecies[Ga1_AN];
     wan4  = WhatSpecies[Ga2_AN];

     leng[3] = Spe_Total_CNO[wan3];
     leng[4] = Spe_Total_CNO[wan4];
     posi[3] = LMindx2LMposi[j1];
     posi[4] = LMindx2LMposi[j2];

     t_posi[4] = leng[1] + leng[2] + leng[3];
     t_leng    = leng[1] + leng[2] + leng[3] + leng[4];

     for (k=1; k<=4; k++){
     for (l=1; l<=4; l++){

       for (m=0; m<leng[k]; m++){
       for (n=0; n<leng[l]; n++){
         temp_M1[t_posi[k]+m][t_posi[l]+n] = H_full[posi[k]+m][posi[l]+n];
       }
       }

     }
     }
/*
     printf("### Fock matrix (bondind NBO:%d , anti-bonding NBO:%d) \n",i,j);
     for (k=0; k<t_leng; k++){
     for (m=0; m<t_leng; m++){
       printf("%9.5f",temp_M1[k][m]);
     }
     printf("\n");
     }
     printf("\n");
*/
     /* construction of pair-vector for bond NBO & antibond NBO */

     for (k=0; k<t_leng; k++){
       temp_M2[k][0] = 0.0;
       temp_M2[k][1] = 0.0;
     }

     for (k=0; k<leng[1]+leng[2]; k++){
       temp_M2[k][0] = NBO_bond_vec[i][k];
     }

     for (k=0; k<leng[3]+leng[4]; k++){
       temp_M2[t_posi[3]+k][1] = NBO_anti_vec[j][k];
     }

     /* construction of Fock matrix for NBO */

     for (k=0; k<t_leng; k++){
     for (l=0; l<=1; l++){
       tmp_ele1 = 0.0;
       for (m=0; m<t_leng; m++){
         tmp_ele1 += temp_M1[k][m] * temp_M2[m][l];
       }
       temp_M10[k][l] = tmp_ele1;
     }
     }

     for (k=0; k<=1; k++){
     for (l=0; l<=1; l++){
       tmp_ele1 = 0.0;
       for (m=0; m<t_leng; m++){
         tmp_ele1 += temp_M2[m][k] * temp_M10[m][l];
       }
       Fij[k][l] = tmp_ele1;
     }
     }

     tmp_ele1 = Pop_bond_NBO[i] * Fij[0][1] * Fij[0][1] / (Fij[1][1] - Fij[0][0]);

     printf("### bondind NBO:%d (%s%d-%s%d), anti-bonding NBO:%d (%s%d-%s%d))\n",
            i,SpeName[wan1],Gb1_AN,SpeName[wan2],Gb2_AN,
            j,SpeName[wan3],Ga1_AN,SpeName[wan4],Ga2_AN);
     printf("  Fock matrix :\n");
     printf("   F11= %9.5f  F12= %9.5f \n",Fij[0][0],Fij[0][1]);
     printf("   F21= %9.5f  F22= %9.5f \n",Fij[1][0],Fij[1][1]);
     printf("  Population : b-NBO = %9.5f a-NBO = %9.5f \n",Pop_bond_NBO[i],Pop_anti_NBO[j]);
     printf("  Interaction Energy = %9.5f (Hartree)\n",tmp_ele1);
     printf("                     = %9.5f (eV)\n",tmp_ele1*27.2116);
     printf("                     = %9.5f (kcal/mol)\n\n",tmp_ele1*627.509);


   } /* j (for anti-bonding NBO) */
   } /* i (for bonding NBO) */

   /* lone pair and antibonding NBO */

   for (i=1; i<=Num_LonePair; i++){
     i1 = Table_LP[i][0];
   if (i1 < Num_FCenter){
     Gb1_AN = FSCenter[i1];
     wan1  = WhatSpecies[Gb1_AN];

     posi[1] = LMindx2LMposi[i1];
     leng[1] = Spe_Total_CNO[wan1];

     t_posi[1] = 0;
     t_posi[2] = leng[1];

   for (j=1; j<=Num_NBO; j++){
     j1 = Table_NBO[j][0];
     j2 = Table_NBO[j][1];

     Ga1_AN = FSCenter[j1];
     Ga2_AN = FSCenter[j2];
     wan3  = WhatSpecies[Ga1_AN];
     wan4  = WhatSpecies[Ga2_AN];

     leng[2] = Spe_Total_CNO[wan3];
     leng[3] = Spe_Total_CNO[wan4];
     posi[2] = LMindx2LMposi[j1];
     posi[3] = LMindx2LMposi[j2];

     t_posi[3] = leng[1] + leng[2];
     t_leng    = leng[1] + leng[2] + leng[3];

     for (k=1; k<=3; k++){
     for (l=1; l<=3; l++){

       for (m=0; m<leng[k]; m++){
       for (n=0; n<leng[l]; n++){
         temp_M1[t_posi[k]+m][t_posi[l]+n] = H_full[posi[k]+m][posi[l]+n];
       }
       }

     }
     }
/*
     printf("### Fock matrix (Lone Pair:%d , anti-bonding NBO:%d) \n",i,j);
     for (k=0; k<t_leng; k++){
     for (m=0; m<t_leng; m++){
       printf("%9.5f",temp_M1[k][m]);
     }
     printf("\n");
     }
     printf("\n");
*/
     /** construction of pair-vector for lone pair & antibond NBO **/

     for (k=0; k<t_leng; k++){
       temp_M2[k][0] = 0.0;
       temp_M2[k][1] = 0.0;
     }

     for (k=0; k<leng[1]; k++){
       temp_M2[k][0] = LonePair_vec[i][k];
     }

     for (k=0; k<leng[2]+leng[3]; k++){
       temp_M2[t_posi[2]+k][1] = NBO_anti_vec[j][k];
     }
/*
     for (k=0; k<leng[1]+leng[2]+leng[3]; k++){
       printf("## TEST ##  %9.5f %9.5f \n",temp_M2[k][0],temp_M2[k][1]);
     }
     printf("\n");
*/
     /* construction of Fock matrix for NBO */

     for (k=0; k<t_leng; k++){
     for (l=0; l<=1; l++){
       tmp_ele1 = 0.0;
       for (m=0; m<t_leng; m++){
         tmp_ele1 += temp_M1[k][m] * temp_M2[m][l];
       }
       temp_M10[k][l] = tmp_ele1;
     }
     }

     for (k=0; k<=1; k++){
     for (l=0; l<=1; l++){
       tmp_ele1 = 0.0;
       for (m=0; m<t_leng; m++){
         tmp_ele1 += temp_M2[m][k] * temp_M10[m][l];
       }
       Fij[k][l] = tmp_ele1;
     }
     }

     tmp_ele1 = Pop_LonePair[i] * Fij[0][1] * Fij[0][1] / (Fij[1][1] - Fij[0][0]);

     printf("### lone pair:%d (%s%d), anti-bonding NBO:%d (%s%d-%s%d)) \n",
            i,SpeName[wan1],Gb1_AN,
            j,SpeName[wan3],Ga1_AN,SpeName[wan4],Ga2_AN);
     printf("  Fock matrix :\n");
     printf("   F11= %9.5f  F12= %9.5f \n",Fij[0][0],Fij[0][1]);
     printf("   F21= %9.5f  F22= %9.5f \n",Fij[1][0],Fij[1][1]);
     printf("  Population : LP = %9.5f a-NBO = %9.5f \n",Pop_LonePair[i],Pop_anti_NBO[j]);
     printf("  Interaction Energy = %9.5f (Hartree)\n",   tmp_ele1);
     printf("                     = %9.5f (eV)\n",        tmp_ele1*27.2116);
     printf("                     = %9.5f (kcal/mol)\n\n",tmp_ele1*627.509);

   } /* j (for anti-bonding NBO) */
   }
   } /* i (for LP) */


  /***  Freeing arrays  ***/

    free(Posi2Gatm);
    free(Num_LP);
    free(Num_NHO);
    free(Pop_bond_NBO);
    free(Pop_anti_NBO);
    free(Pop_LonePair);

    for (i=0; i<=Num_FSCenter*10; i++){
     free(Table_NBO[i]);
    }
     free(Table_NBO);

    for (i=0; i<=Num_FSCenter*5; i++){
     free(Table_LP[i]);
    }
     free(Table_LP);

    for (i=0; i<=Num_FSCenter; i++){
     free(NHO_indexL2G[i]);
    }
     free(NHO_indexL2G);

    for (i=0; i<=Num_FSCenter; i++){
    for (j=0; j<=MaxLeng_a; j++){
     free(pLP_vec[i][j]);
    }
     free(pLP_vec[i]);
    }
     free(pLP_vec);

    for (i=0; i<=Num_FSCenter; i++){
    for (j=0; j<=MaxLeng_a; j++){
     free(pNHO_vec[i][j]);
    }
     free(pNHO_vec[i]);
    }
     free(pNHO_vec);

    for (i=0; i<=Num_FSCenter; i++){
    for (j=0; j<=MaxLeng_a; j++){
    for (k=0; k<=MaxLeng_a; k++){
     free(pLP_mat[i][j][k]);
    }
     free(pLP_mat[i][j]);
    }
     free(pLP_mat[i]);
    }
     free(pLP_mat);

    for (i=0; i<=MaxLeng_a*2; i++){
     free(pNHO_mat[i]);
    }
     free(pNHO_mat);

    for (i=0; i<=MaxLeng_a; i++){
     free(X_mat[i]);
    }
     free(X_mat);

    for (i=0; i<=Num_FSCenter; i++){
    for (j=0; j<=MaxLeng_a; j++){
     free(NHO_mat[i][j]);
    }
     free(NHO_mat[i]);
    }
     free(NHO_mat);

    for (i=0; i<=MaxLeng_a*2; i++){
     free(Bond_Pair_vec[i]);
    }
     free(Bond_Pair_vec);

    for (i=0; i<=Num_FSCenter*10; i++){
     free(NBO_bond_vec[i]);
    }
     free(NBO_bond_vec);

    for (i=0; i<=Num_FSCenter*10; i++){
     free(NBO_anti_vec[i]);
    }
     free(NBO_anti_vec);

    for (i=0; i<=Num_FSCenter*5; i++){
     free(LonePair_vec[i]);
    }
     free(LonePair_vec);

    for (i=0; i<=Num_FSCenter; i++){
    for (j=0; j<=Num_FSCenter; j++){
    for (k=0; k<=MaxLeng_a*2; k++){
     free(P_pair[i][j][k]);
    }
     free(P_pair[i][j]);
    }
     free(P_pair[i]);
    }
     free(P_pair);

    for (i=0; i<=MaxLeng_a*5; i++){
     free(temp_M1[i]);
    }
     free(temp_M1);

    for (i=0; i<=MaxLeng_a*5; i++){
     free(temp_M2[i]);
    }
     free(temp_M2);

    for (i=0; i<=MaxLeng_a*5; i++){
     free(temp_M10[i]);
    }
     free(temp_M10);

    free(temp_M7);
    free(temp_M8);


} /** end of "Calc_NBO" **/

