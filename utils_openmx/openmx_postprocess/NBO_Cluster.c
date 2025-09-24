/**********************************************************************
  NBO-13.c:

     NBO-1.c is a subroutine to calculate atomic localized orbitals,
     NAOs (Natural Atomic Orbitals), which are precursors of
     subsequent localized orbitals, NHOs (Natural Hybrid Orbitals)
     and NBOs (Natural Bond Orbitals).

       Reference: J. Chem. Phys., vol. 83, pp 735-746 (19.5)

    Log of NBO-8.c:

    22/Feb/2012 -- Released by T. Ohwaki (NISSAN Research Center, Jpn)
    16/Jul/2012 --  

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "f77func.h"

#define  measure_time 0

void Calc_NAO_Cluster(double *****CDM){

  /* loop */
  int spin0,spin,Gc_AN,Gh_AN,Gb_AN,h_AN,Mc_AN,L1,M1,i,j,i2,j2,k,l,is;

  /* temporary */
  int tnum,pnum,snum,qnum,num;
  int wan1,wan2,wan3,wan4,tmp_num1,tmp_num2,tmp_num3,mtps1,mtps2;
  int posi1,posi2,leng1,leng2,leng3,Mmax1,Mmax2;
  int outputlev = 1;  
  double tmp_ele0,tmp_ele1,tmp_ele2,tmp_ele3,tmp_ele4,sum;
  double tmp_ele00,tmp_ele01,tmp_ele02,tmp_ele03;
  double tmp_ele10,tmp_ele11,tmp_ele12,tmp_ele13;
  double tmp_ele20,tmp_ele21,tmp_ele22,tmp_ele23;
  double ***temp_M1,***temp_M2,***temp_M3,***temp_M4;
  double *temp_V1,*temp_V2,*temp_V3;

  /* BLAS */
  int M,N,K,lda,ldb,ldc;
  double alpha,beta;

  /* data */
  int *ID2nump,*ID2num,*ID2posi,**ID2G,*MatomN;
  int *Leng_a,**Leng_al,**Leng_alm,*Posi_a,**Posi_al,***Posi_alm,*Posi2Gatm;
  int Lmax,Mulmax,SizeMat,***M_or_R1,*M_or_R2,*S_or_L;
  int Num_NMB,**Num_NMB1,*NMB_posi1,*NMB_posi2;
  int Num_NRB,**Num_NRB1,Num_NRB2,Num_NRB3;
  int *NRB_posi1,*NRB_posi2,*NRB_posi3,*NRB_posi4;
  int *NRB_posi5,*NRB_posi6,*NRB_posi7,*NRB_posi8,*NRB_posi9;
  int MaxLeng_a,MaxLeng_al,MaxLeng_alm;
  double thredens = NAO_Occ_or_Ryd/(1.0+SpinP_switch);
  double *Tmp_Vec0,*Tmp_Vec1,*Tmp_Vec2,*Tmp_Vec3;
  double ***S_full,***S_full0,***D_full,***H_full,***H_full0;
  double ***P_full,***P_full0,***T_full,**W_full;
  double ******P_alm,******S_alm;
  double ***P_alm2,***S_alm2,***N_alm2,**W_alm2;
  double ***N_tran1,***N_tran2;
  double **W_NMB,***S_NMB;
  double ***Sw1,***Uw1,**Rw1,***Ow_NMB2;
  double ***Sw2,***Uw2,**Rw2,***O_sym;
  double **W_NRB,**W_NRB2,***S_NRB,***S_NRB2,***Ow_NRB2;
  /*
    double ***temp_M5,***temp_M6;
    double ***N_diag,***N_ryd,***N_red,***Ow_NMB,,***Ow_NRB,***O_schm,***O_schm2,***O_sym;
  */
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


  ID2nump = (int*)malloc(sizeof(int)*(numprocs));
  for (i=0; i<numprocs; i++){
    ID2nump[i] = 0;
  }

  ID2num  = (int*)malloc(sizeof(int)*(numprocs));
  for (i=0; i<numprocs; i++){
    ID2num[i] = 0;
  }

  ID2posi = (int*)malloc(sizeof(int)*(numprocs));
  for (i=0; i<numprocs; i++){
    ID2posi[i] = 0;
  }

  MatomN = (int*)malloc(sizeof(int)*(numprocs));
  for (i=0; i<numprocs; i++){
    MatomN[i] = 0;
  }

  ID2G = (int**)malloc(sizeof(int*)*(numprocs));
  for (i=0; i<numprocs; i++){
    ID2G[i] = (int*)malloc(sizeof(int)*(atomnum+1));
    for (j=0; j<=atomnum; j++){
      ID2G[i][j] = 0;
    }
  }


  tnum = 0;

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    ID = G2ID[Gc_AN];

    if (myid==ID){
      num = 0;
      wan1 = WhatSpecies[Gc_AN];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        wan2 = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[wan1]; i++){
          for (j=0; j<Spe_Total_CNO[wan2]; j++){
            num++;
          }
	}
      } /* h_AN */

      tnum += num;
      ID2nump[ID] += num;
    }
  } /* Gc_AN */

  MPI_Reduce(&tnum, &pnum, 1, MPI_INT, MPI_SUM, Host_ID, mpi_comm_level1);
  MPI_Gather(&ID2nump[myid], 1, MPI_INT, &ID2num[0], 1, MPI_INT, Host_ID, mpi_comm_level1);

  for (k=1; k<numprocs; k++){
    ID2posi[k] = ID2posi[k-1] + ID2num[k-1];
  }

  if (myid==Host_ID){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      ID = G2ID[Gc_AN];
      ID2G[ID][MatomN[ID]] = Gc_AN;
      MatomN[ID]++;
    }
  }

  if (myid==Host_ID && outputlev==1){
    for (k=0; k<numprocs; k++){
      printf("MatomN[%3d] = %3d \n",k,MatomN[k]);fflush(stdout);
    }
    for (k=0; k<numprocs; k++){
      for (i=0; i<MatomN[k]; i++){
	printf("ID2G[%3d][%3d] = %3d \n",k,i,ID2G[k][i]);fflush(stdout);
      }
    }
    for (k=0; k<numprocs; k++){
      printf("ID2posi[%3d], ID2num[%3d] = %4d %4d \n",k,k,ID2posi[k],ID2num[k]);fflush(stdout);
    }
  }

  Tmp_Vec0 = (double*)malloc(sizeof(double)*ID2nump[myid]);
  Tmp_Vec1 = (double*)malloc(sizeof(double)*pnum);
  Tmp_Vec2 = (double*)malloc(sizeof(double)*pnum);
  Tmp_Vec3 = (double*)malloc(sizeof(double)*pnum);


  for (spin0=0; spin0<=SpinP_switch; spin0++){

    /***************************************************************
                          overlap matrix
    ****************************************************************/

    tnum = 0;

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      ID = G2ID[Gc_AN];

      if (myid==ID){
	num = 0;
	Mc_AN = F_G2M[Gc_AN];
	wan1 = WhatSpecies[Gc_AN];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];

          for (i=0; i<Spe_Total_CNO[wan1]; i++){
	    for (j=0; j<Spe_Total_CNO[wan2]; j++){
	      Tmp_Vec0[num+tnum] = OLP[0][Mc_AN][h_AN][i][j];
	      num++;
	    }
          }

	} /* h_AN */

	tnum += num;
 
      }  

    } /* Gc_AN */

    MPI_Gatherv(&Tmp_Vec0[0], ID2nump[myid], MPI_DOUBLE, 
		&Tmp_Vec1[0], ID2num, ID2posi, MPI_DOUBLE, Host_ID, mpi_comm_level1);

    if (myid==Host_ID && outputlev==2){
      for (k=0; k<pnum; k++){
	printf("Tmp_Vec1[%3d] = %9.5f \n",k,Tmp_Vec1[k]);fflush(stdout);
      }
    }

    /***************************************************************
                         density matrix
    ****************************************************************/

    tnum = 0;

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      ID = G2ID[Gc_AN];

      if (myid==ID){
	num = 0;
	Mc_AN = F_G2M[Gc_AN];
	wan1 = WhatSpecies[Gc_AN];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];

          for (i=0; i<Spe_Total_CNO[wan1]; i++){
	    for (j=0; j<Spe_Total_CNO[wan2]; j++){
	      Tmp_Vec0[num+tnum] = CDM[spin0][Mc_AN][h_AN][i][j];
	      num++;
	    }
          }

	} /* h_AN */

	tnum += num;

      }
    } /* Gc_AN */


    MPI_Gatherv(&Tmp_Vec0[0], ID2nump[myid], MPI_DOUBLE,
		&Tmp_Vec2[0], ID2num, ID2posi, MPI_DOUBLE, Host_ID, mpi_comm_level1);

    if (myid==Host_ID && outputlev==2){
      for (k=0; k<pnum; k++){
	printf("Tmp_Vec2[%3d] = %9.5f \n",k,Tmp_Vec2[k]);fflush(stdout);
      }
    }

    /***************************************************************
                        Hamiltonian matrix
    ****************************************************************/

    tnum = 0;

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      ID = G2ID[Gc_AN];

      if (myid==ID){
	num = 0;
	Mc_AN = F_G2M[Gc_AN];
	wan1 = WhatSpecies[Gc_AN];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];

          for (i=0; i<Spe_Total_CNO[wan1]; i++){
	    for (j=0; j<Spe_Total_CNO[wan2]; j++){
	      Tmp_Vec0[num+tnum] = H[spin0][Mc_AN][h_AN][i][j];
	      num++;
	    }
          }

	} /* h_AN */

	tnum += num;

      }
    } /* Gc_AN */

    MPI_Gatherv(&Tmp_Vec0[0], ID2nump[myid], MPI_DOUBLE,
		&Tmp_Vec3[0], ID2num, ID2posi, MPI_DOUBLE, Host_ID, mpi_comm_level1);

    if (myid==Host_ID && outputlev==2){
      for (k=0; k<pnum; k++){
	printf("Tmp_Vec3[%3d] = %9.5f \n",k,Tmp_Vec3[k]);fflush(stdout);
      }
    }

    if (myid == Host_ID){

      if (spin0==0){

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

      if (SpinP_switch==1 && spin0==0){

	printf("\n");fflush(stdout);
	printf(" *********************************************** \n");fflush(stdout);
	printf("                   for up-spin                   \n");fflush(stdout);
	printf(" *********************************************** \n");fflush(stdout);
	printf("\n");fflush(stdout);

      }
      else if (SpinP_switch==1 && spin0==1){

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
      else if (SpinP_switch==1 && spin0==0){

	printf("\n");fflush(stdout);
	printf(" *********************************************** \n");fflush(stdout);
	printf("        STAGE-1: NAO & NPA (for up-spin)         \n");fflush(stdout);
	printf(" *********************************************** \n");fflush(stdout);
	printf("\n");fflush(stdout);

      }
      else if (SpinP_switch==1 && spin0==1){

	printf("\n");fflush(stdout);
	printf(" *********************************************** \n");fflush(stdout);
	printf("        STAGE-1: NAO & NPA (for down-spin)       \n");fflush(stdout);
	printf(" *********************************************** \n");fflush(stdout);
	printf("\n");fflush(stdout);

      }
#endif

      /***  Maximum number of L (angular momentum)  ***/

      Lmax = 0;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	if (Lmax < Spe_MaxL_Basis[wan1]) Lmax = Spe_MaxL_Basis[wan1];
      }

      printf(" Maximum number of L = %d \n",Lmax);fflush(stdout);

      /***  Maximum number of multiplicity for magnetic momentum  ***/

      Mulmax = 0;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  if (Mulmax < Spe_Num_Basis[wan1][L1]) Mulmax = Spe_Num_Basis[wan1][L1];
	} /* L1 */
      } /* Gc_AN */

      printf(" Maximum number of M = %d \n",Mulmax);fflush(stdout);

      /***  Total number of basis sets (= size of full matrix)  ***/

      SizeMat = 0; 
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	SizeMat += Spe_Total_NO[wan1];
      }

      Posi2Gatm = (int*)malloc(sizeof(int)*SizeMat);

      num = 0;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (i=0; i<Spe_Total_NO[wan1]; i++){
	  Posi2Gatm[num] = Gc_AN;
	  num++;
	}
      }


      printf(" Size of full matrix = %d \n\n",SizeMat);fflush(stdout);

      /***  Allocation of arrays (1)  ***/

      Leng_a = (int*)malloc(sizeof(int)*(atomnum+1));

      Leng_al = (int**)malloc(sizeof(int*)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
	Leng_al[i] = (int*)malloc(sizeof(int)*(Lmax+1));
	for (j=0; j<=Lmax; j++){
	  Leng_al[i][j] = 0;
	}
      }

      Leng_alm = (int**)malloc(sizeof(int*)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
	Leng_alm[i] = (int*)malloc(sizeof(int)*(Lmax+1));
	for (j=0; j<=Lmax; j++){
	  Leng_alm[i][j] = 0;
	}
      }

      Posi_a = (int*)malloc(sizeof(int)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
	Posi_a[i] = 0;
      }

      Posi_al = (int**)malloc(sizeof(int*)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
	Posi_al[i] = (int*)malloc(sizeof(int)*(Lmax+1));
	for (j=0; j<=Lmax; j++){
	  Posi_al[i][j] = 0;
	}
      }

      Posi_alm = (int***)malloc(sizeof(int**)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
	Posi_alm[i] = (int**)malloc(sizeof(int*)*(Lmax+1));
	for (j=0; j<=Lmax; j++){
	  Posi_alm[i][j] = (int*)malloc(sizeof(int)*(Lmax*2+2));
	  for (k=0; k<=Lmax*2+1; k++){
	    Posi_alm[i][j][k] = 0;
	  }
	}
      }

      Num_NMB1 = (int**)malloc(sizeof(int*)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
	Num_NMB1[i] = (int*)malloc(sizeof(int)*(Lmax+1));
	for (j=0; j<=Lmax; j++){
	  Num_NMB1[i][j] = 0;
	}
      }

      Num_NRB1 = (int**)malloc(sizeof(int*)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
	Num_NRB1[i] = (int*)malloc(sizeof(int)*(Lmax+1));
	for (j=0; j<=Lmax; j++){
	  Num_NRB1[i][j] = 0;
	}
      }

      S_full = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	S_full[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  S_full[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    S_full[spin][i][j] = 0.0;
	  }
	}
      }

      S_full0 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	S_full0[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  S_full0[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    S_full0[spin][i][j] = 0.0;
	  }
	}
      }

      D_full = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	D_full[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  D_full[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    D_full[spin][i][j] = 0.0;
	  }
	}
      }

      P_full = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	P_full[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  P_full[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    P_full[spin][i][j] = 0.0;
	  }
	}
      }

      P_full0 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	P_full0[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  P_full0[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    P_full0[spin][i][j] = 0.0;
	  }
	}
      }

      H_full = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	H_full[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  H_full[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    H_full[spin][i][j] = 0.0;
	  }
	}
      }

      H_full0 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	H_full0[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  H_full0[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    H_full0[spin][i][j] = 0.0;
	  }
	}
      }

      T_full = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	T_full[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  T_full[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    T_full[spin][i][j] = 0.0;
	  }
	}
      }

      W_full = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	W_full[spin] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  W_full[spin][i] = 0.0;
	}
      }

      N_tran1 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	N_tran1[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  N_tran1[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    N_tran1[spin][i][j] = 0.0;
	  }
	}
      }

      N_tran2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	N_tran2[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  N_tran2[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    N_tran2[spin][i][j] = 0.0;
	  }
	}
      }
      /*
	N_diag = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	N_diag[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	N_diag[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	N_diag[spin][i][j] = 0.0;
	}
	}
	}
      */
      /*
	N_ryd = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	N_ryd[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	N_ryd[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	N_ryd[spin][i][j] = 0.0;
	}
	}
	}
      */
      /*
	N_red = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	N_red[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	N_red[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	N_red[spin][i][j] = 0.0;
	}
	}
	}
      */
      /*
	O_schm = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	O_schm[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	O_schm[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	O_schm[spin][i][j] = 0.0;
	}
	}
	}
      */
      /*
	O_schm2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	O_schm2[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	O_schm2[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	O_schm2[spin][i][j] = 0.0;
	}
	}
	}
      */
      /*
	Ow_NMB = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	Ow_NMB[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	Ow_NMB[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	Ow_NMB[spin][i][j] = 0.0;
	}
	}
	}
      */
      /*
	Ow_NRB = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	Ow_NRB[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	Ow_NRB[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	Ow_NRB[spin][i][j] = 0.0;
	}
	}
	}
      */
      /*
	O_sym = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	O_sym[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	O_sym[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	O_sym[spin][i][j] = 0.0;
	}
	}
	}
      */
      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  N_tran1[spin][i][i]  = 1.0;
	  N_tran2[spin][i][i]  = 1.0;
	  /*   N_diag[spin][i][i]  = 1.0; */
	  /*   N_ryd[spin][i][i]   = 1.0; */
	  /*   N_red[spin][i][i]   = 1.0; */
	  /*   O_schm[spin][i][i]  = 1.0; */
	  /*   O_schm2[spin][i][i] = 1.0; */
	  /*   Ow_NMB[spin][i][i]  = 1.0; */
	  /*   Ow_NRB[spin][i][i]  = 1.0; */
	  /*   O_sym[spin][i][i]   = 1.0; */
	  T_full[spin][i][i]  = 1.0;
	}
      }

      temp_M1 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	temp_M1[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  temp_M1[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    temp_M1[spin][i][j] = 0.0;
	  }
	}
      }

      temp_M2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	temp_M2[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  temp_M2[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    temp_M2[spin][i][j] = 0.0;
	  }
	}
      }

      temp_M3 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	temp_M3[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  temp_M3[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    temp_M3[spin][i][j] = 0.0;
	  }
	}
      }

      temp_M4 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	temp_M4[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	  temp_M4[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	  for (j=0; j<=SizeMat; j++){
	    temp_M4[spin][i][j] = 0.0;
	  }
	}
      }
      /*
	temp_M5 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	temp_M5[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	temp_M5[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	temp_M5[spin][i][j] = 0.0;
	}
	}
	}
      */
      /*
	temp_M6 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	temp_M6[spin] = (double**)malloc(sizeof(double*)*(SizeMat+1));
	for (i=0; i<=SizeMat; i++){
	temp_M6[spin][i] = (double*)malloc(sizeof(double)*(SizeMat+1));
	for (j=0; j<=SizeMat; j++){
	temp_M6[spin][i][j] = 0.0;
	}
	}
	}
      */
      temp_V1 = (double*)malloc(sizeof(double)*(SizeMat*SizeMat+1));
      for (j=0; j<=(SizeMat*SizeMat); j++){
	temp_V1[j] = 0.0;
      }

      temp_V2 = (double*)malloc(sizeof(double)*(SizeMat*SizeMat+1));
      for (j=0; j<=(SizeMat*SizeMat); j++){
	temp_V2[j] = 0.0;
      }

      temp_V3 = (double*)malloc(sizeof(double)*(SizeMat*SizeMat+1));
      for (j=0; j<=(SizeMat*SizeMat); j++){
	temp_V3[j] = 0.0;
      }


      /**************************************************************************
   1. Setting density & overlap matrices P & S
      **************************************************************************/

      if (myid == Host_ID) printf("<< 1 >> Setting density & overlap matrices P & S \n");
      if (measure_time==1) dtime(&Stime1);

      /***  Table functions of atomic positions on full matrix  ***/

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	Leng_a[Gc_AN] = Spe_Total_NO[wan1];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  Leng_alm[Gc_AN][L1] = Spe_Num_Basis[wan1][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    Leng_al[Gc_AN][L1] += Leng_alm[Gc_AN][L1];
	  } /* M1 */
	} /* L1 */
      } /* Gc_AN */

      if (outputlev==1){
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  printf(" Leng_a[%d] = %d \n",Gc_AN,Leng_a[Gc_AN]);fflush(stdout);
	} /* Gc_AN */

	printf("\n");fflush(stdout);

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  wan1 = WhatSpecies[Gc_AN];
	  for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	    printf(" Leng_al[%d][%d] = %d \n",Gc_AN,L1,Leng_al[Gc_AN][L1]);fflush(stdout);
	  } /* L1 */
	} /* Gc_AN */

	printf("\n");fflush(stdout);

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  wan1 = WhatSpecies[Gc_AN];
	  for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	    printf(" Leng_alm[%d][%d] = %d \n",Gc_AN,L1,Leng_alm[Gc_AN][L1]);fflush(stdout);
	  } /* L1 */
	} /* Gc_AN */

	printf("\n");fflush(stdout);
      }

      MaxLeng_a   = 0;
      MaxLeng_al  = 0;
      MaxLeng_alm = 0;

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	if (MaxLeng_a < Leng_a[Gc_AN]) MaxLeng_a = Leng_a[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  if (MaxLeng_al  < Leng_al[Gc_AN][L1])  MaxLeng_al  = Leng_al[Gc_AN][L1];
	  if (MaxLeng_alm < Leng_alm[Gc_AN][L1]) MaxLeng_alm = Leng_alm[Gc_AN][L1];
	} /* L1 */
      } /* Gc_AN */

      if (outputlev==1){
	printf(" MaxLeng_a   = %d \n",MaxLeng_a  );fflush(stdout);
	printf(" MaxLeng_al  = %d \n",MaxLeng_al );fflush(stdout);
	printf(" MaxLeng_alm = %d \n",MaxLeng_alm);fflush(stdout);
      }

      tmp_num1 =0;
      tmp_num2 =0;
      tmp_num3 =0;

      for (Gc_AN=2; Gc_AN<=atomnum; Gc_AN++){
	tmp_num1 += Leng_a[Gc_AN-1];
	Posi_a[Gc_AN] = tmp_num1;
      } /* Gc_AN */

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  tmp_num2 += Leng_al[Gc_AN][L1];
	  if (L1 != Spe_MaxL_Basis[wan1])                     Posi_al[Gc_AN][L1+1] = tmp_num2;
	  if (L1 == Spe_MaxL_Basis[wan1] && Gc_AN != atomnum) Posi_al[Gc_AN+1][0]  = tmp_num2;
	} /* L1 */
      } /* Gc_AN */

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  for (M1=1; M1<=2*L1+1; M1++){
	    tmp_num3 += Leng_alm[Gc_AN][L1];
	    if (M1 != 2*L1+1)                               Posi_alm[Gc_AN][L1][M1+1] = tmp_num3;
	    if (M1 == 2*L1+1 && L1 != Spe_MaxL_Basis[wan1]) Posi_alm[Gc_AN][L1+1][1]  = tmp_num3;
	    if (M1 == 2*L1+1 && L1 == Spe_MaxL_Basis[wan1]
		&& Gc_AN != atomnum)           Posi_alm[Gc_AN+1][0][1]   = tmp_num3;
	  }
	}
      }

      if (outputlev==1){
	printf("\n ### Check Positions on Matrices ###\n");fflush(stdout);

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  printf(" Posi_a[%d] = %d \n",Gc_AN,Posi_a[Gc_AN]);fflush(stdout);
	} /* Gc_AN */

	printf("\n");fflush(stdout);

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  wan1 = WhatSpecies[Gc_AN];
	  for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	    printf(" Posi_al[%d][%d] = %d \n",Gc_AN,L1,Posi_al[Gc_AN][L1]);fflush(stdout);
	  } /* L1 */
	} /* Gc_AN */

	printf("\n");fflush(stdout);

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  wan1 = WhatSpecies[Gc_AN];
	  for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	    for (M1=1; M1<=2*L1+1; M1++){
	      printf(" Posi_alm[%d][%d][%d] = %d \n",Gc_AN,L1,M1,Posi_alm[Gc_AN][L1][M1]);fflush(stdout);
	    } /* M1 */
	  } /* L1 */
	} /* Gc_AN */

	printf("\n");fflush(stdout);
      }

      /***  Bond-order matrix D, overlap matrix S, Hamiltonian matrix H  ***/

      for (ID=0; ID<numprocs; ID++){
	num = 0;
	for (k=0; k<MatomN[ID]; k++){
	  Gc_AN = ID2G[ID][k];
	  wan1 = WhatSpecies[Gc_AN];
	  mtps1 = Posi_a[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2  = WhatSpecies[Gh_AN];
	    mtps2 = Posi_a[Gh_AN];
	    for (i=0; i<Spe_Total_NO[wan1]; i++){
	      for (j=0; j<Spe_Total_NO[wan2]; j++){
		S_full[0][mtps1+i][mtps2+j] = Tmp_Vec1[ID2posi[ID]+num];
		D_full[0][mtps1+i][mtps2+j] = Tmp_Vec2[ID2posi[ID]+num]*(2.0-(double)SpinP_switch);
		H_full[0][mtps1+i][mtps2+j] = Tmp_Vec3[ID2posi[ID]+num];
		num++;
	      } /* j */
	    } /* i */
	  } /* h_AN */
	}
      }

      int SizeMat0;
      SizeMat0 = 18;

      if (outputlev==1){
	printf("### Overlap Matrix (full size) ###\n");fflush(stdout);
	for (i=0; i<SizeMat0; i++){
	  for (j=0; j<SizeMat0; j++){
	    printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	  } /* j */
	  printf("\n");fflush(stdout);
	} /* i */

	printf("\n");fflush(stdout);

	printf("### Bond-Order Matrix (full size) ###\n");fflush(stdout);
	for (i=0; i<SizeMat0; i++){
	  for (j=0; j<SizeMat0; j++){
	    printf("%9.5f",D_full[0][i][j]);fflush(stdout);
	  } /* j */
	  printf("\n");fflush(stdout);
	} /* i */

	printf("\n");fflush(stdout);

	printf("### Hamiltonian Matrix (full size) ###\n");fflush(stdout);
	for (i=0; i<SizeMat0; i++){
	  for (j=0; j<SizeMat0; j++){
	    printf("%9.5f",H_full[0][i][j]);fflush(stdout);
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
        tmp_ele1 += D_full[0][i][k] * S_full[0][k][j];
	} 
	temp_M1[0][i][j] = tmp_ele1;
	} 
	} 

	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	tmp_ele1 = 0.0;
	for (k=0; k<SizeMat; k++){
	tmp_ele1 += S_full[0][i][k] * temp_M1[0][k][j];
	} 
	P_full[0][i][j] = tmp_ele1;
	} 
	} 
      */

      /* BLAS */
      if(1>0){
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){ 
	    temp_V1[tnum] = S_full[0][i][j];
	    temp_V2[tnum] = D_full[0][i][j];
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
	    P_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}
      }
      else{
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_M2[0][j][i] = S_full[0][i][j];
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele0 = 0.0;
	    tmp_ele1 = 0.0;
	    tmp_ele2 = 0.0;
	    tmp_ele3 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 += D_full[0][i][k]   * temp_M2[0][j][k];
	      tmp_ele1 += D_full[0][i][k+1] * temp_M2[0][j][k+1];
	      tmp_ele2 += D_full[0][i][k+2] * temp_M2[0][j][k+2];
	      tmp_ele3 += D_full[0][i][k+3] * temp_M2[0][j][k+3];
	    }
	    temp_M1[0][j][i] = tmp_ele0 + tmp_ele1 + tmp_ele2 + tmp_ele3;
	  }
	}

	is = SizeMat - SizeMat%4;

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele0 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 += D_full[0][i][k] * temp_M2[0][j][k];
	    }
	    temp_M1[0][j][i] += tmp_ele0;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele0 = 0.0;
	    tmp_ele1 = 0.0;
	    tmp_ele2 = 0.0;
	    tmp_ele3 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 += S_full[0][i][k]   * temp_M1[0][j][k];
	      tmp_ele1 += S_full[0][i][k+1] * temp_M1[0][j][k+1];
	      tmp_ele2 += S_full[0][i][k+2] * temp_M1[0][j][k+2];
	      tmp_ele3 += S_full[0][i][k+3] * temp_M1[0][j][k+3];
	    }
	    P_full[0][i][j] = tmp_ele0 + tmp_ele1 + tmp_ele2 + tmp_ele3;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele0 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 += S_full[0][i][k] * temp_M1[0][j][k];
	    }
	    P_full[0][i][j] += tmp_ele0;
	  }
	}
      }

      if (outputlev==1){
	printf("### Density Matrix (full size) ###\n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	  } /* j */
	  printf("\n");fflush(stdout);
	} /* i */

	printf("\n");fflush(stdout);
      }

      /***  Rearrangement of overlap, density & Hamiltonian matrices  ***/
      /*
	tmp_num3 = -1;
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	posi1 = Posi_al[Gc_AN][L1];
	tmp_num1 = Spe_Num_Basis[wan1][L1];
	for (M1=1; M1<=2*L1+1; M1++){
	for (k=1; k<=tmp_num1; k++){
	tmp_num3 += 1;
        if(1<0){
	printf("L1=%d, M1=%d, posi1=%d, k=%d, tmp_num3=%d, %4d\n",
	L1,M1,posi1,k,tmp_num3,posi1+(2*L1+1)*(k-1)+(M1-1));fflush(stdout); 
        }
	for (j=0; j<SizeMat; j++){       
	temp_M1[0][j][tmp_num3] = S_full[0][j][posi1+(2*L1+1)*(k-1)+(M1-1)];
	temp_M3[0][j][tmp_num3] = P_full[0][j][posi1+(2*L1+1)*(k-1)+(M1-1)]; 
	temp_M5[0][j][tmp_num3] = H_full[0][j][posi1+(2*L1+1)*(k-1)+(M1-1)];
	}
	}
	}
	}
	}

	tmp_num3 = -1;
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	posi1 = Posi_al[Gc_AN][L1];
	tmp_num1 = Spe_Num_Basis[wan1][L1];
	for (M1=1; M1<=2*L1+1; M1++){
	for (k=1; k<=tmp_num1; k++){
	tmp_num3 += 1;
        if(1<0){
	printf("L1=%d, M1=%d, posi1=%d, k=%d, tmp_num3=%d, %4d\n",
	L1,M1,posi1,k,tmp_num3,posi1+(2*L1+1)*(k-1)+(M1-1));fflush(stdout);
        }
	for (j=0; j<SizeMat; j++){
	temp_M2[0][tmp_num3][j] = temp_M1[0][posi1+(2*L1+1)*(k-1)+(M1-1)][j];
	temp_M4[0][tmp_num3][j] = temp_M3[0][posi1+(2*L1+1)*(k-1)+(M1-1)][j];
	temp_M6[0][tmp_num3][j] = temp_M5[0][posi1+(2*L1+1)*(k-1)+(M1-1)][j];
	}
	}
	}
	}
	}

	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	S_full[0][i][j] = temp_M2[0][i][j];
	P_full[0][i][j] = temp_M4[0][i][j];
	H_full[0][i][j] = temp_M6[0][i][j];
	S_full0[0][i][j] = S_full[0][i][j];
	P_full0[0][i][j] = P_full[0][i][j];
	H_full0[0][i][j] = H_full[0][i][j];
	}
	}
      */
      /***  Overlap  Matrix  ***/

      tmp_num3 = -1;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  posi1 = Posi_al[Gc_AN][L1];
	  tmp_num1 = Spe_Num_Basis[wan1][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    for (k=1; k<=tmp_num1; k++){
	      tmp_num3 += 1;
	      for (j=0; j<SizeMat; j++){
		temp_M1[0][j][tmp_num3] = S_full[0][j][posi1+(2*L1+1)*(k-1)+(M1-1)];
	      }
	    }
	  }
	}
      }

      tmp_num3 = -1;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  posi1 = Posi_al[Gc_AN][L1];
	  tmp_num1 = Spe_Num_Basis[wan1][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    for (k=1; k<=tmp_num1; k++){
	      tmp_num3 += 1;
	      for (j=0; j<SizeMat; j++){
		temp_M2[0][tmp_num3][j] = temp_M1[0][posi1+(2*L1+1)*(k-1)+(M1-1)][j];
	      }
	    }
	  }
	}
      }

      for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	  S_full[0][i][j] = temp_M2[0][i][j];
	  S_full0[0][i][j] = S_full[0][i][j];
	}
      }


      /***  Density Matrix  ***/

      tmp_num3 = -1;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  posi1 = Posi_al[Gc_AN][L1];
	  tmp_num1 = Spe_Num_Basis[wan1][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    for (k=1; k<=tmp_num1; k++){
	      tmp_num3 += 1;
	      for (j=0; j<SizeMat; j++){
		temp_M1[0][j][tmp_num3] = P_full[0][j][posi1+(2*L1+1)*(k-1)+(M1-1)];
	      }
	    }
	  }
	}
      }

      tmp_num3 = -1;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  posi1 = Posi_al[Gc_AN][L1];
	  tmp_num1 = Spe_Num_Basis[wan1][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    for (k=1; k<=tmp_num1; k++){
	      tmp_num3 += 1;
	      for (j=0; j<SizeMat; j++){
		temp_M2[0][tmp_num3][j] = temp_M1[0][posi1+(2*L1+1)*(k-1)+(M1-1)][j];
	      }
	    }
	  }
	}
      }

      for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	  P_full[0][i][j] = temp_M2[0][i][j];
	  P_full0[0][i][j] = P_full[0][i][j];
	}
      }


      /***  Hamiltonian Matrix  ***/

      tmp_num3 = -1;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  posi1 = Posi_al[Gc_AN][L1];
	  tmp_num1 = Spe_Num_Basis[wan1][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    for (k=1; k<=tmp_num1; k++){
	      tmp_num3 += 1;
	      for (j=0; j<SizeMat; j++){
		temp_M1[0][j][tmp_num3] = H_full[0][j][posi1+(2*L1+1)*(k-1)+(M1-1)];
	      }
	    }
	  }
	}
      }

      tmp_num3 = -1;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  posi1 = Posi_al[Gc_AN][L1];
	  tmp_num1 = Spe_Num_Basis[wan1][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    for (k=1; k<=tmp_num1; k++){
	      tmp_num3 += 1;
	      for (j=0; j<SizeMat; j++){
		temp_M2[0][tmp_num3][j] = temp_M1[0][posi1+(2*L1+1)*(k-1)+(M1-1)][j];
	      }
	    }
	  }
	}
      }

      for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	  H_full[0][i][j] = temp_M2[0][i][j];
	  H_full0[0][i][j] = H_full[0][i][j];
	}
      }

      SizeMat0 = 18;

      if (outputlev==1){
	printf("### Overlap Matrix (full size) 2 ###\n");fflush(stdout);
	for (i=0; i<SizeMat0; i++){
	  for (j=0; j<SizeMat0; j++){
	    printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	  } /* j */
	  printf("\n");fflush(stdout);
	} /* i */

	printf("\n");

	printf("### Density Matrix (full size) 2 ###\n");fflush(stdout);
	for (i=0; i<SizeMat0; i++){
	  for (j=0; j<SizeMat0; j++){
	    printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	  } /* j */
	  printf("\n");fflush(stdout);
	} /* i */

	printf("\n");

	printf("### Hamiltonian Matrix (full size) 2 ###\n");fflush(stdout);
	for (i=0; i<SizeMat0; i++){
	  for (j=0; j<SizeMat0; j++){
	    printf("%9.5f",H_full[0][i][j]);fflush(stdout);
	  } /* j */
	  printf("\n");fflush(stdout);
	} /* i */

	printf("\n");
      }


      /***  Allocation of arrays (2)  ***/

      P_alm = (double******)malloc(sizeof(double*****)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	P_alm[spin] = (double*****)malloc(sizeof(double****)*(atomnum+1));
	for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
	  P_alm[spin][Gc_AN] = (double****)malloc(sizeof(double***)*(Lmax+1));
	  for (L1=0; L1<=Lmax; L1++){
	    P_alm[spin][Gc_AN][L1] = (double***)malloc(sizeof(double**)*(2*Lmax+2));
	    for (M1=0; M1<=2*Lmax+1; M1++){
	      P_alm[spin][Gc_AN][L1][M1] = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
	      for (i=0; i<=MaxLeng_alm; i++){
		P_alm[spin][Gc_AN][L1][M1][i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
		for (j=0; j<=MaxLeng_alm; j++){
		  P_alm[spin][Gc_AN][L1][M1][i][j] = 0.0;
		}
	      }
	    }
	  }
	}
      }

      S_alm = (double******)malloc(sizeof(double*****)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	S_alm[spin] = (double*****)malloc(sizeof(double****)*(atomnum+1));
	for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
	  S_alm[spin][Gc_AN] = (double****)malloc(sizeof(double***)*(Lmax+1));
	  for (L1=0; L1<=Lmax; L1++){
	    S_alm[spin][Gc_AN][L1] = (double***)malloc(sizeof(double**)*(2*Lmax+2));
	    for (M1=0; M1<=2*Lmax+1; M1++){
	      S_alm[spin][Gc_AN][L1][M1] = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
	      for (i=0; i<=MaxLeng_alm; i++){
		S_alm[spin][Gc_AN][L1][M1][i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
		for (j=0; j<=MaxLeng_alm; j++){
		  S_alm[spin][Gc_AN][L1][M1][i][j] = 0.0;
		}
	      }
	    }
	  }
	}
      }

      /***  Atomic & anglar- & magetic-momentum block matrices  ***/

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  leng1 = Leng_alm[Gc_AN][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    posi1 = Posi_alm[Gc_AN][L1][M1];
	    for (i=0; i<leng1; i++){
	      for (j=0; j<leng1; j++){
		P_alm[0][Gc_AN][L1][M1][i][j] = P_full[0][i+posi1][j+posi1];
		S_alm[0][Gc_AN][L1][M1][i][j] = S_full[0][i+posi1][j+posi1];
	      } /* j */
	    } /* i */
	  } /* M1 */
	} /* L1 */
      } /* Gc_AN */

      if (measure_time==1){
	dtime(&Etime1);
	time1 = Etime1 - Stime1;
      }

      /**************************************************************************
   2. Intraatomic orthogonalization
      **************************************************************************/

      if (myid == Host_ID) printf("<< 2 >> Intraatomic orthogonalization \n");

      /***  Allocation of arrays (3)  ***/

      P_alm2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	P_alm2[spin] = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
	for (i=0; i<=MaxLeng_alm; i++){
	  P_alm2[spin][i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
	  for (j=0; j<=MaxLeng_alm; j++){
	    P_alm2[spin][i][j] = 0.0;
	  }
	}
      }

      S_alm2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	S_alm2[spin] = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
	for (i=0; i<=MaxLeng_alm; i++){
	  S_alm2[spin][i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
	  for (j=0; j<=MaxLeng_alm; j++){
	    S_alm2[spin][i][j] = 0.0;
	  }
	}
      }

      N_alm2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	N_alm2[spin] = (double**)malloc(sizeof(double*)*(MaxLeng_alm+1));
	for (i=0; i<=MaxLeng_alm; i++){
	  N_alm2[spin][i] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
	  for (j=0; j<=MaxLeng_alm; j++){
	    N_alm2[spin][i][j] = 0.0;
	  }
	}
      }

      W_alm2 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	W_alm2[spin] = (double*)malloc(sizeof(double)*(MaxLeng_alm+1));
	for (i=0; i<=MaxLeng_alm; i++){
	  W_alm2[spin][i] = 0.0;
	}
      }

      NRB_posi3= (int*)malloc(sizeof(int)*(MaxLeng_alm+1));
      for (i=0; i<=MaxLeng_alm; i++){
	NRB_posi3[i] = 0;
      }

      M_or_R1= (int***)malloc(sizeof(int**)*(atomnum+1));
      for (i=0; i<=atomnum; i++){
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

      if (myid == Host_ID) printf("<< 2-2 >> Symmetry averaging of P & S \n");
      if (measure_time==1) dtime(&Stime1);

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  leng1 = Leng_alm[Gc_AN][L1];
	  for (i=0; i<leng1; i++){
	    for (j=0; j<leng1; j++){
	      tmp_ele1 = 0.0;
	      tmp_ele2 = 0.0;
	      for (M1=1; M1<=L1*2+1; M1++){
		tmp_ele1 += P_alm[0][Gc_AN][L1][M1][i][j];
		tmp_ele2 += S_alm[0][Gc_AN][L1][M1][i][j];
	      } /* M1 */
	      tmp_ele1 /= L1*2.0+1.0;
	      tmp_ele2 /= L1*2.0+1.0;
	      for (M1=1; M1<=L1*2+1; M1++){
		P_alm[0][Gc_AN][L1][M1][i][j] = tmp_ele1;
		S_alm[0][Gc_AN][L1][M1][i][j] = tmp_ele2;
	      } /* M1 */
	    } /* j */
	  } /* i */
	} /* L1 */
      } /* Gc_AN */

      if (measure_time==1){
	dtime(&Etime1);
	time2 = Etime1 - Stime1;
      }

      /*******************************
         2-3. Formation of pre-NAOs
      *******************************/

      if (myid == Host_ID) printf("<< 2-3 >> Formation of pre-NAOs \n");
      if (measure_time==1) dtime(&Stime1);

      /***  Atom & angular-momentum loop  ***/

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  Mmax1 = Leng_alm[Gc_AN][L1];

	  /***  Transferring of matrices (1)  ***/

	  for (i=0; i<Mmax1; i++){
	    for (j=0; j<Mmax1; j++){
	      P_alm2[0][i+1][j+1] = P_alm[0][Gc_AN][L1][1][i][j];
	      S_alm2[0][i+1][j+1] = S_alm[0][Gc_AN][L1][1][i][j];
	    } /* j */
	  } /* i */

	  if (outputlev==1){
	    printf("### S_alm2[%d][%d] ### \n",Gc_AN,L1);fflush(stdout);
	    for (i=0; i<Mmax1; i++){
	      for (j=0; j<Mmax1; j++){
		printf("%9.5f",S_alm2[0][i+1][j+1]);fflush(stdout);
	      } /* j */
	      printf("\n");fflush(stdout);
	    } /* j */

	    printf("\n");fflush(stdout);

	    printf("### P_alm2[%d][%d] ### \n",Gc_AN,L1);fflush(stdout);
	    for (i=0; i<Mmax1; i++){
	      for (j=0; j<Mmax1; j++){
		printf("%9.5f",P_alm2[0][i+1][j+1]);fflush(stdout);
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

	  Eigen_lapack(S_alm2[0],W_alm2[0],Mmax1,Mmax1);

	  if (outputlev==1){
	    for (k=1; k<=Mmax1; k++){
	      printf("k W %2d %15.12f\n",k,W_alm2[0][k]);
	    }
	  }

	  /* check ill-conditioned eigenvalues */

	  for (k=1; k<=Mmax1; k++){
	    if (W_alm2[0][k]<1.0e-14) {
	      printf("Found ill-conditioned eigenvalues (1)\n");
	      printf("Stopped calculation\n");
	      exit(1);
	    }
	  }

	  for (k=1; k<=Mmax1; k++){
	    temp_V1[k] = 1.0/sqrt(W_alm2[0][k]);
	  }

	  /***  Calculations of eigenvalues  ***/

	  for (i=1; i<=Mmax1; i++){
	    for (j=1; j<=Mmax1; j++){
	      sum = 0.0;
	      for (k=1; k<=Mmax1; k++){
		sum = sum + P_alm2[0][i][k]*S_alm2[0][k][j]*temp_V1[j];
	      }
	      N_alm2[0][i][j] = sum;
	    }
	  }

	  for (i=1; i<=Mmax1; i++){
	    for (j=1; j<=Mmax1; j++){
	      sum = 0.0;
	      for (k=1; k<=Mmax1; k++){
		sum = sum + temp_V1[i]*S_alm2[0][k][i]*N_alm2[0][k][j];   
	      }
	      temp_M1[0][i][j] = sum;
	    }
	  }

	  if (outputlev==1){
	    printf("\n ##### TEST 1 ##### \n");
	    for (i=1; i<=Mmax1; i++){
	      for (j=1; j<=Mmax1; j++){
		printf("%9.5f",temp_M1[0][i][j]);
	      }
	      printf("\n");
	    }
	  } 

	  Eigen_lapack(temp_M1[0],W_alm2[0],Mmax1,Mmax1);

	  if (outputlev==1){
	    printf("\n ##### TEST 2 ##### \n");
	    for (k=1; k<=Mmax1; k++){
	      printf("k W %2d %9.5f\n",k,W_alm2[0][k]);
	    }

	    printf("\n ##### TEST 3 ##### \n");
	    for (i=1; i<=Mmax1; i++){
	      for (j=1; j<=Mmax1; j++){
		printf("%9.5f",temp_M1[0][i][j]);
	      }
	      printf("\n");
	    }
	    printf("\n");
	  }

	  /***  Transformation to the original eigen vectors  ***/

	  for (i=1; i<=Mmax1; i++){
	    for (j=1; j<=Mmax1; j++){
	      N_alm2[0][i][j] = 0.0;
	    }
	  }

	  for (i=1; i<=Mmax1; i++){
	    for (j=1; j<=Mmax1; j++){
	      sum = 0.0;
	      for (k=1; k<=Mmax1; k++){
		sum = sum + S_alm2[0][i][k]*temp_V1[k]*temp_M1[0][k][j];
	      }
	      N_alm2[0][i][j] = sum;
	    }
	  }

	  /* printing out eigenvalues and the eigenvectors */

	  if (outputlev==1){
	    for (i=1; i<=Mmax1; i++){
	      printf("%ith eigenvalue of HC=eSC: %15.12f\n",i,W_alm2[0][i]);
	    }

	    for (i=1; i<=Mmax1; i++){
	      printf("%ith eigenvector: ",i);
	      printf("{");
	      for (j=1; j<=Mmax1; j++){
		printf("%15.12f,",N_alm2[0][i][j]);
	      }
	      printf("}\n");
	    }

	    printf("\n");
	  }

	  /***  Selection of NMB & NRB orbitals  ***/

	  for (i=1; i<=Mmax1; i++){
	    if (W_alm2[0][i] >= thredens) {M_or_R1[Gc_AN][L1][i-1] = 1;}
	    else                          {M_or_R1[Gc_AN][L1][i-1] = 0;}
	  } /* i */

	  leng1 = Leng_alm[Gc_AN][L1];

	  for (M1=1; M1<=L1*2+1; M1++){
	    posi1 = Posi_alm[Gc_AN][L1][M1];
	    for (i=0; i<leng1; i++){
	      W_full[0][posi1+i] = W_alm2[0][i+1];
	      if (W_alm2[0][i+1] >= thredens) {M_or_R2[posi1+i] = 1;}
	      else                            {M_or_R2[posi1+i] = 0;}
	      for (j=0; j<leng1; j++){
		N_tran1[0][posi1+i][posi1+j] = N_alm2[0][i+1][j+1];
	      } /* j */
	    } /* i */
	  } /* M1 */

	} /* L1 */
      } /* Gc_AN */

      if (outputlev==1){
	printf("### T_full (1) (= N_diag) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",N_tran1[0][i][j]);fflush(stdout);
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

      if (myid == Host_ID) printf("<< 2-4 >> Construction of S & P for pre-NAOs \n");
      if (measure_time==1) dtime(&Stime1);
      /*
	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full[0][i][k] * N_diag[0][k][j];
        tmp_ele2 += P_full[0][i][k] * N_diag[0][k][j];
	} 
        temp_M1[0][i][j] = tmp_ele1;
        temp_M2[0][i][j] = tmp_ele2;
	} 
	} 

	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += N_diag[0][k][i] * temp_M1[0][k][j];
        tmp_ele2 += N_diag[0][k][i] * temp_M2[0][k][j];
	} 
        S_full[0][i][j] = tmp_ele1;
        P_full[0][i][j] = tmp_ele2;
	} 
	} 
      */

      /* BLAS */
      /*  N_diag * S * N_diag  */
      if(1>0){
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V1[tnum] = N_tran1[0][i][j];
	    temp_V2[tnum] = S_full[0][i][j];
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
	    S_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  N_diag * P * N_diag  */
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = P_full[0][i][j];
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
	    P_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}
      }
      else{
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_M3[0][j][i] = N_tran1[0][i][j];
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M3[0][j][k];
	      tmp_ele1 = temp_M3[0][j][k+1];
	      tmp_ele2 = temp_M3[0][j][k+2];
	      tmp_ele3 = temp_M3[0][j][k+3];

	      tmp_ele00 += S_full[0][i][k]   * tmp_ele0;
	      tmp_ele01 += S_full[0][i][k+1] * tmp_ele1;
	      tmp_ele02 += S_full[0][i][k+2] * tmp_ele2;
	      tmp_ele03 += S_full[0][i][k+3] * tmp_ele3;

	      tmp_ele10 += P_full[0][i][k]   * tmp_ele0;
	      tmp_ele11 += P_full[0][i][k+1] * tmp_ele1;
	      tmp_ele12 += P_full[0][i][k+2] * tmp_ele2;
	      tmp_ele13 += P_full[0][i][k+3] * tmp_ele3;
	    }
	    temp_M1[0][j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    temp_M2[0][j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	  }
	}

	is = SizeMat - SizeMat%4;

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M3[0][j][k];
	      tmp_ele00 += S_full[0][i][k] * tmp_ele0;
	      tmp_ele10 += P_full[0][i][k] * tmp_ele0;
	    }
	    temp_M1[0][j][i] += tmp_ele00;
	    temp_M2[0][j][i] += tmp_ele10;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M3[0][i][k];
	      tmp_ele1 = temp_M3[0][i][k+1];
	      tmp_ele2 = temp_M3[0][i][k+2];
	      tmp_ele3 = temp_M3[0][i][k+3];

	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele01 += tmp_ele1 * temp_M1[0][j][k+1];
	      tmp_ele02 += tmp_ele2 * temp_M1[0][j][k+2];
	      tmp_ele03 += tmp_ele3 * temp_M1[0][j][k+3];

	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	      tmp_ele11 += tmp_ele1 * temp_M2[0][j][k+1];
	      tmp_ele12 += tmp_ele2 * temp_M2[0][j][k+2];
	      tmp_ele13 += tmp_ele3 * temp_M2[0][j][k+3];
	    }
	    S_full[0][i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    P_full[0][i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M3[0][i][k];
	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	    }
	    S_full[0][i][j] += tmp_ele00;
	    P_full[0][i][j] += tmp_ele10;
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
	    printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### P_full (1) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### W_full (1) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  printf("%9.5f \n",W_full[0][i]);fflush(stdout);
	}
      }

      /***  Allocation of arrays (4)  ***/

      S_NMB = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	S_NMB[spin] = (double**)malloc(sizeof(double*)*(Num_NMB+1));
	for (i=0; i<=Num_NMB; i++){
	  S_NMB[spin][i] = (double*)malloc(sizeof(double)*(Num_NMB+1));
	  for (j=0; j<=Num_NMB; j++){
	    S_NMB[spin][i][j] = 0.0;
	  }
	}
      }

      W_NMB = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	W_NMB[spin] = (double*)malloc(sizeof(double)*(Num_NMB+1));
	for (i=0; i<=Num_NMB; i++){
	  W_NMB[spin][i] = 0.0;
	}
      }

      Ow_NMB2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	Ow_NMB2[spin] = (double**)malloc(sizeof(double*)*(Num_NMB+1));
	for (i=0; i<=Num_NMB; i++){
	  Ow_NMB2[spin][i] = (double*)malloc(sizeof(double)*(Num_NMB+1));
	  for (j=0; j<=Num_NMB; j++){
	    Ow_NMB2[spin][i][j] = 0.0;
	  }
	}
      }

      Sw1 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	Sw1[spin] = (double**)malloc(sizeof(double*)*(Num_NMB+1));
	for (i=0; i<=Num_NMB; i++){
	  Sw1[spin][i] = (double*)malloc(sizeof(double)*(Num_NMB+1));
	  for (j=0; j<=Num_NMB; j++){
	    Sw1[spin][i][j] = 0.0;
	  }
	}
      }

      Uw1 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	Uw1[spin] = (double**)malloc(sizeof(double*)*(Num_NMB+1));
	for (i=0; i<=Num_NMB; i++){
	  Uw1[spin][i] = (double*)malloc(sizeof(double)*(Num_NMB+1));
	  for (j=0; j<=Num_NMB; j++){
	    Uw1[spin][i][j] = 0.0;
	  }
	}
      }

      Rw1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	Rw1[spin] = (double*)malloc(sizeof(double)*(Num_NMB+1));
	for (i=0; i<=Num_NMB; i++){
	  Rw1[spin][i] = 0.0;
	}
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
          W_NMB[0][tmp_num1] = W_full[0][i];
          NMB_posi1[tmp_num1] = i;
	  if(1<0) printf(" ** NMB_posi1[%d] = %d \n",tmp_num1,i);fflush(stdout);
          NMB_posi2[i] = tmp_num1;
          tmp_num2 = -1;
	  for (j=0; j<SizeMat; j++){
	    if (M_or_R2[j]==1){
	      tmp_num2 += 1;
	      S_NMB[0][tmp_num1][tmp_num2] = S_full[0][i][j];
	    }
	  } /* j */
        }
      } /* i */

      if (outputlev==1){
	printf("\n ### S_NMB ### \n");fflush(stdout);
	for (i=0; i<Num_NMB; i++){
	  for (j=0; j<Num_NMB; j++){        
	    printf("%9.5f",S_NMB[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### W_NMB ### \n");fflush(stdout);
	for (i=0; i<Num_NMB; i++){
	  printf("%9.5f \n",W_NMB[0][i]);fflush(stdout);
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

      if (myid == Host_ID) printf("<< 3-1 >> OWSO of NMB orbitals (Ow) \n");
      if (measure_time==1) dtime(&Stime1);

      /***  Sw (= W * S * W) Matrix  ***/
      /*
	for (i=0; i<Num_NMB; i++){
	for (j=0; j<Num_NMB; j++){
        Sw1[0][i+1][j+1] = W_NMB[0][i] * S_NMB[0][i][j] * W_NMB[0][j];
        Uw1[0][i+1][j+1] = Sw1[0][i+1][j+1];
	} 
	} 
      */
      for (i=0; i<Num_NMB; i++){
        tmp_ele0 = W_NMB[0][i];
	for (j=0; j<Num_NMB; j++){
	  tmp_ele2 = tmp_ele0 * S_NMB[0][i][j] * W_NMB[0][j];
	  Sw1[0][i+1][j+1] = tmp_ele2;
	  Uw1[0][i+1][j+1] = tmp_ele2;
	} 
      } 

      if (outputlev==1){
	printf("\n ### Sw1 ### \n");fflush(stdout);
	for (i=1; i<=Num_NMB; i++){
	  for (j=1; j<=Num_NMB; j++){
	    printf("%9.5f",Sw1[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      }

      /***  Eigenvalue problem, Sw * Uw = Rw * Uw  ***/

      Eigen_lapack(Uw1[0], Rw1[0], Num_NMB, Num_NMB);

      if (outputlev==1){
	printf("\n ### Uw1 ### \n");fflush(stdout);
	for (i=1; i<=Num_NMB; i++){
	  for (j=1; j<=Num_NMB; j++){
	    printf("%9.5f",Uw1[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### Rw1 (1) ### \n");fflush(stdout);
	for (i=1; i<=Num_NMB; i++){
	  printf("%9.5f \n",Rw1[0][i]);fflush(stdout);
	}
      }

      for (l=1; l<=Num_NMB; l++){
	if (Rw1[0][l]<1.0e-14) {
          printf("Found ill-conditioned eigenvalues (2)\n");
          printf("Stopped calculation\n");
          exit(1);
	}
      }

      /***  Rw^{-1/2}  ***/

      for (l=1; l<=Num_NMB; l++){
	Rw1[0][l] = 1.0/sqrt(Rw1[0][l]);
      }

      if (outputlev==1){
	printf("\n ### Rw1 (2) ### \n");fflush(stdout);
	for (i=1; i<=Num_NMB; i++){
	  printf("%9.5f \n",Rw1[0][i]);fflush(stdout);
	}
      }

      /***  Sw^{-1/2} = Uw * Rw^{-1/2} * Uw^{t}  ***/
      /*
	for (i=1; i<=Num_NMB; i++){
	for (j=1; j<=Num_NMB; j++){
        temp_M1[0][i][j] = Rw1[0][i] * Uw1[0][j][i];
	} 
	} 

	for (i=1; i<=Num_NMB; i++){
	for (j=1; j<=Num_NMB; j++){
        tmp_ele1 = 0.0;
	for (k=1; k<=Num_NMB; k++){
        tmp_ele1 += Uw1[0][i][k] * temp_M1[0][k][j];
	} 
        Sw1[0][i][j] = tmp_ele1;
	} 
	} 
      */
      for (i=1; i<=Num_NMB; i++){
        tmp_ele0 = Rw1[0][i];
	for (j=1; j<=Num_NMB; j++){
	  temp_M1[0][j][i] = tmp_ele0 * Uw1[0][j][i];
	} 
      } 

      for (i=1; i<=Num_NMB; i++){
	for (j=1; j<=Num_NMB; j++){
	  tmp_ele1 = 0.0;
	  for (k=1; k<=Num_NMB; k++){
	    tmp_ele1 += Uw1[0][i][k] * temp_M1[0][j][k];
	  } 
	  Sw1[0][i][j] = tmp_ele1;
	} 
      } 

      if (outputlev==1){
	printf("\n ### Sw1^{-1/2} ### \n");fflush(stdout);
	for (i=1; i<=Num_NMB; i++){
	  for (j=1; j<=Num_NMB; j++){
	    printf("%9.5f",Sw1[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      }

      /***  Ow = W * Sw^{-1/2}  ***/
      /*
	for (i=0; i<Num_NMB; i++){
	for (j=0; j<Num_NMB; j++){
        Ow_NMB2[0][i+1][j+1] = W_NMB[0][i] * Sw1[0][i+1][j+1];
	} 
	} 
      */
      for (i=0; i<Num_NMB; i++){
        tmp_ele0 = W_NMB[0][i];
	for (j=0; j<Num_NMB; j++){
	  Ow_NMB2[0][i+1][j+1] = tmp_ele0 * Sw1[0][i+1][j+1];
	} /* j */
      } /* i */

      if (outputlev==1){
	printf("\n ### Ow_NMB2 ### \n");fflush(stdout);
	for (i=1; i<=Num_NMB; i++){
	  for (j=1; j<=Num_NMB; j++){
	    printf("%9.5f",Ow_NMB2[0][i][j]);fflush(stdout);
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

      if (myid == Host_ID) printf("<< 3-2 >> Schmidt interatomic orthogonalization of NRB to NMB (O_schm) \n");
      if (measure_time==1) dtime(&Stime1);

      /***  Overlap matrix for NMB and NRB  ***/

      for (i=0; i<Num_NMB; i++){
	for (j=0; j<Num_NMB; j++){
	  N_tran2[0][NMB_posi1[i]][NMB_posi1[j]] = Ow_NMB2[0][i+1][j+1];
	} /* j */
      } /* i */

      if (outputlev==1){
	printf("\n ### Ow_NMB ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",N_tran2[0][i][j]);fflush(stdout);
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
        tmp_ele1 += S_full[0][i][k] * Ow_NMB[0][k][j];
        tmp_ele2 += P_full[0][i][k] * Ow_NMB[0][k][j];
        tmp_ele3 += N_diag[0][i][k] * Ow_NMB[0][k][j];
	} 
        temp_M1[0][i][j] = tmp_ele1;
        temp_M2[0][i][j] = tmp_ele2;
        T_full[0][i][j]  = tmp_ele3;
	} 
	} 

	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += Ow_NMB[0][k][i] * temp_M1[0][k][j];
        tmp_ele2 += Ow_NMB[0][k][i] * temp_M2[0][k][j];
	} 
        S_full[0][i][j] = tmp_ele1;
        P_full[0][i][j] = tmp_ele2;
        O_schm[0][i][j] = T_full[0][i][j];
	} 
	} 
      */

      /* BLAS */
      /*  Ow_NMB * S * Ow_NMB  */
      if(1>0){
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V1[tnum] = N_tran2[0][i][j];
	    temp_V2[tnum] = S_full[0][i][j];
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
	    S_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  Ow_NMB * P * Ow_NMB  */
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = P_full[0][i][j];
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
	    P_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  T * Ow_NMB  */

	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = N_tran1[0][i][j];
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
	    T_full[0][j][i] = temp_V3[tnum];
	    tnum++;
	  }
	}
      }
      else{
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_M3[0][j][i] = N_tran2[0][i][j];
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M3[0][j][k];
	      tmp_ele1 = temp_M3[0][j][k+1];
	      tmp_ele2 = temp_M3[0][j][k+2];
	      tmp_ele3 = temp_M3[0][j][k+3];

	      tmp_ele00 += S_full[0][i][k]   * tmp_ele0;
	      tmp_ele01 += S_full[0][i][k+1] * tmp_ele1;
	      tmp_ele02 += S_full[0][i][k+2] * tmp_ele2;
	      tmp_ele03 += S_full[0][i][k+3] * tmp_ele3;

	      tmp_ele10 += P_full[0][i][k]   * tmp_ele0;
	      tmp_ele11 += P_full[0][i][k+1] * tmp_ele1;
	      tmp_ele12 += P_full[0][i][k+2] * tmp_ele2;
	      tmp_ele13 += P_full[0][i][k+3] * tmp_ele3;

	      tmp_ele20 += N_tran1[0][i][k]   * tmp_ele0;
	      tmp_ele21 += N_tran1[0][i][k+1] * tmp_ele1;
	      tmp_ele22 += N_tran1[0][i][k+2] * tmp_ele2;
	      tmp_ele23 += N_tran1[0][i][k+3] * tmp_ele3;
	    }

	    temp_M1[0][j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    temp_M2[0][j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	    T_full[0][i][j]  = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
	  }
	}

	is = SizeMat - SizeMat%4;

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M3[0][j][k];
	      tmp_ele00 += S_full[0][i][k] * tmp_ele0;
	      tmp_ele10 += P_full[0][i][k] * tmp_ele0;
	      tmp_ele20 += N_tran1[0][i][k] * tmp_ele0;
	    }
	    temp_M1[0][j][i] += tmp_ele00;
	    temp_M2[0][j][i] += tmp_ele10;
	    T_full[0][i][j]  += tmp_ele20;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M3[0][i][k];
	      tmp_ele1 = temp_M3[0][i][k+1];
	      tmp_ele2 = temp_M3[0][i][k+2];
	      tmp_ele3 = temp_M3[0][i][k+3];

	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele01 += tmp_ele1 * temp_M1[0][j][k+1];
	      tmp_ele02 += tmp_ele2 * temp_M1[0][j][k+2];
	      tmp_ele03 += tmp_ele3 * temp_M1[0][j][k+3];

	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	      tmp_ele11 += tmp_ele1 * temp_M2[0][j][k+1];
	      tmp_ele12 += tmp_ele2 * temp_M2[0][j][k+2];
	      tmp_ele13 += tmp_ele3 * temp_M2[0][j][k+3];

	    }
	    S_full[0][i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    P_full[0][i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M3[0][i][k];
	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	    }
	    S_full[0][i][j] += tmp_ele00;
	    P_full[0][i][j] += tmp_ele10;
	  }
	}
      }

      for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	  N_tran1[0][i][j] = T_full[0][i][j];
	}
      }


      if (outputlev==1){
	printf("\n ### S_full (2) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### P_full (2) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### T_full (2) (= N_diag * Ow_NMB) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",T_full[0][i][j]);fflush(stdout);
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
	      tmp_ele1 += S_full0[0][i][k] * T_full[0][k][j];
	    } /* k */
	    temp_M1[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele1 = 0.0;
	    for (k=0; k<SizeMat; k++){
	      tmp_ele1 += T_full[0][k][i] * temp_M1[0][k][j];
	    } /* k */
	    temp_M3[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */

	printf("\n ### S_full (TEST 1) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",temp_M3[0][i][j]);fflush(stdout);
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
	    tmp_ele1 += T_full[0][k][NMB_posi1[j]] * S_full[0][NMB_posi1[j]][NRB_posi1[i]];
	  } /* j */
	  N_tran1[0][k][NRB_posi1[i]] -= tmp_ele1;
	} /* k */
      } /* i */

      if (outputlev==1){
	printf("\n ### O_schm ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",N_tran1[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      }

      for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	  N_tran2[0][i][j]=0.0;
	}
      }
      for (i=0; i<SizeMat; i++){
	N_tran2[0][i][i]=1.0;
      }

      if (measure_time==1){
	dtime(&Etime1);
	time6 = Etime1 - Stime1;
      }

      /****************************************************
     3-3. Symmetry averaging of NRB orbitals (N_ryd)
      ****************************************************/

      if (myid == Host_ID) printf("<< 3-3 >> Symmetry averaging of NRB orbitals (N_ryd) \n");
      if (measure_time==1) dtime(&Stime1);

      /***  Overlap & density matrices (full size)  ***/
      /*
	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += S_full0[0][i][k] * O_schm[0][k][j];
        tmp_ele2 += P_full0[0][i][k] * O_schm[0][k][j];
	} 
        temp_M1[0][i][j] = tmp_ele1;
        temp_M2[0][i][j] = tmp_ele2;
	} 
	} 

	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += O_schm[0][k][i] * temp_M1[0][k][j];
        tmp_ele2 += O_schm[0][k][i] * temp_M2[0][k][j];
	} 
        S_full[0][i][j] = tmp_ele1;
        P_full[0][i][j] = tmp_ele2;
        T_full[0][i][j] = O_schm[0][i][j];
	} 
	} 
      */

      /* BLAS */
      /*  O_schm * S * O_schm  */
      if(1>0){
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V1[tnum] = N_tran1[0][i][j];
	    temp_V2[tnum] = S_full0[0][i][j];
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
	    S_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  O_schm * P * O_schm  */
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = P_full0[0][i][j];
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
	    P_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}
      }
      else{

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_M3[0][j][i] = N_tran1[0][i][j];
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){

	      tmp_ele0 = temp_M3[0][j][k];
	      tmp_ele1 = temp_M3[0][j][k+1];
	      tmp_ele2 = temp_M3[0][j][k+2];
	      tmp_ele3 = temp_M3[0][j][k+3];

	      tmp_ele00 += S_full0[0][i][k]   * tmp_ele0;
	      tmp_ele01 += S_full0[0][i][k+1] * tmp_ele1;
	      tmp_ele02 += S_full0[0][i][k+2] * tmp_ele2;
	      tmp_ele03 += S_full0[0][i][k+3] * tmp_ele3;

	      tmp_ele10 += P_full0[0][i][k]   * tmp_ele0;
	      tmp_ele11 += P_full0[0][i][k+1] * tmp_ele1;
	      tmp_ele12 += P_full0[0][i][k+2] * tmp_ele2;
	      tmp_ele13 += P_full0[0][i][k+3] * tmp_ele3;

	    }
	    temp_M1[0][j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    temp_M2[0][j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	  }
	}

	is = SizeMat - SizeMat%4;

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M3[0][j][k];
	      tmp_ele00 += S_full0[0][i][k] * tmp_ele0;
	      tmp_ele10 += P_full0[0][i][k] * tmp_ele0;
	    }
	    temp_M1[0][j][i] += tmp_ele00;
	    temp_M2[0][j][i] += tmp_ele10;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M3[0][i][k];
	      tmp_ele1 = temp_M3[0][i][k+1];
	      tmp_ele2 = temp_M3[0][i][k+2];
	      tmp_ele3 = temp_M3[0][i][k+3];

	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele01 += tmp_ele1 * temp_M1[0][j][k+1];
	      tmp_ele02 += tmp_ele2 * temp_M1[0][j][k+2];
	      tmp_ele03 += tmp_ele3 * temp_M1[0][j][k+3];

	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	      tmp_ele11 += tmp_ele1 * temp_M2[0][j][k+1];
	      tmp_ele12 += tmp_ele2 * temp_M2[0][j][k+2];
	      tmp_ele13 += tmp_ele3 * temp_M2[0][j][k+3];
	    }
	    S_full[0][i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    P_full[0][i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M3[0][i][k];
	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	    }
	    S_full[0][i][j] += tmp_ele00;
	    P_full[0][i][j] += tmp_ele10;
	  }
	}
      }

      for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	  T_full[0][i][j] = N_tran1[0][i][j];
	}
      }


      if (outputlev==1){
	printf("\n ### S_full (3) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### P_full (3) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### T_full (3) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",T_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      }

      /***  Atomic & anglar- & magetic-momentum block matrices  ***/

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  leng1 = Leng_alm[Gc_AN][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    posi1 = Posi_alm[Gc_AN][L1][M1];
	    for (i=0; i<leng1; i++){
	      for (j=0; j<leng1; j++){
		P_alm[0][Gc_AN][L1][M1][i][j] = P_full[0][i+posi1][j+posi1];
		S_alm[0][Gc_AN][L1][M1][i][j] = S_full[0][i+posi1][j+posi1];
	      } /* j */
	    } /* i */
	  } /* M1 */
	} /* L1 */
      } /* Gc_AN */

      /***  Partitioning and symmetry averaging of P & S  ***/

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  leng1 = Leng_alm[Gc_AN][L1];
	  for (i=0; i<leng1; i++){
	    for (j=0; j<leng1; j++){
	      tmp_ele1 = 0.0;
	      tmp_ele2 = 0.0;
	      if (M_or_R1[Gc_AN][L1][i]==0 && M_or_R1[Gc_AN][L1][j]==0){
		for (M1=1; M1<=L1*2+1; M1++){
		  tmp_ele1 += P_alm[0][Gc_AN][L1][M1][i][j];
		  tmp_ele2 += S_alm[0][Gc_AN][L1][M1][i][j];
		} /* M1 */
		tmp_ele1 /= L1*2.0+1.0;
		tmp_ele2 /= L1*2.0+1.0;
		for (M1=1; M1<=L1*2+1; M1++){
		  P_alm[0][Gc_AN][L1][M1][i][j] = tmp_ele1;
		  S_alm[0][Gc_AN][L1][M1][i][j] = tmp_ele2;
		} /* M1 */
	      }     
	    } /* j */
	  } /* i */
	} /* L1 */
      } /* Gc_AN */

      if (1<0){
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  wan1 = WhatSpecies[Gc_AN];
	  for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	    leng1 = Leng_alm[Gc_AN][L1];
	    for (M1=1; M1<=L1*2+1; M1++){
	      for (i=0; i<leng1; i++){
		for (j=0; j<leng1; j++){
		  printf("%9.5f",S_alm[0][Gc_AN][L1][M1][i][j]);fflush(stdout);
		}
		printf("\n");fflush(stdout);
	      }
	    } /* M1 */
	  } /* L1 */
	} /* Gc_AN */
      }

      /***  Atom & anglar-momentum loop  ***/

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  Mmax1 = Leng_alm[Gc_AN][L1];

	  /***  Transferring of matrices (1)  ***/

	  tmp_num1 = -1;

	  for (i=0; i<Mmax1; i++){
	    if (M_or_R1[Gc_AN][L1][i]==0){
	      tmp_num1 += 1;
	      NRB_posi3[tmp_num1+1] = i;
	      tmp_num2 = -1;
	      for (j=0; j<Mmax1; j++){
		if (M_or_R1[Gc_AN][L1][j]==0){
		  tmp_num2 += 1;
		  P_alm2[0][tmp_num1+1][tmp_num2+1] = P_alm[0][Gc_AN][L1][1][i][j];
		  S_alm2[0][tmp_num1+1][tmp_num2+1] = S_alm[0][Gc_AN][L1][1][i][j];
		}
	      } /* j */
	    }
	  } /* i */

	  Mmax2 = tmp_num1 + 1;
	  if (outputlev==1) printf("\n  Mmax2 = %d \n",Mmax2);fflush(stdout);

	  if (Mmax2 != 0){

	    if (outputlev==1){
	      printf("### S_alm2[%d][%d] ### \n",Gc_AN,L1);fflush(stdout);
	      for (i=0; i<Mmax2; i++){
		for (j=0; j<Mmax2; j++){
		  printf("%16.12f",S_alm2[0][i+1][j+1]);fflush(stdout);
		} /* j */
		printf("\n");fflush(stdout);
	      } /* j */

	      printf("\n");fflush(stdout);

	      printf("### P_alm2[%d][%d] ### \n",Gc_AN,L1);fflush(stdout);
	      for (i=0; i<Mmax2; i++){
		for (j=0; j<Mmax2; j++){
		  printf("%16.12f",P_alm2[0][i+1][j+1]);fflush(stdout);
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

	    Eigen_lapack(S_alm2[0],W_alm2[0],Mmax2,Mmax2);

	    if (outputlev==1){
	      for (k=1; k<=Mmax2; k++){
		printf("k W %2d %15.12f\n",k,W_alm2[0][k]);
	      }
	    }

	    /* check ill-conditioned eigenvalues */

	    for (k=1; k<=Mmax2; k++){
	      if (W_alm2[0][k]<1.0e-14) {
		printf("Found ill-conditioned eigenvalues (3)\n");
		printf("Stopped calculation\n");
		exit(1);
	      }
	    }

	    for (k=1; k<=Mmax2; k++){
	      temp_V1[k] = 1.0/sqrt(W_alm2[0][k]);
	    }

	    /***  Calculations of eigenvalues  ***/

	    for (i=1; i<=Mmax2; i++){
	      for (j=1; j<=Mmax2; j++){
		sum = 0.0;
		for (k=1; k<=Mmax2; k++){
		  sum = sum + P_alm2[0][i][k]*S_alm2[0][k][j]*temp_V1[j];
		}
		N_alm2[0][i][j] = sum;
	      }
	    }

	    for (i=1; i<=Mmax2; i++){
	      for (j=1; j<=Mmax2; j++){
		sum = 0.0;
		for (k=1; k<=Mmax2; k++){
		  sum = sum + temp_V1[i]*S_alm2[0][k][i]*N_alm2[0][k][j];
		}
		temp_M1[0][i][j] = sum;
	      }
	    }

	    if (outputlev==1){
	      printf("\n ##### TEST 1 ##### \n");
	      for (i=1; i<=Mmax2; i++){
		for (j=1; j<=Mmax2; j++){
		  printf("%9.5f",temp_M1[0][i][j]);
		}
		printf("\n");
	      }
	    }

	    if (outputlev==1) printf("\n Diagonalize the D matrix \n");

	    Eigen_lapack(temp_M1[0],W_alm2[0],Mmax2,Mmax2);

	    if (outputlev==1){
	      printf("\n ##### TEST 2 ##### \n");
	      for (k=1; k<=Mmax2; k++){
		printf("k W %2d %9.5f\n",k,W_alm2[0][k]);
	      }

	      printf("\n ##### TEST 3 ##### \n");
	      for (i=1; i<=Mmax2; i++){
		for (j=1; j<=Mmax2; j++){
		  printf("%9.5f",temp_M1[0][i][j]);
		}
		printf("\n");
	      }
	      printf("\n");
	    }

	    /***  Transformation to the original eigen vectors  ***/

	    for (i=1; i<=Mmax2; i++){
	      for (j=1; j<=Mmax2; j++){
		N_alm2[0][i][j] = 0.0;
	      }
	    }

	    for (i=1; i<=Mmax2; i++){
	      for (j=1; j<=Mmax2; j++){
		sum = 0.0;
		for (k=1; k<=Mmax2; k++){
		  sum = sum + S_alm2[0][i][k]*temp_V1[k]*temp_M1[0][k][j];
		}
		/*  N_alm2[0][j][Mmax1-i+1] = sum;  */
		N_alm2[0][i][j] = sum;
	      }
	    }

	    /* printing out eigenvalues and the eigenvectors */

	    if (outputlev==1){
	      for (i=1; i<=Mmax2; i++){
		printf("%ith eigenvalue of HC=eSC: %15.12f\n",i,W_alm2[0][i]);
	      }

	      for (i=1; i<=Mmax2; i++){
		printf("%ith eigenvector: ",i);
		printf("{");
		for (j=1; j<=Mmax2; j++){
		  printf("%7.4f,",N_alm2[0][i][j]);
		}
		printf("}\n");
	      }
	    }

	    /***  Transferring of matrices (2)  ***/

	    for (M1=1; M1<=L1*2+1; M1++){
	      posi1 = Posi_alm[Gc_AN][L1][M1];
	      for (i=1; i<=Mmax2; i++){
		W_full[0][posi1+NRB_posi3[i]] = W_alm2[0][i];
		for (j=1; j<=Mmax2; j++){
		  N_tran2[0][posi1+NRB_posi3[i]][posi1+NRB_posi3[j]] = N_alm2[0][i][j];
		} /* j */
	      } /* i */
	    } /* M1 */

	  } /* if Mmax2 != 0 */

	} /* L1 */
      } /* Gc_AN */

      if (outputlev==1){
	printf("\n ### N_ryd ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",N_tran2[0][i][j]);fflush(stdout);
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
        tmp_ele1 += S_full[0][i][k] * N_ryd[0][k][j];
        tmp_ele2 += P_full[0][i][k] * N_ryd[0][k][j];
        tmp_ele3 += T_full[0][i][k] * N_ryd[0][k][j];
	} 
        temp_M1[0][i][j] = tmp_ele1;
        temp_M2[0][i][j] = tmp_ele2;
        temp_M3[0][i][j] = tmp_ele3;
	} 
	} 

	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += N_ryd[0][k][i] * temp_M1[0][k][j];
        tmp_ele2 += N_ryd[0][k][i] * temp_M2[0][k][j];
	} 
        S_full[0][i][j] = tmp_ele1;
        P_full[0][i][j] = tmp_ele2;
        T_full[0][i][j] = temp_M3[0][i][j];
	} 
	} 
      */

      /* BLAS */
      /*  N_ryd * S * N_ryd  */
      if(1>0){
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V1[tnum] = N_tran2[0][i][j];
	    temp_V2[tnum] = S_full[0][i][j];
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
	    S_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  N_ryd * P * N_ryd  */
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = P_full[0][i][j];
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
	    P_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  T * N_ryd  */

	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = T_full[0][j][i];
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
	    T_full[0][j][i] = temp_V3[tnum];
	    tnum++;
	  }
	}
      }
      else{
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_M4[0][j][i] = N_tran2[0][i][j];
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M4[0][j][k];
	      tmp_ele1 = temp_M4[0][j][k+1];
	      tmp_ele2 = temp_M4[0][j][k+2];
	      tmp_ele3 = temp_M4[0][j][k+3];

	      tmp_ele00 += S_full[0][i][k]   * tmp_ele0;
	      tmp_ele01 += S_full[0][i][k+1] * tmp_ele1;
	      tmp_ele02 += S_full[0][i][k+2] * tmp_ele2;
	      tmp_ele03 += S_full[0][i][k+3] * tmp_ele3;

	      tmp_ele10 += P_full[0][i][k]   * tmp_ele0;
	      tmp_ele11 += P_full[0][i][k+1] * tmp_ele1;
	      tmp_ele12 += P_full[0][i][k+2] * tmp_ele2;
	      tmp_ele13 += P_full[0][i][k+3] * tmp_ele3;

	      tmp_ele20 += T_full[0][i][k]   * tmp_ele0;
	      tmp_ele21 += T_full[0][i][k+1] * tmp_ele1;
	      tmp_ele22 += T_full[0][i][k+2] * tmp_ele2;
	      tmp_ele23 += T_full[0][i][k+3] * tmp_ele3;
	    }
	    temp_M1[0][j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    temp_M2[0][j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	    temp_M3[0][i][j] = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
	  }
	}

	is = SizeMat - SizeMat%4;

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M4[0][j][k];
	      tmp_ele00 += S_full[0][i][k] * tmp_ele0;
	      tmp_ele10 += P_full[0][i][k] * tmp_ele0;
	      tmp_ele20 += T_full[0][i][k] * tmp_ele0;
	    }
	    temp_M1[0][j][i] += tmp_ele00;
	    temp_M2[0][j][i] += tmp_ele10;
	    temp_M3[0][i][j] += tmp_ele20;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M4[0][i][k];
	      tmp_ele1 = temp_M4[0][i][k+1];
	      tmp_ele2 = temp_M4[0][i][k+2];
	      tmp_ele3 = temp_M4[0][i][k+3];

	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele01 += tmp_ele1 * temp_M1[0][j][k+1];
	      tmp_ele02 += tmp_ele2 * temp_M1[0][j][k+2];
	      tmp_ele03 += tmp_ele3 * temp_M1[0][j][k+3];

	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	      tmp_ele11 += tmp_ele1 * temp_M2[0][j][k+1];
	      tmp_ele12 += tmp_ele2 * temp_M2[0][j][k+2];
	      tmp_ele13 += tmp_ele3 * temp_M2[0][j][k+3];
	    }
	    S_full[0][i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    P_full[0][i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M4[0][i][k];
	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	    }
	    S_full[0][i][j] += tmp_ele00;
	    P_full[0][i][j] += tmp_ele10;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    T_full[0][i][j] = temp_M3[0][i][j];
	  }
	}
      }

      if (outputlev==1){
	printf("\n ### S_full (4) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### P_full (4) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### W_full (4) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  printf("%9.5f \n",W_full[0][i]);fflush(stdout);
	}

	printf("\n ### T_full (4) (= N_diag * Ow_NMB * O_schm * N_ryd) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",T_full[0][i][j]);fflush(stdout);
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
	      tmp_ele1 += S_full0[0][i][k] * T_full[0][k][j];
	    } /* k */
	    temp_M1[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele1 = 0.0;
	    for (k=0; k<SizeMat; k++){
	      tmp_ele1 += T_full[0][k][i] * temp_M1[0][k][j];
	    } /* k */
	    temp_M3[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */

	printf("\n ### S_full (TEST 2) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",temp_M3[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      }
      /* ### TEST ### */


      /***  Allocation of arrays (5)  ***/

      S_NRB = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	S_NRB[spin] = (double**)malloc(sizeof(double*)*(Num_NRB+1));
	for (i=0; i<=Num_NRB; i++){
	  S_NRB[spin][i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
	  for (j=0; j<=Num_NRB; j++){
	    S_NRB[spin][i][j] = 0.0;
	  }
	}
      }

      S_NRB2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	S_NRB2[spin] = (double**)malloc(sizeof(double*)*(Num_NRB+1));
	for (i=0; i<=Num_NRB; i++){
	  S_NRB2[spin][i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
	  for (j=0; j<=Num_NRB; j++){
	    S_NRB2[spin][i][j] = 0.0;
	  }
	}
      }

      W_NRB = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	W_NRB[spin] = (double*)malloc(sizeof(double)*(Num_NRB+1));
	for (i=0; i<=Num_NRB; i++){
	  W_NRB[spin][i] = 0.0;
	}
      }

      W_NRB2 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	W_NRB2[spin] = (double*)malloc(sizeof(double)*(Num_NRB+1));
	for (i=0; i<=Num_NRB; i++){
	  W_NRB2[spin][i] = 0.0;
	}
      }

      Ow_NRB2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	Ow_NRB2[spin] = (double**)malloc(sizeof(double*)*(Num_NRB+1));
	for (i=0; i<=Num_NRB; i++){
	  Ow_NRB2[spin][i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
	  for (j=0; j<=Num_NRB; j++){
	    Ow_NRB2[spin][i][j] = 0.0;
	  }
	}
      }

      Sw2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	Sw2[spin] = (double**)malloc(sizeof(double*)*(Num_NRB+1));
	for (i=0; i<=Num_NRB; i++){
	  Sw2[spin][i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
	  for (j=0; j<=Num_NRB; j++){
	    Sw2[spin][i][j] = 0.0;
	  }
	}
      }

      Uw2 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	Uw2[spin] = (double**)malloc(sizeof(double*)*(Num_NRB+1));
	for (i=0; i<=Num_NRB; i++){
	  Uw2[spin][i] = (double*)malloc(sizeof(double)*(Num_NRB+1));
	  for (j=0; j<=Num_NRB; j++){
	    Uw2[spin][i][j] = 0.0;
	  }
	}
      }

      Rw2 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
      for (spin=0; spin<=SpinP_switch; spin++){
	Rw2[spin] = (double*)malloc(sizeof(double)*(Num_NRB+1));
	for (i=0; i<=Num_NRB; i++){
	  Rw2[spin][i] = 0.0;
	}
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
          W_NRB[0][tmp_num1] = W_full[0][i];
          NRB_posi8[tmp_num1] = i;
          NRB_posi9[i] = tmp_num1;
          tmp_num2 = -1;
	  for (j=0; j<SizeMat; j++){
	    if (M_or_R2[j]==0){
	      tmp_num2 += 1;
	      S_NRB[0][tmp_num1][tmp_num2] = S_full[0][i][j];
	    }
	  } /* j */
        }
      } /* i */

      if (outputlev==1){
	printf("\n ### S_NRB (5) ### \n");fflush(stdout);
	for (i=0; i<Num_NRB; i++){
	  for (j=0; j<Num_NRB; j++){
	    printf("%9.5f",S_NRB[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
	printf("\n");
      }

      if (outputlev==1) printf("\n ### W_NRB ### \n");fflush(stdout);
      tmp_num1 = -1;
      tmp_num2 = -1;
      for (i=0; i<Num_NRB; i++){
	if(1<0) printf("%9.5f \n",W_NRB[0][i]);fflush(stdout);
        if(W_NRB[0][i] >= 1.0e-3){
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
	printf("\n Num_NRB2 = %d   ",Num_NRB2);fflush(stdout);
	printf("\n Num_NRB3 = %d \n",Num_NRB3);fflush(stdout);
      }

      if (Num_NRB2 == 0 && Num_NRB3 == 0){
	printf("\n ### STOP (Num_NRB2=0 & Num_NRB3=0) (ID=%d) \n",myid);
	exit(0);
      }

      for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	  N_tran1[0][i][j]=0.0;
	}
      }
      for (i=0; i<SizeMat; i++){
	N_tran1[0][i][i]=1.0;
      }

      if (measure_time==1){
	dtime(&Etime1);
	time7 = Etime1 - Stime1;
      }

      /************************************
     3-4. OWSO of NRB orbitals (Ow)
      ************************************/

      if (myid == Host_ID) printf("<< 3-4 >> OWSO of NRB orbitals (Ow) \n");
      if (measure_time==1) dtime(&Stime1);

      /***  Sw (= W * S * W) Matrix  ***/
      /*
	for (i=0; i<Num_NRB; i++){
	for (j=0; j<Num_NRB; j++){
	if(S_or_L[i] == 1 && S_or_L[j] == 1){
        k = NRB_posi4[i];
        l = NRB_posi4[j];
        Sw2[0][k+1][l+1] = W_NRB[0][i] * S_NRB[0][i][j] * W_NRB[0][j];
        Uw2[0][k+1][l+1] = Sw2[0][k+1][l+1];
        W_NRB2[0][k+1]   = W_NRB[0][i];
	}
	} 
	} 
      */

      for (i=0; i<Num_NRB; i++){
        tmp_ele1 = W_NRB[0][i];
	for (j=0; j<Num_NRB; j++){
	  if(S_or_L[i] == 1 && S_or_L[j] == 1){
	    k = NRB_posi4[i];
	    l = NRB_posi4[j];
	    tmp_ele2 = tmp_ele1 * S_NRB[0][i][j] * W_NRB[0][j];
	    Sw2[0][k+1][l+1] = tmp_ele2;
	    Uw2[0][k+1][l+1] = tmp_ele2;
	    W_NRB2[0][k+1]   = tmp_ele1;
	  }
	} 
      } 

      if (outputlev==1){
	printf("\n ### Sw2 ### \n");fflush(stdout);
	for (i=1; i<=Num_NRB2; i++){
	  for (j=1; j<=Num_NRB2; j++){
	    printf("%9.5f",Sw2[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      }
  
      /***  Eigenvalue problem, Sw * Uw = Rw * Uw  ***/

      Eigen_lapack(Uw2[0], Rw2[0], Num_NRB2, Num_NRB2);

      if (outputlev==1){
	printf("\n ### Uw2 ### \n");fflush(stdout);
	for (i=1; i<=Num_NRB2; i++){
	  for (j=1; j<=Num_NRB2; j++){
	    printf("%9.5f",Uw2[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      } 

      if (outputlev==1){
	printf("\n ### Rw2 (1) ### \n");fflush(stdout);
	for (i=1; i<=Num_NRB2; i++){
	  printf("%9.5f \n",Rw2[0][i]);fflush(stdout);
	}
      }

      for (l=1; l<=Num_NRB2; l++){
	if (Rw2[0][l]<1.0e-14) {
          printf("Found ill-conditioned eigenvalues (4)\n");
          printf("Stopped calculation\n");
          exit(1);
	}
      }

      /***  Rw^{-1/2}  ***/

      for (l=1; l<=Num_NRB2; l++){
	Rw2[0][l] = 1.0/sqrt(Rw2[0][l]);
      }

      if (outputlev==1){
	printf("\n ### Rw2 (2) ### \n");fflush(stdout);
	for (i=1; i<=Num_NRB2; i++){
	  printf("%9.5f \n",Rw2[0][i]);fflush(stdout);
	}
      }

      /***  Sw^{-1/2} = Uw * Rw^{-1/2} * Uw^{t}  ***/
      /*
	for (i=1; i<=Num_NRB2; i++){
	for (j=1; j<=Num_NRB2; j++){
        temp_M1[0][i][j] = Rw2[0][i] * Uw2[0][j][i];
	} 
	} 

	for (i=1; i<=Num_NRB2; i++){
	for (j=1; j<=Num_NRB2; j++){
        tmp_ele1 = 0.0;
	for (k=1; k<=Num_NRB2; k++){
        tmp_ele1 += Uw2[0][i][k] * temp_M1[0][k][j];
	} 
        Sw2[0][i][j] = tmp_ele1;
	} 
	} 
      */

      for (i=1; i<=Num_NRB2; i++){
	for (j=1; j<=Num_NRB2; j++){
	  temp_M2[0][j][i] = Uw2[0][i][j];
	}
      }

      for (i=1; i<=Num_NRB2; i++){
        tmp_ele0 = Rw2[0][i];
	for (j=1; j<=Num_NRB2; j++){
	  temp_M1[0][j][i] = tmp_ele0 * temp_M2[0][i][j];
	} /* j */
      } /* i */

      /* BLAS */
      if(1>0){
	tnum = 0;
	for (i=1; i<=Num_NRB2; i++){
	  for (j=1; j<=Num_NRB2; j++){
	    temp_V1[tnum] =     Uw2[0][j][i];
	    temp_V2[tnum] = temp_M1[0][j][i];
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
	    Sw2[0][i][j] = temp_V3[tnum];
	    tnum++;
	  }
	}

      }
      else{
	for (i=1; i<=Num_NRB2; i++){
	  for (j=1; j<=Num_NRB2; j++){
	    tmp_ele1 = 0.0;
	    for (k=1; k<=Num_NRB2; k++){
	      tmp_ele1 += Uw2[0][i][k] * temp_M1[0][j][k];
	    } /* k */
	    Sw2[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */
      }


      if (outputlev==1){
	printf("\n ### Sw2^{-1/2} ### \n");fflush(stdout);
	for (i=1; i<=Num_NRB2; i++){
	  for (j=1; j<=Num_NRB2; j++){
	    printf("%19.10f",Sw2[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### W_NRB2 ### \n");fflush(stdout);
	for (i=1; i<=Num_NRB2; i++){
	  printf("%9.5f \n",W_NRB2[0][i]);fflush(stdout);
	}
      }

      /***  Ow = W * Sw^{-1/2}  ***/
      /*
	for (i=1; i<=Num_NRB2; i++){
	for (j=1; j<=Num_NRB2; j++){
        Ow_NRB2[0][i][j] = W_NRB2[0][i] * Sw2[0][i][j];
	} 
	}
      */

      for (i=1; i<=Num_NRB2; i++){
        tmp_ele0 = W_NRB2[0][i];
	for (j=1; j<=Num_NRB2; j++){
	  Ow_NRB2[0][i][j] = tmp_ele0 * Sw2[0][i][j];
	} /* j */
      } /* i */

      if (outputlev==1){
	printf("\n ### Ow_NRB2 ### \n");fflush(stdout);
	for (i=1; i<=Num_NRB2; i++){
	  for (j=1; j<=Num_NRB2; j++){
	    printf("%9.5f",Ow_NRB2[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      }

      for (i=1; i<=Num_NRB2; i++){
	for (j=1; j<=Num_NRB2; j++){
	  k = NRB_posi5[i];
	  l = NRB_posi5[j];
	  N_tran1[0][NRB_posi1[k]][NRB_posi1[l]] = Ow_NRB2[0][i][j];
	} /* j */
      } /* i */

      if (outputlev==1){
	printf("\n ### Ow_NRB ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",N_tran1[0][i][j]);fflush(stdout);
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
        tmp_ele1 += S_full[0][i][k] * Ow_NRB[0][k][j];
        tmp_ele2 += P_full[0][i][k] * Ow_NRB[0][k][j];
        tmp_ele3 += T_full[0][i][k] * Ow_NRB[0][k][j];
	} 
        temp_M1[0][i][j] = tmp_ele1;
        temp_M2[0][i][j] = tmp_ele2;
        temp_M3[0][i][j] = tmp_ele3;
	} 
	} 

	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += Ow_NRB[0][k][i] * temp_M1[0][k][j];
        tmp_ele2 += Ow_NRB[0][k][i] * temp_M2[0][k][j];
	} 
        S_full[0][i][j] = tmp_ele1;
        P_full[0][i][j] = tmp_ele2;
        T_full[0][i][j] = temp_M3[0][i][j];
        O_schm2[0][i][j] = T_full[0][i][j];
	} 
	} 
      */

      /* BLAS */
      /*  Ow_NRB * S * Ow_NRB  */
      if(1>0){
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V1[tnum] = N_tran1[0][i][j];
	    temp_V2[tnum] = S_full[0][i][j];
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
	    S_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  Ow_NRB * P * Ow_NRB  */
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = P_full[0][i][j];
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
	    P_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  T * Ow_NRB  */

	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = T_full[0][j][i];
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
	    T_full[0][j][i] = temp_V3[tnum];
	    N_tran2[0][j][i] = temp_V3[tnum];
	    tnum++;
	  }
	}
      }
      else{
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_M4[0][j][i] = N_tran1[0][i][j];
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M4[0][j][k];
	      tmp_ele1 = temp_M4[0][j][k+1];
	      tmp_ele2 = temp_M4[0][j][k+2];
	      tmp_ele3 = temp_M4[0][j][k+3];

	      tmp_ele00 += S_full[0][i][k]   * tmp_ele0;
	      tmp_ele01 += S_full[0][i][k+1] * tmp_ele1;
	      tmp_ele02 += S_full[0][i][k+2] * tmp_ele2;
	      tmp_ele03 += S_full[0][i][k+3] * tmp_ele3;

	      tmp_ele10 += P_full[0][i][k]   * tmp_ele0;
	      tmp_ele11 += P_full[0][i][k+1] * tmp_ele1;
	      tmp_ele12 += P_full[0][i][k+2] * tmp_ele2;
	      tmp_ele13 += P_full[0][i][k+3] * tmp_ele3;

	      tmp_ele20 += T_full[0][i][k]   * tmp_ele0;
	      tmp_ele21 += T_full[0][i][k+1] * tmp_ele1;
	      tmp_ele22 += T_full[0][i][k+2] * tmp_ele2;
	      tmp_ele23 += T_full[0][i][k+3] * tmp_ele3;
	    }
	    temp_M1[0][j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    temp_M2[0][j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	    temp_M3[0][i][j] = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
	  }
	}

	is = SizeMat - SizeMat%4;

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M4[0][j][k];
	      tmp_ele00 += S_full[0][i][k] * tmp_ele0;
	      tmp_ele10 += P_full[0][i][k] * tmp_ele0;
	      tmp_ele20 += T_full[0][i][k] * tmp_ele0;
	    }
	    temp_M1[0][j][i] += tmp_ele00;
	    temp_M2[0][j][i] += tmp_ele10;
	    temp_M3[0][i][j] += tmp_ele20;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M4[0][i][k];
	      tmp_ele1 = temp_M4[0][i][k+1];
	      tmp_ele2 = temp_M4[0][i][k+2];
	      tmp_ele3 = temp_M4[0][i][k+3];

	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele01 += tmp_ele1 * temp_M1[0][j][k+1];
	      tmp_ele02 += tmp_ele2 * temp_M1[0][j][k+2];
	      tmp_ele03 += tmp_ele3 * temp_M1[0][j][k+3];

	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	      tmp_ele11 += tmp_ele1 * temp_M2[0][j][k+1];
	      tmp_ele12 += tmp_ele2 * temp_M2[0][j][k+2];
	      tmp_ele13 += tmp_ele3 * temp_M2[0][j][k+3];
	    }
	    S_full[0][i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    P_full[0][i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M4[0][i][k];
	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	    }
	    S_full[0][i][j] += tmp_ele00;
	    P_full[0][i][j] += tmp_ele10;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele0 = temp_M3[0][i][j];
	    T_full[0][i][j]  = tmp_ele0;
	    N_tran2[0][i][j] = tmp_ele0;
	  }
	}
      }

      /* ### TEST ### */
      if (outputlev==1){
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele1 = 0.0;
	    for (k=0; k<SizeMat; k++){
	      tmp_ele1 += S_full0[0][i][k] * T_full[0][k][j];
	    } /* k */
	    temp_M1[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele1 = 0.0;
	    for (k=0; k<SizeMat; k++){
	      tmp_ele1 += T_full[0][k][i] * temp_M1[0][k][j];
	    } /* k */
	    temp_M3[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */

	printf("\n ### S_full (TEST 3) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",temp_M3[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      }
      /* ### TEST ### */

      if (outputlev==1){
	printf("\n ### S_full (5) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### P_full (5) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### T_full (5) (= N_diag * Ow_NMB * O_schm * N_ryd * Ow_NRB) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",T_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}
      }

      /***  Treatment of low-occupied NRBs  ***/

      if(Num_NRB3>0){

	printf("\n ** Schmidt orthogonalization of low-occupied NRBs to high ones **\n");fflush(stdout);

	for (i=0; i<Num_NRB3; i++){
	  i2 = NRB_posi8[NRB_posi7[i+1]];
	  for (k=0; k<SizeMat;  k++){
	    tmp_ele1 = 0.0;
	    for (j=0; j<Num_NRB2; j++){
	      j2 = NRB_posi8[NRB_posi5[j+1]];
	      tmp_ele1 += T_full[0][k][j2] * S_full[0][j2][i2];
	    } /* j */
	    N_tran2[0][k][i2] -= tmp_ele1;
	  } /* k */
	} /* i */

	if(outputlev == 1){
	  printf("\n ### O_schm2 ### \n");fflush(stdout);
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      printf("%9.5f",N_tran2[0][i][j]);fflush(stdout);
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
	  tmp_ele1 += S_full0[0][i][k] * O_schm2[0][k][j];
	  tmp_ele2 += P_full0[0][i][k] * O_schm2[0][k][j];
	  }
	  temp_M1[0][i][j] = tmp_ele1;
	  temp_M2[0][i][j] = tmp_ele2;
	  }
	  }

	  for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	  tmp_ele1 = 0.0;
	  tmp_ele2 = 0.0;
	  for (k=0; k<SizeMat; k++){
	  tmp_ele1 += O_schm2[0][k][i] * temp_M1[0][k][j];
	  tmp_ele2 += O_schm2[0][k][i] * temp_M2[0][k][j];
	  } 
	  S_full[0][i][j] = tmp_ele1;
	  P_full[0][i][j] = tmp_ele2;
	  T_full[0][i][j] = O_schm2[0][i][j];
	  } 
	  } 
	*/

	/* BLAS */
	/*  O_schm * S * O_schm  */
	if(1>0){
	  tnum = 0;
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      temp_V1[tnum] = N_tran2[0][i][j];
	      temp_V2[tnum] = S_full0[0][i][j];
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
	      S_full[0][i][j] = temp_V2[tnum];
	      tnum++;
	    }
	  }

	  /*  O_schm * P * O_schm  */
	  tnum = 0;
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      temp_V2[tnum] = P_full0[0][i][j];
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
	      P_full[0][i][j] = temp_V2[tnum];
	      tnum++;
	    }
	  }
	}
	else{
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      temp_M3[0][j][i] = N_tran2[0][i][j];
	    }
	  }

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	      tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	      tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	      tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	      for (k=0; k<(SizeMat-3); k+=4){

		tmp_ele0 = temp_M3[0][j][k];
		tmp_ele1 = temp_M3[0][j][k+1];
		tmp_ele2 = temp_M3[0][j][k+2];
		tmp_ele3 = temp_M3[0][j][k+3];

		tmp_ele00 += S_full0[0][i][k]   * tmp_ele0;
		tmp_ele01 += S_full0[0][i][k+1] * tmp_ele1;
		tmp_ele02 += S_full0[0][i][k+2] * tmp_ele2;
		tmp_ele03 += S_full0[0][i][k+3] * tmp_ele3;

		tmp_ele10 += P_full0[0][i][k]   * tmp_ele0;
		tmp_ele11 += P_full0[0][i][k+1] * tmp_ele1;
		tmp_ele12 += P_full0[0][i][k+2] * tmp_ele2;
		tmp_ele13 += P_full0[0][i][k+3] * tmp_ele3;

	      }
	      temp_M1[0][j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	      temp_M2[0][j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	    }
	  }

	  is = SizeMat - SizeMat%4;

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	      for (k=is; k<SizeMat; k++){
		tmp_ele0 = temp_M3[0][j][k];
		tmp_ele00 += S_full0[0][i][k] * tmp_ele0;
		tmp_ele10 += P_full0[0][i][k] * tmp_ele0;
	      }
	      temp_M1[0][j][i] += tmp_ele00;
	      temp_M2[0][j][i] += tmp_ele10;
	    }
	  }

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	      tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	      tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	      tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	      for (k=0; k<(SizeMat-3); k+=4){
		tmp_ele0 = temp_M3[0][i][k];
		tmp_ele1 = temp_M3[0][i][k+1];
		tmp_ele2 = temp_M3[0][i][k+2];
		tmp_ele3 = temp_M3[0][i][k+3];

		tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
		tmp_ele01 += tmp_ele1 * temp_M1[0][j][k+1];
		tmp_ele02 += tmp_ele2 * temp_M1[0][j][k+2];
		tmp_ele03 += tmp_ele3 * temp_M1[0][j][k+3];

		tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
		tmp_ele11 += tmp_ele1 * temp_M2[0][j][k+1];
		tmp_ele12 += tmp_ele2 * temp_M2[0][j][k+2];
		tmp_ele13 += tmp_ele3 * temp_M2[0][j][k+3];
	      }
	      S_full[0][i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	      P_full[0][i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	    }
	  }

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	      for (k=is; k<SizeMat; k++){
		tmp_ele0 = temp_M3[0][i][k];
		tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
		tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	      }
	      S_full[0][i][j] += tmp_ele00;
	      P_full[0][i][j] += tmp_ele10;
	    }
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    T_full[0][i][j] = N_tran2[0][i][j];
	  }
	}





	/* ### TEST ### */
	if (outputlev==1){
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele1 = 0.0;
	      for (k=0; k<SizeMat; k++){
		tmp_ele1 += S_full0[0][i][k] * T_full[0][k][j];
	      } /* k */
	      temp_M1[0][i][j] = tmp_ele1;
	    } /* j */
	  } /* i */

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele1 = 0.0;
	      for (k=0; k<SizeMat; k++){
		tmp_ele1 += T_full[0][k][i] * temp_M1[0][k][j];
	      } /* k */
	      temp_M3[0][i][j] = tmp_ele1;
	    } /* j */
	  } /* i */

	  printf("\n ### S_full (TEST 2) ### \n");fflush(stdout);
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      printf("%9.5f",temp_M3[0][i][j]);fflush(stdout);
	    }
	    printf("\n");
	  }
	}
	/* ### TEST ### */


	if (outputlev==1){
	  printf("\n ### S_full (6) ### \n");fflush(stdout);
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	    }
	    printf("\n");
	  }

	  printf("\n ### P_full (6) ### \n");fflush(stdout);
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	    }
	    printf("\n");
	  }

	  printf("\n ### T_full (6) ### \n");fflush(stdout);
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      printf("%9.5f",T_full[0][i][j]);fflush(stdout);
	    }
	    printf("\n");
	  }
	}

	printf("\n ** Loewdin symmetric orthogonalization **\n");fflush(stdout);


	/***  Set of overlap matrix for low-occpied NRBs (S_NRB2)  ***/

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    N_tran1[0][i][j]=0.0;
	  }
	}
	for (i=0; i<SizeMat; i++){
	  N_tran1[0][i][i]=1.0;
	}

	for (i=1; i<=Num_NRB3; i++){
	  i2 = NRB_posi8[NRB_posi7[i]];
	  for (j=1; j<=Num_NRB3; j++){
	    j2 = NRB_posi8[NRB_posi7[j]];
	    S_NRB2[0][i][j] = S_full[0][i2][j2];
	    Uw2[0][i][j]    = S_NRB2[0][i][j];
	  } /* j */
	} /* i */

	if (outputlev==1){
	  printf("\n ### S_NRB2 ### \n");fflush(stdout);
	  for (i=1; i<=Num_NRB3; i++){
	    for (j=1; j<=Num_NRB3; j++){
	      printf("%9.5f",S_NRB2[0][i][j]);fflush(stdout);
	    } /* j */
	    printf("\n");
	  } /* i */
	}

	/***  Eigenvalue problem, S_NRB2 * Uw = Rw * Uw  ***/

	Eigen_lapack(Uw2[0], Rw2[0], Num_NRB3,Num_NRB3);

	if (outputlev==1){
	  printf("\n ### Uw1 ### \n");fflush(stdout);
	  for (i=1; i<=Num_NRB3; i++){
	    for (j=1; j<=Num_NRB3; j++){
	      printf("%9.5f",Uw2[0][i][j]);fflush(stdout);
	    }
	    printf("\n");
	  }
	}

	if (outputlev==1){
	  printf("\n ### Rw1 (1) ### \n");fflush(stdout);
	  for (i=1; i<=Num_NRB3; i++){
	    printf("%9.5f \n",Rw2[0][i]);fflush(stdout);
	  }
	}

	for (l=1; l<=Num_NRB3; l++){
	  if (Rw2[0][l]<1.0e-14) {
	    printf("Found ill-conditioned eigenvalues (5)\n");
	    printf("Stopped calculation\n");
	    exit(1);
	  }
	}

	/***  Rw^{-1/2}  ***/

	for (l=1; l<=Num_NRB3; l++){
	  Rw2[0][l] = 1.0/sqrt(Rw2[0][l]);
	}

	if (outputlev==1){
	  printf("\n ### Rw1 (2) ### \n");fflush(stdout);
	  for (i=1; i<=Num_NRB3; i++){
	    printf("%9.5f \n",Rw2[0][i]);fflush(stdout);
	  }
	}

	/***  Sw^{-1/2} = Uw * Rw^{-1/2} * Uw^{t}  ***/

	for (i=1; i<=Num_NRB3; i++){
	  for (j=1; j<=Num_NRB3; j++){
	    temp_M1[0][i][j] = Rw2[0][i] * Uw2[0][j][i];
	  } /* j */
	} /* i */

	for (i=1; i<=Num_NRB3; i++){
	  for (j=1; j<=Num_NRB3; j++){
	    tmp_ele1 = 0.0;
	    for (k=1; k<=Num_NRB3; k++){
	      tmp_ele1 += Uw2[0][i][k] * temp_M1[0][k][j];
	    } /* k */
	    Sw2[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */

	if (outputlev==1){
	  printf("\n ### Sw2^{-1/2} ### \n");fflush(stdout);
	  for (i=1; i<=Num_NRB3; i++){
	    for (j=1; j<=Num_NRB3; j++){
	      printf("%9.5f",Sw2[0][i][j]);fflush(stdout);
	    }
	    printf("\n");
	  }
	}

	for (i=1; i<=Num_NRB3; i++){
	  i2 = NRB_posi8[NRB_posi7[i]];
	  for (j=1; j<=Num_NRB3; j++){
	    j2 = NRB_posi8[NRB_posi7[j]];
	    N_tran1[0][i2][j2] = Sw2[0][i][j];
	  } /* j */
	} /* i */

	if (outputlev==1){

	  printf("\n ### O_sym ### \n");fflush(stdout);
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      printf("%9.5f",N_tran1[0][i][j]);fflush(stdout);
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
	  tmp_ele1 += S_full[0][i][k] * O_sym[0][k][j];
	  tmp_ele2 += P_full[0][i][k] * O_sym[0][k][j];
	  tmp_ele3 += T_full[0][i][k] * O_sym[0][k][j];
	  }
	  temp_M1[0][i][j] = tmp_ele1;
	  temp_M2[0][i][j] = tmp_ele2;
	  temp_M3[0][i][j] = tmp_ele3;
	  }
	  }

	  for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	  tmp_ele1 = 0.0;
	  tmp_ele2 = 0.0;
	  for (k=0; k<SizeMat; k++){
	  tmp_ele1 += O_sym[0][k][i] * temp_M1[0][k][j];
	  tmp_ele2 += O_sym[0][k][i] * temp_M2[0][k][j];
	  }
	  S_full[0][i][j] = tmp_ele1;
	  P_full[0][i][j] = tmp_ele2;
	  T_full[0][i][j] = temp_M3[0][i][j];
	  }
	  }
	*/

	/* BLAS */
	/*  Ow_NRB * S * Ow_NRB  */
	if(1>0){
	  tnum = 0;
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      temp_V1[tnum] = N_tran1[0][i][j];
	      temp_V2[tnum] = S_full[0][i][j];
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
	      S_full[0][i][j] = temp_V2[tnum];
	      tnum++;
	    }
	  }

	  /*  Ow_NRB * P * Ow_NRB  */
	  tnum = 0;
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      temp_V2[tnum] = P_full[0][i][j];
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
	      P_full[0][i][j] = temp_V2[tnum];
	      tnum++;
	    }
	  }

	  /*  T * Ow_NRB  */

	  tnum = 0;
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      temp_V2[tnum] = T_full[0][j][i];
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
	      T_full[0][j][i] = temp_V3[tnum];
	      tnum++;
	    }
	  }
	}
	else{
	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      temp_M4[0][j][i] = N_tran1[0][i][j];
	    }
	  }

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	      tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
	      tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
	      tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
	      for (k=0; k<(SizeMat-3); k+=4){
		tmp_ele0 = temp_M4[0][j][k];
		tmp_ele1 = temp_M4[0][j][k+1];
		tmp_ele2 = temp_M4[0][j][k+2];
		tmp_ele3 = temp_M4[0][j][k+3];

		tmp_ele00 += S_full[0][i][k]   * tmp_ele0;
		tmp_ele01 += S_full[0][i][k+1] * tmp_ele1;
		tmp_ele02 += S_full[0][i][k+2] * tmp_ele2;
		tmp_ele03 += S_full[0][i][k+3] * tmp_ele3;

		tmp_ele10 += P_full[0][i][k]   * tmp_ele0;
		tmp_ele11 += P_full[0][i][k+1] * tmp_ele1;
		tmp_ele12 += P_full[0][i][k+2] * tmp_ele2;
		tmp_ele13 += P_full[0][i][k+3] * tmp_ele3;

		tmp_ele20 += T_full[0][i][k]   * tmp_ele0;
		tmp_ele21 += T_full[0][i][k+1] * tmp_ele1;
		tmp_ele22 += T_full[0][i][k+2] * tmp_ele2;
		tmp_ele23 += T_full[0][i][k+3] * tmp_ele3;
	      }
	      temp_M1[0][j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	      temp_M2[0][j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	      temp_M3[0][i][j] = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
	    }
	  }

	  is = SizeMat - SizeMat%4;

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	      for (k=is; k<SizeMat; k++){
		tmp_ele0 = temp_M4[0][j][k];
		tmp_ele00 += S_full[0][i][k] * tmp_ele0;
		tmp_ele10 += P_full[0][i][k] * tmp_ele0;
		tmp_ele20 += T_full[0][i][k] * tmp_ele0;
	      }
	      temp_M1[0][j][i] += tmp_ele00;
	      temp_M2[0][j][i] += tmp_ele10;
	      temp_M3[0][i][j] += tmp_ele20;
	    }
	  }

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	      tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	      tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	      tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	      for (k=0; k<(SizeMat-3); k+=4){
		tmp_ele0 = temp_M4[0][i][k];
		tmp_ele1 = temp_M4[0][i][k+1];
		tmp_ele2 = temp_M4[0][i][k+2];
		tmp_ele3 = temp_M4[0][i][k+3];

		tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
		tmp_ele01 += tmp_ele1 * temp_M1[0][j][k+1];
		tmp_ele02 += tmp_ele2 * temp_M1[0][j][k+2];
		tmp_ele03 += tmp_ele3 * temp_M1[0][j][k+3];

		tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
		tmp_ele11 += tmp_ele1 * temp_M2[0][j][k+1];
		tmp_ele12 += tmp_ele2 * temp_M2[0][j][k+2];
		tmp_ele13 += tmp_ele3 * temp_M2[0][j][k+3];
	      }
	      S_full[0][i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	      P_full[0][i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	    }
	  }

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	      for (k=is; k<SizeMat; k++){
		tmp_ele0 = temp_M4[0][i][k];
		tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
		tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	      }
	      S_full[0][i][j] += tmp_ele00;
	      P_full[0][i][j] += tmp_ele10;
	    }
	  }

	  for (i=0; i<SizeMat; i++){
	    for (j=0; j<SizeMat; j++){
	      T_full[0][i][j] = temp_M3[0][i][j];
	    }
	  }
	}

      } /* if Num_NRB3 > 0 */

      if (outputlev==1){
	printf("\n ### S_full (7) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### P_full (7) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

	printf("\n ### T_full (7) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",T_full[0][i][j]);fflush(stdout);
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

      if (myid == Host_ID) printf("<< 3-5 >> Symmetry averaging of whole orbitals to construct NAOs (N_red) \n");
      if (measure_time==1) dtime(&Stime1);

      for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
	  N_tran2[0][i][j]=0.0;
	}
      }
      for (i=0; i<SizeMat; i++){
	N_tran2[0][i][i]=1.0;
      }
      /***  Atomic & anglar- & magetic-momentum block matrices  ***/

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  leng1 = Leng_alm[Gc_AN][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    posi1 = Posi_alm[Gc_AN][L1][M1];
	    for (i=0; i<leng1; i++){
	      for (j=0; j<leng1; j++){
		P_alm[0][Gc_AN][L1][M1][i][j] = P_full[0][i+posi1][j+posi1];
		S_alm[0][Gc_AN][L1][M1][i][j] = S_full[0][i+posi1][j+posi1];
	      } /* j */
	    } /* i */
	  } /* M1 */
	} /* L1 */
      } /* Gc_AN */

      /***  Partitioning and symmetry averaging of P & S  ***/

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  leng1 = Leng_alm[Gc_AN][L1];
	  for (i=0; i<leng1; i++){
	    for (j=0; j<leng1; j++){
	      tmp_ele1 = 0.0;
	      tmp_ele2 = 0.0;
	      for (M1=1; M1<=L1*2+1; M1++){
		tmp_ele1 += P_alm[0][Gc_AN][L1][M1][i][j];
		tmp_ele2 += S_alm[0][Gc_AN][L1][M1][i][j];
	      } /* M1 */
	      tmp_ele1 /= L1*2.0+1.0;
	      tmp_ele2 /= L1*2.0+1.0;
	      for (M1=1; M1<=L1*2+1; M1++){
		P_alm[0][Gc_AN][L1][M1][i][j] = tmp_ele1;
		S_alm[0][Gc_AN][L1][M1][i][j] = tmp_ele2;
	      } /* M1 */
	    } /* j */
	  } /* i */
	} /* L1 */
      } /* Gc_AN */

      /***  Atom & anglar-momentum loop  ***/

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  Mmax1 = Leng_alm[Gc_AN][L1];

	  /***  Transferring of matrices (1)  ***/

	  for (i=0; i<Mmax1; i++){
	    for (j=0; j<Mmax1; j++){
	      P_alm2[0][i+1][j+1] = P_alm[0][Gc_AN][L1][1][i][j];
	      S_alm2[0][i+1][j+1] = S_alm[0][Gc_AN][L1][1][i][j];
	    } /* j */
	  } /* i */

	  if (outputlev==1){
	    printf("\n### S_alm2[%d][%d] ### \n",Gc_AN,L1);fflush(stdout);
	    for (i=0; i<Mmax1; i++){
	      for (j=0; j<Mmax1; j++){
		printf("%9.5f",S_alm2[0][i+1][j+1]);fflush(stdout);
	      } /* j */
	      printf("\n");fflush(stdout);
	    } /* j */

	    printf("\n");fflush(stdout);

	    printf("### P_alm2[%d][%d] ### \n",Gc_AN,L1);fflush(stdout);
	    for (i=0; i<Mmax1; i++){
	      for (j=0; j<Mmax1; j++){
		printf("%9.5f",P_alm2[0][i+1][j+1]);fflush(stdout);
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

	  Eigen_lapack(S_alm2[0],W_alm2[0],Mmax1,Mmax1);

	  if (outputlev==1){
	    for (k=1; k<=Mmax1; k++){
	      printf("k W %2d %15.12f\n",k,W_alm2[0][k]);
	    }
	  }

	  /* check ill-conditioned eigenvalues */

	  for (k=1; k<=Mmax1; k++){
	    if (W_alm2[0][k]<1.0e-14) {
	      printf("Found ill-conditioned eigenvalues (6)\n");
	      printf("Stopped calculation\n");
	      exit(1);
	    }
	  }

	  for (k=1; k<=Mmax1; k++){
	    temp_V1[k] = 1.0/sqrt(W_alm2[0][k]);
	  }

	  /***  Calculations of eigenvalues  ***/

	  for (i=1; i<=Mmax1; i++){
	    for (j=1; j<=Mmax1; j++){
	      sum = 0.0;
	      for (k=1; k<=Mmax1; k++){
		sum = sum + P_alm2[0][i][k]*S_alm2[0][k][j]*temp_V1[j];
	      }
	      N_alm2[0][i][j] = sum;
	    }
	  }

	  for (i=1; i<=Mmax1; i++){
	    for (j=1; j<=Mmax1; j++){
	      sum = 0.0;
	      for (k=1; k<=Mmax1; k++){
		sum = sum + temp_V1[i]*S_alm2[0][k][i]*N_alm2[0][k][j];
	      }
	      temp_M1[0][i][j] = sum;
	    }
	  }

	  if (outputlev==1){
	    printf("\n ##### TEST 1 ##### \n");
	    for (i=1; i<=Mmax1; i++){
	      for (j=1; j<=Mmax1; j++){
		printf("%9.5f",temp_M1[0][i][j]);
	      }
	      printf("\n");
	    }
	  }

	  Eigen_lapack(temp_M1[0],W_alm2[0],Mmax1,Mmax1);

	  if (outputlev==1){
	    printf("\n ##### TEST 2 ##### \n");
	    for (k=1; k<=Mmax1; k++){
	      printf("k W %2d %9.5f\n",k,W_alm2[0][k]);
	    }

	    printf("\n ##### TEST 3 ##### \n");
	    for (i=1; i<=Mmax1; i++){
	      for (j=1; j<=Mmax1; j++){
		printf("%9.5f",temp_M1[0][i][j]);
	      }
	      printf("\n");
	    }
	    printf("\n");
	  }

	  /***  Transformation to the original eigen vectors  ***/

	  for (i=1; i<=Mmax1; i++){
	    for (j=1; j<=Mmax1; j++){
	      N_alm2[0][i][j] = 0.0;
	    }
	  }

	  for (i=1; i<=Mmax1; i++){
	    for (j=1; j<=Mmax1; j++){
	      sum = 0.0;
	      for (k=1; k<=Mmax1; k++){
		sum = sum + S_alm2[0][i][k]*temp_V1[k]*temp_M1[0][k][j];
	      }
	      N_alm2[0][i][j] = sum;
	    }
	  }

	  /* printing out eigenvalues and the eigenvectors */

	  if (outputlev==1){
	    for (i=1; i<=Mmax1; i++){
	      printf("%ith eigenvalue of HC=eSC: %15.12f\n",i,W_alm2[0][i]);
	    }

	    for (i=1; i<=Mmax1; i++){
	      printf("%ith eigenvector: ",i);
	      printf("{");
	      for (j=1; j<=Mmax1; j++){
		printf("%7.4f,",N_alm2[0][i][j]);
	      }
	      printf("}\n");
	    }

	    printf("\n");
	  }

	  /***  Selection of NMB & NRB orbitals (2)  ***/

	  for (M1=1; M1<=L1*2+1; M1++){
	    posi1 = Posi_alm[Gc_AN][L1][M1];
	    for (i=0; i<Mmax1; i++){
	      W_full[0][posi1+i] = W_alm2[0][i+1];
	      for (j=0; j<Mmax1; j++){
		N_tran2[0][posi1+i][posi1+j] = N_alm2[0][i+1][j+1];
	      } /* j */
	    } /* i */
	  } /* M1 */

	} /* L1 */
      } /* Gc_AN */

      if (outputlev==1){
	printf("### N_red ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",N_tran2[0][i][j]);fflush(stdout);
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
        tmp_ele1 += S_full[0][i][k] * N_red[0][k][j];
        tmp_ele2 += P_full[0][i][k] * N_red[0][k][j];
        tmp_ele3 += T_full[0][i][k] * N_red[0][k][j];
	} 
        temp_M1[0][i][j] = tmp_ele1;
        temp_M2[0][i][j] = tmp_ele2;
        temp_M3[0][i][j] = tmp_ele3;
	} 
	} 

	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
        tmp_ele2 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += N_red[0][k][i] * temp_M1[0][k][j];
        tmp_ele2 += N_red[0][k][i] * temp_M2[0][k][j];
	} 
        S_full[0][i][j] = tmp_ele1;
        P_full[0][i][j] = tmp_ele2;
        T_full[0][i][j] = temp_M3[0][i][j];
	} 
	} 
      */

      /* BLAS */
      /*  N_red * S * N_red  */
      if(1>0){
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V1[tnum] = N_tran2[0][i][j];
	    temp_V2[tnum] = S_full[0][i][j];
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
	    S_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  N_red * P * N_red  */
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = P_full[0][i][j];
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
	    P_full[0][i][j] = temp_V2[tnum];
	    tnum++;
	  }
	}

	/*  T * N_red  */

	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V2[tnum] = T_full[0][j][i];
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
	    T_full[0][j][i] = temp_V3[tnum];
	    tnum++;
	  }
	}
      }
      else{
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_M4[0][j][i] = N_tran2[0][i][j];
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0; tmp_ele21 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0; tmp_ele22 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0; tmp_ele23 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M4[0][j][k];
	      tmp_ele1 = temp_M4[0][j][k+1];
	      tmp_ele2 = temp_M4[0][j][k+2];
	      tmp_ele3 = temp_M4[0][j][k+3];

	      tmp_ele00 += S_full[0][i][k]   * tmp_ele0;
	      tmp_ele01 += S_full[0][i][k+1] * tmp_ele1;
	      tmp_ele02 += S_full[0][i][k+2] * tmp_ele2;
	      tmp_ele03 += S_full[0][i][k+3] * tmp_ele3;

	      tmp_ele10 += P_full[0][i][k]   * tmp_ele0;
	      tmp_ele11 += P_full[0][i][k+1] * tmp_ele1;
	      tmp_ele12 += P_full[0][i][k+2] * tmp_ele2;
	      tmp_ele13 += P_full[0][i][k+3] * tmp_ele3;

	      tmp_ele20 += T_full[0][i][k]   * tmp_ele0;
	      tmp_ele21 += T_full[0][i][k+1] * tmp_ele1;
	      tmp_ele22 += T_full[0][i][k+2] * tmp_ele2;
	      tmp_ele23 += T_full[0][i][k+3] * tmp_ele3;
	    }
	    temp_M1[0][j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    temp_M2[0][j][i] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	    temp_M3[0][i][j] = tmp_ele20 + tmp_ele21 + tmp_ele22 + tmp_ele23;
	  }
	}

	is = SizeMat - SizeMat%4;

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0; tmp_ele20 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M4[0][j][k];
	      tmp_ele00 += S_full[0][i][k] * tmp_ele0;
	      tmp_ele10 += P_full[0][i][k] * tmp_ele0;
	      tmp_ele20 += T_full[0][i][k] * tmp_ele0;
	    }
	    temp_M1[0][j][i] += tmp_ele00;
	    temp_M2[0][j][i] += tmp_ele10;
	    temp_M3[0][i][j] += tmp_ele20;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    tmp_ele01 = 0.0; tmp_ele11 = 0.0;
	    tmp_ele02 = 0.0; tmp_ele12 = 0.0;
	    tmp_ele03 = 0.0; tmp_ele13 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele0 = temp_M4[0][i][k];
	      tmp_ele1 = temp_M4[0][i][k+1];
	      tmp_ele2 = temp_M4[0][i][k+2];
	      tmp_ele3 = temp_M4[0][i][k+3];

	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele01 += tmp_ele1 * temp_M1[0][j][k+1];
	      tmp_ele02 += tmp_ele2 * temp_M1[0][j][k+2];
	      tmp_ele03 += tmp_ele3 * temp_M1[0][j][k+3];

	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	      tmp_ele11 += tmp_ele1 * temp_M2[0][j][k+1];
	      tmp_ele12 += tmp_ele2 * temp_M2[0][j][k+2];
	      tmp_ele13 += tmp_ele3 * temp_M2[0][j][k+3];
	    }
	    S_full[0][i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	    P_full[0][i][j] = tmp_ele10 + tmp_ele11 + tmp_ele12 + tmp_ele13;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; tmp_ele10 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele0 = temp_M4[0][i][k];
	      tmp_ele00 += tmp_ele0 * temp_M1[0][j][k];
	      tmp_ele10 += tmp_ele0 * temp_M2[0][j][k];
	    }
	    S_full[0][i][j] += tmp_ele00;
	    P_full[0][i][j] += tmp_ele10;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    T_full[0][i][j] = temp_M3[0][i][j];
	  }
	}
      }

      /* ### TEST ### */
      if (outputlev==1){
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele1 = 0.0;
	    for (k=0; k<SizeMat; k++){
	      tmp_ele1 += S_full0[0][i][k] * T_full[0][k][j];
	    } /* k */
	    temp_M1[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele1 = 0.0;
	    for (k=0; k<SizeMat; k++){
	      tmp_ele1 += T_full[0][k][i] * temp_M1[0][k][j];
	    } /* k */
	    temp_M3[0][i][j] = tmp_ele1;
	  } /* j */
	} /* i */

	printf("\n ### S_full (TEST 4) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",temp_M3[0][i][j]);fflush(stdout);
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

      if (myid == Host_ID) printf("<< 3-6 >> Energy of each NAO \n");
      if (measure_time==1) dtime(&Stime1);
      /*
	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += H_full0[0][i][k] * T_full[0][k][j];
	} 
        temp_M1[0][i][j] = tmp_ele1;
	} 
	} 

	for (i=0; i<SizeMat; i++){
	for (j=0; j<SizeMat; j++){
        tmp_ele1 = 0.0;
	for (k=0; k<SizeMat; k++){
        tmp_ele1 += T_full[0][k][i] * temp_M1[0][k][j];
	} 
        temp_M3[0][i][j] = tmp_ele1;
        H_full[0][i][j] = temp_M3[0][i][j];
	} 
	} 
      */

      /* BLAS */
      if(1>0){
	tnum = 0;
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_V1[tnum] = T_full[0][j][i];
	    temp_V2[tnum] = H_full0[0][i][j];
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
	    H_full[0][i][j]  = temp_V2[tnum];
	    tnum++;
	  }
	}
      }
      else{
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    temp_M2[0][j][i] = T_full[0][i][j];
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0;
	    tmp_ele01 = 0.0;
	    tmp_ele02 = 0.0;
	    tmp_ele03 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele00 += H_full0[0][i][k]   * temp_M2[0][j][k];
	      tmp_ele01 += H_full0[0][i][k+1] * temp_M2[0][j][k+1];
	      tmp_ele02 += H_full0[0][i][k+2] * temp_M2[0][j][k+2];
	      tmp_ele03 += H_full0[0][i][k+3] * temp_M2[0][j][k+3];
	    } 
	    temp_M1[0][j][i] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	  } 
	} 

	is = SizeMat - SizeMat%4;

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0; 
	    for (k=is; k<SizeMat; k++){
	      tmp_ele00 += H_full0[0][i][k] * temp_M2[0][j][k];
	    } 
	    temp_M1[0][j][i] += tmp_ele00;
	  }
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0;
	    tmp_ele01 = 0.0;
	    tmp_ele02 = 0.0;
	    tmp_ele03 = 0.0;
	    for (k=0; k<(SizeMat-3); k+=4){
	      tmp_ele00 += temp_M2[0][i][k]   * temp_M1[0][j][k];
	      tmp_ele01 += temp_M2[0][i][k+1] * temp_M1[0][j][k+1];
	      tmp_ele02 += temp_M2[0][i][k+2] * temp_M1[0][j][k+2];
	      tmp_ele03 += temp_M2[0][i][k+3] * temp_M1[0][j][k+3];
	    } 
	    H_full[0][i][j] = tmp_ele00 + tmp_ele01 + tmp_ele02 + tmp_ele03;
	  } 
	}

	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    tmp_ele00 = 0.0;
	    for (k=is; k<SizeMat; k++){
	      tmp_ele00 += temp_M2[0][i][k] * temp_M1[0][j][k];
	    }
	    H_full[0][i][j] += tmp_ele00;
	  }
	}
      } 

      if (outputlev==1){
	/*
	  printf("\n ### T^t * H_full * T (TEST 4) ### \n");fflush(stdout);
	  for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	  printf("%9.5f",H_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	  }
	*/
	/*
	  printf("\n ### S_full (10) ### \n");fflush(stdout);
	  for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	  printf("%9.5f",S_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	  }
	*/
	/*
	  printf("\n ### P_full (10) ### \n");fflush(stdout);
	  for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	  printf("%9.5f",P_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	  }
	*/
	printf("\n ### T_full (10) ### \n");fflush(stdout);
	for (i=0; i<SizeMat; i++){
	  for (j=0; j<SizeMat; j++){
	    printf("%9.5f",T_full[0][i][j]);fflush(stdout);
	  }
	  printf("\n");
	}

      }
      printf("\n");fflush(stdout);


      if (measure_time==1){
	dtime(&Etime1);
	time10 = Etime1 - Stime1;
      }

      double *NAP;
      NAP = (double*)malloc(sizeof(double)*(atomnum+1));

      k=0;
      tmp_ele2 = 0.0;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	leng1 = Leng_a[Gc_AN];
	tmp_ele1 = 0.0;
	for (i=0; i<leng1; i++){
	  tmp_ele1 += P_full[0][k+i][k+i];
	} /* i */
	k += leng1;
	NAP[Gc_AN] = tmp_ele1;
	tmp_ele2 += tmp_ele1;
	printf("%4d %3s : %12.8f \n",Gc_AN,SpeName[wan1],tmp_ele1);fflush(stdout);
      } /* Gc_AN */
      printf("------------------------\n");fflush(stdout);
      printf("  Total  : %12.8f \n",tmp_ele2);fflush(stdout);
      printf("\n");fflush(stdout);


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


      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        k=0;
        wan1 = WhatSpecies[Gc_AN];

        printf(" ## Global atom num.: %d ( %s ) / NP = %8.4f\n",
               Gc_AN,SpeName[wan1],NAP[Gc_AN]);fflush(stdout);
        printf("-----------------");
        for (i=0; i<Leng_a[Gc_AN]; i++){
          printf("--------");
        }
        printf("\n");fflush(stdout);

        printf(" NP in NAO       ");fflush(stdout);
        for (i=0; i<Leng_a[Gc_AN]; i++){
          printf("%8.4f",P_full[0][Posi_a[Gc_AN]+i][Posi_a[Gc_AN]+i]);fflush(stdout);
        }
        printf("\n");fflush(stdout);

        printf(" Energy (Hartree)");fflush(stdout);
        for (i=0; i<Leng_a[Gc_AN]; i++){
          printf("%8.4f",H_full[0][Posi_a[Gc_AN]+i][Posi_a[Gc_AN]+i]);fflush(stdout);
        }
        printf("\n");fflush(stdout);
        printf("-----------------");
        for (i=0; i<Leng_a[Gc_AN]; i++){
          printf("--------");
        }
        printf("\n");fflush(stdout);

	for (L1=0; L1<=Spe_MaxL_Basis[wan1]; L1++){
	  leng1 = Leng_alm[Gc_AN][L1];
	  for (M1=1; M1<=2*L1+1; M1++){
	    for (i=1; i<=leng1; i++){
	      printf(" %s %3s   ",Name_Multiple[i],Name_Angular[L1][M1-1]);fflush(stdout);
	      for (j=0; j<Leng_a[Gc_AN]; j++){
		printf("%8.4f",T_full[0][Posi_a[Gc_AN]+k][Posi_a[Gc_AN]+j]);fflush(stdout);
	      } /* j */
	      k++;
	      printf("\n");fflush(stdout);
	    } /* i */
	  } /* M1 */
	} /* L1 */
	printf("\n");fflush(stdout);
      } /* Gc_AN */
      printf("\n");fflush(stdout);

      free(NAP);


      if(NAO_only==0){

	printf("******************************************\n");
	printf("           NATURAL HYBRID ORBITAL         \n");
	printf("******************************************\n");

	/**  Allocation of arrays **/

	int *Num_LP,*Num_NHO,Num_LonePair=0,Num_NBO;
	int **Table_NBO,**Table_LP,**NHO_indexL2G;
	int Gb1_AN,Gb2_AN,Ga1_AN,Ga2_AN,Nc,GN;
	int posi[5],t_posi[5],leng[5],t_leng,m,n;
	double ***pLP_vec,***pNHO_vec,**Bond_Pair_vec;
	double ****pLP_mat,**pNHO_mat,**X_mat,***NHO_mat,****P_pair;
	double **NBO_bond_vec,**NBO_anti_vec,**LonePair_vec,Fij[2][2];
	double *Pop_bond_NBO,*Pop_anti_NBO,*Pop_LonePair;
	double ***temp_M8,***temp_M9,***temp_M10;

	Num_LP = (int*)malloc(sizeof(int)*(atomnum+1));
	for (i=0; i<=atomnum; i++){
	  Num_LP[i] = 0;
	}

	Num_NHO = (int*)malloc(sizeof(int)*(atomnum+1));
	for (i=0; i<=atomnum; i++){
	  Num_NHO[i] = 0;
	}

	/*******************************************************************

           Table_NBO[i] : Table for i th bonding and anti-bonding NBO's

           Table_NBO[i][0] : atom-1 (global number)
           Table_NBO[i][1] : atom-2 (global number)
           Table_NBO[i][2] : index of NHO on atom-1
           Table_NBO[i][3] : index of NHO on atom-2

	*********************************************************************/

	Table_NBO = (int**)malloc(sizeof(int*)*(atomnum*10+1));
	for (i=0; i<=atomnum*10; i++){
	  Table_NBO[i] = (int*)malloc(sizeof(int)*(6));
	  for (j=0; j<=5; j++){
	    Table_NBO[i][j] = 0;
	  }
	}

	/*******************************************************************

           Table_LP[i] : Table for i th LP's
           Table_LP[i][0] : atom (global number)

	*********************************************************************/

	Table_LP = (int**)malloc(sizeof(int*)*(atomnum*5+1));
	for (i=0; i<=atomnum*5; i++){
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

	Pop_bond_NBO = (double*)malloc(sizeof(double)*(atomnum*10+1));
	for (i=0; i<=atomnum*10; i++){
	  Pop_bond_NBO[i] = 0.0;
	}

	Pop_anti_NBO = (double*)malloc(sizeof(double)*(atomnum*10+1));
	for (i=0; i<=atomnum*10; i++){
	  Pop_anti_NBO[i] = 0.0;
	}

	Pop_LonePair = (double*)malloc(sizeof(double)*(atomnum*10+1));
	for (i=0; i<=atomnum*10; i++){
	  Pop_LonePair[i] = 0.0;
	}

	pLP_vec = (double***)malloc(sizeof(double**)*(atomnum+1));
	for (i=0; i<=atomnum; i++){
	  pLP_vec[i] = (double**)malloc(sizeof(double*)*(MaxLeng_a+1));
	  for (j=0; j<=MaxLeng_a; j++){
	    pLP_vec[i][j] = (double*)malloc(sizeof(double)*(MaxLeng_a+1));
	    for (k=0; k<=MaxLeng_a; k++){
	      pLP_vec[i][j][k] = 0.0;
	    }
	  }
	}

	pNHO_vec = (double***)malloc(sizeof(double**)*(atomnum+1));
	for (i=0; i<=atomnum; i++){
	  pNHO_vec[i] = (double**)malloc(sizeof(double*)*(MaxLeng_a+1));
	  for (j=0; j<=MaxLeng_a; j++){
	    pNHO_vec[i][j] = (double*)malloc(sizeof(double)*(MaxLeng_a+1));
	    for (k=0; k<=MaxLeng_a; k++){
	      pNHO_vec[i][j][k] = 0.0;
	    }
	  }
	}

	pLP_mat = (double****)malloc(sizeof(double***)*(atomnum+1));
	for (i=0; i<=atomnum; i++){
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

	X_mat = (double**)malloc(sizeof(double*)*(MaxLeng_a+1));
	for (i=0; i<=MaxLeng_a; i++){
	  X_mat[i] = (double*)malloc(sizeof(double)*(MaxLeng_a+1));
	  for (j=0; j<=MaxLeng_a; j++){
	    X_mat[i][j] = 0.0;
	  }
	}

	NHO_mat = (double***)malloc(sizeof(double**)*(atomnum+1));
	for (i=0; i<=atomnum; i++){
	  NHO_mat[i] = (double**)malloc(sizeof(double*)*(MaxLeng_a+1));
	  for (j=0; j<=MaxLeng_a; j++){
	    NHO_mat[i][j] = (double*)malloc(sizeof(double)*(MaxLeng_a+1));
	    for (k=0; k<=MaxLeng_a; k++){
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

	NBO_bond_vec = (double**)malloc(sizeof(double*)*(atomnum*10+1));
	for (i=0; i<=atomnum*10; i++){
	  NBO_bond_vec[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
	  for (j=0; j<=MaxLeng_a*2; j++){
	    NBO_bond_vec[i][j] = 0.0;
	  }
	}

	NBO_anti_vec = (double**)malloc(sizeof(double*)*(atomnum*10+1));
	for (i=0; i<=atomnum*10; i++){
	  NBO_anti_vec[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
	  for (j=0; j<=MaxLeng_a*2; j++){
	    NBO_anti_vec[i][j] = 0.0;
	  }
	}

	LonePair_vec = (double**)malloc(sizeof(double*)*(atomnum*5+1));
	for (i=0; i<=atomnum*5; i++){
	  LonePair_vec[i] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
	  for (j=0; j<=MaxLeng_a*2; j++){
	    LonePair_vec[i][j] = 0.0;
	  }
	}

	P_pair = (double****)malloc(sizeof(double***)*(atomnum+1));
	for (i=0; i<=atomnum; i++){
	  P_pair[i] = (double***)malloc(sizeof(double**)*(atomnum+1));
	  for (j=0; j<=atomnum; j++){
	    P_pair[i][j] = (double**)malloc(sizeof(double*)*(MaxLeng_a*2+1));
	    for (k=0; k<=MaxLeng_a*2; k++){
	      P_pair[i][j][k] = (double*)malloc(sizeof(double)*(MaxLeng_a*2+1));
	      for (l=0; l<=MaxLeng_a*2; l++){
		P_pair[i][j][k][l] = 0.0;
	      }
	    }
	  }
	}

	temp_M8 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	  temp_M8[spin] = (double**)malloc(sizeof(double*)*(MaxLeng_a*5+1));
	  for (i=0; i<=MaxLeng_a*5; i++){
	    temp_M8[spin][i] = (double*)malloc(sizeof(double)*(MaxLeng_a*5+1));
	    for (j=0; j<=MaxLeng_a*5; j++){
	      temp_M8[spin][i][j] = 0.0;
	    }
	  }
	}

	temp_M9 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	  temp_M9[spin] = (double**)malloc(sizeof(double*)*(MaxLeng_a*5+1));
	  for (i=0; i<=MaxLeng_a*5; i++){
	    temp_M9[spin][i] = (double*)malloc(sizeof(double)*(MaxLeng_a*5+1));
	    for (j=0; j<=MaxLeng_a*5; j++){
	      temp_M9[spin][i][j] = 0.0;
	    }
	  }
	}

	temp_M10 = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
	for (spin=0; spin<=SpinP_switch; spin++){
	  temp_M10[spin] = (double**)malloc(sizeof(double*)*(MaxLeng_a*5+1));
	  for (i=0; i<=MaxLeng_a*5; i++){
	    temp_M10[spin][i] = (double*)malloc(sizeof(double)*(MaxLeng_a*5+1));
	    for (j=0; j<=MaxLeng_a*5; j++){
	      temp_M10[spin][i][j] = 0.0;
	    }
	  }
	}


	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  wan1 = WhatSpecies[Gc_AN];
	  leng1 = Leng_a[Gc_AN];
	  posi1 = Posi_a[Gc_AN];
	  tnum = 0;

	  /** (1) Diagonalization of atom-block density matrices **/

	  for (i=0; i<leng1; i++){
	    for (j=0; j<leng1; j++){
	      temp_M1[0][i+1][j+1] = P_full[0][i+posi1][j+posi1];
	    } /* j */
	  } /* i */

	  /*
	    printf("### Density Matrix of Atom %d ###\n",Gc_AN);fflush(stdout);
	    for (i=0; i<leng1; i++){
	    for (j=0; j<leng1; j++){
	    printf("%9.5f",temp_M1[0][i+1][j+1]);fflush(stdout);
	    } 
	    printf("\n");fflush(stdout);
	    } 
	    printf("\n");fflush(stdout);
	  */

	  Eigen_lapack(temp_M1[0],temp_V1,leng1,leng1);

	  if(1>0){
	    for (i=0; i<leng1; i++){
	      printf("## atom-DM ev.(Gc_AN=%d): %9.5f \n",Gc_AN,temp_V1[i+1]);
	    }
	    printf("\n");
	  }
  
	  /** (2) Finding lone pairs **/

	  for (k=0; k<leng1; k++){
	    if (temp_V1[k+1] >= 1.77/(1.0+SpinP_switch)){
	      tnum++;

	      for (i=0; i<leng1; i++){
		pLP_vec[Gc_AN][tnum][i] = temp_M1[0][i+1][k+1];
	      }

	      for (i=0; i<leng1; i++){
		for (j=0; j<leng1; j++){
		  pLP_mat[Gc_AN][tnum][i][j] = temp_V1[k+1] 
		    * pLP_vec[Gc_AN][tnum][i] 
		    * pLP_vec[Gc_AN][tnum][j];
		}
	      }

	    }
	  }
	  Num_LP[Gc_AN] = tnum;

	  /*
	    if (Num_LP[Gc_AN] != 0){
	    printf("### Lone-pair vector (atom: %d) ###\n",Gc_AN);fflush(stdout);
	    for (k=1; k<=Num_LP[Gc_AN]; k++){
	    for (i=0; i<leng1; i++){
	    printf("%9.5f\n",pLP_vec[Gc_AN][k][i]);fflush(stdout);
	    }
	    printf("\n");fflush(stdout);
	    }
	    printf("\n");fflush(stdout);

	    printf("### Density Matrix for Lone-Pair (atom: %d) ###\n",Gc_AN);fflush(stdout);
	    for (k=1; k<=Num_LP[Gc_AN]; k++){
	    for (i=0; i<leng1; i++){
	    for (j=0; j<leng1; j++){
	    printf("%9.5f",pLP_mat[Gc_AN][k][i][j]);fflush(stdout);
	    } 
	    printf("\n");fflush(stdout);
	    } 
	    printf("\n");fflush(stdout);
	    } 
	    }
	    else {
	    printf("### Lone-pair not fouond on atom %d ###\n",Gc_AN);fflush(stdout);
	    }
	  */

	} /* Gc_AN */ 

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  Num_LonePair += Num_LP[Gc_AN];
	}


	/** (3) Construction of atom-pair density matrix without LP's **/

	Num_NBO = 0;

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  wan1 = WhatSpecies[Gc_AN];
	  leng1 = Leng_a[Gc_AN];
	  posi1 = Posi_a[Gc_AN];
	  tnum = 0;

	  for (i=0; i<leng1; i++){
	    for (j=0; j<leng1; j++){
	      temp_M1[0][i+1][j+1] = P_full[0][i+posi1][j+posi1];
	      if (Num_LP[Gc_AN] != 0){
		for (k=1; k<=Num_LP[Gc_AN]; k++){
		  temp_M1[0][i+1][j+1] -= pLP_mat[Gc_AN][k][i][j];
		}
	      }
	    }
	  }

	  for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];

	    if (Gh_AN > Gc_AN){
	      wan2 = WhatSpecies[Gh_AN];
	      leng2 = Leng_a[Gh_AN];
	      posi2 = Posi_a[Gh_AN];

	      for (i=0; i<leng2; i++){
		for (j=0; j<leng2; j++){
		  temp_M1[0][leng1+i+1][leng1+j+1] = P_full[0][i+posi2][j+posi2];
		  if (Num_LP[Gh_AN] != 0){
		    for (k=1; k<=Num_LP[Gh_AN]; k++){
		      temp_M1[0][leng1+i+1][leng1+j+1] -= pLP_mat[Gh_AN][k][i][j];
		    }
		  }
		}
	      }

	      for (i=0; i<leng2; i++){
		for (j=0; j<leng1; j++){
		  temp_M1[0][leng1+i+1][j+1] = P_full[0][i+posi2][j+posi1];
		  temp_M1[0][j+1][leng1+i+1] = P_full[0][j+posi1][i+posi2];
		}
	      }

	      for (i=0; i<leng1+leng2; i++){
		for (j=0; j<leng1+leng2; j++){
		  temp_M2[0][i+1][j+1]       = temp_M1[0][i+1][j+1];
		  P_pair[Gc_AN][Gh_AN][i][j] = temp_M1[0][i+1][j+1];
		}
	      }

	      /*
		printf("### Density Matrix of (%d,%d) Atom Pair w/o LP's ###\n",
		Gc_AN,Gh_AN);fflush(stdout);
		for (i=0; i<leng1+leng2; i++){
		for (j=0; j<leng1+leng2; j++){
		printf("%9.5f",temp_M1[0][i+1][j+1]);fflush(stdout);
		} 
		printf("\n");fflush(stdout);
		} 
		printf("\n");fflush(stdout);
	      */

	      /** (5) Orthogonalization of atom-pair matrices **/

	      Eigen_lapack(temp_M2[0],temp_V1,leng1+leng2,leng1+leng2);

	      if (outputlev==0){
		printf("### pNHO without LP's (%d,%d) ###\n",Gc_AN,Gh_AN);fflush(stdout);
		for (k=1; k<=leng1+leng2; k++){
		  printf("k W %2d %15.12f\n",k,temp_V1[k]);
		}
		printf("\n");fflush(stdout);
	      }

	      if (outputlev==1){
		printf("### pNHO without LP's (%d,%d) ###\n",Gc_AN,Gh_AN);fflush(stdout);

		for (k=1; k<=leng1+leng2; k++){
		  printf("k W (1) %2d %15.12f\n",k,temp_V1[k]);
		}
		printf("\n");fflush(stdout);

		for (i=0; i<leng1+leng2; i++){
		  for (j=0; j<leng1+leng2; j++){
		    printf("%9.5f",temp_M2[0][i+1][j+1]);fflush(stdout);
		  } /* j */
		  printf("\n");fflush(stdout);
		} /* i */
		printf("\n");fflush(stdout);
	      }

	      /** (6) Finding doublly occupied states **/

	      for (k=0; k<leng1+leng2; k++){
		if (temp_V1[k+1] >= 1.80/(1.0+(double)SpinP_switch)){
		  Num_NBO++;
		  Num_NHO[Gc_AN]++;  tnum = Num_NHO[Gc_AN];
		  Num_NHO[Gh_AN]++;  pnum = Num_NHO[Gh_AN];
		  Table_NBO[Num_NBO][0] = Gc_AN;
		  Table_NBO[Num_NBO][1] = Gh_AN;
		  Table_NBO[Num_NBO][2] = tnum;
		  Table_NBO[Num_NBO][3] = pnum;

		  for (i=0; i<leng1; i++){
		    pNHO_vec[Gc_AN][tnum][i]= temp_M2[0][i+1][k+1];
		  }

		  for (i=0; i<leng2; i++){
		    pNHO_vec[Gh_AN][pnum][i]= temp_M2[0][i+leng1+1][k+1];
		  }

		}
	      }

	    } /* if (Gh_AN > Gc_AN) */
	  } /* h_AN */
	} /* Gc_AN */

	Total_Num_NBO = Num_NBO;

	printf("### Table of NBO ###\n");fflush(stdout);
	for (i=1; i<=Num_NBO; i++){
	  printf(" NBO: %d | atom1: %d (%d) | atom2: %d (%d)\n",
		 i,Table_NBO[i][0],Table_NBO[i][2],Table_NBO[i][1],Table_NBO[i][3]);
	}
	printf("\n");fflush(stdout);

	printf("### Number of NHO's & LP's ###\n");fflush(stdout);
	for (i=1; i<=atomnum; i++){
	  printf("  %d :  %d  %d\n",i,Num_NHO[i],Num_LP[i]);fflush(stdout);
	}
	printf("\n");fflush(stdout);

	/** (7) Symmetrically orthogonalization of pre-NHOs in each atom **/

	tnum = 0; /* for counting the num. of LP's in system */

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  wan1 = WhatSpecies[Gc_AN];
	  leng1 = Leng_a[Gc_AN];
	  leng2 = Num_NHO[Gc_AN];
	  leng3 = Num_LP[Gc_AN];

	  /* construction of pNHO matrix C */

	  for (i=0; i<leng1; i++){
	    for (j=1; j<=leng2; j++){
	      pNHO_mat[i][j-1] = pNHO_vec[Gc_AN][j][i];
	    }
	  }

	  if (leng3 != 0){
	    for (i=0; i<leng1; i++){
	      for (j=1; j<=leng3; j++){
		pNHO_mat[i][leng2+j-1] = pLP_vec[Gc_AN][j][i];
	      }
	    }
	  }

	  if (outputlev==0){
	    printf("### pNHO Matrix (Gatom = %d) ###\n",Gc_AN);fflush(stdout);
	    for (i=0; i<leng1; i++){
	      for (j=0; j<leng2+leng3; j++){
		printf("%9.5f",pNHO_mat[i][j]);fflush(stdout);
	      } 
	      printf("\n");fflush(stdout);
	    } 
	    printf("\n");fflush(stdout);
	  }

	  /* overlap matrix of pNHO */

	  for (i=0; i<leng2+leng3; i++){
	    for (j=0; j<leng2+leng3; j++){
	      tmp_ele1 = 0.0;
	      for (k=0; k<leng1; k++){
		tmp_ele1 += pNHO_mat[k][i] * pNHO_mat[k][j];
	      }
	      temp_M2[0][i+1][j+1] = tmp_ele1;
	    }
	  }

	  /*
	    printf("### Overlap Matrix of pNHO (Gatom = %d) ###\n",Gc_AN);fflush(stdout);
	    for (i=0; i<leng2+leng3; i++){
	    for (j=0; j<leng2+leng3; j++){
	    printf("%9.5f",temp_M2[0][i+1][j+1]);fflush(stdout);
	    } 
	    printf("\n");fflush(stdout);
	    } 
	    printf("\n");fflush(stdout);
	  */

	  /* diagonalization of overlap matrix */

	  Eigen_lapack(temp_M2[0],temp_V1,leng2+leng3,leng2+leng3);

	  if (outputlev==1){
	    for (k=1; k<=leng2+leng3; k++){
	      printf("k W (2) %2d %15.12f\n",k,temp_V1[k]);
	    }
	    printf("\n");fflush(stdout);
	  }

	  /* check ill-conditioned eigenvalues */

	  for (k=1; k<=leng2+leng3; k++){
	    if (temp_V1[k]<1.0e-14) {
	      printf("Found ill-conditioned eigenvalues (7)\n");
	      printf("Stopped calculation\n");
	      exit(1);
	    }
	  }

	  /* construction of transfer matrix X */

	  for (k=1; k<=leng2+leng3; k++){
	    temp_V1[k] = 1.0/sqrt(temp_V1[k]);
	  }

	  for (i=1; i<=leng2+leng3; i++){
	    for (j=1; j<=leng2+leng3; j++){
	      temp_M1[0][i][j] = temp_V1[i] * temp_M2[0][j][i];
	    }
	  }

	  for (i=1; i<=leng2+leng3; i++){
	    for (j=1; j<=leng2+leng3; j++){
	      tmp_ele1 = 0.0;
	      for (k=1; k<=leng2+leng3; k++){
		tmp_ele1 += temp_M2[0][i][k] * temp_M1[0][k][j];
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
	      NHO_mat[Gc_AN][i][j] = tmp_ele1;
	    }
	  } 

	  /* extraction of LP vector */

	  if (leng3 != 0){
	    for (j=1; j<=leng3; j++){
	      tnum++;
	      for (i=0; i<leng1; i++){
		LonePair_vec[tnum][i] = NHO_mat[Gc_AN][i][j+leng2-1];
		Table_LP[tnum][0] = Gc_AN;
	      }
	    }
	  }

	  /* overlap matrix of NHO */

	  for (i=0; i<leng2+leng3; i++){
	    for (j=0; j<leng2+leng3; j++){
	      tmp_ele1 = 0.0;
	      for (k=0; k<leng1; k++){
		tmp_ele1 += NHO_mat[Gc_AN][k][i] * NHO_mat[Gc_AN][k][j];
	      }
	      temp_M2[0][i+1][j+1] = tmp_ele1;
	    }
	  }


	  if (outputlev==1){
	    printf("### Overlap Matrix of NHO (Gatom = %d) ###\n",Gc_AN);fflush(stdout);
	    for (i=0; i<leng2+leng3; i++){
	      for (j=0; j<leng2+leng3; j++){
		printf("%9.5f",temp_M2[0][i+1][j+1]);fflush(stdout);
	      } 
	      printf("\n");fflush(stdout);
	    } 
	    printf("\n");fflush(stdout);
	  }

	  /* NHO */

	  if (outputlev==1){
	    printf("### NHO Matrix (Gatom = %d) ###\n",Gc_AN);fflush(stdout);
	    for (i=0; i<leng1; i++){
	      for (j=0; j<leng2+leng3; j++){
		printf("%9.5f",NHO_mat[Gc_AN][i][j]);fflush(stdout);
	      } 
	      printf("\n");fflush(stdout);
	    } 
	    printf("\n");fflush(stdout);
	  }

	} /* Gc_AN */


	/* Cuber-data output of NHO */

	printf("### Cube-data of NHOs ###\n\n");fflush(stdout);

	tnum = -1;
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  leng1 = Leng_a[Gc_AN];
	  leng2 = Num_NHO[Gc_AN];
	  leng3 = Num_LP[Gc_AN];
	  posi1 = Posi_a[Gc_AN];

	  for (j=0; j<leng2+leng3; j++){
	    tnum++;
	    pnum = -1;
	    for (i=0; i<SizeMat; i++){
	      tmp_ele1 = 0.0;
	      Gh_AN = Posi2Gatm[i];
	      if (i==0) Gb_AN = Gh_AN;

	      if (Gh_AN == Gb_AN) pnum++;
	      if (Gh_AN != Gb_AN) pnum = 0;

	      for (k=0; k<leng1; k++){
		tmp_ele1 += T_full[0][i][posi1+k] * NHO_mat[Gc_AN][k][j];
	      }

	      NHOs_Coef[spin0][tnum][Gh_AN][pnum] = tmp_ele1;
	      Gb_AN = Gh_AN;
	    }
	  }

	  Num_NHOs[Gc_AN] = leng2+leng3;
	} 

	Total_Num_NHO = tnum;

	/*** Rearranging order of basis ***/

	printf("# Rearranging order of basis in Cube-data #\n\n");fflush(stdout);

	for (i=0; i<=Total_Num_NHO; i++){
	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	    wan1 = WhatSpecies[Gc_AN];
	    Lmax = Spe_MaxL_Basis[wan1];
	    tnum = 0;
	    pnum = 0;

	    for (L1=0; L1<=Lmax; L1++){
	      Mulmax = Spe_Num_Basis[wan1][L1];
	      pnum = tnum;
	      tnum = 0;

	      for (j=0; j<Mulmax*(2*L1+1); j++){
		temp_V1[j] = NHOs_Coef[spin0][i][Gc_AN][j+pnum];
		tnum++;
	      }

	      if (Mulmax > 1){
		qnum = 0;
		for (k=0; k<Mulmax; k++){
		  for (j=0; j<(2*L1+1); j++){
		    temp_V2[qnum] = temp_V1[k + j*Mulmax];
		    qnum++;
		  }
		}

		for (j=0; j<Mulmax*(2*L1+1); j++){
		  NHOs_Coef[spin0][i][Gc_AN][j+pnum] = temp_V2[j];
		}
	      }

	    }
	  }
	}


	pnum = -1;
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  tnum = Num_NHO[Gc_AN];
	  for (i=1; i<=tnum; i++){
	    pnum++;
	    NHO_indexL2G[Gc_AN][i] = pnum;
	    /*
	      printf("### NHO_indexL2G[%d][%d] = %d ###\n",Gc_AN,i,pnum);fflush(stdout);
	    */
	  }
	  pnum += Num_LP[Gc_AN]; 
	}
	printf("\n");


	printf("###  Total Num. of NHOs = %d \n\n",Total_Num_NHO+1);fflush(stdout);

	printf("******************************************\n");
	printf("           NATURAL BOND ORBITAL           \n");
	printf("******************************************\n");

	for (i=1; i<=Num_NBO; i++){
	  Gc_AN = Table_NBO[i][0];
	  Gh_AN = Table_NBO[i][1];
	  tnum  = Table_NBO[i][2];
	  pnum  = Table_NBO[i][3];

	  snum = NHO_indexL2G[Gc_AN][tnum];
	  qnum = NHO_indexL2G[Gh_AN][pnum];

	  wan1 = WhatSpecies[Gc_AN];
	  wan2 = WhatSpecies[Gh_AN];

	  /*
	    printf("### NHO_index : %d , %d\n\n",snum,qnum);fflush(stdout);
	  */
	  leng1 = Leng_a[Gc_AN];
	  leng2 = Leng_a[Gh_AN];

	  /* construction of bond-pair NHO vector V_nho */

	  for (j=0; j<=MaxLeng_a*2; j++){
	    Bond_Pair_vec[j][1] = 0.0;
	    Bond_Pair_vec[j][2] = 0.0;
	  }

	  for (j=0; j<leng1; j++){
	    Bond_Pair_vec[j][1] = NHO_mat[Gc_AN][j][tnum-1];
	  }

	  for (j=0; j<leng2; j++){
	    Bond_Pair_vec[leng1+j][2] = NHO_mat[Gh_AN][j][pnum-1];
	  }

	  if(outputlev==1){
	    printf("### Bond_Pair_vector (%d,%d)(%d) ###\n",Gc_AN,Gh_AN,i);
	    for (k=0; k<leng1+leng2; k++){
	      for (j=1; j<=2; j++){
		printf("%9.5f",Bond_Pair_vec[k][j]);
	      } /* j */
	      printf("\n");
	    } /* i */
	    printf("\n");
	  }

	  /* V_nho^t * P_pair * V_nho  =>  2x2 P-matrix */

	  for (j=0; j<leng1+leng2; j++){
	    for (k=1; k<=2; k++){
	      tmp_ele1 = 0.0;
	      for (l=0; l<leng1+leng2; l++){
		tmp_ele1 += P_pair[Gc_AN][Gh_AN][j][l] * Bond_Pair_vec[l][k];
	      }
	      temp_M1[0][j][k] = tmp_ele1;
	    }
	  }

	  for (j=1; j<=2; j++){
	    for (k=1; k<=2; k++){
	      tmp_ele1 = 0.0;
	      for (l=0; l<leng1+leng2; l++){
		tmp_ele1 += Bond_Pair_vec[l][j] * temp_M1[0][l][k];
	      }
	      temp_M2[0][j][k] = tmp_ele1;
	    }
	  }

	  /*
	    printf("### 2x2 P-matrix (%d,%d)(%d) ###\n",Gc_AN,Gh_AN,i);
	    printf("  %9.5f  %9.5f\n",temp_M2[0][1][1],temp_M2[0][1][2]);
	    printf("  %9.5f  %9.5f\n",temp_M2[0][2][1],temp_M2[0][2][2]);
	    printf("\n");
	  */

	  /* diagonalization of 2x2 P-matrix */

	  Eigen_lapack(temp_M2[0],temp_V1,2,2);

	  printf("### NBO:%d (%s%d-%s%d) ###\n",i,SpeName[wan1],Gc_AN,SpeName[wan2],Gh_AN);
	  printf("  elec. in bond orb.:      %9.5f\n",temp_V1[2]);
	  printf("  elec. in anti-bond orb.: %9.5f\n",temp_V1[1]);
	  printf("\n");
	  printf("  %9.5f  %9.5f\n",temp_M2[0][1][1],temp_M2[0][1][2]);
	  printf("  %9.5f  %9.5f\n",temp_M2[0][2][1],temp_M2[0][2][2]);
	  printf("\n");

	  Pop_bond_NBO[i] = temp_V1[2];
	  Pop_anti_NBO[i] = temp_V1[1];

	  /* construction of NBO vectors */

	  for (j=0; j<leng1; j++){
	    NBO_bond_vec[i][j]       = NHO_mat[Gc_AN][j][tnum-1] * temp_M2[0][1][2];
	    NBO_anti_vec[i][j]       = NHO_mat[Gc_AN][j][tnum-1] * temp_M2[0][1][1];
	  }
	  for (j=0; j<leng2; j++){
	    NBO_bond_vec[i][leng1+j] = NHO_mat[Gh_AN][j][pnum-1] * temp_M2[0][2][2];
	    NBO_anti_vec[i][leng1+j] = NHO_mat[Gh_AN][j][pnum-1] * temp_M2[0][2][1];
	  }

	  /* Cube-data of NBOs */

	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	    for (j=0; j<Leng_a[Gc_AN]; j++){
	      NBOs_Coef_b[spin0][i-1][Gc_AN][j] = NHOs_Coef[spin0][snum][Gc_AN][j] * temp_M2[0][1][2]
		+ NHOs_Coef[spin0][qnum][Gc_AN][j] * temp_M2[0][2][2];
	      NBOs_Coef_a[spin0][i-1][Gc_AN][j] = NHOs_Coef[spin0][snum][Gc_AN][j] * temp_M2[0][1][1]
		+ NHOs_Coef[spin0][qnum][Gc_AN][j] * temp_M2[0][2][1];
	    }
	  }

	  if(1<0){
	    printf("### NBO vector (%d) ###\n",i);
	    for (j=0; j<leng1+leng2; j++){
	      printf("  %9.5f  %9.5f\n",NBO_bond_vec[i][j],NBO_anti_vec[i][j]);
	    }
	    printf("\n");
	  }
	} /* i (NBO-loop) */


	/* population on LP */

	for (i=1; i<=Num_LonePair; i++){
	  Gc_AN = Table_LP[i][0];
	  posi1 = Posi_a[Gc_AN];
	  leng1 = Leng_a[Gc_AN];
	  wan1 = WhatSpecies[Gc_AN];

	  for (j=0; j<leng1; j++){
	    for (k=0; k<leng1; k++){
	      temp_M1[0][j][k] = P_full[0][j+posi1][k+posi1];
	    } /* j */
	  } /* i */

	  for (j=0; j<leng1; j++){ 
	    tmp_ele1 = 0.0;
	    for (k=0; k<leng1; k++){
	      tmp_ele1 += temp_M1[0][j][k] * LonePair_vec[i][k];
	    }
	    temp_M2[0][j][0] = tmp_ele1;
	  }

	  tmp_ele1 = 0.0;
	  for (j=0; j<leng1; j++){
	    tmp_ele1 += LonePair_vec[i][j] * temp_M2[0][j][0];
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

	} /* i (LP-loop) */


	printf("******************************************\n");
	printf("     Orbital Interaction Analysis         \n");
	printf("     based on 2nd Perturbation Theory     \n");
	printf("******************************************\n");
	printf("\n");

	/*** construction of 2x2 Fock matrix for NBO basis ***/

	/** bonding NBO and antibonding NBO **/

	for (i=1; i<=Num_NBO; i++){
	  Gb1_AN = Table_NBO[i][0];
	  Gb2_AN = Table_NBO[i][1];

	  wan1 = WhatSpecies[Gb1_AN];
	  wan2 = WhatSpecies[Gb2_AN];

	  posi[1] = Posi_a[Gb1_AN]; leng[1] = Leng_a[Gb1_AN];
	  posi[2] = Posi_a[Gb2_AN]; leng[2] = Leng_a[Gb2_AN];

	  t_posi[1] = 0;
	  t_posi[2] = leng[1];
	  t_posi[3] = leng[1] + leng[2];

	  for (j=1; j<=Num_NBO; j++){
	    Ga1_AN = Table_NBO[j][0];
	    Ga2_AN = Table_NBO[j][1];

	    wan3 = WhatSpecies[Ga1_AN];
	    wan4 = WhatSpecies[Ga2_AN];

	    posi[3] = Posi_a[Ga1_AN]; leng[3] = Leng_a[Ga1_AN];
	    posi[4] = Posi_a[Ga2_AN]; leng[4] = Leng_a[Ga2_AN];

	    t_posi[4] = leng[1] + leng[2] + leng[3];
	    t_leng    = leng[1] + leng[2] + leng[3] + leng[4];

	    for (k=1; k<=4; k++){
	      for (l=1; l<=4; l++){

		for (m=0; m<leng[k]; m++){
		  for (n=0; n<leng[l]; n++){
		    temp_M8[0][t_posi[k]+m][t_posi[l]+n] = H_full[0][posi[k]+m][posi[l]+n];
		  }
		}

	      }
	    }
	    /*
	      printf("### Fock matrix (bondind NBO:%d , anti-bonding NBO:%d) \n",i,j);
	      for (k=0; k<t_leng; k++){
	      for (m=0; m<t_leng; m++){
	      printf("%9.5f",temp_M8[0][k][m]);
	      }
	      printf("\n");
	      }
	      printf("\n");
	    */
	    /* construction of pair-vector for bond NBO & antibond NBO */

	    for (k=0; k<t_leng; k++){
	      temp_M9[0][k][0] = 0.0;
	      temp_M9[0][k][1] = 0.0;
	    }

	    for (k=0; k<leng[1]+leng[2]; k++){
	      temp_M9[0][k][0] = NBO_bond_vec[i][k];
	    }

	    for (k=0; k<leng[3]+leng[4]; k++){
	      temp_M9[0][t_posi[3]+k][1] = NBO_anti_vec[j][k];
	    }

	    /* construction of Fock matrix for NBO */

	    for (k=0; k<t_leng; k++){
	      for (l=0; l<=1; l++){
		tmp_ele1 = 0.0;
		for (m=0; m<t_leng; m++){
		  tmp_ele1 += temp_M8[0][k][m] * temp_M9[0][m][l];
		}
		temp_M10[0][k][l] = tmp_ele1;
	      }
	    }

	    for (k=0; k<=1; k++){
	      for (l=0; l<=1; l++){
		tmp_ele1 = 0.0;
		for (m=0; m<t_leng; m++){
		  tmp_ele1 += temp_M9[0][m][k] * temp_M10[0][m][l];
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
	    printf("  Population : b-NBO = %9.5f, a-NBO = %9.5f \n",Pop_bond_NBO[i],Pop_anti_NBO[j]);
	    printf("  Interaction Energy = %9.5f (Hartree)\n",tmp_ele1);
	    printf("                     = %9.5f (eV)\n",tmp_ele1*27.2116);
	    printf("                     = %9.5f (kcal/mol)\n\n",tmp_ele1*627.509);


	  } /* j (for anti-bonding NBO) */
	} /* i (for bonding NBO) */

	/* lone pair and antibonding NBO */

	for (i=1; i<=Num_LonePair; i++){
	  Gb1_AN = Table_LP[i][0];

	  wan1 = WhatSpecies[Gb1_AN];

	  posi[1] = Posi_a[Gb1_AN]; leng[1] = Leng_a[Gb1_AN];

	  t_posi[1] = 0;
	  t_posi[2] = leng[1];

	  for (j=1; j<=Num_NBO; j++){
	    Ga1_AN = Table_NBO[j][0];
	    Ga2_AN = Table_NBO[j][1];

	    wan3 = WhatSpecies[Ga1_AN];
	    wan4 = WhatSpecies[Ga2_AN];

	    posi[2] = Posi_a[Ga1_AN]; leng[2] = Leng_a[Ga1_AN];
	    posi[3] = Posi_a[Ga2_AN]; leng[3] = Leng_a[Ga2_AN];

	    t_posi[3] = leng[1] + leng[2];
	    t_leng    = leng[1] + leng[2] + leng[3];

	    for (k=1; k<=3; k++){
	      for (l=1; l<=3; l++){

		for (m=0; m<leng[k]; m++){
		  for (n=0; n<leng[l]; n++){
		    temp_M8[0][t_posi[k]+m][t_posi[l]+n] = H_full[0][posi[k]+m][posi[l]+n];
		  }
		}

	      }
	    }
	    /*
	      printf("### Fock matrix (Lone Pair:%d , anti-bonding NBO:%d) \n",i,j);
	      for (k=0; k<t_leng; k++){
	      for (m=0; m<t_leng; m++){
	      printf("%9.5f",temp_M8[0][k][m]);
	      }
	      printf("\n");
	      }
	      printf("\n");
	    */
	    /** construction of pair-vector for lone pair & antibond NBO **/

	    for (k=0; k<t_leng; k++){
	      temp_M9[0][k][0] = 0.0;
	      temp_M9[0][k][1] = 0.0;
	    }

	    for (k=0; k<leng[1]; k++){
	      temp_M9[0][k][0] = LonePair_vec[i][k];
	    }

	    for (k=0; k<leng[2]+leng[3]; k++){
	      temp_M9[0][t_posi[2]+k][1] = NBO_anti_vec[j][k];
	    }
	    /*
	      for (k=0; k<leng[1]+leng[2]+leng[3]; k++){
	      printf("## TEST ##  %9.5f %9.5f \n",temp_M9[0][k][0],temp_M9[0][k][1]);       
	      }
	      printf("\n");
	    */
	    /* construction of Fock matrix for NBO */

	    for (k=0; k<t_leng; k++){
	      for (l=0; l<=1; l++){
		tmp_ele1 = 0.0;
		for (m=0; m<t_leng; m++){
		  tmp_ele1 += temp_M8[0][k][m] * temp_M9[0][m][l];
		}
		temp_M10[0][k][l] = tmp_ele1;
	      }
	    }

	    for (k=0; k<=1; k++){
	      for (l=0; l<=1; l++){
		tmp_ele1 = 0.0;
		for (m=0; m<t_leng; m++){
		  tmp_ele1 += temp_M9[0][m][k] * temp_M10[0][m][l];
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
	    printf("  Population : LP = %9.5f, a-NBO = %9.5f \n",Pop_LonePair[i],Pop_anti_NBO[j]);
	    printf("  Interaction Energy = %9.5f (Hartree)\n",   tmp_ele1);
	    printf("                     = %9.5f (eV)\n",        tmp_ele1*27.2116);
	    printf("                     = %9.5f (kcal/mol)\n\n",tmp_ele1*627.509);

	  } /* j (for anti-bonding NBO) */
	} /* i (for LP) */



	free(Num_LP);
	free(Num_NHO);
	free(Pop_bond_NBO);
	free(Pop_anti_NBO);
	free(Pop_LonePair);

	for (i=0; i<=atomnum*10; i++){
	  free(Table_NBO[i]);
	}
	free(Table_NBO);

	for (i=0; i<=atomnum*5; i++){
	  free(Table_LP[i]);
	}
	free(Table_LP);

	for (i=0; i<=atomnum; i++){
	  free(NHO_indexL2G[i]);
	}
	free(NHO_indexL2G);

	for (i=0; i<=atomnum; i++){
	  for (j=0; j<=MaxLeng_a; j++){
	    free(pLP_vec[i][j]);
	  }
	  free(pLP_vec[i]);
	}
	free(pLP_vec);

	for (i=0; i<=atomnum; i++){
	  for (j=0; j<=MaxLeng_a; j++){
	    free(pNHO_vec[i][j]);
	  }
	  free(pNHO_vec[i]);
	}
	free(pNHO_vec);

	for (i=0; i<=atomnum; i++){
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

	for (i=0; i<=atomnum; i++){
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

	for (i=0; i<=atomnum*10; i++){
	  free(NBO_bond_vec[i]);
	}
	free(NBO_bond_vec);

	for (i=0; i<=atomnum*10; i++){
	  free(NBO_anti_vec[i]);
	}
	free(NBO_anti_vec);

	for (i=0; i<=atomnum*5; i++){
	  free(LonePair_vec[i]);
	}
	free(LonePair_vec);

	for (i=0; i<=atomnum; i++){
	  for (j=0; j<=atomnum; j++){
	    for (k=0; k<=MaxLeng_a*2; k++){
	      free(P_pair[i][j][k]);
	    }
	    free(P_pair[i][j]);
	  }
	  free(P_pair[i]);
	}
	free(P_pair);

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=0; i<=MaxLeng_a*5; i++){
	    free(temp_M8[spin][i]);
	  }
	  free(temp_M8[spin]);
	}
	free(temp_M8);

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=0; i<=MaxLeng_a*5; i++){
	    free(temp_M9[spin][i]);
	  }
	  free(temp_M9[spin]);
	}
	free(temp_M9);

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=0; i<=MaxLeng_a*5; i++){
	    free(temp_M10[spin][i]);
	  }
	  free(temp_M10[spin]);
	}
	free(temp_M10);


      } /* NAO_only==0 */

      /***  Freeing arrays (1)  ***/

      free(Leng_a);
      free(Posi2Gatm);

      for (i=0; i<=atomnum; i++){
	free(Leng_al[i]);
      }
      free(Leng_al);

      for (i=0; i<=atomnum; i++){
	free(Leng_alm[i]);
      }
      free(Leng_alm);

      free(Posi_a);

      for (i=0; i<=atomnum; i++){
	free(Posi_al[i]);
      }
      free(Posi_al);

      for (i=0; i<=atomnum; i++){
	for (j=0; j<=Lmax; j++){
	  free(Posi_alm[i][j]);
	}
	free(Posi_alm[i]);
      }
      free(Posi_alm);

      for (i=0; i<=atomnum; i++){
	free(Num_NMB1[i]);
      }
      free(Num_NMB1);

      for (i=0; i<=atomnum; i++){
	free(Num_NRB1[i]);
      }
      free(Num_NRB1);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(S_full[spin][i]);
	}
	free(S_full[spin]);
      }
      free(S_full);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(S_full0[spin][i]);
	}
	free(S_full0[spin]);
      }
      free(S_full0);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(D_full[spin][i]);
	}
	free(D_full[spin]);
      }
      free(D_full);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(P_full[spin][i]);
	}
	free(P_full[spin]);
      }
      free(P_full);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(P_full0[spin][i]);
	}
	free(P_full0[spin]);
      }
      free(P_full0);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(H_full[spin][i]);
	}
	free(H_full[spin]);
      }
      free(H_full);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(H_full0[spin][i]);
	}
	free(H_full0[spin]);
      }
      free(H_full0);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(T_full[spin][i]);
	}
	free(T_full[spin]);
      }
      free(T_full);

      for (spin=0; spin<=SpinP_switch; spin++){
	free(W_full[spin]);
      }
      free(W_full);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(N_tran1[spin][i]);
	}
	free(N_tran1[spin]);
      }
      free(N_tran1);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(N_tran2[spin][i]);
	}
	free(N_tran2[spin]);
      }
      free(N_tran2);
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(N_diag[spin][i]);
	}
	free(N_diag[spin]);
	}
	free(N_diag);
      */
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(N_ryd[spin][i]);
	}
	free(N_ryd[spin]);
	}
	free(N_ryd);
      */
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(N_red[spin][i]);
	}
	free(N_red[spin]);
	}
	free(N_red);
      */
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(O_schm[spin][i]);
	}
	free(O_schm[spin]);
	}
	free(O_schm);
      */
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(O_schm2[spin][i]);
	}
	free(O_schm2[spin]);
	}
	free(O_schm2);
      */
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(Ow_NMB[spin][i]);
	}
	free(Ow_NMB[spin]);
	}
	free(Ow_NMB);
      */
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(Ow_NRB[spin][i]);
	}
	free(Ow_NRB[spin]);
	}
	free(Ow_NRB);
      */
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(O_sym[spin][i]);
	}
	free(O_sym[spin]);
	}
	free(O_sym);
      */
      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(temp_M1[spin][i]);
	}
	free(temp_M1[spin]);
      }
      free(temp_M1);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(temp_M2[spin][i]);
	}
	free(temp_M2[spin]);
      }
      free(temp_M2);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(temp_M3[spin][i]);
	}
	free(temp_M3[spin]);
      }
      free(temp_M3);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	  free(temp_M4[spin][i]);
	}
	free(temp_M4[spin]);
      }
      free(temp_M4);
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(temp_M5[spin][i]);
	}
	free(temp_M5[spin]);
	}
	free(temp_M5);
      */
      /*
	for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=SizeMat; i++){
	free(temp_M6[spin][i]);
	}
	free(temp_M6[spin]);
	}
	free(temp_M6);
      */
      free(temp_V1);
      free(temp_V2);
      free(temp_V3);

      /***  Freeing arrays (2)  ***/

      for (spin=0; spin<=SpinP_switch; spin++){
	for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
	  for (L1=0; L1<=Lmax; L1++){
	    for (M1=0; M1<=2*Lmax+1; M1++){
	      for (i=0; i<=MaxLeng_alm; i++){
		free(P_alm[spin][Gc_AN][L1][M1][i]);
	      }
	      free(P_alm[spin][Gc_AN][L1][M1]);
	    }
	    free(P_alm[spin][Gc_AN][L1]);
	  }
	  free(P_alm[spin][Gc_AN]);
	}
	free(P_alm[spin]);
      }
      free(P_alm);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
	  for (L1=0; L1<=Lmax; L1++){
	    for (M1=0; M1<=2*Lmax+1; M1++){
	      for (i=0; i<=MaxLeng_alm; i++){
		free(S_alm[spin][Gc_AN][L1][M1][i]);
	      }
	      free(S_alm[spin][Gc_AN][L1][M1]);
	    }
	    free(S_alm[spin][Gc_AN][L1]);
	  }
	  free(S_alm[spin][Gc_AN]);
	}
	free(S_alm[spin]);
      }
      free(S_alm);

      /***  Freeing arrays (3)  ***/

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=MaxLeng_alm; i++){
	  free(P_alm2[spin][i]);
	}
	free(P_alm2[spin]);
      }
      free(P_alm2);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=MaxLeng_alm; i++){
	  free(S_alm2[spin][i]);
	}
	free(S_alm2[spin]);
      }
      free(S_alm2);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=MaxLeng_alm; i++){
	  free(N_alm2[spin][i]);
	}
	free(N_alm2[spin]);
      }
      free(N_alm2);

      for (spin=0; spin<=SpinP_switch; spin++){
	free(W_alm2[spin]);
      }
      free(W_alm2);

      for (i=0; i<=atomnum; i++){
	for (j=0; j<=Lmax; j++){
	  free(M_or_R1[i][j]);
	}
	free(M_or_R1[i]);
      }
      free(M_or_R1);

      free(M_or_R2);

      /***  Freeing arrays (4)  ***/

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=Num_NMB; i++){
	  free(S_NMB[spin][i]);
	}
	free(S_NMB[spin]);
      }
      free(S_NMB);

      for (spin=0; spin<=SpinP_switch; spin++){
	free(W_NMB[spin]);
      }
      free(W_NMB);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=Num_NMB; i++){
	  free(Ow_NMB2[spin][i]);
	}
	free(Ow_NMB2[spin]);
      }
      free(Ow_NMB2);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=Num_NMB; i++){
	  free(Sw1[spin][i]);
	}
	free(Sw1[spin]);
      }
      free(Sw1);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=Num_NMB; i++){
	  free(Uw1[spin][i]);
	}
	free(Uw1[spin]);
      }
      free(Uw1);

      for (spin=0; spin<=SpinP_switch; spin++){
	free(Rw1[spin]);
      }
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

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=Num_NRB; i++){
	  free(S_NRB[spin][i]);
	}
	free(S_NRB[spin]);
      }
      free(S_NRB);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=Num_NRB; i++){
	  free(S_NRB2[spin][i]);
	}
	free(S_NRB2[spin]);
      }
      free(S_NRB2);

      for (spin=0; spin<=SpinP_switch; spin++){
	free(W_NRB[spin]);
      }
      free(W_NRB);

      for (spin=0; spin<=SpinP_switch; spin++){
	free(W_NRB2[spin]);
      }
      free(W_NRB2);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=Num_NRB; i++){
	  free(Ow_NRB2[spin][i]);
	}
	free(Ow_NRB2[spin]);
      }
      free(Ow_NRB2);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=Num_NRB; i++){
	  free(Sw2[spin][i]);
	}
	free(Sw2[spin]);
      }
      free(Sw2);

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<=Num_NRB; i++){
	  free(Uw2[spin][i]);
	}
	free(Uw2[spin]);
      }
      free(Uw2);

      for (spin=0; spin<=SpinP_switch; spin++){
	free(Rw2[spin]);
      }
      free(Rw2);

    } /* myid == Host_ID */

  } /* spin0 */

  free(ID2nump);
  free(ID2num);
  free(ID2posi);
  free(MatomN);

  for (i=0; i<numprocs; i++){
    free(ID2G[i]);
  }
  free(ID2G);

  free(Tmp_Vec0);
  free(Tmp_Vec1);
  free(Tmp_Vec2);
  free(Tmp_Vec3);



  if (measure_time==1) dtime(&EtimeF);


  if (measure_time==1 && myid == Host_ID){

    printf(" ######## Elapsed time in NAO calc. (sec.) ########\n\n");
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

  if(NAO_only==0){

    MPI_Barrier(mpi_comm_level1);

    /* Cube-data of Orbitals */

    if(NHO_fileout==1){
      MPI_Bcast(&Total_Num_NHO, 1, MPI_INT, Host_ID, mpi_comm_level1);

      for (spin0=0; spin0<=SpinP_switch; spin0++){
	for (i=0; i<Total_Num_NHO; i++){
	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	    wan1 = WhatSpecies[Gc_AN];
	    MPI_Bcast(&NHOs_Coef[spin0][i][Gc_AN][0], Spe_Total_CNO[wan1], MPI_DOUBLE, Host_ID, mpi_comm_level1);
	  }
	}
      }
    }

    if(NBO_fileout==1){
      MPI_Bcast(&Total_Num_NBO, 1, MPI_INT, Host_ID, mpi_comm_level1);

      for (spin0=0; spin0<=SpinP_switch; spin0++){
	for (i=0; i<Total_Num_NBO; i++){
	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	    wan1 = WhatSpecies[Gc_AN];
	    MPI_Bcast(&NBOs_Coef_b[spin0][i][Gc_AN][0], Spe_Total_CNO[wan1], MPI_DOUBLE, Host_ID, mpi_comm_level1);
	  }
	}
      }

      for (spin0=0; spin0<=SpinP_switch; spin0++){
	for (i=0; i<Total_Num_NBO; i++){
	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	    wan1 = WhatSpecies[Gc_AN];
	    MPI_Bcast(&NBOs_Coef_a[spin0][i][Gc_AN][0], Spe_Total_CNO[wan1], MPI_DOUBLE, Host_ID, mpi_comm_level1);
	  }
	}
      }
    }

  } /* NAO_only==0 */

} /** end of "Calc_NAO" **/

