/**********************************************************************
  Set_CoreHoleMatrix.c:

     Set_CoreHoleMatrix.c is a subroutine to calculate matrix elements
     to create a core hole by penalizing the occupation to a target state.

  Log of Set_CoreHoleMatrix.c:

     11/May/2016  Released by T. Ozaki

***********************************************************************/

#define  measure_time   0

#include <stdio.h>
#include <stdlib.h>
#include<string.h> 
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include <sys/stat.h>
#include <unistd.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

static void Multiply_OLP(int Mc_AN, int Mj_AN, int k, int kl, 
                         int Cwan, int Hwan, int wakg, dcomplex ***NLH);


double Set_CoreHoleMatrix(double *****HCH)
{
  static int firsttime=1;
  double TStime,TEtime;
  int i,j,kl,n,m,Rn;
  int Mc_AN,Gc_AN,h_AN,k,Cwan,Gh_AN,Hwan,so;
  int tno0,tno1,tno2,i1,j1,p,ct_AN,spin;
  int fan,jg,kg,wakg,jg0,Mj_AN0,j0;
  int Mj_AN,num,size1,size2;
  int *Snd_OLP_Size,*Rcv_OLP_Size;
  int Original_Mc_AN,po; 
  double rcutA,rcutB,rcut,dmp;
  double rs,rs2,rs4,rs5,rs6,al;
  double fac2,fac,lx,ly,lz,r;
  double time1,time2,time3;
  double stime,etime;
  dcomplex ***NLH;
  double *tmp_array;
  double *tmp_array2;
  double Stime_atom,Etime_atom;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*************************************************************
    set scaled OLP = OLP_CH
    OLP_CH = exp(-(r/al)^6) * OLP where al = 4.0

    We store only limited elements of HCH so that the structure 
    of the matrix HCH can be the same as that of OLP and H.
    The treatment violates the positivity of the penalty 
    functional, and in fact we see that the penalty functional 
    becomes negative when the penalty scheme for delocalized 
    states is applied.

    A way of avoiding the situation is to make OLP short-ranged. 
    To do that we introduce a scaled OLP defined by
    OLP_CH = exp(-(r/al)^6) * OLP.
    The treatment may correspond to introduction of short-ranged 
    projector functions. Since exp(-(r/al)^6) keeps almost unity 
    below al, if the targeted states are localized, the treatment
    may not change largely the result. It was actually confirmed 
    that the guess is true for N-1s and Ce-4f states, while 
    the population to the targeted states seems to occur for 
    delocalized states.
  *************************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      Rn = ncn[Gc_AN][h_AN];
      r = Dis[Gc_AN][h_AN];

      if (h_AN==0){

	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){
            OLP_CH[0][Mc_AN][h_AN][i][j] = OLP[0][Mc_AN][h_AN][i][j]; 
            OLP_CH[1][Mc_AN][h_AN][i][j] = OLP[1][Mc_AN][h_AN][i][j]; 
            OLP_CH[2][Mc_AN][h_AN][i][j] = OLP[2][Mc_AN][h_AN][i][j]; 
            OLP_CH[3][Mc_AN][h_AN][i][j] = OLP[3][Mc_AN][h_AN][i][j]; 
	  }
	}
      }
      else{

	lx = (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])/r;
	ly = (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])/r;
	lz = (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])/r;

	/*
        al = 0.3; 
        rs = r/al;
        rs2 = rs*rs;
        rs4 = rs2*rs2;
        rs5 = rs*rs4;
        rs6 = rs2*rs4;
        fac = exp(-rs6);
        fac2 = -6.0*rs5/al*fac;
	*/

        /* In Ver. 3.9, the scaling is not applied. Therefore, we set that fac=1 and fac2=0. */

        fac  = 1.0;
        fac2 = 0.0;

	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            OLP_CH[0][Mc_AN][h_AN][i][j] = fac*OLP[0][Mc_AN][h_AN][i][j]; 
            OLP_CH[1][Mc_AN][h_AN][i][j] = fac2*lx*OLP[0][Mc_AN][h_AN][i][j] + fac*OLP[1][Mc_AN][h_AN][i][j]; 
            OLP_CH[2][Mc_AN][h_AN][i][j] = fac2*ly*OLP[0][Mc_AN][h_AN][i][j] + fac*OLP[2][Mc_AN][h_AN][i][j]; 
            OLP_CH[3][Mc_AN][h_AN][i][j] = fac2*lz*OLP[0][Mc_AN][h_AN][i][j] + fac*OLP[3][Mc_AN][h_AN][i][j];
	  }
	}
      }

    }
  }

  /* one-dimensionalize the Mc_AN and h_AN loops */

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
    }
  }  

  OneD2Mc_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));
  OneD2h_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD2Mc_AN[OneD_Nloop] = Mc_AN; 
      OneD2h_AN[OneD_Nloop] = h_AN; 
      OneD_Nloop++;
    }
  }

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITH COMMUNICATION

     MPI: communicate only for k=0
     OLP_CH
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

  /* allocation of arrays */

  NLH = (dcomplex***)malloc(sizeof(dcomplex**)*3); 
  for (k=0; k<3; k++){
    NLH[k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]); 
    for (i=0; i<List_YOUSO[7]; i++){
      NLH[k][i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]); 
    }
  }

  Snd_OLP_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_OLP_Size = (int*)malloc(sizeof(int)*numprocs);

  for (ID=0; ID<numprocs; ID++){
    F_Snd_Num_WK[ID] = 0;
    F_Rcv_Num_WK[ID] = 0;
  }

  do {

    /***********************************                                                            
          set the size of data                                                                      
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /* find the data size to send the block data */

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

        size1 = 0;
        n = F_Snd_Num_WK[IDS];

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

        Snd_OLP_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_OLP_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_OLP_Size[IDR] = size2;
      }
      else{
        Rcv_OLP_Size[IDR] = 0;
      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) )  MPI_Wait(&request,&stat);

    } /* ID */

    /***********************************
               data transfer
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /******************************
         sending of the data 
      ******************************/

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = Snd_OLP_Size[IDS];

	/* allocation of the array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to the vector array */

	num = 0;
	n = F_Snd_Num_WK[IDS];

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
	      tmp_array[num] = OLP_CH[0][Mc_AN][h_AN][i][j];
	      num++;
	    } 
	  } 
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /******************************
        receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        
	size2 = Rcv_OLP_Size[IDR];
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	/* store */

	num = 0;
	n = F_Rcv_Num_WK[IDR];
	Original_Mc_AN = F_TopMAN[IDR] + n;

	Gc_AN = Rcv_GAN[IDR][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_NO[Hwan];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      OLP_CH[0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
	      num++;
	    }
	  }
	}

	/* free tmp_array2 */
	free(tmp_array2);

	/*****************************************************************
                           multiplying overlap integrals
	*****************************************************************/

        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  dtime(&Stime_atom);

          Gc_AN = M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          fan = FNAN[Gc_AN];
          rcutA = Spe_Atom_Cut1[Cwan];

          n = F_Rcv_Num_WK[IDR];
          jg = Rcv_GAN[IDR][n];

          for (j0=0; j0<=fan; j0++){

            jg0 = natn[Gc_AN][j0];
            Mj_AN0 = F_G2M[jg0];

            po = 0;
            if (Original_Mc_AN==Mj_AN0){
              po = 1;
              j = j0;
            }

            if (po==1){

	      Hwan = WhatSpecies[jg];
              rcutB = Spe_Atom_Cut1[Hwan];
              rcut = rcutA + rcutB;

	      for (m=0; m<Spe_Total_NO[Cwan]; m++){
		for (n=0; n<Spe_Total_NO[Hwan]; n++){

		  NLH[0][m][n].r = 0.0;     /* <up|VNL|up> */
		  NLH[1][m][n].r = 0.0;     /* <dn|VNL|dn> */
		  NLH[2][m][n].r = 0.0;     /* <up|VNL|dn> */
		  NLH[0][m][n].i = 0.0;
		  NLH[1][m][n].i = 0.0;
		  NLH[2][m][n].i = 0.0;
		}
	      }

              for (k=0; k<=fan; k++){

                kg = natn[Gc_AN][k];
                wakg = WhatSpecies[kg];
                kl = RMI1[Mc_AN][j][k];

                if (0<=kl){

                  Multiply_OLP(Mc_AN, Matomnum+1, k, kl, Cwan, Hwan, wakg, NLH);

		} /* if (0<=kl) */

	      } /* k */

              /****************************************************
                      adding NLH to HCH
  
                 HNL[0] and iHNL[0] for up-up
                 HNL[1] and iHNL[1] for dn-dn
                 HNL[2] and iHNL[2] for up-dn
	      ****************************************************/

              dmp = dampingF(rcut,Dis[Gc_AN][j]);

	      for (p=0; p<List_YOUSO[5]; p++){
		for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
		  for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){

		    HCH[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].r*F_CH_flag;

		    if (SO_switch==1){
		      iHCH[p][Mc_AN][j][i1][j1]  = dmp*NLH[p][i1][j1].i*F_CH_flag;
		      iHNL[p][Mc_AN][j][i1][j1] += dmp*NLH[p][i1][j1].i*F_CH_flag;
		    }
		  }
		}
	      }

	    } /* if (po==1) */
	  } /* j0 */

          dtime(&Etime_atom);
          time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

        /********************************************                                               
            increment of F_Rcv_Num_WK[IDR]                                                          
	********************************************/

        F_Rcv_Num_WK[IDR]++;
      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) {

        MPI_Wait(&request,&stat);
        free(tmp_array);  /* freeing of array */

        /********************************************                                               
             increment of F_Snd_Num_WK[IDS]                                                         
	********************************************/

        F_Snd_Num_WK[IDS]++;
      }

    } /* ID */

    /*****************************************************                                          
      check whether all the communications have finished                                            
    *****************************************************/

    po = 0;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) po += F_Snd_Num[IDS]-F_Snd_Num_WK[IDS];
      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ) po += F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR];
    }

  } while (po!=0);

  /* freeing of array */

  free(Rcv_OLP_Size);
  free(Snd_OLP_Size);

  for (k=0; k<3; k++){
    for (i=0; i<List_YOUSO[7]; i++){
      free(NLH[k][i]);
    }
    free(NLH[k]);
  }
  free(NLH);

  if (measure_time){
    dtime(&etime);
    time2 = etime - stime;
  }

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITHOUT COMMUNICATION
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

  /* allocation of arrays */

#pragma omp parallel shared(Matomnum,time_per_atom,HCH,iHCH,iHNL,F_CH_flag,List_YOUSO,Dis,SpinP_switch,Spe_Total_NO,OLP,Spe_VPS_List,VPS_j_dependency,RMI1,F_G2M,natn,Spe_Atom_Cut1,FNAN,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,SO_switch) 
  {
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,j,Gc_AN,Cwan,fan,jg,i1,j1,i;
    int Mj_AN,Hwan,k,kg,wakg,kl;
    int p,m,n,L,L1,L2,L3;
    double rcutA,rcutB,rcut,sum,ene;
    double Stime_atom, Etime_atom;
    double ene_m,ene_p,dmp;
    double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
    double PFp,PFm;
    dcomplex ***NLH;
    dcomplex sum0,sum1,sum2; 

    /* allocation of arrays */

    NLH = (dcomplex***)malloc(sizeof(dcomplex**)*3); 
    for (k=0; k<3; k++){
      NLH[k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]); 
      for (i=0; i<List_YOUSO[7]; i++){
	NLH[k][i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]); 
      }
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /* one-dimensionalized loop */

    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      dtime(&Stime_atom);

      /* get Mc_AN and j */

      Mc_AN = OneD2Mc_AN[Nloop];
      j     = OneD2h_AN[Nloop];

      /* set data on Mc_AN */
    
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      rcutA = Spe_Atom_Cut1[Cwan];

      /* set data on j */
  
      jg = natn[Gc_AN][j];
      Mj_AN = F_G2M[jg];

      if (Mj_AN<=Matomnum){

	Hwan = WhatSpecies[jg];
	rcutB = Spe_Atom_Cut1[Hwan];
	rcut = rcutA + rcutB;

	for (m=0; m<Spe_Total_NO[Cwan]; m++){
	  for (n=0; n<Spe_Total_NO[Hwan]; n++){
	    NLH[0][m][n].r = 0.0;     /* <up|VNL|up> */
	    NLH[1][m][n].r = 0.0;     /* <dn|VNL|dn> */
	    NLH[2][m][n].r = 0.0;     /* <up|VNL|dn> */
	    NLH[0][m][n].i = 0.0;
	    NLH[1][m][n].i = 0.0;
	    NLH[2][m][n].i = 0.0;
	  }
	}

	for (k=0; k<=fan; k++){

	  kg = natn[Gc_AN][k];
	  wakg = WhatSpecies[kg];
	  kl = RMI1[Mc_AN][j][k];

	  if (0<=kl){

	    Multiply_OLP(Mc_AN, Mj_AN, k, kl, Cwan, Hwan, wakg, NLH);

	  } /* if (0<=kl) */

	} /* k */

	/****************************************************
                       adding NLH to HNL
  
                 HNL[0] and iHNL[0] for up-up
                 HNL[1] and iHNL[1] for dn-dn
                 HNL[2] and iHNL[2] for up-dn
	****************************************************/

	dmp = dampingF(rcut,Dis[Gc_AN][j]);

	for (p=0; p<List_YOUSO[5]; p++){
	  for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
	    for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){

	      HCH[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].r*F_CH_flag;

	      if (SO_switch==1){
		iHCH[p][Mc_AN][j][i1][j1]  = dmp*NLH[p][i1][j1].i*F_CH_flag;
		iHNL[p][Mc_AN][j][i1][j1] += dmp*NLH[p][i1][j1].i*F_CH_flag;
	      }
	    }
	  }
	}

      } /* if (Mj_AN<=Matomnum) */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Nloop */

    /* freeing of array */

    for (k=0; k<3; k++){
      for (i=0; i<List_YOUSO[7]; i++){
	free(NLH[k][i]);
      }
      free(NLH[k]);
    }
    free(NLH);

#pragma omp barrier
#pragma omp flush(HNL,iHNL)

  } /* #pragma omp parallel */

  /****************************************************
                    freeing of arrays:
  ****************************************************/

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  if (measure_time){
    dtime(&etime);
    time3 = etime - stime;
  }

  if (measure_time){
    printf("Set_Nonlocal: myid=%2d time2=%10.5f time3=%10.5f\n",
           myid,time2,time3);
  }

  /****************************************************
   MPI_Barrier
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  /* for time */
  dtime(&TEtime);
  return TEtime - TStime;
} 






void Multiply_OLP(int Mc_AN, int Mj_AN, int k, int kl, 
                  int Cwan, int Hwan, int wakg, dcomplex ***NLH)
{
  int m,n,L,L1,L2,spin,target_spin;
  int i,j,apply_flag,l,mul;
  int done_flag,GA_AN,GB_AN;
  double **penalty;
  double penalty_value;
  double sum,sumMul[2],ene;
  double tmp0;
  double d12_m12,d12_p12;
  double d32_m12,d32_p12,d32_m32,d32_p32;
  double d52_m12,d52_p12,d52_m32,d52_p32,d52_m52,d52_p52;
  double d72_m12,d72_p12,d72_m32,d72_p32,d72_m52,d72_p52,d72_m72,d72_p72;
  dcomplex sum0,sum1,sum2; 

  /* allocation of array */

  penalty = (double**)malloc(sizeof(double*)*2);
  for (i=0; i<2; i++){
    penalty[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  penalty_value = penalty_value_CoreHole;

  /****************************************************
              l-dependent non-local part
  ****************************************************/
        
  if (VPS_j_dependency[wakg]==0){

    for (m=0; m<Spe_Total_NO[Cwan]; m++){
      for (n=0; n<Spe_Total_NO[Hwan]; n++){

	sumMul[0] = 0.0;
	sumMul[1] = 0.0;

        /* if (GB_AN==Core_Hole_Atom), then perform multiplication */

        GA_AN = M2G[Mc_AN];
        GB_AN = natn[GA_AN][k];

        if (GB_AN==Core_Hole_Atom){
          
          /* set penalty */

  	  for (i=0; i<Spe_Total_NO[wakg]; i++){
            penalty[0][i] = 0.0;
            penalty[1][i] = 0.0;
	  }

          L = 0;
          for (l=0; l<=Spe_MaxL_Basis[wakg]; l++){
            for (mul=0; mul<Spe_Num_Basis[wakg][l]; mul++){

              apply_flag = 0; 

	      if      ((strcmp(Core_Hole_Orbital,"s")==0) && l==0 && mul==0) {

		if (Core_Hole_J==1) target_spin = 0;
                else                target_spin = 1;

                apply_flag = 1; 
              } 

	      else if ((strcmp(Core_Hole_Orbital,"p")==0) && l==1 && mul==0) {

		if (Core_Hole_J<=3) target_spin = 0;
                else                target_spin = 1;

                apply_flag = 1; 
              }

	      else if ((strcmp(Core_Hole_Orbital,"d")==0) && l==2 && mul==0) {

		if (Core_Hole_J<=5) target_spin = 0;
                else                target_spin = 1;

                apply_flag = 1; 
              }

	      else if ((strcmp(Core_Hole_Orbital,"f")==0) && l==3 && mul==0) { 
		
                if (Core_Hole_J<=7) target_spin = 0;
                else                target_spin = 1;

                apply_flag = 1; 
              }

              /* set the penalty into all the states speficied by l and mul */
            
              if (apply_flag==1 && Core_Hole_J==0){
		for (i=0; i<(2*l+1); i++){
		  penalty[0][L+i] = penalty_value;
		  penalty[1][L+i] = penalty_value;
		}
              }

              /* set the penalty into one of the states speficied by l and mul */

              else if (apply_flag==1){
                penalty[target_spin][L+(Core_Hole_J-1) % (2*l+1)] = penalty_value;
	      }

              /* increment of L */

              L += 2*l+1; 
	    }
	  }

	  /* calculate the multiplication */

	  for (i=0; i<Spe_Total_NO[wakg]; i++){
	    sumMul[0] += penalty[0][i]*OLP_CH[0][Mc_AN][k][m][i]*OLP_CH[0][Mj_AN][kl][n][i];
	    sumMul[1] += penalty[1][i]*OLP_CH[0][Mc_AN][k][m][i]*OLP_CH[0][Mj_AN][kl][n][i];
	  }

	  NLH[0][m][n].r += sumMul[0];    /* <up|VNL|up> */
	  NLH[1][m][n].r += sumMul[1];    /* <dn|VNL|dn> */

	}
      }
    }

  } /* if */

  /****************************************************
               j-dependent non-local part
  ****************************************************/

  else if (VPS_j_dependency[wakg]==1){

    for (m=0; m<Spe_Total_NO[Cwan]; m++){
      for (n=0; n<Spe_Total_NO[Hwan]; n++){

	sum0 = Complex(0.0,0.0);
	sum1 = Complex(0.0,0.0);
	sum2 = Complex(0.0,0.0);

        /* if (GB_AN==Core_Hole_Atom), then perform multiplication */

        GA_AN = M2G[Mc_AN];
        GB_AN = natn[GA_AN][k];

        if (GB_AN==Core_Hole_Atom){

          L = 0;
          for (l=0; l<=Spe_MaxL_Basis[wakg]; l++){
            for (mul=0; mul<Spe_Num_Basis[wakg][l]; mul++){

              apply_flag = 0; 

	      if      ((strcmp(Core_Hole_Orbital,"s")==0) && l==0 && mul==0) {

                L2 = 0; 
                apply_flag = 1; 
              } 

	      else if ((strcmp(Core_Hole_Orbital,"p")==0) && l==1 && mul==0) {

                L2 = 2; 
                apply_flag = 1; 
              }

	      else if ((strcmp(Core_Hole_Orbital,"d")==0) && l==2 && mul==0) {
 
                L2 = 4; 
                apply_flag = 1; 
              }

	      else if ((strcmp(Core_Hole_Orbital,"f")==0) && l==3 && mul==0) { 

                L2 = 6; 
                apply_flag = 1; 
              }

              if (apply_flag==1){

		/****************************************************
                        set coefficients related to penalty
		****************************************************/

		if      (L2==0){

                  if (Core_Hole_J==0){
		    /* for Core_Hole_J==0 */ 
		    d12_p12 = penalty_value;
		    d12_m12 = penalty_value;
		  }

                  else{
		    /* for Core_Hole_J!=0 */ 
		    d12_p12 = penalty_value*(Core_Hole_J==1);
		    d12_m12 = penalty_value*(Core_Hole_J==2);
		  }
		}

		else if (L2==2){

                  if (Core_Hole_J==0){
		    /* for Core_Hole_J==0 */ 
		    d32_p32 = penalty_value/3.0;
		    d32_p12 = penalty_value/3.0;
		    d32_m12 = penalty_value/3.0;
		    d32_m32 = penalty_value/3.0;

		    d12_p12 = penalty_value/3.0;
		    d12_m12 = penalty_value/3.0;
		  }

                  else{
		    /* for Core_Hole_J!=0 */ 
		    d32_p32 = penalty_value/3.0*(Core_Hole_J==1);
		    d32_p12 = penalty_value/3.0*(Core_Hole_J==2);
		    d32_m12 = penalty_value/3.0*(Core_Hole_J==3);
		    d32_m32 = penalty_value/3.0*(Core_Hole_J==4);

		    d12_p12 = penalty_value/3.0*(Core_Hole_J==5);
		    d12_m12 = penalty_value/3.0*(Core_Hole_J==6);
		  }
		}

 	        else if (L2==4){

                  if (Core_Hole_J==0){
		    /* for Core_Hole_J==0 */ 
		    d52_p52 = penalty_value/5.0;
		    d52_p32 = penalty_value/5.0;
		    d52_p12 = penalty_value/5.0;
		    d52_m12 = penalty_value/5.0;
		    d52_m32 = penalty_value/5.0;
		    d52_m52 = penalty_value/5.0;

		    d32_p32 = penalty_value/5.0;
		    d32_p12 = penalty_value/5.0;
		    d32_m12 = penalty_value/5.0;
		    d32_m32 = penalty_value/5.0;
		  }

                  else{
		    /* for Core_Hole_J!=0 */ 
		    d52_p52 = penalty_value/5.0*(Core_Hole_J==1);
		    d52_p32 = penalty_value/5.0*(Core_Hole_J==2);
		    d52_p12 = penalty_value/5.0*(Core_Hole_J==3);
		    d52_m12 = penalty_value/5.0*(Core_Hole_J==4);
		    d52_m32 = penalty_value/5.0*(Core_Hole_J==5);
		    d52_m52 = penalty_value/5.0*(Core_Hole_J==6);

		    d32_p32 = penalty_value/5.0*(Core_Hole_J==7);
		    d32_p12 = penalty_value/5.0*(Core_Hole_J==8);
		    d32_m12 = penalty_value/5.0*(Core_Hole_J==9);
		    d32_m32 = penalty_value/5.0*(Core_Hole_J==10);
		  }
		}

                else if (L2==6){

                  if (Core_Hole_J==0){
		    /* for Core_Hole_J==0 */ 
		    d72_p72 = penalty_value/7.0;
		    d72_p52 = penalty_value/7.0;
		    d72_p32 = penalty_value/7.0;
		    d72_p12 = penalty_value/7.0;
		    d72_m12 = penalty_value/7.0;
		    d72_m32 = penalty_value/7.0;
		    d72_m52 = penalty_value/7.0;
		    d72_m72 = penalty_value/7.0;

		    d52_p52 = penalty_value/7.0;
		    d52_p32 = penalty_value/7.0;
		    d52_p12 = penalty_value/7.0;
		    d52_m12 = penalty_value/7.0;
		    d52_m32 = penalty_value/7.0;
		    d52_m52 = penalty_value/7.0;
		  }

                  else{
		    /* for Core_Hole_J!=0 */ 
		    d72_p72 = penalty_value/7.0*(Core_Hole_J==1);
		    d72_p52 = penalty_value/7.0*(Core_Hole_J==2);
		    d72_p32 = penalty_value/7.0*(Core_Hole_J==3);
		    d72_p12 = penalty_value/7.0*(Core_Hole_J==4);
		    d72_m12 = penalty_value/7.0*(Core_Hole_J==5);
		    d72_m32 = penalty_value/7.0*(Core_Hole_J==6);
		    d72_m52 = penalty_value/7.0*(Core_Hole_J==7);
		    d72_m72 = penalty_value/7.0*(Core_Hole_J==8);

		    d52_p52 = penalty_value/7.0*(Core_Hole_J==9);
		    d52_p32 = penalty_value/7.0*(Core_Hole_J==10);
		    d52_p12 = penalty_value/7.0*(Core_Hole_J==11);
		    d52_m12 = penalty_value/7.0*(Core_Hole_J==12);
		    d52_m32 = penalty_value/7.0*(Core_Hole_J==13);
		    d52_m52 = penalty_value/7.0*(Core_Hole_J==14);
		  }
		}

		/****************************************************
                         off-diagonal contribution on up-dn
                         for spin non-collinear
		****************************************************/

		if (SpinP_switch==3){

		  /***************
                          p
		  ***************/ 

		  if (L2==2){

		    /* real contribution of l+1/2 to off-diagonal up-down matrix */ 
		    sum2.r += 
		       d32_m12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      -d32_p12*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L  ]; 

		    /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */ 
		    sum2.i +=
		      -d32_m12*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      +d32_p12*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]; 

		    /* real contribution of l-1/2 for to off-diagonal up-down matrix */ 
		    sum2.r -=
		       d12_m12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      -d12_p12*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L  ]; 

		    /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */ 
		    sum2.i -=
		      -d12_m12*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      +d12_p12*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]; 
		  }

		  /***************
                         d
		  ***************/ 

		  else if (L2==4){

		    /* real contribution of l+1/2 to off diagonal up-down matrix */ 

		    sum2.r +=
		      -sqrt(3.0)*d52_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      +sqrt(3.0)*d52_m12*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L  ]
		                +d52_m32*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+3]
		                -d52_p32*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+1]
		                +d52_m32*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+4]
		                -d52_p32*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+2];

		    /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

		    sum2.i +=
		       sqrt(3.0)*d52_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+4]
		      -sqrt(3.0)*d52_m12*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L  ]
		                +d52_m32*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+4]
		                -d52_p32*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+1]
		                -d52_m32*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+3]
		                +d52_p32*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+2];

		    /* real contribution of l-1/2 for to diagonal up-down matrix */ 

		    sum2.r -=
		      -sqrt(3.0)*d32_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      +sqrt(3.0)*d32_m12*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L  ]
		                +d32_m32*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+3]
		                -d32_p32*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+1]
		                +d32_m32*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+4]
		                -d32_p32*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+2];

		    /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

		    sum2.i -=
		       sqrt(3.0)*d32_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+4]
		      -sqrt(3.0)*d32_m12*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L  ]
		                +d32_m32*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+4]
		                -d32_p32*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+1]
		                -d32_m32*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+3]
		                +d32_p32*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+2];

		  }

		  /***************
                         f
		  ***************/ 

		  else if (L2==6){

		    /* real contribution of l+1/2 to off diagonal up-down matrix */ 

		    sum2.r += 
		      -sqrt(6.0)*d72_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+1]
		      +sqrt(6.0)*d72_m12*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L  ]
		      -sqrt(2.5)*d72_p32*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      +sqrt(2.5)*d72_m32*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+1]
		      -sqrt(2.5)*d72_p32*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+4]
		      +sqrt(2.5)*d72_m32*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      -sqrt(1.5)*d72_p52*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+5]
		      +sqrt(1.5)*d72_m52*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      -sqrt(1.5)*d72_p52*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+6]
		      +sqrt(1.5)*d72_m52*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+4];

		    /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

		    sum2.i +=
		       sqrt(6.0)*d72_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      -sqrt(6.0)*d72_m12*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L  ]
		      +sqrt(2.5)*d72_p32*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+4]
		      -sqrt(2.5)*d72_m32*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+1]
		      -sqrt(2.5)*d72_p32*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      +sqrt(2.5)*d72_m32*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      +sqrt(1.5)*d72_p52*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+6]
		      -sqrt(1.5)*d72_m52*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      -sqrt(1.5)*d72_p52*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+5]
		      +sqrt(1.5)*d72_m52*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+4];

		    /* real contribution of l-1/2 for to off-diagonal up-down matrix */ 

		    sum2.r -= 
		      -sqrt(6.0)*d52_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+1]
		      +sqrt(6.0)*d52_m12*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L  ]
		      -sqrt(2.5)*d52_p32*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      +sqrt(2.5)*d52_m32*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+1]
		      -sqrt(2.5)*d52_p32*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+4]
		      +sqrt(2.5)*d52_m32*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      -sqrt(1.5)*d52_p52*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+5]
		      +sqrt(1.5)*d52_m52*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      -sqrt(1.5)*d52_p52*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+6]
		      +sqrt(1.5)*d52_m52*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+4];

		    /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

		    sum2.i -=
		       sqrt(6.0)*d52_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      -sqrt(6.0)*d52_m12*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L  ]
		      +sqrt(2.5)*d52_p32*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+4]
		      -sqrt(2.5)*d52_m32*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+1]
		      -sqrt(2.5)*d52_p32*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      +sqrt(2.5)*d52_m32*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+2]
		      +sqrt(1.5)*d52_p52*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+6]
		      -sqrt(1.5)*d52_m52*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+3]
		      -sqrt(1.5)*d52_p52*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+5]
		      +sqrt(1.5)*d52_m52*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+4];
		  }

		} /* if (SpinP_switch==3) */ 

		/****************************************************
                    off-diagonal contribution on up-up and dn-dn
		****************************************************/

		/* p */ 

		if (L2==2){

		  /* contribution of l+1/2 for up spin */ 

		  tmp0 =
		    +( d32_m12-3.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    +(-d32_m12+3.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L  ]; 

		  sum0.i += 0.5*tmp0;

		  /* contribution of l+1/2 for down spin */ 

		  tmp0 =
		    +(-d32_p12+3.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    +(+d32_p12-3.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L  ]; 
                  
		  sum1.i += 0.5*tmp0;

		  /* contribution of l-1/2 for up spin */

		  tmp0 = 
		     d12_m12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    -d12_m12*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L  ];

		  sum0.i += tmp0;

		  /* contribution of l-1/2 for down spin */ 

		  tmp0 = 
		    -d12_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    +d12_p12*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L  ];

		  sum1.i += tmp0;
		}

		/* d */ 

		else if (L2==4){

		  /* contribution of l+1/2 for up spin */ 

		  tmp0 =
		    +(     d52_m32-5.0*d52_p52)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		    +(    -d52_m32+5.0*d52_p52)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    +( 2.0*d52_m12-4.0*d52_p32)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+4]
		    +(-2.0*d52_m12+4.0*d52_p32)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+3]; 

		  sum0.i += 0.5*tmp0;

		  /* contribution of l+1/2 for down spin */ 

		  tmp0 =
		    -(     d52_p32-5.0*d52_m52)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		    -(    -d52_p32+5.0*d52_m52)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    -( 2.0*d52_p12-4.0*d52_m32)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+4]
		    -(-2.0*d52_p12+4.0*d52_m32)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+3]; 

		  sum1.i += 0.5*tmp0;

		  /* contribution of l-1/2 for up spin */ 

		  tmp0 =
		     (         4.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		    +(        -4.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    +( 3.0*d32_m12-d32_p32)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+4]
		    +(-3.0*d32_m12+d32_p32)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+3]; 

		  sum0.i += 0.5*tmp0;

		  /* contribution of l-1/2 for down spin */ 

		  tmp0 =
		     (        -4.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		    +(        +4.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    -( 3.0*d32_p12-d32_m32)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+4]
		    -(-3.0*d32_p12+d32_m32)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+3]; 

		  sum1.i += 0.5*tmp0;

		}

		/* f */ 

		else if (L2==6){

		  /* contribution of l+1/2 for up spin */ 

		  tmp0 =
		     ( 3.0*d72_m12-5.0*d72_p32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		    +(-3.0*d72_m12+5.0*d72_p32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    +( 2.0*d72_m32-6.0*d72_p52)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+4]
		    +(-2.0*d72_m32+6.0*d72_p52)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+3]
		    +( 1.0*d72_m52-7.0*d72_p72)*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+6]
		    +(-1.0*d72_m52+7.0*d72_p72)*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+5];

		  sum0.i += 0.5*tmp0;

		  /* contribution of l+1/2 for down spin */ 

		  tmp0 =
		    -( 3.0*d72_p12-5.0*d72_m32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		    -(-3.0*d72_p12+5.0*d72_m32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    -( 2.0*d72_p32-6.0*d72_m52)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+4]
		    -(-2.0*d72_p32+6.0*d72_m52)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+3]
		    -( 1.0*d72_p52-7.0*d72_m72)*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+6]
		    -(-1.0*d72_p52+7.0*d72_m72)*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+5];

		  sum1.i += 0.5*tmp0;

		  /* contribution of l-1/2 for up spin */ 

		  tmp0 =
		     ( 4.0*d52_m12-2.0*d52_p32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		    +(-4.0*d52_m12+2.0*d52_p32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    +( 5.0*d52_m32-1.0*d52_p52)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+4]
		    +(-5.0*d52_m32+1.0*d52_p52)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+3]
		    +(+6.0*d52_m52            )*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+6]
		    +(-6.0*d52_m52            )*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+5];

		  sum0.i += 0.5*tmp0;

		  /* contribution of l-1/2 for down spin */ 

		  tmp0 =
		    -( 4.0*d52_p12-2.0*d52_m32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+2]
		    -(-4.0*d52_p12+2.0*d52_m32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+1]
		    -( 5.0*d52_p32-1.0*d52_m52)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+4]
		    -(-5.0*d52_p32+1.0*d52_m52)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+3]
		    -(+6.0*d52_p52            )*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+6]
		    -(-6.0*d52_p52            )*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+5];

		  sum1.i += 0.5*tmp0;

		}

		/****************************************************
                      diagonal contribution on up-up and dn-dn
		****************************************************/

                /* s */

                if (L2==0){

		  /* VNL for j=l+1/2 */

                  sum0.r += d12_p12*OLP_CH[0][Mc_AN][k][m][L]*OLP_CH[0][Mj_AN][kl][n][L];
                  sum1.r += d12_m12*OLP_CH[0][Mc_AN][k][m][L]*OLP_CH[0][Mj_AN][kl][n][L];

		  /* note that VNL for j=l-1/2 is zero */
		}

                /* p */

                else if (L2==2){

		  /* VNL for j=l+1/2 */
                  sum0.r += 0.5*(d32_m12+3.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum0.r += 0.5*(d32_m12+3.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum0.r += 0.5*(        4.0*d32_p12)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];

                  sum1.r += 0.5*(d32_p12+3.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum1.r += 0.5*(d32_p12+3.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum1.r += 0.5*(        4.0*d32_m12)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];

		  /* VNL for j=l-1/2 */
                  sum0.r += d12_m12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum0.r += d12_m12*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum0.r += d12_p12*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];

                  sum1.r += d12_p12*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum1.r += d12_p12*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum1.r += d12_m12*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];
		}

                /* d */

                else if (L2==4){

		  /* VNL for j=l+1/2 */
                  sum0.r += 0.5*(            6.0*d52_p12)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum0.r += 0.5*(1.0*d52_m32+5.0*d52_p52)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum0.r += 0.5*(1.0*d52_m32+5.0*d52_p52)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];
                  sum0.r += 0.5*(2.0*d52_m12+4.0*d52_p32)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+3];
                  sum0.r += 0.5*(2.0*d52_m12+4.0*d52_p32)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+4];

                  sum1.r += 0.5*(            6.0*d52_m12)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum1.r += 0.5*(1.0*d52_p32+5.0*d52_m52)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum1.r += 0.5*(1.0*d52_p32+5.0*d52_m52)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];
                  sum1.r += 0.5*(2.0*d52_p12+4.0*d52_m32)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+3];
                  sum1.r += 0.5*(2.0*d52_p12+4.0*d52_m32)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+4];

		  /* VNL for j=l-1/2 */
                  sum0.r += 0.5*(            4.0*d32_p12)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum0.r += 0.5*(            4.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum0.r += 0.5*(            4.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];
                  sum0.r += 0.5*(3.0*d32_m12+1.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+3];
                  sum0.r += 0.5*(3.0*d32_m12+1.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+4];

                  sum1.r += 0.5*(            4.0*d32_m12)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum1.r += 0.5*(            4.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum1.r += 0.5*(            4.0*d32_p32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];
                  sum1.r += 0.5*(3.0*d32_p12+1.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+3];
                  sum1.r += 0.5*(3.0*d32_p12+1.0*d32_m32)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+4];
		}

                /* f */

                else if (L2==6){

		  /* VNL for j=l+1/2 */
                  sum0.r += 0.5*(            8.0*d72_p12)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum0.r += 0.5*(3.0*d72_m12+5.0*d72_p32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum0.r += 0.5*(3.0*d72_m12+5.0*d72_p32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];
                  sum0.r += 0.5*(2.0*d72_m32+6.0*d72_p52)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+3];
                  sum0.r += 0.5*(2.0*d72_m32+6.0*d72_p52)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+4];
                  sum0.r += 0.5*(1.0*d72_m52+7.0*d72_p72)*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+5];
                  sum0.r += 0.5*(1.0*d72_m52+7.0*d72_p72)*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+6];

                  sum1.r += 0.5*(            8.0*d72_m12)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum1.r += 0.5*(3.0*d72_p12+5.0*d72_m32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum1.r += 0.5*(3.0*d72_p12+5.0*d72_m32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];
                  sum1.r += 0.5*(2.0*d72_p32+6.0*d72_m52)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+3];
                  sum1.r += 0.5*(2.0*d72_p32+6.0*d72_m52)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+4];
                  sum1.r += 0.5*(1.0*d72_p52+7.0*d72_m72)*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+5];
                  sum1.r += 0.5*(1.0*d72_p52+7.0*d72_m72)*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+6];

		  /* VNL for j=l-1/2 */
                  sum0.r += 0.5*(            6.0*d52_p12)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum0.r += 0.5*(4.0*d52_m12+2.0*d52_p32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum0.r += 0.5*(4.0*d52_m12+2.0*d52_p32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];
                  sum0.r += 0.5*(5.0*d52_m32+1.0*d52_p52)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+3];
                  sum0.r += 0.5*(5.0*d52_m32+1.0*d52_p52)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+4];
                  sum0.r += 0.5*(6.0*d52_m52            )*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+5];
                  sum0.r += 0.5*(6.0*d52_m52            )*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+6];

                  sum1.r += 0.5*(            6.0*d52_m12)*OLP_CH[0][Mc_AN][k][m][L  ]*OLP_CH[0][Mj_AN][kl][n][L  ];
                  sum1.r += 0.5*(4.0*d52_p12+2.0*d52_m32)*OLP_CH[0][Mc_AN][k][m][L+1]*OLP_CH[0][Mj_AN][kl][n][L+1];
                  sum1.r += 0.5*(4.0*d52_p12+2.0*d52_m32)*OLP_CH[0][Mc_AN][k][m][L+2]*OLP_CH[0][Mj_AN][kl][n][L+2];
                  sum1.r += 0.5*(5.0*d52_p32+1.0*d52_m52)*OLP_CH[0][Mc_AN][k][m][L+3]*OLP_CH[0][Mj_AN][kl][n][L+3];
                  sum1.r += 0.5*(5.0*d52_p32+1.0*d52_m52)*OLP_CH[0][Mc_AN][k][m][L+4]*OLP_CH[0][Mj_AN][kl][n][L+4];
                  sum1.r += 0.5*(6.0*d52_p52            )*OLP_CH[0][Mc_AN][k][m][L+5]*OLP_CH[0][Mj_AN][kl][n][L+5];
                  sum1.r += 0.5*(6.0*d52_p52            )*OLP_CH[0][Mc_AN][k][m][L+6]*OLP_CH[0][Mj_AN][kl][n][L+6];
		}

	      } /* if (apply_flag==1) */ 

	      /* increment of L */
 
              L += 2*l + 1; 

            } /* mul */
	  } /* l */

	} /* if (GB_AN==Core_Hole_Atom) */

	NLH[0][m][n].r += sum0.r;    /* <up|VNL|up> */
	NLH[1][m][n].r += sum1.r;    /* <dn|VNL|dn> */
	NLH[2][m][n].r += sum2.r;    /* <up|VNL|dn> */

	NLH[0][m][n].i += sum0.i;    /* <up|VNL|up> */
	NLH[1][m][n].i += sum1.i;    /* <dn|VNL|dn> */
	NLH[2][m][n].i += sum2.i;    /* <up|VNL|dn> */

      } /* n */
    } /* m */

  } /* else if */

  /* freeing of array */

  for (i=0; i<2; i++){
    free(penalty[i]);
  }
  free(penalty);

}




