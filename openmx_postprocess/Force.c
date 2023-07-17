/**********************************************************************
  Force.c:
  
     Force.c is a subroutine to calculate force on atoms.

  Log of Force.c:

     22/Nov/2001  Released by T. Ozaki
     18/Apr/2013  Force3() modified by A.M. Ito

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time   0


static void dH_U_full(int Mc_AN, int h_AN, int q_AN,
		      double *****OLP, double ****v_eff,
		      double ***Hx, double ***Hy, double ***Hz);

static void dH_U_NC_full(int Mc_AN, int h_AN, int q_AN,
			 double *****OLP, dcomplex *****NC_v_eff,
			 dcomplex ****Hx, dcomplex ****Hy, dcomplex ****Hz);
		
static void dHNL(int where_flag,
		 int Mc_AN, int h_AN, int q_AN,
		 double ******DS_NL1,
		 dcomplex ***Hx, dcomplex ***Hy, dcomplex ***Hz);

static void dHVNA(int where_flag, int Mc_AN, int h_AN, int q_AN,
                  Type_DS_VNA *****DS_VNA1, 
                  double *****TmpHVNA2, double *****TmpHVNA3,
                  double **Hx, double **Hy, double **Hz);


static void dHNL_SO(
	     double *sumx0r,
	     double *sumy0r, 
	     double *sumz0r, 
	     double *sumx1r,
	     double *sumy1r, 
	     double *sumz1r, 
	     double *sumx2r,
	     double *sumy2r, 
	     double *sumz2r, 
	     double *sumx0i,
	     double *sumy0i, 
	     double *sumz0i, 
	     double *sumx1i,
	     double *sumy1i, 
	     double *sumz1i, 
	     double *sumx2i,
	     double *sumy2i, 
	     double *sumz2i, 
             double fugou, 
	     double PFp,
	     double PFm,
	     double ene_p,
	     double ene_m,
	     int l2, int *l,
	     int Mc_AN, int k,  int m,
	     int Mj_AN, int kl, int n,
	     double ******DS_NL1);

static void dHCH(int where_flag,
                 int Mc_AN, int h_AN, int q_AN,
                 double *****OLP1,
                 dcomplex ***Hx, dcomplex ***Hy, dcomplex ***Hz);

static void dHCH_SO(double *sumx0r, double *sumx0i, double *sumy0r, double *sumy0i, double *sumz0r, double *sumz0i,
		    double *sumx1r, double *sumx1i, double *sumy1r, double *sumy1i, double *sumz1r, double *sumz1i,
		    double *sumx2r, double *sumx2i, double *sumy2r, double *sumy2i, double *sumz2r, double *sumz2i,
		    double fugou,
		    int Mc_AN, int k,  int m,
		    int Mj_AN, int kl, int n,
		    int kg, int wakg,
		    double penalty_value, 
		    double *****OLP1);

static void MPI_OLP(double *****OLP1);
static void Force3();
static void Force4();
static void Force4B(double *****CDM0);

static void Force_HNL(double *****CDM0, double *****iDM0);
static void Force_CoreHole(double *****CDM0, double *****iDM0);

double Force(double *****H0,
             double ******DS_NL,
             double *****OLP,
             double *****CDM,
             double *****EDM)
{
  static int firsttime=1;
  int Nc,GNc,GRc,Cwan,s1,s2,BN_AB;
  int Mc_AN,Gc_AN,MNc,start_q_AN;
  double x,y,z,dx,dy,dz,tmp0,tmp1,tmp2,tmp3;
  double xx,r2,tot_den;
  double sumx,sumy,sumz,r,dege,pref;
  int i,j,k,l,Hwan,Qwan,so,p0,q,q0;
  int h_AN,Gh_AN,q_AN,Gq_AN;
  int ian,jan,kl,spin,spinmax,al,be,p,size_CDM0,size_iDM0;
  int tno0,tno1,tno2,Mh_AN,Mq_AN,n,num,size1,size2;
  int wanA,wanB,Gc_BN;
  int XC_P_switch;
  double time0;
  double dum,dge;
  double dEx,dEy,dEz;
  double Cxyz[4];
  double *Fx,*Fy,*Fz;
  dcomplex ***Hx;
  dcomplex ***Hy;
  dcomplex ***Hz;
  double ***HUx;
  double ***HUy;
  double ***HUz;
  dcomplex ****NC_HUx;
  dcomplex ****NC_HUy;
  dcomplex ****NC_HUz;
  double **HVNAx;
  double **HVNAy;
  double **HVNAz;
  double *****CDM0;
  double *****iDM0;
  double *tmp_array;
  double *tmp_array2;
  double Re00x,Re00y,Re00z;
  double Re11x,Re11y,Re11z;
  double Re01x,Re01y,Re01z;
  double Im00x,Im00y,Im00z;
  double Im11x,Im11y,Im11z;
  double Im01x,Im01y,Im01z;
  int *Snd_CDM0_Size,*Rcv_CDM0_Size;
  int *Snd_iDM0_Size,*Rcv_iDM0_Size;
  double TStime,TEtime;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom, Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;
  double stime,etime;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /****************************************************
   allocation of arrays:
  ****************************************************/

  Fx = (double*)malloc(sizeof(double)*(Matomnum+1));
  Fy = (double*)malloc(sizeof(double)*(Matomnum+1));
  Fz = (double*)malloc(sizeof(double)*(Matomnum+1));

  HVNAx = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (j=0; j<List_YOUSO[7]; j++){
    HVNAx[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  HVNAy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (j=0; j<List_YOUSO[7]; j++){
    HVNAy[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  HVNAz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (j=0; j<List_YOUSO[7]; j++){
    HVNAz[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  /* CDM0 */  
  size_CDM0 = 0;
  CDM0 = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
  for (k=0; k<=SpinP_switch; k++){
    CDM0[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = F_M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_CNO[Cwan];  
      }    

      CDM0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_CNO[Hwan];
        } 

        CDM0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          CDM0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          size_CDM0 += tno1;  
        }
      }
    }
  }

  Snd_CDM0_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_CDM0_Size = (int*)malloc(sizeof(int)*numprocs);

  /* iDM0 */  

  if ( SO_switch==1 || (Hub_U_switch==1 && SpinP_switch==3) || 1<=Constraint_NCS_switch
       || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 ){

    size_iDM0 = 0;
    iDM0 = (double*****)malloc(sizeof(double****)*2); 
    for (k=0; k<2; k++){
      iDM0[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = F_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];  
	}    

	iDM0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  } 

	  iDM0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    iDM0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    size_iDM0 += tno1;  
	  }
	}
      }
    }

    Snd_iDM0_Size = (int*)malloc(sizeof(int)*numprocs);
    Rcv_iDM0_Size = (int*)malloc(sizeof(int)*numprocs);
  }

  /****************************************************
                      PrintMemory
  ****************************************************/

  if (firsttime) {
    PrintMemory("Force: Hx",sizeof(dcomplex)*List_YOUSO[7]*List_YOUSO[7],NULL);
    PrintMemory("Force: Hy",sizeof(dcomplex)*List_YOUSO[7]*List_YOUSO[7],NULL);
    PrintMemory("Force: Hz",sizeof(dcomplex)*List_YOUSO[7]*List_YOUSO[7],NULL);
    PrintMemory("Force: CDM0",sizeof(double)*size_CDM0,NULL);
    if ( SO_switch==1 || (Hub_U_switch==1 && SpinP_switch==3) || 1<=Constraint_NCS_switch
         || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){
      PrintMemory("Force: iDM0",sizeof(double)*size_iDM0,NULL);
    }
    firsttime=0;
  }

  /****************************************************
    CDM to CDM0
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
  
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[Cwan];
  
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
  
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno2 = Spe_Total_CNO[Hwan];
  
      for (spin=0; spin<=SpinP_switch; spin++){
        for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
            CDM0[spin][Mc_AN][h_AN][i][j] = CDM[spin][Mc_AN][h_AN][i][j]; 
	  }
        }
      }
    }
  }
  
  /****************************************************
    iDM to iDM0
  ****************************************************/

  if ( SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1 && SpinP_switch==3) || 1<=Constraint_NCS_switch
       || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
	    iDM0[0][Mc_AN][h_AN][i][j] = iDM[0][0][Mc_AN][h_AN][i][j]; 
	    iDM0[1][Mc_AN][h_AN][i][j] = iDM[0][1][Mc_AN][h_AN][i][j]; 
	  }
	}
      }
    }
  }

  /****************************************************
   MPI:

   CDM0
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
 
        Snd_CDM0_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_CDM0_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR]!=0){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_CDM0_Size[IDR] = size2;
      }
      else{
        Rcv_CDM0_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

    }
    else{
      Snd_CDM0_Size[IDS] = 0;
      Rcv_CDM0_Size[IDR] = 0;
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

        size1 = Snd_CDM0_Size[IDS];

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
                  tmp_array[num] = CDM[spin][Mc_AN][h_AN][i][j];
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

        size2 = Rcv_CDM0_Size[IDR];
        
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
                  CDM0[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
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

  /****************************************************
   MPI:

   iDM0
  ****************************************************/

  if ( SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1 && SpinP_switch==3) || 1<=Constraint_NCS_switch
      || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

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
	  for (so=0; so<2; so++){
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
		    size1++; 
		  } 
		} 
	      }
	    }
	  }
 
	  Snd_iDM0_Size[IDS] = size1;
	  MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);

	}
	else{
	  Snd_iDM0_Size[IDS] = 0;
	}

	/* receiving of size of data */

	if (F_Rcv_Num[IDR]!=0){
	  MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	  Rcv_iDM0_Size[IDR] = size2;
	}
	else{
	  Rcv_iDM0_Size[IDR] = 0;
	}

	if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

      }
      else{
	Snd_iDM0_Size[IDS] = 0;
	Rcv_iDM0_Size[IDR] = 0;
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

	  size1 = Snd_iDM0_Size[IDS];

	  /* allocation of array */

	  tmp_array = (double*)malloc(sizeof(double)*size1);

	  /* multidimentional array to vector array */

	  num = 0;
	  for (so=0; so<2; so++){
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
		    tmp_array[num] = iDM[0][so][Mc_AN][h_AN][i][j];
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

	  size2 = Rcv_iDM0_Size[IDR];
        
	  /* allocation of array */
	  tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
	  MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	  num = 0;
	  for (so=0; so<2; so++){
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
		    iDM0[so][Mc_AN][h_AN][i][j] = tmp_array2[num];
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

  } /* if ( SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1 && SpinP_switch==3) 
     || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) */

  /****************************************************
                      #1 of force

              -\int \delta V_H drho_a/dx dr
                         and 
                force induced from PCC
              +\int V_XC drho_pcc/dx dr
  ****************************************************/

  if (myid==Host_ID && 0<level_stdout){
    printf("  Force calculation #1\n");fflush(stdout);
  }

  dtime(&stime);

  /*********************************************************
   set RefVxc_Grid, where the CA-LDA exchange-correlation 
   functional is alway used.
  *********************************************************/

  XC_P_switch = 1;
  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
    tot_den = ADensity_Grid_B[BN_AB] + ADensity_Grid_B[BN_AB];
    if (PCC_switch==1) {
      tot_den += PCCDensity_Grid_B[0][BN_AB] + PCCDensity_Grid_B[1][BN_AB];
    }
    RefVxc_Grid_B[BN_AB] = XC_Ceperly_Alder(tot_den,XC_P_switch);
  }
  
  Data_Grid_Copy_B2C_1( RefVxc_Grid_B,  RefVxc_Grid  );
  Data_Grid_Copy_B2C_1( dVHart_Grid_B,  dVHart_Grid  );
  Data_Grid_Copy_B2C_2( Vxc_Grid_B,     Vxc_Grid     );
  Data_Grid_Copy_B2C_2( Density_Grid_B, Density_Grid );

#pragma omp parallel shared(myid,Spe_OpenCore_flag,Spe_Atomic_PCC,Spe_VPS_RV,Spe_VPS_XV,Spe_Num_Mesh_VPS,Spe_PAO_RV,Spe_Atomic_Den,Spe_PAO_XV,Spe_Num_Mesh_PAO,time_per_atom,level_stdout,GridVol,Vxc_Grid,RefVxc_Grid,SpinP_switch,F_Vxc_flag,PCC_switch,dVHart_Grid,F_dVHart_flag,Gxyz,atv,MGridListAtom,CellListAtom,GridListAtom,GridN_Atom,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,sumx,sumy,sumz,Nc,GNc,GRc,MNc,Cxyz,x,y,z,dx,dy,dz,r,r2,tmp0,tmp1,tmp2,xx)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      sumx = 0.0;
      sumy = 0.0;
      sumz = 0.0;

      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

	GNc = GridListAtom[Mc_AN][Nc]; 
	GRc = CellListAtom[Mc_AN][Nc];
	MNc = MGridListAtom[Mc_AN][Nc]; 

	Get_Grid_XYZ(GNc,Cxyz);
	x = Cxyz[1] + atv[GRc][1]; 
	y = Cxyz[2] + atv[GRc][2]; 
	z = Cxyz[3] + atv[GRc][3];

	dx = Gxyz[Gc_AN][1] - x;
	dy = Gxyz[Gc_AN][2] - y;
	dz = Gxyz[Gc_AN][3] - z;
        r2 = dx*dx + dy*dy + dz*dz;
	r = sqrt(r2);
        xx = 0.5*log(r2);

	/* for empty atoms */
	if (r<1.0e-10) r = 1.0e-10;

	if (1.0e-14<r){

	  tmp0 = Dr_KumoF( Spe_Num_Mesh_PAO[Cwan], xx, r, 
                           Spe_PAO_XV[Cwan], Spe_PAO_RV[Cwan], Spe_Atomic_Den[Cwan]);

	  tmp1 = dVHart_Grid[MNc]*tmp0/r*F_dVHart_flag;
	  sumx += tmp1*dx;
	  sumy += tmp1*dy;
	  sumz += tmp1*dz;

          /* contribution of Exc^(0) */

	  tmp1 = RefVxc_Grid[MNc]*tmp0/r*F_Vxc_flag;
	  sumx += tmp1*dx;
	  sumy += tmp1*dy;
	  sumz += tmp1*dz;

	  /* partial core correction */
	  if (PCC_switch==1){

	    tmp0 = 0.5*F_Vxc_flag*Dr_KumoF( Spe_Num_Mesh_VPS[Cwan], xx, r, 
                                            Spe_VPS_XV[Cwan], Spe_VPS_RV[Cwan], Spe_Atomic_PCC[Cwan]);

	    if (SpinP_switch==0){
              tmp2 = 2.0*Vxc_Grid[0][MNc];
	    }
	    else {
              if (Spe_OpenCore_flag[Cwan]==0){
                tmp2 = Vxc_Grid[0][MNc] + Vxc_Grid[1][MNc];
	      }
              else if (Spe_OpenCore_flag[Cwan]==1){
                tmp2 = 2.0*Vxc_Grid[0][MNc];
	      }
              else if (Spe_OpenCore_flag[Cwan]==-1){
                tmp2 = 2.0*Vxc_Grid[1][MNc];
	      }
	    }

	    tmp1 = tmp2*tmp0/r;
	    sumx -= tmp1*dx;
	    sumy -= tmp1*dy;
	    sumz -= tmp1*dz;

            /* contribution of Exc^(0) */

	    tmp2 = 2.0*RefVxc_Grid[MNc];
	    tmp1 = tmp2*tmp0/r;
	    sumx += tmp1*dx;
	    sumy += tmp1*dy;
	    sumz += tmp1*dz;
	  }
	}
      }

      Gxyz[Gc_AN][17] = -sumx*GridVol;
      Gxyz[Gc_AN][18] = -sumy*GridVol;
      Gxyz[Gc_AN][19] = -sumz*GridVol;

      if (2<=level_stdout){
	printf("<Force>  force(1) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	       myid,Mc_AN,Gc_AN,-sumx*GridVol,-sumy*GridVol,-sumz*GridVol);fflush(stdout);
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

  } /* #pragma omp parallel */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for force#1=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
     added by T.Ohwaki

                      #1' of force
     contribution from an artificial wall applied 
     in the ESM method so that atoms cannot go beyond 
     the boundary of the unit cell along the a-axis.
  ****************************************************/

  if (ESM_switch!=0){

    double fx,xb,x0,x,a;

    /* modified by AdvanceSoft */
    xb = Grid_Origin[ESM_direction] + tv[1][ESM_direction];
    a = ESM_wall_height/pow(1.89,3.0);

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      x = Gxyz[Gc_AN][ESM_direction];
      x0 = xb - ESM_wall_position;
      dx = x - x0;

      if (0.0<dx){
        fx = 3.0*a*dx*dx;
      }
      else {
        fx = 0.0;
      }

      Gxyz[Gc_AN][16+ESM_direction] += fx;

      /*                                                                                                                          
      printf("Gc_AN=%2d fx=%15.12f\n",Gc_AN,fx);fflush(stdout);                                                                   
      */

      /* add an artifical force if required. */

      if(Arti_Force==1){
        if(Gc_AN==1) Gxyz[1][16+ESM_direction] += Arti_Grad;
        if(myid==0) printf("    adding force at the proc. 'Force #1' \n");
      }
    }
  }

  /****************************************************
   contraction

   H0
   OLP
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  
  if (Cnt_switch==1){

    Cont_Matrix0(H0[0],CntH0[0]);
    Cont_Matrix0(H0[1],CntH0[1]);
    Cont_Matrix0(H0[2],CntH0[2]);
    Cont_Matrix0(H0[3],CntH0[3]);

    Cont_Matrix0(OLP[0],CntOLP[0]);
    Cont_Matrix0(OLP[1],CntOLP[1]);
    Cont_Matrix0(OLP[2],CntOLP[2]);
    Cont_Matrix0(OLP[3],CntOLP[3]);
  }

  if ( Hub_U_switch==1 && Hub_U_occupation==1 ){
    MPI_OLP(OLP);
  }

  MPI_Barrier(mpi_comm_level1);

  /****************************************************
                      #2 of force

   kinetic operator and contribution from the Hubbard 
   term with the full representation in calculating 
   the occupation number
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Force calculation #2\n");fflush(stdout);
  }

#pragma omp parallel shared(time_per_atom,Gxyz,myid,level_stdout,iDM0,CDM0,CntH0,H0,F_Kin_flag,NC_v_eff,v_eff,OLP,Hub_U_occupation,Cnt_switch,F_NL_flag,List_YOUSO,RMI1,Zeeman_NCO_switch,Zeeman_NCS_switch,Constraint_NCS_switch,F_U_flag,Hub_U_switch,SO_switch,SpinP_switch,Spe_Total_CNO,F_G2M,natn,FNAN,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,dEx,dEy,dEz,h_AN,Gh_AN,Mh_AN,Hwan,ian,start_q_AN,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,so,i,j,k,Hx,Hy,Hz,HUx,HUy,HUz,NC_HUx,NC_HUy,NC_HUz,s1,s2,pref,spinmax,spin)
  {

    /* allocation of arrays */

    Hx = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hx[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hx[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hy = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hy[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hy[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hz = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hz[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hz[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    if (   (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
	   && (Hub_U_occupation==1 || Hub_U_occupation==2)
	   && SpinP_switch!=3 ){

      HUx = (double***)malloc(sizeof(double**)*3);
      for (i=0; i<3; i++){
	HUx[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	for (j=0; j<List_YOUSO[7]; j++){
	  HUx[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	}
      }

      HUy = (double***)malloc(sizeof(double**)*3);
      for (i=0; i<3; i++){
	HUy[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	for (j=0; j<List_YOUSO[7]; j++){
	  HUy[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	}
      }

      HUz = (double***)malloc(sizeof(double**)*3);
      for (i=0; i<3; i++){
	HUz[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	for (j=0; j<List_YOUSO[7]; j++){
	  HUz[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	}
      }
    }

    if (   (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
	   && (Hub_U_occupation==1 || Hub_U_occupation==2)
	   && SpinP_switch==3 ){

      NC_HUx = (dcomplex****)malloc(sizeof(dcomplex***)*2);
      for (i=0; i<2; i++){
	NC_HUx[i] = (dcomplex***)malloc(sizeof(dcomplex**)*2);
	for (j=0; j<2; j++){
	  NC_HUx[i][j] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (k=0; k<List_YOUSO[7]; k++){
	    NC_HUx[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}
      }

      NC_HUy = (dcomplex****)malloc(sizeof(dcomplex***)*2);
      for (i=0; i<2; i++){
	NC_HUy[i] = (dcomplex***)malloc(sizeof(dcomplex**)*2);
	for (j=0; j<2; j++){
	  NC_HUy[i][j] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (k=0; k<List_YOUSO[7]; k++){
	    NC_HUy[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}
      }

      NC_HUz = (dcomplex****)malloc(sizeof(dcomplex***)*2);
      for (i=0; i<2; i++){
	NC_HUz[i] = (dcomplex***)malloc(sizeof(dcomplex**)*2);
	for (j=0; j<2; j++){
	  NC_HUz[i][j] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (k=0; k<List_YOUSO[7]; k++){
	    NC_HUz[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}
      }
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];
	Hwan = WhatSpecies[Gh_AN];
	ian = Spe_Total_CNO[Hwan];

	if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1)
	 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) )
	  start_q_AN = 0;
	else 
	  start_q_AN = h_AN;

	for (q_AN=start_q_AN; q_AN<=FNAN[Gc_AN]; q_AN++){

	  Gq_AN = natn[Gc_AN][q_AN];
	  Mq_AN = F_G2M[Gq_AN];
	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];

	  if (0<=kl){

	    for (so=0; so<3; so++){
	      for (i=0; i<List_YOUSO[7]; i++){
		for (j=0; j<List_YOUSO[7]; j++){
		  Hx[so][i][j] = Complex(0.0,0.0);
		  Hy[so][i][j] = Complex(0.0,0.0);
		  Hz[so][i][j] = Complex(0.0,0.0);
		}
	      }
	    }

	    /****************************************************
             Contribution from LDA+U with 'full'treatment for
             counting the occupation number  
	    ****************************************************/

	    if ( (Hub_U_switch==1 && F_U_flag==1) || 1<=Constraint_NCS_switch
               || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 ){

	      /* full treatment and collinear case */

	      if (Hub_U_occupation==1 && SpinP_switch!=3){

		/* initialize HUx, HUy, and HUz */

		for (so=0; so<3; so++){
		  for (i=0; i<List_YOUSO[7]; i++){
		    for (j=0; j<List_YOUSO[7]; j++){
		      HUx[so][i][j] = 0.0;
		      HUy[so][i][j] = 0.0;
		      HUz[so][i][j] = 0.0;
		    }
		  }
		}

		dH_U_full(Mc_AN,h_AN,q_AN,OLP,v_eff,HUx,HUy,HUz);

		/* add the contribution to Hx, Hy, and Hz */

		if (SpinP_switch==0) spinmax = 0;
		else                 spinmax = 1;

		for (spin=0; spin<=spinmax; spin++){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      Hx[spin][i][j].r += HUx[spin][i][j];
		      Hy[spin][i][j].r += HUy[spin][i][j];
		      Hz[spin][i][j].r += HUz[spin][i][j];
		    }
		  }
		}
	      }

	      /* full treatment and non-collinear case */

	      else if (Hub_U_occupation==1 && SpinP_switch==3){

		/* initialize NC_HUx, NC_HUy, and NC_HUz */

		for (s1=0; s1<2; s1++){
		  for (s2=0; s2<2; s2++){
		    for (i=0; i<List_YOUSO[7]; i++){
		      for (j=0; j<List_YOUSO[7]; j++){
			NC_HUx[s1][s2][i][j] = Complex(0.0,0.0);
			NC_HUy[s1][s2][i][j] = Complex(0.0,0.0);
			NC_HUz[s1][s2][i][j] = Complex(0.0,0.0);
		      }
		    }
		  }
		}

		dH_U_NC_full(Mc_AN,h_AN,q_AN,OLP,NC_v_eff,NC_HUx,NC_HUy,NC_HUz);

		/******************************************************
                       add the contribution to Hx, Hy, and Hz

                       Hx[0] 00
                       Hx[1] 11
                       Hx[2] 01
		******************************************************/

		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){

		    Hx[0][i][j].r += NC_HUx[0][0][i][j].r;
		    Hy[0][i][j].r += NC_HUy[0][0][i][j].r;
		    Hz[0][i][j].r += NC_HUz[0][0][i][j].r;

		    Hx[1][i][j].r += NC_HUx[1][1][i][j].r;
		    Hy[1][i][j].r += NC_HUy[1][1][i][j].r;
		    Hz[1][i][j].r += NC_HUz[1][1][i][j].r;

		    Hx[2][i][j].r += NC_HUx[0][1][i][j].r;
		    Hy[2][i][j].r += NC_HUy[0][1][i][j].r;
		    Hz[2][i][j].r += NC_HUz[0][1][i][j].r;

		    Hx[0][i][j].i += NC_HUx[0][0][i][j].i;
		    Hy[0][i][j].i += NC_HUy[0][0][i][j].i;
		    Hz[0][i][j].i += NC_HUz[0][0][i][j].i;

		    Hx[1][i][j].i += NC_HUx[1][1][i][j].i;
		    Hy[1][i][j].i += NC_HUy[1][1][i][j].i;
		    Hz[1][i][j].i += NC_HUz[1][1][i][j].i;

		    Hx[2][i][j].i += NC_HUx[0][1][i][j].i;
		    Hy[2][i][j].i += NC_HUy[0][1][i][j].i;
		    Hz[2][i][j].i += NC_HUz[0][1][i][j].i;
		  }
		}
	      }

	    }

	    /****************************************************
                               H0 = dKinetic
	    ****************************************************/

	    if (F_Kin_flag==1){  

	      /* in case of no obital optimization */

	      if (Cnt_switch==0){
		if (h_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      Hx[0][i][j].r += H0[1][Mc_AN][q_AN][i][j];
		      Hy[0][i][j].r += H0[2][Mc_AN][q_AN][i][j];
		      Hz[0][i][j].r += H0[3][Mc_AN][q_AN][i][j];

		      Hx[1][i][j].r += H0[1][Mc_AN][q_AN][i][j];
		      Hy[1][i][j].r += H0[2][Mc_AN][q_AN][i][j];
		      Hz[1][i][j].r += H0[3][Mc_AN][q_AN][i][j];
		    }
		  }
		} 

		else if (h_AN!=0 && q_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      Hx[0][i][j].r += H0[1][Mc_AN][h_AN][j][i];
		      Hy[0][i][j].r += H0[2][Mc_AN][h_AN][j][i];
		      Hz[0][i][j].r += H0[3][Mc_AN][h_AN][j][i];

		      Hx[1][i][j].r += H0[1][Mc_AN][h_AN][j][i];
		      Hy[1][i][j].r += H0[2][Mc_AN][h_AN][j][i];
		      Hz[1][i][j].r += H0[3][Mc_AN][h_AN][j][i];
		    }
		  }
		}
	      }

	      /* in case of obital optimization */

	      else{

		if (h_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){

		      Hx[0][i][j].r += CntH0[1][Mc_AN][q_AN][i][j];
		      Hy[0][i][j].r += CntH0[2][Mc_AN][q_AN][i][j];
		      Hz[0][i][j].r += CntH0[3][Mc_AN][q_AN][i][j];

		      Hx[1][i][j].r += CntH0[1][Mc_AN][q_AN][i][j];
		      Hy[1][i][j].r += CntH0[2][Mc_AN][q_AN][i][j];
		      Hz[1][i][j].r += CntH0[3][Mc_AN][q_AN][i][j];

		    }
		  }
		} 

		else if (h_AN!=0 && q_AN==0){
		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){

		      Hx[0][i][j].r += CntH0[1][Mc_AN][h_AN][j][i];
		      Hy[0][i][j].r += CntH0[2][Mc_AN][h_AN][j][i];
		      Hz[0][i][j].r += CntH0[3][Mc_AN][h_AN][j][i];

		      Hx[1][i][j].r += CntH0[1][Mc_AN][h_AN][j][i];
		      Hy[1][i][j].r += CntH0[2][Mc_AN][h_AN][j][i];
		      Hz[1][i][j].r += CntH0[3][Mc_AN][h_AN][j][i];

		    }
		  }
		}
	      } 

            } /* if F_Kin_flag */

	    /****************************************************
                              \sum rho*dH
	    ****************************************************/

	    /* non-spin polarization */

	    if (SpinP_switch==0){

              if (q_AN==h_AN) pref = 2.0;
              else            pref = 4.0; 

	      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){
		  dEx += pref*CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r;
		  dEy += pref*CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r;
		  dEz += pref*CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r;
		}
	      }
	    }

	    /* collinear spin polarized or non-colliear without SO and LDA+U */

	    else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
		 && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

              if (q_AN==h_AN) pref = 1.0;
              else            pref = 2.0; 

	      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		  dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r
		               + CDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].r);
		  dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r
		               + CDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].r);
		  dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r
		               + CDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].r);
		}
	      }
	    }

	    /* spin collinear with spin-orbit coupling */

	    else if ( SpinP_switch==1 && SO_switch==1 ){
	      printf("Spin-orbit coupling is not supported for collinear DFT calculations.\n");fflush(stdout);
	      MPI_Finalize();
	      exit(0);
	    }

	    /* spin non-collinear with spin-orbit coupling or with LDA+U */

	    else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
		  || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) ){

	      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		  dEx +=   CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r
		         - iDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].i
		         + CDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].r
		         - iDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].i
		     + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hx[2][i][j].r
		     - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hx[2][i][j].i;

		  dEy +=   CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r
		         - iDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].i
		         + CDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].r
		         - iDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].i
		     + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hy[2][i][j].r
		     - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hy[2][i][j].i; 

		  dEz +=   CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r
		         - iDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].i
		         + CDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].r
		         - iDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].i
		     + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hz[2][i][j].r
		     - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hz[2][i][j].i; 

		}
	      }
	    }

	  }  /* if (0<=kl) */
	}  /* q_AN */
      }  /* h_AN */

      /****************************************************
                        #2 of Force
      ****************************************************/

      if (2<=level_stdout){
	printf("<Force>  force(2) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	       myid,Mc_AN,Gc_AN,dEx,dEy,dEz);fflush(stdout);
      }

      Gxyz[Gc_AN][17] += dEx;
      Gxyz[Gc_AN][18] += dEy;
      Gxyz[Gc_AN][19] += dEz;

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hx[i][j]);
      }
      free(Hx[i]);
    }
    free(Hx);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hy[i][j]);
      }
      free(Hy[i]);
    }
    free(Hy);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hz[i][j]);
      }
      free(Hz[i]);
    }
    free(Hz);

    if (   (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
	   && (Hub_U_occupation==1 || Hub_U_occupation==2)
	   && SpinP_switch!=3 ){

      for (i=0; i<3; i++){
	for (j=0; j<List_YOUSO[7]; j++){
	  free(HUx[i][j]);
	}
	free(HUx[i]);
      }
      free(HUx);

      for (i=0; i<3; i++){
	for (j=0; j<List_YOUSO[7]; j++){
	  free(HUy[i][j]);
	}
	free(HUy[i]);
      }
      free(HUy);

      for (i=0; i<3; i++){
	for (j=0; j<List_YOUSO[7]; j++){
	  free(HUz[i][j]);
	}
	free(HUz[i]);
      }
      free(HUz);
    }

    if (   (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
	   && (Hub_U_occupation==1 || Hub_U_occupation==2)
	   && SpinP_switch==3 ){

      for (i=0; i<2; i++){
	for (j=0; j<2; j++){
	  for (k=0; k<List_YOUSO[7]; k++){
	    free(NC_HUx[i][j][k]);
	  }
	  free(NC_HUx[i][j]);
	}
	free(NC_HUx[i]);
      }
      free(NC_HUx);

      for (i=0; i<2; i++){
	for (j=0; j<2; j++){
	  for (k=0; k<List_YOUSO[7]; k++){
	    free(NC_HUy[i][j][k]);
	  }
	  free(NC_HUy[i][j]);
	}
	free(NC_HUy[i]);
      }
      free(NC_HUy);

      for (i=0; i<2; i++){
	for (j=0; j<2; j++){
	  for (k=0; k<List_YOUSO[7]; k++){
	    free(NC_HUz[i][j][k]);
	  }
	  free(NC_HUz[i][j]);
	}
	free(NC_HUz[i]);
      }
      free(NC_HUz);
    }

  } /* #pragma omp parallel */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for force#2=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                      #3 of Force

               dn/dx * (VNA + dVH + Vxc)
            or 
               dn/dx * (dVH + Vxc)
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Force calculation #3\n");fflush(stdout);
  }

  Force3();

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for force#3=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                      #4 of Force

       Force4:   n * dVNA/dx
       Force4B:  from separable VNA projectors
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Force calculation #4\n");fflush(stdout);
  }

  if (ProExpn_VNA==0 && F_VNA_flag==1){
    Force4();
  }
  else if (ProExpn_VNA==1 && F_VNA_flag==1){
    Force4B(CDM0);
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                      #5 of Force

               Contribution from overlap
  ****************************************************/

  dtime(&stime);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Force calculation #5\n");fflush(stdout);
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Fx[Mc_AN] = 0.0;
    Fy[Mc_AN] = 0.0;
    Fz[Mc_AN] = 0.0;
  }

#pragma omp parallel shared(time_per_atom,Fx,Fy,Fz,CntOLP,OLP,Cnt_switch,EDM,SpinP_switch,Spe_Total_CNO,natn,FNAN,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,dum,dx,dy,dz)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){

	    if (SpinP_switch==0){
	      dum = 2.0*EDM[0][Mc_AN][h_AN][i][j];
	    }
	    else if (SpinP_switch==1 || SpinP_switch==3){
	      dum = EDM[0][Mc_AN][h_AN][i][j] + EDM[1][Mc_AN][h_AN][i][j];
	    }

	    if (Cnt_switch==0){
	      dx = dum*OLP[1][Mc_AN][h_AN][i][j];
	      dy = dum*OLP[2][Mc_AN][h_AN][i][j];
	      dz = dum*OLP[3][Mc_AN][h_AN][i][j];
	    }
	    else{
	      dx = dum*CntOLP[1][Mc_AN][h_AN][i][j];
	      dy = dum*CntOLP[2][Mc_AN][h_AN][i][j];
	      dz = dum*CntOLP[3][Mc_AN][h_AN][i][j];
	    } 

	    Fx[Mc_AN] = Fx[Mc_AN] - 2.0*dx;
	    Fy[Mc_AN] = Fy[Mc_AN] - 2.0*dy;
	    Fz[Mc_AN] = Fz[Mc_AN] - 2.0*dz;

	  }
	}
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    } 

  } /* #pragma omp parallel */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for force#5=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                  add #5 of Force
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    Gxyz[Gc_AN][17] += Fx[Mc_AN];
    Gxyz[Gc_AN][18] += Fy[Mc_AN];
    Gxyz[Gc_AN][19] += Fz[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(5) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
             myid,Mc_AN,Gc_AN,Fx[Mc_AN],Fy[Mc_AN],Fz[Mc_AN]);fflush(stdout);
    }
  }

  /****************************************************************
   In case that the dual representation is used for evaluation of 
   the occupation number in the LDA+U method, the following force
   term is added.  
  ****************************************************************/

  if (   (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
	 && (Hub_U_occupation==1 || Hub_U_occupation==2)
	 && SpinP_switch!=3 ){

    HUx = (double***)malloc(sizeof(double**)*3);
    for (i=0; i<3; i++){
      HUx[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	HUx[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }

    HUy = (double***)malloc(sizeof(double**)*3);
    for (i=0; i<3; i++){
      HUy[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	HUy[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }

    HUz = (double***)malloc(sizeof(double**)*3);
    for (i=0; i<3; i++){
      HUz[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	HUz[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }
  }

  if ( (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) 
        && F_U_flag==1 && Hub_U_occupation==2){

    if (myid==Host_ID)  printf("  Force calculation for LDA_U with dual\n");fflush(stdout);

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Fx[Mc_AN] = 0.0;
      Fy[Mc_AN] = 0.0;
      Fz[Mc_AN] = 0.0;
    }

    /****************************************************
      if (SpinP_switch!=3)

      collinear case
    ****************************************************/

    if (SpinP_switch!=3){

      if (SpinP_switch==0){
	spinmax = 0;
	dege = 2.0;
      }
      else{
	spinmax = 1;
	dege = 1.0;
      }

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
 
	for (spin=0; spin<=spinmax; spin++){

	  for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    Mh_AN = F_G2M[Gh_AN];
	    Hwan = WhatSpecies[Gh_AN];

	    /* non-orbital optimization */

	    if (Cnt_switch==0){

	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){

		  tmp1 = 0.0; 
		  tmp2 = 0.0; 
		  tmp3 = 0.0; 

		  for (k=0; k<Spe_Total_NO[Cwan]; k++){
		    tmp1 += v_eff[spin][Mc_AN][i][k]*OLP[1][Mc_AN][h_AN][k][j];
		    tmp2 += v_eff[spin][Mc_AN][i][k]*OLP[2][Mc_AN][h_AN][k][j];
		    tmp3 += v_eff[spin][Mc_AN][i][k]*OLP[3][Mc_AN][h_AN][k][j];
		  }

		  for (k=0; k<Spe_Total_NO[Hwan]; k++){
		    tmp1 += v_eff[spin][Mh_AN][k][j]*OLP[1][Mc_AN][h_AN][i][k];
		    tmp2 += v_eff[spin][Mh_AN][k][j]*OLP[2][Mc_AN][h_AN][i][k];
		    tmp3 += v_eff[spin][Mh_AN][k][j]*OLP[3][Mc_AN][h_AN][i][k];
		  }

		  dx = tmp1*dege*CDM[spin][Mc_AN][h_AN][i][j];
		  dy = tmp2*dege*CDM[spin][Mc_AN][h_AN][i][j];
		  dz = tmp3*dege*CDM[spin][Mc_AN][h_AN][i][j];

		  Fx[Mc_AN] += dx;
		  Fy[Mc_AN] += dy;
		  Fz[Mc_AN] += dz;
		}
	      }
	    }

	    /* orbital optimization */

	    else if (Cnt_switch==1){

	      /* HUx, HUy, HUz for primitive orbital */

	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){

		  tmp1 = 0.0; 
		  tmp2 = 0.0; 
		  tmp3 = 0.0; 

		  for (k=0; k<Spe_Total_NO[Cwan]; k++){
		    tmp1 += v_eff[spin][Mc_AN][i][k]*OLP[1][Mc_AN][h_AN][k][j];
		    tmp2 += v_eff[spin][Mc_AN][i][k]*OLP[2][Mc_AN][h_AN][k][j];
		    tmp3 += v_eff[spin][Mc_AN][i][k]*OLP[3][Mc_AN][h_AN][k][j];
		  }

		  for (k=0; k<Spe_Total_NO[Hwan]; k++){
		    tmp1 += v_eff[spin][Mh_AN][k][j]*OLP[1][Mc_AN][h_AN][i][k];
		    tmp2 += v_eff[spin][Mh_AN][k][j]*OLP[2][Mc_AN][h_AN][i][k];
		    tmp3 += v_eff[spin][Mh_AN][k][j]*OLP[3][Mc_AN][h_AN][i][k];
		  }

		  HUx[0][i][j] = tmp1;
		  HUy[0][i][j] = tmp2;
		  HUz[0][i][j] = tmp3;
		}
	      }

	      /* contract HUx, HUy, HUz */

	      for (al=0; al<Spe_Total_CNO[Cwan]; al++){
		for (be=0; be<Spe_Total_CNO[Hwan]; be++){

		  tmp1 = 0.0; 
		  tmp2 = 0.0; 
		  tmp3 = 0.0; 

		  for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
		    p0 = Spe_Trans_Orbital[Cwan][al][p];
		    for (q=0; q<Spe_Specified_Num[Hwan][be]; q++){
		      q0 = Spe_Trans_Orbital[Hwan][be][q];
		      tmp0 = CntCoes[Mc_AN][al][p]*CntCoes[Mh_AN][be][q]; 
		      tmp1 += tmp0*HUx[0][p0][q0];
		      tmp2 += tmp0*HUy[0][p0][q0];
		      tmp3 += tmp0*HUz[0][p0][q0];
		    }
		  }

		  dx = tmp1*dege*CDM[spin][Mc_AN][h_AN][al][be];
		  dy = tmp2*dege*CDM[spin][Mc_AN][h_AN][al][be];
		  dz = tmp3*dege*CDM[spin][Mc_AN][h_AN][al][be];

		  Fx[Mc_AN] += dx;
		  Fy[Mc_AN] += dy;
		  Fz[Mc_AN] += dz;
		}
	      }

	    }

	  }
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      } 

    }

    /****************************************************
      if (SpinP_switch==3)

      spin non-collinear
    ****************************************************/

    else {

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	dtime(&Stime_atom);

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Mh_AN = F_G2M[Gh_AN];
	  Hwan = WhatSpecies[Gh_AN];

          kl = RMI1[Mc_AN][h_AN][0];

	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      Re00x = 0.0;  Re00y = 0.0;   Re00z = 0.0; 
	      Re11x = 0.0;  Re11y = 0.0;   Re11z = 0.0; 
	      Re01x = 0.0;  Re01y = 0.0;   Re01z = 0.0; 

	      Im00x = 0.0;  Im00y = 0.0;   Im00z = 0.0; 
	      Im11x = 0.0;  Im11y = 0.0;   Im11z = 0.0; 
	      Im01x = 0.0;  Im01y = 0.0;   Im01z = 0.0; 

	      for (k=0; k<Spe_Total_NO[Cwan]; k++){

		Re00x += NC_v_eff[0][0][Mc_AN][i][k].r * OLP[1][Mc_AN][h_AN][k][j];
		Re00y += NC_v_eff[0][0][Mc_AN][i][k].r * OLP[2][Mc_AN][h_AN][k][j];
		Re00z += NC_v_eff[0][0][Mc_AN][i][k].r * OLP[3][Mc_AN][h_AN][k][j];

		Re11x += NC_v_eff[1][1][Mc_AN][i][k].r * OLP[1][Mc_AN][h_AN][k][j];
		Re11y += NC_v_eff[1][1][Mc_AN][i][k].r * OLP[2][Mc_AN][h_AN][k][j];
		Re11z += NC_v_eff[1][1][Mc_AN][i][k].r * OLP[3][Mc_AN][h_AN][k][j];

		Re01x += NC_v_eff[0][1][Mc_AN][i][k].r * OLP[1][Mc_AN][h_AN][k][j];
		Re01y += NC_v_eff[0][1][Mc_AN][i][k].r * OLP[2][Mc_AN][h_AN][k][j];
		Re01z += NC_v_eff[0][1][Mc_AN][i][k].r * OLP[3][Mc_AN][h_AN][k][j];

		Im00x += NC_v_eff[0][0][Mc_AN][i][k].i * OLP[1][Mc_AN][h_AN][k][j];
		Im00y += NC_v_eff[0][0][Mc_AN][i][k].i * OLP[2][Mc_AN][h_AN][k][j];
		Im00z += NC_v_eff[0][0][Mc_AN][i][k].i * OLP[3][Mc_AN][h_AN][k][j];

		Im11x += NC_v_eff[1][1][Mc_AN][i][k].i * OLP[1][Mc_AN][h_AN][k][j];
		Im11y += NC_v_eff[1][1][Mc_AN][i][k].i * OLP[2][Mc_AN][h_AN][k][j];
		Im11z += NC_v_eff[1][1][Mc_AN][i][k].i * OLP[3][Mc_AN][h_AN][k][j];

		Im01x += NC_v_eff[0][1][Mc_AN][i][k].i * OLP[1][Mc_AN][h_AN][k][j];
		Im01y += NC_v_eff[0][1][Mc_AN][i][k].i * OLP[2][Mc_AN][h_AN][k][j];
		Im01z += NC_v_eff[0][1][Mc_AN][i][k].i * OLP[3][Mc_AN][h_AN][k][j];

	      }

	      for (k=0; k<Spe_Total_NO[Hwan]; k++){

		Re00x += NC_v_eff[0][0][Mh_AN][k][j].r * OLP[1][Mc_AN][h_AN][i][k];
		Re00y += NC_v_eff[0][0][Mh_AN][k][j].r * OLP[2][Mc_AN][h_AN][i][k];
		Re00z += NC_v_eff[0][0][Mh_AN][k][j].r * OLP[3][Mc_AN][h_AN][i][k];

		Re11x += NC_v_eff[1][1][Mh_AN][k][j].r * OLP[1][Mc_AN][h_AN][i][k];
		Re11y += NC_v_eff[1][1][Mh_AN][k][j].r * OLP[2][Mc_AN][h_AN][i][k];
		Re11z += NC_v_eff[1][1][Mh_AN][k][j].r * OLP[3][Mc_AN][h_AN][i][k];

		Re01x += NC_v_eff[0][1][Mh_AN][k][j].r * OLP[1][Mc_AN][h_AN][i][k];
		Re01y += NC_v_eff[0][1][Mh_AN][k][j].r * OLP[2][Mc_AN][h_AN][i][k];
		Re01z += NC_v_eff[0][1][Mh_AN][k][j].r * OLP[3][Mc_AN][h_AN][i][k];

		Im00x += NC_v_eff[0][0][Mh_AN][k][j].i * OLP[1][Mc_AN][h_AN][i][k];
		Im00y += NC_v_eff[0][0][Mh_AN][k][j].i * OLP[2][Mc_AN][h_AN][i][k];
		Im00z += NC_v_eff[0][0][Mh_AN][k][j].i * OLP[3][Mc_AN][h_AN][i][k];

		Im11x += NC_v_eff[1][1][Mh_AN][k][j].i * OLP[1][Mc_AN][h_AN][i][k];
		Im11y += NC_v_eff[1][1][Mh_AN][k][j].i * OLP[2][Mc_AN][h_AN][i][k];
		Im11z += NC_v_eff[1][1][Mh_AN][k][j].i * OLP[3][Mc_AN][h_AN][i][k];

		Im01x += NC_v_eff[0][1][Mh_AN][k][j].i * OLP[1][Mc_AN][h_AN][i][k];
		Im01y += NC_v_eff[0][1][Mh_AN][k][j].i * OLP[2][Mc_AN][h_AN][i][k];
		Im01z += NC_v_eff[0][1][Mh_AN][k][j].i * OLP[3][Mc_AN][h_AN][i][k];

	      }

	      dx =      Re00x*CDM0[0][Mc_AN][h_AN][i][j]
                  +     Re11x*CDM0[1][Mc_AN][h_AN][i][j]
                  + 2.0*Re01x*CDM0[2][Mc_AN][h_AN][i][j]
                  -     Im00x*iDM0[0][Mc_AN][h_AN][i][j]
                  -     Im11x*iDM0[1][Mc_AN][h_AN][i][j]
	  	  - 2.0*Im01x*CDM0[3][Mc_AN][h_AN][i][j];

	      dy =      Re00y*CDM0[0][Mc_AN][h_AN][i][j]
                  +     Re11y*CDM0[1][Mc_AN][h_AN][i][j]
                  + 2.0*Re01y*CDM0[2][Mc_AN][h_AN][i][j]
                  -     Im00y*iDM0[0][Mc_AN][h_AN][i][j]
                  -     Im11y*iDM0[1][Mc_AN][h_AN][i][j]
	  	  - 2.0*Im01y*CDM0[3][Mc_AN][h_AN][i][j];

	      dz =      Re00z*CDM0[0][Mc_AN][h_AN][i][j]
                  +     Re11z*CDM0[1][Mc_AN][h_AN][i][j]
                  + 2.0*Re01z*CDM0[2][Mc_AN][h_AN][i][j]
                  -     Im00z*iDM0[0][Mc_AN][h_AN][i][j]
                  -     Im11z*iDM0[1][Mc_AN][h_AN][i][j]
	  	  - 2.0*Im01z*CDM0[3][Mc_AN][h_AN][i][j];

	      Fx[Mc_AN] += 0.5*dx;
	      Fy[Mc_AN] += 0.5*dy;
	      Fz[Mc_AN] += 0.5*dz;

	      Re00x = 0.0;  Re00y = 0.0;   Re00z = 0.0; 
	      Re11x = 0.0;  Re11y = 0.0;   Re11z = 0.0; 
	      Re01x = 0.0;  Re01y = 0.0;   Re01z = 0.0; 

	      Im00x = 0.0;  Im00y = 0.0;   Im00z = 0.0; 
	      Im11x = 0.0;  Im11y = 0.0;   Im11z = 0.0; 
	      Im01x = 0.0;  Im01y = 0.0;   Im01z = 0.0; 

	      for (k=0; k<Spe_Total_NO[Hwan]; k++){

		Re00x += NC_v_eff[0][0][Mh_AN][j][k].r * OLP[1][Mc_AN][h_AN][i][k];
		Re00y += NC_v_eff[0][0][Mh_AN][j][k].r * OLP[2][Mc_AN][h_AN][i][k];
		Re00z += NC_v_eff[0][0][Mh_AN][j][k].r * OLP[3][Mc_AN][h_AN][i][k];

		Re11x += NC_v_eff[1][1][Mh_AN][j][k].r * OLP[1][Mc_AN][h_AN][i][k];
		Re11y += NC_v_eff[1][1][Mh_AN][j][k].r * OLP[2][Mc_AN][h_AN][i][k];
		Re11z += NC_v_eff[1][1][Mh_AN][j][k].r * OLP[3][Mc_AN][h_AN][i][k];

		Re01x += NC_v_eff[0][1][Mh_AN][j][k].r * OLP[1][Mc_AN][h_AN][i][k];
		Re01y += NC_v_eff[0][1][Mh_AN][j][k].r * OLP[2][Mc_AN][h_AN][i][k];
		Re01z += NC_v_eff[0][1][Mh_AN][j][k].r * OLP[3][Mc_AN][h_AN][i][k];

		Im00x += NC_v_eff[0][0][Mh_AN][j][k].i * OLP[1][Mc_AN][h_AN][i][k];
		Im00y += NC_v_eff[0][0][Mh_AN][j][k].i * OLP[2][Mc_AN][h_AN][i][k];
		Im00z += NC_v_eff[0][0][Mh_AN][j][k].i * OLP[3][Mc_AN][h_AN][i][k];

		Im11x += NC_v_eff[1][1][Mh_AN][j][k].i * OLP[1][Mc_AN][h_AN][i][k];
		Im11y += NC_v_eff[1][1][Mh_AN][j][k].i * OLP[2][Mc_AN][h_AN][i][k];
		Im11z += NC_v_eff[1][1][Mh_AN][j][k].i * OLP[3][Mc_AN][h_AN][i][k];

		Im01x += NC_v_eff[0][1][Mh_AN][j][k].i * OLP[1][Mc_AN][h_AN][i][k];
		Im01y += NC_v_eff[0][1][Mh_AN][j][k].i * OLP[2][Mc_AN][h_AN][i][k];
		Im01z += NC_v_eff[0][1][Mh_AN][j][k].i * OLP[3][Mc_AN][h_AN][i][k];

	      }

	      for (k=0; k<Spe_Total_NO[Cwan]; k++){

		Re00x += NC_v_eff[0][0][Mc_AN][k][i].r * OLP[1][Mc_AN][h_AN][k][j];
		Re00y += NC_v_eff[0][0][Mc_AN][k][i].r * OLP[2][Mc_AN][h_AN][k][j];
		Re00z += NC_v_eff[0][0][Mc_AN][k][i].r * OLP[3][Mc_AN][h_AN][k][j];

		Re11x += NC_v_eff[1][1][Mc_AN][k][i].r * OLP[1][Mc_AN][h_AN][k][j];
		Re11y += NC_v_eff[1][1][Mc_AN][k][i].r * OLP[2][Mc_AN][h_AN][k][j];
		Re11z += NC_v_eff[1][1][Mc_AN][k][i].r * OLP[3][Mc_AN][h_AN][k][j];

		Re01x += NC_v_eff[0][1][Mc_AN][k][i].r * OLP[1][Mc_AN][h_AN][k][j];
		Re01y += NC_v_eff[0][1][Mc_AN][k][i].r * OLP[2][Mc_AN][h_AN][k][j];
		Re01z += NC_v_eff[0][1][Mc_AN][k][i].r * OLP[3][Mc_AN][h_AN][k][j];

		Im00x += NC_v_eff[0][0][Mc_AN][k][i].i * OLP[1][Mc_AN][h_AN][k][j];
		Im00y += NC_v_eff[0][0][Mc_AN][k][i].i * OLP[2][Mc_AN][h_AN][k][j];
		Im00z += NC_v_eff[0][0][Mc_AN][k][i].i * OLP[3][Mc_AN][h_AN][k][j];

		Im11x += NC_v_eff[1][1][Mc_AN][k][i].i * OLP[1][Mc_AN][h_AN][k][j];
		Im11y += NC_v_eff[1][1][Mc_AN][k][i].i * OLP[2][Mc_AN][h_AN][k][j];
		Im11z += NC_v_eff[1][1][Mc_AN][k][i].i * OLP[3][Mc_AN][h_AN][k][j];

		Im01x += NC_v_eff[0][1][Mc_AN][k][i].i * OLP[1][Mc_AN][h_AN][k][j];
		Im01y += NC_v_eff[0][1][Mc_AN][k][i].i * OLP[2][Mc_AN][h_AN][k][j];
		Im01z += NC_v_eff[0][1][Mc_AN][k][i].i * OLP[3][Mc_AN][h_AN][k][j];

	      }

	      dx =      Re00x*CDM0[0][Mh_AN][kl][j][i]
                  +     Re11x*CDM0[1][Mh_AN][kl][j][i]
                  + 2.0*Re01x*CDM0[2][Mh_AN][kl][j][i]
                  -     Im00x*iDM0[0][Mh_AN][kl][j][i]
                  -     Im11x*iDM0[1][Mh_AN][kl][j][i]
	  	  - 2.0*Im01x*CDM0[3][Mh_AN][kl][j][i];

	      dy =      Re00y*CDM0[0][Mh_AN][kl][j][i]
                  +     Re11y*CDM0[1][Mh_AN][kl][j][i]
                  + 2.0*Re01y*CDM0[2][Mh_AN][kl][j][i]
                  -     Im00y*iDM0[0][Mh_AN][kl][j][i]
                  -     Im11y*iDM0[1][Mh_AN][kl][j][i]
	  	  - 2.0*Im01y*CDM0[3][Mh_AN][kl][j][i];

	      dz =      Re00z*CDM0[0][Mh_AN][kl][j][i]
                  +     Re11z*CDM0[1][Mh_AN][kl][j][i]
                  + 2.0*Re01z*CDM0[2][Mh_AN][kl][j][i]
                  -     Im00z*iDM0[0][Mh_AN][kl][j][i]
                  -     Im11z*iDM0[1][Mh_AN][kl][j][i]
	  	  - 2.0*Im01z*CDM0[3][Mh_AN][kl][j][i];

	      Fx[Mc_AN] += 0.5*dx;
	      Fy[Mc_AN] += 0.5*dy;
	      Fz[Mc_AN] += 0.5*dz;
               
	    }
	  }
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      } 
    }

    /****************************************************
      add the contribution
    ****************************************************/

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];

      Gxyz[Gc_AN][17] += Fx[Mc_AN];
      Gxyz[Gc_AN][18] += Fy[Mc_AN];
      Gxyz[Gc_AN][19] += Fz[Mc_AN];

      if (2<=level_stdout){
	printf("<Force>  force(LDA_U_dual) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	       myid,Mc_AN,Gc_AN,Fx[Mc_AN],Fy[Mc_AN],Fz[Mc_AN]);fflush(stdout);
      }
    }

  } /* if ( (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) 
       && F_U_flag==1 && Hub_U_occupation==2) */

  /****************************************************
                 Force arising from HNL
  ****************************************************/

  Force_HNL(CDM0, iDM0);

  /****************************************************
        Force arising from the penalty functional 
        to create a core hole 
  ****************************************************/

  if (core_hole_state_flag==1){
    Force_CoreHole(CDM0, iDM0);
  }

  /****************************************************
   freeing of arrays:
  ****************************************************/

  if (   (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
	 && (Hub_U_occupation==1 || Hub_U_occupation==2)
	 && SpinP_switch!=3 ){

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(HUx[i][j]);
      }
      free(HUx[i]);
    }
    free(HUx);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(HUy[i][j]);
      }
      free(HUy[i]);
    }
    free(HUy);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(HUz[i][j]);
      }
      free(HUz[i]);
    }
    free(HUz);
  }

  free(Fx);
  free(Fy);
  free(Fz);

  for (j=0; j<List_YOUSO[7]; j++){
    free(HVNAx[j]);
  }
  free(HVNAx);

  for (j=0; j<List_YOUSO[7]; j++){
    free(HVNAy[j]);
  }
  free(HVNAy);

  for (j=0; j<List_YOUSO[7]; j++){
    free(HVNAz[j]);
  }
  free(HVNAz);

  /* CDM0 */  
  for (k=0; k<=SpinP_switch; k++){
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = F_M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_CNO[Cwan];  
      }    

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_CNO[Hwan];
        } 

        for (i=0; i<tno0; i++){
          free(CDM0[k][Mc_AN][h_AN][i]);
        }
        free(CDM0[k][Mc_AN][h_AN]);
      }
      free(CDM0[k][Mc_AN]);
    }
    free(CDM0[k]);
  }
  free(CDM0);

  free(Snd_CDM0_Size);
  free(Rcv_CDM0_Size);

  /* iDM0 */  
  if ( SO_switch==1 || (Hub_U_switch==1 && SpinP_switch==3) || 1<=Constraint_NCS_switch
      || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 ){

    for (k=0; k<2; k++){

      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = F_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];  
	}    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  } 

	  for (i=0; i<tno0; i++){
	    free(iDM0[k][Mc_AN][h_AN][i]);
	  }
          free(iDM0[k][Mc_AN][h_AN]);
	}
        free(iDM0[k][Mc_AN]);
      }
      free(iDM0[k]);
    }
    free(iDM0);

    free(Snd_iDM0_Size);
    free(Rcv_iDM0_Size);
  }

  /* for time */

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}



void Force3()
{
  /****************************************************
    	#3 of Force
   	dn/dx * (VNA + dVH + Vxc)
        or
        dn/dx * (dVH + Vxc)
  ****************************************************/
  /* for OpenMP */

  /* MPI */
  int numprocs, myid;
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /**********************************************************
              main loop for calculation of force #3
  **********************************************************/
  /* shared memory for force */

  double** Vpot_grid = (double**)malloc(sizeof(double*)*(SpinP_switch + 1));
  {
    double* p2 = (double*)malloc(sizeof(double)*(SpinP_switch + 1)*Max_GridN_Atom);

    int spin;
    for (spin = 0; spin<(SpinP_switch + 1); spin++) {
      Vpot_grid[spin] = p2;
      p2 += Max_GridN_Atom;
    }
  }

  double*** dChi0 = (double***)malloc(sizeof(double**)*Max_GridN_Atom);
  {
    double** p2 = (double**)malloc(sizeof(double*)*Max_GridN_Atom*List_YOUSO[7]);
    double* p = (double*)malloc(sizeof(double)*Max_GridN_Atom*List_YOUSO[7] * 3);
    int Nc;
    for (Nc = 0; Nc<Max_GridN_Atom; Nc++) {
      dChi0[Nc] = p2;
      p2 += List_YOUSO[7];
      int i;
      for (i = 0; i<List_YOUSO[7]; i++) {
	dChi0[Nc][i] = p;
	p += 3;
      }
    }
  }

  double sumx = 0.0; /* this must be defined out of parallel pragma */
  double sumy = 0.0;
  double sumz = 0.0;

#pragma omp parallel
  {		

    /* allocation of arrays */

    double** dorbs0 = (double**)malloc(sizeof(double*) * 4);
    {
      int i;
      for (i = 0; i<4; i++) {
	dorbs0[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }
    double* orbs1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

    struct WORK_DORBITAL work_dObs;
    Get_dOrbitals_init(&work_dObs);

    double time1 = 0.0;
    double time2 = 0.0;
    double last_time;
    double current_time;


    int Mc_AN;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {

      int Gc_AN = M2G[Mc_AN];
      int Cwan = WhatSpecies[Gc_AN];
      int NO0 = Spe_Total_CNO[Cwan];


      /***********************************
                   calc dOrb0
      ***********************************/
#pragma omp barrier   /* this barrier is necessary to wait (1)clearing sumx, sumy, sumz, and (2)initializing work_dObs */
      dtime(&last_time);

      int Nc;
#pragma omp for
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++) {

	int GNc = GridListAtom[Mc_AN][Nc];
	int GRc = CellListAtom[Mc_AN][Nc];
	int MNc = MGridListAtom[Mc_AN][Nc];

	double Cxyz[4];
	Get_Grid_XYZ(GNc, Cxyz);
	double x = Cxyz[1] + atv[GRc][1];
	double y = Cxyz[2] + atv[GRc][2];
	double z = Cxyz[3] + atv[GRc][3];
	double dx = x - Gxyz[Gc_AN][1];
	double dy = y - Gxyz[Gc_AN][2];
	double dz = z - Gxyz[Gc_AN][3];

	if (Cnt_switch == 0) {
	  /* AITUNE201704 : Get_dOrbitals(Cwan, dx, dy, dz, dorbs0); */
	  /* Get_dOrbitals_work(Cwan, dx, dy, dz, dorbs0, work_dObs); */

          /* start: direct inlining of Get_dOrbitals_work */

          int wan=Cwan;
          double x=dx,y=dy,z=dz; 
	  int i, i1, i2, i3, i4, j, l, l1;
	  int po, L0, Mul0, M0;
	  int mp_min, mp_max, m;
	  double dum, dum1, dum2, dum3, dum4;
	  double siQ, coQ, siP, coP, a, b, c;
	  double dx, rm, tmp0, tmp1, id, d;
	  double drx, dry, drz, R, Q, P, Rmin;
	  double S_coordinate[3];
	  double **RF = work_dObs.RF;
	  double **dRF = work_dObs.dRF;
	  double **AF = work_dObs.AF;
	  double **dAFQ = work_dObs.dAFQ;
	  double **dAFP = work_dObs.dAFP;
	  double h1, h2, h3, f1, f2, f3, f4, dfx, dfx2;
	  double g1, g2, x1, x2, y1, y2, y12, y22, f, df, df2;
	  double dRx, dRy, dRz, dQx, dQy, dQz, dPx, dPy, dPz;
	  double dChiR, dChiQ, dChiP, h, sum0, sum1;
	  double SH[Supported_MaxL * 2 + 1][2];
	  double dSHt[Supported_MaxL * 2 + 1][2];
	  double dSHp[Supported_MaxL * 2 + 1][2];

	  /* start calc. */

	  Rmin = 10e-14;

	  xyz2spherical(x, y, z, 0.0, 0.0, 0.0, S_coordinate);
	  R = S_coordinate[0];
	  Q = S_coordinate[1];
	  P = S_coordinate[2];

	  if (R < Rmin) {
	    x = x + Rmin;
	    y = y + Rmin;
	    z = z + Rmin;
	    xyz2spherical(x, y, z, 0.0, 0.0, 0.0, S_coordinate);
	    R = S_coordinate[0];
	    Q = S_coordinate[1];
	    P = S_coordinate[2];
	  }

	  po = 0;
	  mp_min = 0;
	  mp_max = Spe_Num_Mesh_PAO[wan] - 1;

	  if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan] - 1] < R) {

	    for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
	      for (Mul0 = 0; Mul0 < Spe_Num_Basis[wan][L0]; Mul0++) {
		RF[L0][Mul0] = 0.0;
		dRF[L0][Mul0] = 0.0;
	      }
	    }

	    po = 1;
	  }

	  else if (R < Spe_PAO_RV[wan][0]) {

	    m = 4;
	    rm = Spe_PAO_RV[wan][m];

	    h1 = Spe_PAO_RV[wan][m - 1] - Spe_PAO_RV[wan][m - 2];
	    h2 = Spe_PAO_RV[wan][m] - Spe_PAO_RV[wan][m - 1];
	    h3 = Spe_PAO_RV[wan][m + 1] - Spe_PAO_RV[wan][m];

	    x1 = rm - Spe_PAO_RV[wan][m - 1];
	    x2 = rm - Spe_PAO_RV[wan][m];
	    y1 = x1 / h2;
	    y2 = x2 / h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1 / h2 / dum;
	    dum2 = h2 / h1 / dum;
	    dum = h2 + h3;
	    dum3 = h2 / h3 / dum;
	    dum4 = h3 / h2 / dum;

	    for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
	      for (Mul0 = 0; Mul0 < Spe_Num_Basis[wan][L0]; Mul0++) {

		f1 = Spe_PAO_RWF[wan][L0][Mul0][m - 2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m - 1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][m + 1];

		if (m == 1) {
		  h1 = -(h2 + h3);
		  f1 = f4;
		} else if (m == (Spe_Num_Mesh_PAO[wan] - 1)) {
		  h3 = -(h1 + h2);
		  f4 = f1;
		}

		dum = f3 - f2;
		g1 = dum*dum1 + (f2 - f1)*dum2;
		g2 = (f4 - f3)*dum3 + dum*dum4;

		f = y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		df = 2.0*y2 / h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y22*(2.0*f2 + h2*g1) / h2
		  + 2.0*y1 / h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		  - y12*(2.0*f3 - h2*g2) / h2;

		if (L0 == 0) {
		  a = 0.0;
		  b = 0.5*df / rm;
		  c = 0.0;
		  d = f - b*rm*rm;
		}

		else if (L0 == 1) {
		  a = (rm*df - f) / (2.0*rm*rm*rm);
		  b = 0.0;
		  c = df - 3.0*a*rm*rm;
		  d = 0.0;
		}

		else {
		  b = (3.0*f - rm*df) / (rm*rm);
		  a = (f - b*rm*rm) / (rm*rm*rm);
		  c = 0.0;
		  d = 0.0;
		}

		RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;
		dRF[L0][Mul0] = 3.0*a*R*R + 2.0*b*R + c;

	      }
	    }

	  }

	  else {

	    do {
	      m = (mp_min + mp_max) / 2;
	      if (Spe_PAO_RV[wan][m] < R)
		mp_min = m;
	      else
		mp_max = m;
	    } while ((mp_max - mp_min) != 1);
	    m = mp_max;

	    h1 = Spe_PAO_RV[wan][m - 1] - Spe_PAO_RV[wan][m - 2];
	    h2 = Spe_PAO_RV[wan][m] - Spe_PAO_RV[wan][m - 1];
	    h3 = Spe_PAO_RV[wan][m + 1] - Spe_PAO_RV[wan][m];

	    x1 = R - Spe_PAO_RV[wan][m - 1];
	    x2 = R - Spe_PAO_RV[wan][m];
	    y1 = x1 / h2;
	    y2 = x2 / h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1 / h2 / dum;
	    dum2 = h2 / h1 / dum;
	    dum = h2 + h3;
	    dum3 = h2 / h3 / dum;
	    dum4 = h3 / h2 / dum;

	    for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
	      for (Mul0 = 0; Mul0 < Spe_Num_Basis[wan][L0]; Mul0++) {

		f1 = Spe_PAO_RWF[wan][L0][Mul0][m - 2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m - 1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][m + 1];

		if (m == 1) {
		  h1 = -(h2 + h3);
		  f1 = f4;
		} else if (m == (Spe_Num_Mesh_PAO[wan] - 1)) {
		  h3 = -(h1 + h2);
		  f4 = f1;
		}

		dum = f3 - f2;
		g1 = dum*dum1 + (f2 - f1)*dum2;
		g2 = (f4 - f3)*dum3 + dum*dum4;

		f = y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		df = 2.0*y2 / h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y2*y2*(2.0*f2 + h2*g1) / h2
		  + 2.0*y1 / h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		  - y1*y1*(2.0*f3 - h2*g2) / h2;

		RF[L0][Mul0] = f;
		dRF[L0][Mul0] = df;

	      }
	    }
	  }

	  /* dr/dx,y,z, dQ/dx,y,z, dP/dx,y,z and dAngular */

	  if (po == 0) {

	    /* Angular */
	    siQ = sin(Q);
	    coQ = cos(Q);
	    siP = sin(P);
	    coP = cos(P);

	    dRx = siQ*coP;
	    dRy = siQ*siP;
	    dRz = coQ;

	    if (Rmin < R) {
	      dQx = coQ*coP / R;
	      dQy = coQ*siP / R;
	      dQz = -siQ / R;
	    } else {
	      dQx = 0.0;
	      dQy = 0.0;
	      dQz = 0.0;
	    }

	    /* RICS note 72P */

	    if (Rmin < R) {
	      dPx = -siP / R;
	      dPy = coP / R;
	      dPz = 0.0;
	    } else {
	      dPx = 0.0;
	      dPy = 0.0;
	      dPz = 0.0;
	    }

	    for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
	      if (L0 == 0) {
		AF[0][0] = 0.282094791773878;
		dAFQ[0][0] = 0.0;
		dAFP[0][0] = 0.0;
	      } else if (L0 == 1) {
		dum = 0.48860251190292*siQ;

		AF[1][0] = dum*coP;
		AF[1][1] = dum*siP;
		AF[1][2] = 0.48860251190292*coQ;

		dAFQ[1][0] = 0.48860251190292*coQ*coP;
		dAFQ[1][1] = 0.48860251190292*coQ*siP;
		dAFQ[1][2] = -0.48860251190292*siQ;

		dAFP[1][0] = -0.48860251190292*siP;
		dAFP[1][1] = 0.48860251190292*coP;
		dAFP[1][2] = 0.0;
	      } else if (L0 == 2) {

		dum1 = siQ*siQ;
		dum2 = 1.09254843059208*siQ*coQ;
		AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		AF[2][2] = 1.09254843059208*dum1*siP*coP;
		AF[2][3] = dum2*coP;
		AF[2][4] = dum2*siP;

		dAFQ[2][0] = -1.89234939151512*siQ*coQ;
		dAFQ[2][1] = 1.09254843059208*siQ*coQ*(1.0 - 2.0*siP*siP);
		dAFQ[2][2] = 2.18509686118416*siQ*coQ*siP*coP;
		dAFQ[2][3] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*coP;
		dAFQ[2][4] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*siP;

		/* RICS note 72P */

		dAFP[2][0] = 0.0;
		dAFP[2][1] = -2.18509686118416*siQ*siP*coP;
		dAFP[2][2] = 1.09254843059208*siQ*(1.0 - 2.0*siP*siP);
		dAFP[2][3] = -1.09254843059208*coQ*siP;
		dAFP[2][4] = 1.09254843059208*coQ*coP;
	      }

	      else if (L0 == 3) {
		AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP - siP*siP);
		AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);

		dAFQ[3][0] = 0.373176332590116*siQ*(-15.0*coQ*coQ + 3.0);
		dAFQ[3][1] = 0.457045799464466*coP*coQ*(15.0*coQ*coQ - 11.0);
		dAFQ[3][2] = 0.457045799464466*siP*coQ*(15.0*coQ*coQ - 11.0);
		dAFQ[3][3] = 1.44530572132028*(coP*coP - siP*siP)*siQ*(2.0*coQ*coQ - siQ*siQ);
		dAFQ[3][4] = 2.89061144264055*coP*siP*siQ*(2.0*coQ*coQ - siQ*siQ);
		dAFQ[3][5] = 1.770130769779932*coP*coQ*siQ*siQ*(-3.0 + 4.0*coP*coP);
		dAFQ[3][6] = 1.770130769779932*coQ*siP*siQ*siQ*(3.0 - 4.0*siP*siP);

		/* RICS note 72P */

		dAFP[3][0] = 0.0;
		dAFP[3][1] = 0.457045799464466*siP*(-5.0*coQ*coQ + 1.0);
		dAFP[3][2] = 0.457045799464466*coP*(5.0*coQ*coQ - 1.0);
		dAFP[3][3] = -5.781222885281120*coP*coQ*siP*siQ;
		dAFP[3][4] = 2.89061144264055*coQ*siQ*(coP*coP - siP*siP);
		dAFP[3][5] = 1.770130769779932*siP*siQ*siQ*(1.0 - 4.0*coP*coP);
		dAFP[3][6] = 1.770130769779932*coP*siQ*siQ*(1.0 - 4.0*siP*siP);
	      }

	      else if (4 <= L0) {

		/* calculation of complex spherical harmonics functions */
		for (m = -L0; m <= L0; m++) {
		  ComplexSH(L0, m, Q, P, SH[L0 + m], dSHt[L0 + m], dSHp[L0 + m]);
		}

		/* transformation of complex to real */
		for (i = 0; i < (L0 * 2 + 1); i++) {

		  sum0 = 0.0;
		  sum1 = 0.0;
		  for (j = 0; j < (L0 * 2 + 1); j++) {
		    sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1];
		    sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0];
		  }
		  AF[L0][i] = sum0 + sum1;

		  sum0 = 0.0;
		  sum1 = 0.0;
		  for (j = 0; j < (L0 * 2 + 1); j++) {
		    sum0 += Comp2Real[L0][i][j].r*dSHt[j][0] - Comp2Real[L0][i][j].i*dSHt[j][1];
		    sum1 += Comp2Real[L0][i][j].r*dSHt[j][1] + Comp2Real[L0][i][j].i*dSHt[j][0];
		  }
		  dAFQ[L0][i] = sum0 + sum1;

		  sum0 = 0.0;
		  sum1 = 0.0;
		  for (j = 0; j < (L0 * 2 + 1); j++) {
		    sum0 += Comp2Real[L0][i][j].r*dSHp[j][0] - Comp2Real[L0][i][j].i*dSHp[j][1];
		    sum1 += Comp2Real[L0][i][j].r*dSHp[j][1] + Comp2Real[L0][i][j].i*dSHp[j][0];
		  }
		  dAFP[L0][i] = sum0 + sum1;
		}

	      }
	    }
	  }

	  /* Chi */
	  i1 = -1;
	  for (L0 = 0; L0 <= Spe_MaxL_Basis[wan]; L0++) {
	    for (Mul0 = 0; Mul0 < Spe_Num_Basis[wan][L0]; Mul0++) {
	      for (M0 = 0; M0 <= 2 * L0; M0++) {

		i1++;

		dChiR = dRF[L0][Mul0] * AF[L0][M0];
		dChiQ = RF[L0][Mul0] * dAFQ[L0][M0];
		dChiP = RF[L0][Mul0] * dAFP[L0][M0];

		dorbs0[0][i1] = RF[L0][Mul0] * AF[L0][M0];
		dorbs0[1][i1] = -dRx*dChiR - dQx*dChiQ - dPx*dChiP;
		dorbs0[2][i1] = -dRy*dChiR - dQy*dChiQ - dPy*dChiP;
		dorbs0[3][i1] = -dRz*dChiR - dQz*dChiQ - dPz*dChiP;
	      }
	    }
	  }

          /* end: direct inlining of Get_dOrbitals_work */

	} 
        else {
	  Get_Cnt_dOrbitals(Mc_AN, dx, dy, dz, dorbs0);
	}

	int k;
	for (k = 0; k<3; k++) {
	  int i;
	  for (i = 0; i<NO0; i++) {
	    dChi0[Nc][i][k] = dorbs0[k + 1][i];
	  }
	}

	if (SpinP_switch == 0 || SpinP_switch == 1) {

	  int spin;
	  for (spin = 0; spin <= SpinP_switch; spin++) {
						
	    double Vpt;
	    if (0 <= MNc) {
	      if (E_Field_switch == 1) {

		if (ProExpn_VNA == 0) {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc]
		    + F_VNA_flag*VNA_Grid[MNc]
		    + F_VEF_flag*VEF_Grid[MNc];

		} else {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc]
		    + F_VEF_flag*VEF_Grid[MNc];

		}

	      } else {
		if (ProExpn_VNA == 0) {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc]
		    + F_VNA_flag*VNA_Grid[MNc];

		} else {

		  Vpt = F_dVHart_flag*dVHart_Grid[MNc]
		    + F_Vxc_flag*Vxc_Grid[spin][MNc];

		}
	      }
	    } else {
	      Vpt = 0.0;
	    }

	    if (SpinP_switch == 0) {
	      Vpot_grid[0][Nc] = 4.0 * Vpt;
	    } else if (SpinP_switch == 1) {
	      Vpot_grid[spin][Nc] = 2.0 * Vpt;
	    }

	  }
	}
 
        else if (SpinP_switch == 3) {

	  /* spin non-collinear */
					
	  double ReVpt11;
	  double ReVpt22;
	  double ReVpt21;
	  double ImVpt21;

	  if (0 <= MNc) {
	    if (E_Field_switch == 1) {

	      if (ProExpn_VNA == 0) {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[0][MNc]
		  + F_VNA_flag*VNA_Grid[MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[1][MNc]
		  + F_VNA_flag*VNA_Grid[MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];
	      } else {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[0][MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[1][MNc]
		  + F_VEF_flag*VEF_Grid[MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];
	      }

	    } else {

	      if (ProExpn_VNA == 0) {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[0][MNc]
		  + F_VNA_flag*VNA_Grid[MNc];

		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc]
		  + F_Vxc_flag*Vxc_Grid[1][MNc]
		  + F_VNA_flag*VNA_Grid[MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];

	      } else {

		ReVpt11 = F_dVHart_flag*dVHart_Grid[MNc] + F_Vxc_flag*Vxc_Grid[0][MNc];
		ReVpt22 = F_dVHart_flag*dVHart_Grid[MNc] + F_Vxc_flag*Vxc_Grid[1][MNc];

		ReVpt21 = F_Vxc_flag*Vxc_Grid[2][MNc];
		ImVpt21 = -F_Vxc_flag*Vxc_Grid[3][MNc];
	      }

	    }
	  } else {
	    ReVpt11 = 0.0;
	    ReVpt22 = 0.0;
	    ReVpt21 = 0.0;
	    ImVpt21 = 0.0;
	  }

	  Vpot_grid[0][Nc] = 2.0 * ReVpt11;
	  Vpot_grid[1][Nc] = 2.0 * ReVpt22;
	  Vpot_grid[2][Nc] = 4.0 * ReVpt21;
	  Vpot_grid[3][Nc] = 4.0 * ImVpt21;

	}

      }/* Nc, here omp barrier is called implicitly because of end of for loop */

      dtime(&current_time);
      time1 += current_time - last_time;
      last_time = current_time;

      int h_AN;
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

	int Gh_AN = natn[Gc_AN][h_AN];
	int Mh_AN = F_G2M[Gh_AN];
	int Rnh = ncn[Gc_AN][h_AN];
	int Hwan = WhatSpecies[Gh_AN];
	int NO1 = Spe_Total_CNO[Hwan];

	int Nog;
#pragma omp for reduction (+:sumx, sumy, sumz)
	for (Nog = 0; Nog<NumOLG[Mc_AN][h_AN]; Nog++) {

	  int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

	  double** const ai_dorbs0 = dChi0[Nc];

	  /* set orbs1 */

	  if (G2ID[Gh_AN] == myid) {
	    int j;
	    for (j = 0; j<NO1; j++) {
	      orbs1[j] = Orbs_Grid[Mh_AN][Nh][j];
	    }
	  } else {
	    int j;
	    for (j = 0; j<NO1; j++) {
	      orbs1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	    }
	  }

	  int spin;
	  for (spin = 0; spin <= SpinP_switch; spin++) {

	    double tmpx = 0.0;
	    double tmpy = 0.0;
	    double tmpz = 0.0;

	    int i;
	    for (i = 0; i<NO0; i++) {
	      double tmp0 = 0.0;
	      int j;
	      for (j = 0; j<NO1; j++) {
		tmp0 += orbs1[j] * DM[0][spin][Mc_AN][h_AN][i][j];
	      }

	      tmpx += ai_dorbs0[i][0] * tmp0;
	      tmpy += ai_dorbs0[i][1] * tmp0;
	      tmpz += ai_dorbs0[i][2] * tmp0;
	    }

	    /* due to difference in the definition between density matrix and density */
	    double Vpt = Vpot_grid[spin][Nc];
	    sumx += tmpx * Vpt;
	    sumy += tmpy * Vpt;
	    sumz += tmpz * Vpt;


	  }/* spin */
	}/* Nog */
      }/* h_AN, here omp barrier is called implicitly because of end of for loop  */

      /***********************************
                  calc force #3
      ***********************************/

#pragma omp master
      {
	Gxyz[Gc_AN][17] += sumx*GridVol;
	Gxyz[Gc_AN][18] += sumy*GridVol;
	Gxyz[Gc_AN][19] += sumz*GridVol;

	sumx = 0.0;
	sumy = 0.0;
	sumz = 0.0;
      }

      dtime(&current_time);
      time2 += current_time - last_time;
      last_time = current_time;


#pragma omp master
      if (2 <= level_stdout) {
	printf("<Force>  force(3) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	       myid, Mc_AN, Gc_AN, sumx*GridVol, sumy*GridVol, sumz*GridVol); fflush(stdout);

	printf("<Force>  force(3) myid=%2d  GridN_Atom[Gc_AN] = %d, FNAN[Gc_AN] = %d\n",
               myid, GridN_Atom[Gc_AN], FNAN[Gc_AN]);
				
      }
    } /* Mc_AN */

#if measure_time
#pragma omp master
    printf("<Force>  force(3) myid=%2d  time1, 2 = %lf [s], %lf [s]\n", myid, time1, time2);
#endif

    /* freeing of arrays */

    free(orbs1);

    int i;
    for (i = 0; i<4; i++) {
      free(dorbs0[i]);
    }
    free(dorbs0);

    Get_dOrbitals_free(work_dObs);

  } /* #pragma omp parallel */
  /* free */
  free(dChi0[0][0]);
  free(dChi0[0]);
  free(dChi0);

  free(Vpot_grid[0]);
  free(Vpot_grid);
}






void Force4()
{
  /****************************************************
                      #4 of Force

                      n * dVNA/dx
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,Hwan,NO0,NO1;
  int i,j,k,Nc,Nh,GNc,GRc,MNc;
  int h_AN,Gh_AN,Mh_AN,Rnh,spin,Nog;
  double sum,tmp0,r,dx,dy,dz;
  double dvx,dvy,dvz;
  double sumx,sumy,sumz;
  double x,y,z,den;
  double Cxyz[4];

  /**********************************************************
              main loop for calculation of force #4
  **********************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_CNO[Cwan];

    /***********************************
                 summation 
    ***********************************/

    sumx = 0.0;
    sumy = 0.0;
    sumz = 0.0;

    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

      GNc = GridListAtom[Mc_AN][Nc]; 
      GRc = CellListAtom[Mc_AN][Nc];
      MNc = MGridListAtom[Mc_AN][Nc];

      Get_Grid_XYZ(GNc,Cxyz);
      x = Cxyz[1] + atv[GRc][1];
      y = Cxyz[2] + atv[GRc][2];
      z = Cxyz[3] + atv[GRc][3];
      dx = Gxyz[Gc_AN][1] - x;
      dy = Gxyz[Gc_AN][2] - y;
      dz = Gxyz[Gc_AN][3] - z;
      r = sqrt(dx*dx + dy*dy + dz*dz);

      /* for empty atoms or finite elemens basis */
      if (r<1.0e-10) r = 1.0e-10;

      if (1.0e-14<r){
        tmp0 = Dr_VNAF(Cwan,r);
        dvx = tmp0*dx/r;
        dvy = tmp0*dy/r;
        dvz = tmp0*dz/r;
      }
      else{
        dvx = 0.0;
        dvy = 0.0;
        dvz = 0.0;
      }

      den = Density_Grid[0][MNc] + Density_Grid[1][MNc];
      sumx += den*dvx;
      sumy += den*dvy;
      sumz += den*dvz;
    }

    Gxyz[Gc_AN][17] += sumx*GridVol;
    Gxyz[Gc_AN][18] += sumy*GridVol;
    Gxyz[Gc_AN][19] += sumz*GridVol;

    /*
    if (2<=level_stdout){
      printf("<Force>  force(4) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
              myid,Mc_AN,Gc_AN,sumx*GridVol,sumy*GridVol,sumz*GridVol);fflush(stdout);
    }
    */

  }
}





void Force_HNL(double *****CDM0, double *****iDM0)
{
  /****************************************************
                  Force arising from HNL
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,i,j,h_AN,q_AN,Mq_AN,start_q_AN;
  int jan,kl,km,kl1,Qwan,Gq_AN,Gh_AN,Mh_AN,Hwan,ian;
  int l1,l2,l3,l,LL,Mul1,tno0,ncp,so;
  int tno1,tno2,size1,size2,n,kk,num,po,po1,po2;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  int **S_array,**R_array;
  int S_comm_flag,R_comm_flag;
  int SA_num,q,Sc_AN,GSc_AN,smul;
  int Sc_wan,Sh_AN,GSh_AN,Sh_wan;
  int Sh_AN2,fan,jg,j0,jg0,Mj_AN0;
  int Original_Mc_AN;

  double rcutA,rcutB,rcut;
  double dEx,dEy,dEz,ene,pref;
  double Stime_atom, Etime_atom;
  dcomplex ***Hx,***Hy,***Hz;
  dcomplex ***Hx0,***Hy0,***Hz0;
  dcomplex ***Hx1,***Hy1,***Hz1;
  int *Snd_DS_NL_Size,*Rcv_DS_NL_Size;  
  int *Indicator;
  double *tmp_array;
  double *tmp_array2;

  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop,ODNloop;
  int *OneD2h_AN,*OneD2q_AN;
  double *dEx_threads;
  double *dEy_threads;
  double *dEz_threads;
  double stime,etime;
  double stime1,etime1;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&stime);

  /****************************
       allocation of arrays 
  *****************************/

  Indicator = (int*)malloc(sizeof(int)*numprocs);

  S_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    S_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  R_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    R_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  Snd_DS_NL_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_DS_NL_Size = (int*)malloc(sizeof(int)*numprocs);

  /* initialize the temporal array storing the force contribution */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];
    Gxyz[Gc_AN][41] = 0.0;
    Gxyz[Gc_AN][42] = 0.0;
    Gxyz[Gc_AN][43] = 0.0;
  }

  /*************************************************************
                    contraction of DS_NL
     Note: DS_NL is overwritten by CntDS_NL in Cont_Matrix1().
  *************************************************************/

  if (Cnt_switch==1){
    for (so=0; so<(SO_switch+1); so++){
      Cont_Matrix1(DS_NL[so][0],CntDS_NL[so][0]);
      Cont_Matrix1(DS_NL[so][1],CntDS_NL[so][1]);
      Cont_Matrix1(DS_NL[so][2],CntDS_NL[so][2]);
      Cont_Matrix1(DS_NL[so][3],CntDS_NL[so][3]);
    }
  }

  /*****************************************}********************** 
      THE FIRST CASE:
      In case of I=i or I=j 
      for d [ \sum_k <i|k>ek<k|j> ]/dRI  
  ****************************************************************/

  /*******************************************************
   *******************************************************
       multiplying overlap integrals WITH COMMUNICATION

       In case of I=i or I=j 
       for d [ \sum_k <i|k>ek<k|j> ]/dRI  
   *******************************************************
   *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

  Hx0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hy0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hz0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hx1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hy1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hz1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

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
	  tno2 = Spe_Total_VPS_Pro[Hwan];
	  size1 += (VPS_j_dependency[Hwan]+1)*tno1*tno2;
	}

	Snd_DS_NL_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_DS_NL_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DS_NL_Size[IDR] = size2;
      }
      else{
	Rcv_DS_NL_Size[IDR] = 0;
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

	size1 = Snd_DS_NL_Size[IDS];

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
	  tno2 = Spe_Total_VPS_Pro[Hwan];

	  for (so=0; so<=VPS_j_dependency[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp_array[num] = DS_NL[so][0][Mc_AN][h_AN][i][j];
		num++;
	      } 
	    } 
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /******************************
        receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        
	size2 = Rcv_DS_NL_Size[IDR];
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
	  tno2 = Spe_Total_VPS_Pro[Hwan];

	  for (so=0; so<=VPS_j_dependency[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		DS_NL[so][0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
		num++;
	      }
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

	  dEx = 0.0; 
	  dEy = 0.0; 
	  dEz = 0.0; 

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  fan = FNAN[Gc_AN];

	  h_AN = 0;
	  Gh_AN = natn[Gc_AN][h_AN];
	  Mh_AN = F_G2M[Gh_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  ian = Spe_Total_CNO[Hwan];

	  n = F_Rcv_Num_WK[IDR];
	  jg = Rcv_GAN[IDR][n];

	  for (j0=0; j0<=fan; j0++){

	    jg0 = natn[Gc_AN][j0];
	    Mj_AN0 = F_G2M[jg0];

	    po2 = 0;
	    if (Original_Mc_AN==Mj_AN0){
	      po2 = 1;
	      q_AN = j0;
	    }

	    if (po2==1){

	      Gq_AN = natn[Gc_AN][q_AN];
	      Mq_AN = F_G2M[Gq_AN];
	      Qwan = WhatSpecies[Gq_AN];
	      jan = Spe_Total_CNO[Qwan];
	      kl = RMI1[Mc_AN][h_AN][q_AN];

 	      dHNL(0,Mc_AN,h_AN,q_AN,DS_NL,Hx0,Hy0,Hz0);

	      /* contribution of force = Trace(CDM0*dH) */
	      /* spin non-polarization */

	      if (SpinP_switch==0){

                if (q_AN==h_AN) pref = 2.0;
                else            pref = 4.0; 

		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){

		    dEx += pref*CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r;
		    dEy += pref*CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r;
		    dEz += pref*CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r;
		  }
		}
	      }

	      /* collinear spin polarized or non-colliear without SO and LDA+U */

	      else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
		   && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

		if (q_AN==h_AN) pref = 1.0;
		else            pref = 2.0; 

		for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		  for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		    dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
			         + CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r);
		    dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
				 + CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r);
		    dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
				 + CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r);
		  }
		}
	      }

	      /* spin non-collinear with spin-orbit coupling or with LDA+U */

	      else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
                     || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

                if (q_AN==h_AN){

		  for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		    for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		      dEx +=
			  CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
			- iDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].i
			+ CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r
			- iDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hx0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hx0[2][i][j].i;

		      dEy += 
			  CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
			- iDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].i
			+ CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r
			- iDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hy0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hy0[2][i][j].i; 

		      dEz +=  
			  CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
			- iDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].i
			+ CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r
			- iDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hz0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hz0[2][i][j].i; 

		    }
		  }
		}

                else {

		  for (i=0; i<Spe_Total_CNO[Hwan]; i++){  /* Hwan */
		    for (j=0; j<Spe_Total_CNO[Qwan]; j++){ /* Qwan  */

		      dEx += 
			  CDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].r
			- iDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].r
			- iDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hx0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hx0[2][i][j].i;

		      dEy += 
			  CDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].r
			- iDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].r
			- iDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].i
	 	    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hy0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hy0[2][i][j].i; 

		      dEz +=
			  CDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].r
			- iDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].r
			- iDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].i
	 	    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hz0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hz0[2][i][j].i;

		    } /* j */
		  } /* i */

		  dHNL(0,Mc_AN,q_AN,h_AN,DS_NL,Hx1,Hy1,Hz1);
		  kl1 = RMI1[Mc_AN][q_AN][h_AN];

		  for (i=0; i<Spe_Total_CNO[Qwan]; i++){ /* Qwan */
		    for (j=0; j<Spe_Total_CNO[Hwan]; j++){ /* Hwan */

		      dEx += 
			  CDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].r
			- iDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].r
			- iDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hx1[2][i][j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hx1[2][i][j].i;

		      dEy += 
			  CDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].r
			- iDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].r
			- iDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hy1[2][i][j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hy1[2][i][j].i; 

		      dEz +=
			  CDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].r
			- iDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].r
			- iDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hz1[2][i][j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hz1[2][i][j].i;

		    } /* j */
		  } /* i */

                }
	      }

	    } /* if (po2==1) */
	  } /* j0 */             

	  /* force from #4B */

	  Gxyz[Gc_AN][41] += dEx;
	  Gxyz[Gc_AN][42] += dEy;
	  Gxyz[Gc_AN][43] += dEz;

	  /* timing */
	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

	  /********************************************
            increment of F_Rcv_Num_WK[IDR] 
	  ********************************************/

	F_Rcv_Num_WK[IDR]++;

      } /* if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) */

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

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hx0[i][j]);
    }
    free(Hx0[i]);
  }
  free(Hx0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hy0[i][j]);
    }
    free(Hy0[i]);
  }
  free(Hy0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hz0[i][j]);
    }
    free(Hz0[i]);
  }
  free(Hz0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hx1[i][j]);
    }
    free(Hx1[i]);
  }
  free(Hx1);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hy1[i][j]);
    }
    free(Hy1[i]);
  }
  free(Hy1);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hz1[i][j]);
    }
    free(Hz1[i]);
  }
  free(Hz1);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part1 of force_NL=%18.5f\n",etime-stime);fflush(stdout);
  } 

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HNL1) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }
  }

  /*******************************************************
   *******************************************************
     THE FIRST CASE:
     multiplying overlap integrals WITHOUT COMMUNICATION

     In case of I=i or I=j 
     for d [ \sum_k <i|k>ek<k|j> ]/dRI  
   *******************************************************
   *******************************************************/

  dtime(&stime);

#pragma omp parallel shared(time_per_atom,Gxyz,CDM0,SpinP_switch,SO_switch,Hub_U_switch,F_U_flag,Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch,DS_NL,RMI1,FNAN,Spe_Total_CNO,WhatSpecies,F_G2M,natn,M2G,Matomnum,List_YOUSO,F_NL_flag) private(Hx0,Hy0,Hz0,Hx1,Hy1,Hz1,OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,kl1,i,j,kk,pref)
  {

    /* allocation of array */

    Hx0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hy0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hz0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hx1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hy1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hz1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      dEx = 0.0; 
      dEy = 0.0; 
      dEz = 0.0; 

      Gc_AN = M2G[Mc_AN];
      h_AN = 0;
      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Hwan = WhatSpecies[Gh_AN];
      ian = Spe_Total_CNO[Hwan];

      for (q_AN=0; q_AN<=FNAN[Gc_AN]; q_AN++){

	Gq_AN = natn[Gc_AN][q_AN];
	Mq_AN = F_G2M[Gq_AN];

	if (Mq_AN<=Matomnum){

	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];

          dHNL(0,Mc_AN,h_AN,q_AN,DS_NL,Hx0,Hy0,Hz0);

	  if (SpinP_switch==0){

            if (q_AN==h_AN) pref = 2.0;
            else            pref = 4.0; 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){

		dEx += pref*CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r;
		dEy += pref*CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r;
		dEz += pref*CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r;
	      }
	    }
	  }

          /* collinear spin polarized or non-colliear without SO and LDA+U */

	  else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
	        && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

	    if (q_AN==h_AN) pref = 1.0;
	    else            pref = 2.0; 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){

		dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
			     + CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r);
		dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
			     + CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r);
		dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
			     + CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r);
	      }
	    }
	  }

	  /* spin non-collinear with spin-orbit coupling or with LDA+U */

	  else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
		|| 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

            if (q_AN==h_AN){

	      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		  dEx += 
                      CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hx0[2][i][j].r
		- 2.0*CDM0[3][Mh_AN][kl][i][j]*Hx0[2][i][j].i;

		  dEy += 
                      CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].i
		+ 2.0*CDM0[2][Mh_AN][kl][i][j]*Hy0[2][i][j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hy0[2][i][j].i; 

		  dEz +=
                      CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hz0[2][i][j].r
		- 2.0*CDM0[3][Mh_AN][kl][i][j]*Hz0[2][i][j].i;

		}
	      }
            }

            else{

              for (i=0; i<Spe_Total_CNO[Hwan]; i++){  /* Hwan */
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){ /* Qwan  */

		  dEx += 
                      CDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].i
		+ 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hx0[2][i][j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hx0[2][i][j].i;

		  dEy += 
                      CDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].i
		+ 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hy0[2][i][j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hy0[2][i][j].i; 

		  dEz +=
                      CDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hz0[2][i][j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hz0[2][i][j].i;

		} /* j */
	      } /* i */

              dHNL(0,Mc_AN,q_AN,h_AN,DS_NL,Hx1,Hy1,Hz1);

       	      kl1 = RMI1[Mc_AN][q_AN][h_AN];
	      for (i=0; i<Spe_Total_CNO[Qwan]; i++){ /* Qwan */
		for (j=0; j<Spe_Total_CNO[Hwan]; j++){ /* Hwan */

		  dEx += 
		      CDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].i
		+ 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hx1[2][i][j].r
		- 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hx1[2][i][j].i;

		  dEy += 
                      CDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].i
	        + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hy1[2][i][j].r
	        - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hy1[2][i][j].i; 

		  dEz +=
                      CDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].i
	        + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hz1[2][i][j].r
		- 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hz1[2][i][j].i;

		} /* j */
	      } /* i */

	    } 
	  }
	}
      }

      /* force from #4B */

      if (F_NL_flag==1){
        Gxyz[Gc_AN][41] += dEx;
        Gxyz[Gc_AN][42] += dEy;
        Gxyz[Gc_AN][43] += dEz;
      }

      /* timing */
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of array */

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hx0[i][j]);
      }
      free(Hx0[i]);
    }
    free(Hx0);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hy0[i][j]);
      }
      free(Hy0[i]);
    }
    free(Hy0);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hz0[i][j]);
      }
      free(Hz0[i]);
    }
    free(Hz0);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hx1[i][j]);
      }
      free(Hx1[i]);
    }
    free(Hx1);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hy1[i][j]);
      }
      free(Hy1[i]);
    }
    free(Hy1);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hz1[i][j]);
      }
      free(Hz1[i]);
    }
    free(Hz1);

  } /* #pragma omp parallel */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of force_NL=%18.5f\n",etime-stime);fflush(stdout);
  } 

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HNL2) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }
  }

  /*************************************************************
     THE SECOND CASE:
     In case of I=k with I!=i and I!=j
     d [ \sum_k <i|k>ek<k|j> ]/dRI  
  *************************************************************/

  /************************************************************ 
     MPI communication of DS_NL whose basis part is not located 
     on own site but projector part is located on own site. 
  ************************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

  for (ID=0; ID<numprocs; ID++) Indicator[ID] = 0;

  for (Mc_AN=1; Mc_AN<=Max_Matomnum; Mc_AN++){

    if (Mc_AN<=Matomnum)  Gc_AN = M2G[Mc_AN];
    else                  Gc_AN = 0;

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      i = Indicator[IDS]; 
      po = 0;

      Gh_AN = Pro_Snd_GAtom[IDS][i]; 

      if (Gh_AN!=0){

	/* find the range with the same global atomic number */

	do {

	  i++;
	  if (Gh_AN!=Pro_Snd_GAtom[IDS][i]) po = 1;
	} while(po==0);

	i--;
	SA_num = i - Indicator[IDS] + 1;

	/* find the data size to send the block data */

	size1 = 0;
	for (q=Indicator[IDS]; q<=(Indicator[IDS]+SA_num-1); q++){

	  Sc_AN = Pro_Snd_MAtom[IDS][q]; 
	  GSc_AN = F_M2G[Sc_AN];
	  Sc_wan = WhatSpecies[GSc_AN];
	  tno1 = Spe_Total_CNO[Sc_wan];

	  Sh_AN = Pro_Snd_LAtom[IDS][q]; 
	  GSh_AN = natn[GSc_AN][Sh_AN];
	  Sh_wan = WhatSpecies[GSh_AN];
	  tno2 = Spe_Total_VPS_Pro[Sh_wan];
          smul = (VPS_j_dependency[Sh_wan]+1);

	  size1 += smul*4*tno1*tno2;
	  size1 += 3;
	}

      } /* if (Gh_AN!=0) */

      else {
	SA_num = 0;
	size1 = 0;
      }
        
      S_array[IDS][0] = Gh_AN;
      S_array[IDS][1] = SA_num;
      S_array[IDS][2] = size1;

      if (ID!=0){
	MPI_Isend(&S_array[IDS][0], 3, MPI_INT, IDS, tag, mpi_comm_level1, &request);
	MPI_Recv( &R_array[IDR][0], 3, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	MPI_Wait(&request,&stat);
      }
      else {
	R_array[myid][0] = S_array[myid][0];
	R_array[myid][1] = S_array[myid][1];
	R_array[myid][2] = S_array[myid][2];
      }

      if (R_array[IDR][0]==Gc_AN) R_comm_flag = 1;
      else                        R_comm_flag = 0;

      if (ID!=0){
	MPI_Isend(&R_comm_flag, 1, MPI_INT, IDR, tag, mpi_comm_level1, &request);
	MPI_Recv( &S_comm_flag, 1, MPI_INT, IDS, tag, mpi_comm_level1, &stat);
	MPI_Wait(&request,&stat);
      }
      else{
	S_comm_flag = R_comm_flag;
      }

      /*****************************************
                    send the data
      *****************************************/
        
      /* if (S_comm_flag==1) then, send data to IDS */
         
      if (S_comm_flag==1){

	/* allocate tmp_array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (q=Indicator[IDS]; q<=(Indicator[IDS]+SA_num-1); q++){

	  Sc_AN = Pro_Snd_MAtom[IDS][q]; 
	  GSc_AN = F_M2G[Sc_AN];
	  Sc_wan = WhatSpecies[GSc_AN];
	  tno1 = Spe_Total_CNO[Sc_wan];

	  Sh_AN = Pro_Snd_LAtom[IDS][q]; 
	  GSh_AN = natn[GSc_AN][Sh_AN];
	  Sh_wan = WhatSpecies[GSh_AN];
	  tno2 = Spe_Total_VPS_Pro[Sh_wan];
	  Sh_AN2 = Pro_Snd_LAtom2[IDS][q]; 

	  tmp_array[num] = (double)Sc_AN;  num++;
	  tmp_array[num] = (double)Sh_AN;  num++; 
	  tmp_array[num] = (double)Sh_AN2; num++; 

	  for (so=0; so<=VPS_j_dependency[Sh_wan]; so++){
	    for (kk=0; kk<=3; kk++){
	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){
		  tmp_array[num] = DS_NL[so][kk][Sc_AN][Sh_AN][i][j];
		  num++;
		}
	      }
	    }
	  }
	}

	if (ID!=0){
	  MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
	}

	/* update Indicator[IDS] */

	Indicator[IDS] += SA_num;

      } /* if (S_comm_flag==1) */ 

      /*****************************************
                   receive the data
      *****************************************/

      /* if (R_comm_flag==1) then, receive the data from IDR */

      if (R_comm_flag==1){

	size2 = R_array[IDR][2]; 
	tmp_array2 = (double*)malloc(sizeof(double)*size2);

	if (ID!=0){
	  MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
	}
	else{
	  for (i=0; i<size2; i++) tmp_array2[i] = tmp_array[i];
	}

	/* store */

	num = 0;

	for (n=0; n<R_array[IDR][1]; n++){
            
	  Sc_AN  = (int)tmp_array2[num]; num++;
	  Sh_AN  = (int)tmp_array2[num]; num++;
	  Sh_AN2 = (int)tmp_array2[num]; num++;

	  GSc_AN = natn[Gc_AN][Sh_AN2]; 
	  Sc_wan = WhatSpecies[GSc_AN]; 
	  tno1 = Spe_Total_CNO[Sc_wan];

	  GSh_AN = natn[GSc_AN][Sh_AN];
	  Sh_wan = WhatSpecies[GSh_AN];
	  tno2 = Spe_Total_VPS_Pro[Sh_wan];

	  for (so=0; so<=VPS_j_dependency[Sh_wan]; so++){
	    for (kk=0; kk<=3; kk++){
	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){
		  DS_NL[so][kk][Matomnum+1][Sh_AN2][i][j] = tmp_array2[num];
		  num++;
		}
	      }
	    }
	  }
	}
      
	/* free tmp_array2 */
	free(tmp_array2);
 
      } /* if (R_comm_flag==1) */

      if (S_comm_flag==1){
	if (ID!=0) MPI_Wait(&request,&stat);
	free(tmp_array);  /* freeing of array */
      }
      
    } /* ID */

    if (Mc_AN<=Matomnum){ 

      /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
      {
	Nthrds0 = omp_get_num_threads();
      }

      /* allocation of arrays */
      dEx_threads = (double*)malloc(sizeof(double)*Nthrds0);
      dEy_threads = (double*)malloc(sizeof(double)*Nthrds0);
      dEz_threads = (double*)malloc(sizeof(double)*Nthrds0);

      for (Nloop=0; Nloop<Nthrds0; Nloop++){
	dEx_threads[Nloop] = 0.0;
	dEy_threads[Nloop] = 0.0;
	dEz_threads[Nloop] = 0.0;
      }

      /* one-dimensionalize the h_AN and q_AN loops */ 

      OneD2h_AN = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+1)*(FNAN[Gc_AN]+2));
      OneD2q_AN = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+1)*(FNAN[Gc_AN]+2));

      ODNloop = 0;
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1)
	 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) 
         || (Solver==5 || Solver==8 || Solver==11) )
	  start_q_AN = 0;
	else
	  start_q_AN = h_AN;

	for (q_AN=start_q_AN; q_AN<=FNAN[Gc_AN]; q_AN++){

	  kl = RMI1[Mc_AN][h_AN][q_AN];

	  if (0<=kl){
	    OneD2h_AN[ODNloop] = h_AN;
	    OneD2q_AN[ODNloop] = q_AN; 
	    ODNloop++;      
	  }
	}
      }

#pragma omp parallel shared(ODNloop,OneD2h_AN,OneD2q_AN,Mc_AN,Gc_AN,dEx_threads,dEy_threads,dEz_threads,CDM0,SpinP_switch,SO_switch,Hub_U_switch,Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch,DS_NL,RMI1,Spe_Total_CNO,WhatSpecies,F_G2M,natn,FNAN,List_YOUSO,Solver,F_NL_flag,F_U_flag) private(OMPID,Nthrds,Nprocs,Hx,Hy,Hz,i,j,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,km,Nloop,pref)
      {
          
	/* allocation of arrays */

	Hx = (dcomplex***)malloc(sizeof(dcomplex**)*3);
	for (i=0; i<3; i++){
	  Hx[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    Hx[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}

	Hy = (dcomplex***)malloc(sizeof(dcomplex**)*3);
	for (i=0; i<3; i++){
	  Hy[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    Hy[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}

	Hz = (dcomplex***)malloc(sizeof(dcomplex**)*3);
	for (i=0; i<3; i++){
	  Hz[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    Hz[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}

	/* get info. on OpenMP */ 
          
	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (Nloop=OMPID*ODNloop/Nthrds; Nloop<(OMPID+1)*ODNloop/Nthrds; Nloop++){

	  /* get h_AN and q_AN */

	  h_AN = OneD2h_AN[Nloop];
	  q_AN = OneD2q_AN[Nloop];

	  /* set informations on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Mh_AN = F_G2M[Gh_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  ian = Spe_Total_CNO[Hwan];

	  /* set informations on q_AN */

	  Gq_AN = natn[Gc_AN][q_AN];
	  Mq_AN = F_G2M[Gq_AN];
	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];
          km = RMI1[Mc_AN][q_AN][h_AN];

	  if (0<=kl){

            dHNL(1,Mc_AN,h_AN,q_AN,DS_NL,Hx,Hy,Hz);

	    /* contribution of force = Trace(CDM0*dH) */

	    /* spin non-polarization */

	    if (SpinP_switch==0){

              if (Solver==5 || Solver==8 || Solver==11){
	        pref = 2.0;
              }
              else {
	        if (q_AN==h_AN) pref = 2.0;
  	        else            pref = 4.0; 
              }

	      for (i=0; i<ian; i++){
		for (j=0; j<jan; j++){
		  dEx_threads[OMPID] += pref*CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r;
		  dEy_threads[OMPID] += pref*CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r;
		  dEz_threads[OMPID] += pref*CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r;
		}
	      }

	    }

            /* collinear spin polarized or non-colliear without SO and LDA+U */

	    else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
	          && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

              if (Solver==5 || Solver==8 || Solver==11){
	        pref = 1.0;
              }
              else {
	        if (q_AN==h_AN) pref = 1.0;
  	        else            pref = 2.0; 
              }

	      for (i=0; i<ian; i++){
		for (j=0; j<jan; j++){

		  dEx_threads[OMPID] += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r
					      + CDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].r);
		  dEy_threads[OMPID] += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r
					      + CDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].r);
		  dEz_threads[OMPID] += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r
					      + CDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].r);

		}
	      }
	    }

	    /* spin non-collinear with spin-orbit coupling or with LDA+U */

	    else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
 		   || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

	      pref = 1.0; 

	      for (i=0; i<ian; i++){
	        for (j=0; j<jan; j++){

		  dEx_threads[OMPID] += 
	        pref*(CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].i
 	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hx[2][i][j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hx[2][i][j].i);

		  dEy_threads[OMPID] += 
		pref*(CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r
	            - iDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hy[2][i][j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hy[2][i][j].i); 

		  dEz_threads[OMPID] +=
		pref*(CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hz[2][i][j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hz[2][i][j].i); 

		}
	      }
	    }

	  } /* if (0<=kl) */
	} /* Nloop */

        /* freeing of arrays */

	for (i=0; i<3; i++){
	  for (j=0; j<List_YOUSO[7]; j++){
	    free(Hx[i][j]);
	  }
	  free(Hx[i]);
	}
	free(Hx);

	for (i=0; i<3; i++){
	  for (j=0; j<List_YOUSO[7]; j++){
	    free(Hy[i][j]);
	  }
	  free(Hy[i]);
	}
	free(Hy);

	for (i=0; i<3; i++){
	  for (j=0; j<List_YOUSO[7]; j++){
	    free(Hz[i][j]);
	  }
	  free(Hz[i]);
	}
	free(Hz);

      } /* #pragma omp parallel */

      /* sum of dEx_threads */

      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;

      if (F_NL_flag==1){
        for (Nloop=0; Nloop<Nthrds0; Nloop++){
	  dEx += dEx_threads[Nloop];
	  dEy += dEy_threads[Nloop];
	  dEz += dEz_threads[Nloop];
        }

        /* force from #4B */

        Gxyz[Gc_AN][41] += dEx;
        Gxyz[Gc_AN][42] += dEy;
        Gxyz[Gc_AN][43] += dEz;
      }

      if (2<=level_stdout){
        printf("<Force>  force(HNL3) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	       myid,Mc_AN,Gc_AN,dEx,dEy,dEz);fflush(stdout);
      }

      /* freeing of array */
      free(OneD2q_AN);
      free(OneD2h_AN);
      free(dEx_threads);
      free(dEy_threads);
      free(dEz_threads);

    } /* if (Mc_AN<=Matomnum) */

  } /* Mc_AN */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part3 of force_NL=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /********************************************************
    adding Gxyz[Gc_AN][41,42,43] to Gxyz[Gc_AN][17,18,19]
  ********************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HNL) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }

    Gxyz[Gc_AN][17] += Gxyz[Gc_AN][41];
    Gxyz[Gc_AN][18] += Gxyz[Gc_AN][42];
    Gxyz[Gc_AN][19] += Gxyz[Gc_AN][43];
  }

  /***********************************
            freeing of arrays 
  ************************************/

  free(Indicator);

  for (ID=0; ID<numprocs; ID++){
    free(S_array[ID]);
  }
  free(S_array);

  for (ID=0; ID<numprocs; ID++){
    free(R_array[ID]);
  }
  free(R_array);

  free(Snd_DS_NL_Size);
  free(Rcv_DS_NL_Size);
}





void Force4B(double *****CDM0)
{
  /****************************************************
                      #4 of Force

            by the projector expansion of VNA
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,i,j,h_AN,q_AN,start_q_AN,Mq_AN;
  int jan,kl,Qwan,Gq_AN,Gh_AN,Mh_AN,Hwan,ian;
  int l1,l2,l3,l,LL,Mul1,Num_RVNA,tno0,ncp;
  int tno1,tno2,size1,size2,n,kk,num,po,po1,po2;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  int **S_array,**R_array;
  int S_comm_flag,R_comm_flag;
  int SA_num,q,Sc_AN,GSc_AN;
  int Sc_wan,Sh_AN,GSh_AN,Sh_wan;
  int Sh_AN2,fan,jg,j0,jg0,Mj_AN0;
  int Original_Mc_AN;

  double rcutA,rcutB,rcut;
  double dEx,dEy,dEz,ene,pref;
  double Stime_atom, Etime_atom;
  double **HVNAx,**HVNAy,**HVNAz;
  int *VNA_List;
  int *VNA_List2;
  int *Snd_DS_VNA_Size,*Rcv_DS_VNA_Size;  
  int *Indicator;
  Type_DS_VNA *tmp_array;
  Type_DS_VNA *tmp_array2;

  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop,ODNloop;
  int *OneD2h_AN,*OneD2q_AN;
  double *dEx_threads;
  double *dEy_threads;
  double *dEz_threads;
  double stime,etime;
  double stime1,etime1;

  MPI_Status stat;
  MPI_Request request;

  static int counter=0;

  counter++;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&stime);

  /****************************
       allocation of arrays 
  *****************************/

  Indicator = (int*)malloc(sizeof(int)*numprocs);

  S_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    S_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  R_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    R_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  Snd_DS_VNA_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_DS_VNA_Size = (int*)malloc(sizeof(int)*numprocs);

  VNA_List  = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) ); 
  VNA_List2 = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) );

  /* initialize the temporal array storing the force contribution */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];
    Gxyz[Gc_AN][41] = 0.0;
    Gxyz[Gc_AN][42] = 0.0;
    Gxyz[Gc_AN][43] = 0.0;
  }

  /*************************************************************
                 contraction of DS_VNA and HVNA2
  *************************************************************/

  if (Cnt_switch==1 && ProExpn_VNA==1){

    Cont_Matrix2(DS_VNA[0],CntDS_VNA[0]);
    Cont_Matrix2(DS_VNA[1],CntDS_VNA[1]);
    Cont_Matrix2(DS_VNA[2],CntDS_VNA[2]);
    Cont_Matrix2(DS_VNA[3],CntDS_VNA[3]);

    Cont_Matrix3(HVNA2[1],CntHVNA2[1]);
    Cont_Matrix3(HVNA2[2],CntHVNA2[2]);
    Cont_Matrix3(HVNA2[3],CntHVNA2[3]);

    Cont_Matrix4(HVNA3[1],CntHVNA3[1]);
    Cont_Matrix4(HVNA3[2],CntHVNA3[2]);
    Cont_Matrix4(HVNA3[3],CntHVNA3[3]);
  }

  /*************************************************************
                  make VNA_List and VNA_List2
  *************************************************************/

  l = 0;
  for (i=0; i<=List_YOUSO[35]; i++){     /* max L */
    for (j=0; j<List_YOUSO[34]; j++){    /* # of radial projectors */
      VNA_List[l]  = i;
      VNA_List2[l] = j;
      l++;
    }
  }

  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35] + 1);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part1 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*****************************************************
   if orbital optimization
   copy CntDS_VNA[0] into DS_VNA[0]
  *****************************************************/

  if (Cnt_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = F_M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_CNO[Cwan];  

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<tno0; i++){

	  l = 0;
	  for (l1=0; l1<Num_RVNA; l1++){

	    l2 = 2*VNA_List[l1];
	    for (l3=0; l3<=l2; l3++){
	      DS_VNA[0][Mc_AN][h_AN][i][l] = CntDS_VNA[0][Mc_AN][h_AN][i][l];
	      l++;
	    }
	  }
	}
      }
    }
  }

  /*****************************************************
     (1) pre-multiplying DS_VNA[kk] with ene
     (2) copy DS_VNA[kk] or CntDS_VNA[kk] into DS_VNA[kk]
  *****************************************************/

  dtime(&stime);

#pragma omp parallel shared(CntDS_VNA,DS_VNA,Cnt_switch,VNA_proj_ene,VNA_List2,VNA_List,Num_RVNA,natn,FNAN,Spe_Total_CNO,WhatSpecies,F_M2G,Matomnum) private(kk,OMPID,Nthrds,Nprocs,Gc_AN,Cwan,tno0,Mc_AN,h_AN,Gh_AN,Hwan,i,l,l1,LL,Mul1,ene,l2,l3)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (kk=1; kk<=3; kk++){
      for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

	Gc_AN = F_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];  

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  for (i=0; i<tno0; i++){

	    l = 0;
	    for (l1=0; l1<Num_RVNA; l1++){

	      LL   = VNA_List[l1];
	      Mul1 = VNA_List2[l1];

	      ene = VNA_proj_ene[Hwan][LL][Mul1];
	      l2 = 2*VNA_List[l1];

	      if (Cnt_switch==0){
		for (l3=0; l3<=l2; l3++){
		  DS_VNA[kk][Mc_AN][h_AN][i][l] = ene*DS_VNA[kk][Mc_AN][h_AN][i][l];
		  l++;
		}
	      }
 
	      else{
		for (l3=0; l3<=l2; l3++){
		  DS_VNA[kk][Mc_AN][h_AN][i][l] = ene*CntDS_VNA[kk][Mc_AN][h_AN][i][l];
		  l++;
		}
	      }
	    }
	  } 

	} /* h_AN */
      } /* Mc_AN */
    } /* kk */

  } /* #pragma omp parallel */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*****************************************}********************** 
      THE FIRST CASE:
      In case of I=i or I=j 
      for d [ \sum_k <i|k>ek<k|j> ]/dRI  
  ****************************************************************/

  /*******************************************************
   *******************************************************
      multiplying overlap integrals WITH COMMUNICATION
   *******************************************************
   *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

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
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	  size1 += tno1*tno2; 
	}
 
	Snd_DS_VNA_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_DS_VNA_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DS_VNA_Size[IDR] = size2;
      }
      else{
	Rcv_DS_VNA_Size[IDR] = 0;
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

	size1 = Snd_DS_VNA_Size[IDS];

	/* allocation of the array */

	tmp_array = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size1);

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
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      tmp_array[num] = DS_VNA[0][Mc_AN][h_AN][i][j];
	      num++;
	    } 
	  } 
	}

	MPI_Isend(&tmp_array[0], size1, MPI_Type_DS_VNA, IDS, tag, mpi_comm_level1, &request);
      }

      /******************************
          receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        
	size2 = Rcv_DS_VNA_Size[IDR];
	tmp_array2 = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_Type_DS_VNA, IDR, tag, mpi_comm_level1, &stat);

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
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      DS_VNA[0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
	      num++;
	    }
	  }
	}

	/* free tmp_array2 */
	free(tmp_array2);

	/*****************************************
               multiplying overlap integrals
	*****************************************/

#pragma omp parallel shared(List_YOUSO,time_per_atom,Gxyz,CDM0,SpinP_switch,CntHVNA2,HVNA2,DS_VNA,Cnt_switch,RMI1,Original_Mc_AN,IDR,Rcv_GAN,F_Rcv_Num_WK,Spe_Total_CNO,F_G2M,natn,FNAN,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,Mc_AN,Cwan,fan,h_AN,Gh_AN,Mh_AN,Hwan,ian,n,jg,j0,jg0,Mj_AN0,po2,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,HVNAx,HVNAy,HVNAz,i,j)
	{

	  /* allocation of array */

	  HVNAx = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    HVNAx[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	  }

	  HVNAy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    HVNAy[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	  }

	  HVNAz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    HVNAz[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	  }

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

	    dtime(&Stime_atom);

	    dEx = 0.0; 
	    dEy = 0.0; 
	    dEz = 0.0; 

	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    fan = FNAN[Gc_AN];

	    h_AN = 0;
	    Gh_AN = natn[Gc_AN][h_AN];
	    Mh_AN = F_G2M[Gh_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    ian = Spe_Total_CNO[Hwan];

	    n = F_Rcv_Num_WK[IDR];
	    jg = Rcv_GAN[IDR][n];

	    for (j0=0; j0<=fan; j0++){

	      jg0 = natn[Gc_AN][j0];
	      Mj_AN0 = F_G2M[jg0];

	      po2 = 0;
	      if (Original_Mc_AN==Mj_AN0){
		po2 = 1;
		q_AN = j0;
	      }

	      if (po2==1){

		Gq_AN = natn[Gc_AN][q_AN];
		Mq_AN = F_G2M[Gq_AN];
		Qwan = WhatSpecies[Gq_AN];
		jan = Spe_Total_CNO[Qwan];
		kl = RMI1[Mc_AN][h_AN][q_AN];

		if (Cnt_switch==0) {
		  dHVNA(0,Mc_AN,h_AN,q_AN,DS_VNA,HVNA2,HVNA3,HVNAx,HVNAy,HVNAz);
		}
		else { 
		  dHVNA(0,Mc_AN,h_AN,q_AN,DS_VNA,CntHVNA2,CntHVNA3,HVNAx,HVNAy,HVNAz);
		}

		/* contribution of force = Trace(CDM0*dH) */
		/* spin non-polarization */

		if (SpinP_switch==0){

		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      if (q_AN==h_AN){

			dEx += 2.0*CDM0[0][Mh_AN][kl][i][j]*HVNAx[i][j];
			dEy += 2.0*CDM0[0][Mh_AN][kl][i][j]*HVNAy[i][j];
			dEz += 2.0*CDM0[0][Mh_AN][kl][i][j]*HVNAz[i][j];
		      }
		      else{
			dEx += 4.0*CDM0[0][Mh_AN][kl][i][j]*HVNAx[i][j];
			dEy += 4.0*CDM0[0][Mh_AN][kl][i][j]*HVNAy[i][j];
			dEz += 4.0*CDM0[0][Mh_AN][kl][i][j]*HVNAz[i][j];
		      }
		    }
		  }
		}

		/* else */

		else{

		  for (i=0; i<ian; i++){
		    for (j=0; j<jan; j++){
		      if (q_AN==h_AN){
			dEx += (  CDM0[0][Mh_AN][kl][i][j]
			        + CDM0[1][Mh_AN][kl][i][j] )*HVNAx[i][j];
			dEy += (  CDM0[0][Mh_AN][kl][i][j]
			        + CDM0[1][Mh_AN][kl][i][j] )*HVNAy[i][j];
			dEz += (  CDM0[0][Mh_AN][kl][i][j]
				+ CDM0[1][Mh_AN][kl][i][j] )*HVNAz[i][j];
		      }
		      else{
			dEx += 2.0*(  CDM0[0][Mh_AN][kl][i][j]
				    + CDM0[1][Mh_AN][kl][i][j] )*HVNAx[i][j];
			dEy += 2.0*(  CDM0[0][Mh_AN][kl][i][j]
				    + CDM0[1][Mh_AN][kl][i][j] )*HVNAy[i][j];
			dEz += 2.0*(  CDM0[0][Mh_AN][kl][i][j]
				    + CDM0[1][Mh_AN][kl][i][j] )*HVNAz[i][j];
		      } 
		    }
		  }
		}

	      } /* if (po2==1) */
	    } /* j0 */             

	      /* force from #4B */

	    Gxyz[Gc_AN][41] += dEx;
	    Gxyz[Gc_AN][42] += dEy;
	    Gxyz[Gc_AN][43] += dEz;

	    /* timing */
	    dtime(&Etime_atom);
	    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	  } /* Mc_AN */

	    /* freeing of array */

	  for (j=0; j<List_YOUSO[7]; j++){
	    free(HVNAx[j]);
	  }
	  free(HVNAx);

	  for (j=0; j<List_YOUSO[7]; j++){
	    free(HVNAy[j]);
	  }
	  free(HVNAy);

	  for (j=0; j<List_YOUSO[7]; j++){
	    free(HVNAz[j]);
	  }
	  free(HVNAz);

	} /* #pragma omp parallel */

	  /********************************************
            increment of F_Rcv_Num_WK[IDR] 
	  ********************************************/

	F_Rcv_Num_WK[IDR]++;

      } /* if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) */

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

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part3 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*******************************************************
   *******************************************************
      THE FIRST CASE:
      multiplying overlap integrals WITHOUT COMMUNICATION
   *******************************************************
   *******************************************************/

  dtime(&stime);

#pragma omp parallel shared(time_per_atom,Gxyz,CDM0,SpinP_switch,CntHVNA2,HVNA2,DS_VNA,Cnt_switch,RMI1,FNAN,Spe_Total_CNO,WhatSpecies,F_G2M,natn,M2G,Matomnum,List_YOUSO) private(HVNAx,HVNAy,HVNAz,OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,i,j,kk)
  {

    /* allocation of array */

    HVNAx = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      HVNAx[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }

    HVNAy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      HVNAy[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }

    HVNAz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      HVNAz[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      dEx = 0.0; 
      dEy = 0.0; 
      dEz = 0.0; 

      Gc_AN = M2G[Mc_AN];
      h_AN = 0;
      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Hwan = WhatSpecies[Gh_AN];
      ian = Spe_Total_CNO[Hwan];

      for (q_AN=h_AN; q_AN<=FNAN[Gc_AN]; q_AN++){

	Gq_AN = natn[Gc_AN][q_AN];
	Mq_AN = F_G2M[Gq_AN];

	if (Mq_AN<=Matomnum){

	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];

	  if (Cnt_switch==0) {
	    dHVNA(0,Mc_AN,h_AN,q_AN,DS_VNA,HVNA2,HVNA3,HVNAx,HVNAy,HVNAz);
	  }
	  else { 
	    dHVNA(0,Mc_AN,h_AN,q_AN,DS_VNA,CntHVNA2,CntHVNA3,HVNAx,HVNAy,HVNAz);
	  }

	  if (SpinP_switch==0){

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){
		if (q_AN==h_AN){
		  dEx += 2.0*CDM0[0][Mh_AN][kl][i][j]*HVNAx[i][j];
		  dEy += 2.0*CDM0[0][Mh_AN][kl][i][j]*HVNAy[i][j];
		  dEz += 2.0*CDM0[0][Mh_AN][kl][i][j]*HVNAz[i][j];
		}
		else{
		  dEx += 4.0*CDM0[0][Mh_AN][kl][i][j]*HVNAx[i][j];
		  dEy += 4.0*CDM0[0][Mh_AN][kl][i][j]*HVNAy[i][j];
		  dEz += 4.0*CDM0[0][Mh_AN][kl][i][j]*HVNAz[i][j];
		}

	      }
	    }
	  }

	  /* else */

	  else{

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){
		if (q_AN==h_AN){
		  dEx += (  CDM0[0][Mh_AN][kl][i][j]
			  + CDM0[1][Mh_AN][kl][i][j] )*HVNAx[i][j];
		  dEy += (  CDM0[0][Mh_AN][kl][i][j]
			  + CDM0[1][Mh_AN][kl][i][j] )*HVNAy[i][j];
		  dEz += (  CDM0[0][Mh_AN][kl][i][j]
			  + CDM0[1][Mh_AN][kl][i][j] )*HVNAz[i][j];
		}
		else{
		  dEx += 2.0*(  CDM0[0][Mh_AN][kl][i][j]
			      + CDM0[1][Mh_AN][kl][i][j] )*HVNAx[i][j];
		  dEy += 2.0*(  CDM0[0][Mh_AN][kl][i][j]
			      + CDM0[1][Mh_AN][kl][i][j] )*HVNAy[i][j];
		  dEz += 2.0*(  CDM0[0][Mh_AN][kl][i][j]
			      + CDM0[1][Mh_AN][kl][i][j] )*HVNAz[i][j];
		}
	      }
	    }
	  }
	}
      }

      /* force from #4B */

      Gxyz[Gc_AN][41] += dEx;
      Gxyz[Gc_AN][42] += dEy;
      Gxyz[Gc_AN][43] += dEz;

      /* timing */
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

      /* freeing of array */

    for (j=0; j<List_YOUSO[7]; j++){
      free(HVNAx[j]);
    }
    free(HVNAx);

    for (j=0; j<List_YOUSO[7]; j++){
      free(HVNAy[j]);
    }
    free(HVNAy);

    for (j=0; j<List_YOUSO[7]; j++){
      free(HVNAz[j]);
    }
    free(HVNAz);

  } /* #pragma omp parallel */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part4 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*************************************************************
     THE SECOND CASE:
     In case of I=k with I!=i and I!=j
     d [ \sum_k <i|k>ek<k|j> ]/dRI  
  *************************************************************/

  /************************************************************ 
     MPI communication of DS_VNA whose basis part is not located 
     on own site but projector part is located on own site. 
  ************************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

  for (ID=0; ID<numprocs; ID++) Indicator[ID] = 0;

  for (Mc_AN=1; Mc_AN<=Max_Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    dtime(&stime1);

    if (Mc_AN<=Matomnum)  Gc_AN = M2G[Mc_AN];
    else                  Gc_AN = 0;

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      i = Indicator[IDS]; 
      po = 0;

      Gh_AN = Pro_Snd_GAtom[IDS][i]; 

      if (Gh_AN!=0){

	/* find the range with the same global atomic number */

	do {

	  i++;
	  if (Gh_AN!=Pro_Snd_GAtom[IDS][i]) po = 1;
	} while(po==0);

	i--;
	SA_num = i - Indicator[IDS] + 1;

	/* find the data size to send the block data */

	size1 = 0;
	for (q=Indicator[IDS]; q<=(Indicator[IDS]+SA_num-1); q++){

	  Sc_AN = Pro_Snd_MAtom[IDS][q]; 
	  GSc_AN = F_M2G[Sc_AN];
	  Sc_wan = WhatSpecies[GSc_AN];
	  tno1 = Spe_Total_CNO[Sc_wan];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	  size1 += 4*tno1*tno2;
	  size1 += 3;
	}

      } /* if (Gh_AN!=0) */

      else {
	SA_num = 0;
	size1 = 0;
      }
        
      S_array[IDS][0] = Gh_AN;
      S_array[IDS][1] = SA_num;
      S_array[IDS][2] = size1;

      if (ID!=0){
	MPI_Isend(&S_array[IDS][0], 3, MPI_INT, IDS, tag, mpi_comm_level1, &request);
	MPI_Recv( &R_array[IDR][0], 3, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	MPI_Wait(&request,&stat);
      }
      else {
	R_array[myid][0] = S_array[myid][0];
	R_array[myid][1] = S_array[myid][1];
	R_array[myid][2] = S_array[myid][2];
      }

      if (R_array[IDR][0]==Gc_AN) R_comm_flag = 1;
      else                        R_comm_flag = 0;

      if (ID!=0){
	MPI_Isend(&R_comm_flag, 1, MPI_INT, IDR, tag, mpi_comm_level1, &request);
	MPI_Recv( &S_comm_flag, 1, MPI_INT, IDS, tag, mpi_comm_level1, &stat);
	MPI_Wait(&request,&stat);
      }
      else{
	S_comm_flag = R_comm_flag;
      }

      /*
  if (counter==2 && ID==8){
    printf("QQQ4 myid=%2d\n",myid);fflush(stdout);
    MPI_Finalize();
    exit(0);
  }
      */

      /*****************************************
                       send the data
      *****************************************/
        
      /* if (S_comm_flag==1) then, send data to IDS */
         
      if (S_comm_flag==1){

	/* allocate tmp_array */

	tmp_array = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (q=Indicator[IDS]; q<=(Indicator[IDS]+SA_num-1); q++){

	  Sc_AN = Pro_Snd_MAtom[IDS][q]; 
	  GSc_AN = F_M2G[Sc_AN];
	  Sc_wan = WhatSpecies[GSc_AN];
	  tno1 = Spe_Total_CNO[Sc_wan];

	  Sh_AN = Pro_Snd_LAtom[IDS][q]; 
	  GSh_AN = natn[GSc_AN][Sh_AN];
	  Sh_wan = WhatSpecies[GSh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  Sh_AN2 = Pro_Snd_LAtom2[IDS][q]; 

	  tmp_array[num] = (Type_DS_VNA)Sc_AN;  num++;
	  tmp_array[num] = (Type_DS_VNA)Sh_AN;  num++; 
	  tmp_array[num] = (Type_DS_VNA)Sh_AN2; num++; 

	  for (kk=0; kk<=3; kk++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp_array[num] = DS_VNA[kk][Sc_AN][Sh_AN][i][j];
		num++;
	      }
	    }
	  }
	}

	if (ID!=0){
	  MPI_Isend(&tmp_array[0], size1, MPI_Type_DS_VNA, IDS, tag, mpi_comm_level1, &request);
	}

	/* update Indicator[IDS] */

	Indicator[IDS] += SA_num;

      } /* if (S_comm_flag==1) */ 

      /*****************************************
                   receive the data
      *****************************************/

      /* if (R_comm_flag==1) then, receive the data from IDR */

      if (R_comm_flag==1){

	size2 = R_array[IDR][2]; 
	tmp_array2 = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size2);

	if (ID!=0){
	  MPI_Recv(&tmp_array2[0], size2, MPI_Type_DS_VNA, IDR, tag, mpi_comm_level1, &stat);
	}
	else{
	  for (i=0; i<size2; i++) tmp_array2[i] = tmp_array[i];
	}

	/* store */

	num = 0;

	for (n=0; n<R_array[IDR][1]; n++){
            
	  Sc_AN  = (int)tmp_array2[num]; num++;
	  Sh_AN  = (int)tmp_array2[num]; num++;
	  Sh_AN2 = (int)tmp_array2[num]; num++;

	  GSc_AN = natn[Gc_AN][Sh_AN2]; 
	  Sc_wan = WhatSpecies[GSc_AN]; 

	  tno1 = Spe_Total_CNO[Sc_wan];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  for (kk=0; kk<=3; kk++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		DS_VNA[kk][Matomnum+1][Sh_AN2][i][j] = tmp_array2[num];
		num++;
	      }
	    }
	  }
	}         

	/* free tmp_array2 */
	free(tmp_array2);
 
      } /* if (R_comm_flag==1) */

      if (S_comm_flag==1){
	if (ID!=0) MPI_Wait(&request,&stat);
	free(tmp_array);  /* freeing of array */
      }

    } /* ID */

    dtime(&etime1);
    if(myid==0 && measure_time){
      printf("Time for part5A of force#4=%18.5f\n",etime1-stime1);fflush(stdout);
    } 

    dtime(&stime1);

    if (Mc_AN<=Matomnum){ 

      /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
      {
	Nthrds0 = omp_get_num_threads();
      }

      /* allocation of arrays */
      dEx_threads = (double*)malloc(sizeof(double)*Nthrds0);
      dEy_threads = (double*)malloc(sizeof(double)*Nthrds0);
      dEz_threads = (double*)malloc(sizeof(double)*Nthrds0);

      for (Nloop=0; Nloop<Nthrds0; Nloop++){
	dEx_threads[Nloop] = 0.0;
	dEy_threads[Nloop] = 0.0;
	dEz_threads[Nloop] = 0.0;
      }

      /* one-dimensionalize the h_AN and q_AN loops */ 

      OneD2h_AN = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+1)*(FNAN[Gc_AN]+2));
      OneD2q_AN = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+1)*(FNAN[Gc_AN]+2));

      ODNloop = 0;
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if ( Solver==5 || Solver==8 || Solver==11)
	  start_q_AN = 0;
	else
	  start_q_AN = h_AN;

	for (q_AN=start_q_AN; q_AN<=FNAN[Gc_AN]; q_AN++){

	  kl = RMI1[Mc_AN][h_AN][q_AN];

	  if (0<=kl){
	    OneD2h_AN[ODNloop] = h_AN;
	    OneD2q_AN[ODNloop] = q_AN; 
	    ODNloop++;      
	  }
	}
      }

#pragma omp parallel shared(ODNloop,OneD2h_AN,OneD2q_AN,Mc_AN,Gc_AN,dEx_threads,dEy_threads,dEz_threads,CDM0,SpinP_switch,CntHVNA2,HVNA2,DS_VNA,Cnt_switch,RMI1,Spe_Total_CNO,WhatSpecies,F_G2M,natn,FNAN,List_YOUSO,Solver) private(OMPID,Nthrds,Nprocs,HVNAx,HVNAy,HVNAz,i,j,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,Nloop,pref)
      {
          
	/* allocation of arrays */
           
	HVNAx = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	for (j=0; j<List_YOUSO[7]; j++){
	  HVNAx[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	}

	HVNAy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	for (j=0; j<List_YOUSO[7]; j++){
	  HVNAy[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	}

	HVNAz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
	for (j=0; j<List_YOUSO[7]; j++){
	  HVNAz[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	}

	/* get info. on OpenMP */ 
          
	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (Nloop=OMPID*ODNloop/Nthrds; Nloop<(OMPID+1)*ODNloop/Nthrds; Nloop++){

	  /* get h_AN and q_AN */

	  h_AN = OneD2h_AN[Nloop];
	  q_AN = OneD2q_AN[Nloop];

	  /* set informations on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Mh_AN = F_G2M[Gh_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  ian = Spe_Total_CNO[Hwan];

	  /* set informations on q_AN */

	  Gq_AN = natn[Gc_AN][q_AN];
	  Mq_AN = F_G2M[Gq_AN];
	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];

	  if (0<=kl){

	    if (Cnt_switch==0)
	      dHVNA(1,Mc_AN,h_AN,q_AN,DS_VNA,HVNA2,HVNA3,HVNAx,HVNAy,HVNAz);
	    else 
	      dHVNA(1,Mc_AN,h_AN,q_AN,DS_VNA,CntHVNA2,CntHVNA3,HVNAx,HVNAy,HVNAz);

	    /* contribution of force = Trace(CDM0*dH) */

	    /* spin non-polarization */

	    if (SpinP_switch==0){

              if (Solver==5 || Solver==8 || Solver==11){
	        pref = 2.0;
              }
              else {
	        if (q_AN==h_AN) pref = 2.0;
  	        else            pref = 4.0; 
              }

	      for (i=0; i<ian; i++){
		for (j=0; j<jan; j++){
		  dEx_threads[OMPID] += pref*CDM0[0][Mh_AN][kl][i][j]*HVNAx[i][j];
		  dEy_threads[OMPID] += pref*CDM0[0][Mh_AN][kl][i][j]*HVNAy[i][j];
		  dEz_threads[OMPID] += pref*CDM0[0][Mh_AN][kl][i][j]*HVNAz[i][j];
		}
	      }
	    }

	    /* else */

	    else{

              if (Solver==5 || Solver==8 || Solver==11){
	        pref = 1.0;
              }
              else {
	        if (q_AN==h_AN) pref = 1.0;
  	        else            pref = 2.0; 
              }

	      for (i=0; i<ian; i++){
		for (j=0; j<jan; j++){
		  dEx_threads[OMPID] += pref*(   CDM0[0][Mh_AN][kl][i][j]
			 	               + CDM0[1][Mh_AN][kl][i][j] )*HVNAx[i][j];
		  dEy_threads[OMPID] += pref*(   CDM0[0][Mh_AN][kl][i][j]
					       + CDM0[1][Mh_AN][kl][i][j] )*HVNAy[i][j];
		  dEz_threads[OMPID] += pref*(   CDM0[0][Mh_AN][kl][i][j]
					       + CDM0[1][Mh_AN][kl][i][j] )*HVNAz[i][j];
		}
	      }
	    }

	  } /* if (0<=kl) */

	} /* Nloop */

	  /* freeing of arrays */

	for (j=0; j<List_YOUSO[7]; j++){
	  free(HVNAx[j]);
	}
	free(HVNAx);

	for (j=0; j<List_YOUSO[7]; j++){
	  free(HVNAy[j]);
	}
	free(HVNAy);

	for (j=0; j<List_YOUSO[7]; j++){
	  free(HVNAz[j]);
	}
	free(HVNAz);

      } /* #pragma omp parallel */

	/* sum of dEx_threads */

      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;

      for (Nloop=0; Nloop<Nthrds0; Nloop++){
	dEx += dEx_threads[Nloop];
	dEy += dEy_threads[Nloop];
	dEz += dEz_threads[Nloop];
      }

      /* force from #4B */

      Gxyz[Gc_AN][41] += dEx;
      Gxyz[Gc_AN][42] += dEy;
      Gxyz[Gc_AN][43] += dEz;

      /* timing */
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      /* freeing of array */
      free(OneD2q_AN);
      free(OneD2h_AN);
      free(dEx_threads);
      free(dEy_threads);
      free(dEz_threads);

    } /* if (Mc_AN<=Matomnum) */

    dtime(&etime1);
    if(myid==0 && measure_time){
      printf("Time for part5B of force#4=%18.5f\n",etime1-stime1);fflush(stdout);
    } 

  } /* Mc_AN */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part5 of force#4=%18.5f\n",etime-stime);fflush(stdout);
  } 

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(4B) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }

    Gxyz[Gc_AN][17] += Gxyz[Gc_AN][41];
    Gxyz[Gc_AN][18] += Gxyz[Gc_AN][42];
    Gxyz[Gc_AN][19] += Gxyz[Gc_AN][43];
  }

  /***********************************
            freeing of arrays 
  ************************************/

  free(Indicator);

  for (ID=0; ID<numprocs; ID++){
    free(S_array[ID]);
  }
  free(S_array);

  for (ID=0; ID<numprocs; ID++){
    free(R_array[ID]);
  }
  free(R_array);

  free(Snd_DS_VNA_Size);
  free(Rcv_DS_VNA_Size);

  free(VNA_List);
  free(VNA_List2);

}




void dHNL(int where_flag,
          int Mc_AN, int h_AN, int q_AN,
          double ******DS_NL1,
          dcomplex ***Hx, dcomplex ***Hy, dcomplex ***Hz)
{
  int i,j,k,m,n,l,kg,kan,so,deri_kind;
  int ig,ian,jg,jan,kl,kl1,kl2;
  int wakg,l1,l2,l3,Gc_AN,Mi_AN,Mi_AN2,Mj_AN,Mj_AN2;
  int Rni,Rnj,somax;
  double PF[2],sumx,sumy,sumz,ene,dmp,deri_dmp;
  double tmpx,tmpy,tmpz,tmp,r;
  double x0,y0,z0,x1,y1,z1,dx,dy,dz;
  double rcuti,rcutj,rcut;
  double PFp,PFm,ene_p,ene_m;
  dcomplex sumx0,sumy0,sumz0;
  dcomplex sumx1,sumy1,sumz1;
  dcomplex sumx2,sumy2,sumz2;

  /****************************************************
   start calc.
  ****************************************************/

  Gc_AN = M2G[Mc_AN];
  ig = natn[Gc_AN][h_AN];
  Rni = ncn[Gc_AN][h_AN];
  Mi_AN = F_G2M[ig];
  ian = Spe_Total_CNO[WhatSpecies[ig]];
  rcuti = Spe_Atom_Cut1[WhatSpecies[ig]];

  jg = natn[Gc_AN][q_AN];
  Rnj = ncn[Gc_AN][q_AN];
  Mj_AN = F_G2M[jg];
  jan = Spe_Total_CNO[WhatSpecies[jg]];
  rcutj = Spe_Atom_Cut1[WhatSpecies[jg]];

  rcut = rcuti + rcutj;
  kl = RMI1[Mc_AN][h_AN][q_AN];
  dmp = dampingF(rcut,Dis[ig][kl]);

  for (so=0; so<3; so++){
    for (i=0; i<List_YOUSO[7]; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	Hx[so][i][j] = Complex(0.0,0.0);
	Hy[so][i][j] = Complex(0.0,0.0);
	Hz[so][i][j] = Complex(0.0,0.0);
      }
    }
  }

  if (h_AN==0){

    /****************************************************
                          dH*ep*H
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl = RMI1[Mc_AN][q_AN][k];

      /****************************************************
                   l-dependent non-local part
      ****************************************************/

      if (0<=kl && VPS_j_dependency[wakg]==0 && where_flag==0){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene = Spe_VNLE[0][wakg][l1-1];
	      if      (Spe_VPS_List[wakg][l1]==0) l2 = 0;
	      else if (Spe_VPS_List[wakg][l1]==1) l2 = 2;
	      else if (Spe_VPS_List[wakg][l1]==2) l2 = 4;
	      else if (Spe_VPS_List[wakg][l1]==3) l2 = 6;

              if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
              else                 Mj_AN2 = Matomnum + 1; 

	      for (l3=0; l3<=l2; l3++){
		sumx += ene*DS_NL1[0][1][Mc_AN][k][m][l]*DS_NL1[0][0][Mj_AN2][kl][n][l];
		sumy += ene*DS_NL1[0][2][Mc_AN][k][m][l]*DS_NL1[0][0][Mj_AN2][kl][n][l];
		sumz += ene*DS_NL1[0][3][Mc_AN][k][m][l]*DS_NL1[0][0][Mj_AN2][kl][n][l];
		l++;
	      }
	    }

	    Hx[0][m][n].r += sumx;
	    Hy[0][m][n].r += sumy;
	    Hz[0][m][n].r += sumz;

	    Hx[1][m][n].r += sumx;
	    Hy[1][m][n].r += sumy;
	    Hz[1][m][n].r += sumz;

	  } /* n */
	} /* m */

      } /* if */

      /****************************************************
                   j-dependent non-local part
      ****************************************************/

      else if ( 0<=kl && VPS_j_dependency[wakg]==1 && where_flag==0 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

            if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
            else                 Mj_AN2 = Matomnum + 1; 

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene_p = Spe_VNLE[0][wakg][l1-1];
	      ene_m = Spe_VNLE[1][wakg][l1-1];

	      if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	      else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
                      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
                      &sumx0.i,&sumy0.i,&sumz0.i,
                      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      1.0,
                      PFp, PFm,
                      ene_p,ene_m,
                      l2, &l,
                      Mc_AN ,k, m,
                      Mj_AN2,kl,n,
                      DS_NL1);
	    }

            if (q_AN==0){

	      l = 0;
	      for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

		ene_p = Spe_VNLE[0][wakg][l1-1];
		ene_m = Spe_VNLE[1][wakg][l1-1];

		if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
		else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
		else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
		else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

		dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
			&sumx1.r,&sumy1.r,&sumz1.r,
			&sumx2.r,&sumy2.r,&sumz2.r,
			&sumx0.i,&sumy0.i,&sumz0.i,
			&sumx1.i,&sumy1.i,&sumz1.i,
                        &sumx2.i,&sumy2.i,&sumz2.i,
			-1.0,
			PFp, PFm,
			ene_p,ene_m,
			l2, &l,
			Mj_AN2, kl, n,
			Mc_AN,  k,  m,
			DS_NL1);

	      }
	    }

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    } /* k */

    /****************************************************
                           H*ep*dH 
    ****************************************************/

    /* h_AN==0 && q_AN==0 */

    if (q_AN==0 && VPS_j_dependency[wakg]==0){

      for (m=0; m<ian; m++){
        for (n=m; n<jan; n++){

          tmpx = Hx[0][m][n].r + Hx[0][n][m].r;
          Hx[0][m][n].r = tmpx; 
          Hx[0][n][m].r = tmpx;
          Hx[1][m][n].r = tmpx; 
          Hx[1][n][m].r = tmpx;

          tmpy = Hy[0][m][n].r + Hy[0][n][m].r;
          Hy[0][m][n].r = tmpy; 
          Hy[0][n][m].r = tmpy;
          Hy[1][m][n].r = tmpy; 
          Hy[1][n][m].r = tmpy;

          tmpz = Hz[0][m][n].r + Hz[0][n][m].r;
          Hz[0][m][n].r = tmpz; 
          Hz[0][n][m].r = tmpz;
          Hz[1][m][n].r = tmpz; 
          Hz[1][n][m].r = tmpz;
        }
      }
    }

    else if (where_flag==1){

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl = RMI1[Mc_AN][q_AN][0];

      /****************************************************
                   l-dependent non-local part
      ****************************************************/

      if (VPS_j_dependency[wakg]==0){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

            if (Mj_AN<=Matomnum){
              Mj_AN2 = Mj_AN;
 	      kl2 = RMI1[Mc_AN][q_AN][0];
	    }
            else{
              Mj_AN2 = Matomnum + 1; 
	      kl2 = RMI1[Mc_AN][0][q_AN];
	    }

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene = Spe_VNLE[0][wakg][l1-1];
	      if      (Spe_VPS_List[wakg][l1]==0) l2 = 0;
	      else if (Spe_VPS_List[wakg][l1]==1) l2 = 2;
	      else if (Spe_VPS_List[wakg][l1]==2) l2 = 4;
	      else if (Spe_VPS_List[wakg][l1]==3) l2 = 6;

	      for (l3=0; l3<=l2; l3++){

		sumx -= ene*DS_NL1[0][0][Mc_AN][0][m][l]*DS_NL1[0][1][Mj_AN2][kl2][n][l];
		sumy -= ene*DS_NL1[0][0][Mc_AN][0][m][l]*DS_NL1[0][2][Mj_AN2][kl2][n][l];
		sumz -= ene*DS_NL1[0][0][Mc_AN][0][m][l]*DS_NL1[0][3][Mj_AN2][kl2][n][l];
		l++;
	      }
	    }

	    Hx[0][m][n].r += sumx;
	    Hy[0][m][n].r += sumy;
	    Hz[0][m][n].r += sumz;

	    Hx[1][m][n].r += sumx;
	    Hy[1][m][n].r += sumy;
	    Hz[1][m][n].r += sumz;
	  }
	}
      }

      /****************************************************
                   j-dependent non-local part
      ****************************************************/

      else if ( VPS_j_dependency[wakg]==1 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

            if (Mj_AN<=Matomnum){
              Mj_AN2 = Mj_AN;
 	      kl2 = RMI1[Mc_AN][q_AN][0];
	    }
            else{
              Mj_AN2 = Matomnum + 1; 
	      kl2 = RMI1[Mc_AN][0][q_AN];
	    }

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene_p = Spe_VNLE[0][wakg][l1-1];
	      ene_m = Spe_VNLE[1][wakg][l1-1];

	      if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	      else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

       	      /* 1 */

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
                      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
                      &sumx0.i,&sumy0.i,&sumz0.i,
                      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      -1.0,
                      PFp, PFm,
                      -ene_p,-ene_m,  
                      l2, &l,
                      Mj_AN2,kl2,n,
                      Mc_AN, 0,  m,
                      DS_NL1);
	    }

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    }

  } /* if (h_AN==0) */


  else if (where_flag==0){

    /****************************************************
       H*ep*dH

       if (h_AN!=0 && where_flag==0)
       This happens 
       only if 
       ( SpinP_switch==3
         &&
         (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
  	   || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1
           || Zeeman_NCO_switch==1)
         && 
         q_AN==0
       )
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl = RMI1[Mc_AN][h_AN][k];

      if (Mi_AN<=Matomnum) Mi_AN2 = Mi_AN;
      else                 Mi_AN2 = Matomnum + 1; 

      if (0<=kl && VPS_j_dependency[wakg]==1){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene_p = Spe_VNLE[0][wakg][l1-1];
	      ene_m = Spe_VNLE[1][wakg][l1-1];

	      if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	      else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
                      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
                      &sumx0.i,&sumy0.i,&sumz0.i,
                      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      -1.0,
                      PFp, PFm,
                      ene_p, ene_m,
                      l2, &l,
                      Mj_AN,  k,  n,
                      Mi_AN2, kl, m,
                      DS_NL1);
	    }

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    }     

  }

  /* if (h_AN!=0 && where_flag==1) */

  else {

    /****************************************************
                           dH*ep*H
    ****************************************************/

    kg = natn[Gc_AN][0];
    wakg = WhatSpecies[kg];
    kan = Spe_Total_VPS_Pro[wakg];
    kl1 = RMI1[Mc_AN][0][h_AN];
    kl2 = RMI1[Mc_AN][0][q_AN];

    /****************************************************
                   l-dependent non-local part
    ****************************************************/

    if (VPS_j_dependency[wakg]==0){

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx = 0.0;
	  sumy = 0.0;
	  sumz = 0.0;

	  l = 0;
	  for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	    ene = Spe_VNLE[0][wakg][l1-1];
	    if      (Spe_VPS_List[wakg][l1]==0) l2 = 0;
	    else if (Spe_VPS_List[wakg][l1]==1) l2 = 2;
	    else if (Spe_VPS_List[wakg][l1]==2) l2 = 4;
	    else if (Spe_VPS_List[wakg][l1]==3) l2 = 6;

	    for (l3=0; l3<=l2; l3++){
	      sumx -= ene*DS_NL1[0][1][Matomnum+1][kl1][m][l]*DS_NL1[0][0][Matomnum+1][kl2][n][l];
	      sumy -= ene*DS_NL1[0][2][Matomnum+1][kl1][m][l]*DS_NL1[0][0][Matomnum+1][kl2][n][l];
	      sumz -= ene*DS_NL1[0][3][Matomnum+1][kl1][m][l]*DS_NL1[0][0][Matomnum+1][kl2][n][l];
	      l++;
	    }
	  }

	  Hx[0][m][n].r = sumx;
	  Hy[0][m][n].r = sumy;
	  Hz[0][m][n].r = sumz;

	  Hx[1][m][n].r = sumx;
	  Hy[1][m][n].r = sumy;
	  Hz[1][m][n].r = sumz;

	  Hx[2][m][n].r = 0.0;         
	  Hy[2][m][n].r = 0.0;         
	  Hz[2][m][n].r = 0.0;         

	  Hx[0][m][n].i = 0.0;
	  Hy[0][m][n].i = 0.0;
	  Hz[0][m][n].i = 0.0;

	  Hx[1][m][n].i = 0.0;
	  Hy[1][m][n].i = 0.0;
	  Hz[1][m][n].i = 0.0;

	  Hx[2][m][n].i = 0.0;
	  Hy[2][m][n].i = 0.0;
	  Hz[2][m][n].i = 0.0;

	}
      }
    }

    /****************************************************
                 j-dependent non-local part
    ****************************************************/

    else if ( VPS_j_dependency[wakg]==1 ){

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx0 = Complex(0.0,0.0);
	  sumy0 = Complex(0.0,0.0);
	  sumz0 = Complex(0.0,0.0);

	  sumx1 = Complex(0.0,0.0);
	  sumy1 = Complex(0.0,0.0);
	  sumz1 = Complex(0.0,0.0);

          sumx2 = Complex(0.0,0.0);
	  sumy2 = Complex(0.0,0.0);
	  sumz2 = Complex(0.0,0.0);

	  l = 0;
	  for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	    ene_p = Spe_VNLE[0][wakg][l1-1];
	    ene_m = Spe_VNLE[1][wakg][l1-1];

	    if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	    else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	    else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	    else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

   	    /* 2 */

	    dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
		    &sumx1.r,&sumy1.r,&sumz1.r,
                    &sumx2.r,&sumy2.r,&sumz2.r,
		    &sumx0.i,&sumy0.i,&sumz0.i,
		    &sumx1.i,&sumy1.i,&sumz1.i,
                    &sumx2.i,&sumy2.i,&sumz2.i,
                    1.0, 
		    PFp, PFm,
		    -ene_p,-ene_m,
		    l2, &l,
		    Matomnum+1, kl1,m,
		    Matomnum+1, kl2,n,
		    DS_NL1);
	  }

	  Hx[0][m][n].r = sumx0.r;     /* up-up */
	  Hy[0][m][n].r = sumy0.r;     /* up-up */
	  Hz[0][m][n].r = sumz0.r;     /* up-up */

	  Hx[1][m][n].r = sumx1.r;     /* dn-dn */
	  Hy[1][m][n].r = sumy1.r;     /* dn-dn */
	  Hz[1][m][n].r = sumz1.r;     /* dn-dn */

	  Hx[2][m][n].r = sumx2.r;     /* up-dn */
	  Hy[2][m][n].r = sumy2.r;     /* up-dn */
	  Hz[2][m][n].r = sumz2.r;     /* up-dn */

	  Hx[0][m][n].i = sumx0.i;     /* up-up */
	  Hy[0][m][n].i = sumy0.i;     /* up-up */
	  Hz[0][m][n].i = sumz0.i;     /* up-up */

	  Hx[1][m][n].i = sumx1.i;     /* dn-dn */
	  Hy[1][m][n].i = sumy1.i;     /* dn-dn */
	  Hz[1][m][n].i = sumz1.i;     /* dn-dn */

	  Hx[2][m][n].i = sumx2.i;     /* up-dn */
	  Hy[2][m][n].i = sumy2.i;     /* up-dn */
	  Hz[2][m][n].i = sumz2.i;     /* up-dn */

	}
      }
    }

    /****************************************************
                           H*ep*dH
    ****************************************************/

    if (q_AN!=0) {

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl1 = RMI1[Mc_AN][0][h_AN];
      kl2 = RMI1[Mc_AN][0][q_AN];

      /****************************************************
                     l-dependent non-local part
      ****************************************************/

      if (VPS_j_dependency[wakg]==0){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

	      ene = Spe_VNLE[0][wakg][l1-1];
	      if      (Spe_VPS_List[wakg][l1]==0) l2 = 0;
	      else if (Spe_VPS_List[wakg][l1]==1) l2 = 2;
	      else if (Spe_VPS_List[wakg][l1]==2) l2 = 4;
	      else if (Spe_VPS_List[wakg][l1]==3) l2 = 6;

	      for (l3=0; l3<=l2; l3++){
		sumx -= ene*DS_NL1[0][0][Matomnum+1][kl1][m][l]*DS_NL1[0][1][Matomnum+1][kl2][n][l];
		sumy -= ene*DS_NL1[0][0][Matomnum+1][kl1][m][l]*DS_NL1[0][2][Matomnum+1][kl2][n][l];
		sumz -= ene*DS_NL1[0][0][Matomnum+1][kl1][m][l]*DS_NL1[0][3][Matomnum+1][kl2][n][l];
		l++;
	      }
	    }

	    Hx[0][m][n].r += sumx;
	    Hy[0][m][n].r += sumy;
	    Hz[0][m][n].r += sumz;

	    Hx[1][m][n].r += sumx;
	    Hy[1][m][n].r += sumy;
	    Hz[1][m][n].r += sumz;
	  }
	}
      }

      /****************************************************
                    j-dependent non-local part
      ****************************************************/

      else if ( VPS_j_dependency[wakg]==1 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);
	    sumy0 = Complex(0.0,0.0);
	    sumz0 = Complex(0.0,0.0);

	    sumx1 = Complex(0.0,0.0);
	    sumy1 = Complex(0.0,0.0);
	    sumz1 = Complex(0.0,0.0);

	    sumx2 = Complex(0.0,0.0);
	    sumy2 = Complex(0.0,0.0);
	    sumz2 = Complex(0.0,0.0);

	    l = 0;
	    for (l1=1; l1<=Spe_Num_RVPS[wakg]; l1++){

 	      ene_p = Spe_VNLE[0][wakg][l1-1];
	      ene_m = Spe_VNLE[1][wakg][l1-1];

	      if      (Spe_VPS_List[wakg][l1]==0) { l2=0; PFp=1.0;     PFm=0.0;     }  
	      else if (Spe_VPS_List[wakg][l1]==1) { l2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	      else if (Spe_VPS_List[wakg][l1]==2) { l2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	      else if (Spe_VPS_List[wakg][l1]==3) { l2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	      /* 4 */

	      dHNL_SO(&sumx0.r,&sumy0.r,&sumz0.r,
		      &sumx1.r,&sumy1.r,&sumz1.r,
                      &sumx2.r,&sumy2.r,&sumz2.r,
		      &sumx0.i,&sumy0.i,&sumz0.i,
		      &sumx1.i,&sumy1.i,&sumz1.i,
                      &sumx2.i,&sumy2.i,&sumz2.i,
                      -1.0,
		      PFp, PFm,
		      -ene_p,-ene_m,
		      l2, &l,
		      Matomnum+1, kl2,n,
		      Matomnum+1, kl1,m,
		      DS_NL1);
	    }

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    }

  } /* else */

  /****************************************************
               contribution by dampingF 
  ****************************************************/

  /* Qij * dH/dx  */

  for (so=0; so<3; so++){
    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){

        Hx[so][m][n].r = dmp*Hx[so][m][n].r;
        Hy[so][m][n].r = dmp*Hy[so][m][n].r;
        Hz[so][m][n].r = dmp*Hz[so][m][n].r;

        Hx[so][m][n].i = dmp*Hx[so][m][n].i;
        Hy[so][m][n].i = dmp*Hy[so][m][n].i;
        Hz[so][m][n].i = dmp*Hz[so][m][n].i;
      }
    }
  }

  /* dQij/dx * H */

  if ( (h_AN==0 && q_AN!=0) || (h_AN!=0 && q_AN==0) ){

    if      (h_AN==0)   kl = q_AN;
    else if (q_AN==0)   kl = h_AN;

    if      (SpinP_switch==0) somax = 0;
    else if (SpinP_switch==1) somax = 1;
    else if (SpinP_switch==3) somax = 2;

    r = Dis[Gc_AN][kl];

    if (rcut<=r) {
      deri_dmp = 0.0;
      tmp = 0.0; 
    }
    else {
      deri_dmp = deri_dampingF(rcut,r);
      tmp = deri_dmp/dmp;
    }

    x0 = Gxyz[ig][1] + atv[Rni][1];
    y0 = Gxyz[ig][2] + atv[Rni][2];
    z0 = Gxyz[ig][3] + atv[Rni][3];

    x1 = Gxyz[jg][1] + atv[Rnj][1];
    y1 = Gxyz[jg][2] + atv[Rnj][2];
    z1 = Gxyz[jg][3] + atv[Rnj][3];

    /* for empty atoms or finite elemens basis */
    if (r<1.0e-10) r = 1.0e-10;

    if (h_AN==0 && q_AN!=0){
      dx = tmp*(x0-x1)/r;
      dy = tmp*(y0-y1)/r;
      dz = tmp*(z0-z1)/r;
    }

    else if (h_AN!=0 && q_AN==0){
      dx = tmp*(x1-x0)/r;
      dy = tmp*(y1-y0)/r;
      dz = tmp*(z1-z0)/r;
    }

    if (SpinP_switch==0 || SpinP_switch==1){

      if (h_AN==0){ 
        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r += HNL[so][Mc_AN][kl][m][n]*dx;
	      Hy[so][m][n].r += HNL[so][Mc_AN][kl][m][n]*dy;
	      Hz[so][m][n].r += HNL[so][Mc_AN][kl][m][n]*dz;
	    }
	  }
        }
      }

      else if (q_AN==0){ 
        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r += HNL[so][Mc_AN][kl][n][m]*dx;
	      Hy[so][m][n].r += HNL[so][Mc_AN][kl][n][m]*dy;
	      Hz[so][m][n].r += HNL[so][Mc_AN][kl][n][m]*dz;
	    }
	  }
        }
      }
    }

    else if (SpinP_switch==3){

      if (h_AN==0){ 
        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r +=  HNL[so][Mc_AN][kl][m][n]*dx;
	      Hy[so][m][n].r +=  HNL[so][Mc_AN][kl][m][n]*dy;
	      Hz[so][m][n].r +=  HNL[so][Mc_AN][kl][m][n]*dz;
	    }
	  }
        }
      }

      else if (q_AN==0){ 
        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r +=  HNL[so][Mc_AN][kl][n][m]*dx;
	      Hy[so][m][n].r +=  HNL[so][Mc_AN][kl][n][m]*dy;
	      Hz[so][m][n].r +=  HNL[so][Mc_AN][kl][n][m]*dz;
	    }
	  }
        }
      }

      if (SO_switch==1){

        if (h_AN==0){ 
	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		Hx[so][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dx;
		Hy[so][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dy;
		Hz[so][m][n].i += iHNL[so][Mc_AN][kl][m][n]*dz;
	      }
	    }
	  }
	}

        else if (q_AN==0){ 
	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		Hx[so][m][n].i += iHNL[so][Mc_AN][kl][n][m]*dx;
		Hy[so][m][n].i += iHNL[so][Mc_AN][kl][n][m]*dy;
		Hz[so][m][n].i += iHNL[so][Mc_AN][kl][n][m]*dz;
	      }
	    }
	  }
	}

      }
    }
  }

}





void dHVNA(int where_flag, int Mc_AN, int h_AN, int q_AN,
           Type_DS_VNA *****DS_VNA1, 
           double *****TmpHVNA2, double *****TmpHVNA3, 
           double **Hx, double **Hy, double **Hz)
{
  int i,j,k,m,n,l,kg,kan,so,deri_kind;
  int ig,ian,jg,jan,kl,kl1,kl2,Rni,Rnj;
  int wakg,l1,l2,l3,Gc_AN,Mi_AN,Mj_AN,Mj_AN2,num_projectors;
  double sumx,sumy,sumz,ene,rcuti,rcutj,rcut;
  double tmpx,tmpy,tmpz,dmp,deri_dmp,tmp;
  double dx,dy,dz,x0,y0,z0,x1,y1,z1,r;
  double PFp,PFm,ene_p,ene_m;
  double sumx0,sumy0,sumz0;
  double sumx1,sumy1,sumz1;
  double sumx2,sumy2,sumz2;
  int L,LL,Mul1,Num_RVNA;

  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35] + 1);
  num_projectors = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

  /****************************************************
   start calc.
  ****************************************************/

  Gc_AN = M2G[Mc_AN];
  ig = natn[Gc_AN][h_AN];
  Rni = ncn[Gc_AN][h_AN];
  Mi_AN = F_G2M[ig];
  ian = Spe_Total_CNO[WhatSpecies[ig]];
  rcuti = Spe_Atom_Cut1[WhatSpecies[ig]];

  jg = natn[Gc_AN][q_AN];
  Rnj = ncn[Gc_AN][q_AN];
  Mj_AN = F_G2M[jg];
  jan = Spe_Total_CNO[WhatSpecies[jg]];
  rcutj = Spe_Atom_Cut1[WhatSpecies[jg]];

  rcut = rcuti + rcutj;
  kl = RMI1[Mc_AN][h_AN][q_AN];
  dmp = dampingF(rcut,Dis[ig][kl]);

  for (m=0; m<ian; m++){
    for (n=0; n<jan; n++){
      Hx[m][n] = 0.0;
      Hy[m][n] = 0.0;
      Hz[m][n] = 0.0;
    }
  }

  /****************************************************
    two-center integral with orbitals on one-center 
    
    in case of h_AN==0 && q_AN==0
  ****************************************************/

  if (h_AN==0 && q_AN==0 && where_flag==0){

    for (k=1; k<=FNAN[Gc_AN]; k++){
      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
	  Hx[m][n] += TmpHVNA2[1][Mc_AN][k][m][n];
	  Hy[m][n] += TmpHVNA2[2][Mc_AN][k][m][n];
	  Hz[m][n] += TmpHVNA2[3][Mc_AN][k][m][n];
        }
      }      
    }
  }

  /****************************************************
    two-center integral with orbitals on one-center 
    
    in case of h_AN==q_AN && h_AN!=0 
  ****************************************************/

  else if (h_AN==q_AN && h_AN!=0){

    kl = RMI1[Mc_AN][h_AN][0];

    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){

	Hx[m][n] = -TmpHVNA3[1][Mc_AN][h_AN][m][n];
	Hy[m][n] = -TmpHVNA3[2][Mc_AN][h_AN][m][n];
	Hz[m][n] = -TmpHVNA3[3][Mc_AN][h_AN][m][n];
      }
    } 
  }

  /****************************************************
             two and three center integrals
             with orbitals on two-center
  ****************************************************/

  else{

    if (h_AN==0){

      /****************************************************
			   dH*ep*H
      ****************************************************/

      for (k=0; k<=FNAN[Gc_AN]; k++){

	kg = natn[Gc_AN][k];
	wakg = WhatSpecies[kg];
	kl = RMI1[Mc_AN][q_AN][k];

	/****************************************************
			     non-local part
	****************************************************/

	if (0<=kl && where_flag==0){

          if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
          else                 Mj_AN2 = Matomnum+1; 

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      sumx = 0.0;
	      sumy = 0.0;
	      sumz = 0.0;

	      for (l=0; l<num_projectors; l++){
		sumx += DS_VNA1[1][Mc_AN][k][m][l]*DS_VNA1[0][Mj_AN2][kl][n][l];
		sumy += DS_VNA1[2][Mc_AN][k][m][l]*DS_VNA1[0][Mj_AN2][kl][n][l];
		sumz += DS_VNA1[3][Mc_AN][k][m][l]*DS_VNA1[0][Mj_AN2][kl][n][l];
	      }

	      Hx[m][n] += sumx;
	      Hy[m][n] += sumy;
	      Hz[m][n] += sumz;

	    } /* n */
	  } /* m */

	} /* if */

      } /* k */

      /****************************************************
 		  	     H*ep*dH 
      ****************************************************/

      /* non-local part */

      if (q_AN==0){

	for (m=0; m<ian; m++){
	  for (n=m; n<jan; n++){
            
            tmpx = Hx[m][n] + Hx[n][m];
	    Hx[m][n] = tmpx;
	    Hx[n][m] = tmpx;

            tmpy = Hy[m][n] + Hy[n][m];
	    Hy[m][n] = tmpy;
	    Hy[n][m] = tmpy;

            tmpz = Hz[m][n] + Hz[n][m];
	    Hz[m][n] = tmpz;
	    Hz[n][m] = tmpz;
	  }
	}

      }

      else if (where_flag==1) {

	kg = natn[Gc_AN][0];
	wakg = WhatSpecies[kg];

	/****************************************************
			     non-local part
	****************************************************/

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

            if (Mj_AN<=Matomnum){
              Mj_AN2 = Mj_AN;
 	      kl = RMI1[Mc_AN][q_AN][0];
	    }
            else{
              Mj_AN2 = Matomnum+1; 
	      kl = RMI1[Mc_AN][0][q_AN];
            }

	    for (l=0; l<num_projectors; l++){
	      sumx -= DS_VNA1[0][Mc_AN][0][m][l]*DS_VNA1[1][Mj_AN2][kl][n][l];
	      sumy -= DS_VNA1[0][Mc_AN][0][m][l]*DS_VNA1[2][Mj_AN2][kl][n][l];
	      sumz -= DS_VNA1[0][Mc_AN][0][m][l]*DS_VNA1[3][Mj_AN2][kl][n][l];
	    }

	    Hx[m][n] += sumx;
	    Hy[m][n] += sumy;
	    Hz[m][n] += sumz;

	  }
	}

      }

    } /* if (h_AN==0) */

    else {

      /****************************************************
			   dH*ep*H
      ****************************************************/

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kl1 = RMI1[Mc_AN][0][h_AN];
      kl2 = RMI1[Mc_AN][0][q_AN];

      /****************************************************
 		         non-local part
      ****************************************************/

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx = 0.0;
	  sumy = 0.0;
	  sumz = 0.0;

          for (l=0; l<num_projectors; l++){
	    sumx -= DS_VNA1[1][Matomnum+1][kl1][m][l]*DS_VNA1[0][Matomnum+1][kl2][n][l];
	    sumy -= DS_VNA1[2][Matomnum+1][kl1][m][l]*DS_VNA1[0][Matomnum+1][kl2][n][l];
	    sumz -= DS_VNA1[3][Matomnum+1][kl1][m][l]*DS_VNA1[0][Matomnum+1][kl2][n][l];
	  }

	  Hx[m][n] = sumx;
	  Hy[m][n] = sumy;
	  Hz[m][n] = sumz;
	}
      }

      /****************************************************
			   H*ep*dH
      ****************************************************/

      if (q_AN!=0){

	kg = natn[Gc_AN][0];
	wakg = WhatSpecies[kg];
	kl1 = RMI1[Mc_AN][0][h_AN];
	kl2 = RMI1[Mc_AN][0][q_AN];

	/****************************************************
			    non-local part
	****************************************************/

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

	    for (l=0; l<num_projectors; l++){
	      sumx -= DS_VNA1[0][Matomnum+1][kl1][m][l]*DS_VNA1[1][Matomnum+1][kl2][n][l];
	      sumy -= DS_VNA1[0][Matomnum+1][kl1][m][l]*DS_VNA1[2][Matomnum+1][kl2][n][l];
	      sumz -= DS_VNA1[0][Matomnum+1][kl1][m][l]*DS_VNA1[3][Matomnum+1][kl2][n][l];
	    }

	    Hx[m][n] += sumx;
	    Hy[m][n] += sumy;
	    Hz[m][n] += sumz;

	  }
	}
      }

    }
  }

  /****************************************************
                contribution by dampingF 
  ****************************************************/

  /* Qij * dH/dx  */

  for (m=0; m<ian; m++){
    for (n=0; n<jan; n++){
      Hx[m][n] = dmp*Hx[m][n];
      Hy[m][n] = dmp*Hy[m][n];
      Hz[m][n] = dmp*Hz[m][n];
    }
  }

  /* dQij/dx * H */

  if ( (h_AN==0 && q_AN!=0) || (h_AN!=0 && q_AN==0) ){

    if      (h_AN==0)   kl = q_AN;
    else if (q_AN==0)   kl = h_AN;

    r = Dis[Gc_AN][kl];
 
    if (rcut<=r) {
      deri_dmp = 0.0;
      tmp = 0.0; 
    }
    else {
      deri_dmp = deri_dampingF(rcut,r);
      tmp = deri_dmp/dmp;
    }

    x0 = Gxyz[ig][1] + atv[Rni][1];
    x1 = Gxyz[jg][1] + atv[Rnj][1];

    y0 = Gxyz[ig][2] + atv[Rni][2];
    y1 = Gxyz[jg][2] + atv[Rnj][2];

    z0 = Gxyz[ig][3] + atv[Rni][3];
    z1 = Gxyz[jg][3] + atv[Rnj][3];

    /* for empty atoms or finite elemens basis */
    if (r<1.0e-10) r = 1.0e-10;

    if ( h_AN==0 ){
      dx = tmp*(x0-x1)/r;
      dy = tmp*(y0-y1)/r;
      dz = tmp*(z0-z1)/r;
    }

    else if ( q_AN==0 ){
      dx = tmp*(x1-x0)/r;
      dy = tmp*(y1-y0)/r;
      dz = tmp*(z1-z0)/r;
    }

    if (h_AN==0){ 
      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
	  Hx[m][n] += HVNA[Mc_AN][kl][m][n]*dx;
	  Hy[m][n] += HVNA[Mc_AN][kl][m][n]*dy;
	  Hz[m][n] += HVNA[Mc_AN][kl][m][n]*dz;
        }
      }
    }

    else if (q_AN==0){ 
      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
	  Hx[m][n] += HVNA[Mc_AN][kl][n][m]*dx;
	  Hy[m][n] += HVNA[Mc_AN][kl][n][m]*dy;
	  Hz[m][n] += HVNA[Mc_AN][kl][n][m]*dz;
        }
      }
    }
  }

}











void dHNL_SO(
	     double *sumx0r,
	     double *sumy0r, 
	     double *sumz0r, 
	     double *sumx1r,
	     double *sumy1r, 
	     double *sumz1r, 
	     double *sumx2r,
	     double *sumy2r, 
	     double *sumz2r, 
	     double *sumx0i,
	     double *sumy0i, 
	     double *sumz0i, 
	     double *sumx1i,
	     double *sumy1i, 
	     double *sumz1i, 
	     double *sumx2i,
	     double *sumy2i, 
	     double *sumz2i, 
             double fugou,
	     double PFp,
	     double PFm,
	     double ene_p,
	     double ene_m,
	     int l2, int *l,
	     int Mc_AN, int k,  int m,
	     int Mj_AN, int kl, int n,
	     double ******DS_NL1)
{

  int l3,i;
  double tmpx,tmpy,tmpz;
  double tmp0,tmp1,tmp2;
  double tmp3,tmp4,tmp5,tmp6;
  double deri[4];

  /****************************************************
    off-diagonal contribution to up-dn matrix
    for spin non-collinear
  ****************************************************/

  if (SpinP_switch==3){

    /* p */ 
    if (l2==2){

      /* real contribution of l+1/2 to off diagonal up-down matrix */ 
      tmpx = 
        fugou*
        ( ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
         -ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] );      

      tmpy = 
        fugou*
        ( ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
         -ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] );      

      tmpz = 
        fugou*
        ( ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
         -ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] );      

      *sumx2r += tmpx;
      *sumy2r += tmpy;
      *sumz2r += tmpz;

      /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

      tmpx = 
        fugou*
        ( -ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
          +ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1] );

      tmpy = 
        fugou*
        ( -ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
          +ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1] );

      tmpz = 
        fugou*
        ( -ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
          +ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1] );

      *sumx2i += tmpx;
      *sumy2i += tmpy;
      *sumz2i += tmpz;

       /* real contribution of l-1/2 for to diagonal up-down matrix */ 

      tmpx = 
        fugou*
        ( ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
         -ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] );

      tmpy = 
        fugou*
        ( ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
         -ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] );

      tmpz = 
        fugou*
        ( ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
         -ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] );

      *sumx2r -= tmpx;
      *sumy2r -= tmpy;
      *sumz2r -= tmpz;

       /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

      tmpx = 
        fugou*
        ( -ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
          +ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1] );   

      tmpy = 
        fugou*
        ( -ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
          +ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1] );   

      tmpz = 
        fugou*
        ( -ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
          +ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1] );   

      *sumx2i -= tmpx;
      *sumy2i -= tmpy;
      *sumz2i -= tmpz;

    }

    /* d */ 
    if (l2==4){

       tmp0 = sqrt(3.0);
       tmp1 = ene_p/5.0; 
       tmp2 = tmp0*tmp1;

       /* real contribution of l+1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
            ( -tmp2*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
              +tmp2*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2r += deri[1];
       *sumy2r += deri[2];
       *sumz2r += deri[3];

       /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
            ( +tmp2*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
              -tmp2*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
              -tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
              +tmp1*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2i += deri[1];
       *sumy2i += deri[2];
       *sumz2i += deri[3];

       /* real contribution of l-1/2 for to diagonal up-down matrix */ 

       tmp1 = ene_m/5.0; 
       tmp2 = tmp0*tmp1;

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
            ( -tmp2*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
              +tmp2*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
              +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2r -= deri[1];
       *sumy2r -= deri[2];
       *sumz2r -= deri[3];

       /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
	    ( +tmp2*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
              -tmp2*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
              +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
              -tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
	      +tmp1*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+2] );
       }
       *sumx2i -= deri[1];
       *sumy2i -= deri[2];
       *sumz2i -= deri[3];
    }

    /* f */ 
    if (l2==6){

      /* real contribution of l+1/2 to off diagonal up-down matrix */ 

      tmp0 = sqrt(6.0);
      tmp1 = sqrt(3.0/2.0);
      tmp2 = sqrt(5.0/2.0);

      tmp3 = ene_p/7.0; 
      tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
      tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
      tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

       for (i=1; i<=3; i++){
         deri[i] = 
	   fugou*
             ( -tmp6*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
               +tmp6*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
               +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
	       +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
               -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+5]
	       +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
               -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
               +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+4] ); 
       }
       *sumx2r += deri[1];
       *sumy2r += deri[2];
       *sumz2r += deri[3];

       /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
	   fugou*
	     ( +tmp6*DS_NL1[0][i][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
               -tmp6*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l  ]
               +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
               -tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
	       +tmp5*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
               +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
	       -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
               -tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+5]
               +tmp4*DS_NL1[0][i][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+4] );
       }
       *sumx2i += deri[1];
       *sumy2i += deri[2];
       *sumz2i += deri[3];

       /* real contribution of l-1/2 for to diagonal up-down matrix */ 

       tmp3 = ene_m/7.0; 
       tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
       tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
       tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

       for (i=1; i<=3; i++){
         deri[i] = 
	   fugou*
	     ( -tmp6*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
               +tmp6*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
               -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
               +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
               -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
	       +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
               -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+5]
	       +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
               -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
               +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+4] );
       }
       *sumx2r -= deri[1];
       *sumy2r -= deri[2];
       *sumz2r -= deri[3];

       /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

       for (i=1; i<=3; i++){
         deri[i] = 
            fugou* 
	      ( +tmp6*DS_NL1[1][i][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
                -tmp6*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l  ]
                +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
                -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
                -tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
	        +tmp5*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
                +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
	        -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
                -tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+5]
                +tmp4*DS_NL1[1][i][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+4] );
       }
       *sumx2i -= deri[1];
       *sumy2i -= deri[2];
       *sumz2i -= deri[3];
	   
    }

  }

  /****************************************************
      off-diagonal contribution on up-up and dn-dn
  ****************************************************/

  /* p */ 
  if (l2==2){
 
    tmpx =
       fugou*
       ( ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
        -ene_p/3.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] ); 

    tmpy =
       fugou*
       ( ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
        -ene_p/3.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] ); 

    tmpz =
       fugou*
       ( ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l  ]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
        -ene_p/3.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l  ] ); 

    /* contribution of l+1/2 for up spin */ 
    *sumx0i += -tmpx;
    *sumy0i += -tmpy;
    *sumz0i += -tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += tmpx;
    *sumy1i += tmpy;
    *sumz1i += tmpz;

    tmpx =
       fugou*
       ( ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
        -ene_m/3.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] ); 

    tmpy =
       fugou*
       ( ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
        -ene_m/3.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] ); 

    tmpz =
       fugou*
       ( ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l  ]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
        -ene_m/3.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l  ] ); 

    /* contribution of l-1/2 for up spin */
    *sumx0i += tmpx;
    *sumy0i += tmpy;
    *sumz0i += tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += -tmpx;
    *sumy1i += -tmpy;
    *sumz1i += -tmpz;
  }

  /* d */ 
  else if (l2==4){

    tmpx =
       fugou*
       (
       ene_p*2.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*2.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*1.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*1.0/5.0*DS_NL1[0][1][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3] ); 

    tmpy =
       fugou*
       (
       ene_p*2.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*2.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*1.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*1.0/5.0*DS_NL1[0][2][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3] ); 

    tmpz =
       fugou*
       (
       ene_p*2.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*2.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*1.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*1.0/5.0*DS_NL1[0][3][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3] ); 

    /* contribution of l+1/2 for up spin */ 
    *sumx0i += -tmpx;
    *sumy0i += -tmpy;
    *sumz0i += -tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += tmpx;
    *sumy1i += tmpy;
    *sumz1i += tmpz;

    tmpx =
       fugou*
       (
       ene_m*2.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*2.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*1.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*1.0/5.0*DS_NL1[1][1][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3] ); 

    tmpy = 
       fugou*
       (
       ene_m*2.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*2.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*1.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*1.0/5.0*DS_NL1[1][2][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3] ); 

    tmpz =
       fugou*
       (
       ene_m*2.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*2.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*1.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*1.0/5.0*DS_NL1[1][3][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3] ); 

    /* contribution of l-1/2 for up spin */ 
    *sumx0i += tmpx;
    *sumy0i += tmpy;
    *sumz0i += tmpz;

    /* contribution of l-1/2 for down spin */ 
    *sumx1i += -tmpx;
    *sumy1i += -tmpy;
    *sumz1i += -tmpz;

  }
  
  /* f */ 
  else if (l2==6){

    tmpx =
       fugou*
       (
       ene_p*1.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*1.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*2.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*2.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
      +ene_p*3.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
      -ene_p*3.0/7.0*DS_NL1[0][1][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+5] );

    tmpy = 
       fugou*
       (
       ene_p*1.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*1.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*2.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*2.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
      +ene_p*3.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
      -ene_p*3.0/7.0*DS_NL1[0][2][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+5] );

    tmpz =
       fugou*
       (
       ene_p*1.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+1]*DS_NL1[0][0][Mj_AN][kl][n][*l+2]
      -ene_p*1.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+2]*DS_NL1[0][0][Mj_AN][kl][n][*l+1]
      +ene_p*2.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+3]*DS_NL1[0][0][Mj_AN][kl][n][*l+4]
      -ene_p*2.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+4]*DS_NL1[0][0][Mj_AN][kl][n][*l+3]
      +ene_p*3.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+5]*DS_NL1[0][0][Mj_AN][kl][n][*l+6]
      -ene_p*3.0/7.0*DS_NL1[0][3][Mc_AN][k][m][*l+6]*DS_NL1[0][0][Mj_AN][kl][n][*l+5] );

    /* contribution of l+1/2 for up spin */ 
    *sumx0i += -tmpx;
    *sumy0i += -tmpy;
    *sumz0i += -tmpz;

    /* contribution of l+1/2 for down spin */ 
    *sumx1i += tmpx;
    *sumy1i += tmpy;
    *sumz1i += tmpz;

    tmpx =
       fugou*
       (
       ene_m*1.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*1.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*2.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*2.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
      +ene_m*3.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
      -ene_m*3.0/7.0*DS_NL1[1][1][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+5] );

    tmpy =
       fugou*
       (
       ene_m*1.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*1.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*2.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*2.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
      +ene_m*3.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
      -ene_m*3.0/7.0*DS_NL1[1][2][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+5] );

    tmpz =
       fugou*
       (
       ene_m*1.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+1]*DS_NL1[1][0][Mj_AN][kl][n][*l+2]
      -ene_m*1.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+2]*DS_NL1[1][0][Mj_AN][kl][n][*l+1]
      +ene_m*2.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+3]*DS_NL1[1][0][Mj_AN][kl][n][*l+4]
      -ene_m*2.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+4]*DS_NL1[1][0][Mj_AN][kl][n][*l+3]
      +ene_m*3.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+5]*DS_NL1[1][0][Mj_AN][kl][n][*l+6]
      -ene_m*3.0/7.0*DS_NL1[1][3][Mc_AN][k][m][*l+6]*DS_NL1[1][0][Mj_AN][kl][n][*l+5] );

    /* contribution of l-1/2 for up spin */ 
    *sumx0i += tmpx;
    *sumy0i += tmpy;
    *sumz0i += tmpz;

    /* contribution of l-1/2 for down spin */ 
    *sumx1i += -tmpx;
    *sumy1i += -tmpy;
    *sumz1i += -tmpz;
  }

  /****************************************************
      diagonal contribution on up-up and dn-dn
  ****************************************************/

  for (l3=0; l3<=l2; l3++){

    /* VNL for j=l+1/2 */
              
    tmpx = PFp*ene_p*DS_NL1[0][1][Mc_AN][k][m][*l]*DS_NL1[0][0][Mj_AN][kl][n][*l];
    tmpy = PFp*ene_p*DS_NL1[0][2][Mc_AN][k][m][*l]*DS_NL1[0][0][Mj_AN][kl][n][*l];
    tmpz = PFp*ene_p*DS_NL1[0][3][Mc_AN][k][m][*l]*DS_NL1[0][0][Mj_AN][kl][n][*l];

    *sumx0r += tmpx;
    *sumy0r += tmpy;
    *sumz0r += tmpz;

    *sumx1r += tmpx;
    *sumy1r += tmpy;
    *sumz1r += tmpz;

    /* VNL for j=l-1/2 */
              
    tmpx = PFm*ene_m*DS_NL1[1][1][Mc_AN][k][m][*l]*DS_NL1[1][0][Mj_AN][kl][n][*l];
    tmpy = PFm*ene_m*DS_NL1[1][2][Mc_AN][k][m][*l]*DS_NL1[1][0][Mj_AN][kl][n][*l];
    tmpz = PFm*ene_m*DS_NL1[1][3][Mc_AN][k][m][*l]*DS_NL1[1][0][Mj_AN][kl][n][*l];

    *sumx0r += tmpx;
    *sumy0r += tmpy;
    *sumz0r += tmpz;

    *sumx1r += tmpx;
    *sumy1r += tmpy;
    *sumz1r += tmpz;

    *l = *l + 1;
  }
}





void dH_U_full(int Mc_AN, int h_AN, int q_AN,
               double *****OLP, double ****v_eff,
               double ***Hx, double ***Hy, double ***Hz)
{
  int i,j,k,m,n,kg,kan,so,deri_kind,Mk_AN;
  int ig,ian,jg,jan,kl,kl1,kl2,spin,spinmax;
  int wakg,l1,l2,l3,Gc_AN,Mi_AN,Mj_AN;
  int Rwan,Lwan,p,p0;
  double PF[2],sumx,sumy,sumz,ene;
  double tmpx,tmpy,tmpz;
  double Lsum0,Lsum1,Lsum2,Lsum3;
  double Rsum0,Rsum1,Rsum2,Rsum3;
  double PFp,PFm,ene_p,ene_m;
  double ***Hx2,***Hy2,***Hz2;
  double sumx0,sumy0,sumz0;
  double sumx1,sumy1,sumz1;
  double sumx2,sumy2,sumz2;

  /****************************************************
   allocation of arrays:

   double Hx2[3][List_YOUSO[7]][List_YOUSO[7]];
   double Hy2[3][List_YOUSO[7]][List_YOUSO[7]];
   double Hz2[3][List_YOUSO[7]][List_YOUSO[7]];
  ****************************************************/

  Hx2 = (double***)malloc(sizeof(double**)*3);
  for (i=0; i<3; i++){
    Hx2[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hx2[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }
  }

  Hy2 = (double***)malloc(sizeof(double**)*3);
  for (i=0; i<3; i++){
    Hy2[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hy2[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }
  }

  Hz2 = (double***)malloc(sizeof(double**)*3);
  for (i=0; i<3; i++){
    Hz2[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hz2[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }
  }
  
  /****************************************************
   start calc.
  ****************************************************/

  if (SpinP_switch==0) spinmax = 0;
  else                 spinmax = 1;

  Gc_AN = M2G[Mc_AN];
  ig = natn[Gc_AN][h_AN];
  Lwan = WhatSpecies[ig];
  Mi_AN = F_G2M[ig]; /* F_G2M should be used */ 
  ian = Spe_Total_CNO[Lwan];
  jg = natn[Gc_AN][q_AN];
  Rwan = WhatSpecies[jg];
  Mj_AN = F_G2M[jg]; /* F_G2M should be used */
  jan = Spe_Total_CNO[Rwan];

  if (h_AN==0){

    /****************************************************
                          dS*ep*S
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      Mk_AN = F_G2M[kg];  /* F_G2M should be used */
      wakg = WhatSpecies[kg];
      kan = Spe_Total_NO[wakg];
      kl = RMI1[Mc_AN][q_AN][k];

      /****************************************************
                  derivative at h_AN (=Mc_AN)                  
      ****************************************************/

      if (0<=kl){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

            for (spin=0; spin<=spinmax; spin++){

	      sumx = 0.0;
	      sumy = 0.0;
	      sumz = 0.0;

              if (Cnt_switch==0){

		for (l1=0; l1<kan; l1++){
		  for (l2=0; l2<kan; l2++){
		    ene = v_eff[spin][Mk_AN][l1][l2];
		    sumx += ene*OLP[1][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		    sumy += ene*OLP[2][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		    sumz += ene*OLP[3][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		  }
		}

	      }

              else if (Cnt_switch==1){

		for (l1=0; l1<kan; l1++){
		  for (l2=0; l2<kan; l2++){
		    Lsum1 = 0.0; 
		    Lsum2 = 0.0; 
		    Lsum3 = 0.0; 
		    for (p=0; p<Spe_Specified_Num[Lwan][m]; p++){
		      p0 = Spe_Trans_Orbital[Lwan][m][p];
		      Lsum1 += CntCoes[Mc_AN][m][p]*OLP[1][Mc_AN][k][p0][l1];
		      Lsum2 += CntCoes[Mc_AN][m][p]*OLP[2][Mc_AN][k][p0][l1];
		      Lsum3 += CntCoes[Mc_AN][m][p]*OLP[3][Mc_AN][k][p0][l1];
		    }

		    Rsum0 = 0.0; 
		    for (p=0; p<Spe_Specified_Num[Rwan][n]; p++){
		      p0 = Spe_Trans_Orbital[Rwan][n][p];
		      Rsum0 += CntCoes[Mj_AN][n][p]*OLP[0][Mj_AN][kl][p0][l2];
		    }

		    ene = v_eff[spin][Mk_AN][l1][l2];
		    sumx += ene*Lsum1*Rsum0;
		    sumy += ene*Lsum2*Rsum0;
		    sumz += ene*Lsum3*Rsum0;
		  }
		}

	      }

	      if (k==0){
		Hx[spin][m][n] = sumx;
		Hy[spin][m][n] = sumy;
		Hz[spin][m][n] = sumz;

		Hx[2][m][n] = 0.0; 
		Hy[2][m][n] = 0.0; 
		Hz[2][m][n] = 0.0; 
	      }
	      else {
		Hx[spin][m][n] += sumx;
		Hy[spin][m][n] += sumy;
		Hz[spin][m][n] += sumz;
	      }
	    }
	  }
	}
      } /* if */
    } /* k */

    /****************************************************
                          S*ep*dS 
    ****************************************************/

    if (q_AN==0){
      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
          Hx2[0][m][n] = Hx[0][m][n];
          Hy2[0][m][n] = Hy[0][m][n];
          Hz2[0][m][n] = Hz[0][m][n];

          Hx2[1][m][n] = Hx[1][m][n];
          Hy2[1][m][n] = Hy[1][m][n];
          Hz2[1][m][n] = Hz[1][m][n];
        }
      }
      for (m=0; m<ian; m++){
        for (n=0; n<jan; n++){
          Hx[0][m][n] = Hx2[0][m][n] + Hx2[0][n][m];
          Hy[0][m][n] = Hy2[0][m][n] + Hy2[0][n][m];
          Hz[0][m][n] = Hz2[0][m][n] + Hz2[0][n][m];

          Hx[1][m][n] = Hx2[1][m][n] + Hx2[1][n][m];
          Hy[1][m][n] = Hy2[1][m][n] + Hy2[1][n][m];
          Hz[1][m][n] = Hz2[1][m][n] + Hz2[1][n][m];
        }
      }
    }

    else {

      kg = natn[Gc_AN][0];
      Mk_AN = F_G2M[kg]; /* F_G2M should be used */
      wakg = WhatSpecies[kg];
      kan = Spe_Total_NO[wakg];
      kl = RMI1[Mc_AN][q_AN][0];

      /****************************************************
                        derivative at k=0
      ****************************************************/

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

          for (spin=0; spin<=spinmax; spin++){

  	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

            if (Cnt_switch==0){

	      for (l1=0; l1<kan; l1++){
  	        for (l2=0; l2<kan; l2++){
		  ene = v_eff[spin][Mk_AN][l1][l2];
		  sumx -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[1][Mj_AN][kl][n][l2];
		  sumy -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[2][Mj_AN][kl][n][l2];
		  sumz -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[3][Mj_AN][kl][n][l2];
		}
	      }

	    }

            else if (Cnt_switch==1){

              for (l1=0; l1<kan; l1++){
                for (l2=0; l2<kan; l2++){

		  Lsum0 = 0.0; 

		  for (p=0; p<Spe_Specified_Num[Lwan][m]; p++){
		    p0 = Spe_Trans_Orbital[Lwan][m][p];
		    Lsum0 += CntCoes[Mc_AN][m][p]*OLP[0][Mc_AN][0][p0][l1];
		  }

		  Rsum1 = 0.0; 
		  Rsum2 = 0.0; 
		  Rsum3 = 0.0; 

		  for (p=0; p<Spe_Specified_Num[Rwan][n]; p++){
		    p0 = Spe_Trans_Orbital[Rwan][n][p];
		    Rsum1 += CntCoes[Mj_AN][n][p]*OLP[1][Mj_AN][kl][p0][l2];
		    Rsum2 += CntCoes[Mj_AN][n][p]*OLP[2][Mj_AN][kl][p0][l2];
		    Rsum3 += CntCoes[Mj_AN][n][p]*OLP[3][Mj_AN][kl][p0][l2];
		  }

		  ene = v_eff[spin][Mk_AN][l1][l2];
		  sumx -= ene*Lsum0*Rsum1;
		  sumy -= ene*Lsum0*Rsum2;
		  sumz -= ene*Lsum0*Rsum3;
		}
	      }
	    }


	    Hx[spin][m][n] += sumx;
	    Hy[spin][m][n] += sumy;
	    Hz[spin][m][n] += sumz;
	  }
	}
      }
    }

  } /* if (h_AN==0) */

  else {

    /****************************************************
                           dS*ep*S
    ****************************************************/

    kg = natn[Gc_AN][0];
    Mk_AN = F_G2M[kg]; /* F_G2M should be used */
    wakg = WhatSpecies[kg];
    kan = Spe_Total_NO[wakg];
    kl1 = RMI1[Mc_AN][h_AN][0];
    kl2 = RMI1[Mc_AN][q_AN][0];

    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){

        for (spin=0; spin<=spinmax; spin++){

   	  sumx = 0.0;
	  sumy = 0.0;
	  sumz = 0.0;

          if (Cnt_switch==0){

            for (l1=0; l1<kan; l1++){
              for (l2=0; l2<kan; l2++){
		ene = v_eff[spin][Mk_AN][l1][l2];
		sumx -= ene*OLP[1][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
		sumy -= ene*OLP[2][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
		sumz -= ene*OLP[3][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	      }
	    }
	  }

          else if (Cnt_switch==1){

            for (l1=0; l1<kan; l1++){
              for (l2=0; l2<kan; l2++){

		Lsum1 = 0.0; 
		Lsum2 = 0.0; 
		Lsum3 = 0.0; 
		for (p=0; p<Spe_Specified_Num[Lwan][m]; p++){
		  p0 = Spe_Trans_Orbital[Lwan][m][p];
		  Lsum1 += CntCoes[Mi_AN][m][p]*OLP[1][Mi_AN][kl1][p0][l1];
		  Lsum2 += CntCoes[Mi_AN][m][p]*OLP[2][Mi_AN][kl1][p0][l1];
		  Lsum3 += CntCoes[Mi_AN][m][p]*OLP[3][Mi_AN][kl1][p0][l1];
		}

		Rsum0 = 0.0; 
		for (p=0; p<Spe_Specified_Num[Rwan][n]; p++){
		  p0 = Spe_Trans_Orbital[Rwan][n][p];
		  Rsum0 += CntCoes[Mj_AN][n][p]*OLP[0][Mj_AN][kl2][p0][l2];
		}

		ene = v_eff[spin][Mk_AN][l1][l2];
		sumx -= ene*Lsum1*Rsum0;
		sumy -= ene*Lsum2*Rsum0;
		sumz -= ene*Lsum3*Rsum0;
	      }
	    }

	  }


	  Hx[spin][m][n] = sumx;
	  Hy[spin][m][n] = sumy;
	  Hz[spin][m][n] = sumz;

	  Hx[2][m][n] = 0.0;         
	  Hy[2][m][n] = 0.0;         
	  Hz[2][m][n] = 0.0;         
	}
      }
    }

    /****************************************************
                           S*ep*dS
    ****************************************************/

    if (q_AN==0){

      for (k=0; k<=FNAN[Gc_AN]; k++){
        kg = natn[Gc_AN][k];
        Mk_AN = F_G2M[kg]; /* F_G2M should be used */
        wakg = WhatSpecies[kg];
        kan = Spe_Total_NO[wakg];
        kl1 = RMI1[Mc_AN][h_AN][k];
        kl2 = RMI1[Mc_AN][q_AN][k];

        if (0<=kl1){

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

              for (spin=0; spin<=spinmax; spin++){

	        sumx = 0.0;
	        sumy = 0.0;
	        sumz = 0.0;

                if (Cnt_switch==0){

		  for (l1=0; l1<kan; l1++){
		    for (l2=0; l2<kan; l2++){
		      ene = v_eff[spin][Mk_AN][l1][l2];
		      sumx += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		      sumy += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		      sumz += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];
		    }
		  }

		}

                else if (Cnt_switch==1){

                  for (l1=0; l1<kan; l1++){
                    for (l2=0; l2<kan; l2++){

		      Lsum0 = 0.0; 

		      for (p=0; p<Spe_Specified_Num[Lwan][m]; p++){
			p0 = Spe_Trans_Orbital[Lwan][m][p];
			Lsum0 += CntCoes[Mi_AN][m][p]*OLP[0][Mi_AN][kl1][p0][l1];
		      }

		      Rsum1 = 0.0; 
		      Rsum2 = 0.0; 
		      Rsum3 = 0.0; 

		      for (p=0; p<Spe_Specified_Num[Rwan][n]; p++){
			p0 = Spe_Trans_Orbital[Rwan][n][p];
			Rsum1 += CntCoes[Mj_AN][n][p]*OLP[1][Mj_AN][kl2][p0][l2];
			Rsum2 += CntCoes[Mj_AN][n][p]*OLP[2][Mj_AN][kl2][p0][l2];
			Rsum3 += CntCoes[Mj_AN][n][p]*OLP[3][Mj_AN][kl2][p0][l2];
		      }

		      ene = v_eff[spin][Mk_AN][l1][l2];
		      sumx += ene*Lsum0*Rsum1;
		      sumy += ene*Lsum0*Rsum2;
		      sumz += ene*Lsum0*Rsum3;

		    }
		  }
		}

		Hx[spin][m][n] += sumx;
		Hy[spin][m][n] += sumy;
		Hz[spin][m][n] += sumz;
	      }
	    }
	  }
	}

      }
    } /* if (q_AN==0) */

    else {

      kg = natn[Gc_AN][0];
      Mk_AN = F_G2M[kg]; /* F_G2M should be used */
      wakg = WhatSpecies[kg];
      kan = Spe_Total_NO[wakg];
      kl1 = RMI1[Mc_AN][h_AN][0];
      kl2 = RMI1[Mc_AN][q_AN][0];

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

          for (spin=0; spin<=spinmax; spin++){

 	    sumx = 0.0;
	    sumy = 0.0;
	    sumz = 0.0;

            if (Cnt_switch==0){

              for (l1=0; l1<kan; l1++){
                for (l2=0; l2<kan; l2++){
		  ene = v_eff[spin][Mk_AN][l1][l2];
		  sumx -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		  sumy -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		  sumz -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];
		}
	      }
	    }

            else if (Cnt_switch==1){

              for (l1=0; l1<kan; l1++){
                for (l2=0; l2<kan; l2++){

		  Lsum0 = 0.0; 

		  for (p=0; p<Spe_Specified_Num[Lwan][m]; p++){
		    p0 = Spe_Trans_Orbital[Lwan][m][p];
		    Lsum0 += CntCoes[Mi_AN][m][p]*OLP[0][Mi_AN][kl1][p0][l1];
		  }

		  Rsum1 = 0.0; 
		  Rsum2 = 0.0; 
		  Rsum3 = 0.0; 

		  for (p=0; p<Spe_Specified_Num[Rwan][n]; p++){
		    p0 = Spe_Trans_Orbital[Rwan][n][p];
		    Rsum1 += CntCoes[Mj_AN][n][p]*OLP[1][Mj_AN][kl2][p0][l2];
		    Rsum2 += CntCoes[Mj_AN][n][p]*OLP[2][Mj_AN][kl2][p0][l2];
		    Rsum3 += CntCoes[Mj_AN][n][p]*OLP[3][Mj_AN][kl2][p0][l2];
		  }

		  ene = v_eff[spin][Mk_AN][l1][l2];
		  sumx -= ene*Lsum0*Rsum1;
		  sumy -= ene*Lsum0*Rsum2;
		  sumz -= ene*Lsum0*Rsum3;
		}
	      }
	    }

	    Hx[spin][m][n] += sumx;
	    Hy[spin][m][n] += sumy;
	    Hz[spin][m][n] += sumz;
	  }
	}
      }
    }
  }

  /****************************************************
   freeing of arrays:

   double Hx2[3][List_YOUSO[7]][List_YOUSO[7]];
   double Hy2[3][List_YOUSO[7]][List_YOUSO[7]];
   double Hz2[3][List_YOUSO[7]][List_YOUSO[7]];
  ****************************************************/

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hx2[i][j]);
    }
    free(Hx2[i]);
  }
  free(Hx2);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hy2[i][j]);
    }
    free(Hy2[i]);
  }
  free(Hy2);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hz2[i][j]);
    }
    free(Hz2[i]);
  }
  free(Hz2);
}







void dH_U_NC_full(int Mc_AN, int h_AN, int q_AN,
                  double *****OLP, dcomplex *****NC_v_eff,
                  dcomplex ****Hx, dcomplex ****Hy, dcomplex ****Hz)
{
  int i,j,k,m,n,kg,kan,so,deri_kind,Mk_AN;
  int ig,ian,jg,jan,kl,kl1,kl2,spin;
  int wakg,l1,l2,l3,Gc_AN,Mi_AN,Mj_AN;
  int Rwan,Lwan,p,p0,s1,s2;
  double PF[2],sumx,sumy,sumz,ene;
  double tmpx,tmpy,tmpz;
  double Lsum0,Lsum1,Lsum2,Lsum3;
  double Rsum0,Rsum1,Rsum2,Rsum3;
  double PFp,PFm,ene_p,ene_m;
  double Re00x,Re00y,Re00z;
  double Re11x,Re11y,Re11z;
  double Re01x,Re01y,Re01z;
  double Re10x,Re10y,Re10z;
  double Im00x,Im00y,Im00z;
  double Im11x,Im11y,Im11z;
  double Im01x,Im01y,Im01z;
  double Im10x,Im10y,Im10z;

  /****************************************************
   start calc.
  ****************************************************/

  Gc_AN = M2G[Mc_AN];
  ig = natn[Gc_AN][h_AN];
  Lwan = WhatSpecies[ig];
  Mi_AN = F_G2M[ig]; /* F_G2M should be used */ 
  ian = Spe_Total_CNO[Lwan];
  jg = natn[Gc_AN][q_AN];
  Rwan = WhatSpecies[jg];
  Mj_AN = F_G2M[jg]; /* F_G2M should be used */
  jan = Spe_Total_CNO[Rwan];

  if (h_AN==0){

    /****************************************************
                          dS*ep*S
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      Mk_AN = F_G2M[kg];  /* F_G2M should be used */
      wakg = WhatSpecies[kg];
      kan = Spe_Total_NO[wakg];
      kl = RMI1[Mc_AN][q_AN][k];

      /****************************************************
                  derivative at h_AN (=Mc_AN)                  
      ****************************************************/

      if (0<=kl){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    Re00x = 0.0;     Re00y = 0.0;     Re00z = 0.0;
	    Re11x = 0.0;     Re11y = 0.0;     Re11z = 0.0;
	    Re01x = 0.0;     Re01y = 0.0;     Re01z = 0.0;
	    Re10x = 0.0;     Re10y = 0.0;     Re10z = 0.0;

	    Im00x = 0.0;     Im00y = 0.0;     Im00z = 0.0;
	    Im11x = 0.0;     Im11y = 0.0;     Im11z = 0.0;
	    Im01x = 0.0;     Im01y = 0.0;     Im01z = 0.0;
	    Im10x = 0.0;     Im10y = 0.0;     Im10z = 0.0;

	    for (l1=0; l1<kan; l1++){
	      for (l2=0; l2<kan; l2++){

		ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
		Re00x += ene*OLP[1][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Re00y += ene*OLP[2][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Re00z += ene*OLP[3][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];

		ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
		Re11x += ene*OLP[1][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Re11y += ene*OLP[2][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Re11z += ene*OLP[3][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];

		ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
		Re01x += ene*OLP[1][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Re01y += ene*OLP[2][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Re01z += ene*OLP[3][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];

		ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
		Re10x += ene*OLP[1][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Re10y += ene*OLP[2][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Re10z += ene*OLP[3][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];

		ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
		Im00x += ene*OLP[1][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Im00y += ene*OLP[2][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Im00z += ene*OLP[3][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];

		ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
		Im11x += ene*OLP[1][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Im11y += ene*OLP[2][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Im11z += ene*OLP[3][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];

		ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
		Im01x += ene*OLP[1][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Im01y += ene*OLP[2][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Im01z += ene*OLP[3][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];

		ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
		Im10x += ene*OLP[1][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Im10y += ene*OLP[2][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];
		Im10z += ene*OLP[3][Mc_AN][k][m][l1]*OLP[0][Mj_AN][kl][n][l2];

	      }
	    }


	    if (k==0){
	      Hx[0][0][m][n] = Complex(Re00x,Im00x);
	      Hy[0][0][m][n] = Complex(Re00y,Im00y);
	      Hz[0][0][m][n] = Complex(Re00z,Im00z);

	      Hx[1][1][m][n] = Complex(Re11x,Im11x);
	      Hy[1][1][m][n] = Complex(Re11y,Im11y);
	      Hz[1][1][m][n] = Complex(Re11z,Im11z);

	      Hx[0][1][m][n] = Complex(Re01x,Im01x);
	      Hy[0][1][m][n] = Complex(Re01y,Im01y);
	      Hz[0][1][m][n] = Complex(Re01z,Im01z);

	      Hx[1][0][m][n] = Complex(Re10x,Im10x);
	      Hy[1][0][m][n] = Complex(Re10y,Im10y);
	      Hz[1][0][m][n] = Complex(Re10z,Im10z);
	    }
	    else{

	      Hx[0][0][m][n].r += Re00x;  Hx[0][0][m][n].i += Im00x;
	      Hy[0][0][m][n].r += Re00y;  Hy[0][0][m][n].i += Im00y;
	      Hz[0][0][m][n].r += Re00z;  Hz[0][0][m][n].i += Im00z;

	      Hx[1][1][m][n].r += Re11x;  Hx[1][1][m][n].i += Im11x;
	      Hy[1][1][m][n].r += Re11y;  Hy[1][1][m][n].i += Im11y;
	      Hz[1][1][m][n].r += Re11z;  Hz[1][1][m][n].i += Im11z;

	      Hx[0][1][m][n].r += Re01x;  Hx[0][1][m][n].i += Im01x;
	      Hy[0][1][m][n].r += Re01y;  Hy[0][1][m][n].i += Im01y;
	      Hz[0][1][m][n].r += Re01z;  Hz[0][1][m][n].i += Im01z;

	      Hx[1][0][m][n].r += Re10x;  Hx[1][0][m][n].i += Im10x;
	      Hy[1][0][m][n].r += Re10y;  Hy[1][0][m][n].i += Im10y;
	      Hz[1][0][m][n].r += Re10z;  Hz[1][0][m][n].i += Im10z;
	    }

	  } /* n */
	} /* m */
      } /* if */
    } /* k */

    /****************************************************
                          S*ep*dS 
    ****************************************************/

    /* ????? */

    if (q_AN==0){
 
      for (s1=0; s1<2; s1++){
	for (s2=0; s2<2; s2++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      Hx[s1][s2][m][n].r = 2.0*Hx[s1][s2][m][n].r;
	      Hy[s1][s2][m][n].r = 2.0*Hy[s1][s2][m][n].r;
	      Hz[s1][s2][m][n].r = 2.0*Hz[s1][s2][m][n].r;

	      Hx[s1][s2][m][n].i = 2.0*Hx[s1][s2][m][n].i;
	      Hy[s1][s2][m][n].i = 2.0*Hy[s1][s2][m][n].i;
	      Hz[s1][s2][m][n].i = 2.0*Hz[s1][s2][m][n].i;
	    }
	  }
	}
      }
    }

    else {

      kg = natn[Gc_AN][0];
      Mk_AN = F_G2M[kg]; /* F_G2M should be used */
      wakg = WhatSpecies[kg];
      kan = Spe_Total_NO[wakg];
      kl = RMI1[Mc_AN][q_AN][0];

      /****************************************************
                        derivative at k=0
      ****************************************************/

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  Re00x = 0.0;     Re00y = 0.0;     Re00z = 0.0;
	  Re11x = 0.0;     Re11y = 0.0;     Re11z = 0.0;
	  Re01x = 0.0;     Re01y = 0.0;     Re01z = 0.0;
	  Re10x = 0.0;     Re10y = 0.0;     Re10z = 0.0;

	  Im00x = 0.0;     Im00y = 0.0;     Im00z = 0.0;
	  Im11x = 0.0;     Im11y = 0.0;     Im11z = 0.0;
	  Im01x = 0.0;     Im01y = 0.0;     Im01z = 0.0;
	  Im10x = 0.0;     Im10y = 0.0;     Im10z = 0.0;

	  for (l1=0; l1<kan; l1++){
	    for (l2=0; l2<kan; l2++){

	      ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
	      Re00x -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[1][Mj_AN][kl][n][l2];
	      Re00y -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[2][Mj_AN][kl][n][l2];
	      Re00z -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[3][Mj_AN][kl][n][l2];

	      ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
	      Re11x -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[1][Mj_AN][kl][n][l2];
	      Re11y -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[2][Mj_AN][kl][n][l2];
	      Re11z -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[3][Mj_AN][kl][n][l2];

	      ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
	      Re01x -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[1][Mj_AN][kl][n][l2];
	      Re01y -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[2][Mj_AN][kl][n][l2];
	      Re01z -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[3][Mj_AN][kl][n][l2];

	      ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
	      Re10x -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[1][Mj_AN][kl][n][l2];
	      Re10y -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[2][Mj_AN][kl][n][l2];
	      Re10z -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[3][Mj_AN][kl][n][l2];

	      ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
	      Im00x -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[1][Mj_AN][kl][n][l2];
	      Im00y -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[2][Mj_AN][kl][n][l2];
	      Im00z -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[3][Mj_AN][kl][n][l2];

	      ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
	      Im11x -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[1][Mj_AN][kl][n][l2];
	      Im11y -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[2][Mj_AN][kl][n][l2];
	      Im11z -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[3][Mj_AN][kl][n][l2];

	      ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
	      Im01x -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[1][Mj_AN][kl][n][l2];
	      Im01y -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[2][Mj_AN][kl][n][l2];
	      Im01z -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[3][Mj_AN][kl][n][l2];

	      ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
	      Im10x -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[1][Mj_AN][kl][n][l2];
	      Im10y -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[2][Mj_AN][kl][n][l2];
	      Im10z -= ene*OLP[0][Mc_AN][0][m][l1]*OLP[3][Mj_AN][kl][n][l2];

	    }
	  }

	  Hx[0][0][m][n].r += Re00x;  Hx[0][0][m][n].i += Im00x;
	  Hy[0][0][m][n].r += Re00y;  Hy[0][0][m][n].i += Im00y;
	  Hz[0][0][m][n].r += Re00z;  Hz[0][0][m][n].i += Im00z;

	  Hx[1][1][m][n].r += Re11x;  Hx[1][1][m][n].i += Im11x;
	  Hy[1][1][m][n].r += Re11y;  Hy[1][1][m][n].i += Im11y;
	  Hz[1][1][m][n].r += Re11z;  Hz[1][1][m][n].i += Im11z;

	  Hx[0][1][m][n].r += Re01x;  Hx[0][1][m][n].i += Im01x;
	  Hy[0][1][m][n].r += Re01y;  Hy[0][1][m][n].i += Im01y;
	  Hz[0][1][m][n].r += Re01z;  Hz[0][1][m][n].i += Im01z;

	  Hx[1][0][m][n].r += Re10x;  Hx[1][0][m][n].i += Im10x;
	  Hy[1][0][m][n].r += Re10y;  Hy[1][0][m][n].i += Im10y;
	  Hz[1][0][m][n].r += Re10z;  Hz[1][0][m][n].i += Im10z;
	}
      }
    }

  } /* if (h_AN==0) */

  else {

    /****************************************************
                           dS*ep*S
    ****************************************************/

    kg = natn[Gc_AN][0];
    Mk_AN = F_G2M[kg]; /* F_G2M should be used */
    wakg = WhatSpecies[kg];
    kan = Spe_Total_NO[wakg];
    kl1 = RMI1[Mc_AN][h_AN][0];
    kl2 = RMI1[Mc_AN][q_AN][0];

    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){

	Re00x = 0.0;     Re00y = 0.0;     Re00z = 0.0;
	Re11x = 0.0;     Re11y = 0.0;     Re11z = 0.0;
	Re01x = 0.0;     Re01y = 0.0;     Re01z = 0.0;
	Re10x = 0.0;     Re10y = 0.0;     Re10z = 0.0;

	Im00x = 0.0;     Im00y = 0.0;     Im00z = 0.0;
	Im11x = 0.0;     Im11y = 0.0;     Im11z = 0.0;
	Im01x = 0.0;     Im01y = 0.0;     Im01z = 0.0;
	Im10x = 0.0;     Im10y = 0.0;     Im10z = 0.0;

	for (l1=0; l1<kan; l1++){
	  for (l2=0; l2<kan; l2++){

	    ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
	    Re00x -= ene*OLP[1][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Re00y -= ene*OLP[2][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Re00z -= ene*OLP[3][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];

	    ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
	    Re11x -= ene*OLP[1][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Re11y -= ene*OLP[2][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Re11z -= ene*OLP[3][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];

	    ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
	    Re01x -= ene*OLP[1][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Re01y -= ene*OLP[2][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Re01z -= ene*OLP[3][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];

	    ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
	    Re10x -= ene*OLP[1][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Re10y -= ene*OLP[2][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Re10z -= ene*OLP[3][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];

	    ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
	    Im00x -= ene*OLP[1][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Im00y -= ene*OLP[2][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Im00z -= ene*OLP[3][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];

	    ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
	    Im11x -= ene*OLP[1][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Im11y -= ene*OLP[2][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Im11z -= ene*OLP[3][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];

	    ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
	    Im01x -= ene*OLP[1][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Im01y -= ene*OLP[2][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Im01z -= ene*OLP[3][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];

	    ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
	    Im10x -= ene*OLP[1][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Im10y -= ene*OLP[2][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];
	    Im10z -= ene*OLP[3][Mi_AN][kl1][m][l1]*OLP[0][Mj_AN][kl2][n][l2];

	  }
	}

	Hx[0][0][m][n] = Complex(Re00x,Im00x);
	Hy[0][0][m][n] = Complex(Re00y,Im00y);
	Hz[0][0][m][n] = Complex(Re00z,Im00z);

	Hx[1][1][m][n] = Complex(Re11x,Im11x);
	Hy[1][1][m][n] = Complex(Re11y,Im11y);
	Hz[1][1][m][n] = Complex(Re11z,Im11z);

	Hx[0][1][m][n] = Complex(Re01x,Im01x);
	Hy[0][1][m][n] = Complex(Re01y,Im01y);
	Hz[0][1][m][n] = Complex(Re01z,Im01z);

	Hx[1][0][m][n] = Complex(Re10x,Im10x);
	Hy[1][0][m][n] = Complex(Re10y,Im10y);
	Hz[1][0][m][n] = Complex(Re10z,Im10z);
      }
    }

    /****************************************************
                           S*ep*dS
    ****************************************************/

    if (q_AN==0){

      for (k=0; k<=FNAN[Gc_AN]; k++){
        kg = natn[Gc_AN][k];
        Mk_AN = F_G2M[kg]; /* F_G2M should be used */
        wakg = WhatSpecies[kg];
        kan = Spe_Total_NO[wakg];
        kl1 = RMI1[Mc_AN][h_AN][k];
        kl2 = RMI1[Mc_AN][q_AN][k];

        if (0<=kl1){

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      Re00x = 0.0;     Re00y = 0.0;     Re00z = 0.0;
	      Re11x = 0.0;     Re11y = 0.0;     Re11z = 0.0;
	      Re01x = 0.0;     Re01y = 0.0;     Re01z = 0.0;
	      Re10x = 0.0;     Re10y = 0.0;     Re10z = 0.0;

	      Im00x = 0.0;     Im00y = 0.0;     Im00z = 0.0;
	      Im11x = 0.0;     Im11y = 0.0;     Im11z = 0.0;
	      Im01x = 0.0;     Im01y = 0.0;     Im01z = 0.0;
	      Im10x = 0.0;     Im10y = 0.0;     Im10z = 0.0;

	      for (l1=0; l1<kan; l1++){
		for (l2=0; l2<kan; l2++){

		  ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
		  Re00x += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		  Re00y += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		  Re00z += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

		  ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
		  Re11x += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		  Re11y += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		  Re11z += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

		  ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
		  Re01x += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		  Re01y += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		  Re01z += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

		  ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
		  Re10x += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		  Re10y += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		  Re10z += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

		  ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
		  Im00x += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		  Im00y += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		  Im00z += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

		  ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
		  Im11x += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		  Im11y += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		  Im11z += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

		  ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
		  Im01x += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		  Im01y += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		  Im01z += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

		  ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
		  Im10x += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
		  Im10y += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
		  Im10z += ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

		}
	      }

	      Hx[0][0][m][n].r += Re00x;  Hx[0][0][m][n].i += Im00x;
	      Hy[0][0][m][n].r += Re00y;  Hy[0][0][m][n].i += Im00y;
	      Hz[0][0][m][n].r += Re00z;  Hz[0][0][m][n].i += Im00z;

	      Hx[1][1][m][n].r += Re11x;  Hx[1][1][m][n].i += Im11x;
	      Hy[1][1][m][n].r += Re11y;  Hy[1][1][m][n].i += Im11y;
	      Hz[1][1][m][n].r += Re11z;  Hz[1][1][m][n].i += Im11z;

	      Hx[0][1][m][n].r += Re01x;  Hx[0][1][m][n].i += Im01x;
	      Hy[0][1][m][n].r += Re01y;  Hy[0][1][m][n].i += Im01y;
	      Hz[0][1][m][n].r += Re01z;  Hz[0][1][m][n].i += Im01z;

	      Hx[1][0][m][n].r += Re10x;  Hx[1][0][m][n].i += Im10x;
	      Hy[1][0][m][n].r += Re10y;  Hy[1][0][m][n].i += Im10y;
	      Hz[1][0][m][n].r += Re10z;  Hz[1][0][m][n].i += Im10z;

	    }
	  }
	}

      }
    } /* if (q_AN==0) */

    else {

      kg = natn[Gc_AN][0];
      Mk_AN = F_G2M[kg]; /* F_G2M should be used */
      wakg = WhatSpecies[kg];
      kan = Spe_Total_NO[wakg];
      kl1 = RMI1[Mc_AN][h_AN][0];
      kl2 = RMI1[Mc_AN][q_AN][0];

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  Re00x = 0.0;     Re00y = 0.0;     Re00z = 0.0;
	  Re11x = 0.0;     Re11y = 0.0;     Re11z = 0.0;
	  Re01x = 0.0;     Re01y = 0.0;     Re01z = 0.0;
	  Re10x = 0.0;     Re10y = 0.0;     Re10z = 0.0;

	  Im00x = 0.0;     Im00y = 0.0;     Im00z = 0.0;
	  Im11x = 0.0;     Im11y = 0.0;     Im11z = 0.0;
	  Im01x = 0.0;     Im01y = 0.0;     Im01z = 0.0;
	  Im10x = 0.0;     Im10y = 0.0;     Im10z = 0.0;

	  for (l1=0; l1<kan; l1++){
	    for (l2=0; l2<kan; l2++){

	      ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
	      Re00x -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
	      Re00y -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
	      Re00z -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

	      ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
	      Re11x -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
	      Re11y -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
	      Re11z -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

	      ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
	      Re01x -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
	      Re01y -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
	      Re01z -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

	      ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
	      Re10x -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
	      Re10y -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
	      Re10z -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

	      ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
	      Im00x -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
	      Im00y -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
	      Im00z -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

	      ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
	      Im11x -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
	      Im11y -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
	      Im11z -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

	      ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
	      Im01x -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
	      Im01y -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
	      Im01z -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

	      ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
	      Im10x -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[1][Mj_AN][kl2][n][l2];
	      Im10y -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[2][Mj_AN][kl2][n][l2];
	      Im10z -= ene*OLP[0][Mi_AN][kl1][m][l1]*OLP[3][Mj_AN][kl2][n][l2];

	    }
	  }

	  Hx[0][0][m][n].r += Re00x;  Hx[0][0][m][n].i += Im00x;
	  Hy[0][0][m][n].r += Re00y;  Hy[0][0][m][n].i += Im00y;
	  Hz[0][0][m][n].r += Re00z;  Hz[0][0][m][n].i += Im00z;

	  Hx[1][1][m][n].r += Re11x;  Hx[1][1][m][n].i += Im11x;
	  Hy[1][1][m][n].r += Re11y;  Hy[1][1][m][n].i += Im11y;
	  Hz[1][1][m][n].r += Re11z;  Hz[1][1][m][n].i += Im11z;

	  Hx[0][1][m][n].r += Re01x;  Hx[0][1][m][n].i += Im01x;
	  Hy[0][1][m][n].r += Re01y;  Hy[0][1][m][n].i += Im01y;
	  Hz[0][1][m][n].r += Re01z;  Hz[0][1][m][n].i += Im01z;

	  Hx[1][0][m][n].r += Re10x;  Hx[1][0][m][n].i += Im10x;
	  Hy[1][0][m][n].r += Re10y;  Hy[1][0][m][n].i += Im10y;
	  Hz[1][0][m][n].r += Re10z;  Hz[1][0][m][n].i += Im10z;
	  
	}
      }
    }
  }

}







void dHCH(int where_flag,
          int Mc_AN, int h_AN, int q_AN,
          double *****OLP1,
          dcomplex ***Hx, dcomplex ***Hy, dcomplex ***Hz)
{
  int i,j,k,m,n,l,kg,kan,so,deri_kind,L;
  int ig,ian,jg,jan,kl,kl1,kl2,mul;
  int wakg,l1,l2,l3,Gc_AN,Mi_AN,Mi_AN2,Mj_AN,Mj_AN2;
  int Rni,Rnj,somax,apply_flag,target_spin;
  double **penalty;
  double penalty_value;
  double PF[2],sumx[2],sumy[2],sumz[2],ene,dmp,deri_dmp;
  double tmpx,tmpy,tmpz,tmp,r;
  double x0,y0,z0,x1,y1,z1,dx,dy,dz;
  double rcuti,rcutj,rcut;
  double PFp,PFm,ene_p,ene_m;
  dcomplex sumx0,sumy0,sumz0;
  dcomplex sumx1,sumy1,sumz1;
  dcomplex sumx2,sumy2,sumz2;

  /****************************************************
   start calc.
  ****************************************************/

  /* set penalty */

  penalty_value = penalty_value_CoreHole;

  penalty = (double**)malloc(sizeof(double*)*2);
  for (i=0; i<2; i++){
    penalty[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  wakg = WhatSpecies[Core_Hole_Atom];

  /* set penalty */

  if (VPS_j_dependency[wakg]==0){

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
  }

  /* get information of relevant atoms */

  Gc_AN = M2G[Mc_AN];
  ig = natn[Gc_AN][h_AN];
  Rni = ncn[Gc_AN][h_AN];
  Mi_AN = F_G2M[ig];
  ian = Spe_Total_CNO[WhatSpecies[ig]];
  rcuti = Spe_Atom_Cut1[WhatSpecies[ig]];

  jg = natn[Gc_AN][q_AN];
  Rnj = ncn[Gc_AN][q_AN];
  Mj_AN = F_G2M[jg];
  jan = Spe_Total_CNO[WhatSpecies[jg]];
  rcutj = Spe_Atom_Cut1[WhatSpecies[jg]];

  rcut = rcuti + rcutj;
  kl = RMI1[Mc_AN][h_AN][q_AN];
  dmp = dampingF(rcut,Dis[ig][kl]);

  for (so=0; so<3; so++){
    for (i=0; i<List_YOUSO[7]; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	Hx[so][i][j] = Complex(0.0,0.0);
	Hy[so][i][j] = Complex(0.0,0.0);
	Hz[so][i][j] = Complex(0.0,0.0);
      }
    }
  }

  if (h_AN==0){

    /****************************************************
                          dH*ep*H
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_CNO[wakg];
      kl = RMI1[Mc_AN][q_AN][k];

      /****************************************************
                   l-dependent non-local part
      ****************************************************/

      if (0<=kl && VPS_j_dependency[wakg]==0 && where_flag==0){

        if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
        else                 Mj_AN2 = Matomnum + 1; 

        if (kg==Core_Hole_Atom){

	  /* calculate the multiplication */

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      sumx[0] = 0.0; sumx[1] = 0.0;
	      sumy[0] = 0.0; sumy[1] = 0.0;
	      sumz[0] = 0.0; sumz[1] = 0.0;

	      for (i=0; i<Spe_Total_CNO[wakg]; i++){
		sumx[0] += penalty[0][i]*OLP1[1][Mc_AN][k][m][i]*OLP1[0][Mj_AN2][kl][n][i];
		sumy[0] += penalty[0][i]*OLP1[2][Mc_AN][k][m][i]*OLP1[0][Mj_AN2][kl][n][i];
		sumz[0] += penalty[0][i]*OLP1[3][Mc_AN][k][m][i]*OLP1[0][Mj_AN2][kl][n][i];

		sumx[1] += penalty[1][i]*OLP1[1][Mc_AN][k][m][i]*OLP1[0][Mj_AN2][kl][n][i];
		sumy[1] += penalty[1][i]*OLP1[2][Mc_AN][k][m][i]*OLP1[0][Mj_AN2][kl][n][i];
		sumz[1] += penalty[1][i]*OLP1[3][Mc_AN][k][m][i]*OLP1[0][Mj_AN2][kl][n][i];
	      }

	      Hx[0][m][n].r += sumx[0];
	      Hy[0][m][n].r += sumy[0];
	      Hz[0][m][n].r += sumz[0];

	      Hx[1][m][n].r += sumx[1];
	      Hy[1][m][n].r += sumy[1];
	      Hz[1][m][n].r += sumz[1];

	    } /* n */
	  } /* m */

	} /* if (kg==Core_Hole_Atom) */

      } /* if */

      /****************************************************
                   j-dependent non-local part
      ****************************************************/

      else if ( 0<=kl && VPS_j_dependency[wakg]==1 && where_flag==0 ){

        if (Mj_AN<=Matomnum) Mj_AN2 = Mj_AN;
        else                 Mj_AN2 = Matomnum + 1; 

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

            sumx0 = Complex(0.0,0.0); sumy0 = Complex(0.0,0.0); sumz0 = Complex(0.0,0.0);
            sumx1 = Complex(0.0,0.0); sumy1 = Complex(0.0,0.0); sumz1 = Complex(0.0,0.0);
            sumx2 = Complex(0.0,0.0); sumy2 = Complex(0.0,0.0); sumz2 = Complex(0.0,0.0);

	    dHCH_SO( &sumx0.r,&sumx0.i, &sumy0.r,&sumy0.i, &sumz0.r,&sumz0.i, 
                     &sumx1.r,&sumx1.i, &sumy1.r,&sumy1.i, &sumz1.r,&sumz1.i, 
                     &sumx2.r,&sumx2.i, &sumy2.r,&sumy2.i, &sumz2.r,&sumz2.i,
                     1.0,
                     Mc_AN ,k, m,
                     Mj_AN2,kl,n,
                     kg, wakg,
                     penalty_value,
                     OLP1 );

            if (q_AN==0){

	      dHCH_SO( &sumx0.r,&sumx0.i, &sumy0.r,&sumy0.i, &sumz0.r,&sumz0.i, 
                       &sumx1.r,&sumx1.i, &sumy1.r,&sumy1.i, &sumz1.r,&sumz1.i, 
                       &sumx2.r,&sumx2.i, &sumy2.r,&sumy2.i, &sumz2.r,&sumz2.i,
                       -1.0,
                       Mj_AN2, kl,n,
                       Mc_AN ,k, m, 
                       kg, wakg,
                       penalty_value,
                       OLP1 );
	    }

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    } /* k */

    /****************************************************
                           H*ep*dH 
    ****************************************************/

    /* h_AN==0 && q_AN==0 */

    if (q_AN==0 && VPS_j_dependency[wakg]==0){

      for (m=0; m<ian; m++){
        for (n=m; n<jan; n++){

          tmpx = Hx[0][m][n].r + Hx[0][n][m].r;
          Hx[0][m][n].r = tmpx; 
          Hx[0][n][m].r = tmpx;

          tmpy = Hy[0][m][n].r + Hy[0][n][m].r;
          Hy[0][m][n].r = tmpy; 
          Hy[0][n][m].r = tmpy;

          tmpz = Hz[0][m][n].r + Hz[0][n][m].r;
          Hz[0][m][n].r = tmpz; 
          Hz[0][n][m].r = tmpz;

          tmpx = Hx[1][m][n].r + Hx[1][n][m].r;
          Hx[1][m][n].r = tmpx; 
          Hx[1][n][m].r = tmpx;

          tmpy = Hy[1][m][n].r + Hy[1][n][m].r;
          Hy[1][m][n].r = tmpy; 
          Hy[1][n][m].r = tmpy;

          tmpz = Hz[1][m][n].r + Hz[1][n][m].r;
          Hz[1][m][n].r = tmpz; 
          Hz[1][n][m].r = tmpz;
        }
      }
    }

    else if (where_flag==1){

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl = RMI1[Mc_AN][q_AN][0];

      /****************************************************
                   l-dependent non-local part
      ****************************************************/

      if (VPS_j_dependency[wakg]==0){

	if (Mj_AN<=Matomnum){
	  Mj_AN2 = Mj_AN;
	  kl2 = RMI1[Mc_AN][q_AN][0];
	}
	else{
	  Mj_AN2 = Matomnum + 1; 
	  kl2 = RMI1[Mc_AN][0][q_AN];
	}

        if (kg==Core_Hole_Atom){

	  /* calculate the multiplication */

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      sumx[0] = 0.0; sumx[1] = 0.0;
	      sumy[0] = 0.0; sumy[1] = 0.0;
	      sumz[0] = 0.0; sumz[1] = 0.0;

	      for (i=0; i<Spe_Total_CNO[wakg]; i++){
		sumx[0] -= penalty[0][i]*OLP1[0][Mc_AN][0][m][i]*OLP1[1][Mj_AN2][kl2][n][i];
		sumy[0] -= penalty[0][i]*OLP1[0][Mc_AN][0][m][i]*OLP1[2][Mj_AN2][kl2][n][i];
		sumz[0] -= penalty[0][i]*OLP1[0][Mc_AN][0][m][i]*OLP1[3][Mj_AN2][kl2][n][i];

		sumx[1] -= penalty[1][i]*OLP1[0][Mc_AN][0][m][i]*OLP1[1][Mj_AN2][kl2][n][i];
		sumy[1] -= penalty[1][i]*OLP1[0][Mc_AN][0][m][i]*OLP1[2][Mj_AN2][kl2][n][i];
		sumz[1] -= penalty[1][i]*OLP1[0][Mc_AN][0][m][i]*OLP1[3][Mj_AN2][kl2][n][i];
	      }

	      Hx[0][m][n].r += sumx[0];
	      Hy[0][m][n].r += sumy[0];
	      Hz[0][m][n].r += sumz[0];

	      Hx[1][m][n].r += sumx[1];
	      Hy[1][m][n].r += sumy[1];
	      Hz[1][m][n].r += sumz[1];
	    }
	  }

	} /* if (kg==Core_Hole_Atom) */
      }

      /****************************************************
                   j-dependent non-local part
      ****************************************************/

      else if ( VPS_j_dependency[wakg]==1 ){

	if (Mj_AN<=Matomnum){
	  Mj_AN2 = Mj_AN;
	  kl2 = RMI1[Mc_AN][q_AN][0];
	}
	else{
	  Mj_AN2 = Matomnum + 1; 
	  kl2 = RMI1[Mc_AN][0][q_AN];
	}

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);  sumy0 = Complex(0.0,0.0);  sumz0 = Complex(0.0,0.0);
	    sumx1 = Complex(0.0,0.0);  sumy1 = Complex(0.0,0.0);  sumz1 = Complex(0.0,0.0);
	    sumx2 = Complex(0.0,0.0);  sumy2 = Complex(0.0,0.0);  sumz2 = Complex(0.0,0.0);

  	    /* 1 */

	    dHCH_SO( &sumx0.r,&sumx0.i, &sumy0.r,&sumy0.i, &sumz0.r,&sumz0.i, 
                     &sumx1.r,&sumx1.i, &sumy1.r,&sumy1.i, &sumz1.r,&sumz1.i, 
                     &sumx2.r,&sumx2.i, &sumy2.r,&sumy2.i, &sumz2.r,&sumz2.i,
                     -1.0,
                     Mj_AN2,kl2,n,
                     Mc_AN, 0,  m,
                     kg, wakg,
                     -penalty_value,
                     OLP1 );

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    }

  } /* if (h_AN==0) */


  else if (where_flag==0){

    /****************************************************
       H*ep*dH

       if (h_AN!=0 && where_flag==0)
       This happens 
       only if 
       ( SpinP_switch==3
         &&
         (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
  	   || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1
           || Zeeman_NCO_switch==1)
         && 
         q_AN==0
       )
    ****************************************************/

    for (k=0; k<=FNAN[Gc_AN]; k++){

      kg = natn[Gc_AN][k];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl = RMI1[Mc_AN][h_AN][k];

      if (Mi_AN<=Matomnum) Mi_AN2 = Mi_AN;
      else                 Mi_AN2 = Matomnum + 1; 

      if (0<=kl && VPS_j_dependency[wakg]==1){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);  sumy0 = Complex(0.0,0.0);  sumz0 = Complex(0.0,0.0);
	    sumx1 = Complex(0.0,0.0);  sumy1 = Complex(0.0,0.0);  sumz1 = Complex(0.0,0.0);
	    sumx2 = Complex(0.0,0.0);  sumy2 = Complex(0.0,0.0);  sumz2 = Complex(0.0,0.0);

	    dHCH_SO( &sumx0.r,&sumx0.i, &sumy0.r,&sumy0.i, &sumz0.r,&sumz0.i, 
                     &sumx1.r,&sumx1.i, &sumy1.r,&sumy1.i, &sumz1.r,&sumz1.i, 
                     &sumx2.r,&sumx2.i, &sumy2.r,&sumy2.i, &sumz2.r,&sumz2.i,
                     -1.0,
                     Mj_AN,  k,  n,
                     Mi_AN2, kl, m,
                     kg, wakg,
                     penalty_value,
                     OLP1 );

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    }     

  }

  /* if (h_AN!=0 && where_flag==1) */

  else {

    /****************************************************
                           dH*ep*H
    ****************************************************/

    kg = natn[Gc_AN][0];
    wakg = WhatSpecies[kg];
    kan = Spe_Total_VPS_Pro[wakg];
    kl1 = RMI1[Mc_AN][0][h_AN];
    kl2 = RMI1[Mc_AN][0][q_AN];

    /****************************************************
                   l-dependent non-local part
    ****************************************************/

    if (VPS_j_dependency[wakg]==0){

      if (kg==Core_Hole_Atom){

	/* calculate the multiplication */

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx[0] = 0.0; sumx[1] = 0.0;
	    sumy[0] = 0.0; sumy[1] = 0.0;
	    sumz[0] = 0.0; sumz[1] = 0.0;

	    for (i=0; i<Spe_Total_CNO[wakg]; i++){

	      sumx[0] -= penalty[0][i]*OLP1[1][Matomnum+1][kl1][m][i]*OLP1[0][Matomnum+1][kl2][n][i];
	      sumy[0] -= penalty[0][i]*OLP1[2][Matomnum+1][kl1][m][i]*OLP1[0][Matomnum+1][kl2][n][i];
	      sumz[0] -= penalty[0][i]*OLP1[3][Matomnum+1][kl1][m][i]*OLP1[0][Matomnum+1][kl2][n][i];

	      sumx[1] -= penalty[1][i]*OLP1[1][Matomnum+1][kl1][m][i]*OLP1[0][Matomnum+1][kl2][n][i];
	      sumy[1] -= penalty[1][i]*OLP1[2][Matomnum+1][kl1][m][i]*OLP1[0][Matomnum+1][kl2][n][i];
	      sumz[1] -= penalty[1][i]*OLP1[3][Matomnum+1][kl1][m][i]*OLP1[0][Matomnum+1][kl2][n][i];
	    }

	    Hx[0][m][n].r = sumx[0];
	    Hy[0][m][n].r = sumy[0];
	    Hz[0][m][n].r = sumz[0];

	    Hx[1][m][n].r = sumx[1];
	    Hy[1][m][n].r = sumy[1];
	    Hz[1][m][n].r = sumz[1];

	    Hx[2][m][n].r = 0.0;         
	    Hy[2][m][n].r = 0.0;         
	    Hz[2][m][n].r = 0.0;         

	    Hx[0][m][n].i = 0.0;
	    Hy[0][m][n].i = 0.0;
	    Hz[0][m][n].i = 0.0;

	    Hx[1][m][n].i = 0.0;
	    Hy[1][m][n].i = 0.0;
	    Hz[1][m][n].i = 0.0;

	    Hx[2][m][n].i = 0.0;
	    Hy[2][m][n].i = 0.0;
	    Hz[2][m][n].i = 0.0;

	  }
	}
      }
    }

    /****************************************************
                 j-dependent non-local part
    ****************************************************/

    else if ( VPS_j_dependency[wakg]==1 ){

      for (m=0; m<ian; m++){
	for (n=0; n<jan; n++){

	  sumx0 = Complex(0.0,0.0);  sumy0 = Complex(0.0,0.0);  sumz0 = Complex(0.0,0.0);
	  sumx1 = Complex(0.0,0.0);  sumy1 = Complex(0.0,0.0);  sumz1 = Complex(0.0,0.0);
	  sumx2 = Complex(0.0,0.0);  sumy2 = Complex(0.0,0.0);  sumz2 = Complex(0.0,0.0);

          /* 2 */

	  dHCH_SO( &sumx0.r,&sumx0.i, &sumy0.r,&sumy0.i, &sumz0.r,&sumz0.i, 
                   &sumx1.r,&sumx1.i, &sumy1.r,&sumy1.i, &sumz1.r,&sumz1.i, 
                   &sumx2.r,&sumx2.i, &sumy2.r,&sumy2.i, &sumz2.r,&sumz2.i, 
		   1.0,
		   Matomnum+1, kl1,m,
		   Matomnum+1, kl2,n,
		   kg, wakg,
		   -penalty_value,
		   OLP1 );

	  Hx[0][m][n].r = sumx0.r;     /* up-up */
	  Hy[0][m][n].r = sumy0.r;     /* up-up */
	  Hz[0][m][n].r = sumz0.r;     /* up-up */

	  Hx[1][m][n].r = sumx1.r;     /* dn-dn */
	  Hy[1][m][n].r = sumy1.r;     /* dn-dn */
	  Hz[1][m][n].r = sumz1.r;     /* dn-dn */

	  Hx[2][m][n].r = sumx2.r;     /* up-dn */
	  Hy[2][m][n].r = sumy2.r;     /* up-dn */
	  Hz[2][m][n].r = sumz2.r;     /* up-dn */

	  Hx[0][m][n].i = sumx0.i;     /* up-up */
	  Hy[0][m][n].i = sumy0.i;     /* up-up */
	  Hz[0][m][n].i = sumz0.i;     /* up-up */

	  Hx[1][m][n].i = sumx1.i;     /* dn-dn */
	  Hy[1][m][n].i = sumy1.i;     /* dn-dn */
	  Hz[1][m][n].i = sumz1.i;     /* dn-dn */

	  Hx[2][m][n].i = sumx2.i;     /* up-dn */
	  Hy[2][m][n].i = sumy2.i;     /* up-dn */
	  Hz[2][m][n].i = sumz2.i;     /* up-dn */

	}
      }
    }

    /****************************************************
                           H*ep*dH
    ****************************************************/

    if (q_AN!=0) {

      kg = natn[Gc_AN][0];
      wakg = WhatSpecies[kg];
      kan = Spe_Total_VPS_Pro[wakg];
      kl1 = RMI1[Mc_AN][0][h_AN];
      kl2 = RMI1[Mc_AN][0][q_AN];

      /****************************************************
                     l-dependent non-local part
      ****************************************************/

      if (VPS_j_dependency[wakg]==0){

        if (kg==Core_Hole_Atom){

  	  /* calculate the multiplication */

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      sumx[0] = 0.0; sumx[1] = 0.0;
	      sumy[0] = 0.0; sumy[1] = 0.0;
	      sumz[0] = 0.0; sumz[1] = 0.0;

  	      for (i=0; i<Spe_Total_CNO[wakg]; i++){

		sumx[0] -= penalty[0][i]*OLP1[0][Matomnum+1][kl1][m][i]*OLP1[1][Matomnum+1][kl2][n][i];
		sumy[0] -= penalty[0][i]*OLP1[0][Matomnum+1][kl1][m][i]*OLP1[2][Matomnum+1][kl2][n][i];
		sumz[0] -= penalty[0][i]*OLP1[0][Matomnum+1][kl1][m][i]*OLP1[3][Matomnum+1][kl2][n][i];

		sumx[1] -= penalty[1][i]*OLP1[0][Matomnum+1][kl1][m][i]*OLP1[1][Matomnum+1][kl2][n][i];
		sumy[1] -= penalty[1][i]*OLP1[0][Matomnum+1][kl1][m][i]*OLP1[2][Matomnum+1][kl2][n][i];
		sumz[1] -= penalty[1][i]*OLP1[0][Matomnum+1][kl1][m][i]*OLP1[3][Matomnum+1][kl2][n][i];
	      }

	      Hx[0][m][n].r += sumx[0];
	      Hy[0][m][n].r += sumy[0];
	      Hz[0][m][n].r += sumz[0];

	      Hx[1][m][n].r += sumx[1];
	      Hy[1][m][n].r += sumy[1];
	      Hz[1][m][n].r += sumz[1];
	    }
	  }

	}
      }

      /****************************************************
                    j-dependent non-local part
      ****************************************************/

      else if ( VPS_j_dependency[wakg]==1 ){

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

	    sumx0 = Complex(0.0,0.0);  sumy0 = Complex(0.0,0.0);  sumz0 = Complex(0.0,0.0);
	    sumx1 = Complex(0.0,0.0);  sumy1 = Complex(0.0,0.0);  sumz1 = Complex(0.0,0.0);
	    sumx2 = Complex(0.0,0.0);  sumy2 = Complex(0.0,0.0);  sumz2 = Complex(0.0,0.0);

            /* 4 */

	    dHCH_SO( &sumx0.r,&sumx0.i, &sumy0.r,&sumy0.i, &sumz0.r,&sumz0.i, 
                     &sumx1.r,&sumx1.i, &sumy1.r,&sumy1.i, &sumz1.r,&sumz1.i, 
                     &sumx2.r,&sumx2.i, &sumy2.r,&sumy2.i, &sumz2.r,&sumz2.i, 
		     -1.0,
		     Matomnum+1, kl2,n,
		     Matomnum+1, kl1,m,
		     kg, wakg,
		     -penalty_value,
		     OLP1 );

	    Hx[0][m][n].r += sumx0.r;     /* up-up */
	    Hy[0][m][n].r += sumy0.r;     /* up-up */
	    Hz[0][m][n].r += sumz0.r;     /* up-up */

	    Hx[1][m][n].r += sumx1.r;     /* dn-dn */
	    Hy[1][m][n].r += sumy1.r;     /* dn-dn */
	    Hz[1][m][n].r += sumz1.r;     /* dn-dn */

	    Hx[2][m][n].r += sumx2.r;     /* up-dn */
	    Hy[2][m][n].r += sumy2.r;     /* up-dn */
	    Hz[2][m][n].r += sumz2.r;     /* up-dn */

	    Hx[0][m][n].i += sumx0.i;     /* up-up */
	    Hy[0][m][n].i += sumy0.i;     /* up-up */
	    Hz[0][m][n].i += sumz0.i;     /* up-up */

	    Hx[1][m][n].i += sumx1.i;     /* dn-dn */
	    Hy[1][m][n].i += sumy1.i;     /* dn-dn */
	    Hz[1][m][n].i += sumz1.i;     /* dn-dn */

	    Hx[2][m][n].i += sumx2.i;     /* up-dn */
	    Hy[2][m][n].i += sumy2.i;     /* up-dn */
	    Hz[2][m][n].i += sumz2.i;     /* up-dn */

	  }
	}
      }

    }

  } /* else */

  /****************************************************
               contribution by dampingF 
  ****************************************************/

  /* Qij * dH/dx  */

  for (so=0; so<3; so++){
    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){

        Hx[so][m][n].r = dmp*Hx[so][m][n].r;
        Hy[so][m][n].r = dmp*Hy[so][m][n].r;
        Hz[so][m][n].r = dmp*Hz[so][m][n].r;

        Hx[so][m][n].i = dmp*Hx[so][m][n].i;
        Hy[so][m][n].i = dmp*Hy[so][m][n].i;
        Hz[so][m][n].i = dmp*Hz[so][m][n].i;
      }
    }
  }

  /* dQij/dx * H */

  if ( (h_AN==0 && q_AN!=0) || (h_AN!=0 && q_AN==0) ){

    if      (h_AN==0)   kl = q_AN;
    else if (q_AN==0)   kl = h_AN;

    if      (SpinP_switch==0) somax = 0;
    else if (SpinP_switch==1) somax = 1;
    else if (SpinP_switch==3) somax = 2;

    r = Dis[Gc_AN][kl];

    if (rcut<=r) {
      deri_dmp = 0.0;
      tmp = 0.0;
    }
    else {
      deri_dmp = deri_dampingF(rcut,r);
      tmp = deri_dmp/dmp;
    }

    x0 = Gxyz[ig][1] + atv[Rni][1];
    y0 = Gxyz[ig][2] + atv[Rni][2];
    z0 = Gxyz[ig][3] + atv[Rni][3];

    x1 = Gxyz[jg][1] + atv[Rnj][1];
    y1 = Gxyz[jg][2] + atv[Rnj][2];
    z1 = Gxyz[jg][3] + atv[Rnj][3];

    /* for empty atoms or finite elemens basis */
    if (r<1.0e-10) r = 1.0e-10;

    if (h_AN==0 && q_AN!=0){
      dx = tmp*(x0-x1)/r;
      dy = tmp*(y0-y1)/r;
      dz = tmp*(z0-z1)/r;
    }

    else if (h_AN!=0 && q_AN==0){
      dx = tmp*(x1-x0)/r;
      dy = tmp*(y1-y0)/r;
      dz = tmp*(z1-z0)/r;
    }

    if (SpinP_switch==0 || SpinP_switch==1){

      if (h_AN==0){ 
        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r += HCH[so][Mc_AN][kl][m][n]*dx;
	      Hy[so][m][n].r += HCH[so][Mc_AN][kl][m][n]*dy;
	      Hz[so][m][n].r += HCH[so][Mc_AN][kl][m][n]*dz;
	    }
	  }
        }
      }

      else if (q_AN==0){ 
        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r += HCH[so][Mc_AN][kl][n][m]*dx;
	      Hy[so][m][n].r += HCH[so][Mc_AN][kl][n][m]*dy;
	      Hz[so][m][n].r += HCH[so][Mc_AN][kl][n][m]*dz;
	    }
	  }
        }
      }
    }

    else if (SpinP_switch==3){

      if (h_AN==0){ 
        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r +=  HCH[so][Mc_AN][kl][m][n]*dx;
	      Hy[so][m][n].r +=  HCH[so][Mc_AN][kl][m][n]*dy;
	      Hz[so][m][n].r +=  HCH[so][Mc_AN][kl][m][n]*dz;
	    }
	  }
        }
      }

      else if (q_AN==0){ 
        for (so=0; so<=somax; so++){
	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      Hx[so][m][n].r +=  HCH[so][Mc_AN][kl][n][m]*dx;
	      Hy[so][m][n].r +=  HCH[so][Mc_AN][kl][n][m]*dy;
	      Hz[so][m][n].r +=  HCH[so][Mc_AN][kl][n][m]*dz;
	    }
	  }
        }
      }

      if (SO_switch==1){

        if (h_AN==0){ 
	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		Hx[so][m][n].i += iHCH[so][Mc_AN][kl][m][n]*dx;
		Hy[so][m][n].i += iHCH[so][Mc_AN][kl][m][n]*dy;
		Hz[so][m][n].i += iHCH[so][Mc_AN][kl][m][n]*dz;
	      }
	    }
	  }
	}

        else if (q_AN==0){ 
	  for (so=0; so<=somax; so++){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		Hx[so][m][n].i += iHCH[so][Mc_AN][kl][n][m]*dx;
		Hy[so][m][n].i += iHCH[so][Mc_AN][kl][n][m]*dy;
		Hz[so][m][n].i += iHCH[so][Mc_AN][kl][n][m]*dz;
	      }
	    }
	  }
	}

      }
    }
  }

  /* freeing of array */

  for (i=0; i<2; i++){
    free(penalty[i]);
  }
  free(penalty);

}





void dHCH_SO(double *sumx0r, double *sumx0i, double *sumy0r, double *sumy0i, double *sumz0r, double *sumz0i,
             double *sumx1r, double *sumx1i, double *sumy1r, double *sumy1i, double *sumz1r, double *sumz1i,
             double *sumx2r, double *sumx2i, double *sumy2r, double *sumy2i, double *sumz2r, double *sumz2i,
             double fugou,
	     int Mc_AN, int k,  int m,
	     int Mj_AN, int kl, int n,
             int kg, int wakg,
             double penalty_value, 
	     double *****OLP1)
{
  int L,L2,l,mul,apply_flag;
  double d12_m12,d12_p12;
  double d32_m12,d32_p12,d32_m32,d32_p32;
  double d52_m12,d52_p12,d52_m32,d52_p32,d52_m52,d52_p52;
  double d72_m12,d72_p12,d72_m32,d72_p32,d72_m52,d72_p52,d72_m72,d72_p72;

  if (kg!=Core_Hole_Atom) return;

  /****************************************************
                 set penalty coefficients
  ****************************************************/

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
	    *sumx2r += fugou*(
               d32_m12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
 	      -d32_p12*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	    *sumy2r += fugou*(
               d32_m12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
 	      -d32_p12*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	    *sumz2r += fugou*(
               d32_m12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
 	      -d32_p12*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	    /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */ 
	    *sumx2i += fugou*(
	      -d32_m12*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	      +d32_p12*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1] ); 

	    *sumx2i += fugou*(
	      -d32_m12*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	      +d32_p12*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1] ); 

	    *sumx2i += fugou*(
	      -d32_m12*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	      +d32_p12*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1] ); 

	    /* real contribution of l-1/2 for to off-diagonal up-down matrix */ 
	    *sumx2r -= fugou*(
 	       d12_m12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
	      -d12_p12*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	    *sumy2r -= fugou*(
 	       d12_m12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
	      -d12_p12*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	    *sumz2r -= fugou*(
 	       d12_m12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
	      -d12_p12*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	    /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */ 
	    *sumx2i -= fugou*(
	      -d12_m12*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	      +d12_p12*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1] );

	    *sumy2i -= fugou*(
	      -d12_m12*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	      +d12_p12*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1] );

	    *sumz2i -= fugou*(
	      -d12_m12*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	      +d12_p12*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1] );
	  }

	  /***************
                 d
	  ***************/ 

	  else if (L2==4){

	    /* real contribution of l+1/2 to off diagonal up-down matrix */ 

	    *sumx2r += fugou*(
	      -sqrt(3.0)*d52_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(3.0)*d52_m12*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d52_m32*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      -d52_p32*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      +d52_m32*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d52_p32*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2] );

	    *sumy2r += fugou*(
	      -sqrt(3.0)*d52_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(3.0)*d52_m12*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d52_m32*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      -d52_p32*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      +d52_m32*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d52_p32*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2] );

	    *sumz2r += fugou*(
	      -sqrt(3.0)*d52_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(3.0)*d52_m12*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d52_m32*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      -d52_p32*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      +d52_m32*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d52_p32*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2] );

	    /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

	    *sumx2i += fugou*(
	       sqrt(3.0)*d52_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(3.0)*d52_m12*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d52_m32*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d52_p32*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -d52_m32*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
	      +d52_p32*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2] );

	    *sumy2i += fugou*(
	       sqrt(3.0)*d52_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(3.0)*d52_m12*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d52_m32*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d52_p32*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -d52_m32*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
	      +d52_p32*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2] );

	    *sumz2i += fugou*(
	       sqrt(3.0)*d52_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(3.0)*d52_m12*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d52_m32*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d52_p32*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -d52_m32*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
	      +d52_p32*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2] );

	    /* real contribution of l-1/2 for to diagonal up-down matrix */ 

	    *sumx2r -= fugou*(
	      -sqrt(3.0)*d32_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(3.0)*d32_m12*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d32_m32*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      -d32_p32*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      +d32_m32*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d32_p32*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2] );

	    *sumy2r -= fugou*(
	      -sqrt(3.0)*d32_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(3.0)*d32_m12*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d32_m32*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      -d32_p32*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      +d32_m32*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d32_p32*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2] );

	    *sumz2r -= fugou*(
	      -sqrt(3.0)*d32_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(3.0)*d32_m12*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d32_m32*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      -d32_p32*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      +d32_m32*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d32_p32*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2] );

	    /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

	    *sumx2i -= fugou*(
 	       sqrt(3.0)*d32_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(3.0)*d32_m12*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d32_m32*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d32_p32*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -d32_m32*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
              +d32_p32*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2] );

	    *sumy2i -= fugou*(
 	       sqrt(3.0)*d32_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(3.0)*d32_m12*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d32_m32*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d32_p32*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -d32_m32*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
              +d32_p32*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2] );

	    *sumz2i -= fugou*(
 	       sqrt(3.0)*d32_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(3.0)*d32_m12*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L  ]
	      +d32_m32*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -d32_p32*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -d32_m32*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
              +d32_p32*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2] );
	  }

	  /***************
                         f
	  ***************/ 

	  else if (L2==6){

	    /* real contribution of l+1/2 to off diagonal up-down matrix */ 

	    *sumx2r += fugou*(
	      -sqrt(6.0)*d72_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	      +sqrt(6.0)*d72_m12*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ]
	      -sqrt(2.5)*d72_p32*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d72_m32*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d72_p32*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      +sqrt(2.5)*d72_m32*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(1.5)*d72_p52*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d72_m52*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d72_p52*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+6]
	      +sqrt(1.5)*d72_m52*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+4] );

	    *sumy2r += fugou*(
	      -sqrt(6.0)*d72_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	      +sqrt(6.0)*d72_m12*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ]
	      -sqrt(2.5)*d72_p32*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d72_m32*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d72_p32*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      +sqrt(2.5)*d72_m32*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(1.5)*d72_p52*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d72_m52*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d72_p52*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+6]
	      +sqrt(1.5)*d72_m52*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+4] );

	    *sumz2r += fugou*(
	      -sqrt(6.0)*d72_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	      +sqrt(6.0)*d72_m12*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ]
	      -sqrt(2.5)*d72_p32*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d72_m32*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d72_p32*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      +sqrt(2.5)*d72_m32*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(1.5)*d72_p52*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d72_m52*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d72_p52*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+6]
	      +sqrt(1.5)*d72_m52*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+4] );

	    /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

	    *sumx2i += fugou*(
	       sqrt(6.0)*d72_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(6.0)*d72_m12*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ]
	      +sqrt(2.5)*d72_p32*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(2.5)*d72_m32*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d72_p32*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d72_m32*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2]
	      +sqrt(1.5)*d72_p52*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+6]
	      -sqrt(1.5)*d72_m52*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d72_p52*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d72_m52*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+4] );

	    *sumy2i += fugou*(
	       sqrt(6.0)*d72_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(6.0)*d72_m12*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ]
	      +sqrt(2.5)*d72_p32*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(2.5)*d72_m32*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d72_p32*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d72_m32*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2]
	      +sqrt(1.5)*d72_p52*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+6]
	      -sqrt(1.5)*d72_m52*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d72_p52*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d72_m52*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+4] );

	    *sumz2i += fugou*(
	       sqrt(6.0)*d72_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(6.0)*d72_m12*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ]
	      +sqrt(2.5)*d72_p32*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(2.5)*d72_m32*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d72_p32*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d72_m32*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2]
	      +sqrt(1.5)*d72_p52*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+6]
	      -sqrt(1.5)*d72_m52*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d72_p52*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d72_m52*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+4] );

	    /* real contribution of l-1/2 for to off-diagonal up-down matrix */ 

	    *sumx2r -= fugou*( 
	      -sqrt(6.0)*d52_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	      +sqrt(6.0)*d52_m12*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ]
	      -sqrt(2.5)*d52_p32*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d52_m32*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d52_p32*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      +sqrt(2.5)*d52_m32*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(1.5)*d52_p52*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d52_m52*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d52_p52*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+6]
	      +sqrt(1.5)*d52_m52*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+4] );

	    *sumy2r -= fugou*( 
	      -sqrt(6.0)*d52_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	      +sqrt(6.0)*d52_m12*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ]
	      -sqrt(2.5)*d52_p32*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d52_m32*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d52_p32*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      +sqrt(2.5)*d52_m32*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(1.5)*d52_p52*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d52_m52*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d52_p52*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+6]
	      +sqrt(1.5)*d52_m52*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+4] );

	    *sumz2r -= fugou*( 
	      -sqrt(6.0)*d52_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	      +sqrt(6.0)*d52_m12*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ]
	      -sqrt(2.5)*d52_p32*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d52_m32*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d52_p32*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+4]
	      +sqrt(2.5)*d52_m32*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(1.5)*d52_p52*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d52_m52*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d52_p52*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+6]
	      +sqrt(1.5)*d52_m52*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+4] );

	    /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

	    *sumx2i -= fugou*(
	       sqrt(6.0)*d52_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(6.0)*d52_m12*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ]
	      +sqrt(2.5)*d52_p32*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(2.5)*d52_m32*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d52_p32*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d52_m32*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2]
	      +sqrt(1.5)*d52_p52*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+6]
	      -sqrt(1.5)*d52_m52*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d52_p52*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d52_m52*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+4] );

	    *sumy2i -= fugou*(
	       sqrt(6.0)*d52_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(6.0)*d52_m12*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ]
	      +sqrt(2.5)*d52_p32*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(2.5)*d52_m32*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d52_p32*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d52_m32*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2]
	      +sqrt(1.5)*d52_p52*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+6]
	      -sqrt(1.5)*d52_m52*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d52_p52*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d52_m52*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+4] );

	    *sumz2i -= fugou*(
	       sqrt(6.0)*d52_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+2]
	      -sqrt(6.0)*d52_m12*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L  ]
	      +sqrt(2.5)*d52_p32*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+4]
	      -sqrt(2.5)*d52_m32*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+1]
	      -sqrt(2.5)*d52_p32*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+3]
	      +sqrt(2.5)*d52_m32*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+2]
	      +sqrt(1.5)*d52_p52*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+6]
	      -sqrt(1.5)*d52_m52*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+3]
	      -sqrt(1.5)*d52_p52*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+5]
	      +sqrt(1.5)*d52_m52*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+4] );
	  }

	} /* if (SpinP_switch==3) */ 

	/****************************************************
            off-diagonal contribution on up-up and dn-dn
	****************************************************/

	/* p */ 

	if (L2==2){

	  /* contribution of l+1/2 for up spin */ 

	  *sumx0i += 0.5*fugou*(
	    +( d32_m12-3.0*d32_p32)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    +(-d32_m12+3.0*d32_p32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	  *sumy0i += 0.5*fugou*(
	    +( d32_m12-3.0*d32_p32)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    +(-d32_m12+3.0*d32_p32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	  *sumz0i += 0.5*fugou*(
	    +( d32_m12-3.0*d32_p32)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    +(-d32_m12+3.0*d32_p32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	  /* contribution of l+1/2 for down spin */ 

	  *sumx1i += 0.5*fugou*(
	    +(-d32_p12+3.0*d32_m32)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    +(+d32_p12-3.0*d32_m32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	  *sumy1i += 0.5*fugou*(
	    +(-d32_p12+3.0*d32_m32)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    +(+d32_p12-3.0*d32_m32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	  *sumz1i += 0.5*fugou*(
	    +(-d32_p12+3.0*d32_m32)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    +(+d32_p12-3.0*d32_m32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] ); 

	  /* contribution of l-1/2 for up spin */

	  *sumx0i += fugou*(
	     d12_m12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    -d12_m12*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] );

	  *sumy0i += fugou*(
	     d12_m12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    -d12_m12*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] );

	  *sumz0i += fugou*(
	     d12_m12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    -d12_m12*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] );

	  /* contribution of l-1/2 for down spin */ 

	  *sumx1i += fugou*(
	    -d12_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    +d12_p12*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] );

	  *sumy1i += fugou*(
	    -d12_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    +d12_p12*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] );

	  *sumz1i += fugou*(
	    -d12_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L+1]
	    +d12_p12*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L  ] );
	}

	/* d */ 

	else if (L2==4){

	  /* contribution of l+1/2 for up spin */ 

	  *sumx0i += 0.5*fugou*(
	    +(     d52_m32-5.0*d52_p52)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(    -d52_m32+5.0*d52_p52)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 2.0*d52_m12-4.0*d52_p32)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-2.0*d52_m12+4.0*d52_p32)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  *sumy0i += 0.5*fugou*(
	    +(     d52_m32-5.0*d52_p52)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(    -d52_m32+5.0*d52_p52)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 2.0*d52_m12-4.0*d52_p32)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-2.0*d52_m12+4.0*d52_p32)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  *sumz0i += 0.5*fugou*(
	    +(     d52_m32-5.0*d52_p52)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(    -d52_m32+5.0*d52_p52)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 2.0*d52_m12-4.0*d52_p32)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-2.0*d52_m12+4.0*d52_p32)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  /* contribution of l+1/2 for down spin */ 

	  *sumx1i += 0.5*fugou*(
	    -(     d52_p32-5.0*d52_m52)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    -(    -d52_p32+5.0*d52_m52)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 2.0*d52_p12-4.0*d52_m32)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    -(-2.0*d52_p12+4.0*d52_m32)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  *sumy1i += 0.5*fugou*(
	    -(     d52_p32-5.0*d52_m52)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    -(    -d52_p32+5.0*d52_m52)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 2.0*d52_p12-4.0*d52_m32)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    -(-2.0*d52_p12+4.0*d52_m32)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  *sumz1i += 0.5*fugou*(
	    -(     d52_p32-5.0*d52_m52)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    -(    -d52_p32+5.0*d52_m52)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 2.0*d52_p12-4.0*d52_m32)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    -(-2.0*d52_p12+4.0*d52_m32)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  /* contribution of l-1/2 for up spin */ 

	  *sumx0i += 0.5*fugou*(
	     (         4.0*d32_m32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(        -4.0*d32_m32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 3.0*d32_m12-d32_p32)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-3.0*d32_m12+d32_p32)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  *sumy0i += 0.5*fugou*(
	     (         4.0*d32_m32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(        -4.0*d32_m32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 3.0*d32_m12-d32_p32)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-3.0*d32_m12+d32_p32)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  *sumz0i += 0.5*fugou*(
	     (         4.0*d32_m32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(        -4.0*d32_m32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 3.0*d32_m12-d32_p32)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-3.0*d32_m12+d32_p32)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  /* contribution of l-1/2 for down spin */ 

	  *sumx1i += 0.5*fugou*(
	     (        -4.0*d32_p32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(        +4.0*d32_p32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 3.0*d32_p12-d32_m32)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
            -(-3.0*d32_p12+d32_m32)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  *sumy1i += 0.5*fugou*(
	     (        -4.0*d32_p32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(        +4.0*d32_p32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 3.0*d32_p12-d32_m32)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
            -(-3.0*d32_p12+d32_m32)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 

	  *sumz1i += 0.5*fugou*(
	     (        -4.0*d32_p32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(        +4.0*d32_p32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 3.0*d32_p12-d32_m32)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
            -(-3.0*d32_p12+d32_m32)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3] ); 
	}

	/* f */ 

	else if (L2==6){

	  /* contribution of l+1/2 for up spin */ 

	  *sumx0i += 0.5*fugou*(
	     ( 3.0*d72_m12-5.0*d72_p32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(-3.0*d72_m12+5.0*d72_p32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 2.0*d72_m32-6.0*d72_p52)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-2.0*d72_m32+6.0*d72_p52)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    +( 1.0*d72_m52-7.0*d72_p72)*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
            +(-1.0*d72_m52+7.0*d72_p72)*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  *sumy0i += 0.5*fugou*(
	     ( 3.0*d72_m12-5.0*d72_p32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(-3.0*d72_m12+5.0*d72_p32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 2.0*d72_m32-6.0*d72_p52)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-2.0*d72_m32+6.0*d72_p52)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    +( 1.0*d72_m52-7.0*d72_p72)*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
            +(-1.0*d72_m52+7.0*d72_p72)*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  *sumz0i += 0.5*fugou*(
	     ( 3.0*d72_m12-5.0*d72_p32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(-3.0*d72_m12+5.0*d72_p32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 2.0*d72_m32-6.0*d72_p52)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-2.0*d72_m32+6.0*d72_p52)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    +( 1.0*d72_m52-7.0*d72_p72)*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
            +(-1.0*d72_m52+7.0*d72_p72)*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  /* contribution of l+1/2 for down spin */ 

	  *sumx1i += 0.5*fugou*(
	    -( 3.0*d72_p12-5.0*d72_m32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    -(-3.0*d72_p12+5.0*d72_m32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 2.0*d72_p32-6.0*d72_m52)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    -(-2.0*d72_p32+6.0*d72_m52)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    -( 1.0*d72_p52-7.0*d72_m72)*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
	    -(-1.0*d72_p52+7.0*d72_m72)*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  *sumy1i += 0.5*fugou*(
	    -( 3.0*d72_p12-5.0*d72_m32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    -(-3.0*d72_p12+5.0*d72_m32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 2.0*d72_p32-6.0*d72_m52)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    -(-2.0*d72_p32+6.0*d72_m52)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    -( 1.0*d72_p52-7.0*d72_m72)*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
	    -(-1.0*d72_p52+7.0*d72_m72)*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  *sumz1i += 0.5*fugou*(
	    -( 3.0*d72_p12-5.0*d72_m32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    -(-3.0*d72_p12+5.0*d72_m32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 2.0*d72_p32-6.0*d72_m52)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    -(-2.0*d72_p32+6.0*d72_m52)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    -( 1.0*d72_p52-7.0*d72_m72)*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
	    -(-1.0*d72_p52+7.0*d72_m72)*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  /* contribution of l-1/2 for up spin */ 

	  *sumx0i += 0.5*fugou*(
	     ( 4.0*d52_m12-2.0*d52_p32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(-4.0*d52_m12+2.0*d52_p32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 5.0*d52_m32-1.0*d52_p52)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-5.0*d52_m32+1.0*d52_p52)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    +(+6.0*d52_m52            )*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
	    +(-6.0*d52_m52            )*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  *sumy0i += 0.5*fugou*(
	     ( 4.0*d52_m12-2.0*d52_p32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(-4.0*d52_m12+2.0*d52_p32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 5.0*d52_m32-1.0*d52_p52)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-5.0*d52_m32+1.0*d52_p52)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    +(+6.0*d52_m52            )*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
	    +(-6.0*d52_m52            )*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  *sumz0i += 0.5*fugou*(
	     ( 4.0*d52_m12-2.0*d52_p32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    +(-4.0*d52_m12+2.0*d52_p32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    +( 5.0*d52_m32-1.0*d52_p52)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    +(-5.0*d52_m32+1.0*d52_p52)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    +(+6.0*d52_m52            )*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
	    +(-6.0*d52_m52            )*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  /* contribution of l-1/2 for down spin */ 

	  *sumx1i += 0.5*fugou*(
	    -( 4.0*d52_p12-2.0*d52_m32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    -(-4.0*d52_p12+2.0*d52_m32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 5.0*d52_p32-1.0*d52_m52)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    -(-5.0*d52_p32+1.0*d52_m52)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    -(+6.0*d52_p52            )*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
	    -(-6.0*d52_p52            )*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  *sumy1i += 0.5*fugou*(
	    -( 4.0*d52_p12-2.0*d52_m32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    -(-4.0*d52_p12+2.0*d52_m32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 5.0*d52_p32-1.0*d52_m52)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    -(-5.0*d52_p32+1.0*d52_m52)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    -(+6.0*d52_p52            )*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
	    -(-6.0*d52_p52            )*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );

	  *sumz1i += 0.5*fugou*(
	    -( 4.0*d52_p12-2.0*d52_m32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+2]
	    -(-4.0*d52_p12+2.0*d52_m32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+1]
	    -( 5.0*d52_p32-1.0*d52_m52)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+4]
	    -(-5.0*d52_p32+1.0*d52_m52)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+3]
	    -(+6.0*d52_p52            )*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+6]
	    -(-6.0*d52_p52            )*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+5] );
	}

	/****************************************************
             diagonal contribution on up-up and dn-dn
	****************************************************/

	/* s */

	if (L2==0){

	  /* VNL for j=l+1/2 */

	  *sumx0r += d12_p12*OLP1[1][Mc_AN][k][m][L]*OLP1[0][Mj_AN][kl][n][L];
	  *sumy0r += d12_p12*OLP1[2][Mc_AN][k][m][L]*OLP1[0][Mj_AN][kl][n][L];
	  *sumz0r += d12_p12*OLP1[3][Mc_AN][k][m][L]*OLP1[0][Mj_AN][kl][n][L];

	  *sumx1r += d12_m12*OLP1[1][Mc_AN][k][m][L]*OLP1[0][Mj_AN][kl][n][L];
	  *sumy1r += d12_m12*OLP1[2][Mc_AN][k][m][L]*OLP1[0][Mj_AN][kl][n][L];
	  *sumz1r += d12_m12*OLP1[3][Mc_AN][k][m][L]*OLP1[0][Mj_AN][kl][n][L];

	  /* note that VNL for j=l-1/2 is zero */
	}

	/* p */

	else if (L2==2){

	  /* VNL for j=l+1/2 */
	  *sumx0r += 0.5*(d32_m12+3.0*d32_p32)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx0r += 0.5*(d32_m12+3.0*d32_p32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx0r += 0.5*(        4.0*d32_p12)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumy0r += 0.5*(d32_m12+3.0*d32_p32)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy0r += 0.5*(d32_m12+3.0*d32_p32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy0r += 0.5*(        4.0*d32_p12)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumz0r += 0.5*(d32_m12+3.0*d32_p32)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz0r += 0.5*(d32_m12+3.0*d32_p32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz0r += 0.5*(        4.0*d32_p12)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumx1r += 0.5*(d32_p12+3.0*d32_m32)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx1r += 0.5*(d32_p12+3.0*d32_m32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx1r += 0.5*(        4.0*d32_m12)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumy1r += 0.5*(d32_p12+3.0*d32_m32)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy1r += 0.5*(d32_p12+3.0*d32_m32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy1r += 0.5*(        4.0*d32_m12)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumz1r += 0.5*(d32_p12+3.0*d32_m32)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz1r += 0.5*(d32_p12+3.0*d32_m32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz1r += 0.5*(        4.0*d32_m12)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  /* VNL for j=l-1/2 */
	  *sumx0r += d12_m12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx0r += d12_m12*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx0r += d12_p12*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumy0r += d12_m12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy0r += d12_m12*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy0r += d12_p12*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumz0r += d12_m12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz0r += d12_m12*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz0r += d12_p12*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumx1r += d12_p12*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx1r += d12_p12*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx1r += d12_m12*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumy1r += d12_p12*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy1r += d12_p12*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy1r += d12_m12*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];

	  *sumz1r += d12_p12*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz1r += d12_p12*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz1r += d12_m12*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	}

	/* d */

	else if (L2==4){

	  /* VNL for j=l+1/2 */
	  *sumx0r += 0.5*(            6.0*d52_p12)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx0r += 0.5*(1.0*d52_m32+5.0*d52_p52)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx0r += 0.5*(1.0*d52_m32+5.0*d52_p52)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumx0r += 0.5*(2.0*d52_m12+4.0*d52_p32)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumx0r += 0.5*(2.0*d52_m12+4.0*d52_p32)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumy0r += 0.5*(            6.0*d52_p12)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy0r += 0.5*(1.0*d52_m32+5.0*d52_p52)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy0r += 0.5*(1.0*d52_m32+5.0*d52_p52)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumy0r += 0.5*(2.0*d52_m12+4.0*d52_p32)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumy0r += 0.5*(2.0*d52_m12+4.0*d52_p32)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumz0r += 0.5*(            6.0*d52_p12)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz0r += 0.5*(1.0*d52_m32+5.0*d52_p52)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz0r += 0.5*(1.0*d52_m32+5.0*d52_p52)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumz0r += 0.5*(2.0*d52_m12+4.0*d52_p32)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumz0r += 0.5*(2.0*d52_m12+4.0*d52_p32)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumx1r += 0.5*(            6.0*d52_m12)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx1r += 0.5*(1.0*d52_p32+5.0*d52_m52)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx1r += 0.5*(1.0*d52_p32+5.0*d52_m52)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumx1r += 0.5*(2.0*d52_p12+4.0*d52_m32)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumx1r += 0.5*(2.0*d52_p12+4.0*d52_m32)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumy1r += 0.5*(            6.0*d52_m12)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy1r += 0.5*(1.0*d52_p32+5.0*d52_m52)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy1r += 0.5*(1.0*d52_p32+5.0*d52_m52)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumy1r += 0.5*(2.0*d52_p12+4.0*d52_m32)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumy1r += 0.5*(2.0*d52_p12+4.0*d52_m32)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumz1r += 0.5*(            6.0*d52_m12)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz1r += 0.5*(1.0*d52_p32+5.0*d52_m52)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz1r += 0.5*(1.0*d52_p32+5.0*d52_m52)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumz1r += 0.5*(2.0*d52_p12+4.0*d52_m32)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumz1r += 0.5*(2.0*d52_p12+4.0*d52_m32)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  /* VNL for j=l-1/2 */
	  *sumx0r += 0.5*(            4.0*d32_p12)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx0r += 0.5*(            4.0*d32_m32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx0r += 0.5*(            4.0*d32_m32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumx0r += 0.5*(3.0*d32_m12+1.0*d32_p32)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumx0r += 0.5*(3.0*d32_m12+1.0*d32_p32)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumy0r += 0.5*(            4.0*d32_p12)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy0r += 0.5*(            4.0*d32_m32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy0r += 0.5*(            4.0*d32_m32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumy0r += 0.5*(3.0*d32_m12+1.0*d32_p32)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumy0r += 0.5*(3.0*d32_m12+1.0*d32_p32)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumz0r += 0.5*(            4.0*d32_p12)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz0r += 0.5*(            4.0*d32_m32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz0r += 0.5*(            4.0*d32_m32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumz0r += 0.5*(3.0*d32_m12+1.0*d32_p32)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumz0r += 0.5*(3.0*d32_m12+1.0*d32_p32)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumx1r += 0.5*(            4.0*d32_m12)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx1r += 0.5*(            4.0*d32_p32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx1r += 0.5*(            4.0*d32_p32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumx1r += 0.5*(3.0*d32_p12+1.0*d32_m32)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumx1r += 0.5*(3.0*d32_p12+1.0*d32_m32)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumy1r += 0.5*(            4.0*d32_m12)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy1r += 0.5*(            4.0*d32_p32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy1r += 0.5*(            4.0*d32_p32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumy1r += 0.5*(3.0*d32_p12+1.0*d32_m32)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumy1r += 0.5*(3.0*d32_p12+1.0*d32_m32)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];

	  *sumz1r += 0.5*(            4.0*d32_m12)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz1r += 0.5*(            4.0*d32_p32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz1r += 0.5*(            4.0*d32_p32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumz1r += 0.5*(3.0*d32_p12+1.0*d32_m32)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumz1r += 0.5*(3.0*d32_p12+1.0*d32_m32)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	}

	/* f */

	else if (L2==6){

	  /* VNL for j=l+1/2 */
	  *sumx0r += 0.5*(            8.0*d72_p12)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx0r += 0.5*(3.0*d72_m12+5.0*d72_p32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx0r += 0.5*(3.0*d72_m12+5.0*d72_p32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumx0r += 0.5*(2.0*d72_m32+6.0*d72_p52)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumx0r += 0.5*(2.0*d72_m32+6.0*d72_p52)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumx0r += 0.5*(1.0*d72_m52+7.0*d72_p72)*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumx0r += 0.5*(1.0*d72_m52+7.0*d72_p72)*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumy0r += 0.5*(            8.0*d72_p12)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy0r += 0.5*(3.0*d72_m12+5.0*d72_p32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy0r += 0.5*(3.0*d72_m12+5.0*d72_p32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumy0r += 0.5*(2.0*d72_m32+6.0*d72_p52)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumy0r += 0.5*(2.0*d72_m32+6.0*d72_p52)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumy0r += 0.5*(1.0*d72_m52+7.0*d72_p72)*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumy0r += 0.5*(1.0*d72_m52+7.0*d72_p72)*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumz0r += 0.5*(            8.0*d72_p12)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz0r += 0.5*(3.0*d72_m12+5.0*d72_p32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz0r += 0.5*(3.0*d72_m12+5.0*d72_p32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumz0r += 0.5*(2.0*d72_m32+6.0*d72_p52)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumz0r += 0.5*(2.0*d72_m32+6.0*d72_p52)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumz0r += 0.5*(1.0*d72_m52+7.0*d72_p72)*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumz0r += 0.5*(1.0*d72_m52+7.0*d72_p72)*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumx1r += 0.5*(            8.0*d72_m12)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx1r += 0.5*(3.0*d72_p12+5.0*d72_m32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx1r += 0.5*(3.0*d72_p12+5.0*d72_m32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumx1r += 0.5*(2.0*d72_p32+6.0*d72_m52)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumx1r += 0.5*(2.0*d72_p32+6.0*d72_m52)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumx1r += 0.5*(1.0*d72_p52+7.0*d72_m72)*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumx1r += 0.5*(1.0*d72_p52+7.0*d72_m72)*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumy1r += 0.5*(            8.0*d72_m12)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy1r += 0.5*(3.0*d72_p12+5.0*d72_m32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy1r += 0.5*(3.0*d72_p12+5.0*d72_m32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumy1r += 0.5*(2.0*d72_p32+6.0*d72_m52)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumy1r += 0.5*(2.0*d72_p32+6.0*d72_m52)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumy1r += 0.5*(1.0*d72_p52+7.0*d72_m72)*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumy1r += 0.5*(1.0*d72_p52+7.0*d72_m72)*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumz1r += 0.5*(            8.0*d72_m12)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz1r += 0.5*(3.0*d72_p12+5.0*d72_m32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz1r += 0.5*(3.0*d72_p12+5.0*d72_m32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumz1r += 0.5*(2.0*d72_p32+6.0*d72_m52)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumz1r += 0.5*(2.0*d72_p32+6.0*d72_m52)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumz1r += 0.5*(1.0*d72_p52+7.0*d72_m72)*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumz1r += 0.5*(1.0*d72_p52+7.0*d72_m72)*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  /* VNL for j=l-1/2 */
	  *sumx0r += 0.5*(            6.0*d52_p12)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx0r += 0.5*(4.0*d52_m12+2.0*d52_p32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx0r += 0.5*(4.0*d52_m12+2.0*d52_p32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumx0r += 0.5*(5.0*d52_m32+1.0*d52_p52)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumx0r += 0.5*(5.0*d52_m32+1.0*d52_p52)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumx0r += 0.5*(6.0*d52_m52            )*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumx0r += 0.5*(6.0*d52_m52            )*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumy0r += 0.5*(            6.0*d52_p12)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy0r += 0.5*(4.0*d52_m12+2.0*d52_p32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy0r += 0.5*(4.0*d52_m12+2.0*d52_p32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumy0r += 0.5*(5.0*d52_m32+1.0*d52_p52)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumy0r += 0.5*(5.0*d52_m32+1.0*d52_p52)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumy0r += 0.5*(6.0*d52_m52            )*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumy0r += 0.5*(6.0*d52_m52            )*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumz0r += 0.5*(            6.0*d52_p12)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz0r += 0.5*(4.0*d52_m12+2.0*d52_p32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz0r += 0.5*(4.0*d52_m12+2.0*d52_p32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumz0r += 0.5*(5.0*d52_m32+1.0*d52_p52)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumz0r += 0.5*(5.0*d52_m32+1.0*d52_p52)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumz0r += 0.5*(6.0*d52_m52            )*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumz0r += 0.5*(6.0*d52_m52            )*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumx1r += 0.5*(            6.0*d52_m12)*OLP1[1][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumx1r += 0.5*(4.0*d52_p12+2.0*d52_m32)*OLP1[1][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumx1r += 0.5*(4.0*d52_p12+2.0*d52_m32)*OLP1[1][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumx1r += 0.5*(5.0*d52_p32+1.0*d52_m52)*OLP1[1][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumx1r += 0.5*(5.0*d52_p32+1.0*d52_m52)*OLP1[1][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumx1r += 0.5*(6.0*d52_p52            )*OLP1[1][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumx1r += 0.5*(6.0*d52_p52            )*OLP1[1][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumy1r += 0.5*(            6.0*d52_m12)*OLP1[2][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumy1r += 0.5*(4.0*d52_p12+2.0*d52_m32)*OLP1[2][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumy1r += 0.5*(4.0*d52_p12+2.0*d52_m32)*OLP1[2][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumy1r += 0.5*(5.0*d52_p32+1.0*d52_m52)*OLP1[2][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumy1r += 0.5*(5.0*d52_p32+1.0*d52_m52)*OLP1[2][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumy1r += 0.5*(6.0*d52_p52            )*OLP1[2][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumy1r += 0.5*(6.0*d52_p52            )*OLP1[2][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];

	  *sumz1r += 0.5*(            6.0*d52_m12)*OLP1[3][Mc_AN][k][m][L  ]*OLP1[0][Mj_AN][kl][n][L  ];
	  *sumz1r += 0.5*(4.0*d52_p12+2.0*d52_m32)*OLP1[3][Mc_AN][k][m][L+1]*OLP1[0][Mj_AN][kl][n][L+1];
	  *sumz1r += 0.5*(4.0*d52_p12+2.0*d52_m32)*OLP1[3][Mc_AN][k][m][L+2]*OLP1[0][Mj_AN][kl][n][L+2];
	  *sumz1r += 0.5*(5.0*d52_p32+1.0*d52_m52)*OLP1[3][Mc_AN][k][m][L+3]*OLP1[0][Mj_AN][kl][n][L+3];
	  *sumz1r += 0.5*(5.0*d52_p32+1.0*d52_m52)*OLP1[3][Mc_AN][k][m][L+4]*OLP1[0][Mj_AN][kl][n][L+4];
	  *sumz1r += 0.5*(6.0*d52_p52            )*OLP1[3][Mc_AN][k][m][L+5]*OLP1[0][Mj_AN][kl][n][L+5];
	  *sumz1r += 0.5*(6.0*d52_p52            )*OLP1[3][Mc_AN][k][m][L+6]*OLP1[0][Mj_AN][kl][n][L+6];
	}

      } /* if (apply_flag==1) */ 

      /* increment of L */
 
      L += 2*l + 1; 

    } /* mul */
  } /* l */

}




void MPI_OLP(double *****OLP1)
{
  int i,j,h_AN,Gh_AN,Hwan,n;
  int tno1,tno2,Mc_AN,Gc_AN,Cwan;
  int num,k,size1,size2;
  double *tmp_array;
  double *tmp_array2;
  int *Snd_S_Size,*Rcv_S_Size;
  int numprocs,myid,ID,IDS,IDR,tag=999;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  Snd_S_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_S_Size = (int*)malloc(sizeof(int)*numprocs);

  /******************************************************************
   MPI

   OLP1[1], OLP1[2], and OLP1[3]

   note:

   OLP is used in DC and GDC method, where overlap integrals
   of Matomnum+MatomnumF+MatomnumS+1 are stored.
   However, overlap integrals of Matomnum+MatomnumF+1 are
   stored in Force.c. So, F_TopMAN should be used to refer
   overlap integrals in Force.c, while S_TopMAN should be
   used in DC and GDC routines.

   Although OLP is used in Eff_Hub_Pot.c, the usage is 
   consistent with that of DC and GDC routines by the following 
   reason:

   DC or GDC:      OLP + Spe_Total_NO   if no orbital optimization 
                CntOLP + Spe_Total_CNO  if orbital optimization

   Eff_Hub_Pot:    OLP + Spe_Total_NO   always since the U-potential
                                        affects to primitive orbital 

   If no orbital optimization, both the usages are consistent.
   If orbital optimization, CntOLP and OLP are used in DC(GDC) and 
   Eff_Hub_Pot.c, respectively. Therefore, there is no conflict. 
  *******************************************************************/

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
            size1 += 4*tno1*tno2;
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

        for (k=0; k<=3; k++){
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
		  tmp_array[num] = OLP1[k][Mc_AN][h_AN][i][j];
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
          
	size2 = Rcv_S_Size[IDR];
        
	/* allocation of array */
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
         
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	num = 0;

        for (k=0; k<=3; k++){
	  Mc_AN = F_TopMAN[IDR] - 1; /* F_TopMAN should be used. */
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
		  OLP1[k][Mc_AN][h_AN][i][j] = tmp_array2[num];
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
 
  /****************************************************
    freeing of arrays:
  ****************************************************/

  free(Snd_S_Size);
  free(Rcv_S_Size);
}




void Force_CoreHole(double *****CDM0, double *****iDM0)
{
  /****************************************************
        Force arising from the penalty functional 
        to create a core hole 
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,i,j,h_AN,q_AN,Mq_AN,start_q_AN;
  int jan,kl,km,kl1,Qwan,Gq_AN,Gh_AN,Mh_AN,Hwan,ian;
  int l1,l2,l3,l,LL,Mul1,tno0,ncp,so;
  int tno1,tno2,size1,size2,n,kk,num,po,po1,po2;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  int **S_array,**R_array;
  int S_comm_flag,R_comm_flag;
  int SA_num,q,Sc_AN,GSc_AN,smul;
  int Sc_wan,Sh_AN,GSh_AN,Sh_wan;
  int Sh_AN2,fan,jg,j0,jg0,Mj_AN0;
  int Original_Mc_AN;

  double rcutA,rcutB,rcut;
  double dEx,dEy,dEz,ene,pref;
  double Stime_atom, Etime_atom;
  dcomplex ***Hx,***Hy,***Hz;
  dcomplex ***Hx0,***Hy0,***Hz0;
  dcomplex ***Hx1,***Hy1,***Hz1;
  int *Snd_OLP_Size,*Rcv_OLP_Size;  
  int *Indicator;
  double *tmp_array;
  double *tmp_array2;

  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop,ODNloop;
  int *OneD2h_AN,*OneD2q_AN;
  double *dEx_threads;
  double *dEy_threads;
  double *dEz_threads;
  double stime,etime;
  double stime1,etime1;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&stime);

  /****************************
       allocation of arrays 
  *****************************/

  Indicator = (int*)malloc(sizeof(int)*numprocs);

  S_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    S_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  R_array = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    R_array[ID] = (int*)malloc(sizeof(int)*3);
  }

  Snd_OLP_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_OLP_Size = (int*)malloc(sizeof(int)*numprocs);

  /* initialize the temporal array storing the force contribution */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];
    Gxyz[Gc_AN][41] = 0.0;
    Gxyz[Gc_AN][42] = 0.0;
    Gxyz[Gc_AN][43] = 0.0;
  }

  /*****************************************}********************** 
      THE FIRST CASE:
      In case of I=i or I=j 
      for d [ \sum_k <i|k>ek<k|j> ]/dRI  
  ****************************************************************/

  /*******************************************************
   *******************************************************
       multiplying overlap integrals WITH COMMUNICATION

       In case of I=i or I=j 
       for d [ \sum_k <i|k>ek<k|j> ]/dRI  
   *******************************************************
   *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

  Hx0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hy0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hz0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hx1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hy1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

  Hz1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
  for (i=0; i<3; i++){
    Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
    for (j=0; j<List_YOUSO[7]; j++){
      Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
    }
  }

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
	tno1 = Spe_Total_CNO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[Hwan];
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
	tno1 = Spe_Total_CNO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[Hwan];

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
	tno1 = Spe_Total_CNO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[Hwan];

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

	  dEx = 0.0; 
	  dEy = 0.0; 
	  dEz = 0.0; 

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  fan = FNAN[Gc_AN];

	  h_AN = 0;
	  Gh_AN = natn[Gc_AN][h_AN];
	  Mh_AN = F_G2M[Gh_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  ian = Spe_Total_CNO[Hwan];

	  n = F_Rcv_Num_WK[IDR];
	  jg = Rcv_GAN[IDR][n];

	  for (j0=0; j0<=fan; j0++){

	    jg0 = natn[Gc_AN][j0];
	    Mj_AN0 = F_G2M[jg0];

	    po2 = 0;
	    if (Original_Mc_AN==Mj_AN0){
	      po2 = 1;
	      q_AN = j0;
	    }

	    if (po2==1){

	      Gq_AN = natn[Gc_AN][q_AN];
	      Mq_AN = F_G2M[Gq_AN];
	      Qwan = WhatSpecies[Gq_AN];
	      jan = Spe_Total_CNO[Qwan];
	      kl = RMI1[Mc_AN][h_AN][q_AN];

 	      dHCH(0,Mc_AN,h_AN,q_AN,OLP_CH,Hx0,Hy0,Hz0);

	      /* contribution of force = Trace(CDM0*dH) */
	      /* spin non-polarization */

	      if (SpinP_switch==0){

                if (q_AN==h_AN) pref = 2.0;
                else            pref = 4.0; 

		for (i=0; i<ian; i++){
		  for (j=0; j<jan; j++){

		    dEx += pref*CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r;
		    dEy += pref*CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r;
		    dEz += pref*CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r;
		  }
		}
	      }

	      /* collinear spin polarized or non-colliear without SO and LDA+U */

	      else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
		   && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

		if (q_AN==h_AN) pref = 1.0;
		else            pref = 2.0; 

		for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		  for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		    dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
			         + CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r);
		    dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
				 + CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r);
		    dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
				 + CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r);
		  }
		}
	      }

	      /* spin non-collinear with spin-orbit coupling or with LDA+U */

	      else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
                     || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

                if (q_AN==h_AN){

		  for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		    for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		      dEx +=
			  CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
			- iDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].i
			+ CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r
			- iDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hx0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hx0[2][i][j].i;

		      dEy += 
			  CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
			- iDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].i
			+ CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r
			- iDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hy0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hy0[2][i][j].i; 

		      dEz +=  
			  CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
			- iDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].i
			+ CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r
			- iDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hz0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hz0[2][i][j].i; 

		    }
		  }
		}

                else {

		  for (i=0; i<Spe_Total_CNO[Hwan]; i++){  /* Hwan */
		    for (j=0; j<Spe_Total_CNO[Qwan]; j++){ /* Qwan  */

		      dEx += 
			  CDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].r
			- iDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].r
			- iDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].i
		    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hx0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hx0[2][i][j].i;

		      dEy += 
			  CDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].r
			- iDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].r
			- iDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].i
	 	    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hy0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hy0[2][i][j].i; 

		      dEz +=
			  CDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].r
			- iDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].i
			+ CDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].r
			- iDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].i
	 	    + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hz0[2][i][j].r
		    - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hz0[2][i][j].i;

		    } /* j */
		  } /* i */

		  dHCH(0,Mc_AN,q_AN,h_AN,OLP_CH,Hx1,Hy1,Hz1);

		  kl1 = RMI1[Mc_AN][q_AN][h_AN];

		  for (i=0; i<Spe_Total_CNO[Qwan]; i++){ /* Qwan */
		    for (j=0; j<Spe_Total_CNO[Hwan]; j++){ /* Hwan */

		      dEx += 
			  CDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].r
			- iDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].r
			- iDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hx1[2][i][j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hx1[2][i][j].i;

		      dEy += 
			  CDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].r
			- iDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].r
			- iDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hy1[2][i][j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hy1[2][i][j].i; 

		      dEz +=
			  CDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].r
			- iDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].i
			+ CDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].r
			- iDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].i
		    + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hz1[2][i][j].r
		    - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hz1[2][i][j].i;

		    } /* j */
		  } /* i */

                }
	      }

	    } /* if (po2==1) */
	  } /* j0 */             

	  /* force from #4B */

	  Gxyz[Gc_AN][41] += dEx;
	  Gxyz[Gc_AN][42] += dEy;
	  Gxyz[Gc_AN][43] += dEz;

	  /* timing */
	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

	  /********************************************
            increment of F_Rcv_Num_WK[IDR] 
	  ********************************************/

	F_Rcv_Num_WK[IDR]++;

      } /* if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) */

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

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hx0[i][j]);
    }
    free(Hx0[i]);
  }
  free(Hx0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hy0[i][j]);
    }
    free(Hy0[i]);
  }
  free(Hy0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hz0[i][j]);
    }
    free(Hz0[i]);
  }
  free(Hz0);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hx1[i][j]);
    }
    free(Hx1[i]);
  }
  free(Hx1);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hy1[i][j]);
    }
    free(Hy1[i]);
  }
  free(Hy1);

  for (i=0; i<3; i++){
    for (j=0; j<List_YOUSO[7]; j++){
      free(Hz1[i][j]);
    }
    free(Hz1[i]);
  }
  free(Hz1);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part1 of force_HCH=%18.5f\n",etime-stime);fflush(stdout);
  } 

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HCH1) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }
  }

  /*******************************************************
   *******************************************************
     THE FIRST CASE:
     multiplying overlap integrals WITHOUT COMMUNICATION

     In case of I=i or I=j 
     for d [ \sum_k <i|k>ek<k|j> ]/dRI  
   *******************************************************
   *******************************************************/

  dtime(&stime);

#pragma omp parallel shared(time_per_atom,Gxyz,CDM0,SpinP_switch,SO_switch,Hub_U_switch,F_U_flag,Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch,OLP_CH,RMI1,FNAN,Spe_Total_CNO,WhatSpecies,F_G2M,natn,M2G,Matomnum,List_YOUSO,F_NL_flag) private(Hx0,Hy0,Hz0,Hx1,Hy1,Hz1,OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Etime_atom,dEx,dEy,dEz,Gc_AN,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,kl1,i,j,kk,pref)
  {

    /* allocation of array */

    Hx0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hx0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hx0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hy0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hy0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hy0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hz0 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hz0[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hz0[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hx1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hx1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hx1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hy1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hy1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hy1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    Hz1 = (dcomplex***)malloc(sizeof(dcomplex**)*3);
    for (i=0; i<3; i++){
      Hz1[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	Hz1[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
      }
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      dEx = 0.0; 
      dEy = 0.0; 
      dEz = 0.0; 

      Gc_AN = M2G[Mc_AN];
      h_AN = 0;
      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Hwan = WhatSpecies[Gh_AN];
      ian = Spe_Total_CNO[Hwan];

      for (q_AN=0; q_AN<=FNAN[Gc_AN]; q_AN++){

	Gq_AN = natn[Gc_AN][q_AN];
	Mq_AN = F_G2M[Gq_AN];

	if (Mq_AN<=Matomnum){

	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];

          dHCH(0,Mc_AN,h_AN,q_AN,OLP_CH,Hx0,Hy0,Hz0);

	  if (SpinP_switch==0){

            if (q_AN==h_AN) pref = 2.0;
            else            pref = 4.0; 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){

		dEx += pref*CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r;
		dEy += pref*CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r;
		dEz += pref*CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r;
	      }
	    }
	  }

          /* collinear spin polarized or non-colliear without SO and LDA+U */

	  else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
	        && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

	    if (q_AN==h_AN) pref = 1.0;
	    else            pref = 2.0; 

	    for (i=0; i<ian; i++){
	      for (j=0; j<jan; j++){

		dEx += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
			     + CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r);
		dEy += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
			     + CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r);
		dEz += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
			     + CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r);
	      }
	    }
	  }

	  /* spin non-collinear with spin-orbit coupling or with LDA+U */

	  else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
		|| 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

            if (q_AN==h_AN){

	      for (i=0; i<Spe_Total_CNO[Hwan]; i++){
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){

		  dEx += 
                      CDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hx0[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hx0[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hx0[2][i][j].r
		- 2.0*CDM0[3][Mh_AN][kl][i][j]*Hx0[2][i][j].i;

		  dEy += 
                      CDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hy0[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hy0[1][i][j].i
		+ 2.0*CDM0[2][Mh_AN][kl][i][j]*Hy0[2][i][j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hy0[2][i][j].i; 

		  dEz +=
                      CDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hz0[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hz0[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hz0[2][i][j].r
		- 2.0*CDM0[3][Mh_AN][kl][i][j]*Hz0[2][i][j].i;

		}
	      }
            }

            else{

              for (i=0; i<Spe_Total_CNO[Hwan]; i++){  /* Hwan */
		for (j=0; j<Spe_Total_CNO[Qwan]; j++){ /* Qwan  */

		  dEx += 
                      CDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*Hx0[0][i][j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*Hx0[1][i][j].i
		+ 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hx0[2][i][j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hx0[2][i][j].i;

		  dEy += 
                      CDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*Hy0[0][i][j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*Hy0[1][i][j].i
		+ 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hy0[2][i][j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hy0[2][i][j].i; 

		  dEz +=
                      CDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].r
		    - iDM0[0][Mh_AN][kl ][i][j]*Hz0[0][i][j].i
		    + CDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].r
		    - iDM0[1][Mh_AN][kl ][i][j]*Hz0[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl ][i][j]*Hz0[2][i][j].r
                - 2.0*CDM0[3][Mh_AN][kl ][i][j]*Hz0[2][i][j].i;

		} /* j */
	      } /* i */

              dHCH(0,Mc_AN,q_AN,h_AN,OLP_CH,Hx1,Hy1,Hz1);

       	      kl1 = RMI1[Mc_AN][q_AN][h_AN];

	      for (i=0; i<Spe_Total_CNO[Qwan]; i++){ /* Qwan */
		for (j=0; j<Spe_Total_CNO[Hwan]; j++){ /* Hwan */

		  dEx += 
		      CDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*Hx1[0][i][j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*Hx1[1][i][j].i
		+ 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hx1[2][i][j].r
		- 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hx1[2][i][j].i;

		  dEy += 
                      CDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*Hy1[0][i][j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*Hy1[1][i][j].i
	        + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hy1[2][i][j].r
	        - 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hy1[2][i][j].i; 

		  dEz +=
                      CDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].r
		    - iDM0[0][Mq_AN][kl1][i][j]*Hz1[0][i][j].i
		    + CDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].r
		    - iDM0[1][Mq_AN][kl1][i][j]*Hz1[1][i][j].i
	        + 2.0*CDM0[2][Mq_AN][kl1][i][j]*Hz1[2][i][j].r
		- 2.0*CDM0[3][Mq_AN][kl1][i][j]*Hz1[2][i][j].i;

		} /* j */
	      } /* i */

	    } 
	  }
	}
      }

      /* force from #4B */

      if (F_NL_flag==1){
        Gxyz[Gc_AN][41] += dEx;
        Gxyz[Gc_AN][42] += dEy;
        Gxyz[Gc_AN][43] += dEz;
      }

      /* timing */
      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of array */

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hx0[i][j]);
      }
      free(Hx0[i]);
    }
    free(Hx0);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hy0[i][j]);
      }
      free(Hy0[i]);
    }
    free(Hy0);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hz0[i][j]);
      }
      free(Hz0[i]);
    }
    free(Hz0);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hx1[i][j]);
      }
      free(Hx1[i]);
    }
    free(Hx1);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hy1[i][j]);
      }
      free(Hy1[i]);
    }
    free(Hy1);

    for (i=0; i<3; i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(Hz1[i][j]);
      }
      free(Hz1[i]);
    }
    free(Hz1);

  } /* #pragma omp parallel */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of force_HCH=%18.5f\n",etime-stime);fflush(stdout);
  } 

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HCH2) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }
  }

  /*************************************************************
     THE SECOND CASE:
     In case of I=k with I!=i and I!=j
     d [ \sum_k <i|k>ek<k|j> ]/dRI  
  *************************************************************/

  /************************************************************ 
    MPI communication of OLP_CH whose basis part is not located 
    on own site but projector part is located on own site. 
  ************************************************************/

  MPI_Barrier(mpi_comm_level1);
  dtime(&stime);

  for (ID=0; ID<numprocs; ID++) Indicator[ID] = 0;

  for (Mc_AN=1; Mc_AN<=Max_Matomnum; Mc_AN++){

    if (Mc_AN<=Matomnum)  Gc_AN = M2G[Mc_AN];
    else                  Gc_AN = 0;

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      i = Indicator[IDS]; 
      po = 0;

      Gh_AN = Pro_Snd_GAtom[IDS][i]; 

      if (Gh_AN!=0){

	/* find the range with the same global atomic number */

	do {

	  i++;
	  if (Gh_AN!=Pro_Snd_GAtom[IDS][i]) po = 1;
	} while(po==0);

	i--;
	SA_num = i - Indicator[IDS] + 1;

	/* find the data size to send the block data */

	size1 = 0;
	for (q=Indicator[IDS]; q<=(Indicator[IDS]+SA_num-1); q++){

	  Sc_AN = Pro_Snd_MAtom[IDS][q]; 
	  GSc_AN = F_M2G[Sc_AN];
	  Sc_wan = WhatSpecies[GSc_AN];
	  tno1 = Spe_Total_CNO[Sc_wan];

	  Sh_AN = Pro_Snd_LAtom[IDS][q]; 
	  GSh_AN = natn[GSc_AN][Sh_AN];
	  Sh_wan = WhatSpecies[GSh_AN];
	  tno2 = Spe_Total_CNO[Sh_wan];

	  size1 += 4*tno1*tno2;
	  size1 += 3;
	}

      } /* if (Gh_AN!=0) */

      else {
	SA_num = 0;
	size1 = 0;
      }
        
      S_array[IDS][0] = Gh_AN;
      S_array[IDS][1] = SA_num;
      S_array[IDS][2] = size1;

      if (ID!=0){
	MPI_Isend(&S_array[IDS][0], 3, MPI_INT, IDS, tag, mpi_comm_level1, &request);
	MPI_Recv( &R_array[IDR][0], 3, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	MPI_Wait(&request,&stat);
      }
      else {
	R_array[myid][0] = S_array[myid][0];
	R_array[myid][1] = S_array[myid][1];
	R_array[myid][2] = S_array[myid][2];
      }

      if (R_array[IDR][0]==Gc_AN) R_comm_flag = 1;
      else                        R_comm_flag = 0;

      if (ID!=0){
	MPI_Isend(&R_comm_flag, 1, MPI_INT, IDR, tag, mpi_comm_level1, &request);
	MPI_Recv( &S_comm_flag, 1, MPI_INT, IDS, tag, mpi_comm_level1, &stat);
	MPI_Wait(&request,&stat);
      }
      else{
	S_comm_flag = R_comm_flag;
      }

      /*****************************************
                    send the data
      *****************************************/
        
      /* if (S_comm_flag==1) then, send data to IDS */
         
      if (S_comm_flag==1){

	/* allocate tmp_array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (q=Indicator[IDS]; q<=(Indicator[IDS]+SA_num-1); q++){

	  Sc_AN = Pro_Snd_MAtom[IDS][q]; 
	  GSc_AN = F_M2G[Sc_AN];
	  Sc_wan = WhatSpecies[GSc_AN];
	  tno1 = Spe_Total_CNO[Sc_wan];

	  Sh_AN = Pro_Snd_LAtom[IDS][q]; 
	  GSh_AN = natn[GSc_AN][Sh_AN];
	  Sh_wan = WhatSpecies[GSh_AN];
	  tno2 = Spe_Total_CNO[Sh_wan];
	  Sh_AN2 = Pro_Snd_LAtom2[IDS][q]; 

	  tmp_array[num] = (double)Sc_AN;  num++;
	  tmp_array[num] = (double)Sh_AN;  num++; 
	  tmp_array[num] = (double)Sh_AN2; num++; 

	  for (kk=0; kk<=3; kk++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp_array[num] = OLP_CH[kk][Sc_AN][Sh_AN][i][j];
		num++;
	      }
	    }
	  }
	}

	if (ID!=0){
	  MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
	}

	/* update Indicator[IDS] */

	Indicator[IDS] += SA_num;

      } /* if (S_comm_flag==1) */ 

      /*****************************************
                   receiving the data
      *****************************************/

      /* if (R_comm_flag==1) then, receive the data from IDR */

      if (R_comm_flag==1){

	size2 = R_array[IDR][2]; 
	tmp_array2 = (double*)malloc(sizeof(double)*size2);

	if (ID!=0){
	  MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
	}
	else{
	  for (i=0; i<size2; i++) tmp_array2[i] = tmp_array[i];
	}

	/* store */

	num = 0;

	for (n=0; n<R_array[IDR][1]; n++){
            
	  Sc_AN  = (int)tmp_array2[num]; num++;
	  Sh_AN  = (int)tmp_array2[num]; num++;
	  Sh_AN2 = (int)tmp_array2[num]; num++;

	  GSc_AN = natn[Gc_AN][Sh_AN2]; 
	  Sc_wan = WhatSpecies[GSc_AN]; 
	  tno1 = Spe_Total_CNO[Sc_wan];

	  GSh_AN = natn[GSc_AN][Sh_AN];
	  Sh_wan = WhatSpecies[GSh_AN];
	  tno2 = Spe_Total_CNO[Sh_wan];

	  for (kk=0; kk<=3; kk++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		OLP_CH[kk][Matomnum+1][Sh_AN2][i][j] = tmp_array2[num];
		num++;
	      }
	    }
	  }
	}

	/* free tmp_array2 */
	free(tmp_array2);
 
      } /* if (R_comm_flag==1) */

      if (S_comm_flag==1){
	if (ID!=0) MPI_Wait(&request,&stat);
	free(tmp_array);  /* freeing of array */
      }
      
    } /* ID */

    if (Mc_AN<=Matomnum){ 

      /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
      {
	Nthrds0 = omp_get_num_threads();
      }

      /* allocation of arrays */
      dEx_threads = (double*)malloc(sizeof(double)*Nthrds0);
      dEy_threads = (double*)malloc(sizeof(double)*Nthrds0);
      dEz_threads = (double*)malloc(sizeof(double)*Nthrds0);

      for (Nloop=0; Nloop<Nthrds0; Nloop++){
	dEx_threads[Nloop] = 0.0;
	dEy_threads[Nloop] = 0.0;
	dEz_threads[Nloop] = 0.0;
      }

      /* one-dimensionalize the h_AN and q_AN loops */ 

      OneD2h_AN = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+1)*(FNAN[Gc_AN]+2));
      OneD2q_AN = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+1)*(FNAN[Gc_AN]+2));

      ODNloop = 0;
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1)
	 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) 
         || (Solver==5 || Solver==8 || Solver==11) )
	  start_q_AN = 0;
	else
	  start_q_AN = h_AN;

	for (q_AN=start_q_AN; q_AN<=FNAN[Gc_AN]; q_AN++){

	  kl = RMI1[Mc_AN][h_AN][q_AN];

	  if (0<=kl){
	    OneD2h_AN[ODNloop] = h_AN;
	    OneD2q_AN[ODNloop] = q_AN; 
	    ODNloop++;      
	  }
	}
      }

#pragma omp parallel shared(ODNloop,OneD2h_AN,OneD2q_AN,Mc_AN,Gc_AN,dEx_threads,dEy_threads,dEz_threads,CDM0,SpinP_switch,SO_switch,Hub_U_switch,Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch,OLP_CH,RMI1,Spe_Total_CNO,WhatSpecies,F_G2M,natn,FNAN,List_YOUSO,Solver,F_NL_flag,F_U_flag) private(OMPID,Nthrds,Nprocs,Hx,Hy,Hz,i,j,h_AN,Gh_AN,Mh_AN,Hwan,ian,q_AN,Gq_AN,Mq_AN,Qwan,jan,kl,km,Nloop,pref)
      {
          
	/* allocation of arrays */

	Hx = (dcomplex***)malloc(sizeof(dcomplex**)*3);
	for (i=0; i<3; i++){
	  Hx[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    Hx[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}

	Hy = (dcomplex***)malloc(sizeof(dcomplex**)*3);
	for (i=0; i<3; i++){
	  Hy[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    Hy[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}

	Hz = (dcomplex***)malloc(sizeof(dcomplex**)*3);
	for (i=0; i<3; i++){
	  Hz[i] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
	  for (j=0; j<List_YOUSO[7]; j++){
	    Hz[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]);
	  }
	}

	/* get info. on OpenMP */ 
          
	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (Nloop=OMPID*ODNloop/Nthrds; Nloop<(OMPID+1)*ODNloop/Nthrds; Nloop++){

	  /* get h_AN and q_AN */

	  h_AN = OneD2h_AN[Nloop];
	  q_AN = OneD2q_AN[Nloop];

	  /* set informations on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Mh_AN = F_G2M[Gh_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  ian = Spe_Total_CNO[Hwan];

	  /* set informations on q_AN */

	  Gq_AN = natn[Gc_AN][q_AN];
	  Mq_AN = F_G2M[Gq_AN];
	  Qwan = WhatSpecies[Gq_AN];
	  jan = Spe_Total_CNO[Qwan];
	  kl = RMI1[Mc_AN][h_AN][q_AN];
          km = RMI1[Mc_AN][q_AN][h_AN];

	  if (0<=kl){

            dHCH(1,Mc_AN,h_AN,q_AN,OLP_CH,Hx,Hy,Hz);

	    /* contribution of force = Trace(CDM0*dH) */

	    /* spin non-polarization */

	    if (SpinP_switch==0){

              if (Solver==5 || Solver==8 || Solver==11){
	        pref = 2.0;
              }
              else {
	        if (q_AN==h_AN) pref = 2.0;
  	        else            pref = 4.0; 
              }

	      for (i=0; i<ian; i++){
		for (j=0; j<jan; j++){
		  dEx_threads[OMPID] += pref*CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r;
		  dEy_threads[OMPID] += pref*CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r;
		  dEz_threads[OMPID] += pref*CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r;
		}
	      }

	    }

            /* collinear spin polarized or non-colliear without SO and LDA+U */

	    else if (SpinP_switch==1 || (SpinP_switch==3 && SO_switch==0 && Hub_U_switch==0
	          && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)){ 

              if (Solver==5 || Solver==8 || Solver==11){
	        pref = 1.0;
              }
              else {
	        if (q_AN==h_AN) pref = 1.0;
  	        else            pref = 2.0; 
              }

	      for (i=0; i<ian; i++){
		for (j=0; j<jan; j++){

		  dEx_threads[OMPID] += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r
					      + CDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].r);
		  dEy_threads[OMPID] += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r
					      + CDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].r);
		  dEz_threads[OMPID] += pref*(  CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r
					      + CDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].r);

		}
	      }
	    }

	    /* spin non-collinear with spin-orbit coupling or with LDA+U */

	    else if ( SpinP_switch==3 && (SO_switch==1 || (Hub_U_switch==1 && F_U_flag==1) 
 		   || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){

	      pref = 1.0; 

	      for (i=0; i<ian; i++){
	        for (j=0; j<jan; j++){

		  dEx_threads[OMPID] += 
	        pref*(CDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hx[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hx[1][i][j].i
 	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hx[2][i][j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hx[2][i][j].i);

		  dEy_threads[OMPID] += 
		pref*(CDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].r
	            - iDM0[0][Mh_AN][kl][i][j]*Hy[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hy[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hy[2][i][j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hy[2][i][j].i); 

		  dEz_threads[OMPID] +=
		pref*(CDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].r
		    - iDM0[0][Mh_AN][kl][i][j]*Hz[0][i][j].i
		    + CDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].r
		    - iDM0[1][Mh_AN][kl][i][j]*Hz[1][i][j].i
	        + 2.0*CDM0[2][Mh_AN][kl][i][j]*Hz[2][i][j].r
	        - 2.0*CDM0[3][Mh_AN][kl][i][j]*Hz[2][i][j].i); 

		}
	      }
	    }

	  } /* if (0<=kl) */
	} /* Nloop */

        /* freeing of arrays */

	for (i=0; i<3; i++){
	  for (j=0; j<List_YOUSO[7]; j++){
	    free(Hx[i][j]);
	  }
	  free(Hx[i]);
	}
	free(Hx);

	for (i=0; i<3; i++){
	  for (j=0; j<List_YOUSO[7]; j++){
	    free(Hy[i][j]);
	  }
	  free(Hy[i]);
	}
	free(Hy);

	for (i=0; i<3; i++){
	  for (j=0; j<List_YOUSO[7]; j++){
	    free(Hz[i][j]);
	  }
	  free(Hz[i]);
	}
	free(Hz);

      } /* #pragma omp parallel */

      /* sum of dEx_threads */

      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;

      if (F_NL_flag==1){
        for (Nloop=0; Nloop<Nthrds0; Nloop++){
	  dEx += dEx_threads[Nloop];
	  dEy += dEy_threads[Nloop];
	  dEz += dEz_threads[Nloop];
        }

        /* force from #4B */

        Gxyz[Gc_AN][41] += dEx;
        Gxyz[Gc_AN][42] += dEy;
        Gxyz[Gc_AN][43] += dEz;
      }

      if (2<=level_stdout){
        printf("<Force>  force(HCH3) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	       myid,Mc_AN,Gc_AN,dEx,dEy,dEz);fflush(stdout);
      }

      /* freeing of array */
      free(OneD2q_AN);
      free(OneD2h_AN);
      free(dEx_threads);
      free(dEy_threads);
      free(dEz_threads);

    } /* if (Mc_AN<=Matomnum) */

  } /* Mc_AN */

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part3 of force_NL=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /********************************************************
    adding Gxyz[Gc_AN][41,42,43] to Gxyz[Gc_AN][17,18,19]
  ********************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Force>  force(HCH) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Mc_AN,Gc_AN,Gxyz[Gc_AN][41],Gxyz[Gc_AN][42],Gxyz[Gc_AN][43]);fflush(stdout);
    }

    Gxyz[Gc_AN][17] += Gxyz[Gc_AN][41];
    Gxyz[Gc_AN][18] += Gxyz[Gc_AN][42];
    Gxyz[Gc_AN][19] += Gxyz[Gc_AN][43];
  }

  /***********************************
            freeing of arrays 
  ************************************/

  free(Indicator);

  for (ID=0; ID<numprocs; ID++){
    free(S_array[ID]);
  }
  free(S_array);

  for (ID=0; ID<numprocs; ID++){
    free(R_array[ID]);
  }
  free(R_array);

  free(Snd_OLP_Size);
  free(Rcv_OLP_Size);
}

