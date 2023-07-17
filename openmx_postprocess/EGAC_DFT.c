/**********************************************************************
  EGAC_DFT.c:

   EGAC_DFT.c is a subroutine to perform a divide and conquer method 
   of embedding Green functions with analytic continuation. 

  Log of EGAC_DFT.c:

     10/Nov/2017  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>
#include <complex.h>

#define  measure_time   0



static void EGAC_Col(char *mode,
		     int SCF_iter,
		     double *****Hks, double ****OLP0,
		     double *****CDM,
		     double *****EDM,
		     double Eele0[2], double Eele1[2]);

static void Calc_GF( char *mode,
                     int myid,
                     int numprocs,
                     double *tmp_ga_array, 
                     double *tmp_ga_array2, 
                     int *MP, int *MP_A,
                     dcomplex *GA,
                     dcomplex *GB,
                     dcomplex *B,
                     dcomplex *Mat, 
                     dcomplex *Mat2, 
                     double Trial_ChemP );

static void Mixing_Sigma(int myid, int numprocs, int job_id, int method, double alphai);

static double Calc_Number_of_Electrons();
static double Find_Trial_ChemP_by_Muller(
        int SCF_iter, int num_loop,   
        double pTrial_ChemP[4], double pDiff_Num[5], 
        double DeltaMu[7], double DeltaN[3], double DeltaNp[3]);
static dcomplex Tfunc(int k, dcomplex z0);
static dcomplex Tfunc_deri(int k, dcomplex z0);




double EGAC_DFT( char *mode,
                 int SCF_iter,
                 int SpinP_switch,
                 double *****Hks,
                 double *****ImNL,
                 double ****OLP0,
                 double *****CDM,
                 double *****EDM,
                 double Eele0[2], double Eele1[2] )
{
  double time0,TStime,TEtime;

  dtime(&TStime);

  /****************************************************
          collinear without spin-orbit coupling
  ****************************************************/

  if ( SpinP_switch==0 || SpinP_switch==1 ){

    EGAC_Col(mode,SCF_iter, Hks, OLP0, CDM, EDM, Eele0, Eele1);
  }

  /****************************************************
   non-collinear with and without spin-orbit coupling
  ****************************************************/

  else if (SpinP_switch==3){

  }

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}



static void EGAC_Col(char *mode,
		     int SCF_iter,
		     double *****Hks, double ****OLP0,
		     double *****CDM,
		     double *****EDM,
		     double Eele0[2], double Eele1[2])
{
  static int firsttime=1;
  int i,j,k,i0,j0,Mc_AN,Gc_AN,h_AN,Gh_AN,Lc_AN,tno1,tno2;
  int m,n,num,num2,spin,Cwan,Hwan,size1,size2;
  int job_id,job_gid,N3[4],lpole,pole;
  int kl,ig,jg,ian,jan,Anum,Bnum,Gi,ih;
  int *MP,*MP_A;
  double *tmp_array,*tmp_array2;
  double *tmp_ga_array,*tmp_ga_array2;
  int ID,IDS,IDR,myid,numprocs,tag;
  dcomplex *GA,*GB,*B,*Mat,*Mat2;
  int wan,iter_calc_G,iter_ChemP;
  int end_outer_flag,end_inner_flag,norecalc_flag;
  double TZ,Trial_ChemP,Nele,Nele_prev,x3,y3;
  static double pTrial_ChemP[4],pDiff_Num[5]; 
  static double DeltaMu[7],DeltaN[3],DeltaNp[3];  
  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  tmp_array  = (double*)malloc(sizeof(double)*Max_Snd_OLP_EGAC_Size*(SpinP_switch+1));
  tmp_array2 = (double*)malloc(sizeof(double)*Max_Rcv_OLP_EGAC_Size*(SpinP_switch+1));

  tmp_ga_array  = (double*)malloc(sizeof(double)*Max_Snd_GA_EGAC_Size*2);
  tmp_ga_array2 = (double*)malloc(sizeof(double)*Max_Rcv_GA_EGAC_Size*2);

  MP = (int*)malloc(sizeof(int)*List_YOUSO[2]);
  MP_A = (int*)malloc(sizeof(int)*Max_ONAN);

  GA = (dcomplex*)malloc(sizeof(dcomplex)*Max_dim_GA_EGAC*Max_dim_GA_EGAC);
  GB = (dcomplex*)malloc(sizeof(dcomplex)*Max_dim_GA_EGAC*Max_dim_GD_EGAC);
  B = (dcomplex*)malloc(sizeof(dcomplex)*Max_dim_GA_EGAC*Max_dim_GD_EGAC);

  if (Max_dim_GA_EGAC<Max_dim_GD_EGAC){
    Mat  = (dcomplex*)malloc(sizeof(dcomplex)*Max_dim_GD_EGAC*Max_dim_GD_EGAC);
    Mat2 = (dcomplex*)malloc(sizeof(dcomplex)*Max_dim_GD_EGAC*Max_dim_GD_EGAC);
  }
  else{
    Mat  = (dcomplex*)malloc(sizeof(dcomplex)*Max_dim_GA_EGAC*Max_dim_GA_EGAC);
    Mat2 = (dcomplex*)malloc(sizeof(dcomplex)*Max_dim_GA_EGAC*Max_dim_GA_EGAC);
  }

  /****************************************************
   initialize arrays if SCF_iter==1
  ****************************************************/

  if (SCF_iter==1){

    /* GD_EGAC */
      
    for (job_id=0; job_id<EGAC_Num; job_id++){ /* job_id: local job_id */

      job_gid = job_id + EGAC_Top[myid]; /* job_gid: global job_id */
      GN2N_EGAC(job_gid,N3);

      Gc_AN = N3[1];
      Cwan = WhatSpecies[Gc_AN]; 
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];        
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
	    GD_EGAC[job_id][h_AN][i][j].r = 0.0;
	    GD_EGAC[job_id][h_AN][i][j].i = 0.0;
	  }
	}
      } 
    }

    /* Sigma_EGAC */

    for (m=0; m<DIIS_History_EGAC; m++){
      for (job_id=0; job_id<EGAC_Num; job_id++){
	for (i=0; i<dim_GD_EGAC[job_id]*dim_GD_EGAC[job_id]; i++){
	  Sigma_EGAC[m][job_id][i].r = 0.0;
	  Sigma_EGAC[m][job_id][i].i = 0.0;
	}
      }  
    }

  }

  /****************************************************
   MPI

   Hks
  ****************************************************/

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    /*****************************
           sending of data 
    *****************************/

    if ( Num_Snd_HS_EGAC[IDS]!=0 ){

      size1 = Snd_OLP_EGAC_Size[IDS]*(SpinP_switch+1);

      /* multidimentional array to vector array */

      num = 0;
      for (spin=0; spin<=SpinP_switch; spin++){
	for (n=0; n<Num_Snd_HS_EGAC[IDS]; n++){

	  Mc_AN = Indx_Snd_HS_EGAC[IDS][n];
	  Gc_AN = M2G[Mc_AN];
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

      if (ID!=0) MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
    }

    /*****************************
       receiving of block data
    *****************************/

    if ( Num_Rcv_HS_EGAC[IDR]!=0 ){

      size2 = Rcv_OLP_EGAC_Size[IDR]*(SpinP_switch+1);

      if (ID!=0)
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
      else{ 
        for (i=0; i<size2; i++) tmp_array2[i] = tmp_array[i];
      }

      num = 0;
      for (spin=0; spin<=SpinP_switch; spin++){

	Lc_AN = Top_Index_HS_EGAC[IDR];

	for (n=0; n<Num_Rcv_HS_EGAC[IDR]; n++){

	  Gc_AN = Indx_Rcv_HS_EGAC[IDR][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		H_EGAC[spin][Lc_AN][h_AN][i][j] = tmp_array2[num];
		num++;
	      }
	    }
	  }

	  Lc_AN++;
	}        
      }
    }

    if ( Num_Snd_HS_EGAC[IDS]!=0 && ID!=0 ) MPI_Wait(&request,&stat);
  }

  /****************************************************
   MPI

   OLP0
  ****************************************************/

  if (SCF_iter==1){

    tag = 999;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /*****************************
              sending of data 
      *****************************/

      if ( Num_Snd_HS_EGAC[IDS]!=0 ){

	size1 = Snd_OLP_EGAC_Size[IDS];

	/* multidimentional array to vector array */

	num = 0;

	for (n=0; n<Num_Snd_HS_EGAC[IDS]; n++){

	  Mc_AN = Indx_Snd_HS_EGAC[IDS][n];
	  Gc_AN = M2G[Mc_AN];
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

        if (ID!=0) MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if ( Num_Rcv_HS_EGAC[IDR]!=0 ){
          
	size2 = Rcv_OLP_EGAC_Size[IDR];

        if (ID!=0)
          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
        else{ 
          for (i=0; i<size2; i++) tmp_array2[i] = tmp_array[i];
	}

	num = 0;
	Lc_AN = Top_Index_HS_EGAC[IDR];

	for (n=0; n<Num_Rcv_HS_EGAC[IDR]; n++){

	  Gc_AN = Indx_Rcv_HS_EGAC[IDR][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		OLP_EGAC[Lc_AN][h_AN][i][j] = tmp_array2[num];
		num++;
	      }
	    }
	  }

	  Lc_AN++;
	}        
      }

      if ( Num_Snd_HS_EGAC[IDS]!=0 && ID!=0 ) MPI_Wait(&request,&stat);
    }
  }

  /****************************************************
                  total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ = TZ + Spe_Core_Charge[wan];
  }

  /************************************************************************
                       Calculation of Green functions
  *************************************************************************/

  /* loop for adjusting chemical potential */

  Trial_ChemP = ChemP;
  iter_ChemP = 1;
  end_outer_flag = 0;

  if ( strcasecmp(mode,"oneshot")==0 ){

    do {

      /* loop for Green functions */

      iter_calc_G = 1;
      end_inner_flag = 0;
      norecalc_flag = 0;
      Nele_prev = 0.0;

      do {

	if (iter_ChemP==1 && iter_calc_G<=4){
	  Calc_GF("recalc", myid, numprocs,tmp_ga_array,tmp_ga_array2,MP,MP_A,GA,GB,B,Mat,Mat2,Trial_ChemP );
	}

	else{
	  Calc_GF("norecalc", myid, numprocs,tmp_ga_array,tmp_ga_array2,MP,MP_A,GA,GB,B,Mat,Mat2,Trial_ChemP );
	  norecalc_flag = 1;
	}

	/* calculate the number of electrons */
	Nele = Calc_Number_of_Electrons();
    
	if (myid==Host_ID){
	  printf("GGG iter_ChemP=%3d iter_calc_G=%2d Trial_ChemP=%15.12f Nele=%15.12f\n",
		 iter_ChemP,iter_calc_G,Trial_ChemP,Nele);
	}

	/* check convergence */

	if (fabs(Nele-Nele_prev)<1.0e-8){
	  end_inner_flag = 1;
	}
	else if (scf_GF_EGAC==0 && norecalc_flag==1){
	  end_inner_flag = 1;
	}

	Nele_prev = Nele;
 
	/* increment of iter_calc_G */
	iter_calc_G++; 

      } while (iter_calc_G<50 && end_inner_flag==0);

      /* set x3 and y3 */

      x3 = Trial_ChemP;
      y3 = -TZ + Nele + system_charge;
      pTrial_ChemP[3] = x3;
      pDiff_Num[3] = y3;

      Trial_ChemP = Find_Trial_ChemP_by_Muller( SCF_iter, iter_ChemP,
						pTrial_ChemP,  pDiff_Num, 
						DeltaMu, DeltaN, DeltaNp );

      /* check its convergence */

      if (    fabs(y3)<1.0e-8
	      || (fabs(y3)<1.0e-7 && SCF_iter<5) 
	      || fabs(Trial_ChemP-x3)<1.0e-10 ){

	end_outer_flag = 1;
      }
      else {
	iter_ChemP++;
      }

    } while (iter_ChemP<20 && end_outer_flag==0);

  }


  else if ( strcasecmp(mode,"force")==0 ){

    Calc_GF("force", myid, numprocs,tmp_ga_array,tmp_ga_array2,MP,MP_A,GA,GB,B,Mat,Mat2,Trial_ChemP );
  }

  else if ( strcasecmp(mode,"scf")==0 ){

    do {

      /* loop for Green functions */

      iter_calc_G = 1;
      end_inner_flag = 0;
      norecalc_flag = 0;
      Nele_prev = 0.0;

      do {

	/* Calculation of Green functions in a non-self consistent way */
     
	if (scf_GF_EGAC==0){

	  if (SCF_iter==1 && iter_ChemP==1 && iter_calc_G<=1){
	    Calc_GF("recalc_nomix", myid, numprocs,tmp_ga_array,tmp_ga_array2,MP,MP_A,GA,GB,B,Mat,Mat2,Trial_ChemP );
	  }

	  else if ( SCF_iter<=4 && iter_ChemP<=1 && iter_calc_G<=1 ){
	    Calc_GF("recalc", myid, numprocs,tmp_ga_array,tmp_ga_array2,MP,MP_A,GA,GB,B,Mat,Mat2,Trial_ChemP );
	  }

	  else{
	    Calc_GF("norecalc", myid, numprocs,tmp_ga_array,tmp_ga_array2,MP,MP_A,GA,GB,B,Mat,Mat2,Trial_ChemP );
	    norecalc_flag = 1;
	  }
	}

	/* Calculation of Green functions in a self consistent way */

	else {

	  if (SCF_iter==1 && iter_ChemP==1 && iter_calc_G<=2){
	    Calc_GF("recalc_nomix", myid, numprocs,tmp_ga_array,tmp_ga_array2,MP,MP_A,GA,GB,B,Mat,Mat2,Trial_ChemP );
	  }
	  else {
	    Calc_GF("recalc", myid, numprocs,tmp_ga_array,tmp_ga_array2,MP,MP_A,GA,GB,B,Mat,Mat2,Trial_ChemP );
	  }
	}

	/* calculate the number of electrons */
	Nele = Calc_Number_of_Electrons();
    
	if (myid==Host_ID){
	  printf("GGG iter_ChemP=%3d iter_calc_G=%2d Trial_ChemP=%15.12f Nele=%15.12f\n",
		 iter_ChemP,iter_calc_G,Trial_ChemP,Nele);
	}

	/* check convergence */

	if (fabs(Nele-Nele_prev)<1.0e-8){
	  end_inner_flag = 1;
	}
	else if (scf_GF_EGAC==0 && norecalc_flag==1){
	  end_inner_flag = 1;
	}

	Nele_prev = Nele;
 
	/* increment of iter_calc_G */
	iter_calc_G++; 

      } while (iter_calc_G<50 && end_inner_flag==0);

      /* set x3 and y3 */

      x3 = Trial_ChemP;
      y3 = -TZ + Nele + system_charge;
      pTrial_ChemP[3] = x3;
      pDiff_Num[3] = y3;

      Trial_ChemP = Find_Trial_ChemP_by_Muller( SCF_iter, iter_ChemP,
						pTrial_ChemP,  pDiff_Num, 
						DeltaMu, DeltaN, DeltaNp );

      /* check its convergence */

      if (    fabs(y3)<1.0e-9
	      || (fabs(y3)<1.0e-7 && SCF_iter<5) 
	      || fabs(Trial_ChemP-x3)<1.0e-10 ){

	end_outer_flag = 1;
      }
      else {
	iter_ChemP++;
      }

    } while (iter_ChemP<20 && end_outer_flag==0);

  } /* else if ( strcasecmp(mode,"scf")==0 ) */

  /* update the new ChemP */

  ChemP = Trial_ChemP;

  /************************************************************************
      MPI communication of DM_Snd_EGAC             
  *************************************************************************/

  /*
  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<Matomnum_DM_Snd_EGAC; Mc_AN++){

      Gc_AN = M2G_DM_Snd_EGAC[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
	    printf("ABC6 spin=%2d Mc_AN=%2d h_AN=%2d i=%2d j=%2d %15.12f\n",
		   spin,Mc_AN,h_AN,i,j,DM_Snd_EGAC[spin][Mc_AN][h_AN][i][j]);fflush(stdout);
	  }
	}      
      }
    }
  }
  */

  /* initialize CDM */

  for (spin=0; spin<=SpinP_switch; spin++){
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
            CDM[spin][Mc_AN][h_AN][i][j] = 0.0;
	  }
	}      
      }
    }
  }

  /* MPI communication */
  
  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    /*****************************
           sending of data 
    *****************************/

    if ( Num_Snd_DM_EGAC[IDS]!=0 ){

      size1 = Snd_DM_EGAC_Size[IDS];

      /* multidimentional array to vector array */

      num = 0;
      for (spin=0; spin<=SpinP_switch; spin++){
	for (n=0; n<Num_Snd_DM_EGAC[IDS]; n++){

	  Gc_AN = Indx_Snd_DM_EGAC[IDS][n];
	  Mc_AN = G2M_DM_Snd_EGAC[Gc_AN];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp_array[num] = DM_Snd_EGAC[spin][Mc_AN][h_AN][i][j];
		num++;
	      } 
	    } 
	  }
	}
      }

      if (ID!=0) MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
    }

    /*****************************
       receiving of block data
    *****************************/

    if ( Num_Rcv_DM_EGAC[IDR]!=0 ){

      size2 = Rcv_DM_EGAC_Size[IDR];

      if (ID!=0)
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
      else{ 
        for (i=0; i<size2; i++) tmp_array2[i] = tmp_array[i];
      }

      num = 0;
      for (spin=0; spin<=SpinP_switch; spin++){
	for (n=0; n<Num_Rcv_DM_EGAC[IDR]; n++){

          Gc_AN = Indx_Rcv_DM_EGAC[IDR][n];

          Mc_AN = F_G2M[Gc_AN];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		CDM[spin][Mc_AN][h_AN][i][j] += tmp_array2[num];
		num++;
	      }
	    }
	  }
	}        
      }
    }

    if ( Num_Snd_DM_EGAC[IDS]!=0 && ID!=0 ) MPI_Wait(&request,&stat);
  }

  /*
  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];

        printf("CDM myid =%2d spin=%2d Mc_AN=%2d Gc_AN=%2d h_AN=%2d Gh_AN=%2d\n",
                myid,spin,Mc_AN,Gc_AN,h_AN,Gh_AN);fflush(stdout);

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
            printf("%15.12f ",CDM[spin][Mc_AN][h_AN][i][j]);fflush(stdout);
	  }
          printf("\n");fflush(stdout);
	}      
      }
    }
  }
  */

  /****************************************************
    freeing of arrays:
  ****************************************************/

  free(tmp_array);
  free(tmp_array2);
  free(tmp_ga_array);
  free(tmp_ga_array2);
  free(MP_A);
  free(MP);
  free(GA);
  free(GB);
  free(B);
  free(Mat);
  free(Mat2);

}




static void Calc_GF( char *mode,
                     int myid,
                     int numprocs,
                     double *tmp_ga_array, 
                     double *tmp_ga_array2, 
                     int *MP, int *MP_A,
                     dcomplex *GA,
                     dcomplex *GB,
                     dcomplex *B,
                     dcomplex *Mat, 
                     dcomplex *Mat2, 
                     double Trial_ChemP )
{
  int m,n,i,j,tno1,tno2,Anum,Bnum;
  int h_AN,Gh_AN,Cwan,Hwan,Gi,ih,ian,jan;
  int ig,jg,kl,Mc_AN,k,i0,j0,Gc_AN;
  int spin,job_id,job_gid,num,num2,N3[4];
  int size1,size2,lpole,pole,sigma_loop;
  int M,N,K,LDA,LDB,LDC;  
  int ID,IDS,IDR,tag;
  int dim_matD,dim_matA;
  char nanchar[300];
  dcomplex al,be,alpha,weight;
  MPI_Status stat;
  MPI_Request request;

  /*******************************************************************************
      MPI communication of Green's funcitons associated with the outer region
  *******************************************************************************/

  if ( strcasecmp(mode,"recalc")==0 || 
       strcasecmp(mode,"recalc_nomix")==0 ) {

    tag = 999;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /*****************************
           sending of data 
      *****************************/

      if ( Num_Snd_GA_EGAC[IDS]!=0 ){

	size1 = Snd_GA_EGAC_Size[IDS];

	/* multidimentional array to vector array */

	num = 0;
	for (m=0; m<Num_Snd_GA_EGAC[IDS]; m++){

	  n = Indx_Snd_GA_EGAC[IDS][m]; /* n: global job_id */
	  GN2N_EGAC(n,N3);
	  job_id = n - EGAC_Top[myid]; /* job_id: local job_id */ 

	  Gc_AN = N3[1];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];

	  for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];

	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp_ga_array[num] = GD_EGAC[job_id][h_AN][i][j].r; num++; 
		tmp_ga_array[num] = GD_EGAC[job_id][h_AN][i][j].i; num++;
	      } 
	    } 
	  }
	}

	if (ID!=0) MPI_Isend(&tmp_ga_array[0], size1*2, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
       receiving of block data
      *****************************/

      if ( Num_Rcv_GA_EGAC[IDR]!=0 ){

	size2 = Rcv_GA_EGAC_Size[IDR];

	if (ID!=0)
	  MPI_Recv(&tmp_ga_array2[0], size2*2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
	else{ 
	  for (i=0; i<(size2*2); i++) tmp_ga_array2[i] = tmp_ga_array[i];
	}

	num = 0;
	for (m=0; m<Num_Rcv_GA_EGAC[IDR]; m++){

	  k = Top_Index_GA_EGAC[IDR] + m;       /* serial index of GA */
	  job_gid = Indx_Rcv_GA_EGAC[IDR][m];   /* job_gid */
	  GN2N_EGAC(job_gid,N3); 
	  Gc_AN = N3[1]; 
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];

	  for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];

	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		GA_EGAC[k][h_AN][i][j].r = tmp_ga_array2[num]; num++; 
		GA_EGAC[k][h_AN][i][j].i = tmp_ga_array2[num]; num++;
	      } 
	    } 
	  }
	}
      }

      if ( Num_Snd_GA_EGAC[IDS]!=0 && ID!=0 ) MPI_Wait(&request,&stat);
    }
  }

  /*******************************************
           initialize DM_Snd_EGAC
  *******************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<Matomnum_DM_Snd_EGAC; Mc_AN++){

      Gc_AN = M2G_DM_Snd_EGAC[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
            DM_Snd_EGAC[spin][Mc_AN][h_AN][i][j] = 0.0;
	  }
	}      
      }
    }
  }

  /*******************************************
                 loop of job_id 
  *******************************************/

  for (job_id=0; job_id<EGAC_Num; job_id++){

    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);

    Gc_AN = N3[1];
    spin  = N3[2];
    pole = N3[3];

    if (EGAC_method[pole]==1){

      alpha.r  = Trial_ChemP + EGAC_zp[pole].r/Beta;
      alpha.i  =               EGAC_zp[pole].i/Beta;
      weight.r = -2.0*EGAC_Rp[pole].r/Beta;
      weight.i = -2.0*EGAC_Rp[pole].i/Beta;
    }

    else if (EGAC_method[pole]==2){

      alpha.r  = EGAC_zp[pole].r;
      alpha.i  = EGAC_zp[pole].i;
      weight.r = EGAC_Rp[pole].r; 
      weight.i = EGAC_Rp[pole].i;
    }

    /* MP */
    
    dim_matD = 0;
    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      MP[i] = dim_matD;
      Gi = natn[Gc_AN][i];
      Cwan = WhatSpecies[Gi];
      dim_matD += Spe_Total_CNO[Cwan];
    }

    /* MP_A */

    dim_matA = 0;
    for (i0=0; i0<ONAN[Gc_AN]; i0++){
      MP_A[i0] = dim_matA;
      ig = natn_onan[Gc_AN][i0];
      Cwan = WhatSpecies[ig]; 
      dim_matA += Spe_Total_CNO[Cwan];
    }

#define Sigma_ref(m,job_id,i,j)  Sigma_EGAC[m][job_id][dim_matD*(j)+(i)]
#define fGD_ref(job_id,i,j)      fGD_EGAC[job_id][dim_matD*(j)+(i)]
#define GA_ref(i,j)              GA[dim_matA*(j)+(i)]
#define GB_ref(i,j)              GB[dim_matA*(j)+(i)]
#define B_ref(i,j)               B[dim_matA*(j)+(i)]

    /*********************************************
             self-energy is recalculated
    *********************************************/

    if ( strcasecmp(mode,"recalc")==0 ||
         strcasecmp(mode,"recalc_nomix")==0 ) {

      /* construction of GA */

      for (i0=0; i0<ONAN[Gc_AN]; i0++){

	i = i0 + FNAN[Gc_AN] + SNAN[Gc_AN] + 1;
	ig = natn_onan[Gc_AN][i0];
	ian = Spe_Total_CNO[WhatSpecies[ig]];
	k = L2L_ONAN[job_id][i0];
	Anum = MP_A[i0];

	for (j0=0; j0<ONAN[Gc_AN]; j0++){
      
	  j = j0 + FNAN[Gc_AN] + SNAN[Gc_AN] + 1;
	  jg = natn_onan[Gc_AN][j0];
	  jan = Spe_Total_CNO[WhatSpecies[jg]];
	  Bnum = MP_A[j0];
	  kl = RMI2_EGAC[job_id][i][j]; 

	  if (0<=kl){

	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		GA_ref(Anum+m,Bnum+n) = GA_EGAC[k][kl][m][n];
	      }
	    }
	  }

	  else{
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		GA_ref(Anum+m,Bnum+n).r = 0.0;
		GA_ref(Anum+m,Bnum+n).i = 0.0;
	      }
	    }
	  }

	}
      }

      /* construction of B */

      for (i0=0; i0<ONAN[Gc_AN]; i0++){

	i = i0 + FNAN[Gc_AN] + SNAN[Gc_AN] + 1;
	ig = natn_onan[Gc_AN][i0];
	ian = Spe_Total_CNO[WhatSpecies[ig]];
	ih = G2M_EGAC[ig];
	Anum = MP_A[i0];

	for (j=0; j<=(FNAN[Gc_AN]+SNAN[Gc_AN]); j++){

	  jg = natn[Gc_AN][j];
	  jan = Spe_Total_CNO[WhatSpecies[jg]];
	  Bnum = MP[j];
	  kl = RMI1_EGAC[job_id][i][j]; 

	  if (0<=kl){

	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){

		B_ref(Anum+m,Bnum+n).r = alpha.r*OLP_EGAC[ih][kl][m][n] - H_EGAC[spin][ih][kl][m][n];
		B_ref(Anum+m,Bnum+n).i = alpha.i*OLP_EGAC[ih][kl][m][n];
	      }
	    }
	  }

	  else{

	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		B_ref(Anum+m,Bnum+n).r = 0.0;
		B_ref(Anum+m,Bnum+n).i = 0.0;
	      }
	    }
	  }

	}
      }

      /* C*GA -> Mat */

      al.r = 1.0; al.i = 0.0;
      be.r = 0.0; be.i = 0.0;
      M = dim_matD; N = dim_matA; K = dim_matA; LDA = K; LDB = K; LDC = M;
      if (M!=0 && N!=0 && K!=0){
	F77_NAME(zgemm,ZGEMM)("T","N",&M,&N,&K,&al,B,&LDA,GA,&LDB,&be,Mat,&LDC);
      }

      /* Mat*B -> Mat2 */    

      al.r = 1.0; al.i = 0.0;
      be.r = 0.0; be.i = 0.0;
      M = dim_matD; N = dim_matD; K = dim_matA; LDA = M; LDB = K; LDC = M;
      if (M!=0 && N!=0 && K!=0){
	F77_NAME(zgemm,ZGEMM)("N","N",&M,&N,&K,&al,Mat,&LDA,B,&LDB,&be,Mat2,&LDC);
      }
      else{
	for (i=0; i<dim_matD*dim_matD; i++){
	  Mat2[i].r = 0.0;
	  Mat2[i].i = 0.0;
	}     
      }

      /* fGD_EGAC*Sigma -> Mat */

      al.r = 1.0; al.i = 0.0;
      be.r = 0.0; be.i = 0.0;
      M = dim_matD; N = dim_matD; K = dim_matD; LDA = K; LDB = K; LDC = M;
      if (M!=0 && N!=0 && K!=0){
	F77_NAME(zgemm,ZGEMM)("N","N",&M,&N,&K,&al,fGD_EGAC[job_id],&LDA,Sigma_EGAC[0][job_id],&LDB,&be,Mat,&LDC); 
      }
      
      /* Mat + I -> Mat */

#define Mat_ref(i,j)  Mat[dim_matD*(j)+(i)]

      for (i=0; i<dim_matD; i++){
        Mat_ref(i,i).r = Mat_ref(i,i).r + 1.0;
      }

      /* Mat^-1 -> Mat */

       Lapack_LU_Zinverse(dim_matD,Mat);

      /* shift of Sigma_EGAC */

      for (m=(DIIS_History_EGAC-1); 0<m; m--){
	for (i=0; i<dim_matD*dim_matD; i++){
	  Sigma_EGAC[m][job_id][i] = Sigma_EGAC[m-1][job_id][i];
	}
      }

      /* Mat2*Mat -> Sigma_EGAC[0][job_id] */

      al.r = 1.0; al.i = 0.0;
      be.r = 0.0; be.i = 0.0;
      M = dim_matD; N = dim_matD; K = dim_matD; LDA = K; LDB = K; LDC = M;
      if (M!=0 && N!=0 && K!=0){
	F77_NAME(zgemm,ZGEMM)("N","N",&M,&N,&K,&al,Mat2,&LDA,Mat,&LDB,&be,Sigma_EGAC[0][job_id],&LDC); 
      }

      /* mixing of Sigma_EGAC */

      if ( strcasecmp(mode,"recalc")==0){
        Mixing_Sigma( myid, numprocs, job_id, EGAC_method[pole], alpha.i );
      }
      else if ( strcasecmp(mode,"recalc_nomix")==0){
	for (i=0; i<dim_matD*dim_matD; i++){
	  Sigma_EGAC[1][job_id][i] = Sigma_EGAC[0][job_id][i];
	}     
      }

    } /* end of if (strcasecmp(mode,"recalc")==0 ||  strcasecmp(mode,"recalc_nomix")==0) */


    /* Sigma_EGAC */
    /*
    for (m=0; m<DIIS_History_EGAC; m++){
      for (i=0; i<dim_GD_EGAC[job_id]*dim_GD_EGAC[job_id]; i++){
	Sigma_EGAC[m][job_id][i].r = 0.0;
	Sigma_EGAC[m][job_id][i].i = 0.0;
      }
    }
    */

    /*************************************************
             calculation of (D - Sigma)^{-1}
    *************************************************/

    /* construction of D */

    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){

      ig = natn[Gc_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i];
      ih = G2M_EGAC[ig];

      for (j=0; j<=(FNAN[Gc_AN]+SNAN[Gc_AN]); j++){

	jg = natn[Gc_AN][j];
        jan = Spe_Total_CNO[WhatSpecies[jg]];
        Bnum = MP[j];
        kl = RMI1_EGAC[job_id][i][j];

        if (0<=kl){

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

              fGD_ref(job_id,Anum+m,Bnum+n).r = alpha.r*OLP_EGAC[ih][kl][m][n] - H_EGAC[spin][ih][kl][m][n];
	      fGD_ref(job_id,Anum+m,Bnum+n).i = alpha.i*OLP_EGAC[ih][kl][m][n];
	    }
	  }
	}

        else{

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
              fGD_ref(job_id,Anum+m,Bnum+n).r = 0.0;
	      fGD_ref(job_id,Anum+m,Bnum+n).i = 0.0;
	    }
	  }
	} 

      }
    }    

    /* GD - Sigma -> fGD_EGAC */

    for (i=0; i<dim_matD; i++){
      for (j=0; j<dim_matD; j++){
	fGD_ref(job_id,i,j).r = fGD_ref(job_id,i,j).r - Sigma_ref(0,job_id,i,j).r;
	fGD_ref(job_id,i,j).i = fGD_ref(job_id,i,j).i - Sigma_ref(0,job_id,i,j).i;
      }
    }


    /*
    if (job_gid==0){

      printf("VVV1 Sigma.r\n");
      for (i=0; i<dim_matD; i++){
	for (j=0; j<dim_matD; j++){
	  printf("%10.5f ",Sigma_ref(0,job_id,i,j).r);
	}
        printf("\n");
      }
    }
    */

    /* calculate the inverse of (GD - Sigma)^{-1} */

    Lapack_LU_Zinverse(dim_matD,fGD_EGAC[job_id]);

    /* calculation of the partial DM */

    Mc_AN = G2M_DM_Snd_EGAC[Gc_AN]; 
    Cwan = WhatSpecies[Gc_AN];
    tno1 = Spe_Total_CNO[Cwan];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno2 = Spe_Total_CNO[Hwan];
      Bnum = MP[h_AN];

      for (i=0; i<tno1; i++){
	for (j=0; j<tno2; j++){

	  DM_Snd_EGAC[spin][Mc_AN][h_AN][i][j] += fGD_ref(job_id,i,Bnum+j).r*weight.r
	                                        - fGD_ref(job_id,i,Bnum+j).i*weight.i;
	}
      }
    }

    /* copy the Green's function associated with the central atom to GD_EGAC */

    for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno2 = Spe_Total_CNO[Hwan];
      Bnum = MP[h_AN];
       
      for (i=0; i<tno1; i++){
        for (j=0; j<tno2; j++){

          GD_EGAC[job_id][h_AN][i][j] = fGD_ref(job_id,i,Bnum+j);

          sprintf(nanchar,"%8.4f",GD_EGAC[job_id][h_AN][i][j].r);
	  if (strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	      || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

	    printf("ABC1 myid=%2d job_id=%2d job_gid=%2d Gc_AN=%2d i=%2d j=%2d alpha=%15.12f %15.12f\n",
		   myid,job_id,job_gid,Gc_AN,i,j,alpha.r,alpha.i );fflush(stdout);

            MPI_Finalize();
            exit(0);

	  }

	}
      }
    }

  } /* end of job_id */  

}







double Calc_Number_of_Electrons()
{
  int spin,Mc_AN,Gc_AN,Cwan,tno1,h_AN;
  int Gh_AN,Hwan,tno2,i,j,Mc_AN_OLP;
  double my_N,N;

  my_N  = 0.0;

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=0; Mc_AN<Matomnum_DM_Snd_EGAC; Mc_AN++){

      Gc_AN = M2G_DM_Snd_EGAC[Mc_AN];
      Mc_AN_OLP = G2M_EGAC[Gc_AN]; 

      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];

	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
            my_N  +=  DM_Snd_EGAC[spin][Mc_AN][h_AN][i][j]*OLP_EGAC[Mc_AN_OLP][h_AN][i][j];
	  }
	}      
      }
    }
  }

  MPI_Allreduce(&my_N,  &N,  1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  if (SpinP_switch==0) N *= 2.0;

  return N;  
}


static void Mixing_Sigma(int myid, int numprocs, int job_id, int method, double alphai)
{
  int i,dim_matD;
  double coe0,coe1;
  double cutval;

  coe0 = 1.0;
  coe1 = 1.0 - coe0;

  /* simple mixing of Sigma_EGAC */

  dim_matD = dim_GD_EGAC[job_id];

  for (i=0; i<dim_matD*dim_matD; i++){
    Sigma_EGAC[0][job_id][i].r = coe0*Sigma_EGAC[0][job_id][i].r + coe1*Sigma_EGAC[1][job_id][i].r;  
    Sigma_EGAC[0][job_id][i].i = coe0*Sigma_EGAC[0][job_id][i].i + coe1*Sigma_EGAC[1][job_id][i].i;  
  }

  /*
  cutval = 1.0e+5;

  if (method==1){

    if (scf_GF_EGAC==0){
      coe0 = 0.0;
      coe1 = 1.0 - coe0;
    }
    else {
      if (alphai<10.0){
        coe0 = 0.3;
        coe1 = 1.0 - coe0;
      }
      else{
        coe0 = 0.0;
        coe1 = 1.0 - coe0;
      }
    }
  }

  else if (method==2){
    coe0 = 0.0;
    coe1 = 1.0 - coe0;
  }
  */

  /* simple mixing of Sigma_EGAC */

  /*
  dim_matD = dim_GD_EGAC[job_id];

  for (i=0; i<dim_matD*dim_matD; i++){
    Sigma_EGAC[0][job_id][i].r = coe0*Sigma_EGAC[0][job_id][i].r + coe1*Sigma_EGAC[1][job_id][i].r;  
    Sigma_EGAC[0][job_id][i].i = coe0*Sigma_EGAC[0][job_id][i].i + coe1*Sigma_EGAC[1][job_id][i].i;  

    if (cutval<fabs(Sigma_EGAC[0][job_id][i].r)) Sigma_EGAC[0][job_id][i].r = sgn(Sigma_EGAC[0][job_id][i].r)*cutval;
    if (cutval<fabs(Sigma_EGAC[0][job_id][i].i)) Sigma_EGAC[0][job_id][i].i = sgn(Sigma_EGAC[0][job_id][i].i)*cutval;
  } 
  */    


}




static double Find_Trial_ChemP_by_Muller(
        int SCF_iter, int num_loop,   
        double pTrial_ChemP[4], double pDiff_Num[5], 
        double DeltaMu[7], double DeltaN[3], double DeltaNp[3])
{
  static double stepw;
  double y31,x0,x1,x2,x3,y0,y1,y2,y3;
  int i,i0,i1,i2,num4,po3,po4;
  double a,b,c,d,g0,g1,g2,dFa,dFb,dFc,dF,dF0,F;
  double xx[5],yy[5],zz[5],yy0[5];
  double z0,z1,z2,a1,a2,a3,det;
  double a11,a12,a13,a21,a22,a23,a31,a32,a33;
  double scaling4;
  double Trial_ChemP,Trial_ChemP0;
  int flag_nan;
  char nanchar[300];
    
  Trial_ChemP = pTrial_ChemP[3];

  x0 = pTrial_ChemP[0];
  x1 = pTrial_ChemP[1];
  x2 = pTrial_ChemP[2];
  x3 = pTrial_ChemP[3];

  y0  = pDiff_Num[0];
  y1  = pDiff_Num[1];
  y2  = pDiff_Num[2];
  y3  = pDiff_Num[3];
  y31 = pDiff_Num[4];

  /*******************************************************
               store history of stepw and y3
  *******************************************************/

  if (num_loop==2){

    DeltaMu[2] = DeltaMu[1];
    DeltaN[2]  = DeltaN[1];
    DeltaNp[2] = DeltaNp[1];

    DeltaMu[1] = DeltaMu[0];
    DeltaN[1]  = DeltaN[0];
    DeltaNp[1] = DeltaNp[0];
  
    DeltaMu[0] = DeltaMu[6];
    DeltaN[0]  = y31;
    DeltaNp[0] = y3;
  }

  /*******************************************************
                estimate a new chemical potential
  *******************************************************/

  /* num_loop==1 */

  if (num_loop==1){

    y31 = y3; 

    if (SCF_iter<4){

      stepw = fabs(y3);
      if (0.02<stepw) stepw = -sgn(y3)*0.02;
      else            stepw = -(y3 + 10.0*y3*y3);

      Trial_ChemP = Trial_ChemP + stepw;
    }

    else {

      /* if (4<=SCF_iter), estimate Trial_ChemP using an extrapolation. */

      z0 = DeltaNp[0];
      z1 = DeltaNp[1];
      z2 = DeltaNp[2];

      a11 = DeltaN[0]; a12 = DeltaMu[0]; a13 = DeltaN[0]*DeltaMu[0];
      a21 = DeltaN[1]; a22 = DeltaMu[1]; a23 = DeltaN[1]*DeltaMu[1];
      a31 = DeltaN[2]; a32 = DeltaMu[2]; a33 = DeltaN[2]*DeltaMu[2];

      det = a11*a22*a33+a21*a32*a13+a31*a12*a23-a11*a32*a23-a31*a22*a13-a21*a12*a33;

      a1 = ((a22*a33-a23*a32)*z0 + (a13*a32-a12*a33)*z1 + (a12*a23-a13*a22)*z2)/det;
      a2 = ((a23*a31-a21*a33)*z0 + (a11*a33-a13*a31)*z1 + (a13*a21-a11*a23)*z2)/det;
      a3 = ((a21*a32-a22*a31)*z0 + (a12*a31-a11*a32)*z1 + (a11*a22-a12*a21)*z2)/det;

      stepw = -a1*y3/(a2+a3*y3);

      flag_nan = 0;
      sprintf(nanchar,"%8.4f",stepw);

      if (strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	  || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

	flag_nan = 1;

	stepw = fabs(y3);
	if (0.02<stepw) stepw = -sgn(y3)*0.02;
	else            stepw = -(y3 + y3*y3);
	Trial_ChemP = Trial_ChemP + stepw;
      }
      else {
	Trial_ChemP = Trial_ChemP + stepw;
      }

    }

    x0 = x3;
    y0 = y3;
  } 

  /* num_loop==2 */

  else if (num_loop==2){

    x1 = x3;
    y1 = y3;

    Trial_ChemP0 = (x1*y0-x0*y1)/(y0-y1);

    if ( 0.1<fabs(Trial_ChemP-Trial_ChemP0) ){
      Trial_ChemP = Trial_ChemP + 0.1*sgn(Trial_ChemP0 - Trial_ChemP);
    }
    else {

      Trial_ChemP = Trial_ChemP0;
    }
  }

  /* num_loop==3 */

  else if (num_loop==3){

    x2 = x3;
    y2 = y3;

    a = y0/(x0-x2)/(x0-x1) - y1/(x1-x2)/(x0-x1) + y2/(x0-x2)/(x1-x2);
    b = -(x2+x1)*y0/(x0-x2)/(x0-x1) + (x0+x2)*y1/(x1-x2)/(x0-x1) - (x1+x0)*y2/(x0-x2)/(x1-x2);
    c = x1*x2*y0/(x0-x2)/(x0-x1) - x2*x0*y1/(x1-x2)/(x0-x1) + x0*x1*y2/(x0-x2)/(x1-x2);

    /* refinement of a, b, and c by the iterative method */

    po4 = 0;
    num4 = 0;
    scaling4 = 0.3;
    dF = 1000.0;

    do {

      g0 = a*x0*x0 + b*x0 + c - y0;
      g1 = a*x1*x1 + b*x1 + c - y1;
      g2 = a*x2*x2 + b*x2 + c - y2;

      dFa = 2.0*(g0*x0*x0 + g1*x1*x1 + g2*x2*x2);   
      dFb = 2.0*(g0*x0 + g1*x1 + g2*x2);
      dFc = 2.0*(g0 + g1 + g2);

      F = g0*g0 + g1*g1 + g2*g2;
      dF0 = dF;
      dF = sqrt(fabs(dFa*dFa + dFb*dFb + dFc*dFc));

      if (dF0<dF) scaling4 /= 1.5;

      /*
        printf("3 F=%18.15f dF=%18.14f scaling4=%15.12f\n",F,dF,scaling4);
      */

      a -= scaling4*dFa;
      b -= scaling4*dFb;
      c -= scaling4*dFc;

      if (dF<1.0e-11) po4 = 1;
      num4++; 

    } while (po4==0 && num4<100);

    /* calculte Trial_ChemP */

    /*
      printf("3 a=%18.15f\n",a); 
      printf("3 b=%18.15f\n",b); 
      printf("3 c=%18.15f\n",c); 
    */

    if (0.0<=b)
      Trial_ChemP0 = -2.0*c/(b+sqrt(fabs(b*b-4*a*c)));
    else 
      Trial_ChemP0 = (-b+sqrt(fabs(b*b-4*a*c)))/(2.0*a); 

    flag_nan = 0;
    sprintf(nanchar,"%8.4f",Trial_ChemP0);

    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	   || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;

      stepw = fabs(y3);
      if (0.02<stepw) stepw = -sgn(y3)*0.02;
      else            stepw = -(y3 + y3*y3);
      Trial_ChemP = Trial_ChemP + stepw;
    }
    else {

      if ( 0.02<fabs(Trial_ChemP-Trial_ChemP0) ){
	Trial_ChemP = Trial_ChemP + 0.02*sgn(Trial_ChemP0 - Trial_ChemP);
      }
      else {
	Trial_ChemP = Trial_ChemP0;
      }
    }

    /*
      printf("3 Trial_ChemP0=%18.15f\n",Trial_ChemP0); 
      printf("3 Trial_ChemP =%18.15f\n",Trial_ChemP); 
    */

  }

  /* if (4<=num_loop) */

  else {

    Trial_ChemP0 = Trial_ChemP;

    xx[1] = x0;
    xx[2] = x1;
    xx[3] = x2;
    xx[4] = x3;

    yy[1] = y0;
    yy[2] = y1;
    yy[3] = y2; 
    yy[4] = y3; 

    qsort_double(4, xx, yy);      

    /* check whether the zero point is passed or not. */

    po3 = 0; 
    for (i=1; i<4; i++){
      if (yy[i]*yy[i+1]<0.0){
	i0 = i;
	po3 = 1;
      }
    } 

    /*
      for (i=1; i<=4; i++){
      printf("num_loop=%2d i0=%2d i=%2d xx=%18.15f yy=%18.15f\n",num_loop,i0,i,xx[i],yy[i]);
      }
    */
      
    if (po3==1){

      if (i0==1){
	x0 = xx[i0+0];
	x1 = xx[i0+1];
	x2 = xx[i0+2];
 
	y0 = yy[i0+0];
	y1 = yy[i0+1];
	y2 = yy[i0+2];
      }

      else if (i0==2) {

	if (fabs(yy[i0-1])<fabs(yy[i0+2])){
          
	  x0 = xx[i0-1];
	  x1 = xx[i0+0];
	  x2 = xx[i0+1];
 
	  y0 = yy[i0-1];
	  y1 = yy[i0+0];
	  y2 = yy[i0+1];
	}
	else {
	  x0 = xx[i0+0];
	  x1 = xx[i0+1];
	  x2 = xx[i0+2];
 
	  y0 = yy[i0+0];
	  y1 = yy[i0+1];
	  y2 = yy[i0+2];
	}
      }

      else if (i0==3) {
	x0 = xx[i0-1];
	x1 = xx[i0+0];
	x2 = xx[i0+1];
 
	y0 = yy[i0-1];
	y1 = yy[i0+0];
	y2 = yy[i0+1];
      }

      a = y0/(x0-x2)/(x0-x1) - y1/(x1-x2)/(x0-x1) + y2/(x0-x2)/(x1-x2);
      b = -(x2+x1)*y0/(x0-x2)/(x0-x1) + (x0+x2)*y1/(x1-x2)/(x0-x1) - (x1+x0)*y2/(x0-x2)/(x1-x2);
      c = x1*x2*y0/(x0-x2)/(x0-x1) - x2*x0*y1/(x1-x2)/(x0-x1) + x0*x1*y2/(x0-x2)/(x1-x2);

      /*
	printf("x0=%18.15f y0=%18.15f\n",x0,y0); 
	printf("x1=%18.15f y1=%18.15f\n",x1,y1); 
	printf("x2=%18.15f y2=%18.15f\n",x2,y2); 

	printf("a=%18.15f\n",a); 
	printf("b=%18.15f\n",b); 
	printf("c=%18.15f\n",c); 
      */

      /* refinement of a, b, and c by the iterative method */

      po4 = 0;
      num4 = 0;
      scaling4 = 0.3;
      dF = 1000.0;

      do {

	g0 = a*x0*x0 + b*x0 + c - y0;
	g1 = a*x1*x1 + b*x1 + c - y1;
	g2 = a*x2*x2 + b*x2 + c - y2;

	dFa = 2.0*(g0*x0*x0 + g1*x1*x1 + g2*x2*x2);   
	dFb = 2.0*(g0*x0 + g1*x1 + g2*x2);
	dFc = 2.0*(g0 + g1 + g2);

	F = g0*g0 + g1*g1 + g2*g2;
	dF0 = dF;
	dF = sqrt(fabs(dFa*dFa + dFb*dFb + dFc*dFc));

	if (dF0<dF) scaling4 /= 1.5;

	/*
          printf("F=%18.15f dF=%18.14f scaling4=%15.12f\n",F,dF,scaling4);
	*/

	a -= scaling4*dFa;
	b -= scaling4*dFb;
	c -= scaling4*dFc;

	if (dF<1.0e-11) po4 = 1;
	num4++; 

      } while (po4==0 && num4<100);

      /* calculte Trial_ChemP */

      if (0.0<=b)
	Trial_ChemP0 = -2.0*c/(b+sqrt(fabs(b*b-4*a*c)));
      else 
	Trial_ChemP0 = (-b+sqrt(fabs(b*b-4*a*c)))/(2.0*a); 

      flag_nan = 0;
      sprintf(nanchar,"%8.4f",Trial_ChemP0);

      if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	     || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

	flag_nan = 1;

	stepw = fabs(y3);
	if (0.02<stepw) stepw = -sgn(y3)*0.02;
	else            stepw = -(y3 + y3*y3);
	Trial_ChemP = Trial_ChemP + stepw;
      }
      else {

	if ( 0.02<fabs(Trial_ChemP-Trial_ChemP0) ){
	  Trial_ChemP = Trial_ChemP + 0.02*sgn(Trial_ChemP0 - Trial_ChemP);
	}
	else {
	  Trial_ChemP = Trial_ChemP0;
	}
      }

      /*
        printf("F=%18.15f dF=%18.14f Trial_ChemP=%18.15f\n",F,dF,Trial_ChemP);
      */
    }

    else {

      yy[1] = fabs(y0);
      yy[2] = fabs(y1);
      yy[3] = fabs(y2); 
      yy[4] = fabs(y3); 

      zz[1] = 1;
      zz[2] = 2;
      zz[3] = 3;
      zz[4] = 4;

      xx[1] = x0;
      xx[2] = x1;
      xx[3] = x2;
      xx[4] = x3;

      yy0[1] = y0;
      yy0[2] = y1;
      yy0[3] = y2; 
      yy0[4] = y3; 

      qsort_double(4, yy, zz);

      i0 = (int)zz[1];
      i1 = (int)zz[2];
      i2 = (int)zz[3];

      x0 = xx[i0];
      x1 = xx[i1];
      x2 = xx[i2];

      y0 = yy0[i0];
      y1 = yy0[i1];
      y2 = yy0[i2];

      Trial_ChemP0 = (x1*y0-x0*y1)/(y0-y1);

      flag_nan = 0;
      sprintf(nanchar,"%8.4f",Trial_ChemP0);

      if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	     || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

	flag_nan = 1;

	stepw = fabs(y3);
	if (0.02<stepw) stepw = -sgn(y3)*0.02;
	else            stepw = -(y3 + y3*y3);
	Trial_ChemP = Trial_ChemP + stepw;
      }
      else {

	if ( 0.02<fabs(Trial_ChemP-Trial_ChemP0) ){
	  Trial_ChemP = Trial_ChemP + 0.02*sgn(Trial_ChemP0 - Trial_ChemP);
	}
	else {
	  Trial_ChemP = Trial_ChemP0;
	}
      }

    } /* else */
  } /* else */

  /* store pTrial_ChemP and pDiff_Num */

  pTrial_ChemP[0] = x0;
  pTrial_ChemP[1] = x1;
  pTrial_ChemP[2] = x2;
  pTrial_ChemP[3] = x3;

  pDiff_Num[0] = y0;
  pDiff_Num[1] = y1;
  pDiff_Num[2] = y2;
  pDiff_Num[3] = y3;
  pDiff_Num[4] = y31;

  DeltaMu[6] = stepw;

  /* return Traial_ChemP */
  return Trial_ChemP;
}
