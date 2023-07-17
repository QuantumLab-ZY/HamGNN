/**********************************************************************
  Set_Hamiltonian.c:

     Set_Hamiltonian.c is a subroutine to make Hamiltonian matrix
     within LDA or GGA.

  Log of Set_Hamiltonian.c:

     24/April/2002  Released by T. Ozaki
     17/April/2013  Modified by A.M. Ito

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time   0

void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind);


double Set_Hamiltonian(char *mode,
                       int MD_iter,
                       int SCF_iter,
                       int SCF_iter0,
                       int TRAN_Poisson_flag2,
                       int SucceedReadingDMfile,
                       int Cnt_kind,
                       double *****H0,
                       double *****HNL,
                       double *****CDM,
		       double *****H)
{
  /***************************************************************
      Cnt_kind
        0:  Uncontracted Hamiltonian    
        1:  Contracted Hamiltonian    
  ***************************************************************/

  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int i,j,k,Cwan,Hwan,NO0,NO1,spin,N,NOLG;
  int Nc,Ncs,GNc,GRc,Nog,Nh,MN,XC_P_switch;
  double TStime,TEtime;
  int numprocs,myid;
  double time0,time1,time2,mflops;
  long Num_C0,Num_C1;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  if (myid==Host_ID && strcasecmp(mode,"stdout")==0 && 0<level_stdout ){
    printf("<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...\n");fflush(stdout);
  }

  /*****************************************************
                adding H0+HNL+(HCH) to H 
  *****************************************************/

  if(measure_time) dtime(&time1);

  /* spin non-collinear */

  if (SpinP_switch==3){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){

            if (ProExpn_VNA==0){
              H[0][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
 	                              + F_NL_flag*HNL[0][Mc_AN][h_AN][i][j];
              H[1][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
		                      + F_NL_flag*HNL[1][Mc_AN][h_AN][i][j];
              H[2][Mc_AN][h_AN][i][j] = F_NL_flag*HNL[2][Mc_AN][h_AN][i][j];
              H[3][Mc_AN][h_AN][i][j] = 0.0;
	    }
            else{
              H[0][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
	 	                      + F_VNA_flag*HVNA[Mc_AN][h_AN][i][j]
		                      + F_NL_flag*HNL[0][Mc_AN][h_AN][i][j];
              H[1][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
		                      + F_VNA_flag*HVNA[Mc_AN][h_AN][i][j]
		                      + F_NL_flag*HNL[1][Mc_AN][h_AN][i][j];
              H[2][Mc_AN][h_AN][i][j] = F_NL_flag*HNL[2][Mc_AN][h_AN][i][j];
              H[3][Mc_AN][h_AN][i][j] = 0.0;
            }

            /* Effective Hubbard Hamiltonain --- added by MJ */

	    if( (Hub_U_switch==1 || 1<=Constraint_NCS_switch) && F_U_flag==1 && 2<=SCF_iter ){
	      H[0][Mc_AN][h_AN][i][j] += H_Hub[0][Mc_AN][h_AN][i][j];
	      H[1][Mc_AN][h_AN][i][j] += H_Hub[1][Mc_AN][h_AN][i][j];
	      H[2][Mc_AN][h_AN][i][j] += H_Hub[2][Mc_AN][h_AN][i][j];
	    }

	    /* core hole Hamiltonain */

	    if (core_hole_state_flag==1){
	      H[0][Mc_AN][h_AN][i][j] += HCH[0][Mc_AN][h_AN][i][j];
	      H[1][Mc_AN][h_AN][i][j] += HCH[1][Mc_AN][h_AN][i][j];
	      H[2][Mc_AN][h_AN][i][j] += HCH[2][Mc_AN][h_AN][i][j];
	    }

          }
        }
      }
    }
  }

  /* spin collinear */

  else{

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){
            for (spin=0; spin<=SpinP_switch; spin++){

              if (ProExpn_VNA==0){
                H[spin][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
		                           + F_NL_flag*HNL[spin][Mc_AN][h_AN][i][j];
	      }
              else{
                H[spin][Mc_AN][h_AN][i][j] = F_Kin_flag*H0[0][Mc_AN][h_AN][i][j]
		                           + F_VNA_flag*HVNA[Mc_AN][h_AN][i][j]
		                           + F_NL_flag*HNL[spin][Mc_AN][h_AN][i][j];
              }

	      /* Effective Hubbard Hamiltonain --- added by MJ */
	      if ( (Hub_U_switch==1 || 1<=Constraint_NCS_switch) && F_U_flag==1 && 2<=SCF_iter ){
		H[spin][Mc_AN][h_AN][i][j] += H_Hub[spin][Mc_AN][h_AN][i][j];
	      }

	      /* core hole Hamiltonain */
	      if (core_hole_state_flag==1){
		H[spin][Mc_AN][h_AN][i][j] += HCH[spin][Mc_AN][h_AN][i][j];
	      }

            }
          }
        }
      }
    }

  }

  if(measure_time){ 
    dtime(&time2);
    printf("myid=%4d Time1=%18.10f\n",myid,time2-time1);fflush(stdout);
  }

  if (Cnt_kind==1) {
    Contract_Hamiltonian( H, CntH, OLP, CntOLP );
    if (SO_switch==1) Contract_iHNL(iHNL,iCntHNL);
  }

  /*****************************************************
   calculation of Vpot;
  *****************************************************/

  if(myid==0 && measure_time)  dtime(&time1);

  XC_P_switch = 1;
  Set_Vpot(MD_iter,SCF_iter,SCF_iter0,TRAN_Poisson_flag2,XC_P_switch);

  if(measure_time){ 
    dtime(&time2);
    printf("myid=%4d Time2=%18.10f\n",myid,time2-time1);fflush(stdout);
  }

  /*****************************************************
   calculation of matrix elements for dVH + Vxc (+ VNA)
  *****************************************************/

  Calc_MatrixElements_dVH_Vxc_VNA(Cnt_kind);

  /* for time */
  if (measure_time) dtime(&time1);
  MPI_Barrier(mpi_comm_level1);
  if(measure_time) {
    dtime(&time2);
    printf("myid=%4d Time Barrier=%18.10f\n",myid,time2-time1);
    fflush(stdout);
  }

  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}


void Calc_MatrixElements_dVH_Vxc_VNA(int Cnt_kind)
{
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int Nh0,Nh1,Nh2,Nh3;
  int Nc0,Nc1,Nc2,Nc3;
  int MN0,MN1,MN2,MN3;
  int Nloop,OneD_Nloop;
  int *OneD2spin,*OneD2Mc_AN,*OneD2h_AN;
  int numprocs,myid;
  double time0,time1,time2,mflops;

  if(measure_time) dtime(&time1);

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* one-dimensionalization of loops */

  Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Nloop++;
    }
  }

  OneD2Mc_AN = (int*)malloc(sizeof(int)*Nloop);
  OneD2h_AN = (int*)malloc(sizeof(int)*Nloop);

  Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      OneD2Mc_AN[Nloop] = Mc_AN;
      OneD2h_AN[Nloop] = h_AN;
      Nloop++;
    }
  }

  OneD_Nloop = Nloop;

  if(measure_time){ 
    dtime(&time2);
    printf("myid=%4d Time3=%18.10f\n",myid,time2-time1);fflush(stdout);
  }

  /* numerical integration */

  if(measure_time) dtime(&time1);

#pragma omp parallel 
  {
    int Nloop,spin,Mc_AN,h_AN,Gh_AN,Mh_AN,Hwan,NOLG;
    int Gc_AN,Cwan,NO0,NO1,spin0=-1,Mc_AN0=0;
    int i,j,Nc,MN,GNA,Nog,Nh,OMPID,Nthrds;
    int M,N,K,lda,ldb,ldc,ii,jj;
    double alpha,beta,Vpot;
    double sum0,sum1,sum2,sum3,sum4;
    double *ChiV0,*Chi1,*ChiV0_2,*C;

    /* allocation of arrays */

    /* AITUNE */
    double **AI_tmpH[4];
    {
      /* get size of temporary buffer */
      int AI_MaxNO = 0;
      if (Cnt_kind==0){
	int spe;
	for(spe = 0; spe < SpeciesNum; spe++){
	  if(AI_MaxNO < Spe_Total_NO[spe]){ AI_MaxNO = Spe_Total_NO[spe];}
	}
      }else{
	int spe;
	for(spe = 0; spe < SpeciesNum; spe++){
	  if(AI_MaxNO < Spe_Total_CNO[spe]){ AI_MaxNO = Spe_Total_CNO[spe];}
	}
      }
			
      int spin;
      for (spin=0; spin<=SpinP_switch; spin++){
	AI_tmpH[spin] = (double**)malloc(sizeof(double*)*AI_MaxNO);

	int i;
	double *p = (double*)malloc(sizeof(double)*AI_MaxNO*AI_MaxNO);
	for(i = 0; i < AI_MaxNO; i++){
	  AI_tmpH[spin][i] = p;
	  p += AI_MaxNO;
	}
      }			
    }
    /* AITUNE */

    /* starting of one-dimensionalized loop */

#pragma omp for schedule(static,1)  /* guided */  /* AITUNE */
    for (Nloop = 0; Nloop < OneD_Nloop; Nloop++){ /* AITUNE */

      int Mc_AN = OneD2Mc_AN[Nloop];
      int h_AN = OneD2h_AN[Nloop];
      int Gc_AN = M2G[Mc_AN];    
      int Gh_AN = natn[Gc_AN][h_AN];
      int Mh_AN = F_G2M[Gh_AN];
      int Cwan = WhatSpecies[Gc_AN];
      int Hwan = WhatSpecies[Gh_AN];
      int GNA = GridN_Atom[Gc_AN];
      int NOLG = NumOLG[Mc_AN][h_AN]; 

      int NO0,NO1;
      if (Cnt_kind==0){
	NO0 = Spe_Total_NO[Cwan];
	NO1 = Spe_Total_NO[Hwan];
      }
      else{
	NO0 = Spe_Total_CNO[Cwan];
	NO1 = Spe_Total_CNO[Hwan];
      }

      /* quadrature for Hij  */

      /* AITUNE change order of loop */
      if(SpinP_switch==0){
	/* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */
		
	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
	    }
	  }
	}

	int Nog;
	for (Nog=0; Nog<NOLG; Nog++){

	  int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  int MN = MGridListAtom[Mc_AN][Nc];
	  int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
		
	  double AI_tmp_GVVG = GridVol * Vpot_Grid[0][MN];


	  if (G2ID[Gh_AN]==myid){
	    int i;
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
	      int j;

	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	  }else{
	    int i;
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
	      int j;

	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	  }
		
	}/* Nog */
	
	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	}

      }else if (SpinP_switch == 1){
	  
	/* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */
		
	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
	    }
	  }
	}

	int Nog;
	for (Nog=0; Nog<NOLG; Nog++){

	  int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  int MN = MGridListAtom[Mc_AN][Nc];
	  int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
		
	  double AI_tmp_GVVG = GridVol * Vpot_Grid[0][MN];
	  double AI_tmp_GVVG1 = GridVol * Vpot_Grid[1][MN];


	  if (G2ID[Gh_AN]==myid){
				
	    int i;
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
	      int j;
	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }

	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG1 * Orbs_Grid[Mc_AN][Nc][i];

	      int j;
	      for (j=0; j<NO1; j++){
		AI_tmpH[1][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	  }else{
	    int i;
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];
	      int j;
	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
			
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG1 * Orbs_Grid[Mc_AN][Nc][i];
	      int j;
	      for (j=0; j<NO1; j++){
		AI_tmpH[1][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	  }

		
	}/* Nog */
	
	/* AITUNE copy from temporary buffer */

	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
	    }
	  }
	}

      }

      else{ /* SpinP_switch==3 */
	 
	/* AITUNE temporary buffer for "unroll-Jammed" HLO optimization by Intel */

	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
	      AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
	      AI_tmpH[2][i][j] = H[2][Mc_AN][h_AN][i][j];
	      AI_tmpH[3][i][j] = H[3][Mc_AN][h_AN][i][j];
	    }
	  }
	}

        else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
	      AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
	      AI_tmpH[2][i][j] = CntH[2][Mc_AN][h_AN][i][j];
	      AI_tmpH[3][i][j] = CntH[3][Mc_AN][h_AN][i][j];
	    }
	  }
	}

	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = H[0][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[1][i][j] = H[1][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[2][i][j] = H[2][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[3][i][j] = H[3][Mc_AN][h_AN][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[0][i][j] = CntH[0][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[1][i][j] = CntH[1][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[2][i][j] = CntH[2][Mc_AN][h_AN][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      AI_tmpH[3][i][j] = CntH[3][Mc_AN][h_AN][i][j];
	    }
	  }
			
	}

	int Nog;

	for (Nog=0; Nog<NOLG; Nog++){

	  int Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  int MN = MGridListAtom[Mc_AN][Nc];
	  int Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
		
	  double AI_tmp_GVVG = GridVol * Vpot_Grid[0][MN];
	  double AI_tmp_GVVG1 = GridVol * Vpot_Grid[1][MN];
	  double AI_tmp_GVVG2 = GridVol * Vpot_Grid[2][MN];
	  double AI_tmp_GVVG3 = GridVol * Vpot_Grid[3][MN];

	  if (G2ID[Gh_AN]==myid){

	    int i;
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG1 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[1][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG2 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[2][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	    for (i=0; i<NO0; i++){

	      double AI_tmp_i = AI_tmp_GVVG3 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[3][i][j] += AI_tmp_i * Orbs_Grid[Mh_AN][Nh][j];
	      }		
	    }
			
	  }

          else{

	    int i;
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[0][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
			
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG1 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[1][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG2 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[2][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
		
	      }
	    }			
	    for (i=0; i<NO0; i++){
					
	      double AI_tmp_i = AI_tmp_GVVG3 * Orbs_Grid[Mc_AN][Nc][i];

	      for (j=0; j<NO1; j++){
		AI_tmpH[3][i][j] += AI_tmp_i * Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	  }

		
	} /* Nog */
	
	/* AITUNE copy from temporary buffer */

	if (Cnt_kind==0){
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      H[3][Mc_AN][h_AN][i][j] = AI_tmpH[3][i][j];
	    }
	  }
	}else{
	  int i;
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[0][Mc_AN][h_AN][i][j] = AI_tmpH[0][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[1][Mc_AN][h_AN][i][j] = AI_tmpH[1][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[2][Mc_AN][h_AN][i][j] = AI_tmpH[2][i][j];
	    }
	  }
	  for (i=0; i<NO0; i++){
	    int j;
	    for (j=0; j<NO1; j++){
	      CntH[3][Mc_AN][h_AN][i][j] = AI_tmpH[3][i][j];
	    }
	  }
	}
      }
      /* AITUNE change order of loop */

    } /* Nloop */

    /* freeing of arrays */
    {
      int spin;
      for (spin=0; spin<=SpinP_switch; spin++){
	free(AI_tmpH[spin][0]);
	free(AI_tmpH[spin]);
      }
    }

  } /* pragma omp parallel */ 

  /* freeing of arrays */

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  if(measure_time){ 
    dtime(&time2);
    printf("myid=%4d Time4=%18.10f\n",myid,time2-time1);fflush(stdout);
  }
}
