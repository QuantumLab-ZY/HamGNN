/**********************************************************************
  Total_Energy.c:

     Total_Energy.c is a subrutine to calculate the total energy

  Log of Total_Energy.c:

     22/Nov/2001  Released by T. Ozaki
     19/Feb/2006  The subroutine name 'Correction_Energy' was changed 
                  to 'Total_Energy'

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time   0
#define  Num_Leb_Grid  590

static double Calc_Ecore();
static double Calc_EH0(int MD_iter);
static double Calc_Ekin();
static double Calc_Ena();
static double Calc_Enl();
static double Calc_ECH();
static void Calc_EXC_EH1(double ECE[],double *****CDM);
static double Calc_Ehub();   /* --- added by MJ  */
static double Calc_EdftD();  /* added by okuno */
static double Calc_EdftD3(); /* added by Ellner */
static void EH0_TwoCenter(int Gc_AN, int h_AN, double VH0ij[4]);
static void EH0_TwoCenter_at_Cutoff(int wan1, int wan2, double VH0ij[4]);
static void Set_Lebedev_Grid();
static void Energy_Decomposition(double ECE[]);

double Leb_Grid_XYZW[Num_Leb_Grid][4];

/* for OpenMP */
int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;




double Total_Energy(int MD_iter, double *****CDM, double ECE[])
{ 
  double time0;
  double TStime,TEtime;
  int numprocs,myid;
  int Mc_AN,Gc_AN,h_AN;
  double stime,etime;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  /****************************************************
   For OpenMP:
   making of arrays of the one-dimensionalized loop
  ****************************************************/

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

  /****************************************************
               core-core repulsion energy
  ****************************************************/

  dtime(&stime);

  ECE[0] = Calc_Ecore();

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Ecore=%18.5f\n",etime-stime);fflush(stdout);
  } 
  
  /****************************************************
              EH0 = -1/2\int n^a(r) V^a_H dr
  ****************************************************/

  dtime(&stime);

  ECE[1] = Calc_EH0(MD_iter);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for EH0=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
                    kinetic energy
  ****************************************************/

  dtime(&stime);

  if (F_Kin_flag==1)  ECE[2] = Calc_Ekin();

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Ekin=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
              neutral atom potential energy
  ****************************************************/

  dtime(&stime);

  if (F_VNA_flag==1 && ProExpn_VNA==1)  ECE[3] = Calc_Ena();

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Ena=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
           non-local pseudo potential energy
  ****************************************************/

  dtime(&stime);

  if (F_NL_flag==1)  ECE[4] = Calc_Enl();

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for Enl=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
        The penalty term to create a core hole 
  ****************************************************/

  dtime(&stime);

  if (core_hole_state_flag==1 && F_CH_flag==1){
    ECE[14] = Calc_ECH();
  }

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for ECH=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
     EXC = \sum_{\sigma} (n_{\sigma}(r)+n_pcc(r)\epsilon_{xc}
     EH1 = 1/2\int {n(r)-n_a(r)} \delta V_H(r) dr
     if (ProExpn_VNA==0) Ena = \int n(r) Vna(r) dr
  ****************************************************/

  dtime(&stime);

  Calc_EXC_EH1(ECE,CDM);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for EXC_EH1=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /****************************************************
    LDA+U energy   --- added by MJ
  ****************************************************/

  if (F_U_flag==1){
    if (Hub_U_switch==1)  ECE[8] = Calc_Ehub();
    else                  ECE[8] = 0.0;
  }

  /*********************************************************
   DFT-D2 and D3 by Grimme (implemented by Okuno and Ellner)
  *********************************************************/

  if(F_dftD_flag==1){
    if(dftD_switch==1){
      if(version_dftD==2) ECE[13] = Calc_EdftD(); 
      if(version_dftD==3) ECE[13] = Calc_EdftD3(); 
    }
    else ECE[13] = 0.0;
  }

  /****************************************************
    Stress coming from volume terms
          
    ECE[3]; Una
    ECE[5]; UH1

    The volume term coming from Exc is added in 
    Calc_EXC_EH1 
  ****************************************************/

  if (scf_stress_flag){

    if (ProExpn_VNA==0){
      double s3,s5;

      s3 = (double)F_VNA_flag*ECE[3]; 
      s5 = (double)F_dVHart_flag*ECE[5];

      Stress_Tensor[0] += s3 + s5;
      Stress_Tensor[4] += s3 + s5;
      Stress_Tensor[8] += s3 + s5;
    }
    else{
      double s5;

      s5 = (double)F_dVHart_flag*ECE[5];
      Stress_Tensor[0] += s5;
      Stress_Tensor[4] += s5;
      Stress_Tensor[8] += s5;
    }

    {
      int i,j;
      double tmp;

      /* symmetrization of stress tensor */

      for (i=0; i<3; i++){
	for (j=(i+1); j<3; j++){
	  tmp = 0.5*(Stress_Tensor[3*i+j]+Stress_Tensor[3*j+i]);
	  Stress_Tensor[3*i+j] = tmp;
	  Stress_Tensor[3*j+i] = tmp;
	}
      }

      /* show the stress tensor including all the contributions */

      if (myid==Host_ID && 0<level_stdout){

	printf("\n*******************************************************\n"); fflush(stdout);
	printf("               Stress tensor (Hartree/bohr^3)            \n"); fflush(stdout);
	printf("*******************************************************\n\n"); fflush(stdout);

	for (i=0; i<3; i++){
	  for (j=0; j<3; j++){
	    printf("%17.8f ", Stress_Tensor[3*i+j]/Cell_Volume);
	  }
	  printf("\n");fflush(stdout);
	}
      }
    }
  }


  /*********************************************************
                decomposition of total energy 
  *********************************************************/

  if (Energy_Decomposition_flag==1){

    Energy_Decomposition(ECE);
  }

  /****************************************************
   freeing of arrays
  ****************************************************/

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  /* computational time */

  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}








double Calc_Ekin()
{
  /****************************************************
          calculate the kinetic energy, Ekin
  ****************************************************/

  int i,j,spin,spinmax;
  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  double My_Ekin,Ekin,Zc,Zh,dum,dum2;
  int numprocs,myid;
  double Stime_atom, Etime_atom;
  double *My_Ekin_threads;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
      conventional calculation of kinetic energy
  ****************************************************/

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */
  My_Ekin_threads = (double*)malloc(sizeof(double)*Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_Ekin_threads[Nloop] = 0.0;

  if (SpinP_switch==0 || SpinP_switch==1){

    for (spin=0; spin<=SpinP_switch; spin++){

#pragma omp parallel shared(SpinP_switch,time_per_atom,spin,CntH0,H0,DM,My_Ekin_threads,Spe_Total_CNO,Cnt_switch,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,i,j,Etime_atom)
      {

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	/* one-dimensionalized loop */

	for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	  dtime(&Stime_atom);

	  /* get Mc_AN and h_AN */

	  Mc_AN = OneD2Mc_AN[Nloop];
	  h_AN  = OneD2h_AN[Nloop];

	  /* set data on Mc_AN */

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];

	  /* set data on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  if (Cnt_switch==0){
	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Ekin_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*H0[0][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  else{

	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Ekin_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*CntH0[0][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
	}

        if (SpinP_switch==0) My_Ekin_threads[OMPID] = 2.0*My_Ekin_threads[OMPID];

      } /* #pragma omp parallel */
    } /* spin */ 

  }
  else if (SpinP_switch==3){

#pragma omp parallel shared(time_per_atom,H0,DM,My_Ekin_threads,Spe_Total_CNO,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,Gc_AN,Cwan,h_AN,Gh_AN,Hwan,i,j,Etime_atom)
    {

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* one-dimensionalized loop */

      for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	dtime(&Stime_atom);

	/* get Mc_AN and h_AN */

	Mc_AN = OneD2Mc_AN[Nloop];
	h_AN  = OneD2h_AN[Nloop];

	/* set data on Mc_AN */

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	/* set data on h_AN */

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){
	    My_Ekin_threads[OMPID] += (DM[0][0][Mc_AN][h_AN][i][j] + DM[0][1][Mc_AN][h_AN][i][j])*H0[0][Mc_AN][h_AN][i][j];
	  }
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }

    } /* #pragma omp parallel */
  }

  /* sum of My_Ekin_threads */
  My_Ekin = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_Ekin += My_Ekin_threads[Nloop];
  }

  /* sum of My_Ekin */
  MPI_Allreduce(&My_Ekin, &Ekin, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* freeing of array */
  free(My_Ekin_threads);

  /****************************************************
                      return Ekin
  ****************************************************/

  return Ekin;  
}





double Calc_Ena()
{
  /****************************************************
     calculate the neutral atom potential energy, Ena
  ****************************************************/

  int i,j,spin,spinmax;
  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  double My_Ena,Ena,Zc,Zh,dum,dum2;
  int numprocs,myid;
  double Stime_atom, Etime_atom;
  double *My_Ena_threads;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /**********************************************************
   conventional calculation of neutral atom potential energy
  **********************************************************/

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */
  My_Ena_threads = (double*)malloc(sizeof(double)*Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_Ena_threads[Nloop] = 0.0;

  if (SpinP_switch==0 || SpinP_switch==1){

    if (Cnt_switch==1){
      /* temporaly, we borrow the CntH0 matrix */
      Cont_Matrix0(HVNA,CntH0[0]);
    }

    for (spin=0; spin<=SpinP_switch; spin++){

#pragma omp parallel shared(spin,SpinP_switch,time_per_atom,CntH0,HVNA,DM,My_Ena_threads,Spe_Total_CNO,Cnt_switch,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,i,j,Etime_atom)
      {

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	/* one-dimensionalized loop */

	for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	  dtime(&Stime_atom);

	  /* get Mc_AN and h_AN */

	  Mc_AN = OneD2Mc_AN[Nloop];
	  h_AN  = OneD2h_AN[Nloop];

	  /* set data on Mc_AN */

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];

	  /* set data on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  if (Cnt_switch==0){
	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Ena_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*HVNA[Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  else {

	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Ena_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*CntH0[0][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
	}

        if (SpinP_switch==0) My_Ena_threads[OMPID] = 2.0*My_Ena_threads[OMPID];

      } /* #pragma omp parallel */
    } /* spin */

  }
  else if (SpinP_switch==3){

#pragma omp parallel shared(time_per_atom,HVNA,DM,Spe_Total_CNO,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,Gc_AN,h_AN,Cwan,Gh_AN,Hwan,i,j,Etime_atom)
    {

      /* get info. on OpenMP */ 

      OMPID  = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* one-dimensionalized loop */

      for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	dtime(&Stime_atom);

	/* get Mc_AN and h_AN */

	Mc_AN = OneD2Mc_AN[Nloop];
	h_AN  = OneD2h_AN[Nloop];

	/* set data on Mc_AN */

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	/* set data on h_AN */

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){
	    My_Ena_threads[OMPID] += (DM[0][0][Mc_AN][h_AN][i][j]+DM[0][1][Mc_AN][h_AN][i][j])*HVNA[Mc_AN][h_AN][i][j];
	  }
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Nloop */
    } /* #pragma omp parallel */ 
  } /* else if (SpinP_switch==3) */


  /* sum of My_Ena_threads */
  My_Ena = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_Ena += My_Ena_threads[Nloop];
  }

  /* sum of My_Ena */
  MPI_Allreduce(&My_Ena, &Ena, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* freeing of array */
  free(My_Ena_threads);

  /****************************************************
                      return Ena
  ****************************************************/

  return Ena;  
}





double Calc_Enl()
{
  /****************************************************
     calculate the non-local pseudo potential energy
  ****************************************************/

  int i,j,spin,spinmax;
  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  double My_Enl,Enl,Zc,Zh,dum,dum2;
  int numprocs,myid;
  double Stime_atom, Etime_atom;
  double *My_Enl_threads;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*******************************************************************
   conventional calculation of the non-local pseudo potential energy
  *******************************************************************/

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */
  My_Enl_threads = (double*)malloc(sizeof(double)*Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_Enl_threads[Nloop] = 0.0;

  if (SpinP_switch==0 || SpinP_switch==1){

    for (spin=0; spin<=SpinP_switch; spin++){

      if (Cnt_switch==1){
        /* temporaly, borrow the CntH0 matrix */
        Cont_Matrix0(HNL[spin],CntH0[0]);
      }

#pragma omp parallel shared(spin,SpinP_switch,time_per_atom,CntH0,HNL,DM,My_Enl_threads,Spe_Total_CNO,Cnt_switch,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,Etime_atom,i,j)
      {

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	/* one-dimensionalized loop */

	for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	  dtime(&Stime_atom);

	  /* get Mc_AN and h_AN */

	  Mc_AN = OneD2Mc_AN[Nloop];
	  h_AN  = OneD2h_AN[Nloop];

	  /* set data on Mc_AN */

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];

	  /* set data on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  if (Cnt_switch==0){
	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Enl_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*HNL[spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  else {
	    for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	      for (j=0; j<Spe_Total_CNO[Hwan]; j++){
		My_Enl_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*CntH0[0][Mc_AN][h_AN][i][j];
	      }
	    }
	  }

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Nloop */

        if (SpinP_switch==0) My_Enl_threads[OMPID] = 2.0*My_Enl_threads[OMPID];

      } /* #pragma omp parallel */
    } /* spin */

  }
  else if (SpinP_switch==3){

#pragma omp parallel shared(time_per_atom,iHNL0,HNL,iDM,DM,My_Enl_threads,Spe_Total_CNO,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Etime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,i,j)
    {

      /* get info. on OpenMP */ 

      OMPID  = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* one-dimensionalized loop */

      for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	dtime(&Stime_atom);

	/* get Mc_AN and h_AN */

	Mc_AN = OneD2Mc_AN[Nloop];
	h_AN  = OneD2h_AN[Nloop];

	/* set data on Mc_AN */

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	/* set data on h_AN */

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){

	    My_Enl_threads[OMPID] += 
	         DM[0][0][Mc_AN][h_AN][i][j]*  HNL[0][Mc_AN][h_AN][i][j]
	      - iDM[0][0][Mc_AN][h_AN][i][j]*iHNL0[0][Mc_AN][h_AN][i][j]
	      +  DM[0][1][Mc_AN][h_AN][i][j]*  HNL[1][Mc_AN][h_AN][i][j]
	      - iDM[0][1][Mc_AN][h_AN][i][j]*iHNL0[1][Mc_AN][h_AN][i][j]
	   + 2.0*DM[0][2][Mc_AN][h_AN][i][j]*  HNL[2][Mc_AN][h_AN][i][j] 
	   - 2.0*DM[0][3][Mc_AN][h_AN][i][j]*iHNL0[2][Mc_AN][h_AN][i][j];
 
	  }
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Nloop */
    } /* #pragma omp parallel */
  }

  /* sum of My_Enl_threads */
  My_Enl = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_Enl += My_Enl_threads[Nloop];
  }

  /* sum of My_Enl */
  MPI_Allreduce(&My_Enl, &Enl, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* freeing of array */
  free(My_Enl_threads);

  /****************************************************
                      return Enl
  ****************************************************/

  return Enl;  
}


double Calc_ECH()
{
  /****************************************************
    calculate the penalty term to create a core-hole 
  ****************************************************/

  int i,j,spin,spinmax;
  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  double My_ECH,ECH,Zc,Zh,dum,dum2;
  int numprocs,myid;
  double Stime_atom, Etime_atom;
  double *My_ECH_threads;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*******************************************************************
            conventional calculation of the penalty term 
  *******************************************************************/

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */
  My_ECH_threads = (double*)malloc(sizeof(double)*Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_ECH_threads[Nloop] = 0.0;

  if (SpinP_switch==0 || SpinP_switch==1){

    for (spin=0; spin<=SpinP_switch; spin++){

#pragma omp parallel shared(spin,SpinP_switch,time_per_atom,HCH,DM,My_ECH_threads,Spe_Total_CNO,Cnt_switch,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,Etime_atom,i,j)
      {

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	/* one-dimensionalized loop */

	for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	  dtime(&Stime_atom);

	  /* get Mc_AN and h_AN */

	  Mc_AN = OneD2Mc_AN[Nloop];
	  h_AN  = OneD2h_AN[Nloop];

	  /* set data on Mc_AN */

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];

	  /* set data on h_AN */

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];

	  for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	    for (j=0; j<Spe_Total_CNO[Hwan]; j++){
	      My_ECH_threads[OMPID] += DM[0][spin][Mc_AN][h_AN][i][j]*HCH[spin][Mc_AN][h_AN][i][j];
	    }
	  }

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Nloop */

        if (SpinP_switch==0) My_ECH_threads[OMPID] = 2.0*My_ECH_threads[OMPID];

      } /* #pragma omp parallel */
    } /* spin */

  }
  else if (SpinP_switch==3){

#pragma omp parallel shared(time_per_atom,iHCH,HCH,iDM,DM,My_ECH_threads,Spe_Total_CNO,natn,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop) private(OMPID,Nthrds,Nprocs,Nloop,Stime_atom,Etime_atom,Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN,Hwan,i,j)
    {

      /* get info. on OpenMP */ 

      OMPID  = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      /* one-dimensionalized loop */

      for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

	dtime(&Stime_atom);

	/* get Mc_AN and h_AN */

	Mc_AN = OneD2Mc_AN[Nloop];
	h_AN  = OneD2h_AN[Nloop];

	/* set data on Mc_AN */

	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];

	/* set data on h_AN */

	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];

	for (i=0; i<Spe_Total_CNO[Cwan]; i++){
	  for (j=0; j<Spe_Total_CNO[Hwan]; j++){

	    My_ECH_threads[OMPID] += 
	         DM[0][0][Mc_AN][h_AN][i][j]* HCH[0][Mc_AN][h_AN][i][j]
	      - iDM[0][0][Mc_AN][h_AN][i][j]*iHCH[0][Mc_AN][h_AN][i][j]
	      +  DM[0][1][Mc_AN][h_AN][i][j]* HCH[1][Mc_AN][h_AN][i][j]
	      - iDM[0][1][Mc_AN][h_AN][i][j]*iHCH[1][Mc_AN][h_AN][i][j]
	   + 2.0*DM[0][2][Mc_AN][h_AN][i][j]* HCH[2][Mc_AN][h_AN][i][j] 
	   - 2.0*DM[0][3][Mc_AN][h_AN][i][j]*iHCH[2][Mc_AN][h_AN][i][j];

	  }
	}

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Nloop */
    } /* #pragma omp parallel */
  }

  /* sum of My_ECH_threads */
  My_ECH = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_ECH += My_ECH_threads[Nloop];
  }

  /* sum of My_ECH */
  MPI_Allreduce(&My_ECH, &ECH, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* freeing of array */
  free(My_ECH_threads);

  /****************************************************
                      return ECH
  ****************************************************/

  return ECH;  
}





double Calc_Ecore()
{
  /****************************************************
                         Ecore
  ****************************************************/

  int Mc_AN,Gc_AN,Cwan,Rn,h_AN,Gh_AN,Hwan;
  int i,spin,spinmax;
  double My_Ecore,Ecore,Zc,Zh,dum,dum2;
  double *My_Ecore_threads;
  double TmpEcore,dEx,dEy,dEz,r,lx,ly,lz;
  int numprocs,myid;
  double Stime_atom,Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && 0<level_stdout){
    printf("  Force calculation #6\n");fflush(stdout);
  }

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */
  My_Ecore_threads = (double*)malloc(sizeof(double)*Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_Ecore_threads[Nloop] = 0.0;

#pragma omp parallel shared(level_stdout,time_per_atom,atv,Gxyz,Dis,ncn,natn,FNAN,Spe_Core_Charge,WhatSpecies,M2G,Matomnum,My_Ecore_threads,DecEscc,Energy_Decomposition_flag,SpinP_switch,Spe_MaxL_Basis,Spe_Num_Basis) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,Cwan,Zc,dEx,dEy,dEz,h_AN,Gh_AN,Rn,Hwan,Zh,r,lx,ly,lz,dum,dum2,Etime_atom,TmpEcore)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      Zc = Spe_Core_Charge[Cwan];
      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;
      TmpEcore = 0.0;       

      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Rn = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	Zh = Spe_Core_Charge[Hwan];
	r = Dis[Gc_AN][h_AN];

	/* for empty atoms or finite elemens basis */
	if (r<1.0e-10) r = 1.0e-10;

	lx = (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])/r;
	ly = (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])/r;
	lz = (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])/r;
	dum = Zc*Zh/r;
	dum2 = dum/r;
        TmpEcore += dum; 
	dEx = dEx - lx*dum2;
	dEy = dEy - ly*dum2;
	dEz = dEz - lz*dum2;

      } /* h_AN */

      /****************************************************
                        #6 of force
         Contribution from the core-core repulsions
      ****************************************************/

      My_Ecore_threads[OMPID] += 0.50*TmpEcore;
      Gxyz[Gc_AN][17] += dEx;
      Gxyz[Gc_AN][18] += dEy;
      Gxyz[Gc_AN][19] += dEz;

      if (Energy_Decomposition_flag==1){

        DecEscc[0][Mc_AN][0] = 0.25*TmpEcore;
        DecEscc[1][Mc_AN][0] = 0.25*TmpEcore;
      } 

      if (2<=level_stdout){
	printf("<Total_Ene>  force(6) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	       myid,Mc_AN,Gc_AN,dEx,dEy,dEz);fflush(stdout);
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

  } /* #pragma omp parallel */

  My_Ecore = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_Ecore += My_Ecore_threads[Nloop];
  }

  MPI_Allreduce(&My_Ecore, &Ecore, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* freeing of array */
  free(My_Ecore_threads);

  return Ecore;  
}








double Calc_EH0(int MD_iter)
{
  /****************************************************
              EH0 = -1/2\int n^a(r) V^a_H dr
  ****************************************************/

  int Mc_AN,Gc_AN,h_AN,Gh_AN,num,gnum,i;
  int wan,wan1,wan2,Nd,n1,n2,n3,spin,spinmax;
  double bc,dx,x,y,z,r1,r2,rho0,xx;
  double Scale_Grid_Ecut,TmpEH0;
  double EH0ij[4],My_EH0,EH0,tmp0;
  double *Fx,*Fy,*Fz,*g0;
  double dEx,dEy,dEz,Dx,Sx;
  double Z1,Z2,factor;
  double My_dEx,My_dEy,My_dEz;
  int numprocs,myid,ID;
  double stime,etime;
  double Stime_atom, Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;
  double *My_EH0_threads;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
     allocation of arrays:

    double Fx[Matomnum+1];
    double Fy[Matomnum+1];
    doubel Fz[Matomnum+1];
  ****************************************************/

  Fx = (double*)malloc(sizeof(double)*(Matomnum+1));
  Fy = (double*)malloc(sizeof(double)*(Matomnum+1));
  Fz = (double*)malloc(sizeof(double)*(Matomnum+1));

  /****************************************************
             Set of atomic density on grids
  ****************************************************/

  if (MD_iter==1){

    dtime(&stime);

    Scale_Grid_Ecut = 16.0*600.0;

    /* estimate the size of an array g0 */

    Max_Nd = 0;
    for (wan=0; wan<SpeciesNum; wan++){
      Spe_Spe2Ban[wan] = wan;
      bc = Spe_Atom_Cut1[wan];
      dx = PI/sqrt(Scale_Grid_Ecut);
      Nd = 2*(int)(bc/dx) + 1;
      if (Max_Nd<Nd) Max_Nd = Nd;
    }

    /* estimate sizes of arrays GridX,Y,Z_EH0, Arho_EH0, and Wt_EH0 */

    Max_TGN_EH0 = 0;
    for (wan=0; wan<SpeciesNum; wan++){

      Spe_Spe2Ban[wan] = wan;
      bc = Spe_Atom_Cut1[wan];
      dx = PI/sqrt(Scale_Grid_Ecut);

      Nd = 2*(int)(bc/dx) + 1;
      dx = 2.0*bc/(double)(Nd-1);
      gnum = Nd*CoarseGL_Mesh;

      if (Max_TGN_EH0<gnum) Max_TGN_EH0 = gnum;

      if (2<=level_stdout){
        printf("<Calc_EH0> A spe=%2d 1D-grids=%2d 3D-grids=%2d\n",wan,Nd,gnum);fflush(stdout);
      }
    }
    
    /* allocation of arrays GridX,Y,Z_EH0, Arho_EH0, and Wt_EH0 */

    Max_TGN_EH0 += 10; 

    Allocate_Arrays(4);

    /* calculate GridX,Y,Z_EH0 and Wt_EH0 */

#pragma omp parallel shared(Spe_Num_Mesh_PAO,Spe_PAO_XV,Spe_Atomic_Den,Max_Nd,level_stdout,TGN_EH0,Wt_EH0,Arho_EH0,GridZ_EH0,GridY_EH0,GridX_EH0,Scale_Grid_Ecut,Spe_Atom_Cut1,dv_EH0,Spe_Spe2Ban,SpeciesNum,CoarseGL_Abscissae,CoarseGL_Weight) private(OMPID,Nthrds,Nprocs,wan,bc,dx,Nd,gnum,n1,n2,n3,x,y,z,tmp0,r1,rho0,g0,Sx,Dx,xx)
    {

      int l,p;

      /* allocation of arrays g0 */

      g0 = (double*)malloc(sizeof(double)*Max_Nd);

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();
    
      for (wan=OMPID*SpeciesNum/Nthrds; wan<(OMPID+1)*SpeciesNum/Nthrds; wan++){

	Spe_Spe2Ban[wan] = wan;
	bc = Spe_Atom_Cut1[wan];
	dx = PI/sqrt(Scale_Grid_Ecut);
	Nd = 2*(int)(bc/dx) + 1;
	dx = 2.0*bc/(double)(Nd-1);
	dv_EH0[wan] = dx;

	for (n1=0; n1<Nd; n1++){
	  g0[n1] = dx*(double)n1 - bc;
	}

	gnum = 0; 
        y = 0.0;

        Sx = Spe_Atom_Cut1[wan] + 0.0;
        Dx = Spe_Atom_Cut1[wan] - 0.0;

        for (n3=0; n3<Nd; n3++){

          z = g0[n3];
          tmp0 = z*z;

  	  for (n1=0; n1<CoarseGL_Mesh; n1++){

            x = 0.50*(Dx*CoarseGL_Abscissae[n1] + Sx);
            xx = 0.5*log(x*x + tmp0);

	    GridX_EH0[wan][gnum] = x;
	    GridY_EH0[wan][gnum] = y;
	    GridZ_EH0[wan][gnum] = z;

      	    rho0 = KumoF( Spe_Num_Mesh_PAO[wan], xx, 
                          Spe_PAO_XV[wan], Spe_PAO_RV[wan], Spe_Atomic_Den[wan]);

	    Arho_EH0[wan][gnum] = rho0;
   	    Wt_EH0[wan][gnum] = PI*x*CoarseGL_Weight[n1]*Dx;

            /* increment of gnum */

	    gnum++;  
	  }

	} /* n3 */

	TGN_EH0[wan] = gnum;

	if (2<=level_stdout){
	  printf("<Calc_EH0> B spe=%2d 1D-grids=%2d 3D-grids=%2d\n",wan,Nd,gnum);fflush(stdout);
	}

      } /* wan */

      /* free */
      free(g0);

    } /* #pragma omp parallel */

    dtime(&etime);
    if(myid==0 && measure_time){
      printf("Time for part1 of EH0=%18.5f\n",etime-stime);fflush(stdout);
    } 

  } /* if (MD_iter==1) */

  /****************************************************
    calculation of scaling factors:
  ****************************************************/

  if (MD_iter==1){

    for (wan1=0; wan1<SpeciesNum; wan1++){

      r1 = Spe_Atom_Cut1[wan1];
      Z1 = Spe_Core_Charge[wan1];

      for (wan2=0; wan2<SpeciesNum; wan2++){

	/* EH0_TwoCenter_at_Cutoff is parallelized by OpenMP */      
	EH0_TwoCenter_at_Cutoff(wan1, wan2, EH0ij);

	r2 = Spe_Atom_Cut1[wan2];
	Z2 = Spe_Core_Charge[wan2];
	tmp0 = Z1*Z2/(r1+r2);

	if (1.0e-20<fabs(EH0ij[0])){ 
	  EH0_scaling[wan1][wan2] = tmp0/EH0ij[0];
	}
	else{
	  EH0_scaling[wan1][wan2] = 0.0;
	}

      }
    }
  }

  /****************************************************
                -1/2\int n^a(r) V^a_H dr
  ****************************************************/

  dtime(&stime);

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Fx[Mc_AN] = 0.0;
    Fy[Mc_AN] = 0.0;
    Fz[Mc_AN] = 0.0;
  }

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */
  My_EH0_threads = (double*)malloc(sizeof(double)*Nthrds0);
  for (Nloop=0; Nloop<Nthrds0; Nloop++) My_EH0_threads[Nloop] = 0.0;

#pragma omp parallel shared(time_per_atom,RMI1,EH0_scaling,natn,FNAN,WhatSpecies,M2G,Matomnum,My_EH0_threads,DecEscc,Energy_Decomposition_flag,List_YOUSO,Spe_MaxL_Basis,Spe_Num_Basis,SpinP_switch) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,wan1,h_AN,Gh_AN,wan2,factor,Etime_atom,TmpEH0)
  {

    int l,p;
    double EH0ij[4];    

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
  
    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      TmpEH0 = 0.0; 

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	wan2 = WhatSpecies[Gh_AN];

	if (h_AN==0) factor = 1.0;
	else         factor = EH0_scaling[wan1][wan2];

	EH0_TwoCenter(Gc_AN, h_AN, EH0ij);
        TmpEH0 -= 0.250*factor*EH0ij[0];
	Fx[Mc_AN] = Fx[Mc_AN] - 0.5*factor*EH0ij[1];
	Fy[Mc_AN] = Fy[Mc_AN] - 0.5*factor*EH0ij[2];
	Fz[Mc_AN] = Fz[Mc_AN] - 0.5*factor*EH0ij[3];

	if (h_AN==0) factor = 1.0;
	else         factor = EH0_scaling[wan2][wan1];

	EH0_TwoCenter(Gh_AN, RMI1[Mc_AN][h_AN][0], EH0ij);
        TmpEH0 -= 0.250*factor*EH0ij[0];
	Fx[Mc_AN] = Fx[Mc_AN] + 0.5*factor*EH0ij[1];
	Fy[Mc_AN] = Fy[Mc_AN] + 0.5*factor*EH0ij[2];
	Fz[Mc_AN] = Fz[Mc_AN] + 0.5*factor*EH0ij[3];

      } /* h_AN */

      My_EH0_threads[OMPID] += TmpEH0;

      if (Energy_Decomposition_flag==1){

	DecEscc[0][Mc_AN][0] += 0.5*TmpEH0;
	DecEscc[1][Mc_AN][0] += 0.5*TmpEH0;
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

  } /* #pragma omp parallel */

  /* sum of My_EH0_threads */
  My_EH0 = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    My_EH0 += My_EH0_threads[Nloop];
  }

  /* sum of My_EH0 */
  MPI_Allreduce(&My_EH0, &EH0, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* freeing of array */
  free(My_EH0_threads);

  dtime(&etime);
  if(myid==0 && measure_time){
    printf("Time for part2 of EH0=%18.5f\n",etime-stime);fflush(stdout);
  } 

  /*******************************************************
                      #7 of force
   contribution from the classical Coulomb energy between
   the neutral atomic charge and the neutral potential 
  *******************************************************/

  if (myid==Host_ID && 0<level_stdout){
    printf("  Force calculation #7\n");fflush(stdout);
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    if (2<=level_stdout){
      printf("<Total_Ene>  force(7) myid=%2d  Mc_AN=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
              myid,Mc_AN,Gc_AN,Fx[Mc_AN],Fy[Mc_AN],Fz[Mc_AN]);fflush(stdout);
    }

    Gxyz[Gc_AN][17] += Fx[Mc_AN];
    Gxyz[Gc_AN][18] += Fy[Mc_AN];
    Gxyz[Gc_AN][19] += Fz[Mc_AN];
  }

  /****************************************************
   MPI, Gxyz[Gc_AN][17-19]
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /****************************************************
    freeing of arrays:

    double Fx[Matomnum+1];
    double Fy[Matomnum+1];
    doubel Fz[Matomnum+1];
  ****************************************************/

  free(Fx);
  free(Fy);
  free(Fz);

  /* return */

  return EH0;  
}


void Calc_EXC_EH1(double ECE[],double *****CDM)
{
  /************************************************************
     EXC = \sum_{\sigma} (n_{\sigma}(r)+n_pcc(r)\epsilon_{xc}
     EH1 = 1/2\int {n(r)-n_a(r)} \delta V_H dr
  ************************************************************/

  static int firsttime=1;
  int i,spin,spinmax,XC_P_switch;
  int numS,numR,My_GNum,BN_AB;
  int n,n1,n2,n3,Ng1,Ng2,Ng3,j,k;
  int GNc,GRc,MNc;
  int GN,GNs,BN,DN,LN,N2D,n2D,N3[4];
  double EXC[2],EH1,sum,tot_den;
  double My_EXC[2],My_EH1;
  double My_EXC_VolumeTerm[2];
  double EXC_VolumeTerm[2];
  double My_Eef,Eef;
  double My_Ena,Ena;
  double sum_charge,My_charge;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Cxyz[4],gradxyz[4];
  double Stime_atom,Etime_atom;
  double time0,time1;
  double sden[2],tden,aden,pden[2];

  /* dipole moment */
  int Gc_AN,Mc_AN,spe;
  double x,y,z,den,charge,cden_BG;
  double E_dpx,E_dpy,E_dpz; 
  double E_dpx_BG,E_dpy_BG,E_dpz_BG; 
  double C_dpx,C_dpy,C_dpz;
  double My_E_dpx_BG,My_E_dpy_BG,My_E_dpz_BG; 
  double My_E_dpx,My_E_dpy,My_E_dpz; 
  double My_C_dpx,My_C_dpy,My_C_dpz;
  double AU2Debye,AbsD;
  double x0,y0,z0,r;
  double rs,re,ts,te,ps,pe;
  double Sp,Dp,St,Dt,Sr,Dr;
  double r1,dx,dy,dz,dx1,dy1,dz1;
  double x1,y1,z1,den0,exc0,w;
  double sumr,sumt;
  double sumrx,sumtx;
  double sumry,sumty;
  double sumrz,sumtz;
  double gden0,vxc0;
  double *My_sumr,*My_sumrx,*My_sumry,*My_sumrz;
  int ir,ia,Cwan,Rn,Hwan,Gh_AN,h_AN;
  char file_DPM[YOUSO10] = ".dpm";
  FILE *fp_DPM;
  char buf[fp_bsize];          /* setvbuf */
  MPI_Status stat;
  MPI_Request request;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if      (SpinP_switch==0) spinmax = 0;
  else if (SpinP_switch==1) spinmax = 1;
  else if (SpinP_switch==3) spinmax = 1;

  /****************************************************
   set Vxc_Grid
  ****************************************************/

  XC_P_switch = 0;
  Set_XC_Grid(2, XC_P_switch,XC_switch,
              Density_Grid_D[0],Density_Grid_D[1],
              Density_Grid_D[2],Density_Grid_D[3],
              Vxc_Grid_D[0], Vxc_Grid_D[1],
	      Vxc_Grid_D[2], Vxc_Grid_D[3],
              NULL,NULL);

  /****************************************************
             copy Vxc_Grid_D to Vxc_Grid_B
  ****************************************************/

  Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
  Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
  Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

  for (n=0; n<Num_Rcv_Grid_B2D[myid]; n++){
    DN = Index_Rcv_Grid_B2D[myid][n];
    BN = Index_Snd_Grid_B2D[myid][n];

    i = DN/(Ng2*Ng3);
    j = (DN-i*Ng2*Ng3)/Ng3;
    k = DN - i*Ng2*Ng3 - j*Ng3; 

    if ( !(i<=1 || (Ng1-2)<=i || j<=1 || (Ng2-2)<=j || k<=1 || (Ng3-2)<=k)){
      for (spin=0; spin<=SpinP_switch; spin++){
        Vxc_Grid_B[spin][BN] = Vxc_Grid_D[spin][DN];
      }
    }
  }

  /*********************************************************
   set RefVxc_Grid, where the CA-LDA exchange-correlation 
   functional is alway used.
  *********************************************************/

  XC_P_switch = 0;
  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
    tot_den = ADensity_Grid_B[BN_AB] + ADensity_Grid_B[BN_AB];
    if (PCC_switch==1) {
      tot_den += PCCDensity_Grid_B[0][BN_AB] + PCCDensity_Grid_B[1][BN_AB];
    }
    RefVxc_Grid_B[BN_AB] = XC_Ceperly_Alder(tot_den,XC_P_switch);
  }

  /****************************************************
        calculations of Ena, Eef, EH1, and EXC
  ****************************************************/

  My_Ena = 0.0;
  My_Eef = 0.0;
  My_EH1 = 0.0;
  My_EXC[0] = 0.0;
  My_EXC[1] = 0.0;

  for (BN=0; BN<My_NumGridB_AB; BN++){

    sden[0] = Density_Grid_B[0][BN];
    sden[1] = Density_Grid_B[1][BN];
    tden = sden[0] + sden[1];
    aden = ADensity_Grid_B[BN];
    pden[0] = PCCDensity_Grid_B[0][BN];
    pden[1] = PCCDensity_Grid_B[1][BN];

    /* if (ProExpn_VNA==off), Ena is calculated here. */
    if (ProExpn_VNA==0) My_Ena += tden*VNA_Grid_B[BN];

    /* electric energy by electric field */
    if (E_Field_switch==1) My_Eef += tden*VEF_Grid_B[BN];

    /* EH1 = 1/2\int \delta n(r) \delta V_H dr */
    My_EH1 += (tden - 2.0*aden)*dVHart_Grid_B[BN];

    /*   EXC = \sum_{\sigma} (n_{\sigma}+n_pcc)\epsilon_{xc}
              -(n_{atom}+n_pcc)\epsilon_{xc}(n_{atom})

        calculation of the difference between the xc energies 
        calculated by wave-function-charge and atomic charge
        on the coarse grid.  */

    for (spin=0; spin<=spinmax; spin++){
      My_EXC[spin] += (sden[spin]+pden[spin])*Vxc_Grid_B[spin][BN]
                     -(aden+pden[spin])*RefVxc_Grid_B[BN];
    }

  } /* BN */

  /****************************************************
       multiplying GridVol and MPI communication
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  /* if (ProExpn_VNA==off), Ena is calculated here. */

  if (ProExpn_VNA==0){

    if (F_VNA_flag==1){
      My_Ena *= GridVol;
      MPI_Allreduce(&My_Ena, &Ena, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      ECE[3] = Ena;
    }
    else{
      ECE[3] = 0.0;
    }
  }

  /* electric energy by electric field */
  if (E_Field_switch==1){
    My_Eef *= GridVol;
    MPI_Allreduce(&My_Eef, &Eef, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    ECE[12] = Eef;
  }
  else {
    ECE[12] = 0.0;
  }
  if (F_VEF_flag==0){
    ECE[12] = 0.0;
  }

  /* EH1 = 1/2\int \delta n(r) \delta V_H dr */
  My_EH1 *= (0.5*GridVol);

  /************************************************************
    EXC = \sum_{\sigma} n_{\sigma}\epsilon_{xc}
       - n_{atom}\epsilon_{xc}(n_{atom})

       calculation of the difference between the xc energies 
       calculated by wave-function-charge and atomic charge
       on the coarse grid.  

       My_EXC_VolumeTerm will be used to take account of 
       volume term for stress. 
  *************************************************************/

  My_EXC[0] *= GridVol;
  My_EXC[1] *= GridVol;

  My_EXC_VolumeTerm[0] = My_EXC[0];
  My_EXC_VolumeTerm[1] = My_EXC[1];

  /****************************************************
    calculation of Exc^(0) and its contribution 
    to forces on the fine mesh
  ****************************************************/

  Set_Lebedev_Grid();

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* initialize the temporal array storing the force contribution */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    Gxyz[Gc_AN][41] = 0.0;
    Gxyz[Gc_AN][42] = 0.0;
    Gxyz[Gc_AN][43] = 0.0;
  }

  /* start calc. */

  rs = 0.0;
  sum = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    re = Spe_Atom_Cut1[Cwan];
    Sr = re + rs;
    Dr = re - rs;

    /* allocation of arrays */

    double *My_sumr,**My_sumrx,**My_sumry,**My_sumrz;

    My_sumr = (double*)malloc(sizeof(double)*Nthrds0);
    for (i=0; i<Nthrds0; i++) My_sumr[i]  = 0.0;

    My_sumrx = (double**)malloc(sizeof(double*)*Nthrds0);
    for (i=0; i<Nthrds0; i++){
      My_sumrx[i] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      for (j=0; j<(FNAN[Gc_AN]+1); j++){
        My_sumrx[i][j] = 0.0;
      }
    }

    My_sumry = (double**)malloc(sizeof(double*)*Nthrds0);
    for (i=0; i<Nthrds0; i++){
      My_sumry[i] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      for (j=0; j<(FNAN[Gc_AN]+1); j++){
        My_sumry[i][j] = 0.0;
      }
    }

    My_sumrz = (double**)malloc(sizeof(double*)*Nthrds0);
    for (i=0; i<Nthrds0; i++){
      My_sumrz[i] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      for (j=0; j<(FNAN[Gc_AN]+1); j++){
        My_sumrz[i][j] = 0.0;
      }
    }

#pragma omp parallel shared(Spe_Atomic_Den2,Spe_PAO_XV,Spe_Num_Mesh_PAO,Leb_Grid_XYZW,My_sumr,My_sumrx,My_sumry,My_sumrz,Dr,Sr,CoarseGL_Abscissae,CoarseGL_Weight,Gxyz,Gc_AN,FNAN,natn,ncn,WhatSpecies,atv,F_Vxc_flag,Cwan,PCC_switch) private(OMPID,Nthrds,Nprocs,ir,ia,r,w,sumt,sumtx,sumty,sumtz,x,x0,y0,z0,h_AN,Gh_AN,Rn,Hwan,x1,y1,z1,dx,dy,dz,r1,den,den0,gden0,dx1,dy1,dz1,exc0,vxc0)
    {

      double *gx,*gy,*gz,dexc0;
      double *sum_gx,*sum_gy,*sum_gz;

      gx = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      gy = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      gz = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      sum_gx = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      sum_gy = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      sum_gz = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (ir=(OMPID*CoarseGL_Mesh/Nthrds); ir<((OMPID+1)*CoarseGL_Mesh/Nthrds); ir++){

	r = 0.50*(Dr*CoarseGL_Abscissae[ir] + Sr);
	sumt  = 0.0; 

	for (i=0; i<(FNAN[Gc_AN]+1); i++){
	  sum_gx[i] = 0.0;
	  sum_gy[i] = 0.0;
	  sum_gz[i] = 0.0;
	}

	for (ia=0; ia<Num_Leb_Grid; ia++){

	  x0 = r*Leb_Grid_XYZW[ia][0] + Gxyz[Gc_AN][1];
	  y0 = r*Leb_Grid_XYZW[ia][1] + Gxyz[Gc_AN][2];
	  z0 = r*Leb_Grid_XYZW[ia][2] + Gxyz[Gc_AN][3];

          /* calculate rho_atom + rho_pcc */ 

	  den = 0.0;

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    Rn = ncn[Gc_AN][h_AN]; 
	    Hwan = WhatSpecies[Gh_AN];

	    x1 = Gxyz[Gh_AN][1] + atv[Rn][1];
	    y1 = Gxyz[Gh_AN][2] + atv[Rn][2];
	    z1 = Gxyz[Gh_AN][3] + atv[Rn][3];
            
	    dx = x1 - x0;
	    dy = y1 - y0;
	    dz = z1 - z0;

	    x = 0.5*log(dx*dx + dy*dy + dz*dz);

	    /* calculate density */

	    den += KumoF( Spe_Num_Mesh_PAO[Hwan], x, 
                          Spe_PAO_XV[Hwan], Spe_PAO_RV[Hwan], Spe_Atomic_Den2[Hwan])*F_Vxc_flag;

	    if (h_AN==0) den0 = den;

	    /* calculate gradient of density */

	    if (h_AN!=0){
	      r1 = sqrt(dx*dx + dy*dy + dz*dz);
	      gden0 = Dr_KumoF( Spe_Num_Mesh_PAO[Hwan], x, r1, 
				Spe_PAO_XV[Hwan], Spe_PAO_RV[Hwan], Spe_Atomic_Den2[Hwan])*F_Vxc_flag;

	      gx[h_AN] = gden0/r1*dx;
	      gy[h_AN] = gden0/r1*dy;
	      gz[h_AN] = gden0/r1*dz;
	    }

	  } /* h_AN */

	  /* calculate the CA-LDA exchange-correlation energy density */
	  exc0 = XC_Ceperly_Alder(den,0);

	  /* calculate the CA-LDA exchange-correlation potential */
	  dexc0 = XC_Ceperly_Alder(den,3);

	  /* Lebedev quadrature */

          w = Leb_Grid_XYZW[ia][3];
	  sumt += w*den0*exc0;

	  for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
            sum_gx[h_AN] += w*den0*dexc0*gx[h_AN]; 
            sum_gy[h_AN] += w*den0*dexc0*gy[h_AN]; 
            sum_gz[h_AN] += w*den0*dexc0*gz[h_AN]; 
	  }

	} /* ia */

	/* r for Gauss-Legendre quadrature */

        w = r*r*CoarseGL_Weight[ir]; 
	My_sumr[OMPID]  += w*sumt;

	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	  My_sumrx[OMPID][h_AN] += w*sum_gx[h_AN];
	  My_sumry[OMPID][h_AN] += w*sum_gy[h_AN];
	  My_sumrz[OMPID][h_AN] += w*sum_gz[h_AN];
	}

      } /* ir */

      free(gx);
      free(gy);
      free(gz);
      free(sum_gx);
      free(sum_gy);
      free(sum_gz);

    } /* #pragma omp */

    sumr = 0.0;
    for (Nloop=0; Nloop<Nthrds0; Nloop++){
      sumr += My_sumr[Nloop];
    }
    sum += 2.0*PI*Dr*sumr;

    /* add force */

    for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

      sumrx = 0.0;
      sumry = 0.0;
      sumrz = 0.0;
      for (Nloop=0; Nloop<Nthrds0; Nloop++){
	sumrx += My_sumrx[Nloop][h_AN];
	sumry += My_sumry[Nloop][h_AN];
	sumrz += My_sumrz[Nloop][h_AN];
      }

      Gh_AN = natn[Gc_AN][h_AN];

      Gxyz[Gh_AN][41] += 2.0*PI*Dr*sumrx;
      Gxyz[Gh_AN][42] += 2.0*PI*Dr*sumry;
      Gxyz[Gh_AN][43] += 2.0*PI*Dr*sumrz;

      Gxyz[Gc_AN][41] -= 2.0*PI*Dr*sumrx;
      Gxyz[Gc_AN][42] -= 2.0*PI*Dr*sumry;
      Gxyz[Gc_AN][43] -= 2.0*PI*Dr*sumrz;
    }

    /* freeing of arrays */

    free(My_sumr);

    for (i=0; i<Nthrds0; i++){
      free(My_sumrx[i]);
    }
    free(My_sumrx);

    for (i=0; i<Nthrds0; i++){
      free(My_sumry[i]);
    }
    free(My_sumry);

    for (i=0; i<Nthrds0; i++){
      free(My_sumrz[i]);
    }
    free(My_sumrz);

  } /* Mc_AN */

  /* add Exc^0 calculated on the fine mesh to My_EXC */

  My_EXC[0] += 0.5*sum;
  My_EXC[1] += 0.5*sum;

  /* MPI: Gxyz[][41,42,43] */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    MPI_Allreduce(&Gxyz[Gc_AN][41], &gradxyz[0], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    Gxyz[Gc_AN][17] += gradxyz[0];
    Gxyz[Gc_AN][18] += gradxyz[1];
    Gxyz[Gc_AN][19] += gradxyz[2];

    if (2<=level_stdout){
      printf("<Total_Ene>  force(8) myid=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
	     myid,Gc_AN,gradxyz[0],gradxyz[1],gradxyz[2]);
    }
  }

  /****************************************************
   MPI, Gxyz[Gc_AN][17-19]
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);

    if (2<=level_stdout && myid==Host_ID){
      printf("<Total_Ene>  force(t) myid=%2d Gc_AN=%2d  %15.12f %15.12f %15.12f\n",
              myid,Gc_AN,Gxyz[Gc_AN][17],Gxyz[Gc_AN][18],Gxyz[Gc_AN][19]);fflush(stdout);
    }
  }

  /****************************************************
   MPI:

   EH1, EXC
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&My_EH1, &EH1, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  for (spin=0; spin<=spinmax; spin++){
    MPI_Allreduce(&My_EXC[spin], &EXC[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
    MPI_Allreduce(&My_EXC_VolumeTerm[spin], &EXC_VolumeTerm[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0){
    ECE[5] = EH1;
    ECE[6] = EXC[0];
    ECE[7] = EXC[0];
    EXC_VolumeTerm[1] = EXC_VolumeTerm[0];
  }
  else if (SpinP_switch==1 || SpinP_switch==3) {
    ECE[5] = EH1;
    ECE[6] = EXC[0];
    ECE[7] = EXC[1];
  }

  if (F_dVHart_flag==0){
    ECE[5] = 0.0;
  }

  if (F_Vxc_flag==0){
    ECE[6] = 0.0;
    ECE[7] = 0.0;
  }

  if (F_Vxc_flag==1){
    Stress_Tensor[0] += EXC_VolumeTerm[0] + EXC_VolumeTerm[1];
    Stress_Tensor[4] += EXC_VolumeTerm[0] + EXC_VolumeTerm[1];
    Stress_Tensor[8] += EXC_VolumeTerm[0] + EXC_VolumeTerm[1];
  }

  /****************************************************
             calculation of dipole moment
  ****************************************************/

  /* contribution from electron density */

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  My_E_dpx = 0.0;
  My_E_dpy = 0.0;
  My_E_dpz = 0.0;

  My_E_dpx_BG = 0.0;
  My_E_dpy_BG = 0.0;
  My_E_dpz_BG = 0.0; 

  for (BN=0; BN<My_NumGridB_AB; BN++){

    GN = BN + GNs;     
    n1 = GN/(Ngrid2*Ngrid3);    
    n2 = (GN - n1*Ngrid2*Ngrid3)/Ngrid3;
    n3 = GN - n1*Ngrid2*Ngrid3 - n2*Ngrid3; 

    x = (double)n1*gtv[1][1] + (double)n2*gtv[2][1]
      + (double)n3*gtv[3][1] + Grid_Origin[1];
    y = (double)n1*gtv[1][2] + (double)n2*gtv[2][2]
      + (double)n3*gtv[3][2] + Grid_Origin[2];
    z = (double)n1*gtv[1][3] + (double)n2*gtv[2][3]
      + (double)n3*gtv[3][3] + Grid_Origin[3];

    den = Density_Grid_B[0][BN] + Density_Grid_B[1][BN];
   
    My_E_dpx += den*x;
    My_E_dpy += den*y;
    My_E_dpz += den*z; 

    My_E_dpx_BG += x;
    My_E_dpy_BG += y;
    My_E_dpz_BG += z; 
    
  } /* BN */

  MPI_Allreduce(&My_E_dpx, &E_dpx, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_E_dpy, &E_dpy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_E_dpz, &E_dpz, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  MPI_Allreduce(&My_E_dpx_BG, &E_dpx_BG, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_E_dpy_BG, &E_dpy_BG, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_E_dpz_BG, &E_dpz_BG, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  E_dpx = E_dpx*GridVol;
  E_dpy = E_dpy*GridVol;
  E_dpz = E_dpz*GridVol;

  cden_BG = system_charge/Cell_Volume; 

  E_dpx_BG = E_dpx_BG*GridVol*cden_BG;
  E_dpy_BG = E_dpy_BG*GridVol*cden_BG;
  E_dpz_BG = E_dpz_BG*GridVol*cden_BG;

  /* contribution from core charge */

  My_C_dpx = 0.0;
  My_C_dpy = 0.0;
  My_C_dpz = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    x = Gxyz[Gc_AN][1];
    y = Gxyz[Gc_AN][2];
    z = Gxyz[Gc_AN][3];

    spe = WhatSpecies[Gc_AN];
    charge = Spe_Core_Charge[spe];
    My_C_dpx += charge*x;
    My_C_dpy += charge*y;
    My_C_dpz += charge*z;
  }

  MPI_Allreduce(&My_C_dpx, &C_dpx, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_C_dpy, &C_dpy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_C_dpz, &C_dpz, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  AU2Debye = 2.54174776;

  dipole_moment[0][1] = AU2Debye*(C_dpx - E_dpx - E_dpx_BG);
  dipole_moment[0][2] = AU2Debye*(C_dpy - E_dpy - E_dpy_BG);
  dipole_moment[0][3] = AU2Debye*(C_dpz - E_dpz - E_dpz_BG);

  dipole_moment[1][1] = AU2Debye*C_dpx;
  dipole_moment[1][2] = AU2Debye*C_dpy;
  dipole_moment[1][3] = AU2Debye*C_dpz;

  dipole_moment[2][1] = -AU2Debye*E_dpx;
  dipole_moment[2][2] = -AU2Debye*E_dpy;
  dipole_moment[2][3] = -AU2Debye*E_dpz;

  dipole_moment[3][1] = -AU2Debye*E_dpx_BG;
  dipole_moment[3][2] = -AU2Debye*E_dpy_BG;
  dipole_moment[3][3] = -AU2Debye*E_dpz_BG;

  AbsD = sqrt( dipole_moment[0][1]*dipole_moment[0][1]
             + dipole_moment[0][2]*dipole_moment[0][2]
             + dipole_moment[0][3]*dipole_moment[0][3] );

  if (myid==Host_ID){

    if (0<level_stdout){
      printf("\n*******************************************************\n"); fflush(stdout);
      printf("                  Dipole moment (Debye)                 \n");  fflush(stdout);
      printf("*******************************************************\n\n"); fflush(stdout);

      printf(" Absolute D %17.8f\n\n",AbsD);
      printf("                      Dx                Dy                Dz\n"); fflush(stdout);
      printf(" Total       %17.8f %17.8f %17.8f\n",
	     dipole_moment[0][1],dipole_moment[0][2],dipole_moment[0][3]);fflush(stdout);
      printf(" Core        %17.8f %17.8f %17.8f\n",
	     dipole_moment[1][1],dipole_moment[1][2],dipole_moment[1][3]);fflush(stdout);
      printf(" Electron    %17.8f %17.8f %17.8f\n",
	     dipole_moment[2][1],dipole_moment[2][2],dipole_moment[2][3]);fflush(stdout);
      printf(" Back ground %17.8f %17.8f %17.8f\n",
	     dipole_moment[3][1],dipole_moment[3][2],dipole_moment[3][3]);fflush(stdout);
    }

    /********************************************************
             write the dipole moments to a file
    ********************************************************/

    fnjoint(filepath,filename,file_DPM);

    if ((fp_DPM = fopen(file_DPM,"w")) != NULL){

#ifdef xt3
      setvbuf(fp_DPM,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      fprintf(fp_DPM,"\n");
      fprintf(fp_DPM,"***********************************************************\n");
      fprintf(fp_DPM,"***********************************************************\n");
      fprintf(fp_DPM,"                    Dipole moment (Debye)                  \n");
      fprintf(fp_DPM,"***********************************************************\n");
      fprintf(fp_DPM,"***********************************************************\n\n");

      fprintf(fp_DPM," Absolute D %17.8f\n\n",AbsD);
      fprintf(fp_DPM,"                      Dx                Dy                Dz\n");
      fprintf(fp_DPM," Total       %17.8f %17.8f %17.8f\n",
                dipole_moment[0][1],dipole_moment[0][2],dipole_moment[0][3]);
      fprintf(fp_DPM," Core        %17.8f %17.8f %17.8f\n",
                dipole_moment[1][1],dipole_moment[1][2],dipole_moment[1][3]);
      fprintf(fp_DPM," Electron    %17.8f %17.8f %17.8f\n",
                dipole_moment[2][1],dipole_moment[2][2],dipole_moment[2][3]);
      fprintf(fp_DPM," Back ground %17.8f %17.8f %17.8f\n",
                dipole_moment[3][1],dipole_moment[3][2],dipole_moment[3][3]);

      fclose(fp_DPM);
    }
    else{
      printf("Failure of saving the DPM file.\n");fflush(stdout);
    }
  }

}



void EH0_TwoCenter(int Gc_AN, int h_AN, double VH0ij[4])
{ 
  int n1,ban;
  int Gh_AN,Rn,wan1,wan2;
  double dv,x,y,z,r,r2,xx,va0,rho0,dr_va0;
  double z2,sum,sumr,sumx,sumy,sumz,wt;

  Gh_AN = natn[Gc_AN][h_AN];
  Rn = ncn[Gc_AN][h_AN];
  wan1 = WhatSpecies[Gc_AN];
  ban = Spe_Spe2Ban[wan1];
  wan2 = WhatSpecies[Gh_AN];
  dv = dv_EH0[ban];
  
  sum = 0.0;
  sumr = 0.0;

  for (n1=0; n1<TGN_EH0[ban]; n1++){
    x = GridX_EH0[ban][n1];
    y = GridY_EH0[ban][n1];
    z = GridZ_EH0[ban][n1];
    rho0 = Arho_EH0[ban][n1];
    wt = Wt_EH0[ban][n1];
    z2 = z - Dis[Gc_AN][h_AN];
    r2 = x*x + y*y + z2*z2;
    r = sqrt(r2);
    xx = 0.5*log(r2);

    /* for empty atoms or finite elemens basis */
    if (r<1.0e-10) r = 1.0e-10;

    va0 = VH_AtomF(wan2, 
                   Spe_Num_Mesh_VPS[wan2], xx, r, 
                   Spe_VPS_XV[wan2], Spe_VPS_RV[wan2], Spe_VH_Atom[wan2]);

    sum += wt*va0*rho0;

    if (h_AN!=0 && 1.0e-14<r){
      dr_va0 = Dr_VH_AtomF(wan2, 
                           Spe_Num_Mesh_VPS[wan2], xx, r, 
                           Spe_VPS_XV[wan2], Spe_VPS_RV[wan2], Spe_VH_Atom[wan2]);

      sumr -= wt*rho0*dr_va0*z2/r;
    }
  }

  sum  = sum*dv;

  if (h_AN!=0){

    /* for empty atoms or finite elemens basis */
    r = Dis[Gc_AN][h_AN];
    if (r<1.0e-10) r = 1.0e-10;

    x = Gxyz[Gc_AN][1] - (Gxyz[Gh_AN][1] + atv[Rn][1]);
    y = Gxyz[Gc_AN][2] - (Gxyz[Gh_AN][2] + atv[Rn][2]);
    z = Gxyz[Gc_AN][3] - (Gxyz[Gh_AN][3] + atv[Rn][3]);
    sumr = sumr*dv;
    sumx = sumr*x/r;
    sumy = sumr*y/r;
    sumz = sumr*z/r;
  }
  else{
    sumx = 0.0;
    sumy = 0.0;
    sumz = 0.0;
  }

  VH0ij[0] = sum;
  VH0ij[1] = sumx;
  VH0ij[2] = sumy;
  VH0ij[3] = sumz;
}














void EH0_TwoCenter_at_Cutoff(int wan1, int wan2, double VH0ij[4])
{ 
  int n1,ban;
  double dv,x,y,z,r1,r2,va0,rho0,dr_va0,rcut;
  double z2,sum,sumr,sumx,sumy,sumz,wt,r,xx;
  /* for OpenMP */
  int OMPID,Nthrds,Nthrds0,Nprocs,Nloop;
  double *my_sum_threads;

  ban = Spe_Spe2Ban[wan1];
  dv  = dv_EH0[ban];

  rcut = Spe_Atom_Cut1[wan1] + Spe_Atom_Cut1[wan2];

  /* get Nthrds0 */  
#pragma omp parallel shared(Nthrds0)
  {
    Nthrds0 = omp_get_num_threads();
  }

  /* allocation of array */
  my_sum_threads = (double*)malloc(sizeof(double)*Nthrds0);

  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    my_sum_threads[Nloop] = 0.0;
  }

#pragma omp parallel shared(Spe_VH_Atom,Spe_VPS_XV,Spe_VPS_RV,Spe_Num_Mesh_VPS,wan2,Wt_EH0,my_sum_threads,rcut,Arho_EH0,GridZ_EH0,GridY_EH0,GridX_EH0,TGN_EH0,ban) private(n1,OMPID,Nthrds,Nprocs,x,y,z,rho0,wt,z2,r2,va0,r,xx)
  {
    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (n1=OMPID*TGN_EH0[ban]/Nthrds; n1<(OMPID+1)*TGN_EH0[ban]/Nthrds; n1++){

      x = GridX_EH0[ban][n1];
      y = GridY_EH0[ban][n1];
      z = GridZ_EH0[ban][n1];
      rho0 = Arho_EH0[ban][n1];
      wt = Wt_EH0[ban][n1];
      z2 = z - rcut;
      r2 = x*x + y*y + z2*z2;
      r = sqrt(r2);
      xx = 0.5*log(r2);

      va0 = VH_AtomF(wan2, 
                     Spe_Num_Mesh_VPS[wan2], xx, r, 
                     Spe_VPS_XV[wan2], Spe_VPS_RV[wan2], Spe_VH_Atom[wan2]);

      my_sum_threads[OMPID] += wt*va0*rho0;
    }

  } /* #pragma omp parallel */

  sum  = 0.0;
  for (Nloop=0; Nloop<Nthrds0; Nloop++){
    sum += my_sum_threads[Nloop];
  }

  sum  = sum*dv;
  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;

  VH0ij[0] = sum;
  VH0ij[1] = sumx;
  VH0ij[2] = sumy;
  VH0ij[3] = sumz;

  /* freeing of array */
  free(my_sum_threads);
}





double Calc_Ehub()
{   
 /****************************************************
         LDA+U energy correction added by MJ
  ****************************************************/

  int Mc_AN,Gc_AN,wan1;
  int cnt1,cnt2,l1,mul1,m1,l2,mul2,m2;
  int spin,max_spin;
  double My_Ehub,Ehub,Uvalue,tmpv,sum;
  int numprocs,myid,ID;

  /* added by S.Ryee */
  int on_off,cnt_start,tmp_l1,ii,jj,kk,ll;	
  double Jvalue,tmpEhub1,tmpEhub2,tmpEhub3,tmpEhub4,trace_spin,trace_opp_spin;
  int dd;
  int NZUJ;
  dcomplex N_00_ac,N_11_ac,N_00_bd,N_11_bd,N_01_ac,N_10_ac,N_01_bd,N_10_bd;
  dcomplex AMF_00_ac,AMF_11_ac,AMF_00_bd,AMF_11_bd,AMF_01_ac,AMF_10_ac,AMF_01_bd,AMF_10_bd;
  dcomplex trace_N00,trace_N11,trace_N01,trace_N10;
  /*******************/ 


  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

 /****************************************************
                 caculation of My_Ehub
  ****************************************************/

  if      (SpinP_switch==0) max_spin = 0;
  else if (SpinP_switch==1) max_spin = 1;
  else if (SpinP_switch==3) max_spin = 1;

  My_Ehub = 0.0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];

   /****************************************************
                     collinear case
    ****************************************************/
    if (SpinP_switch!=3){
      switch (Hub_Type){
      case 1:		/* Dudarev form */
        for (spin=0; spin<=max_spin; spin++){

          /* Hubbard term, 0.5*Tr(N) */

	  cnt1 = 0;
	  for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
	    for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){

	      Uvalue = Hub_U_Basis[wan1][l1][mul1];
	      for(m1=0; m1<(2*l1+1); m1++){

                tmpv = 0.5*Uvalue*DM_onsite[0][spin][Mc_AN][cnt1][cnt1];
	        My_Ehub += tmpv;

	        cnt1++;
	      }
	    }
	  }


          /* Hubbard term, -0.5*Tr(N*N) */

	  cnt1 = 0;
	  for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
	    for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){
	      for(m1=0; m1<(2*l1+1); m1++){

                sum = 0.0;  

	        cnt2 = 0;
	        for(l2=0; l2<=Spe_MaxL_Basis[wan1]; l2++ ){
		  for(mul2=0; mul2<Spe_Num_Basis[wan1][l2]; mul2++){
		    for(m2=0; m2<(2*l2+1); m2++){

		      if (l1==l2 && mul1==mul2){
                      
		        Uvalue = Hub_U_Basis[wan1][l1][mul1];
		        sum -= 0.5*Uvalue*DM_onsite[0][spin][Mc_AN][cnt1][cnt2]*
			                  DM_onsite[0][spin][Mc_AN][cnt2][cnt1];
		      }

		      cnt2++;
		    }
		  }
	        }

                My_Ehub += sum;

	        cnt1++;
	      }
	    }
	  }
	} /* spin */
      break;

      case 2:		/* general form by S.Ryee */
	/* U Energy */

	for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++){
	  for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){
	    Uvalue = Hub_U_Basis[wan1][l1][mul1];
	    Jvalue = Hund_J_Basis[wan1][l1][mul1];
            NZUJ = Nonzero_UJ[wan1][l1][mul1];
            if(NZUJ>0){
	      cnt_start = 0;
	      switch (mul1){
	      case 0:	/* mul1 = 0 */
	        if(l1 > 0){
		  for(tmp_l1=0; tmp_l1<l1; tmp_l1++){
		    cnt_start += (2*tmp_l1+1)*Spe_Num_Basis[wan1][tmp_l1];
		  }
	        }
	        else{	/* l1 = 0 */
		  cnt_start = 0;
	        }
	      break;
	
	      case 1:	/* mul1 = 1 */
	        if(l1 > 0){
		  for(tmp_l1=0; tmp_l1<l1; tmp_l1++){
		    cnt_start += (2*tmp_l1+1)*Spe_Num_Basis[wan1][tmp_l1];
		  }
		  cnt_start += (2*l1+1)*mul1;
	        }
	        else{	/* l1 = 0 */
		  cnt_start = (2*l1+1)*mul1;
	        }
	      break;
	      }	/* switch (mul1) */


	      trace_spin = 0.0;
	      trace_opp_spin = 0.0;
	      for(ii=0; ii<(2*l1+1); ii++){
	        trace_spin += DM_onsite[0][0][Mc_AN][cnt_start+ii][cnt_start+ii];
	        trace_opp_spin += DM_onsite[0][1][Mc_AN][cnt_start+ii][cnt_start+ii];
	      }
	      /* dc Energy */
	      if(dc_Type==1){  /* sFLL */
	        My_Ehub -= 0.5*(Uvalue)*(trace_spin+trace_opp_spin)*(trace_spin+trace_opp_spin-1.0) 
		          -0.5*(Jvalue)*(trace_spin*(trace_spin-1.0)+trace_opp_spin*(trace_opp_spin-1.0));
	      } /* sFLL */

              if(dc_Type==3){  /* cFLL */
	        My_Ehub -= 0.5*(Uvalue)*(trace_spin+trace_opp_spin)*(trace_spin+trace_opp_spin-1.0) 
		          -0.25*(Jvalue)*(trace_spin+trace_opp_spin)*(trace_spin+trace_opp_spin-2.0);
	      } /* cFLL */


             /*f(dc_Type==3){  
	        My_Ehub -= dc_alpha[count]*(0.5*(Uvalue)*(trace_spin+trace_opp_spin)*(trace_spin+trace_opp_spin-1.0) 
		                    -0.5*(Jvalue)*(trace_spin*(trace_spin-1.0)+trace_opp_spin*(trace_opp_spin-1.0)));
   	      } */

              /* loop start for interaction energy */
	      for(ii=0; ii<(2*l1+1); ii++){
                for(jj=0; jj<(2*l1+1); jj++){
	          for(kk=0; kk<(2*l1+1); kk++){
	            for(ll=0; ll<(2*l1+1); ll++){
                      switch(dc_Type){
                      case 1:  /* sFLL */
		        tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                      (DM_onsite[0][0][Mc_AN][cnt_start+ii][cnt_start+kk]*
			    	       DM_onsite[0][1][Mc_AN][cnt_start+jj][cnt_start+ll]
                                      +DM_onsite[0][1][Mc_AN][cnt_start+ii][cnt_start+kk]*
                                       DM_onsite[0][0][Mc_AN][cnt_start+jj][cnt_start+ll]);
		        tmpEhub2 = 0.5*(Coulomb_Array[NZUJ][ii][jj][kk][ll]-Coulomb_Array[NZUJ][ii][jj][ll][kk])*
				      (DM_onsite[0][0][Mc_AN][cnt_start+ii][cnt_start+kk]*
				       DM_onsite[0][0][Mc_AN][cnt_start+jj][cnt_start+ll]
                                      +DM_onsite[0][1][Mc_AN][cnt_start+ii][cnt_start+kk]*
                                       DM_onsite[0][1][Mc_AN][cnt_start+jj][cnt_start+ll]);
	                My_Ehub += tmpEhub1 + tmpEhub2;
                      break;
              
                      case 2:  /* sAMF */
                        tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                      (AMF_Array[NZUJ][0][0][ii][kk]*
			    	       AMF_Array[NZUJ][1][0][jj][ll]
                                      +AMF_Array[NZUJ][1][0][ii][kk]*
                                       AMF_Array[NZUJ][0][0][jj][ll]);
		        tmpEhub2 = 0.5*(Coulomb_Array[NZUJ][ii][jj][kk][ll]-Coulomb_Array[NZUJ][ii][jj][ll][kk])*
				      (AMF_Array[NZUJ][0][0][ii][kk]*
				       AMF_Array[NZUJ][0][0][jj][ll]
                                      +AMF_Array[NZUJ][1][0][ii][kk]*
                                       AMF_Array[NZUJ][1][0][jj][ll]);
	                My_Ehub += tmpEhub1 + tmpEhub2;
                      break;

                      case 3:  /* cFLL */
		        tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                      (DM_onsite[0][0][Mc_AN][cnt_start+ii][cnt_start+kk]*
			    	       DM_onsite[0][1][Mc_AN][cnt_start+jj][cnt_start+ll]
                                      +DM_onsite[0][1][Mc_AN][cnt_start+ii][cnt_start+kk]*
                                       DM_onsite[0][0][Mc_AN][cnt_start+jj][cnt_start+ll]);
		        tmpEhub2 = 0.5*(Coulomb_Array[NZUJ][ii][jj][kk][ll]-Coulomb_Array[NZUJ][ii][jj][ll][kk])*
				      (DM_onsite[0][0][Mc_AN][cnt_start+ii][cnt_start+kk]*
				       DM_onsite[0][0][Mc_AN][cnt_start+jj][cnt_start+ll]
                                      +DM_onsite[0][1][Mc_AN][cnt_start+ii][cnt_start+kk]*
                                       DM_onsite[0][1][Mc_AN][cnt_start+jj][cnt_start+ll]);
	                My_Ehub += tmpEhub1 + tmpEhub2;
                      break;

                      case 4:  /* cAMF */
                        tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                      (AMF_Array[NZUJ][0][0][ii][kk]*
			    	       AMF_Array[NZUJ][1][0][jj][ll]
                                      +AMF_Array[NZUJ][1][0][ii][kk]*
                                       AMF_Array[NZUJ][0][0][jj][ll]);
		        tmpEhub2 = 0.5*(Coulomb_Array[NZUJ][ii][jj][kk][ll]-Coulomb_Array[NZUJ][ii][jj][ll][kk])*
				      (AMF_Array[NZUJ][0][0][ii][kk]*
				       AMF_Array[NZUJ][0][0][jj][ll]
                                      +AMF_Array[NZUJ][1][0][ii][kk]*
                                       AMF_Array[NZUJ][1][0][jj][ll]);
	                My_Ehub += tmpEhub1 + tmpEhub2;
                      break;

                /*      case 3:  
		        tmpEhub1 = 0.5*dc_alpha[count]*(Coulomb_Array[count][ii][jj][kk][ll]*
                                      (DM_onsite[0][0][Mc_AN][cnt_start+ii][cnt_start+kk]*
			    	       DM_onsite[0][1][Mc_AN][cnt_start+jj][cnt_start+ll]
                                      +DM_onsite[0][1][Mc_AN][cnt_start+ii][cnt_start+kk]*
                                       DM_onsite[0][0][Mc_AN][cnt_start+jj][cnt_start+ll]));
		        tmpEhub2 = 0.5*dc_alpha[count]*((Coulomb_Array[count][ii][jj][kk][ll]-Coulomb_Array[count][ii][jj][ll][kk])*
				      (DM_onsite[0][0][Mc_AN][cnt_start+ii][cnt_start+kk]*
				       DM_onsite[0][0][Mc_AN][cnt_start+jj][cnt_start+ll]
                                      +DM_onsite[0][1][Mc_AN][cnt_start+ii][cnt_start+kk]*
                                       DM_onsite[0][1][Mc_AN][cnt_start+jj][cnt_start+ll]));
                        tmpEhub3 = 0.5*(1.0-dc_alpha[count])*(Coulomb_Array[count][ii][jj][kk][ll]*
                                      (AMF_Array[count][0][0][ii][kk]*
			    	       AMF_Array[count][1][0][jj][ll]
                                      +AMF_Array[count][1][0][ii][kk]*
                                       AMF_Array[count][0][0][jj][ll]));
		        tmpEhub4 = 0.5*(1.0-dc_alpha[count])*((Coulomb_Array[count][ii][jj][kk][ll]-Coulomb_Array[count][ii][jj][ll][kk])*
				      (AMF_Array[count][0][0][ii][kk]*
				       AMF_Array[count][0][0][jj][ll]
                                      +AMF_Array[count][1][0][ii][kk]*
                                       AMF_Array[count][1][0][jj][ll]));
                        My_Ehub += tmpEhub1 + tmpEhub2 + tmpEhub3 + tmpEhub4;
                      break; */
                      } /* switch dc_Type */
		    }
		  }
	        }
	      }      
	      
            }
	  }	/* mul1 */
	}	/* l1 */
      break;
 
      } /* Hub_Type */
    } /* SpinP_switch */ 

   /****************************************************
                     non-collinear case
    ****************************************************/

    else {
    
      switch (Hub_Type){ 
      case 1:	/* Dudarev form */
        /* Hubbard term, 0.5*Tr(N) */

        cnt1 = 0;
        for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
  	  for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){

	    Uvalue = Hub_U_Basis[wan1][l1][mul1];
	    for(m1=0; m1<(2*l1+1); m1++){

	      tmpv = 0.5*Uvalue*( NC_OcpN[0][0][0][Mc_AN][cnt1][cnt1].r
			      + NC_OcpN[0][1][1][Mc_AN][cnt1][cnt1].r);
              My_Ehub += tmpv;

	      cnt1++;
	    }
	  }
        }

        /* Hubbard term, -0.5*Tr(N*N) */

        cnt1 = 0;
        for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
 	  for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){
	    for(m1=0; m1<(2*l1+1); m1++){

              sum = 0.0;  

	      cnt2 = 0;
	      for(l2=0; l2<=Spe_MaxL_Basis[wan1]; l2++ ){
	        for(mul2=0; mul2<Spe_Num_Basis[wan1][l2]; mul2++){
		  for(m2=0; m2<(2*l2+1); m2++){

		    if (l1==l2 && mul1==mul2){

		      Uvalue = Hub_U_Basis[wan1][l1][mul1];

		      sum -= 0.5*Uvalue*( NC_OcpN[0][0][0][Mc_AN][cnt1][cnt2].r*
			                NC_OcpN[0][0][0][Mc_AN][cnt1][cnt2].r
					    +
				        NC_OcpN[0][0][0][Mc_AN][cnt1][cnt2].i*
					NC_OcpN[0][0][0][Mc_AN][cnt1][cnt2].i
					    +
				        NC_OcpN[0][0][1][Mc_AN][cnt1][cnt2].r*
					NC_OcpN[0][0][1][Mc_AN][cnt1][cnt2].r
					    +
					NC_OcpN[0][0][1][Mc_AN][cnt1][cnt2].i*
					NC_OcpN[0][0][1][Mc_AN][cnt1][cnt2].i
					    +
					NC_OcpN[0][1][0][Mc_AN][cnt1][cnt2].r*
					NC_OcpN[0][1][0][Mc_AN][cnt1][cnt2].r
					    +
					NC_OcpN[0][1][0][Mc_AN][cnt1][cnt2].i*
					NC_OcpN[0][1][0][Mc_AN][cnt1][cnt2].i
					    +
					NC_OcpN[0][1][1][Mc_AN][cnt1][cnt2].r*
					NC_OcpN[0][1][1][Mc_AN][cnt1][cnt2].r
					    +
					NC_OcpN[0][1][1][Mc_AN][cnt1][cnt2].i*
					NC_OcpN[0][1][1][Mc_AN][cnt1][cnt2].i );

		    }

		    cnt2++;
		  }
	        }
	      }

              My_Ehub += sum;

	      cnt1++;
	    } /* m1 */
	  } /* mul1 */
        } /* l1 */
      break;

      case 2:	/* general form by S.Ryee */

        /* U Energy */
        for(l1=0; l1<=Spe_MaxL_Basis[wan1]; l1++ ){
          for(mul1=0; mul1<Spe_Num_Basis[wan1][l1]; mul1++){
           Uvalue = Hub_U_Basis[wan1][l1][mul1];
           Jvalue = Hund_J_Basis[wan1][l1][mul1];
           NZUJ = Nonzero_UJ[wan1][l1][mul1];
           if(NZUJ>0){
            cnt_start = 0;
            switch (mul1){
            case 0:   /* mul1 = 0 */
              if(l1 > 0){
                for(tmp_l1=0; tmp_l1<l1; tmp_l1++){
                  cnt_start += (2*tmp_l1+1)*Spe_Num_Basis[wan1][tmp_l1];
                }
              }
              else{   /* l1 = 0 */
                cnt_start = 0;
              }
            break;

            case 1:   /* mul1 = 1 */
              if(l1 > 0){
                for(tmp_l1=0; tmp_l1<l1; tmp_l1++){
                  cnt_start += (2*tmp_l1+1)*Spe_Num_Basis[wan1][tmp_l1];
                }
                cnt_start += (2*l1+1)*mul1;
              }
              else{   /* l1 = 0 */
                cnt_start = (2*l1+1)*mul1;
              }
            break;
            } /* switch (mul1) */


	    trace_N00.r = 0.0;
            trace_N00.i = 0.0;
	    trace_N11.r = 0.0;
	    trace_N11.i = 0.0;
            trace_N01.r = 0.0;
            trace_N01.i = 0.0;
            trace_N10.r = 0.0;
            trace_N10.i = 0.0;

            for(dd=0; dd<(2*l1+1); dd++){
              trace_N00.r += NC_OcpN[0][0][0][Mc_AN][cnt_start+dd][cnt_start+dd].r;
	      trace_N00.i += NC_OcpN[0][0][0][Mc_AN][cnt_start+dd][cnt_start+dd].i;
	      trace_N11.r += NC_OcpN[0][1][1][Mc_AN][cnt_start+dd][cnt_start+dd].r;
	      trace_N11.i += NC_OcpN[0][1][1][Mc_AN][cnt_start+dd][cnt_start+dd].i;

              trace_N01.r += NC_OcpN[0][0][1][Mc_AN][cnt_start+dd][cnt_start+dd].r;
              trace_N01.i += NC_OcpN[0][0][1][Mc_AN][cnt_start+dd][cnt_start+dd].i;
              trace_N10.r += NC_OcpN[0][1][0][Mc_AN][cnt_start+dd][cnt_start+dd].r;
              trace_N10.i += NC_OcpN[0][1][0][Mc_AN][cnt_start+dd][cnt_start+dd].i;
            }
	    /* Double counting energy */
	    if(dc_Type==1){  /* sFLL */
	      My_Ehub -= 0.5*(Uvalue)*(trace_N00.r+trace_N11.r)*(trace_N00.r+trace_N11.r-1.0)
		        -0.5*(Jvalue)*(Cmul(trace_N00,Csub(trace_N00,Complex(1.0,0.0))).r
				    +Cmul(trace_N11,Csub(trace_N11,Complex(1.0,0.0))).r);
	      My_Ehub -=-0.5*(Jvalue)*(Cmul(trace_N01,trace_N10).r + Cmul(trace_N10,trace_N01).r);
	
	    }  /* sFLL */

            if(dc_Type==3){ /* cFLL */
              My_Ehub -= 0.5*(Uvalue)*(trace_N00.r+trace_N11.r)*(trace_N00.r+trace_N11.r-1.0)
		        -0.25*(Jvalue)*(trace_N00.r+trace_N11.r)*(trace_N00.r+trace_N11.r-2.0);
	    }  /* cFLL */


          /*  if(dc_Type==3){  
	      My_Ehub -= dc_alpha[count]*(0.5*(Uvalue)*(trace_N00.r+trace_N11.r)*(trace_N00.r+trace_N11.r-1.0)
		                  -0.5*(Jvalue)*(Cmul(trace_N00,Csub(trace_N00,Complex(1.0,0.0))).r
				  +Cmul(trace_N11,Csub(trace_N11,Complex(1.0,0.0))).r));
	      My_Ehub -=-dc_alpha[count]*(0.5*(Jvalue)*(Cmul(trace_N01,trace_N10).r + Cmul(trace_N10,trace_N01).r));
            }  */

            /* loop start for interaction energy */
            for(ii=0; ii<(2*l1+1); ii++){
    	      for(jj=0; jj<(2*l1+1); jj++){
	        for(kk=0; kk<(2*l1+1); kk++){
	          for(ll=0; ll<(2*l1+1); ll++){
                    switch(dc_Type){
                    case 1:  /* sFLL */
		      N_00_ac.r=NC_OcpN[0][0][0][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_00_ac.i=NC_OcpN[0][0][0][Mc_AN][cnt_start+ii][cnt_start+kk].i;
		      N_11_ac.r=NC_OcpN[0][1][1][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_11_ac.i=NC_OcpN[0][1][1][Mc_AN][cnt_start+ii][cnt_start+kk].i;

		      N_00_bd.r=NC_OcpN[0][0][0][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_00_bd.i=NC_OcpN[0][0][0][Mc_AN][cnt_start+jj][cnt_start+ll].i;
		      N_11_bd.r=NC_OcpN[0][1][1][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_11_bd.i=NC_OcpN[0][1][1][Mc_AN][cnt_start+jj][cnt_start+ll].i;

		      N_01_ac.r=NC_OcpN[0][0][1][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_01_ac.i=NC_OcpN[0][0][1][Mc_AN][cnt_start+ii][cnt_start+kk].i;
		      N_10_ac.r=NC_OcpN[0][1][0][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_10_ac.i=NC_OcpN[0][1][0][Mc_AN][cnt_start+ii][cnt_start+kk].i;

		      N_01_bd.r=NC_OcpN[0][0][1][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_01_bd.i=NC_OcpN[0][0][1][Mc_AN][cnt_start+jj][cnt_start+ll].i;
		      N_10_bd.r=NC_OcpN[0][1][0][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_10_bd.i=NC_OcpN[0][1][0][Mc_AN][cnt_start+jj][cnt_start+ll].i;

                      /* diagonal term */
   		      tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      /* off-diagonal term */
		      tmpEhub2 = -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);
		      /* LDA+U energy */
		      My_Ehub += tmpEhub1 + tmpEhub2;	
                    break;
                   
                    case 2:  /* sAMF */
                      N_00_ac.r=AMF_Array[NZUJ][0][0][ii][kk];
		      N_00_ac.i=AMF_Array[NZUJ][0][1][ii][kk];
		      N_11_ac.r=AMF_Array[NZUJ][1][0][ii][kk];
		      N_11_ac.i=AMF_Array[NZUJ][1][1][ii][kk];

		      N_00_bd.r=AMF_Array[NZUJ][0][0][jj][ll];
		      N_00_bd.i=AMF_Array[NZUJ][0][1][jj][ll];
		      N_11_bd.r=AMF_Array[NZUJ][1][0][jj][ll];
		      N_11_bd.i=AMF_Array[NZUJ][1][1][jj][ll];

		      N_01_ac.r=AMF_Array[NZUJ][2][0][ii][kk];
		      N_01_ac.i=AMF_Array[NZUJ][2][1][ii][kk];
		      N_10_ac.r=AMF_Array[NZUJ][3][0][ii][kk];
		      N_10_ac.i=AMF_Array[NZUJ][3][1][ii][kk];

		      N_01_bd.r=AMF_Array[NZUJ][2][0][jj][ll];
		      N_01_bd.i=AMF_Array[NZUJ][2][1][jj][ll];
		      N_10_bd.r=AMF_Array[NZUJ][3][0][jj][ll];
		      N_10_bd.i=AMF_Array[NZUJ][3][1][jj][ll];

                      /* diagonal term */
		      tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      /* off-diagonal term */
		      tmpEhub2 = -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);
		      /* LDA+U energy */
		      My_Ehub += tmpEhub1 + tmpEhub2;	
                    break;

                    case 3:  /* cFLL */
		      N_00_ac.r=NC_OcpN[0][0][0][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_00_ac.i=NC_OcpN[0][0][0][Mc_AN][cnt_start+ii][cnt_start+kk].i;
		      N_11_ac.r=NC_OcpN[0][1][1][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_11_ac.i=NC_OcpN[0][1][1][Mc_AN][cnt_start+ii][cnt_start+kk].i;

		      N_00_bd.r=NC_OcpN[0][0][0][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_00_bd.i=NC_OcpN[0][0][0][Mc_AN][cnt_start+jj][cnt_start+ll].i;
		      N_11_bd.r=NC_OcpN[0][1][1][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_11_bd.i=NC_OcpN[0][1][1][Mc_AN][cnt_start+jj][cnt_start+ll].i;

		      N_01_ac.r=NC_OcpN[0][0][1][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_01_ac.i=NC_OcpN[0][0][1][Mc_AN][cnt_start+ii][cnt_start+kk].i;
		      N_10_ac.r=NC_OcpN[0][1][0][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_10_ac.i=NC_OcpN[0][1][0][Mc_AN][cnt_start+ii][cnt_start+kk].i;

		      N_01_bd.r=NC_OcpN[0][0][1][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_01_bd.i=NC_OcpN[0][0][1][Mc_AN][cnt_start+jj][cnt_start+ll].i;
		      N_10_bd.r=NC_OcpN[0][1][0][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_10_bd.i=NC_OcpN[0][1][0][Mc_AN][cnt_start+jj][cnt_start+ll].i;

                      /* diagonal term */
   		      tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      /* off-diagonal term */
		      tmpEhub2 = -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);
		      /* LDA+U energy */
		      My_Ehub += tmpEhub1 + tmpEhub2;	
                    break;
 
                    case 4:  /* cAMF */
                      N_00_ac.r=AMF_Array[NZUJ][0][0][ii][kk];
		      N_00_ac.i=AMF_Array[NZUJ][0][1][ii][kk];
		      N_11_ac.r=AMF_Array[NZUJ][1][0][ii][kk];
		      N_11_ac.i=AMF_Array[NZUJ][1][1][ii][kk];

		      N_00_bd.r=AMF_Array[NZUJ][0][0][jj][ll];
		      N_00_bd.i=AMF_Array[NZUJ][0][1][jj][ll];
		      N_11_bd.r=AMF_Array[NZUJ][1][0][jj][ll];
		      N_11_bd.i=AMF_Array[NZUJ][1][1][jj][ll];

		      N_01_ac.r=AMF_Array[NZUJ][2][0][ii][kk];
		      N_01_ac.i=AMF_Array[NZUJ][2][1][ii][kk];
		      N_10_ac.r=AMF_Array[NZUJ][3][0][ii][kk];
		      N_10_ac.i=AMF_Array[NZUJ][3][1][ii][kk];

		      N_01_bd.r=AMF_Array[NZUJ][2][0][jj][ll];
		      N_01_bd.i=AMF_Array[NZUJ][2][1][jj][ll];
		      N_10_bd.r=AMF_Array[NZUJ][3][0][jj][ll];
		      N_10_bd.i=AMF_Array[NZUJ][3][1][jj][ll];

                      /* diagonal term */
		      tmpEhub1 = 0.5*Coulomb_Array[NZUJ][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      /* off-diagonal term */
		      tmpEhub2 = -0.5*Coulomb_Array[NZUJ][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);
		      /* LDA+U energy */
		      My_Ehub += tmpEhub1 + tmpEhub2;	
                    break;
                /*    case 3:  
		      N_00_ac.r=NC_OcpN[0][0][0][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_00_ac.i=NC_OcpN[0][0][0][Mc_AN][cnt_start+ii][cnt_start+kk].i;
		      N_11_ac.r=NC_OcpN[0][1][1][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_11_ac.i=NC_OcpN[0][1][1][Mc_AN][cnt_start+ii][cnt_start+kk].i;

		      N_00_bd.r=NC_OcpN[0][0][0][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_00_bd.i=NC_OcpN[0][0][0][Mc_AN][cnt_start+jj][cnt_start+ll].i;
		      N_11_bd.r=NC_OcpN[0][1][1][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_11_bd.i=NC_OcpN[0][1][1][Mc_AN][cnt_start+jj][cnt_start+ll].i;

		      N_01_ac.r=NC_OcpN[0][0][1][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_01_ac.i=NC_OcpN[0][0][1][Mc_AN][cnt_start+ii][cnt_start+kk].i;
		      N_10_ac.r=NC_OcpN[0][1][0][Mc_AN][cnt_start+ii][cnt_start+kk].r;
		      N_10_ac.i=NC_OcpN[0][1][0][Mc_AN][cnt_start+ii][cnt_start+kk].i;

		      N_01_bd.r=NC_OcpN[0][0][1][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_01_bd.i=NC_OcpN[0][0][1][Mc_AN][cnt_start+jj][cnt_start+ll].i;
		      N_10_bd.r=NC_OcpN[0][1][0][Mc_AN][cnt_start+jj][cnt_start+ll].r;
		      N_10_bd.i=NC_OcpN[0][1][0][Mc_AN][cnt_start+jj][cnt_start+ll].i;

                      AMF_00_ac.r=AMF_Array[count][0][0][ii][kk];
		      AMF_00_ac.i=AMF_Array[count][0][1][ii][kk];
		      AMF_11_ac.r=AMF_Array[count][1][0][ii][kk];
		      AMF_11_ac.i=AMF_Array[count][1][1][ii][kk];

		      AMF_00_bd.r=AMF_Array[count][0][0][jj][ll];
		      AMF_00_bd.i=AMF_Array[count][0][1][jj][ll];
		      AMF_11_bd.r=AMF_Array[count][1][0][jj][ll];
		      AMF_11_bd.i=AMF_Array[count][1][1][jj][ll];

		      AMF_01_ac.r=AMF_Array[count][2][0][ii][kk];
		      AMF_01_ac.i=AMF_Array[count][2][1][ii][kk];
		      AMF_10_ac.r=AMF_Array[count][3][0][ii][kk];
		      AMF_10_ac.i=AMF_Array[count][3][1][ii][kk];

		      AMF_01_bd.r=AMF_Array[count][2][0][jj][ll];
		      AMF_01_bd.i=AMF_Array[count][2][1][jj][ll];
		      AMF_10_bd.r=AMF_Array[count][3][0][jj][ll];
		      AMF_10_bd.i=AMF_Array[count][3][1][jj][ll];

		      tmpEhub1 = 0.5*Coulomb_Array[count][ii][jj][kk][ll]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r
			            +Cmul(N_00_ac,N_11_bd).r + Cmul(N_11_ac,N_00_bd).r)
			        -0.5*Coulomb_Array[count][ii][jj][ll][kk]*
                                    (Cmul(N_00_ac,N_00_bd).r + Cmul(N_11_ac,N_11_bd).r);
		      tmpEhub2 = -0.5*Coulomb_Array[count][ii][jj][ll][kk]*
                                     (Cmul(N_01_ac,N_10_bd).r + Cmul(N_10_ac,N_01_bd).r);

		      tmpEhub3 = 0.5*Coulomb_Array[count][ii][jj][kk][ll]*
                                    (Cmul(AMF_00_ac,AMF_00_bd).r + Cmul(AMF_11_ac,AMF_11_bd).r
			            +Cmul(AMF_00_ac,AMF_11_bd).r + Cmul(AMF_11_ac,AMF_00_bd).r)
			        -0.5*Coulomb_Array[count][ii][jj][ll][kk]*
                                    (Cmul(AMF_00_ac,AMF_00_bd).r + Cmul(AMF_11_ac,AMF_11_bd).r);
		      tmpEhub4 = -0.5*Coulomb_Array[count][ii][jj][ll][kk]*
                                     (Cmul(AMF_01_ac,AMF_10_bd).r + Cmul(AMF_10_ac,AMF_01_bd).r);

                      My_Ehub += dc_alpha[count]*(tmpEhub1+tmpEhub2) + (1.0-dc_alpha[count])*(tmpEhub3+tmpEhub4);
                    break; */
		    } /* dc switch */
		  }
		}
	      }
	    }
           } /* Uvalue != 0.0 || Jvalue != 0.0 */

          } /* mul1 */
        } /* l1 */

      break;
      } /* Hub_Type */

    } /* SpinP_switch */
  
  } /* Mc_AN */
 

  if (SpinP_switch==0) My_Ehub = 2.0*My_Ehub;

 /****************************************************
                      MPI My_Ehub
  ****************************************************/

  MPI_Allreduce(&My_Ehub, &Ehub, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* if (F_U_flag==0) */
  if (F_U_flag==0) Ehub = 0.0;
 
  return Ehub;  
}







/* okuno */
double Calc_EdftD()
{
  /************************************************
     The subroutine calculates the semiemprical 
     vdW correction to DFT-GGA proposed by 
     S. Grimme, J. Comput. Chem. 27, 1787 (2006).
  *************************************************/

  double My_EdftD,EdftD;
  double rij[4],fdamp,fdamp2;
  double rij0[4],par;
  double dist,dist6,dist2;
  double exparg,expval;
  double rcut_dftD2;
  int numprocs,myid,ID;
  int Mc_AN,Gc_AN,wanA,wanB;
  int Gc_BN;
  int nrm,nr;
  int i,j;
  int n1,n2,n3;
  int per_flag1,per_flag2;
  int n1_max,n2_max,n3_max; 
  double test_ene;
  double dblcnt_factor;
  double E,dEx,dEy,dEz,dist7;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  My_EdftD = 0.0;
  EdftD    = 0.0;
  rcut_dftD2 = rcut_dftD*rcut_dftD;

  dblcnt_factor = 0.5;

  /* here we calculate DFT-D dispersion energy */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    E = 0.0;
    dEx = 0.0;
    dEy = 0.0;
    dEz = 0.0;

    Gc_AN = M2G[Mc_AN];
    wanA = WhatSpecies[Gc_AN];
    per_flag1 = (int)Gxyz[Gc_AN][60]; 

    for(Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){

      wanB = WhatSpecies[Gc_BN];
      per_flag2 = (int)Gxyz[Gc_BN][60]; 

      rij0[1] = Gxyz[Gc_AN][1] - Gxyz[Gc_BN][1];
      rij0[2] = Gxyz[Gc_AN][2] - Gxyz[Gc_BN][2];
      rij0[3] = Gxyz[Gc_AN][3] - Gxyz[Gc_BN][3];

      par = beta_dftD/(Rsum_dftD[wanA][wanB]);

      if (per_flag1==0 && per_flag2==0){
        n1_max = 0;
        n2_max = 0;
        n3_max = 0;
      }
      else if (per_flag1==0 && per_flag2==1){
        n1_max = n1_DFT_D;
        n2_max = n2_DFT_D;
        n3_max = n3_DFT_D;
      }
      else if (per_flag1==1 && per_flag2==0){
        n1_max = 0;
        n2_max = 0;
        n3_max = 0;
      }
      else if (per_flag1==1 && per_flag2==1){
        n1_max = n1_DFT_D;
        n2_max = n2_DFT_D;
        n3_max = n3_DFT_D;
      }

      /*
      printf("Gc_AN=%2d Gc_BN=%2d %2d %2d %2d %2d %2d\n",Gc_AN,Gc_BN,per_flag1,per_flag2,n1_max,n2_max,n3_max);
      */

      for (n1=-n1_max; n1<=n1_max; n1++){
	for (n2=-n2_max; n2<=n2_max; n2++){
	  for (n3=-n3_max; n3<=n3_max; n3++){
            
            /* for double counting */
            if((!(abs(n1)+abs(n2)+abs(n3))==0) && (per_flag1==0 && per_flag2==1) ){
	      dblcnt_factor = 1.0;
	    }
            else{
	      dblcnt_factor = 0.5;
            }

	    rij[1] = rij0[1] - ( (double)n1*tv[1][1]
	                       + (double)n2*tv[2][1] 
	                       + (double)n3*tv[3][1] ); 

	    rij[2] = rij0[2] - ( (double)n1*tv[1][2]
			       + (double)n2*tv[2][2] 
			       + (double)n3*tv[3][2] ); 

	    rij[3] = rij0[3] - ( (double)n1*tv[1][3]
			       + (double)n2*tv[2][3] 
			       + (double)n3*tv[3][3] ); 

            dist2 = rij[1]*rij[1] + rij[2]*rij[2] + rij[3]*rij[3];

            if (0.1<dist2 && dist2<=rcut_dftD2){

	      dist  = sqrt(dist2); 
	      dist6 = dist2*dist2*dist2;            

	      /* calculate the vdW energy */
	      exparg = -beta_dftD*((dist/Rsum_dftD[wanA][wanB])-1.0);
	      expval = exp(exparg);
	      fdamp = scal6_dftD/(1.0+expval);

	      E -= dblcnt_factor*C6ij_dftD[wanA][wanB]/dist6*fdamp;

	      /* calculate the gradient of the vdW energy */

              dist7 = dist6 * dist;
	      fdamp2 = C6ij_dftD[wanA][wanB]*fdamp/dist6*(expval*par/(1.0+expval) - 6.0/dist);
              dEx -= fdamp2*rij[1]/dist;
              dEy -= fdamp2*rij[2]/dist;
              dEz -= fdamp2*rij[3]/dist;
	    }

	  } /* n3 */
	} /* n2 */
      } /* n1 */
    } /* Gc_BN */

    My_EdftD += E;

    /* energy decomposition */

    if (Energy_Decomposition_flag==1){

      DecEvdw[0][Mc_AN][0] = E;
      DecEvdw[1][Mc_AN][0] = E;
    }

    /* gradients from two-body terms */

    Gxyz[Gc_AN][17] += dEx;
    Gxyz[Gc_AN][18] += dEy;
    Gxyz[Gc_AN][19] += dEz;

    /*
    printf("Gc_AN=%2d dEx=%15.12f dEy=%15.12f dEz=%15.12f\n",Gc_AN,dEx,dEy,dEz);
    */

  } /* Mc_AN */

  MPI_Allreduce(&My_EdftD, &EdftD, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  return EdftD;
}
/* okuno */



/* Ellner */
double Calc_EdftD3()
{
  /***********************************************************************
   The subroutine calculates the semiemprical DFTD3 vdW correction
   DFTD3 with zero damping:
    Grimme, S. et al. H. J. Chem. Phys. (2010), 132, 154104
   DFTD3 with BJ damping
    Becke, A. D.; Johnson, E. R. J. Chem. Phys. 2005, 122, 154101
    Johnson, E. R.; Becke, A. D. J. Chem. Phys. 2005, 123, 024101
    Johnson, E. R.; Becke, A. D. J. Chem. Phys. 2006, 124, 174104
  ************************************************************************/

  /* VARIABLES DECLARATOIN */
  double My_EdftD,EdftD; /* energy */
  double E; /* atomic energy */
  double rij[4],fdamp,fdamp6,fdamp8,t6,t62,t8,t82,dE6,dE8,dEC,**dEC0; /* interaction */
  double rij0[4]; /* positions */
  double dist,dist2,dist5,dist6,dist7,dist8; /**/
  double rcut2, cncut2; /* cutoff values */
  int numprocs,myid,ID; /* MPI */
  int Mc_AN,Gc_AN,Gc_BN,Gc_CN,wanA,wanB,iZ; /* atom counting and species */
  int i,j; /* dummy vars */
  int n1,n2,n3,n1_max,n2_max,n3_max; /* PBC */
  double per_flagA, per_flagB, dblcnt_factor; /* double counting */
  double dEx,dEy,dEz; /* gradients*/
  double xn, *CN, *****dCN; /* Coordination number */
  double exparg,expval, powarg, powval; /**/
  double Z, W, C6_ref, dAi, dBj, Lij, C6, C8, **dC6ij, dZi, dZj, dWi, dWj; /* Gaussian distance C6, C8 parameter */
  double C8C6;

  /* START: for printing gradients ERASE 
     double *xgrad,*ygrad,*zgrad;  
     END: for printing gradients ERASE */

  /* MPI AND INITIALIZATION */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  n1_max = n1_CN_DFT_D;
  n2_max = n2_CN_DFT_D;
  n3_max = n3_CN_DFT_D;
  My_EdftD = 0.0;
  EdftD    = 0.0;
  rcut2 = rcut_dftD*rcut_dftD;
  cncut2 = cncut_dftD*cncut_dftD;

  CN = (double*)malloc(sizeof(double)*(atomnum+1));
  dC6ij = (double**)malloc(sizeof(double*)*(atomnum+1));
  dEC0 = (double**)malloc(sizeof(double*)*(atomnum+1));  
  dCN  = (double*****)malloc(sizeof(double****)*(atomnum+1));
  for(Gc_AN=0; Gc_AN<atomnum+1; Gc_AN++){
    dC6ij[Gc_AN]=(double*)malloc(sizeof(double)*(atomnum+1));            
    dEC0[Gc_AN]=(double*)malloc(sizeof(double*)*(atomnum+1));

    for (i=0; i<(atomnum+1); i++){
      dC6ij[Gc_AN][i] = 0.0;
      dEC0[Gc_AN][i]  = 0.0;
    }
      
    dCN[Gc_AN] =(double****)malloc(sizeof(double***)*(atomnum+1));      
    for(Gc_BN=0; Gc_BN<atomnum+1; Gc_BN++){    
      dCN[Gc_AN][Gc_BN] =(double***)malloc(sizeof(double**)*(2*n1_max+1));      
      for (n1=0; n1<=2*n1_max; n1++){  
	dCN[Gc_AN][Gc_BN][n1] =(double**)malloc(sizeof(double*)*(2*n2_max+1));      
	for (n2=0; n2<=2*n2_max; n2++){
	  dCN[Gc_AN][Gc_BN][n1][n2] =(double*)malloc(sizeof(double)*(2*n3_max+1));
          for (i=0; i<(2*n3_max+1); i++) dCN[Gc_AN][Gc_BN][n1][n2][i] = 0.0;

	} /* n2 */
      } /* n1 */
    } /* Gc_BN */
  } /* Gc_AN */

  /* START: for printing gradients ERASE 
     xgrad = (double*)malloc(sizeof(double)*(atomnum+1));
     ygrad = (double*)malloc(sizeof(double)*(atomnum+1));
     zgrad = (double*)malloc(sizeof(double)*(atomnum+1));
     END: for printing gradients ERASE */
  
  /* Compute coordination numbers CN_A and derivative dCN_AB/dr_AB by adding an inverse damping function */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];
    if ( iZ>0 ) {
      xn=0.0;
      for(Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){
	wanB = WhatSpecies[Gc_BN];
	iZ = Spe_WhatAtom[wanB];
	if ( iZ>0 ) {
	  per_flagB = (int)Gxyz[Gc_BN][60]; 
	  rij0[1] = Gxyz[Gc_AN][1] - Gxyz[Gc_BN][1];
	  rij0[2] = Gxyz[Gc_AN][2] - Gxyz[Gc_BN][2];
	  rij0[3] = Gxyz[Gc_AN][3] - Gxyz[Gc_BN][3];
	  if (per_flagB==0){
	    n1_max = 0;
	    n2_max = 0;
	    n3_max = 0;
	  }
	  else {
	    n1_max = n1_CN_DFT_D;
	    n2_max = n2_CN_DFT_D;
	    n3_max = n3_CN_DFT_D;
	  }
	  for (n1=-n1_max; n1<=n1_max; n1++){
	    for (n2=-n2_max; n2<=n2_max; n2++){
	      for (n3=-n3_max; n3<=n3_max; n3++){
		rij[1] = rij0[1] - ( (double)n1*tv[1][1]
				     + (double)n2*tv[2][1] 
				     + (double)n3*tv[3][1] ); 
		rij[2] = rij0[2] - ( (double)n1*tv[1][2]
				     + (double)n2*tv[2][2] 
				     + (double)n3*tv[3][2] ); 
		rij[3] = rij0[3] - ( (double)n1*tv[1][3]
				     + (double)n2*tv[2][3] 
				     + (double)n3*tv[3][3] ); 
		dist2 = rij[1]*rij[1] + rij[2]*rij[2] + rij[3]*rij[3];
		if (dist2<cncut2 && dist2>0.1){
		  dist  = sqrt(dist2);           
		  exparg = -k1_dftD*((rcovab_dftD[wanA][wanB]/dist)-1.0); /* Rsum is scaled by k2 */
		  expval = exp(exparg);
		  fdamp = 1.0/(1.0+expval);
		  xn+=fdamp;
		  dCN[Gc_AN][Gc_BN][n1+n1_CN_DFT_D][n2+n2_CN_DFT_D][n3+n3_CN_DFT_D]=-fdamp*fdamp*expval*k1_dftD*rcovab_dftD[wanA][wanB]/dist2;
		}
		else{
		  dCN[Gc_AN][Gc_BN][n1+n1_CN_DFT_D][n2+n2_CN_DFT_D][n3+n3_CN_DFT_D]=0.0;
		}
	      } /* n3 */
	    } /* n2 */
	  } /* n1 */
	}
      } /* Gc_BN */
      CN[Gc_AN] = xn;
    }
  } /* Mc_AN */

  /*MPI BROADCAST CN NUMBERS - MPI_Barrier(mpi_comm_level1); */
  MPI_Barrier(mpi_comm_level1);  /* NOT SURE IF NEEDED! */ 
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];
    if ( iZ>0 ) {
      ID = G2ID[Gc_AN];
      MPI_Bcast(&CN[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
      for (Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){
	wanB = WhatSpecies[Gc_BN];
	iZ = Spe_WhatAtom[wanB];
	if ( iZ>0 ) {
	  per_flagB = (int)Gxyz[Gc_BN][60]; 
	  if (per_flagB==0){
	    n1_max = 0;
	    n2_max = 0;
	    n3_max = 0;
	  }
	  else {
	    n1_max = n1_CN_DFT_D;
	    n2_max = n2_CN_DFT_D;
	    n3_max = n3_CN_DFT_D;
	  }
	  for (n1=-n1_max; n1<=n1_max; n1++){
	    for (n2=-n2_max; n2<=n2_max; n2++){
	      for (n3=-n3_max; n3<=n3_max; n3++){
		MPI_Bcast(&dCN[Gc_AN][Gc_BN][n1+n1_CN_DFT_D][n2+n2_CN_DFT_D][n3+n3_CN_DFT_D], 1, MPI_DOUBLE, ID, mpi_comm_level1); 
	      } /* n3 */
	    } /* n2 */
	  } /* n1 */
	}
      } /* Gc_BN */
    }
  } /* Gc_AN */

  /* Calculate energy and collect gradients two body terms C_ij*d(f_ij/r_ij)/dr_ij also dCi_ij dCj_ij dEC0_ij needed in gradients of 3 body terms */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];

    if ( iZ>0 ) {

      dblcnt_factor=0.5;
      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;
      E = 0.0;        

      per_flagA = (int)Gxyz[Gc_AN][60]; 
      for(Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){
	wanB = WhatSpecies[Gc_BN];
	iZ = Spe_WhatAtom[wanB];
	if ( iZ>0 ) {
	  dEC0[Gc_AN][Gc_BN]=0.0;
	  per_flagB = (int)Gxyz[Gc_BN][60]; 
	  rij0[1] = Gxyz[Gc_AN][1] - Gxyz[Gc_BN][1];
	  rij0[2] = Gxyz[Gc_AN][2] - Gxyz[Gc_BN][2];
	  rij0[3] = Gxyz[Gc_AN][3] - Gxyz[Gc_BN][3];
	  /* Calculate C6, C8 coefficient and derivatives with Gaussian-distance (L) */
	  Z = 0.0;
	  W = 0.0;
	  dZi=0.0;
	  dZj=0.0;
	  dWi=0.0;
	  dWj=0.0;
	  for (i=0; i<maxcn_dftD[wanA]; i++){
	    for (j=0; j<maxcn_dftD[wanB]; j++){
	      C6_ref=C6ab_dftD[wanA][wanB][i][j][0];
	      if (C6_ref>1.0e-12){
		dAi = CN[Gc_AN] - C6ab_dftD[wanA][wanB][i][j][1];
		dBj = CN[Gc_BN] - C6ab_dftD[wanA][wanB][i][j][2];
		exparg = -k3_dftD*( dAi*dAi + dBj*dBj );
		Lij=exp(exparg);
		Z += C6_ref*Lij;
		W += Lij;
		dZi+=C6_ref*Lij*2.0*k3_dftD*dAi;
		dZj+=C6_ref*Lij*2.0*k3_dftD*dBj;
		dWi+=Lij*2.0*k3_dftD*dAi;
		dWj+=Lij*2.0*k3_dftD*dBj;                    
	      }
	    } /* CN_j */
	  } /* CN_i */

	  if (W>1.0e-12){

	    C6 = Z/W;
	    C8 = 3.0*C6*r2r4ab_dftD[wanA][wanB];
            C8C6 = 3.0*r2r4ab_dftD[wanA][wanB];
	    dC6ij[Gc_AN][Gc_BN]=((dZi*W)-(dWi*Z))/(W*W);
	  }
	  else{
	    C6 = 0.0;
	    C8 = 0.0;
            C8C6 = 3.0*r2r4ab_dftD[wanA][wanB];
	    dC6ij[Gc_AN][Gc_BN]=0.0;
	  } 
	  /*  CALCULATE ENERGY AND TWO FIRST PART OF GRADIENTS*/
	  if (per_flagB==0){
	    n1_max = 0;
	    n2_max = 0;
	    n3_max = 0;
	  }
	  else {
	    n1_max = n1_DFT_D;
	    n2_max = n2_DFT_D;
	    n3_max = n3_DFT_D;
	  }
	  for (n1=-n1_max; n1<=n1_max; n1++){
	    for (n2=-n2_max; n2<=n2_max; n2++){
	      for (n3=-n3_max; n3<=n3_max; n3++){

		/* for double counting */

		if((!(abs(n1)+abs(n2)+abs(n3))==0) && (per_flagA==0 && per_flagB==1) ){
		  dblcnt_factor = 1.0;
		}
		else{
		  dblcnt_factor = 0.5;
		}

		rij[1] = rij0[1] - ( (double)n1*tv[1][1]
				     + (double)n2*tv[2][1] 
				     + (double)n3*tv[3][1] ); 
		rij[2] = rij0[2] - ( (double)n1*tv[1][2]
				     + (double)n2*tv[2][2] 
				     + (double)n3*tv[3][2] ); 
		rij[3] = rij0[3] - ( (double)n1*tv[1][3]
				     + (double)n2*tv[2][3] 
				     + (double)n3*tv[3][3] ); 

		dist2 = rij[1]*rij[1] + rij[2]*rij[2] + rij[3]*rij[3];

		if (0.1<dist2 && dist2<rcut2){

		  dist  = sqrt(dist2); 
		  dist5 = dist2*dist2*dist;
		  dist6 = dist2*dist2*dist2;
		  dist7 = dist6*dist;
		  dist8 = dist6*dist2;

		  if (DFTD3_damp_dftD == 1){ /*DFTD3 ZERO DAMPING*/

		    /* calculate the vdW energy of E6 and grad of f6/r6 term*/
		    powarg = dist/(sr6_dftD*r0ab_dftD[wanB][wanA]);
		    powval = pow(powarg,-alp6_dftD);
		    fdamp6 = 1.0/(1.0+6.0*powval);
		    E -= dblcnt_factor*s6_dftD*C6*fdamp6/dist6;
		    dE6=(s6_dftD*C6*fdamp6/dist6)*(6.0/dist)*(-1.0+alp6_dftD*powval*fdamp6);

		    /* calculate the vdW energy of E8 and grad of f8/r8 term*/                
		    powarg = dist/(sr8_dftD*r0ab_dftD[wanB][wanA]);
		    powval = pow(powarg,-alp8_dftD);
		    fdamp8 = 1.0/(1.0+6.0*powval);
		    E -= dblcnt_factor*s8_dftD*C8*fdamp8/dist8;
		    dE8=(s8_dftD*C8*fdamp8/dist8)*(2.0/dist)*(-4.0+3.0*alp8_dftD*powval*fdamp8);
		    dEC0[Gc_AN][Gc_BN]+=s6_dftD*fdamp6/dist6+s8_dftD*3.0*r2r4ab_dftD[wanA][wanB]*fdamp8/dist8;

		  } /* END IF ZERO DAMPING */

		  if (DFTD3_damp_dftD == 2){ /*DFTD3 BJ DAMPING*/

		    fdamp = (a1_dftD*sqrt(C8C6)+a2_dftD);
		    fdamp6=fdamp*fdamp*fdamp*fdamp*fdamp*fdamp;
		    fdamp8=fdamp6*fdamp*fdamp;
		    t6=dist6 + fdamp6;
		    t62=t6*t6;
		    t8=dist8 + fdamp8;
		    t82=t8*t8;
		    E -= dblcnt_factor*s6_dftD*C6/t6;                
		    dE6=-s6_dftD*C6*6.0*dist5/t62;
		    E -= dblcnt_factor*s8_dftD*C8/t8;
		    dE8=-s8_dftD*C8*8.0*dist7/t82;
		    dEC0[Gc_AN][Gc_BN]+=s6_dftD/t6+s8_dftD*3.0*r2r4ab_dftD[wanA][wanB]/t8;
		  } /* IF BJ DAMPING */

		  dEx -= (dE6+dE8)*rij[1]/dist;          
		  dEy -= (dE6+dE8)*rij[2]/dist;
		  dEz -= (dE6+dE8)*rij[3]/dist;

		} /* if dist2 < rcut */
	      } /* n3 */
	    } /* n2 */
	  } /* n1 */
	}
      } /* Gc_BN */

      My_EdftD += E; 

      /* energy decomposition */
 
      if (Energy_Decomposition_flag==1){

        DecEvdw[0][Mc_AN][0] = E;
        DecEvdw[1][Mc_AN][0] = E;
      }

      /* gradients from two-body terms */

      Gxyz[Gc_AN][17] += dEx;
      Gxyz[Gc_AN][18] += dEy;
      Gxyz[Gc_AN][19] += dEz;

      /* START: for printing gradients ERASE 
	 xgrad[Gc_AN]=dEx;ygrad[Gc_AN]=dEy;zgrad[Gc_AN]=dEz;
	 END: for printing gradients ERASE */
    }

  } /* Mc_AN */

  /*MPI BROADCAST GRADIENTS AND REDUCE ENERGIES - MPI_Barrier(mpi_comm_level1); */    
  MPI_Allreduce(&My_EdftD, &EdftD, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];
    if ( iZ>0 ) {
      ID = G2ID[Gc_AN];
      MPI_Bcast(&CN[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
      for (Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){
	wanB = WhatSpecies[Gc_BN];
	iZ = Spe_WhatAtom[wanB];
	if ( iZ>0 ) {
	  MPI_Bcast(&dC6ij[Gc_AN][Gc_BN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
	  MPI_Bcast(&dEC0[Gc_AN][Gc_BN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
	}
      } /* Gc_BN */
    }
  } /* Gc_AN */
  MPI_Barrier(mpi_comm_level1); /* NOT SURE IF ITS NEEDED! */

  /* Calculate three body terms of gradients */
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wanA = WhatSpecies[Gc_AN];
    iZ = Spe_WhatAtom[wanA];
    if ( iZ>0 ) {
      dEx = 0.0;
      dEy = 0.0;
      dEz = 0.0;

      for(Gc_BN=1; Gc_BN<=atomnum; Gc_BN++){

	wanB = WhatSpecies[Gc_BN];
	iZ = Spe_WhatAtom[wanB];

	if ( iZ>0 ) {
	  per_flagB = (int)Gxyz[Gc_BN][60];
	  rij0[1] = Gxyz[Gc_AN][1] - Gxyz[Gc_BN][1];
	  rij0[2] = Gxyz[Gc_AN][2] - Gxyz[Gc_BN][2];
	  rij0[3] = Gxyz[Gc_AN][3] - Gxyz[Gc_BN][3];
	  if (per_flagB==0){
	    n1_max = 0;
	    n2_max = 0;
	    n3_max = 0;
	  }
	  else {
	    n1_max = n1_CN_DFT_D;
	    n2_max = n2_CN_DFT_D;
	    n3_max = n3_CN_DFT_D;
	  }

	  for (n1=-n1_max; n1<=n1_max; n1++){
	    for (n2=-n2_max; n2<=n2_max; n2++){
	      for (n3=-n3_max; n3<=n3_max; n3++){

		dEC=0.0;
		rij[1] = rij0[1] - ( (double)n1*tv[1][1]
				     + (double)n2*tv[2][1] 
				     + (double)n3*tv[3][1] ); 
		rij[2] = rij0[2] - ( (double)n1*tv[1][2]
				     + (double)n2*tv[2][2] 
				     + (double)n3*tv[3][2] ); 
		rij[3] = rij0[3] - ( (double)n1*tv[1][3]
				     + (double)n2*tv[2][3] 
				     + (double)n3*tv[3][3] ); 

		dist2 = rij[1]*rij[1] + rij[2]*rij[2] + rij[3]*rij[3];

		if (0.1<dist2 && dist2<cncut2){

		  /* calculate grad of C6 term: dEC=dEC0_ik*dC6_ik*dCN_ij+dEC0_jk*dC6_jk*dCN_ij */
		  dist=sqrt(dist2);
		  for(Gc_CN=1; Gc_CN<=atomnum; Gc_CN++){

		    dEC += dEC0[Gc_AN][Gc_CN]*dC6ij[Gc_AN][Gc_CN]*dCN[Gc_AN][Gc_BN][n1+n1_CN_DFT_D][n2+n2_CN_DFT_D][n3+n3_CN_DFT_D];
		    dEC += dEC0[Gc_BN][Gc_CN]*dC6ij[Gc_BN][Gc_CN]*dCN[Gc_AN][Gc_BN][n1+n1_CN_DFT_D][n2+n2_CN_DFT_D][n3+n3_CN_DFT_D];

		  } /* Gc_CN */

		  dEx += dEC*rij[1]/dist;
		  dEy += dEC*rij[2]/dist;
		  dEz += dEC*rij[3]/dist;

		} /* if dist2 < cn_thr */

	      } /* n3 */
	    } /* n2 */
	  } /* n1 */
	}
      } /* Gc_BN */
 
      Gxyz[Gc_AN][17] += dEx;
      Gxyz[Gc_AN][18] += dEy;
      Gxyz[Gc_AN][19] += dEz;

      /* START: for printing gradients ERASE
	 xgrad[Gc_AN]+=dEx;ygrad[Gc_AN]+=dEy;zgrad[Gc_AN]+=dEz;
	 END: for printing gradients ERASE */
    }
  } /* Mc_AN */

  /* START: for printing gradients ERASE 
     MPI_Barrier(mpi_comm_level1);
     for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
     ID = G2ID[Gc_AN];
     MPI_Bcast(&xgrad[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
     MPI_Bcast(&ygrad[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
     MPI_Bcast(&zgrad[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
     }
     if(myid==0){
     printf("DFTD3: ATOM NUMBER, COORDINATION NUMBER, GRADIENTS (X, Y, Z)\n");
     for(Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
     printf("%4d %10.6e %+12.8e %+12.8e %+12.8e \n",Gc_AN, CN[Gc_AN],xgrad[Gc_AN],ygrad[Gc_AN],zgrad[Gc_AN]);fflush(stdout);
     }
     }
     free(xgrad);free(ygrad);free(zgrad);
     END: for printing gradients ERASE */
  
  /* free arrays */  
  for(Gc_AN=0; Gc_AN<atomnum+1; Gc_AN++){
    for(Gc_BN=0; Gc_BN<atomnum+1; Gc_BN++){
      for (n1=0; n1<=2*n1_CN_DFT_D; n1++){
        for (n2=0; n2<=2*n2_CN_DFT_D; n2++){
          free(dCN[Gc_AN][Gc_BN][n1][n2]);
        } /* n2 */
        free(dCN[Gc_AN][Gc_BN][n1]);
      } /* n1 */
      free(dCN[Gc_AN][Gc_BN]);
    } /* Gc_BN */
    free(dC6ij[Gc_AN]);
    free(dEC0[Gc_AN]);
    free(dCN[Gc_AN]);
  } /* Gc_AN */
  free(dC6ij);
  free(dEC0);
  free(dCN);
  free(CN);
  return EdftD;
}
/* Ellner */


void Energy_Decomposition(double ECE[])
{
  static int firsttime=1;
  int i,spin,spinmax,XC_P_switch;
  int numS,numR,My_GNum,BN_AB,max_ene;
  int n,n1,n2,n3,Ng1,Ng2,Ng3,j,k;
  int GN,GNs,BN,DN,LN,N2D,n2D,N3[4];
  double intVH1[2],intVxc[2];
  double My_intVH1[2],My_intVxc[2];
  double c[2],Etot;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Cxyz[4],gradxyz[4];
  double Stime_atom,Etime_atom;
  double time0,time1;
  double sden[2],tden,aden,pden[2];
  int tnoA,tnoB,wanA,wanB,Mc_AN,Gc_AN,h_AN;
  double sum,tsum,Total_Mul_up,Total_Mul_dn;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* calculation of total energy  */

  max_ene = 14;

  Etot = 0.0; 
  for (i=0; i<=max_ene; i++){
    Etot += ECE[i];
  }
  Etot = Etot - ECE[0] - ECE[1] - ECE[13]; 

  Total_Mul_up = 0.0;
  Total_Mul_dn = 0.0;

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    Total_Mul_up += InitN_USpin[Gc_AN];
    Total_Mul_dn += InitN_DSpin[Gc_AN];
  }
  
  if      (SpinP_switch==0){
    c[0] = 0.5*(Etot - Uele)/(Total_Mul_up+1.0e-15);
  }
  else if (SpinP_switch==1) {
    c[0] = (Etot - Uele)/(Total_Mul_up+Total_Mul_dn+1.0e-15);        
    c[1] = c[0];

  }
  else if (SpinP_switch==3) {
    c[0] = (Etot - Uele)/(Total_Mul_up+Total_Mul_dn+1.0e-15);        
    c[1] = c[0];
  }

  /****************************************************
       calculations of DecEkin, DecEv, and DecEcon
  ****************************************************/

  if (SpinP_switch==0 || SpinP_switch==1){

    for (spin=0; spin<=SpinP_switch; spin++){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];
	wanA = WhatSpecies[Gc_AN];
	tnoA = Spe_Total_CNO[wanA];

        for (i=0; i<tnoA; i++){

          DecEkin[spin][Mc_AN][i] = 0.0;
          DecEv[spin][Mc_AN][i]   = 0.0;
          DecEcon[spin][Mc_AN][i] = 0.0;

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    wanB = WhatSpecies[natn[Gc_AN][h_AN]];
	    tnoB = Spe_Total_CNO[wanB];
	    for (j=0; j<tnoB; j++){

	      DecEkin[spin][Mc_AN][i] += DM[0][spin][Mc_AN][h_AN][i][j]*H0[0][Mc_AN][h_AN][i][j];


	      DecEv[spin][Mc_AN][i]   += DM[0][spin][Mc_AN][h_AN][i][j]*(H[spin][Mc_AN][h_AN][i][j]-H0[0][Mc_AN][h_AN][i][j]);
	      DecEcon[spin][Mc_AN][i] += DM[0][spin][Mc_AN][h_AN][i][j]*OLP[0][Mc_AN][h_AN][i][j]*c[spin];
	    }
	  }
	}
      }
    }
  }

  else if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wanA = WhatSpecies[Gc_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (i=0; i<tnoA; i++){

	DecEkin[0][Mc_AN][i] = 0.0;
	DecEkin[1][Mc_AN][i] = 0.0;

	DecEv[0][Mc_AN][i]   = 0.0;
	DecEv[1][Mc_AN][i]   = 0.0;

	DecEcon[0][Mc_AN][i] = 0.0;
	DecEcon[1][Mc_AN][i] = 0.0;

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  wanB = WhatSpecies[natn[Gc_AN][h_AN]];
	  tnoB = Spe_Total_CNO[wanB];
	  for (j=0; j<tnoB; j++){

	    DecEkin[0][Mc_AN][i] += DM[0][0][Mc_AN][h_AN][i][j]*H0[0][Mc_AN][h_AN][i][j];
	    DecEkin[1][Mc_AN][i] += DM[0][1][Mc_AN][h_AN][i][j]*H0[0][Mc_AN][h_AN][i][j];

	    DecEcon[0][Mc_AN][i] += DM[0][0][Mc_AN][h_AN][i][j]*OLP[0][Mc_AN][h_AN][i][j]*c[0];
	    DecEcon[1][Mc_AN][i] += DM[0][1][Mc_AN][h_AN][i][j]*OLP[0][Mc_AN][h_AN][i][j]*c[1];

	    DecEv[0][Mc_AN][i] += 
                    DM[0][0][Mc_AN][h_AN][i][j]*(H[0][Mc_AN][h_AN][i][j]-H0[0][Mc_AN][h_AN][i][j])
	         - iDM[0][0][Mc_AN][h_AN][i][j]*iHNL[0][Mc_AN][h_AN][i][j]
                 +  DM[0][1][Mc_AN][h_AN][i][j]*(H[1][Mc_AN][h_AN][i][j]-H0[0][Mc_AN][h_AN][i][j])
	         - iDM[0][1][Mc_AN][h_AN][i][j]*iHNL[1][Mc_AN][h_AN][i][j]
	      + 2.0*DM[0][2][Mc_AN][h_AN][i][j]*H[2][Mc_AN][h_AN][i][j]
              - 2.0*DM[0][3][Mc_AN][h_AN][i][j]*(H[3][Mc_AN][h_AN][i][j]+iHNL[2][Mc_AN][h_AN][i][j]);

	    DecEv[1][Mc_AN][i] = 0.0; 

	  }
	}
      }
    }

  }

}





void Set_Lebedev_Grid()
{
  /* 590 */

  if (Num_Leb_Grid==590){

    Leb_Grid_XYZW[   0][0]= 1.000000000000000;
    Leb_Grid_XYZW[   0][1]= 0.000000000000000;
    Leb_Grid_XYZW[   0][2]= 0.000000000000000;
    Leb_Grid_XYZW[   0][3]= 0.000309512129531;

    Leb_Grid_XYZW[   1][0]=-1.000000000000000;
    Leb_Grid_XYZW[   1][1]= 0.000000000000000;
    Leb_Grid_XYZW[   1][2]= 0.000000000000000;
    Leb_Grid_XYZW[   1][3]= 0.000309512129531;

    Leb_Grid_XYZW[   2][0]= 0.000000000000000;
    Leb_Grid_XYZW[   2][1]= 1.000000000000000;
    Leb_Grid_XYZW[   2][2]= 0.000000000000000;
    Leb_Grid_XYZW[   2][3]= 0.000309512129531;

    Leb_Grid_XYZW[   3][0]= 0.000000000000000;
    Leb_Grid_XYZW[   3][1]=-1.000000000000000;
    Leb_Grid_XYZW[   3][2]= 0.000000000000000;
    Leb_Grid_XYZW[   3][3]= 0.000309512129531;

    Leb_Grid_XYZW[   4][0]= 0.000000000000000;
    Leb_Grid_XYZW[   4][1]= 0.000000000000000;
    Leb_Grid_XYZW[   4][2]= 1.000000000000000;
    Leb_Grid_XYZW[   4][3]= 0.000309512129531;

    Leb_Grid_XYZW[   5][0]= 0.000000000000000;
    Leb_Grid_XYZW[   5][1]= 0.000000000000000;
    Leb_Grid_XYZW[   5][2]=-1.000000000000000;
    Leb_Grid_XYZW[   5][3]= 0.000309512129531;

    Leb_Grid_XYZW[   6][0]= 0.577350269189626;
    Leb_Grid_XYZW[   6][1]= 0.577350269189626;
    Leb_Grid_XYZW[   6][2]= 0.577350269189626;
    Leb_Grid_XYZW[   6][3]= 0.001852379698597;

    Leb_Grid_XYZW[   7][0]=-0.577350269189626;
    Leb_Grid_XYZW[   7][1]= 0.577350269189626;
    Leb_Grid_XYZW[   7][2]= 0.577350269189626;
    Leb_Grid_XYZW[   7][3]= 0.001852379698597;

    Leb_Grid_XYZW[   8][0]= 0.577350269189626;
    Leb_Grid_XYZW[   8][1]=-0.577350269189626;
    Leb_Grid_XYZW[   8][2]= 0.577350269189626;
    Leb_Grid_XYZW[   8][3]= 0.001852379698597;

    Leb_Grid_XYZW[   9][0]=-0.577350269189626;
    Leb_Grid_XYZW[   9][1]=-0.577350269189626;
    Leb_Grid_XYZW[   9][2]= 0.577350269189626;
    Leb_Grid_XYZW[   9][3]= 0.001852379698597;

    Leb_Grid_XYZW[  10][0]= 0.577350269189626;
    Leb_Grid_XYZW[  10][1]= 0.577350269189626;
    Leb_Grid_XYZW[  10][2]=-0.577350269189626;
    Leb_Grid_XYZW[  10][3]= 0.001852379698597;

    Leb_Grid_XYZW[  11][0]=-0.577350269189626;
    Leb_Grid_XYZW[  11][1]= 0.577350269189626;
    Leb_Grid_XYZW[  11][2]=-0.577350269189626;
    Leb_Grid_XYZW[  11][3]= 0.001852379698597;

    Leb_Grid_XYZW[  12][0]= 0.577350269189626;
    Leb_Grid_XYZW[  12][1]=-0.577350269189626;
    Leb_Grid_XYZW[  12][2]=-0.577350269189626;
    Leb_Grid_XYZW[  12][3]= 0.001852379698597;

    Leb_Grid_XYZW[  13][0]=-0.577350269189626;
    Leb_Grid_XYZW[  13][1]=-0.577350269189626;
    Leb_Grid_XYZW[  13][2]=-0.577350269189626;
    Leb_Grid_XYZW[  13][3]= 0.001852379698597;

    Leb_Grid_XYZW[  14][0]= 0.704095493822747;
    Leb_Grid_XYZW[  14][1]= 0.704095493822747;
    Leb_Grid_XYZW[  14][2]= 0.092190407076898;
    Leb_Grid_XYZW[  14][3]= 0.001871790639278;

    Leb_Grid_XYZW[  15][0]=-0.704095493822747;
    Leb_Grid_XYZW[  15][1]= 0.704095493822747;
    Leb_Grid_XYZW[  15][2]= 0.092190407076898;
    Leb_Grid_XYZW[  15][3]= 0.001871790639278;

    Leb_Grid_XYZW[  16][0]= 0.704095493822747;
    Leb_Grid_XYZW[  16][1]=-0.704095493822747;
    Leb_Grid_XYZW[  16][2]= 0.092190407076898;
    Leb_Grid_XYZW[  16][3]= 0.001871790639278;

    Leb_Grid_XYZW[  17][0]=-0.704095493822747;
    Leb_Grid_XYZW[  17][1]=-0.704095493822747;
    Leb_Grid_XYZW[  17][2]= 0.092190407076898;
    Leb_Grid_XYZW[  17][3]= 0.001871790639278;

    Leb_Grid_XYZW[  18][0]= 0.704095493822747;
    Leb_Grid_XYZW[  18][1]= 0.704095493822747;
    Leb_Grid_XYZW[  18][2]=-0.092190407076898;
    Leb_Grid_XYZW[  18][3]= 0.001871790639278;

    Leb_Grid_XYZW[  19][0]=-0.704095493822747;
    Leb_Grid_XYZW[  19][1]= 0.704095493822747;
    Leb_Grid_XYZW[  19][2]=-0.092190407076898;
    Leb_Grid_XYZW[  19][3]= 0.001871790639278;

    Leb_Grid_XYZW[  20][0]= 0.704095493822747;
    Leb_Grid_XYZW[  20][1]=-0.704095493822747;
    Leb_Grid_XYZW[  20][2]=-0.092190407076898;
    Leb_Grid_XYZW[  20][3]= 0.001871790639278;

    Leb_Grid_XYZW[  21][0]=-0.704095493822747;
    Leb_Grid_XYZW[  21][1]=-0.704095493822747;
    Leb_Grid_XYZW[  21][2]=-0.092190407076898;
    Leb_Grid_XYZW[  21][3]= 0.001871790639278;

    Leb_Grid_XYZW[  22][0]= 0.704095493822747;
    Leb_Grid_XYZW[  22][1]= 0.092190407076898;
    Leb_Grid_XYZW[  22][2]= 0.704095493822747;
    Leb_Grid_XYZW[  22][3]= 0.001871790639278;

    Leb_Grid_XYZW[  23][0]=-0.704095493822747;
    Leb_Grid_XYZW[  23][1]= 0.092190407076898;
    Leb_Grid_XYZW[  23][2]= 0.704095493822747;
    Leb_Grid_XYZW[  23][3]= 0.001871790639278;

    Leb_Grid_XYZW[  24][0]= 0.704095493822747;
    Leb_Grid_XYZW[  24][1]=-0.092190407076898;
    Leb_Grid_XYZW[  24][2]= 0.704095493822747;
    Leb_Grid_XYZW[  24][3]= 0.001871790639278;

    Leb_Grid_XYZW[  25][0]=-0.704095493822747;
    Leb_Grid_XYZW[  25][1]=-0.092190407076898;
    Leb_Grid_XYZW[  25][2]= 0.704095493822747;
    Leb_Grid_XYZW[  25][3]= 0.001871790639278;

    Leb_Grid_XYZW[  26][0]= 0.704095493822747;
    Leb_Grid_XYZW[  26][1]= 0.092190407076898;
    Leb_Grid_XYZW[  26][2]=-0.704095493822747;
    Leb_Grid_XYZW[  26][3]= 0.001871790639278;

    Leb_Grid_XYZW[  27][0]=-0.704095493822747;
    Leb_Grid_XYZW[  27][1]= 0.092190407076898;
    Leb_Grid_XYZW[  27][2]=-0.704095493822747;
    Leb_Grid_XYZW[  27][3]= 0.001871790639278;

    Leb_Grid_XYZW[  28][0]= 0.704095493822747;
    Leb_Grid_XYZW[  28][1]=-0.092190407076898;
    Leb_Grid_XYZW[  28][2]=-0.704095493822747;
    Leb_Grid_XYZW[  28][3]= 0.001871790639278;

    Leb_Grid_XYZW[  29][0]=-0.704095493822747;
    Leb_Grid_XYZW[  29][1]=-0.092190407076898;
    Leb_Grid_XYZW[  29][2]=-0.704095493822747;
    Leb_Grid_XYZW[  29][3]= 0.001871790639278;

    Leb_Grid_XYZW[  30][0]= 0.092190407076898;
    Leb_Grid_XYZW[  30][1]= 0.704095493822747;
    Leb_Grid_XYZW[  30][2]= 0.704095493822747;
    Leb_Grid_XYZW[  30][3]= 0.001871790639278;

    Leb_Grid_XYZW[  31][0]=-0.092190407076898;
    Leb_Grid_XYZW[  31][1]= 0.704095493822747;
    Leb_Grid_XYZW[  31][2]= 0.704095493822747;
    Leb_Grid_XYZW[  31][3]= 0.001871790639278;

    Leb_Grid_XYZW[  32][0]= 0.092190407076898;
    Leb_Grid_XYZW[  32][1]=-0.704095493822747;
    Leb_Grid_XYZW[  32][2]= 0.704095493822747;
    Leb_Grid_XYZW[  32][3]= 0.001871790639278;

    Leb_Grid_XYZW[  33][0]=-0.092190407076898;
    Leb_Grid_XYZW[  33][1]=-0.704095493822747;
    Leb_Grid_XYZW[  33][2]= 0.704095493822747;
    Leb_Grid_XYZW[  33][3]= 0.001871790639278;

    Leb_Grid_XYZW[  34][0]= 0.092190407076898;
    Leb_Grid_XYZW[  34][1]= 0.704095493822747;
    Leb_Grid_XYZW[  34][2]=-0.704095493822747;
    Leb_Grid_XYZW[  34][3]= 0.001871790639278;

    Leb_Grid_XYZW[  35][0]=-0.092190407076898;
    Leb_Grid_XYZW[  35][1]= 0.704095493822747;
    Leb_Grid_XYZW[  35][2]=-0.704095493822747;
    Leb_Grid_XYZW[  35][3]= 0.001871790639278;

    Leb_Grid_XYZW[  36][0]= 0.092190407076898;
    Leb_Grid_XYZW[  36][1]=-0.704095493822747;
    Leb_Grid_XYZW[  36][2]=-0.704095493822747;
    Leb_Grid_XYZW[  36][3]= 0.001871790639278;

    Leb_Grid_XYZW[  37][0]=-0.092190407076898;
    Leb_Grid_XYZW[  37][1]=-0.704095493822747;
    Leb_Grid_XYZW[  37][2]=-0.704095493822747;
    Leb_Grid_XYZW[  37][3]= 0.001871790639278;

    Leb_Grid_XYZW[  38][0]= 0.680774406645524;
    Leb_Grid_XYZW[  38][1]= 0.680774406645524;
    Leb_Grid_XYZW[  38][2]= 0.270356088359165;
    Leb_Grid_XYZW[  38][3]= 0.001858812585438;

    Leb_Grid_XYZW[  39][0]=-0.680774406645524;
    Leb_Grid_XYZW[  39][1]= 0.680774406645524;
    Leb_Grid_XYZW[  39][2]= 0.270356088359165;
    Leb_Grid_XYZW[  39][3]= 0.001858812585438;

    Leb_Grid_XYZW[  40][0]= 0.680774406645524;
    Leb_Grid_XYZW[  40][1]=-0.680774406645524;
    Leb_Grid_XYZW[  40][2]= 0.270356088359165;
    Leb_Grid_XYZW[  40][3]= 0.001858812585438;

    Leb_Grid_XYZW[  41][0]=-0.680774406645524;
    Leb_Grid_XYZW[  41][1]=-0.680774406645524;
    Leb_Grid_XYZW[  41][2]= 0.270356088359165;
    Leb_Grid_XYZW[  41][3]= 0.001858812585438;

    Leb_Grid_XYZW[  42][0]= 0.680774406645524;
    Leb_Grid_XYZW[  42][1]= 0.680774406645524;
    Leb_Grid_XYZW[  42][2]=-0.270356088359165;
    Leb_Grid_XYZW[  42][3]= 0.001858812585438;

    Leb_Grid_XYZW[  43][0]=-0.680774406645524;
    Leb_Grid_XYZW[  43][1]= 0.680774406645524;
    Leb_Grid_XYZW[  43][2]=-0.270356088359165;
    Leb_Grid_XYZW[  43][3]= 0.001858812585438;

    Leb_Grid_XYZW[  44][0]= 0.680774406645524;
    Leb_Grid_XYZW[  44][1]=-0.680774406645524;
    Leb_Grid_XYZW[  44][2]=-0.270356088359165;
    Leb_Grid_XYZW[  44][3]= 0.001858812585438;

    Leb_Grid_XYZW[  45][0]=-0.680774406645524;
    Leb_Grid_XYZW[  45][1]=-0.680774406645524;
    Leb_Grid_XYZW[  45][2]=-0.270356088359165;
    Leb_Grid_XYZW[  45][3]= 0.001858812585438;

    Leb_Grid_XYZW[  46][0]= 0.680774406645524;
    Leb_Grid_XYZW[  46][1]= 0.270356088359165;
    Leb_Grid_XYZW[  46][2]= 0.680774406645524;
    Leb_Grid_XYZW[  46][3]= 0.001858812585438;

    Leb_Grid_XYZW[  47][0]=-0.680774406645524;
    Leb_Grid_XYZW[  47][1]= 0.270356088359165;
    Leb_Grid_XYZW[  47][2]= 0.680774406645524;
    Leb_Grid_XYZW[  47][3]= 0.001858812585438;

    Leb_Grid_XYZW[  48][0]= 0.680774406645524;
    Leb_Grid_XYZW[  48][1]=-0.270356088359165;
    Leb_Grid_XYZW[  48][2]= 0.680774406645524;
    Leb_Grid_XYZW[  48][3]= 0.001858812585438;

    Leb_Grid_XYZW[  49][0]=-0.680774406645524;
    Leb_Grid_XYZW[  49][1]=-0.270356088359165;
    Leb_Grid_XYZW[  49][2]= 0.680774406645524;
    Leb_Grid_XYZW[  49][3]= 0.001858812585438;

    Leb_Grid_XYZW[  50][0]= 0.680774406645524;
    Leb_Grid_XYZW[  50][1]= 0.270356088359165;
    Leb_Grid_XYZW[  50][2]=-0.680774406645524;
    Leb_Grid_XYZW[  50][3]= 0.001858812585438;

    Leb_Grid_XYZW[  51][0]=-0.680774406645524;
    Leb_Grid_XYZW[  51][1]= 0.270356088359165;
    Leb_Grid_XYZW[  51][2]=-0.680774406645524;
    Leb_Grid_XYZW[  51][3]= 0.001858812585438;

    Leb_Grid_XYZW[  52][0]= 0.680774406645524;
    Leb_Grid_XYZW[  52][1]=-0.270356088359165;
    Leb_Grid_XYZW[  52][2]=-0.680774406645524;
    Leb_Grid_XYZW[  52][3]= 0.001858812585438;

    Leb_Grid_XYZW[  53][0]=-0.680774406645524;
    Leb_Grid_XYZW[  53][1]=-0.270356088359165;
    Leb_Grid_XYZW[  53][2]=-0.680774406645524;
    Leb_Grid_XYZW[  53][3]= 0.001858812585438;

    Leb_Grid_XYZW[  54][0]= 0.270356088359165;
    Leb_Grid_XYZW[  54][1]= 0.680774406645524;
    Leb_Grid_XYZW[  54][2]= 0.680774406645524;
    Leb_Grid_XYZW[  54][3]= 0.001858812585438;

    Leb_Grid_XYZW[  55][0]=-0.270356088359165;
    Leb_Grid_XYZW[  55][1]= 0.680774406645524;
    Leb_Grid_XYZW[  55][2]= 0.680774406645524;
    Leb_Grid_XYZW[  55][3]= 0.001858812585438;

    Leb_Grid_XYZW[  56][0]= 0.270356088359165;
    Leb_Grid_XYZW[  56][1]=-0.680774406645524;
    Leb_Grid_XYZW[  56][2]= 0.680774406645524;
    Leb_Grid_XYZW[  56][3]= 0.001858812585438;

    Leb_Grid_XYZW[  57][0]=-0.270356088359165;
    Leb_Grid_XYZW[  57][1]=-0.680774406645524;
    Leb_Grid_XYZW[  57][2]= 0.680774406645524;
    Leb_Grid_XYZW[  57][3]= 0.001858812585438;

    Leb_Grid_XYZW[  58][0]= 0.270356088359165;
    Leb_Grid_XYZW[  58][1]= 0.680774406645524;
    Leb_Grid_XYZW[  58][2]=-0.680774406645524;
    Leb_Grid_XYZW[  58][3]= 0.001858812585438;

    Leb_Grid_XYZW[  59][0]=-0.270356088359165;
    Leb_Grid_XYZW[  59][1]= 0.680774406645524;
    Leb_Grid_XYZW[  59][2]=-0.680774406645524;
    Leb_Grid_XYZW[  59][3]= 0.001858812585438;

    Leb_Grid_XYZW[  60][0]= 0.270356088359165;
    Leb_Grid_XYZW[  60][1]=-0.680774406645524;
    Leb_Grid_XYZW[  60][2]=-0.680774406645524;
    Leb_Grid_XYZW[  60][3]= 0.001858812585438;

    Leb_Grid_XYZW[  61][0]=-0.270356088359165;
    Leb_Grid_XYZW[  61][1]=-0.680774406645524;
    Leb_Grid_XYZW[  61][2]=-0.680774406645524;
    Leb_Grid_XYZW[  61][3]= 0.001858812585438;

    Leb_Grid_XYZW[  62][0]= 0.637254693925875;
    Leb_Grid_XYZW[  62][1]= 0.637254693925875;
    Leb_Grid_XYZW[  62][2]= 0.433373868777154;
    Leb_Grid_XYZW[  62][3]= 0.001852028828296;

    Leb_Grid_XYZW[  63][0]=-0.637254693925875;
    Leb_Grid_XYZW[  63][1]= 0.637254693925875;
    Leb_Grid_XYZW[  63][2]= 0.433373868777154;
    Leb_Grid_XYZW[  63][3]= 0.001852028828296;

    Leb_Grid_XYZW[  64][0]= 0.637254693925875;
    Leb_Grid_XYZW[  64][1]=-0.637254693925875;
    Leb_Grid_XYZW[  64][2]= 0.433373868777154;
    Leb_Grid_XYZW[  64][3]= 0.001852028828296;

    Leb_Grid_XYZW[  65][0]=-0.637254693925875;
    Leb_Grid_XYZW[  65][1]=-0.637254693925875;
    Leb_Grid_XYZW[  65][2]= 0.433373868777154;
    Leb_Grid_XYZW[  65][3]= 0.001852028828296;

    Leb_Grid_XYZW[  66][0]= 0.637254693925875;
    Leb_Grid_XYZW[  66][1]= 0.637254693925875;
    Leb_Grid_XYZW[  66][2]=-0.433373868777154;
    Leb_Grid_XYZW[  66][3]= 0.001852028828296;

    Leb_Grid_XYZW[  67][0]=-0.637254693925875;
    Leb_Grid_XYZW[  67][1]= 0.637254693925875;
    Leb_Grid_XYZW[  67][2]=-0.433373868777154;
    Leb_Grid_XYZW[  67][3]= 0.001852028828296;

    Leb_Grid_XYZW[  68][0]= 0.637254693925875;
    Leb_Grid_XYZW[  68][1]=-0.637254693925875;
    Leb_Grid_XYZW[  68][2]=-0.433373868777154;
    Leb_Grid_XYZW[  68][3]= 0.001852028828296;

    Leb_Grid_XYZW[  69][0]=-0.637254693925875;
    Leb_Grid_XYZW[  69][1]=-0.637254693925875;
    Leb_Grid_XYZW[  69][2]=-0.433373868777154;
    Leb_Grid_XYZW[  69][3]= 0.001852028828296;

    Leb_Grid_XYZW[  70][0]= 0.637254693925875;
    Leb_Grid_XYZW[  70][1]= 0.433373868777154;
    Leb_Grid_XYZW[  70][2]= 0.637254693925875;
    Leb_Grid_XYZW[  70][3]= 0.001852028828296;

    Leb_Grid_XYZW[  71][0]=-0.637254693925875;
    Leb_Grid_XYZW[  71][1]= 0.433373868777154;
    Leb_Grid_XYZW[  71][2]= 0.637254693925875;
    Leb_Grid_XYZW[  71][3]= 0.001852028828296;

    Leb_Grid_XYZW[  72][0]= 0.637254693925875;
    Leb_Grid_XYZW[  72][1]=-0.433373868777154;
    Leb_Grid_XYZW[  72][2]= 0.637254693925875;
    Leb_Grid_XYZW[  72][3]= 0.001852028828296;

    Leb_Grid_XYZW[  73][0]=-0.637254693925875;
    Leb_Grid_XYZW[  73][1]=-0.433373868777154;
    Leb_Grid_XYZW[  73][2]= 0.637254693925875;
    Leb_Grid_XYZW[  73][3]= 0.001852028828296;

    Leb_Grid_XYZW[  74][0]= 0.637254693925875;
    Leb_Grid_XYZW[  74][1]= 0.433373868777154;
    Leb_Grid_XYZW[  74][2]=-0.637254693925875;
    Leb_Grid_XYZW[  74][3]= 0.001852028828296;

    Leb_Grid_XYZW[  75][0]=-0.637254693925875;
    Leb_Grid_XYZW[  75][1]= 0.433373868777154;
    Leb_Grid_XYZW[  75][2]=-0.637254693925875;
    Leb_Grid_XYZW[  75][3]= 0.001852028828296;

    Leb_Grid_XYZW[  76][0]= 0.637254693925875;
    Leb_Grid_XYZW[  76][1]=-0.433373868777154;
    Leb_Grid_XYZW[  76][2]=-0.637254693925875;
    Leb_Grid_XYZW[  76][3]= 0.001852028828296;

    Leb_Grid_XYZW[  77][0]=-0.637254693925875;
    Leb_Grid_XYZW[  77][1]=-0.433373868777154;
    Leb_Grid_XYZW[  77][2]=-0.637254693925875;
    Leb_Grid_XYZW[  77][3]= 0.001852028828296;

    Leb_Grid_XYZW[  78][0]= 0.433373868777154;
    Leb_Grid_XYZW[  78][1]= 0.637254693925875;
    Leb_Grid_XYZW[  78][2]= 0.637254693925875;
    Leb_Grid_XYZW[  78][3]= 0.001852028828296;

    Leb_Grid_XYZW[  79][0]=-0.433373868777154;
    Leb_Grid_XYZW[  79][1]= 0.637254693925875;
    Leb_Grid_XYZW[  79][2]= 0.637254693925875;
    Leb_Grid_XYZW[  79][3]= 0.001852028828296;

    Leb_Grid_XYZW[  80][0]= 0.433373868777154;
    Leb_Grid_XYZW[  80][1]=-0.637254693925875;
    Leb_Grid_XYZW[  80][2]= 0.637254693925875;
    Leb_Grid_XYZW[  80][3]= 0.001852028828296;

    Leb_Grid_XYZW[  81][0]=-0.433373868777154;
    Leb_Grid_XYZW[  81][1]=-0.637254693925875;
    Leb_Grid_XYZW[  81][2]= 0.637254693925875;
    Leb_Grid_XYZW[  81][3]= 0.001852028828296;

    Leb_Grid_XYZW[  82][0]= 0.433373868777154;
    Leb_Grid_XYZW[  82][1]= 0.637254693925875;
    Leb_Grid_XYZW[  82][2]=-0.637254693925875;
    Leb_Grid_XYZW[  82][3]= 0.001852028828296;

    Leb_Grid_XYZW[  83][0]=-0.433373868777154;
    Leb_Grid_XYZW[  83][1]= 0.637254693925875;
    Leb_Grid_XYZW[  83][2]=-0.637254693925875;
    Leb_Grid_XYZW[  83][3]= 0.001852028828296;

    Leb_Grid_XYZW[  84][0]= 0.433373868777154;
    Leb_Grid_XYZW[  84][1]=-0.637254693925875;
    Leb_Grid_XYZW[  84][2]=-0.637254693925875;
    Leb_Grid_XYZW[  84][3]= 0.001852028828296;

    Leb_Grid_XYZW[  85][0]=-0.433373868777154;
    Leb_Grid_XYZW[  85][1]=-0.637254693925875;
    Leb_Grid_XYZW[  85][2]=-0.637254693925875;
    Leb_Grid_XYZW[  85][3]= 0.001852028828296;

    Leb_Grid_XYZW[  86][0]= 0.504441970780036;
    Leb_Grid_XYZW[  86][1]= 0.504441970780036;
    Leb_Grid_XYZW[  86][2]= 0.700768575373573;
    Leb_Grid_XYZW[  86][3]= 0.001846715956151;

    Leb_Grid_XYZW[  87][0]=-0.504441970780036;
    Leb_Grid_XYZW[  87][1]= 0.504441970780036;
    Leb_Grid_XYZW[  87][2]= 0.700768575373573;
    Leb_Grid_XYZW[  87][3]= 0.001846715956151;

    Leb_Grid_XYZW[  88][0]= 0.504441970780036;
    Leb_Grid_XYZW[  88][1]=-0.504441970780036;
    Leb_Grid_XYZW[  88][2]= 0.700768575373573;
    Leb_Grid_XYZW[  88][3]= 0.001846715956151;

    Leb_Grid_XYZW[  89][0]=-0.504441970780036;
    Leb_Grid_XYZW[  89][1]=-0.504441970780036;
    Leb_Grid_XYZW[  89][2]= 0.700768575373573;
    Leb_Grid_XYZW[  89][3]= 0.001846715956151;

    Leb_Grid_XYZW[  90][0]= 0.504441970780036;
    Leb_Grid_XYZW[  90][1]= 0.504441970780036;
    Leb_Grid_XYZW[  90][2]=-0.700768575373573;
    Leb_Grid_XYZW[  90][3]= 0.001846715956151;

    Leb_Grid_XYZW[  91][0]=-0.504441970780036;
    Leb_Grid_XYZW[  91][1]= 0.504441970780036;
    Leb_Grid_XYZW[  91][2]=-0.700768575373573;
    Leb_Grid_XYZW[  91][3]= 0.001846715956151;

    Leb_Grid_XYZW[  92][0]= 0.504441970780036;
    Leb_Grid_XYZW[  92][1]=-0.504441970780036;
    Leb_Grid_XYZW[  92][2]=-0.700768575373573;
    Leb_Grid_XYZW[  92][3]= 0.001846715956151;

    Leb_Grid_XYZW[  93][0]=-0.504441970780036;
    Leb_Grid_XYZW[  93][1]=-0.504441970780036;
    Leb_Grid_XYZW[  93][2]=-0.700768575373573;
    Leb_Grid_XYZW[  93][3]= 0.001846715956151;

    Leb_Grid_XYZW[  94][0]= 0.504441970780036;
    Leb_Grid_XYZW[  94][1]= 0.700768575373573;
    Leb_Grid_XYZW[  94][2]= 0.504441970780036;
    Leb_Grid_XYZW[  94][3]= 0.001846715956151;

    Leb_Grid_XYZW[  95][0]=-0.504441970780036;
    Leb_Grid_XYZW[  95][1]= 0.700768575373573;
    Leb_Grid_XYZW[  95][2]= 0.504441970780036;
    Leb_Grid_XYZW[  95][3]= 0.001846715956151;

    Leb_Grid_XYZW[  96][0]= 0.504441970780036;
    Leb_Grid_XYZW[  96][1]=-0.700768575373573;
    Leb_Grid_XYZW[  96][2]= 0.504441970780036;
    Leb_Grid_XYZW[  96][3]= 0.001846715956151;

    Leb_Grid_XYZW[  97][0]=-0.504441970780036;
    Leb_Grid_XYZW[  97][1]=-0.700768575373573;
    Leb_Grid_XYZW[  97][2]= 0.504441970780036;
    Leb_Grid_XYZW[  97][3]= 0.001846715956151;

    Leb_Grid_XYZW[  98][0]= 0.504441970780036;
    Leb_Grid_XYZW[  98][1]= 0.700768575373573;
    Leb_Grid_XYZW[  98][2]=-0.504441970780036;
    Leb_Grid_XYZW[  98][3]= 0.001846715956151;

    Leb_Grid_XYZW[  99][0]=-0.504441970780036;
    Leb_Grid_XYZW[  99][1]= 0.700768575373573;
    Leb_Grid_XYZW[  99][2]=-0.504441970780036;
    Leb_Grid_XYZW[  99][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 100][0]= 0.504441970780036;
    Leb_Grid_XYZW[ 100][1]=-0.700768575373573;
    Leb_Grid_XYZW[ 100][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 100][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 101][0]=-0.504441970780036;
    Leb_Grid_XYZW[ 101][1]=-0.700768575373573;
    Leb_Grid_XYZW[ 101][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 101][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 102][0]= 0.700768575373573;
    Leb_Grid_XYZW[ 102][1]= 0.504441970780036;
    Leb_Grid_XYZW[ 102][2]= 0.504441970780036;
    Leb_Grid_XYZW[ 102][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 103][0]=-0.700768575373573;
    Leb_Grid_XYZW[ 103][1]= 0.504441970780036;
    Leb_Grid_XYZW[ 103][2]= 0.504441970780036;
    Leb_Grid_XYZW[ 103][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 104][0]= 0.700768575373573;
    Leb_Grid_XYZW[ 104][1]=-0.504441970780036;
    Leb_Grid_XYZW[ 104][2]= 0.504441970780036;
    Leb_Grid_XYZW[ 104][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 105][0]=-0.700768575373573;
    Leb_Grid_XYZW[ 105][1]=-0.504441970780036;
    Leb_Grid_XYZW[ 105][2]= 0.504441970780036;
    Leb_Grid_XYZW[ 105][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 106][0]= 0.700768575373573;
    Leb_Grid_XYZW[ 106][1]= 0.504441970780036;
    Leb_Grid_XYZW[ 106][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 106][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 107][0]=-0.700768575373573;
    Leb_Grid_XYZW[ 107][1]= 0.504441970780036;
    Leb_Grid_XYZW[ 107][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 107][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 108][0]= 0.700768575373573;
    Leb_Grid_XYZW[ 108][1]=-0.504441970780036;
    Leb_Grid_XYZW[ 108][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 108][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 109][0]=-0.700768575373573;
    Leb_Grid_XYZW[ 109][1]=-0.504441970780036;
    Leb_Grid_XYZW[ 109][2]=-0.504441970780036;
    Leb_Grid_XYZW[ 109][3]= 0.001846715956151;

    Leb_Grid_XYZW[ 110][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 110][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 110][2]= 0.802836877335274;
    Leb_Grid_XYZW[ 110][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 111][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 111][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 111][2]= 0.802836877335274;
    Leb_Grid_XYZW[ 111][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 112][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 112][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 112][2]= 0.802836877335274;
    Leb_Grid_XYZW[ 112][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 113][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 113][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 113][2]= 0.802836877335274;
    Leb_Grid_XYZW[ 113][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 114][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 114][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 114][2]=-0.802836877335274;
    Leb_Grid_XYZW[ 114][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 115][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 115][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 115][2]=-0.802836877335274;
    Leb_Grid_XYZW[ 115][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 116][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 116][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 116][2]=-0.802836877335274;
    Leb_Grid_XYZW[ 116][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 117][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 117][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 117][2]=-0.802836877335274;
    Leb_Grid_XYZW[ 117][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 118][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 118][1]= 0.802836877335274;
    Leb_Grid_XYZW[ 118][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 118][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 119][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 119][1]= 0.802836877335274;
    Leb_Grid_XYZW[ 119][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 119][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 120][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 120][1]=-0.802836877335274;
    Leb_Grid_XYZW[ 120][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 120][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 121][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 121][1]=-0.802836877335274;
    Leb_Grid_XYZW[ 121][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 121][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 122][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 122][1]= 0.802836877335274;
    Leb_Grid_XYZW[ 122][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 122][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 123][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 123][1]= 0.802836877335274;
    Leb_Grid_XYZW[ 123][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 123][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 124][0]= 0.421576178401097;
    Leb_Grid_XYZW[ 124][1]=-0.802836877335274;
    Leb_Grid_XYZW[ 124][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 124][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 125][0]=-0.421576178401097;
    Leb_Grid_XYZW[ 125][1]=-0.802836877335274;
    Leb_Grid_XYZW[ 125][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 125][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 126][0]= 0.802836877335274;
    Leb_Grid_XYZW[ 126][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 126][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 126][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 127][0]=-0.802836877335274;
    Leb_Grid_XYZW[ 127][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 127][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 127][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 128][0]= 0.802836877335274;
    Leb_Grid_XYZW[ 128][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 128][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 128][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 129][0]=-0.802836877335274;
    Leb_Grid_XYZW[ 129][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 129][2]= 0.421576178401097;
    Leb_Grid_XYZW[ 129][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 130][0]= 0.802836877335274;
    Leb_Grid_XYZW[ 130][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 130][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 130][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 131][0]=-0.802836877335274;
    Leb_Grid_XYZW[ 131][1]= 0.421576178401097;
    Leb_Grid_XYZW[ 131][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 131][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 132][0]= 0.802836877335274;
    Leb_Grid_XYZW[ 132][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 132][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 132][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 133][0]=-0.802836877335274;
    Leb_Grid_XYZW[ 133][1]=-0.421576178401097;
    Leb_Grid_XYZW[ 133][2]=-0.421576178401097;
    Leb_Grid_XYZW[ 133][3]= 0.001818471778163;

    Leb_Grid_XYZW[ 134][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 134][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 134][2]= 0.883078727934133;
    Leb_Grid_XYZW[ 134][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 135][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 135][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 135][2]= 0.883078727934133;
    Leb_Grid_XYZW[ 135][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 136][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 136][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 136][2]= 0.883078727934133;
    Leb_Grid_XYZW[ 136][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 137][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 137][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 137][2]= 0.883078727934133;
    Leb_Grid_XYZW[ 137][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 138][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 138][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 138][2]=-0.883078727934133;
    Leb_Grid_XYZW[ 138][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 139][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 139][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 139][2]=-0.883078727934133;
    Leb_Grid_XYZW[ 139][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 140][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 140][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 140][2]=-0.883078727934133;
    Leb_Grid_XYZW[ 140][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 141][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 141][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 141][2]=-0.883078727934133;
    Leb_Grid_XYZW[ 141][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 142][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 142][1]= 0.883078727934133;
    Leb_Grid_XYZW[ 142][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 142][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 143][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 143][1]= 0.883078727934133;
    Leb_Grid_XYZW[ 143][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 143][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 144][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 144][1]=-0.883078727934133;
    Leb_Grid_XYZW[ 144][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 144][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 145][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 145][1]=-0.883078727934133;
    Leb_Grid_XYZW[ 145][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 145][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 146][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 146][1]= 0.883078727934133;
    Leb_Grid_XYZW[ 146][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 146][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 147][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 147][1]= 0.883078727934133;
    Leb_Grid_XYZW[ 147][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 147][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 148][0]= 0.331792073647212;
    Leb_Grid_XYZW[ 148][1]=-0.883078727934133;
    Leb_Grid_XYZW[ 148][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 148][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 149][0]=-0.331792073647212;
    Leb_Grid_XYZW[ 149][1]=-0.883078727934133;
    Leb_Grid_XYZW[ 149][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 149][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 150][0]= 0.883078727934133;
    Leb_Grid_XYZW[ 150][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 150][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 150][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 151][0]=-0.883078727934133;
    Leb_Grid_XYZW[ 151][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 151][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 151][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 152][0]= 0.883078727934133;
    Leb_Grid_XYZW[ 152][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 152][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 152][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 153][0]=-0.883078727934133;
    Leb_Grid_XYZW[ 153][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 153][2]= 0.331792073647212;
    Leb_Grid_XYZW[ 153][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 154][0]= 0.883078727934133;
    Leb_Grid_XYZW[ 154][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 154][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 154][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 155][0]=-0.883078727934133;
    Leb_Grid_XYZW[ 155][1]= 0.331792073647212;
    Leb_Grid_XYZW[ 155][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 155][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 156][0]= 0.883078727934133;
    Leb_Grid_XYZW[ 156][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 156][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 156][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 157][0]=-0.883078727934133;
    Leb_Grid_XYZW[ 157][1]=-0.331792073647212;
    Leb_Grid_XYZW[ 157][2]=-0.331792073647212;
    Leb_Grid_XYZW[ 157][3]= 0.001749564657281;

    Leb_Grid_XYZW[ 158][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 158][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 158][2]= 0.941414158220403;
    Leb_Grid_XYZW[ 158][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 159][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 159][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 159][2]= 0.941414158220403;
    Leb_Grid_XYZW[ 159][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 160][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 160][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 160][2]= 0.941414158220403;
    Leb_Grid_XYZW[ 160][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 161][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 161][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 161][2]= 0.941414158220403;
    Leb_Grid_XYZW[ 161][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 162][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 162][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 162][2]=-0.941414158220403;
    Leb_Grid_XYZW[ 162][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 163][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 163][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 163][2]=-0.941414158220403;
    Leb_Grid_XYZW[ 163][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 164][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 164][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 164][2]=-0.941414158220403;
    Leb_Grid_XYZW[ 164][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 165][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 165][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 165][2]=-0.941414158220403;
    Leb_Grid_XYZW[ 165][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 166][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 166][1]= 0.941414158220403;
    Leb_Grid_XYZW[ 166][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 166][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 167][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 167][1]= 0.941414158220403;
    Leb_Grid_XYZW[ 167][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 167][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 168][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 168][1]=-0.941414158220403;
    Leb_Grid_XYZW[ 168][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 168][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 169][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 169][1]=-0.941414158220403;
    Leb_Grid_XYZW[ 169][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 169][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 170][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 170][1]= 0.941414158220403;
    Leb_Grid_XYZW[ 170][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 170][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 171][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 171][1]= 0.941414158220403;
    Leb_Grid_XYZW[ 171][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 171][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 172][0]= 0.238473670142189;
    Leb_Grid_XYZW[ 172][1]=-0.941414158220403;
    Leb_Grid_XYZW[ 172][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 172][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 173][0]=-0.238473670142189;
    Leb_Grid_XYZW[ 173][1]=-0.941414158220403;
    Leb_Grid_XYZW[ 173][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 173][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 174][0]= 0.941414158220403;
    Leb_Grid_XYZW[ 174][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 174][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 174][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 175][0]=-0.941414158220403;
    Leb_Grid_XYZW[ 175][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 175][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 175][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 176][0]= 0.941414158220403;
    Leb_Grid_XYZW[ 176][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 176][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 176][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 177][0]=-0.941414158220403;
    Leb_Grid_XYZW[ 177][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 177][2]= 0.238473670142189;
    Leb_Grid_XYZW[ 177][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 178][0]= 0.941414158220403;
    Leb_Grid_XYZW[ 178][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 178][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 178][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 179][0]=-0.941414158220403;
    Leb_Grid_XYZW[ 179][1]= 0.238473670142189;
    Leb_Grid_XYZW[ 179][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 179][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 180][0]= 0.941414158220403;
    Leb_Grid_XYZW[ 180][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 180][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 180][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 181][0]=-0.941414158220403;
    Leb_Grid_XYZW[ 181][1]=-0.238473670142189;
    Leb_Grid_XYZW[ 181][2]=-0.238473670142189;
    Leb_Grid_XYZW[ 181][3]= 0.001617210647254;

    Leb_Grid_XYZW[ 182][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 182][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 182][2]= 0.978480583762694;
    Leb_Grid_XYZW[ 182][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 183][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 183][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 183][2]= 0.978480583762694;
    Leb_Grid_XYZW[ 183][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 184][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 184][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 184][2]= 0.978480583762694;
    Leb_Grid_XYZW[ 184][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 185][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 185][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 185][2]= 0.978480583762694;
    Leb_Grid_XYZW[ 185][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 186][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 186][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 186][2]=-0.978480583762694;
    Leb_Grid_XYZW[ 186][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 187][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 187][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 187][2]=-0.978480583762694;
    Leb_Grid_XYZW[ 187][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 188][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 188][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 188][2]=-0.978480583762694;
    Leb_Grid_XYZW[ 188][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 189][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 189][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 189][2]=-0.978480583762694;
    Leb_Grid_XYZW[ 189][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 190][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 190][1]= 0.978480583762694;
    Leb_Grid_XYZW[ 190][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 190][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 191][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 191][1]= 0.978480583762694;
    Leb_Grid_XYZW[ 191][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 191][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 192][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 192][1]=-0.978480583762694;
    Leb_Grid_XYZW[ 192][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 192][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 193][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 193][1]=-0.978480583762694;
    Leb_Grid_XYZW[ 193][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 193][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 194][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 194][1]= 0.978480583762694;
    Leb_Grid_XYZW[ 194][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 194][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 195][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 195][1]= 0.978480583762694;
    Leb_Grid_XYZW[ 195][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 195][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 196][0]= 0.145903644915776;
    Leb_Grid_XYZW[ 196][1]=-0.978480583762694;
    Leb_Grid_XYZW[ 196][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 196][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 197][0]=-0.145903644915776;
    Leb_Grid_XYZW[ 197][1]=-0.978480583762694;
    Leb_Grid_XYZW[ 197][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 197][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 198][0]= 0.978480583762694;
    Leb_Grid_XYZW[ 198][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 198][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 198][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 199][0]=-0.978480583762694;
    Leb_Grid_XYZW[ 199][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 199][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 199][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 200][0]= 0.978480583762694;
    Leb_Grid_XYZW[ 200][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 200][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 200][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 201][0]=-0.978480583762694;
    Leb_Grid_XYZW[ 201][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 201][2]= 0.145903644915776;
    Leb_Grid_XYZW[ 201][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 202][0]= 0.978480583762694;
    Leb_Grid_XYZW[ 202][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 202][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 202][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 203][0]=-0.978480583762694;
    Leb_Grid_XYZW[ 203][1]= 0.145903644915776;
    Leb_Grid_XYZW[ 203][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 203][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 204][0]= 0.978480583762694;
    Leb_Grid_XYZW[ 204][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 204][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 204][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 205][0]=-0.978480583762694;
    Leb_Grid_XYZW[ 205][1]=-0.145903644915776;
    Leb_Grid_XYZW[ 205][2]=-0.145903644915776;
    Leb_Grid_XYZW[ 205][3]= 0.001384737234852;

    Leb_Grid_XYZW[ 206][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 206][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 206][2]= 0.996278129754016;
    Leb_Grid_XYZW[ 206][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 207][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 207][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 207][2]= 0.996278129754016;
    Leb_Grid_XYZW[ 207][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 208][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 208][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 208][2]= 0.996278129754016;
    Leb_Grid_XYZW[ 208][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 209][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 209][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 209][2]= 0.996278129754016;
    Leb_Grid_XYZW[ 209][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 210][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 210][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 210][2]=-0.996278129754016;
    Leb_Grid_XYZW[ 210][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 211][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 211][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 211][2]=-0.996278129754016;
    Leb_Grid_XYZW[ 211][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 212][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 212][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 212][2]=-0.996278129754016;
    Leb_Grid_XYZW[ 212][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 213][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 213][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 213][2]=-0.996278129754016;
    Leb_Grid_XYZW[ 213][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 214][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 214][1]= 0.996278129754016;
    Leb_Grid_XYZW[ 214][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 214][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 215][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 215][1]= 0.996278129754016;
    Leb_Grid_XYZW[ 215][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 215][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 216][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 216][1]=-0.996278129754016;
    Leb_Grid_XYZW[ 216][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 216][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 217][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 217][1]=-0.996278129754016;
    Leb_Grid_XYZW[ 217][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 217][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 218][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 218][1]= 0.996278129754016;
    Leb_Grid_XYZW[ 218][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 218][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 219][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 219][1]= 0.996278129754016;
    Leb_Grid_XYZW[ 219][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 219][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 220][0]= 0.060950341155072;
    Leb_Grid_XYZW[ 220][1]=-0.996278129754016;
    Leb_Grid_XYZW[ 220][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 220][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 221][0]=-0.060950341155072;
    Leb_Grid_XYZW[ 221][1]=-0.996278129754016;
    Leb_Grid_XYZW[ 221][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 221][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 222][0]= 0.996278129754016;
    Leb_Grid_XYZW[ 222][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 222][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 222][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 223][0]=-0.996278129754016;
    Leb_Grid_XYZW[ 223][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 223][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 223][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 224][0]= 0.996278129754016;
    Leb_Grid_XYZW[ 224][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 224][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 224][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 225][0]=-0.996278129754016;
    Leb_Grid_XYZW[ 225][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 225][2]= 0.060950341155072;
    Leb_Grid_XYZW[ 225][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 226][0]= 0.996278129754016;
    Leb_Grid_XYZW[ 226][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 226][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 226][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 227][0]=-0.996278129754016;
    Leb_Grid_XYZW[ 227][1]= 0.060950341155072;
    Leb_Grid_XYZW[ 227][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 227][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 228][0]= 0.996278129754016;
    Leb_Grid_XYZW[ 228][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 228][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 228][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 229][0]=-0.996278129754016;
    Leb_Grid_XYZW[ 229][1]=-0.060950341155072;
    Leb_Grid_XYZW[ 229][2]=-0.060950341155072;
    Leb_Grid_XYZW[ 229][3]= 0.000976433116505;

    Leb_Grid_XYZW[ 230][0]= 0.611684344200988;
    Leb_Grid_XYZW[ 230][1]= 0.791101929626902;
    Leb_Grid_XYZW[ 230][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 230][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 231][0]=-0.611684344200988;
    Leb_Grid_XYZW[ 231][1]= 0.791101929626902;
    Leb_Grid_XYZW[ 231][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 231][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 232][0]= 0.611684344200988;
    Leb_Grid_XYZW[ 232][1]=-0.791101929626902;
    Leb_Grid_XYZW[ 232][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 232][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 233][0]=-0.611684344200988;
    Leb_Grid_XYZW[ 233][1]=-0.791101929626902;
    Leb_Grid_XYZW[ 233][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 233][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 234][0]= 0.791101929626902;
    Leb_Grid_XYZW[ 234][1]= 0.611684344200988;
    Leb_Grid_XYZW[ 234][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 234][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 235][0]=-0.791101929626902;
    Leb_Grid_XYZW[ 235][1]= 0.611684344200988;
    Leb_Grid_XYZW[ 235][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 235][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 236][0]= 0.791101929626902;
    Leb_Grid_XYZW[ 236][1]=-0.611684344200988;
    Leb_Grid_XYZW[ 236][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 236][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 237][0]=-0.791101929626902;
    Leb_Grid_XYZW[ 237][1]=-0.611684344200988;
    Leb_Grid_XYZW[ 237][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 237][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 238][0]= 0.611684344200988;
    Leb_Grid_XYZW[ 238][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 238][2]= 0.791101929626902;
    Leb_Grid_XYZW[ 238][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 239][0]=-0.611684344200988;
    Leb_Grid_XYZW[ 239][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 239][2]= 0.791101929626902;
    Leb_Grid_XYZW[ 239][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 240][0]= 0.611684344200988;
    Leb_Grid_XYZW[ 240][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 240][2]=-0.791101929626902;
    Leb_Grid_XYZW[ 240][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 241][0]=-0.611684344200988;
    Leb_Grid_XYZW[ 241][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 241][2]=-0.791101929626902;
    Leb_Grid_XYZW[ 241][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 242][0]= 0.791101929626902;
    Leb_Grid_XYZW[ 242][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 242][2]= 0.611684344200988;
    Leb_Grid_XYZW[ 242][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 243][0]=-0.791101929626902;
    Leb_Grid_XYZW[ 243][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 243][2]= 0.611684344200988;
    Leb_Grid_XYZW[ 243][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 244][0]= 0.791101929626902;
    Leb_Grid_XYZW[ 244][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 244][2]=-0.611684344200988;
    Leb_Grid_XYZW[ 244][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 245][0]=-0.791101929626902;
    Leb_Grid_XYZW[ 245][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 245][2]=-0.611684344200988;
    Leb_Grid_XYZW[ 245][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 246][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 246][1]= 0.611684344200988;
    Leb_Grid_XYZW[ 246][2]= 0.791101929626902;
    Leb_Grid_XYZW[ 246][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 247][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 247][1]=-0.611684344200988;
    Leb_Grid_XYZW[ 247][2]= 0.791101929626902;
    Leb_Grid_XYZW[ 247][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 248][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 248][1]= 0.611684344200988;
    Leb_Grid_XYZW[ 248][2]=-0.791101929626902;
    Leb_Grid_XYZW[ 248][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 249][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 249][1]=-0.611684344200988;
    Leb_Grid_XYZW[ 249][2]=-0.791101929626902;
    Leb_Grid_XYZW[ 249][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 250][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 250][1]= 0.791101929626902;
    Leb_Grid_XYZW[ 250][2]= 0.611684344200988;
    Leb_Grid_XYZW[ 250][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 251][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 251][1]=-0.791101929626902;
    Leb_Grid_XYZW[ 251][2]= 0.611684344200988;
    Leb_Grid_XYZW[ 251][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 252][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 252][1]= 0.791101929626902;
    Leb_Grid_XYZW[ 252][2]=-0.611684344200988;
    Leb_Grid_XYZW[ 252][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 253][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 253][1]=-0.791101929626902;
    Leb_Grid_XYZW[ 253][2]=-0.611684344200988;
    Leb_Grid_XYZW[ 253][3]= 0.001857161196774;

    Leb_Grid_XYZW[ 254][0]= 0.396475534819986;
    Leb_Grid_XYZW[ 254][1]= 0.918045287711454;
    Leb_Grid_XYZW[ 254][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 254][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 255][0]=-0.396475534819986;
    Leb_Grid_XYZW[ 255][1]= 0.918045287711454;
    Leb_Grid_XYZW[ 255][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 255][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 256][0]= 0.396475534819986;
    Leb_Grid_XYZW[ 256][1]=-0.918045287711454;
    Leb_Grid_XYZW[ 256][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 256][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 257][0]=-0.396475534819986;
    Leb_Grid_XYZW[ 257][1]=-0.918045287711454;
    Leb_Grid_XYZW[ 257][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 257][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 258][0]= 0.918045287711454;
    Leb_Grid_XYZW[ 258][1]= 0.396475534819986;
    Leb_Grid_XYZW[ 258][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 258][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 259][0]=-0.918045287711454;
    Leb_Grid_XYZW[ 259][1]= 0.396475534819986;
    Leb_Grid_XYZW[ 259][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 259][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 260][0]= 0.918045287711454;
    Leb_Grid_XYZW[ 260][1]=-0.396475534819986;
    Leb_Grid_XYZW[ 260][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 260][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 261][0]=-0.918045287711454;
    Leb_Grid_XYZW[ 261][1]=-0.396475534819986;
    Leb_Grid_XYZW[ 261][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 261][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 262][0]= 0.396475534819986;
    Leb_Grid_XYZW[ 262][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 262][2]= 0.918045287711454;
    Leb_Grid_XYZW[ 262][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 263][0]=-0.396475534819986;
    Leb_Grid_XYZW[ 263][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 263][2]= 0.918045287711454;
    Leb_Grid_XYZW[ 263][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 264][0]= 0.396475534819986;
    Leb_Grid_XYZW[ 264][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 264][2]=-0.918045287711454;
    Leb_Grid_XYZW[ 264][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 265][0]=-0.396475534819986;
    Leb_Grid_XYZW[ 265][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 265][2]=-0.918045287711454;
    Leb_Grid_XYZW[ 265][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 266][0]= 0.918045287711454;
    Leb_Grid_XYZW[ 266][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 266][2]= 0.396475534819986;
    Leb_Grid_XYZW[ 266][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 267][0]=-0.918045287711454;
    Leb_Grid_XYZW[ 267][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 267][2]= 0.396475534819986;
    Leb_Grid_XYZW[ 267][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 268][0]= 0.918045287711454;
    Leb_Grid_XYZW[ 268][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 268][2]=-0.396475534819986;
    Leb_Grid_XYZW[ 268][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 269][0]=-0.918045287711454;
    Leb_Grid_XYZW[ 269][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 269][2]=-0.396475534819986;
    Leb_Grid_XYZW[ 269][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 270][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 270][1]= 0.396475534819986;
    Leb_Grid_XYZW[ 270][2]= 0.918045287711454;
    Leb_Grid_XYZW[ 270][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 271][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 271][1]=-0.396475534819986;
    Leb_Grid_XYZW[ 271][2]= 0.918045287711454;
    Leb_Grid_XYZW[ 271][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 272][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 272][1]= 0.396475534819986;
    Leb_Grid_XYZW[ 272][2]=-0.918045287711454;
    Leb_Grid_XYZW[ 272][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 273][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 273][1]=-0.396475534819986;
    Leb_Grid_XYZW[ 273][2]=-0.918045287711454;
    Leb_Grid_XYZW[ 273][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 274][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 274][1]= 0.918045287711454;
    Leb_Grid_XYZW[ 274][2]= 0.396475534819986;
    Leb_Grid_XYZW[ 274][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 275][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 275][1]=-0.918045287711454;
    Leb_Grid_XYZW[ 275][2]= 0.396475534819986;
    Leb_Grid_XYZW[ 275][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 276][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 276][1]= 0.918045287711454;
    Leb_Grid_XYZW[ 276][2]=-0.396475534819986;
    Leb_Grid_XYZW[ 276][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 277][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 277][1]=-0.918045287711454;
    Leb_Grid_XYZW[ 277][2]=-0.396475534819986;
    Leb_Grid_XYZW[ 277][3]= 0.001705153996396;

    Leb_Grid_XYZW[ 278][0]= 0.172478200990772;
    Leb_Grid_XYZW[ 278][1]= 0.985013335028002;
    Leb_Grid_XYZW[ 278][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 278][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 279][0]=-0.172478200990772;
    Leb_Grid_XYZW[ 279][1]= 0.985013335028002;
    Leb_Grid_XYZW[ 279][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 279][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 280][0]= 0.172478200990772;
    Leb_Grid_XYZW[ 280][1]=-0.985013335028002;
    Leb_Grid_XYZW[ 280][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 280][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 281][0]=-0.172478200990772;
    Leb_Grid_XYZW[ 281][1]=-0.985013335028002;
    Leb_Grid_XYZW[ 281][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 281][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 282][0]= 0.985013335028002;
    Leb_Grid_XYZW[ 282][1]= 0.172478200990772;
    Leb_Grid_XYZW[ 282][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 282][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 283][0]=-0.985013335028002;
    Leb_Grid_XYZW[ 283][1]= 0.172478200990772;
    Leb_Grid_XYZW[ 283][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 283][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 284][0]= 0.985013335028002;
    Leb_Grid_XYZW[ 284][1]=-0.172478200990772;
    Leb_Grid_XYZW[ 284][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 284][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 285][0]=-0.985013335028002;
    Leb_Grid_XYZW[ 285][1]=-0.172478200990772;
    Leb_Grid_XYZW[ 285][2]= 0.000000000000000;
    Leb_Grid_XYZW[ 285][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 286][0]= 0.172478200990772;
    Leb_Grid_XYZW[ 286][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 286][2]= 0.985013335028002;
    Leb_Grid_XYZW[ 286][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 287][0]=-0.172478200990772;
    Leb_Grid_XYZW[ 287][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 287][2]= 0.985013335028002;
    Leb_Grid_XYZW[ 287][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 288][0]= 0.172478200990772;
    Leb_Grid_XYZW[ 288][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 288][2]=-0.985013335028002;
    Leb_Grid_XYZW[ 288][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 289][0]=-0.172478200990772;
    Leb_Grid_XYZW[ 289][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 289][2]=-0.985013335028002;
    Leb_Grid_XYZW[ 289][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 290][0]= 0.985013335028002;
    Leb_Grid_XYZW[ 290][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 290][2]= 0.172478200990772;
    Leb_Grid_XYZW[ 290][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 291][0]=-0.985013335028002;
    Leb_Grid_XYZW[ 291][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 291][2]= 0.172478200990772;
    Leb_Grid_XYZW[ 291][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 292][0]= 0.985013335028002;
    Leb_Grid_XYZW[ 292][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 292][2]=-0.172478200990772;
    Leb_Grid_XYZW[ 292][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 293][0]=-0.985013335028002;
    Leb_Grid_XYZW[ 293][1]= 0.000000000000000;
    Leb_Grid_XYZW[ 293][2]=-0.172478200990772;
    Leb_Grid_XYZW[ 293][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 294][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 294][1]= 0.172478200990772;
    Leb_Grid_XYZW[ 294][2]= 0.985013335028002;
    Leb_Grid_XYZW[ 294][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 295][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 295][1]=-0.172478200990772;
    Leb_Grid_XYZW[ 295][2]= 0.985013335028002;
    Leb_Grid_XYZW[ 295][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 296][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 296][1]= 0.172478200990772;
    Leb_Grid_XYZW[ 296][2]=-0.985013335028002;
    Leb_Grid_XYZW[ 296][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 297][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 297][1]=-0.172478200990772;
    Leb_Grid_XYZW[ 297][2]=-0.985013335028002;
    Leb_Grid_XYZW[ 297][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 298][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 298][1]= 0.985013335028002;
    Leb_Grid_XYZW[ 298][2]= 0.172478200990772;
    Leb_Grid_XYZW[ 298][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 299][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 299][1]=-0.985013335028002;
    Leb_Grid_XYZW[ 299][2]= 0.172478200990772;
    Leb_Grid_XYZW[ 299][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 300][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 300][1]= 0.985013335028002;
    Leb_Grid_XYZW[ 300][2]=-0.172478200990772;
    Leb_Grid_XYZW[ 300][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 301][0]= 0.000000000000000;
    Leb_Grid_XYZW[ 301][1]=-0.985013335028002;
    Leb_Grid_XYZW[ 301][2]=-0.172478200990772;
    Leb_Grid_XYZW[ 301][3]= 0.001300321685886;

    Leb_Grid_XYZW[ 302][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 302][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 302][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 302][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 303][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 303][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 303][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 303][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 304][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 304][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 304][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 304][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 305][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 305][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 305][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 305][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 306][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 306][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 306][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 306][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 307][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 307][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 307][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 307][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 308][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 308][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 308][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 308][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 309][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 309][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 309][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 309][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 310][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 310][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 310][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 310][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 311][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 311][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 311][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 311][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 312][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 312][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 312][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 312][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 313][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 313][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 313][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 313][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 314][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 314][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 314][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 314][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 315][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 315][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 315][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 315][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 316][0]= 0.561026380862206;
    Leb_Grid_XYZW[ 316][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 316][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 316][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 317][0]=-0.561026380862206;
    Leb_Grid_XYZW[ 317][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 317][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 317][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 318][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 318][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 318][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 318][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 319][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 319][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 319][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 319][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 320][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 320][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 320][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 320][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 321][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 321][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 321][2]= 0.749310611904116;
    Leb_Grid_XYZW[ 321][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 322][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 322][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 322][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 322][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 323][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 323][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 323][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 323][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 324][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 324][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 324][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 324][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 325][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 325][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 325][2]=-0.749310611904116;
    Leb_Grid_XYZW[ 325][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 326][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 326][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 326][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 326][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 327][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 327][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 327][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 327][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 328][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 328][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 328][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 328][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 329][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 329][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 329][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 329][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 330][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 330][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 330][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 330][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 331][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 331][1]= 0.749310611904116;
    Leb_Grid_XYZW[ 331][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 331][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 332][0]= 0.351828092773352;
    Leb_Grid_XYZW[ 332][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 332][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 332][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 333][0]=-0.351828092773352;
    Leb_Grid_XYZW[ 333][1]=-0.749310611904116;
    Leb_Grid_XYZW[ 333][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 333][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 334][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 334][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 334][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 334][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 335][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 335][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 335][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 335][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 336][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 336][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 336][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 336][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 337][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 337][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 337][2]= 0.351828092773352;
    Leb_Grid_XYZW[ 337][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 338][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 338][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 338][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 338][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 339][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 339][1]= 0.561026380862206;
    Leb_Grid_XYZW[ 339][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 339][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 340][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 340][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 340][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 340][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 341][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 341][1]=-0.561026380862206;
    Leb_Grid_XYZW[ 341][2]=-0.351828092773352;
    Leb_Grid_XYZW[ 341][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 342][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 342][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 342][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 342][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 343][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 343][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 343][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 343][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 344][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 344][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 344][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 344][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 345][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 345][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 345][2]= 0.561026380862206;
    Leb_Grid_XYZW[ 345][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 346][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 346][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 346][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 346][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 347][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 347][1]= 0.351828092773352;
    Leb_Grid_XYZW[ 347][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 347][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 348][0]= 0.749310611904116;
    Leb_Grid_XYZW[ 348][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 348][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 348][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 349][0]=-0.749310611904116;
    Leb_Grid_XYZW[ 349][1]=-0.351828092773352;
    Leb_Grid_XYZW[ 349][2]=-0.561026380862206;
    Leb_Grid_XYZW[ 349][3]= 0.001842866472905;

    Leb_Grid_XYZW[ 350][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 350][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 350][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 350][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 351][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 351][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 351][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 351][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 352][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 352][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 352][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 352][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 353][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 353][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 353][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 353][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 354][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 354][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 354][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 354][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 355][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 355][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 355][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 355][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 356][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 356][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 356][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 356][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 357][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 357][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 357][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 357][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 358][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 358][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 358][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 358][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 359][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 359][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 359][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 359][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 360][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 360][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 360][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 360][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 361][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 361][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 361][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 361][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 362][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 362][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 362][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 362][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 363][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 363][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 363][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 363][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 364][0]= 0.474239284255198;
    Leb_Grid_XYZW[ 364][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 364][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 364][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 365][0]=-0.474239284255198;
    Leb_Grid_XYZW[ 365][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 365][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 365][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 366][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 366][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 366][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 366][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 367][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 367][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 367][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 367][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 368][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 368][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 368][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 368][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 369][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 369][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 369][2]= 0.840047488359050;
    Leb_Grid_XYZW[ 369][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 370][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 370][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 370][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 370][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 371][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 371][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 371][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 371][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 372][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 372][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 372][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 372][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 373][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 373][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 373][2]=-0.840047488359050;
    Leb_Grid_XYZW[ 373][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 374][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 374][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 374][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 374][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 375][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 375][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 375][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 375][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 376][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 376][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 376][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 376][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 377][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 377][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 377][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 377][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 378][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 378][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 378][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 378][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 379][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 379][1]= 0.840047488359050;
    Leb_Grid_XYZW[ 379][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 379][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 380][0]= 0.263471665593795;
    Leb_Grid_XYZW[ 380][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 380][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 380][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 381][0]=-0.263471665593795;
    Leb_Grid_XYZW[ 381][1]=-0.840047488359050;
    Leb_Grid_XYZW[ 381][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 381][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 382][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 382][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 382][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 382][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 383][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 383][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 383][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 383][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 384][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 384][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 384][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 384][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 385][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 385][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 385][2]= 0.263471665593795;
    Leb_Grid_XYZW[ 385][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 386][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 386][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 386][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 386][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 387][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 387][1]= 0.474239284255198;
    Leb_Grid_XYZW[ 387][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 387][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 388][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 388][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 388][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 388][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 389][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 389][1]=-0.474239284255198;
    Leb_Grid_XYZW[ 389][2]=-0.263471665593795;
    Leb_Grid_XYZW[ 389][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 390][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 390][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 390][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 390][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 391][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 391][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 391][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 391][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 392][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 392][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 392][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 392][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 393][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 393][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 393][2]= 0.474239284255198;
    Leb_Grid_XYZW[ 393][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 394][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 394][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 394][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 394][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 395][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 395][1]= 0.263471665593795;
    Leb_Grid_XYZW[ 395][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 395][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 396][0]= 0.840047488359050;
    Leb_Grid_XYZW[ 396][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 396][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 396][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 397][0]=-0.840047488359050;
    Leb_Grid_XYZW[ 397][1]=-0.263471665593795;
    Leb_Grid_XYZW[ 397][2]=-0.474239284255198;
    Leb_Grid_XYZW[ 397][3]= 0.001802658934377;

    Leb_Grid_XYZW[ 398][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 398][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 398][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 398][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 399][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 399][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 399][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 399][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 400][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 400][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 400][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 400][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 401][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 401][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 401][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 401][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 402][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 402][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 402][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 402][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 403][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 403][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 403][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 403][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 404][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 404][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 404][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 404][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 405][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 405][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 405][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 405][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 406][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 406][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 406][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 406][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 407][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 407][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 407][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 407][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 408][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 408][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 408][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 408][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 409][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 409][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 409][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 409][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 410][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 410][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 410][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 410][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 411][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 411][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 411][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 411][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 412][0]= 0.598412649788538;
    Leb_Grid_XYZW[ 412][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 412][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 412][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 413][0]=-0.598412649788538;
    Leb_Grid_XYZW[ 413][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 413][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 413][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 414][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 414][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 414][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 414][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 415][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 415][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 415][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 415][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 416][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 416][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 416][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 416][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 417][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 417][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 417][2]= 0.780320742479920;
    Leb_Grid_XYZW[ 417][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 418][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 418][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 418][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 418][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 419][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 419][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 419][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 419][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 420][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 420][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 420][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 420][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 421][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 421][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 421][2]=-0.780320742479920;
    Leb_Grid_XYZW[ 421][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 422][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 422][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 422][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 422][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 423][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 423][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 423][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 423][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 424][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 424][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 424][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 424][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 425][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 425][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 425][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 425][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 426][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 426][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 426][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 426][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 427][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 427][1]= 0.780320742479920;
    Leb_Grid_XYZW[ 427][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 427][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 428][0]= 0.181664084036021;
    Leb_Grid_XYZW[ 428][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 428][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 428][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 429][0]=-0.181664084036021;
    Leb_Grid_XYZW[ 429][1]=-0.780320742479920;
    Leb_Grid_XYZW[ 429][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 429][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 430][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 430][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 430][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 430][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 431][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 431][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 431][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 431][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 432][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 432][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 432][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 432][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 433][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 433][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 433][2]= 0.181664084036021;
    Leb_Grid_XYZW[ 433][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 434][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 434][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 434][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 434][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 435][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 435][1]= 0.598412649788538;
    Leb_Grid_XYZW[ 435][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 435][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 436][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 436][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 436][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 436][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 437][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 437][1]=-0.598412649788538;
    Leb_Grid_XYZW[ 437][2]=-0.181664084036021;
    Leb_Grid_XYZW[ 437][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 438][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 438][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 438][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 438][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 439][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 439][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 439][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 439][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 440][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 440][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 440][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 440][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 441][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 441][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 441][2]= 0.598412649788538;
    Leb_Grid_XYZW[ 441][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 442][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 442][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 442][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 442][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 443][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 443][1]= 0.181664084036021;
    Leb_Grid_XYZW[ 443][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 443][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 444][0]= 0.780320742479920;
    Leb_Grid_XYZW[ 444][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 444][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 444][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 445][0]=-0.780320742479920;
    Leb_Grid_XYZW[ 445][1]=-0.181664084036021;
    Leb_Grid_XYZW[ 445][2]=-0.598412649788538;
    Leb_Grid_XYZW[ 445][3]= 0.001849830560444;

    Leb_Grid_XYZW[ 446][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 446][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 446][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 446][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 447][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 447][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 447][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 447][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 448][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 448][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 448][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 448][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 449][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 449][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 449][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 449][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 450][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 450][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 450][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 450][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 451][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 451][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 451][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 451][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 452][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 452][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 452][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 452][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 453][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 453][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 453][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 453][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 454][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 454][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 454][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 454][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 455][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 455][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 455][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 455][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 456][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 456][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 456][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 456][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 457][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 457][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 457][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 457][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 458][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 458][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 458][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 458][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 459][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 459][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 459][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 459][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 460][0]= 0.379103540769556;
    Leb_Grid_XYZW[ 460][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 460][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 460][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 461][0]=-0.379103540769556;
    Leb_Grid_XYZW[ 461][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 461][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 461][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 462][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 462][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 462][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 462][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 463][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 463][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 463][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 463][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 464][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 464][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 464][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 464][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 465][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 465][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 465][2]= 0.909213475092374;
    Leb_Grid_XYZW[ 465][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 466][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 466][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 466][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 466][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 467][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 467][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 467][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 467][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 468][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 468][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 468][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 468][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 469][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 469][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 469][2]=-0.909213475092374;
    Leb_Grid_XYZW[ 469][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 470][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 470][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 470][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 470][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 471][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 471][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 471][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 471][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 472][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 472][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 472][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 472][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 473][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 473][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 473][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 473][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 474][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 474][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 474][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 474][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 475][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 475][1]= 0.909213475092374;
    Leb_Grid_XYZW[ 475][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 475][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 476][0]= 0.172079522565688;
    Leb_Grid_XYZW[ 476][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 476][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 476][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 477][0]=-0.172079522565688;
    Leb_Grid_XYZW[ 477][1]=-0.909213475092374;
    Leb_Grid_XYZW[ 477][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 477][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 478][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 478][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 478][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 478][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 479][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 479][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 479][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 479][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 480][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 480][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 480][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 480][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 481][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 481][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 481][2]= 0.172079522565688;
    Leb_Grid_XYZW[ 481][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 482][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 482][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 482][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 482][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 483][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 483][1]= 0.379103540769556;
    Leb_Grid_XYZW[ 483][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 483][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 484][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 484][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 484][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 484][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 485][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 485][1]=-0.379103540769556;
    Leb_Grid_XYZW[ 485][2]=-0.172079522565688;
    Leb_Grid_XYZW[ 485][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 486][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 486][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 486][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 486][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 487][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 487][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 487][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 487][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 488][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 488][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 488][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 488][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 489][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 489][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 489][2]= 0.379103540769556;
    Leb_Grid_XYZW[ 489][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 490][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 490][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 490][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 490][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 491][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 491][1]= 0.172079522565688;
    Leb_Grid_XYZW[ 491][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 491][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 492][0]= 0.909213475092374;
    Leb_Grid_XYZW[ 492][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 492][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 492][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 493][0]=-0.909213475092374;
    Leb_Grid_XYZW[ 493][1]=-0.172079522565688;
    Leb_Grid_XYZW[ 493][2]=-0.379103540769556;
    Leb_Grid_XYZW[ 493][3]= 0.001713904507107;

    Leb_Grid_XYZW[ 494][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 494][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 494][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 494][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 495][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 495][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 495][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 495][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 496][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 496][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 496][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 496][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 497][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 497][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 497][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 497][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 498][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 498][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 498][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 498][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 499][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 499][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 499][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 499][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 500][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 500][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 500][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 500][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 501][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 501][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 501][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 501][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 502][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 502][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 502][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 502][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 503][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 503][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 503][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 503][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 504][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 504][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 504][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 504][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 505][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 505][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 505][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 505][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 506][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 506][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 506][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 506][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 507][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 507][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 507][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 507][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 508][0]= 0.277867319058624;
    Leb_Grid_XYZW[ 508][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 508][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 508][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 509][0]=-0.277867319058624;
    Leb_Grid_XYZW[ 509][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 509][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 509][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 510][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 510][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 510][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 510][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 511][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 511][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 511][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 511][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 512][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 512][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 512][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 512][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 513][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 513][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 513][2]= 0.957102074310073;
    Leb_Grid_XYZW[ 513][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 514][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 514][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 514][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 514][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 515][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 515][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 515][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 515][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 516][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 516][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 516][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 516][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 517][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 517][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 517][2]=-0.957102074310073;
    Leb_Grid_XYZW[ 517][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 518][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 518][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 518][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 518][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 519][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 519][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 519][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 519][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 520][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 520][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 520][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 520][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 521][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 521][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 521][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 521][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 522][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 522][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 522][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 522][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 523][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 523][1]= 0.957102074310073;
    Leb_Grid_XYZW[ 523][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 523][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 524][0]= 0.082130215819325;
    Leb_Grid_XYZW[ 524][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 524][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 524][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 525][0]=-0.082130215819325;
    Leb_Grid_XYZW[ 525][1]=-0.957102074310073;
    Leb_Grid_XYZW[ 525][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 525][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 526][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 526][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 526][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 526][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 527][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 527][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 527][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 527][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 528][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 528][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 528][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 528][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 529][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 529][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 529][2]= 0.082130215819325;
    Leb_Grid_XYZW[ 529][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 530][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 530][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 530][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 530][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 531][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 531][1]= 0.277867319058624;
    Leb_Grid_XYZW[ 531][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 531][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 532][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 532][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 532][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 532][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 533][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 533][1]=-0.277867319058624;
    Leb_Grid_XYZW[ 533][2]=-0.082130215819325;
    Leb_Grid_XYZW[ 533][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 534][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 534][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 534][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 534][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 535][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 535][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 535][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 535][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 536][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 536][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 536][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 536][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 537][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 537][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 537][2]= 0.277867319058624;
    Leb_Grid_XYZW[ 537][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 538][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 538][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 538][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 538][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 539][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 539][1]= 0.082130215819325;
    Leb_Grid_XYZW[ 539][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 539][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 540][0]= 0.957102074310073;
    Leb_Grid_XYZW[ 540][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 540][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 540][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 541][0]=-0.957102074310073;
    Leb_Grid_XYZW[ 541][1]=-0.082130215819325;
    Leb_Grid_XYZW[ 541][2]=-0.277867319058624;
    Leb_Grid_XYZW[ 541][3]= 0.001555213603397;

    Leb_Grid_XYZW[ 542][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 542][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 542][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 542][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 543][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 543][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 543][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 543][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 544][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 544][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 544][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 544][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 545][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 545][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 545][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 545][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 546][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 546][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 546][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 546][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 547][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 547][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 547][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 547][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 548][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 548][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 548][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 548][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 549][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 549][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 549][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 549][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 550][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 550][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 550][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 550][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 551][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 551][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 551][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 551][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 552][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 552][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 552][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 552][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 553][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 553][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 553][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 553][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 554][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 554][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 554][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 554][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 555][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 555][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 555][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 555][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 556][0]= 0.503356427107512;
    Leb_Grid_XYZW[ 556][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 556][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 556][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 557][0]=-0.503356427107512;
    Leb_Grid_XYZW[ 557][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 557][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 557][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 558][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 558][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 558][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 558][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 559][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 559][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 559][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 559][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 560][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 560][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 560][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 560][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 561][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 561][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 561][2]= 0.859379855890721;
    Leb_Grid_XYZW[ 561][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 562][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 562][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 562][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 562][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 563][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 563][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 563][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 563][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 564][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 564][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 564][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 564][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 565][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 565][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 565][2]=-0.859379855890721;
    Leb_Grid_XYZW[ 565][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 566][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 566][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 566][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 566][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 567][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 567][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 567][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 567][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 568][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 568][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 568][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 568][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 569][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 569][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 569][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 569][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 570][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 570][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 570][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 570][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 571][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 571][1]= 0.859379855890721;
    Leb_Grid_XYZW[ 571][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 571][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 572][0]= 0.089992058420749;
    Leb_Grid_XYZW[ 572][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 572][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 572][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 573][0]=-0.089992058420749;
    Leb_Grid_XYZW[ 573][1]=-0.859379855890721;
    Leb_Grid_XYZW[ 573][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 573][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 574][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 574][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 574][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 574][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 575][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 575][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 575][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 575][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 576][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 576][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 576][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 576][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 577][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 577][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 577][2]= 0.089992058420749;
    Leb_Grid_XYZW[ 577][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 578][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 578][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 578][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 578][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 579][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 579][1]= 0.503356427107512;
    Leb_Grid_XYZW[ 579][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 579][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 580][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 580][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 580][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 580][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 581][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 581][1]=-0.503356427107512;
    Leb_Grid_XYZW[ 581][2]=-0.089992058420749;
    Leb_Grid_XYZW[ 581][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 582][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 582][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 582][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 582][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 583][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 583][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 583][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 583][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 584][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 584][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 584][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 584][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 585][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 585][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 585][2]= 0.503356427107512;
    Leb_Grid_XYZW[ 585][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 586][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 586][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 586][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 586][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 587][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 587][1]= 0.089992058420749;
    Leb_Grid_XYZW[ 587][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 587][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 588][0]= 0.859379855890721;
    Leb_Grid_XYZW[ 588][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 588][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 588][3]= 0.001802239128009;

    Leb_Grid_XYZW[ 589][0]=-0.859379855890721;
    Leb_Grid_XYZW[ 589][1]=-0.089992058420749;
    Leb_Grid_XYZW[ 589][2]=-0.503356427107512;
    Leb_Grid_XYZW[ 589][3]= 0.001802239128009;

  }

}
