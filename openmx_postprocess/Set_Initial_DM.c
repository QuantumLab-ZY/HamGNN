/**********************************************************************
  Set_Initial_DM.c:

     Set_Initial_DM.c is a subroutine to set an initial density matrix.

  Log of Set_Initial_DM.c:

     7/Feb./2012  Released by T.Ozaki

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


double Set_Initial_DM(double *****CDM, double *****H)
{
  int i,j,spin,spinmax,po,loopN;
  int Mc_AN,Gc_AN,h_AN,Gh_AN,Cwan,Hwan;
  double cp,cp_min,cp_max,ns,tns,dn,ff,x,TZ;
  double spin_degeneracy;
  double TStime,TEtime,time0;
  int myid,numprocs;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* measure TStime */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* initialize CDM */

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        for (i=0; i<Spe_Total_NO[Cwan]; i++){
          for (j=0; j<Spe_Total_NO[Hwan]; j++){
            CDM[spin][Mc_AN][h_AN][i][j] = 0.0;
          }
        }
      }
    }
  }

  /* set diagonal terms of CDM */

  if      (SpinP_switch==0){
    spinmax = 0;
    spin_degeneracy = 2.0;
  }
  else if (SpinP_switch==1){
    spinmax = 1;
    spin_degeneracy = 1.0;
  }
  else if (SpinP_switch==3){
    spinmax = 1;
    spin_degeneracy = 1.0;
  }

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    Cwan = WhatSpecies[i];
    TZ = TZ + Spe_Core_Charge[Cwan];
  }

  /* CDM[0] and CDM[1] */

  cp_max =  30.0;
  cp_min = -30.0;
  loopN = 0;
  po = 0;

  do {

    cp = 0.50*(cp_max + cp_min);
    ns = 0.0;

    for (spin=0; spin<=spinmax; spin++){ 
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];

	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  x = (H[spin][Mc_AN][0][i][i] - cp)*Beta;
	  ff = 1.0/(1.0 + exp(x));
	  CDM[spin][Mc_AN][0][i][i] = ff; 
	  ns += spin_degeneracy*ff;
	}

	/*
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  printf("Gc_AN=%2d i=%2d CDM=%15.12f\n",Gc_AN,i,CDM[spin][Mc_AN][0][i][i]);
	} 
	*/     

      } /* Mc_AN */
    } /* spin */

    MPI_Allreduce(&ns, &tns, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    dn = TZ - tns - system_charge;  

    if (0.0<=dn) cp_min = cp;
    else         cp_max = cp;
 
    if (fabs(dn)<1.0e-12) po = 1;

    printf("loopN=%3d cp=%15.12f TZ=%15.12f dn=%15.12f\n",loopN,cp,TZ,dn);

    loopN++;

  } while (po==0 && loopN<1000);

  /* CDM[2] */
  if (SpinP_switch==3){
  }

  /* calculate time0 */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}
