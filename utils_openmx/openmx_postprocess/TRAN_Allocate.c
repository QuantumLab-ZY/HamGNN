/**********************************************************************
  TRAN_Allocate.c:

  TRAN_Allocate.c is a set of subroutines to allocate and deallocate 
  arrays used for the NEGF calculation.

  TRAN_Allocate_Atoms:          called from TRAN_Input_Atoms
  TRAN_Deallocate_Atoms:        called from openmx
  TRAN_Allocate_Cregion:        called from DFT
  TRAN_Deallocate_Cregion:      called from DFT
  TRAN_Allocate_Lead_Region:    called from DFT
  TRAN_Deallocate_Lead_Region:  called from DFT

  Log of TRAN_Allocate.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "tran_variables.h"


/* (de)allocate memory used in TRAN_Input_std_Atoms */

void TRAN_Allocate_Atoms(int atomnum)
{
  TRAN_region      = (int*)malloc(sizeof(int)*(atomnum+1));
  TRAN_Original_Id = (int*)malloc(sizeof(int)*(atomnum+1));
}

void TRAN_Deallocate_Atoms( void )
{
  free(TRAN_region);
  free(TRAN_Original_Id);
}


/* (de)allocate memory to calculate green function  */



void TRAN_Allocate_Cregion(MPI_Comm mpi_comm_level1,  
                           int SpinP_switch, 
			   int atomnum,
			   int *WhatSpecies,
			   int *Spe_Total_CNO 
			   /* no explicit output */
			   ) 
{
  int *MP; /* dummy */
  int i,k;
  int numprocs,myid;

  /* MPI */ 
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  TRAN_Set_MP(0, atomnum, WhatSpecies, Spe_Total_CNO, &NUM_c, &i);

  if (myid==Host_ID){
    printf("<TRAN_Allocate_Cregion: NUM_c=%d NUM_e=%d %d>\n",NUM_c, NUM_e[0],NUM_e[1]);
  }

  /* allocate */
  SCC = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  SCL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0]);
  SCR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1]);
  HCC = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));
  HCL = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));
  HCR = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));
  for (k=0; k<=SpinP_switch; k++) {
    HCC[k] = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
    HCL[k] = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0]);
    HCR[k] = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1]);
  }
}





void TRAN_Deallocate_Cregion(int SpinP_switch)
{
  int k;

  for (k=SpinP_switch; k>=0; k--) {
    free(HCR[k]);
    free(HCL[k]);
    free(HCC[k]);
  }
  free(HCR);
  free(HCL);
  free(HCC);
  free(SCR);
  free(SCL);
  free(SCC);
}








void TRAN_Allocate_Lead_Region( MPI_Comm mpi_comm_level1 ) 
{
  int *MP; /* dummy */
  int i,k,n2,iside,num[2];
  int numprocs,myid;

  /* MPI */ 
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){
    printf("<TRAN_Allocate_Lead_Region>\n");
  }

  /* MPI */ 
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  iside = 0; 
  TRAN_Set_MP(0, atomnum_e[iside], WhatSpecies_e[iside], Spe_Total_CNO_e[iside], &num[iside], &i);

  iside = 1; 
  TRAN_Set_MP(0, atomnum_e[iside], WhatSpecies_e[iside], Spe_Total_CNO_e[iside], &num[iside], &i);

  NUM_e[0] = num[0];
  NUM_e[1] = num[1];

  /* allocate */

  S00_e = (dcomplex**)malloc(sizeof(dcomplex*)*2);
  for (iside=0; iside<=1; iside++){
    n2 = num[iside] + 1;
    S00_e[iside] = (dcomplex*)malloc(sizeof(dcomplex)*n2*n2);
  }

  S01_e = (dcomplex**)malloc(sizeof(dcomplex*)*2);
  for (iside=0; iside<=1; iside++){
    n2 = num[iside] + 1;
    S01_e[iside] = (dcomplex*)malloc(sizeof(dcomplex)*n2*n2);
  }

  H00_e = (dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (iside=0; iside<=1; iside++){
    H00_e[iside] = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch_e[iside]+1));
    for (k=0; k<=SpinP_switch_e[iside]; k++) {
      n2 = num[iside] + 1;
      H00_e[iside][k] = (dcomplex*)malloc(sizeof(dcomplex)*n2*n2);
    }
  }

  H01_e = (dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (iside=0; iside<=1; iside++){
    H01_e[iside] = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch_e[iside]+1));
    for (k=0; k<=SpinP_switch_e[iside]; k++) {
      n2 = num[iside] + 1;
      H01_e[iside][k] = (dcomplex*)malloc(sizeof(dcomplex)*n2*n2);
    }
  }
}



void TRAN_Deallocate_Lead_Region()
{
  int k,iside;

  for (iside=0; iside<=1; iside++){
    free(S00_e[iside]);
  }
  free(S00_e);

  for (iside=0; iside<=1; iside++){
    free(S01_e[iside]);
  }
  free(S01_e);

  for (iside=0; iside<=1; iside++){
    for (k=0;k<=SpinP_switch_e[iside]; k++) {
      free(H00_e[iside][k]);
    }
    free(H00_e[iside]);
  }
  free(H00_e);

  for (iside=0; iside<=1; iside++){
    for (k=0; k<=SpinP_switch_e[iside]; k++) {
      free(H01_e[iside][k]);
    }
    free(H01_e[iside]);
  }
  free(H01_e);
}


