/**********************************************************************
  TRAN_Allocate_NC.c:

  TRAN_Allocate_NC.c is a set of subroutines to allocate and deallocate 
  arrays used for the NEGF calculation.

  TRAN_Allocate_Cregion_NC:        called from DFT
  TRAN_Deallocate_Cregion_NC:      called from DFT
  TRAN_Allocate_Lead_Region_NC:    called from DFT
  TRAN_Deallocate_Lead_Region_NC:  called from DFT

  Log of TRAN_Allocate_NC.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "tran_variables.h"


/* (de)allocate memory used in TRAN_Input_std_Atoms */
/*
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

*/
/* (de)allocate memory to calculate green function  */



void TRAN_Allocate_Cregion_NC(MPI_Comm mpi_comm_level1,  
                           int SpinP_switch, 
			   int atomnum,
			   int *WhatSpecies,
			   int *Spe_Total_CNO 
			   /* no explicit output */
			   ) 
{
  int *MP; /* dummy */
  int i;
  int numprocs,myid;

  /* MPI */ 
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  TRAN_Set_MP(0, atomnum, WhatSpecies, Spe_Total_CNO, &NUM_c, &i);
  NUM_c=2*NUM_c;

  if (myid==Host_ID){
    printf("<TRAN_Allocate_Cregion: NUM_c=%d NUM_e=%d %d>\n",NUM_c, NUM_e[0],NUM_e[1]);
  }

  /* allocate */
  SCC_nc = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  SCL_nc = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0]);
  SCR_nc = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1]);
  HCC_nc = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  HCL_nc = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0]);
  HCR_nc = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1]);

}





void TRAN_Deallocate_Cregion_NC(int SpinP_switch)
{
  
  free(HCR_nc);
  free(HCL_nc);
  free(HCC_nc);
  free(SCR_nc);
  free(SCL_nc);
  free(SCC_nc);
}








void TRAN_Allocate_Lead_Region_NC( MPI_Comm mpi_comm_level1 ) 
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
  num[iside]=2*num[iside];

  iside = 1; 
  TRAN_Set_MP(0, atomnum_e[iside], WhatSpecies_e[iside], Spe_Total_CNO_e[iside], &num[iside], &i);
  num[iside]=2*num[iside];

  NUM_e[0] = num[0];
  NUM_e[1] = num[1];

  /* allocate */

  S00_nc_e = (dcomplex**)malloc(sizeof(dcomplex*)*2);
  for (iside=0; iside<=1; iside++){
    n2 = num[iside] + 1;
    S00_nc_e[iside] = (dcomplex*)malloc(sizeof(dcomplex)*n2*n2);
  }

  S01_nc_e = (dcomplex**)malloc(sizeof(dcomplex*)*2);
  for (iside=0; iside<=1; iside++){
    n2 = num[iside] + 1;
    S01_nc_e[iside] = (dcomplex*)malloc(sizeof(dcomplex)*n2*n2);
  }

  H00_nc_e = (dcomplex**)malloc(sizeof(dcomplex*)*2);
  for (iside=0; iside<=1; iside++){
      n2 = num[iside] + 1;
      H00_nc_e[iside] = (dcomplex*)malloc(sizeof(dcomplex)*n2*n2);
  }

  H01_nc_e = (dcomplex**)malloc(sizeof(dcomplex*)*2);
  for (iside=0; iside<=1; iside++){
      n2 = num[iside] + 1;
      H01_nc_e[iside] = (dcomplex*)malloc(sizeof(dcomplex)*n2*n2);
  }
}



void TRAN_Deallocate_Lead_Region_NC()
{
  int iside;

  for (iside=0; iside<=1; iside++){
    free(S00_nc_e[iside]);
  }
  free(S00_nc_e);

  for (iside=0; iside<=1; iside++){
    free(S01_nc_e[iside]);
  }
  free(S01_nc_e);

  for (iside=0; iside<=1; iside++){
    free(H00_nc_e[iside]);
  }
  free(H00_nc_e);

  for (iside=0; iside<=1; iside++){
    free(H01_nc_e[iside]);
  }
  free(H01_nc_e);
}


