/**********************************************************************
  TRAN_Check_Inputs.c:

  TRAN_Check_Input.c is a subroutine to check th input data.

  Log of TRAN_Check_Input.c:

     06/Oct./2008  Released by H.Kino and T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Inputtools.h"
#include "openmx_common.h"
#include <mpi.h>
#include "tran_prototypes.h"
#include "tran_variables.h"

void TRAN_Check_Input(  MPI_Comm comm1, int Solver )
{
  int po=0;
  FILE *fp;
  char *s_vec[20];
  int i_vec[20];
  double r_vec[20];

  int myid,i,j,spe,spe_e; 
  char Species[YOUSO10];
  double Length_C, Length_L, Length_R;
  double angleCL, angleCR;
  double Lsign, Rsign; 

  if (Solver!=4) return; 

  MPI_Comm_rank(comm1,&myid);

  /* left */

  for (i=1; i<=Latomnum; i++){

    j = TRAN_Original_Id[i];
    spe   = WhatSpecies[i];
    spe_e = WhatSpecies_e[0][j];

    if (Spe_Total_NO_e[0][spe_e]!=Spe_Total_NO[spe]){

      if (myid==Host_ID){
        printf("The specification of species in the LEFT lead is inconsistent.\n");
        printf("Probably the basis set is different from that used in the band calculation.\n");
      }

      po++;
    }
  }

  /* right */

  for (i=1; i<=Ratomnum; i++){

    j = TRAN_Original_Id[Catomnum+Latomnum+i];
    spe   = WhatSpecies[Catomnum+Latomnum+i];
    spe_e = WhatSpecies_e[1][j];

    if (Spe_Total_NO_e[1][spe_e]!=Spe_Total_NO[spe]){

      if (myid==Host_ID){
        printf("The specification of species in the RIGHT lead is inconsistent.\n");
        printf("Probably the basis set is different from that used in the band calculation.\n");
      }

      po++;
    }
  }

  /* check po */

  if (po!=0){
    MPI_Finalize();
    exit(0);
  } 
}
