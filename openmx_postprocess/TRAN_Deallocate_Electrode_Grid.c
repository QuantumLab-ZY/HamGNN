/**********************************************************************
  TRAN_Deallocate_Electrode_Grid.c:

  TRAN_Deallocate_Electrode_Grid.c is a subroutine to deallocate arrays
  storing data of electrodes.

  Log of TRAN_Deallocate_Electrode_Grid.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tran_variables.h"


void TRAN_Deallocate_Electrode_Grid(int Ngrid2)
{

  int side,spin,n1,n2;

  for (side=0; side<2; side++) {

    for (spin=0; spin<=SpinP_switch_e[side]; spin++) {
      free( ElectrodeDensity_Grid[side][spin] );
    }

    free( ElectrodeDensity_Grid[side] );
    free( ElectrodedVHart_Grid[side] ); 
    free( ElectrodeADensity_Grid[side] );

    for (n1=0; n1<Ngrid1_e[side]; n1++){
      for (n2=0; n2<Ngrid2; n2++){
	free(VHart_Boundary[side][n1][n2]);
      }
      free(VHart_Boundary[side][n1]);
    }
    free(VHart_Boundary[side]);

    for (n1=0; n1<IntNgrid1_e[side]; n1++){
      for (n2=0; n2<Ngrid2; n2++){
	free(dDen_IntBoundary[side][n1][n2]);
      }
      free(dDen_IntBoundary[side][n1]);
    }
    free(dDen_IntBoundary[side]);
  }

}


