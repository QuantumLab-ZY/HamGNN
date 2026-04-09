/**********************************************************************
  TRAN_Add_ADensity_Lead.c:

  TRAN_Add_ADensity_Lead.c is a subroutine to correct atomic charge
  density near the boundary region of the extended system
  The super position of atomic charge density from that of electrodes 
  is added to the regions [0:TRAN_grid_bound[0]] and 
  [TRAN_grid_bound[1]:Ngrid1-1].

  Log of TRAN_Add_ADensity_Lead.c:

     24/July/2008  Released by T.Ozaki
     24/Apr/2012   Modified by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "tran_variables.h"
#include "tran_prototypes.h"


void TRAN_Add_ADensity_Lead(
            MPI_Comm comm1,
            int SpinP_switch,
            int Ngrid1,
            int Ngrid2,
            int Ngrid3,
            int My_NumGridB_AB,
            double *ADensity_Grid_B)

#define grid_e_ref(i,j,k)  ( ((i)-l1[0]) *Ngrid2*Ngrid3+(j)*Ngrid3+(k) )

{
  int side,l1[2],N2D,GNs;
  int i,j,k,spin,GN,BN_AB;
  int myid,numprocs;

  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

  if (myid==Host_ID){
    printf("<TRAN_Add_ADensity_Lead>\n");
  }

  /* set N2D and GNs */

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  /***********************************************************
    add contribution to atomic charge density from electrodes

    side=0 -> left lead
    side=1 -> right lead
  ***********************************************************/

  for (side=0; side<=1; side++){

    if (side==0){
      l1[0] = 0;
      l1[1] = TRAN_grid_bound[0]; 
    }
    else{
      l1[0] = TRAN_grid_bound[1];
      l1[1] = Ngrid1-1;
    }

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

      GN = BN_AB + GNs;     
      i = GN/(Ngrid2*Ngrid3);    
      j = (GN - i*Ngrid2*Ngrid3)/Ngrid3;
      k = GN - i*Ngrid2*Ngrid3 - j*Ngrid3; 

      if ( l1[0]<=i && i<=l1[1] ) {

	ADensity_Grid_B[BN_AB] += ElectrodeADensity_Grid[side][ grid_e_ref(i,j,k) ];
      }
    }
  }
}
