/**********************************************************************
  TRAN_Check_Region_Lead.c:

  TRAN_Check_Region_Lead.c is a subroutine to check whether the lead
  is set up properly or not.

   output: none
   return value:  1 = OK
                  0 = NG

   purpose:
   to confirm that ...|L0|L1|L2|L3|... has overlapping PAO only between L1 and L2.

  Log of TRAN_Check_Region_Lead.o

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "tran_variables.h"


int TRAN_Check_Region_Lead(int atomnum,
			   int *WhatSpecies, 
			   double *Spe_Atom_Cut1,
			   double **Gxyz,
			   double tv[4][4])
{
  int ct_AN;
  int wanA;
  double rcutA;
  int ct_BN;
  int wanB;
  double rcutB;
  double rcutAB;
  double len,dx,dy,dz;
  int i;

  /* this routine only in the case of TRAN_output_hks!=0 */

  if ( TRAN_output_hks==0 && TRAN_output_TranMain==0 ) {
    return 1;
  }

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++) {

    wanA = WhatSpecies[ct_AN];
    rcutA = Spe_Atom_Cut1[wanA];

    for (ct_BN=1; ct_BN<=atomnum; ct_BN++) {

      wanB = WhatSpecies[ct_BN];
      rcutB = Spe_Atom_Cut1[wanB];
      rcutAB = rcutA + rcutB;

      dx = Gxyz[ct_AN][1] - (Gxyz[ct_BN][1] + 2.0*tv[1][1]);
      dy = Gxyz[ct_AN][2] - (Gxyz[ct_BN][2] + 2.0*tv[1][2]);
      dz = Gxyz[ct_AN][3] - (Gxyz[ct_BN][3] + 2.0*tv[1][3]);

      len = sqrt( dx*dx + dy*dy + dz*dz );

      if ( len <= rcutAB ) { 
	printf("\n\nTRAN_Check_Region_Lead()\n");
	printf("\nThe length between atomA=%d and atomB=%d is too short for the transport calculation.\n",
               ct_AN, ct_BN); 
	printf("distance=%lf rcutA=%lf rcutB=%lf\n",len,rcutA,rcutB);
	return 0; 
      }

    } /* ct_BN */
  } /* ct_AN */

  return 1;
}



