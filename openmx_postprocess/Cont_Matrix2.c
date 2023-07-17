/**********************************************************************
  Cont_Matrix2.c:

     Cont_Matrix2.c is a subroutine to contract a Matrix2 (HVNA).

  Log of Cont_Matrix2.c:

     12/Apr./2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"

void Cont_Matrix2(Type_DS_VNA ****Mat, Type_DS_VNA ****CMat)
{
  int i,j;
  int spin,Mc_AN,Gc_AN,h_AN,Gh_AN,L,L1,M1;
  int p,q,p0,q0,al,be,Cwan,Hwan;
  double sumS,tmp0;
  int *VNA_List;
  int *VNA_List2;
  int Num_RVNA;

  Num_RVNA = List_YOUSO[34]*(List_YOUSO[35] + 1);

  VNA_List  = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) ); 
  VNA_List2 = (int*)malloc(sizeof(int)*(List_YOUSO[34]*(List_YOUSO[35] + 1)+2) ); 
  
  L = 0;
  for (i=0; i<=List_YOUSO[35]; i++){   /* max L                  */
    for (j=0; j<List_YOUSO[34]; j++){  /* # of radial projectors */
      VNA_List[L]  = i;
      VNA_List2[L] = j;
      L++;
    }
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];

      for (al=0; al<Spe_Total_CNO[Cwan]; al++){
	be = -1;
	for (L=0; L<Num_RVNA; L++){
          L1 = VNA_List[L];
	  for (M1=-L1; M1<=L1; M1++){
	    be++;
	    sumS = 0.0;

	    for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	      p0 = Spe_Trans_Orbital[Cwan][al][p];
	      sumS += CntCoes[Mc_AN][al][p]*Mat[Mc_AN][h_AN][p0][be];
	    }

	    CMat[Mc_AN][h_AN][al][be] = sumS;

	  }
	}
      }
    }
  }

  free(VNA_List);
  free(VNA_List2);
}
