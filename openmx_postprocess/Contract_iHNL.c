/**********************************************************************
  Contract_iHNL.c:

     Contract_iHNL.c is a subroutine to contract the iHNL matrix

  Log of Contract_iHNL.c:

     8/Jan/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"

void Contract_iHNL(double *****iHNL, double *****iCntHNL)
{
  int i1,Gc_AN,Mc_AN,h_AN,Mh_AN,Gh_AN;
  int p,q,p0,q0,al,be,Cwan,Hwan;
  double sumiHNL[5],sumS,tmp0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      Mh_AN = F_G2M[Gh_AN];

      for (al=0; al<Spe_Total_CNO[Cwan]; al++){
        for (be=0; be<Spe_Total_CNO[Hwan]; be++){

          for (i1=0; i1<List_YOUSO[5]; i1++) sumiHNL[i1] = 0.0;

          for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
            p0 = Spe_Trans_Orbital[Cwan][al][p];

            for (q=0; q<Spe_Specified_Num[Hwan][be]; q++){
              q0 = Spe_Trans_Orbital[Hwan][be][q];
              tmp0 = CntCoes[Mc_AN][al][p]*CntCoes[Mh_AN][be][q]; 

              for (i1=0; i1<List_YOUSO[5]; i1++){
                sumiHNL[i1] += tmp0*iHNL[i1][Mc_AN][h_AN][p0][q0];
              }
            }
          }

          for (i1=0; i1<List_YOUSO[5]; i1++){
            iCntHNL[i1][Mc_AN][h_AN][al][be] = sumiHNL[i1];
	  }

	}
      }      

    }
  }
}











