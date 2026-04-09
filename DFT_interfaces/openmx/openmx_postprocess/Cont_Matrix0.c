/**********************************************************************
  Cont_Matrix0.c:

     Cont_Matrix0.c is a subroutine to contract a Matrix0 (eg. H)

  Log of Cont_Matrix0.c:

     8/Jan/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"

void Cont_Matrix0(double ****Mat, double ****CMat)
{
  int spin,Mc_AN,Gc_AN,h_AN,Gh_AN,Mh_AN,L,L1,M1;
  int p,q,p0,q0,al,be,Cwan,Hwan;
  double sumS,tmp0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      Mh_AN = F_G2M[Gh_AN];

      for (al=0; al<Spe_Total_CNO[Cwan]; al++){
	for (be=0; be<Spe_Total_CNO[Hwan]; be++){

	  sumS = 0.0;
	  for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	    p0 = Spe_Trans_Orbital[Cwan][al][p];
	    for (q=0; q<Spe_Specified_Num[Hwan][be]; q++){
	      q0 = Spe_Trans_Orbital[Hwan][be][q];
	      tmp0 = CntCoes[Mc_AN][al][p]*CntCoes[Mh_AN][be][q]; 
	      sumS += tmp0*Mat[Mc_AN][h_AN][p0][q0];
	    }
	  }
	  CMat[Mc_AN][h_AN][al][be] = sumS;
	}
      }
    }
  }

}
