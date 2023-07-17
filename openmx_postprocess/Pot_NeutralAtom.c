/**********************************************************************
  Pot_NeutralAtom.c:

     Pot_NeutralAtom.c is a subroutine to calculate the neutral atom
     potential at (Gx,Gy,Gz).

  Log of Pot_NeutralAtom.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

double Pot_NeutralAtom(int ct_AN, double Gx, double Gy, double Gz)
{
  int h_AN,Gh_AN,wan,Rn;
  double sum,x,y,z,dx,dy,dz,r;

  sum = 0.0;
  for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
    Gh_AN = natn[ct_AN][h_AN];
    Rn = ncn[ct_AN][h_AN];
    wan = WhatSpecies[Gh_AN];

    x = Gxyz[Gh_AN][1] + atv[Rn][1];
    y = Gxyz[Gh_AN][2] + atv[Rn][2];
    z = Gxyz[Gh_AN][3] + atv[Rn][3];
    
    dx = Gx - x;
    dy = Gy - y;
    dz = Gz - z;
    r = sqrt(dx*dx + dy*dy + dz*dz);  

    sum = sum + VNAF(wan,r);
  }

  return sum;
}

