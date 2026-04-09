/**********************************************************************
  Fuzzy_Weight.c:

     Fuzzy_Weight.c is a subrutine to calculate the fuzzy weight
     at position (x,y,z).

  Log of Fuzzy_Weight.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"


#define Degree_Smear  5

static double Smear(double mu);
static double Smear_Beck(double x);

double Fuzzy_Weight(int ct_AN, int Mc_AN, int Rn, double x, double y, double z)
{
  int i,j,p_AN,Gp_AN,Gs_AN,s_AN,Rnp,Rns;
  double dx,dy,dz,xp,yp,zp,xs,ys,zs,*Pn;
  double mu,Rij,ri,rj,Denominator,Wn;

  /* allocation of array */
  Pn = (double*)malloc(sizeof(double)*List_YOUSO[8]);

  /* Correction of coordinates to the original cell */
  x = x - atv[Rn][1];
  y = y - atv[Rn][2];
  z = z - atv[Rn][3];

  /* Pn */
  for (p_AN=0; p_AN<=FNAN[ct_AN]; p_AN++){
    Gp_AN = natn[ct_AN][p_AN];
    Rnp = ncn[ct_AN][p_AN];
    xp = Gxyz[Gp_AN][1] + atv[Rnp][1];
    yp = Gxyz[Gp_AN][2] + atv[Rnp][2];
    zp = Gxyz[Gp_AN][3] + atv[Rnp][3];
    dx = x - xp;
    dy = y - yp;
    dz = z - zp;
    ri = dx*dx + dy*dy + dz*dz;
  
    Pn[p_AN] = 1.0;
  
    for (s_AN=1; s_AN<=FNAN[Gp_AN]; s_AN++){

      Gs_AN = natn[Gp_AN][s_AN];
      Rns = ncn[Gp_AN][s_AN];
      xs = Gxyz[Gs_AN][1] + atv[Rns][1] + atv[Rnp][1];
      ys = Gxyz[Gs_AN][2] + atv[Rns][2] + atv[Rnp][2];
      zs = Gxyz[Gs_AN][3] + atv[Rns][3] + atv[Rnp][3];

      dx = xp - xs;
      dy = yp - ys;
      dz = zp - zs;
      Rij = dx*dx + dy*dy + dz*dz;

      dx = x - xs;
      dy = y - ys;
      dz = z - zs;
      rj = dx*dx + dy*dy + dz*dz;
      mu = sqrt(ri/Rij) - sqrt(rj/Rij);
      Pn[p_AN] = Pn[p_AN]*Smear(mu);
    }

  } 

  /* Wn */

  Denominator = 0.0;
  for (p_AN=0; p_AN<=FNAN[ct_AN]; p_AN++){
    Denominator = Denominator + Pn[p_AN];
  }

  if (fabs(Denominator)<1.0e-14)
    Wn = 0.0;
  else 
    Wn = Pn[0]/Denominator;

  /* freeing of array */
  free(Pn);

  return Wn;
}


double Smear(double mu)
{
  int i,j,k;
  double f0,f1,result;

  f0 = Smear_Beck(mu);
  for (i=1; i<=Degree_Smear; i++){
    f1 = Smear_Beck(f0);
    f0 = f1;
  }

  result = 0.50*(1.0 - f0);
  return result;
}

double Smear_Beck(double x)
{
  double result;

  result = 1.50*x - 0.50*x*x*x;
  return result;
}

