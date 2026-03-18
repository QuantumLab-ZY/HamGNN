/**********************************************************************
  RF_BesselF.c:

     RF_BesselF.c is a subroutine to calculate radial part of PAO of 
     atom specified by "Gensi" in k-space.

  Log of RF_BesselF.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

double RF_BesselF(int Gensi, int GL, int Mul, double R)
{
  int mp_min,mp_max,m,po;
  double h1,h2,h3,f1,f2,f3,f4;
  double g1,g2,x1,x2,y1,y2,f;
  double result;

  mp_min = 0;
  mp_max = Ngrid_NormK - 1;
  po = 0;

  if (R<NormK[0]){
    m = 1;
  }
  else if (NormK[mp_max]<R){
    result = 0.0;
    po = 1;
  }
  else{
    do{
      m = (mp_min + mp_max)/2;
      if (NormK[m]<R)
        mp_min = m;
      else 
        mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;

    if (m<2)
      m = 2;
    else if (Ngrid_NormK<=m)
      m = Ngrid_NormK - 2;
  }

  /****************************************************
                 Spline like interpolation
  ****************************************************/

  if (po==0){

    if (m==1){
      h2 = NormK[m]   - NormK[m-1];
      h3 = NormK[m+1] - NormK[m];

      f2 = Spe_RF_Bessel[Gensi][GL][Mul][m-1];
      f3 = Spe_RF_Bessel[Gensi][GL][Mul][m];
      f4 = Spe_RF_Bessel[Gensi][GL][Mul][m+1];

      h1 = -(h2+h3);
      f1 = f4;
    }
    else if (m==(Ngrid_NormK-1)){
      h1 = NormK[m-1] - NormK[m-2];
      h2 = NormK[m]   - NormK[m-1];

      f1 = Spe_RF_Bessel[Gensi][GL][Mul][m-2];
      f2 = Spe_RF_Bessel[Gensi][GL][Mul][m-1];
      f3 = Spe_RF_Bessel[Gensi][GL][Mul][m];

      h3 = -(h1+h2);
      f4 = f1;
    }
    else{
      h1 = NormK[m-1] - NormK[m-2];
      h2 = NormK[m]   - NormK[m-1];
      h3 = NormK[m+1] - NormK[m];

      f1 = Spe_RF_Bessel[Gensi][GL][Mul][m-2];
      f2 = Spe_RF_Bessel[Gensi][GL][Mul][m-1];
      f3 = Spe_RF_Bessel[Gensi][GL][Mul][m];
      f4 = Spe_RF_Bessel[Gensi][GL][Mul][m+1];
    }

    /****************************************************
                Calculate the value at R
    ****************************************************/

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = R - NormK[m-1];
    x2 = R - NormK[m];
    y1 = x1/h2;
    y2 = x2/h2;

    f =  y2*y2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y1*y1*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    result = f;
  }
  
  return result;
}
