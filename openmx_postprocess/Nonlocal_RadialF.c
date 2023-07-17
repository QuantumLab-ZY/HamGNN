#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

double Nonlocal_RadialF(int spe, int l, int so, double R)
{
  int mp_min,mp_max,m,po;
  double h1,h2,h3,f1,f2,f3,f4;
  double g1,g2,x1,x2,y1,y2,f;
  double result;

  mp_min = 0;
  mp_max = Spe_Num_Mesh_VPS[spe] - 1;
  po = 0;

  if (R<Spe_VPS_RV[spe][0]){

    if (Spe_VPS_List[spe][l+1]==0){
      m = 1;
    }
    else{
      result = 0.0;
      po = 1;
    }

  }
  else if (Spe_VPS_RV[spe][Spe_Num_Mesh_VPS[spe]-1]<R){
    result = 0.0;
    po = 1;
  }
  else{
    do{
      m = (mp_min + mp_max)/2;
      if (Spe_VPS_RV[spe][m]<R)
        mp_min = m;
      else 
        mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;
  }

  /****************************************************
                 Spline like interpolation
  ****************************************************/

  if (po==0){

    if (m==1){
      h2 = Spe_VPS_RV[spe][m]   - Spe_VPS_RV[spe][m-1];
      h3 = Spe_VPS_RV[spe][m+1] - Spe_VPS_RV[spe][m];

      f2 = Spe_VNL[so][spe][l][m-1];
      f3 = Spe_VNL[so][spe][l][m];
      f4 = Spe_VNL[so][spe][l][m+1];

      h1 = -(h2+h3);
      f1 = f4;
    }
    else if (m==(Spe_Num_Mesh_VPS[spe]-1)){
      h1 = Spe_VPS_RV[spe][m-1] - Spe_VPS_RV[spe][m-2];
      h2 = Spe_VPS_RV[spe][m]   - Spe_VPS_RV[spe][m-1];

      f1 = Spe_VNL[so][spe][l][m-2];
      f2 = Spe_VNL[so][spe][l][m-1];
      f3 = Spe_VNL[so][spe][l][m];

      h3 = -(h1+h2);
      f4 = f1;
    }
    else{
      h1 = Spe_VPS_RV[spe][m-1] - Spe_VPS_RV[spe][m-2];
      h2 = Spe_VPS_RV[spe][m]   - Spe_VPS_RV[spe][m-1];
      h3 = Spe_VPS_RV[spe][m+1] - Spe_VPS_RV[spe][m];

      f1 = Spe_VNL[so][spe][l][m-2];
      f2 = Spe_VNL[so][spe][l][m-1];
      f3 = Spe_VNL[so][spe][l][m];
      f4 = Spe_VNL[so][spe][l][m+1];
    } 

    /****************************************************
                Calculate the value at R
    ****************************************************/

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = R - Spe_VPS_RV[spe][m-1];
    x2 = R - Spe_VPS_RV[spe][m];
    y1 = x1/h2;
    y2 = x2/h2;

    f =  y2*y2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y1*y1*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    result = f;

  }

  return result;

}




