#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Tools_BandCalc.h"
#include "EigenValue_Problem.h"



void func_Newton(double *k1, double E1, double *k2, double E2, double *k3, double *E3, double EF, int l, int loopMAX){
  int i,j,k;
  double d0,d1,d2,d3;

  if (loopMAX <1) loopMAX = 1;

  for (i=0; i<loopMAX; i++){
    d0 = (E2*k1[0] - E1*k2[0])/(E2 - E1);
    if ((d0<k1[0] && d0<k2[0]) || (d0>k1[0] && d0>k2[0])) d0 = (k1[0]+k2[0])*0.5;
    d1 = (E2*k1[1] - E1*k2[1])/(E2 - E1);
    if ((d1<k1[1] && d1<k2[1]) || (d1>k1[1] && d1>k2[1])) d1 = (k1[1]+k2[1])*0.5;
    d2 = (E2*k1[2] - E1*k2[2])/(E2 - E1);
    if ((d2<k1[2] && d2<k2[2]) || (d2>k1[2] && d2>k2[2])) d2 = (k1[2]+k2[2])*0.5;

    if(switch_Eigen_Newton==1){ 

      /* Disabled by N. Yamaguchi ***
      d3 = (E2*E2 - E1*E1)/(E2 - E1);
      * ***/

      /* Added by N. Yamaguchi ***/
      d3=E1+E2;
      /* ***/

      if ((d3<E1 && d3<E2) || (d3>E1 && d3>E2)) d3 = (E1+E2)*0.5;
    }else if(switch_Eigen_Newton==0){
      EigenValue_Problem(d0, d1, d2, 0);
      d3 = EIGEN[l];
    }//

    if ((d3-EF)*(E1-EF)<0){
      k2[0] = d0;    k2[1] = d1;    k2[2] = d2;    E2 = d3;
    }else if((d3-EF)*(E2-EF)<0){
      k1[0] = d0;    k1[1] = d1;    k1[2] = d2;    E1 = d3;
    }
    if(fabs(d3-EF)<=1.0e-5){  break;  }
  }//i
  k3[0] = d0;    k3[1] = d1;    k3[2] = d2;    E3[0] = d3;

}

