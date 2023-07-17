/**********************************************************************
  XC_Ceperly_Alder.c:

     XC_Ceperly_Alder.c is a subroutine to calculate an exchange-
     correlation potential for a given density "den" by the local
     density approximation, which is based on the original works
     by Ceperly and Alder and parametrized by Perdew and Zunger.

  Log of XC_Ceperly_Alder.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h> 
#include "openmx_common.h"

double XC_Ceperly_Alder(double den, int P_switch)
{

  /****************************************************
     P_switch:
      0  \epsilon_XC (XC energy density)  
      1  \mu_XC      (XC potential)  
      2  \epsilon_XC - \mu_XC 
      3  derivative of XC energy density w.r.t. den
  ****************************************************/

  double dum,rs,coe;
  double Ex,Ec,dEx,dEc;
  double tmp0,tmp1; 
  double result;

  /****************************************************
                     Non-relativisic
  ****************************************************/

  if (den<=1.0e-15){
    result = 0.0;
  }
  else{

    coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0); */
    rs = coe*pow(den,-0.3333333333333333333);

    tmp0 = 0.458165293632163/rs;
    Ex = -tmp0;
    dEx = tmp0/rs;

    if (1.0<=rs){
      tmp0 = sqrt(rs);  
      dum = (1.0 + 1.0529*tmp0 + 0.3334*rs);
      tmp1 = 0.1423/dum;
      Ec = -tmp1;
      dEc = tmp1/dum*(0.52645/tmp0 + 0.3334);
    }
    else{
      tmp0 = log(rs);
      Ec = -0.0480 + 0.0311*tmp0 + rs*(0.0020*tmp0 - 0.0116);
      dEc = 0.0311/rs + 0.0020*tmp0 - 0.0096;
    }

    /*
    printf("Ex=%15.12f %15.12f\n",Ex,Ex-0.33333333333333333333*rs*dEx);
    */
  
    if      (P_switch==0)
      result = Ex + Ec;
    else if (P_switch==1)
      result = Ex + Ec - 0.33333333333333333333*rs*(dEx + dEc);
    else if (P_switch==2)
      result = 0.3333333333333333333*rs*(dEx + dEc);
    else if (P_switch==3)
      result = -0.3333333333333333333/(coe*coe*coe)*rs*rs*rs*rs*(dEx + dEc);

  }

  return result;
} 











