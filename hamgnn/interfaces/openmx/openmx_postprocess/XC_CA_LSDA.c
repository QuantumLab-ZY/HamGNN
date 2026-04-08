/**********************************************************************
  XC_CA_LSDA.c:

     XC_CA_LSDA is a subroutine to calculate an exchange-
     correlation potential for a given density "den" by the local
     spin density approximation, which is based on the original works
     by Ceperly and Alder and parametrized by Perdew and Zunger.

  Log of XC_CA_LSDA.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h> 
#include "openmx_common.h"


/* input argument "SCF_iter" was added by S.Ryee for LDA+U */
void XC_CA_LSDA(int SCF_iter, double den0, double den1, double XC[2],int P_switch)
{

  /****************************************************
          P_switch:
              0  \epsilon_XC (Exc, XC energy density)  
              1  \mu_XC      (Vxc, XC potential)  
              2  \epsilon_XC - \mu_XC (Exc-Vxc) 
  ****************************************************/

  double dum,rs,coe,tden,min_den;
  double Vxc,Ex,Ec,dEx,dEc,Exc,dExc;
  double ExP,EcP,dExP,dEcP;
  double ExF,EcF,dExF,dEcF;
  double zeta,fzeta,dfzeta;
  double tmp0,tmp1,tmp2;
  double z0,z1,z02,z04,z12,z14; 

  /****************************************************
              Non-relativisic formalism
  ****************************************************/

  /****************************************************
     total density (tden) = den0 + den1
     zeta = (den0-den1)/tden
     rs = pow(3.0/4.0/PI,1.0/3.0)*tden^{-1/3}
  *****************************************************/
 
  min_den = 1.0e-15;
  tden = den0 + den1;
  if (tden<min_den){
    XC[0] = 0.0;
    XC[1] = 0.0;
  }
  
  else{
    zeta = (den0 - den1)/tden;  
    if (1.0<zeta)  zeta =  1.0 - min_den;
    if (zeta<-1.0) zeta = -1.0 + min_den;

    if((dc_Type==3 || dc_Type==4) && SCF_iter>1){  /* for LDA+U with cFLL by S.Ryee */
      zeta = 0.0;
    }

    coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0); */
    if (tden<min_den)
      rs = 62035.04908994;  /* coe*pow(1.0e-15,-1.0/3.0); */
    else
      rs = coe*pow(tden,-0.3333333333333333333);

    /*****************************************************
     exchange energy density for the para magnetic state
     ExP = -3/4*(3/pi)^(1/3)*tden^{1/3}
         = -3/4*(9/4/pi/pi)^(1/3)/rs
         = -0.458165293632163/rs 
    *****************************************************/

    tmp0 = 0.458165293632163/rs;
    ExP  = -tmp0;
    dExP = tmp0/rs;

    /*****************************************************
     exchange energy density for the ferro magnetic state
     ExF = 2^(1/3)*ExP
         = 1.25992104989487*ExP
    *****************************************************/
  
    ExF  = 1.25992104989487*ExP; 
    dExF = 1.25992104989487*dExP;

    /*****************************************************
     correlation energy density for the para magnetic state
     1<=rs
     EcP = -0.1423/(1+1.0529*sqrt(rs) + 0.3334*rs)
     0<=rs<=1
     EcP = 0.0311*ln(rs)-0.048+0.0020*rs*ln(rs)-0.0116*rs 

     correlation energy density for the ferro magnetic state
     1<=rs
     EcF = -0.0843/(1+1.3981*sqrt(rs) + 0.2611*rs)
     0<=rs<=1
     EcF = 0.01555*ln(rs)-0.0269+0.0007*rs*ln(rs)-0.0048*rs 
    *****************************************************/

    if (1.0<=rs){
      tmp0 = sqrt(rs);  

      dum = (1.0 + 1.0529*tmp0 + 0.3334*rs);
      tmp1 = 0.1423/dum;
      EcP = -tmp1;
      dEcP = tmp1/dum*(0.52645/tmp0 + 0.3334);

      dum = (1.0 + 1.3981*tmp0 + 0.2611*rs);
      tmp1 = 0.0843/dum;
      EcF = -tmp1;
      dEcF = tmp1/dum*(0.69905/tmp0 + 0.2611);
    }
    else{
      tmp0 = log(rs);

      EcP = -0.0480 + 0.0311*tmp0 + rs*(0.0020*tmp0 - 0.0116);
      dEcP = 0.0311/rs + 0.0020*tmp0 - 0.0096;

      EcF = -0.0269 + 0.01555*tmp0 + rs*(0.0007*tmp0 - 0.0048);
      dEcF = 0.01555/rs + 0.0007*tmp0 - 0.0041;
    }

    if (P_switch==0){

      /*****************************************************
       z0 = (1 + zeta)^{4/3}
       z1 = (1 - zeta)^{4/3}
       fzeta = (z0 + z1 - 2)/(2*(2^{1/3}-1))
             = 1.92366105093154*(z0 + z1 - 2)
      *****************************************************/

      z0 = pow(1.0+zeta,1.33333333333333333);
      z1 = pow(1.0-zeta,1.33333333333333333);
      fzeta = 1.92366105093154*(z0 + z1 - 2.0);

      /*****************************************************
        exchange-correration energy density       
        Ex = ExP + (ExF - ExP)*fzeta
        Ec = EcP + (EcF - EcP)*fzeta
        Exc = Ex + Ec
            = ExP + EcP + (ExF + EcF - ExP - EcP)*fzeta       
      *****************************************************/

      Exc = ExP + EcP + (ExF + EcF - ExP - EcP)*fzeta;

      /*
      printf("ABC %15.12f\n",ExP + (ExF - ExP)*fzeta);
      */

      XC[0] = Exc;
      XC[1] = Exc;
    }
    else if (P_switch==1){

      /*****************************************************
       z0 = (1 + zeta)^{1/3}
       z1 = (1 - zeta)^{1/3}
       fzeta = (z0^4 + z1^4 - 2)/(2*(2^{1/3}-1))
             = 1.92366105093154*(z0^4 + z1^4 - 2)
       dfzeta = 2.56488140124205*(z0 - z1)
      *****************************************************/

      z0 = pow(1.0+zeta,0.33333333333333333);
      z1 = pow(1.0-zeta,0.33333333333333333);

      z02 = z0*z0;
      z04 = z02*z02;
      z12 = z1*z1;
      z14 = z12*z12;
      fzeta  = 1.92366105093154*(z04 + z14 - 2.0);
      dfzeta = 2.56488140124205*(z0 - z1);

      /*****************************************************
        exchange-correration potential
        Vxc+- = Exc + tden*dExc/drho +- dExc/dzeta*(1-+zeta) 

        Ex = ExP + (ExF - ExP)*fzeta
        Ec = EcP + (EcF - EcP)*fzeta
        Exc = Ex + Ec
            = ExP + EcP + (ExF + EcF - ExP - EcP)*fzeta       

        dEx  = dExP + (dExF - dExP)*fzeta
        dEc  = dEcP + (dEcF - dEcP)*fzeta
        dExc = dEx + dEc
             = dExP + dEcP + (dExF + dEcF - dExP - dEcP)*fzeta       

        tden*dExc/drho = -1/3*rs*dExc/drs
      *****************************************************/

      tmp0 = ExF + EcF - ExP - EcP;
      Exc = ExP + EcP + tmp0*fzeta;
      dExc = dExP + dEcP + (dExF + dEcF - dExP - dEcP)*fzeta;
      Vxc = Exc - 0.33333333333333333333*rs*dExc;
 
      XC[0] = Vxc + tmp0*( 1.0 - zeta)*dfzeta;
      XC[1] = Vxc + tmp0*(-1.0 - zeta)*dfzeta;


    }
    else if (P_switch==2){

      /*****************************************************
       z0 = (1 + zeta)^{1/3}
       z1 = (1 - zeta)^{1/3}
       fzeta = (z0^4 + z1^4 - 2)/(2*(2^{1/3}-1))
             = 1.92366105093154*(z0^4 + z1^4 - 2)
       dfzeta = 2.56488140124205*(z0 - z1)
      *****************************************************/

      z0 = pow(1.0+zeta,0.33333333333333333);
      z1 = pow(1.0-zeta,0.33333333333333333);
      z02 = z0*z0;
      z04 = z02*z02;
      z12 = z1*z1;
      z14 = z12*z12;
      fzeta  = 1.92366105093154*(z04 + z14 - 2.0);
      dfzeta = 2.56488140124205*(z0 - z1);

      /*****************************************************
        Exc - Vxc 
      *****************************************************/

      tmp0 = ExF + EcF - ExP - EcP;
      Exc = ExP + EcP + tmp0*fzeta;
      dExc = dExP + dEcP + (dExF + dEcF - dExP - dEcP)*fzeta;
      Vxc = Exc - 0.33333333333333333333*rs*dExc;
      tmp1 = 0.33333333333333333333*rs*dExc; 
      tmp2 = tmp0*dfzeta;
      XC[0] = tmp1 - tmp2*( 1.0 - zeta);
      XC[1] = tmp1 - tmp2*(-1.0 - zeta);
    }
  }

} 

