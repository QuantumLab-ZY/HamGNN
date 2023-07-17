/**********************************************************************
  XC_PW92C.c:

     XC_PW92C.c is a subroutine to calculate the correlation
     energy density and potential developed by Perdew and Wang 
     (Ref: J.P.Perdew and Y.Wang, PRB, 45, 13244 (1992)) 
     for given up (dens[0]) and down (dens[1]) densities.

     This routine was written by T.Ozaki, based on the original fortran 
     code provided by the SIESTA group through their website.
     Thanks to them.

  Log of PW92C.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "openmx_common.h"

#define den_min       1.0e-14
#define den_min_half  0.5*1.0e-14

/* input argument "SCF_iter" was added by S.Ryee for LDA+U */
void XC_PW92C(int SCF_iter, double dens[2], double Ec[1], double Vc[2])
{
  int i;
  double dtot,rs,srs,zeta,coe;
  double dum,dum1,dum2,b,c,dbdrs,dcdrs;
  double tmp0,tmp1,tmp2,tmp12,tmp14,tmp22,tmp24;
  double fpp0,f,dfdz;
  double G[3],dGdrs[3];
  double dEcdd[2];
  double dEcdrs,dEcdz;
  double drsdd,dzdd[2];

  /****************************************************
               parameters from Table I of
            Perdew and Wang, PRB, 45, 13244 (92)
  ****************************************************/

  double p[3]      = {1.0000000,  1.0000000,  1.0000000};
  double A[3]      = {0.0310910,  0.0155450,  0.0168870};
  double alpha1[3] = {0.2137000,  0.2054800,  0.1112500};
  double beta1[3]  = {7.5957000, 14.1189000, 10.3570000};
  double beta2[3]  = {3.5876000,  6.1977000,  3.6231000};
  double beta3[3]  = {1.6382000,  3.3662000,  0.8802600};
  double beta4[3]  = {0.4929400,  0.6251700,  0.4967100};

  /****************************************************
                      zeta and rs
  ****************************************************/

  coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0);   */
  dtot = dens[0] + dens[1];
     
  if (dtot<=den_min){
    rs = 6203.504908994;  /* coe*pow(1.0e-12,-1.0/3.0); */
    dens[0] = den_min_half;
    dens[1] = den_min_half;
    dtot = den_min;
  }
  else
    rs = coe*pow(dtot,-0.33333333333333333333333);

  tmp0 = 1.0/dtot;  
  zeta = tmp0*(dens[0] - dens[1]);

  if (0.99<zeta) zeta =  0.99;
  if (zeta<-1.0) zeta = -0.99;

  if((dc_Type==3 || dc_Type==4) && SCF_iter>1){  /* for LDA+U with cFLL by S.Ryee */
    zeta=0.0;
  }

  drsdd = -0.3333333333333333333333*rs*tmp0;
  dzdd[0] = tmp0*( 1.0 - zeta);
  dzdd[1] = tmp0*(-1.0 - zeta);

  /****************************************************
                    eps_c(rs,0)=G(0)
                    eps_c(rs,1)=G(1)
                   -alpha_c(rs)=G(2)
                    using eq.(10) in 
          Perdew and Wang, PRB, 45, 13244 (1992))
  ****************************************************/

  srs = sqrt(rs);

  for (i=0; i<=2; i++){
    b = beta1[i]*srs + rs*(beta2[i] + beta3[i]*srs + beta4[i]*rs);

    dbdrs =  beta1[i]*0.50/srs
           + beta2[i]
           + beta3[i]*1.50*srs
           + beta4[i]*2.0*rs;

    c = 1.0 + 1.0/(2.0*A[i]*b);
    dcdrs = -(c - 1.0)*dbdrs/b;
    dum = log(c);
    dum1 = 1.0 + alpha1[i]*rs;
    G[i] = -2.0*A[i]*dum1*dum;
    dGdrs[i] = -2.0*A[i]*(alpha1[i]*dum + dum1*dcdrs/c);
  }

  /****************************************************
            f''(0) and f(zeta) from eq.(9)
  ****************************************************/

  c = 1.92366105093154;       /* 1/(2*(2^{1/3}-1)) */
  fpp0 = 1.70992093416137;
  dum1 = 1.0 + zeta;
  dum2 = 1.0 - zeta;

  tmp1  = pow(dum1,0.333333333333333333);
  tmp2  = pow(dum2,0.333333333333333333);

  tmp12 = tmp1*tmp1;
  tmp22 = tmp2*tmp2;
  tmp14 = tmp12*tmp12;
  tmp24 = tmp22*tmp22;

  f = (tmp14 + tmp24 - 2.0)*c;
  dfdz = 1.333333333333333333*(tmp1 - tmp2)*c;

  /****************************************************
               eps_c(rs,zeta) from eq.(8)
  ****************************************************/

  dum1 = zeta*zeta*zeta;
  dum  = dum1*zeta;

  Ec[0] = G[0] - G[2]*f/fpp0*(1.0 - dum) + (G[1] - G[0])*f*dum;

  dEcdrs =   dGdrs[0] - dGdrs[2]*f/fpp0*(1.0 - dum)
          + (dGdrs[1] - dGdrs[0])*f*dum;

  dEcdz = - G[2]/fpp0*(dfdz*(1.0 - dum) - f*4.0*dum1)
          + (G[1] - G[0])*(dfdz*dum + f*4.0*dum1);

  /*
  printf("rs     = %18.15f\n",rs);
  printf("zeta   = %18.15f\n",zeta);
  printf("Ec     = %18.15f\n",Ec[0]);
  printf("Ec2    = %18.15f  %18.15f\n",
         -G[2]*f/fpp0*(1.0 - dum),-dGdrs[2]*f/fpp0*(1.0 - dum));
  printf("Ec3    = %18.15f %18.15f\n",
         (G[1] - G[0])*f*dum,(dGdrs[1] - dGdrs[0])*f*dum);

  printf("G0 dGdrs0    = %18.15f %18.15f\n",G[0],dGdrs[0]);
  printf("G1 dGdrs1    = %18.15f %18.15f\n",G[1],dGdrs[1]);
  printf("G2 dGdrs2    = %18.15f %18.15f\n",G[2],dGdrs[2]);

  printf("dEcdrs = %18.15f\n",dEcdrs);
  printf("dEcdz  = %18.15f\n",dEcdz);
  */

  /****************************************************
                Find correlation potential
  ****************************************************/

  dum = dEcdrs*drsdd;
  dEcdd[0] = dum + dEcdz*dzdd[0];
  dEcdd[1] = dum + dEcdz*dzdd[1];
  Vc[0] = Ec[0] + dtot*dEcdd[0];
  Vc[1] = Ec[0] + dtot*dEcdd[1];

} 
