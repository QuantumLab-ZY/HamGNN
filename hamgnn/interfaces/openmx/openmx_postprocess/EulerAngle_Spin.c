/**********************************************************************
  EulerAngle_Spin.c:

     EulerAngle_Spin.c is a subroutine to find the Euler angle of spin
     orientation from the density matrix.

  Log of EulerAngle_Spin.c:

     15/Feb/2006  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "openmx_common.h"


void EulerAngle_Spin( int quickcalc_flag,
                      double Re11, double Re22,
                      double Re12, double Im12,
                      double Re21, double Im21,
                      double Nup[2], double Ndown[2],
                      double t[2], double p[2] )
{
  double phi,theta,d1,d2,d3,d4;
  double cop,sip,sit,cot,tmp,tmp1,tmp2,prod;
  double mx,my,mz,tn,absm;
  double S_coordinate[3];

  mx =  2.0*Re12;
  my = -2.0*Im12;
  mz = Re11 - Re22;
  
  xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);

  absm = S_coordinate[0];
  theta = S_coordinate[1];
  phi = S_coordinate[2]; 
  tn = Re11 + Re22;

  Nup[0]   = 0.5*(tn + absm);
  Nup[1]   = 0.0;
    
  Ndown[0] = 0.5*(tn - absm);
  Ndown[1] = 0.0;
  
  t[0] = theta;
  t[1] = 0.0;
  
  p[0] = phi;
  p[1] = 0.0;


  /*
  if ((Re11+1.0e-10)<Re22){

    Nup[0]   = 0.5*(tn - absm);
    Nup[1]   = 0.0;
    
    Ndown[0] = 0.5*(tn + absm);
    Ndown[1] = 0.0;

    t[0] = 3.0*PI + theta;
    t[1] = 0.0;
  
    p[0] = phi;
    p[1] = 0.0;
  }

  else {

    Nup[0]   = 0.5*(tn + absm);
    Nup[1]   = 0.0;
    
    Ndown[0] = 0.5*(tn - absm);
    Ndown[1] = 0.0;
  
    t[0] = theta;
    t[1] = 0.0;
  
    p[0] = phi;
    p[1] = 0.0;
  }
  */

}
